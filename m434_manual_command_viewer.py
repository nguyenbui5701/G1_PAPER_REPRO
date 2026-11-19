#!/usr/bin/env python3
"""M4.34: manually commanded G1 MJX diagnostic viewer (no PPO training).

Runs the existing 401M PPO checkpoint in original nominal MJX physics. A
separate CPU MuJoCo model renders poses only. Keyboard commands replace the
sampled joystick target in `state.info['command']` AND the actor/critic
observation command slots. This makes the experiment a *controlled-command
DIAGNOSTIC*, not a paper-evaluation replay or a 1:1 M4.32 comparison.

Keys: W/S increment/decrement forward velocity (x), A/D increment/decrement
lateral (y), J/L increment/decrement yaw rate, X zero all commands,
Space pause/resume, R reset, Q quit. Commands persist until changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

from m433_visual_validation import preflight

WORK = Path(__file__).resolve().parent
OUT_DIR = WORK / 'm434_manual_control'
DT_EXPECTED = 0.02
# Audited state observation layout: linvel(3), gyro(3), gravity(3), command(3).
CMD_START, CMD_END = 9, 12


def adjust_command(old, key):
    """Pure helper, independent of JAX/GUI, usable for test/preflight."""
    cmd = list(old)
    if key == 'w': cmd[0] = min(0.6, round(cmd[0]+0.1, 4))
    elif key == 's': cmd[0] = max(-0.6, round(cmd[0]-0.1, 4))
    elif key == 'a': cmd[1] = min(0.3, round(cmd[1]+0.1, 4))
    elif key == 'd': cmd[1] = max(-0.3, round(cmd[1]-0.1, 4))
    elif key == 'j': cmd[2] = min(0.8, round(cmd[2]+0.2, 4))
    elif key == 'l': cmd[2] = max(-0.8, round(cmd[2]-0.2, 4))
    elif key == 'x': cmd = [0.0, 0.0, 0.0]
    else: return None
    return cmd


def patch_obs_and_info(state, command, jnp):
    """Update *both* command metadata and actor/critic observations.

    Do not modify environment physics, reward, termination or source code.
    We override the source's sampled command after every environment step.
    """
    if tuple(state.info['command'].shape) != (3,):
        raise RuntimeError('INVALID_ENV_INFO_COMMAND_SHAPE')
    obs = {}
    for name, value in state.obs.items():
        if name not in ('state', 'privileged_state'):
            raise RuntimeError(f'UNEXPECTED_OBS_KEY: {name}')
        if value.ndim != 1 or value.shape[0] < CMD_END:
            raise RuntimeError(f'INVALID_OBS_LAYOUT: {name} {value.shape}')
        obs[name] = value.at[CMD_START:CMD_END].set(command)
    if set(obs) != {'state', 'privileged_state'}:
        raise RuntimeError('OBSERVATION_KEYS_CHANGED')
    info = {**state.info, 'command': command}
    return state.replace(info=info, obs=obs)


def validate_initial_observation(state, jnp):
    """Verify source's known command offset BEFORE any command injection."""
    original = state.info['command']
    if tuple(original.shape) != (3,):
        raise RuntimeError('SOURCE_COMMAND_SHAPE_CHANGED')
    if set(state.obs) != {'state', 'privileged_state'}:
        raise RuntimeError('SOURCE_OBSERVATION_KEYS_CHANGED')
    for name in ('state', 'privileged_state'):
        observed = state.obs[name]
        if observed.ndim != 1 or observed.shape[0] < CMD_END:
            raise RuntimeError(f'SOURCE_OBSERVATION_SHAPE_CHANGED {name}')
        if not bool(jnp.allclose(observed[CMD_START:CMD_END], original, rtol=0, atol=1e-6)):
            raise RuntimeError(f'SOURCE_COMMAND_NOT_AT_OBS_9_12 {name}')


def run(args, ctx):
    from paper_config import BASE, ENV
    from train import check_source
    check_source()
    sys.path.insert(0, str(BASE / 'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('NVCC_COMPAT:', apply(), flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import locomotion
    import mujoco
    import mujoco.viewer
    from m426_heldout_det_stoch import restore_params

    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU_BACKEND_REQUIRED')
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('GRAPHICAL_DESKTOP_REQUIRED')
    env = locomotion.load(ENV, config=locomotion.get_default_config(ENV))
    dt = float(env.dt)
    if not (0 < dt <= 0.1):
        raise RuntimeError(f'INVALID_ENV_DT {dt}')
    print('M434_SOURCE_PUSH_ENABLED', bool(env._config.push_config.enable), flush=True)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    test_state = reset_fn(jax.random.PRNGKey(999))
    validate_initial_observation(test_state, jnp)
    obs_size = jax.tree_util.tree_map(lambda a: a.shape, test_state.obs)
    net = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ctx['report']['scaled_config']['network_factory'],
    )
    params = restore_params(ctx['checkpoint_path'].read_bytes(),
                            serialization, running_statistics, jax, jnp)
    policy = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))

    model = env.mj_model
    if not isinstance(model, mujoco.MjModel):
        raise RuntimeError('CPU_VISUAL_MODEL_NOT_AVAILABLE')
    view_data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

    def mirror(state):
        np.copyto(view_data.qpos, np.asarray(jax.device_get(state.data.qpos)))
        np.copyto(view_data.qvel, np.asarray(jax.device_get(state.data.qvel)))
        if getattr(state.data, 'time', None) is not None:
            view_data.time = float(jax.device_get(state.data.time))
        mujoco.mj_forward(model, view_data)

    control = {'cmd': [0.0, 0.0, 0.0], 'paused': False, 'reset': False,
               'quit': False, 'events': []}
    lock = threading.Lock()
    samples = []
    episode_events = []
    steps = 0
    episode_index = 0
    finished = False
    stop_reason = None

    def reset_episode():
        # Same reset key logic as M4.33, but commands are user-controlled.
        key = jax.random.PRNGKey(args.seed)
        key, reset_key = jax.random.split(key)
        state = reset_fn(reset_key)
        validate_initial_observation(state, jnp)
        state = patch_obs_and_info(state, jnp.zeros((3,), dtype=jnp.float32), jnp)
        return state, key

    def on_key(keycode):
        nonlocal steps
        try:
            key = chr(keycode).lower()
        except (ValueError, OverflowError):
            key = ''
        with lock:
            if keycode == 32:
                control['paused'] = not control['paused']
                return
            if key == 'r':
                control['reset'] = True
                return
            if key == 'q':
                control['quit'] = True
                return
            next_cmd = adjust_command(control['cmd'], key)
            if next_cmd is not None:
                control['cmd'] = next_cmd
                evt = {'step_when_pressed': steps, 'key': key, 'command': list(next_cmd)}
                control['events'].append(evt)
                print(f"M434_COMMAND key={key} vx={next_cmd[0]:+.2f} "
                      f"vy={next_cmd[1]:+.2f} yaw={next_cmd[2]:+.2f}", flush=True)

    state, key = reset_episode()
    mirror(state)
    print('M434_VIEWER_START: W/S=Vx, A/D=Vy, J/L=yaw, X=zero, '
          'Space=pause, R=reset, Q=quit', flush=True)
    print('M434_NOTICE manual commands differ from M4.32; NOT_A_MATCHED_REPLAY', flush=True)
    try:
        with mujoco.viewer.launch_passive(model, view_data, key_callback=on_key,
                                          show_left_ui=True, show_right_ui=False) as viewer:
            if pelvis_id >= 0:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = pelvis_id
                    viewer.cam.distance = 3.0
                    viewer.cam.azimuth = 140
                    viewer.cam.elevation = -15
            viewer.sync()
            while viewer.is_running():
                with lock:
                    paused = control['paused']
                    reset_requested = control['reset']
                    quit_requested = control['quit']
                    cmd = list(control['cmd'])
                    if reset_requested:
                        control['reset'] = False
                        control['cmd'] = [0.0, 0.0, 0.0]
                        cmd = [0.0, 0.0, 0.0]
                if quit_requested:
                    stop_reason = 'keyboard_quit'
                    break
                if reset_requested:
                    episode_events.append({'episode': episode_index, 'end_step': steps,
                                           'reset_pressed': True})
                    episode_index += 1
                    state, key = reset_episode()
                    steps, finished = 0, False
                    with viewer.lock():
                        mirror(state)
                    viewer.sync()
                    print('M434_EPISODE_RESET', flush=True)
                if finished or paused:
                    viewer.sync()
                    time.sleep(0.04)
                    continue
                t0 = time.perf_counter()
                command = jnp.asarray(cmd, dtype=jnp.float32)
                # Force current command into both observations and environment
                # metadata, even though the source samples its own every 500 steps.
                state = patch_obs_and_info(state, command, jnp)
                local_v = np.asarray(jax.device_get(env.get_local_linvel(state.data, 'pelvis')))
                local_w = np.asarray(jax.device_get(env.get_gyro(state.data, 'pelvis')))
                pos = np.asarray(jax.device_get(state.data.qpos[:3]))
                key, action_key = jax.random.split(key)
                action, _ = policy(state.obs, action_key)
                if not bool(jnp.all(jnp.isfinite(action))):
                    raise RuntimeError(f'NONFINITE_ACTION step={steps+1}')
                state = step_fn(state, action)
                steps += 1
                samples.append({'episode': episode_index, 'step': steps,
                                'time_s': round((steps-1)*dt,5),
                                'command': cmd, 'actual_xy': [float(x) for x in local_v[:2]],
                                'actual_yaw': float(local_w[2]),
                                'root_xyz': [float(x) for x in pos],
                                'done_after_step': bool(state.done)})
                with viewer.lock():
                    mirror(state)
                viewer.sync()
                if steps == 1 or steps % 50 == 0:
                    print('M434_PROGRESS step={} cmd=[{:+.2f},{:+.2f},{:+.2f}] '
                          'actual=[{:+.2f},{:+.2f},{:+.2f}] z={:.3f}'.format(
                              steps, *cmd, *local_v[:2], local_w[2], pos[2]), flush=True)
                if bool(state.done) or steps >= args.max_steps:
                    finished = True
                    stop_reason = 'termination' if bool(state.done) else 'max_steps'
                    print('M434_EPISODE_END', f'steps={steps}',
                          f'terminated={bool(state.done)}', flush=True)
                delay = dt / args.speed - (time.perf_counter()-t0)
                if delay > 0:
                    time.sleep(delay)
    finally:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / (f'm434_manual_seed{args.train_seed}_{args.seed}_'
                         f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.json')
        with lock:
            events = list(control['events'])
        payload = {'status':'M434_MANUAL_VIEWER_CLOSED',
                   'experiment_class':'DIAGNOSTIC_NOT_PAPER_METRIC',
                   'policy':'deterministic', 'training_seed':args.train_seed,
                   'eval_reset_seed':args.seed, 'checkpoint_sha256':ctx['checkpoint_sha256'],
                   'max_steps':args.max_steps, 'physics':'nominal_original_push_enabled',
                   'commands_overridden':True, 'no_training':True,
                   'control_keys':'W/S Vx; A/D Vy; J/L yaw; X stop; Space pause; R reset; Q quit',
                   'stop_reason':stop_reason,
                   'events':events, 'episode_events':episode_events,
                   'samples':samples}
        with out.open('x',encoding='utf-8') as f:
            json.dump(payload,f,indent=2,ensure_ascii=False)
            f.write('\n')
        print('M434_OUTPUT',out,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inspect','run'),required=True)
    parser.add_argument('--train-seed',type=int,choices=(0,1),default=1)
    parser.add_argument('--seed',type=int,default=5186)
    parser.add_argument('--speed',type=float,default=1.0)
    parser.add_argument('--max-steps',type=int,default=1000)
    args=parser.parse_args()
    if not math.isfinite(args.speed) or not (0.1 <= args.speed <= 5.0):
        parser.error('--speed must be between 0.1 and 5.0')
    if args.max_steps<1 or args.max_steps>5000:
        parser.error('--max-steps must be in [1,5000]')
    ctx=preflight(args.train_seed,args.seed,'deterministic')
    if args.mode=='inspect':
        assert adjust_command([0.,0.,0.],'s')==[-0.1,0.,0.]
        assert adjust_command([0.,0.,0.],'j')==[0.,0.,0.2]
        assert adjust_command([0.1,-0.1,-0.2],'x')==[0.,0.,0.]
        print('M434_INSPECT_PASS NO_GPU_ROLLOUT NO_TRAINING',flush=True)
        return
    run(args,ctx)

if __name__=='__main__':
    main()
