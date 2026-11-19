#!/usr/bin/env python3
"""M4.35: G1 stability diagnosis — original source push ON vs push OFF.

Read-only policy; MJX physics remain unchanged except runtime push_config.enable
for the OFF comparison. Fixed deterministic commands are injected into both
actor observation and state.info exactly as in M4.34. This is NOT a paper
benchmark nor a training configuration change.

Files stay inside G1_PAPER_REPRO/. It requires local m433_visual_validation.py,
m434_manual_command_viewer.py, m426_heldout_det_stoch.py, train.py, paper_config.py.

--mode inspect: checkpoint/provenance/CLI checks; no JAX training or rollout.
--mode run: nominal deterministic paired rollout, no GUI, writes JSON once.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'm435_push_ab_stability'
CMD = {
    'stand': (0., 0., 0.),
    'backward': (-0.3, 0., 0.),
    'forward': (0.3, 0., 0.),
    'yaw_right': (0., 0., -0.4),
    'yaw_left': (0., 0., 0.4),
}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def run(args, ctx, selected):
    # The existing shim MUST execute before importing JAX.
    from m434_manual_command_viewer import patch_obs_and_info, validate_initial_observation
    from paper_config import BASE, ENV
    from train import check_source
    check_source()
    sys.path.insert(0, str(BASE / 'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('NVCC_COMPAT:', apply(), flush=True)

    import jax
    import jax.numpy as jnp
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import locomotion
    from m426_heldout_det_stoch import restore_params

    if jax.default_backend() != 'gpu':
        raise RuntimeError('M435_REQUIRES_GPU_BACKEND')
    if OUT.exists() and list(OUT.glob('m435*.json')):
        print('M435_NOTE existing output retained; new run creates a distinct timestamped file', flush=True)

    probe_cfg = locomotion.get_default_config(ENV)
    if bool(probe_cfg.push_config.enable) is not True:
        raise RuntimeError('PINNED_SOURCE_DEFAULT_PUSH_NOT_ENABLED; refusing A/B')
    probe_env = locomotion.load(ENV, config=probe_cfg)
    test_state = jax.jit(probe_env.reset)(jax.random.PRNGKey(999))
    validate_initial_observation(test_state, jnp)
    network = ppo_networks.make_ppo_networks(
        observation_size=jax.tree_util.tree_map(lambda a: a.shape, test_state.obs),
        action_size=probe_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ctx['report']['scaled_config']['network_factory'],
    )
    params = restore_params(ctx['checkpoint_path'].read_bytes(),
                            serialization, running_statistics, jax, jnp)
    policy = jax.jit(ppo_networks.make_inference_fn(network)(params, deterministic=True))

    results = []
    for case in selected:
        for push_enabled in (True, False):
            cfg = locomotion.get_default_config(ENV)
            cfg.push_config.enable = push_enabled  # this *one* runtime toggle
            env = locomotion.load(ENV, config=cfg)
            reset, step_fn = jax.jit(env.reset), jax.jit(env.step)
            key = jax.random.PRNGKey(args.seed)
            key, reset_key = jax.random.split(key)
            state = reset(reset_key)
            validate_initial_observation(state, jnp)
            command = jnp.array(CMD[case], dtype=jnp.float32)
            if not math.isclose(float(env.dt), 0.02, rel_tol=0, abs_tol=1e-6):
                raise RuntimeError(f'EXPECTED_DT_0.02 got {float(env.dt)}')
            frames = []
            push_events = []
            done = False
            print(f'M435_START case={case} push_enabled={push_enabled} seed={args.seed}', flush=True)
            for step_index in range(1, args.max_steps + 1):
                state = patch_obs_and_info(state, command, jnp)
                body_velocity = jax.device_get(env.get_local_linvel(state.data, 'pelvis'))
                gyro = jax.device_get(env.get_gyro(state.data, 'pelvis'))
                root_height = float(state.data.qpos[2])
                key, action_key = jax.random.split(key)
                action, _ = policy(state.obs, action_key)
                if not bool(jnp.all(jnp.isfinite(action))):
                    raise RuntimeError(f'NONFINITE_ACTION case={case} push={push_enabled} step={step_index}')
                state = step_fn(state, action)
                done = bool(state.done)
                # Source stores the push direction vector, not its sampled magnitude.
                push = [float(v) for v in jax.device_get(state.info['push'])]
                if len(push) != 2 or not all(math.isfinite(v) for v in push):
                    raise RuntimeError('SOURCE_PUSH_RECORD_INVALID')
                if math.hypot(*push) > 0.01:
                    push_events.append({'step': step_index, 'direction_xy': push})
                    print(f'M435_PUSH_EVENT case={case} enabled={push_enabled} step={step_index} direction={push}', flush=True)
                v = [float(v) for v in body_velocity[:2]]
                yaw = float(gyro[2])
                frame = {'step': step_index, 'root_z': root_height,
                         'actual_xy': v, 'actual_yaw': yaw,
                         'done': done}
                if not all(math.isfinite(v) for v in (root_height, *v, yaw)):
                    raise RuntimeError('NONFINITE_TELEMETRY')
                frames.append(frame)
                if step_index % 200 == 0 or done:
                    print(f'M435_PROGRESS case={case} push={push_enabled} step={step_index} root_z={root_height:.3f} done={done}', flush=True)
                if done:
                    break

            if not push_enabled and push_events:
                raise RuntimeError('PUSH_OFF_ARM_STILL_HAS_PUSH_EVENTS')
            last = frames[-1]
            diag = {
                'case': case, 'command': list(CMD[case]),
                'push_enabled': push_enabled,
                'steps': len(frames), 'terminated': done,
                'completed_1000_steps': len(frames) == 1000 and not done,
                'push_event_steps': [p['step'] for p in push_events],
                'push_events': push_events,
                'root_z_min': min(x['root_z'] for x in frames),
                'root_z_last': last['root_z'],
                'mean_actual_xy': [mean([x['actual_xy'][k] for x in frames]) for k in range(2)],
                'mean_actual_yaw': mean([x['actual_yaw'] for x in frames]),
                'xy_rmse': math.sqrt(mean([sum((x['actual_xy'][k] - CMD[case][k])**2 for k in range(2)) for x in frames])),
                'yaw_rmse': math.sqrt(mean([(x['actual_yaw'] - CMD[case][2])**2 for x in frames])),
                'frames': frames,
            }
            results.append(diag)
            print('M435_RESULT',f'case={case}',f'push={push_enabled}',
                  f'steps={len(frames)}',f'push_events={diag["push_event_steps"]}',
                  f'root_z_min={diag["root_z_min"]:.3f}', flush=True)

    comparisons = []
    for case in selected:
        on, off = [x for x in results if x['case'] == case and x['push_enabled']], [x for x in results if x['case'] == case and not x['push_enabled']]
        if len(on) != 1 or len(off) != 1:
            raise RuntimeError(f'INCOMPLETE_PAIRED_CASE {case}')
        comparisons.append({'case': case,
                            'push_on_steps': on[0]['steps'],
                            'push_off_steps': off[0]['steps'],
                            'difference_off_minus_on': off[0]['steps']-on[0]['steps'],
                            'push_on_events': on[0]['push_event_steps']})
    OUT.mkdir(parents=True, exist_ok=True)
    outfile = OUT / ('m435_push_ab_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    data = {
        'status': 'M435_PUSH_AB_COMPLETED',
        'checkpoint_sha256': ctx['checkpoint_sha256'],
        'train_seed': args.train_seed,
        'eval_reset_seed': args.seed,
        'environment': ENV,
        'max_steps': args.max_steps,
        'training_repeated': False,
        'reward_termination_modified': False,
        'domain_randomization': False,
        'experiment_type': 'DIAGNOSTIC_FIXED_COMMAND_NOT_PAPER_EVALUATION',
        'only_ab_difference': 'source default push_config.enable True vs False at environment construction',
        'push_event_note': 'source info.push is the push direction vector, not the sampled magnitude',
        'cases': selected,
        'paired_comparison': comparisons,
        'rollouts': results,
    }
    with outfile.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('M435_PUSH_AB_COMPLETED',flush=True)
    print('OUTPUT:',outfile,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('inspect','run'))
    parser.add_argument('--train-seed', type=int, choices=(0,1), default=1)
    parser.add_argument('--seed', type=int, default=5186)
    parser.add_argument('--cases', default='stand,backward,yaw_right')
    parser.add_argument('--max-steps', type=int, default=1000)
    args = parser.parse_args()
    cases = args.cases.split(',')
    if not cases or len(set(cases)) != len(cases) or any(x not in CMD for x in cases):
        parser.error('cases must be unique, comma-separated names: ' + ','.join(CMD))
    if args.max_steps < 10 or args.max_steps > 1000:
        parser.error('--max-steps must be from 10 to 1000')
    from m433_visual_validation import preflight
    ctx = preflight(args.train_seed, args.seed, 'deterministic')
    if args.mode == 'inspect':
        print('M435_INSPECT_PASS NO_ROLLOUT NO_TRAINING',flush=True)
        print('CASE_LIST:',cases,'CHECKPOINT_SHA256:',ctx['checkpoint_sha256'],flush=True)
        return
    run(args,ctx,cases)

if __name__ == '__main__':
    main()
