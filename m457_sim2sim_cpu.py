#!/usr/bin/env python3
"""M4.57 | Read-only G1 MJX -> native MuJoCo CPU sim-to-sim diagnostic.

Pinned G1JoystickFlatTerrain (2025), deterministic seed-1 PPO 401M checkpoint.
Native MuJoCo CPU independently advances physics with mujoco.mj_step;
policy inference uses the existing JAX/GPU checkpoint and observation normalization.
CPU observations are reconstructed from native MjData sensors (NOT MJX poses).
A separate MJX reference is advanced in parallel from the same randomized reset.

Experimental scope: fixed commands and push OFF; nominal diagnostic, not a
paper-exact benchmark, pure CPU policy deployment, or sim-to-real validation.
No training, source/model/reward/PD changes or dependency installations.

Modes: inspect (no JAX/GPU), smoke (one control step), run (full episodes).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'm457_sim2sim_cpu'
FINAL_STEP = 401_080_320
SEED1_SHA256 = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
COMMANDS = {
    'stand': (0.0, 0.0, 0.0),
    'forward': (0.3, 0.0, 0.0),
    'backward': (-0.3, 0.0, 0.0),
    'yaw_left': (0.0, 0.0, 0.4),
    'yaw_right': (0.0, 0.0, -0.4),
}


def preflight(seed: int) -> dict:
    # Same provenance logic already used by M4.34/M4.37; does not import JAX.
    from m433_visual_validation import preflight as existing_preflight
    from train import check_source
    check_source()
    ctx = existing_preflight(1, seed, 'deterministic')
    if ctx['checkpoint_sha256'] != SEED1_SHA256:
        raise RuntimeError('M457_SEED1_SHA256_CHANGED')
    if int(ctx['report']['checkpoints'][-1]['step']) != FINAL_STEP:
        raise RuntimeError('M457_CHECKPOINT_STEP_CHANGED')
    print('M457_PREFLIGHT_PASS', 'seed=',seed,
          'checkpoint=',ctx['checkpoint_sha256'],flush=True)
    return ctx


def average(vals):
    return float(sum(vals) / len(vals)) if vals else None


def rmse(vals):
    return math.sqrt(sum(float(x)*float(x) for x in vals)/len(vals)) if vals else None


def require_finite(values, why):
    import numpy as np
    if not np.all(np.isfinite(values)):
        raise RuntimeError('M457_NONFINITE_'+why)


def run(args, ctx, cases):
    # The non-idempotent NVCC shim must run ONCE before any JAX or Playground import.
    from paper_config import BASE, ENV
    sys.path.insert(0, str(BASE / 'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('NVCC_COMPAT:', apply(), flush=True)

    import numpy as np
    import jax
    import jax.numpy as jnp
    import mujoco
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import locomotion
    from m426_heldout_det_stoch import restore_params
    from m434_manual_command_viewer import patch_obs_and_info, validate_initial_observation

    if jax.default_backend() != 'gpu':
        raise RuntimeError('M457_GPU_BACKEND_REQUIRED_FOR_PINNED_POLICY')

    cfg = locomotion.get_default_config(ENV)
    if not bool(cfg.push_config.enable):
        raise RuntimeError('M457_EXPECTED_SOURCE_DEFAULT_PUSH_ON')
    cfg.push_config.enable = False  # controlled nominal diagnostic in both engines
    env = locomotion.load(ENV, config=cfg)
    model = env.mj_model
    if not isinstance(model, mujoco.MjModel):
        raise RuntimeError('M457_MODEL_NOT_NATIVE_MUJOCO')
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise RuntimeError('M457_G1_MODEL_DIMS_CHANGED')
    n_substeps = int(env.n_substeps)
    if (n_substeps != 10 or abs(float(env.dt)-.02) > 1e-7 or
            abs(float(model.opt.timestep)-.002) > 1e-7):
        raise RuntimeError('M457_INCONSISTENT_SIMULATION_TIMESTEPS')

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    # Use EXACT same checkpoint restoration and actor factory as earlier M4.37.
    test_state = reset_fn(jax.random.PRNGKey(999))
    validate_initial_observation(test_state, jnp)
    shapes = jax.tree_util.tree_map(lambda z: z.shape, test_state.obs)
    # Pinned G1 joystick.py: feet velocity has 2x3=6 values, not 4x3.
    # Critic obs = 103 actor obs + 113 privileged extras = 216.
    if shapes != {'state': (103,), 'privileged_state': (216,)}:
        raise RuntimeError(f'M457_OBSERVATION_SHAPES_CHANGED {shapes}')
    net = ppo_networks.make_ppo_networks(
        observation_size=shapes,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ctx['report']['scaled_config']['network_factory'])
    params = restore_params(ctx['checkpoint_path'].read_bytes(),
                            serialization, running_statistics, jax, jnp)
    policy = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))
    default_pose = np.asarray(jax.device_get(env._default_pose), dtype=np.float64)
    if default_pose.shape != (model.nu,):
        raise RuntimeError('M457_DEFAULT_POSE_ACTUATOR_MISMATCH')

    def sensor(cpu_data, name):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise RuntimeError('M457_SENSOR_NOT_FOUND: '+name)
        adr = int(model.sensor_adr[sensor_id])
        dim = int(model.sensor_dim[sensor_id])
        return np.asarray(cpu_data.sensordata[adr:adr+dim],dtype=np.float64).copy()

    # Validate sensor bindings BEFORE any run, rather than guessing IMU units.
    mandatory = (
        'gyro_pelvis','local_linvel_pelvis','upvector_torso',
        'accelerometer_pelvis','global_angvel_pelvis',
    )
    for sensor_name in mandatory:
        sensor_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,sensor_name)
        if sensor_id < 0:
            raise RuntimeError('M457_MISSING_SOURCE_SENSOR: '+sensor_name)
    pelvis_imu_site = int(env._pelvis_imu_site_id)
    feet_ids = [int(x) for x in np.asarray(env._feet_geom_id)]
    floor_id = int(env._floor_geom_id)
    feet_velocity_adr = np.asarray(jax.device_get(env._foot_linvel_sensor_adr),dtype=int)
    if feet_velocity_adr.shape != (2,3):
        raise RuntimeError('M457_FOOT_VELOCITY_SENSOR_SHAPE_CHANGED')
    right_foot_id = int(env._right_foot_geom_id)
    left_foot_id = int(env._left_foot_geom_id)
    right_shin_id = int(env._right_shin_geom_id)
    left_shin_id = int(env._left_shin_geom_id)

    def is_colliding(cpu_data, g1, g2):
        # Source collision.geoms_colliding checks min(contact.dist) < 0.
        return any(
            ((int(c.geom1)==g1 and int(c.geom2)==g2) or
             (int(c.geom1)==g2 and int(c.geom2)==g1)) and float(c.dist)<0
            for c in cpu_data.contact[:cpu_data.ncon]
        )

    def cpu_contacts(cpu_data):
        return np.array([is_colliding(cpu_data,g,floor_id) for g in feet_ids],dtype=bool)

    def cpu_termination(cpu_data):
        predicates={
            'torso_upside_down':float(sensor(cpu_data,'upvector_torso')[-1]) < 0.0,
            'foot_foot_collision':is_colliding(cpu_data,right_foot_id,left_foot_id),
            'left_foot_right_shin_collision':is_colliding(cpu_data,left_foot_id,right_shin_id),
            'right_foot_left_shin_collision':is_colliding(cpu_data,right_foot_id,left_shin_id),
            'qpos_nan':bool(np.isnan(cpu_data.qpos).any()),
            'qvel_nan':bool(np.isnan(cpu_data.qvel).any()),
        }
        return any(predicates.values()), [k for k,v in predicates.items() if v]

    # Reproduce the source's RNG splitting and five independent observation
    # noise streams. Sampling remains in JAX; all physical sensors are from CPU.
    @jax.jit
    def cpu_noise_and_advance_rng(rng):
        rng, push1_rng, push2_rng = jax.random.split(rng,3)
        # push1_rng/push2_rng are deliberately unused, as push is OFF.
        rng, kgyro = jax.random.split(rng)
        rng, kgravity = jax.random.split(rng)
        rng, kjoint = jax.random.split(rng)
        rng, kjointvel = jax.random.split(rng)
        rng, klinvel = jax.random.split(rng)
        rng, cmd_rng = jax.random.split(rng)
        return (rng,
                jax.random.uniform(kgyro, (3,)),
                jax.random.uniform(kgravity, (3,)),
                jax.random.uniform(kjoint, (29,)),
                jax.random.uniform(kjointvel, (29,)),
                jax.random.uniform(klinvel, (3,)))

    def cpu_observation(cpu_data, info, command, contact):
        noise_result = jax.device_get(cpu_noise_and_advance_rng(info['rng']))
        info['rng'] = noise_result[0]
        scale = cfg.noise_config.scales
        level = float(cfg.noise_config.level)
        gyro = sensor(cpu_data, 'gyro_pelvis')
        gravity = np.asarray(cpu_data.site_xmat[pelvis_imu_site],dtype=np.float64).reshape(3,3).T @ np.array([0.,0.,-1.])
        joint_pos = np.asarray(cpu_data.qpos[7:],dtype=np.float64)
        joint_vel = np.asarray(cpu_data.qvel[6:],dtype=np.float64)
        linvel = sensor(cpu_data,'local_linvel_pelvis')
        def noisy(x, i, s):
            return x + (2*np.asarray(noise_result[i],dtype=np.float64)-1)*level*float(s)
        phase = np.concatenate([np.cos(info['phase']),np.sin(info['phase'])])
        obs = np.hstack((
            noisy(linvel,5,scale.linvel),
            noisy(gyro,1,scale.gyro),
            noisy(gravity,2,scale.gravity),
            command,
            noisy(joint_pos,3,scale.joint_pos)-default_pose,
            noisy(joint_vel,4,scale.joint_vel),
            info['last_act'],phase,
        )).astype(np.float32)
        accelerometer=sensor(cpu_data,'accelerometer_pelvis')
        global_angvel=sensor(cpu_data,'global_angvel_pelvis')
        feet_vel=np.asarray(cpu_data.sensordata[feet_velocity_adr],dtype=np.float64).ravel()
        privileged=np.hstack((
            obs, gyro,accelerometer,gravity,linvel,global_angvel,
            joint_pos-default_pose,joint_vel,[cpu_data.qpos[2]],
            cpu_data.actuator_force,contact.astype(np.float64),
            feet_vel,info['feet_air_time'],
        )).astype(np.float32)
        # CPU uses the same two 3D foot-velocity sensors as source MJX.
        if obs.shape!=(103,) or privileged.shape!=(216,):
            raise RuntimeError(f'M457_CPU_OBS_SHAPE_MISMATCH: {obs.shape},{privileged.shape}')
        require_finite(obs,'CPU_OBS')
        require_finite(privileged,'CPU_PRIVILEGED_OBS')
        return {'state':jnp.asarray(obs),'privileged_state':jnp.asarray(privileged)}

    def make_cpu_info(state):
        return {
            'rng':state.info['rng'],
            'phase':np.asarray(jax.device_get(state.info['phase']),dtype=np.float32).copy(),
            'phase_dt':np.asarray(jax.device_get(state.info['phase_dt']),dtype=np.float32).copy(),
            'last_act':np.asarray(jax.device_get(state.info['last_act']),dtype=np.float32).copy(),
            'feet_air_time':np.asarray(jax.device_get(state.info['feet_air_time']),dtype=np.float32).copy(),
            'last_contact':np.asarray(jax.device_get(state.info['last_contact']),dtype=bool).copy(),
        }

    def cpu_step(cpu_data, info, action, command):
        # Exactly the source's position-control action convention. No custom PD.
        np.copyto(cpu_data.ctrl, default_pose + action.astype(np.float64)*float(cfg.action_scale))
        for _ in range(n_substeps):
            mujoco.mj_step(model,cpu_data)  # CPU independently advances physics
        require_finite(cpu_data.qpos,'CPU_QPOS')
        require_finite(cpu_data.qvel,'CPU_QVEL')
        contact = cpu_contacts(cpu_data)
        info['feet_air_time'] += np.float32(env.dt)
        obs_next=cpu_observation(cpu_data,info,command,contact)
        # Source phase update happens AFTER computing next observation.
        p=info['phase'] + info['phase_dt']
        info['phase']=np.fmod(p+np.float32(np.pi),np.float32(2*np.pi))-np.float32(np.pi)
        info['last_act']=np.asarray(action,dtype=np.float32).copy()
        info['feet_air_time'] *= (~contact).astype(np.float32)
        info['last_contact']=contact.copy()
        done,reasons=cpu_termination(cpu_data)
        return obs_next,done,reasons

    def cpu_measure(cpu_data):
        v=sensor(cpu_data,'local_linvel_pelvis')
        w=sensor(cpu_data,'gyro_pelvis')
        return {'root_z_m':float(cpu_data.qpos[2]),'vx_mps':float(v[0]),
                'vy_mps':float(v[1]),'yaw_rate_radps':float(w[2]),
                'foot_floor_contact':[bool(x) for x in cpu_contacts(cpu_data)]}

    def mjx_measure(state):
        d=state.data
        v=np.asarray(jax.device_get(env.get_local_linvel(d,'pelvis')))
        w=np.asarray(jax.device_get(env.get_gyro(d,'pelvis')))
        return {'root_z_m':float(jax.device_get(d.qpos[2])),'vx_mps':float(v[0]),
                'vy_mps':float(v[1]),'yaw_rate_radps':float(w[2])}

    runs=[]
    limit=1 if args.mode=='smoke' else args.max_steps
    for case in cases:
        command_np=np.asarray(COMMANDS[case],dtype=np.float32)
        command_jax=jnp.asarray(command_np)
        key=jax.random.PRNGKey(args.seed)
        key,reset_key=jax.random.split(key)
        state=reset_fn(reset_key)
        validate_initial_observation(state,jnp)
        cpu_data=mujoco.MjData(model)
        np.copyto(cpu_data.qpos,np.asarray(jax.device_get(state.data.qpos)))
        np.copyto(cpu_data.qvel,np.asarray(jax.device_get(state.data.qvel)))
        np.copyto(cpu_data.ctrl,np.asarray(jax.device_get(state.data.ctrl)))
        if model.na:
            np.copyto(cpu_data.act,np.asarray(jax.device_get(state.data.act)))
        cpu_data.time=float(jax.device_get(state.data.time))
        mujoco.mj_forward(model,cpu_data)
        # Diagnostic only: inspect native sensor parity at the shared reset.
        # Initial *policy* observation below intentionally reuses MJX; this
        # check is for raw sensor values, not noisy actor-observation parity.
        initial_sensor_error = {
            'local_linvel_pelvis_max_abs':float(np.max(np.abs(
                np.asarray(jax.device_get(env.get_local_linvel(state.data,'pelvis')))
                - sensor(cpu_data,'local_linvel_pelvis')))),
            'gyro_pelvis_max_abs':float(np.max(np.abs(
                np.asarray(jax.device_get(env.get_gyro(state.data,'pelvis')))
                - sensor(cpu_data,'gyro_pelvis')))),
            'upvector_torso_max_abs':float(np.max(np.abs(
                np.asarray(jax.device_get(env.get_gravity(state.data,'torso')))
                - sensor(cpu_data,'upvector_torso')))),
        }
        info=make_cpu_info(state)
        # At reset the CPU and MJX start from the same qpos/qvel, and we reuse
        # the source-generated initial observation for the FIRST policy action.
        cpu_obs=patch_obs_and_info(state,command_jax,jnp).obs
        ref_alive=True
        cpu_alive=True
        mjx_frames=[]
        cpu_frames=[]
        paired_errors=[]
        paired_action_errors=[]
        cpu_terminal_reasons=[]
        first_action_max_abs=None
        print(f'M457_START case={case} seed={args.seed} mode={args.mode} max_steps={limit}',flush=True)

        ref_rng=key
        cpu_rng=key
        for step_idx in range(1,limit+1):
            ref_action_this_step = None
            if ref_alive:
                state=patch_obs_and_info(state,command_jax,jnp)
                ref_rng,act_rng=jax.random.split(ref_rng)
                action_jax,_=policy(state.obs,act_rng)
                action_ref=np.asarray(jax.device_get(action_jax),dtype=np.float64)
                ref_action_this_step = action_ref
                require_finite(action_ref,'MJX_ACTION')
                state=step_fn(state,action_jax)
                ref_frame=mjx_measure(state)
                ref_frame.update({'step':step_idx,'terminated':bool(state.done)})
                mjx_frames.append(ref_frame)
                ref_alive=not bool(state.done)
                if bool(np.any(np.asarray(jax.device_get(state.info['push'])))):
                    raise RuntimeError('M457_UNEXPECTED_PUSH_MJX')
            if cpu_alive:
                cpu_rng,cpu_act_rng=jax.random.split(cpu_rng)
                cpu_action_jax,_=policy(cpu_obs,cpu_act_rng)
                cpu_action=np.asarray(jax.device_get(cpu_action_jax),dtype=np.float64)
                require_finite(cpu_action,'CPU_POLICY_ACTION')
                if ref_action_this_step is not None:
                    paired_action_errors.append({
                        'step':step_idx,
                        'max_abs_action':float(np.max(np.abs(cpu_action-ref_action_this_step))),
                    })
                if step_idx == 1:
                    first_action_max_abs = float(np.max(np.abs(cpu_action-action_ref)))
                    if first_action_max_abs > 1e-6:
                        raise RuntimeError(
                            f'M457_INITIAL_ACTION_MISMATCH: {first_action_max_abs}')
                cpu_obs,cpu_done,cpu_terminal_reasons=cpu_step(cpu_data,info,cpu_action,command_np)
                cpu_frame=cpu_measure(cpu_data)
                cpu_frame.update({'step':step_idx,'terminated':bool(cpu_done)})
                cpu_frames.append(cpu_frame)
                cpu_alive=not cpu_done
            if len(cpu_frames)==len(mjx_frames)==step_idx:
                mq=np.asarray(jax.device_get(state.data.qpos),dtype=np.float64)
                mv=np.asarray(jax.device_get(state.data.qvel),dtype=np.float64)
                dq=np.abs(mq-cpu_data.qpos)
                dv=np.abs(mv-cpu_data.qvel)
                mo=np.asarray(jax.device_get(state.obs['state']),dtype=np.float64)
                co=np.asarray(jax.device_get(cpu_obs['state']),dtype=np.float64)
                if mo.shape!=(103,) or co.shape!=(103,):
                    raise RuntimeError('M457_DIAG_ACTOR_OBSERVATION_SHAPE_CHANGED')
                do=np.abs(mo-co)
                paired_errors.append({
                    'step':step_idx,
                    'max_abs_qpos':float(dq.max()),
                    'qpos_max_index':int(dq.argmax()),
                    'root_xyz_max_abs':float(dq[:3].max()),
                    'root_quat_max_abs':float(dq[3:7].max()),
                    'joint_qpos_max_abs':float(dq[7:].max()),
                    'max_abs_qvel':float(dv.max()),
                    'qvel_max_index':int(dv.argmax()),
                    'base_qvel_max_abs':float(dv[:6].max()),
                    'joint_qvel_max_abs':float(dv[6:].max()),
                    'root_z_abs':float(dq[2]),
                    'actor_obs_max_abs':float(do.max()),
                    'actor_obs_linvel_max_abs':float(do[0:3].max()),
                    'actor_obs_gyro_max_abs':float(do[3:6].max()),
                    'actor_obs_gravity_max_abs':float(do[6:9].max()),
                    'actor_obs_command_max_abs':float(do[9:12].max()),
                    'actor_obs_jointpos_max_abs':float(do[12:41].max()),
                    'actor_obs_jointvel_max_abs':float(do[41:70].max()),
                    'actor_obs_lastact_max_abs':float(do[70:99].max()),
                    'actor_obs_phase_max_abs':float(do[99:103].max()),
                })
            if (step_idx==1 or step_idx%100==0 or
                    (not ref_alive and len(mjx_frames)==step_idx) or
                    (not cpu_alive and len(cpu_frames)==step_idx)):
                print(f'M457_PROGRESS {case} step={step_idx} mjx_alive={ref_alive} cpu_alive={cpu_alive} '
                      f'root_z_mjx={mjx_frames[-1]["root_z_m"]:.4f} '
                      f'root_z_cpu={cpu_frames[-1]["root_z_m"]:.4f}',flush=True)
            if not ref_alive and not cpu_alive:
                break

        def metrics(frames):
            xy_err=[math.hypot(f['vx_mps']-command_np[0],f['vy_mps']-command_np[1]) for f in frames]
            yaw_err=[f['yaw_rate_radps']-command_np[2] for f in frames]
            return {'steps':len(frames),
                    'terminated':bool(frames[-1]['terminated']),
                    'mean_vx_mps':average([f['vx_mps'] for f in frames]),
                    'mean_yaw_radps':average([f['yaw_rate_radps'] for f in frames]),
                    'xy_rmse_mps':rmse(xy_err),
                    'yaw_rmse_radps':rmse(yaw_err),
                    'root_z_min_m':min(f['root_z_m'] for f in frames)}

        outcome={'case':case,'command':list(COMMANDS[case]),
                 'mjx':metrics(mjx_frames),'cpu':metrics(cpu_frames),
                 'cpu_terminal_reasons':cpu_terminal_reasons if not cpu_alive else [],
                 'first_action_max_abs_difference':first_action_max_abs,
                 'initial_sensor_max_abs_difference':initial_sensor_error,
                 'paired_action_errors':paired_action_errors,
                 'paired_error_first_step':paired_errors[0] if paired_errors else None,
                 'paired_error_last_step':paired_errors[-1] if paired_errors else None,
                 'paired_error_sample_count':len(paired_errors),
                 'paired_error_samples':paired_errors,
                 'both_complete_horizon':(
                     len(mjx_frames)==limit and len(cpu_frames)==limit and ref_alive and cpu_alive),
                 'mjx_frames':mjx_frames,'cpu_frames':cpu_frames}
        runs.append(outcome)
        print(f'M457_RESULT case={case} '
              f'mjx_steps={len(mjx_frames)} cpu_steps={len(cpu_frames)} '
              f'mjx_done={not ref_alive} cpu_done={not cpu_alive} '
              f'cpu_reasons={outcome["cpu_terminal_reasons"]}',flush=True)

    OUTPUT.mkdir(parents=True,exist_ok=True)
    output_path=OUTPUT/('m457_'+args.mode+f'_seed{args.seed}_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')
    payload={
        'status':'M457_SMOKE_COMPLETED' if args.mode=='smoke' else 'M457_SIM2SIM_DIAGNOSTIC_COMPLETED',
        'experiment_class':'NOMINAL_FIXED_COMMAND_SIM_TO_SIM_NOT_PAPER_BENCHMARK',
        'source':'pinned MuJoCo Playground 2025 G1JoystickFlatTerrain',
        'training_repeated':False,'ppo_or_reward_modified':False,
        'source_model_modified':False,'domain_randomization':False,
        'push_enabled':False,'cpu_physics_integrator':'mujoco.mj_step',
        'cpu_observation_source':'native MjData sensors; source-order G1 actor/critic observation',
        'policy_inference_backend':'JAX CUDA (NOT pure CPU policy inference)',
        'eval_seed':args.seed,'train_seed':1,'checkpoint_step':FINAL_STEP,
        'checkpoint_sha256':ctx['checkpoint_sha256'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ctrl_dt_s':float(env.dt),'sim_dt_s':float(model.opt.timestep),
        'n_substeps':n_substeps,'max_steps':limit,'cases':cases,'results':runs,
        'limits':[ 'Both simulators share exact model and reset state; only physics engine differs.',
                   'Native CPU observation manually mirrors pinned source; still requires numerical parity validation.',
                   'MJX reference and CPU policies run independently after reset; trajectories may diverge.',
                   'Fixed commands, push disabled: diagnostic only; no real-world deployment implied.' ],
    }
    with output_path.open('x',encoding='utf-8') as f:
        json.dump(payload,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print('M457_OUTPUT',output_path,flush=True)
    print('M457_DIAGNOSTIC_SENSOR_AND_OBSERVATION_ERRORS_SAVED',flush=True)
    print(payload['status'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inspect','smoke','run'),required=True)
    parser.add_argument('--seed',type=int,default=5186)
    parser.add_argument('--cases',default='stand,forward,yaw_left')
    parser.add_argument('--max-steps',type=int,default=400)
    args=parser.parse_args()
    cases=args.cases.split(',')
    if not cases or len(cases)!=len(set(cases)) or any(c not in COMMANDS for c in cases):
        parser.error('Invalid cases. Choose unique: '+','.join(COMMANDS))
    if not (1<=args.max_steps<=1000):
        parser.error('max-steps must be 1..1000')
    ctx=preflight(args.seed)
    if args.mode=='inspect':
        print('M457_CASES:',cases,'seed=',args.seed,'max_steps=',args.max_steps,flush=True)
        print('M457_INSPECT_PASS NO_GPU_ROLLOUT NO_TRAINING',flush=True)
        return
    run(args,ctx,cases)


if __name__=='__main__':
    main()
