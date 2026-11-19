#!/usr/bin/env python3
"""M4.37 — G1 gait kinematics audit (diagnostic, NOT paper evaluation).

Uses exactly the M4.35 deterministic-policy MJX evaluation loop plus read-only
kinematic sampling. Never edits the original MuJoCo Playground source, PPO,
reward, termination, model, or saved checkpoints. Source push is configured
at environment construction (ON/OFF), and a fixed command is injected into
both info and the actor/critic observations as in M4.35.

Measures floor-foot geom collision contacts, foot-site world velocity during
contact (slip PROXY, not contact-point slip), foot-site swing Z (clearance PROXY,
not lowest-point clearance), pelvis/torso tilt, joint angles, velocity tracking,
height and push timing. Uses initial 1000-step episode horizon.

Run from G1_PAPER_REPRO with the existing pinned project environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

from m435_push_ab_stability import CMD

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'm437_gait_kinematics'
EXPECTED_SHA = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
DEFAULT_STEPS = 1000
JOINTS = [f'{side}_{name}_joint' for side in ('left', 'right')
          for name in ('knee', 'hip_pitch', 'ankle_pitch')]
SIDES = ('left', 'right')


def safe_number(n):
    v = float(n)
    if not math.isfinite(v):
        raise RuntimeError(f'M437_NONFINITE_TELEMETRY: {v!r}')
    return v


def summary(values):
    if not values:
        return {'count': 0, 'mean': None, 'min': None, 'max': None, 'p95': None}
    ordered = sorted(float(x) for x in values)
    return {'count': len(ordered), 'mean': statistics.fmean(ordered),
            'min': ordered[0], 'max': ordered[-1],
            'p95': ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)]}


def episode_stats(frames, command, ground_z):
    if not frames:
        raise ValueError('M437_EMPTY_FRAMES')
    s = {'sample_count': len(frames), 'root_height_m':summary([f['root_z_m'] for f in frames]),
         'torso_roll_deg':summary([f['torso_roll_deg'] for f in frames]),
         'torso_pitch_deg':summary([f['torso_pitch_deg'] for f in frames]),
         'torso_tilt_deg':summary([f['torso_tilt_deg'] for f in frames]),
         'foot': {}, 'joints_rad': {},
         'mean_local_vel_x_mps':statistics.fmean(f['actual_body_velocity'][0] for f in frames),
         'mean_local_vel_y_mps':statistics.fmean(f['actual_body_velocity'][1] for f in frames),
         'mean_body_yaw_rate_radps':statistics.fmean(f['actual_body_yaw_rate_radps'] for f in frames)}
    s['xy_rmse_mps']=math.sqrt(statistics.fmean(
        (f['actual_body_velocity'][0]-command[0])**2 +
        (f['actual_body_velocity'][1]-command[1])**2 for f in frames))
    s['yaw_rmse_radps']=math.sqrt(statistics.fmean(
        (f['actual_body_yaw_rate_radps']-command[2])**2 for f in frames))

    s['gait_phase_unwrapped_progress_rad'] = {}
    for side_idx, side in enumerate(SIDES):
        phase=[f['source_phase_rad'][side_idx] for f in frames]
        diffs=[(b-a+math.pi)%(2*math.pi)-math.pi for a,b in zip(phase,phase[1:])]
        s['gait_phase_unwrapped_progress_rad'][side]=sum(diffs)
    for side in SIDES:
        on = [f for f in frames if f['foot'][side]['floor_geom_contact']]
        off = [f for f in frames if not f['foot'][side]['floor_geom_contact']]
        on_speeds = [f['foot'][side]['site_world_xy_speed_mps'] for f in on]
        off_z = [f['foot'][side]['site_height_above_floor_plane_m'] for f in off]
        contacts = [bool(f['foot'][side]['floor_geom_contact']) for f in frames]
        touchdown_events = sum(not a and b for a,b in zip(contacts, contacts[1:]))
        liftoff_events = sum(a and not b for a,b in zip(contacts,contacts[1:]))
        landing_positions=[frames[i]['foot'][side]['site_world_xyz_m'][:2]
                           for i in range(1,len(frames)) if not contacts[i-1] and contacts[i]]
        same_foot_landing_distances=[math.dist(a,b)
                                     for a,b in zip(landing_positions,landing_positions[1:])]
        s['foot'][side] = {
            'geom_contact_fraction':len(on)/len(frames),
            'stance_site_xy_speed_mps':summary(on_speeds),
            'stance_site_speed_gt_0p1_fraction':sum(v>.1 for v in on_speeds)/len(on_speeds) if on_speeds else None,
            'swing_site_height_m':summary(off_z),
            'swing_site_height_gt_0p05_fraction':sum(z>.05 for z in off_z)/len(off_z) if off_z else None,
            'touchdown_count':touchdown_events,
            'liftoff_count':liftoff_events,
            'successive_touchdown_site_xy_displacement_m':summary(same_foot_landing_distances),
            'note':'Foot-site velocity during foot-geom floor-contact is a slip proxy; not an exact contact patch speed.',
        }
    for name in JOINTS:
        s['joints_rad'][name] = summary([f['joint_positions_rad'][name] for f in frames])
    knee_l = s['joints_rad']['left_knee_joint']
    knee_r = s['joints_rad']['right_knee_joint']
    s['knee_peak_to_peak_rad'] = {
        'left':knee_l['max']-knee_l['min'],
        'right':knee_r['max']-knee_r['min'],
        'absolute_difference':abs((knee_l['max']-knee_l['min']) - (knee_r['max']-knee_r['min']))}
    return s


def inspect(args):
    from m433_visual_validation import preflight
    ctx=preflight(args.train_seed,args.seed,'deterministic')
    if args.train_seed==1 and ctx['checkpoint_sha256'] != EXPECTED_SHA:
        raise RuntimeError('M437_SEED1_CHECKPOINT_HASH_CHANGED')
    print('M437_CHECKPOINT_OK',ctx['checkpoint_sha256'], flush=True)
    print('M437_CASES',args.cases,'PUSH',args.push,'SEED',args.seed,flush=True)
    print('M437_INSPECT_PASS NO_GPU_ROLLOUT NO_TRAINING',flush=True)
    return ctx


def run(args, ctx):
    # Compat shim MUST run before importing any JAX module.
    from m434_manual_command_viewer import patch_obs_and_info, validate_initial_observation
    from paper_config import BASE, ENV
    from train import check_source
    check_source()
    sys.path.insert(0, str(BASE / 'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('NVCC_COMPAT',apply(),flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import locomotion
    from mujoco_playground._src.collision import geoms_colliding
    from m426_heldout_det_stoch import restore_params

    if jax.default_backend() != 'gpu':
        raise RuntimeError('M437_GPU_BACKEND_REQUIRED')
    default_cfg=locomotion.get_default_config(ENV)
    if bool(default_cfg.push_config.enable) is not True:
        raise RuntimeError('M437_SOURCE_DEFAULT_PUSH_CHANGED')
    baseline=locomotion.load(ENV, config=default_cfg)
    test_state=jax.jit(baseline.reset)(jax.random.PRNGKey(999))
    validate_initial_observation(test_state,jnp)
    network=ppo_networks.make_ppo_networks(
        observation_size=jax.tree_util.tree_map(lambda x:x.shape,test_state.obs),
        action_size=baseline.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ctx['report']['scaled_config']['network_factory'])
    params=restore_params(ctx['checkpoint_path'].read_bytes(),
                          serialization,running_statistics,jax,jnp)
    policy=jax.jit(ppo_networks.make_inference_fn(network)(params,deterministic=True))
    parsed_cases=args.cases.split(',')
    push_settings = [False] if args.push=='off' else ([True] if args.push=='on' else [True,False])
    all_cases=[]
    for case in parsed_cases:
        for push_enabled in push_settings:
            cfg=locomotion.get_default_config(ENV)
            cfg.push_config.enable=push_enabled  # only physics setting toggled
            env=locomotion.load(ENV, config=cfg)
            model=env.mj_model
            dt=float(env.dt)
            if abs(dt-0.02)>1e-6:
                raise RuntimeError(f'M437_UNEXPECTED_DT: {dt}')
            ids=[int(v) for v in env._feet_site_id]
            geoms=[int(v) for v in env._feet_geom_id]
            floor_id=int(env._floor_geom_id)
            sensor_indices=env._foot_linvel_sensor_adr
            if len(ids)!=2 or len(geoms)!=2 or tuple(sensor_indices.shape)!=(2,3):
                raise RuntimeError('M437_FOOT_GEOMETRY_OR_SENSOR_LAYOUT_CHANGED')
            posadr={name:int(model.joint(name).qposadr) for name in JOINTS}
            ground_z=float(model.geom_pos[floor_id,2])
            imu_id=int(env._torso_imu_site_id)
            floor_name='floor'
            geom_names=[model.geom(v).name for v in geoms]
            site_names=[model.site(v).name for v in ids]
            # The source's array order is not assumed to be left/right.
            def side_of_name(name):
                n=name.lower()
                if n.startswith('left_') or n.startswith('l_'):
                    return 'left'
                if n.startswith('right_') or n.startswith('r_'):
                    return 'right'
                raise RuntimeError(f'M437_FOOT_SITE_SIDE_UNRESOLVED: {name}')
            side_order=[side_of_name(n) for n in site_names]
            geom_order=[side_of_name(n) for n in geom_names]
            if set(side_order)!=set(SIDES) or side_order != geom_order:
                raise RuntimeError(f'M437_FOOT_SITE_GEOM_ORDER_MISMATCH: sites={site_names} geoms={geom_names}')

            # Capture data at the start of each physics step, as done by M4.35.
            # geoms_colliding() matches the source G1 foot-floor contact helper.
            def diag_fn(state):
                data=state.data
                contact=jnp.stack([geoms_colliding(data,g,floor_id) for g in geoms])
                feet=data.site_xpos[jnp.asarray(ids,dtype=jnp.int32)]
                foot_vel=data.sensordata[sensor_indices]
                mat=data.site_xmat[imu_id]
                roll=jnp.arctan2(mat[2,1],mat[2,2])
                pitch=jnp.arctan2(-mat[2,0],jnp.sqrt(mat[2,1]**2+mat[2,2]**2))
                # Relative torso tilt from a gravity/up vector; yaw-invariant.
                tilt=jnp.arccos(jnp.clip(mat[2,2],-1.,1.))
                return {
                    'foot_contact':contact,
                    'foot_site_pos':feet,
                    'foot_site_linvel':foot_vel,
                    'joint_qpos':jnp.stack([data.qpos[posadr[name]] for name in JOINTS]),
                    'root_z':data.qpos[2],
                    'source_phase':state.info['phase'],
                    'torso_rpt':jnp.stack([roll,pitch,tilt]),
                    'actual_v':env.get_local_linvel(data,'pelvis')[:2],
                    'actual_yaw':env.get_gyro(data,'pelvis')[2],
                }
            diagnostic=jax.jit(diag_fn)
            reset=jax.jit(env.reset)
            step_fn=jax.jit(env.step)
            key=jax.random.PRNGKey(args.seed)
            key,reset_key=jax.random.split(key)
            state=reset(reset_key)
            validate_initial_observation(state,jnp)
            command=jnp.asarray(CMD[case],dtype=jnp.float32)
            frames=[]
            pushes=[]
            terminated=False
            print(f'M437_START case={case} push={push_enabled} seed={args.seed}',flush=True)
            for index in range(1,args.max_steps+1):
                state=patch_obs_and_info(state,command,jnp)
                measurements=jax.device_get(diagnostic(state))
                contact=np.asarray(measurements['foot_contact'],dtype=bool)
                site_pos=np.asarray(measurements['foot_site_pos'])
                foot_vel=np.asarray(measurements['foot_site_linvel'])
                angles=np.asarray(measurements['joint_qpos'])
                torso=np.asarray(measurements['torso_rpt'])
                vxvy=np.asarray(measurements['actual_v'])
                yaw=safe_number(measurements['actual_yaw'])
                root_z=safe_number(measurements['root_z'])
                frame={'step':index,'time_s':round((index-1)*dt,5),
                       'command':list(CMD[case]),'actual_body_velocity':[safe_number(x) for x in vxvy],
                       'actual_body_yaw_rate_radps':yaw,'root_z_m':root_z,
                       'source_phase_rad':[safe_number(x) for x in np.asarray(measurements['source_phase'])],
                       'torso_roll_deg':math.degrees(safe_number(torso[0])),
                       'torso_pitch_deg':math.degrees(safe_number(torso[1])),
                       'torso_tilt_deg':math.degrees(safe_number(torso[2])),
                       'joint_positions_rad':{name:safe_number(angles[k]) for k,name in enumerate(JOINTS)},
                       'foot':{side:{'floor_geom_contact':bool(contact[k]),
                                     'site_world_xyz_m':[safe_number(x) for x in site_pos[k]],
                                     'site_height_above_floor_plane_m':safe_number(site_pos[k,2]-ground_z),
                                     'site_world_xy_speed_mps':math.hypot(
                                         safe_number(foot_vel[k,0]),safe_number(foot_vel[k,1]))}
                               for k,side in enumerate(side_order)}}
                key,action_key=jax.random.split(key)
                action,_=policy(state.obs,action_key)
                if not bool(jnp.all(jnp.isfinite(action))):
                    raise RuntimeError(f'M437_NONFINITE_ACTION: case={case} step={index}')
                state=step_fn(state,action)
                terminated=bool(state.done)
                push_xy=[safe_number(v) for v in jax.device_get(state.info['push'])]
                if math.hypot(*push_xy)>0.01:
                    pushes.append({'step':index,'direction_xy':push_xy})
                frame['done_after_step']=terminated
                frames.append(frame)
                if index%200==0 or terminated:
                    print('M437_PROGRESS',case,'push',push_enabled,'step',index,
                          'z',round(root_z,3),'done',terminated,flush=True)
                if terminated:
                    break
            if not push_enabled and pushes:
                raise RuntimeError('M437_UNEXPECTED_PUSH_WHEN_OFF')
            result={'case':case,'command':list(CMD[case]),'push_enabled':push_enabled,
                    'steps':len(frames),'terminated':terminated,
                    'completed_1000_steps':len(frames)==1000 and not terminated,
                    'push_events':pushes,
                    'summary':episode_stats(frames,CMD[case],ground_z),
                    'source_geometry':{'floor_geom':floor_name,'floor_z_m':ground_z,
                                       'feet_geoms':geom_names,'feet_sites':site_names,
                                       'joint_names':JOINTS},
                    'samples':frames}
            all_cases.append(result)
            print(f'M437_RESULT {case} push={push_enabled} steps={len(frames)} '
                  f'contacts_L={result["summary"]["foot"]["left"]["geom_contact_fraction"]:.3f} '
                  f'contacts_R={result["summary"]["foot"]["right"]["geom_contact_fraction"]:.3f}',flush=True)

    OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/('m437_gait_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')
    payload={'status':'M437_GAIT_KINEMATICS_COMPLETED',
             'experiment_class':'READ_ONLY_KINEMATICS_DIAGNOSTIC_NOT_PAPER_BENCHMARK',
             'environment':ENV,'checkpoint_step':401080320,
             'train_seed':args.train_seed,'reset_seed':args.seed,
             'checkpoint_sha256':ctx['checkpoint_sha256'],
             'max_steps':args.max_steps,'no_training':True,
             'policy_mode':'deterministic', 'domain_randomization':False,
             'push_toggle_only_at_env_creation':True,
             'joint_convention':'model.joint(name).qposadr; signed radians in MuJoCo joint coordinates',
             'contacts_note':'Source G1 geoms_colliding(foot_geom, floor_geom) only; other support contacts omitted.',
             'slip_note':'foot-site XY world velocity while foot-geom floor contact is true. NOT true contact-point slip; differs from source feet_slip reward formulation.',
             'clearance_note':'foot-site height above plane geom_position.z during non-contact. NOT lowest foot surface clearance.',
             'tilt_note':'torso IMU site orientation Euler roll/pitch and absolute tilt relative to global Z; site alignment may not equal anatomical torso angles.',
             'phase_note':'Source phase advances even for zero commands in this pinned revision; recorded phases are diagnostic only, no gait phase override.',
             'rollouts':all_cases}
    with out.open('x',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print('M437_GAIT_KINEMATICS_COMPLETED',flush=True)
    print('OUTPUT:',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('inspect','run'))
    parser.add_argument('--train-seed',type=int,choices=(0,1),default=1)
    parser.add_argument('--seed',type=int,default=5186)
    parser.add_argument('--cases',default='stand,backward,yaw_right')
    parser.add_argument('--push',choices=('on','off','both'),default='off')
    parser.add_argument('--max-steps',type=int,default=1000)
    args=parser.parse_args()
    cases=args.cases.split(',')
    if len(cases)!=len(set(cases)) or not cases or any(c not in CMD for c in cases):
        parser.error('Invalid cases: '+','.join(CMD))
    if not (10<=args.max_steps<=1000):
        parser.error('max-steps must be 10..1000')
    ctx=inspect(args)
    if args.mode=='run':
        run(args,ctx)

if __name__=='__main__':
    main()
