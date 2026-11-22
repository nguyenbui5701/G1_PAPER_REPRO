#!/usr/bin/env python3
"""M4.63 Gate 3: matched C/P controlled checkpoint evaluation on pinned MJX GPU.

- Two frozen policies: C=401M, P=M4.63 push-exposure smoke.
- NO training, randomizer, source modification, installing dependencies, or CPU sim.
- Fixed task commands: stand, straight, turn_left, turn_right.
- Push OFF baseline + push ON fork from SAME full prepush JAX state per policy/seed.
- Push is instantaneous qvel-world-XY delta (m/s), not Newton force.
- Commands are explicitly synchronized to source observation indices 9:12.
- Original source random push disabled in all evaluations.

This is a diagnostic evaluation, NOT a paper-exact benchmark, and does not
measure native MuJoCo CPU sim-to-sim agreement. It intentionally reports a
separate recovery definition and stores per-run outcomes and prefix audit.
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PIN = '9bcbe736177472ba35d60773e3030a36e37962b7'
BASE_SHA = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
TRAIN_SHA = 'e20613d42bf0868a13003e0312998816cc8d3593bb5ccaf9f1d070fd09c44bd7'
H_SHA = 'a3dd9f811364820197f73e29926c7f1565ba25fb6103d4e094703da7de6b6dcd'
PROFILES = {
    'stand': (0.0,0.0,0.0),
    'straight': (0.4,0.0,0.0),
    'forward_slow': (0.2,0.0,0.0),
    'forward_fast': (0.6,0.0,0.0),
    'strafe': (0.0,0.35,0.0),
    'turn_left': (0.4,0.0,0.8),
    'turn_right': (0.4,0.0,-0.8),
}
DIRECTIONS = {
    'left': (0.0,1.0), 'right': (0.0,-1.0),
    'forward': (1.0,0.0), 'backward': (-1.0,0.0),
}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read_m463_smoke_report(path, checkpoint):
    """Fail closed: only the verified M4.63 easy smoke checkpoint is accepted."""
    report=json.loads(Path(path).read_text(encoding='utf8'))
    mode = report.get('mode')

    modes = {
        'smoke': ('M463_COVERAGE_SMOKE_PASS', 262144, 1),
        'probe5': ('M463_COVERAGE_PROBE5_PASS', 1310720, 5),
        'probe2': ('M463_COVERAGE_PROBE2_PASS', 524288, 2),
        'probe15': ('M463_COVERAGE_PROBE15_PASS', 3932160, 15),
    }

    if mode not in modes:
        raise RuntimeError('M463_GATE3_UNSUPPORTED_TRAINING_MODE')

    expected_status, expected_steps, expected_rollouts = modes[mode]
    checks={
        'status': report.get('status') == expected_status,
        'stage': report.get('stage') in ('easy', 'easy_repeat', 'easy_repeat_delayed'),
        'delayed_stage_config': (
        report.get('stage') != 'easy_repeat_delayed'
        or (
            report.get('stage_magnitude_range_mps') == [0.2, 0.6]
            and report.get('stage_repeat_probability') == 0.25
            and report.get('stage_repeat_gap_step_range') == [90, 115]
            and report.get('hardware_scaled_config', {}).get(
                'num_resets_per_eval'
            ) == 0
            and report.get('coverage', {}).get('counts', {}).get(
                'training/m463_repeat_per_rollout', 0
            ) > 0
        )
    ),
        'mode': report.get('mode') == mode,
        'commit':report.get('source_commit')==PIN,
        'base_sha':report.get('source_checkpoint_sha256')==BASE_SHA,
        'steps': report.get('steps_observed') == expected_steps,
        'rollouts': report.get('rollouts_target') == expected_rollouts,
        'train_seed':report.get('training_seed')==1,
        'warmstart_kind':report.get('trained_from')=='401M_WEIGHTS_ONLY_ADAM_RESTART',
        'exposure':report.get('coverage_gate_pass') is True,
        'memory':report.get('memory_gate_pass') is True,
        'weights_finite':report.get('weights_finite') is True,
        'callback':report.get('callback_vs_final_equal') is True,
        'tensor_count':report.get('final_tensor_count')==25,
        'train_push_on':report.get('source_push_disabled_in_custom_train_env') is True,
        'checkpoint_entries': (
            len(report.get('checkpoints') or [])
            == (1 if mode == 'probe2' else 0)
        ),      

        'checkpoint_schedule': [
            c.get('steps')
            for c in (report.get('checkpoints') or [])
        ] == (
            [expected_steps]
            if mode == 'probe2'
            else [
                262144 * i
                for i in range(1, expected_rollouts + 1)
            ]
        ),
        'repeat_stage_config': (
            report.get('stage') != 'easy_repeat'
            or (
                report.get('stage_magnitude_range_mps') == [0.2, 0.6]
                and report.get('stage_repeat_probability') == 0.25
                and report.get('coverage', {}).get('counts', {}).get(
                    'training/m463_repeat_per_rollout', 0
                ) > 0
                and report.get('hardware_scaled_config', {}).get(
                    'num_resets_per_eval'
                ) == (1 if mode == 'probe2' else 0)
            )
        ),
        'probe5_no_host_reset': (
            mode != 'probe5'
            or report.get('hardware_scaled_config', {}).get('num_resets_per_eval') == 0
        ),

        'probe5_t0_gate': (
            mode != 'probe5'
            or report.get('m465_t0_gate', {}).get('recovery_not_proven') is True
        ),
        'probe2_H_provenance': (
            mode != 'probe2' or (
                report.get('stage') == 'easy_repeat'
                and report.get('training_performed') is True
                and report.get('hardware_scaled_config', {}).get('num_evals') == 2
                and report.get('hardware_scaled_config', {}).get('num_resets_per_eval') == 1
                and report.get('hardware_scaled_config', {}).get('num_envs') == 8192
                and report.get('hardware_scaled_config', {}).get('unroll_length') == 32
                and report.get('hardware_scaled_config', {}).get('num_timesteps') == 524288
                and report.get('instrumented_brax', {}).get('original_brax_sha256') == TRAIN_SHA
                and report.get('instrumented_brax', {}).get('instrumented_sha256')
                    == '5a06e3a0b22647890e5ff906365cb07ff4d94f39b721716a0a3c840454ea6083'
                and len(report.get('checkpoints') or []) == 1
                and report['checkpoints'][0].get('sha256') == H_SHA
                and report.get('coverage', {}).get('counts', {}).get(
                    'training/m463_repeat_stand_right_support_gap_30_40_per_rollout', 0
                ) > 0
            )
        ),
    }
    if report.get('stage') == 'easy_mixed_time':
        checks.update({
            'status': (
                report.get('status')
                == 'M463_COVERAGE_PROBE15_PASS'
            ),
            'stage': report.get('stage') == 'easy_mixed_time',
            'mode': report.get('mode') == 'probe15',
            'steps': report.get('steps_observed') == 3932160,
            'rollouts': report.get('rollouts_target') == 15,
            'training_performed': (
                report.get('training_performed') is True
            ),
            'mixed_time_config': (
                report.get('first_push_windows_steps')
                == [[12, 24], [200, 219]]
                and report.get('first_push_late_probability') == 0.5
                and report.get('stage_repeat_probability') == 0.0
                and report.get('stage_magnitude_range_mps')
                == [0.2, 0.6]
                and report.get('probability_nominal_episode') == 0.2
                and report.get(
                    'hardware_scaled_config', {}
                ).get('num_resets_per_eval') == 0
            ),
        })
    if not all(checks.values()):
        raise RuntimeError('M463_GATE3_INVALID_SMOKE_REPORT '+str({k:v for k,v in checks.items() if not v}))
    ck=report['checkpoints'][-1]
    expected_steps = (
        3932160
        if report.get('stage') == 'easy_mixed_time'
        else report['steps_observed']
    )
    if (ck.get('steps')!=expected_steps or ck.get('file')!=Path(checkpoint).name
            or ck.get('sha256')!=sha256(checkpoint)):
        raise RuntimeError('M463_GATE3_SMOKE_CHECKPOINT_SHA_OR_NAME_MISMATCH')
    cov=report.get('coverage') or {}
    if not cov.get('passed') or cov.get('push_events_observed',0)<=0:
        raise RuntimeError('M463_GATE3_PUSH_EXPOSURE_NOT_VERIFIED')
    return report


def parse_items(text, allowed, name):
    items=[x.strip() for x in text.split(',') if x.strip()]
    if not items or len(items)!=len(set(items)) or any(x not in allowed for x in items):
        raise ValueError(f'Invalid --{name}: {text} ; allowed {list(allowed)}')
    return items


def new_body_delta(direction, heading, magnitude):
    bx,by=DIRECTIONS[direction]
    c,s=math.cos(heading),math.sin(heading)
    return (magnitude*(c*bx-s*by),magnitude*(s*bx+c*by))


def yaw_from_root_quat(q):
    w,x,y,z=map(float,q)
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def compute_recovery(pre_frames,post_frames,window=100,consecutive=10):
    """Separate, explicitly defined band-recovery proxy. Not original M461 metric."""
    import statistics
    if len(pre_frames)<50:return {'recovered':None,'reason':'insufficient_prepush_baseline'}
    keys=('vx','vy','yaw_rate','height','tilt_deg')
    median={k:statistics.median(f[k] for f in pre_frames[-50:]) for k in keys}
    tolerances={'vx':0.1,'vy':0.1,'yaw_rate':0.1,'height':0.05,'tilt_deg':5.0}
    count=0
    for i,fr in enumerate(post_frames[:window]):
        if all(abs(fr[k]-median[k])<=tolerances[k] for k in keys):
            count+=1
            if count>=consecutive:
                return {'recovered':True,'recovery_s':(i+1)*0.02,'band_center':median,'band_tolerance':tolerances}
        else:count=0
    return {'recovered':False,'recovery_s':None,'band_center':median,'band_tolerance':tolerances}


def aggregate_frames(frames, terminated, steps, max_steps, push_step=None):

    # Không có dữ liệu frames
    if not frames:
        return {
            'steps': steps,
            'terminated': terminated,
            'completed': False,
            'n_samples': 0
        }

    import statistics

    def rmse(key):
        return math.sqrt(
            statistics.mean(fr[key] ** 2 for fr in frames)
        )

    # Các chỉ số đánh giá gốc - giữ nguyên
    out = {
        'steps': steps,
        'terminated': terminated,
        'completed': not terminated and steps == max_steps,
        'n_samples': len(frames),

        'xy_tracking_rmse_mps': rmse('xy_err'),
        'yaw_tracking_rmse_radps': rmse('yaw_err'),

        'max_tilt_deg': max(
            fr['tilt_deg'] for fr in frames
        ),

        'min_root_height_m': min(
            fr['height'] for fr in frames
        ),

        'termination_within_2s': (
            terminated
            and push_step is not None
            and steps - push_step + 1 <= 100
        ),

        'termination_within_5s': (
            terminated
            and push_step is not None
            and steps - push_step + 1 <= 250
        )
    }

    # M4.70-B: Phân tích nguyên nhân termination
    if terminated:

        flags = (
            'cause_inverted_torso',
            'cause_foot_foot',
            'cause_left_foot_right_shin',
            'cause_right_foot_left_shin',
            'cause_nan_qpos',
            'cause_nan_qvel',
        )

        # Lưu nguyên nhân tại bước cuối
        out['terminal_source_flags'] = {
            k: bool(frames[-1][k])
            for k in flags
        }

        # Kiểm tra tính nhất quán với source
        if not any(out['terminal_source_flags'].values()):
            raise RuntimeError(
                'M470_TERMINATED_WITH_NO_SOURCE_FLAG'
            )

        # Lưu các chỉ số 25 bước cuối
        keys = (
            'height',
            'tilt_deg',
            'vx',
            'vy',
            'yaw_rate',
            'reward_total',
            'reward_tracking_lin',
            'reward_tracking_ang',
            'reward_orientation',
            'reward_termination',
        )

        out['preterminal_last_25'] = [
            {k: frame[k] for k in keys}
            for frame in frames[-25:]
        ]

                # M4.70-C: bounded foot-geometry trace
        foot_keys = (
            'height',
            'tilt_deg',
            'vx',
            'vy',
            'yaw_rate',
            'cause_foot_foot',
            'reward_total',
            'reward_tracking_lin',
            'reward_tracking_ang',
            'reward_orientation',
            'reward_termination',

            'lf_geom_x_m',
            'lf_geom_y_m',
            'lf_geom_z_m',
            'rf_geom_x_m',
            'rf_geom_y_m',
            'rf_geom_z_m',

            'foot_center_xy_dist_m',
            'foot_center_3d_dist_m',
            'foot_lateral_sep_body_m',
            'foot_forward_sep_body_m',

            'lf_site_z_m',
            'rf_site_z_m',
            'lf_ground_contact',
            'rf_ground_contact',

            'phase_left_post_rad',
            'phase_right_post_rad',

            'foot_pair_present',
            'foot_pair_signed_dist_m',
        )

        out['precollision_foot_last_50'] = [
            {k: frame[k] for k in foot_keys}
            for frame in frames[-50:]
        ]

        out['precollision_first_step'] = (
            steps - min(len(frames), 50) + 1
        )
        # M4.70-D: same-time geometry comparison for single/double push
    if push_step is not None:
        start_step = steps - len(frames) + 1

        compare_keys = (
            'height',
            'vx',
            'vy',
            'yaw_rate',
            'foot_center_xy_dist_m',
            'foot_lateral_sep_body_m',
            'foot_forward_sep_body_m',
            'lf_site_z_m',
            'rf_site_z_m',
            'lf_ground_contact',
            'rf_ground_contact',
            'phase_left_post_rad',
            'phase_right_post_rad',
            'reward_tracking_lin',
            'cause_foot_foot',
            'foot_pair_signed_dist_m',
        )

        window = [
            {
                'step': start_step + i,
                **{k: frame[k] for k in compare_keys}
            }
            for i, frame in enumerate(frames)
            if 330 <= start_step + i <= 385
        ]

        if window:
            out['second_event_foot_window_330_385'] = window
    # Chỉ return một lần ở cuối
    return out



def matched_off_recovery(off_post,pushed,window_steps=100,consecutive=10,dt_s=0.02):
    """Recovery to contemporaneous OFF trajectory, not a stationary median.

    Paired within a single policy, same full pre-push MJX snapshot. One metric
    is not a substitute for survival; a later termination is reported separately.
    This metric is *not* the same as the existing prepush-median recovery proxy.
    """
    if window_steps<consecutive or consecutive<1:
        raise ValueError('window_steps must be >= consecutive >= 1')
    keys=('vx','vy','yaw_rate','height','tilt_deg')
    tolerances={'vx':0.1,'vy':0.1,'yaw_rate':0.1,'height':0.05,'tilt_deg':5.0}
    n=min(len(off_post),len(pushed),window_steps)
    maximum={k:0.0 for k in keys}
    count=0
    when=None
    for i in range(n):
        diffs={k:abs(float(pushed[i][k])-float(off_post[i][k])) for k in keys}
        if any(not math.isfinite(v) for v in diffs.values()):
            raise ValueError('nonfinite matched recovery data')
        for k in keys: maximum[k]=max(maximum[k],diffs[k])
        if all(diffs[k]<=tolerances[k] for k in keys):
            count+=1
            if count>=consecutive and when is None:
                when=(i+1)*dt_s
        else:
            count=0
    return {'recovered':when is not None,'recovery_s':when,
            'comparison':'ON minus contemporaneous OFF of same policy',
            'n_common_steps':n,'window_steps':window_steps,'consecutive':consecutive,
            'bands':tolerances,'max_abs_deviation_in_common_window':maximum}


def fixed_window_tracking(off_post,pushed,window_steps=100):
    """Compute fair fixed-horizon RMSE only if both traces reach window length."""
    if len(off_post)<window_steps or len(pushed)<window_steps:
        return {'comparable':False,'reason':'ON_or_OFF_terminated_before_window',
                'window_steps':window_steps,
                'off_available':len(off_post),'on_available':len(pushed)}
    def rmse(trace,key):
        return math.sqrt(sum(float(f[key])**2 for f in trace[:window_steps])/window_steps)
    return {'comparable':True,'window_steps':window_steps,
            'on_xy_rmse_mps':rmse(pushed,'xy_err'),
            'off_xy_rmse_mps':rmse(off_post,'xy_err'),
            'on_yaw_rmse_radps':rmse(pushed,'yaw_err'),
            'off_yaw_rmse_radps':rmse(off_post,'yaw_err')}

def run(args):
    if args.output.exists():raise FileExistsError('Refuse to overwrite output: '+str(args.output))
    # Use the same pinned runner environment; apply compatibility BEFORE JAX import.
    from train import check_source, build_paper_config
    from paper_config import BASE, ENV, PIN as SOURCE_PIN
    check_source()
    if SOURCE_PIN!=PIN:raise RuntimeError('M462_SOURCE_PIN_MISMATCH')
    sys.path.insert(0,str(BASE/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('M463_GATE3_NVCC',apply(),flush=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks, train as ppo
    from mujoco_playground import locomotion
    from mujoco_playground._src import collision
    if jax.default_backend()!='gpu':raise RuntimeError('M463_GATE3_REQUIRES_GPU')
    if sha256(inspect.getfile(ppo.train))!=TRAIN_SHA:raise RuntimeError('M462_PINNED_BRAX_SOURCE_MISMATCH')
    paper=build_paper_config()
    if sha256(args.checkpoint_c)!=BASE_SHA:raise RuntimeError('M462_FROZEN_401M_SHA_MISMATCH')
    report_p=read_m463_smoke_report(args.report_p,args.checkpoint_p)
    if report_p.get('instrumented_brax',{}).get('original_brax_sha256')!=TRAIN_SHA:
        raise RuntimeError('M463_GATE3_TRAIN_BRAX_MISMATCH')

    cfg=locomotion.get_default_config(ENV)
    if not bool(cfg.push_config.enable) or list(cfg.push_config.interval_range)!=[5.0,10.0] or list(cfg.push_config.magnitude_range)!=[0.1,2.0]:
        raise RuntimeError('M462_EVAL_PINNED_SOURCE_PUSH_CONFIG_CHANGED')
    if float(cfg.reward_config.scales.base_height)!=0.:
        raise RuntimeError('M462_EVAL_SOURCE_REWARD_CONFIG_CHANGED')
    cfg.push_config.enable=False
    env=locomotion.load(ENV,config=cfg)
    if env.action_size!=29:raise RuntimeError('M462_ACTION_DIM_CHANGED')
    if abs(float(env.dt)-0.02)>1e-6:raise RuntimeError('M462_CTRL_DT_CHANGED')
    obs_shape={'state':(103,),'privileged_state':(216,)}
    ppo_nets=networks.make_ppo_networks(obs_shape,env.action_size,
                preprocess_observations_fn=running_statistics.normalize,
                **dict(paper.network_factory))
    norm_template=running_statistics.init_state({k:jnp.zeros(s,dtype=jnp.float32) for k,s in obs_shape.items()})
    actor_template=ppo_nets.policy_network.init(jax.random.PRNGKey(11))
    critic_template=ppo_nets.value_network.init(jax.random.PRNGKey(17))
    def load_policy(path):
        from m462_typed_orbax_stage import checkpoint_parts
        n,p,v=checkpoint_parts(serialization.msgpack_restore(Path(path).read_bytes()))
        norm=serialization.from_state_dict(norm_template,n)
        actor=serialization.from_state_dict(actor_template,p)
        critic=serialization.from_state_dict(critic_template,v)
        leaves=jax.tree_util.tree_leaves((norm,actor,critic))
        if len(leaves)!=25:raise RuntimeError('M462_WRONG_CHECKPOINT_LEAF_COUNT')
        if any(not np.all(np.isfinite(np.asarray(jax.device_get(x)))) for x in leaves):
            raise RuntimeError('M462_NONFINITE_POLICY')
        inf=networks.make_inference_fn(ppo_nets)((norm,actor),deterministic=True)
        return inf
    policies={name:load_policy(path) for name,path in (
        ('C',args.checkpoint_c),('P',args.checkpoint_p))}

    results=[]
    for seed in args.seeds:
      for profile in args.profiles:
        command=jnp.asarray(PROFILES[profile],dtype=jnp.float32)
        for name,policy in policies.items():
            # JIT step/diagnostics; a single MJX state (no vectorized envs) is active at a time.
            def one_step(state):
                info=dict(state.info);info['command']=command
                obs=dict(state.obs)
                # Pinned G1 actor state: local v(3), gyro(3), projected g(3), command(3).
                obs['state']=obs['state'].at[9:12].set(command)
                obs['privileged_state']=obs['privileged_state'].at[9:12].set(command)
                state=state.replace(info=info,obs=obs)
                action,_=policy(state.obs,jax.random.PRNGKey(0))
                next_state=env.step(state,action)
                d = next_state.data

                ff = env.get_gravity(d, 'torso')[-1] < 0.0

                cff = collision.geoms_colliding(
                    d, env._right_foot_geom_id, env._left_foot_geom_id
                )

                cfs1 = collision.geoms_colliding(
                    d, env._left_foot_geom_id, env._right_shin_geom_id
                )

                cfs2 = collision.geoms_colliding(
                    d, env._right_foot_geom_id, env._left_shin_geom_id
                )

                nan_qpos = jnp.isnan(d.qpos).any()
                nan_qvel = jnp.isnan(d.qvel).any()
                vel=env.get_local_linvel(next_state.data,'pelvis')
                yaw_rate=env.get_gyro(next_state.data,'pelvis')[2]
                gravity_z=env.get_gravity(next_state.data,'torso')[-1]
                diag={'vx':vel[0],'vy':vel[1],'yaw_rate':yaw_rate,
                      'height':next_state.data.qpos[2],
                      'tilt_deg':jnp.degrees(jnp.arccos(jnp.clip(gravity_z,-1.,1.))),
                      'xy_err':jnp.linalg.norm(vel[:2]-command[:2]),
                      'yaw_err':jnp.abs(yaw_rate-command[2]),
                      'done': next_state.done,

                    'cause_fall': ff.astype(jnp.int32),
                    'cause_foot_foot': cff.astype(jnp.int32),
                    'cause_left_foot_right_shin': cfs1.astype(jnp.int32),
                    'cause_right_foot_left_shin': cfs2.astype(jnp.int32),
                    'cause_nan_qpos': nan_qpos.astype(jnp.int32),
                    'cause_nan_qvel': nan_qvel.astype(jnp.int32),
                    
                    'reward_total': next_state.reward,
                    'reward_tracking_lin':
                        next_state.metrics['reward/tracking_lin_vel'],
                    'reward_tracking_ang':
                        next_state.metrics['reward/tracking_ang_vel'],
                    'reward_orientation':
                        next_state.metrics['reward/orientation'],
                    'reward_termination':
                        next_state.metrics['reward/termination'],}
                # M4.70-B — Source-exact termination diagnostics
                d = next_state.data

                diag.update({
                    'cause_inverted_torso': (
                        env.get_gravity(d, 'torso')[-1] < 0.0
                    ).astype(jnp.float32),

                    'cause_foot_foot': collision.geoms_colliding(
                        d,
                        env._right_foot_geom_id,
                        env._left_foot_geom_id
                    ).astype(jnp.float32),

                    'cause_left_foot_right_shin': collision.geoms_colliding(
                        d,
                        env._left_foot_geom_id,
                        env._right_shin_geom_id
                    ).astype(jnp.float32),

                    'cause_right_foot_left_shin': collision.geoms_colliding(
                        d,
                        env._right_foot_geom_id,
                        env._left_shin_geom_id
                    ).astype(jnp.float32),

                    'cause_nan_qpos': jnp.isnan(
                        d.qpos
                    ).any().astype(jnp.float32),

                    'cause_nan_qvel': jnp.isnan(
                        d.qvel
                    ).any().astype(jnp.float32),

                    'reward_total': next_state.reward,

                    'reward_tracking_lin':
                        next_state.metrics['reward/tracking_lin_vel'],

                    'reward_tracking_ang':
                        next_state.metrics['reward/tracking_ang_vel'],

                    'reward_orientation':
                        next_state.metrics['reward/orientation'],

                    'reward_termination':
                        next_state.metrics['reward/termination'],
                })
                                # M4.70-C: read-only pre-collision foot geometry
                d = next_state.data

                lf = d.geom_xpos[env._left_foot_geom_id]
                rf = d.geom_xpos[env._right_foot_geom_id]
                lr = lf - rf

                qw, qx, qy, qz = d.qpos[3:7]

                yaw = jnp.arctan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz)
                )

                cy = jnp.cos(yaw)
                sy = jnp.sin(yaw)

                # Source-exact foot contact pair identification
                geom_pairs = d.contact.geom
                lid = env._left_foot_geom_id
                rid = env._right_foot_geom_id

                pair_mask = (
                    ((geom_pairs[:, 0] == lid) &
                     (geom_pairs[:, 1] == rid))
                    |
                    ((geom_pairs[:, 0] == rid) &
                     (geom_pairs[:, 1] == lid))
                )

                pair_present = jnp.any(pair_mask)

                min_pair_dist = jnp.min(
                    jnp.where(
                        pair_mask,
                        d.contact.dist,
                        1.0e4
                    )
                )

                diag.update({
                    'lf_geom_x_m': lf[0],
                    'lf_geom_y_m': lf[1],
                    'lf_geom_z_m': lf[2],

                    'rf_geom_x_m': rf[0],
                    'rf_geom_y_m': rf[1],
                    'rf_geom_z_m': rf[2],

                    'foot_center_xy_dist_m':
                        jnp.linalg.norm(lr[:2]),

                    'foot_center_3d_dist_m':
                        jnp.linalg.norm(lr),

                    'foot_lateral_sep_body_m':
                        -sy * lr[0] + cy * lr[1],

                    'foot_forward_sep_body_m':
                        cy * lr[0] + sy * lr[1],

                    'lf_site_z_m':
                        d.site_xpos[env._feet_site_id[0], 2],

                    'rf_site_z_m':
                        d.site_xpos[env._feet_site_id[1], 2],

                    'lf_ground_contact':
                        collision.geoms_colliding(
                            d, lid, env._floor_geom_id
                        ).astype(jnp.float32),

                    'rf_ground_contact':
                        collision.geoms_colliding(
                            d, rid, env._floor_geom_id
                        ).astype(jnp.float32),

                    'phase_left_post_rad':
                        next_state.info['phase'][0],

                    'phase_right_post_rad':
                        next_state.info['phase'][1],

                    'foot_pair_present':
                        pair_present.astype(jnp.float32),

                    'foot_pair_signed_dist_m':
                        jnp.where(
                            pair_present,
                            min_pair_dist,
                            0.0
                        ),
                })
                return next_state,diag
            step_fn=jax.jit(one_step)
            st=env.reset(jax.random.PRNGKey(seed))
            frames=[]
            snapshot=None
            stop=0
            off_terminated=False
            for i in range(1,args.max_steps+1):
                st,diag=step_fn(st)
                data=jax.device_get(diag)
                row={k:float(v) for k,v in data.items() if k!='done'}
                frames.append(row);stop=i
                if i==args.push_step-1:
                    # Functional copy of entire state pytree, including info RNG and dynamics.
                    snapshot=jax.tree_util.tree_map(lambda x:x,st)
                if bool(data['done']):
                    off_terminated=True
                    break
            off=aggregate_frames(frames,off_terminated,stop,args.max_steps)
            off['type']='nominal_no_push'
            for direction in args.directions:
                case={'policy':name,'seed':seed,'profile':profile,'command':list(PROFILES[profile]),
                      'push_direction_body':direction,'push_step':args.push_step,
                      'nominal':off,'push_event_exposed':False}
                if snapshot is None or (off_terminated and stop<args.push_step):
                    case['not_exposed_reason']='terminated_before_push'
                    results.append(case)
                    print('M463_GATE3_SKIP',name,seed,profile,direction,'baseline_terminated',stop,flush=True)
                    continue
                qpos=np.asarray(jax.device_get(snapshot.data.qpos[:7]))
                heading=yaw_from_root_quat(qpos[3:7])
                delta=new_body_delta(direction,heading,args.push_magnitude)
                kicked=jax.tree_util.tree_map(lambda x:x,snapshot)
                kicked=kicked.replace(data=kicked.data.replace(qvel=kicked.data.qvel.at[:2].add(
                    jnp.asarray(delta,dtype=kicked.data.qvel.dtype))))
                pushed=[];last=args.push_step-1;terminated=False
                second_snapshot=None
                for i in range(args.push_step,args.max_steps+1):
                    kicked,diag=step_fn(kicked)
                    val=jax.device_get(diag)
                    pushed.append({k:float(v) for k,v in val.items() if k!='done'})
                    last=i
                    if (
                        args.second_push_step is not None
                        and i == args.second_push_step - 1
                        and not bool(val['done'])
):                  
                        second_snapshot = jax.tree_util.tree_map(
                            lambda x: x, kicked
                        )
                    if bool(val['done']):
                        terminated=True;break
                stats=aggregate_frames(pushed,terminated,last,args.max_steps,args.push_step)
                off_post=aggregate_frames(frames[args.push_step-1:],off_terminated,stop,args.max_steps,args.push_step)
                recovery=compute_recovery(frames[:args.push_step-1],pushed)
                off_post_frames=frames[args.push_step-1:]
                case.update(push_event_exposed=True,push_world_dqvel_xy_mps=list(delta),
                            push_body_heading_yaw_rad=heading,push=stats,
                            control_off_post_event=off_post,
                            recovery_2s_proxy=recovery,
                            matched_off_recovery_2s=matched_off_recovery(off_post_frames,pushed,100),
                            matched_off_recovery_5s=matched_off_recovery(off_post_frames,pushed,250),
                            fixed_window_tracking_2s=fixed_window_tracking(off_post_frames,pushed,100),
                            full_pre_push_snapshot_identity_by_construction=True)
                if args.second_push_step is not None:
                    second = {
                        'scheduled_step': args.second_push_step,
                        'magnitude_mps': args.second_push_magnitude,
                        'same_body_direction': True,
                        'exposed': False,
                    }

                    if second_snapshot is None:
                        second['reason'] = (
                            'single_push_terminated_before_second_event'
                        )
                    else:
                        heading2 = yaw_from_root_quat(
                            np.asarray(
                                jax.device_get(
                                    second_snapshot.data.qpos[3:7]
                                )
                            )
                        )

                        delta2 = new_body_delta(
                            direction,
                            heading2,
                            args.second_push_magnitude,
                        )

                        twice = second_snapshot.replace(
                            data=second_snapshot.data.replace(
                                qvel=second_snapshot.data.qvel.at[:2].add(
                                    jnp.asarray(
                                        delta2,
                                        dtype=second_snapshot.data.qvel.dtype,
                                    )
                                )
                            )
                        )

                        frames2 = []
                        last2 = args.second_push_step - 1
                        done2 = False

                        for j in range(
                            args.second_push_step, args.max_steps + 1
                        ):
                            twice, d2 = step_fn(twice)
                            v2 = jax.device_get(d2)

                            frames2.append({
                                k: float(v)
                                for k, v in v2.items()
                                if k != 'done'
                            })

                            last2 = j

                            if bool(v2['done']):
                                done2 = True
                                break
                            
                        first_only = pushed[
                            args.second_push_step - args.push_step:
                        ]

                        def recovery_vs_one_push(window):
                            if (
                                len(first_only) < window
                                or len(frames2) < window
                            ):
                                return {
                                    'recovered': None,
                                    'comparable': False,
                                    'reason': 'early_termination',
                                    'window_steps': window,
                                }

                            return {
                                **matched_off_recovery(
                                    first_only, frames2, window
                                ),
                                'comparable': True,
                            }

                        second.update(
                            exposed=True,
                            heading_rad=heading2,
                            world_dqvel_xy_mps=list(delta2),
                            double_push_after_second=aggregate_frames(
                                frames2,
                                done2,
                                last2,
                                args.max_steps,
                                args.second_push_step,
                            ),
                            single_push_survived_horizon=bool(
                                stats['completed']
                            ),
                            matched_single_recovery_2s=(
                                recovery_vs_one_push(100)
                            ),
                            matched_single_recovery_5s=(
                                recovery_vs_one_push(250)
                            ),
                            fixed_window_tracking_2s=(
                                fixed_window_tracking(
                                    first_only, frames2, 100
                                )
                            ),
                        )

                    case['second_push'] = second
                results.append(case)
                print('M463_GATE3_RESULT',name,seed,profile,direction,
                    'nominal',off['steps'],'pushed',last,'terminated',terminated,
                    'recovered',recovery['recovered'],flush=True)
    outcome={'status':'M463_GATE3_CP_FIXED_EVALUATION_COMPLETED',
        'class':'EXPLORATORY_MJX_GPU_ONLY_NOT_PAPER_EXACT_NOT_SIM_TO_SIM',
        'source_commit':PIN,'brax_train_sha256':TRAIN_SHA,
        'checkpoint_sha256':{'C':sha256(args.checkpoint_c),'P':sha256(args.checkpoint_p)},
        'train_steps_P': report_p['steps_observed'],
        'policy_P_is': (
            'M470H_PROBE2'
            if report_p['mode'] == 'probe2'
            else 'M463_PREVIOUS'
        ),
        'directions':args.directions,'max_steps':args.max_steps,'push_step':args.push_step,
        'push_magnitude_mps':args.push_magnitude,'dt_s':float(env.dt),
        'random_source_push_enabled':False,'domain_randomization_enabled':False,
        'deterministic_actor':True,'training_performed':False,'number_of_envs':1,
        'checkpoint_policy_layout_verified':True,'evaluation_jax_version':jax.__version__,
        'eval_metric_extension':'M463_GATE3_MATCHED_OFF_RECOVERY_V1',
        'second_push_step': args.second_push_step,
        'second_push_magnitude_mps': (
            args.second_push_magnitude
            if args.second_push_step is not None
            else None
        ),
        'second_push_comparison': (
            'two pushes versus one push from '
            'the same second-event pre-state'
        ),
        'caveats':['C and P see the same reset seed and command, but their physical states may differ before push.',
            'Within each policy/seed, ON forks directly from the full OFF state at push_step-1.',
            'Commands manually overridden in info and both observation arrays at indices 9:12; original policy obs noise retained.',
            'Only MJX GPU physics, no native MuJoCo CPU integration in this diagnostic.',
            'The push-vs-OFF comparison uses the same time window [push_step,max_steps] for tracking metrics.',
            'Push is fixed one-time qvel delta; not force in Newtons and not original randomized training push.',
            'Recovery metric is an explicit separate proxy; not identical to M4.61 recovery.',
            'M4.63 Gate 3 additional matched-OFF dynamic recovery uses same-policy contemporaneous OFF, a DIFFERENT metric from prepush-median recovery.',
            'Fixed 100-step tracking RMSE is reported only for cases reaching the full window; censored otherwise.',
            'Evaluation seeds were chosen for diagnostic; independent final holdout must be locked before policy selection.',
            'A survival at 1000 steps is right-censored, not indefinite robustness.'],
        'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:
        json.dump(outcome,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print('M463_GATE3_DONE',args.output,'cases',len(results),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','run'),required=True)
    p.add_argument('--checkpoint-c',type=Path,required=True)
    p.add_argument('--checkpoint-p',type=Path,required=True)
    p.add_argument('--report-p',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=str,default='7301,7331,7357')
    p.add_argument('--profiles',type=str,default='stand,forward_fast,turn_left,turn_right,strafe')
    p.add_argument('--directions',type=str,default='left,right')
    p.add_argument('--push-step',type=int,default=300)
    p.add_argument('--push-magnitude',type=float,default=0.35)
    p.add_argument('--second-push-step', type=int, default=None)
    p.add_argument('--second-push-magnitude', type=float, default=0.35)
    p.add_argument('--max-steps',type=int,default=1000)
    p.add_argument('--max-cases',type=int,default=120)
    args=p.parse_args()
    try:
        args.seeds=[int(x.strip()) for x in args.seeds.split(',') if x.strip()]
        args.profiles=parse_items(args.profiles,PROFILES,'profiles')
        args.directions=parse_items(args.directions,DIRECTIONS,'directions')
    except ValueError as e:p.error(str(e))
    if len(args.seeds)!=len(set(args.seeds)) or not args.seeds or any(s<0 or s>2**31-1 for s in args.seeds):p.error('invalid seeds')
    if not (51<=args.push_step<=args.max_steps-250):p.error('push-step requires >=50 prepush frames and 250 postpush steps')
    if not (0.1<=args.push_magnitude<=2.0) or not math.isfinite(args.push_magnitude):p.error('push magnitude must be 0.1..2.0 m/s')
    if args.second_push_step is not None:
        if not (
            args.push_step + 15
            <= args.second_push_step
            <= args.max_steps - 250
        ):
            p.error(
                'second push requires >=15 steps gap '
                'and >=250 steps post-event'
            )

        if not (
            0.1 <= args.second_push_magnitude <= 2.0
        ) or not math.isfinite(args.second_push_magnitude):
            p.error('invalid second push magnitude')
    if args.max_steps>5000 or args.max_steps<350:p.error('max steps must be in [350,5000]')
    count=len(args.seeds)*len(args.profiles)*len(args.directions)*2
    if count>args.max_cases or args.max_cases>300:p.error(f'{count} cases requested, max-cases={args.max_cases}')
    if args.mode=='plan':
        print('M463_GATE3_PLAN',json.dumps({'cases':count,'seeds':args.seeds,'profiles':args.profiles,
            'directions':args.directions,'max_steps':args.max_steps,'push_step':args.push_step,
            'push_magnitude_mps':args.push_magnitude,'one_MJX_env_at_time':True,
            'no_training':True,'output':str(args.output)}));return
    for path in (args.checkpoint_c,args.checkpoint_p,args.report_p):
        if not path.is_file():p.error('missing input: '+str(path))
    run(args)

if __name__=='__main__':main()
