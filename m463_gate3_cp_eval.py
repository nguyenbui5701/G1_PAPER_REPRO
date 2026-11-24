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
M470_10M_SHA = '0a011b6a3011f126a29832e4faa18972d7fbdddef77b63414219a15abab8314a'
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

    if mode == 'longtrain':
        cfg = report.get('hardware_scaled_config') or {}
        cov = report.get('coverage') or {}
        counts = cov.get('counts') or {}
        entries = report.get('checkpoints') or []

        checks = {
            'status': report.get('status') == 'M463_COVERAGE_LONGTRAIN_PASS',
            'stage': report.get('stage') == 'easy_repeat_support',
            'independent': (
                report.get('independent_budget') == '10M'
                and report.get('experiment_design')
                    == 'INDEPENDENT_C401M_SINGLE_PROCESS'
            ),
            'source': (
                report.get('source_commit') == PIN
                and report.get('source_checkpoint_sha256') == BASE_SHA
            ),
            'warmstart': (
                report.get('trained_from')
                    == 'C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE'
                and report.get('cumulative_steps_start') == 0
                and report.get('resume_fullstate_sha256') is None
            ),
            'budget': (
                report.get('steps_observed') == 10485760
                and report.get('rollouts_target') == 40
                and report.get('training_seed') == 1
            ),
            'ppo_config': all(
                cfg.get(key) == value
                for key, value in {
                    'num_envs': 8192,
                    'unroll_length': 32,
                    'batch_size': 1024,
                    'num_minibatches': 8,
                    'num_updates_per_batch': 5,
                    'num_evals': 6,
                    'num_resets_per_eval': 1,
                    'num_timesteps': 10485760,
                }.items()
            ),
            'training': (
                report.get('training_performed') is True
                and report.get('source_push_disabled_in_custom_train_env') is True
            ),
            'health': (
                report.get('weights_finite') is True
                and report.get('memory_gate_pass') is True
                and report.get('coverage_gate_pass') is True
                and report.get('callback_vs_final_equal') is True
                and report.get('final_tensor_count') == 25
            ),
            'exposure': (
                cov.get('passed') is True
                and cov.get('push_events_observed', 0) > 0
                and counts.get(
                    'training/m463_repeat_side_matched_per_rollout', 0
                ) > 0
            ),
            'checkpoints': (
                [x.get('steps') for x in entries]
                == [2097152 * i for i in range(1, 6)]
            ),
        }

        if not all(checks.values()):
            raise RuntimeError(
                'M470_10M_INVALID_REPORT '
                + str([k for k, v in checks.items() if not v])
            )

        ck = entries[-1]
        if (
            ck.get('file') != Path(checkpoint).name
            or ck.get('sha256') != M470_10M_SHA
            or sha256(checkpoint) != M470_10M_SHA
        ):
            raise RuntimeError('M470_10M_CHECKPOINT_SHA_MISMATCH')

        return report

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


# M470-H2A: Recovery Command Governor
H2A_CONFIG = {
    'entry_velocity': 0.45,
    'entry_tilt': 20.0,
    'exit_velocity': 0.20,
    'exit_tilt': 10.0,
    'entry_ticks': 2,
    'exit_ticks': 10,
    'velocity_gain': 0.5,
    'max_bias': 0.25,
    'yaw_multiplier': 0.5,
    'xy_slew': 0.05,
    'yaw_slew': 0.10,
    'max_active_ticks': 100,
    'max_return_ticks': 10,
}


def h2a_clip(value, lower, upper):
    return max(lower, min(upper, value))


def h2a_clip_norm(x, y, limit):
    norm = math.hypot(x, y)
    if norm > limit:
        scale = limit / norm
        return x * scale, y * scale
    return x, y


def h2a_initial_state(goal):
    return {
        'mode': 'NORMAL',
        'high': 0,
        'low': 0,
        'age': 0,
        'forced': False,
        'return_ticks': 0,
        'effective': tuple(goal),
    }


def h2a_tick(obs, goal, previous, enabled):
    cfg = H2A_CONFIG
    goal = tuple(float(x) for x in goal)

    vx, vy = float(obs[0]), float(obs[1])
    gx, gy, gz = map(float, obs[6:9])

    if not all(math.isfinite(x) for x in (
        vx, vy, gx, gy, gz, *goal
    )):
        raise RuntimeError('H2A_NONFINITE_INPUT')

    gravity_norm = math.sqrt(gx*gx + gy*gy + gz*gz)

    if gravity_norm <= 1e-6:
        raise RuntimeError('H2A_INVALID_GRAVITY')

    # Actor observation: upright is negative Z.
    tilt = math.degrees(math.acos(
        h2a_clip(-gz / gravity_norm, -1.0, 1.0)
    ))

    ex = vx - goal[0]
    ey = vy - goal[1]

    error = math.hypot(ex, ey)

    risky = (
        error >= cfg['entry_velocity']
        or tilt >= cfg['entry_tilt']
    )

    clear = (
        error <= cfg['exit_velocity']
        and tilt <= cfg['exit_tilt']
    )

    metrics = {
        'sensor_velocity_error': error,
        'sensor_tilt_deg': tilt,
        'risk_high': risky,
        'risk_clear': clear,
        'timeout_event': False,
        'intervention': False,
    }

    if not enabled:
        return h2a_initial_state(goal), metrics

    g = dict(previous)
    old_mode = g['mode']

    g['high'] = (
        min(g['high'] + 1, cfg['entry_ticks'])
        if risky else 0
    )

    g['low'] = (
        min(g['low'] + 1, cfg['exit_ticks'])
        if clear else 0
    )

    if g['mode'] == 'LOCKOUT':
        if g['low'] >= cfg['exit_ticks']:
            g = h2a_initial_state(goal)

        metrics['intervention'] = True
        return g, metrics

    if g['mode'] == 'NORMAL':
        if g['high'] >= cfg['entry_ticks']:
            g['mode'] = 'RECOVERY'
            g['age'] = 0
            g['low'] = 0
            g['forced'] = False

    if g['mode'] in ('RECOVERY', 'RETURN'):

        # Timeout persists across RECOVERY/RETURN reentry.
        if (
            g['age'] >= cfg['max_active_ticks']
            and not g['forced']
        ):
            g['forced'] = True
            g['mode'] = 'RETURN'
            g['return_ticks'] = 0
            metrics['timeout_event'] = True

        if not g['forced']:
            if (
                g['mode'] == 'RECOVERY'
                and g['low'] >= cfg['exit_ticks']
            ):
                g['mode'] = 'RETURN'

            elif (
                g['mode'] == 'RETURN'
                and g['high'] >= cfg['entry_ticks']
            ):
                g['mode'] = 'RECOVERY'

    if g['mode'] == 'RECOVERY':
        bx, by = h2a_clip_norm(
            cfg['velocity_gain'] * ex,
            cfg['velocity_gain'] * ey,
            cfg['max_bias'],
        )

        target = (
            h2a_clip(goal[0] + bx, -1.0, 1.0),
            h2a_clip(goal[1] + by, -0.5, 0.5),
            h2a_clip(
                cfg['yaw_multiplier'] * goal[2],
                -1.0, 1.0
            ),
        )
    else:
        target = goal

    px, py, pw = g['effective']

    dx = target[0] - px
    dy = target[1] - py
    dw = target[2] - pw

    if math.hypot(dx, dy) <= cfg['xy_slew']:
        nx, ny = target[:2]
    else:
        dx, dy = h2a_clip_norm(
            dx, dy, cfg['xy_slew']
        )
        nx, ny = px + dx, py + dy

    if abs(dw) <= cfg['yaw_slew']:
        nw = target[2]
    else:
        nw = pw + h2a_clip(
            dw,
            -cfg['yaw_slew'],
            cfg['yaw_slew']
        )

    g['effective'] = (nx, ny, nw)

    metrics['intervention'] = (
        old_mode != 'NORMAL'
        or g['mode'] != 'NORMAL'
    )

    if g['mode'] in ('RECOVERY', 'RETURN'):
        if g['forced']:
            g['return_ticks'] += 1

            if g['return_ticks'] > cfg['max_return_ticks']:
                raise RuntimeError('H2A_RETURN_BOUND_EXCEEDED')
        else:
            g['age'] += 1

    if g['mode'] == 'RETURN':
        if g['effective'] == goal:
            if g['forced']:
                g['mode'] = 'LOCKOUT'
                g['high'] = 0
                g['low'] = 0
            else:
                g = h2a_initial_state(goal)

    return g, metrics


def h2a_recovery(nominal, pushed, event_step, window):
    start = event_step - 1
    end = start + window

    # Full recovery requires survival through the window.
    if len(nominal) < end or len(pushed) < end:
        return {
            'success': False,
            'recovery_s': None,
            'reason': 'insufficient_survival',
        }

    off = nominal[start:end]
    on = pushed[start:end]

    if any(r['done'] for r in off + on):
        return {
            'success': False,
            'recovery_s': None,
            'reason': 'terminated_in_window',
        }

    bands = {
        'vx': 0.10,
        'vy': 0.10,
        'yaw_rate': 0.10,
        'height': 0.05,
        'tilt_deg': 5.0,
    }

    count = 0

    for i, (a, b) in enumerate(zip(off, on)):
        matched = all(
            abs(a[k] - b[k]) <= tolerance
            for k, tolerance in bands.items()
        )

        goal_error_ok = (
            b['xy_err'] <= 0.20
            and b['yaw_err'] <= 0.30
        )

        returned = (
            b['governor_mode'] == 'NORMAL'
            and b['effective_command'] == b['goal_command']
        )

        if matched and goal_error_ok and returned:
            count += 1

            if count >= 10:
                return {
                    'success': True,
                    'recovery_s': (i + 1) * 0.02,
                    'reason': 'matched_goal_recovery',
                }
        else:
            count = 0

    return {
        'success': False,
        'recovery_s': None,
        'reason': 'recovery_band_not_reached',
    }


def h2a_evaluate_group(
    args, env, step_fn, command, seed, profile,
    jax, jnp, np
):
    goal = tuple(
        float(x) for x in jax.device_get(command)
    )

    def clone(state):
        return jax.tree_util.tree_map(lambda x: x, state)

    def kick(state, delta):
        return state.replace(
            data=state.data.replace(
                qvel=state.data.qvel.at[:2].add(
                    jnp.asarray(
                        delta,
                        dtype=state.data.qvel.dtype,
                    )
                )
            )
        )

    def advance(
        state, governor, first, last,
        enabled, quiet=False, capture_step=None
    ):
        frames = []
        snapshot = None

        for i in range(first, last + 1):
            obs = np.asarray(
                jax.device_get(state.obs['state'])
            )

            if obs.shape != (103,):
                raise RuntimeError('H2A_BAD_OBS_SHAPE')

            governor, risk = h2a_tick(
                obs, goal, governor, enabled
            )

            effective = governor['effective']

            if quiet and (
                risk['intervention']
                or effective != goal
            ):
                return {
                    'frames': frames,
                    'state': state,
                    'governor': governor,
                    'snapshot': snapshot,
                    'blocked': {
                        'step': i,
                        'reason': 'PREPUSH_INTERVENTION',
                        'risk': risk,
                        'governor': governor,
                    },
                }

            state, diag = step_fn(
                state,
                jnp.asarray(
                    effective,
                    dtype=command.dtype,
                )
            )

            data = jax.device_get(diag)

            if not bool(data['h2a_state_finite']):
                raise RuntimeError(
                    'H2A_NONFINITE_TRANSITION'
                )

            row = {}

            for key, value in data.items():
                arr = np.asarray(value)

                if not np.isfinite(arr).all():
                    raise RuntimeError(
                        'H2A_NONFINITE_DIAG ' + key
                    )

                row[key] = (
                    float(arr)
                    if arr.ndim == 0
                    else arr.tolist()
                )

            row.update({
                'step': i,
                'done': bool(data['done']),
                'goal_command': list(goal),
                'effective_command': list(effective),
                'governor_mode': governor['mode'],
                'governor_age': governor['age'],
                **risk,
            })

            frames.append(row)

            if row['done']:
                break

            if i == capture_step:
                snapshot = (
                    clone(state),
                    dict(governor),
                )

        return {
            'frames': frames,
            'state': state,
            'governor': governor,
            'snapshot': snapshot,
            'blocked': None,
        }

    def summarize(frames):
        terminated = bool(
            frames and frames[-1]['done']
        )

        stats = aggregate_frames(
            frames,
            terminated,
            len(frames),
            args.max_steps,
        )

        stats.update({
            'intervention_steps': sum(
                bool(r['intervention']) for r in frames
            ),
            'timeout_events': sum(
                bool(r['timeout_event']) for r in frames
            ),
            'joint_limit_excess_max': max(
                r['h2a_joint_limit_excess']
                for r in frames
            ),
            'ctrl_limit_excess_max': max(
                r['h2a_ctrl_limit_excess']
                for r in frames
            ),
            'force_limit_excess_max': max(
                r['h2a_force_limit_excess']
                for r in frames
            ),
        })

        return stats

    initial = env.reset(jax.random.PRNGKey(seed))

    # Run one genuine shared prefix.
    # Governor ON is evaluated without any push knowledge.
    prefix = advance(
        initial,
        h2a_initial_state(goal),
        1,
        args.push_step - 1,
        enabled=True,
        quiet=True,
    )

    record = {
        'seed': seed,
        'profile': profile,
        'goal_command': list(goal),
        'shared_prefix': prefix['frames'],
        'blocked': prefix['blocked'],
        'arms': {},
    }

    if prefix['blocked'] is not None:
        return record

    if (
        len(prefix['frames']) != args.push_step - 1
        or prefix['frames'][-1]['done']
    ):
        record['blocked'] = {
            'reason': 'NOMINAL_TERMINATED_BEFORE_PUSH',
        }
        return record

    heading = yaw_from_root_quat(
        np.asarray(
            jax.device_get(
                prefix['state'].data.qpos[3:7]
            )
        )
    )

    record['first_heading_rad'] = heading

    for enabled in (False, True):
        arm_name = 'ON' if enabled else 'OFF'

        def initial_branch():
            return (
                clone(prefix['state']),
                dict(prefix['governor'])
                if enabled
                else h2a_initial_state(goal),
            )

        state, governor = initial_branch()

        nominal = advance(
            state, governor,
            args.push_step, args.max_steps,
            enabled,
        )

        nominal_full = (
            prefix['frames'] + nominal['frames']
        )

        arm = {
            'nominal_frames': nominal['frames'],
            'nominal_summary': summarize(nominal_full),
            'directions': {},
        }

        for direction in args.directions:

            # Both controllers receive identical world delta-v.
            delta1 = new_body_delta(
                direction,
                heading,
                args.push_magnitude,
            )

            # Second direction is also based on FIRST heading.
            delta2 = new_body_delta(
                direction,
                heading,
                args.second_push_magnitude,
            )

            state, governor = initial_branch()

            single = advance(
                kick(state, delta1),
                governor,
                args.push_step,
                args.max_steps,
                enabled,
                capture_step=args.second_push_step - 1,
            )

            single_full = (
                prefix['frames'] + single['frames']
            )

            item = {
                'first_delta_world_mps': list(delta1),
                'second_delta_world_mps': list(delta2),
                'single_frames': single['frames'],
                'single_summary': summarize(single_full),
                'single_recovery_2s': h2a_recovery(
                    nominal_full,
                    single_full,
                    args.push_step, 100,
                ),
                'single_recovery_5s': h2a_recovery(
                    nominal_full,
                    single_full,
                    args.push_step, 250,
                ),
                'second_push_exposed': False,
            }

            if single['snapshot'] is None:
                # Failure before second event:
                # retain it in the double-push denominator.
                item.update({
                    'double_frames_after_second': [],
                    'double_summary': summarize(single_full),
                    'double_recovery_2s': {
                        'success': False,
                        'reason': 'terminated_before_second',
                    },
                    'double_recovery_5s': {
                        'success': False,
                        'reason': 'terminated_before_second',
                    },
                })

            else:
                state2, governor2 = single['snapshot']

                double = advance(
                    kick(clone(state2), delta2),
                    dict(governor2),
                    args.second_push_step,
                    args.max_steps,
                    enabled,
                )

                n = args.second_push_step - 1

                double_full = (
                    single_full[:n] + double['frames']
                )

                item.update({
                    'second_push_exposed': True,
                    'double_frames_after_second':
                        double['frames'],
                    'double_summary': summarize(double_full),
                    'double_recovery_2s': h2a_recovery(
                        nominal_full,
                        double_full,
                        args.second_push_step, 100,
                    ),
                    'double_recovery_5s': h2a_recovery(
                        nominal_full,
                        double_full,
                        args.second_push_step, 250,
                    ),
                })

            arm['directions'][direction] = item

        record['arms'][arm_name] = arm

    return record


def h2a_write(args, results, jax_version, blocked=False):
    status = (
        'M470_H2A_EVALUATION_BLOCKED'
        if blocked
        else 'M470_H2A_EVALUATION_COMPLETED'
    )

    outcome = {
        'status': status,
        'source_commit': PIN,
        'brax_train_sha256': TRAIN_SHA,
        'checkpoint_sha256': sha256(args.checkpoint_c),
        'jax_version': jax_version,
        'training_performed': False,
        'deterministic_actor': True,
        'random_source_push_enabled': False,
        'domain_randomization_enabled': False,
        'governor_config': H2A_CONFIG,
        'seeds': args.seeds,
        'profiles': args.profiles,
        'directions': args.directions,
        'push_step': args.push_step,
        'push_magnitude_mps': args.push_magnitude,
        'second_push_step': args.second_push_step,
        'second_push_magnitude_mps':
            args.second_push_magnitude,
        'max_steps': args.max_steps,
        'dt_s': 0.02,
        'expected_logical_trajectories': 60,
        'completed_groups': sum(
            r['blocked'] is None for r in results
        ),
        'results': results,
    }

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with args.output.open('x', encoding='utf8') as f:
        json.dump(
            outcome,
            f,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        f.write('\n')

    print(status, args.output, flush=True)

    if blocked:
        raise RuntimeError('M470_H2A_PREFIX_GATE_NO_GO')


H2B_CAL_SEEDS = (7301, 7303)
H2B_DEV_SEEDS = (7331, 7357)
H2B_PROFILES = ('stand', 'forward_fast', 'turn_left')
H2B_CONFIG = {
    'dt': 0.02,
    'gravity': 9.81,
    'height_min': 0.35,
    'height_max': 1.0,
    'reference_min_count': 8,
    'age_edges': [10, 30, 75],
    'phase_bins': 4,
    'entry_residual_m': 0.04,
    'exit_residual_m': 0.015,
    'entry_ticks': 2,
    'exit_ticks': 10,
    'residual_gain_per_s': 2.0,
    'max_bias': 0.25,
    'xy_slew': 0.05,
    'max_active_ticks': 100,
    'max_return_ticks': 10,
}


class H2BReferenceUnavailable(RuntimeError):
    pass


def h2b_capture_point(raw, previous=None):
    """Pure calculation on ONE cached sample; no simulator or kick inputs."""
    cfg = H2B_CONFIG
    age = int(raw['age'])
    c = tuple(map(float, raw['com_world']))
    p = tuple(map(float, raw['imu_world']))
    rot = tuple(tuple(map(float, row)) for row in raw['imu_rotation'])
    local_v = tuple(map(float, raw['local_velocity']))
    left = tuple(map(float, raw['left_foot_world']))
    right = tuple(map(float, raw['right_foot_world']))
    phase = float(raw['phase_left'])
    floor_z = float(raw['floor_z'])
    values = (*c, *p, *local_v, *left, *right, phase, floor_z,
              *(v for row in rot for v in row))
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError('H2B_NONFINITE_SENSOR_SAMPLE')
    rel = tuple(c[i] - p[i] for i in range(3))
    if previous is None:
        if age != 0:
            raise RuntimeError('H2B_HISTORY_MISSING')
        rel_rate = (0.0, 0.0, 0.0)
    else:
        if age != previous['age'] + 1:
            raise RuntimeError('H2B_HISTORY_NOT_CONSECUTIVE')
        rel_rate = tuple((rel[i] - previous['relative_com_world'][i])
                         / cfg['dt'] for i in range(3))
    v_imu = tuple(sum(rot[i][j] * local_v[j] for j in range(3))
                  for i in range(3))
    v_com = tuple(v_imu[i] + rel_rate[i] for i in range(3))
    lc, rc = map(bool, raw['contact'])
    if lc and not rc:
        mode, support = 'LEFT', left
    elif rc and not lc:
        mode, support = 'RIGHT', right
    else:
        mode = 'BOTH' if lc and rc else 'NO_SUPPORT_PROXY'
        support = tuple((left[i] + right[i]) / 2 for i in range(3))
    h_raw = c[2] - floor_z
    if h_raw <= 0 or not math.isfinite(h_raw):
        raise RuntimeError('H2B_INVALID_COM_HEIGHT')
    h = h2a_clip(h_raw, cfg['height_min'], cfg['height_max'])
    omega = math.sqrt(cfg['gravity'] / h)
    # Heading frame: horizontal projection of IMU +X, not full 3D tilt.
    hx, hy = rot[0][0], rot[1][0]
    if math.hypot(hx, hy) <= 1e-6:
        raise RuntimeError('H2B_DEGENERATE_HEADING')
    yaw = math.atan2(hy, hx)
    cy, sy = math.cos(yaw), math.sin(yaw)
    xi = (c[0] + v_com[0]/omega, c[1] + v_com[1]/omega)
    x, y = xi[0]-support[0], xi[1]-support[1]
    d = (cy*x + sy*y, -sy*x + cy*y)
    if not all(math.isfinite(v) for v in (*d, *v_com, *rel_rate)):
        raise RuntimeError('H2B_NONFINITE_ESTIMATE')
    return {
        **raw,
        'age': age,
        'relative_com_world': rel,
        'relative_com_rate_world': rel_rate,
        'derivative_reset': previous is None,
        'imu_velocity_world': v_imu,
        'estimated_com_velocity_world': v_com,
        'contact_mode': mode,
        'support_center_world': support,
        'height_raw': h_raw,
        'height_used': h,
        'omega': omega,
        'heading_rad': yaw,
        'capture_point_world': xi,
        'd_heading': d,
    }


def h2b_make_sampler(env, jax, jnp, np, collision):
    """Make a causal idealized sensor-proxy sampler. Call only reset/poststep."""
    model = env.mj_model
    masses = np.asarray(model.body_mass, dtype=np.float64)
    if (masses.shape != (model.nbody,) or not np.isfinite(masses).all()
            or masses[0] != 0 or (masses < 0).any() or masses.sum() <= 0):
        raise RuntimeError('H2B_INVALID_BODY_MASSES')
    # Pinned G1 flat scene contains only robot positive-mass bodies.
    # Check all positive-mass bodies descend from pelvis.
    root = int(model.body('pelvis').id)
    for i in np.flatnonzero(masses > 0):
        parent = int(i)
        while parent not in (0, root):
            parent = int(model.body_parentid[parent])
        if parent != root:
            raise RuntimeError('H2B_NON_ROBOT_MASS')
    weights = jnp.asarray(masses / masses.sum(), dtype=jnp.float32)
    imu = int(env._pelvis_imu_site_id)
    left, right = int(env._left_foot_geom_id), int(env._right_foot_geom_id)
    floor = int(env._floor_geom_id)
    if int(model.geom_type[floor]) != 0:  # mjGEOM_PLANE in pinned MuJoCo
        raise RuntimeError('H2B_EXPECTED_FLAT_PLANE')

    @jax.jit
    def raw_sample(state):
        data = state.data
        obs = state.obs['state']
        if obs.shape != (103,) or data.xipos.shape != (model.nbody, 3):
            raise RuntimeError('H2B_OBSERVATION_OR_XIPOS_SHAPE')
        # Never use qvel directly, nor regenerate actor observation/RNG.
        return {
            'com_world': jnp.sum(weights[:, None]*data.xipos, axis=0),
            'imu_world': data.site_xpos[imu],
            'imu_rotation': data.site_xmat[imu].reshape(3, 3),
            'local_velocity': obs[0:3],
            'phase_cos': obs[99],
            'phase_sin': obs[101],
            'left_foot_world': data.geom_xpos[left],
            'right_foot_world': data.geom_xpos[right],
            'floor_z': data.geom_xpos[floor, 2],
            'floor_normal': data.geom_xmat[floor].reshape(3, 3)[:, 2],
            'contact': jnp.stack([
                collision.geoms_colliding(data, left, floor),
                collision.geoms_colliding(data, right, floor),
            ]),
        }

    def sample(state, age, previous=None):
        d = jax.device_get(raw_sample(state))
        if not all(np.isfinite(np.asarray(v)).all() for v in d.values()):
            raise RuntimeError('H2B_NONFINITE_RAW_SAMPLE')
        if not np.allclose(d['floor_normal'], [0, 0, 1], rtol=0, atol=1e-6):
            raise RuntimeError('H2B_NONHORIZONTAL_FLOOR')
        r = np.asarray(d['imu_rotation'])
        if (not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-4)
                or abs(float(np.linalg.det(r))-1) > 1e-4):
            raise RuntimeError('H2B_INVALID_ROTATION')
        cos, sin = float(d['phase_cos']), float(d['phase_sin'])
        if abs(math.hypot(cos, sin)-1.0) > 1e-4:
            raise RuntimeError('H2B_INVALID_PHASE')
        raw = {k: np.asarray(v).tolist() for k, v in d.items()}
        raw['age'] = age
        raw['phase_left'] = math.atan2(sin, cos)
        return h2b_capture_point(raw, previous)

    return sample


def h2b_bucket(profile, sample):
    age = sample['age']
    if age < 0 or profile not in H2B_PROFILES:
        raise RuntimeError('H2B_BAD_BUCKET_INPUT')
    age_bin = sum(age >= edge for edge in H2B_CONFIG['age_edges'])
    p = (sample['phase_left'] + math.pi) % (2*math.pi)
    phase_bin = min(3, int(p / (math.pi/2)))
    mode = sample['contact_mode']
    if mode not in ('LEFT', 'RIGHT', 'BOTH', 'NO_SUPPORT_PROXY'):
        raise RuntimeError('H2B_BAD_CONTACT_MODE')
    # No fallback across profile or contact state.
    prefix = f'{profile}|{mode}|'
    return (f'{prefix}{age_bin}|{phase_bin}',
            f'{prefix}{age_bin}|*', f'{prefix}*|*')


def h2b_fit_reference(calibration):
    expected = {(s, p) for s in H2B_CAL_SEEDS for p in H2B_PROFILES}
    actual = {(g['seed'], g['profile']) for g in calibration}
    if len(calibration) != 6 or actual != expected:
        raise RuntimeError('H2B_CALIBRATION_GROUPS_INVALID')
    bins = {}
    for group in calibration:
        frames = group['frames']
        if (group['blocked'] is not None or len(frames) != 1000
                or any(row['done'] for row in frames)):
            raise RuntimeError('H2B_CALIBRATION_INCOMPLETE')
        for i, row in enumerate(frames):
            sample = row['h2b_sample']
            if sample['age'] != i or row['step'] != i+1:
                raise RuntimeError('H2B_CALIBRATION_HISTORY_INVALID')
            xy = sample['d_heading']
            if len(xy) != 2 or not all(math.isfinite(x) for x in xy):
                raise RuntimeError('H2B_BAD_CALIBRATION_ESTIMATE')
            for key in h2b_bucket(group['profile'], sample):
                b = bins.setdefault(key, {'count': 0, 'lower': list(xy),
                                          'upper': list(xy)})
                b['count'] += 1
                b['lower'] = [min(a, x) for a, x in zip(b['lower'], xy)]
                b['upper'] = [max(a, x) for a, x in zip(b['upper'], xy)]
    return bins


def h2b_risk_residual(profile, sample, reference):
    for level, key in enumerate(h2b_bucket(profile, sample)):
        b = reference.get(key)
        if b is not None and b['count'] >= H2B_CONFIG['reference_min_count']:
            d = sample['d_heading']
            r = tuple(x - h2a_clip(x, lo, hi)
                      for x, lo, hi in zip(d, b['lower'], b['upper']))
            return {'reference_key': key, 'reference_backoff': level,
                    'reference_count': b['count'],
                    'reference_lower': list(b['lower']),
                    'reference_upper': list(b['upper']),
                    'residual_xy_m': r, 'residual_norm_m': math.hypot(*r)}
    raise H2BReferenceUnavailable('|'.join(h2b_bucket(profile, sample)))


def h2b_tick(residual, goal, previous, enabled):
    cfg = H2B_CONFIG
    goal = tuple(map(float, goal))
    rx, ry = map(float, residual['residual_xy_m'])
    rho = math.hypot(rx, ry)
    if not all(math.isfinite(x) for x in (rx, ry, rho, *goal)):
        raise RuntimeError('H2B_NONFINITE_CONTROL')
    risky = rho >= cfg['entry_residual_m']
    clear = rho <= cfg['exit_residual_m']
    metrics = {**residual, 'risk_high': risky, 'risk_clear': clear,
               'timeout_event': False, 'intervention': False}
    if not enabled:
        return h2a_initial_state(goal), metrics
    g = dict(previous)
    old_mode = g['mode']
    g['high'] = min(g['high']+1, cfg['entry_ticks']) if risky else 0
    g['low'] = min(g['low']+1, cfg['exit_ticks']) if clear else 0
    if g['mode'] == 'LOCKOUT':
        if g['low'] >= cfg['exit_ticks']:
            g = h2a_initial_state(goal)
        metrics['intervention'] = True
        return g, metrics
    if g['mode'] == 'NORMAL' and g['high'] >= cfg['entry_ticks']:
        g['mode'], g['age'], g['low'], g['forced'] = 'RECOVERY', 0, 0, False
    if g['mode'] in ('RECOVERY', 'RETURN'):
        if g['age'] >= cfg['max_active_ticks'] and not g['forced']:
            g['forced'], g['mode'], g['return_ticks'] = True, 'RETURN', 0
            metrics['timeout_event'] = True
        if not g['forced']:
            if g['mode'] == 'RECOVERY' and g['low'] >= cfg['exit_ticks']:
                g['mode'] = 'RETURN'
            elif g['mode'] == 'RETURN' and g['high'] >= cfg['entry_ticks']:
                g['mode'] = 'RECOVERY'
    if g['mode'] == 'RECOVERY':
        bx, by = h2a_clip_norm(cfg['residual_gain_per_s']*rx,
                               cfg['residual_gain_per_s']*ry, cfg['max_bias'])
        target = (h2a_clip(goal[0]+bx, -1., 1.),
                  h2a_clip(goal[1]+by, -.5, .5), goal[2])
    else:
        target = goal
    px, py, _ = g['effective']
    dx, dy = target[0]-px, target[1]-py
    if math.hypot(dx, dy) <= cfg['xy_slew']:
        nx, ny = target[:2]
    else:
        dx, dy = h2a_clip_norm(dx, dy, cfg['xy_slew'])
        nx, ny = px+dx, py+dy
    g['effective'] = (nx, ny, goal[2])
    metrics['intervention'] = old_mode != 'NORMAL' or g['mode'] != 'NORMAL'
    if g['mode'] in ('RECOVERY', 'RETURN'):
        if g['forced']:
            g['return_ticks'] += 1
            if g['return_ticks'] > cfg['max_return_ticks']:
                raise RuntimeError('H2B_RETURN_BOUND_EXCEEDED')
        else:
            g['age'] += 1
    if g['mode'] == 'RETURN' and g['effective'] == goal:
        if g['forced']:
            g['mode'], g['high'], g['low'] = 'LOCKOUT', 0, 0
        else:
            g = h2a_initial_state(goal)
    return g, metrics


def h2b_fork(ctx, jax):
    # MJX arrays are immutable; dict/list metadata copied without resampling.
    import copy
    state, cache, governor = ctx
    return (jax.tree_util.tree_map(lambda x: x, state),
            copy.deepcopy(cache), dict(governor))


def h2b_kick(ctx, delta, jnp):
    state, cache, governor = ctx
    qvel = state.data.qvel.at[:2].add(jnp.asarray(delta, dtype=state.data.qvel.dtype))
    # Only physics state changes. Cached sensor observation and previous
    # estimator/supervisor history must be untouched until AFTER step_fn.
    return state.replace(data=state.data.replace(qvel=qvel)), cache, governor


def h2b_advance(ctx, first, last, *, profile, goal, command, enabled,
                reference, sampler, step_fn, jax, jnp, np,
                quiet=False, capture_step=None):
    state, cache, governor = ctx
    frames, snapshot, blocked = [], None, None
    for i in range(first, last+1):
        if cache['age'] != i-1:
            raise RuntimeError('H2B_CACHE_AGE_MISMATCH')
        if reference is None:  # calibration ONLY
            if enabled or quiet:
                raise RuntimeError('H2B_UNFROZEN_REFERENCE_CONTROL')
            governor = h2a_initial_state(goal)
            risk = {'risk_high': False, 'risk_clear': False,
                    'timeout_event': False, 'intervention': False}
        else:
            try:
                residual = h2b_risk_residual(profile, cache, reference)
            except H2BReferenceUnavailable as e:
                blocked = {'step': i, 'reason': 'REFERENCE_UNAVAILABLE',
                           'detail': str(e), 'sample': cache}
                break
            governor, risk = h2b_tick(residual, goal, governor, enabled)
        effective = governor['effective']
        if quiet and (risk['intervention'] or effective != goal):
            blocked = {'step': i, 'reason': 'PREPUSH_INTERVENTION',
                       'sample': cache, 'risk': risk, 'governor': governor}
            break
        state, diag = step_fn(state, jnp.asarray(effective, dtype=command.dtype))
        data = jax.device_get(diag)
        if not bool(data['h2a_state_finite']):
            raise RuntimeError('H2B_NONFINITE_TRANSITION')
        row = {}
        for key, val in data.items():
            a = np.asarray(val)
            if not np.isfinite(a).all():
                raise RuntimeError('H2B_NONFINITE_DIAG '+key)
            row[key] = float(a) if a.ndim == 0 else a.tolist()
        row.update({'step': i, 'done': bool(data['done']),
                    'goal_command': list(goal), 'effective_command': list(effective),
                    'governor_mode': governor['mode'], 'governor_age': governor['age'],
                    'h2b_sample': cache, **risk})
        frames.append(row)
        if row['done'] or i == 1000:
            break
        # Exactly one new sensor sample after completed physics; use previous
        # sample for derivative. Never sample on a kick or branch boundary.
        cache = sampler(state, i, cache)
        if i == capture_step:
            snapshot = h2b_fork((state, cache, governor), jax)
    return {'frames': frames, 'ctx': (state, cache, governor),
            'snapshot': snapshot, 'blocked': blocked}


def h2b_summary(frames, max_steps):
    if not frames:
        raise RuntimeError('H2B_EMPTY_TRACE')
    stats = aggregate_frames(frames, bool(frames[-1]['done']), len(frames), max_steps)
    stats.update({
        'intervention_steps': sum(bool(r['intervention']) for r in frames),
        'timeout_events': sum(bool(r['timeout_event']) for r in frames),
        'joint_limit_excess_max': max(r['h2a_joint_limit_excess'] for r in frames),
        'ctrl_limit_excess_max': max(r['h2a_ctrl_limit_excess'] for r in frames),
        'force_limit_excess_max': max(r['h2a_force_limit_excess'] for r in frames),
    })
    return stats


def h2b_calibrate(args, env, sampler, step_fn, command, seed, profile, jax, jnp, np):
    goal = tuple(map(float, jax.device_get(command)))
    state = env.reset(jax.random.PRNGKey(seed))
    ctx = state, sampler(state, 0), h2a_initial_state(goal)
    run = h2b_advance(ctx, 1, 1000, profile=profile, goal=goal, command=command,
                     enabled=False, reference=None, sampler=sampler,
                     step_fn=step_fn, jax=jax, jnp=jnp, np=np)
    frames = run['frames']
    blocked = run['blocked']
    if len(frames) != 1000 or frames[-1]['done']:
        blocked = {'reason': 'CALIBRATION_NOMINAL_TERMINATION'}
    return {'seed': seed, 'profile': profile, 'goal_command': list(goal),
            'frames': frames, 'blocked': blocked,
            'summary': h2b_summary(frames, args.max_steps)}


def h2b_evaluate_group(args, env, sampler, step_fn, command, seed, profile,
                       reference, jax, jnp, np):
    goal = tuple(map(float, jax.device_get(command)))
    state = env.reset(jax.random.PRNGKey(seed))
    initial = state, sampler(state, 0), h2a_initial_state(goal)
    common = dict(profile=profile, goal=goal, command=command, reference=reference,
                  sampler=sampler, step_fn=step_fn, jax=jax, jnp=jnp, np=np)
    prefix = h2b_advance(initial, 1, 299, enabled=True, quiet=True, **common)
    pre = prefix['frames']
    result = {'seed': seed, 'profile': profile, 'goal_command': list(goal),
              'shared_prefix': pre, 'blocked': prefix['blocked'], 'arms': {}}
    if result['blocked'] is not None:
        return result
    if len(pre) != 299 or pre[-1]['done']:
        result['blocked'] = {'reason': 'NOMINAL_TERMINATED_BEFORE_PUSH'}
        return result
    heading = yaw_from_root_quat(np.asarray(jax.device_get(prefix['ctx'][0].data.qpos[3:7])))
    result['first_heading_rad'] = heading

    def initial_branch(enabled):
        state, cache, gov = h2b_fork(prefix['ctx'], jax)
        return state, cache, gov if enabled else h2a_initial_state(goal)

    def record_block(run, arm_name, kind, direction=None):
        if run['blocked'] is None:
            return False
        result['blocked'] = {**run['blocked'], 'arm': arm_name,
                             'kind': kind, 'direction': direction}
        return True

    for enabled in (False, True):
        name = 'ON' if enabled else 'OFF'
        arm = {'directions': {}}
        result['arms'][name] = arm
        nominal = h2b_advance(initial_branch(enabled), 300, 1000,
                              enabled=enabled, **common)
        arm['nominal_frames'] = nominal['frames']
        if record_block(nominal, name, 'nominal'):
            return result
        nf = pre + nominal['frames']
        arm['nominal_summary'] = h2b_summary(nf, args.max_steps)
        for direction in args.directions:
            d1 = new_body_delta(direction, heading, args.push_magnitude)
            d2 = new_body_delta(direction, heading, args.second_push_magnitude)
            item = {'first_delta_world_mps': list(d1), 'second_delta_world_mps': list(d2)}
            arm['directions'][direction] = item
            single = h2b_advance(h2b_kick(initial_branch(enabled), d1, jnp),
                                 300, 1000, enabled=enabled, capture_step=334, **common)
            item['single_frames'] = single['frames']
            if record_block(single, name, 'single', direction):
                return result
            sf = pre + single['frames']
            item['single_summary'] = h2b_summary(sf, args.max_steps)
            for sec, ticks in ((2, 100), (5, 250)):
                item[f'single_recovery_{sec}s'] = h2a_recovery(nf, sf, 300, ticks)
            exposed = single['snapshot'] is not None
            item['second_push_exposed'] = exposed
            if exposed:
                double = h2b_advance(h2b_kick(h2b_fork(single['snapshot'], jax), d2, jnp),
                                     335, 1000, enabled=enabled, **common)
                item['double_frames_after_second'] = double['frames']
                if record_block(double, name, 'double', direction):
                    return result
                df = sf[:334] + double['frames']
            else:
                if not sf[-1]['done'] or len(sf) > 334:
                    raise RuntimeError('H2B_MISSING_SECOND_SNAPSHOT')
                item['double_frames_after_second'] = []
                df = sf
            item['double_summary'] = h2b_summary(df, args.max_steps)
            for sec, ticks in ((2, 100), (5, 250)):
                item[f'double_recovery_{sec}s'] = (
                    h2a_recovery(nf, df, 335, ticks) if exposed else
                    {'success': False, 'reason': 'terminated_before_second'})
    return result


def h2b_write_progress(path, report, initial=False):
    """Only update THIS run's report. Initial creation exclusive; atomic later updates."""
    path = Path(path)
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if initial:
        with path.open('x', encoding='utf8') as f:
            f.write(payload)
        return
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('x', encoding='utf8') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def h2b_freeze_reference(path, calibration):
    payload = {'format': 'M470_H2B_REFERENCE_V1', 'source_commit': PIN,
               'checkpoint_sha256': BASE_SHA, 'config': H2B_CONFIG,
               'calibration_seeds': list(H2B_CAL_SEEDS),
               'profiles': list(H2B_PROFILES), 'bins': h2b_fit_reference(calibration)}
    canonical = (json.dumps(payload, sort_keys=True, separators=(',', ':'),
                            allow_nan=False)+'\n').encode('utf8')
    with Path(path).open('xb') as f:
        f.write(canonical)
    digest = hashlib.sha256(canonical).hexdigest()
    if sha256(path) != digest:
        raise RuntimeError('H2B_REFERENCE_SHA_MISMATCH')
    return digest


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
    report_p = None

    if not (args.h2a or args.h2b):
        report_p = read_m463_smoke_report(
            args.report_p,
            args.checkpoint_p,
        )

        if report_p.get(
            'instrumented_brax', {}
        ).get('original_brax_sha256') != TRAIN_SHA:
            raise RuntimeError(
                'M463_GATE3_TRAIN_BRAX_MISMATCH'
            )

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
    policies = {
        'C': load_policy(args.checkpoint_c)
    }

    if not (args.h2a or args.h2b):
        policies['P'] = load_policy(
            args.checkpoint_p
        )
    results=[]
    if args.h2b:
        reference_path = args.output.with_suffix('.reference.json')
        if reference_path.exists() or args.output.with_name(args.output.name+'.tmp').exists():
            raise FileExistsError('H2B output/reference already exists')
        sampler = h2b_make_sampler(env, jax, jnp, np, collision)
        session = {
            'status': 'M470_H2B_IN_PROGRESS',
            'source_commit': PIN, 'brax_train_sha256': TRAIN_SHA,
            'checkpoint_sha256': BASE_SHA,
            'evaluator_sha256': sha256(Path(__file__)),
            'jax_version': jax.__version__, 'training_performed': False,
            'deterministic_actor': True, 'random_source_push_enabled': False,
            'domain_randomization_enabled': False,
            'governor_config': H2B_CONFIG,
            'calibration_seeds': list(H2B_CAL_SEEDS), 'seeds': args.seeds,
            'profiles': args.profiles, 'directions': args.directions,
            'push_step': args.push_step, 'push_magnitude_mps': args.push_magnitude,
            'second_push_step': args.second_push_step,
            'second_push_magnitude_mps': args.second_push_magnitude,
            'max_steps': args.max_steps, 'dt_s': float(env.dt),
            'reference_sha256': None, 'calibration': [], 'results': results,
            'expected_calibration_trajectories': 6,
            'expected_logical_trajectories': 60,
        }
        h2b_write_progress(args.output, session, initial=True)
        run_seeds = list(H2B_CAL_SEEDS) + args.seeds
    else:
        run_seeds = args.seeds
    for seed in run_seeds:
      for profile in args.profiles:
        command=jnp.asarray(PROFILES[profile],dtype=jnp.float32)
        for name,policy in policies.items():
            # JIT step/diagnostics; a single MJX state (no vectorized envs) is active at a time.
            def one_step(state, effective_command=None):
                cmd = (
                    command
                    if effective_command is None
                    else effective_command
                )

                info = dict(state.info)
                info['command'] = cmd

                obs = dict(state.obs)           

                obs['state'] = obs['state'].at[9:12].set(cmd)

                obs['privileged_state'] = (
                    obs['privileged_state'].at[9:12].set(cmd)
                )
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

                if args.h2a or args.h2b:
                    model = env.mj_model

                    def limit_excess(values, limits, mask):
                        limits = jnp.asarray(limits)
                        mask = jnp.asarray(mask, dtype=bool)

                        excess = jnp.maximum(
                            limits[:, 0] - values,
                            values - limits[:, 1],
                        )

                        return jnp.max(
                            jnp.where(
                                mask,
                                jnp.maximum(excess, 0.0),
                                0.0,
                            )
                        )

                    diag.update({
                        'h2a_action': action,

                        'h2a_joint_limit_excess': limit_excess(
                            d.qpos[7:],
                            model.jnt_range[1:],
                            model.jnt_limited[1:],
                        ),

                        'h2a_ctrl_limit_excess': limit_excess(
                            d.ctrl,
                            model.actuator_ctrlrange,
                            model.actuator_ctrllimited,
                        ),

                        'h2a_force_limit_excess': limit_excess(
                            d.actuator_force,
                            model.actuator_forcerange,
                            model.actuator_forcelimited,
                        ),

                        'h2a_state_finite': (
                            jnp.isfinite(d.qpos).all()
                            & jnp.isfinite(d.qvel).all()
                            & jnp.isfinite(action).all()
                            & jnp.isfinite(d.ctrl).all()
                            & jnp.isfinite(d.actuator_force).all()
                        ),
                    })

                return next_state,diag
            step_fn=jax.jit(one_step)
            if args.h2b:
                if seed in H2B_CAL_SEEDS:
                    group = h2b_calibrate(args, env, sampler, step_fn, command,
                                          seed, profile, jax, jnp, np)
                    session['calibration'].append(group)
                    if group['blocked'] is not None:
                        session['status'] = 'M470_H2B_CALIBRATION_BLOCKED'
                    h2b_write_progress(args.output, session)
                    print('M470_H2B_CALIBRATION_GROUP', seed, profile,
                          group['blocked'], flush=True)
                    if group['blocked'] is not None:
                        raise RuntimeError('M470_H2B_CALIBRATION_NO_GO')
                    if len(session['calibration']) == 6:
                        digest = h2b_freeze_reference(reference_path, session['calibration'])
                        session['reference_sha256'] = digest
                        h2b_write_progress(args.output, session)
                        print('M470_H2B_REFERENCE_FROZEN', digest, flush=True)
                else:
                    if len(session['calibration']) != 6 or session['reference_sha256'] is None:
                        raise RuntimeError('H2B_EVALUATION_BEFORE_FREEZE')
                    if sha256(reference_path) != session['reference_sha256']:
                        raise RuntimeError('H2B_REFERENCE_CHANGED')
                    frozen = json.loads(reference_path.read_text(encoding='utf8'))
                    group = h2b_evaluate_group(args, env, sampler, step_fn, command,
                                               seed, profile, frozen['bins'], jax, jnp, np)
                    results.append(group)
                    if group['blocked'] is not None:
                        reason = group['blocked']['reason']
                        session['status'] = ('M470_H2B_REFERENCE_INCONCLUSIVE'
                            if reason == 'REFERENCE_UNAVAILABLE' else 'M470_H2B_EVALUATION_BLOCKED')
                    h2b_write_progress(args.output, session)
                    print('M470_H2B_EVALUATION_GROUP', seed, profile,
                          group['blocked'], flush=True)
                    if group['blocked'] is not None:
                        raise RuntimeError('M470_H2B_BOUNDED_STOP')
                continue
            if args.h2b:
                if len(session['calibration']) != 6 or len(results) != 6:
                    raise RuntimeError('H2B_INCOMPLETE_GROUPS')
                if sha256(reference_path) != session['reference_sha256']:
                    raise RuntimeError('H2B_REFERENCE_CHANGED')
                if sha256(args.checkpoint_c) != BASE_SHA:
                    raise RuntimeError('H2B_BASELINE_CHANGED')
                session['status'] = 'M470_H2B_EVALUATION_COMPLETED'
                h2b_write_progress(args.output, session)
                print(session['status'], args.output, flush=True)
                return
            if args.h2a:
                result = h2a_evaluate_group(
                    args,
                    env,
                    step_fn,
                    command,
                    seed,
                    profile,
                    jax,
                    jnp,
                    np,
                )

                results.append(result)

                print(
                    'M470_H2A_GROUP',
                    seed,
                    profile,
                    'blocked', result['blocked'],
                    flush=True,
                )

                if result['blocked'] is not None:
                    return h2a_write(
                        args,
                        results,
                        jax.__version__,
                        blocked=True,
                    )

                continue
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
                        before_second = pushed[
                            args.second_push_step - args.push_step - 1
                        ]                   

                        second['pre_second_push_support'] = {
                            'step': args.second_push_step - 1,
                            **{
                                key: before_second[key]
                                for key in (
                                    'lf_ground_contact',
                                    'rf_ground_contact',
                                    'phase_left_post_rad',
                                    'phase_right_post_rad',
                                    'height',
                                    'tilt_deg',
                                )
                            },
                        }
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
    if args.h2b:
        if len(session['calibration']) != 6 or len(results) != 6:
            raise RuntimeError('H2B_INCOMPLETE_GROUPS')
        if sha256(reference_path) != session['reference_sha256']:
            raise RuntimeError('H2B_REFERENCE_CHANGED')
        if sha256(args.checkpoint_c) != BASE_SHA:
            raise RuntimeError('H2B_BASELINE_CHANGED')
        session['status'] = 'M470_H2B_EVALUATION_COMPLETED'
        h2b_write_progress(args.output, session)
        print(session['status'], args.output, flush=True)
        return
    if args.h2a:
        if len(results) != 6:
            raise RuntimeError(
                'M470_H2A_EXPECTED_SIX_GROUPS'
            )

        return h2a_write(
            args,
            results,
            jax.__version__,
        )
    outcome={'status':'M463_GATE3_CP_FIXED_EVALUATION_COMPLETED',
        'class':'EXPLORATORY_MJX_GPU_ONLY_NOT_PAPER_EXACT_NOT_SIM_TO_SIM',
        'source_commit':PIN,'brax_train_sha256':TRAIN_SHA,
        'checkpoint_sha256':{'C':sha256(args.checkpoint_c),'P':sha256(args.checkpoint_p)},
        'train_steps_P': report_p['steps_observed'],
        'policy_P_is': (
            'M470_INDEPENDENT_C401M_PLUS10M'
            if report_p['mode'] == 'longtrain'
            else 'M470H_PROBE2'
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
    p.add_argument('--checkpoint-p', type=Path)
    p.add_argument('--report-p', type=Path)
    p.add_argument('--h2b', action='store_true',
                   help='Support-aware DCM command residual, frozen C401M')
    p.add_argument(
        '--h2a',
        action='store_true',
        help='Frozen C401M recovery governor comparison',
    )
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
    if args.h2a and args.h2b:
        p.error('H2A and H2B are mutually exclusive')
    if args.h2b:
        if (args.seeds != list(H2B_DEV_SEEDS)
                or args.profiles != list(H2B_PROFILES)
                or args.directions != ['left', 'right']
                or args.push_step != 300 or args.push_magnitude != 0.50
                or args.second_push_step != 335 or args.second_push_magnitude != 0.35
                or args.max_steps != 1000):
            p.error('M470_H2B_FROZEN_PROTOCOL_MISMATCH')
        if args.checkpoint_p is not None or args.report_p is not None:
            p.error('H2B uses frozen C401M only')
        if args.mode == 'plan':
            print('M470_H2B_PLAN', json.dumps({
                'calibration_seeds': list(H2B_CAL_SEEDS), 'evaluation_seeds': args.seeds,
                'calibration_trajectories': 6, 'logical_trajectories': 60,
                'max_physical_transitions': 49014, 'training': False,
                'governor_config': H2B_CONFIG, 'output': str(args.output),
            }))
            return
        if not args.checkpoint_c.is_file():
            p.error('Missing frozen C401M checkpoint')
        run(args)
        return
    if args.h2a:
        if (
            args.seeds != [7331, 7357]
            or args.profiles != [
                'stand',
                'forward_fast',
                'turn_left',
            ]
            or args.directions != ['left', 'right']
            or args.push_step != 300
            or args.push_magnitude != 0.50
            or args.second_push_step != 335
            or args.second_push_magnitude != 0.35
            or args.max_steps != 1000
        ):
            p.error('M470_H2A_FROZEN_PROTOCOL_MISMATCH')

        if (
            args.checkpoint_p is not None
            or args.report_p is not None
        ):
            p.error('H2A uses frozen C401M only')

        if args.mode == 'plan':
            print(
                'M470_H2A_PLAN',
                json.dumps({
                    'groups': 6,
                    'logical_trajectories': 60,
                    'training': False,
                    'governor_config': H2A_CONFIG,
                    'output': str(args.output),
                }),
            )
            return

        if not args.checkpoint_c.is_file():
            p.error('Missing frozen C401M checkpoint')

        run(args)
        return

    if args.checkpoint_p is None or args.report_p is None:
        p.error(
            'Legacy evaluation requires '
            '--checkpoint-p and --report-p'
        )

    if args.mode=='plan':
        print('M463_GATE3_PLAN',json.dumps({'cases':count,'seeds':args.seeds,'profiles':args.profiles,
            'directions':args.directions,'max_steps':args.max_steps,'push_step':args.push_step,
            'push_magnitude_mps':args.push_magnitude,'one_MJX_env_at_time':True,
            'no_training':True,'output':str(args.output)}));return
    for path in (args.checkpoint_c,args.checkpoint_p,args.report_p):
        if not path.is_file():p.error('missing input: '+str(path))
    run(args)

if __name__=='__main__':main()
