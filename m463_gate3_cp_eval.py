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


# M470-H3-M1-3: frozen three-policy, original-source controlled screen.
# Isolated from the legacy and H2 evaluators; no optimizer or governor.
H3_EVAL_SEEDS = (7331, 7357, 7601, 7613)
H3_EVAL_PROFILES = ('stand', 'forward_fast', 'turn_left')
H3_EVAL_POLICIES = ('C', 'OLD', 'H3')
H3_EVAL_LOGICAL = 180
H3_EVAL_PHYSICAL = 134424
H3_EVAL_SHA = 'aaa4e14b0142d83843892105e47cfe25aed655bc71c20e04ac4ac4239dbb75a4'
H3_EVAL_REPORT_SHA = '84c285315e109ca8716476b0f527873046b441d35e3fe963570e5f2019fe3530'
H3_EVAL_MANIFEST_SHA = 'b7e99cf8a440ce2991d972b4edba93c0c709589afe73290acf2308c7dce6bcb5'
H3_EVAL_FULLSTATE_SHA = '2a2fba49a6ceb940dd1ee8c72dd1858d288172fbd022acd5bedd625ff694dc68'
H3_EVAL_OLD_REPORT_SHA = '04f7c71b3dd4b9d24e06b13991a9e4bb4e6c482d217929a74aa40537e9b62f74'
H3_EVAL_FLAGS = ('cause_inverted_torso', 'cause_foot_foot',
                'cause_left_foot_right_shin', 'cause_right_foot_left_shin',
                'cause_nan_qpos', 'cause_nan_qvel')
H3_EVAL_LIMITS = {'h2a_joint_limit_excess': .001,
                 'h2a_ctrl_limit_excess': .001,
                 'h2a_force_limit_excess': .1}


def h3_require(ok, label):
    if not ok:
        raise RuntimeError('M470_H3_EVAL_'+label)


def h3_validate_protocol(args):
    h3_require(args.h3_m1_eval and not args.h2a and not args.h2b, 'MODE_CONFLICT')
    h3_require(tuple(args.seeds)==H3_EVAL_SEEDS
               and tuple(args.profiles)==H3_EVAL_PROFILES
               and tuple(args.directions)==('left','right')
               and (args.push_step,args.push_magnitude,args.second_push_step,
                    args.second_push_magnitude,args.max_steps,args.max_cases)
                   ==(300,.50,335,.35,1000,H3_EVAL_LOGICAL), 'FROZEN_PROTOCOL')
    h3_require(all(p is not None for p in
                   (args.checkpoint_c,args.checkpoint_p,args.report_p,
                    args.checkpoint_h3,args.report_h3)), 'MISSING_INPUT')
    h3_require(36*(1000+2*(701+666))==H3_EVAL_PHYSICAL,'BUDGET_FORMULA')


def h3_read_inputs(args):
    """Byte-pinned evidence; OLD10M still uses the unmodified legacy validator."""
    from m463_push_ppo_train import H3_SPEC,H3_FP,h3_audit_counts,m470_fingerprint
    h3_validate_protocol(args)
    manifest=args.checkpoint_h3.parent/'m470_fullstate_step_010485760.json'
    pins={'C_policy':(args.checkpoint_c,BASE_SHA),
          'OLD_policy':(args.checkpoint_p,M470_10M_SHA),
          'OLD_report':(args.report_p,H3_EVAL_OLD_REPORT_SHA),
          'H3_policy':(args.checkpoint_h3,H3_EVAL_SHA),
          'H3_report':(args.report_h3,H3_EVAL_REPORT_SHA),
          'H3_manifest':(manifest,H3_EVAL_MANIFEST_SHA)}
    for k,(p,expected) in pins.items():
        h3_require(p.is_file() and not p.is_symlink() and sha256(p)==expected,'INPUT_'+k)
    old=read_m463_smoke_report(args.report_p,args.checkpoint_p)
    r=json.loads(args.report_h3.read_bytes())
    m=json.loads(manifest.read_bytes())
    h3_require(r['status']=='M463_COVERAGE_LONGTRAIN_PASS'
               and r['experiment_design']=='INDEPENDENT_C401M_SINGLE_PROCESS'
               and r['stage']=='easy_repeat_support' and r['independent_budget']=='10M'
               and r['trained_from']=='C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE'
               and r['source_commit']==PIN and r['source_checkpoint_sha256']==BASE_SHA
               and r['orbax_sha256']==old['orbax_sha256']
               and r['instrumented_brax']['original_brax_sha256']==TRAIN_SHA
               and r['instrumented_brax']['instrumented_sha256']=='0ff25f52fac87ff112579ad7f91f35ffa5b0a2d654e22d74c45be26dd0850f7a'
               and r['runner_sha256']=='56dad9056cd1bef579839eddbf9e21f2117b8634c190f833606736014667b56c'
               and r['training_seed']==1 and r['cumulative_steps_start']==0
               and r['resume_fullstate_sha256'] is None
               and r['train_steps_target']==r['steps_observed']==r['run_steps_observed']==10485760
               and r['rollouts_target']==40 and r['rollouts_per_epoch']==8
               and r['hardware_scaled_config']==old['hardware_scaled_config']
               and m470_fingerprint(r['hardware_scaled_config'])==H3_FP
               and r['h3_m1_enabled'] is True and r['h3_m1_spec']==H3_SPEC
               and r['h1_m1_enabled'] is False and r['h1_m2_enabled'] is False
               and r['training_performed'] is True and r['weights_finite'] is True
               and r['final_tensor_count']==25 and r['callback_vs_final_equal'] is True
               and r['coverage_gate_pass'] is True and r['memory_gate_pass'] is True
               and r['eval_source_reward_unchanged'] is True
               and r['PD_termination_source_unchanged'] is True
               and old['reward_PD_termination_source_unchanged'] is True
               and r['gpu_total_mib']==12288 and 0<r['gpu_peak_mib']<.95*12288
               and r['gpu_monitor_errors']==[], 'H3_REPORT_CONTRACT')
    a=r['h3_independent_acceptance_basis']
    h3_require(a['old10m_report_sha256']==H3_EVAL_OLD_REPORT_SHA
               and a['old10m_policy_sha256']==M470_10M_SHA
               and a['longtrain_brax_comparison']['normalized_generated_matches_old10m'] is True
               and a['historical_exact_gate']=='FAILED_85891a1'
               and a['diagnostic_control_transitions_reexecuted']==0
               and a['independent_executable_exact_equal'] is False,'H3_COMPARATOR')
    audit=h3_audit_counts(r['coverage']['counts'])
    h3_require(audit['passed'] and audit==r['h3_m1_training_transition_audit'],'H3_AUDIT')
    steps=list(range(2097152,10485761,2097152))
    h3_require([x['steps'] for x in r['checkpoints']]==steps
               and [x['steps'] for x in r['fullstate_checkpoints']]==steps
               and r['checkpoints'][-1]['file']==args.checkpoint_h3.name
               and r['checkpoints'][-1]['sha256']==H3_EVAL_SHA
               and r['fullstate_checkpoints'][-1]['sha256']==H3_EVAL_FULLSTATE_SHA
               and m['fullstate_sha256']==H3_EVAL_FULLSTATE_SHA
               and m['cumulative_steps']==10485760 and m['parent_resume_sha256'] is None
               and m['h3_m1_enabled'] is True and m['h3_m1_spec']==H3_SPEC
               and m['source_checkpoint_sha256']==BASE_SHA
               and m['runner_sha256']==r['runner_sha256']
               and m['config_fingerprint']==H3_FP
               and m['instrumented_brax_sha256']==r['instrumented_brax']['instrumented_sha256'],
               'H3_CHECKPOINT_MANIFEST')
    return {k:expected for k,(_,expected) in pins.items()}


def h3_write(path, report, initial=False):
    # Compact raw scalar rows, shared prefixes stored only once.
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    if initial:
        h3_require(not path.exists() and not tmp.exists(),'OUTPUT_EXISTS')
        target=path
    else:
        target=tmp
    with target.open('x',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    if not initial:
        os.replace(tmp,path)


def h3_gpu_check(monitor):
    g=monitor.snapshot()
    h3_require(g['total_mib']==12288 and g['peak_used_mib'] is not None
               and 0<g['peak_used_mib']<.95*12288 and g['samples']>0
               and not g['sampler_errors'],'GPU_GATE')
    return g


def h3_group(args,env,step_fn,jax,jnp,np,session,name,seed,profile,
             physical_cap=H3_EVAL_PHYSICAL):
    """One nominal rollout, two single suffixes, two double suffixes at most."""
    group={'policy':name,'seed':seed,'profile':profile,'command':list(PROFILES[profile]),
           'nominal_frames':[],'directions':{},'complete':False}
    session['groups'].append(group)
    start=session['physical_transitions_attempted']
    def clone(st):
        return jax.tree_util.tree_map(lambda x:x,st)
    def advance(st,first,last,rows,capture_step=None):
        snap=None
        for i in range(first,last+1):
            h3_require(session['physical_transitions_attempted']<physical_cap,'PHYSICAL_BUDGET')
            session['physical_transitions_attempted']+=1
            st,diag=step_fn(st)
            raw=jax.device_get(diag)
            h3_require(bool(raw['h2a_state_finite']) and bool(raw['h3_finite']),'NONFINITE_STATE')
            row={}
            for k,v in raw.items():
                a=np.asarray(v)
                h3_require(a.ndim==0 and np.isfinite(a).all(),'NONFINITE_OR_NONSCALAR_'+k)
                row[k]=float(a)
            h3_require(row['done'] in (0.,1.) and
                       bool(row['done'])==any(bool(row[k]) for k in H3_EVAL_FLAGS),
                       'TERMINATION_FLAGS')
            h3_require(all(row[k] in (0.,1.) for k in ('lf_ground_contact','rf_ground_contact')),
                       'CONTACT_SCHEMA')
            cols=sorted(row)
            if session['frame_columns'] is None:
                session['frame_columns']=cols
            h3_require(cols==session['frame_columns'],'FRAME_SCHEMA')
            rows.append([row[k] for k in cols])
            session['physical_transitions_recorded']+=1
            if row['done']:
                break
            if i==capture_step:
                snap=clone(st)
            if i%100==0:
                session['gpu']=h3_gpu_check(args.h3_monitor)
        return snap
    def kick(snapshot,direction,magnitude,step):
        st=clone(snapshot)
        heading=yaw_from_root_quat(jax.device_get(st.data.qpos[3:7]))
        requested=new_body_delta(direction,heading,magnitude)
        before=np.asarray(jax.device_get(st.data.qvel))
        qvel=st.data.qvel.at[:2].add(jnp.asarray(requested,dtype=st.data.qvel.dtype))
        after=np.asarray(jax.device_get(qvel))
        actual=(after[:2].astype(np.float64)-before[:2].astype(np.float64))
        h3_require(np.isfinite(actual).all() and np.array_equal(before[2:],after[2:])
                   and np.allclose(actual,requested,atol=2e-6,rtol=0),'ACTUAL_KICK')
        return st.replace(data=st.data.replace(qvel=qvel)), {
            'exposed':True,'step':step,'heading_rad':heading,
            'requested_world_dqvel_xy_mps':list(requested),
            'actual_world_dqvel_xy_mps':actual.tolist(),
            'actual_magnitude_mps':float(np.linalg.norm(actual)),
            'full_state_rng_fork_by_construction':True}
    st=env.reset(jax.random.PRNGKey(seed))
    h3_require(st.obs['state'].shape==(103,) and st.obs['privileged_state'].shape==(216,)
               and not bool(jax.device_get(st.done)), 'RESET_SCHEMA')
    snapshot=advance(st,1,1000,group['nominal_frames'],299)
    for direction in args.directions:
        item={'first':{'exposed':False,'step':300,'reason':'nominal_terminated_before_push'},
              'second':{'exposed':False,'step':335,'reason':'first_not_exposed'},
              'single_suffix':[],'double_suffix':[]}
        group['directions'][direction]=item
        if snapshot is None:
            continue
        first,item['first']=kick(snapshot,direction,.50,300)
        second_snapshot=advance(first,300,1000,item['single_suffix'],334)
        item['second']['reason']='single_terminated_before_second'
        if second_snapshot is not None:
            second,item['second']=kick(second_snapshot,direction,.35,335)
            advance(second,335,1000,item['double_suffix'])
    group['physical_transitions']=session['physical_transitions_attempted']-start
    h3_require(group['physical_transitions']==len(group['nominal_frames'])+sum(
        len(v['single_suffix'])+len(v['double_suffix']) for v in group['directions'].values())
        and group['physical_transitions']<=3734,'GROUP_BUDGET')
    group['complete']=True
    session['logical_trajectories_completed']+=5
    return group


def h3_decode(group,columns):
    """Reconstruct logical full traces; never count reused prefixes as physical."""
    def rows(values):
        return [dict(zip(columns,row)) for row in values]
    nom=rows(group['nominal_frames'])
    h3_require(bool(nom),'EMPTY_NOMINAL')
    traces={('nominal','none'):(nom,True)}
    for d in ('left','right'):
        item=group['directions'][d]
        f=bool(item['first']['exposed'])
        e=bool(item['second']['exposed'])
        single=nom[:299]+rows(item['single_suffix']) if f else nom
        double=single[:334]+rows(item['double_suffix']) if e else single
        h3_require((not e or f) and (not f or (len(nom)>=299 and not nom[298]['done']))
                   and (not e or (len(single)>=334 and not single[333]['done']))
                   and bool(item['single_suffix'])==f and bool(item['double_suffix'])==e,
                   'FORK_LENGTH')
        for trace in (single,double):
            h3_require(0<len(trace)<=1000 and
                       (len(trace)==1000 or bool(trace[-1]['done'])) and
                       not any(bool(x['done']) for x in trace[:-1]),'TRACE_END')
        traces[('single',d)]=(single,f)
        traces[('double',d)]=(double,e)
    h3_require(len(nom)<=1000 and (len(nom)==1000 or bool(nom[-1]['done']))
               and not any(bool(x['done']) for x in nom[:-1]),'NOMINAL_END')
    return traces


def h3_recovery(control,trace,step,window,exposed):
    a,b=control[step-1:step-1+window],trace[step-1:step-1+window]
    if not exposed or len(a)!=window or len(b)!=window or a[-1]['done'] or b[-1]['done']:
        return {'recovered':False,'comparable':False,'recovery_s':None,
                'reason':'nonexposed_or_terminated_within_window',
                'window_steps':window,'n_common_steps':min(len(a),len(b))}
    return {**matched_off_recovery(a,b,window), 'comparable':True}


def h3_summary(trace,exposed,kind,control):
    step={'nominal':None,'single':300,'double':335}[kind]
    out=aggregate_frames(trace,bool(trace[-1]['done']),len(trace),1000,step if exposed else None)
    if step is not None and not exposed:
        out['termination_within_2s']=None
        out['termination_within_5s']=None
    out['exposed']=exposed
    out['survived']=bool(exposed and out['completed'])
    out['full_observed_safety_max']={k:max(f[k] for f in trace) for k in H3_EVAL_LIMITS}
    if step is not None:
        for sec,window in ((2,100),(5,250)):
            out['recovery_'+str(sec)+'s']=h3_recovery(control,trace,step,window,exposed)
            a,b=control[step-1:],trace[step-1:]
            # A terminal frame at the last required tick is not a full valid window.
            out['tracking_'+str(sec)+'s']=(fixed_window_tracking(a,b,window)
                if exposed and len(a)>=window and len(b)>=window
                and not a[window-1]['done'] and not b[window-1]['done'] else
                {'comparable':False,'reason':'nonexposed_or_early_termination','window_steps':window})
    return out


def h3_tracking_limit(old,new):
    return new<=old+(.0005 if old<.01 else .05*old)


def h3_standing(trace,first):
    x=trace[first-1:]
    h3_require(len(x)==1001-first,'STANDING_WINDOW')
    n=len(x)
    mean=lambda k:sum(f[k] for f in x)/n
    return {'first_step':first,'last_step':1000,'samples':n,
            'pose_l1_mean':mean('h3_pose_l1'),
            'swing_fraction':sum((2-f['lf_ground_contact']-f['rf_ground_contact'])/2 for f in x)/n,
            'xy_rmse':math.sqrt(sum(f['xy_err']**2 for f in x)/n),
            'yaw_rmse':math.sqrt(sum(f['yaw_err']**2 for f in x)/n),
            'contact_transitions':{k:sum(a[k]!=b[k] for a,b in zip(x,x[1:]))
                                   for k in ('lf_ground_contact','rf_ground_contact')},
            'root_xy_path_m':sum(math.hypot(b['h3_root_x']-a['h3_root_x'],
                                            b['h3_root_y']-a['h3_root_y'])
                                 for a,b in zip(x,x[1:]))}


def h3_decide(session, *, policies=H3_EVAL_POLICIES, comparator='OLD', candidate='H3',
              logical_cap=H3_EVAL_LOGICAL, physical_cap=H3_EVAL_PHYSICAL):
    # H3 defaults preserve the original labels, case ordering and gates.
    only_old=comparator+'_ONLY'; only_new=candidate+'_ONLY'
    groups={(g['seed'],g['profile'],g['policy']):g for g in session['groups']}
    expected={(s,p,n) for s in H3_EVAL_SEEDS for p in H3_EVAL_PROFILES for n in policies}
    h3_require(set(groups)==expected and len(session['groups'])==len(expected)
               and all(g['complete'] for g in groups.values()),'GROUP_COMPLETENESS')
    h3_require(session['logical_trajectories_completed']==logical_cap
               and session['physical_transitions_attempted']==session['physical_transitions_recorded']
               ==sum(g['physical_transitions'] for g in groups.values())
               and session['physical_transitions_recorded']<=physical_cap,'FINAL_BUDGET')
    summaries=[]
    paired=[]
    hard=[]
    primary={'survival':{comparator:0,candidate:0},'recovery5':{comparator:0,candidate:0}}
    moving={p:{comparator:0,candidate:0} for p in H3_EVAL_PROFILES if p!='stand'}
    standing={k:0 for k in ('nominal','single','double')}
    nominal={n:0 for n in policies}
    for seed in H3_EVAL_SEEDS:
        for profile in H3_EVAL_PROFILES:
            traces={n:h3_decode(groups[(seed,profile,n)],session['frame_columns']) for n in policies}
            for kind,d in [('nominal','none')]+[(k,d) for d in ('left','right') for k in ('single','double')]:
                row={'seed':seed,'profile':profile,'kind':kind,'direction':d,'policies':{}}
                for n in policies:
                    trace,exposed=traces[n][(kind,d)]
                    control=traces[n][('single',d) if kind=='double' else ('nominal','none')][0]
                    summary=h3_summary(trace,exposed,kind,control)
                    row['policies'][n]=summary
                    if kind=='nominal':
                        nominal[n]+=int(summary['survived'])
                o,h=row['policies'][comparator],row['policies'][candidate]
                old_t,new_t=traces[comparator][(kind,d)][0],traces[candidate][(kind,d)][0]
                key=[seed,profile,kind,d]
                n=min(len(old_t),len(new_t))
                safety={}
                for k,tol in H3_EVAL_LIMITS.items():
                    x=max(f[k] for f in old_t[:n]); y=max(f[k] for f in new_t[:n])
                    safety[k]={comparator:x,candidate:y,'tolerance':tol,'pass':y<=x+tol}
                    if not safety[k]['pass']:
                        hard.append({'case':key,'gate':k})
                flags=lambda summary:{k for k,v in summary.get('terminal_source_flags',{}).items() if v}
                new_flags=sorted(flags(h)-flags(o))
                if new_flags:
                    hard.append({'case':key,'gate':'new_critical_terminal_flags','flags':new_flags})
                lost=o['survived'] and not h['survived']
                if lost and (kind=='single' or (kind=='double' and profile!='stand')):
                    hard.append({'case':key,'gate':'lost_survival'})
                comparison={'case':key,'common_prefix_samples':n,'safety':safety,
                            'new_critical_terminal_flags':new_flags,
                            'survival':{n:row['policies'][n]['survived'] for n in policies}}
                if kind!='nominal':
                    comparison['exposed']={n:row['policies'][n]['exposed'] for n in policies}
                    for sec in (2,5):
                        comparison['recovery'+str(sec)]={n:row['policies'][n]['recovery_'+str(sec)+'s']['recovered']
                                                           for n in policies}
                if policies==H4_EVAL_POLICIES and kind!='nominal':
                    # Actual support before each push, not a randomized or imposed state.
                    comparison['prepush_support']={}
                    for label in policies:
                        group=groups[(seed,profile,label)]
                        event=group['directions'][d]['first' if kind=='single' else 'second']
                        prefix=traces[label][('nominal','none') if kind=='single' else ('single',d)][0]
                        tick=299 if kind=='single' else 334
                        if event['exposed']:
                            f=prefix[tick-1]
                            comparison['prepush_support'][label]={
                                'step':tick,'left':int(f['lf_ground_contact']),
                                'right':int(f['rf_ground_contact']),
                                'phase_left_rad':f['phase_left_post_rad'],
                                'phase_right_rad':f['phase_right_post_rad']}
                        else:
                            comparison['prepush_support'][label]=None
                comparison['discordance']={}
                for metric in ('survival','exposed','recovery2','recovery5'):
                    if metric in comparison:
                        a,b=(comparison[metric][n] for n in (comparator,candidate))
                        comparison['discordance'][metric]=(
                            'BOTH' if a and b else only_old if a else only_new if b else 'NEITHER')
                if kind=='nominal':
                    for k in ('xy_tracking_rmse_mps','yaw_tracking_rmse_radps'):
                        if not h3_tracking_limit(o[k],h[k]):
                            hard.append({'case':key,'gate':'nominal_'+k})
                if kind=='double':
                    if profile=='stand':
                        for n in (comparator,candidate):
                            primary['survival'][n]+=int(row['policies'][n]['survived'])
                            primary['recovery5'][n]+=int(comparison['recovery5'][n])
                    else:
                        for n in (comparator,candidate):
                            moving[profile][n]+=int(comparison['recovery5'][n])
                if profile=='stand':
                    if o['survived'] and h['survived']:
                        first={'nominal':101,'single':400,'double':435}[kind]
                        x,y=h3_standing(old_t,first),h3_standing(new_t,first)
                        standing[kind]+=1
                        tests={k:h3_tracking_limit(x[k],y[k]) for k in ('pose_l1_mean','xy_rmse','yaw_rmse')}
                        tests['swing_fraction']=y['swing_fraction']<=x['swing_fraction']+.02
                        comparison['standing']={comparator:x,candidate:y,'checks':tests}
                        for k,v in tests.items():
                            if not v:
                                hard.append({'case':key,'gate':'standing_'+k})
                    else:
                        comparison['standing']={'comparable':False,'reason':'not_common_full_survivors'}
                summaries.append(row)
                paired.append(comparison)
    for n,count in nominal.items():
        if count!=12:
            hard.append({'policy':n,'gate':'nominal_12_of_12','observed':count})
    for p,v in moving.items():
        if v[candidate]<v[comparator]:
            hard.append({'profile':p,'gate':'moving_double_recovery5_net_loss'})
    ceiling=[]
    primary_pass=True
    for k,v in primary.items():
        v['gain']=v[candidate]-v[comparator]
        v['pass']=v['gain']>=2
        if v[comparator]>6:
            ceiling.append(k)
        primary_pass=primary_pass and v['pass']
    missing=[k for k,v in standing.items() if v<(4 if k=='nominal' else 1)]
    # A safety failure is never masked by a primary ceiling or missing window.
    decision=('NO_GO' if hard else 'INCONCLUSIVE' if ceiling else
              'NO_GO' if not primary_pass else 'INCONCLUSIVE' if missing else 'DEV_PASS')
    discordance={}
    for pair in paired:
        _,profile,kind,_=pair['case']
        key=profile+'/'+kind
        dst=discordance.setdefault(key,{})
        for metric,label in pair['discordance'].items():
            counts=dst.setdefault(metric,{k:0 for k in ('BOTH',only_old,only_new,'NEITHER')})
            counts[label]+=1
    return {'decision':decision,'primary_stand_double':primary,'primary_ceiling':ceiling,
            'discordance_counts':discordance,
            'moving_double_recovery5':moving,'nominal_survivors':nominal,
            'standing_common_windows':standing,'missing_standing_windows':missing,
            'hard_failures':hard,'paired_cases':paired,'logical_outcomes':summaries,
            'promotion_authorized':False,'further_training_authorized':False}


def h3_verify_brax_train_source(networks):
    """Verify installed PPO source bytes WITHOUT executing its training imports."""
    from importlib.util import find_spec
    spec=find_spec('brax.training.agents.ppo.train')
    h3_require(spec is not None and spec.origin is not None,'BRAX_TRAIN_SOURCE_MISSING')
    path=Path(spec.origin).resolve(strict=True)
    expected=Path(networks.__file__).resolve(strict=True).with_name('train.py')
    h3_require(path==expected and path.is_file(),'BRAX_TRAIN_SOURCE_LOCATION')
    actual=sha256(path)
    if actual!=TRAIN_SHA:
        raise RuntimeError('M462_PINNED_BRAX_SOURCE_MISMATCH')
    return {'path':str(path),'sha256':actual,'train_module_import_required':False}


def h3_run(args):
    import shutil
    from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
    pins=h3_read_inputs(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    h3_require(shutil.disk_usage(args.output.parent).free>=2*1024**3,'DISK_RESERVE')
    g=sample_gpu()
    h3_require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288,'GPU_PREFLIGHT')
    report={'status':'M470_H3_EVAL_IN_PROGRESS','source_commit':PIN,
            'brax_train_sha256':TRAIN_SHA,'evaluator_sha256':sha256(Path(__file__)),
            'input_sha256':pins,'seeds':args.seeds,'profiles':args.profiles,
            'directions':args.directions,'push_step':300,'push_magnitude_mps':.50,
            'second_push_step':335,'second_push_magnitude_mps':.35,'max_steps':1000,
            'dt_s':.02,'training_performed':False,'deterministic_actor':True,
            'domain_randomization_enabled':False,'random_source_push_enabled':False,
            'original_reward_PD_termination':True,'expected_logical_trajectories':180,
            'max_physical_transitions':H3_EVAL_PHYSICAL,'physical_transitions_attempted':0,
            'physical_transitions_recorded':0,'logical_trajectories_completed':0,
            'frame_columns':None,'groups':[],'gpu_initial':g,'promotion_authorized':False}
    h3_write(args.output,report,initial=True)
    monitor=GPUMonitor()
    args.h3_monitor=monitor
    args.h3_session=report
    try:
        with monitor:
            run(args)
        report['gpu']=h3_gpu_check(monitor)
        h3_require(h3_read_inputs(args)==pins,'INPUT_IMMUTABILITY')
        h3_require(report['logical_trajectories_completed']==180
                   and report['physical_transitions_attempted']<=H3_EVAL_PHYSICAL,
                   'FINAL_CONTRACT')
        report['status']='M470_H3_EVAL_COMPLETED'
        h3_write(args.output,report)
        print('M470_H3_EVAL_COMPLETED',report['decision']['decision'],
              report['logical_trajectories_completed'],report['physical_transitions_attempted'],
              flush=True)
    except BaseException as e:
        report['status']='M470_H3_EVAL_FAILED'
        report['error']=type(e).__name__+': '+str(e)
        report['gpu']=monitor.snapshot()
        try:
            h3_write(args.output,report)
        except BaseException as write_error:
            print('M470_H3_EVAL_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise



# H4-M1-3: matched development only. Separate explicit execution GO required.
H4_EVAL_EXECUTION_AUTHORIZED = True
H4_EVAL_POLICIES = ('CTRL', 'CAND')
H4_EVAL_LOGICAL = 120
H4_EVAL_PHYSICAL = 89616
H4_EVAL_ROOT = Path(__file__).resolve().parent
H4_EVAL_OUTPUT = Path('m463_runs/m470_h4_m1_3_dev_v1/report.json')
H4_EVAL_ARTIFACTS = {
    'report.json': (
        '2ab06f3cc92d546f49eb5b40551a1a8238398106afc547822d0ac2b73529a059',
        '9c49c3cbea6cd3aae24435c4d8d4ee6f2cf09d35f4fbf1425d78f53a4a56307c'),
    'm463_easy_repeat_support_longtrain_step_010485760.msgpack': (
        'addc8e7217008555c4b4391f93dd3e7a3083e7b23052ec6ddf9b928fbc9fff33',
        'a6ce74e03179202cb2bd9098e96a431e9e4265f373c9f092c43eb34ae558b296'),
    'm470_fullstate_step_010485760.msgpack': (
        '9b19312da08d71513ad8fa6638c632401bec20e09399fdf4a9f6f6e2b2619c80',
        '7cbc24cb55767a9bcd6faa470ece3dbd78256f1bf5b6d1f074b67e2dea41124c'),
    'm470_fullstate_step_010485760.json': (
        '723ba76257cd0ec660ef6dd78d69d6546f80eec21467584d8fdd62295d69a119',
        '015484fe8501d530c7dcd4742f22e50215667428e0e441e3ea1240dd81ce4959'),
    'attempt.json': (
        '4ea87a9695dbea17bc9e14052223f86f5126aca90a6b422e5e5959e8253c0bd2',
        'e8e30dcf15876808f78917479a7a7e7044ec60c701207582aa0bee0cd0d22130'),
    'dispatch.started.json': (
        'a067569b1dc7f281fd4147402f3fc3d0907027f7c21a251119dbe13765b88aa1',
        '3fd18fbfc5c033233e6dd86435c1cc0739d1b3d82619a0cbf37953b67e2387f8'),
    'attempt_result.json': (
        'ceaa02e61c04dda2f35a8b4a1f020fac74d1bb9f60673ac3d2d1d2ecfae8e334',
        'fa7307b72904b56c7d9b8cb3dfdf388785ec46ab9bccacd190a62715d2b6ac25'),
    'm463_brax_ppo_instrumented.py': (
        '8f30e5c695211b21815375c37b1bda67fcda98b2ee259a703e87b182c3402062',
        '8f30e5c695211b21815375c37b1bda67fcda98b2ee259a703e87b182c3402062'),
}
H4_EVAL_POLICY_FILE = 'm463_easy_repeat_support_longtrain_step_010485760.msgpack'
H4_EVAL_DR = '60a4e6999c4fbad089439fabe9de951ba063bc68db1380b46ad47cb2c2f96ebf'
H4_EVAL_COMMON = 'ac0ee95f9272cc58049e439db65f3bc68194e21b4784a114097334519d3df26b'


def h4e_require(ok, label):
    if not ok:
        raise RuntimeError('M470_H4_EVAL_'+label)


def h4e_paths(arm):
    return H4_EVAL_ROOT / 'm463_runs' / ('m470_h4_m1_2r_'+arm+'_v1')


def h4e_validate_protocol(args):
    h4e_require(args.h4_m1_eval and not any((args.h3_m1_eval, args.h2a,
                 args.h2b, args.h4_technical)), 'MODE_CONFLICT')
    h4e_require(all(getattr(args,k) is None for k in (
        'checkpoint_c','checkpoint_p','report_p','checkpoint_h3','report_h3',
        'h4_reward_report')), 'LEGACY_INPUT_CONFLICT')
    h4e_require(tuple(args.seeds)==H3_EVAL_SEEDS
                 and tuple(args.profiles)==H3_EVAL_PROFILES
                 and tuple(args.directions)==('left','right')
                 and (args.push_step,args.push_magnitude,args.second_push_step,
                      args.second_push_magnitude,args.max_steps,args.max_cases)
                 ==(300,.50,335,.35,1000,H4_EVAL_LOGICAL), 'FROZEN_PROTOCOL')
    expected=H4_EVAL_ROOT/H4_EVAL_OUTPUT
    h4e_require(args.output.absolute()==expected and args.output.resolve()==expected,
                 'FROZEN_OUTPUT')
    h4e_require(24*5==H4_EVAL_LOGICAL and
                 24*(1000+2*(701+666))==H4_EVAL_PHYSICAL,'BUDGET_FORMULA')


def h4e_read_inputs(args):
    # Exact accepted evidence bytes; no training module import or artifact restore.
    h4e_validate_protocol(args)
    pins={}
    reports=[]
    for i,arm in enumerate(('control','candidate')):
        base=h4e_paths(arm)
        h4e_require(base.resolve()==base and base.is_dir(),'INPUT_DIRECTORY_'+arm)
        for name,expected in H4_EVAL_ARTIFACTS.items():
            p=base/name
            h4e_require(p.is_file() and not p.is_symlink() and sha256(p)==expected[i],
                         'INPUT_'+arm+'_'+name)
            pins[arm+'/'+name]=expected[i]
        r=json.loads((base/'report.json').read_bytes())
        result=json.loads((base/'attempt_result.json').read_bytes())
        auth=r['h4x_authorization']
        op=r['h4x_operational']
        budget={'training':10485760,'optimizer':1600,'evaluation':6000,
                'train_reset_lanes':24576,'eval_reset_lanes':6}
        initial=[e for e in op['events'] if e['event']=='initial']
        h4e_require(r['status']=='M463_COVERAGE_LONGTRAIN_PASS'
                     and r['h4_arm']==arm and r['h4x_exploratory'] is True
                     and r['source_commit']==PIN and r['source_checkpoint_sha256']==BASE_SHA
                     and r['training_seed']==1 and r['trained_from']=='C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE'
                     and r['resume_fullstate_sha256'] is None and r['cumulative_steps_start']==0
                     and r['steps_observed']==r['run_steps_observed']==10485760
                     and r['reward_PD_termination_source_unchanged'] is True
                     and r['h4_final_exposure']['passed'] is True
                     and r['coverage_gate_pass'] is True and r['memory_gate_pass'] is True
                     and r['weights_finite'] is True and r['final_tensor_count']==25
                     and r['callback_vs_final_equal'] is True
                     and op['complete'] is True and op['recorded']==op['attempted']==budget
                     and len(initial)==1 and initial[0]['dr_sha256']==H4_EVAL_DR
                     and initial[0]['adam']==0 and initial[0]['restored_leaves']==25
                     and result['status']=='COMPLETE' and result['authorization']==auth
                     and auth['arm']==arm and auth['id']=='H4-M1-2R-V1'
                     and auth['source_sha256']==H4_EVAL_COMMON
                     and auth['generated_sha256']==H4_EVAL_ARTIFACTS['m463_brax_ppo_instrumented.py'][i]
                     and r['checkpoints'][-1]['steps']==10485760
                     and r['checkpoints'][-1]['file']==H4_EVAL_POLICY_FILE
                     and r['checkpoints'][-1]['sha256']==H4_EVAL_ARTIFACTS[H4_EVAL_POLICY_FILE][i],
                     'REPORT_CONTRACT_'+arm)
        reports.append(r)
    ctrl,cand=reports
    h4e_require(ctrl['h4_common_contract']==cand['h4_common_contract']
                 and ctrl['hardware_scaled_config']==cand['hardware_scaled_config']
                 and ctrl['train_reward_config']==cand['train_reward_config']
                 and cand['h4_control_report_sha256']==H4_EVAL_ARTIFACTS['report.json'][0]
                 and cand['h4x_authorization']['evidence_sha256']==H4_EVAL_ARTIFACTS['report.json'][0]
                 and cand['h4_matched_nominal_pass'] is True,'MATCHED_PAIR')
    # Frozen helper source digests in the matched training contract exclude this evaluator.
    for name,expected in ctrl['h4_common_contract']['source_digests'].items():
        p=H4_EVAL_ROOT/name
        h4e_require(p.is_file() and not p.is_symlink() and sha256(p)==expected,
                     'HELPER_SOURCE_'+name)
        pins['source/'+name]=expected
    return pins


def h4e_run(args):
    h4e_require(H4_EVAL_EXECUTION_AUTHORIZED is True,'EXECUTION_HOLD')
    pins=h4e_read_inputs(args)
    import shutil
    from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
    h4e_require(shutil.disk_usage(args.output.parent.parent).free>=2*1024**3,'DISK_RESERVE')
    # Exclusive directory reservation. Never retry by deleting this directory.
    args.output.parent.mkdir(exist_ok=False)
    report={'status':'M470_H4_EVAL_IN_PROGRESS','source_commit':PIN,
            'brax_train_sha256':TRAIN_SHA,'evaluator_sha256':sha256(Path(__file__)),
            'input_sha256':pins,'seeds':args.seeds,'profiles':args.profiles,
            'directions':args.directions,'push_step':300,'push_magnitude_mps':.50,
            'second_push_step':335,'second_push_magnitude_mps':.35,'max_steps':1000,
            'dt_s':.02,'training_performed':False,'deterministic_actor':True,
            'domain_randomization_enabled':False,'random_source_push_enabled':False,
            'original_reward_PD_termination':True,'expected_logical_trajectories':H4_EVAL_LOGICAL,
            'max_physical_transitions':H4_EVAL_PHYSICAL,'physical_transitions_attempted':0,
            'physical_transitions_recorded':0,'logical_trajectories_completed':0,
            'frame_columns':None,'groups':[],'promotion_authorized':False}
    h3_write(args.output,report,initial=True)
    monitor=GPUMonitor()
    args.h3_monitor=monitor
    args.h3_session=report
    try:
        g=sample_gpu()
        h4e_require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288,'GPU_PREFLIGHT')
        report['gpu_initial']=g
        with monitor:
            run(args)
        report['gpu']=h3_gpu_check(monitor)
        h4e_require(h4e_read_inputs(args)==pins,'INPUT_IMMUTABILITY')
        h4e_require(report['logical_trajectories_completed']==H4_EVAL_LOGICAL
                     and report['physical_transitions_attempted']==report['physical_transitions_recorded']
                     <=H4_EVAL_PHYSICAL,'FINAL_CONTRACT')
        report['status']='M470_H4_EVAL_COMPLETED'
        h3_write(args.output,report)
        print('M470_H4_EVAL_COMPLETED',report['decision']['decision'],
              report['logical_trajectories_completed'],report['physical_transitions_attempted'],flush=True)
    except BaseException as e:
        report['status']='M470_H4_EVAL_FAILED'
        report['error']=type(e).__name__+': '+str(e)
        report['gpu']=monitor.snapshot()
        try:
            h3_write(args.output,report)
        except BaseException as write_error:
            print('M470_H4_EVAL_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise


def run(args):
    h4_mode=bool(getattr(args,'h4_m1_eval',False))
    if h4_mode:
        h4e_require(H4_EVAL_EXECUTION_AUTHORIZED is True,'EXECUTION_HOLD')
        h4e_validate_protocol(args)
        h4e_require(getattr(args,'h3_session',None) is not None,'SESSION_REQUIRED')
        h4e_require(args.output.is_file() and not args.output.is_symlink()
                     and h4e_read_inputs(args)==args.h3_session['input_sha256'],
                     'SESSION_INPUT_PROVENANCE')
    h3_mode=bool(getattr(args,'h3_m1_eval',False)) or h4_mode
    if args.output.exists() and not (h3_mode and getattr(args,'h3_session',None) is not None):
        raise FileExistsError('Refuse to overwrite output: '+str(args.output))
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
    if h3_mode:
        from brax.training.agents.ppo import networks
        args.h3_session['brax_source_verification']=h3_verify_brax_train_source(networks)
        print('M470_H4_EVAL_BRAX_SOURCE_PASS' if h4_mode else 'M470_H3_EVAL_BRAX_SOURCE_PASS',
              args.h3_session['brax_source_verification'],flush=True)
    else:
        from brax.training.agents.ppo import networks, train as ppo
    from mujoco_playground import locomotion
    from mujoco_playground._src import collision
    if jax.default_backend()!='gpu':raise RuntimeError('M463_GATE3_REQUIRES_GPU')
    if h3_mode:
        h3_require(jax.__version__=='0.4.35' and jax.process_count()==1
                   and jax.local_device_count()==1,'RUNTIME_PIN')
        args.h3_session['jax_version']=jax.__version__
    if not h3_mode and sha256(inspect.getfile(ppo.train))!=TRAIN_SHA:raise RuntimeError('M462_PINNED_BRAX_SOURCE_MISMATCH')
    paper=build_paper_config()
    if not h4_mode and sha256(args.checkpoint_c)!=BASE_SHA:raise RuntimeError('M462_FROZEN_401M_SHA_MISMATCH')
    report_p = None

    if not (args.h2a or args.h2b or h4_mode):
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
    if h3_mode:
        reward_report=h4e_paths('control')/'report.json' if h4_mode else args.report_h3
        original_reward=json.loads(reward_report.read_bytes())['train_reward_config']
        h3_require(cfg.reward_config.to_dict()==original_reward,'ORIGINAL_REWARD_CONFIG')
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
    if h4_mode:
        policies={n:load_policy(h4e_paths(arm)/H4_EVAL_POLICY_FILE)
                  for n,arm in zip(H4_EVAL_POLICIES,('control','candidate'))}
    else:
        policies = {
            'C': load_policy(args.checkpoint_c)
        }

        if not (args.h2a or args.h2b):
            policies['P'] = load_policy(
                args.checkpoint_p
            )
        if h3_mode:
            policies['OLD']=policies.pop('P')
            policies['H3']=load_policy(args.checkpoint_h3)
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

                if args.h2a or args.h2b or h3_mode:
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
                        **({} if h3_mode else {'h2a_action': action}),

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

                if h3_mode:
                    diag.update({
                        'h3_pose_l1': jnp.sum(jnp.abs(d.qpos[7:]-env._default_pose)),
                        'h3_root_x': d.qpos[0], 'h3_root_y': d.qpos[1],
                        'h3_finite': (jnp.isfinite(d.qacc).all()
                            & jnp.isfinite(d.sensordata).all()
                            & jnp.isfinite(state.obs['state']).all()
                            & jnp.isfinite(state.obs['privileged_state']).all()
                            & jnp.isfinite(next_state.obs['state']).all()
                            & jnp.isfinite(next_state.obs['privileged_state']).all()
                            & jnp.isfinite(next_state.reward)),
                    })
                return next_state,diag
            step_fn=jax.jit(one_step)
            if h3_mode:
                session=args.h3_session
                g=h3_group(args,env,step_fn,jax,jnp,np,session,name,seed,profile,
                           H4_EVAL_PHYSICAL if h4_mode else H3_EVAL_PHYSICAL)
                session['gpu']=h3_gpu_check(args.h3_monitor)
                h3_write(args.output,session)
                print('M470_H4_EVAL_GROUP' if h4_mode else 'M470_H3_EVAL_GROUP',
                      name,seed,profile,g['physical_transitions'],
                      session['logical_trajectories_completed'],
                      session['physical_transitions_attempted'],flush=True)
                continue
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
    if h3_mode:
        args.h3_session['decision']=(h3_decide(args.h3_session,policies=H4_EVAL_POLICIES,
            comparator='CTRL',candidate='CAND',logical_cap=H4_EVAL_LOGICAL,
            physical_cap=H4_EVAL_PHYSICAL) if h4_mode else h3_decide(args.h3_session))
        return
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



# H4-M1-1: isolated non-training technical validation. Never call ppo.train.
H4_TECH_SPEC={'lanes':4,'ticks':640,'unroll':32,'arms':['control','candidate'],
              'reset_seed':913811,'dr_seed':913812,'actor_seed':913813,
              'fixture_physical':4,'max_physical':5124,
              'synthetic_boundary_keys':32,'synthetic_step_max':119,
              'synthetic_scan_steps':40,'key_only_scan_steps':1280}
H4_TECH_REWARD_SHA='84c285315e109ca8716476b0f527873046b441d35e3fe963570e5f2019fe3530'
H4_TECH_ACTING_BLOB='ec6f81d3e3c48e96a22f7df292fb1a98b63f938d'
H4_TECH_SOURCE_SHA={
    'm463_coverage_env.py':'5a995942b87ba9514d9187fc7b864f33ee6eafa4d4994662f1faeb5c96e74e27',
    'm463_instrument_brax.py':'565465e7a5d73d68e1073322496d163015006d9ac78e16a45315dc255215c5e7',
    'm463_push_ppo_train.py':'6775ea45912d9ab98d273232fcfc86011759427ef905ea61e836159590df2887'}


def h4t_require(ok,label):
    if not bool(ok):
        raise RuntimeError('M470_H4_TECH_'+label)


def h4t_inputs(args):
    pins={}
    for name,p,expected in (
        ('C401M',args.checkpoint_c,BASE_SHA),
        ('reward_reference',args.h4_reward_report,H4_TECH_REWARD_SHA)):
        h4t_require(p is not None and p.is_file() and not p.is_symlink(),'INPUT_'+name)
        pins[name]=sha256(p)
        h4t_require(pins[name]==expected,'SHA_'+name)
    for name,expected in H4_TECH_SOURCE_SHA.items():
        pins[name]=sha256(Path(__file__).with_name(name))
        h4t_require(pins[name]==expected,'SOURCE_SHA_'+name)
    return pins


def h4t_carry(jax,jp,np):
    # Exact integer carry and RNG, on real JAX/pmap. No model/optimizer.
    def step(c,_):
        x, key, nested=c
        key,sub=jax.random.split(key)
        bits=jax.random.bits(sub,(),dtype=jp.uint32)
        n=nested['n']+1
        return (x+bits,key,{'n':n,'v':nested['v']+jp.arange(3,dtype=jp.int32)*n}), n
    def scan(c,length):
        return jax.lax.scan(step,c,(),length=length)
    init=(jp.zeros((1,),jp.uint32),jax.random.split(jax.random.PRNGKey(913810),1),
          {'n':jp.zeros((1,),jp.int32),'v':jp.zeros((1,3),jp.int32)})
    full,mfull=jax.pmap(lambda c:scan(c,20))(init)
    short=jax.pmap(lambda c:scan(c,5))
    c=init
    chunks=[]
    for _ in range(4):
        c,m=short(c)
        chunks.append(m)
    def host(v):
        return np.asarray(jax.device_get(v))
    h4t_require(jax.tree_util.tree_structure(c)==jax.tree_util.tree_structure(full),'CARRY_TREE')
    for x,y in zip(jax.tree_util.tree_leaves(c),jax.tree_util.tree_leaves(full)):
        h4t_require(np.array_equal(host(x),host(y)),'CARRY_EXACT')
    h4t_require(np.array_equal(host(jp.concatenate(chunks,axis=1)),host(mfull)),'CARRY_METRIC_ROWS')
    h4t_require(float(jp.mean(mfull))==float(jp.mean(jp.stack([jp.mean(x) for x in chunks]))),'CARRY_METRIC_MEAN')
    return {'exact_nested_integer_rng_carry':True,'exact_metric_rows_and_mean':True,
            'synthetic_scan_steps':40,'actual_optimizer_tested':False}


def h4t_scheduler_fixtures(jax,jp,np):
    """Actual CoverageJoystick scheduler with a synthetic, non-physics base step.
    No modifications to installed modules on disk; mock scope ends before MJX.
    """
    from unittest.mock import patch
    from flax import struct
    from mujoco_playground._src import mjx_env,wrapper
    import m463_coverage_env as ce

    @struct.dataclass
    class Data:
        qpos: object
        qvel: object
        support: object
        terminal: object

    def reset(self,key):
        z=jp.asarray(0,jp.int32)
        data=Data(jp.zeros(36,jp.float32).at[3].set(1.),
                  jp.zeros(35,jp.float32),jp.asarray(3,jp.int32),z)
        obs={'state':jp.zeros(103,jp.float32),'privileged_state':jp.zeros(216,jp.float32)}
        return mjx_env.State(data,obs,jp.asarray(0.,jp.float32),jp.asarray(0.,jp.float32),
             {'reward/lin_vel_z':jp.asarray(0.,jp.float32),
              'reward/feet_clearance':jp.asarray(0.,jp.float32)},
             {'rng':key,'command':ce.COMMANDS[0],'step':z,'steps':z})
    def base_step(self,state,action):
        # Deliberately synthetic dynamics, reward and refresh. Tests only
        # scheduler's handling of source-returned events, not physics fidelity.
        info=dict(state.info)
        info['rng'],_=jax.random.split(info['rng'])
        obs=dict(state.obs)
        obs['state']=obs['state'].at[9:12].set(info['command'])
        obs['privileged_state']=obs['privileged_state'].at[9:12].set(info['command'])
        refresh=info['step']>=500
        info['command']=jp.where(refresh,ce.COMMANDS[3],info['command'])
        info['step']=jp.where(refresh | (state.data.terminal>0),0,info['step']+1)
        return state.replace(info=info,obs=obs,done=state.data.terminal.astype(jp.float32))
    def collision(data,geom,floor):
        return (data.support & jp.asarray(1<<int(geom),jp.int32))>0
    def make(arm):
        env=object.__new__(ce.CoverageJoystick)
        env._h4_arm=arm
        env._h3_m1=env._h1_m1=env._h1_m2=False
        env._m463_stage='easy_repeat_support'
        env._m463_min_mag,env._m463_max_mag,env._m463_repeat_probability=ce.STAGES[env._m463_stage]
        env._feet_geom_id=(0,1)
        env._floor_geom_id=2
        return env
    action=jp.zeros(29,jp.float32)
    calls=0
    def scalar(st,k):
        return int(np.asarray(st.info[k]))
    def expect(st,**kw):
        for k,v in kw.items():
            h4t_require(scalar(st,k)==v,'FIXTURE_'+k)
    def exact(a,b):
        h4t_require(jax.tree_util.tree_structure(a)==jax.tree_util.tree_structure(b),'FIXTURE_TREE')
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            h4t_require(np.array_equal(np.asarray(x),np.asarray(y)),'FIXTURE_CONTROL_DRIFT')
    with patch.object(ce.joystick.Joystick,'reset',reset),patch.object(
            ce.joystick.Joystick,'step',base_step),patch.object(ce,'geoms_colliding',collision):
        legacy,control,candidate=map(make,(None,'control','candidate'))
        keys=jax.random.split(jax.random.PRNGKey(913814),32)
        def boundary(env):
            return jax.jit(jax.vmap(lambda k:env.step(env.reset(k),action)))(keys)
        l,c,b=map(boundary,(legacy,control,candidate))
        calls+=96
        exact(l,c.replace(info={k:c.info[k] for k in l.info}))
        counts=[0]*4
        for i,key in enumerate(keys):
            _,ks,kd,kp,kt,*_=jax.random.split(key,8)
            disabled=bool(jax.random.bernoulli(ks,p=.20))
            pick=float(jax.random.uniform(jax.random.fold_in(ks,41),()))
            cls=0 if disabled else 1 if pick<.5 else 2 if pick<.7 else 3
            age=int(jax.random.randint(jax.random.fold_in(kt,42),(),240,301)) if cls>=2 else int(jax.random.randint(kt,(),12,25))
            direction=int(jax.random.randint(jax.random.fold_in(kd,43),(),2,4)) if cls>=2 else int(jax.random.randint(kd,(),0,6))
            h4t_require(int(b.info['h4_class'][i])==cls and int(b.info['m463_next'][i])==age
                         and int(b.info['m463_direction'][i])==direction,'FIXTURE_ASSIGNMENT')
            h4t_require(int(b.info['h4_assigned'][i])==int(not disabled),'FIXTURE_ASSIGNED')
            counts[cls]+=1
        h4t_require(not np.any(np.asarray(b.info['h4_bad'])),'FIXTURE_BOUNDARY_BAD')
        fn=jax.jit(lambda st:candidate.step(st,action))
        def state(age=240,cls=3,support=3,terminal=0,**fields):
            st=candidate.reset(jax.random.PRNGKey(913815))
            info=dict(st.info)
            defaults={'steps':age,'step':age,'m463_age':age,'m463_next':age,
                      'm463_direction':2,'m463_phase_target':0,'h4_class':cls,
                      'h4_first_pending':1}
            defaults.update(fields)
            for k,v in defaults.items():
                info[k]=jp.asarray(v,dtype=info[k].dtype)
            return st.replace(info=info,data=st.data.replace(
                support=jp.asarray(support,jp.int32),terminal=jp.asarray(terminal,jp.int32)))
        labels=[]
        def check(label,st,**expected):
            nonlocal calls
            h4t_require(calls<H4_TECH_SPEC['synthetic_step_max'],'FIXTURE_BUDGET')
            out=fn(st);calls+=1
            expect(out,h4_bad=0,**expected)
            labels.append(label)
            return out
        single=check('mature_single_first',state(cls=2),m463_event=1,m463_repeat=0,
                     h4_repeat_assigned=0,h4_tail_kind=1,h4_tick_age=0)
        h4t_require(.45<=float(single.info['m463_magnitude'])<=.55,'FIXTURE_FIRST_MAG')
        check('single_no_repeat',state(age=300,cls=2,m463_push_count=2,
              m463_next=2**30,h4_first_pending=0),m463_event=0,m463_repeat=0)
        check('nominal_no_push',state(cls=0,m463_no_push=1,h4_first_pending=0),
              m463_event=0,m463_repeat=0)
        first=check('mature_double_first',state(age=300),m463_event=1,h4_repeat_assigned=1,
                    h4_repeat_pending=1,h4_tail_kind=1,h4_tick_age=0)
        h4t_require(330<=scalar(first,'m463_next')<=332,'FIXTURE_REPEAT_TARGET')
        for side in (1,2):
            common=dict(m463_direction=side+1,m463_push_count=1,m463_last_event_age=300,
                        m463_next=330,h4_first_pending=0,h4_repeat_pending=1,
                        h4_first_age=300,h4_first_mag=.50,h4_first_motion=0,
                        h4_tail_kind=1,h4_tail_age=34,h4_tail_same=1,h4_tail_motion=0)
            match=check('repeat_match_'+str(side),state(age=335,support=side,**common),
                    m463_event=1,m463_repeat=1,m463_repeat_side_matched=1,h4_event_qual=side,
                    h4_close_repeat=1,h4_tick_age=0,h4_tail_kind=2,h4_repeat_pending=0)
            h4t_require(.30<=float(match.info['m463_magnitude'])<=.40,'FIXTURE_REPEAT_MAG')
            check('repeat_wait_'+str(side),state(age=330,support=3,**common),
                    m463_event=0,h4_repeat_due=1,h4_repeat_pending=1)
            check('repeat_fallback_'+str(side),state(age=338,support=3,**common),
                    m463_event=1,m463_repeat_side_fallback=1,h4_event_qual=0,h4_close_repeat=1)
        check('terminal_first',state(terminal=1),m463_event=1,h4_event_terminal=1,
              h4_repeat_closed=1,h4_tail_kind=0)
        check('unreached_first_terminal',state(terminal=1,m463_next=300),
              m463_event=0,h4_first_closed=1,h4_first_pending=0)
        check('unreached_repeat_terminal',state(age=320,terminal=1,m463_push_count=1,
              m463_last_event_age=300,m463_next=330,h4_first_pending=0,h4_repeat_pending=1,
              h4_tail_kind=1,h4_tail_age=19,h4_tail_same=1,h4_tail_motion=0),
              m463_event=0,h4_repeat_closed=1,h4_repeat_pending=0,h4_close_terminal=1)
        for w in (10,25,50,100,250):
            common=dict(m463_push_count=2,m463_next=2**30,h4_first_pending=0,
                        h4_tail_kind=2,h4_tail_cell=1,h4_tail_age=w-1,
                        h4_tail_same=1,h4_tail_motion=0)
            check('tail_'+str(w),state(age=400,**common),
                  h4_tick_age=w,h4_tick_kind=2,h4_tick_survived=1,
                  h4_close_complete=int(w==250),h4_tail_kind=0 if w==250 else 2)
            if w==250:
                check('terminal_at_250',state(age=400,terminal=1,**common),
                      h4_tick_age=250,h4_tick_survived=0,h4_close_terminal=1,h4_close_complete=0)
        fresh=check('source_refresh',state(age=500,m463_push_count=2,m463_next=2**30,
                    h4_first_pending=0,h4_tail_kind=2,h4_tail_same=1,h4_tail_motion=0),
                    h4_refresh=1,h4_changed=1,h4_tick_same=1)
        # The synthetic base step does not implement EpisodeWrapper's steps.
        fresh=fresh.replace(info=dict(fresh.info,steps=jp.asarray(501,jp.int32)))
        check('command_mismatch_next',fresh,h4_command_mismatch=1,h4_tick_same=0)
        # Real EpisodeWrapper and BraxAutoResetWrapper over synthetic
        # dynamics. Force one source terminal then inspect the next boundary.
        w=wrapper.wrap_for_brax_training(candidate,episode_length=1000,action_repeat=1)
        st=jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(913816),1))
        info=dict(st.info)
        for k,v in {'steps':240,'m463_age':240,'m463_next':240,
                    'h4_class':3,'h4_first_pending':1,'m463_phase_target':0,
                    'm463_direction':2}.items():
            info[k]=jp.full_like(info[k],v)
        st=st.replace(info=info,data=st.data.replace(terminal=jp.ones((1,),jp.int32)))
        wf=jax.jit(w.step)
        act=jp.zeros((1,29),jp.float32)
        terminal=wf(st,act);calls+=1
        h4t_require(float(terminal.done[0])==1 and int(terminal.info['h4_event_terminal'][0])==1
                     and int(terminal.info['h4_repeat_closed'][0])==1
                     and int(terminal.info['h4_tail_kind'][0])==0,'WRAP_FIXTURE_TERMINAL')
        exact(terminal.data,terminal.info['first_state'])
        exact(terminal.obs,terminal.info['first_obs'])
        again=wf(terminal,act);calls+=1
        h4t_require(float(again.done[0])==0 and int(again.info['steps'][0])==1
                     and int(again.info['h4_age'][0])==0 and int(again.info['h4_boundary'][0])==1
                     and int(again.info['h4_close_boundary'][0])==0
                     and int(again.info['m463_event'][0])==0,'WRAP_FIXTURE_NEXT_BOUNDARY')
        h4t_require(calls==H4_TECH_SPEC['synthetic_step_max'],'FIXTURE_BUDGET')
        labels.append('actual_wrapper_synthetic_terminal_reset')
    return {'synthetic_calls':calls,'boundary_class_counts':counts,
            'control_legacy_exact_on_shared_synthetic_inputs':True,'case_labels':labels,
            'physics_performed':False}


def h4t_actual(args,r,monitor,persist):
    import ast
    import copy
    from train import check_source,build_paper_config
    from paper_config import BASE,ENV,PIN as SOURCE_PIN
    check_source()
    h4t_require(SOURCE_PIN==PIN,'SOURCE_PIN')
    sys.path.insert(0,str(BASE/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('M470_H4_TECH_NVCC',apply(),flush=True)
    import jax
    import jax.numpy as jp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    from brax.training.types import Transition
    from mujoco_playground import locomotion
    from mujoco_playground._src import wrapper
    import m463_coverage_env as ce
    from m463_instrument_brax import EXT_FIELDS,H4_EXT_FIELDS,H4_METRICS_SOURCE
    from m463_push_ppo_train import H4_EXECUTION_AUTHORIZED
    h4t_require(H4_EXECUTION_AUTHORIZED is False,'TRAIN_HOLD')
    h4t_require(jax.default_backend()=='gpu' and jax.__version__=='0.4.35'
                 and jax.process_count()==1 and jax.local_device_count()==1,'GPU_RUNTIME')
    r['brax_source']=h3_verify_brax_train_source(networks)
    r['jax_version']=jax.__version__
    # Exact pinned actor_step body, without importing acting.py's unrelated
    # legacy brax.v1 dependency (previous dm_env failure). No replacement body.
    acting=Path(r['brax_source']['path']).parents[2]/'acting.py'
    raw=acting.read_bytes()
    blob=hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
    h4t_require(blob==H4_TECH_ACTING_BLOB,'ACTING_SOURCE_PIN')
    tree=ast.parse(raw)
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='actor_step')
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    ns={'Transition':Transition}
    exec(compile(ast.fix_missing_locations(mod),str(acting),'exec'),ns)
    actor_step=ns['actor_step']
    r['actor_step_blob']=blob
    r['actor_step_module_imported']='brax.training.acting' in sys.modules
    h4t_require(not r['actor_step_module_imported'] and 'brax.training.agents.ppo.train' not in sys.modules,'TRAIN_IMPORT')
    r['carry']=h4t_carry(jax,jp,np)
    r['scheduler_fixtures']=h4t_scheduler_fixtures(jax,jp,np)
    h3_gpu_check(monitor)
    persist()
    print('M470_H4_TECH_CARRY_AND_FIXTURES_PASS',flush=True)

    paper=build_paper_config()
    cfg=locomotion.get_default_config(ENV)
    reward=json.loads(args.h4_reward_report.read_bytes())['train_reward_config']
    h4t_require(cfg.reward_config.to_dict()==reward and bool(cfg.push_config.enable)
                 and list(cfg.push_config.interval_range)==[5.,10.]
                 and list(cfg.push_config.magnitude_range)==[.1,2.],'SOURCE_CONFIG')
    cfg.push_config.enable=False
    randomizer=locomotion.get_domain_randomizer(ENV)
    h4t_require(randomizer is not None,'DR')
    obs_shape={'state':(103,),'privileged_state':(216,)}
    nets=networks.make_ppo_networks(obs_shape,29,
           preprocess_observations_fn=running_statistics.normalize,**dict(paper.network_factory))
    norm_template=running_statistics.init_state({k:jp.zeros(v,jp.float32) for k,v in obs_shape.items()})
    from m462_typed_orbax_stage import checkpoint_parts
    n,p,v=checkpoint_parts(serialization.msgpack_restore(args.checkpoint_c.read_bytes()))
    norm=serialization.from_state_dict(norm_template,n)
    actor=serialization.from_state_dict(nets.policy_network.init(jax.random.PRNGKey(11)),p)
    critic=serialization.from_state_dict(nets.value_network.init(jax.random.PRNGKey(17)),v)
    leaves=jax.tree_util.tree_leaves((norm,actor,critic))
    h4t_require(len(leaves)==25 and all(np.isfinite(np.asarray(jax.device_get(x))).all() for x in leaves),'POLICY')
    policy=networks.make_inference_fn(nets)((norm,actor),deterministic=True)
    fields=('truncation',)+EXT_FIELDS+H4_EXT_FIELDS
    reducer_ns={'jnp':jp}
    exec(compile('def reduce(extras):\n    unroll_length=32\n    coverage={}\n'+H4_METRICS_SOURCE+'    return coverage\n',
                 '<exact H4 metrics source>','exec'),reducer_ns)
    reduce=jax.jit(reducer_ns['reduce'])
    keys=jax.random.split(jax.random.PRNGKey(H4_TECH_SPEC['reset_seed']),4)
    dr_keys=jax.random.split(jax.random.PRNGKey(H4_TECH_SPEC['dr_seed']),4)
    cols=list(fields)+['done','prev_done','wrapper_steps','physical_finite','reset_data_obs_equal']
    r['columns']=cols
    def host(v):
        return np.asarray(jax.device_get(v))
    def make(arm):
        env=ce.CoverageJoystick(stage='easy_repeat_support',config=copy.deepcopy(cfg),h4_arm=arm)
        h4t_require(env.action_size==29 and float(env.dt)==.02,'ENV_CONFIG')
        return wrapper.wrap_for_brax_training(env,episode_length=1000,action_repeat=1,
               randomization_fn=functools.partial(randomizer,rng=dr_keys))
    def step(env,carry,_):
        st,key=carry
        key,sub=jax.random.split(key)
        prev=st.done
        nxt,tr=actor_step(env,st,policy,sub,extra_fields=fields)
        extra=tr.extras['state_extras']
        finite=jp.ones((4,),dtype=jp.bool_)
        for leaf in (nxt.data.qpos,nxt.data.qvel,nxt.data.qacc,nxt.data.ctrl,
                     nxt.data.actuator_force,nxt.data.sensordata,nxt.obs['state'],
                     nxt.obs['privileged_state'],tr.reward,tr.action):
            finite=finite & jp.all(jp.isfinite(leaf).reshape((4,-1)),axis=1)
        equal=jp.ones((4,),dtype=jp.bool_)
        for x,y in [(nxt.data.qpos,nxt.info['first_state'].qpos),
                    (nxt.data.qvel,nxt.info['first_state'].qvel),
                    (nxt.obs['state'],nxt.info['first_obs']['state']),
                    (nxt.obs['privileged_state'],nxt.info['first_obs']['privileged_state'])]:
            equal=equal & jp.all((x==y).reshape((4,-1)),axis=1)
        row={**extra,'done':nxt.done,'prev_done':prev,'wrapper_steps':nxt.info['steps'],
             'physical_finite':finite,'reset_data_obs_equal':equal}
        return (nxt,key),row
    def checked(carry,rows,n):
        data={k:host(v) for k,v in rows.items()}
        h4t_require(all(v.shape==(n,4) and np.isfinite(v).all() for v in data.values()),'ROW_SCHEMA')
        h4t_require(np.all(data['physical_finite']) and not np.any(data['h4_bad'])
                     and not np.any(data['truncation']),'FINITE_OR_BAD')
        h4t_require(np.array_equal(data['done'],data['h4_source_terminal']),'DONE')
        h4t_require(np.all(data['reset_data_obs_equal'][data['done']>0]),'AUTORESET_DATA_OBS')
        h4t_require(np.array_equal(data['wrapper_steps'],data['h4_age']+1),'WRAPPER_AGE')
        h4t_require(np.all(data['h4_boundary'][data['prev_done']>0]==1),'NATURAL_BOUNDARY')
        h4t_require(np.all(data['h4_tick_age'][data['m463_event']>0]==0),'EVENT_NOT_TAIL')
        h4t_require(np.all(data['h4_tick_survived'][data['done']>0]==0),'TERMINAL_NOT_SURVIVED')
        return data
    def charge(n):
        h3_gpu_check(monitor)
        h4t_require(r['physical_attempted']+n<=H4_TECH_SPEC['max_physical'],'BUDGET')
        r['physical_attempted']+=n
        persist()
    for arm in H4_TECH_SPEC['arms']:
        env=make(arm)
        initial=jax.jit(env.reset)(keys)
        carry=(initial,jax.random.PRNGKey(H4_TECH_SPEC['actor_seed']))
        unroll=jax.jit(lambda c:jax.lax.scan(lambda c,x:step(env,c,x),c,(),length=32))
        record={'arm':arm,'rows':[],'rollout_counts':[],'chunk_keys':[]}
        expected_key=carry[1]
        key_advance=jax.jit(lambda k:jax.lax.scan(lambda c,_:(jax.random.split(c)[0],None),k,(),length=160)[0])
        r['natural'].append(record)
        for i in range(20):
            charge(128)
            carry,rows=unroll(carry)
            data=checked(carry,rows,32)
            counts={k:float(host(v)) for k,v in reduce({k:rows[k].T for k in fields}).items()}
            h4t_require(counts['m463_h4_bad_per_rollout']==0,'METRIC_BAD')
            for k,m in [('first_events',(data['m463_event']>0)&(data['m463_repeat']==0)),
                        ('repeat_events',data['m463_repeat']>0),
                        ('close_complete',data['h4_close_complete']>0),
                        ('close_repeat',data['h4_close_repeat']>0)]:
                h4t_require(counts['m463_h4_'+k+'_per_rollout']==int(m.sum()),'METRIC_COUNT_'+k)
            record['rows'].extend(np.stack([data[k] for k in cols],axis=-1).tolist())
            record['rollout_counts'].append(counts)
            r['physical_recorded']+=128
            if (i+1)%5==0:
                expected_key=key_advance(expected_key)
                h4t_require(np.array_equal(host(carry[1]),host(expected_key)),'NATURAL_CHUNK_RNG')
                record['chunk_keys'].append(host(carry[1]).tolist())
                h3_gpu_check(monitor)
                persist()
                print('M470_H4_TECH_CHUNK',arm,i+1,r['physical_recorded'],flush=True)
        pending={k:host(carry[0].info[k]).tolist() for k in (
            'h4_first_pending','h4_repeat_pending','h4_tail_kind','h4_class',
            'h4_repeat_origin_motion','m463_direction')}
        record['pending_at_end']=pending
        record['audit']=h4t_natural_audit(record,cols)
        persist()
        # Separate deliberate scheduler-state fixtures on real reset physics.
        # No pose, command, reward, model, policy or source RNG edit.
        if arm=='candidate':
            st=initial
            info=dict(st.info)
            pre=host(info['command'])
            motion=np.argmin(np.abs(pre[:,None,:]-host(ce.COMMANDS)[None,:,:]).sum(axis=-1),axis=-1)
            vals={'steps':[300,300,338,338],'m463_age':[300,300,338,338],
                  'm463_next':[300,300,330,330],'m463_direction':[2,3,2,3],
                  'm463_phase_target':[0,0,0,0],'h4_class':[2,3,3,3],
                  'm463_push_count':[0,0,1,1],'m463_last_event_age':[-1,-1,300,300],
                  'h4_first_pending':[1,1,0,0],'h4_repeat_pending':[0,0,1,1],
                  'h4_first_age':[-1,-1,300,300],'h4_first_mag':[0,0,.5,.5],
                  'h4_first_motion':[-1,-1,int(motion[2]),int(motion[3])],
                  'h4_tail_kind':[0,0,1,1],'h4_tail_age':[0,0,37,37],
                  'h4_tail_same':[0,0,1,1],'h4_tail_motion':[-1,-1,int(motion[2]),int(motion[3])]}
            for k,val in vals.items():
                info[k]=jp.asarray(val,dtype=info[k].dtype)
            st=st.replace(info=info)
            charge(4)
            c,rows=jax.jit(lambda c:jax.lax.scan(lambda c,x:step(env,c,x),c,(),length=1))(
                (st,jax.random.PRNGKey(H4_TECH_SPEC['actor_seed'])))
            d=checked(c,rows,1)
            h4t_require(np.array_equal(d['m463_event'][0],[1,1,1,1])
                         and np.array_equal(d['m463_repeat'][0],[0,0,1,1]),'REAL_FIXTURE_EVENT')
            h4t_require(np.all((d['m463_magnitude'][0,:2]>=.45)&(d['m463_magnitude'][0,:2]<=.55))
                         and np.all((d['m463_magnitude'][0,2:]>=.30)&(d['m463_magnitude'][0,2:]<=.40)),
                         'REAL_FIXTURE_MAG')
            h4t_require(np.all(d['m463_repeat_gap'][0,2:]==38),'REAL_FIXTURE_GAP')
            mag=np.hypot(d['h4_actual_dx'],d['h4_actual_dy'])
            h4t_require(np.all(np.abs(mag-d['m463_magnitude'])<=3e-6),'REAL_FIXTURE_ACTUAL_MAG')
            h4t_require(np.all(d['m463_repeat_side_matched'][0,2:]+d['m463_repeat_side_fallback'][0,2:]==1),'REAL_FIXTURE_SIDE')
            r['real_fixture_rows']=np.stack([d[k] for k in cols],axis=-1).tolist()
            r['physical_recorded']+=4
            persist()
    h4t_require(r['physical_attempted']==r['physical_recorded']==5124,'FINAL_BUDGET')
    h4t_require('brax.training.agents.ppo.train' not in sys.modules,'TRAIN_IMPORT_FINAL')


def h4t_natural_audit(record,cols):
    """Independent host event/tail reconciliation on raw natural rows."""
    import numpy as np
    x=np.asarray(record['rows'])
    h4t_require(x.shape==(640,4,len(cols)),'NATURAL_ROWS')
    idx={k:i for i,k in enumerate(cols)}
    def col(k):
        return x[:,:,idx[k]]
    def count(k):
        return int(col(k).sum())
    event=col('m463_event')>0
    repeat=col('m463_repeat')>0
    p=record['pending_at_end']
    first=int((event&~repeat).sum())
    second=int(repeat.sum())
    h4t_require(count('h4_assigned')==first+count('h4_first_closed')+sum(p['h4_first_pending']),'FIRST_LEDGER')
    h4t_require(count('h4_repeat_assigned')==second+count('h4_repeat_closed')+sum(p['h4_repeat_pending']),'REPEAT_LEDGER')
    h4t_require(first+second==sum(count(k) for k in ('h4_event_terminal','h4_close_repeat',
        'h4_close_terminal','h4_close_complete','h4_close_boundary'))+sum(v>0 for v in p['h4_tail_kind']),'TAIL_LEDGER')
    h4t_require(count('h4_close_repeat')==second and count('h4_close_boundary')==0,'TAIL_CLOSURE')
    h4t_require(count('h4_close_complete')==int(((col('h4_tick_age')==250)&(col('h4_tick_survived')>0)).sum()),'TAIL_COMPLETE')
    for lane in range(4):
        tail=0
        kind=0
        for tick in range(640):
            def v(k):return int(x[tick,lane,idx[k]])
            if v('h4_boundary'):
                h4t_require(kind==0,'STALE_TAIL')
            if v('m463_event'):
                h4t_require(v('h4_tick_age')==0 and v('h4_close_repeat')==int(kind>0),'EVENT_TAIL')
                tail=0
                kind=0 if v('done') else 2 if v('m463_repeat') else 1
            elif kind:
                tail+=1
                h4t_require(v('h4_tick_age')==tail and v('h4_tick_kind')==kind
                             and v('h4_tick_survived')==int(not v('done')),'RAW_TAIL')
                if v('done') or tail==250:
                    kind=tail=0
            else:
                h4t_require(v('h4_tick_age')==0 and v('h4_tick_kind')==0,'NO_TAIL')
            h4t_require(v('h4_tail_kind')==kind and v('h4_tail_age')==tail,'TAIL_STATE')
        h4t_require(kind==p['h4_tail_kind'][lane],'PENDING_TAIL')
    cls=col('h4_class')
    h4t_require(not np.any(event&(cls==0)) and not np.any(repeat&(cls==2)),'CLASS_EVENT')
    if record['arm']=='control':
        h4t_require(not np.any(cls>=2),'CONTROL_CLASS')
    mature=event&(cls>=2)&~repeat
    h4t_require(np.all((col('h4_age')[mature]>=240)&(col('h4_age')[mature]<=308)),'MATURE_AGE')
    # Run the exact production ledger on the collected metrics. The
    # training-budget/exposure floors are not applicable to this diagnostic;
    # every conservation and bounds check remains required. Never use this
    # filtered result as a production training gate.
    from m463_push_ppo_train import h4_epoch_audit
    combined={}
    h4t_require(len(record['rollout_counts'])==20,'ROLLOUT_COUNT')
    for row in record['rollout_counts']:
        for k,v in row.items():
            if k.endswith('_per_rollout'):
                h4t_require(np.isfinite(v) and 0<=v<=128 and v==int(v),'ROLLOUT_INTEGER')
                combined[k]=combined.get(k,0)+int(v)
    q={'first_pending':sum(p['h4_first_pending']),
       'repeat_pending':sum(p['h4_repeat_pending']),
       'tail_pending':sum(v>0 for v in p['h4_tail_kind'])}
    for ci in range(4):
        for k in ('first_pending','repeat_pending'):
            q[f'class_{ci}_{k}']=sum(int(v>0 and c==ci)
                for v,c in zip(p['h4_'+k],p['h4_class']))
    for m in (0,3,4):
        for side in (1,2):
            q[f'target_{m}_{side}_pending']=sum(int(v>0 and c==3 and o==m and d==side+1)
                for v,c,o,d in zip(p['h4_repeat_pending'],p['h4_class'],
                        p['h4_repeat_origin_motion'],p['m463_direction']))
    excluded={'step_partition','nominal_fraction'}
    if record['arm']=='candidate':
        excluded.update(f'stand_{side}_{k}' for side in (1,2) for k in ('exposure','tail100'))
        excluded.update(f'turn_{m}_{side}' for m in (3,4) for side in (1,2))
    audited=h4_epoch_audit(combined,q,record['arm'])['checks']
    h4t_require(excluded.issubset(audited),'TECH_LEDGER_EXCLUSION_SCHEMA')
    selected={k:v for k,v in audited.items() if k not in excluded}
    h4t_require(all(selected.values()),'PRODUCTION_CONSERVATION '+str([k for k,v in selected.items() if not v]))
    h4t_require(sum(combined[f'm463_h4_class_{c}_steps_per_rollout'] for c in range(4))==2560,'TECH_PARTITION')
    return {'production_conservation_checks':selected,
            'training_only_gates_not_applicable':sorted(excluded),
            'first':first,'repeat':second,'natural_terminals':count('h4_source_terminal'),
            'natural_boundaries':count('h4_boundary'),'refresh':count('h4_refresh'),
            'mature_first':int(mature.sum()),'mature_repeat':count('h4_mature_repeat'),
            'qualified_stand_repeats':int((col('h4_event_qual')>0).sum()),
            'tail_250_executed':int((col('h4_tick_age')==250).sum()),
            'tail_250_survived':count('h4_close_complete'),'ledger_pass':True}


def h4t_run(args):
    import shutil
    from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
    pins=h4t_inputs(args)
    h4t_require(not args.output.exists(),'OUTPUT_EXISTS')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    h4t_require(shutil.disk_usage(args.output.parent).free>=2*1024**3,'DISK')
    gpu=sample_gpu()
    h4t_require(gpu['total_mib']==12288 and 0<=gpu['used_mib']<.95*12288,'GPU_PREFLIGHT')
    r={'status':'M470_H4_TECH_IN_PROGRESS','spec':H4_TECH_SPEC,'input_sha256':pins,
       'evaluator_sha256':sha256(Path(__file__)),'source_commit':PIN,'gpu_initial':gpu,
       'physical_attempted':0,'physical_recorded':0,'natural':[],
       'training_performed':False,'ppo_steps':0,'optimizer_updates':0,
       'promotion_authorized':False,'efficacy_evaluated':False}
    h3_write(args.output,r,initial=True)
    def persist():h3_write(args.output,r)
    monitor=GPUMonitor()
    try:
        with monitor:
            h4t_actual(args,r,monitor,persist)
        r['gpu']=h3_gpu_check(monitor)
        h4t_require(h4t_inputs(args)==pins,'INPUT_IMMUTABILITY')
        r['status']='M470_H4_TECH_COMPLETED'
        persist()
        print('M470_H4_TECH_COMPLETED',r['physical_recorded'],flush=True)
    except BaseException as exc:
        r['status']='M470_H4_TECH_FAILED'
        r['error']=type(exc).__name__+': '+str(exc)
        r['gpu']=monitor.snapshot()
        persist()
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','run'),required=True)
    p.add_argument('--checkpoint-c',type=Path)
    p.add_argument('--h4-m1-eval',action='store_true')
    p.add_argument('--checkpoint-p', type=Path)
    p.add_argument('--report-p', type=Path)
    p.add_argument('--h4-technical',action='store_true')
    p.add_argument('--h4-reward-report',type=Path)
    p.add_argument('--h3-m1-eval',action='store_true')
    p.add_argument('--checkpoint-h3',type=Path)
    p.add_argument('--report-h3',type=Path)
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
    if not args.h4_m1_eval and args.checkpoint_c is None:
        p.error('--checkpoint-c is required outside H4 matched evaluation')
    if args.h4_m1_eval and args.h4_technical:
        p.error('M470_H4_EVAL_MODE_CONFLICT')
    if args.h4_technical:
        if (args.h3_m1_eval or args.h2a or args.h2b or args.checkpoint_p is not None
                or args.report_p is not None or args.checkpoint_h3 is not None
                or args.report_h3 is not None or args.h4_reward_report is None
                or any(getattr(args,k)!=p.get_default(k) for k in (
                    'seeds','profiles','directions','push_step','push_magnitude',
                    'second_push_step','second_push_magnitude','max_steps','max_cases'))):
            p.error('M470_H4_TECH_FROZEN_FLAGS')
        if args.mode=='plan':
            print('M470_H4_TECH_PLAN',json.dumps(H4_TECH_SPEC,sort_keys=True))
            return
        h4t_run(args)
        return
    if args.h4_reward_report is not None:
        p.error('M470_H4_TECH_REPORT_REQUIRES_FLAG')
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
    if args.h4_m1_eval:
        try:
            h4e_validate_protocol(args)
            if args.mode=='run':
                h4e_require(H4_EVAL_EXECUTION_AUTHORIZED is True,'EXECUTION_HOLD')
        except RuntimeError as e:
            p.error(str(e))
        if args.mode=='plan':
            print('M470_H4_EVAL_PLAN',json.dumps({
                'policies':list(H4_EVAL_POLICIES),'seeds':args.seeds,
                'profiles':args.profiles,'directions':args.directions,
                'logical_trajectories':H4_EVAL_LOGICAL,'max_physical_transitions':H4_EVAL_PHYSICAL,
                'training_performed':False,'execution_authorized':H4_EVAL_EXECUTION_AUTHORIZED,
                'output':str(args.output)}))
            return
        h4e_run(args)
        return
    if args.h3_m1_eval:
        try:
            h3_validate_protocol(args)
        except RuntimeError as e:
            p.error(str(e))
        if args.mode=='plan':
            print('M470_H3_EVAL_PLAN',json.dumps({
                'policies':list(H3_EVAL_POLICIES),'seeds':args.seeds,
                'profiles':args.profiles,'directions':args.directions,
                'logical_trajectories':180,'max_physical_transitions':134424,
                'training_performed':False,'output':str(args.output)}))
            return
        h3_run(args)
        return
    if args.checkpoint_h3 is not None or args.report_h3 is not None:
        p.error('H3 inputs require --h3-m1-eval')
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
