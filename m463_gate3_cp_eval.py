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
             physical_cap=H3_EVAL_PHYSICAL,*,rich=None):
    """One nominal rollout, two single suffixes, two double suffixes at most."""
    group={'policy':name,'seed':seed,'profile':profile,'command':list(PROFILES[profile]),
           'nominal_frames':[],'directions':{},'complete':False}
    session['groups'].append(group)
    start=session['physical_transitions_attempted']
    def clone(st):
        return jax.tree_util.tree_map(lambda x:x,st)
    def advance(st,first,last,rows,capture_step=None,rich_direction=None,rich_kind=None):
        snap=None
        for i in range(first,last+1):
            h3_require(session['physical_transitions_attempted']<physical_cap,'PHYSICAL_BUDGET')
            session['physical_transitions_attempted']+=1
            capture=(rich is not None and rich_direction is not None and
                ((rich_kind=='single' and i==334) or (rich_kind=='double' and 335<=i<=385)))
            if capture:
                st,diag,packed=step_fn(st,placement_capture=True)
            else:
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
            if capture:
                rich(name,seed,profile,rich_direction,i,rows[-1],jax.device_get(packed))
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
    if rich is not None:
        h4m6_eval_require(session['reset_lanes_attempted']<24,'DEV_RESET_CAP')
        session['reset_lanes_attempted']+=1
        h3_write(args.output,session)
    st=env.reset(jax.random.PRNGKey(seed))
    h3_require(st.obs['state'].shape==(103,) and st.obs['privileged_state'].shape==(216,)
               and not bool(jax.device_get(st.done)), 'RESET_SCHEMA')
    if rich is not None:
        session['reset_lanes_recorded']+=1
        h3_write(args.output,session)
    snapshot=advance(st,1,1000,group['nominal_frames'],299)
    for direction in args.directions:
        item={'first':{'exposed':False,'step':300,'reason':'nominal_terminated_before_push'},
              'second':{'exposed':False,'step':335,'reason':'first_not_exposed'},
              'single_suffix':[],'double_suffix':[]}
        group['directions'][direction]=item
        if snapshot is None:
            continue
        first,item['first']=kick(snapshot,direction,.50,300)
        rd=direction if rich is not None and profile=='stand' and (seed,direction) in H4M6_MECHANISM_LANES else None
        second_snapshot=advance(first,300,1000,item['single_suffix'],334,rd,'single')
        item['second']['reason']='single_terminated_before_second'
        if second_snapshot is not None:
            second,item['second']=kick(second_snapshot,direction,.35,335)
            advance(second,335,1000,item['double_suffix'],rich_direction=rd,rich_kind='double')
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
H4_EVAL_EXECUTION_AUTHORIZED = False
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


# H4-M2 staged offline pairing: never dispatch dynamics from this entry.
H4CC_EVAL_EXECUTION_AUTHORIZED = False
H4CC_EVAL_OUTPUT = Path('m463_runs/m470_h4_m2_dev_v1/report.json')
H4CC_ARCHIVE_SHA = '32bf04d64bafd7e80b250b862493b54a7c8db6af1bcf1010bf0df984a904f40c'
H4CC_ARCHIVE_CTRL_PHYSICAL = 41052

def h4cc_eval_require(ok,label):
    if not ok:
        raise RuntimeError('M470_H4CC_EVAL_'+label)

def h4cc_read_archive(path):
    path=Path(path)
    h4cc_eval_require(path.is_file() and not path.is_symlink(),'ARCHIVE_FILE')
    h4cc_eval_require(sha256(path)==H4CC_ARCHIVE_SHA,'ARCHIVE_SHA')
    r=json.loads(path.read_bytes())
    h4cc_eval_require(r['status']=='M470_H4_EVAL_COMPLETED' and
        r['input_sha256']['control/'+H4_EVAL_POLICY_FILE]==H4_EVAL_ARTIFACTS[H4_EVAL_POLICY_FILE][0]
        and r['input_sha256']['control/report.json']==H4_EVAL_ARTIFACTS['report.json'][0],
        'ARCHIVE_PROVENANCE')
    ctrl=[g for g in r['groups'] if g['policy']=='CTRL']
    h4cc_eval_require(len(ctrl)==12 and all(g['complete'] for g in ctrl) and
        sum(g['physical_transitions'] for g in ctrl)==H4CC_ARCHIVE_CTRL_PHYSICAL,
        'ARCHIVE_LEDGER')
    return r

def h4cc_pair_decisions(archive, fresh):
    """Offline-only computation on already accepted raw reports.

    Caller must first verify immutable final REF/CCRC report and policy pins,
    exact dependency/physics equivalence and all technical/exposure gates.
    This helper is not a provenance acceptance or execution authorization.
    """
    for k in ('source_commit','brax_train_sha256','jax_version','seeds','profiles',
              'directions','push_step','push_magnitude_mps','second_push_step',
              'second_push_magnitude_mps','max_steps','dt_s','frame_columns',
              'training_performed','deterministic_actor','domain_randomization_enabled',
              'random_source_push_enabled','original_reward_PD_termination'):
        h4cc_eval_require(k in archive and k in fresh and archive[k]==fresh[k],
                          'PROTOCOL '+k)
    h4cc_eval_require(fresh['logical_trajectories_completed']==120 and
        fresh['physical_transitions_attempted']==fresh['physical_transitions_recorded']==
        sum(g['physical_transitions'] for g in fresh['groups'])<=89616 and
        len(fresh['groups'])==24 and {g['policy'] for g in fresh['groups']}=={'REF','CCRC'},
        'FRESH_LEDGER')
    def view(old_groups,old_label):
        # Use exact existing H4 decision function including support logging.
        groups=[dict(g,policy='CTRL') for g in old_groups]+[
            dict(g,policy='CAND') for g in fresh['groups'] if g['policy']=='CCRC']
        n=sum(g['physical_transitions'] for g in groups)
        session=dict(fresh,groups=groups,logical_trajectories_completed=120,
                     physical_transitions_attempted=n,physical_transitions_recorded=n)
        d=h3_decide(session,policies=H4_EVAL_POLICIES,comparator='CTRL',
                    candidate='CAND',logical_cap=120,physical_cap=89616)
        return {'comparison_roles':{'CTRL':old_label,'CAND':'CCRC'},'decision':d,
                'combined_physical_rows':n}
    official=view([g for g in archive['groups'] if g['policy']=='CTRL'],'ARCHIVED_H4_CTRL')
    incremental=view([g for g in fresh['groups'] if g['policy']=='REF'],'FRESH_REF')
    o=official['decision']; i=incremental['decision']
    primary=i['primary_stand_double']
    incremental_result=('NO_GO' if i['hard_failures'] else
        'INCONCLUSIVE' if any(v['CTRL']==8 for v in primary.values()) else
        'NO_GO' if not all(v['gain']>=1 for v in primary.values()) else
        'INCONCLUSIVE' if i['missing_standing_windows'] else 'DEV_PASS')
    result=('NO_GO' if o['decision']=='NO_GO' or incremental_result=='NO_GO' else
            'DEV_PASS' if o['decision']==incremental_result=='DEV_PASS' else 'INCONCLUSIVE')
    incremental['incremental_decision']=incremental_result
    incremental['frozen_plus_two_decision_informational']=i['decision']
    return {'decision':result,'official':official,'incremental':incremental,
            'archived_ctrl_physical':H4CC_ARCHIVE_CTRL_PHYSICAL,
            'new_physical':fresh['physical_transitions_recorded'],
            'promotion_authorized':False,'further_training_authorized':False}

def h4cc_eval_authorization():
    import re
    import m463_push_ppo_train as trainer
    rules=(H4_EVAL_ROOT/'G1_RESEARCH_RULES.md').read_text('utf8')
    h4cc_eval_require(not any((H4CC_EVAL_EXECUTION_AUTHORIZED,H4_EVAL_EXECUTION_AUTHORIZED,
        trainer.H4CC_EXECUTION_AUTHORIZED,trainer.H4_EXECUTION_AUTHORIZED,
        trainer.H4R_EXECUTION_AUTHORIZED,trainer.H4C_EXECUTION_AUTHORIZED)),
        'BROAD_HOLD_REQUIRED')
    h4cc_eval_require(not any(line.startswith(('H4X_ACTIVE_GO:','H4CC_ACTIVE_GO:','H4CC_TECH_ACTIVE_GO:'))
        for line in rules.splitlines()),'TRAINING_AUTHORIZATION_CONFLICT')
    lines=[line for line in rules.splitlines() if line.startswith('H4CC_EVAL_ACTIVE_GO:')]
    h4cc_eval_require(len(lines)==1,'EXECUTION_HOLD')
    m=re.fullmatch(r'H4CC_EVAL_ACTIVE_GO: (H4-M2-CCRC-V1) '
        r'([0-9a-f]{64}) ([0-9a-f]{64}) ([0-9a-f]{64}) ([0-9a-f]{64})',lines[0])
    h4cc_eval_require(m is not None,'AUTHORIZATION_SCHEMA')
    ident,source,ref,ccrc,archive=m.groups()
    identity=trainer.h4cc_source_identity(H4_EVAL_ROOT)
    h4cc_eval_require(source==identity['current_sha256']
        and ref==trainer.h4cc_approved_sha(rules,'REF_REPORT')
        and ccrc==trainer.h4cc_approved_sha(rules,'CCRC_REPORT')
        and archive==H4CC_ARCHIVE_SHA,'AUTHORIZATION_BINDING')
    return {'id':ident,'source_sha256':source,'ref_report_sha256':ref,
            'ccrc_report_sha256':ccrc,'archive_sha256':archive}

def h4cc_read_inputs(args):
    """Read accepted files only, no policy restore, runtime or simulator."""
    import m463_push_ppo_train as trainer
    auth=h4cc_eval_authorization()
    identity=trainer.h4cc_source_identity(H4_EVAL_ROOT)
    archive_path=H4_EVAL_ROOT/H4_EVAL_OUTPUT
    archive=h4cc_read_archive(archive_path)
    reports={}
    pins={'archive/report.json':H4CC_ARCHIVE_SHA}
    for arm in ('ref','ccrc'):
        p=H4_EVAL_ROOT/trainer.H4CC_PATHS[arm]/'report.json'
        reports[arm]=trainer.h4cc_read_final_report(p,auth[arm+'_report_sha256'],arm,
                         reports['ref']['h4_common_contract'] if arm=='ccrc' else None,
                         contract_arm='ref' if arm=='ccrc' else None)
        pins[arm+'/report.json']=auth[arm+'_report_sha256']
        for name in ('m463_brax_ppo_instrumented.py','attempt.json',
                     'dispatch.started.json','attempt_result.json'):
            pins[arm+'/'+name]=sha256(p.parent/name)
        for pol,full in zip(reports[arm]['checkpoints'],reports[arm]['fullstate_checkpoints']):
            for name in (pol['file'],full['file'],full['manifest']):
                pins[arm+'/'+name]=sha256(p.parent/name)
    ref,t=reports['ref'],reports['ccrc']
    a,b=ref['h4x_authorization'],t['h4x_authorization']
    h4cc_eval_require(a==trainer.h4cc_matched_ref_authorization(b,identity)
        and b['source_sha256']==auth['source_sha256']
        and b['evidence_sha256']==auth['ref_report_sha256'],'MATCHED_AUTHORIZATION')
    h4cc_eval_require(trainer.h4cc_matched_nominal(ref['h4_final_counts'],t['h4_final_counts'])
        and trainer.h4cc_exposure_decision(ref['h4cc_final_counts'],t['h4cc_final_counts'])
            ==t['h4cc_exposure_decision']
        and t['h4cc_exposure_decision']['decision']=='PASS','EXPOSURE_HYPOTHESIS')
    historical=json.loads((h4e_paths('control')/'report.json').read_bytes())
    h4cc_eval_require(ref['h4x_brax_dependencies']==t['h4x_brax_dependencies']
        ==historical['h4x_brax_dependencies'],'MATCHED_DEPENDENCY_PINS')
    # Historical modified helper files are not expected to match new CCRC.
    # Unchanged runtime helpers must match archived bytes exactly.
    for name in ('train.py','paper_config.py','m462_ppo_warmstart_run.py','m462_typed_orbax_stage.py'):
        expected=archive['input_sha256']['source/'+name]
        p=H4_EVAL_ROOT/name
        h4cc_eval_require(p.is_file() and not p.is_symlink() and sha256(p)==expected,
                          'ARCHIVE_HELPER '+name)
        pins['source/'+name]=expected
    for name,expected in identity['current_files'].items():
        p=H4_EVAL_ROOT/name
        h4cc_eval_require(p.is_file() and not p.is_symlink() and sha256(p)==expected,
                          'FRESH_HELPER '+name)
        pins['source/'+name]=expected
    # Match installed source files by pinned Git blob without importing PPO.
    from m463_instrument_brax import h4c_git_blob
    train_path=Path(archive['brax_source_verification']['path'])
    h4cc_eval_require(train_path.is_file() and sha256(train_path)==TRAIN_SHA,
                      'ARCHIVE_BRAX_TRAIN')
    brax_root=train_path.resolve().parents[3]
    for name,entry in ref['h4x_brax_dependencies'].items():
        p=Path(entry['path'])
        intended=brax_root.joinpath(*name.split('.')[1:]).with_suffix('.py')
        h4cc_eval_require(p.resolve()==intended and p.is_file()
            and h4c_git_blob(p.read_bytes())==entry['blob'],'DEPENDENCY '+name)
        pins['dependency/'+name]=sha256(p)
    wrapper=brax_root/'envs/wrappers/training.py'
    h4cc_eval_require(wrapper.is_file() and h4c_git_blob(wrapper.read_bytes())==
        '2c09c8268a29d14c832dd07050575e4541ee98af','DEPENDENCY_WRAPPER')
    pins['dependency/episode_wrapper']=sha256(wrapper)
    return pins,archive

def h4cc_eval_plan(args):
    h4cc_eval_require(args.mode in ('plan','run'),'MODE')
    h4cc_eval_require(not any(getattr(args,k) for k in
        ('h4_m1_eval','h4_technical','h3_m1_eval','h2a','h2b','checkpoint_c',
         'checkpoint_p','report_p','checkpoint_h3','report_h3','h4_reward_report')),
         'MODE_CONFLICT')
    seeds=','.join(map(str,args.seeds)) if isinstance(args.seeds,list) else args.seeds
    profiles=','.join(args.profiles) if isinstance(args.profiles,list) else args.profiles
    directions=','.join(args.directions) if isinstance(args.directions,list) else args.directions
    h4cc_eval_require((seeds,profiles,directions,args.push_step,
        args.push_magnitude,args.second_push_step,args.second_push_magnitude,
        args.max_steps,args.max_cases)==('7331,7357,7601,7613','stand,forward_fast,turn_left',
        'left,right',300,.50,335,.35,1000,120),'FROZEN_PROTOCOL')
    h4cc_eval_require(args.output.absolute()==H4_EVAL_ROOT/H4CC_EVAL_OUTPUT and
        args.output.resolve()==H4_EVAL_ROOT/H4CC_EVAL_OUTPUT,'FROZEN_OUTPUT')
    return {'policies':['REF','CCRC'],'official_reference':'ARCHIVED_H4_CTRL',
        'logical_trajectories':120,'max_physical_transitions':89616,
        'training_performed':False,'execution_authorized':False,
        'archive_sha256':H4CC_ARCHIVE_SHA,'output':str(args.output)}


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
    cc=bool(getattr(args,'h4cc_eval',False))
    if cc:
        h4cc_eval_plan(args)
        auth=h4cc_eval_authorization()
        pins,archive=h4cc_read_inputs(args)
        args.h4cc_archive=archive
    else:
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
    if cc:
        report.update(status='M470_H4CC_EVAL_IN_PROGRESS',authorization=auth,
                      archived_ctrl_physical=H4CC_ARCHIVE_CTRL_PHYSICAL)
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
        if cc:
            marker=args.output.parent/'dispatch.started.json'
            h4cc_eval_require(h4cc_eval_authorization()==auth and
                h4cc_read_inputs(args)[0]==pins and marker.is_file() and not marker.is_symlink()
                and json.loads(marker.read_bytes())=={'authorization':auth},'INPUT_IMMUTABILITY')
        else:
            h4e_require(h4e_read_inputs(args)==pins,'INPUT_IMMUTABILITY')
        h4e_require(report['logical_trajectories_completed']==H4_EVAL_LOGICAL
                     and report['physical_transitions_attempted']==report['physical_transitions_recorded']
                     <=H4_EVAL_PHYSICAL,'FINAL_CONTRACT')
        report['status']='M470_H4CC_EVAL_COMPLETED' if cc else 'M470_H4_EVAL_COMPLETED'
        h3_write(args.output,report)
        print(report['status'],report['decision']['decision'],
              report['logical_trajectories_completed'],report['physical_transitions_attempted'],flush=True)
    except BaseException as e:
        report['status']='M470_H4CC_EVAL_FAILED' if cc else 'M470_H4_EVAL_FAILED'
        report['error']=type(e).__name__+': '+str(e)
        report['gpu']=monitor.snapshot()
        try:
            h3_write(args.output,report)
        except BaseException as write_error:
            print('M470_H4_EVAL_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise


def h4_shared_frozen_loader(action_size,paper,networks,running_statistics,serialization,jax,jnp,np,
                            audit=None,template_event=None):
    obs_shape={'state':(103,),'privileged_state':(216,)}
    ppo_nets=networks.make_ppo_networks(obs_shape,action_size,
                preprocess_observations_fn=running_statistics.normalize,
                **dict(paper.network_factory))
    norm_template=running_statistics.init_state({k:jnp.zeros(s,dtype=jnp.float32) for k,s in obs_shape.items()})
    if template_event is not None: template_event('actor','attempted')
    actor_template=ppo_nets.policy_network.init(jax.random.PRNGKey(11))
    if template_event is not None:
        jax.block_until_ready(actor_template)
        template_event('actor','recorded')
    if template_event is not None: template_event('critic','attempted')
    critic_template=ppo_nets.value_network.init(jax.random.PRNGKey(17))
    if template_event is not None:
        jax.block_until_ready(critic_template)
        template_event('critic','recorded')
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
        if audit is not None: audit(path,(norm,actor,critic),(norm_template,actor_template,critic_template))
        inf=networks.make_inference_fn(ppo_nets)((norm,actor),deterministic=True)
        return inf
    return load_policy


def run(args):
    m6=bool(getattr(args,'h4m6_phase',None)=='dev')
    if getattr(args,'h4m6_phase',None) is not None:
        h4m6_runtime_gate(args.h4m6_phase)
        h4m6_eval_require(m6,'DEDICATED_TECH_DISPATCH')
    if getattr(args,'h4m5_obs_attribution',False):
        raise RuntimeError('M470_H4M5_DEDICATED_DISPATCH_REQUIRED')
    feasibility=bool(getattr(args,'h4m4_feasibility',False))
    placement=bool(getattr(args,'h4m3_placement',False)) or feasibility
    if feasibility:
        h4m4_enter_runtime(args)
    elif placement:
        h4m3_enter_runtime(args)
    if getattr(args,'h4cc_technical',False):
        raise RuntimeError('M470_H4CC_TECH_USE_DEDICATED_DISPATCH')
    cc=bool(getattr(args,'h4cc_eval',False))
    h4_mode=bool(getattr(args,'h4_m1_eval',False)) or cc or m6
    if m6:
        import m463_push_ppo_train as trainer
        auth=trainer.h4m6_authorization(phase='dev')
        checked,archive,external,exposure=h4m6_dev_inputs(auth)
        session=getattr(args,'h3_session',None)
        h4m6_eval_require(session is not None and session['authorization']==auth
            and session['status']=='M470_H4M6_DEV_IN_PROGRESS'
            and session['physical_transitions_attempted']==session['physical_transitions_recorded']
                ==session['logical_trajectories_completed']==session['reset_lanes_attempted']
                ==session['reset_lanes_recorded']==0 and session['groups']==[]
            and session['input_sha256']==checked and session['exposure']==exposure
            and args.output.is_file() and not args.output.is_symlink()
            and json.loads(args.output.read_bytes())==session
            and json.loads((args.output.parent/'attempt.json').read_bytes())=={'authorization':auth,'reserved':True},
            'DEV_SESSION')
        args.h4m6_archive=archive
        args.h4m6_external=external
        h4m6_write_bounded(args.output.parent/'dispatch.started.json',{'authorization':auth},exclusive=True)
    elif cc:
        h4cc_eval_plan(args)
        auth=h4cc_eval_authorization()
        session=getattr(args,'h3_session',None)
        validated_pins,validated_archive=h4cc_read_inputs(args)
        h4cc_eval_require(session is not None and session['authorization']==auth
            and args.output.is_file() and not args.output.is_symlink()
            and session['status']=='M470_H4CC_EVAL_IN_PROGRESS'
            and session['physical_transitions_attempted']==session['physical_transitions_recorded']
                ==session['logical_trajectories_completed']==0 and session['groups']==[]
            and validated_pins==session['input_sha256'],'SESSION_PROVENANCE')
        # Never accept a caller-provided archive as the official comparator.
        args.h4cc_archive=validated_archive
        # Independent exclusive dispatch reservation, before runtime imports.
        with (args.output.parent/'dispatch.started.json').open('x',encoding='utf8') as f:
            json.dump({'authorization':auth},f,sort_keys=True,allow_nan=False)
    elif h4_mode:
        h4e_require(H4_EVAL_EXECUTION_AUTHORIZED is True,'EXECUTION_HOLD')
        h4e_validate_protocol(args)
        h4e_require(getattr(args,'h3_session',None) is not None,'SESSION_REQUIRED')
        h4e_require(args.output.is_file() and not args.output.is_symlink()
                     and h4e_read_inputs(args)==args.h3_session['input_sha256'],
                     'SESSION_INPUT_PROVENANCE')
    h3_mode=bool(getattr(args,'h3_m1_eval',False)) or h4_mode or placement
    if args.output.exists() and not (h3_mode and getattr(args,'h3_session',None) is not None):
        raise FileExistsError('Refuse to overwrite output: '+str(args.output))
    # Use the same pinned runner environment; apply compatibility BEFORE JAX import.
    from train import check_source, build_paper_config
    from paper_config import BASE, ENV, PIN as SOURCE_PIN
    check_source()
    if SOURCE_PIN!=PIN:raise RuntimeError('M462_SOURCE_PIN_MISMATCH')
    sys.path.insert(0,str(BASE/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    if placement:
        h4m3_no_runtime_imports()
        h4m3_require(str(Path(inspect.getfile(apply)).resolve())==
            args.h3_session['external_sources']['nvcc_compat']['path'] and
            sha256(inspect.getfile(apply))==args.h3_session['external_sources']['nvcc_compat']['sha256'],
            'NVCC_RUNTIME_PIN')
    if m6:
        h4m3_no_runtime_imports()
        nvcc=args.h4m6_external['common']['historical']['nvcc_compat']
        h4m6_eval_require(str(Path(inspect.getfile(apply)).resolve())==nvcc['path']
            and sha256(inspect.getfile(apply))==nvcc['sha256'],'DEV_NVCC_PIN')
    print('M463_GATE3_NVCC',apply(),flush=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    if h3_mode:
        from brax.training.agents.ppo import networks
        args.h3_session['brax_source_verification']=h3_verify_brax_train_source(networks)
        if placement:
            h4m3_require(args.h3_session['brax_source_verification']['path']
                ==args.h3_session['brax_train_path'],'RUNTIME_BRAX_PATH')
            h4m3_runtime_dependencies(args.h3_session)
            if feasibility:
                h4m4_require(args.h3_session['brax_source_verification']
                    ==args.h4m4_approved_brax,'BRAX_ARCHIVE_IDENTITY')
        if m6:
            h4m6_eval_require(args.h3_session['brax_source_verification']==args.h4m6_archive['brax_source_verification'],'DEV_BRAX_SOURCE')
            from importlib.util import find_spec
            for name,pin in args.h4m6_external['training'].items():
                spec=find_spec(name) if name!='brax.training.agents.ppo.train' else None
                p=Path(pin['path'])
                h4m6_eval_require(p.is_file() and sha256(p)==pin['sha256']
                    and (spec is None if name=='brax.training.agents.ppo.train' else
                        spec is not None and Path(spec.origin).resolve()==p),'DEV_RUNTIME_DEP '+name)
            h4m6_eval_require('brax.training.agents.ppo.train' not in sys.modules,'DEV_TRAIN_IMPORT')
        if cc:
            h4cc_eval_require(args.h3_session['brax_source_verification']==args.h4cc_archive['brax_source_verification'],
                              'RUNTIME_BRAX_EQUIVALENCE')
            from importlib.util import find_spec
            deps=args.h3_session['input_sha256']
            root=Path(args.h3_session['brax_source_verification']['path']).resolve().parents[3]
            for key,expected in deps.items():
                if not key.startswith('dependency/') or key=='dependency/episode_wrapper':
                    continue
                name=key.removeprefix('dependency/')
                spec=find_spec(name)
                intended=root.joinpath(*name.split('.')[1:]).with_suffix('.py')
                h4cc_eval_require(spec is not None and spec.origin is not None and
                    Path(spec.origin).resolve()==intended and sha256(intended)==expected,
                    'RUNTIME_DEPENDENCY '+name)
        print('M470_H4M3_BRAX_SOURCE_PASS' if placement else
              'M470_H4_EVAL_BRAX_SOURCE_PASS' if h4_mode else 'M470_H3_EVAL_BRAX_SOURCE_PASS',
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

    if not (args.h2a or args.h2b or h4_mode or placement):
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
        reward_report=(H4_EVAL_ROOT/'m463_runs/m470_h4_m2_ccrc_v1/report.json'
                       if placement else h4e_paths('control')/'report.json'
                       if h4_mode else args.report_h3)
        original_reward=json.loads(reward_report.read_bytes())['train_reward_config']
        h3_require(cfg.reward_config.to_dict()==original_reward,'ORIGINAL_REWARD_CONFIG')
    cfg.push_config.enable=False
    env=locomotion.load(ENV,config=cfg)
    if env.action_size!=29:raise RuntimeError('M462_ACTION_DIM_CHANGED')
    if abs(float(env.dt)-0.02)>1e-6:raise RuntimeError('M462_CTRL_DT_CHANGED')
    if placement:
        h4m3_runtime_model_source(env,args.h3_session)
        args.h3_session['manifest']=h4m3_manifest(env,np)
        if feasibility:
            h4m4_require(args.h3_session['manifest']==args.h4m4_approved_manifest,
                'H4M3_MODEL_IDENTITY')
            args.h3_session['kinematics']=h4m4_model_geometry(env,args.h3_session['manifest'],np)
        h4m3_write(args.output,args.h3_session)
    if m6:
        session=args.h3_session
        session['mjx_sources']=args.h4m6_external['common']['historical']['mjx']
        h4m3_runtime_model_source(env,session)
        args.h4m6_rich['manifest']=h4m3_manifest(env,np)
        h4m6_rich_write(args.h4m6_rich)
    load_policy=h4_shared_frozen_loader(env.action_size,paper,networks,running_statistics,
        serialization,jax,jnp,np)
    if m6:
        from m463_push_ppo_train import H4M6_PATHS
        policies={label:load_policy(H4_EVAL_ROOT/H4M6_PATHS[arm]/H4_EVAL_POLICY_FILE)
                  for label,arm in (('REF6','ref'),('RSS','rss'))}
    elif placement:
        policies={n:load_policy(p) for n,p in h4m3_policy_paths(H4_EVAL_ROOT).items()}
    elif cc:
        from m463_push_ppo_train import H4CC_PATHS
        policies={arm.upper():load_policy(H4_EVAL_ROOT/H4CC_PATHS[arm]/H4_EVAL_POLICY_FILE)
                  for arm in ('ref','ccrc')}
    elif h4_mode:
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
    for lane_index,seed in enumerate(run_seeds):
      for profile in args.profiles:
        command=jnp.asarray(PROFILES[profile],dtype=jnp.float32)
        selected=H4M3_LANES[lane_index][0] if placement else None
        for name,policy in ([(selected,policies[selected])] if placement else policies.items()):
            # JIT step/diagnostics; a single MJX state (no vectorized envs) is active at a time.
            def one_step(state, effective_command=None, *, placement_capture=False,
                         probe_residual=None,probe_selected=None):
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
                if placement_capture:
                    pre_placement=h4m3_pre_capture(state)
                action,_=policy(state.obs,jax.random.PRNGKey(0))
                if probe_residual is not None:
                    h4m4_require(feasibility and placement_capture and probe_selected is not None,
                                  'OPT_IN_CAPTURE_REQUIRED')
                    action,probe_meta=h4m4_apply_residual(action,probe_residual,probe_selected,
                        env._default_pose,env.mj_model.actuator_ctrlrange,
                        env.mj_model.actuator_ctrllimited,jnp)
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
                if placement_capture:
                    if probe_residual is not None:
                        return (next_state,diag,h4m3_post_capture(pre_placement,action,next_state,env),
                                probe_meta)
                    return next_state,diag,h4m3_post_capture(pre_placement,action,next_state,env)
                return next_state,diag
            step_fn=(jax.jit(one_step,static_argnames=('placement_capture',))
                     if placement or m6 else jax.jit(one_step))
            if feasibility:
                session=args.h3_session
                h4m4_native_lane(args.h4m4_lanes[lane_index],session,env,step_fn,jax,jnp,np,
                    pulse=lambda: h4m3_progress(args,session))
                h4m3_progress(args,session)
                continue
            if placement:
                session=args.h3_session
                h4m3_require(h4m3_authorization()==session['authorization'],'AUTH_CHANGED')
                h4m3_native_lane(args.h4m3_lanes[lane_index],session,env,step_fn,
                    session['manifest'],jax,jnp,np,
                    pulse=lambda: h4m3_progress(args,session))
                h4m3_progress(args,session)
                continue
            if h3_mode:
                session=args.h3_session
                if cc:
                    h4cc_eval_require(h4cc_eval_authorization()==session['authorization'],'AUTHORIZATION_CHANGED')
                if m6:
                    h4m6_eval_require(trainer.h4m6_authorization(phase='dev')==auth,'DEV_AUTH_CHANGED')
                    def capture(label,seed,profile,direction,tick,row,raw):
                        h4m6_add_rich(args.h4m6_rich,label,seed,direction,tick,row,raw,np)
                    g=h3_group(args,env,step_fn,jax,jnp,np,session,name,seed,profile,
                               H4_EVAL_PHYSICAL,rich=capture)
                    h4m6_rich_write(args.h4m6_rich)
                else:
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
    if placement:
        return
    if h3_mode:
        if m6:
            session=args.h3_session
            session['decision']=h4m6_dev_decisions(args.h4m6_archive,session)
            session['ledger']=h4m6_dev_ledger(session)
            projection=h4m6_rich_projection(session,args.h4m6_rich)
            endpoints=h4m6_mechanism_endpoints(session,args.h4m6_rich,np)
            session['mechanism']=h4m6_mechanism_decision(session['decision']['decision'],
                session['exposure']['decision'],endpoints)
            session['mechanism']['endpoints']=endpoints
            args.h4m6_rich['projection']=projection
            h4m6_rich_write(args.h4m6_rich)
            session['mechanism_sha256']=sha256(H4_EVAL_ROOT/H4M6_RICH_OUTPUT)
            return
        if cc:
            args.h3_session['decision']=h4cc_pair_decisions(args.h4cc_archive,args.h3_session)
            return
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



# H4-M2-3: isolated diagnostic only, no optimizer or efficacy evaluation.
H4CC_TECH_EXECUTION_AUTHORIZED = False
H4CC_TECH_ID = 'H4-M2-3-TECH-V1'
H4CC_TECH_OUTPUT = Path('m463_runs/m470_h4_m2_tech_v1/report.json')
H4CC_TECH_GENERATED_SHA = 'd5c03ebcbac9638c75b62161e332d15e7b2b3c671196ce1d11dce46bdef42e68'
H4CC_TECH_FIXTURES = (
    (4101,3,2,295,327,335,335),
    (4102,3,3,295,327,335,335),
    (4103,3,2,295,327,335,100),
    (4104,1,3,305,327,335,335),
)
H4CC_TECH_SPEC = {'id':H4CC_TECH_ID,'fixtures':[list(x) for x in H4CC_TECH_FIXTURES],
    'fixture_columns':['seed','class','direction','first_age','next','event_age','source_clock'],
    'arms':['ref','ccrc'],'max_physical':2016,'max_reset_lanes':8,
    'max_lane_steps':252,'hold_ticks':250,'action_repeat':1,'episode_length':1000,
    'deterministic_actor':True,'synthetic_prior_event':True,'prior_actual_pushes':0,
    'dr_seed_fold_in':41,'actor_seed_fold_in':42,'training_performed':False,
    'efficacy_evaluated':False,'optimizer_updates':0,'evaluation_transitions':0,
    'checkpoint_saves':0}

def h4cc_tech_require(ok,label):
    if not bool(ok):
        raise RuntimeError('M470_H4CC_TECH_'+label)

def h4cc_tech_parse_marker(rules):
    import re
    h4cc_tech_require(not any(line.startswith(('H4X_ACTIVE_GO:','H4CC_ACTIVE_GO:',
        'H4CC_EVAL_ACTIVE_GO:')) for line in rules.splitlines()),'AUTH_CONFLICT')
    lines=[line for line in rules.splitlines() if line.startswith('H4CC_TECH_ACTIVE_GO:')]
    h4cc_tech_require(len(lines)==1,'EXECUTION_HOLD')
    m=re.fullmatch(r'H4CC_TECH_ACTIVE_GO: (H4-M2-3-TECH-V1) '
        r'([0-9a-f]{64}) ([0-9a-f]{64}) ([0-9a-f]{64})',lines[0])
    h4cc_tech_require(m is not None,'AUTH_SCHEMA')
    ident,source,generated,static=m.groups()
    approved=re.findall(r'^H4CC_APPROVED_TECH_STATIC_SHA256: ([0-9a-f]{64})$',rules,re.M)
    h4cc_tech_require(approved==[static] and generated==H4CC_TECH_GENERATED_SHA,
                       'AUTH_BINDING')
    return dict(id=ident,source_sha256=source,generated_sha256=generated,
                static_sha256=static,output=str(H4CC_TECH_OUTPUT))

def h4cc_tech_authorization():
    import m463_push_ppo_train as t
    h4cc_tech_require(not any((H4CC_TECH_EXECUTION_AUTHORIZED,H4CC_EVAL_EXECUTION_AUTHORIZED,
        H4_EVAL_EXECUTION_AUTHORIZED,t.H4CC_EXECUTION_AUTHORIZED,t.H4_EXECUTION_AUTHORIZED,
        t.H4R_EXECUTION_AUTHORIZED,t.H4C_EXECUTION_AUTHORIZED)),'BROAD_HOLD_REQUIRED')
    a=h4cc_tech_parse_marker((H4_EVAL_ROOT/'G1_RESEARCH_RULES.md').read_text('utf8'))
    h4cc_tech_require(t.h4x_source_digest(H4_EVAL_ROOT)==a['source_sha256'],'SOURCE_PIN')
    return a

def h4cc_tech_inputs(args,auth):
    import m463_push_ppo_train as t
    from types import SimpleNamespace
    from importlib.machinery import PathFinder
    paths={'C401M':args.checkpoint_c,'reward':args.h4_reward_report,
           'restore':args.h4cc_restore_report,'static':args.h4cc_static_log}
    expected={'C401M':BASE_SHA,'reward':t.H4R_REWARD_SHA,
              'restore':t.H3_CAUSAL_RESTORE_SHA,'static':auth['static_sha256']}
    for k,p in paths.items():
        t.h4x_file_pin(p,expected[k])
    static=paths['static'].read_text('utf8').splitlines()
    h4cc_tech_require(static.count('H4_M2_3A_HELD_STATIC_PASS')==1 and
        'H4CC_TECH_COMMON_SOURCE_SHA256 '+auth['source_sha256'] in static and
        'H4CC_TECH_GENERATED_SHA256 '+auth['generated_sha256'] in static,'STATIC_EVIDENCE')
    bootstrap=t.h4x_bootstrap(SimpleNamespace(checkpoint=paths['C401M'],
        restore_report=paths['restore'],h4_exploratory_reward_report=paths['reward']),h4cc=True)
    for name in ('m463_coverage_env','m463_gate3_cp_eval'):
        spec=PathFinder.find_spec(name,sys.path)
        h4cc_tech_require(spec is not None and spec.origin is not None and
            Path(spec.origin).resolve()==H4_EVAL_ROOT/(name+'.py'),'MODULE_ORIGIN '+name)
    return {'input_sha256':expected,'input_paths':{k:str(p.resolve()) for k,p in paths.items()},
            'bootstrap':bootstrap,'source_sha256':auth['source_sha256']}

def h4cc_tech_fixture_values(fixture):
    seed,cls,direction,first,target,age,clock=fixture
    h4cc_tech_require(fixture in H4CC_TECH_FIXTURES and age==target+8 and
        (30<=target-first<=32 and 30<=age-first<=40 if cls==3
         else 15<=target-first<=30),'FIXTURE_METADATA')
    return {'steps':age,'step':clock,'m463_age':age,'m463_next':target,
            'm463_direction':direction,'m463_phase_target':0,'m463_no_push':0,
            'm463_push_count':1,'m463_last_event_age':first,
            'm463_recent':100-(age-first-1),'h4_class':cls,
            'h4_first_pending':0,'h4_repeat_pending':1,'h4_first_age':first,
            'h4_first_mag':.5,'h4_first_motion':0,'h4_assign_motion':0,
            'h4_tail_kind':1,'h4_tail_cell':0,'h4_tail_age':age-first-1,
            'h4_tail_same':1,'h4_tail_motion':0}

def h4cc_tech_audit(report,*,partial=False):
    """Pure replay of bounded raw rows. No JAX, simulator or policy import."""
    h4cc_tech_require(report['spec']==H4CC_TECH_SPEC,'SPEC')
    lanes=report['lanes']
    h4cc_tech_require(len(lanes)==report['reset_lanes_attempted']==report['reset_lanes_recorded']
        and (0<len(lanes)<=8 if partial else len(lanes)==8),'RESET_LEDGER')
    stand=[0.,0.,0.]
    commands=[stand,[.2,0.,0.],[.6,0.,0.],[.4,0.,.8],[.4,0.,-.8],[0.,.35,0.]]
    # Compare exact float32 values by accepting only their canonical encodings.
    import struct
    import re
    def valid_key(x):
        return isinstance(x,list) and len(x)==2 and all(type(v) is int and 0<=v<2**32 for v in x)
    def valid_sha(x):
        return isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) is not None
    commands=[[struct.unpack('f',struct.pack('f',v))[0] for v in c] for c in commands]
    total=0; distinct_release=0; broad=0; no_refresh_completed=0; terminal_lanes=0
    for index,fixture in enumerate(H4CC_TECH_FIXTURES):
        seed,cls,direction,first,target,age,clock=fixture
        pair=lanes[index*2:index*2+2]
        if not pair:break
        for arm,lane in zip(('ref','ccrc'),pair):
            h4cc_tech_require(lane['arm']==arm and lane['fixture']==list(fixture)
                and lane['metadata']==h4cc_tech_fixture_values(fixture)
                and lane['prior_actual_pushes']==0 and lane['initial_command']==stand
                and lane['native_reset_command'] in commands and valid_key(lane['initial_rng'])
                and valid_sha(lane['initial_sha256']) and valid_sha(lane['model_sha256']),'LANE_ID')
            rows=lane['rows']
            h4cc_tech_require(1<=len(rows)<=252,'LANE_LENGTH')
            total+=len(rows)
            active=False; pending=None; held=stand; same=True; event_side=0
            command=actor=critic=stand; source_clock=clock
            last_key=lane['initial_rng']; last_sha=lane['initial_sha256']
            for i,row in enumerate(rows):
                def req(ok,label):h4cc_tech_require(ok,f'{seed}/{arm}/{i}/'+label)
                m=row['metrics']
                req(row['tick']==i and row['pre_sha256']==last_sha
                    and row['rng_before']==last_key and row['source_clock']==source_clock
                    and row['pre']==command and row['actor']==actor and row['critic']==critic,'CARRY')
                req(row['finite'] is True and row['rng_after']==row['expected_rng']
                    and row['native_draw'] in commands
                    and all(valid_key(row[k]) for k in ('rng_before','rng_after','expected_rng'))
                    and all(valid_sha(row[k]) for k in ('pre_sha256','post_sha256',
                        'action_sha256','physics_sha256')),'FINITE_RNG')
                req(all(type(m[k]) is int and m[k] in (0,1) for k in (
                    'event','repeat','terminal','refresh','bad','cc_bad','start',
                    'event_terminal','tick','tick_same','tick_survived','tail_terminal',
                    'completed','cc_refresh','sample_same','sample_different','deferred',
                    'published','terminal_restored','active','pending_valid','boundary','truncation',
                    'source_terminal','mature','side_matched','side_fallback')),
                    'FLAGS')
                terminal=bool(m['terminal']); refresh=source_clock>=500
                start=(i==0 and cls==3); tick=active and i>0
                scope=start or tick
                req(m['event']==m['repeat']==int(i==0) and m['start']==int(start)
                    and m['mature']==int(start) and m['refresh']==int(refresh)
                    and m['source_terminal']==m['terminal']
                    and m['boundary']==m['truncation']==m['bad']==m['cc_bad']==0
                    and m['age']==age+i and m['wrapper_steps']==age+i+1,'SCHEDULER')
                if i==0:
                    lo,hi=(.30,.40) if cls==3 else (.20,.60)
                    req(m['gap']==age-first and m['direction']==direction
                        and m['side_target']==direction-1 and m['support'] in (0,1,2,3)
                        and m['side_matched']==int(m['support']==direction-1)
                        and m['side_fallback']==1-m['side_matched']
                        and lo<=row['magnitude']<=hi
                        and abs(math.hypot(*row['delta'])-row['magnitude'])<=3e-6,'EVENT')
                    event_side=direction-1 if start and m['side_matched'] else 0
                    req(m['event_qual']==event_side,'EVENT_QUALIFICATION')
                else:
                    req(row['magnitude']==0 and row['delta']==[0.,0.],'EXTRA_PUSH')
                exact=(command==actor==critic==held)
                same=exact and (start or same) if scope else same
                if arm=='ccrc' and scope:req(same,'COMMAND_COHERENCE')
                native=row['native_draw'] if refresh else command
                defer=arm=='ccrc' and scope and refresh and not terminal
                restore=arm=='ccrc' and tick and terminal and pending is not None
                if defer:
                    req(pending is None,'DOUBLE_REFRESH')
                    pending=native
                complete=tick and i==250 and not terminal
                publish=arm=='ccrc' and complete and pending is not None
                keep=(start or (tick and i<250)) and not terminal
                delivered=held if arm=='ccrc' and keep else native
                if publish:delivered=pending
                if restore and not refresh:delivered=pending
                next_obs=stand if terminal else delivered if arm=='ccrc' and (keep or publish) else command
                req(row['post']==delivered and row['next_actor']==row['next_critic']==next_obs,
                    'COMMAND_DELIVERY')
                expect={'event_terminal':start and terminal,'tick':tick,
                    'tick_same':tick and same,'tick_survived':tick and not terminal,
                    'tail_terminal':tick and terminal,'completed':complete,
                    'cc_refresh':scope and refresh,'sample_same':scope and refresh and native==held,
                    'sample_different':scope and refresh and native!=held,
                    'deferred':defer,'published':publish,'terminal_restored':restore,
                    'active':keep,'pending_valid':arm=='ccrc' and keep and pending is not None}
                req(all(m[k]==int(v) for k,v in expect.items()) and
                    m['tick_age']==(i if tick else 0) and m['tick_side']==(event_side if tick else 0)
                    and m['event_side']==event_side,'TAIL_ACCOUNTING')
                if publish and delivered!=held:
                    # Completion alone is insufficient: observe the next physical action.
                    if i+1<len(rows) and rows[i+1]['pre']==rows[i+1]['actor']==rows[i+1]['critic']==delivered:
                        distinct_release+=1
                if seed==4103 and arm=='ccrc' and complete:
                    req(pending is None,'NO_REFRESH_PENDING')
                    no_refresh_completed+=1
                active=keep
                if not keep:pending=None
                req(row['held']==(held if keep else stand) and
                    row['pending']==(pending if pending is not None else stand),'PENDING_VECTOR')
                req(not terminal or (i==len(rows)-1 and not active and
                    row['reset_data_obs_equal'] is True),'TERMINAL_CLOSE')
                if terminal:terminal_lanes+=1
                source_clock=0 if terminal or refresh else source_clock+1
                req(row['next_source_clock']==source_clock,'SOURCE_CLOCK')
                command,actor,critic=delivered,next_obs,next_obs
                last_key=row['rng_after'];last_sha=row['post_sha256']
            h4cc_tech_require(rows[-1]['metrics']['terminal']==1 or len(rows)==252 or
                (partial and lane is lanes[-1]),'SHORT_LANE')
            h4cc_tech_require(lane['physical_attempted']==lane['physical_recorded']==len(rows),'LANE_LEDGER')
        if len(pair)<2:continue
        a,b=pair
        h4cc_tech_require(a['initial_sha256']==b['initial_sha256'] and
            a['initial_rng']==b['initial_rng'] and a['model_sha256']==b['model_sha256'],
            'MATCHED_RESET')
        matched=True
        for x,y in zip(a['rows'],b['rows']):
            if not matched:break
            h4cc_tech_require(x['pre_sha256']==y['pre_sha256'] and
                x['action_sha256']==y['action_sha256'] and x['physics_sha256']==y['physics_sha256']
                and x['rng_after']==y['rng_after'] and x['native_draw']==y['native_draw']
                and x['magnitude']==y['magnitude'] and x['delta']==y['delta'],'MATCHED_PREFIX')
            if (x['post'],x['next_actor'])!=(y['post'],y['next_actor']):
                h4cc_tech_require(cls==3 and y['metrics']['deferred']==1,'UNEXPECTED_DIVERGENCE')
                matched=False
            else:
                h4cc_tech_require(x['post_sha256']==y['post_sha256'],'MATCHED_STATE')
        if matched:
            h4cc_tech_require(len(b['rows'])<=len(a['rows']),'MATCHED_PREFIX_LENGTH')
        if matched and (not partial or b['rows'][-1]['metrics']['terminal']==1 or len(b['rows'])==252):
            h4cc_tech_require(len(a['rows'])==len(b['rows']),'MATCHED_LENGTH')
        if cls==1:
            h4cc_tech_require(matched,'BROAD_NON_TARGET')
            broad+=1
    h4cc_tech_require(total==report['physical_transitions_attempted']==report['physical_transitions_recorded']
        and 0<total<=2016,'PHYSICAL_LEDGER')
    passed=not partial and distinct_release>0 and broad==1
    return {'passed':passed,'decision':'PASS' if passed else 'INCONCLUSIVE',
        'different_draw_release_followed_by_action':distinct_release,'broad_pairs':broad,
        'no_refresh_completed':no_refresh_completed,'terminal_lanes':terminal_lanes,
        'physical_transitions':total}


def h4cc_tech_static_regression():
    """Synthetic report-replay and marker tests only; no physical calls."""
    import copy
    import struct
    passed=[]
    def check(label,ok):
        h4cc_tech_require(ok,'STATIC '+label)
        passed.append(label)
    def reject(label,fn):
        try:
            fn()
        except RuntimeError as exc:
            h4cc_tech_require(str(exc).startswith('M470_H4CC_TECH_'),'STATIC_WRONG_EXCEPTION')
            passed.append(label)
        else:
            raise RuntimeError('M470_H4CC_TECH_STATIC_ACCEPTED '+label)
    stand=[0.,0.,0.]
    move=[struct.unpack('f',struct.pack('f',.6))[0],0.,0.]
    def h(*x):return hashlib.sha256(repr(x).encode()).hexdigest()
    def build(*,different=True,terminal_at=None):
        # Independent synthetic trace: constant reset, fixed event and refresh
        # index, not a real RNG draw, state-machine call, or source execution.
        lanes=[]
        for fixture in H4CC_TECH_FIXTURES:
            seed,cls,direction,first,target,age,clock=fixture
            refresh_i=500-clock
            draw=move if different else stand
            support=0 if seed==4102 else direction-1
            side=direction-1 if cls==3 and support==direction-1 else 0
            for arm in ('ref','ccrc'):
                treatment=(arm=='ccrc' and cls==3)
                initial=h(seed,'reset')
                lane={'fixture':list(fixture),'arm':arm,'metadata':h4cc_tech_fixture_values(fixture),
                    'prior_actual_pushes':0,'initial_command':stand,'native_reset_command':stand,
                    'initial_rng':[seed,0],'initial_sha256':initial,'model_sha256':h(seed,'model'),
                    'rows':[]}
                prev=initial;pre=obs=stand
                stop=terminal_at if terminal_at is not None else 251
                for i in range(stop+1):
                    terminal=terminal_at==i
                    target_event=(cls==3 and i==0)
                    tail=(cls==3 and 1<=i<=250)
                    refresh=i==refresh_i
                    scope=target_event or tail
                    kept=cls==3 and i<250 and not terminal
                    publish=treatment and i==250 and not terminal and refresh_i<=i
                    restored=treatment and tail and terminal and refresh_i<i
                    deferred=treatment and scope and refresh and not terminal
                    valid=treatment and kept and refresh_i<=i
                    if treatment:
                        post=draw if publish or restored or (refresh and terminal) or i>250 and refresh_i<=i else stand
                    else:
                        post=draw if refresh_i<=i else stand
                    next_obs=stand if terminal else post if treatment and (kept or publish) else pre
                    m={'event':int(i==0),'repeat':int(i==0),'terminal':int(terminal),
                        'source_terminal':int(terminal),'mature':int(target_event),'refresh':int(refresh),
                        'bad':0,'cc_bad':0,'boundary':0,'truncation':0,'age':age+i,
                        'wrapper_steps':age+i+1,'gap':age-first if i==0 else 0,
                        'direction':direction,'support':support,'side_target':direction-1 if i==0 else 0,
                        'side_matched':int(i==0 and support==direction-1),
                        'side_fallback':int(i==0 and support!=direction-1),
                        'event_qual':side if i==0 else 0,'event_side':side,
                        'start':int(target_event),'event_terminal':int(target_event and terminal),
                        'tick':int(tail),'tick_age':i if tail else 0,'tick_side':side if tail else 0,
                        'tick_same':int(tail and (treatment or not different or i<=refresh_i)),
                        'tick_survived':int(tail and not terminal),'tail_terminal':int(tail and terminal),
                        'completed':int(tail and i==250 and not terminal),
                        'cc_refresh':int(scope and refresh),'sample_same':int(scope and refresh and not different),
                        'sample_different':int(scope and refresh and different),'deferred':int(deferred),
                        'published':int(publish),'terminal_restored':int(restored),'active':int(kept),
                        'pending_valid':int(valid)}
                    post_sha=h(seed,i,post,next_obs,terminal)
                    row={'tick':i,'metrics':m,'pre_sha256':prev,'post_sha256':post_sha,
                        'rng_before':[seed,i],'rng_after':[seed,i+1],'expected_rng':[seed,i+1],
                        'source_clock':clock+i if i<=refresh_i else i-refresh_i-1,
                        'next_source_clock':0 if terminal or refresh else
                            clock+i+1 if i<refresh_i else i-refresh_i,
                        'pre':pre,'actor':obs,'critic':obs,'post':post,'next_actor':next_obs,
                        'next_critic':next_obs,'native_draw':draw,'held':stand,
                        'pending':draw if valid else stand,'finite':True,'reset_data_obs_equal':terminal,
                        'magnitude':.35 if i==0 else 0.,'delta':[0.,.35 if direction==2 else -.35] if i==0 else stand[:2],
                        'action_sha256':h(seed,i,obs),'physics_sha256':h(seed,i,pre,obs,terminal)}
                    lane['rows'].append(row)
                    pre,obs,prev=post,next_obs,post_sha
                lane['physical_attempted']=lane['physical_recorded']=len(lane['rows'])
                lanes.append(lane)
        total=sum(len(l['rows']) for l in lanes)
        return {'spec':copy.deepcopy(H4CC_TECH_SPEC),'lanes':lanes,
            'physical_transitions_attempted':total,'physical_transitions_recorded':total,
            'reset_lanes_attempted':8,'reset_lanes_recorded':8}
    # Metadata arithmetic is predeclared, not a physical first push.
    check('fixture_schedule',all(h4cc_tech_fixture_values(f)['m463_next']+8==f[5]
                                for f in H4CC_TECH_FIXTURES))
    check('budget',len(H4CC_TECH_FIXTURES)*2*H4CC_TECH_SPEC['max_lane_steps']==2016)
    a,b,c='a'*64,H4CC_TECH_GENERATED_SHA,'c'*64
    approved='H4CC_APPROVED_TECH_STATIC_SHA256: '+c+'\n'
    marker='H4CC_TECH_ACTIVE_GO: '+H4CC_TECH_ID+' '+a+' '+b+' '+c
    check('marker_valid',h4cc_tech_parse_marker(approved+marker)['source_sha256']==a)
    for label,value in (('marker_missing',approved),('marker_duplicate',approved+marker+'\n'+marker),
        ('marker_unapproved',marker),('marker_wrong_generated',approved+marker.replace(b,'d'*64)),
        ('marker_wrong_static',approved+marker[:-64]+'d'*64),
        ('marker_approval_duplicate',approved+approved+marker),
        ('marker_h4x_conflict',approved+marker+'\nH4X_ACTIVE_GO: x'),
        ('marker_train_conflict',approved+marker+'\nH4CC_ACTIVE_GO: x'),
        ('marker_eval_conflict',approved+marker+'\nH4CC_EVAL_ACTIVE_GO: x')):
        reject(label,lambda value=value:h4cc_tech_parse_marker(value))
    report=build()
    audit=h4cc_tech_audit(report)
    check('full_replay',audit['passed'] and audit['physical_transitions']==2016
        and audit['different_draw_release_followed_by_action']==2
        and audit['no_refresh_completed']==1 and audit['broad_pairs']==1)
    check('same_draw_inconclusive',h4cc_tech_audit(build(different=False))['decision']=='INCONCLUSIVE')
    for i in (0,1,200,250):
        audit=h4cc_tech_audit(build(terminal_at=i))
        check('terminal_'+str(i),audit['decision']=='INCONCLUSIVE' and audit['terminal_lanes']==8
              and audit['physical_transitions']==8*(i+1))
    # Partial replay must not require completion or report PASS.
    partial=copy.deepcopy(report);partial['lanes']=partial['lanes'][:2]
    partial['lanes'][1]['rows']=partial['lanes'][1]['rows'][:166]
    partial['lanes'][1]['physical_attempted']=partial['lanes'][1]['physical_recorded']=166
    partial['reset_lanes_attempted']=partial['reset_lanes_recorded']=2
    partial['physical_transitions_attempted']=partial['physical_transitions_recorded']=252+166
    check('partial_replay',not h4cc_tech_audit(partial,partial=True)['passed'])
    reject('partial_not_final',lambda:h4cc_tech_audit(partial))
    def bad(label,mutate):
        r=copy.deepcopy(report);mutate(r)
        reject(label,lambda:h4cc_tech_audit(r))
    bad('attempted_mismatch',lambda r:r.update(physical_transitions_attempted=2017))
    bad('reset_overflow',lambda r:r.update(reset_lanes_attempted=9))
    bad('wrong_seed',lambda r:r['lanes'][0]['fixture'].__setitem__(0,4105))
    bad('rng_mismatch',lambda r:r['lanes'][1]['rows'][165].update(rng_after=[0,0]))
    bad('wrong_pending',lambda r:r['lanes'][1]['rows'][165].update(pending=stand))
    bad('early_release',lambda r:r['lanes'][1]['rows'][165].update(post=move))
    bad('late_release',lambda r:r['lanes'][1]['rows'][250].update(next_actor=stand))
    bad('bad_actor',lambda r:r['lanes'][1]['rows'][251].update(actor=stand))
    bad('bad_support',lambda r:r['lanes'][0]['rows'][0]['metrics'].update(side_matched=0))
    bad('bad_tick_side',lambda r:r['lanes'][1]['rows'][10]['metrics'].update(tick_side=0))
    bad('nonfinite',lambda r:r['lanes'][1]['rows'][1].update(finite=False))
    bad('extra_push',lambda r:r['lanes'][1]['rows'][2]['metrics'].update(event=1))
    bad('pair_mismatch',lambda r:r['lanes'][7]['rows'][2].update(physics_sha256=h('bad')))
    bad('model_mismatch',lambda r:r['lanes'][7].update(model_sha256=h('bad')))
    bad('terminal_then_step',lambda r:r['lanes'][1]['rows'][200]['metrics'].update(terminal=1))
    return {'passed':True,'fixtures':passed,'count':len(passed),'physical_transitions':0}

def h4cc_tech_read_report(path,expected_sha,source,generated):
    import m463_push_ppo_train as t
    h4cc_tech_require(path is not None and path.resolve()==H4_EVAL_ROOT/H4CC_TECH_OUTPUT
        and not path.parent.is_symlink(),'REPORT_PATH')
    t.h4x_file_pin(path,expected_sha)
    r=json.loads(path.read_bytes()); a=r['authorization']
    h4cc_tech_require(r['status']=='M470_H4CC_TECH_COMPLETE' and r['passed'] is True
        and r['source_commit']==PIN and a['id']==H4CC_TECH_ID
        and a['output']==str(H4CC_TECH_OUTPUT) and a['source_sha256']==source
        and a['generated_sha256']==generated==H4CC_TECH_GENERATED_SHA
        and r['provenance']['source_sha256']==source
        and r['provenance']['input_sha256']=={'C401M':BASE_SHA,'reward':t.H4R_REWARD_SHA,
            'restore':t.H3_CAUSAL_RESTORE_SHA,'static':a['static_sha256']}
        and r['brax_source']['sha256']==TRAIN_SHA and r['jax_version']=='0.4.35'
        and r['training_performed'] is False and r['efficacy_evaluated'] is False
        and r['promotion_authorized'] is False and r['further_training_authorized'] is False
        and r['ppo_steps']==r['optimizer_updates']==r['evaluation_transitions']==r['checkpoint_saves']==0
        and r['policy_sha256_before']==r['policy_sha256_after']
        and r['gpu']['total_mib']==12288 and 0<r['gpu']['peak_used_mib']<.95*12288
        and r['gpu']['samples']>0 and not r['gpu']['sampler_errors'],'REPORT_PROVENANCE')
    h4cc_tech_require(set(r['brax_dependencies'])==set(t.H4C_BRAX_BLOBS),'DEPENDENCY_SCHEMA')
    from m463_instrument_brax import h4c_git_blob
    root=Path(r['brax_source']['path']).resolve().parents[3]
    t.h4x_file_pin(Path(r['brax_source']['path']),TRAIN_SHA)
    for name,blob in t.H4C_BRAX_BLOBS.items():
        p=root.joinpath(*name.split('.')[1:]).with_suffix('.py')
        h4cc_tech_require(r['brax_dependencies'][name]=={'path':str(p),'blob':blob}
            and p.is_file() and h4c_git_blob(p.read_bytes())==blob,'REPORT_DEP '+name)
    h4cc_tech_require(set(r['wrapper_pins'])=={'episode','autoreset'},'WRAPPER_SCHEMA')
    for k,blob in (('episode','2c09c8268a29d14c832dd07050575e4541ee98af'),
                   ('autoreset','62e2823b80b692befe29e4640d0b7a816e52271c')):
        item=r['wrapper_pins'][k];p=Path(item['path'])
        h4cc_tech_require(item['blob']==blob and p.is_file() and
            h4c_git_blob(p.read_bytes())==blob,'REPORT_WRAPPER '+k)
    rules=(H4_EVAL_ROOT/'G1_RESEARCH_RULES.md').read_text('utf8')
    h4cc_tech_require(t.h4cc_approved_sha(rules,'TECH_STATIC')==a['static_sha256'],'STATIC_APPROVAL')
    for name,value in (('attempt.json',{'authorization':a,'reserved':True}),
                       ('dispatch.started.json',{'authorization':a}),
                       ('attempt_result.json',{'status':'COMPLETE','authorization':a})):
        p=path.parent/name
        h4cc_tech_require(p.is_file() and not p.is_symlink() and
            json.loads(p.read_bytes())==value,'ATTEMPT '+name)
    h4cc_tech_require(set(r['provenance']['input_paths'])==set(r['provenance']['input_sha256']),'INPUT_PATH_SCHEMA')
    from types import SimpleNamespace
    paths={k:Path(p) for k,p in r['provenance']['input_paths'].items()}
    proxy=SimpleNamespace(checkpoint_c=paths['C401M'],h4_reward_report=paths['reward'],
        h4cc_restore_report=paths['restore'],h4cc_static_log=paths['static'])
    h4cc_tech_require(h4cc_tech_inputs(proxy,a)==r['provenance'],'BOOTSTRAP_REPLAY')
    audit=h4cc_tech_audit(r)
    h4cc_tech_require(audit==r['audit'] and audit['passed'],'REPORT_REPLAY')
    return r

def h4cc_tech_actual(args,r,monitor,persist,*,rss=False,diagnostic=False):
    # Independent one-use dispatch check BEFORE runtime imports.
    import m463_push_ppo_train as t
    h4cc_tech_require(not diagnostic or rss,'DIAG_RSS_REQUIRED')
    if rss and not diagnostic:
        h4m6_runtime_gate('tech')
    auth=h4m6_diag_authorization() if diagnostic else (t.h4m6_authorization(phase='tech') if rss else h4cc_tech_authorization())
    expected_out=H4M6_DIAG_OUTPUT if diagnostic else (H4M6_TECH_OUTPUT if rss else H4CC_TECH_OUTPUT)
    expected_status='M470_H4M6_DIAG_IN_PROGRESS' if diagnostic else ('M470_H4M6_TECH_IN_PROGRESS' if rss else 'M470_H4CC_TECH_IN_PROGRESS')
    h4cc_tech_require(args.output.resolve()==H4_EVAL_ROOT/expected_out
        and r['authorization']==auth and r['status']==expected_status
        and r['lanes']==[] and all(r[k]==0 for k in ('physical_transitions_attempted',
            'physical_transitions_recorded','reset_lanes_attempted','reset_lanes_recorded'))
        and r['provenance']==(h4m6_tech_inputs(args,auth,diagnostic=diagnostic) if rss else h4cc_tech_inputs(args,auth)),'DISPATCH_CONTRACT')
    reservation=args.output.parent/'attempt.json'
    h4cc_tech_require(reservation.is_file() and not reservation.is_symlink()
        and json.loads(reservation.read_bytes())=={'authorization':auth,'reserved':True}
        and args.output.is_file() and not args.output.is_symlink()
        and json.loads(args.output.read_bytes())==r,'DISPATCH_RESERVATION')
    t.h4x_write_json(args.output.parent/'dispatch.started.json',{'authorization':auth},exclusive=True)
    import copy
    from train import check_source,build_paper_config
    from paper_config import BASE,ENV,REPO,PIN as SOURCE_PIN
    check_source()
    h4cc_tech_require(SOURCE_PIN==PIN,'SOURCE')
    sys.path.insert(0,str(BASE/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    r['nvcc']=apply()
    import jax
    import jax.numpy as jp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    from mujoco_playground import locomotion
    from mujoco_playground._src import wrapper
    import m463_coverage_env as ce
    from m462_typed_orbax_stage import checkpoint_parts
    from m463_instrument_brax import h4c_git_blob
    h4cc_tech_require(all(Path(m.__file__).resolve().is_relative_to(REPO.resolve())
        for m in (locomotion,wrapper,ce.joystick)),'PLAYGROUND_IMPORT_ORIGIN')
    h4cc_tech_require(jax.default_backend()=='gpu' and jax.__version__=='0.4.35'
        and jax.process_count()==jax.local_device_count()==1,'GPU_RUNTIME')
    r['jax_version']=jax.__version__;r['brax_source']=h3_verify_brax_train_source(networks)
    root=Path(r['brax_source']['path']).parents[3]
    from importlib.util import find_spec
    deps={}
    for name,blob in t.H4C_BRAX_BLOBS.items():
        p=root.joinpath(*name.split('.')[1:]).with_suffix('.py')
        spec=find_spec(name)
        h4cc_tech_require(spec is not None and spec.origin is not None and
            Path(spec.origin).resolve()==p and p.is_file() and
            h4c_git_blob(p.read_bytes())==blob,'BRAX_DEP '+name)
        deps[name]={'path':str(p),'blob':blob}
    r['brax_dependencies']=deps
    r['wrapper_pins']={'episode':{'path':str(root/'envs/wrappers/training.py'),
        'blob':'2c09c8268a29d14c832dd07050575e4541ee98af'},'autoreset':{
        'path':str(Path(wrapper.__file__).resolve()),'blob':'62e2823b80b692befe29e4640d0b7a816e52271c'}}
    h4cc_tech_require(h4c_git_blob((root/'envs/wrappers/training.py').read_bytes())==
        '2c09c8268a29d14c832dd07050575e4541ee98af','EPISODE_WRAPPER')
    h4cc_tech_require(h4c_git_blob(Path(wrapper.__file__).read_bytes())==
        '62e2823b80b692befe29e4640d0b7a816e52271c','AUTORESET_WRAPPER')
    paper=build_paper_config();cfg=locomotion.get_default_config(ENV)
    reward=json.loads(args.h4_reward_report.read_bytes())['train_reward_config']
    h4cc_tech_require(cfg.reward_config.to_dict()==reward and bool(cfg.push_config.enable)
        and list(cfg.push_config.interval_range)==[5.,10.] and
        list(cfg.push_config.magnitude_range)==[.1,2.],'SOURCE_CONFIG')
    cfg.push_config.enable=False
    shapes={'state':(103,),'privileged_state':(216,)}
    nets=networks.make_ppo_networks(shapes,29,
        preprocess_observations_fn=running_statistics.normalize,**dict(paper.network_factory))
    n,p,v=checkpoint_parts(serialization.msgpack_restore(args.checkpoint_c.read_bytes()))
    norm=serialization.from_state_dict(running_statistics.init_state(
        {k:jp.zeros(s,jp.float32) for k,s in shapes.items()}),n)
    if diagnostic:r['template_initializations_attempted']+=1;persist()
    actor=serialization.from_state_dict(nets.policy_network.init(jax.random.PRNGKey(11)),p)
    if diagnostic:
        jax.block_until_ready(actor);r['template_initializations_recorded']+=1
        r['template_initializations_attempted']+=1;persist()
    critic=serialization.from_state_dict(nets.value_network.init(jax.random.PRNGKey(17)),v)
    if diagnostic:jax.block_until_ready(critic);r['template_initializations_recorded']+=1;persist()
    def host(x):return np.asarray(jax.device_get(x))
    def digest_tree(tree):
        h=hashlib.sha256()
        pairs,treedef=jax.tree_util.tree_flatten_with_path(tree)
        h.update(str(treedef).encode())
        for key,x in pairs:
            a=host(x);h.update(str(key).encode());h.update(a.dtype.str.encode())
            h.update(str(a.shape).encode());h.update(a.tobytes())
        return h.hexdigest()
    params=(norm,actor,critic)
    leaves=jax.tree_util.tree_leaves(params)
    h4cc_tech_require(len(leaves)==25 and all(np.isfinite(host(x)).all() for x in leaves),'POLICY')
    r['policy_sha256_before']=digest_tree(params)
    policy=networks.make_inference_fn(nets)((norm,actor),deterministic=True)
    randomizer=locomotion.get_domain_randomizer(ENV)
    h4cc_tech_require(randomizer is not None,'DR')
    # Raw finite checks BEFORE the pinned AutoReset replaces terminal data.
    class RawAudit(wrapper.Wrapper):
        def raw(self,st):
            return (st.data.qpos,st.data.qvel,st.data.qacc,st.data.ctrl,
                st.data.actuator_force,st.data.sensordata,st.obs['state'],st.obs['privileged_state'])
        def finite(self,st):
            finite=jp.ones_like(st.done,dtype=jp.bool_)
            for x in (*self.raw(st),st.reward,*jax.tree_util.tree_leaves(st.metrics)):
                finite=finite & jp.all(jp.isfinite(x).reshape((1,-1)),axis=1)
            return finite
        def reset(self,rng):
            st=self.env.reset(rng)
            return st.replace(info={**st.info,'h4cc_tech_raw_finite':self.finite(st),
                                    'h4cc_tech_raw':self.raw(st)})
        def step(self,st,act):
            nxt=self.env.step(st,act)
            finite=self.finite(nxt) & jp.all(jp.isfinite(act).reshape((1,-1)),axis=1)
            return nxt.replace(info={**nxt.info,'h4cc_tech_raw_finite':finite,'h4cc_tech_raw':self.raw(nxt)})
    def state_digest(st):
        if rss:
            # Exclude only intervention metadata and its scalar/metric outputs.
            # Keep full physics, obs, RNG and every common CCRC shadow field.
            return digest_tree(st.replace(
                info={k:v for k,v in st.info.items() if not k.startswith('h4m6_')},
                reward=jp.zeros_like(st.reward),
                metrics={k:v for k,v in st.metrics.items() if k!='reward/stand_still'}))
        # Only additive CCRC metadata omitted; all common state bytes compared.
        return digest_tree(st.replace(info={k:v for k,v in st.info.items()
                                            if not k.startswith('h4cc_')}))
    metric_keys={'event':'m463_event','repeat':'m463_repeat','mature':'h4_mature_repeat',
        'refresh':'h4_refresh','bad':'h4_bad','cc_bad':'h4cc_bad','age':'h4_age',
        'source_terminal':'h4_source_terminal','event_qual':'h4_event_qual',
        'boundary':'h4_boundary','truncation':'truncation','wrapper_steps':'steps',
        'gap':'m463_repeat_gap','direction':'m463_direction','support':'m463_support_side',
        'side_target':'m463_repeat_side_target','side_matched':'m463_repeat_side_matched',
        'side_fallback':'m463_repeat_side_fallback'}
    for k in ('start','event_terminal','tick','tick_age','tick_side','event_side','tick_same','tick_survived',
        'tail_terminal','completed','sample_same','sample_different','deferred','published',
        'terminal_restored','active','pending_valid'):
        metric_keys[k]='h4cc_'+k
    metric_keys['cc_refresh']='h4cc_refresh'
    # Independent pure key accounting, not a mutation or extra env sample.
    def key_audit(key):
        key=jax.random.split(key,8)[0]
        key=jax.random.split(key,3)[0]
        for _ in range(5):key=jax.random.split(key)[0]
        key,cmd=jax.random.split(key)
        return key,ce.COMMANDS[jax.random.randint(cmd,(),0,6)]
    key_audit=jax.jit(jax.vmap(key_audit))
    stand=jp.zeros((1,3),jp.float32)
    def sync_obs(obs):
        return {**obs,'state':obs['state'].at[:,9:12].set(stand),
            'privileged_state':obs['privileged_state'].at[:,9:12].set(stand)}
    def vec(x):return host(x)[0].tolist()
    def scalar(x):return int(host(x)[0])
    def guard():
        h4cc_tech_require((h4m6_diag_authorization() if diagnostic else (t.h4m6_authorization(phase='tech') if rss else h4cc_tech_authorization()))==r['authorization'],'AUTH_CHANGED')
        h4cc_tech_require('brax.training.agents.ppo.train' not in sys.modules,'TRAIN_IMPORT')
        h3_gpu_check(monitor)
    cache={}
    if diagnostic:
        import jaxlib
        h4m6_eval_require(jaxlib.__version__=='0.4.35' and jax.device_count()==1,'DIAG_DEVICE')
        historical=json.loads((H4_EVAL_ROOT/'m463_runs/m470_h4_m6_tech_v3/report.json').read_bytes())
        historical={l['arm']:l for l in historical['lanes'] if l['fixture'][0]==4102}
        r['compiler']={'jax':jax.__version__,'jaxlib':jaxlib.__version__,'device':str(jax.devices()[0]),
            'platform_version':jax.devices()[0].client.platform_version,
            'env':{k:v for k,v in os.environ.items() if k.startswith(('JAX_','XLA_','CUDA_','NVIDIA_'))}}
        def capture(st,lane,tick,expected):
            projected=st.replace(info={k:v for k,v in st.info.items() if not k.startswith('h4m6_')},
                reward=jp.zeros_like(st.reward),metrics={k:v for k,v in st.metrics.items() if k!='reward/stand_still'})
            return h4m6_diag_capture(r,cache,lane,tick,{'full':st,'projected':projected,
                'physics':(st.data,st.info['h4cc_tech_raw'])},jax.tree_util.tree_flatten_with_path,host,expected)
    for fixture in ((tuple(H4M6_DIAG_SPEC['fixture']),) if diagnostic else H4CC_TECH_FIXTURES):
        seed=fixture[0]; key=jax.random.PRNGKey(seed)
        dr_keys=jax.random.fold_in(key,H4CC_TECH_SPEC['dr_seed_fold_in'])[None,:]
        shared_step=None;shared_reset=None
        for lane_id in (H4M6_DIAG_ORDER if diagnostic else (('ref','rss') if rss else ('ref','ccrc'))):
            arm=lane_id[:3].lower() if diagnostic else lane_id
            guard()
            if (shared_step is None if diagnostic else (not rss or arm=='ref')):
                env=ce.CoverageJoystick(stage='easy_repeat_support',config=copy.deepcopy(cfg),
                                        h4_arm='candidate',h4cc_arm='ref' if rss else arm,
                                        **({'h4m6_arm':'ref','h4m6_tech_capture':True} if rss else {}))
                h4cc_tech_require(env.action_size==29 and float(env.dt)==.02,'ENV_CONFIG')
                wrapped=wrapper.wrap_for_brax_training(env,episode_length=1000,action_repeat=1,
                    randomization_fn=functools.partial(randomizer,rng=dr_keys))
                # Insert observer between EpisodeWrapper and the unchanged AutoReset.
                h4cc_tech_require(isinstance(wrapped,wrapper.BraxAutoResetWrapper),'WRAPPER_TYPE')
                dr_wrapper=wrapped.env.env
                model_sha=t.h4x_model_digest(jax,np,dr_wrapper._mjx_model_v,dr_wrapper._in_axes)
                wrapped.env=RawAudit(wrapped.env)
                if rss:
                    def shared_one(state,actor_key):
                        action,_=policy(state.obs,actor_key)
                        return wrapped.step(state,action),action
                    shared_step=jax.jit(shared_one)
                    shared_reset=jax.jit(wrapped.reset)
            h4cc_tech_require(r['reset_lanes_attempted']<(4 if diagnostic else 8),'RESET_CAP')
            r['reset_lanes_attempted']+=1;persist()
            st=(shared_reset if rss else jax.jit(wrapped.reset))(key[None,:])
            h4cc_tech_require(bool(host(st.info['h4cc_tech_raw_finite'])[0]) and
                scalar(st.done)==0,'RESET_FINITE')
            info=dict(st.info); original_command=vec(info['command'])
            metadata=h4cc_tech_fixture_values(fixture)
            for k,val in metadata.items():
                info[k]=jp.full_like(info[k],val)
            info['command']=stand
            if rss:
                info['h4m6_tech_treatment']=jp.full_like(info['h4m6_tech_treatment'],int(arm=='rss'))
            obs=sync_obs(st.obs);info['first_obs']=sync_obs(info['first_obs'])
            st=st.replace(info=info,obs=obs)
            lane={'fixture':list(fixture),'arm':arm,'metadata':metadata,'prior_actual_pushes':0,
                'native_reset_command':original_command,'initial_command':vec(info['command']),
                'initial_rng':vec(info['rng']),'initial_sha256':state_digest(st),
                'model_sha256':model_sha,'rows':[],'physical_attempted':0,'physical_recorded':0}
            if diagnostic:
                lane['diag_id']=lane_id
                lane['historical_initial']={k:lane[k]==historical[arm][k] for k in ('initial_sha256','initial_rng','model_sha256')}
                lane['historical_rows']=[]
            r['lanes'].append(lane);r['reset_lanes_recorded']+=1;persist()
            if diagnostic and capture(st,lane,-1,{'projected':lane['initial_sha256']}):
                lane['endpoint']='DIVERGENCE';persist();continue
            if diagnostic:persist()
            def one(state,actor_key):
                action,_=policy(state.obs,actor_key)
                nxt=wrapped.step(state,action)
                return nxt,action
            step=shared_step if rss else jax.jit(one)
            for tick in range(8 if diagnostic else (52 if rss else 252)):
                guard()
                h4cc_tech_require(r['physical_transitions_attempted']<(32 if diagnostic else (416 if rss else 2016)),'PHYSICAL_CAP')
                if diagnostic:
                    h4cc_tech_require(r['key_audits_attempted']<32,'DIAG_KEY_CAP')
                    r['key_audits_attempted']+=1;persist()
                pre=st.info;expected_rng,native_draw=key_audit(pre['rng'])
                if diagnostic:
                    host(expected_rng);host(native_draw);r['key_audits_recorded']+=1
                row={'tick':tick,'pre_sha256':state_digest(st),'rng_before':vec(pre['rng']),
                    'source_clock':scalar(pre['step']),'pre':vec(pre['command']),
                    'actor':vec(st.obs['state'][:,9:12]),'critic':vec(st.obs['privileged_state'][:,9:12]),
                    'expected_rng':vec(expected_rng),'native_draw':vec(native_draw)}
                r['physical_transitions_attempted']+=1;lane['physical_attempted']+=1
                if diagnostic:r['actor_calls_attempted']+=1
                persist()
                actor_key=jax.random.fold_in(jax.random.fold_in(key,H4CC_TECH_SPEC['actor_seed_fold_in']),tick)
                nxt,action=step(st,actor_key)
                m={k:scalar(nxt.info[v]) for k,v in metric_keys.items()}
                m['terminal']=scalar(nxt.done)
                finite=bool(host(nxt.info['h4cc_tech_raw_finite'])[0])
                equal=all(np.array_equal(host(x),host(y)) for x,y in (
                    (nxt.data.qpos,nxt.info['first_state'].qpos),
                    (nxt.data.qvel,nxt.info['first_state'].qvel),
                    (nxt.obs['state'],nxt.info['first_obs']['state']),
                    (nxt.obs['privileged_state'],nxt.info['first_obs']['privileged_state'])))
                row.update(metrics=m,finite=finite,reset_data_obs_equal=equal,
                    post=vec(nxt.info['command']),next_actor=vec(nxt.obs['state'][:,9:12]),
                    next_critic=vec(nxt.obs['privileged_state'][:,9:12]),
                    held=vec(nxt.info['h4cc_held_command']),pending=vec(nxt.info['h4cc_pending_command']),
                    rng_after=vec(nxt.info['rng']),next_source_clock=scalar(nxt.info['step']),
                    magnitude=float(host(nxt.info['m463_magnitude'])[0]),
                    delta=[float(host(nxt.info[k])[0]) for k in ('h4_actual_dx','h4_actual_dy')],
                    action_sha256=digest_tree(action),physics_sha256=digest_tree((nxt.data,nxt.reward,nxt.metrics,
                        nxt.info['h4cc_tech_raw'][:6])),
                    post_sha256=state_digest(nxt))
                if rss:
                    names=ce.H4M6_REWARD_NAMES
                    row.update(rss={k:scalar(nxt.info['h4m6_'+k]) for k in ce.H4M6_INT_FIELDS},
                        reward_names=list(names),raw_rewards=vec(nxt.info['h4m6_raw_rewards']),
                        reward_native_types=['bool' if k in ce.H4M6_BOOL_REWARD_NAMES else 'float32' for k in names],
                        weighted_rewards=[float(host(nxt.metrics['reward/'+k])[0]) for k in names],
                        reward_scales=[float(cfg.reward_config.scales[k]) for k in names],
                        reward=float(host(nxt.reward)[0]),
                        physics_sha256=digest_tree((nxt.data,nxt.info['h4cc_tech_raw'])))
                    row['rss'].update({k:float(host(nxt.info['h4m6_'+k])[0]) for k in ce.H4M6_FLOAT_FIELDS})
                    h4m6_eval_require(scalar(nxt.info['h4m6_tech_treatment'])==int(arm=='rss'),'TECH_TREATMENT_BIT')
                lane['rows'].append(row);lane['physical_recorded']+=1
                r['physical_transitions_recorded']+=1
                if diagnostic:r['actor_calls_recorded']+=1
                persist()
                h4cc_tech_require(finite and not m['bad'] and not m['cc_bad'] and not m['truncation']
                    and row['rng_after']==row['expected_rng'],'LIVE_CONTRACT')
                if diagnostic:
                    h4m6_technical_audit(r,partial=True,diagnostic=True)
                    h4m6_eval_require(r['snapshots'][lane_id+':'+str(tick-1)]['projected']['aggregate']==row['pre_sha256'],'DIAG_PRE_SNAPSHOT')
                    archived=historical[arm]['rows'][tick]
                    lane['historical_rows'].append({k:row[k]==archived[k] for k in ('pre_sha256','post_sha256','physics_sha256','action_sha256')})
                    mismatch=capture(nxt,lane,tick,{'projected':row['post_sha256'],'physics':row['physics_sha256']})
                    if mismatch:lane['endpoint']='DIVERGENCE'
                    elif m['terminal']:lane['endpoint']='TERMINAL'
                    elif tick==7:lane['endpoint']='HORIZON'
                    persist()
                    if mismatch:break
                else:(h4m6_technical_audit if rss else h4cc_tech_audit)(r,partial=True)
                st=nxt
                if m['terminal']:break
    r['policy_sha256_after']=digest_tree(params)
    h4cc_tech_require(r['policy_sha256_after']==r['policy_sha256_before'],'POLICY_MUTATED')
    guard()
    check_source()
    r['audit']=h4m6_diag_ledger(r) if diagnostic else (h4m6_technical_audit if rss else h4cc_tech_audit)(r)

def h4cc_tech_run(args):
    import m463_push_ppo_train as t
    auth=h4cc_tech_authorization()
    provenance=h4cc_tech_inputs(args,auth)
    h4cc_tech_require(args.mode=='run' and getattr(args,'h4cc_technical',False)
        and args.output.absolute()==H4_EVAL_ROOT/H4CC_TECH_OUTPUT
        and args.output.resolve()==H4_EVAL_ROOT/H4CC_TECH_OUTPUT,'RUN_PATH')
    out=args.output.parent
    h4cc_tech_require(out.parent.is_dir() and not out.parent.is_symlink()
        and not out.exists() and not out.is_symlink(),'OUTPUT_CONSUMED')
    t.h4x_storage(out,initial=True)
    out.mkdir(exist_ok=False)
    t.h4x_write_json(out/'attempt.json',{'authorization':auth,'reserved':True},exclusive=True)
    r={'status':'M470_H4CC_TECH_IN_PROGRESS','spec':H4CC_TECH_SPEC,'authorization':auth,
        'provenance':provenance,'source_commit':PIN,'lanes':[],
        'physical_transitions_attempted':0,'physical_transitions_recorded':0,
        'reset_lanes_attempted':0,'reset_lanes_recorded':0,'training_performed':False,
        'efficacy_evaluated':False,'promotion_authorized':False,'further_training_authorized':False,
        'ppo_steps':0,'optimizer_updates':0,'evaluation_transitions':0,'checkpoint_saves':0,'passed':False}
    def persist():t.h4x_write_json(args.output,r)
    t.h4x_write_json(args.output,r,exclusive=True)
    try:
        from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
        gpu=sample_gpu()
        h4cc_tech_require(gpu['total_mib']==12288 and 0<=gpu['used_mib']<.95*12288,'GPU_PREFLIGHT')
        r['gpu_initial']=gpu;persist()
        monitor=GPUMonitor()
        with monitor:
            h4cc_tech_actual(args,r,monitor,persist)
        r['gpu']=h3_gpu_check(monitor)
        h4cc_tech_require(h4cc_tech_authorization()==auth and h4cc_tech_inputs(args,auth)==provenance,
                           'FINAL_IMMUTABILITY')
        r['passed']=r['audit']['passed']
        r['status']='M470_H4CC_TECH_COMPLETE' if r['passed'] else 'M470_H4CC_TECH_INCONCLUSIVE'
        persist()
        t.h4x_write_json(out/'attempt_result.json',{'status':'COMPLETE' if r['passed'] else 'INCONCLUSIVE',
            'authorization':auth},exclusive=True)
        print(r['status'],json.dumps(r['audit'],sort_keys=True),flush=True)
    except BaseException as exc:
        r['status']='M470_H4CC_TECH_FAILED';r['passed']=False
        r['error']=type(exc).__name__+': '+str(exc)
        try:
            persist()
            t.h4x_write_json(out/'attempt_result.json',{'status':'STOPPED','authorization':auth},exclusive=True)
        except Exception as write_error:
            print('M470_H4CC_TECH_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise



# H4-M3-1: held passive placement diagnostic. No runtime authorization here.
H4M3_OUTPUT = Path('m463_runs/m470_h4_m3_placement_v1/report.json')
H4M3_LANES = (('CCRC',7357,'left',385), ('CCRC',7601,'right',370),
               ('CCRC',7613,'left',375), ('C',7357,'left',366),
               ('REF',7331,'left',385))
H4M3_ARCHIVES = (
    'a231ff30abcd053f3e31e147fe4877fe0fe8da996229930c2d3b5fe69510a476',
    'beaef42162c16c8a8397e5e62164e2c09f443c0b8c5fa7bf3960c7917d3030e8')
H4M3_POLICIES = {
    'C': BASE_SHA,
    'REF': '8ecdbd675fac2fc54764938590d6ed67aa390fd29fc63bcb6f6b5fd5283f38d2',
    'CCRC': '1c144a88f4bf9f1de6d9926b60c75ea417b77931b53a427ca22b8eef11ffb12e'}
H4M3_COLUMNS = tuple(('cause_fall cause_foot_foot cause_inverted_torso '
    'cause_left_foot_right_shin cause_nan_qpos cause_nan_qvel cause_right_foot_left_shin '
    'done foot_center_3d_dist_m foot_center_xy_dist_m foot_forward_sep_body_m '
    'foot_lateral_sep_body_m foot_pair_present foot_pair_signed_dist_m '
    'h2a_ctrl_limit_excess h2a_force_limit_excess h2a_joint_limit_excess '
    'h2a_state_finite h3_finite h3_pose_l1 h3_root_x h3_root_y height '
    'lf_geom_x_m lf_geom_y_m lf_geom_z_m lf_ground_contact lf_site_z_m '
    'phase_left_post_rad phase_right_post_rad reward_orientation reward_termination '
    'reward_total reward_tracking_ang reward_tracking_lin rf_geom_x_m rf_geom_y_m '
    'rf_geom_z_m rf_ground_contact rf_site_z_m tilt_deg vx vy xy_err yaw_err yaw_rate').split())
H4M3_CAP = 2180
H4M3_EXPECTED = 2136
H4M3_RICH_CAP = 710
H4M3_REPORT_CAP = 64 * 1024**2


def h4m3_require(ok, label):
    if not ok:
        raise RuntimeError('M470_H4M3_'+label)


H4M3_BASE_POLICY=Path('m430_seed1/checkpoint_pilot_20261114_120515_1722937_step_401080320.msgpack')
H4M3_RICH_RECORD_CAP=(H4M3_REPORT_CAP-4*1024**2)//H4M3_RICH_CAP
H4M3_SOURCE_NAMES=('train.py','paper_config.py','m462_ppo_warmstart_run.py',
    'm462_typed_orbax_stage.py','m463_coverage_env.py','m463_instrument_brax.py',
    'm463_push_ppo_train.py','m463_gate3_cp_eval.py','paper_config_preflight.json')
H4M3_MJX_BLOBS={'types':'16fa3a8142e35ca0353859b19bc3c0270d05730f',
    'forward':'07f33a90c8ed68bc545ba47bfb78d4bfb8db1f52',
    'support':'57c9475bca1a8641f05525da15190c93102a085c',
    'smooth':'f52391ed37074424c9a95ae4cd6b8330ca2be96b',
    'sensor':'5d040052bfac7e8afc4e221bdcae6826344de2bd'}


def h4m3_digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()


def h4m3_source_map(root):
    result={}
    for name in H4M3_SOURCE_NAMES:
        p=Path(root)/name
        h4m3_require(p.is_file() and not p.is_symlink() and p.resolve()==p.absolute(),
                     'SOURCE_FILE_'+name)
        result[name]=sha256(p)
    return result


def h4m3_parse_authorization(rules,source,spec):
    approved=[l.split() for l in rules.splitlines() if l.startswith('H4M3_STATIC_APPROVED:')]
    active=[l.split() for l in rules.splitlines() if l.startswith('H4M3_ACTIVE_GO:')]
    h4m3_require(len(active)==1 and len(approved)==1,'EXECUTION_HOLD')
    h4m3_require(not any(l.startswith(('H4X_ACTIVE_GO:','H4CC_ACTIVE_GO:',
        'H4CC_EVAL_ACTIVE_GO:','H4CC_TECH_ACTIVE_GO:')) for l in rules.splitlines()),
        'CONCURRENT_AUTHORIZATION')
    a=approved[0]; b=active[0]
    h4m3_require(len(a)==4 and a[1]==source and len(a[2])==64
        and all(c in '0123456789abcdef' for c in a[2]),'STATIC_PIN')
    log=Path(a[3])
    h4m3_require(not log.is_absolute() and log.parent==Path('m463_runs')
        and log.name.startswith('m470_h4_m3_1_') and log.suffix=='.log','STATIC_PATH')
    expected=['H4M3_ACTIVE_GO:','H4-M3-PLACEMENT-V1',source,a[2],spec,
              *H4M3_ARCHIVES,*[H4M3_POLICIES[k] for k in ('C','REF','CCRC')],
              str(H4M3_OUTPUT)]
    h4m3_require(b==expected,'AUTHORIZATION_PIN')
    return {'id':b[1],'source_sha256':source,'static_sha256':a[2],
            'static_path':a[3],'spec_sha256':spec,'archive_sha256':list(H4M3_ARCHIVES),
            'policy_sha256':dict(H4M3_POLICIES),'output':str(H4M3_OUTPUT)}


def h4m3_authorization():
    root=Path(__file__).resolve().parent
    rules=(root/'G1_RESEARCH_RULES.md').read_text('utf8')
    # No source or artifact access on held CLI. Both distinct markers required.
    h4m3_require(any(l.startswith('H4M3_ACTIVE_GO:') for l in rules.splitlines())
        and any(l.startswith('H4M3_STATIC_APPROVED:') for l in rules.splitlines()),
        'EXECUTION_HOLD')
    auth=h4m3_parse_authorization(rules,h4m3_digest(h4m3_source_map(root)),
                                  h4m3_digest(h4m3_plan()))
    p=root/auth['static_path']
    h4m3_require(p.is_file() and not p.is_symlink() and p.resolve()==p.absolute()
        and sha256(p)==auth['static_sha256'] and p.stat().st_size<=16*1024**2,
        'STATIC_LOG')
    lines=p.read_text('utf8').splitlines()
    h4m3_require(('H4_M3_1_FULL_STATIC_PASS '+auth['source_sha256']) in lines,
                  'STATIC_ACCEPTANCE')
    h4m3_require(('H4_M3_1_EXTERNAL_SHA256 '+h4m3_digest(h4m3_external_sources(root)))
                  in lines,'STATIC_EXTERNAL_PIN')
    return auth


def h4m3_plan():
    return {'id':'H4-M3-PLACEMENT-V1','lanes':H4M3_LANES,'archive_sha256':H4M3_ARCHIVES,
            'policy_sha256':H4M3_POLICIES,'max_physical':H4M3_CAP,
            'expected_physical':H4M3_EXPECTED,'max_resets':5,'max_suffixes':10,
            'max_rich_records':H4M3_RICH_CAP,'rich_first_tick':295,
            'prefix_end':334,'end_tick':385,'dt_s':.02,'sim_dt_s':.002,
            'first_push':[300,.50],'second_push':[335,.35],
            'training_performed':False,'efficacy_evaluated':False,
            'optimizer_updates':0,'checkpoint_saves':0,'execution_authorized':False}


def h4m3_pinned_json(path, expected):
    path=Path(path)
    h4m3_require(path.is_file() and not path.is_symlink(),'INPUT_FILE')
    raw=path.read_bytes()
    h4m3_require(hashlib.sha256(raw).hexdigest()==expected,'INPUT_SHA256')
    return json.loads(raw)


def h4m3_archive_lanes(current, historical):
    """Only byte-accepted archives may be passed by h4m3_read_archives."""
    import numpy as np
    for report,status in ((current,'M470_H4CC_EVAL_COMPLETED'),
                          (historical,'M470_H3_EVAL_COMPLETED')):
        h4m3_require(report['status']==status and report['source_commit']==PIN
            and report['brax_train_sha256']==TRAIN_SHA
            and tuple(report['frame_columns'])==H4M3_COLUMNS
            and report['deterministic_actor'] is True
            and report['training_performed'] is False
            and report['domain_randomization_enabled'] is False
            and report['random_source_push_enabled'] is False
            and report['original_reward_PD_termination'] is True
            and (report['push_step'],report['push_magnitude_mps'],report['second_push_step'],
                 report['second_push_magnitude_mps'],report['dt_s'],report['max_steps'])
                ==(300,.50,335,.35,.02,1000),'ARCHIVE_PROTOCOL')
    h4m3_require(historical['input_sha256']['C_policy']==H4M3_POLICIES['C'],
                  'BASE_POLICY_PIN')
    for arm in ('REF','CCRC'):
        h4m3_require(current['input_sha256'][arm.lower()+'/'+H4_EVAL_POLICY_FILE]
                      ==H4M3_POLICIES[arm],'FINAL_POLICY_PIN')
    done=H4M3_COLUMNS.index('done')
    flags=[H4M3_COLUMNS.index(k) for k in H3_EVAL_FLAGS]
    result=[]
    for policy,seed,direction,end in H4M3_LANES:
        report=historical if policy=='C' else current
        groups=[g for g in report['groups'] if (g['policy'],g['seed'],g['profile'])
                ==(policy,seed,'stand')]
        h4m3_require(len(groups)==1 and groups[0]['complete'] is True
                      and groups[0]['command']==[0.,0.,0.],'ARCHIVE_GROUP')
        g=groups[0]; d=g['directions'][direction]
        h4m3_require(d['first']['exposed'] is True and d['second']['exposed'] is True
            and d['first']['step']==300 and d['second']['step']==335,'ARCHIVE_KICKS')
        rows={'nominal':g['nominal_frames'][:299],
              'first':d['single_suffix'][:35],
              'single':d['single_suffix'][35:86],
              'double':d['double_suffix'][:51]}
        h4m3_require(tuple(map(len,rows.values()))==(299,35,51,end-334),
                      'ARCHIVE_LENGTH')
        for segment,raw in rows.items():
            a=np.asarray(raw)
            h4m3_require(a.shape==(len(raw),46) and np.isfinite(a).all()
                and np.isin(a[:,done],[0,1]).all()
                and np.array_equal(a[:,done].astype(bool),a[:,flags].astype(bool).any(1))
                and not a[:-1,done].any()
                and bool(a[-1,done])==(segment=='double' and end<385),
                'ARCHIVE_ROWS')
            h4m3_require(np.all(a[:,[H4M3_COLUMNS.index('h2a_state_finite'),
                                      H4M3_COLUMNS.index('h3_finite')]]==1)
                and np.isin(a[:,[H4M3_COLUMNS.index('lf_ground_contact'),
                                  H4M3_COLUMNS.index('rf_ground_contact')]], [0,1]).all(),
                'ARCHIVE_FINITE_CONTACT')
        result.append({'policy':policy,'seed':seed,'direction':direction,'end':end,
                       'rows':rows,'first':d['first'],'second':d['second']})
    h4m3_require(sum(len(v) for lane in result for v in lane['rows'].values())
                  ==H4M3_EXPECTED<=H4M3_CAP,'ARCHIVE_BUDGET')
    return result


def h4m3_read_archives(current_path,historical_path):
    current=h4m3_pinned_json(current_path,H4M3_ARCHIVES[0])
    historical=h4m3_pinned_json(historical_path,H4M3_ARCHIVES[1])
    return h4m3_archive_lanes(current,historical),current,historical


def h4m3_legacy_row(raw,expected,np):
    h4m3_require(tuple(sorted(raw))==H4M3_COLUMNS,'FRAME_SCHEMA')
    for k,v in raw.items():
        a=np.asarray(v)
        h4m3_require(a.ndim==0 and np.isfinite(a).all(),'NONFINITE_OR_NONSCALAR_'+k)
    row=[float(raw[k]) for k in H4M3_COLUMNS]
    for k,a,b in zip(H4M3_COLUMNS,row,expected):
        if a!=b:
            raise RuntimeError('M470_H4M3_ARCHIVE_MISMATCH '+json.dumps(
                {'field':k,'expected':b,'actual':a},allow_nan=False))
    return row


def h4m3_effective_ctrl(raw,ranges,limited,clamp_disabled,np):
    raw=np.asarray(raw); ranges=np.asarray(ranges); limited=np.asarray(limited)
    h4m3_require(raw.shape==(29,) and ranges.shape==(29,2) and limited.shape==(29,)
        and np.isfinite(raw).all() and np.isfinite(ranges).all()
        and np.all(ranges[:,0]<=ranges[:,1]) and np.isin(limited,[0,1]).all(),
        'CTRL_SCHEMA')
    return raw.copy() if clamp_disabled else np.where(limited,
        np.clip(raw,ranges[:,0],ranges[:,1]),raw)


def h4m3_com(mass,inertia,xipos,ximat,cvel,root_com,np):
    """Read-only bodywise algebra, all arrays at the same forward epoch."""
    mass,inertia,xipos,ximat,cvel,root_com=map(np.asarray,
        (mass,inertia,xipos,ximat,cvel,root_com))
    n=len(mass)
    h4m3_require(mass.shape==(n,) and inertia.shape==(n,3) and xipos.shape==(n,3)
        and ximat.shape==(n,3,3) and cvel.shape==(n,6) and root_com.shape==(n,3)
        and all(np.isfinite(a).all() for a in (mass,inertia,xipos,ximat,cvel,root_com))
        and (mass>=0).all() and (inertia>=0).all() and mass.sum()>0,'COM_SCHEMA')
    h4m3_require(np.allclose(ximat@ximat.transpose(0,2,1),np.eye(3),atol=1e-5,rtol=0)
        and np.allclose(np.linalg.det(ximat),1.,atol=1e-5,rtol=0),'COM_ROTATION')
    omega=cvel[:,:3]
    v=cvel[:,3:]+np.cross(omega,xipos-root_com)
    center=np.sum(mass[:,None]*xipos,axis=0)/mass.sum()
    velocity=np.sum(mass[:,None]*v,axis=0)/mass.sum()
    iw=(ximat*inertia[:,None,:])@ximat.transpose(0,2,1)
    momentum=np.sum((iw@omega[:,:,None])[:,:,0]+
        np.cross(xipos-center,mass[:,None]*(v-velocity)),axis=0)
    h4m3_require(all(np.isfinite(a).all() for a in (center,velocity,momentum)),
                  'COM_NONFINITE')
    return center,velocity,momentum


def h4m3_contact_force(efc,address,dim,friction,pyramidal,np):
    """Contact-frame wrench; invalid slot is unavailable, never a zero load."""
    if address<0:
        return None
    efc=np.asarray(efc); friction=np.asarray(friction)
    h4m3_require(dim in (1,3,4,6) and friction.shape==(5,) and efc.ndim==1
        and np.isfinite(efc).all() and np.isfinite(friction).all(),'CONTACT_SCHEMA')
    width=2*(dim-1) if pyramidal and dim>1 else dim
    h4m3_require(address+width<=len(efc),'CONTACT_ADDRESS')
    f=efc[address:address+width]
    wrench=np.zeros(6,dtype=np.float64)
    if pyramidal and dim>1:
        wrench[0]=np.sum(f)
        wrench[1:dim]=(f[::2]-f[1::2])*friction[:dim-1]
    else:
        wrench[:dim]=f
    return wrench


def h4m3_reward_check(metrics,reward,np):
    h4m3_require(bool(metrics) and all(k.startswith('reward/') for k in metrics)
        and all(np.asarray(v).ndim==0 and np.isfinite(v) for v in metrics.values())
        and np.isfinite(reward),'REWARD_SCHEMA')
    # Float32 summation order differs from host float64; no tolerance on legacy rows.
    expected=.02*sum(float(v) for v in metrics.values())
    h4m3_require(abs(expected-float(reward))<=2e-6*max(1.,abs(expected)),
                  'REWARD_SUM')


def h4m3_pre_capture(state):
    # Capture immutable leaves before Joystick mutates state.info and metrics dictionaries.
    return {'actor_input':state.obs['state'],'command':state.info['command'],
            'info_phase':state.info['phase'],'rng':state.info['rng'],
            'time_pre':state.data.time,'qpos_pre':state.data.qpos,
            'qvel_pre':state.data.qvel}


def h4m3_post_capture(pre,action,next_state,env):
    d=next_state.data
    lid,rid=env._left_foot_geom_id,env._right_foot_geom_id
    # Source sampled fields (last forward/solve); qpos/qvel/time post-integration.
    return {**pre,'action':action,'time_post':d.time,'qpos_post':d.qpos,
            'qvel_post':d.qvel,'target':next_state.info['motor_targets'],
            'raw_ctrl':d.ctrl,'actuator_force':d.actuator_force,
            'actuator_length':d.actuator_length,'actuator_velocity':d.actuator_velocity,
            'qfrc_actuator':d.qfrc_actuator,
            'lf_geom_pos':d.geom_xpos[lid],'rf_geom_pos':d.geom_xpos[rid],
            'lf_geom_mat':d.geom_xmat[lid],'rf_geom_mat':d.geom_xmat[rid],
            'feet_sensor_velocity':d.sensordata[env._foot_linvel_sensor_adr],
            'xipos':d.xipos,'ximat':d.ximat,'cvel':d.cvel,
            'subtree_com':d.subtree_com,'contact_geom':d.contact.geom,
            'contact_pos':d.contact.pos,'contact_frame':d.contact.frame,
            'contact_dist':d.contact.dist,'contact_includemargin':d.contact.includemargin,
            'contact_friction':d.contact.friction,
            'contact_dim':d.contact.dim,'contact_address':d.contact.efc_address,
            'efc_force':d.efc_force,
            'reward':{k:v for k,v in next_state.metrics.items() if k.startswith('reward/')}}



def h4m3_read_evidence(root,current_path,historical_path,base_policy):
    """Diagnostic evidence only; not the historical full training-report reader."""
    root=Path(root).resolve()
    h4m3_require(Path(current_path).resolve()==root/H4CC_EVAL_OUTPUT
        and Path(current_path).absolute()==root/H4CC_EVAL_OUTPUT
        and Path(historical_path).resolve()==Path(historical_path).absolute(),'ARCHIVE_PATH')
    lanes,current,historical=h4m3_read_archives(current_path,historical_path)
    pins={'current_archive':H4M3_ARCHIVES[0],'historical_archive':H4M3_ARCHIVES[1]}
    paths={'C_policy':(Path(base_policy),H4M3_POLICIES['C'])}
    for key,expected in current['input_sha256'].items():
        if key.startswith(('ref/','ccrc/')):
            arm,name=key.split('/',1)
            h4m3_require(Path(name).name==name,'ARTIFACT_NAME')
            paths[key]=(root/'m463_runs'/('m470_h4_m2_'+arm+'_v1')/name,expected)
        elif key.startswith('source/') and key!='source/m463_gate3_cp_eval.py':
            name=key.split('/',1)[1]
            h4m3_require(Path(name).name==name,'SOURCE_NAME')
            paths[key]=(root/name,expected)
    for key,(path,expected) in paths.items():
        h4m3_require(path.is_file() and not path.is_symlink()
            and path.resolve()==path.absolute() and sha256(path)==expected,'PIN_'+key)
        pins[key]=expected
    # Keep archived evaluator identity distinct; exact reviewed Git base required.
    archived=subprocess.check_output(['git','-C',str(root),'show',
        'd478c810152d2b7a1119514c94f09028d33f52da:m463_gate3_cp_eval.py'])
    git_blob=hashlib.sha1(('blob '+str(len(archived))+'\0').encode()+archived).hexdigest()
    h4m3_require(git_blob=='db89c4c8516e5875fd2d3c51210a3279fe4bc435'
        and hashlib.sha256(archived).hexdigest()==current['evaluator_sha256'],
        'ARCHIVED_EVALUATOR_PIN')
    train=Path(current['brax_source_verification']['path'])
    h4m3_require(train.is_file() and sha256(train)==TRAIN_SHA,'BRAX_TRAIN_PIN')
    h4m3_require(current['brax_source_verification']==historical['brax_source_verification'],
        'ARCHIVE_BRAX_EQUIVALENCE')
    pins['brax_train']=TRAIN_SHA
    brax_root=train.resolve().parents[3]
    for key,expected in current['input_sha256'].items():
        if not key.startswith('dependency/'):
            continue
        name=key.split('/',1)[1]
        path=(brax_root/'envs/wrappers/training.py' if name=='episode_wrapper' else
              brax_root.joinpath(*name.split('.')[1:]).with_suffix('.py'))
        h4m3_require(path.is_file() and not path.is_symlink() and sha256(path)==expected,
                      'DEPENDENCY_'+name)
        pins[key]=expected
    pins['archived_evaluator']=current['evaluator_sha256']
    pins['diagnostic_evaluator']=sha256(Path(__file__))
    h4m3_require({k.removeprefix('source/') for k in current['input_sha256']
        if k.startswith('source/')}==set(H4M3_SOURCE_NAMES),'ARCHIVED_SOURCE_SET')
    h4m3_require(all(pins.get('source/'+k)==v for k,v in h4m3_source_map(root).items()
        if k!='m463_gate3_cp_eval.py'),'SOURCE_MAP_MATCH')
    h4m3_require(pins['archived_evaluator']!=pins['diagnostic_evaluator'],'NEW_SOURCE_ID')
    return lanes,pins


def h4m3_manifest(env,np):
    """Model metadata only, no physics; exact installed-source checks separate."""
    from mujoco import mjtDisableBit, mjtCone, mjtGeom
    m=env.mj_model
    h4m3_require((m.nu,m.nq,m.nv,m.njnt)==(29,36,35,30)
        and abs(float(env.dt)-.02)<1e-9 and abs(float(m.opt.timestep)-.002)<1e-9
        and env.n_substeps==10,'MODEL_DIMENSIONS')
    ids=np.asarray(m.actuator_trnid)[:,0]
    h4m3_require(sorted(ids.tolist())==list(range(1,30))
        and np.all(np.asarray(m.actuator_trntype)==0)
        and np.all(np.asarray(m.actuator_dyntype)==0)
        and int(m.jnt_type[0])==0 and np.all(np.asarray(m.jnt_type)[1:]==3),
        'ACTUATOR_MAPPING')
    qa=np.asarray(m.jnt_qposadr)[ids]; va=np.asarray(m.jnt_dofadr)[ids]
    h4m3_require(np.array_equal(qa,np.arange(7,36)) and
        np.array_equal(va,np.arange(6,35)), 'TARGET_ORDER')
    h4m3_require(int(m.opt.integrator)==0,'INTEGRATOR_EULER')
    h4m3_require(not (int(m.opt.disableflags)&int(mjtDisableBit.mjDSBL_ACTUATION)),
        'ACTUATION_ENABLED')
    body=np.asarray(m.body_rootid); roots=np.unique(body[1:])
    h4m3_require(len(roots)==1 and roots[0]>0,'G1_ROOT')
    root=int(roots[0])
    h4m3_require(float(m.body_subtreemass[root])>0 and np.isclose(
        np.sum(np.asarray(m.body_mass)[body==root]),m.body_subtreemass[root]),
        'G1_MASS')
    names=('actuator_ctrlrange','actuator_ctrllimited','actuator_forcerange',
           'actuator_forcelimited','actuator_gainprm','actuator_biasprm',
           'actuator_gaintype','actuator_biastype','actuator_gear',
           'actuator_dyntype','actuator_trntype','actuator_trnid','jnt_range',
           'jnt_actfrcrange','jnt_actfrclimited',
           'jnt_limited','jnt_qposadr','jnt_dofadr','body_mass','body_inertia',
           'body_rootid','body_parentid','body_subtreemass','geom_bodyid')
    fields={k:np.asarray(getattr(m,k)).tolist() for k in names}
    floor=int(env._floor_geom_id)
    h4m3_require(int(m.geom_type[floor])==int(mjtGeom.mjGEOM_PLANE) and
        int(m.geom_bodyid[floor])==0 and np.allclose(m.geom_quat[floor],[1,0,0,0],
        atol=1e-7,rtol=0) and np.allclose(m.opt.gravity[:2],0,atol=1e-7,rtol=0)
        and float(m.opt.gravity[2])<0,'FLAT_FLOOR')
    h4m3_require(int(m.opt.cone) in (int(mjtCone.mjCONE_PYRAMIDAL),
        int(mjtCone.mjCONE_ELLIPTIC)),'CONTACT_CONE')
    h4m3_require(all(np.isfinite(np.asarray(v)).all() for v in fields.values()),'MODEL_FINITE')
    return {'fields':fields,'joint_names':[m.joint(i).name for i in range(m.njnt)],
        'actuator_names':[m.actuator(i).name for i in range(m.nu)],
        'qpos_addresses':qa.tolist(),'qvel_addresses':va.tolist(),'root_body_id':root,
        'foot_geom_ids':[int(env._left_foot_geom_id),int(env._right_foot_geom_id)],
        'floor_geom_id':int(env._floor_geom_id),'foot_site_ids':np.asarray(env._feet_site_id).tolist(),
        'feet_sensor_addresses':np.asarray(env._foot_linvel_sensor_adr).tolist(),
        'disableflags':int(m.opt.disableflags),'cone':int(m.opt.cone),
        'clamp_disabled':bool(int(m.opt.disableflags)&int(mjtDisableBit.mjDSBL_CLAMPCTRL)),
        'pyramidal':int(m.opt.cone)==int(mjtCone.mjCONE_PYRAMIDAL),
        'gravity':np.asarray(m.opt.gravity).tolist(),
        'reward_keys':sorted('reward/'+k for k in env._config.reward_config.scales.keys()),
        'integrator':int(m.opt.integrator),'default_pose':np.asarray(env._default_pose).tolist(),
        'dt_s':.02,'sim_dt_s':.002,'substeps':10,
        'stage':{'derived':'last_forward_solve','qpos_qvel':'post_integration'}}


def h4m3_validate_core(raw,row,manifest,np):
    shapes={'actor_input':(103,),'command':(3,),'info_phase':(2,),
        'rng':(2,),'time_pre':(),'time_post':(),'qpos_pre':(36,),
        'qpos_post':(36,),'qvel_pre':(35,),'qvel_post':(35,),
        'action':(29,),'target':(29,),'raw_ctrl':(29,),'actuator_force':(29,),
        'actuator_length':(29,),'actuator_velocity':(29,),'qfrc_actuator':(35,),
        'lf_geom_pos':(3,),'rf_geom_pos':(3,),'lf_geom_mat':(3,3),'rf_geom_mat':(3,3)}
    for k,shape in shapes.items():
        h4m3_require(k in raw,'MISSING_'+k)
        a=np.asarray(raw[k])
        h4m3_require(a.shape==shape and a.dtype.kind in 'fiu' and
            np.isfinite(a).all(),'SHAPE_FINITE_'+k)
    obs=np.asarray(raw['actor_input']); act=np.asarray(raw['action'])
    h4m3_require(np.array_equal(obs[9:12],raw['command']) and
        np.array_equal(raw['command'],[0.,0.,0.]) and np.max(np.abs(act))<=1.,
        'ACTOR_COMMAND_ACTION')
    h4m3_require(np.asarray(raw['rng']).dtype==np.uint32,'RNG_DTYPE')
    target=np.asarray(raw['target'])
    default=np.asarray(manifest['default_pose'],dtype=target.dtype)
    h4m3_require(np.allclose(target,default+.5*act,rtol=0,atol=1e-7)
        and np.array_equal(target,raw['raw_ctrl']),'RAW_TARGET')
    h4m3_require(abs(float(raw['time_post'])-float(raw['time_pre'])-.02)<2e-5,
                  'SAMPLE_TIME')
    h4m3_require(sorted(raw['reward'])==manifest['reward_keys'],'REWARD_KEYS')
    h4m3_reward_check(raw['reward'],row[H4M3_COLUMNS.index('reward_total')],np)



def h4m3_pack_capture(raw,row,manifest,np):
    """Pure host packer: optional COM/contact unavailable explicitly, no inference."""
    h4m3_validate_core(raw,row,manifest,np)
    fields=manifest['fields']
    optional=('xipos','ximat','cvel','subtree_com','contact_geom','contact_pos',
              'contact_frame','contact_dist','contact_includemargin','contact_friction',
              'contact_dim','contact_address','efc_force')
    packed={k:v for k,v in raw.items() if k not in optional}
    ctrl=h4m3_effective_ctrl(raw['raw_ctrl'],fields['actuator_ctrlrange'],
        fields['actuator_ctrllimited'],manifest['clamp_disabled'],np)
    packed['derived_effective_ctrl']=ctrl
    packed['effective_ctrl_semantics']='offline_source_clamp_not_direct_measurement'
    def margins(values,ranges,limited):
        x=np.asarray(values); r=np.asarray(ranges); mask=np.asarray(limited)
        h4m3_require(x.shape==(29,) and r.shape==(29,2) and mask.shape==(29,)
                      and np.isfinite(x).all() and np.isfinite(r).all()
                      and np.isin(mask,[0,1]).all(),'MARGIN_SCHEMA')
        margin=np.minimum(x-r[:,0],r[:,1]-x)
        return [float(v) if use else None for v,use in zip(margin,mask)]
    ids=np.asarray(fields['actuator_trnid'])[:,0]
    packed['joint_margin_post']=margins(np.asarray(raw['qpos_post'])[manifest['qpos_addresses']],
        np.asarray(fields['jnt_range'])[ids],np.asarray(fields['jnt_limited'])[ids])
    packed['raw_ctrl_margin']=margins(raw['raw_ctrl'],fields['actuator_ctrlrange'],
                                      fields['actuator_ctrllimited'])
    packed['force_margin_last_solve']=margins(raw['actuator_force'],fields['actuator_forcerange'],
                                             fields['actuator_forcelimited'])
    packed['joint_actuation_margin_last_solve']=margins(
        np.asarray(raw['qfrc_actuator'])[manifest['qvel_addresses']],
        np.asarray(fields['jnt_actfrcrange'])[ids],
        np.asarray(fields['jnt_actfrclimited'])[ids])
    packed['target_minus_post_joint']=np.asarray(raw['target'])-np.asarray(raw['qpos_post'])[
        manifest['qpos_addresses']]
    packed['actor_phase_features']=np.asarray(raw['actor_input'])[99:103]
    packed['previous_action']=np.asarray(raw['actor_input'])[70:99]
    packed['inference_key']=[0,0]
    packed['sample_stage']=manifest['stage']
    try:
        root=manifest['root_body_id']; roots=np.asarray(fields['body_rootid'])
        mask=roots==root; mass=np.asarray(fields['body_mass'])[mask]
        inertia=np.asarray(fields['body_inertia'])[mask]
        xipos=np.asarray(raw['xipos']); ximat=np.asarray(raw['ximat'])
        cvel=np.asarray(raw['cvel']); subtree=np.asarray(raw['subtree_com'])
        center,velocity,momentum=h4m3_com(mass,inertia,xipos[mask],ximat[mask],
            cvel[mask],subtree[roots[mask]],np)
        h4m3_require(np.allclose(center,subtree[root],atol=1e-5,rtol=0),'COM_ROOT_CHECK')
        packed['com']={'available':True,'position':center,'velocity':velocity,
            'angular_momentum':momentum,'body_xipos':xipos[mask],
            'body_ximat':ximat[mask],'body_cvel':cvel[mask],'subtree_com':subtree[root]}
    except (KeyError,IndexError,ValueError,RuntimeError) as exc:
        packed['com']={'available':False,'reason':str(exc)[:240]}
    try:
        geom=np.asarray(raw['contact_geom']); n=len(geom)
        pos=np.asarray(raw['contact_pos']); frame=np.asarray(raw['contact_frame']).reshape(n,3,3)
        dist=np.asarray(raw['contact_dist']); margin=np.asarray(raw['contact_includemargin'])
        friction=np.asarray(raw['contact_friction']); dim=np.asarray(raw['contact_dim'])
        adr=np.asarray(raw['contact_address']); efc=np.asarray(raw['efc_force'])
        h4m3_require(geom.shape==(n,2) and pos.shape==(n,3) and dist.shape==margin.shape
            ==dim.shape==adr.shape==(n,) and friction.shape==(n,5)
            and geom.dtype.kind in 'iu' and dim.dtype.kind in 'iu' and adr.dtype.kind in 'iu',
            'CONTACT_SLOT_SCHEMA')
        lf,rf=manifest['foot_geom_ids']; floor=manifest['floor_geom_id']
        relevant={frozenset((lf,floor)),frozenset((rf,floor)),frozenset((lf,rf))}
        slots=[]; loads={'left':0.,'right':0.}
        for i,(g1,g2) in enumerate(geom):
            pair=frozenset((int(g1),int(g2)))
            if pair not in relevant:
                continue
            h4m3_require(np.isfinite(dist[i]) and np.isfinite(margin[i]),'CONTACT_DIST')
            if adr[i]<0 or dist[i]>margin[i]:
                continue
            h4m3_require(np.isfinite(pos[i]).all() and np.isfinite(frame[i]).all()
                and np.allclose(frame[i]@frame[i].T,np.eye(3),atol=1e-5,rtol=0),
                'CONTACT_FRAME')
            wrench=h4m3_contact_force(efc,int(adr[i]),int(dim[i]),friction[i],
                                       manifest['pyramidal'],np)
            h4m3_require(wrench is not None and np.isfinite(wrench).all(),'CONTACT_WRENCH')
            normal=frame[i,0]
            if floor in pair:
                upward=normal if g1==floor else -normal
                h4m3_require(upward[2]>.99 and wrench[0]>=-1e-6,'CONTACT_NORMAL')
                loads['left' if lf in pair else 'right']+=float(wrench[0])
            slots.append({'slot':i,'geom':[int(g1),int(g2)],'pos':pos[i],
                'frame':frame[i],'dist':dist[i],'margin':margin[i],
                'dim':int(dim[i]),'address':int(adr[i]),'friction':friction[i],
                'wrench_contact_frame':wrench})
        bw=float(np.asarray(fields['body_mass']).sum())*abs(manifest['gravity'][2])
        h4m3_require(bw>0,'BODY_WEIGHT')
        packed['contact']={'available':True,'slots':slots,'normal_load_N':loads,
            'normal_load_bodyweight':{k:v/bw for k,v in loads.items()},
            'stage':'last_forward_solve'}
    except (KeyError,IndexError,ValueError,RuntimeError) as exc:
        packed['contact']={'available':False,'reason':str(exc)[:240]}
    # Required sensor velocities cannot silently become optional.
    sensor=np.asarray(raw['feet_sensor_velocity'])
    h4m3_require(sensor.shape==np.asarray(manifest['feet_sensor_addresses']).shape
        and np.isfinite(sensor).all(),'FOOT_SENSOR_SCHEMA')
    return packed


def h4m3_native_kick(st,direction,magnitude,tick,jax,jnp,np):
    """Same injection as legacy h3_group plus exact no-other-state audit."""
    base=jax.tree_util.tree_map(lambda x:x,st)
    heading=yaw_from_root_quat(jax.device_get(base.data.qpos[3:7]))
    requested=new_body_delta(direction,heading,magnitude)
    before=np.asarray(jax.device_get(base.data.qvel))
    qvel=base.data.qvel.at[:2].add(jnp.asarray(requested,dtype=base.data.qvel.dtype))
    after=np.asarray(jax.device_get(qvel))
    actual=after[:2].astype(np.float64)-before[:2].astype(np.float64)
    h4m3_require(np.isfinite(actual).all() and np.array_equal(before[2:],after[2:])
        and np.allclose(actual,requested,atol=2e-6,rtol=0),'ACTUAL_KICK')
    out=base.replace(data=base.data.replace(qvel=qvel))
    restored=out.replace(data=out.data.replace(qvel=base.data.qvel))
    h4m3_require(h4m3_tree_equal(base,restored,jax,np)
        and h4m3_tree_equal(base,st,jax,np),'KICK_OTHER_STATE')
    meta={'exposed':True,'step':tick,'heading_rad':heading,
          'requested_world_dqvel_xy_mps':list(requested),
          'actual_world_dqvel_xy_mps':actual.tolist(),
          'actual_magnitude_mps':float(np.linalg.norm(actual)),
          'full_state_rng_fork_by_construction':True}
    return out,meta,{'tick':tick,'qvel_before':before.tolist(),'qvel_after':after.tolist()}


def h4m3_native_lane(lane,session,env,step_fn,manifest,jax,jnp,np,pulse=None):
    h4m3_require(h4m3_authorization()==session['authorization'],'AUTH_CHANGED')
    audits=[]
    def reset(seed):
        state=env.reset(jax.random.PRNGKey(seed))
        h4m3_require(state.obs['state'].shape==(103,) and
            state.obs['privileged_state'].shape==(216,) and
            not bool(jax.device_get(state.done)),'RESET_SCHEMA')
        return state
    def step(st,capture):
        result=step_fn(st,placement_capture=capture)
        return result[0],jax.device_get(result[1]),(
            jax.device_get(result[2]) if capture else None)
    def kick(st,direction,magnitude,tick):
        out,meta,audit=h4m3_native_kick(st,direction,magnitude,tick,jax,jnp,np)
        audits.append(audit)
        session['lanes'][-1]['kick_audits']=list(audits)
        return out,meta
    lane_out=h4m3_replay_lane(lane,session,reset,step,
        lambda st:jax.tree_util.tree_map(lambda x:x,st),kick,
        lambda a,b:h4m3_tree_equal(a,b,jax,np),
        lambda raw,row:h4m3_pack_capture(raw,row,manifest,np),pulse=pulse)
    lane_out['kick_audits']=audits
    return lane_out


def h4m3_tree_equal(a,b,jax,np):
    aa,ta=jax.tree_util.tree_flatten(a); bb,tb=jax.tree_util.tree_flatten(b)
    return ta==tb and len(aa)==len(bb) and all(
        np.asarray(x).dtype==np.asarray(y).dtype and
        np.array_equal(np.asarray(x),np.asarray(y))
        for x,y in zip(jax.device_get(aa),jax.device_get(bb)))


def h4m3_replay_lane(lane,session,reset,step,clone,kick,equal,validate,pulse=None):
    """Host bounded mechanics; callbacks supplied by a future authorized dispatcher.
    No runtime imports, no own reset/physics call outside the frozen ledger.
    """
    h4m3_require(session['resets_attempted']<5 and
        len(session['lanes'])==session['resets_recorded']<5,'RESET_CAP')
    idx=len(session['lanes'])
    h4m3_require((lane['policy'],lane['seed'],lane['direction'],lane['end'])
        ==H4M3_LANES[idx],'LANE_ORDER')
    out={'policy':lane['policy'],'seed':lane['seed'],'direction':lane['direction'],
         'segments':{},'rich':[],'complete':False}
    session['lanes'].append(out)
    session['resets_attempted']+=1
    state=reset(lane['seed'])
    session['resets_recorded']+=1
    import numpy as np
    def advance(st,segment,start):
        suffix=segment in ('single','double')
        if suffix:
            h4m3_require(session.get('suffixes_attempted',0)<10,'SUFFIX_CAP')
            session['suffixes_attempted']=session.get('suffixes_attempted',0)+1
        out['segments'][segment]=[]
        for offset,expected in enumerate(lane['rows'][segment]):
            tick=start+offset; capture=tick>=295
            h4m3_require(session['attempted']<H4M3_CAP,'PHYSICAL_CAP')
            if capture:
                h4m3_require(session['rich_recorded']<H4M3_RICH_CAP,'RICH_CAP')
            session['attempted']+=1
            session['last_attempt']={'lane':idx,'segment':segment,'tick':tick}
            st,raw,rich=step(st,capture)
            row=h4m3_legacy_row(raw,expected,np)
            if capture:
                packed=validate(rich,row)
                h4m3_require(isinstance(packed,dict),'PACK_SCHEMA')
                # Reserve 4 MiB for legacy frames and report metadata.
                encoded=json.dumps(packed,allow_nan=False,separators=(',',':'),
                    default=lambda a:np.asarray(a).tolist())
                size=len(encoded.encode('utf8'))+128
                h4m3_require(size<=H4M3_RICH_RECORD_CAP and
                    session.get('rich_bytes',0)+size<=H4M3_REPORT_CAP-4*1024**2,
                    'RICH_REPORT_CAP')
                payload=json.loads(encoded)
                if segment=='double' and tick==335:
                    first=next(x['data'] for x in out['rich']
                               if x['segment']=='single' and x['tick']==335)
                    for k in ('actor_input','rng','action','target'):
                        h4m3_require(np.array_equal(first[k],payload[k]),'EVENT_EQUAL_'+k)
                session['rich_bytes']=session.get('rich_bytes',0)+size
                out['rich'].append({'segment':segment,'tick':tick,'data':payload})
                session['rich_recorded']+=1
            out['segments'][segment].append(row)
            session['recorded']+=1
            if pulse is not None and tick%100==0:
                pulse()
            if row[H4M3_COLUMNS.index('done')]:
                h4m3_require(offset+1==len(lane['rows'][segment]),'EARLY_TERMINAL')
                break
        if suffix:
            session['suffixes_recorded']=session.get('suffixes_recorded',0)+1
        if pulse is not None:
            pulse()
        return st
    state=advance(state,'nominal',1)
    first,meta=kick(clone(state),lane['direction'],.50,300)
    h4m3_require(meta==lane['first'],'FIRST_KICK_ARCHIVE')
    out['first']=meta
    state=advance(first,'first',300)
    single=clone(state); double=clone(state)
    h4m3_require(equal(single,double) and equal(single,state),'FORK_EQUAL')
    double,meta=kick(double,lane['direction'],.35,335)
    h4m3_require(meta==lane['second'] and equal(single,state),'SECOND_KICK_ARCHIVE')
    out['first']=lane['first']; out['second']=meta
    frozen_double=clone(double)
    advance(single,'single',335)
    h4m3_require(equal(double,frozen_double),'SIBLING_MUTATION')
    advance(double,'double',335)
    s=next(x['data'] for x in out['rich'] if x['segment']=='single' and x['tick']==335)
    d=next(x['data'] for x in out['rich'] if x['segment']=='double' and x['tick']==335)
    for key in ('actor_input','rng','action','target'):
        h4m3_require(np.array_equal(s[key],d[key]),'EVENT_EQUAL_'+key)
    out['complete']=True
    return out



def h4m3_policy_paths(root):
    root=Path(root)
    return {'C':root/H4M3_BASE_POLICY,
            **{a.upper():root/'m463_runs'/('m470_h4_m2_'+a+'_v1')/H4_EVAL_POLICY_FILE
               for a in ('ref','ccrc')}}


def h4m3_git_blob(path):
    raw=Path(path).read_bytes()
    return hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()


def h4m3_installed_mjx():
    # Distribution metadata only; safe before NVCC and runtime imports.
    from importlib.metadata import distribution
    core=distribution('mujoco'); mjx_dist=distribution('mujoco-mjx')
    h4m3_require(core.version==mjx_dist.version=='3.2.7','MUJOCO_VERSION')
    paths={}
    for name,expected in H4M3_MJX_BLOBS.items():
        p=Path(mjx_dist.locate_file('mujoco/mjx/_src/'+name+'.py')).resolve(strict=True)
        h4m3_require(p.is_file() and h4m3_git_blob(p)==expected,'MJX_SOURCE_'+name)
        paths['mujoco.mjx._src.'+name]={'path':str(p),'blob':expected,'sha256':sha256(p)}
    return paths


def h4m3_external_sources(root):
    compat=Path(root).parent/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1/nvcc_compat.py'
    h4m3_require(compat.is_file() and not compat.is_symlink(),'NVCC_COMPAT_FILE')
    return {'nvcc_compat':{'path':str(compat.resolve()),'sha256':sha256(compat)},
            'mjx':h4m3_installed_mjx()}


def h4m3_no_runtime_imports():
    h4m3_require(not any(n.split('.')[0] in
        ('jax','jaxlib','brax','mujoco','mujoco_playground') for n in sys.modules),
        'RUNTIME_PREIMPORTED')


def h4m3_runtime_dependencies(session):
    from importlib.util import find_spec
    root=Path(session['brax_train_path']).resolve().parents[3]
    for key,expected in session['input_sha256'].items():
        if not key.startswith('dependency/'):
            continue
        name=key.split('/',1)[1]
        module='brax.envs.wrappers.training' if name=='episode_wrapper' else name
        path=root.joinpath(*module.split('.')[1:]).with_suffix('.py')
        spec=find_spec(module)
        h4m3_require(spec is not None and spec.origin is not None
            and Path(spec.origin).resolve()==path and sha256(path)==expected,
            'RUNTIME_DEPENDENCY_'+name)


def h4m3_runtime_model_source(env,session):
    from importlib.util import find_spec
    from paper_config import REPO
    from mujoco.mjx._src.types import Data, Contact
    from dataclasses import fields
    for module,entry in session['mjx_sources'].items():
        spec=find_spec(module)
        h4m3_require(spec is not None and spec.origin is not None
            and str(Path(spec.origin).resolve())==entry['path']
            and sha256(entry['path'])==entry['sha256'],'RUNTIME_MJX_'+module)
    joystick=Path(inspect.getfile(type(env))).resolve(strict=True)
    from mujoco_playground._src import mjx_env
    wrapper=Path(inspect.getfile(mjx_env)).resolve(strict=True)
    for p,blob in ((joystick,'561a18948e3467ab924468eaf362150dadb9d4a6'),
                   (wrapper,'916c2f412a7a12568283f68eeb755e5d94ad9c83')):
        h4m3_require(p.is_relative_to(Path(REPO).resolve()) and h4m3_git_blob(p)==blob,
                      'PLAYGROUND_RUNTIME_PIN')
    data_fields={f.name for f in fields(Data)}
    contact_fields={f.name for f in fields(Contact)}
    h4m3_require(set('time qpos qvel ctrl actuator_force actuator_length actuator_velocity '
        'geom_xpos geom_xmat sensordata xipos ximat cvel subtree_com contact efc_force qfrc_actuator'.split())
        <=data_fields and set('geom pos frame dist includemargin friction dim efc_address'.split())
        <=contact_fields,'PINNED_TELEMETRY_FIELDS')
    h4m3_require(float(env._config.action_scale)==.5 and not env._config.push_config.enable,
                  'FROZEN_CONFIG')
    session['playground_runtime']={'joystick':str(joystick),'mjx_env':str(wrapper)}


def h4m3_write(path,report,initial=False):
    import shutil
    path=Path(path)
    h4m3_require(path.parent.is_dir() and not path.parent.is_symlink()
        and shutil.disk_usage(path.parent).free>=2*1024**3+H4M3_REPORT_CAP,'REPORT_STORAGE')
    data=(json.dumps(report,allow_nan=False,separators=(',',':'))+'\n').encode('utf8')
    h4m3_require(len(data)<=H4M3_REPORT_CAP,'FINAL_REPORT_CAP')
    if 'lanes' in report:
        overhead={**report,'lanes':[{**lane,'rich':[]} for lane in report['lanes']]}
        h4m3_require(len(json.dumps(overhead,allow_nan=False,separators=(',',':')).encode())
                     +1<=4*1024**2,'REPORT_METADATA_CAP')
    tmp=path.with_name(path.name+'.tmp')
    h4m3_require(not tmp.exists() and not tmp.is_symlink()
        and not path.is_symlink() and path.exists()!=initial,'REPORT_PATH')
    with (path if initial else tmp).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    if not initial:
        os.replace(tmp,path)


def h4m3_reserve(path,auth):
    h4m3_require(not path.parent.exists() and not path.parent.is_symlink()
        and path.parent.parent.resolve()==path.parent.parent.absolute(),'OUTPUT_CONSUMED')
    path.parent.mkdir(exist_ok=False)
    marker=path.parent/'dispatch.started.json'
    with marker.open('x',encoding='utf8') as f:
        json.dump({'authorization':auth},f,sort_keys=True,allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    return marker


def h4m3_progress(args,session):
    session['gpu']=h3_gpu_check(args.h3_monitor)
    h4m3_write(args.output,session)


def h4m3_final_ledger(session):
    h4m3_require(session['attempted']==session['recorded']==H4M3_EXPECTED
        and session['resets_attempted']==session['resets_recorded']==5
        and session['suffixes_attempted']==session['suffixes_recorded']==10
        and session['rich_recorded']==666 and len(session['lanes'])==5,
        'FINAL_LEDGER')
    total=0; rich=0
    for out,(policy,seed,direction,end) in zip(session['lanes'],H4M3_LANES):
        h4m3_require((out['policy'],out['seed'],out['direction'])==(policy,seed,direction)
            and out['complete'] is True and set(out['segments'])=={'nominal','first','single','double'}
            and tuple(len(out['segments'][k]) for k in ('nominal','first','single','double'))
                ==(299,35,51,end-334),'FINAL_LANE')
        expect=[('nominal',t) for t in range(295,300)]+[('first',t) for t in range(300,335)]
        expect += [('single',t) for t in range(335,386)]+[('double',t) for t in range(335,end+1)]
        h4m3_require([(x['segment'],x['tick']) for x in out['rich']]==expect,'FINAL_RICH_KEYS')
        total+=sum(map(len,out['segments'].values())); rich+=len(out['rich'])
        out['double_end']='terminal' if end<385 else 'censored'
    h4m3_require(total==H4M3_EXPECTED and rich==666,'FINAL_RECOUNT')


def h4m3_enter_runtime(args):
    # Only a separately reserved dispatch may enter existing run() bootstrap.
    auth=h4m3_authorization()
    root=Path(__file__).resolve().parent
    session=getattr(args,'h3_session',None)
    h4m3_no_runtime_imports()
    h4m3_require(not any(getattr(args,k,False) for k in ('h4cc_eval','h4cc_technical',
        'h4_m1_eval','h4_technical','h3_m1_eval','h2a','h2b'))
        and list(args.seeds)==[x[1] for x in H4M3_LANES] and list(args.profiles)==['stand']
        and args.checkpoint_c==root/H4M3_BASE_POLICY
        and args.output.resolve()==root/H4M3_OUTPUT and
        args.output.absolute()==root/H4M3_OUTPUT and session is not None
        and session['authorization']==auth and session['status']=='M470_H4M3_IN_PROGRESS'
        and session['attempted']==session['recorded']==session['resets_attempted']==0
        and session['lanes']==[] and args.output.is_file() and not args.output.is_symlink(),
        'SESSION_REQUIRED')
    marker=args.output.parent/'dispatch.started.json'
    h4m3_require(marker.is_file() and not marker.is_symlink() and
        json.loads(marker.read_bytes())=={'authorization':auth}
        and json.loads(args.output.read_bytes())==session,'DISPATCH_MARKER')
    # A second process cannot enter even with a copied initial session.
    with (args.output.parent/'runtime.entered.json').open('x',encoding='utf8') as f:
        json.dump({'authorization':auth},f,sort_keys=True,allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    lanes,pins=h4m3_read_evidence(root,root/H4CC_EVAL_OUTPUT,
        args.h4m3_historical_archive,root/H4M3_BASE_POLICY)
    h4m3_require(pins==session['input_sha256'] and h4m3_external_sources(root)==session['external_sources']
        and session['external_sources']['mjx']==session['mjx_sources']
        and h4m3_digest(h4m3_source_map(root))==auth['source_sha256'],'PRE_RUNTIME_PINS')
    args.h4m3_lanes=lanes


def h4m3_dispatch(args):
    import shutil
    auth=h4m3_authorization()
    root=Path(__file__).resolve().parent
    h4m3_require(not any(getattr(args,k,False) for k in ('h4cc_eval','h4cc_technical',
        'h4_m1_eval','h4_technical','h3_m1_eval','h2a','h2b'))
        and args.mode=='run' and args.output.absolute()==root/H4M3_OUTPUT
        and args.output.resolve()==root/H4M3_OUTPUT,'DISPATCH_PROTOCOL')
    lanes,pins=h4m3_read_evidence(root,root/H4CC_EVAL_OUTPUT,
        args.h4m3_historical_archive,root/H4M3_BASE_POLICY)
    h4m3_no_runtime_imports()
    external=h4m3_external_sources(root)
    mjx_sources=external['mjx']
    h4m3_require(os.environ.get('JAX_PLATFORMS')=='cuda','CUDA_ENV')
    h4m3_require(h4m3_digest(h4m3_source_map(root))==auth['source_sha256']
        and shutil.disk_usage(args.output.parent.parent).free>=2*1024**3+H4M3_REPORT_CAP,
        'PREFLIGHT_STORAGE_SOURCE')
    from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
    h4m3_no_runtime_imports()
    train_path=json.loads((root/H4CC_EVAL_OUTPUT).read_bytes())['brax_source_verification']['path']
    session={'status':'M470_H4M3_IN_PROGRESS','authorization':auth,'plan':json.loads(json.dumps(h4m3_plan())),
        'input_sha256':pins,'mjx_sources':mjx_sources,'external_sources':external,'brax_train_path':train_path,
        'frame_columns':list(H4M3_COLUMNS),'lanes':[],'attempted':0,'recorded':0,
        'resets_attempted':0,'resets_recorded':0,'suffixes_attempted':0,
        'suffixes_recorded':0,'rich_recorded':0,'rich_bytes':0,
        'training_performed':False,'efficacy_evaluated':False,
        'optimizer_updates':0,'checkpoint_saves':0,'promotion_authorized':False}
    marker=h4m3_reserve(args.output,auth)
    h4m3_write(args.output,session,initial=True)
    args.h3_session=session; args.h4m3_lanes=lanes
    args.checkpoint_c=root/H4M3_BASE_POLICY
    args.seeds=[x[1] for x in H4M3_LANES]; args.profiles=['stand']
    monitor=GPUMonitor(); args.h3_monitor=monitor
    try:
        g=sample_gpu()
        h4m3_require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288,'GPU_PREFLIGHT')
        # Do not mutate the initial report until h4m3_enter_runtime checks it.
        with monitor:
            run(args)
        session['gpu_initial']=g
        session['gpu']=h3_gpu_check(monitor)
        h4m3_require(h4m3_authorization()==auth and
            h4m3_read_evidence(root,root/H4CC_EVAL_OUTPUT,args.h4m3_historical_archive,
                root/H4M3_BASE_POLICY)[1]==pins and h4m3_external_sources(root)==external
            and json.loads(marker.read_bytes())=={'authorization':auth}
            and json.loads((args.output.parent/'runtime.entered.json').read_bytes())
                =={'authorization':auth},'POST_IMMUTABILITY')
        h4m3_final_ledger(session)
        session['status']='M470_H4M3_TECHNICAL_REPLAY_COMPLETED'
        h4m3_write(args.output,session)
        print(session['status'],session['recorded'],session['rich_recorded'],flush=True)
    except BaseException as exc:
        session['status']='M470_H4M3_STOP'
        session['error']=(type(exc).__name__+': '+str(exc))[:4096]
        session['gpu']=monitor.snapshot()
        try:
            h4m3_write(args.output,session)
        except BaseException as write_error:
            print('M470_H4M3_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise


def h4m3_static_archive_check(current_path,historical_path):
    lanes,_,_=h4m3_read_archives(current_path,historical_path)
    import numpy as np
    # Exercise exact row comparator against archive rows and a one-field mutation.
    for lane in lanes:
        for rows in lane['rows'].values():
            for row in rows:
                h4m3_legacy_row(dict(zip(H4M3_COLUMNS,row)),row,np)
    sample=lanes[0]['rows']['nominal'][0]
    raw=dict(zip(H4M3_COLUMNS,sample)); raw['vx']+=1.
    try:
        h4m3_legacy_row(raw,sample,np)
    except RuntimeError as exc:
        h4m3_require('ARCHIVE_MISMATCH' in str(exc),'NEGATIVE_MISMATCH')
    else:
        raise RuntimeError('M470_H4M3_NEGATIVE_MISMATCH')
    return {'lanes':len(lanes),'physical_rows':sum(len(v) for l in lanes
            for v in l['rows'].values()),'negative_mismatch':'PASS',
            'simulation_calls':0,'inference_calls':0}



# H4-M4-1: held support-conditioned action-residual diagnostic.
# Pure helpers only; a separate integration/static approval is mandatory.
H4M4_OUTPUT=Path('m463_runs/m470_h4_m4_feasibility_v1/report.json')
H4M4_REPORT_SHA='329ff0564d47e05e2360fdf95350928895cf515e8cec6ca9aa2182133319258b'
H4M4_PREVIOUS_BLOB='71711dec69336bc67f89530396a8199480cacbed'
H4M4_CAP=2337
H4M4_RICH_CAP=867
H4M4_RECORD_CAP=(60*1024**2)//H4M4_RICH_CAP
H4M4_SUBSET=(0,1,3)
H4M4_WEIGHTS=(.25,.5,.75,1.,1.,1.,1.,.75,.5,.25)
H4M4_VECTORS=((-0.277561,1.,.671888),(-.039760,-1.,.727860),
    (-.238374,1.,.743982),(-.657469,1.,.903112),(-.139246,1.,.008200))
H4M4_JOINTS=('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')


def h4m4_require(ok,label):
    if not ok:
        raise RuntimeError('M470_H4M4_'+label)


def h4m4_plan():
    return {'id':'H4-M4-FEASIBILITY-V1','lanes':H4M3_LANES,
        'archives':[*H4M3_ARCHIVES,H4M4_REPORT_SHA],
        'policy_sha256':H4M3_POLICIES,'fork_after_tick':335,
        'first_push':[300,.5],'second_push':[335,.35],'end_tick':385,
        'arms':['baseline','sham','treatment'],'subset':list(H4M4_SUBSET),
        'directions':H4M4_VECTORS,'pulse_first':336,'pulse_last':345,
        'weights':H4M4_WEIGHTS,'peak_action_residual':.2,'action_scale':.5,
        'max_physical':H4M4_CAP,'fixed_control_physical':2087,
        'max_rich_records':H4M4_RICH_CAP,'max_resets':5,'max_suffixes':15,
        'report_cap':64*1024**2,'record_cap':H4M4_RECORD_CAP,
        'log_cap':16*1024**2,'safety':dict(H3_EVAL_LIMITS),
        'geometry':{'delta':.1,'grid_count':11,'min_gap':.005,'min_z':.005,
            'direction_atol':5e-6,'jacobian_eps':1e-5,'jacobian_atol':1e-6},
        'endpoints':{'placement_tick':354,'momentum_tick':365,'D_gain':.02,
            'load_threshold':.05,'load_samples':3,'C_load_onset_max':350,
            'velocity_gain':.1,'positive_velocity_tolerance':.1,
            'positive_H_tolerance':.5,'min_dose_ticks':3,'min_dose_sum':.4},
        'training_performed':False,'efficacy_evaluated':False,
        'optimizer_updates':0,'checkpoint_saves':0,'execution_authorized':False}


def h4m4_parse_authorization(rules,source,spec):
    approved=[l.split() for l in rules.splitlines() if l.startswith('H4M4_STATIC_APPROVED:')]
    active=[l.split() for l in rules.splitlines() if l.startswith('H4M4_ACTIVE_GO:')]
    h4m4_require(len(active)==len(approved)==1,'EXECUTION_HOLD')
    h4m4_require(not any(l.startswith(('H4M3_ACTIVE_GO:','H4X_ACTIVE_GO:',
        'H4CC_ACTIVE_GO:','H4CC_EVAL_ACTIVE_GO:','H4CC_TECH_ACTIVE_GO:'))
        for l in rules.splitlines()),'CONCURRENT_AUTHORIZATION')
    a,b=approved[0],active[0]
    h4m4_require(len(a)==4 and a[1]==source and len(a[2])==64 and
        all(c in '0123456789abcdef' for c in a[2]),'STATIC_PIN')
    p=Path(a[3])
    h4m4_require(not p.is_absolute() and p.parent==Path('m463_runs')
        and p.name.startswith('m470_h4_m4_1_') and p.suffix=='.log','STATIC_PATH')
    expected=['H4M4_ACTIVE_GO:','H4-M4-FEASIBILITY-V1',source,a[2],spec,
        *H4M3_ARCHIVES,H4M4_REPORT_SHA,*[H4M3_POLICIES[k] for k in ('C','REF','CCRC')],
        str(H4M4_OUTPUT)]
    h4m4_require(b==expected,'AUTHORIZATION_PIN')
    return {'id':b[1],'source_sha256':source,'static_sha256':a[2],
        'static_path':a[3],'spec_sha256':spec,'archive_sha256':[*H4M3_ARCHIVES,H4M4_REPORT_SHA],
        'policy_sha256':dict(H4M3_POLICIES),'output':str(H4M4_OUTPUT)}


def h4m4_authorization():
    root=Path(__file__).resolve().parent
    rules=(root/'G1_RESEARCH_RULES.md').read_text('utf8')
    h4m4_require(any(l.startswith('H4M4_ACTIVE_GO:') for l in rules.splitlines())
        and any(l.startswith('H4M4_STATIC_APPROVED:') for l in rules.splitlines()),
        'EXECUTION_HOLD')
    auth=h4m4_parse_authorization(rules,h4m3_digest(h4m3_source_map(root)),
                                  h4m3_digest(h4m4_plan()))
    p=root/auth['static_path']
    h4m4_require(p.is_file() and not p.is_symlink() and p.resolve()==p.absolute()
        and p.stat().st_size<=16*1024**2 and sha256(p)==auth['static_sha256'],
        'STATIC_LOG')
    lines=p.read_text('utf8').splitlines()
    h4m4_require('H4_M4_1_FULL_STATIC_PASS '+auth['source_sha256'] in lines
        and 'H4_M4_1_EXTERNAL_SHA256 '+h4m3_digest(h4m4_external_sources(root)) in lines,
        'STATIC_ACCEPTANCE')
    return auth


def h4m4_leg_spec(side):
    h4m4_require(side in ('left','right'),'LEG_SIDE')
    s=1 if side=='left' else -1
    return {'offsets':[[0,s*.064452,-.1027],[0,s*.052,-.030465],
        [.025001,0,-.12412],[-.078273,s*.0021489,-.17734],
        [0,-s*.000094445,-.30001],[0,0,-.017558]],
        'quats':[[1,0,0,0],[.996179,0,-.0873386,0],[1,0,0,0],
        [.996179,0,.0873386,0],[1,0,0,0],[1,0,0,0]],
        'axes':[[0,1,0],[1,0,0],[0,0,1],[0,1,0],[0,1,0],[1,0,0]],
        'qpos0':[0.]*6,'foot_pos':[.04,0,-.029],'halfsize':[.09,.03,.008],
        'indices':list(range(0 if side=='left' else 6,6 if side=='left' else 12))}


def h4m4_quat_matrix(q,np):
    q=np.asarray(q,dtype=np.float64)
    h4m4_require(q.shape==(4,) and np.isfinite(q).all()
        and np.linalg.norm(q)>1e-8,'QUAT')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def h4m4_fk(qpos,leg,np,root_override=None):
    qpos=np.asarray(qpos,dtype=np.float64)
    h4m4_require(qpos.shape==(36,) and np.isfinite(qpos).all(),'FK_QPOS')
    root=qpos[:3].copy(); R=h4m4_quat_matrix(qpos[3:7],np)
    if root_override is not None:
        root,R=map(lambda a:np.asarray(a,dtype=np.float64).copy(),root_override)
        h4m4_require(root.shape==(3,) and R.shape==(3,3) and np.isfinite(root).all()
            and np.isfinite(R).all() and np.allclose(R@R.T,np.eye(3),atol=1e-6,rtol=0),
            'ALIGNED_ROOT')
    anchors=[]; axes=[]
    for p,q,a,t,ref in zip(leg['offsets'],leg['quats'],leg['axes'],
                           qpos[7+np.asarray(leg['indices'])],leg['qpos0']):
        a=np.asarray(a,dtype=np.float64)
        h4m4_require(abs(np.linalg.norm(a)-1)<1e-8,'FK_AXIS')
        root=root+R@p; R=R@h4m4_quat_matrix(q,np)
        anchors.append(root.copy()); axes.append(R@a)
        x,y,z=a; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        theta=t-ref
        R=R@(np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K))
    foot=root+R@leg['foot_pos']
    J=np.cross(np.asarray(axes),foot-np.asarray(anchors)).T
    h4m4_require(np.isfinite(J).all() and np.isfinite(foot).all(),'FK_FINITE')
    return foot,R,J


def h4m4_verify_aligned(packed,legs,np):
    # Last-solve inertial pose and actuator length, never stale geometry vs qpos_post.
    co=packed['com']
    R=np.asarray(co['body_ximat'][0])@h4m4_quat_matrix([1,0,-.000399148,0],np).T
    root=np.asarray(co['body_xipos'][0])-R@np.asarray([0,0,-.07605])
    qpos=np.asarray(packed['qpos_post'],dtype=np.float64).copy()
    qpos[7:]=packed['actuator_length']
    errors={}
    for side,short in (('left','lf'),('right','rf')):
        p,Q,_=h4m4_fk(qpos,legs[side],np,root_override=(root,R))
        pe=float(np.linalg.norm(p-np.asarray(packed[short+'_geom_pos'])))
        re=float(np.max(abs(Q-np.asarray(packed[short+'_geom_mat']))))
        h4m4_require(pe<=1e-6 and re<=1e-6,'ALIGNED_FK_'+side)
        errors[side]={'position_m':pe,'rotation_element':re}
    return errors


def h4m4_box_gap(p,R,half,q,S,other,np):
    axes=[*R.T,*S.T,*[np.cross(a,b) for a in R.T for b in S.T]]
    gaps=[]
    for a in axes:
        n=np.linalg.norm(a)
        if n>1e-10:
            a=a/n
            gaps.append(abs((q-p)@a)-np.asarray(half)@abs(R.T@a)
                        -np.asarray(other)@abs(S.T@a))
    return float(max(gaps))


def h4m4_geometry(qpos,legs,manifest,direction,lane_index,np):
    support=H4M3_LANES[lane_index][2]
    recovery='right' if support=='left' else 'left'
    rec=legs[recovery]; sup=legs[support]
    p,R,J=h4m4_fk(qpos,rec,np); q,S,_=h4m4_fk(qpos,sup,np)
    e=np.r_[np.asarray(direction,dtype=np.float64),0.]
    h4m4_require(e.shape==(3,) and np.isfinite(e).all()
        and abs(np.linalg.norm(e)-.35)<2e-6,'PUSH_VECTOR')
    e=e/np.linalg.norm(e)
    ids=np.asarray(rec['indices'])[list(H4M4_SUBSET)]
    JS=J[:,list(H4M4_SUBSET)]; g=JS.T@e; h=JS[2]
    h4m4_require(h@h>1e-8,'VERTICAL_DEGENERATE')
    w=g-h*(h@g)/(h@h)
    h4m4_require(np.max(abs(w))>1e-6,'DIRECTION_DEGENERATE')
    u=w/np.max(abs(w))
    h4m4_require(e@JS@u>0 and abs(h@u)<1e-8
        and np.max(abs(u-np.asarray(H4M4_VECTORS[lane_index])))<=5e-6,
        'DIRECTION_PIN')
    for j,i in enumerate(ids):
        delta=np.zeros(36); delta[7+i]=1e-5
        fd=(h4m4_fk(np.asarray(qpos)+delta,rec,np)[0]
            -h4m4_fk(np.asarray(qpos)-delta,rec,np)[0])/(2e-5)
        h4m4_require(np.max(abs(fd-JS[:,j]))<=1e-6,'JACOBIAN_FD')
    f=manifest['fields']; joints=np.asarray(f['actuator_trnid'])[ids,0]
    ranges=np.asarray(f['jnt_range'])[joints]; limited=np.asarray(f['jnt_limited'])[joints]
    lo=np.maximum(-1.,(np.asarray(f['actuator_ctrlrange'])[ids,0]
        -np.asarray(manifest['default_pose'])[ids])/.5)
    hi=np.minimum(1.,(np.asarray(f['actuator_ctrlrange'])[ids,1]
        -np.asarray(manifest['default_pose'])[ids])/.5)
    mask=np.asarray(f['actuator_ctrllimited'])[ids].astype(bool)
    lo=np.where(mask,lo,-1.); hi=np.where(mask,hi,1.)
    h4m4_require(not manifest['clamp_disabled'] and np.all(lo==-1.)
        and np.all(hi==1.),'ACTION_INTERSECTION')
    min_gap=float('inf'); min_z=float('inf')
    for frac in np.linspace(0.,1.,11):
        pose=np.asarray(qpos,dtype=np.float64).copy(); pose[7+ids]+=.1*frac*u
        h4m4_require(np.all(~limited.astype(bool) | ((pose[7+ids]>=ranges[:,0])
            & (pose[7+ids]<=ranges[:,1]))),'HYPOTHETICAL_JOINT_RANGE')
        pt,T,_=h4m4_fk(pose,rec,np)
        min_gap=min(min_gap,h4m4_box_gap(pt,T,rec['halfsize'],q,S,sup['halfsize'],np))
        min_z=min(min_z,float(pt[2]-np.asarray(rec['halfsize'])@abs(T[2])))
    h4m4_require(min_gap>=.005 and min_z>=.005,'GEOMETRY_GUARD')
    return {'support':support,'recovery':recovery,'indices':ids.tolist(),
        'u':u.tolist(),'e':e.tolist(),'jacobian':JS.tolist(),
        'min_gap':min_gap,'min_z':min_z,'predicted_shift_m':float((pt-p)@e),
        'predicted_z_m':float(pt[2]-p[2]),'lo':lo.tolist(),'hi':hi.tolist()}


def h4m4_pulse(tick,latched,arm,geometry,np):
    r=np.zeros(29,dtype=np.float32)
    h4m4_require(arm in ('baseline','sham','treatment'),'ARM')
    if arm=='treatment' and not latched and 336<=tick<=345:
        r[geometry['indices']]=.2*H4M4_WEIGHTS[tick-336]*np.asarray(geometry['u'])
    return r


def h4m4_apply_residual(actor,requested,selected,default,ctrlrange,limited,xp):
    """Array-only adapter, NumPy fixtures and future JAX trace share this function."""
    r=xp.asarray(requested,dtype=actor.dtype)
    mask=xp.asarray(selected,dtype=bool)
    d=xp.asarray(default,dtype=actor.dtype)
    bounds=xp.asarray(ctrlrange,dtype=actor.dtype)
    limited=xp.asarray(limited,dtype=bool)
    lo=xp.where(limited,xp.maximum(-1.,(bounds[:,0]-d)/.5),-1.)
    hi=xp.where(limited,xp.minimum(1.,(bounds[:,1]-d)/.5),1.)
    base=d+.5*actor
    base_ok=xp.all(~mask | ((actor>=lo)&(actor<=hi)&
        (~limited|((base>=bounds[:,0])&(base<=bounds[:,1])))))
    active=mask & (r!=0)
    denom=xp.where(active,xp.abs(r),1.)
    room=xp.where(active,xp.where(r>0,hi-actor,actor-lo)/denom,1.)
    lam=xp.where(base_ok,xp.maximum(0.,xp.minimum(1.,xp.min(room))),0.)
    delivered=lam*r
    candidate=xp.where(active & (delivered!=0),actor+delivered,actor)
    target=d+.5*candidate
    good=xp.all(~mask | ((candidate>=lo)&(candidate<=hi)&
        (~limited|((target>=bounds[:,0])&(target<=bounds[:,1])))))
    sent=xp.where(good & active & (delivered!=0),candidate,actor)
    scale=xp.where(good,lam,0.)
    reason=xp.where(~xp.any(active),0,xp.where(~base_ok,1,xp.where(~good,2,
        xp.where(scale==0,3,4))))
    return sent,{'actor_action':actor,'requested_residual':r,'lambda':scale,
        'delivered_residual':sent-actor,'sent_action':sent,'guard_code':reason}


def h4m4_external_sources(root):
    # Metadata and file bytes only; safe before NVCC or runtime imports.
    from paper_config import REPO
    root=Path(root).resolve()
    repo=Path(REPO).resolve(strict=True)
    h4m4_require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],
        text=True).strip()==PIN and subprocess.check_output(
        ['git','-C',str(repo),'status','--porcelain'],text=True).strip()=='',
        'PLAYGROUND_TREE')
    g1='mujoco_playground/_src/locomotion/g1/'
    expected={g1+'base.py':'fd59950822eb11c4d17d970cbca8fda9de7d2a6d',
        g1+'g1_constants.py':'59cce8046fd9995fc9d2d87bd0c0fa713ef13c69',
        g1+'joystick.py':'561a18948e3467ab924468eaf362150dadb9d4a6',
        g1+'xmls/scene_mjx_feetonly_flat_terrain.xml':'99b32be5b1a951d4ae148addcbe4ad0f4501fa25',
        g1+'xmls/g1_mjx_feetonly.xml':'4a7c00cdc92e245914d39b86b69b9329a26c35f5',
        'mujoco_playground/_src/mjx_env.py':'916c2f412a7a12568283f68eeb755e5d94ad9c83'}
    files={}
    for name,blob in expected.items():
        p=repo/name
        h4m4_require(p.is_file() and not p.is_symlink() and p.resolve()==p.absolute()
            and h4m3_git_blob(p)==blob,'KINEMATIC_SOURCE_'+name)
        files[name]={'path':str(p),'blob':blob,'sha256':sha256(p)}
    return {**h4m3_external_sources(root),'g1_kinematics':files}


def h4m4_model_geometry(env,manifest,np):
    """Verify compiled model metadata, no reset/forward or simulation."""
    from mujoco import mjtGeom
    from mujoco_playground._src.locomotion.g1 import base, g1_constants
    from paper_config import REPO
    for mod,blob in ((base,'fd59950822eb11c4d17d970cbca8fda9de7d2a6d'),
                     (g1_constants,'59cce8046fd9995fc9d2d87bd0c0fa713ef13c69')):
        path=Path(inspect.getfile(mod)).resolve(strict=True)
        h4m4_require(path.is_relative_to(Path(REPO).resolve()) and
            h4m3_git_blob(path)==blob,'G1_SOURCE_PIN')
    scene=Path(env.xml_path).resolve(strict=True)
    h4m4_require(scene==Path(g1_constants.FEET_ONLY_FLAT_TERRAIN_XML).resolve(strict=True)
        and scene.is_relative_to(Path(REPO).resolve()) and h4m3_git_blob(scene)==
            '99b32be5b1a951d4ae148addcbe4ad0f4501fa25','SCENE_PATH')
    xmls=[p for p in scene.parent.glob('*.xml') if h4m3_git_blob(p)==
          '4a7c00cdc92e245914d39b86b69b9329a26c35f5']
    h4m4_require(len(xmls)==1,'G1_XML_PIN')
    import xml.etree.ElementTree as ET
    includes=[x.attrib['file'] for x in ET.parse(scene).iter('include')]
    h4m4_require(xmls[0].resolve() in [(scene.parent/x).resolve() for x in includes],
        'G1_XML_INCLUDE')
    m=env.mj_model; root=int(m.body('pelvis').id)
    h4m4_require(root==manifest['root_body_id'] and int(m.jnt_bodyid[0])==root
        and int(m.body_parentid[root])==0,'PELVIS_ROOT')
    h4m4_require(np.allclose(m.body_ipos[root],[0,0,-.07605],atol=1e-8,rtol=0)
        and np.allclose(h4m4_quat_matrix(m.body_iquat[root],np),
            h4m4_quat_matrix([1,0,-.000399148,0],np),atol=1e-7,rtol=0),'PELVIS_INERTIA')
    legs={}
    for side in ('left','right'):
        leg=h4m4_leg_spec(side); parent=root
        for k,name in enumerate(H4M4_JOINTS):
            joint=side+'_'+name+'_joint'; body=side+'_'+name+'_link'
            j=int(m.joint(joint).id); b=int(m.body(body).id); a=leg['indices'][k]
            h4m4_require(int(m.body_parentid[b])==parent and int(m.jnt_bodyid[j])==b
                and int(m.actuator_trnid[a,0])==j and int(m.jnt_qposadr[j])==7+a
                and int(m.jnt_dofadr[j])==6+a and m.qpos0[7+a]==0.
                and np.array_equal(m.jnt_pos[j],np.zeros(3))
                and np.allclose(m.jnt_axis[j],leg['axes'][k],atol=1e-8,rtol=0)
                and np.allclose(m.body_pos[b],leg['offsets'][k],atol=1e-8,rtol=0)
                and np.allclose(h4m4_quat_matrix(m.body_quat[b],np),
                    h4m4_quat_matrix(leg['quats'][k],np),atol=1e-7,rtol=0)
                and np.array_equal(m.actuator_gear[a],[1,0,0,0,0,0]),'MODEL_JOINT_'+joint)
            leg['offsets'][k]=m.body_pos[b].tolist()
            leg['quats'][k]=m.body_quat[b].tolist()
            leg['axes'][k]=m.jnt_axis[j].tolist()
            parent=b
        gid=int(m.geom(side+'_foot').id)
        h4m4_require(gid==manifest['foot_geom_ids'][0 if side=='left' else 1]
            and int(m.geom_bodyid[gid])==parent and int(m.geom_type[gid])==int(mjtGeom.mjGEOM_BOX)
            and np.allclose(m.geom_pos[gid],leg['foot_pos'],atol=1e-8,rtol=0)
            and np.allclose(m.geom_size[gid],leg['halfsize'],atol=1e-8,rtol=0)
            and np.allclose(h4m4_quat_matrix(m.geom_quat[gid],np),np.eye(3),atol=1e-8,rtol=0),
            'MODEL_FOOT_'+side)
        leg['foot_pos']=m.geom_pos[gid].tolist(); leg['halfsize']=m.geom_size[gid].tolist()
        legs[side]=leg
    return {'legs':legs,'scene_path':str(scene),'scene_sha256':sha256(scene),
        'xml_path':str(xmls[0].resolve()),'xml_sha256':sha256(xmls[0]),
        'xml_blob':'4a7c00cdc92e245914d39b86b69b9329a26c35f5'}



def h4m4_archive_projection(report,lanes):
    h4m4_require(report['status']=='M470_H4M3_TECHNICAL_REPLAY_COMPLETED'
        and report['frame_columns']==list(H4M3_COLUMNS)
        and report['attempted']==report['recorded']==2136
        and report['resets_attempted']==report['resets_recorded']==5
        and report['suffixes_attempted']==report['suffixes_recorded']==10
        and report['rich_recorded']==666 and len(report['lanes'])==len(lanes)==5,
        'APPROVED_REPORT_LEDGER')
    h4m4_require(report['plan']==json.loads(json.dumps(h4m3_plan()))
        and report['authorization']['policy_sha256']==H4M3_POLICIES
        and report['training_performed'] is False and report['efficacy_evaluated'] is False
        and report['optimizer_updates']==report['checkpoint_saves']==0,
        'APPROVED_REPORT_PROTOCOL')
    result=[]
    for i,(old,archive,key) in enumerate(zip(report['lanes'],lanes,H4M3_LANES)):
        policy,seed,direction,end=key
        h4m4_require((old['policy'],old['seed'],old['direction'])==key[:3]
            and (archive['policy'],archive['seed'],archive['direction'],archive['end'])==key
            and old['segments']==archive['rows'] and old['complete'] is True
            and old['first']==archive['first'] and old['second']==archive['second']
            and len(old['kick_audits'])==2,'APPROVED_ARCHIVE_EQUAL')
        prefix=old['segments']['nominal']+old['segments']['first']+old['segments']['double'][:1]
        suffix=old['segments']['double'][1:]
        rich={x['tick']:x['data'] for x in old['rich'] if x['segment']!='single'}
        h4m4_require(len(prefix)==335 and len(suffix)==end-335
            and sorted(rich)==list(range(295,end+1)) and len(old['rich'])==end-243,
            'APPROVED_RICH_KEYS')
        h4m4_require(all(x['com']['available'] is True and x['contact']['available'] is True
            for x in rich.values()),'APPROVED_TELEMETRY')
        result.append({'index':i,'policy':policy,'seed':seed,'direction':direction,
            'end':end,'prefix':prefix,'suffix':suffix,'rich':rich,
            'first':old['first'],'second':old['second'],'kick_audits':old['kick_audits']})
    return result


def h4m4_read_evidence(root,historical_path):
    root=Path(root)
    report=h4m3_pinned_json(root/H4M3_OUTPUT,H4M4_REPORT_SHA)
    lanes,pins=h4m3_read_evidence(root,root/H4CC_EVAL_OUTPUT,historical_path,
                                root/H4M3_BASE_POLICY)
    old=subprocess.check_output(['git','-C',str(root),'show',
        'dc79dc906a38edc208c0770ea8ea26f7722d8aba:m463_gate3_cp_eval.py'])
    old_blob=hashlib.sha1(('blob '+str(len(old))+'\0').encode()+old).hexdigest()
    old_sha=hashlib.sha256(old).hexdigest()
    h4m4_require(old_blob==H4M4_PREVIOUS_BLOB and old_sha==report['input_sha256']['diagnostic_evaluator']
        and pins['diagnostic_evaluator']!=old_sha and
        {**pins,'diagnostic_evaluator':old_sha}==report['input_sha256'],'OLD_SOURCE_PINS')
    h4m4_require(h4m3_digest({**h4m3_source_map(root),'m463_gate3_cp_eval.py':old_sha})
        ==report['authorization']['source_sha256'],'OLD_COMMON_SOURCE')
    external=h4m3_external_sources(root)
    h4m4_require(external==report['external_sources'],'OLD_EXTERNAL_PINS')
    return h4m4_archive_projection(report,lanes),{
        **pins,'approved_h4m3_report':H4M4_REPORT_SHA,'approved_h4m3_evaluator':old_sha},report


def h4m4_row(raw,np):
    h4m4_require(tuple(sorted(raw))==H4M3_COLUMNS,'FRAME_SCHEMA')
    h4m4_require(all(np.asarray(v).ndim==0 and np.isfinite(v) for v in raw.values()),
        'FRAME_FINITE')
    row=[float(raw[k]) for k in H4M3_COLUMNS]
    for k in ('h2a_state_finite','h3_finite'):
        h4m4_require(raw[k]==1,'NONFINITE_'+k)
    for k in ('done','lf_ground_contact','rf_ground_contact',*H3_EVAL_FLAGS):
        h4m4_require(raw[k] in (0,1),'BOOLEAN_'+k)
    return row


def h4m4_audit_action(packed,meta,request,geometry,manifest,np):
    h4m4_require(isinstance(meta,dict) and set(meta)=={'actor_action','requested_residual',
        'lambda','delivered_residual','sent_action','guard_code'},'RESIDUAL_SCHEMA')
    arr={k:np.asarray(meta[k]) for k in ('actor_action',
        'requested_residual','delivered_residual','sent_action')}
    h4m4_require(all(v.shape==(29,) and v.dtype==np.float32 and np.isfinite(v).all()
        for v in arr.values()),'RESIDUAL_FINITE_DTYPE')
    h4m4_require(np.asarray(meta['lambda']).ndim==0 and np.asarray(meta['guard_code']).ndim==0
        and np.asarray(meta['guard_code']).dtype.kind in 'iu','RESIDUAL_SCALAR_DTYPE')
    actor=arr['actor_action']; sent=arr['sent_action']; r=arr['requested_residual']
    mask=np.zeros(29,bool); mask[geometry['indices']]=True
    h4m4_require(np.array_equal(r,request) and np.all(r[~mask]==0)
        and np.max(abs(r))<=np.float32(.2) and np.max(abs(actor))<=1.
        and np.max(abs(sent))<=1. and np.array_equal(sent[~mask].view(np.uint32),
            actor[~mask].view(np.uint32)) and np.array_equal(sent,np.asarray(packed['action']))
        and np.array_equal(arr['delivered_residual'],sent-actor),'RESIDUAL_IDENTITY')
    lam=float(meta['lambda']); code=int(meta['guard_code'])
    h4m4_require(np.isfinite(lam) and 0<=lam<=1. and code in (0,1,2,3,4)
        and np.max(abs((sent-actor)-lam*r))<=1e-7,'RESIDUAL_SCALE')
    f=manifest['fields']
    ref,expected=h4m4_apply_residual(actor,r,mask,manifest['default_pose'],
        f['actuator_ctrlrange'],f['actuator_ctrllimited'],np)
    h4m4_require(abs(float(expected['lambda'])-lam)<=1e-6
        and int(expected['guard_code'])==code and np.allclose(ref,sent,atol=1e-7,rtol=0),
        'RESIDUAL_RECONCILIATION')
    if not np.any(r) or lam==0:
        h4m4_require(np.array_equal(sent.view(np.uint32),actor.view(np.uint32)),
            'ZERO_DOSE_BITS')
    else:
        ids=np.asarray(geometry['indices'])
        raw=np.asarray(packed['raw_ctrl'])[ids]
        bounds=np.asarray(f['actuator_ctrlrange'])[ids]
        h4m4_require(np.all(raw>=bounds[:,0]) and np.all(raw<=bounds[:,1]),
            'TARGET_HEADROOM')
    return {'actor_action':actor.tolist(),'requested_residual':r.tolist(),
        'lambda':lam,'delivered_residual':(sent-actor).tolist(),'sent_action':sent.tolist(),
        'guard_code':code}


def h4m4_force_discrepancy(packed,manifest,np):
    a=np.asarray(packed['actuator_force'])
    q=np.asarray(packed['qfrc_actuator'])[manifest['qvel_addresses']]
    h4m4_require(a.shape==q.shape==(29,) and np.isfinite(a).all() and
        np.isfinite(q).all(),'FORCE_MAP')
    return float(np.max(abs(a-q)))


def h4m4_replay_lane(lane,session,reset,step,clone,kick,equal,geometry_fn,manifest,np,
                      pulse=None):
    """Host mechanics only. Actual callback adapter must be separately authorized."""
    idx=len(session['lanes'])
    h4m4_require(idx<5 and session['resets_attempted']==session['resets_recorded']==idx
        and lane['index']==idx and (lane['policy'],lane['seed'],lane['direction'],lane['end'])
        ==H4M3_LANES[idx],'LANE_ORDER')
    out={'policy':lane['policy'],'seed':lane['seed'],'direction':lane['direction'],
        'prefix':[],'arms':{},'rich':[],'kicks':[],'complete':False}
    session['lanes'].append(out); session['resets_attempted']+=1
    st=reset(lane['seed']); session['resets_recorded']+=1
    geometry=None
    def advance(state,arm,tick,expected,request=None,latched=False):
        capture=tick>=295
        h4m4_require(session['attempted']<H4M4_CAP and
            (not capture or session['rich_recorded']<H4M4_RICH_CAP),'STEP_CAP')
        session['attempted']+=1
        session['last_attempt']={'lane':idx,'arm':arm,'tick':tick}
        state,raw,packed,meta=step(state,capture,request,geometry)
        row=h4m4_row(raw,np)
        if expected is not None:
            h4m4_require(row==expected,'EXACT_ROW_'+arm+'_'+str(tick))
        if capture:
            h4m4_require(packed['com']['available'] is True and
                packed['contact']['available'] is True,'REQUIRED_COM_CONTACT')
            if expected is not None:
                h4m4_require(packed==lane['rich'][tick],'EXACT_RICH_'+arm+'_'+str(tick))
            if request is None:
                h4m4_require(meta is None,'BASELINE_ADAPTER')
                action=packed['action']; meta={'actor_action':action,'sent_action':action,
                    'requested_residual':[0.]*29,'delivered_residual':[0.]*29,
                    'lambda':0.,'guard_code':0}
            else:
                meta=h4m4_audit_action(packed,meta,request,geometry,manifest,np)
            rec={'arm':arm,'tick':tick,'data':packed,'residual':meta,
                 'latched_before':bool(latched),
                 'force_discrepancy_Nm':h4m4_force_discrepancy(packed,manifest,np)}
            encoded=json.dumps(rec,allow_nan=False,separators=(',',':'))
            size=len(encoded.encode())+128
            h4m4_require(size<=H4M4_RECORD_CAP and session['rich_bytes']+size<=60*1024**2,
                'RICH_REPORT_CAP')
            session['rich_bytes']+=size; session['rich_recorded']+=1
            out['rich'].append(rec)
        session['recorded']+=1
        if arm=='prefix':
            out['prefix'].append(row)
        else:
            out['arms'][arm]['rows'].append(row)
        if pulse is not None and tick%100==0:
            pulse()
        return state,row
    for tick in range(1,336):
        if tick in (300,335):
            st,meta,audit=kick(clone(st),lane['direction'],.5 if tick==300 else .35,tick)
            h4m4_require(meta==lane['first' if tick==300 else 'second'] and
                audit==lane['kick_audits'][0 if tick==300 else 1],'KICK_ARCHIVE')
            out['kicks'].append(audit)
        st,row=advance(st,'prefix',tick,lane['prefix'][tick-1])
        h4m4_require(not row[H4M3_COLUMNS.index('done')],'PREFIX_TERMINAL')
    fork=clone(st); witness=clone(st)
    siblings={arm:clone(st) for arm in ('baseline','sham','treatment')}
    h4m4_require(equal(fork,witness) and all(equal(fork,x) for x in siblings.values()),
        'FORK_EQUAL')
    geometry=geometry_fn(fork,out['rich'][-1]['data'],lane)
    out['geometry']=geometry
    rec=geometry['recovery']; sup=geometry['support']
    contact_key=('lf' if rec=='left' else 'rf')+'_ground_contact'
    support_key=('lf' if sup=='left' else 'rf')+'_ground_contact'
    ri=H4M3_COLUMNS.index(contact_key); si=H4M3_COLUMNS.index(support_key)
    loads=out['rich'][-1]['data']['contact']['normal_load_bodyweight']
    h4m4_require(out['prefix'][-1][ri]==0 and out['prefix'][-1][si]==1
        and loads[rec]==0 and loads[sup]>0,'FORK_SUPPORT')
    if pulse is not None: pulse()
    limits={k:H4M3_COLUMNS.index(k) for k in H3_EVAL_LIMITS}
    for arm in ('baseline','sham','treatment'):
        h4m4_require(session['suffixes_attempted']<15 and equal(fork,witness)
            and equal(siblings[arm],witness),'SIBLING_PRE')
        session['suffixes_attempted']+=1
        branch={'rows':[],'end_reason':None,'dose_ticks':0,'dose_sum':0.,
            'first_contact_tick':None,'safety_violations':[]}
        out['arms'][arm]=branch; state=siblings.pop(arm); latched=False
        stop=lane['end'] if arm!='treatment' else 385
        for tick in range(336,stop+1):
            request=None if arm=='baseline' else h4m4_pulse(tick,latched,arm,geometry,np)
            state,row=advance(state,arm,tick,lane['suffix'][tick-336] if arm!='treatment'
                else None,request,latched)
            rich=out['rich'][-1]; dose=max(map(abs,rich['residual']['delivered_residual']))
            if dose>0:
                branch['dose_ticks']+=1; branch['dose_sum']+=dose
            if row[ri] and not latched:
                latched=True; branch['first_contact_tick']=tick
            if arm=='treatment':
                for k,j in limits.items():
                    if row[j]>max(x[j] for x in out['arms']['baseline']['rows'])+H3_EVAL_LIMITS[k]:
                        branch['safety_violations'].append({'tick':tick,'gate':k})
                if rich['force_discrepancy_Nm']>.1:
                    branch['safety_violations'].append({'tick':tick,'gate':'GENERALIZED_FORCE_CLIP'})
            if row[H4M3_COLUMNS.index('done')]:
                h4m4_require(arm=='treatment' or tick==stop,'CONTROL_EARLY_TERMINAL')
                branch['end_reason']='source_terminal'
                break
            if branch['safety_violations']:
                branch['end_reason']='safety_stop'
                break
        if branch['end_reason'] is None:
            branch['end_reason']='censored'
        branch['end_tick']=335+len(branch['rows'])
        if arm!='treatment':
            h4m4_require(branch['end_tick']==lane['end'] and
                branch['end_reason']==('source_terminal' if lane['end']<385 else 'censored')
                and branch['dose_ticks']==0,'CONTROL_END')
        h4m4_require(equal(fork,witness) and all(equal(x,witness) for x in siblings.values()),
            'SIBLING_MUTATION')
        session['suffixes_recorded']+=1
        if pulse is not None: pulse()
    out['complete']=True
    return out


def h4m4_final_ledger(session):
    h4m4_require(session['resets_attempted']==session['resets_recorded']==5
        and session['suffixes_attempted']==session['suffixes_recorded']==15
        and len(session['lanes'])==5,'FINAL_COUNTS')
    physical=0; rich=0; control=0; treatments=0
    for out,key in zip(session['lanes'],H4M3_LANES):
        h4m4_require((out['policy'],out['seed'],out['direction'])==key[:3]
            and out['complete'] is True and len(out['prefix'])==335
            and set(out['arms'])=={'baseline','sham','treatment'},'FINAL_LANE')
        expected=[('prefix',t) for t in range(295,336)]
        physical+=335; control+=335
        for arm in ('baseline','sham','treatment'):
            b=out['arms'][arm]; n=len(b['rows'])
            h4m4_require(1<=n<=50 and b['end_tick']==335+n,'FINAL_ARM')
            if arm!='treatment':
                h4m4_require(n==key[3]-335 and b['dose_ticks']==0,'FINAL_CONTROL')
                control+=n
            else:
                treatments+=n
            physical+=n
            expected.extend((arm,t) for t in range(336,336+n))
        h4m4_require([(r['arm'],r['tick']) for r in out['rich']]==expected,'FINAL_RICH_KEYS')
        rich+=len(expected)
    h4m4_require(control==2087 and physical==2087+treatments
        and session['attempted']==session['recorded']==physical<=H4M4_CAP
        and session['rich_recorded']==rich==physical-1470<=H4M4_RICH_CAP,
        'FINAL_LEDGER')
    return {'control_physical':control,'treatment_physical':treatments,
            'physical':physical,'rich':rich}



def h4m4_load_summary(samples,side,threshold):
    hits=[r['data']['contact']['normal_load_bodyweight'][side]>=threshold for r in samples]
    start=None; streak=0; longest=0
    for i,hit in enumerate(hits):
        streak=streak+1 if hit else 0
        longest=max(longest,streak)
        if streak==3 and start is None:
            start=samples[i-2]['tick']
    return {'onset':start,'completed':None if start is None else start+2,
            'longest':longest,'count':sum(hits)}


def h4m4_endpoint(sample,origin,geometry,np):
    x=sample['data']; b=origin['data']; side=geometry['recovery']
    foot='lf_geom_pos' if side=='left' else 'rf_geom_pos'
    e=np.asarray(geometry['e']); v=np.asarray(x['com']['velocity'])
    p=np.asarray(x[foot])-np.asarray(x['com']['position'])
    p0=np.asarray(b[foot])-np.asarray(b['com']['position'])
    return {'D_m':float((p-p0)@e),'foot_relative_push_m':float(p@e),
            'v_push_mps':float(v@e),'v_transverse_mps':float(v@np.array([-e[1],e[0],0.])),
            'angular_momentum_norm':float(np.linalg.norm(x['com']['angular_momentum'])),
            'load_recovery_bw':x['contact']['normal_load_bodyweight'][side],
            'load_support_bw':x['contact']['normal_load_bodyweight'][geometry['support']]}


def h4m4_classify(out,np):
    B=out['arms']['baseline']; T=out['arms']['treatment']; geom=out['geometry']
    samples={arm:{r['tick']:r for r in out['rich'] if r['arm']==arm}
             for arm in ('prefix','baseline','sham','treatment')}
    origin=samples['prefix'][334]
    ends={}
    for arm in ('baseline','sham','treatment'):
        ends[arm]={str(t):h4m4_endpoint(samples[arm][t],origin,geom,np)
            if t in samples[arm] else None for t in (354,365,385)}
        ends[arm]['last']=h4m4_endpoint(samples[arm][out['arms'][arm]['end_tick']],
                                       origin,geom,np)
        ends[arm]['load']={str(p):h4m4_load_summary(list(samples[arm].values()),
            geom['recovery'],p/100.) for p in (1,5,10)}
    n=min(len(B['rows']),len(T['rows']))
    safety={}
    for k,tol in H3_EVAL_LIMITS.items():
        j=H4M3_COLUMNS.index(k)
        b=max(r[j] for r in B['rows'][:n]); t=max(r[j] for r in T['rows'][:n])
        bf=max(r[j] for r in B['rows']); tf=max(r[j] for r in T['rows'])
        safety[k]={'baseline_common':b,'treatment_common':t,
            'baseline_full':bf,'treatment_full':tf,'tolerance':tol,
            'pass':t<=b+tol and tf<=bf+tol}
    def flags(branch):
        return {k for k in H3_EVAL_FLAGS if any(r[H4M3_COLUMNS.index(k)]
                                                for r in branch['rows'])}
    new_flags=sorted(flags(T)-flags(B))
    adverse=bool(T['safety_violations'] or not all(x['pass'] for x in safety.values())
                 or new_flags or T['end_tick']<B['end_tick'])
    low_dose=T['dose_ticks']<3 or T['dose_sum']<.4
    b,t=ends['baseline'],ends['treatment']; positive=B['end_reason']=='censored'
    criteria={}
    if positive:
        criteria={'censored385':T['end_reason']=='censored' and T['end_tick']==385,
            'velocity':t['385'] is not None and abs(t['385']['v_push_mps'])
                <=abs(b['385']['v_push_mps'])+.1,
            'angular_momentum':t['385'] is not None and t['385']['angular_momentum_norm']
                <=b['385']['angular_momentum_norm']+.5,'safety':not adverse}
        decision='POSITIVE_PRESERVED' if all(criteria.values()) else 'POSITIVE_ADVERSE'
    else:
        missing=t['354'] is None or t['365'] is None
        load=t['load']['5']
        criteria={'placement':t['354'] is not None and
            t['354']['D_m']-b['354']['D_m']>=.02,
            'load':load['completed'] is not None and load['completed']<=354
                and (out['policy']!='C' or load['onset']<=350),
            'momentum':t['365'] is not None and
                t['365']['v_push_mps']<=b['365']['v_push_mps']-.1 and
                abs(t['365']['v_push_mps'])<=abs(b['365']['v_push_mps'])-.1,
            'safety':not adverse}
        decision=('ADVERSE' if adverse else 'LOCAL_CHAIN_SUPPORTED' if all(criteria.values())
            else 'INCONCLUSIVE_MISSING_ENDPOINT' if missing else 'INCONCLUSIVE_LOW_DOSE'
            if low_dose else 'CHAIN_NOT_SUPPORTED')
    return {'decision':decision,'criteria':criteria,'low_dose':low_dose,
        'endpoints':ends,'safety':safety,'new_critical_flags':new_flags,
        'matched_suffix_samples':n,'treatment_end':T['end_tick'],
        'baseline_end':B['end_tick']}


def h4m4_summary(session,np):
    results=[h4m4_classify(x,np) for x in session['lanes']]
    positives=all(results[i]['decision']=='POSITIVE_PRESERVED' for i in (0,4))
    successes=sum(results[i]['decision']=='LOCAL_CHAIN_SUPPORTED' for i in (1,2))
    decision=('POSITIVE_CONTROL_ADVERSE' if not positives else 'LOCAL_FEASIBILITY_SUPPORTED'
        if successes==2 else 'HETEROGENEOUS_PARTIAL' if successes==1 else
        'INCONCLUSIVE' if any(results[i]['decision'].startswith('INCONCLUSIVE')
            for i in (1,2)) else 'CHAIN_NOT_SUPPORTED_IN_TESTED_ENVELOPE')
    return {'decision':decision,'lanes':results,'primary_chains':successes,
        'positive_preservation':positives,'efficacy_evaluated':False}


def h4m4_native_lane(lane,session,env,step_fn,jax,jnp,np,pulse=None):
    h4m4_require(h4m4_authorization()==session['authorization'],'AUTH_CHANGED')
    manifest=session['manifest']; legs=session['kinematics']['legs']
    def reset(seed):
        st=env.reset(jax.random.PRNGKey(seed))
        h4m4_require(st.obs['state'].shape==(103,) and
            st.obs['privileged_state'].shape==(216,) and not bool(jax.device_get(st.done)),
            'RESET_SCHEMA')
        return st
    def clone(st):
        out=jax.tree_util.tree_map(lambda x:x,st)
        def containers(x):
            if isinstance(x,dict):
                return {id(x)}.union(*(containers(v) for v in x.values()))
            if isinstance(x,list):
                return {id(x)}.union(*(containers(v) for v in x))
            if isinstance(x,tuple):
                return set().union(*(containers(v) for v in x))
            return set()
        for name in ('info','obs','metrics'):
            h4m4_require(not (containers(getattr(st,name)) & containers(getattr(out,name))),
                'CLONE_MUTABLE_ALIAS')
        return out
    def step(st,capture,request,geometry):
        if request is None:
            result=step_fn(st,placement_capture=capture)
        else:
            mask=np.zeros(29,bool); mask[geometry['indices']]=True
            result=step_fn(st,placement_capture=capture,
                probe_residual=jnp.asarray(request),probe_selected=jnp.asarray(mask))
        raw=jax.device_get(result[1])
        packed=None; meta=None
        if capture:
            old=h4m3_pack_capture(jax.device_get(result[2]),h4m4_row(raw,np),manifest,np)
            packed=json.loads(json.dumps(old,allow_nan=False,
                default=lambda a:np.asarray(a).tolist()))
            if request is not None:
                meta=jax.device_get(result[3])
        return result[0],raw,packed,meta
    def geometry(st,packed,lane):
        q=np.asarray(jax.device_get(st.data.qpos))
        h4m4_require(np.array_equal(q,packed['qpos_post']),'FORK_QPOS')
        result=h4m4_geometry(q,legs,manifest,lane['second']['requested_world_dqvel_xy_mps'],
                             lane['index'],np)
        result['aligned_fk']=h4m4_verify_aligned(packed,legs,np)
        return result
    return h4m4_replay_lane(lane,session,reset,step,clone,
        lambda st,d,mag,t:h4m3_native_kick(st,d,mag,t,jax,jnp,np),
        lambda a,b:h4m3_tree_equal(a,b,jax,np),geometry,manifest,np,pulse=pulse)



def h4m4_enter_runtime(args):
    auth=h4m4_authorization()
    h4m3_no_runtime_imports()
    root=Path(__file__).resolve().parent
    session=getattr(args,'h3_session',None)
    h4m4_require(not any(getattr(args,k,False) for k in ('h4m3_placement','h4cc_eval',
        'h4cc_technical','h4_m1_eval','h4_technical','h3_m1_eval','h2a','h2b'))
        and args.output.absolute()==root/H4M4_OUTPUT and args.output.resolve()==root/H4M4_OUTPUT
        and list(args.seeds)==[x[1] for x in H4M3_LANES] and list(args.profiles)==['stand']
        and args.checkpoint_c==root/H4M3_BASE_POLICY and session is not None
        and session['authorization']==auth and session['status']=='M470_H4M4_IN_PROGRESS'
        and session['attempted']==session['recorded']==session['resets_attempted']==0
        and session['lanes']==[] and args.output.is_file() and not args.output.is_symlink(),
        'SESSION_REQUIRED')
    marker=args.output.parent/'dispatch.started.json'
    h4m4_require(marker.is_file() and not marker.is_symlink() and
        json.loads(marker.read_bytes())=={'authorization':auth}
        and json.loads(args.output.read_bytes())==session,'DISPATCH_MARKER')
    with (args.output.parent/'runtime.entered.json').open('x',encoding='utf8') as f:
        json.dump({'authorization':auth},f,sort_keys=True,allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    lanes,pins,approved=h4m4_read_evidence(root,args.h4m4_historical_archive)
    h4m4_require(pins==session['input_sha256']
        and h4m4_external_sources(root)==session['external_sources']
        and session['mjx_sources']==session['external_sources']['mjx']
        and h4m3_digest(h4m3_source_map(root))==auth['source_sha256'],'PRE_RUNTIME_PINS')
    args.h4m4_lanes=lanes
    args.h4m4_approved_manifest=approved['manifest']
    args.h4m4_approved_brax=approved['brax_source_verification']


def h4m4_dispatch(args):
    import shutil
    auth=h4m4_authorization()
    root=Path(__file__).resolve().parent
    h4m4_require(args.mode=='run' and args.output.absolute()==root/H4M4_OUTPUT
        and args.output.resolve()==root/H4M4_OUTPUT and
        not any(getattr(args,k,False) for k in ('h4m3_placement','h4cc_eval',
        'h4cc_technical','h4_m1_eval','h4_technical','h3_m1_eval','h2a','h2b')),
        'DISPATCH_PROTOCOL')
    h4m3_no_runtime_imports()
    lanes,pins,approved=h4m4_read_evidence(root,args.h4m4_historical_archive)
    external=h4m4_external_sources(root)
    h4m4_require(os.environ.get('JAX_PLATFORMS')=='cuda'
        and h4m3_digest(h4m3_source_map(root))==auth['source_sha256']
        and shutil.disk_usage(args.output.parent.parent).free>=2*1024**3+H4M3_REPORT_CAP,
        'PREFLIGHT_STORAGE_SOURCE')
    from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
    h4m3_no_runtime_imports()
    session={'status':'M470_H4M4_IN_PROGRESS','authorization':auth,
        'plan':json.loads(json.dumps(h4m4_plan())),'input_sha256':pins,
        'external_sources':external,'mjx_sources':external['mjx'],
        'brax_train_path':approved['brax_source_verification']['path'],
        'frame_columns':list(H4M3_COLUMNS),'lanes':[],'attempted':0,'recorded':0,
        'resets_attempted':0,'resets_recorded':0,'suffixes_attempted':0,'suffixes_recorded':0,
        'rich_recorded':0,'rich_bytes':0,'training_performed':False,
        'efficacy_evaluated':False,'optimizer_updates':0,'checkpoint_saves':0,
        'promotion_authorized':False}
    marker=h4m3_reserve(args.output,auth)
    h4m3_write(args.output,session,initial=True)
    args.h3_session=session; args.checkpoint_c=root/H4M3_BASE_POLICY
    args.seeds=[x[1] for x in H4M3_LANES]; args.profiles=['stand']
    monitor=GPUMonitor(); args.h3_monitor=monitor
    try:
        gpu=sample_gpu()
        h4m4_require(gpu['total_mib']==12288 and 0<=gpu['used_mib']<.95*12288,'GPU_PREFLIGHT')
        with monitor:
            run(args)
        session['gpu_initial']=gpu; session['gpu']=h3_gpu_check(monitor)
        h4m4_require(h4m4_authorization()==auth and
            h4m4_read_evidence(root,args.h4m4_historical_archive)[1]==pins
            and h4m4_external_sources(root)==external
            and json.loads(marker.read_bytes())=={'authorization':auth}
            and json.loads((args.output.parent/'runtime.entered.json').read_bytes())
                =={'authorization':auth},'POST_IMMUTABILITY')
        session['ledger']=h4m4_final_ledger(session)
        import numpy as np
        session['mechanism']=h4m4_summary(session,np)
        session['status']='M470_H4M4_TECHNICAL_PROBE_COMPLETED'
        h4m3_write(args.output,session)
        print(session['status'],session['ledger'],session['mechanism']['decision'],flush=True)
    except BaseException as exc:
        session['status']='M470_H4M4_STOP'
        session['error']=(type(exc).__name__+': '+str(exc))[:4096]
        session['gpu']=monitor.snapshot()
        try:
            h4m3_write(args.output,session)
        except BaseException as write_error:
            print('M470_H4M4_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise



# H4-M5-1: held observation-channel attribution, no environment path.
H4M5_OUTPUT=Path('m463_runs/m470_h4_m5_obs_attribution_v1/report.json')
H4M5_SPEC_SHA='5b07c4a667198b0df3bed92c8466bef69fbf1f5400d62e11d297cf4e37c9f35e'
H4M5_SPEC={"actor_template_init_cap":1,"checkpoint_saves":0,"critic_template_init_cap":1,"endpoint_calls":104,"execution_authorized":False,"fixed_slices":[[9,12],[99,103]],"groups":[["linvel",0,3],["gyro",3,6],["gravity",6,9],["joint_pos",12,41],["joint_vel",41,70],["history",70,99]],"h4m3_report_sha256":"329ff0564d47e05e2360fdf95350928895cf515e8cec6ca9aa2182133319258b","hybrid_calls":638,"id":"H4-M5-OBS-ATTRIBUTION-V1","inference_cap":742,"lane_ticks":[["CCRC",7357,"left",[336,337,338,339,340,341,342,343,344,345,354,365,385]],["CCRC",7601,"right",[336,337,338,339,340,341,342,343,344,345,354,358,360]],["CCRC",7613,"left",[336,337,338,339,340,341,342,343,344,345,350]],["C",7357,"left",[336,337]],["REF",7331,"left",[336,337,338,339,340,341,342,343,344,345,354,365,385]]],"log_cap":16777216,"logical_slots":832,"masks":[0,1,2,4,8,15,16,31,32,47,48,55,59,61,62,63],"metadata_cap":1048576,"optimizer_updates":0,"original_action_match":"float32_bit_exact","pair_digest_sha256":"8d8233f019f2ca18d174f67e2a0371a502e5f4146b6f9d83f0d32c3974d48c15","physical_transitions":0,"primary_ticks":[337,338,339,340,341,342,343,344,345],"report_cap":4194304,"slot_cap":2048,"source_report_sha256":"83d87cdd5279ab13acb06815083b72f02ecadcabc652cff35b24a82814beb9a5","thresholds":{"ankle_min_abs_delta":0.0001,"ankle_min_each_context_share":0.3,"ankle_min_symmetric_share":0.5,"max_absolute_context_gap_share":0.5,"max_absolute_residual_share":0.25,"min_A":0.05,"min_each_context_share":0.3,"min_opposing_ticks":6,"min_symmetric_share":0.5,"opposing_epsilon":0.0001}}

def h4m5_require(ok,label):
    if not ok:
        raise RuntimeError('M470_H4M5_'+label)


def h4m5_plan():
    result=json.loads(json.dumps(H4M5_SPEC))
    h4m5_require(h4m3_digest(result)==H4M5_SPEC_SHA,'SPEC_PIN')
    return result


def h4m5_vector(value,n,np):
    x=np.asarray(value)
    h4m5_require(x.shape==(n,) and x.dtype.kind=='f' and np.isfinite(x).all(),'ARCHIVE_VECTOR')
    y=np.asarray(value,dtype='<f4')
    h4m5_require(np.isfinite(y).all() and np.array_equal(x,y),'ARCHIVE_FLOAT32')
    return y


def h4m5_mask(pair,mask):
    h4m5_require(mask in H4M5_SPEC['masks'],'MASK')
    b,t=pair['obs']
    x=b.copy()
    for i,(_,lo,hi) in enumerate(H4M5_SPEC['groups']):
        if mask & (1<<i): x[lo:hi]=t[lo:hi]
    return x


def h4m5_pairs(report,old,np):
    p=h4m5_plan()
    h4m5_require(report['status']=='M470_H4M4_TECHNICAL_PROBE_COMPLETED'
        and report['attempted']==report['recorded']==2229
        and report['rich_recorded']==759 and report['optimizer_updates']==report['checkpoint_saves']==0
        and report['training_performed'] is False and report['efficacy_evaluated'] is False
        and report['manifest']==old['manifest'] and len(report['lanes'])==len(old['lanes'])==5,
        'REPORT_CONTRACT')
    m=report['manifest']; f=m['fields']
    h4m5_require(m['joint_names'][f['actuator_trnid'][5][0]]=='left_ankle_roll_joint',
        'ANKLE_MAP')
    out=[]; digest=hashlib.sha256(); counts=[]; active=0; checked=0
    for i,(lane,prev,key) in enumerate(zip(report['lanes'],old['lanes'],p['lane_ticks'])):
        h4m5_require([lane[k] for k in ('policy','seed','direction')]==key[:3]
            and [prev[k] for k in ('policy','seed','direction')]==key[:3],'LANE')
        ds={a:{x['tick']:x for x in lane['rich'] if x['arm']==a}
            for a in ('prefix','baseline','sham','treatment')}
        historical={x['tick']:x['data'] for x in prev['rich'] if x['segment']!='single'}
        h4m5_require(sum(map(len,ds.values()))==len(lane['rich'])
            and lane['prefix']==prev['segments']['nominal']+prev['segments']['first']+prev['segments']['double'][:1]
            and lane['arms']['baseline']['rows']==lane['arms']['sham']['rows']==prev['segments']['double'][1:],
            'ARCHIVE_CONTROL')
        for arm,rows in ds.items():
            for tick,rec in rows.items():
                d=rec['data']; x=h4m5_vector(d['actor_input'],103,np)
                a=h4m5_vector(rec['residual']['actor_action'],29,np)
                sent=h4m5_vector(rec['residual']['sent_action'],29,np)
                h4m5_require(np.max(abs(a))<=1 and np.max(abs(sent))<=1
                    and sent.tobytes()==h4m5_vector(d['action'],29,np).tobytes()
                    and d['inference_key']==[0,0],'ARCHIVE_ACTION')
                if arm!='treatment':
                    h4m5_require(d==historical[tick],'PACKED_CONTROL')
                predecessor=ds['prefix'].get(tick-2) if tick-2<=335 else rows.get(tick-2)
                if predecessor is not None:
                    h4m5_require(x[70:99].tobytes()==h4m5_vector(
                        predecessor['data']['action'],29,np).tobytes(),'HISTORY_T2')
                checked+=1
        g=lane['geometry']; ids=g['indices']; u=np.asarray(g['u'],dtype=np.float64)
        h4m5_require(len(ids)==len(set(ids))==3 and 5 not in ids and all(0<=j<29 for j in ids)
            and u.shape==(3,) and np.isfinite(u).all() and np.linalg.norm(u)>0,'DIRECTION')
        count=0
        for t in key[3]:
            B,T=ds['baseline'][t],ds['treatment'][t]
            h4m5_require(B['data']==ds['sham'][t]['data']
                and B['data']['rng']==T['data']['rng']
                and B['data']['command']==T['data']['command']==[0.,0.,0.],
                'PAIR_CONTEXT')
            x,y=[h4m5_vector(z['data']['actor_input'],103,np) for z in (B,T)]
            a,b=[h4m5_vector(z['residual']['actor_action'],29,np) for z in (B,T)]
            for lo,hi in p['fixed_slices']:
                h4m5_require(x[lo:hi].tobytes()==y[lo:hi].tobytes(),'FIXED_CHANNEL')
            if t==336:
                h4m5_require(x.tobytes()==y.tobytes() and a.tobytes()==b.tobytes(),'IDENTICAL_336')
            if t==337:
                h4m5_require(x[70:99].tobytes()==y[70:99].tobytes(),'HISTORY_337')
            live=bool(np.any(h4m5_vector(T['residual']['delivered_residual'],29,np)))
            active+=live
            pair={'lane':i,'tick':t,'policy':key[0],'obs':(x,y),'actions':(a,b),
                  'indices':ids,'unit':u/np.linalg.norm(u),'active':live}
            pair['ankle_context']={arm:{'q':z['data']['qpos_post'][m['qpos_addresses'][5]],
                'qvel':z['data']['qvel_post'][m['qvel_addresses'][5]],
                'load':z['data']['contact']['normal_load_bodyweight'],
                'raw':z['data']['raw_ctrl'][5],'effective':z['data']['derived_effective_ctrl'][5]}
                for arm,z in (('baseline',B),('treatment',T))}
            digest.update(bytes([i])+t.to_bytes(2,'little')+x.tobytes()+y.tobytes()+a.tobytes()+b.tobytes())
            keys={h4m5_mask(pair,mask).tobytes() for mask in p['masks']}
            count+=len(keys)+int(x.tobytes()==y.tobytes())
            out.append(pair)
        counts.append(count)
    h4m5_require(len(out)==52 and checked==759 and active==37
        and counts==[190,190,158,14,190] and digest.hexdigest()==p['pair_digest_sha256'],'PAIR_DIGEST')
    return out


def h4m5_effects(pair,outputs,manifest,np):
    F={k:np.asarray(outputs[k],dtype=np.float64) for k in H4M5_SPEC['masks']}
    delta=F[63]-F[0]
    db=np.stack([F[1<<i]-F[0] for i in range(6)])
    dt=np.stack([F[63]-F[63^(1<<i)] for i in range(6)])
    phi=(db+dt)/2.; gap=dt-db; E=delta-phi.sum(axis=0)
    IB=F[48]-F[16]-F[32]+F[0]; IT=F[63]-F[47]-F[31]+F[15]
    h4m5_require(np.max(abs(delta-phi.sum(axis=0)-E))<=1e-12,'ARITHMETIC')
    if pair['tick']==336:
        h4m5_require(all(not np.any(x) for x in (delta,db,dt,phi,gap,E,IB,IT)),'ZERO_336')
    if pair['tick']==337:
        h4m5_require(all(not np.any(x) for x in (db[5],dt[5],IB,IT)),'ZERO_HISTORY_337')
    project=lambda x:(x[...,pair['indices']]@pair['unit']).tolist()
    den=float(delta[5]); shares=None
    if abs(den)>=.0001:
        shares={'baseline':(db[:,5]/den).tolist(),'treatment':(dt[:,5]/den).tolist(),
                'symmetric':(phi[:,5]/den).tolist(),
                'material':((phi[:,5]/den>=.5)&(db[:,5]/den>=.3)&(dt[:,5]/den>=.3)).tolist()}
    f=manifest['fields']; lo,hi=f['actuator_ctrlrange'][5]; default=np.float32(manifest['default_pose'][5])
    raw={str(k):float(default+np.float32(.5)*np.float32(v[5])) for k,v in F.items()}
    eff={k:min(hi,max(lo,v)) if f['actuator_ctrllimited'][5] else v for k,v in raw.items()}
    return {'lane':pair['lane'],'tick':pair['tick'],'active':pair['active'],
            # dB/dT vectors are reconstructible from F slots and phi +/- gap/2.
            'delta':delta.tolist(),'phi':phi.tolist(),
            'gap':gap.tolist(),'E':E.tolist(),'I_B':IB.tolist(),'I_T':IT.tolist(),
            'projection':{k:project(v) for k,v in [('delta',delta),('dB',db),('dT',dt),
                ('phi',phi),('gap',gap),('E',E),('I_B',IB),('I_T',IT)]},
            'ankle':{'delta':den,'shares':shares,'raw':raw,'effective':eff,
                     'context':pair['ankle_context']}}


def h4m5_decision(effects,np):
    results=[]
    for lane in range(5):
        a=[x['projection'] for x in effects if x['lane']==lane and x['active'] and x['tick']!=336]
        h4m5_require(len(a)==[9,9,9,1,4][lane],'ACTIVE_COUNT')
        A=-sum(x['delta'] for x in a)
        V=-np.sum([x['phi'] for x in a],axis=0)
        B=-np.sum([x['dB'] for x in a],axis=0); T=-np.sum([x['dT'] for x in a],axis=0)
        residual=sum(abs(x['E']) for x in a); context=sum(abs(x['gap'][4]) for x in a)
        opposing=sum(x['phi'][4]<-.0001 for x in a)
        r={'lane':lane,'n':len(a),'A':A,'group_symmetric':V.tolist(),'group_baseline':B.tolist(),
           'group_treatment':T.tolist(),'absolute_residual':residual,'absolute_velocity_gap':context,
           'opposing_ticks':opposing,'decision':'DESCRIPTIVE_ONLY','criteria':None,'shares':None}
        if A>=.05:
            r['shares']={'symmetric':float(V[4]/A),'baseline':float(B[4]/A),'treatment':float(T[4]/A),
                         'residual':residual/A,'context':context/A}
            c={'magnitude':True,'symmetric':V[4]/A>=.5,'baseline':B[4]/A>=.3,
               'treatment':T[4]/A>=.3,'opposing':opposing>=6,
               'dominance':V[4]>=max(abs(V[j]) for j in (0,1,2,3,5)),
               'residual':residual/A<=.25,'context':context/A<=.5}
            r['criteria']={k:bool(v) for k,v in c.items()}
            if lane in (1,2):
                r['decision']='INCONCLUSIVE_CONTEXT' if not(c['residual'] and c['context']) else (
                    'SUPPORT_LOCAL_VELOCITY' if all(c.values()) else 'NOT_SUPPORTED_IN_TESTED_INPUTS')
        elif lane in (1,2):
            r['decision']='INCONCLUSIVE_CONTEXT';r['criteria']={'magnitude':False}
        results.append(r)
    primary=[results[i]['decision'] for i in (1,2)]
    n=primary.count('SUPPORT_LOCAL_VELOCITY')
    return {'lanes':results,'decision':'SUPPORT_LOCAL_VELOCITY' if n==2 else 'HETEROGENEOUS' if n==1 else
            'INCONCLUSIVE' if 'INCONCLUSIVE_CONTEXT' in primary else 'NOT_SUPPORTED'}


def h4m5_collect(pairs,manifest,session,infer,np,persist):
    """Host ledger; infer callback is the only inference boundary."""
    p=h4m5_plan(); cache=[{} for _ in pairs]; outputs=[{} for _ in pairs]
    def slot(i,mask,phase):
        pair=pairs[i]; x=h4m5_mask(pair,mask); key=x.tobytes()
        prior=cache[i].get(key) if phase=='hybrid' else None
        if prior is None:
            h4m5_require(session['attempted']<p['inference_cap']
                and session[phase+'_attempted']<p['endpoint_calls' if phase=='endpoint' else 'hybrid_calls'],'INFERENCE_CAP')
            session['attempted']+=1;session[phase+'_attempted']+=1
            call=session['attempted'];session['last_attempt']={'pair':i,'mask':mask,'call':call,'phase':phase}
            y=np.asarray(infer(pair['policy'],x.copy()))
            h4m5_require(y.shape==(29,) and y.dtype==np.float32 and np.isfinite(y).all()
                and np.max(abs(y))<=1,'INFERENCE_OUTPUT')
            if phase=='endpoint' and y.tobytes()!=pair['actions'][mask==63].tobytes():
                session['mismatch']={'pair':i,'mask':mask,'expected':pair['actions'][mask==63].tolist(),
                    'actual':y.tolist()}
                raise RuntimeError('M470_H4M5_ORIGINAL_ACTION_MISMATCH')
            session['recorded']+=1;session[phase+'_recorded']+=1
            cache[i].setdefault(key,(call,y.copy()))
        else:
            call,y=prior
        rec={'pair':i,'mask':mask,'input_sha256':hashlib.sha256(key).hexdigest(),
             'call':call,'alias':prior is not None,'action':y.tolist()}
        h4m5_require(len(session['slots'])<p['logical_slots'] and
            len(json.dumps(rec,allow_nan=False,separators=(',',':')).encode())<=p['slot_cap'],'SLOT_CAP')
        session['slots'].append(rec);outputs[i][mask]=y.copy()
        if len(session['slots'])%16==0: persist()
    for i in range(len(pairs)):
        for mask in (0,63):slot(i,mask,'endpoint')
    h4m5_require(session['endpoint_recorded']==session['endpoint_attempted']==104
        and session['hybrid_attempted']==0,'PHASE_A')
    persist()
    for i,pair in enumerate(pairs):
        for mask in p['masks']:
            if mask not in (0,63):slot(i,mask,'hybrid')
        session['effects'].append(h4m5_effects(pair,outputs[i],manifest,np))
        persist()
    h4m5_require(session['attempted']==session['recorded']==742
        and session['hybrid_attempted']==session['hybrid_recorded']==638
        and len(session['slots'])==832 and sum(x['alias'] for x in session['slots'])==90
        and len(session['effects'])==52,'FINAL_LEDGER')
    session['decision']=h4m5_decision(session['effects'],np)



def h4m5_final_ledger(pairs,session,np):
    p=h4m5_plan()
    endpoints=[(i,k) for i in range(52) for k in (0,63)]
    hybrids=[(i,k) for i in range(52) for k in p['masks'] if k not in (0,63)]
    records=session['slots']; calls={}; cache={}; lane_counts=[0]*5
    h4m5_require([(x['pair'],x['mask']) for x in records]==endpoints+hybrids,'SLOT_ORDER')
    for slot in records:
        i=slot['pair']; k=slot['mask']; pair=pairs[i]
        raw=h4m5_mask(pair,k).tobytes(); key=(i,raw)
        h4m5_require(slot['input_sha256']==hashlib.sha256(raw).hexdigest(),'SLOT_INPUT')
        y=h4m5_vector(slot['action'],29,np)
        h4m5_require(np.max(abs(y))<=1,'FINAL_ACTION_ENVELOPE')
        if k in (0,63):
            h4m5_require(not slot['alias'] and y.tobytes()==pair['actions'][k==63].tobytes(),
                'FINAL_ORIGINAL_ACTION')
        prior=cache.get(key)
        if slot['alias']:
            h4m5_require(k not in (0,63) and prior is not None and slot['call']==prior
                and y.tobytes()==h4m5_vector(calls[prior]['action'],29,np).tobytes(),'ALIAS_IDENTITY')
        else:
            h4m5_require(slot['call']==len(calls)+1 and (prior is None or k in (0,63)),
                'CALL_IDENTITY')
            calls[slot['call']]=slot; cache.setdefault(key,slot['call'])
            lane_counts[pair['lane']]+=1
    h4m5_require(len(calls)==742 and lane_counts==[190,190,158,14,190]
        and session['attempted']==session['recorded']==len(calls)
        and session['endpoint_attempted']==session['endpoint_recorded']==104
        and session['hybrid_attempted']==session['hybrid_recorded']==638
        and len(records)==832 and sum(x['alias'] for x in records)==90,'FINAL_COUNTS')
    return {'calls':len(calls),'aliases':90,'slots':832,'lane_calls':lane_counts}


def h4m5_read_evidence(root,historical,np):
    root=Path(root).resolve()
    r=h4m3_pinned_json(root/H4M4_OUTPUT,H4M5_SPEC['source_report_sha256'])
    _,pins,old=h4m4_read_evidence(root,historical)
    old_bytes=subprocess.check_output(['git','-C',str(root),'show',
        '3fa74f887ff3331c1de104d243d386099c38a2a9:m463_gate3_cp_eval.py'])
    old_sha=hashlib.sha256(old_bytes).hexdigest()
    h4m5_require(hashlib.sha1(('blob '+str(len(old_bytes))+'\0').encode()+old_bytes).hexdigest()
        =='4f8b176a402eced1125c9986b3bec3546bd515a9'
        and len(pins)==44 and {**pins,'diagnostic_evaluator':old_sha}==r['input_sha256']
        and h4m3_digest({**h4m3_source_map(root),'m463_gate3_cp_eval.py':old_sha})
            ==r['authorization']['source_sha256']
        and r['authorization']['policy_sha256']==H4M3_POLICIES
        and r['external_sources']==h4m4_external_sources(root),'H4M4_PROVENANCE')
    h4m5_require(r['brax_source_verification']==old['brax_source_verification'],'BRAX_IDENTITY')
    return h4m5_pairs(r,old,np),{**pins,'approved_h4m4_report':H4M5_SPEC['source_report_sha256'],
        'approved_h4m4_evaluator':old_sha},r


def h4m5_find_source(name):
    # PathFinder resolves nested module paths without executing package __init__.
    from importlib.machinery import PathFinder
    path=None; spec=None
    for i in range(1,len(name.split('.'))+1):
        spec=PathFinder.find_spec('.'.join(name.split('.')[:i]),path)
        h4m5_require(spec is not None,'MODULE_SPEC_'+name)
        path=spec.submodule_search_locations
    h4m5_require(spec.origin is not None,'MODULE_ORIGIN_'+name)
    f=Path(spec.origin)
    h4m5_require(f.is_file() and not f.is_symlink() and f.resolve()==f.absolute(),
        'MODULE_FILE_'+name)
    return f


def h4m5_external_sources(root):
    from importlib import metadata
    h4m3_no_runtime_imports()
    external=h4m4_external_sources(root)
    public={
        'brax.training.agents.ppo.networks':'1922bad3499988bc664c3024f9640332d66982d5',
        'brax.training.networks':'ee63e0b5aa964ac9800b797d095a80a628eaf00e',
        'brax.training.distribution':'971a4a9abd0a9145eecec387a203ab7be8261a98',
        'brax.training.acme.running_statistics':'4361e185b49917543dfa759feb521be14aeaa487'}
    modules={}
    for name in (*public,'flax.serialization','jax','jaxlib','brax','flax','numpy',
                 'm462_typed_orbax_stage','m462_ppo_warmstart_run','train','paper_config'):
        path=h4m5_find_source(name)
        if name in public:
            h4m5_require(h4m3_git_blob(path)==public[name],'PUBLIC_SOURCE_'+name)
        if name.startswith('m462_') or name in ('train','paper_config'):
            h4m5_require(path==Path(root)/ (name+'.py'),'LOCAL_IMPORT_'+name)
        modules[name]={'path':str(path),'sha256':sha256(path)}
    versions={n:metadata.version(n) for n in ('jax','jaxlib','brax','flax','numpy','scipy',
        'optax','chex','orbax-checkpoint','ml-collections','etils','msgpack',
        'jax-cuda12-plugin','jax-cuda12-pjrt')}
    h4m5_require(all(versions[k]=='0.4.35' for k in
        ('jax','jaxlib','jax-cuda12-plugin','jax-cuda12-pjrt')) and versions['brax']=='0.12.1',
        'PACKAGE_VERSION')
    # Pin actual resolved package trees, excluding generated bytecode. This includes
    # Python source and shared-library bytes, not just dist-info version strings.
    packages={}
    for name in ('jax','jaxlib','brax','flax','numpy','scipy','optax','chex',
                 'ml_collections','etils','msgpack','mujoco'):
        path=h4m5_find_source(name).parent
        files=sorted(x for x in path.rglob('*') if x.is_file()
            and '__pycache__' not in x.parts and x.suffix not in ('.pyc','.pyo'))
        h4m5_require(files and all(not x.is_symlink() for x in files),'PACKAGE_FILES_'+name)
        h=hashlib.sha256()
        for x in files:
            h.update(x.relative_to(path).as_posix().encode()+b'\0')
            h.update(bytes.fromhex(sha256(x)))
        packages[name]={'path':str(path),'files':len(files),'sha256':h.hexdigest()}
    # Orbax is namespace packaged; pin the actual checkpoint package and serializer.
    path=h4m5_find_source('orbax.checkpoint').parent
    files=sorted(x for x in path.rglob('*') if x.is_file()
        and '__pycache__' not in x.parts and x.suffix not in ('.pyc','.pyo'))
    h4m5_require(files and all(not x.is_symlink() for x in files),'PACKAGE_ORBAX')
    h=hashlib.sha256()
    for x in files:
        h.update(x.relative_to(path).as_posix().encode()+b'\0'); h.update(bytes.fromhex(sha256(x)))
    packages['orbax.checkpoint']={'path':str(path),'files':len(files),'sha256':h.hexdigest()}
    native={}
    for name in ('jax-cuda12-plugin','jax-cuda12-pjrt','numpy','scipy'):
        dist=metadata.distribution(name)
        h4m5_require(dist.files is not None,'DIST_FILE_LIST_'+name)
        files={}
        for entry in sorted(dist.files,key=str):
            rel=str(entry)
            if '__pycache__' in entry.parts or entry.suffix in ('.pyc','.pyo'):
                continue
            if name in ('numpy','scipy') and not rel.startswith(name+'.libs/'):
                continue
            path=Path(dist.locate_file(entry))
            h4m5_require(path.is_file() and not path.is_symlink(),'NATIVE_FILE_'+rel)
            files[rel]={'path':str(path.resolve()),'sha256':sha256(path)}
        h4m5_require(bool(files) or name in ('numpy','scipy'),'NATIVE_DIST_'+name)
        native[name]=files
    h4m3_no_runtime_imports()
    return {'historical':external,'modules':modules,'versions':versions,'packages':packages,
            'native_distributions':native}


def h4m5_parse_authorization(rules,source,external):
    p=h4m5_plan()
    static=[x.split() for x in rules.splitlines() if x.startswith('H4M5_STATIC_APPROVED:')]
    active=[x.split() for x in rules.splitlines() if x.startswith('H4M5_ACTIVE_GO:')]
    h4m5_require(len(static)==len(active)==1,'EXECUTION_HOLD')
    h4m5_require(not any(x.split(':',1)[0].endswith('ACTIVE_GO') and
        x.split(':',1)[0]!='H4M5_ACTIVE_GO' and x.split(':',1)[0].replace('_','').isalnum()
        and x.split(':',1)[0].isupper() for x in rules.splitlines()),'CONCURRENT_GO')
    a,b=static[0],active[0]
    h4m5_require(len(a)==5 and a[1]==source and a[3]==external
        and len(a[2])==64 and all(c in '0123456789abcdef' for c in a[2]),'STATIC_PIN')
    log=Path(a[4])
    h4m5_require(log.parent==Path('m463_runs') and log.name.startswith('m470_h4_m5_1_')
        and log.suffix=='.log','STATIC_PATH')
    expected=['H4M5_ACTIVE_GO:',p['id'],source,a[2],external,H4M5_SPEC_SHA,p['pair_digest_sha256'],
        p['source_report_sha256'],p['h4m3_report_sha256'],*[H4M3_POLICIES[k] for k in ('C','REF','CCRC')],
        str(H4M5_OUTPUT)]
    h4m5_require(b==expected,'AUTHORIZATION_PIN')
    return {'id':p['id'],'source_sha256':source,'static_sha256':a[2],'external_sha256':external,
        'static_path':str(log),'spec_sha256':H4M5_SPEC_SHA,'pair_sha256':p['pair_digest_sha256'],
        'report_sha256':p['source_report_sha256'],'h4m3_report_sha256':p['h4m3_report_sha256'],
        'policy_sha256':dict(H4M3_POLICIES),'output':str(H4M5_OUTPUT)}


def h4m5_authorization(root=None):
    root=Path(__file__).resolve().parent if root is None else Path(root).resolve()
    rules=(root/'G1_RESEARCH_RULES.md').read_text('utf8')
    h4m5_require(any(x.startswith('H4M5_ACTIVE_GO:') for x in rules.splitlines()) and
        any(x.startswith('H4M5_STATIC_APPROVED:') for x in rules.splitlines()),'EXECUTION_HOLD')
    source=h4m3_digest(h4m3_source_map(root))
    external=h4m5_external_sources(root)
    auth=h4m5_parse_authorization(rules,source,h4m3_digest(external))
    path=root/auth['static_path']
    h4m5_require(path.is_file() and not path.is_symlink() and path.resolve()==path.absolute()
        and 0<path.stat().st_size<=H4M5_SPEC['log_cap'] and sha256(path)==auth['static_sha256'],
        'STATIC_LOG')
    lines=path.read_text('utf8').splitlines()
    h4m5_require('H4_M5_1_FULL_STATIC_PASS '+source in lines
        and 'H4_M5_1_EXTERNAL_SHA256 '+auth['external_sha256'] in lines
        and 'H4_M5_1_PAIR_SHA256 '+auth['pair_sha256'] in lines
        and 'H4_M5_1_SPEC_SHA256 '+H4M5_SPEC_SHA in lines,'STATIC_ACCEPTANCE')
    return auth,external


def h4m5_write(path,session,initial=False):
    import shutil
    p=h4m5_plan(); path=Path(path)
    data=(json.dumps(session,allow_nan=False,separators=(',',':'))+'\n').encode()
    metadata={**session,'slots':[]}
    h4m5_require(len(data)<=p['report_cap'] and len(json.dumps(metadata,allow_nan=False,
        separators=(',',':')).encode())+1<=p['metadata_cap'],'REPORT_CAP')
    h4m5_require(path.parent.is_dir() and not path.parent.is_symlink() and
        shutil.disk_usage(path.parent).free>=2*1024**3+p['report_cap'],'REPORT_STORAGE')
    tmp=path.with_name(path.name+'.tmp')
    h4m5_require(not tmp.exists() and not tmp.is_symlink() and not path.is_symlink()
        and path.exists()!=initial,'REPORT_PATH')
    with (path if initial else tmp).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    if not initial: os.replace(tmp,path)


def h4m5_tree_digest(tree,jax,np):
    h=hashlib.sha256(); leaves=jax.tree_util.tree_leaves(tree)
    h.update(str(jax.tree_util.tree_structure(tree)).encode())
    for leaf in leaves:
        a=np.asarray(jax.device_get(leaf))
        h4m5_require(np.isfinite(a).all(),'TREE_FINITE')
        h.update(json.dumps([list(a.shape),str(a.dtype)],separators=(',',':')).encode()+b'\0')
        h.update(a.tobytes())
    return h.hexdigest()


def h4m5_runtime(args,session,pairs,approved,monitor):
    # Re-enter only from reserved dispatcher; rehash before any runtime import.
    root=Path(__file__).resolve().parent
    auth,external=h4m5_authorization(root); h4m3_no_runtime_imports()
    out=root/H4M5_OUTPUT
    h4m5_require(args.output==out and out.is_file() and not out.is_symlink()
        and (out.parent/'dispatch.started.json').is_file()
        and not (out.parent/'dispatch.started.json').is_symlink()
        and session['authorization']==auth
        and session['external']==external and session['status']=='M470_H4M5_IN_PROGRESS'
        and session['attempted']==session['recorded']==0 and session['slots']==[]
        and json.loads(out.read_bytes())==session
        and json.loads((out.parent/'dispatch.started.json').read_bytes())=={'authorization':auth},
        'SESSION_REQUIRED')
    with (out.parent/'runtime.entered.json').open('x',encoding='utf8') as f:
        json.dump({'authorization':auth},f,sort_keys=True,allow_nan=False); f.flush();os.fsync(f.fileno())
    import numpy as np
    historical=root/'m463_runs/m470_h3_m1_3_corrected_20261124_192702_101671859/report.json'
    checked,pins,checked_report=h4m5_read_evidence(root,historical,np)
    h4m5_require(pins==session['input_sha256'] and checked_report==approved
        and all(a['obs'][0].tobytes()==b['obs'][0].tobytes() and
            a['obs'][1].tobytes()==b['obs'][1].tobytes() for a,b in zip(checked,pairs)),
        'PRE_RUNTIME_INPUTS')
    from train import check_source,build_paper_config
    from paper_config import BASE
    check_source()
    sys.path.insert(0,str(BASE/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    compat=external['historical']['nvcc_compat']
    h4m5_require(str(Path(inspect.getfile(apply)).resolve())==compat['path']
        and sha256(inspect.getfile(apply))==compat['sha256'],'NVCC_PIN')
    h4m3_no_runtime_imports()
    print('M470_H4M5_NVCC',apply(),flush=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    h4m5_require(jax.default_backend()=='gpu' and jax.__version__=='0.4.35'
        and jax.process_count()==jax.local_device_count()==1,'RUNTIME_PIN')
    h4m5_require(h3_verify_brax_train_source(networks)==approved['brax_source_verification'],
        'BRAX_TRAIN_PIN')
    def check_origins():
        h4m5_require('brax.training.agents.ppo.train' not in sys.modules,'TRAIN_MODULE_IMPORTED')
        for name,pin in external['modules'].items():
            mod=sys.modules.get(name)
            h4m5_require(mod is not None,'REQUIRED_IMPORTED_MODULE_'+name)
            if mod is not None:
                path=Path(mod.__file__).resolve()
                h4m5_require(str(path)==pin['path'] and sha256(path)==pin['sha256'],
                    'IMPORTED_ORIGIN_'+name)
        native={v['path']:v['sha256'] for dist in external['native_distributions'].values()
                for v in dist.values()}
        for name,mod in list(sys.modules.items()):
            if not getattr(mod,'__file__',None):continue
            if name.startswith(('jax_plugins.','jax_cuda12_plugin','jax_cuda12_pjrt')):
                path=str(Path(mod.__file__).resolve())
                h4m5_require(path in native and sha256(path)==native[path],'CUDA_PLUGIN_ORIGIN_'+name)
            for package,pin in external['packages'].items():
                if name==package or name.startswith(package+'.'):
                    h4m5_require(Path(mod.__file__).resolve().is_relative_to(Path(pin['path'])),
                        'TRANSITIVE_ORIGIN_'+name)
    paper=build_paper_config()
    h4m5_require(dict(paper.network_factory).get('policy_obs_key','state')=='state'
        and dict(paper.network_factory)['value_obs_key']=='privileged_state','NETWORK_KEYS')
    trees={}
    def persist():
        session['gpu']=h3_gpu_check(monitor); h4m5_write(out,session)
    def event(name,stage):
        k=name+'_template_'+stage
        h4m5_require(session[k]==0,'TEMPLATE_CAP');session[k]+=1
    def audit(path,tree,template):
        h4m5_require(jax.tree_util.tree_structure(tree)==jax.tree_util.tree_structure(template),
            'TREE_SCHEMA')
        for a,b in zip(jax.tree_util.tree_leaves(tree),jax.tree_util.tree_leaves(template)):
            x,y=np.asarray(jax.device_get(a)),np.asarray(jax.device_get(b))
            h4m5_require(x.shape==y.shape and x.dtype==y.dtype and np.isfinite(x).all(),'TREE_LEAF')
        h4m5_require(all(np.all(np.asarray(jax.device_get(x))>0)
            for x in jax.tree_util.tree_leaves(tree[0].std)),'NORMALIZER_STD')
        key=str(path);h4m5_require(key not in trees,'DUPLICATE_POLICY')
        trees[key]=(tree,h4m5_tree_digest(tree,jax,np))
    load=h4_shared_frozen_loader(29,paper,networks,running_statistics,serialization,jax,jnp,np,
        audit=audit,template_event=event)
    policies={k:jax.jit(load(v)) for k,v in h4m3_policy_paths(root).items()}
    h4m5_require(len(trees)==3 and all(session[k]==1 for k in
        ('actor_template_attempted','actor_template_recorded','critic_template_attempted','critic_template_recorded')),
        'TEMPLATE_LEDGER')
    session['tree_sha256']={k:v[1] for k,v in trees.items()}
    check_origins(); persist()
    privileged=jnp.zeros((216,),dtype=jnp.float32); key=jax.random.PRNGKey(0)
    def infer(policy,x):
        result,extra=policies[policy]({'state':jnp.asarray(x,dtype=jnp.float32),
            'privileged_state':privileged},key)
        h4m5_require(extra=={},'DETERMINISTIC_EXTRA')
        return np.asarray(jax.device_get(result.block_until_ready()))
    h4m5_collect(pairs,approved['manifest'],session,infer,np,persist)
    session['ledger']=h4m5_final_ledger(pairs,session,np)
    h4m5_require(all(h4m5_tree_digest(tree,jax,np)==digest for tree,digest in trees.values()),
        'TREE_IMMUTABILITY')
    check_origins(); persist()


def h4m5_dispatch(args):
    import shutil
    import numpy as np
    root=Path(__file__).resolve().parent; out=root/H4M5_OUTPUT
    auth,external=h4m5_authorization(root)
    h4m5_require(args.mode=='run' and args.output==out and out.resolve()==out
        and os.environ.get('JAX_PLATFORMS')=='cuda'
        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
        and os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')=='false'
        and shutil.disk_usage(out.parent.parent).free>=2*1024**3+H4M5_SPEC['report_cap'],
        'DISPATCH_PREFLIGHT')
    historical=root/'m463_runs/m470_h3_m1_3_corrected_20261124_192702_101671859/report.json'
    pairs,pins,approved=h4m5_read_evidence(root,historical,np)
    from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
    h4m3_no_runtime_imports()
    session={'status':'M470_H4M5_IN_PROGRESS','authorization':auth,'external':external,
        'input_sha256':pins,'plan':h4m5_plan(),'slots':[],'effects':[],
        'attempted':0,'recorded':0,'endpoint_attempted':0,'endpoint_recorded':0,
        'hybrid_attempted':0,'hybrid_recorded':0,'actor_template_attempted':0,
        'actor_template_recorded':0,'critic_template_attempted':0,'critic_template_recorded':0,
        'physical_transitions':0,'resets':0,'optimizer_updates':0,'checkpoint_saves':0,
        'training_performed':False,'efficacy_evaluated':False,'promotion_authorized':False}
    marker=h4m3_reserve(out,auth);h4m5_write(out,session,initial=True)
    monitor=GPUMonitor()
    try:
        gpu=sample_gpu()
        h4m5_require(gpu['total_mib']==12288 and 0<=gpu['used_mib']<.95*12288,'GPU_PREFLIGHT')
        with monitor:
            h4m5_runtime(args,session,pairs,approved,monitor)
        session['gpu_initial']=gpu;session['gpu']=h3_gpu_check(monitor)
        # No new runtime imports or inference in post-check; external scan occurs
        # in an independent clean interpreter to avoid pre-import contamination.
        command=[sys.executable,'-B','-c',
            'import json;import m463_gate3_cp_eval as e;from pathlib import Path;print(json.dumps(e.h4m5_authorization(Path.cwd()),sort_keys=True))']
        after=json.loads(subprocess.check_output(command,cwd=root,text=True))
        h4m5_require(after==[auth,external] and h4m5_read_evidence(root,historical,np)[1]==pins
            and json.loads(marker.read_bytes())=={'authorization':auth}
            and json.loads((out.parent/'runtime.entered.json').read_bytes())=={'authorization':auth},
            'POST_IMMUTABILITY')
        session['status']='M470_H4M5_TECHNICAL_ATTRIBUTION_COMPLETED'
        h4m5_write(out,session)
        print(session['status'],session['ledger'],session['decision']['decision'],flush=True)
    except BaseException as exc:
        session['status']='M470_H4M5_STOP';session['error']=(type(exc).__name__+': '+str(exc))[:4096]
        session['gpu']=monitor.snapshot()
        try:h4m5_write(out,session)
        except BaseException as err:print('M470_H4M5_PARTIAL_WRITE_FAILED',repr(err),flush=True)
        raise



# H4-M6: independent held technical and development comparison contracts.
H4M6_TECH_OUTPUT=Path('m463_runs/m470_h4_m6_tech_v3/report.json')
H4M6_V1_STOP_PINS={
    'm463_runs/m470_h4_m6_tech_v1/report.json':'733ba14afd9fcba1d6ad17aeafb5ad9f54e24fd5d515595efe4d7b5ccfba18cc',
    'm463_runs/m470_h4_m6_tech_v1/attempt.json':'4bec71aac36e1b7e47b13d932e645eba6d2087d3e6ad3a602b501e465eafb1a1',
    'm463_runs/m470_h4_m6_tech_v1/dispatch.started.json':'e84dd853850fb1eba2e8867f4b30c006a4a562d69a3afffad63f21996f3cbdd9',
    'm463_runs/m470_h4_m6_tech_v1/attempt_result.json':'a1219208982cd65237941304893206309fdb8f23ec98248b55578bc5f09d2bc9',
    'm463_runs/m470_h4_m6_2_technical_20261126_001405_1795626845670742063.log':'6858491e87977eb9b206755e5bd86f22e812fca1e626faf5e0bad0f6677ee700',
    'm463_runs/m470_h4_m6_1_full_static_20261126_000627_892520238.log':'4969f7e244cfe2eb37a57dcff0af44b0344f03917591c8c543f8bf9782d9ef93',
}
H4M6_V2_STOP_PINS={
    'm463_runs/m470_h4_m6_tech_v2/report.json':'2d0281c5885c113d748a9947a74a8ccc7bdbc394834a2870163027bf41800ac8',
    'm463_runs/m470_h4_m6_tech_v2/attempt.json':'24c7d103c006322d56d6827ebd4b43fab741991ef1e3121059c9eed8d2048bc1',
    'm463_runs/m470_h4_m6_tech_v2/dispatch.started.json':'662d00152b3eb8a6706d3930239dbaf07f9dd9bbcb9da5c64acf007c8924c04b',
    'm463_runs/m470_h4_m6_tech_v2/attempt_result.json':'b4a27140b5bbf2e4f2d53a4301bcab759d9d610fdc25ad4014c453b2439c3952',
    'm463_runs/m470_h4_m6_2_technical_v2_20261126_002938_1795627778837596800.log':'c19b4cfd76ba5f631464e2f5963fe239209ba05b9585263bd69f4f8f98cbfe30',
    'm463_runs/m470_h4_m6_2r_full_static_20261126_002445_697451410.log':'84675a65e896ec85cc2f9d9cc5a8416cd22a27cbac0aa8b596d9a27f946e81e6',
}
H4M6_ARITH_PINS={
    'm463_runs/m470_h4_m6_arithmetic_v2/report.json':'77af67e65ef8757ba580a42e84b86c0501fd3e912cc74fea93441700e8e9903c',
    'm463_runs/m470_h4_m6_arithmetic_v2/probe.log':'cac36fa7b239b5fc0e74c679f27308f652cfdfc81f9fa0577922f7fac83cacd0',
    'm463_runs/m470_h4_m6_arithmetic_v2/collector.json':'83a6b5c88324752cfbe71100a287884f5298d54a3a5f2510584296df682ff9e4',
    'm463_runs/m470_h4_m6_arithmetic_v2/attempt.json':'b6ce1009a328508cb0542d07410c43d7d3091c1f7a5e9b1f431a360953302ece',
    'm463_runs/m470_h4_m6_arithmetic_v2/dispatch.started.json':'e303b6c9520a9ed0162eef5881fae2cd515f33b8c4989cf71d3dc2aa583fd1a5',
    'm463_runs/m470_h4_m6_arithmetic_v2/attempt_result.json':'2142a867bfdc97633a72486c40acd5647c422ead467365cf9afddecd624752ef',
}
H4M6_ARITH_AUTH={
    'source_sha256':'75edd5c776bb844f3290bded56ebe1a624fdfcd53ef6f77f4521625439135e18',
    'manifest_sha256':'e305539ec45a12a21ef3a136230024a46a7f1828ba13685c2f30baabcc74247b',
    'marker':'H4M6_ARITH_GO: H4-M6-2P-ARITH-V2 m463_runs/m470_h4_m6_arithmetic_v2 75edd5c776bb844f3290bded56ebe1a624fdfcd53ef6f77f4521625439135e18 e305539ec45a12a21ef3a136230024a46a7f1828ba13685c2f30baabcc74247b',
}
H4M6_ARITH_HLO={
    'A':{'pre':'08a834fa394faee94be072c52e0c30485a448ad0fc564102335e02823a7d0c14',
         'optimized':'5e8312e9229082dc4893ee320baa4ab5bf1c74d8d20a65a65f43202535ec7975'},
    'B':{'pre':'9e69034a19ecdfe8606708055228f82eb848a01ce56e608acbf1eb3c761f43c1',
         'optimized':'59b0d2d2ba43d0d90d899f492faf70f5ec949700adb3185ffd485d9595fbef83'},
}
H4M6_DEV_OUTPUT=Path('m463_runs/m470_h4_m6_dev_v1/report.json')
H4M6_RICH_OUTPUT=Path('m463_runs/m470_h4_m6_dev_v1/mechanism.json')
H4M6_TECH_FIXTURES=H4CC_TECH_FIXTURES
H4M6_MECHANISM_LANES=((7601,'right'),(7613,'left'),(7357,'left'),(7331,'left'))

def h4m6_eval_require(ok,label):
    if not bool(ok):
        raise RuntimeError('M470_H4M6_'+label)

def h4m6_eval_plan(phase):
    from m463_push_ppo_train import h4m6_plan
    h4m6_eval_require(phase in ('tech','dev'),'PHASE')
    result=h4m6_plan(phase)
    if phase=='tech':
        result.update(fixtures=[list(f) for f in H4M6_TECH_FIXTURES],
            arms=['ref','rss'],synthetic_prior_event=True,prior_actual_pushes=0,
            optimizer_updates=0,evaluation_transitions=0,checkpoint_saves=0)
    else:
        result.update(seeds=list(H3_EVAL_SEEDS),profiles=list(H3_EVAL_PROFILES),
            directions=['left','right'],policies=['REF6','RSS'],
            push_step=300,push_magnitude=.50,second_push_step=335,
            second_push_magnitude=.35,max_steps=1000,original_reward=True,
            deterministic_actor=True,domain_randomization=False,
            mechanism_lanes=[list(f) for f in H4M6_MECHANISM_LANES],
            mechanism_ticks=[334,385])
    return result


def h4m6_external_sources(root):
    """Read-only exact installed dependencies; no JAX/MJX imports."""
    import m463_push_ppo_train as t
    external=h4m5_external_sources(root)
    paths={}
    for name,blob in {**t.H4C_BRAX_BLOBS,
        'brax.envs.wrappers.training':'2c09c8268a29d14c832dd07050575e4541ee98af',
        'mujoco_playground._src.wrapper':'62e2823b80b692befe29e4640d0b7a816e52271c'}.items():
        p=h4m5_find_source(name)
        h4m6_eval_require(h4m3_git_blob(p)==blob,'EXTERNAL_BLOB '+name)
        paths[name]={'path':str(p),'blob':blob,'sha256':sha256(p)}
    p=h4m5_find_source('brax.training.agents.ppo.train')
    h4m6_eval_require(sha256(p)==TRAIN_SHA,'TRAIN_PIN')
    paths['brax.training.agents.ppo.train']={'path':str(p),'sha256':sha256(p)}
    h4m3_no_runtime_imports()
    return {'common':external,'training':paths}

def h4m6_native_factor(k,eligible,np):
    """Pinned CUDA arithmetic reference; not a change to native reward."""
    return np.float32(1)-np.float32(.01)*np.float32(51-k) if eligible else np.float32(1)


def h4m6_read_arithmetic(root):
    """Read immutable accepted arithmetic evidence, never execute the probe."""
    import numpy as np
    root=Path(root).resolve()
    req=h4m6_eval_require
    pins={**H4M6_V1_STOP_PINS,**H4M6_V2_STOP_PINS,**H4M6_ARITH_PINS}
    req(len(pins)==18,'ARITH_PIN_COUNT')
    blobs={}
    for rel,expected in pins.items():
        p=root/rel
        req(p.is_file() and not p.is_symlink() and p.absolute()==p.resolve()
            and 0<p.stat().st_size<=16*1024**2,'ARITH_PATH '+rel)
        raw=p.read_bytes()
        req(0<len(raw)<=16*1024**2 and hashlib.sha256(raw).hexdigest()==expected,'ARITH_SHA '+rel)
        blobs[rel]=raw
    old=root/'m463_runs/m470_h4_m6_arithmetic_v1'
    req(not old.exists() and not old.is_symlink(),'ARITH_V1_RESERVATION')
    prefix='m463_runs/m470_h4_m6_arithmetic_v2/'
    def read(name): return json.loads(blobs[prefix+name])
    r=read('report.json'); a=r['authorization']; m=r['manifest']; c=read('collector.json')
    req(a==H4M6_ARITH_AUTH and h4m3_digest(m)==a['manifest_sha256']
        and m['archive_sha256']=={**H4M6_V1_STOP_PINS,**H4M6_V2_STOP_PINS}
        and r['provenance']['archive_sha256']==m['archive_sha256']
        and r['provenance']['external_sha256']==m['external']
        and h4m3_digest(r['provenance']['common_map'])==m['common'],'ARITH_PROVENANCE')
    req(read('attempt.json')=={'authorization':a,'reserved':True}
        and read('dispatch.started.json')=={'authorization':a}
        and read('attempt_result.json')=={'authorization':a,'status':'COMPLETE'},'ARITH_MARKERS')
    log=blobs[prefix+'probe.log']
    req(c=={'authorization':a,'decision':'ARITHMETIC_CONFORMANCE_PASS','producer_rc':0,
        'overflow':False,'error':None,'total_bytes':len(log),'stored_bytes':len(log)},'ARITH_COLLECTOR')
    req(r['schema']=='H4M6_2P_ARITH_REPORT_V1' and r['status']=='COMPLETE'
        and r['passed'] is True and r['post_immutability'] is True
        and r['attempted']==r['recorded']=={'A':4896,'B':2512}
        and r['compiled_signatures']==2 and r['archived_rows_reproduced']==6
        and r['host_factor_negative_rejections']==104
        and r['zero']==dict.fromkeys(('robot_resets','actor_calls','physical_transitions',
            'optimizer_updates','evaluation_transitions','checkpoint_saves'),0)
        and r['compiler']['jax']==r['compiler']['jaxlib']=='0.4.35'
        and r['compiler']['device']=='cuda:0' and not r['gpu']['errors']
        and 0<r['gpu']['peak_used_mib']<.95*12288 and r['gpu']['samples']>0,'ARITH_CONTRACT')
    for graph,hs in H4M6_ARITH_HLO.items():
        for key,h in hs.items():
            req(r['hlo'][graph][key+'_sha256']==h
                and hashlib.sha256(r['hlo'][graph][key].encode()).hexdigest()==h,'ARITH_HLO')
    lines=log.decode('utf8').splitlines(); rows=r['rows']
    req(len(rows)==7408 and len(lines)==14817 and lines[-1]=='ARITHMETIC_CONFORMANCE_PASS','ARITH_LEDGER')
    counts={'A':0,'B':0}; cases={'A':[],'B':[]}; seen={'A':set(),'B':set()}
    for j,row in enumerate(rows):
        graph=row['graph']; case=row['case']; counts[graph]+=1; cases[graph].append(case)
        head=lines[2*j].split(' ',3)
        req(head[:3]==['ATTEMPT',graph,str(counts[graph])] and json.loads(head[3])==case
            and lines[2*j+1].startswith('ROW ') and json.loads(lines[2*j+1][4:])==row,'ARITH_LOG_ROW')
        k=case[1]; eligible=case[2] and not case[3]
        req(row['eligible']==eligible and row['bits']==row['reference']
            and row['bits'][0]==f'{h4m6_native_factor(k,eligible,np).view(np.uint32).item():08x}','ARITH_FACTOR')
        if eligible: seen[graph].add(k)
    req(counts==r['recorded'] and all(seen[g]==set(range(1,51)) for g in counts)
        and all(h4m3_digest(cases[g])==r['case_manifest'][g+'_sha256'] for g in counts),'ARITH_DOMAIN')
    return {'artifact_sha256':dict(pins),'authorization':dict(a),'hlo_sha256':H4M6_ARITH_HLO,
            'factor_contract':'rounded_reciprocal_float32_v1'}


# Isolated H4-M6-2D1 diagnostic. No native execution without independent Rules GO.
H4M6_DIAG_OUTPUT=Path('m463_runs/m470_h4_m6_state_diag_v1/report.json')
H4M6_DIAG_ORDER=('REF0','RSS0','RSS1','REF1')
H4M6_DIAG_SPEC={'id':'H4-M6-2D1-V1','fixture':[4102,3,3,295,327,335,335],
    'order':list(H4M6_DIAG_ORDER),'resets':4,'physical':32,'ticks':8,
    'snapshot_cap':8*1024**2,'host_cap':256*1024**2,'report_cap':16*1024**2,
    'log_cap':16*1024**2,'witness_cap':16,'historical_physical':171,
    'actor_calls':32,'key_audits':32,'template_initializations':2,'archive_count':24}
H4M6_DIAG_V3_PINS={
    'm463_runs/m470_h4_m6_tech_v3/report.json':'98f5fdaffb8a8a85bc4c78f024235f9f59d7acb8818c34a2adf869d68de09d3a',
    'm463_runs/m470_h4_m6_tech_v3/attempt.json':'ee4b9930bcf769dbdc9ef5f46bc04271d4ef75739ba0d20eea6595fa6bc1832a',
    'm463_runs/m470_h4_m6_tech_v3/dispatch.started.json':'cb0ada55ee06fd36c71700f1b28a82f49ad7b80229c8ac703d4e27078533b08e',
    'm463_runs/m470_h4_m6_tech_v3/attempt_result.json':'afd6a0d321e97e04bb34f66f53b31c37a386459d0d2a715d88c4fd9894922eb5',
    'm463_runs/m470_h4_m6_2_technical_v3_20261126_074552_1795653952294940231.log':'0e5de852b4afcb63e3eefe884c797ac9ef695f0fd47fa31ba82a172f50a57ad3',
    'm463_runs/m470_h4_m6_2r2_full_static_20261126_073654_707259590.log':'7e65ed7cd72962e6ab7b27ae53a2848f705d40dc2974a1e0c4dfd4ae4c176b2f',
}
H4M6_DIAG_MJX_BLOBS={'types':'16fa3a8142e35ca0353859b19bc3c0270d05730f',
    'forward':'07f33a90c8ed68bc545ba47bfb78d4bfb8db1f52',
    'dataclasses':'0db4a1419b364030358a6f24ce5ad602aab0d501',
    'solver':'6d3d4f212223afeca8a0020f9e9c9c29855cd19b',
    'constraint':'e778bcd8e5076aa3a9e49ae9b8f97819d74ea884',
    'collision_driver':'7b947ce103ac76b7f6d6a72b760119f1814a28b8'}


def h4m6_diag_authorization(root=None):
    import re
    import m463_push_ppo_train as t
    root=Path(__file__).resolve().parent if root is None else Path(root).resolve()
    lines=(root/'G1_RESEARCH_RULES.md').read_text('utf8').splitlines()
    active=[x for x in lines if x.startswith('H4M6_DIAG_GO:')]
    static=[x for x in lines if x.startswith('H4M6_DIAG_STATIC_APPROVED:')]
    h4m6_eval_require(len(active)==len(static)==1,'DIAG_EXECUTION_HOLD')
    h4m6_eval_require(not any(re.match(r'^(?:[A-Z0-9_]*ACTIVE_GO|H4M6_ARITH_GO):',x)
        for x in lines),'DIAG_OTHER_GO')
    fields=active[0].split()[1:]
    h4m6_eval_require(len(fields)==6 and fields==static[0].split()[1:]
        and fields[0]==H4M6_DIAG_SPEC['id'] and fields[1]==str(H4M6_DIAG_OUTPUT.parent)
        and all(re.fullmatch('[0-9a-f]{64}',x) for x in fields[2:5])
        and re.fullmatch(r'm463_runs/m470_h4_m6_2d1_full_static_[0-9_]+\.log',fields[5]),'DIAG_AUTH')
    _,out,source,manifest,static_sha,static_log=fields
    h4m6_eval_require(source==t.h4m6_source_identity(root)['sha256']
        and manifest==h4m3_digest(H4M6_DIAG_SPEC)
        and t.h4m6_spec_digest()=='65371befdc4cb7265b4ee4e6366fbf94ea1e5478cd6117ed190c492e340a9ae5','DIAG_AUTH_BINDING')
    p=root/static_log
    h4m6_eval_require(p.is_file() and not p.is_symlink() and p.absolute()==p.resolve()
        and 0<p.stat().st_size<=H4M6_DIAG_SPEC['log_cap'] and sha256(p)==static_sha,'DIAG_STATIC')
    rows=p.read_text('utf8').splitlines()
    for k,v in [('SOURCE',source),('MANIFEST',manifest),
        ('SPEC','65371befdc4cb7265b4ee4e6366fbf94ea1e5478cd6117ed190c492e340a9ae5'),
        ('EXTERNAL','97e1686ac016de832166f26c7287992f0b6536a27c504ee6451a5d472597e7da'),
        ('GENERATED','7e2010e84e888e4bfa1622c8eceea86e69b1767d9a89a323efec971611f71b5a'),
        ('V3_REPORT',H4M6_DIAG_V3_PINS['m463_runs/m470_h4_m6_tech_v3/report.json'])]:
        marker='H4_M6_2D1_'+k+'_SHA256 '
        h4m6_eval_require([x for x in rows if x.startswith(marker)]==[marker+v],'DIAG_STATIC_MARKER '+k)
    h4m6_eval_require(not any((t.H4_EXECUTION_AUTHORIZED,t.H4R_EXECUTION_AUTHORIZED,
        t.H4C_EXECUTION_AUTHORIZED,t.H4CC_EXECUTION_AUTHORIZED,H4_EVAL_EXECUTION_AUTHORIZED,
        H4CC_EVAL_EXECUTION_AUTHORIZED,H4CC_TECH_EXECUTION_AUTHORIZED)),'DIAG_LEGACY_HOLD')
    return {'id':t.H4M6_ID,'diagnostic_id':fields[0],'phase':'diag','output':out,
        'source_sha256':source,'manifest_sha256':manifest,'static_sha256':static_sha,
        'static_log':static_log,'spec_sha256':'65371befdc4cb7265b4ee4e6366fbf94ea1e5478cd6117ed190c492e340a9ae5',
        'external_sha256':'97e1686ac016de832166f26c7287992f0b6536a27c504ee6451a5d472597e7da',
        'generated_sha256':'7e2010e84e888e4bfa1622c8eceea86e69b1767d9a89a323efec971611f71b5a'}


def h4m6_diag_archives(root,arithmetic):
    pins=dict(arithmetic['artifact_sha256'],**H4M6_DIAG_V3_PINS)
    h4m6_eval_require(len(pins)==24,'DIAG_ARCHIVE_COUNT')
    for rel,h in H4M6_DIAG_V3_PINS.items():
        p=root/rel
        h4m6_eval_require(p.is_file() and not p.is_symlink() and p.absolute()==p.resolve()
            and 0<p.stat().st_size<=16*1024**2 and sha256(p)==h,'DIAG_ARCHIVE '+rel)
    p=root/'m463_runs/m470_h4_m6_tech_v3'
    v=json.loads((p/'report.json').read_bytes());a=v['authorization']
    h4m6_eval_require(v['status']=='M470_H4M6_TECH_STOP'
        and v['physical_transitions_attempted']==v['physical_transitions_recorded']==164
        and v['reset_lanes_attempted']==v['reset_lanes_recorded']==4
        and all(v[k]==0 for k in ('optimizer_updates','evaluation_transitions','checkpoint_saves'))
        and json.loads((p/'attempt.json').read_bytes())=={'authorization':a,'reserved':True}
        and json.loads((p/'dispatch.started.json').read_bytes())=={'authorization':a}
        and json.loads((p/'attempt_result.json').read_bytes())=={'authorization':a,'status':'STOPPED'},
        'DIAG_ARCHIVE_LEDGER')
    return pins


def h4m6_diag_snapshot(pairs,treedef,host,cap):
    """Pure host capture of existing tree leaves, matching digest_tree exactly."""
    import numpy as np
    structure=str(treedef).encode(); size=len(structure)
    h=hashlib.sha256(structure); result=[]; raw=[]
    for key,x in pairs:
        a=np.asarray(host(x)); path=str(key); dtype=a.dtype.str; shape=list(a.shape)
        h4m6_eval_require(not a.dtype.hasobject and a.dtype.fields is None,'DIAG_LEAF_DTYPE')
        size+=a.nbytes+len(path.encode())+len(dtype)+len(str(a.shape))
        h4m6_eval_require(size<=cap,'DIAG_SNAPSHOT_CAP')
        b=a.tobytes()
        h.update(path.encode());h.update(dtype.encode());h.update(str(a.shape).encode());h.update(b)
        result.append({'path':path,'dtype':dtype,'shape':shape,'nbytes':len(b),
            'sha256':hashlib.sha256(b).hexdigest()})
        raw.append(b)
    h4m6_eval_require(size<=cap,'DIAG_SNAPSHOT_CAP')
    return {'aggregate':h.hexdigest(),'treedef':structure.decode(),
        'treedef_sha256':hashlib.sha256(structure).hexdigest(),'leaves':result,'bytes':size},raw


def h4m6_diag_compare(a,ar,b,br,limit):
    """Compare bytes, not just hashes. Witness order is traversal, not causal time."""
    import numpy as np
    differences=[]; count=0
    def add(w):
        nonlocal count
        count+=1
        if len(differences)<limit:differences.append(w)
    if a['treedef']!=b['treedef']:
        x=a['treedef'].encode();y=b['treedef'].encode()
        i=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)))
        add({'kind':'TREEDEF','byte_offset':i,'a_hex':x[i:i+16].hex(),'b_hex':y[i:i+16].hex()})
    if len(ar)!=len(br):
        add({'kind':'LEAF_COUNT','a':len(ar),'b':len(br)})
    for i,(x,y,u,v) in enumerate(zip(a['leaves'],b['leaves'],ar,br)):
        if (x['path'],x['dtype'],x['shape'],x['nbytes'])!=(y['path'],y['dtype'],y['shape'],y['nbytes']):
            add({'kind':'LEAF_SCHEMA','index':i,'a':x,'b':y});continue
        if u==v:continue
        offset=next(j for j,(c,d) in enumerate(zip(u,v)) if c!=d)
        dt=np.dtype(x['dtype']); idx=offset//dt.itemsize
        ub=u[idx*dt.itemsize:(idx+1)*dt.itemsize];vb=v[idx*dt.itemsize:(idx+1)*dt.itemsize]
        kind='RAW_BYTES'
        if dt.kind=='f' and np.frombuffer(ub,dtype=dt)[0]==0 and np.frombuffer(vb,dtype=dt)[0]==0:
            kind='SIGNED_ZERO'
        add({'kind':kind,'path':x['path'],'dtype':x['dtype'],'shape':x['shape'],
            'element_index':list(map(int,np.unravel_index(idx,x['shape']))) if x['shape'] else [],
            'byte_offset':offset,'differing_bytes':sum(c!=d for c,d in zip(u,v)),
            'a_hex':ub.hex(),'b_hex':vb.hex()})
    h4m6_eval_require(bool(count)==(a['aggregate']!=b['aggregate']),'DIAG_AGGREGATE_WITNESS')
    return {'different':bool(count),'mismatches':count,'witnesses':differences,'truncated':count>len(differences)}


def h4m6_diag_capture(r,cache,lane,tick,trees,flatten,host,expected):
    """Only called after reset/step returned; no graph instrumentation."""
    key=lane['diag_id']+':'+str(tick)
    h4m6_eval_require(key not in cache,'DIAG_DUPLICATE_SNAPSHOT')
    views={}; raw={}; size=0
    for name,tree in trees.items():
        pairs,td=flatten(tree)
        meta,data=h4m6_diag_snapshot(pairs,td,host,H4M6_DIAG_SPEC['snapshot_cap']-size)
        size+=meta['bytes'];views[name]=meta;raw[name]=data
    h4m6_eval_require(all(views[n]['aggregate']==h for n,h in expected.items()),'DIAG_ORIGINAL_HASH')
    total=r['snapshot_bytes']+size
    h4m6_eval_require(total<=H4M6_DIAG_SPEC['host_cap'],'DIAG_HOST_CAP')
    if not cache:
        h4m6_eval_require(size*36<=H4M6_DIAG_SPEC['host_cap'],'DIAG_SCHEMA_ESTIMATE')
    before=(len(r['leaf_schema']),len(r['treedef_table']),len(r['comparisons']))
    schemas={json.dumps(v,sort_keys=True):i for i,v in enumerate(r['leaf_schema'])}
    tdefs={v['text']:i for i,v in enumerate(r['treedef_table'])}
    packed={}
    for name,v in views.items():
        td=v['treedef']
        if td not in tdefs:
            tdefs[td]=len(r['treedef_table']);r['treedef_table'].append({'text':td,'sha256':v['treedef_sha256']})
        leaves=[]
        for leaf in v['leaves']:
            schema={k:x for k,x in leaf.items() if k!='sha256'}; sk=json.dumps(schema,sort_keys=True)
            if sk not in schemas:
                schemas[sk]=len(r['leaf_schema']);r['leaf_schema'].append(schema)
            leaves.append([schemas[sk],leaf['sha256']])
        packed[name]={'aggregate':v['aggregate'],'treedef_id':tdefs[td],'leaves':leaves,'bytes':v['bytes']}
    if not cache:
        estimate=len(json.dumps(packed,separators=(',',':')))*36+len(json.dumps(r['leaf_schema']))+len(json.dumps(r['treedef_table']))
        h4m6_eval_require(estimate<12*1024**2,'DIAG_REPORT_ESTIMATE')
    stop=False
    for prev in r['lanes']:
        if prev is lane:break
        other=prev['diag_id']+':'+str(tick)
        if other not in cache:continue
        old_views,old_raw=cache[other]
        same=prev['arm']==lane['arm']; names=('full','projected','physics') if same else ('projected','physics')
        entry={'a':prev['diag_id'],'b':lane['diag_id'],'tick':tick,'same_arm':same,'views':{}}
        remaining=H4M6_DIAG_SPEC['witness_cap']
        for name in names:
            out=h4m6_diag_compare(old_views[name],old_raw[name],views[name],raw[name],remaining)
            remaining-=len(out['witnesses']);entry['views'][name]=out
        if tick==-1:
            fields=('initial_sha256','initial_rng','model_sha256','native_reset_command','initial_command')
            entry['row_fields']=[n for n in fields if prev[n]!=lane[n]]
        else:
            x=prev['rows'][tick];y=lane['rows'][tick]
            entry['row_fields']=[n for n in x if json.dumps(x[n],sort_keys=True)!=json.dumps(y[n],sort_keys=True)]
        entry['different']=any(v['different'] for v in entry['views'].values()) or bool(entry['row_fields'])
        r['comparisons'].append(entry)
        stop=stop or (entry['different'] and (prev['diag_id']=='REF0' or same))
    r['snapshots'][key]=packed;r['snapshot_bytes']=total
    if len(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode())>14*1024**2:
        del r['snapshots'][key];r['snapshot_bytes']-=size
        del r['leaf_schema'][before[0]:];del r['treedef_table'][before[1]:];del r['comparisons'][before[2]:]
        h4m6_eval_require(False,'DIAG_REPORT_HEADROOM')
    cache[key]=(views,raw)
    return stop


def h4m6_diag_ledger(r,*,partial=False):
    audit=h4m6_technical_audit(r,partial=True,diagnostic=True)
    req=h4m6_eval_require
    req(r['diag_spec']==H4M6_DIAG_SPEC and r['passed'] is False
        and r['physical_transitions_attempted']<=32 and r['reset_lanes_attempted']<=4
        and r['snapshot_bytes']<=H4M6_DIAG_SPEC['host_cap'],'DIAG_LEDGER')
    req(all(l['diag_id']==H4M6_DIAG_ORDER[i] and len(l['rows'])<=8 for i,l in enumerate(r['lanes'])),'DIAG_ORDER')
    req(0<=r['key_audits_recorded']<=r['key_audits_attempted']<=32
        and r['actor_calls_attempted']==r['physical_transitions_attempted']
        and r['actor_calls_recorded']==r['physical_transitions_recorded']
        and 0<=r['template_initializations_recorded']<=r['template_initializations_attempted']<=2,'DIAG_CALLS')
    if not partial:
        req(len(r['lanes'])==4 and audit['outstanding_reset']==audit['outstanding_step']==0
            and r['key_audits_attempted']==r['key_audits_recorded']==r['physical_transitions_recorded']
            and r['actor_calls_recorded']==r['physical_transitions_recorded']
            and r['template_initializations_attempted']==r['template_initializations_recorded']==2,'DIAG_COMPLETE')
        req(all(l.get('endpoint') in ('DIVERGENCE','HORIZON','TERMINAL') and
            (l['endpoint']!='HORIZON' or len(l['rows'])==8)
            and all(l['diag_id']+':'+str(k) in r['snapshots'] for k in range(-1,len(l['rows'])))
            for l in r['lanes']),'DIAG_ENDPOINT')
        expected=sum(1+min(len(a['rows']),len(b['rows'])) for j,b in enumerate(r['lanes']) for a in r['lanes'][:j])
        req(len(r['comparisons'])==expected,'DIAG_COMPARISONS')
    same=any(x['different'] and x['same_arm'] for x in r['comparisons'])
    cross=any(x['different'] and not x['same_arm'] for x in r['comparisons'])
    return dict(audit,diagnostic_complete=not partial,classification=(
        'SAME_ARM_DIVERGENCE_OBSERVED' if same else 'CROSS_ARM_ONLY_DIVERGENCE' if cross else 'NOT_REPRODUCED'))


def h4m6_diag_report_replay(r):
    """Strict host replay of completed diagnostic evidence, never a technical PASS."""
    import re
    import numpy as np
    req=h4m6_eval_require
    issha=lambda x:isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None
    audit=h4m6_diag_ledger(r)
    req(r['audit']==audit and r['status']=='M470_H4M6_DIAG_COMPLETE'
        and r['post_immutability'] is True and r['passed'] is False
        and r['policy_sha256_before']==r['policy_sha256_after']
        and issha(r['policy_sha256_before']) and len(r['provenance']['diagnostic_archive_sha256'])==24
        and r['further_training_authorized'] is False,'DIAG_FINAL_REPORT')
    schemas=r['leaf_schema']; tdefs=r['treedef_table']
    req(len({json.dumps(x,sort_keys=True) for x in schemas})==len(schemas)
        and len({x['text'] for x in tdefs})==len(tdefs),'DIAG_TABLE_UNIQUENESS')
    for x in schemas:
        dt=np.dtype(x['dtype']); shape=x['shape']
        req(isinstance(x['path'],str) and all(type(n) is int and n>=0 for n in shape)
            and not dt.hasobject and dt.fields is None
            and math.prod(shape)*dt.itemsize==x['nbytes'],'DIAG_SCHEMA_REPLAY')
    for x in tdefs:
        req(hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],'DIAG_TREEDEF_REPLAY')
    lane_map={l['diag_id']:l for l in r['lanes']}
    snapshots=r['snapshots'];expected_keys={l['diag_id']+':'+str(k) for l in r['lanes'] for k in range(-1,len(l['rows']))}
    req(set(snapshots)==expected_keys,'DIAG_SNAPSHOT_KEYS')
    total=0
    for key,views in snapshots.items():
        label,k=key.split(':');tick=int(k);lane=lane_map[label];size=0
        req(set(views)=={'full','projected','physics'},'DIAG_VIEWS')
        for v in views.values():
            req(issha(v['aggregate']) and type(v['treedef_id']) is int and 0<=v['treedef_id']<len(tdefs),'DIAG_PACKED_TREE')
            n=len(tdefs[v['treedef_id']]['text'].encode())
            for index,h in v['leaves']:
                req(type(index) is int and 0<=index<len(schemas) and issha(h),'DIAG_PACKED_LEAF')
                x=schemas[index];n+=x['nbytes']+len(x['path'].encode())+len(x['dtype'])+len(str(tuple(x['shape'])))
            req(n==v['bytes'],'DIAG_BYTES_REPLAY');size+=n
        req(size<=H4M6_DIAG_SPEC['snapshot_cap'],'DIAG_SNAPSHOT_REPLAY');total+=size
        if tick==-1:
            req(views['projected']['aggregate']==lane['initial_sha256'],'DIAG_RESET_HASH')
        else:
            row=lane['rows'][tick]
            req(views['projected']['aggregate']==row['post_sha256']
                and views['physics']['aggregate']==row['physics_sha256']
                and snapshots[label+':'+str(tick-1)]['projected']['aggregate']==row['pre_sha256'],'DIAG_ROW_HASH')
    req(total==r['snapshot_bytes']<=H4M6_DIAG_SPEC['host_cap'],'DIAG_TOTAL_BYTES')
    expected={(a['diag_id'],b['diag_id'],k) for j,b in enumerate(r['lanes'])
        for a in r['lanes'][:j] for k in range(-1,min(len(a['rows']),len(b['rows'])))}
    observed=set()
    for e in r['comparisons']:
        a,b,k=e['a'],e['b'],e['tick'];key=(a,b,k)
        req(key in expected and key not in observed,'DIAG_PAIR_KEYS');observed.add(key)
        la,lb=lane_map[a],lane_map[b];same=la['arm']==lb['arm']
        req(e['same_arm']==same and set(e['views'])==set(('full','projected','physics') if same else ('projected','physics')),'DIAG_PAIR_VIEWS')
        aa=snapshots[a+':'+str(k)];bb=snapshots[b+':'+str(k)]
        used=0
        for n,v in e['views'].items():
            x,y=aa[n],bb[n]
            count=int(x['treedef_id']!=y['treedef_id'])+int(len(x['leaves'])!=len(y['leaves']))
            count+=sum(u!=w for u,w in zip(x['leaves'],y['leaves']))
            req(v['mismatches']==count and v['different']==bool(count)==(x['aggregate']!=y['aggregate'])
                and v['truncated']==(count>len(v['witnesses']))
                and len(v['witnesses'])<=count,'DIAG_PAIR_REPLAY')
            for w in v['witnesses']:
                if w['kind'] in ('RAW_BYTES','SIGNED_ZERO'):
                    dt=np.dtype(w['dtype']);u=bytes.fromhex(w['a_hex']);z=bytes.fromhex(w['b_hex'])
                    req(len(u)==len(z)==dt.itemsize and u!=z and 0<w['differing_bytes']
                        and (w['kind']!='SIGNED_ZERO' or (dt.kind=='f' and np.frombuffer(u,dtype=dt)[0]==0
                        and np.frombuffer(z,dtype=dt)[0]==0)),'DIAG_WITNESS_REPLAY')
                else:req(w['kind'] in ('TREEDEF','LEAF_COUNT','LEAF_SCHEMA'),'DIAG_WITNESS_KIND')
            used+=len(v['witnesses'])
        req(used<=H4M6_DIAG_SPEC['witness_cap'],'DIAG_WITNESS_CAP')
        if k==-1:
            fields=[n for n in ('initial_sha256','initial_rng','model_sha256','native_reset_command','initial_command') if la[n]!=lb[n]]
        else:
            x,y=la['rows'][k],lb['rows'][k]
            fields=[n for n in x if json.dumps(x[n],sort_keys=True)!=json.dumps(y[n],sort_keys=True)]
        req(set(e['row_fields'])==set(fields) and len(e['row_fields'])==len(fields)
            and e['different']==(bool(fields) or any(v['different'] for v in e['views'].values())),'DIAG_PAIR_FIELDS')
    req(observed==expected,'DIAG_PAIR_COVERAGE')
    for l in r['lanes']:
        label=l['diag_id']; end=len(l['rows'])-1
        endpoints=[e['tick'] for e in r['comparisons'] if e['b']==label and e['different']
            and (e['a']=='REF0' or e['same_arm'])]
        req((not endpoints and l['endpoint']!='DIVERGENCE') or
            (endpoints and min(endpoints)==end and l['endpoint']=='DIVERGENCE'),'DIAG_ENDPOINT_REPLAY')
        req(l['endpoint']!='TERMINAL' or (bool(l['rows']) and l['rows'][-1]['metrics']['terminal']),'DIAG_TERMINAL_ENDPOINT')
    return audit


def h4m6_diag_dispatch(args):
    """Bounded parent collector; only the reserved child may enter native code."""
    import signal
    root=Path(__file__).resolve().parent
    auth=h4m6_diag_authorization(root)
    path=root/H4M6_DIAG_OUTPUT
    req=h4m6_eval_require
    req(args.mode=='run' and args.h4m6_phase=='diag'
        and args.output.absolute()==args.output.resolve()==path
        and all(isinstance(x,Path) for x in (args.checkpoint_c,args.h4_reward_report,args.h4cc_restore_report))
        and os.environ.get('JAX_PLATFORMS')=='cuda'
        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
        and os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')=='false','DIAG_PREFLIGHT')
    if getattr(args,'h4m6_diag_child',False):
        return h4m6_technical_dispatch(args,diagnostic=True)
    provenance=h4m6_tech_inputs(args,auth,diagnostic=True)
    h4m3_no_runtime_imports()
    h4m6_reserve_tech(path,auth)
    h4m6_write_bounded(path.parent/'collector.started.json',{'authorization':auth,'pid':os.getpid()},exclusive=True)
    log=path.parent/'probe.log';cap=H4M6_DIAG_SPEC['log_cap']
    cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--mode','run',
        '--h4m6-phase','diag','--h4m6-diag-child','--output',str(path),
        '--checkpoint-c',str(args.checkpoint_c),'--h4-reward-report',str(args.h4_reward_report),
        '--h4cc-restore-report',str(args.h4cc_restore_report)]
    total=stored=0;overflow=False;rc=None;error=None;proc=None;decision='STOP'
    def kill():
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    try:
        with log.open('xb') as f:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                data=proc.stdout.read1(65536)
                if not data:break
                total+=len(data);chunk=data[:max(0,cap-stored)];f.write(chunk);stored+=len(chunk)
                if total>cap and not overflow:overflow=True;kill()
            rc=proc.wait();f.flush();os.fsync(f.fileno())
        req(rc==0 and not overflow,'DIAG_BOUNDED_PROCESS')
        req(h4m6_diag_authorization(root)==auth and h4m6_tech_inputs(args,auth,diagnostic=True)==provenance,'DIAG_PARENT_POST')
        req(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=H4M6_DIAG_SPEC['report_cap'],'DIAG_REPORT')
        report=json.loads(path.read_bytes())
        req(report['authorization']==auth and report['provenance']==provenance
            and json.loads((path.parent/'attempt.json').read_bytes())=={'authorization':auth,'reserved':True}
            and json.loads((path.parent/'dispatch.started.json').read_bytes())=={'authorization':auth}
            and json.loads((path.parent/'attempt_result.json').read_bytes())=={'authorization':auth,'status':'COMPLETE'},'DIAG_REPORT_PROVENANCE')
        audit=h4m6_diag_report_replay(report)
        decision='DIAGNOSTIC_COMPLETE'
    except BaseException as exc:
        error=(type(exc).__name__+': '+str(exc))[:4096];kill()
        if proc is not None:rc=proc.wait()
    status={'decision':decision,'producer_rc':rc,'overflow':overflow,'total_bytes':total,
        'stored_bytes':stored,'error':error,'authorization':auth}
    h4m6_write_bounded(path.parent/'collector.json',status,exclusive=True)
    print('BOUNDED_LOG_STATUS',json.dumps(status,sort_keys=True),flush=True)
    for p in sorted(path.parent.iterdir()):
        if p.is_file() and not p.is_symlink():print('ARTIFACT_SHA256',sha256(p),p,flush=True)
    req(decision=='DIAGNOSTIC_COMPLETE','DIAG_COLLECTOR_STOP')


def h4m6_tech_inputs(args,auth,*,diagnostic=False):
    import m463_push_ppo_train as t
    from types import SimpleNamespace
    root=Path(__file__).resolve().parent
    paths={'C401M':args.checkpoint_c,'reward':args.h4_reward_report,
           'restore':args.h4cc_restore_report,'static':root/auth['static_log']}
    expected={'C401M':BASE_SHA,'reward':t.H4R_REWARD_SHA,
              'restore':t.H3_CAUSAL_RESTORE_SHA,'static':auth['static_sha256']}
    for k,p in paths.items():
        t.h4x_file_pin(p,expected[k])
    arithmetic=h4m6_read_arithmetic(root)
    h4m6_eval_require(auth['phase']==('diag' if diagnostic else 'tech') and args.output.resolve()==root/(H4M6_DIAG_OUTPUT if diagnostic else H4M6_TECH_OUTPUT),'TECH_PATH')
    diag_pins=h4m6_diag_archives(root,arithmetic) if diagnostic else None
    pins=t.h4x_bootstrap(SimpleNamespace(checkpoint=paths['C401M'],
        restore_report=paths['restore'],h4_exploratory_reward_report=paths['reward']),
        h4cc=True,h4m6=auth)
    external=h4m6_external_sources(root)
    h4m6_eval_require(h4m3_digest(external)==auth['external_sha256'],'EXTERNAL_PIN')
    from m463_instrument_brax import make_h4m6_source
    gen,original=make_h4m6_source(Path(external['training']['brax.training.agents.ppo.train']['path']).read_bytes())
    h4m6_eval_require(original==TRAIN_SHA and hashlib.sha256(gen.encode()).hexdigest()==auth['generated_sha256'],'GENERATED_PIN')
    result={'input_sha256':expected,'input_paths':{k:str(p.resolve()) for k,p in paths.items()},
            'bootstrap':pins,'source_sha256':auth['source_sha256'],'external':external,
            'v1_stop_sha256':dict(H4M6_V1_STOP_PINS),'reward_scale_contract':'native_bool_float32_v2',
            'arithmetic_evidence':arithmetic}
    if diagnostic:
        extra={}
        for n,b in H4M6_DIAG_MJX_BLOBS.items():
            p=h4m5_find_source('mujoco.mjx._src.'+n)
            h4m6_eval_require(h4m3_git_blob(p)==b,'DIAG_MJX_BLOB '+n)
            extra[n]={'path':str(p),'blob':b,'sha256':sha256(p)}
        result.update(diagnostic_archive_sha256=diag_pins,diagnostic_mjx=extra)
    return result

def h4m6_runtime_gate(phase):
    # No source flag; current Rules has neither approval nor active marker.
    import m463_push_ppo_train as t
    h4m6_eval_require(phase in ('tech','dev'),'RUNTIME_PHASE')
    return t.h4m6_authorization(phase=phase)

def h4m6_write_bounded(path,value,*,exclusive=False,cap=16*1024**2):
    """Atomic bounded JSON; reject aliases and partial temporary-file reuse."""
    import shutil
    path=Path(path)
    h4m6_eval_require(path.absolute()==path.resolve() and not path.is_symlink()
        and path.parent.is_dir() and not path.parent.is_symlink(),'REPORT_PATH')
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
    h4m6_eval_require(0<len(raw)<=cap and shutil.disk_usage(path.parent).free>=2*1024**3+len(raw),
        'REPORT_STORAGE')
    if exclusive:
        with path.open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
    else:
        h4m6_eval_require(path.is_file() and not path.is_symlink(),'REPORT_MISSING')
        tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
    return len(raw)

def h4m6_reserve_tech(path,auth):
    import shutil
    path=Path(path)
    h4m6_eval_require(path.absolute()==path.resolve() and path.parent.parent.is_dir()
        and path.parent.parent.resolve()==path.parent.parent.absolute()
        and not path.parent.exists() and not path.parent.is_symlink()
        and shutil.disk_usage(path.parent.parent).free>=2*1024**3+32*1024**2,
        'TECH_RESERVATION')
    path.parent.mkdir(exist_ok=False)
    h4m6_write_bounded(path.parent/'attempt.json',{'authorization':auth,'reserved':True},exclusive=True)

def h4m6_technical_dispatch(args,*,diagnostic=False):
    if not diagnostic:h4m6_runtime_gate('tech')
    import m463_push_ppo_train as t
    root=Path(__file__).resolve().parent
    auth=h4m6_diag_authorization(root) if diagnostic else t.h4m6_authorization(root,phase='tech')
    path=root/(H4M6_DIAG_OUTPUT if diagnostic else H4M6_TECH_OUTPUT)
    h4m6_eval_require(args.mode=='run' and args.h4m6_phase==('diag' if diagnostic else 'tech')
        and args.output.absolute()==args.output.resolve()==path
        and os.environ.get('JAX_PLATFORMS')=='cuda'
        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
        and os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')=='false','TECH_PREFLIGHT')
    provenance=h4m6_tech_inputs(args,auth,diagnostic=diagnostic)
    h4m3_no_runtime_imports()
    if diagnostic:
        marker=path.parent/'collector.started.json'
        h4m6_eval_require(getattr(args,'h4m6_diag_child',False) and marker.is_file() and not marker.is_symlink()
            and json.loads(marker.read_bytes())=={'authorization':auth,'pid':os.getppid()}
            and (path.parent/'probe.log').is_file() and not (path.parent/'probe.log').is_symlink()
            and json.loads((path.parent/'attempt.json').read_bytes())=={'authorization':auth,'reserved':True},'DIAG_PARENT')
    else:h4m6_reserve_tech(path,auth)
    r={'status':'M470_H4M6_TECH_IN_PROGRESS','spec':t.H4M6_SPEC,
        'authorization':auth,'provenance':provenance,'source_commit':PIN,'lanes':[],
        'physical_transitions_attempted':0,'physical_transitions_recorded':0,
        'reset_lanes_attempted':0,'reset_lanes_recorded':0,
        'optimizer_updates':0,'evaluation_transitions':0,'checkpoint_saves':0,
        'training_performed':False,'efficacy_evaluated':False,'promotion_authorized':False,
        'further_training_authorized':False,'passed':False}
    if diagnostic:r.update(status='M470_H4M6_DIAG_IN_PROGRESS',diag_spec=H4M6_DIAG_SPEC,
        snapshot_bytes=0,snapshots={},comparisons=[],leaf_schema=[],treedef_table=[],
        key_audits_attempted=0,key_audits_recorded=0,actor_calls_attempted=0,actor_calls_recorded=0,
        template_initializations_attempted=0,template_initializations_recorded=0)
    def persist():
        h4m6_write_bounded(path,r)
    h4m6_write_bounded(path,r,exclusive=True)
    monitor=None
    try:
        from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
        h4m3_no_runtime_imports()
        gpu=sample_gpu()
        h4m6_eval_require(gpu['total_mib']==12288 and 0<=gpu['used_mib']<.95*12288,'GPU_PREFLIGHT')
        r['gpu_initial']=gpu;persist()
        monitor=GPUMonitor()
        with monitor:
            h4cc_tech_actual(args,r,monitor,persist,rss=True,diagnostic=diagnostic)
        r['gpu']=h3_gpu_check(monitor)
        h4m6_eval_require((r['audit']==h4m6_diag_ledger(r) and not r['audit']['passed'] if diagnostic else
            r['audit']==h4m6_technical_audit(r) and r['audit']['passed'])
            and r['policy_sha256_before']==r['policy_sha256_after'],'TECH_FINAL_AUDIT')
        # Exact external-source scan in a new interpreter; no second inference or physics.
        command=[sys.executable,'-B','-c',
            'import json,sys;from pathlib import Path;from types import SimpleNamespace;'
            'import m463_gate3_cp_eval as e;import m463_push_ppo_train as t;'+
            ('a=e.h4m6_diag_authorization();' if diagnostic else 'a=t.h4m6_authorization(phase="tech");')+
            'p=json.loads(sys.argv[1]);'
            'args=SimpleNamespace(checkpoint_c=Path(p["C401M"]),h4_reward_report=Path(p["reward"]),'
            'h4cc_restore_report=Path(p["restore"]),output=Path(a["output"])/"report.json");'+
            ('print(json.dumps([a,e.h4m6_tech_inputs(args,a,diagnostic=True)],sort_keys=True))' if diagnostic else
             'print(json.dumps([a,e.h4m6_tech_inputs(args,a)],sort_keys=True))'),
            json.dumps(provenance['input_paths'],sort_keys=True)]
        checked=json.loads(subprocess.check_output(command,cwd=root,text=True))
        h4m6_eval_require(checked==[auth,provenance]
            and json.loads((path.parent/'attempt.json').read_bytes())=={'authorization':auth,'reserved':True}
            and json.loads((path.parent/'dispatch.started.json').read_bytes())=={'authorization':auth},
            'TECH_POST_IMMUTABILITY')
        r['status']='M470_H4M6_DIAG_COMPLETE' if diagnostic else 'M470_H4M6_TECH_COMPLETE'
        r['passed']=not diagnostic
        if diagnostic:r['post_immutability']=True
        persist()
        h4m6_write_bounded(path.parent/'attempt_result.json',{'status':'COMPLETE','authorization':auth},exclusive=True)
        print(r['status'],json.dumps(r['audit'],sort_keys=True),flush=True)
    except BaseException as exc:
        r['status']='M470_H4M6_DIAG_STOP' if diagnostic else 'M470_H4M6_TECH_STOP';r['passed']=False
        r['error']=(type(exc).__name__+': '+str(exc))[:4096]
        if monitor is not None:r['gpu']=monitor.snapshot()
        try:
            r['partial_ledger']=h4m6_diag_ledger(r,partial=True) if diagnostic else h4m6_technical_audit(r,partial=True)
        except BaseException as err:
            r['partial_ledger_error']=(type(err).__name__+': '+str(err))[:4096]
        try:
            persist()
            h4m6_write_bounded(path.parent/'attempt_result.json',{'status':'STOPPED','authorization':auth},exclusive=True)
        except BaseException as err:
            print('M470_H4M6_TECH_PARTIAL_WRITE_FAILED',repr(err),flush=True)
        raise

def h4m6_read_technical(path,expected_sha,auth):
    """Read completed native report under historical tech authorization, no active-marker substitution."""
    import m463_push_ppo_train as t
    from types import SimpleNamespace
    root=Path(__file__).resolve().parent
    path=Path(path)
    h4m6_eval_require(path.absolute()==path.resolve()==root/H4M6_TECH_OUTPUT
        and path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=16*1024**2,'TECH_READ_PATH')
    t.h4x_file_pin(path,expected_sha)
    r=json.loads(path.read_bytes());a=r['authorization']
    common=('id','source_sha256','spec_sha256','generated_sha256','external_sha256','static_sha256','static_log')
    h4m6_eval_require(all(a[k]==auth[k] for k in common) and a['phase']=='tech'
        and a['output']==t.H4M6_PATHS['tech']
        and all(a[k]=='-' for k in ('technical_sha256','ref_sha256','rss_sha256'))
        and auth['technical_sha256']==expected_sha,'TECH_AUTH_HISTORY')
    h4m6_eval_require(r['status']=='M470_H4M6_TECH_COMPLETE' and r['passed'] is True
        and r['spec']==t.H4M6_SPEC and r['source_commit']==PIN
        and r['jax_version']=='0.4.35' and r['brax_source']['sha256']==TRAIN_SHA
        and r['policy_sha256_before']==r['policy_sha256_after']
        and r['gpu']['total_mib']==12288 and 0<r['gpu']['peak_used_mib']<.95*12288
        and r['gpu']['samples']>0 and not r['gpu']['sampler_errors']
        and r['further_training_authorized'] is False,'TECH_REPORT_CONTRACT')
    for name,expected in (('attempt.json',{'authorization':a,'reserved':True}),
        ('dispatch.started.json',{'authorization':a}),
        ('attempt_result.json',{'status':'COMPLETE','authorization':a})):
        p=path.parent/name
        h4m6_eval_require(p.is_file() and not p.is_symlink() and json.loads(p.read_bytes())==expected,'TECH_MARKER '+name)
    p=r['provenance']['input_paths']
    proxy=SimpleNamespace(checkpoint_c=Path(p['C401M']),h4_reward_report=Path(p['reward']),
        h4cc_restore_report=Path(p['restore']),output=path)
    h4m6_eval_require(h4m6_tech_inputs(proxy,a)==r['provenance']
        and h4m6_technical_audit(r)==r['audit'] and r['audit']['passed'],'TECH_REPORT_REPLAY')
    reward=json.loads(Path(p['reward']).read_bytes())['train_reward_config']['scales']
    for lane in r['lanes']:
        for row in lane['rows']:
            h4m6_eval_require(set(row['reward_names'])==set(reward)
                and row['reward_scales']==[float(reward[k]) for k in row['reward_names']],'TECH_FROZEN_REWARD')
    return r

def h4m6_verify_reward_scale(names,raw,scales,w,np):
    """Replay pinned native bool/float32 reward types, not the promoted stack type."""
    boolean_names=('termination','collision')
    for i,name in enumerate(names):
        a,s,b=raw[i],scales[i],w[i]
        if name in boolean_names:
            # A native predicate is exactly False/True, not an arbitrary float.
            if a.tobytes() not in (np.float32(0).tobytes(),np.float32(1).tobytes()):
                return False
            if a==0:
                # Native lowering may select +0 or multiply to -0. Only the
                # inactive boolean term has no source-specified zero sign.
                if b!=0:
                    return False
                continue
            expected=s
        else:
            expected=np.float32(a*s)
        if expected.tobytes()!=b.tobytes():
            return False
    return True


def h4m6_technical_audit(report,*,partial=False,diagnostic=False):
    """Pure replay: exact same-trajectory paired reward isolation, no physics."""
    import numpy as np
    req=h4m6_eval_require
    lanes=report['lanes']
    req(not diagnostic or (partial and len(lanes)<=4 and report['physical_transitions_attempted']<=32
        and report['reset_lanes_attempted']<=4),'DIAG_NOT_TECH_ACCEPTANCE')
    req(len(lanes)<=8 and report['reset_lanes_recorded']==len(lanes)
        and len(lanes)<=report['reset_lanes_attempted']<=8,'TECH_RESET_LEDGER')
    req(sum(len(l['rows']) for l in lanes)==report['physical_transitions_recorded']
        <=report['physical_transitions_attempted']<=416,'TECH_PHYSICAL_LEDGER')
    req(sum(l['physical_attempted'] for l in lanes)==report['physical_transitions_attempted']
        and sum(l['physical_recorded'] for l in lanes)==report['physical_transitions_recorded'],'TECH_LANE_TOTALS')
    outstanding_reset=report['reset_lanes_attempted']-report['reset_lanes_recorded']
    outstanding_step=report['physical_transitions_attempted']-report['physical_transitions_recorded']
    req(outstanding_reset+outstanding_step<=(1 if partial else 0)
        and (not outstanding_step or (lanes and lanes[-1]['physical_attempted']-lanes[-1]['physical_recorded']==1)),
        'TECH_PARTIAL_LEDGER')
    for name in ('optimizer_updates','evaluation_transitions','checkpoint_saves'):
        req(report[name]==0,'TECH_ZERO_'+name)
    req(report['training_performed'] is False and report['efficacy_evaluated'] is False
        and report['promotion_authorized'] is False,'TECH_NO_PROMOTION')
    f32=lambda x:np.asarray(x,dtype=np.float32)
    bits=lambda a,b:f32(a).shape==f32(b).shape and f32(a).tobytes()==f32(b).tobytes()
    pairs=0
    for i,l in enumerate(lanes):
        fixture=tuple(H4M6_DIAG_SPEC['fixture']) if diagnostic else H4M6_TECH_FIXTURES[i//2]
        arm=H4M6_DIAG_ORDER[i][:3].lower() if diagnostic else ('ref','rss')[i%2]
        req(l['fixture']==list(fixture) and l['arm']==arm
            and l['metadata']==h4cc_tech_fixture_values(fixture)
            and l['prior_actual_pushes']==0 and l['initial_command']==[0.,0.,0.]
            and len(l['rows'])==l['physical_recorded']<=l['physical_attempted']<=52,'TECH_LANE')
        seed,cls,direction,first,target,age,clock=fixture
        last_sha=l['initial_sha256']; last_key=l['initial_rng']
        for k,row in enumerate(l['rows']):
            m=row['metrics']; e=row['rss']
            req(row['pre_sha256']==last_sha and row['rng_before']==last_key
                and row['source_clock']==clock+k and m['age']==age+k
                and m['wrapper_steps']==age+k+1 and m['boundary']==m['refresh']==0
                and m['event']==m['repeat']==int(k==0)
                and m['mature']==int(k==0 and cls==3)
                and m['source_terminal']==m['terminal'],'TECH_SCHEDULER')
            if k==0:
                lo,hi=(.30,.40) if cls==3 else (.20,.60)
                req(m['direction']==direction and m['gap']==age-first
                    and m['side_target']==direction-1
                    and m['side_matched']==int(m['support']==direction-1)
                    and m['side_fallback']==1-m['side_matched']
                    and m['event_qual']==(direction-1 if seed==4101 else 0)
                    and lo<=row['magnitude']<=hi
                    and abs(math.hypot(*row['delta'])-row['magnitude'])<=3e-6,'TECH_EVENT')
            else:
                req(row['magnitude']==0 and row['delta']==[0.,0.] and m['event_qual']==0,'TECH_NO_EXTRA_PUSH')
            req(row['next_source_clock']==(0 if m['terminal'] else clock+k+1)
                and row['post']==row['next_actor']==row['next_critic']==[0.,0.,0.],'TECH_COMMAND')
            req(not m['terminal'] or row['reset_data_obs_equal'] is True,'TECH_TERMINAL_RESET')
            last_sha=row['post_sha256']; last_key=row['rng_after']
            req(row['tick']==k and row['finite'] is True and not m['bad'] and not m['cc_bad']
                and not m['truncation'] and row['rng_after']==row['expected_rng']
                and not e['bad'] and row['pre']==row['actor']==row['critic'],'TECH_ROW')
            req(m['terminal'] in (0,1) and (not m['terminal'] or k==len(l['rows'])-1),'TECH_TERMINAL')
            req(e['qualified']==int(k==0 and fixture[0]==4101)
                and e['eligible']==int(fixture[0]==4101 and 1<=k<=50 and not m['terminal']),'TECH_ELIGIBLE')
            req(e['start']==int(e['qualified'] and not m['terminal'])
                and e['tick_age']==(k if e['eligible'] else 0),'TECH_TIMING')
            names=row['reward_names']; raw=f32(row['raw_rewards']); w=f32(row['weighted_rewards'])
            scales=f32(row['reward_scales'])
            req(len(names)==24 and len(set(names))==24 and 'stand_still' in names
                and 'termination' in names and 'collision' in names
                and row['reward_native_types']==['bool' if k in ('termination','collision') else 'float32' for k in names]
                and raw.shape==w.shape==scales.shape==(24,)
                and np.isfinite(raw).all() and np.isfinite(w).all() and np.isfinite(scales).all(),'TECH_REWARD_SCHEMA')
            idx=names.index('stand_still')
            req(scales[idx]==-1 and scales[names.index('termination')]==-100
                and h4m6_verify_reward_scale(names,raw,scales,w,np),'TECH_REWARD_SCALE')
            req(all(np.isfinite(float(e[x])) for x in ('factor','raw_stand','shadow_stand',
                'actual_stand','shadow_relief','delivered_relief','base_reward','expected_reward')),'TECH_FINITE')
            factor=h4m6_native_factor(k,e['eligible'],np)
            req(bits(factor,e['factor']) and bits(raw[idx],e['actual_stand'])
                and bits(w[idx],-f32(e['actual_stand']))
                and bits(e['expected_reward'],row['reward']),'TECH_REWARD_ACCOUNTING')
            h4m3_reward_check({'reward/'+name:float(value) for name,value in zip(names,w)},row['reward'],np)
            base=w.copy();base[idx]=-f32(e['raw_stand'])
            h4m3_reward_check({'reward/'+name:float(value) for name,value in zip(names,base)},e['base_reward'],np)
            req(bits(e['shadow_stand'],factor*f32(e['raw_stand']))
                and bits(e['shadow_relief'],(f32(e['raw_stand'])-f32(e['shadow_stand']))*np.float32(.02))
                and bits(e['delivered_relief'],(f32(e['raw_stand'])-f32(e['actual_stand']))*np.float32(.02)),'TECH_RELIEF_FORMULA')
            req(bits(e['actual_stand'],e['raw_stand'] if arm=='ref' else e['shadow_stand'])
                and (e['eligible'] or (bits(e['raw_stand'],e['actual_stand']) and e['delivered_relief']==0))
                and (arm!='ref' or e['delivered_relief']==0)
                and (not m['terminal'] or e['delivered_relief']==0),'TECH_REWARD_SCOPE')
            if i%2 and not diagnostic:
                ref=lanes[i-1]; req(l['initial_sha256']==ref['initial_sha256']
                    and l['initial_rng']==ref['initial_rng'] and l['model_sha256']==ref['model_sha256'],'TECH_INITIAL_MATCH')
                req(k<len(ref['rows']),'TECH_PAIRED_LENGTH')
                b=ref['rows'][k]; be=b['rss']
                for key in ('pre_sha256','post_sha256','physics_sha256','action_sha256','rng_before',
                    'rng_after','native_draw','metrics','reward_names','reward_native_types','reward_scales','post','next_actor','next_critic'):
                    req(row[key]==b[key],'TECH_PAIRED_NONREWARD '+key)
                req(all(bits(row['raw_rewards'][j],b['raw_rewards'][j]) and
                    bits(row['weighted_rewards'][j],b['weighted_rewards'][j])
                    for j in range(24) if j!=idx),'TECH_OTHER_23_TERMS')
                req({k:v for k,v in e.items() if k not in ('actual_stand','delivered_relief','expected_reward')}
                    =={k:v for k,v in be.items() if k not in ('actual_stand','delivered_relief','expected_reward')},
                    'TECH_PAIRED_WINDOW')
                for key in ('factor','raw_stand','shadow_stand','shadow_relief','base_reward'):
                    req(bits(e[key],be[key]),'TECH_PAIRED_SHADOW '+key)
                if not e['eligible']:
                    req(bits(row['reward'],b['reward']) and bits(raw,b['raw_rewards'])
                        and bits(w,b['weighted_rewards']),'TECH_EXACT_IDENTITY')
                pairs+=1
    if not partial:
        req(len(lanes)==8 and report['reset_lanes_attempted']==report['reset_lanes_recorded']==8
            and report['physical_transitions_attempted']==report['physical_transitions_recorded']
            and all(l['physical_attempted']==l['physical_recorded'] and len(l['rows'])>0
                and (len(l['rows'])==52 or l['rows'][-1]['metrics']['terminal']) for l in lanes)
            and all(len(lanes[i]['rows'])==len(lanes[i+1]['rows']) for i in range(0,8,2)),'TECH_COMPLETE')
        req(all(len(lanes[i]['rows'])==52 and lanes[i]['rows'][50]['rss']['completed']==1
            and lanes[i]['rows'][51]['rss']['eligible']==0
            and lanes[i]['rows'][51]['rss']['factor']==1 for i in (0,1)),'TECH_REQUIRED_RESTORATION')
        req(any(row['rss']['delivered_relief']>0 for row in lanes[1]['rows']),'TECH_NONZERO_RELIEF')
    return {'passed':not partial,'paired_rows':pairs,'physical_recorded':report['physical_transitions_recorded'],
            'reset_lanes_recorded':report['reset_lanes_recorded'],
            'outstanding_reset':outstanding_reset,'outstanding_step':outstanding_step}

def h4m6_dev_decisions(archive,fresh):
    """Reuse the exact official/incremental decision code; no changed efficacy thresholds."""
    req=h4m6_eval_require
    req(len(fresh['groups'])==24 and {g['policy'] for g in fresh['groups']}=={'REF6','RSS'},'DEV_ARMS')
    mapped=dict(fresh,groups=[dict(g,policy={'REF6':'REF','RSS':'CCRC'}[g['policy']])
                              for g in fresh['groups']])
    result=h4cc_pair_decisions(archive,mapped)
    result['official']['comparison_roles']['CAND']='RSS'
    result['incremental']['comparison_roles']={'CTRL':'FRESH_REF6','CAND':'RSS'}
    result['promotion_authorized']=False
    result['further_training_authorized']=False
    return result

def h4m6_mechanism_decision(efficacy,exposure,rows):
    """Decision on verified endpoint evidence only; never manufacture missing endpoints."""
    req=h4m6_eval_require
    expected={(s,d) for s,d in H4M6_MECHANISM_LANES}
    req(len(rows)==4 and {(r['seed'],r['direction']) for r in rows}==expected,'MECHANISM_LANES')
    primary=[]; positive=[]
    for r in rows:
        k=(r['seed'],r['direction'])
        req(r['role']==('primary' if k in H4M6_MECHANISM_LANES[:2] else 'positive'),'MECHANISM_ROLE')
        if r['candidate_adverse'] or not r['safety_preserved']:
            verdict='ADVERSE'
        elif not r['candidate_endpoints_available'] and r['role']=='positive':
            verdict='ADVERSE'
        elif not r['reference_endpoints_available'] or not r['candidate_endpoints_available']:
            verdict='INCONCLUSIVE'
        elif r['role']=='primary':
            if not r['matched_sole_ipsilateral']:
                verdict='INCONCLUSIVE'
            else:
                verdict='PASS' if (r['placement_gain_m']>=.02 and r['load3_complete_tick'] is not None
                    and r['load3_complete_tick']<=354 and r['signed_velocity_gain_mps']>=.10
                    and r['absolute_velocity_gain_mps']>=.10 and r['candidate_survived']
                    and r['candidate_recovery5']) else 'NOT_SUPPORTED'
        else:
            verdict='PASS' if (r['candidate_survived'] and r['candidate_recovery5']
                and r['abs_v385_excess_mps']<=.10 and r['abs_H385_excess']<=.50) else 'ADVERSE'
        (primary if r['role']=='primary' else positive).append(verdict)
    decision=('NO_GO' if efficacy=='NO_GO' or 'ADVERSE' in primary+positive else
              'INCONCLUSIVE' if exposure!='ADEQUATE' or 'INCONCLUSIVE' in primary+positive else
              'LOCAL_H_R3b_SUPPORT' if efficacy=='DEV_PASS' and primary==['PASS','PASS']
                 and positive==['PASS','PASS'] else
              'HETEROGENEOUS' if primary.count('PASS')==1 else
              'NOT_SUPPORTED_IN_TESTED_DESIGN' if primary.count('NOT_SUPPORTED')==2 else 'INCONCLUSIVE')
    return {'decision':decision,'primary':primary,'positive':positive,
            'efficacy':efficacy,'exposure':exposure,'promotion_authorized':False}


def h4m6_dev_inputs(auth):
    """Immutable evidence for the fresh paired dev run; no runtime import."""
    import m463_push_ppo_train as t
    root=H4_EVAL_ROOT
    h4m6_eval_require(auth['phase']=='dev','DEV_AUTH')
    tech=h4m6_read_technical(root/H4M6_TECH_OUTPUT,auth['technical_sha256'],auth)
    external=h4m6_external_sources(root)
    h4m6_eval_require(h4m3_digest(external)==auth['external_sha256']
        and external==tech['provenance']['external'],'DEV_EXTERNAL')
    reports={};pins={'archive/report.json':H4CC_ARCHIVE_SHA,'tech/report.json':auth['technical_sha256']}
    archive=h4cc_read_archive(root/H4_EVAL_OUTPUT)
    for arm in ('ref','rss'):
        out=root/t.H4M6_PATHS[arm]
        r=t.h4m6_read_final_report(out/'report.json',auth[arm+'_sha256'],arm,auth)
        reports[arm]=r
        for name in ('report.json','attempt.json','dispatch.started.json','attempt_result.json',
                     'm463_brax_ppo_instrumented.py'):
            pins[arm+'/'+name]=sha256(out/name)
        for p,f in zip(r['checkpoints'],r['fullstate_checkpoints']):
            for name in (p['file'],f['file'],f['manifest']):
                pins[arm+'/'+name]=sha256(out/name)
    ref,rss=reports['ref'],reports['rss']
    historical=json.loads((h4e_paths('control')/'report.json').read_bytes())
    h4m6_eval_require(ref['h4_common_contract']==rss['h4_common_contract']
        and ref['h4x_brax_dependencies']==rss['h4x_brax_dependencies']==historical['h4x_brax_dependencies'],
        'DEV_MATCHED_CONTRACT')
    exposure=t.h4m6_exposure_comparison(ref['h4m6_final'],rss['h4m6_final'])
    h4m6_eval_require(exposure==rss['h4m6_exposure_comparison'],'DEV_EXPOSURE_REPLAY')
    for name in ('train.py','paper_config.py','m462_ppo_warmstart_run.py','m462_typed_orbax_stage.py'):
        expected=archive['input_sha256']['source/'+name];t.h4x_file_pin(root/name,expected)
        pins['source/'+name]=expected
    for name,value in t.h4m6_source_identity(root)['files'].items():
        pins['source/'+name]=value
    pins['external']=auth['external_sha256']
    h4m3_no_runtime_imports()
    return pins,archive,external,exposure

def h4m6_rich_write(sidecar,*,initial=False):
    meta={k:v for k,v in sidecar.items() if k!='records'}
    h4m6_mechanism_bound(sidecar['records'],meta)
    return h4m6_write_bounded(H4_EVAL_ROOT/H4M6_RICH_OUTPUT,sidecar,
        exclusive=initial,cap=32*1024**2)

def h4m6_add_rich(sidecar,label,seed,direction,tick,row,raw,np):
    req=h4m6_eval_require
    req(label in ('REF6','RSS') and (seed,direction) in H4M6_MECHANISM_LANES
        and 334<=tick<=385 and len(sidecar['records'])<416,'RICH_SCOPE')
    req(not any((r['policy'],r['seed'],r['direction'],r['tick'])==(label,seed,direction,tick)
        for r in sidecar['records']),'RICH_DUPLICATE')
    data=h4m3_pack_capture(raw,row,sidecar['manifest'],np)
    req(data['com']['available'] is True and data['contact']['available'] is True,'RICH_COM_LOAD')
    item={'policy':label,'seed':seed,'direction':direction,'tick':tick,'row':row,'data':data}
    item=json.loads(json.dumps(item,allow_nan=False,default=lambda a:np.asarray(a).tolist()))
    req(len(json.dumps(item,allow_nan=False,separators=(',',':')).encode())+128<=65536,'RICH_RECORD_BYTES')
    sidecar['records'].append(item)

def h4m6_dev_ledger(session,*,partial=False):
    req=h4m6_eval_require
    groups=session['groups']
    expected={(p,s,c) for p in ('REF6','RSS') for s in H3_EVAL_SEEDS for c in H3_EVAL_PROFILES}
    keys={(g['policy'],g['seed'],g['profile']) for g in groups}
    req(len(keys)==len(groups)<=24 and keys<=expected,'DEV_GROUP_KEYS')
    rows=sum(len(g['nominal_frames'])+sum(len(v['single_suffix'])+len(v['double_suffix'])
             for v in g['directions'].values()) for g in groups)
    complete=sum(bool(g['complete']) for g in groups)
    ra=session['reset_lanes_attempted'];rr=session['reset_lanes_recorded']
    pa=session['physical_transitions_attempted'];pr=session['physical_transitions_recorded']
    req(rows==pr<=pa<=89616 and 0<=rr<=ra<=24
        and complete*5==session['logical_trajectories_completed']
        and ra-rr+pa-pr<=(1 if partial else 0),'DEV_LEDGER')
    req(complete<=rr<=len(groups) and len(groups)-complete<=(1 if partial else 0),'DEV_GROUP_LEDGER')
    if not partial:
        req(keys==expected and ra==rr==complete==24,'DEV_COMPLETE')
    return {'physical_recorded':pr,'outstanding_step':pa-pr,'reset_recorded':rr,
            'outstanding_reset':ra-rr,'complete_groups':complete,'logical':complete*5}

def h4m6_rich_projection(session,sidecar):
    req=h4m6_eval_require
    req(tuple(session['frame_columns'])==H4M3_COLUMNS and len(session['groups'])==24,'RICH_FRAME_COLUMNS')
    expected={}
    for g in session['groups']:
        if g['profile']!='stand':continue
        for seed,d in H4M6_MECHANISM_LANES:
            if seed!=g['seed']:continue
            item=g['directions'][d]
            single=item['single_suffix']
            if len(single)>=35:
                expected[(g['policy'],seed,d,334)]=single[34]
            for tick,row in enumerate(item['double_suffix'][:51],335):
                expected[(g['policy'],seed,d,tick)]=row
    actual={(r['policy'],r['seed'],r['direction'],r['tick']):r['row'] for r in sidecar['records']}
    req(len(actual)==len(sidecar['records']) and actual==expected,'RICH_RAW_PROJECTION')
    return {'records':len(actual),'raw_projection_sha256':h4m3_digest(
        [[*k,v] for k,v in sorted(expected.items())]),'extra_physical_transitions':0}

def h4m6_mechanism_endpoints(session,sidecar,np):
    req=h4m6_eval_require
    h4m6_rich_projection(session,sidecar)
    groups={(g['policy'],g['seed']):g for g in session['groups'] if g['profile']=='stand'}
    cases=[]
    for idx,(seed,direction) in enumerate(H4M6_MECHANISM_LANES):
        role='primary' if idx<2 else 'positive'
        ends={}; summaries={}; support={}; load={}
        recovery='right' if direction=='left' else 'left'
        for arm in ('REF6','RSS'):
            g=groups[(arm,seed)]
            traces=h3_decode(g,session['frame_columns'])
            trace,exposed=traces[('double',direction)]
            summaries[arm]=h3_summary(trace,exposed,'double',traces[('single',direction)][0])
            samples={r['tick']:r for r in sidecar['records'] if
                     (r['policy'],r['seed'],r['direction'])==(arm,seed,direction)}
            origin=samples.get(334)
            support[arm]=False if origin is None else (
                origin['row'][H4M3_COLUMNS.index('lf_ground_contact')]==int(direction=='left')
                and origin['row'][H4M3_COLUMNS.index('rf_ground_contact')]==int(direction=='right'))
            ends[arm]={str(t):None for t in (354,365,385)}
            load[arm]=None
            if origin is not None and exposed:
                kick=g['directions'][direction]['second']['actual_world_dqvel_xy_mps']
                e=np.asarray([*kick,0.],dtype=np.float64); norm=np.linalg.norm(e)
                req(np.isfinite(e).all() and norm>0,'RICH_DIRECTION')
                geom={'e':(e/norm).tolist(),'recovery':recovery,'support':direction}
                for tick in (354,365,385):
                    if tick in samples:
                        ends[arm][str(tick)]=h4m4_endpoint(samples[tick],origin,geom,np)
                load[arm]=h4m4_load_summary([samples[t] for t in sorted(samples) if t>=335],recovery,.05)
        b,t=ends['REF6'],ends['RSS'];bs,ts=summaries['REF6'],summaries['RSS']
        required=('354','365') if role=='primary' else ('385',)
        br=all(b[k] is not None for k in required);tr=all(t[k] is not None for k in required)
        pair=next(p for p in session['decision']['incremental']['decision']['paired_cases']
                  if p['case']==[seed,'stand','double',direction])
        safe=all(x['pass'] for x in pair['safety'].values()) and not pair['new_critical_terminal_flags']
        btrace=h3_decode(groups[('REF6',seed)],session['frame_columns'])[('double',direction)][0]
        ttrace=h3_decode(groups[('RSS',seed)],session['frame_columns'])[('double',direction)][0]
        earlier=len(ttrace)<len(btrace) and bool(ttrace[-1]['done'])
        r={'seed':seed,'direction':direction,'role':role,'endpoints':ends,'load':load,
           'matched_sole_ipsilateral':all(support.values()),'initial_support':support,
           'reference_endpoints_available':br,'candidate_endpoints_available':tr,
           'candidate_adverse':earlier or (not tr and bool(ttrace[-1]['done'])),
           'safety_preserved':safe,'candidate_survived':ts['survived'],
           'candidate_recovery5':ts['recovery_5s']['recovered'],
           'reference_survived':bs['survived'],'reference_recovery5':bs['recovery_5s']['recovered'],
           'placement_gain_m':None,'load3_complete_tick':None if load['RSS'] is None else load['RSS']['completed'],
           'signed_velocity_gain_mps':None,'absolute_velocity_gain_mps':None,
           'abs_v385_excess_mps':None,'abs_H385_excess':None}
        if br and tr:
            if role=='primary':
                r.update(placement_gain_m=t['354']['D_m']-b['354']['D_m'],
                    signed_velocity_gain_mps=b['365']['v_push_mps']-t['365']['v_push_mps'],
                    absolute_velocity_gain_mps=abs(b['365']['v_push_mps'])-abs(t['365']['v_push_mps']))
            else:
                r.update(abs_v385_excess_mps=abs(t['385']['v_push_mps'])-abs(b['385']['v_push_mps']),
                    abs_H385_excess=t['385']['angular_momentum_norm']-b['385']['angular_momentum_norm'])
        cases.append(r)
    return cases

def h4m6_dev_dispatch(args):
    h4m6_runtime_gate('dev')
    import m463_push_ppo_train as t
    import shutil
    root=H4_EVAL_ROOT; path=root/H4M6_DEV_OUTPUT
    auth=t.h4m6_authorization(phase='dev')
    h4m6_eval_require(args.mode=='run' and args.h4m6_phase=='dev'
        and args.output.absolute()==args.output.resolve()==path
        and os.environ.get('JAX_PLATFORMS')=='cuda'
        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
        and os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')=='false'
        and shutil.disk_usage(path.parent.parent).free>=2*1024**3+2*128*1024**2+2*32*1024**2,'DEV_PREFLIGHT')
    pins,archive,external,exposure=h4m6_dev_inputs(auth)
    args.seeds=list(H3_EVAL_SEEDS);args.profiles=list(H3_EVAL_PROFILES);args.directions=['left','right']
    args.push_step=300;args.push_magnitude=.50;args.second_push_step=335
    args.second_push_magnitude=.35;args.max_steps=1000;args.max_cases=120
    h4m6_reserve_tech(path,auth)
    session={'status':'M470_H4M6_DEV_IN_PROGRESS','authorization':auth,'source_commit':PIN,
        'brax_train_sha256':TRAIN_SHA,'evaluator_sha256':sha256(Path(__file__)),'input_sha256':pins,
        'seeds':args.seeds,'profiles':args.profiles,'directions':args.directions,'push_step':300,
        'push_magnitude_mps':.50,'second_push_step':335,'second_push_magnitude_mps':.35,
        'max_steps':1000,'dt_s':.02,'training_performed':False,'deterministic_actor':True,
        'domain_randomization_enabled':False,'random_source_push_enabled':False,
        'original_reward_PD_termination':True,'expected_logical_trajectories':120,
        'max_physical_transitions':89616,'physical_transitions_attempted':0,
        'physical_transitions_recorded':0,'logical_trajectories_completed':0,
        'reset_lanes_attempted':0,'reset_lanes_recorded':0,'frame_columns':None,'groups':[],
        'promotion_authorized':False,'exposure':exposure}
    sidecar={'authorization':auth,'input_sha256':pins,'manifest':None,'records':[]}
    h3_write(path,session,initial=True);h4m6_rich_write(sidecar,initial=True)
    args.h3_session=session;args.h4m6_rich=sidecar
    monitor=None
    try:
        from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
        g=sample_gpu()
        h4m6_eval_require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288,'DEV_GPU_PREFLIGHT')
        session['gpu_initial']=g
        monitor=GPUMonitor();args.h3_monitor=monitor
        with monitor:run(args)
        session['gpu']=h3_gpu_check(monitor)
        command=[sys.executable,'-B','-c',
            'import json;import m463_push_ppo_train as t;import m463_gate3_cp_eval as e;'
            'a=t.h4m6_authorization(phase="dev");p,_,x,y=e.h4m6_dev_inputs(a);'
            'print(json.dumps([a,p,x,y],sort_keys=True))']
        h4m6_eval_require(json.loads(subprocess.check_output(command,cwd=root,text=True))==[auth,pins,external,exposure]
            and session['logical_trajectories_completed']==120
            and session['physical_transitions_attempted']==session['physical_transitions_recorded']<=89616
            and session['reset_lanes_attempted']==session['reset_lanes_recorded']==24
            and h4m6_dev_ledger(session)==session['ledger']
            and h4m6_rich_projection(session,sidecar)==sidecar['projection']
            and sidecar['authorization']==auth and sidecar['input_sha256']==pins
            and json.loads((path.parent/'attempt.json').read_bytes())=={'authorization':auth,'reserved':True}
            and json.loads((path.parent/'dispatch.started.json').read_bytes())=={'authorization':auth}
            and sha256(root/H4M6_RICH_OUTPUT)==session['mechanism_sha256'],'DEV_FINAL')
        session['status']='M470_H4M6_DEV_COMPLETE'
        h3_write(path,session)
        h4m6_write_bounded(path.parent/'attempt_result.json',{'status':'COMPLETE','authorization':auth},exclusive=True)
        print(session['status'],session['decision']['decision'],session['mechanism']['decision'],flush=True)
    except BaseException as exc:
        session['status']='M470_H4M6_DEV_STOP';session['error']=(type(exc).__name__+': '+str(exc))[:4096]
        if monitor is not None:session['gpu']=monitor.snapshot()
        try:
            session['partial_ledger']=h4m6_dev_ledger(session,partial=True)
        except BaseException as err:
            session['partial_ledger_error']=(type(err).__name__+': '+str(err))[:4096]
        try:
            h4m6_rich_write(sidecar)
            h3_write(path,session)
            h4m6_write_bounded(path.parent/'attempt_result.json',{'status':'STOPPED','authorization':auth},exclusive=True)
        except BaseException as err:print('M470_H4M6_DEV_PARTIAL_WRITE_FAILED',repr(err),flush=True)
        raise

def h4m6_mechanism_bound(records,metadata):
    """Byte accounting for future exact rich capture, no extra physical steps."""
    req=h4m6_eval_require
    req(len(records)<=416,'MECHANISM_RECORD_CAP')
    encode=lambda x:len(json.dumps(x,separators=(',',':'),allow_nan=False).encode('utf8'))
    sizes=[encode(row)+128 for row in records]
    m=encode(metadata)
    req(all(n<=65536 for n in sizes) and m<=2*1024**2 and sum(sizes)+m<=32*1024**2,
        'MECHANISM_BYTE_CAP')
    return {'records':len(records),'record_bytes':sum(sizes),'metadata_bytes':m,'max_record_bytes':max(sizes,default=0)}

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
    p.add_argument('--h4cc-eval',action='store_true',help='CCRC held plan only')
    p.add_argument('--h4cc-technical',action='store_true',help='CCRC separately held technical diagnostic')
    p.add_argument('--h4cc-restore-report',type=Path)
    p.add_argument('--h4cc-static-log',type=Path)
    p.add_argument('--h4m3-placement',action='store_true')
    p.add_argument('--h4m3-static-archives',action='store_true')
    p.add_argument('--h4m3-historical-archive',type=Path)
    p.add_argument('--h4m4-feasibility',action='store_true')
    p.add_argument('--h4m4-historical-archive',type=Path)
    p.add_argument('--h4m5-obs-attribution',action='store_true')
    p.add_argument('--h4m6-phase',choices=('tech','dev','diag'),help='H4-M6 separately authorized phases')
    p.add_argument('--h4m6-diag-child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.h4m6_diag_child and not (args.mode=='run' and args.h4m6_phase=='diag'):
        p.error('M470_H4M6_DIAG_CHILD_CONFLICT')
    if args.h4m6_phase is not None:
        try:
            if args.mode=='run':
                if args.h4m6_phase=='diag':h4m6_diag_authorization()
                else:h4m6_runtime_gate(args.h4m6_phase)
            allowed=('mode','output','h4m6_phase')
            if args.mode=='run' and args.h4m6_phase in ('tech','diag'):
                allowed+=('checkpoint_c','h4_reward_report','h4cc_restore_report')
            if args.mode=='run' and args.h4m6_phase=='diag':allowed+=('h4m6_diag_child',)
            h4m6_eval_require(all(getattr(args,k)==p.get_default(k) for k in vars(args)
                if k not in allowed),'PLAN_CONFLICT')
            root=Path(__file__).resolve().parent
            expected=root/(H4M6_DIAG_OUTPUT if args.h4m6_phase=='diag' else (H4M6_TECH_OUTPUT if args.h4m6_phase=='tech' else H4M6_DEV_OUTPUT))
            h4m6_eval_require(args.output.absolute()==args.output.resolve()==expected,'PLAN_OUTPUT')
            if args.h4m6_phase=='diag':
                if args.mode=='run':return h4m6_diag_dispatch(args)
                print('M470_H4M6_DIAG_PLAN',json.dumps(dict(H4M6_DIAG_SPEC,execution_authorized=False),sort_keys=True));return
            if args.mode=='run':
                return h4m6_technical_dispatch(args) if args.h4m6_phase=='tech' else h4m6_dev_dispatch(args)
            print('M470_H4M6_PLAN',json.dumps(h4m6_eval_plan(args.h4m6_phase),sort_keys=True))
            return
        except (RuntimeError,ValueError,KeyError,OSError) as exc:
            p.error(str(exc))
    if args.h4m5_obs_attribution:
        try:
            h4m5_require(not any(getattr(args,k) for k in ('h4m4_feasibility',
                'h4m4_historical_archive','h4m3_placement','h4m3_static_archives',
                'h4m3_historical_archive','h4cc_technical','h4cc_eval','h4_m1_eval',
                'h4_technical','h3_m1_eval','h2a','h2b','checkpoint_c','checkpoint_p',
                'report_p','checkpoint_h3','report_h3','h4_reward_report','h4cc_restore_report',
                'h4cc_static_log')) and all(getattr(args,k)==p.get_default(k) for k in
                ('seeds','profiles','directions','push_step','push_magnitude','second_push_step',
                'second_push_magnitude','max_steps','max_cases')),'FROZEN_FLAGS')
            root=Path(__file__).resolve().parent
            h4m5_require(args.output.absolute()==root/H4M5_OUTPUT
                and args.output.resolve()==root/H4M5_OUTPUT,'OUTPUT_PATH')
            if args.mode=='run':
                args.output=root/H4M5_OUTPUT
                return h4m5_dispatch(args)
            print('M470_H4M5_PLAN',json.dumps(h4m5_plan(),sort_keys=True))
            return
        except (RuntimeError,ValueError,KeyError,OSError) as exc:
            p.error(str(exc))
    if args.h4m4_feasibility:
        try:
            h4m4_require(not any(getattr(args,k) for k in ('h4m3_placement',
                'h4m3_static_archives','h4m3_historical_archive','h4cc_technical',
                'h4cc_eval','h4_m1_eval','h4_technical','h3_m1_eval','h2a','h2b',
                'checkpoint_c','checkpoint_p','report_p','checkpoint_h3','report_h3',
                'h4_reward_report','h4cc_restore_report','h4cc_static_log'))
                and all(getattr(args,k)==p.get_default(k) for k in ('seeds','profiles',
                'directions','push_step','push_magnitude','second_push_step',
                'second_push_magnitude','max_steps','max_cases')),'FROZEN_FLAGS')
            root=Path(__file__).resolve().parent
            h4m4_require(args.output.absolute()==root/H4M4_OUTPUT
                and args.output.resolve()==root/H4M4_OUTPUT,'OUTPUT_PATH')
            if args.mode=='run':
                h4m4_authorization()
                h4m4_require(args.h4m4_historical_archive is not None,'HISTORICAL_REQUIRED')
                return h4m4_dispatch(args)
            h4m4_require(args.h4m4_historical_archive is None,'PLAN_FLAGS')
            print('M470_H4M4_PLAN',json.dumps(h4m4_plan(),sort_keys=True))
            return
        except (RuntimeError,ValueError,KeyError,OSError) as exc:
            p.error(str(exc))
    if args.h4m4_historical_archive is not None:
        p.error('M470_H4M4_FLAG_REQUIRED')
    if args.h4m3_placement:
        try:
            h4m3_require(not any(getattr(args,k) for k in ('h4cc_technical',
                'h4cc_eval','h4_m1_eval','h4_technical','h3_m1_eval','h2a','h2b',
                'checkpoint_c','checkpoint_p','report_p','checkpoint_h3','report_h3',
                'h4_reward_report','h4cc_restore_report','h4cc_static_log'))
                and all(getattr(args,k)==p.get_default(k) for k in ('seeds','profiles',
                'directions','push_step','push_magnitude','second_push_step',
                'second_push_magnitude','max_steps','max_cases')),'FROZEN_FLAGS')
            root=Path(__file__).resolve().parent
            h4m3_require(args.output.absolute()==root/H4M3_OUTPUT
                and args.output.resolve()==root/H4M3_OUTPUT,'OUTPUT_PATH')
            if args.mode=='run':
                h4m3_authorization()
                h4m3_require(not args.h4m3_static_archives
                    and args.h4m3_historical_archive is not None,'RUN_FLAGS')
                return h4m3_dispatch(args)
            h4m3_require(args.h4m3_static_archives==(args.h4m3_historical_archive is not None),
                          'STATIC_FLAGS')
            if args.h4m3_static_archives:
                print('M470_H4M3_ARCHIVE_STATIC_PASS',json.dumps(h4m3_static_archive_check(
                    root/H4CC_EVAL_OUTPUT,args.h4m3_historical_archive),sort_keys=True))
            else:
                print('M470_H4M3_PLAN',json.dumps(h4m3_plan(),sort_keys=True))
            return
        except (RuntimeError,ValueError,KeyError,OSError) as exc:
            p.error(str(exc))
    if args.h4m3_static_archives or args.h4m3_historical_archive is not None:
        p.error('M470_H4M3_FLAG_REQUIRED')
    if args.h4cc_technical:
        try:
            h4cc_tech_require(not any(getattr(args,k) for k in ('h4cc_eval','h4_m1_eval',
                'h4_technical','h3_m1_eval','h2a','h2b','checkpoint_p','report_p',
                'checkpoint_h3','report_h3')) and all(getattr(args,k)==p.get_default(k)
                for k in ('seeds','profiles','directions','push_step','push_magnitude',
                          'second_push_step','second_push_magnitude','max_steps','max_cases')),
                'FROZEN_FLAGS')
            h4cc_tech_require(args.output.absolute()==H4_EVAL_ROOT/H4CC_TECH_OUTPUT
                and args.output.resolve()==H4_EVAL_ROOT/H4CC_TECH_OUTPUT,'OUTPUT_PATH')
            if args.mode=='plan':
                print('M470_H4CC_TECH_PLAN',json.dumps(H4CC_TECH_SPEC,sort_keys=True))
                return
            h4cc_tech_authorization()
        except RuntimeError as exc:
            p.error(str(exc))
        return h4cc_tech_run(args)
    if args.h4cc_restore_report is not None or args.h4cc_static_log is not None:
        p.error('M470_H4CC_TECH_FLAG_REQUIRED')
    if args.h4cc_eval:
        try:
            plan=h4cc_eval_plan(args)
        except RuntimeError as exc:
            p.error(str(exc))
        if args.mode=='plan':
            print('M470_H4CC_EVAL_PLAN',json.dumps(plan,sort_keys=True))
            return
        try:
            h4cc_eval_authorization()
        except RuntimeError as exc:
            p.error(str(exc))
        args.seeds=[int(x) for x in args.seeds.split(',')]
        args.profiles=args.profiles.split(',')
        args.directions=args.directions.split(',')
        return h4e_run(args)
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
