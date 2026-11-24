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
import hashlib
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


def rc1_collect_step(env, state, action, previous_action,
                     phase_for_reward, reward_keys, jax):
    """Read-only RC1 telemetry; reward terms are weighted, pre-dt."""
    weighted = {
        key: float(value)
        for key, value in jax.device_get({
            key: state.metrics[f"reward/{key}"]
            for key in reward_keys
        }).items()
    }

    dt = float(env.dt)
    actual_reward = float(state.reward)
    reconstructed = dt * sum(weighted.values())

    if not all(math.isfinite(v) for v in
               [actual_reward, reconstructed, *weighted.values()]):
        raise RuntimeError("RC1_NONFINITE_REWARD")

    error = abs(actual_reward - reconstructed)
    if error > max(2e-5, 1e-5 * abs(actual_reward)):
        raise RuntimeError(f"RC1_REWARD_SUM_MISMATCH {error}")

    act = jax.device_get(action).tolist()
    prev = jax.device_get(previous_action).tolist()
    if len(act) != len(prev):
        raise RuntimeError("RC1_ACTION_SHAPE_CHANGED")

    delta = [float(x-y) for x, y in zip(act, prev)]

    feet = state.data.site_xpos[env._feet_site_id]
    pelvis_r = state.data.site_xmat[env._pelvis_imu_site_id]
    relative_foot = jax.device_get(
        pelvis_r.T @ (feet[0] - feet[1])
    ).tolist()

    contacts = jax.device_get(
        state.info["last_contact"]
    ).tolist()

    if len(contacts) != 2 or len(relative_foot) != 3:
        raise RuntimeError("RC1_FOOT_LAYOUT_CHANGED")

    return {
        "timing": "poststep_state_reward__prestep_phase",
        "weighted_reward_pre_dt": weighted,
        "reward_step": actual_reward,
        "reward_reconstructed": reconstructed,
        "reward_sum_error": error,
        "dt": dt,
        "action": act,
        "action_norm": math.sqrt(sum(float(x*x) for x in act)),
        "action_delta_norm": math.sqrt(sum(x*x for x in delta)),
        "contact_left_right": [bool(x) for x in contacts],
        "foot_left_minus_right_pelvis_xyz": relative_foot,
        "phase_for_reward": jax.device_get(
            phase_for_reward
        ).tolist(),
        "root_z_poststep": float(state.data.qpos[2]),
        "torso_gravity_z_poststep": float(
            env.get_gravity(state.data, "torso")[-1]
        ),
        "pelvis_local_xy_poststep": jax.device_get(
            env.get_local_linvel(state.data, "pelvis")[:2]
        ).tolist(),
        "pelvis_gyro_yaw_poststep": float(
            env.get_gyro(state.data, "pelvis")[2]
        ),
    }


def m470_ab_preflight(which):
    """Frozen C401M / original +10M / H1-M1 +10M; no JAX, no training."""
    from m431_seed1_matched_eval import validate, report_final, SEED1

    frozen = validate()
    c_sha = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
    shas = {
        'C401M': c_sha,
        'OLD10M': '0a011b6a3011f126a29832e4faa18972d7fbdddef77b63414219a15abab8314a',
        'M1_10M': 'd8775da6962ddf43ec509e8734220b32f8719763a5a334963886ea61ecafb5aa',
        'M2_10M': 'd3d5d0e7c8b7d8cd53d072d94b9efd604ed934716304f2d2ecda65562426fae7',
    }
    old_dir = ROOT / 'm463_runs' / 'm470_c401m_independent_10M_20261122_095207_043793917'
    m1_dir = ROOT / 'm463_runs' / 'm470_h1m1_independent_10M_20261123_144920_302107870'
    m2_dir = (
        ROOT / 'm463_runs'
        / 'm470_h1m2_independent_10M_20261124_104054_091938368'
    )
    if which == 'C401M':
        report = json.loads(frozen['report'].read_text(encoding='utf8'))
        entry, checkpoint_path = report_final(report, SEED1)
        if entry['sha256'] != c_sha:
            raise RuntimeError('M470_AB_C401M_SHA_MISMATCH')
    else:
        root = {
            'OLD10M': old_dir,
            'M1_10M': m1_dir,
            'M2_10M': m2_dir,
        }[which]
        report_path = root / 'report.json'
        if report_path.is_symlink() or not report_path.is_file():
            raise RuntimeError('M470_AB_REPORT_MISSING ' + str(report_path))
        report = json.loads(report_path.read_text(encoding='utf8'))
        old_path = old_dir / 'report.json'
        if not old_path.is_file():
            raise RuntimeError('M470_AB_ORIGINAL10M_REPORT_MISSING')
        old = json.loads(old_path.read_text(encoding='utf8'))
        required = {
            'status': 'M463_COVERAGE_LONGTRAIN_PASS',
            'independent_budget': '10M',
            'experiment_design': 'INDEPENDENT_C401M_SINGLE_PROCESS',
            'trained_from': 'C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE',
            'source_checkpoint_sha256': c_sha,
            'training_seed': 1,
            'stage': 'easy_repeat_support',
            'steps_observed': 10485760,
            'rollouts_target': 40,
            'rollouts_per_epoch': 8,
        }
        for label, rep in (('OLD10M', old), (which, report)):
            for k, v in required.items():
                if rep.get(k) != v:
                    raise RuntimeError('M470_AB_REPORT_METADATA_MISMATCH ' + label + ' ' + k)
        if report.get('hardware_scaled_config') != old.get('hardware_scaled_config'):
            raise RuntimeError('M470_AB_PPO_CONFIG_DIFF')
        if report.get('config_fingerprint') != old.get('config_fingerprint'):
            raise RuntimeError('M470_AB_CONFIG_FINGERPRINT_DIFF')
        if bool(report.get('h1_m1_enabled', False)) != (which == 'M1_10M'):
            raise RuntimeError('M470_AB_M1_FLAG_MISMATCH')
        if which == 'M1_10M':
            if (report.get('h1_m1_reward_scale') != -0.25 or
                report.get('train_reward_config', {}).get('scales', {}).get('lin_vel_z') != -0.25 or
                report.get('coverage_gate_pass') is not True or
                report.get('weights_finite') is not True):
                raise RuntimeError('M470_AB_M1_REPORT_GATE_INVALID')
        if which == 'OLD10M' and report.get('reward_PD_termination_source_unchanged') is not True:
            raise RuntimeError('M470_AB_OLD10M_REWARD_CHANGED')
        if bool(report.get('h1_m2_enabled', False)) != (
            which == 'M2_10M'
        ):
            raise RuntimeError('M470_AB_M2_FLAG_MISMATCH')

        if which == 'M2_10M':
            expected_report_sha = (
                '06a87b939b4b36f112808e56992606aaa'
                '441f721f913149dd4d75b0222e42c7b'
            )

            actual_report_sha = hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest()

            if actual_report_sha != expected_report_sha:
                raise RuntimeError(
                    'M470_AB_M2_REPORT_SHA_MISMATCH'
                )

            scales = report.get(
                'train_reward_config', {}
            ).get('scales', {})

            if (
                report.get('h1_m1_enabled') is not False
                or report.get('h1_m2_enabled') is not True
                or report.get('h1_m2_reward_slot') != 'feet_clearance'
                or report.get('h1_m2_reward_scale') != -0.25
                or len(scales) != 24
                or scales.get('feet_clearance') != -0.25
                or scales.get('lin_vel_z') != 0.0
                or report.get('cumulative_steps_start') != 0
                or report.get('resume_fullstate_sha256') is not None
                or report.get('coverage_gate_pass') is not True
                or report.get('memory_gate_pass') is not True
                or report.get('weights_finite') is not True
                or report.get('callback_vs_final_equal') is not True
                or report.get('eval_source_reward_unchanged') is not True
                or report.get('PD_termination_source_unchanged') is not True
            ):
                raise RuntimeError(
                    'M470_AB_M2_REPORT_GATE_INVALID'
                )

            audit = report.get(
                'h1_m2_training_transition_audit', {}
            )

            active = float(audit.get('active_steps', 0))
            penalty = float(
                audit.get('sum_weighted_pre_dt', 0)
            )

            if (
                not math.isfinite(active)
                or not math.isfinite(penalty)
                or active <= 0
                or penalty >= 0
                or audit.get('nonfinite_steps') != 0
            ):
                raise RuntimeError(
                    'M470_AB_M2_TRANSITION_AUDIT_INVALID'
                )
        if not report.get('checkpoints'):
            raise RuntimeError('M470_AB_NO_CHECKPOINTS')
        entry = report['checkpoints'][-1]
        if (int(entry['steps']) != 10485760 or
            entry['sha256'] != shas[which] or
            Path(entry['file']).name != entry['file']):
            raise RuntimeError('M470_AB_FINAL_CHECKPOINT_METADATA')
        checkpoint_path = root / entry['file']
        if (not checkpoint_path.is_file() or checkpoint_path.is_symlink() or
            checkpoint_path.resolve().parent != root.resolve()):
            raise RuntimeError('M470_AB_CHECKPOINT_PATH')
        h = hashlib.sha256()
        with checkpoint_path.open('rb') as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != shas[which]:
            raise RuntimeError('M470_AB_CHECKPOINT_SHA256')
    if which == 'C401M':
        network_report = report
    else:
        network_report = {
            'scaled_config': {
                'network_factory': report['hardware_scaled_config']['network_factory']
            }
        }
    print('M470_AB_CHECKPOINT_PREFLIGHT_PASS', which, shas[which],
          'checkpoint', checkpoint_path, flush=True)
    return {
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': shas[which],
        'report': network_report,
        'm470_policy': which,
    }


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
    from m426_heldout_det_stoch import restore_params, classify_done

    if jax.default_backend() != 'gpu':
        raise RuntimeError('M435_REQUIRES_GPU_BACKEND')
    if OUT.exists() and list(OUT.glob('m435*.json')):
        print('M435_NOTE existing output retained; new run creates a distinct timestamped file', flush=True)

    probe_cfg = locomotion.get_default_config(ENV)
    if bool(probe_cfg.push_config.enable) is not True:
        raise RuntimeError('PINNED_SOURCE_DEFAULT_PUSH_NOT_ENABLED; refusing A/B')

    reward_keys = None
    if args.rc1_telemetry:
        reward_keys = tuple(
            probe_cfg.reward_config.scales.keys()
        )
        if len(reward_keys) != 24:
            raise RuntimeError("RC1_REWARD_KEYS_CHANGED")

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

                if args.rc1_telemetry:
                    rc1_previous_action = state.info["last_act"]
                    rc1_phase = state.info["phase"]

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
                if ctx.get('m470_policy') is not None:
                    frame['root_z_poststep'] = float(state.data.qpos[2])
                
                if args.rc1_telemetry:
                    frame["rc1"] = rc1_collect_step(
                        env, state, action,
                        rc1_previous_action, rc1_phase,
                        reward_keys, jax
                    )

                frames.append(frame)
                if step_index % 200 == 0 or done:
                    print(f'M435_PROGRESS case={case} push={push_enabled} step={step_index} root_z={root_height:.3f} done={done}', flush=True)
                if done:
                    break

            if not push_enabled and push_events:
                raise RuntimeError('PUSH_OFF_ARM_STILL_HAS_PUSH_EVENTS')
            last = frames[-1]
            terminal_cause = classify_done(env, state.data) if done else None
            if done and not terminal_cause['recomputed_done']:
                raise RuntimeError('M470_AB_DONE_PREDICATE_UNEXPLAINED')
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
                'terminal_source_cause': terminal_cause,
            }

            if args.rc1_telemetry:
                diag["rc1_end_category"] = (
                    "source_failure"
                    if done else "evaluation_horizon_censored"
                )
                diag["rc1_source_truncation_signal"] = (
                    float(state.info["truncation"])
                    if "truncation" in state.info else None
                )

            if ctx.get('m470_policy') is not None and push_enabled and push_events:
                pstep = push_events[0]['step']
                diag['first_push_recovery'] = {}
                for horizon, tag in ((100, '2s'), (250, '5s')):
                    window = [f for f in frames if pstep <= f['step'] <= pstep + horizon]
                    survived = len(frames) >= pstep + horizon and not (
                        done and len(frames) <= pstep + horizon)
                    diag['first_push_recovery'][tag] = {
                        'target_steps': horizon,
                        'observed_steps': len(window),
                        'survived': survived,
                        'censored': not survived,
                        'xy_rmse': (math.sqrt(mean([
                            sum((f['actual_xy'][k]-CMD[case][k])**2 for k in range(2))
                            for f in window])) if window else None),
                        'yaw_rmse': (math.sqrt(mean([
                            (f['actual_yaw']-CMD[case][2])**2
                            for f in window])) if window else None),
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
        comparison = {'case': case,
                      'push_on_steps': on[0]['steps'],
                      'push_off_steps': off[0]['steps'],
                      'difference_off_minus_on': off[0]['steps']-on[0]['steps'],
                      'push_on_events': on[0]['push_event_steps']}
        if ctx.get('m470_policy') is not None:
            first_push = (on[0]['push_event_steps'][0]
                          if on[0]['push_event_steps'] else args.max_steps + 1)
            pre_steps = min(first_push - 1,
                            len(on[0]['frames']), len(off[0]['frames']))
            if pre_steps < 1:
                raise RuntimeError('M470_AB_NO_MATCHED_PREPUSH_HISTORY')
            diff = 0.
            first_pre_push_mismatch_step = None
            for index, (a, b) in enumerate(zip(
                    on[0]['frames'][:pre_steps],
                    off[0]['frames'][:pre_steps]), start=1):
                observables_a = (a['root_z'], *a['actual_xy'], a['actual_yaw'])
                observables_b = (b['root_z'], *b['actual_xy'], b['actual_yaw'])
                current_diff = max(abs(x-y) for x, y in zip(
                    observables_a, observables_b))
                diff = max(diff, current_diff)
                if current_diff > 1.e-5 and first_pre_push_mismatch_step is None:
                    first_pre_push_mismatch_step = index
            comparison['prepush_observable_steps'] = pre_steps
            comparison['prepush_observable_max_abs_diff'] = diff
            comparison['prepush_first_mismatch_step'] = first_pre_push_mismatch_step
            comparison['prepush_observables_identical'] = first_pre_push_mismatch_step is None
            comparison['on_off_pair_status'] = (
                'PREPUSH_OBSERVABLE_MATCH_ONLY_NOT_FULLSTATE'
                if first_pre_push_mismatch_step is None
                else 'PREPUSH_NONIDENTICAL_NOT_STRICT_COUNTERFACTUAL')
            if first_pre_push_mismatch_step is not None:
                # No silent pass: store a disqualifying matched-ON/OFF quality flag,
                # but still evaluate the main same-push-ON policy comparison.
                print('M470_AB_PREPUSH_NONIDENTICAL', 'case', case,
                      'first_step', first_pre_push_mismatch_step,
                      'max_diff', diff,
                      'ON_OFF_CAUSAL_MATCH_UNVERIFIED', flush=True)
        comparisons.append(comparison)
    OUT.mkdir(parents=True, exist_ok=True)
    outfile = OUT / ('m435_push_ab_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    data = {
        'status': 'M435_PUSH_AB_COMPLETED',
        'checkpoint_sha256': ctx['checkpoint_sha256'],
        'm470_ab_policy': ctx.get('m470_policy'),
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
        'all_on_off_prepush_observables_identical': (
            all(c.get('prepush_observables_identical') is True
                for c in comparisons) if ctx.get('m470_policy') is not None
            else None),
        'strict_on_off_counterfactual_proven': False,
        'rollouts': results,
    }

    if args.rc1_telemetry:
            data["rc1_telemetry_enabled"] = True
            data["rc1_reward_keys"] = list(reward_keys)
            data["rc1_notes"] = (
                "24 weighted pre-dt reward terms; original source push; "
                "evaluation horizon is not PPO time-limit truncation; "
                "push magnitude not available"
            )
    
    with outfile.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print('M435_PUSH_AB_COMPLETED',flush=True)
    print('OUTPUT:',outfile,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('inspect','run'))
    parser.add_argument('--m470-policy', choices=('C401M','OLD10M','M1_10M','M2_10M'))

    parser.add_argument(
        "--rc1-telemetry",
        action="store_true",
        help="Read-only reward/action/foot telemetry for RC1"
    )

    parser.add_argument('--train-seed', type=int, choices=(0,1), default=1)
    parser.add_argument('--seed', type=int, default=5186)
    parser.add_argument('--cases', default='stand,backward,yaw_right')
    parser.add_argument('--max-steps', type=int, default=1000)
    args = parser.parse_args()

    if args.rc1_telemetry and (
        args.mode != "run"
        or args.m470_policy not in ("C401M", "OLD10M")
    ):
        parser.error(
            "RC1_REQUIRES_RUN_WITH_C401M_OR_OLD10M"
        )

    cases = args.cases.split(',')
    if not cases or len(set(cases)) != len(cases) or any(x not in CMD for x in cases):
        parser.error('cases must be unique, comma-separated names: ' + ','.join(CMD))
    if args.max_steps < 10 or args.max_steps > 1000:
        parser.error('--max-steps must be from 10 to 1000')
    if args.m470_policy is not None:
        if args.train_seed != 1 or args.max_steps != 1000:
            parser.error('M470_AB_REQUIRES_TRAIN_SEED_1_AND_1000_STEPS')
        ctx = m470_ab_preflight(args.m470_policy)
    else:
        from m433_visual_validation import preflight
        ctx = preflight(args.train_seed, args.seed, 'deterministic')
    if args.mode == 'inspect':
        print('M435_INSPECT_PASS NO_ROLLOUT NO_TRAINING',flush=True)
        print('CASE_LIST:',cases,'CHECKPOINT_SHA256:',ctx['checkpoint_sha256'],flush=True)
        return
    run(args,ctx,cases)

if __name__ == '__main__':
    main()
