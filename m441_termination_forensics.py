#!/usr/bin/env python3
"""M4.41 | G1 source-exact termination-cause forensics (MJX, no training).

PAPER-FAITHFUL CONTEXT: existing pinned G1JoystickFlatTerrain, 401M checkpoint,
original source, PPO/reward/PD/termination unchanged. This is a DIAGNOSTIC:
constant commands replace sampled commands, and configured push magnitude is
changed only for each rollout at environment construction. It also records the
SIX original source termination predicates, without changing their logic.

Modes:
  inspect  validate checkpoint/source/input only; no MJX rollout
  run      per-seed sweep, incremental timestamped JSON output
  compare  offline aggregation of completed distinct-seed JSONs

Push magnitude is the world-XY velocity increment configured in source
`qvel[:2] += unit_push_direction * push_magnitude`; this is NOT force (N).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from m435_push_ab_stability import CMD

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'm441_push_threshold'
SRC_SHA = '9d3b6d04fbc86c99f1ebf69bb7e3fc7ae11dbd8588ec3fd2daf4d8e28a41a5a4'
CHECKPOINT_SHA = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
DEFAULT_MAGS = (0., 0.25, 0.5, 1., 1.5, 2.)
EXPECTED_SOURCE_VERSION = '9bcbe736177472ba35d60773e3030a36e37962b7'


def parse_magnitudes(value):
    """Baseline is OFF (0); positive arms use [m, m] with source push ON."""
    try:
        numbers = tuple(float(token.strip()) for token in value.split(','))
    except ValueError as e:
        raise ValueError('Invalid numeric magnitude list') from e
    if (not numbers or numbers[0] != 0.0 or len(numbers) < 2 or
            any(not math.isfinite(x) or not 0 <= x <= 2.0 for x in numbers) or
            any(b <= a for a, b in zip(numbers,numbers[1:]))):
        raise ValueError('Magnitudes must start at zero, rise strictly, and lie in [0,2] m/s')
    return numbers


def parse_cases(value):
    out = value.split(',')
    if not out or len(out) != len(set(out)) or any(c not in CMD for c in out):
        raise ValueError('Cases must be distinct and in '+', '.join(CMD))
    return out


def validated_checkpoint(seed):
    from m433_visual_validation import preflight
    from paper_config import REPO, PIN
    if PIN != EXPECTED_SOURCE_VERSION:
        raise RuntimeError(f'M441_UNEXPECTED_PIN: {PIN}')
    p = REPO / 'mujoco_playground/_src/locomotion/g1/joystick.py'
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != SRC_SHA:
        raise RuntimeError('M441_PINNED_G1_SOURCE_SHA_MISMATCH')
    ctx = preflight(1, seed, 'deterministic')
    if ctx['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise RuntimeError('M441_CHECKPOINT_SHA_MISMATCH')
    return ctx


def finite(x):
    x = float(x)
    if not math.isfinite(x):
        raise RuntimeError(f'M441_NONFINITE_VALUE: {x!r}')
    return x


def mean(values):
    return statistics.fmean(values) if values else None


def summarize_episode(samples, command, events):
    if not samples:
        raise RuntimeError('M441_NO_SAMPLES')
    steps = len(samples)
    xy_rmse = math.sqrt(statistics.fmean(
        (s['v_body_xy_mps'][0]-command[0])**2 +
        (s['v_body_xy_mps'][1]-command[1])**2 for s in samples))
    yaw_rmse = math.sqrt(statistics.fmean(
        (s['yaw_rate_radps']-command[2])**2 for s in samples))
    event_response = []
    for e in events:
        # Samples are POST-step. Include the perturbation frame and 50 following.
        window = [s for s in samples if e['step'] <= s['step'] <= e['step']+50]
        event_response.append({
            'push_step':e['step'],
            'window_steps':len(window),
            'min_root_z_m_0_to_50_steps':min(x['root_z_m'] for x in window),
            'max_torso_tilt_deg_0_to_50_steps':max(x['torso_tilt_deg'] for x in window),
            'min_foot_floor_contacts_0_to_50_steps':min(x['feet_floor_contacts'] for x in window),
            'reaches_50_steps':len(window)==51,
        })
    terminal_flags = samples[-1]['termination_flags']
    triggered = [name for name, value in terminal_flags.items() if value]
    if bool(samples[-1]['done_after_step']) != bool(triggered):
        raise RuntimeError('M441_TERMINATION_FLAGS_DO_NOT_MATCH_DONE')
    return {
        'terminal_predicates':terminal_flags,
        'terminal_reason_codes':triggered,
        'steps':steps,
        'terminated':bool(samples[-1]['done_after_step']),
        'completed_max_steps':not samples[-1]['done_after_step'],
        'root_z_min_m':min(s['root_z_m'] for s in samples),
        'root_z_last_m':samples[-1]['root_z_m'],
        'torso_tilt_max_deg':max(s['torso_tilt_deg'] for s in samples),
        'mean_body_velocity_xy_mps':[mean([s['v_body_xy_mps'][i] for s in samples]) for i in range(2)],
        'mean_yaw_rate_radps':mean([s['yaw_rate_radps'] for s in samples]),
        'xy_rmse_mps':xy_rmse,
        'yaw_rmse_radps':yaw_rmse,
        'steps_with_no_foot_floor_contacts':sum(s['feet_floor_contacts']==0 for s in samples),
        'push_count':len(events),
        'push_event_steps':[e['step'] for e in events],
        'last_push_to_termination_steps':(
            steps - events[-1]['step']
            if samples[-1]['done_after_step'] and events else None),
        'post_push_50_step_windows':event_response,
    }


def atomic_json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.partial')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(args, ctx, cases, magnitudes):
    # Nvcc compatibility must run BEFORE JAX imports, including transitively.
    from m434_manual_command_viewer import patch_obs_and_info, validate_initial_observation
    from paper_config import BASE, ENV
    from train import check_source
    check_source()
    sys.path.insert(0, str(BASE / 'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('M441_NVCC_COMPAT', apply(), flush=True)

    import jax
    import jax.numpy as jnp
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import locomotion
    from mujoco_playground._src.collision import geoms_colliding
    from m426_heldout_det_stoch import restore_params

    if jax.default_backend() != 'gpu':
        raise RuntimeError('M441_GPU_REQUIRED')
    default_config = locomotion.get_default_config(ENV)
    if not bool(default_config.push_config.enable):
        raise RuntimeError('M441_DEFAULT_SOURCE_PUSH_DISABLED')
    original_range = tuple(float(x) for x in default_config.push_config.magnitude_range)
    if len(original_range)!=2 or original_range[0]>=original_range[1]:
        raise RuntimeError('M441_SOURCE_MAGNITUDE_RANGE_INVALID')
    baseline = locomotion.load(ENV, config=default_config)
    reference = jax.jit(baseline.reset)(jax.random.PRNGKey(999))
    validate_initial_observation(reference,jnp)
    network=ppo_networks.make_ppo_networks(
        observation_size=jax.tree_util.tree_map(lambda x:x.shape,reference.obs),
        action_size=baseline.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ctx['report']['scaled_config']['network_factory'])
    params=restore_params(ctx['checkpoint_path'].read_bytes(),
                          serialization,running_statistics,jax,jnp)
    policy=jax.jit(ppo_networks.make_inference_fn(network)(params,deterministic=True))

    outfile = OUT / ('m441_seed'+str(args.seed)+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')
    result = {
        'status':'M441_IN_PROGRESS',
        'experiment_class':'TERMINATION_CAUSE_DIAGNOSTIC_NOT_PAPER_BENCHMARK',
        'training_repeated':False,
        'checkpoint_sha256':ctx['checkpoint_sha256'],
        'checkpoint_step':401080320,
        'source_commit':EXPECTED_SOURCE_VERSION,
        'source_sha256':SRC_SHA,
        'environment':ENV,
        'train_seed':1,
        'reset_seed':args.seed,
        'policy':'deterministic',
        'domain_randomization':False,
        'fixed_commands_injected':True,
        'original_source_default_push_magnitude_range_mps':list(original_range),
        'independent_variable':'push magnitude fixed at [m,m] for each rollout; m=0 disables push',
        'source_push_semantics':'added world-XY qvel in m/s, NOT force in newtons',
        'other_source_config_changed':False,
        'max_steps':args.max_steps,
        'cases':cases,
        'magnitudes_mps':list(magnitudes),
        'sample_timestamp':'POST physics step; push event and response share same step label',
        'termination_audit':'Original G1 source six component predicates, recomputed on same post-step mjx.Data and asserted equal state.done',
        'case_results':[],
    }
    atomic_json_write(outfile,result)
    for case in cases:
        zero_sample_prefix = None
        reference_nonzero_events = None
        reference_interval = None
        for m in magnitudes:
            config = locomotion.get_default_config(ENV)
            config.push_config.enable = m != 0.
            if m != 0.:
                # JAX uniform(minval=maxval) returns a fixed scalar without
                # altering any other source RNG scheduling.
                config.push_config.magnitude_range = [m,m]
            env=locomotion.load(ENV, config=config)
            if abs(float(env.dt)-0.02) > 1e-6:
                raise RuntimeError(f'M441_UNEXPECTED_PHYSICS_DT {env.dt}')
            if env.action_size != baseline.action_size:
                raise RuntimeError('M441_ACTION_SPACE_CHANGED')
            geoms = [int(i) for i in env._feet_geom_id]
            floor_id=int(env._floor_geom_id)
            torso_id=int(env._torso_imu_site_id)
            if len(geoms)!=2:
                raise RuntimeError('M441_FOOT_LAYOUT_CHANGED')
            def diagnostic(st):
                data=st.data
                torso_matrix=data.site_xmat[torso_id]
                tilt=jnp.arccos(jnp.clip(torso_matrix[2,2],-1.,1.))
                contact=jnp.stack([geoms_colliding(data,geom,floor_id) for geom in geoms])
                # EXACT predicates from pinned G1 source `_get_termination`.
                # Do not simplify geometries or replace these with root height.
                terminal_flags = {
                    'torso_upside_down': env.get_gravity(data, 'torso')[-1] < 0.,
                    'foot_foot_collision': geoms_colliding(
                        data, env._right_foot_geom_id, env._left_foot_geom_id),
                    'left_foot_right_shin_collision': geoms_colliding(
                        data, env._left_foot_geom_id, env._right_shin_geom_id),
                    'right_foot_left_shin_collision': geoms_colliding(
                        data, env._right_foot_geom_id, env._left_shin_geom_id),
                    'qpos_nan': jnp.isnan(data.qpos).any(),
                    'qvel_nan': jnp.isnan(data.qvel).any(),
                }
                return {
                    'termination_flags':terminal_flags,
                    'root_z':data.qpos[2],
                    'torso_tilt':tilt,
                    'velocity_xy':env.get_local_linvel(data,'pelvis')[:2],
                    'yaw_rate':env.get_gyro(data,'pelvis')[2],
                    'contacts':contact,
                }
            diagnostics=jax.jit(diagnostic)
            reset_fn=jax.jit(env.reset)
            step_fn=jax.jit(env.step)
            key=jax.random.PRNGKey(args.seed)
            key,reset_key=jax.random.split(key)
            state=reset_fn(reset_key)
            validate_initial_observation(state,jnp)
            interval=int(jax.device_get(state.info['push_interval_steps']))
            if interval < 1:
                raise RuntimeError('M441_PUSH_INTERVAL_NONPOSITIVE')
            if reference_interval is None:
                reference_interval = interval
            elif interval != reference_interval:
                raise RuntimeError('M441_PUSH_SCHEDULE_INTERVAL_CHANGED_BETWEEN_MAGNITUDES')
            command=jnp.asarray(CMD[case],dtype=jnp.float32)
            samples=[]
            events=[]
            print(f'M441_START seed={args.seed} case={case} push_mps={m:.2f} interval={interval}',flush=True)
            for step in range(1,args.max_steps+1):
                state=patch_obs_and_info(state,command,jnp)
                key,action_key=jax.random.split(key)
                action,_=policy(state.obs,action_key)
                if not bool(jnp.all(jnp.isfinite(action))):
                    raise RuntimeError(f'M441_NONFINITE_ACTION {case}/{m}/{step}')
                state=step_fn(state,action)
                measured=jax.device_get(diagnostics(state))
                z=finite(measured['root_z'])
                tilt=math.degrees(finite(measured['torso_tilt']))
                v=[finite(x) for x in measured['velocity_xy']]
                yaw=finite(measured['yaw_rate'])
                contacts=sum(bool(x) for x in measured['contacts'])
                push=[finite(x) for x in jax.device_get(state.info['push'])]
                detected=math.hypot(*push)>0.01
                expected=bool(m>0 and step % interval==0)
                if detected != expected:
                    raise RuntimeError(f'M441_SOURCE_PUSH_TIMING_MISMATCH {case}/{m}/{step}: {detected}!={expected}')
                if detected:
                    norm=math.hypot(*push)
                    if abs(norm-1)>1e-4:
                        raise RuntimeError('M441_PUSH_DIRECTION_NOT_UNIT')
                    events.append({'step':step,'direction_world_xy':push,
                                   'configured_velocity_impulse_mps':[m*x for x in push],
                                   'configured_magnitude_mps':m})
                    print(f'M441_PUSH case={case} magnitude={m:.2f} step={step}',flush=True)
                done=bool(state.done)
                predicates = {name:bool(value) for name,value in
                              measured['termination_flags'].items()}
                if done != any(predicates.values()):
                    raise RuntimeError(f'M441_SOURCE_TERMINATION_PREDICATE_MISMATCH '
                                       f'case={case} m={m} step={step} '
                                       f'done={done} predicates={predicates}')
                samples.append({'step':step,'root_z_m':z,'torso_tilt_deg':tilt,
                                'termination_flags':predicates,
                                'v_body_xy_mps':v,'yaw_rate_radps':yaw,
                                'feet_floor_contacts':contacts,
                                'push_event_this_step':detected,
                                'done_after_step':done})
                if done:
                    break
            # Same fixed seed means exactly the same pre-first-push trajectory.
            # Failure here means the planned matched comparison is invalid.
            pre_first_end=min(interval-1,len(samples),20)
            prefix=[(x['root_z_m'],x['v_body_xy_mps'][0],x['v_body_xy_mps'][1])
                    for x in samples[:pre_first_end]]
            if m==0:
                zero_sample_prefix=prefix
            elif len(prefix)!=len(zero_sample_prefix) or any(
                    abs(a-b)>1e-4 for t0,t1 in zip(zero_sample_prefix,prefix)
                    for a,b in zip(t0,t1)):
                raise RuntimeError(f'M441_PRE_FIRST_PUSH_TRAJECTORY_MISMATCH {case}/{m}')
            # The sampled direction and schedule are a matched seed-specific
            # intervention; verify the common observed event prefix explicitly.
            direction_overlap = 0
            if m > 0:
                if reference_nonzero_events is None:
                    reference_nonzero_events = events
                else:
                    for e, ref in zip(events,reference_nonzero_events):
                        if e['step'] != ref['step'] or any(
                            abs(x-y)>1e-5 for x,y in zip(e['direction_world_xy'],ref['direction_world_xy'])
                        ):
                            raise RuntimeError(f'M441_PUSH_DIRECTION_OR_SCHEDULE_CHANGED {case}/{m}')
                        direction_overlap += 1
            diag=summarize_episode(samples,CMD[case],events)
            diag.update({'case':case,'command':list(CMD[case]),
                         'matching_direction_events_checked':direction_overlap,
                         'push_magnitude_mps':m,'push_enabled':m>0,
                         'push_interval_steps':interval,
                         'pre_first_push_prefix_matches_zero_magnitude':True,
                         'push_events':events,'samples':samples})
            result['case_results'].append(diag)
            atomic_json_write(outfile,result)
            print(f'M441_RESULT seed={args.seed} case={case} mag={m:.2f} '
                  f'steps={diag["steps"]} terminated={diag["terminated"]} '
                  f'min_z={diag["root_z_min_m"]:.3f} max_tilt={diag["torso_tilt_max_deg"]:.1f} '
                          f'terminal_reasons={diag["terminal_reason_codes"]}',flush=True)
    result['status']='M441_PUSH_THRESHOLD_COMPLETED'
    atomic_json_write(outfile,result)
    print('M441_PUSH_THRESHOLD_COMPLETED',flush=True)
    print('OUTPUT:',outfile,flush=True)


def compare(args):
    if not args.inputs:
        raise ValueError('M441_COMPARE_INPUTS_REQUIRED')
    reports=[json.loads(Path(p).read_text(encoding='utf-8')) for p in args.inputs]
    seeds=[]
    for r in reports:
        if r.get('status')!='M441_PUSH_THRESHOLD_COMPLETED':
            raise RuntimeError('M441_COMPARE_REJECTS_PARTIAL_INPUT')
        if r.get('checkpoint_sha256')!=CHECKPOINT_SHA or r.get('source_sha256')!=SRC_SHA:
            raise RuntimeError('M441_COMPARE_PROVENANCE_MISMATCH')
        if r.get('experiment_class')!='TERMINATION_CAUSE_DIAGNOSTIC_NOT_PAPER_BENCHMARK':
            raise RuntimeError('M441_COMPARE_CLASS_MISMATCH')
        if r.get('training_repeated') is not False:
            raise RuntimeError('M441_COMPARE_TRAINING_FLAG_INVALID')
        seed=r['reset_seed'];seeds.append(seed)
        if (r['cases']!=reports[0]['cases'] or
                r['magnitudes_mps']!=reports[0]['magnitudes_mps'] or
                r['max_steps']!=reports[0]['max_steps']):
            raise RuntimeError('M441_COMPARE_DIFFERENT_PROTOCOLS')
        expected={(c,float(m)) for c in r['cases'] for m in r['magnitudes_mps']}
        actual={(x['case'],float(x['push_magnitude_mps'])) for x in r['case_results']}
        if len(actual)!=len(r['case_results']) or expected!=actual:
            raise RuntimeError('M441_COMPARE_MISSING_DUPLICATE_ARM')
    if len(set(seeds))!=len(seeds):
        raise RuntimeError('M441_COMPARE_DUPLICATE_RESET_SEED')
    aggregate=[]
    for case in reports[0]['cases']:
        for m in reports[0]['magnitudes_mps']:
            rows=[next(x for x in r['case_results']
                       if x['case']==case and x['push_magnitude_mps']==m) for r in reports]
            completed=sum(x['completed_max_steps'] and not x['terminated'] for x in rows)
            aggregate.append({
                'case':case,'magnitude_mps':m,
                'episodes':len(rows),
                'completed_count':completed,
                'completion_fraction':completed/len(rows),
                'mean_survival_steps':mean([x['steps'] for x in rows]),
                'min_survival_steps':min(x['steps'] for x in rows),
                'mean_root_z_min_m':mean([x['root_z_min_m'] for x in rows]),
                'mean_xy_rmse_mps':mean([x['xy_rmse_mps'] for x in rows]),
                'mean_yaw_rmse_radps':mean([x['yaw_rmse_radps'] for x in rows]),
                'by_seed':[{ 'seed':seed,'steps':x['steps'],
                             'terminated':x['terminated'],
                             'terminal_reason_codes':x['terminal_reason_codes'],
                             'push_event_steps':x['push_event_steps']}
                           for seed,x in zip(seeds,rows)],
            })
    result={'status':'M441_PUSH_THRESHOLD_COMPARISON_COMPLETED',
            'note':'Fixed-magnitude, small-N controlled diagnostic. Not the original uniform random push distribution; not paper benchmark; no universal safe threshold inferred.',
            'checkpoint_sha256':CHECKPOINT_SHA,'source_sha256':SRC_SHA,
            'reset_seeds':seeds,'num_seeds':len(seeds),
            'cases':reports[0]['cases'],
            'magnitudes_mps':reports[0]['magnitudes_mps'],
            'aggregate':aggregate}
    outfile=OUT/('m441_comparison_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')
    atomic_json_write(outfile,result)
    print('M441_COMPARISON_TABLE (survival / completed N)',flush=True)
    for x in aggregate:
        print(f"{x['case']:>10} mag={x['magnitude_mps']:>4.2f} "
              f"mean_steps={x['mean_survival_steps']:>7.1f} "
              f"completed={x['completed_count']}/{x['episodes']}",flush=True)
    print('M441_PUSH_THRESHOLD_COMPARISON_COMPLETED',flush=True)
    print('OUTPUT:',outfile,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inspect','run','compare'),required=True)
    parser.add_argument('--seed',type=int,default=5001)
    parser.add_argument('--cases',default='stand,backward,yaw_right')
    parser.add_argument('--magnitudes',default=','.join(str(x) for x in DEFAULT_MAGS))
    parser.add_argument('--max-steps',type=int,default=1000)
    parser.add_argument('--inputs',nargs='*',default=[])
    args=parser.parse_args()
    if args.mode=='compare':
        compare(args)
        return
    try:
        cases=parse_cases(args.cases)
        magnitudes=parse_magnitudes(args.magnitudes)
    except ValueError as e:
        parser.error(str(e))
    if not (10<=args.max_steps<=1000):
        parser.error('max-steps must be in [10,1000]')
    ctx=validated_checkpoint(args.seed)
    print(f'M441_PREFLIGHT_PASS seed={args.seed} checkpoint={ctx["checkpoint_sha256"][:16]} '
          f'cases={cases} mag={list(magnitudes)}',flush=True)
    if args.mode=='inspect':
        print('M441_INSPECT_PASS NO_JAX_ROLLOUT NO_TRAINING',flush=True)
    else:
        run(args,ctx,cases,magnitudes)


if __name__=='__main__':
    main()
