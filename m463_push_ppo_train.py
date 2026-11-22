#!/usr/bin/env python3
"""M4.63 source-preserving push-coverage PPO, with mandatory TRAINING exposure gate.

Run only with pinned local project, venv and NVCC compatibility overlays.
Stage 1: easy (0.20-0.60 m/s), 20% nominal episodes, 6 motion commands,
6 push directions, and phase-aware push trigger. Stage 2/3 definitions are
PROTOTYPES, not automatic optimizer-state continuation.

--mode plan      : offline safe plan
--mode smoke     : 1 PPO rollout of 262,144 total steps (32/env)
--mode pilot     : 20 PPO rollouts, ONLY after same-stage verified smoke report

Both smoke and pilot are weights-only warm-start from SAME 401M checkpoint.
No overwrite to baseline or the pinned Playground/Brax source.
"""
from __future__ import annotations

import argparse
import copy
import csv
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PIN = '9bcbe736177472ba35d60773e3030a36e37962b7'
SEED1_SHA = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
TOTAL_STEPS_BATCH = 8192*32
BATCHES={'smoke':1,'probe2': 2,'smoke3':3,'pilot':20}
PREFIX='training/m463_'


def digest(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as fh:
        for buf in iter(lambda:fh.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()


def ensure_smoke_gate(path, stage, orbax_sha):
    if not path or not path.is_file():
        raise RuntimeError('M463_PILOT_NEEDS_SMOKE_REPORT_SAME_STAGE')
    r=json.loads(path.read_text('utf8'))
    if any((r.get('status')!='M463_COVERAGE_SMOKE_PASS',r.get('stage')!=stage,
            r.get('orbax_sha256')!=orbax_sha, r.get('steps_observed')!=TOTAL_STEPS_BATCH,
            not r.get('coverage_gate_pass'))):
        raise RuntimeError('M463_PILOT_SMOKE_GATE_INVALID')
    if r.get('gpu_peak_mib') is None or r.get('gpu_total_mib') is None:
        raise RuntimeError('M463_MISSING_SMOKE_GPU_METRIC')
    if r['gpu_peak_mib']>=r['gpu_total_mib']*0.95:
        raise RuntimeError('M463_SMOKE_VRAM_TOO_HIGH')
    return r


def audit_coverage(history, batches_by_eval):
    """Brax averages train metrics over rollout batches within one eval period.

    Reconstruct actual *observed* event counts by multiplying that mean
    with exact rollout count per eval period, not by PPO gradient updates.
    """
    counters={}
    for row in history:
        if row['additional_steps']==0:
            continue
        m=row['metrics']
        for key,v in m.items():
            if key.startswith(PREFIX):
                counters[key]=counters.get(key,0.0)+float(v)*batches_by_eval
    expect=PREFIX+'events_per_rollout'
    if expect not in counters:
        raise RuntimeError('M463_NO_TRAIN_ROLLOUT_COVERAGE_METRICS')
    n_ev=counters[expect]
        # M4.70-J: event counts are additive, rates are not.
    n_batches = batches_by_eval * sum(
        row['additional_steps'] > 0
        for row in history
    )

    if n_batches <= 0:
        raise RuntimeError('M470J_NO_OBSERVED_BATCHES')

    for suffix in (
        'events_per_1000_steps',
        'episodes_no_push_fraction_per_step',
    ):
        key = PREFIX + suffix
        if key in counters:
            counters[key] /= n_batches
    # Strict minimum for one 8192-env batch. This guards both zero exposure
    # and severe coverage holes due to early termination/reset.
    min_events=100.0 if batches_by_eval==1 else 1000.0
    checks={'nonzero_and_minimum_push_events': n_ev>=min_events}
    for i in range(6):
        checks[f'direction_{i}_exposed']=counters.get(PREFIX+f'direction_{i}_per_rollout',0)>0
        checks[f'motion_{i}_exposed']=counters.get(PREFIX+f'motion_{i}_per_rollout',0)>0
    for i in range(3):
        checks[f'phase_target_{i}_exposed']=counters.get(PREFIX+f'phase_target_{i}_per_rollout',0)>0
    checks['some_phase_matched']=counters.get(PREFIX+'phase_matched_per_rollout',0)>0
    checks['sum_magnitudes_positive']=counters.get(PREFIX+'push_magnitude_sum',0)>0
    checks['direction_sum_reconciles']=abs(sum(counters.get(PREFIX+f'direction_{i}_per_rollout',0) for i in range(6))-n_ev)<0.1
    checks['motion_sum_reconciles']=abs(sum(counters.get(PREFIX+f'motion_{i}_per_rollout',0) for i in range(6))-n_ev)<0.1
    return {'push_events_observed':n_ev,'counts':counters,
            'checks':checks,'passed':all(checks.values()),
            'notes':'Events counted from Brax PPO transition extras, before SGD; no inference from configured push interval.'}


def run(args):
    if args.output_dir.exists():
        raise FileExistsError('M463_REFUSE_TO_OVERWRITE_OUTPUT_DIRECTORY')
    # Import from PRIOR, already proven checkpoint validator and GPU monitor.
    from m462_ppo_warmstart_run import (read_restore_gate, sample_gpu,
                                        GPUMonitor, check_source_and_config)
    from m463_instrument_brax import build_instrumented_ppo

    report_gate,orbax_dir=read_restore_gate(args.restore_report,args.checkpoint)
    if args.mode in ('probe2', 'pilot'):
        ensure_smoke_gate(args.smoke_report,args.stage,report_gate['orbax_dir_sha256'])
    gpu_initial=sample_gpu()
    # This checks source pin, GPU, Brax source SHA, config, domain randomizer.
    api=check_source_and_config('A')
    jax,np=api['jax'],api['numpy']
    from flax import serialization
    from m463_coverage_env import CoverageJoystick, STAGES
    if jax.default_backend()!='gpu':raise RuntimeError('M463_GPU_REQUIRED')
    if args.stage not in STAGES:raise RuntimeError('M463_STAGE_UNKNOWN')
    source_cfg=api['source_cfg']
    train_cfg=copy.deepcopy(source_cfg)
    train_cfg.push_config.enable=False  # avoid source push + explicit push stacking
    if not bool(source_cfg.push_config.enable):
        raise RuntimeError('M463_ORIGINAL_SOURCE_PUSH_WAS_NOT_ENABLED')
    if source_cfg.to_dict()['reward_config']!=train_cfg.to_dict()['reward_config']:
        raise RuntimeError('M463_REWARD_CHANGED')
    train_env=CoverageJoystick(config=train_cfg,stage=args.stage)
    # Original source-driven randomized evaluation; only 2 envs as previous.
    eval_env=api['locomotion'].load(api['env_name'],config=source_cfg)
    scaled=api['scaled']
    scaled.num_timesteps=BATCHES[args.mode]*TOTAL_STEPS_BATCH
    scaled.num_evals = 2 if args.mode in ('smoke', 'probe2', 'smoke3') else 6
    assert scaled.num_envs==8192 and scaled.num_minibatches==8
    assert scaled.unroll_length==32 and scaled.batch_size==1024
    assert scaled.num_updates_per_batch==5 and scaled.action_repeat==1
    cfg=scaled.to_dict()
    if args.mode == 'smoke3' and cfg.get('num_resets_per_eval', 0) != 0:
        raise RuntimeError('M470_SMOKE3_REQUIRES_CONTINUOUS_ENV_STATE')
    net_cfg=cfg.pop('network_factory')
    cfg.update(seed=1,randomization_fn=api['randomizer'],
               restore_checkpoint_path=str(orbax_dir.resolve()))
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    instrumented_ppo,instrument_meta=build_instrumented_ppo(
        api['ppo'],out/'m463_brax_ppo_instrumented.py')
    report_path=out/'report.json'
    history=[];checkpoints=[]
    report={
        'status':'M463_RUNNING','stage':args.stage,'mode':args.mode,
        'experiment_class':'PUSH_COVERAGE_PPO_HARDWARE_SCALED_EXTENSION_NOT_PAPER_EXACT',
        'trained_from':'401M_WEIGHTS_ONLY_ADAM_RESTART','source_commit':PIN,
        'source_checkpoint_sha256':SEED1_SHA,
        'orbax_sha256':report_gate['orbax_dir_sha256'],
        'checkpoint_path':str(args.checkpoint.resolve()),
        'orbax_path':str(orbax_dir.resolve()),
        'training_seed':1,'train_steps_target':scaled.num_timesteps,
        'rollouts_target':BATCHES[args.mode],'hardware_scaled_config':scaled.to_dict(),
        'source_push_disabled_in_custom_train_env':True,
        'source_random_push_enabled_in_eval':True,
        'reward_PD_termination_source_unchanged':True,
        'commands_per_episode':['stand','forward_slow','forward_fast','turn_left','turn_right','strafe'],
        'directions_per_episode':['forward','back','left','right','diag_forward_left','diag_forward_right'],
        'phase_target':['any','single','double'],
        'stage_magnitude_range_mps':list(STAGES[args.stage][:2]),
        'stage_repeat_probability':STAGES[args.stage][2],
        'probability_nominal_episode':0.20,
        'first_push_target_step_range':[12,24],
        'max_phase_wait_steps':8,
        'source_push_semantics':'instantaneous_world_xy_qvel_delta_mps_NOT_force_N',
        'instrumented_brax':instrument_meta,
        'runner_sha256':digest(Path(__file__)),
        'gpu_initial':gpu_initial,
        'training_performed':False,'report_history':history,'checkpoints':checkpoints,
    }
    def persist():
        tmp=report_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
        tmp.replace(report_path)
    persist()
    def convert_metrics(metrics):
        out={}
        for k,v in metrics.items():
            arr=np.asarray(jax.device_get(v))
            if arr.dtype.kind not in 'biuf':
                out[k]={'shape':list(arr.shape),'dtype':str(arr.dtype)}
                continue
            if not np.all(np.isfinite(arr)):
                raise RuntimeError('M463_NONFINITE_METRICS '+k)
            if arr.ndim==0:out[k]=float(arr)
            else:out[k]={'shape':list(arr.shape),'mean':float(arr.mean()),
                         'min':float(arr.min()),'max':float(arr.max())}
        return out
    def progress(step,metrics):
        row={'additional_steps':int(step),'metrics':convert_metrics(metrics)}
        history.append(row);persist()
        print('M463_PROGRESS',args.stage,'steps',step,
              'reward',row['metrics'].get('eval/episode_reward'),
              'train_push_events_mean_batch',row['metrics'].get(PREFIX+'events_per_rollout'),flush=True)
    def save_policy(step,_make_policy,params):
        if int(step)==0:return
        name=f'm463_{args.stage}_{args.mode}_step_{int(step):09d}.msgpack'
        p=out/name
        raw=serialization.to_bytes(params)
        with p.open('xb') as f:f.write(raw)
        checkpoints.append({'steps':int(step),'sha256':hashlib.sha256(raw).hexdigest(),
                            'file':name})
        persist()
        print('M463_CHECKPOINT',int(step),name,flush=True)

    monitor=GPUMonitor()
    try:
        with monitor:
            report['training_performed']=True
            _,params,_=instrumented_ppo.train(
                environment=train_env,eval_env=eval_env,
                wrap_env_fn=api['wrapper'].wrap_for_brax_training,
                network_factory=functools.partial(api['networks'].make_ppo_networks,**net_cfg),
                progress_fn=progress,policy_params_fn=save_policy,
                **cfg)
        gpu=monitor.snapshot()
        report.update(gpu_peak_mib=gpu['peak_used_mib'],gpu_total_mib=gpu['total_mib'],
                      gpu_samples=gpu['samples'],gpu_monitor_errors=gpu['sampler_errors'])
        expected=scaled.num_timesteps
        if not history or history[-1]['additional_steps']!=expected:
            raise RuntimeError('M463_STEP_BUDGET_INCOMPLETE')
        if not checkpoints or checkpoints[-1]['steps']!=expected:
            raise RuntimeError('M463_FINAL_CHECKPOINT_NOT_SAVED')
        if len(params)!=3:raise RuntimeError('M463_UNEXPECTED_FINAL_PARAMS_LAYOUT')
        end_template=(params[0],{'policy':params[1],'value':params[2]})
        restored=serialization.from_bytes(end_template,(out/checkpoints[-1]['file']).read_bytes())
        from m462_typed_orbax_stage import compare_exact
        n=compare_exact(end_template,restored,jax,np,'M463_RETURNED_VS_CALLBACK')
        for leaf in jax.tree_util.tree_leaves(params):
            if not np.all(np.isfinite(np.asarray(jax.device_get(leaf)))):
                raise RuntimeError('M463_NONFINITE_PARAMETERS')
        batches_per_eval=BATCHES[args.mode]//(scaled.num_evals-1)
        if batches_per_eval*(scaled.num_evals-1)!=BATCHES[args.mode]:
            raise RuntimeError('M463_INVALID_EVAL_BATCH_COUNT')
        audit=audit_coverage(history,batches_per_eval)
        if args.mode == 'smoke3':
            c = audit['counts']
        
            t = c.get(PREFIX+'repeat_side_targeted_per_rollout', 0.0)
            m = c.get(PREFIX+'repeat_side_matched_per_rollout', 0.0)
            f = c.get(PREFIX+'repeat_side_fallback_per_rollout', 0.0)
            left = c.get(PREFIX+'repeat_side_target_left_per_rollout', 0.0)
            right = c.get(PREFIX+'repeat_side_target_right_per_rollout', 0.0)
            repeat = c.get(PREFIX+'repeat_per_rollout', 0.0)
        
            audit['checks'].update({
                'repeat_side_targeted_observed': t > 0,
                'repeat_side_matched_observed': m > 0,
                'repeat_side_both_directions_observed': left > 0 and right > 0,
                'repeat_side_counts_reconcile':
                    abs(t-m-f) < 0.1
                    and abs(t-left-right) < 0.1
                    and t <= repeat,
            })
        
            audit['passed'] = all(audit['checks'].values())
        memory_ok=(gpu['peak_used_mib'] is not None and gpu['total_mib'] is not None
                   and gpu['peak_used_mib']<gpu['total_mib']*0.95)
        report.update(steps_observed=expected,final_tensor_count=n,weights_finite=True,
                      callback_vs_final_equal=True,coverage=audit,
                      coverage_gate_pass=audit['passed'],memory_gate_pass=memory_ok,
                      finished_at_unix=time.time())
        ok=audit['passed'] and memory_ok and not gpu['sampler_errors']
        report['status']='M463_COVERAGE_'+args.mode.upper()+'_PASS' if ok else 'M463_COVERAGE_GATE_FAIL'
        persist()
        print('M463_END',report['status'],'training_push_events',audit['push_events_observed'],
              'vram_peak_mib',gpu['peak_used_mib'],flush=True)
        if not ok:raise RuntimeError('M463_EXPOSURE_OR_VRAM_GATE_FAILED')
    except Exception as e:
        if report.get('status')!='M463_COVERAGE_GATE_FAIL':
            gpu=monitor.snapshot()
            report.update(gpu_peak_mib=gpu['peak_used_mib'],gpu_total_mib=gpu['total_mib'],
                          gpu_samples=gpu['samples'],gpu_monitor_errors=gpu['sampler_errors'])
            report['status']='M463_FAILED'
        report['error']=repr(e)
        persist();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','smoke','probe2','smoke3','pilot'],required=True)
    p.add_argument('--stage',choices=['easy','easy_repeat','easy_repeat_support','medium','hard'],default='easy')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--restore-report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke-report',type=Path)
    a=p.parse_args()
    if a.mode=='plan':
        print('M463_COVERAGE_PLAN_NO_GPU',json.dumps({
            'stage':a.stage,'mode':a.mode,'smoke_steps':TOTAL_STEPS_BATCH,
            'pilot_steps':20*TOTAL_STEPS_BATCH,'num_envs':8192,
            'unroll_length':32,'push_target_step':[12,24],
            'nominal_fraction':0.20,'checkpoint':str(a.checkpoint),
            'probe2_steps': 2 * TOTAL_STEPS_BATCH,
            'restore_report':str(a.restore_report),'out_dir':str(a.output_dir),
            'source_not_modified':True,'optimizer_resumed':False},indent=2))
        return
    run(a)


if __name__=='__main__': main()
