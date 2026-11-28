#!/usr/bin/env python3
"""M4.62 PPO A/B weights-only warm-start from validated typed Orbax checkpoint.

A: source random push (5-10s, 0.1-2.0 m/s)
B: source push mechanism, density shifted (3-7s, 0.3-2.0 m/s)

Only --mode smoke is allowed without BOTH A+B validated smoke reports.
Use pinned MuJoCo Playground 2025, Brax 0.12.1, RTX 3060 12GB.
No source edit, no reward/PD/model/termination changes, no baseline overwrite.
"""
from __future__ import annotations

import argparse
import copy
import csv
import functools
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

from m462_typed_orbax_stage import orbax_directory_digest, compare_exact

SEED1_SHA = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
TRAIN_SHA = 'e20613d42bf0868a13003e0312998816cc8d3593bb5ccaf9f1d070fd09c44bd7'
PIN = '9bcbe736177472ba35d60773e3030a36e37962b7'
STEPS_PER_ROLLOUT = 262_144
BUDGET = {'smoke':1,'pilot':20}
B_INTERVAL = [3.0,7.0]
B_MAGNITUDE = [0.3,2.0]


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def sample_gpu():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used',
                                 '--format=csv,noheader,nounits'],text=True,timeout=8)
    rows=[]
    for cells in csv.reader(out.splitlines()):
        if len(cells)<4:continue
        try:
            rows.append({'index':int(cells[0].strip()),'name':cells[1].strip(),
                         'total_mib':int(cells[2].strip()),'used_mib':int(cells[3].strip())})
        except ValueError:continue
    if not rows:raise RuntimeError('M462_NVIDIA_SMI_NO_GPU_DATA')
    # Designed for the one-RTX3060 12GB research workstation.
    valid=[x for x in rows if 11_000 <= x['total_mib'] <= 13_000]
    if len(valid)!=1:
        raise RuntimeError(f'M462_EXPECT_ONE_12GB_GPU_FOUND {rows!r}')
    return valid[0]


class GPUMonitor:
    """Sampling GPU VRAM; samples include other processes, so it is a bound not an exact attribution."""
    def __init__(self):
        self.stop=threading.Event()
        self.samples=[]
        self.errors=[]
        self.thread=threading.Thread(target=self.work,daemon=True)
    def work(self):
        while not self.stop.is_set():
            try:self.samples.append(sample_gpu())
            except Exception as e:self.errors.append(repr(e))
            self.stop.wait(2.0)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self,*unused):
        self.stop.set()
        self.thread.join(timeout=10)
    def snapshot(self):
        return {'samples':len(self.samples),
                'gpu_name':self.samples[-1]['name'] if self.samples else None,
                'total_mib':self.samples[-1]['total_mib'] if self.samples else None,
                'peak_used_mib':max((s['used_mib'] for s in self.samples),default=None),
                'sampler_errors':self.errors[:10]}


def flatten_changed_paths(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):raise RuntimeError('M462_CONFIG_KEYS_CHANGED '+prefix)
        result=[]
        for k in a:
            result.extend(flatten_changed_paths(a[k],b[k],prefix+'.'+k if prefix else k))
        return result
    if a!=b:return [prefix]
    return []


def read_restore_gate(report_path, original_ck):
    report=json.loads(report_path.read_text(encoding='utf8'))
    if report.get('status')!='M462_TYPED_ORBAX_RESTORE_PASS_NO_TRAINING':
        raise RuntimeError('M462_TYPED_RESTORE_REPORT_NOT_PASS')
    if report.get('checkpoint_sha256')!=SEED1_SHA or report.get('source_commit')!=PIN:
        raise RuntimeError('M462_BASELINE_PROVENANCE_INVALID')
    if not report.get('all_restored_tensor_values_bitwise_equal') or not report.get('deterministic_action_equal'):
        raise RuntimeError('M462_STAGE_WEIGHTS_NOT_VERIFIED')
    if report.get('training_performed') or report.get('rollout_performed'):
        raise RuntimeError('M462_RESTORE_GATE_WAS_NOT_ZERO_STEP')
    if file_sha256(original_ck)!=SEED1_SHA:
        raise RuntimeError('M462_BASELINE_CHECKPOINT_CHANGED')
    ck_dir=Path(report['orbax_dir'])
    if not ck_dir.is_dir():raise RuntimeError('M462_TYPED_ORBAX_DIRECTORY_MISSING')
    sha,files=orbax_directory_digest(ck_dir)
    if sha!=report.get('orbax_dir_sha256') or files!=report.get('orbax_file_count'):
        raise RuntimeError('M462_ORBAX_STAGE_CHANGED_AFTER_VERIFICATION')
    return report,ck_dir


def smoke_reports_pass(a_path,b_path,stage_sha):
    for path,expected in ((a_path,'A'),(b_path,'B')):
        if path is None or not path.is_file():
            raise RuntimeError('M462_REQUIRE_BOTH_SMOKE_GATE_REPORTS')
        d=json.loads(path.read_text())
        if d.get('status')!='M462_GPU_SMOKE_PASS' or d.get('arm')!=expected:
            raise RuntimeError('M462_SMOKE_REPORT_NOT_PASS '+str(path))
        if (d.get('steps_observed')!=STEPS_PER_ROLLOUT or d.get('base_checkpoint_sha256')!=SEED1_SHA
                or d.get('orbax_dir_sha256')!=stage_sha):
            raise RuntimeError('M462_SMOKE_GATE_DIFFERENT_BASELINE_OR_BUDGET '+str(path))
        if d.get('gpu_peak_used_mib') is None or d.get('gpu_total_mib') is None:
            raise RuntimeError('M462_NO_GPU_MEMORY_EVIDENCE '+str(path))
        if d['gpu_peak_used_mib']>=int(d['gpu_total_mib']*0.95):
            raise RuntimeError('M462_SMOKE_GPU_MEMORY_TOO_CLOSE_TO_LIMIT '+str(path))


def check_source_and_config(arm):
    from train import check_source, build_paper_config
    from paper_config import PIN as source_pin, BASE, ENV
    check_source()
    if source_pin != PIN:raise RuntimeError('M462_SOURCE_PIN_CHANGED')
    sys.path.insert(0,str(BASE/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('M462_NVCC_COMPAT',apply(),flush=True)

    import jax
    import numpy as np
    from flax import serialization
    from brax.training.agents.ppo import train as ppo, networks
    from mujoco_playground import locomotion, registry, wrapper
    import inspect

    if jax.default_backend()!='gpu':raise RuntimeError('M462_GPU_REQUIRED')
    if file_sha256(inspect.getfile(ppo.train)) != TRAIN_SHA:
        raise RuntimeError('M462_PINNED_BRAX_PPO_SOURCE_DIFFERENT')
    if 'restore_checkpoint_path' not in inspect.signature(ppo.train).parameters:
        raise RuntimeError('M462_BRAX_RESTORE_PATH_NOT_SUPPORTED')
    paper=build_paper_config()
    scaled=copy.deepcopy(paper)
    scaled.num_envs=8192
    scaled.num_eval_envs=2
    scaled.batch_size=1024
    scaled.num_minibatches=8
    scaled.unroll_length=32
    scaled.num_updates_per_batch=5
    scaled.episode_length=1000
    scaled.action_repeat=1
    if (scaled.batch_size*scaled.num_minibatches*scaled.unroll_length*scaled.action_repeat)!=STEPS_PER_ROLLOUT:
        raise RuntimeError('M462_HARDWARE_SCALED_ROLLOUT_CHANGED')
    if scaled.batch_size*scaled.num_minibatches!=scaled.num_envs:
        raise RuntimeError('M462_BAD_BATCH_ENV_ALIGNMENT')

    source_env_cfg=locomotion.get_default_config(ENV)
    train_cfg=copy.deepcopy(source_env_cfg)
    if list(source_env_cfg.push_config.interval_range)!=[5.0,10.0] or list(source_env_cfg.push_config.magnitude_range)!=[0.1,2.0]:
        raise RuntimeError('M462_PINNED_SOURCE_PUSH_DISTRIBUTION_CHANGED')
    if arm=='B':
        train_cfg.push_config.interval_range=list(B_INTERVAL)
        train_cfg.push_config.magnitude_range=list(B_MAGNITUDE)
    changed=flatten_changed_paths(source_env_cfg.to_dict(),train_cfg.to_dict())
    if set(changed)!=({'push_config.interval_range','push_config.magnitude_range'} if arm=='B' else set()):
        raise RuntimeError('M462_TRAIN_ENV_CHANGED_UNEXPECTED_FIELDS '+str(changed))
    if float(train_cfg.reward_config.scales.base_height)!=0 or float(source_env_cfg.reward_config.scales.base_height)!=0:
        raise RuntimeError('M462_REWARD_CONFIG_CHANGED')
    randomizer=registry.get_domain_randomizer(ENV)
    if randomizer is None:raise RuntimeError('M462_DOMAIN_RANDOMIZER_MISSING')
    return {'jax':jax,'numpy':np,'serialization':serialization,
            'ppo':ppo,'networks':networks,'locomotion':locomotion,'wrapper':wrapper,
            'env_name':ENV,'scaled':scaled,'train_cfg':train_cfg,'source_cfg':source_env_cfg,
            'randomizer':randomizer,'changed_paths':changed}


def run(args):
    if args.out_dir.exists():raise FileExistsError('M462_REFUSE_OUTPUT_DIRECTORY_OVERWRITE')
    gate,orbax_dir=read_restore_gate(args.restore_report,args.checkpoint)
    if args.mode=='pilot':
        smoke_reports_pass(args.smoke_a_report,args.smoke_b_report,gate['orbax_dir_sha256'])
    gpu_before=sample_gpu()
    api=check_source_and_config(args.arm)
    jax,np=api['jax'],api['numpy']
    from flax import serialization

    scaled=api['scaled']
    scaled.num_timesteps=BUDGET[args.mode]*STEPS_PER_ROLLOUT
    scaled.num_evals=2 if args.mode=='smoke' else 6  # 1 batch vs 20 batches/5 evaluations
    conf=scaled.to_dict()
    net_cfg=conf.pop('network_factory')
    conf.update(seed=1,randomization_fn=api['randomizer'])
    conf['restore_checkpoint_path']=str(orbax_dir.resolve())
    # B changes only training push configuration; evaluation remains unchanged.
    env=api['locomotion'].load(api['env_name'],config=api['train_cfg'])
    eval_env=api['locomotion'].load(api['env_name'],config=api['source_cfg'])
    if env is eval_env:raise RuntimeError('M462_EVAL_AND_TRAIN_ENV_IDENTICAL')

    args.out_dir.mkdir(parents=True,exist_ok=False)
    report_path=args.out_dir/'report.json'
    history=[]
    checkpoints=[]
    result={
        'status':'M462_GPU_RUN_STARTED','arm':args.arm,'mode':args.mode,
        'source_commit':PIN,'base_checkpoint_sha256':SEED1_SHA,
        'base_checkpoint_step':401_080_320,
        'orbax_dir':str(orbax_dir.resolve()),'orbax_dir_sha256':gate['orbax_dir_sha256'],
        'train_seed':1,'weight_init':'original_401M_weights_only_adam_reset',
        'ppo_training_steps_requested':scaled.num_timesteps,
        'num_rollouts_requested':BUDGET[args.mode],
        'minibatch_optimizer_updates_requested':BUDGET[args.mode]*40,
        'scaled_config':scaled.to_dict(),
        'train_push_interval':list(api['train_cfg'].push_config.interval_range),
        'train_push_magnitude':list(api['train_cfg'].push_config.magnitude_range),
        'evaluation_push_interval':list(api['source_cfg'].push_config.interval_range),
        'evaluation_push_magnitude':list(api['source_cfg'].push_config.magnitude_range),
        'train_config_changed_fields':api['changed_paths'],
        'gpu_before':gpu_before,'gpu_peak_used_mib':None,'gpu_total_mib':None,
        'report_history':history,'checkpoints':checkpoints,
        'experiment_type':'hardware_scaled_extension_not_paper_exact',
        'runner_sha256':file_sha256(__file__),
        'gpu_runtime_measured':False,'training_performed':False,
    }
    def persist():
        tmp=report_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf8')
        tmp.replace(report_path)
    persist()
    def summarise_metrics(metrics):
        summary={}
        for k,v in metrics.items():
            arr=np.asarray(jax.device_get(v))
            if arr.dtype.kind not in ('b','i','u','f'):
                summary[k]={'shape':list(arr.shape),'type':str(arr.dtype)}
                continue
            if not np.all(np.isfinite(arr)):
                raise RuntimeError('M462_NONFINITE_METRIC '+str(k))
            summary[k]=float(arr) if arr.ndim==0 else {
                'shape':list(arr.shape),'mean':float(arr.mean()),
                'min':float(arr.min()),'max':float(arr.max())}
        return summary
    def progress(step,metrics):
        step=int(step)
        item={'additional_steps':step,'metrics':summarise_metrics(metrics)}
        history.append(item)
        persist()
        print('M462_PROGRESS',args.arm,step,'eval_reward',item['metrics'].get('eval/episode_reward'),flush=True)
    def save_policy(step,_make_policy,params):
        step=int(step)
        if step==0:return
        name=f'm462_{args.arm}_{args.mode}_step_{step:09d}.msgpack'
        path=args.out_dir/name
        raw=serialization.to_bytes(params)
        with path.open('xb') as f:f.write(raw)
        checkpoints.append({'additional_steps':step,'name':name,
                            'sha256':hashlib.sha256(raw).hexdigest()})
        persist()
        print('M462_CHECKPOINT',args.arm,step,path,flush=True)

    monitor=GPUMonitor()
    try:
        with monitor:
            result['training_performed']=True
            _,final_params,final_metrics=api['ppo'].train(
                environment=env,eval_env=eval_env,
                wrap_env_fn=api['wrapper'].wrap_for_brax_training,
                network_factory=functools.partial(api['networks'].make_ppo_networks,**net_cfg),
                progress_fn=progress,policy_params_fn=save_policy,
                **conf)
        gpu=monitor.snapshot()
        result.update(gpu_peak_used_mib=gpu['peak_used_mib'],gpu_total_mib=gpu['total_mib'],
                      gpu_name=gpu['gpu_name'],gpu_monitor_samples=gpu['samples'],
                      gpu_monitor_errors=gpu['sampler_errors'],gpu_runtime_measured=True)
        last_step=BUDGET[args.mode]*STEPS_PER_ROLLOUT
        if not history or history[-1]['additional_steps']!=last_step:
            raise RuntimeError('M462_TRAINING_STEP_BUDGET_MISMATCH')
        if not checkpoints or checkpoints[-1]['additional_steps']!=last_step:
            raise RuntimeError('M462_FINAL_CHECKPOINT_MISSING')
        if len(final_params)!=3:raise RuntimeError('M462_PPO_RETURN_SCHEMA_CHANGED')
        end_template=(final_params[0],{'policy':final_params[1],'value':final_params[2]})
        raw=(args.out_dir/checkpoints[-1]['name']).read_bytes()
        from_cb=serialization.from_bytes(end_template,raw)
        n=compare_exact(end_template,from_cb,jax,np,'RETURNED_VS_CALLBACK')
        for leaf in jax.tree_util.tree_leaves(final_params):
            if not np.all(np.isfinite(np.asarray(jax.device_get(leaf)))):
                raise RuntimeError('M462_NONFINITE_TRAINED_PARAMETERS')
        if not gpu['samples'] or gpu['peak_used_mib'] is None:
            raise RuntimeError('M462_MISSING_GPU_MEMORY_MEASUREMENT')
        # 12GB unit is NVIDIA-reported MiB, and peak is device-wide not per-process.
        memory_ok=gpu['peak_used_mib'] < gpu['total_mib']*0.95
        result.update(steps_observed=last_step,final_tensor_comparison_leaf_count=n,
                      final_params_finite=True,returned_vs_callback_tensor_equal=True,
                      estimated_gpu_memory_margin_ok=memory_ok,
                      finished_at_unix=time.time())
        result['status']=('M462_GPU_SMOKE_PASS' if args.mode=='smoke' else 'M462_GPU_PILOT_COMPLETED') if memory_ok else 'M462_GPU_MEMORY_GATE_FAILED'
        persist()
        print('M462_END',result['status'],'arm',args.arm,'steps',last_step,
              'GPU_PEAK_MIB',gpu['peak_used_mib'],'GPU_TOTAL_MIB',gpu['total_mib'],flush=True)
        if not memory_ok:raise RuntimeError('M462_GPU_MEMORY_TOO_CLOSE_TO_CAPACITY')
    except Exception as e:
        gpu=monitor.snapshot()
        result.update(gpu_peak_used_mib=gpu['peak_used_mib'],gpu_total_mib=gpu['total_mib'],
                      gpu_name=gpu['gpu_name'],gpu_monitor_samples=gpu['samples'],
                      gpu_monitor_errors=gpu['sampler_errors'],gpu_runtime_measured=True)
        result['status']='M462_GPU_RUN_FAILED'
        result['error']=repr(e)
        persist()
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('plan','smoke','pilot'),required=True)
    ap.add_argument('--arm',choices=('A','B'),required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--restore-report',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--smoke-a-report',type=Path)
    ap.add_argument('--smoke-b-report',type=Path)
    args=ap.parse_args()
    if args.mode=='plan':
        print('M462_PLAN',json.dumps({'arm':args.arm,'steps_smoke':STEPS_PER_ROLLOUT,
                                    'steps_pilot':20*STEPS_PER_ROLLOUT,
                                    'gpu':'single RTX3060 12GB',
                                    'source_push_change_only_in_B':True,
                                    'train_seed':1,'checkpoint_sha256':SEED1_SHA,
                                    'checkpoint':str(args.checkpoint),'restore_report':str(args.restore_report),
                                    'out_dir':str(args.out_dir)}))
        return
    run(args)

if __name__=='__main__':main()
