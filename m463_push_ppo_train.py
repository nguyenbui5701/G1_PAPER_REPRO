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


H3_SPEC = {'id': 'H3-M1', 'slot': 'stand_still', 'source_scale': -1.0,
           'alpha': 0.5, 'window_ticks': 100, 'dt': 0.02,
           'terminal_factor': 1.0, 'context': 'actual_event_boundary_corrected',
           'actor_critic_context_added': False}
H3_FP = '420ecb3798789b0f90f3ed249572ad8298fad03a5ce56a83a7a83e59ebe90fef'
H3_BAD_KEYS = tuple('h3_' + k for k in (
    'timing_bad_per_rollout', 'formula_bad_per_rollout',
    'forbidden_bad_per_rollout', 'truncation_bad_per_rollout',
    'reward_sum_bad_per_rollout', 'nonfinite_per_rollout'))
H3_POS_KEYS = tuple('h3_' + k for k in (
    'active_per_rollout', 'post_active_per_rollout', 'stand_events_per_rollout',
    'stand_post_steps_per_rollout', 'nominal_steps_per_rollout',
    'nonstand_steps_per_rollout', 'delta_sum_dt'))
H3_METRIC_KEYS = H3_BAD_KEYS + H3_POS_KEYS + (
    'h3_terminal_steps_per_rollout', 'h3_base_sum_pre_dt',
    'h3_modified_sum_pre_dt')


H3_CAUSAL_SHA = '47a1d2d8633ffd223768b9cedbbb6ca12a61f4fddd127fdc73b59e3e3f147a07'
H3_CAUSAL_SOURCE_DIGESTS = {
    'm463_coverage_env.py': 'f59bfdfc8e9000ab52bbeda25abf9d1556cb96d354f133ee18e9efaebef1a582',
    'm463_instrument_brax.py': '16e7e0b024de80e9f2663987d79fe2ac4bedb17a9a95ac6a5f9d7115c6df39fd',
    'm463_push_ppo_train.py': '613d405d5a43f1c8e68ac128b9e551689d8bf42308b7905914171d73e19bc120',
}
H3_CAUSAL_EVALUATOR_SHA = '81d1a0bec04749e649ed252fabb7b77b3a5ea2d9758e5e96d711fccafa91d627'
H3_CAUSAL_BRAX_SHA = 'a2795ba3d305d5695b72107832bfa1531ce6750523167bea6ea584988a6a2840'
H3_CAUSAL_ORIGINAL_BRAX_SHA = 'e20613d42bf0868a13003e0312998816cc8d3593bb5ccaf9f1d070fd09c44bd7'
H3_CAUSAL_RESTORE_SHA = '9c46427c16aa482730e45bc9f79c53f6cb18ec3a267d1fc3253faab695916e81'


H3_SMOKE_SHA = '55e6d2248e17c9a58eaae6c0c0b965792059e3f4edec564b5444791ffa750aca'
H3_SMOKE_RUNNER_SHA = '7a3d475928765fc669ef6f6ca71fd043cc7ad4053b4c6a381cad747dbb6e8031'
H3_OLD10M_SHA = '04f7c71b3dd4b9d24e06b13991a9e4bb4e6c482d217929a74aa40537e9b62f74'
H3_OLD10M_POLICY_SHA = '0a011b6a3011f126a29832e4faa18972d7fbdddef77b63414219a15abab8314a'
H3_OLD10M_BRAX_SHA = '2a7182e7bd8e9134b5317b858cc0e022468a057807c0021c2a6a449ce1be68b5'
H3_OLD10M_RUNNER_SHA = '706a19830e426672efeac4b2832c6b46512973ecc7baff30c82350281153bb2f'
H3_OLD10M_COMMIT = 'aae9e8f484612790029d54d4537993e614477885'


H4_SPEC = {'arms': ['control','candidate'], 'control_episode_mix': [.20,.80,0.,0.],
           'candidate_episode_mix': [.20,.40,.16,.24],
           'rollouts':40,'num_evals':3,'rollouts_per_epoch':20,
           'rollouts_per_chunk':5,'collection_ticks':640,
           'env_steps':10485760,'optimizer_updates_target':1600,
           'first_age':[240,300],'first_magnitude':[.45,.55],
           'repeat_gap_target':[30,32],'repeat_gap_actual':[30,40],
           'repeat_magnitude':[.30,.40],'max_wait':8,
           'tail_windows':[10,25,50,100,250]}


# Deliberately locked for this source revision. Opening H4 requires a
# separately reviewed source change and explicit owner execution GO.
H4_EXECUTION_AUTHORIZED = False
H4_ORBAX_SHA = '5acfabc1376e91a2a9a52b47ac821e296f0e84f5545b646b07c9482ddc462bf3'
H4_EDGES = (5242880,10485760)
H4_FILES = ('m463_coverage_env.py','m463_instrument_brax.py',
            'm463_push_ppo_train.py','train.py','paper_config.py',
            'm462_ppo_warmstart_run.py','m462_typed_orbax_stage.py')


def h4_integer_count(mean, rollouts=5):
    # Each integer rollout count <= 262144; five-count sum <= 1310720
    # is exactly representable in float32. Division rounding < .125 counts.
    x = float(mean)*rollouts
    if not math.isfinite(x) or not 0 <= x <= rollouts*TOTAL_STEPS_BATCH or abs(x-round(x)) > .125:
        raise RuntimeError('M470_H4_NONINTEGER_COUNT')
    return int(round(x))


def h4_epoch_audit(counts, pending, arm, final=False):
    """Integer ledger over real PPO extras and pending pre-reset env info."""
    def c(name):
        return counts['m463_h4_'+name+'_per_rollout']
    def q(name):
        return pending[name]
    if any(type(v) is not int or v<0 for v in (*counts.values(),*pending.values())):
        raise RuntimeError('M470_H4_LEDGER_COUNT_SCHEMA')
    checks = {
        'step_partition':sum(c(f'class_{i}_steps') for i in range(4)) ==
                         (10485760 if final else 5242880),
        'first_conservation': c('assigned') == c('first_events') +
                              c('first_closed') + q('first_pending'),
        'repeat_conservation': c('repeat_assigned') == c('repeat_events') +
                               c('repeat_closed') + q('repeat_pending'),
        'tail_conservation': c('first_events')+c('repeat_events') ==
                             c('event_terminal')+c('close_repeat')+
                             c('close_terminal')+c('close_complete')+
                             c('close_boundary')+q('tail_pending'),
        'no_bad': c('bad') == 0,
        'nominal_fraction': c('class_0_steps') >= .15 *
                            (10485760 if final else 5242880),
        'due_bounds': c('first_events')<=c('first_due')<=c('assigned') and
                      c('repeat_events')<=c('repeat_due')<=c('repeat_assigned'),
    }
    for ci in range(4):
        def cc(k):
            return c(f'class_{ci}_{k}')
        checks[f'class_{ci}_first_ledger'] = (
            cc('assigned') == cc('first_events')+cc('first_closed')+q(f'class_{ci}_first_pending')
            and cc('first_events')<=cc('first_due')<=cc('assigned'))
        checks[f'class_{ci}_repeat_ledger'] = (
            cc('repeat_assigned') == cc('repeat_events')+cc('repeat_closed')+q(f'class_{ci}_repeat_pending')
            and cc('repeat_events')<=cc('repeat_due')<=cc('repeat_assigned'))
        for name in ('assigned','repeat_assigned','first_due','repeat_due',
                     'first_closed','repeat_closed','first_events','repeat_events'):
            checks[f'partition_{name}'] = sum(c(f'class_{j}_{name}') for j in range(4)) == c(name)
    checks['nominal_no_events'] = c('class_0_first_events')+c('class_0_repeat_events')==0
    checks['single_no_repeat'] = c('class_2_repeat_events')+c('class_2_repeat_assigned')==0
    checks['arm_partition'] = (arm=='candidate' or
        sum(c(f'class_{j}_steps') for j in (2,3))==0)
    for motion in (0,3,4):
        for side in (1,2):
            prefix=f'target_{motion}_{side}_'
            n=c(prefix+'repeat_events')
            checks[prefix+'ledger'] = (
                c(prefix+'repeat_assigned') == n+c(prefix+'repeat_closed')+q(prefix+'pending')
                and n<=c(prefix+'repeat_due')<=c(prefix+'repeat_assigned')
                and n==c(prefix+'matched')+c(prefix+'fallback'))
    for kind,total in ((1,c('first_events')),(2,c('repeat_events'))):
        checks[f'age_{kind}_partition'] = sum(c(f'age_{kind}_{lo}_{hi}') for lo,hi in
            ((0,32),(33,239),(240,308),(309,348),(349,639)))==total
        checks[f'support_{kind}_partition'] = sum(c(f'support_{kind}_{side}') for side in range(4))==total
        for w in H4_SPEC['tail_windows']:
            executed=c(f'tail_{kind}_{w}_executed')
            checks[f'tail_{kind}_{w}_bounds'] = (
                0<=c(f'tail_{kind}_{w}_survived')<=executed<=total and
                c(f'tail_{kind}_{w}_same_command')<=executed and
                c(f'tail_{kind}_{w}_same_unroll')<=executed)
            total=executed
    checks['tail_completed_reconcile']=c('close_complete') == sum(
        c(f'tail_{kind}_250_survived') for kind in (1,2))
    checks['tail_repeat_reconcile']=c('close_repeat')==c('repeat_events')
    checks['no_stale_natural_boundary']=c('close_boundary')==0
    checks['repeat_gap_partition']=sum(c(f'gap_{lo}_{hi}') for lo,hi in ((0,29),(30,40),(41,639)))==c('repeat_events')
    checks['repeat_side_bound']=c('repeat_matched')+c('repeat_fallback')<=c('repeat_events')
    checks['broad_partition']=c('broad_events')==c('class_1_first_events')+c('class_1_repeat_events')
    for k in ('motion','direction'):
        checks[f'broad_{k}_partition']=sum(c(f'broad_{k}_{j}') for j in range(6))==c('broad_events')
    for side in (1,2):
        n=c(f'stand_{side}_events')
        checks[f'qual_{side}_matched_bound'] = n<=c(f'target_0_{side}_matched')
        prev=n
        for w in H4_SPEC['tail_windows']:
            count=c(f'stand_{side}_tail_2_{w}')
            checks[f'qual_{side}_{w}_bounds'] = (
                0<=c(f'stand_{side}_survived_2_{w}')<=count<=prev)
            prev=count
    if arm == 'candidate':
        for side in (1,2):
            n=c(f'stand_{side}_events')
            checks[f'stand_{side}_exposure']=n >= (32 if final else 1)
            checks[f'stand_{side}_tail100']=c(f'stand_{side}_tail_2_100') >= (16 if final else 1)
            if final:
                checks[f'stand_{side}_tail250']=c(f'stand_{side}_tail_2_250')>=8
            for motion in (3,4):
                checks[f'turn_{motion}_{side}']=c(f'turn_{motion}_{side}_events') >= (32 if final else 1)
        if final:
            checks['broad_events']=c('broad_events')>=1000
            for j in range(6):
                for k in ('motion','direction'):
                    checks[f'broad_{k}_{j}']=c(f'broad_{k}_{j}')>0
    elif arm != 'control':
        raise RuntimeError('M470_H4_ARM')
    return {'checks':checks, 'passed':all(checks.values())}


def h4_chunk_callback(jax,np,monitor,report,persist,arm,*,h4cc_arm=None,h4m6_arm=None):
    """Prepared callback; no H4 runtime is authorized by this source revision.

    The caller must bind the same monitor and report as the future H4 runner.
    """
    chunks=[]
    epoch_counts={}
    all_counts={}
    all_pending={}
    report['h4_chunks']=chunks
    report['h4_epochs']=[]
    cc_audit = h4cc_chunk_ledger(jax,np,report,h4cc_arm) if h4cc_arm is not None else None
    m6_audit = h4m6_chunk_ledger(jax,np,report,h4m6_arm) if h4m6_arm is not None else None
    if h4m6_arm is not None:
        h4m6_require(h4cc_arm=='ref' and arm=='candidate','CHUNK_ARM')
    def callback(index,state,env_state,metrics):
        expected_index=len(chunks)%4+1
        expected_steps=(len(chunks)+1)*5*TOTAL_STEPS_BATCH
        if index!=expected_index or int(jax.device_get(state.env_steps))!=expected_steps:
            raise RuntimeError('M470_H4_CHUNK_STEP')
        if len(chunks)>=8:
            raise RuntimeError('M470_H4_CHUNK_BUDGET')
        gpu=monitor.snapshot()
        row={'chunk':len(chunks)+1,'steps':expected_steps,'gpu':gpu,
             'counts':{},'passed':False}
        chunks.append(row)
        persist()
        if (gpu['total_mib']!=12288 or gpu['peak_used_mib'] is None or
                not 0<gpu['peak_used_mib']<.95*12288 or gpu['sampler_errors']):
            raise RuntimeError('M470_H4_CHUNK_GPU')
        for leaf in jax.tree_util.tree_leaves(jax.device_get(state)):
            if not np.isfinite(np.asarray(leaf)).all():
                raise RuntimeError('M470_H4_CHUNK_NONFINITE_STATE')
        for k,v in metrics.items():
            a=np.asarray(jax.device_get(v))
            if a.ndim!=0 or not np.isfinite(a).all():
                raise RuntimeError('M470_H4_CHUNK_METRIC_SCHEMA '+k)
            if k.startswith('m463_h4_') and k.endswith('_per_rollout'):
                n=h4_integer_count(a)
                row['counts'][k]=n
                epoch_counts[k]=epoch_counts.get(k,0)+n
                all_counts[k]=all_counts.get(k,0)+n
        required = {'bad','assigned','repeat_assigned','first_events','repeat_events',
                    'first_closed','repeat_closed','first_due','repeat_due',
                    'event_terminal','close_repeat','close_terminal','close_complete',
                    'close_boundary','broad_events'}
        required.update(f'class_{i}_{k}' for i in range(4) for k in
            ('steps','assigned','repeat_assigned','first_due','repeat_due','first_closed',
             'repeat_closed','first_events','repeat_events'))
        required.update(f'target_{m}_{side}_{k}' for m in (0,3,4) for side in (1,2) for k in
            ('repeat_assigned','repeat_due','repeat_closed','repeat_events','matched','fallback'))
        for kind in (1,2):
            for w in H4_SPEC['tail_windows']:
                required.update(f'tail_{kind}_{w}_{v}' for v in
                    ('executed','survived','same_command','same_unroll'))
        for side in (1,2):
            required.add(f'stand_{side}_events')
            required.update(f'stand_{side}_{k}_2_{w}' for w in H4_SPEC['tail_windows'] for k in ('tail','survived'))
            required.update(f'turn_{m}_{side}_events' for m in (3,4))
        required.update(f'broad_{k}_{j}' for j in range(6) for k in ('motion','direction'))
        if any('m463_h4_'+k+'_per_rollout' not in row['counts'] for k in required):
            raise RuntimeError('M470_H4_CHUNK_MISSING_METRICS')
        if sum(row['counts'][f'm463_h4_class_{j}_steps_per_rollout'] for j in range(4))!=5*TOTAL_STEPS_BATCH:
            raise RuntimeError('M470_H4_CHUNK_STEP_PARTITION')
        if row['counts']['m463_h4_bad_per_rollout']!=0:
            raise RuntimeError('M470_H4_CHUNK_INVARIANT')
        # Read only scalar device arrays, not full MJX state. Validate at
        # every host boundary, preserving the actual carry untouched.
        if index==4:
            info=env_state.info
            pending={}
            for k,field in (('first_pending','h4_first_pending'),
                            ('repeat_pending','h4_repeat_pending'),
                            ('tail_pending','h4_tail_kind')):
                arr=np.asarray(jax.device_get(info[field]))
                allowed=(0,1,2) if k=='tail_pending' else (0,1)
                if arr.size!=8192 or not np.isin(arr,allowed).all():
                    raise RuntimeError('M470_H4_PENDING_SCHEMA')
                pending[k]=int(np.count_nonzero(arr))
            hc=np.asarray(jax.device_get(info['h4_class']))
            fp=np.asarray(jax.device_get(info['h4_first_pending']))>0
            rp=np.asarray(jax.device_get(info['h4_repeat_pending']))>0
            origin=np.asarray(jax.device_get(info['h4_repeat_origin_motion']))
            direction=np.asarray(jax.device_get(info['m463_direction']))
            if any(x.shape!=hc.shape for x in (fp,rp,origin,direction)) or not np.isin(hc,(0,1,2,3)).all():
                raise RuntimeError('M470_H4_PENDING_CELL_SCHEMA')
            for ci in range(4):
                pending[f'class_{ci}_first_pending']=int(np.count_nonzero(fp & (hc==ci)))
                pending[f'class_{ci}_repeat_pending']=int(np.count_nonzero(rp & (hc==ci)))
            for motion in (0,3,4):
                for side in (1,2):
                    pending[f'target_{motion}_{side}_pending']=int(np.count_nonzero(
                        rp & (hc==3) & (origin==motion) & (direction==side+1)))
            for k,v in pending.items():
                all_pending[k]=all_pending.get(k,0)+v
            result=h4_epoch_audit(epoch_counts,pending,arm)
            # Early exposure floor only after first epoch; second epoch
            # must satisfy conservation but full-run floor governs exposure.
            if len(chunks)==8:
                result['checks']={k:v for k,v in result['checks'].items()
                    if not k.startswith(('stand_','turn_'))}
                result['passed']=all(result['checks'].values())
            report['h4_epochs'].append({'counts':dict(epoch_counts),
                'pending_at_reset':pending,'audit':result})
            persist()
            if not result['passed']:
                raise RuntimeError('M470_H4_EPOCH_EXPOSURE_OR_LEDGER')
            epoch_counts.clear()
            if len(chunks)==8:
                final=h4_epoch_audit(all_counts,all_pending,arm,final=True)
                report['h4_final_exposure']=final
                report['h4_final_counts']=dict(all_counts)
                report['h4_final_pending']=dict(all_pending)
                persist()
                if not final['passed']:
                    raise RuntimeError('M470_H4_FINAL_EXPOSURE')
        if cc_audit is not None:
            cc_audit(index,env_state,metrics,row['counts'])
        if m6_audit is not None:
            m6_audit(index,env_state,metrics,row['counts'])
        row['passed']=True
        persist()
    return callback


def h4_matched_nominal_check(control_counts, candidate_counts):
    """Requires separately provenance-validated matched-control counts."""
    k='m463_h4_class_0_steps_per_rollout'
    a,b=control_counts[k],candidate_counts[k]
    if not (0<=a<=10485760 and 0<=b<=10485760):
        raise RuntimeError('M470_H4_NOMINAL_COUNT_SCHEMA')
    return b>=.15*10485760 and 5*b>=4*a


def h4_common_contract(config,reward,instrument):
    if (m470_fingerprint(config)!=H3_FP or config.get('num_timesteps')!=10485760
            or config.get('num_evals')!=3 or config.get('num_resets_per_eval')!=1
            or config.get('episode_length')!=1000):
        raise RuntimeError('M470_H4_CONFIG_CONTRACT')
    if instrument.get('original_brax_sha256')!=H3_CAUSAL_ORIGINAL_BRAX_SHA:
        raise RuntimeError('M470_H4_BRAX_PIN')
    return {'source_commit':PIN,'checkpoint_sha256':SEED1_SHA,
            'restore_report_sha256':H3_CAUSAL_RESTORE_SHA,'orbax_sha256':H4_ORBAX_SHA,
            'seed':1,'config':json.loads(json.dumps(config,allow_nan=False)),
            'reward':json.loads(json.dumps(reward,allow_nan=False)),
            'source_digests':{name:digest(Path(__file__).with_name(name)) for name in H4_FILES},
            'generated_brax_sha256':instrument['instrumented_sha256'],
            'original_brax_sha256':instrument['original_brax_sha256']}


def h4_verify_ledger(report):
    chunks=report['h4_chunks']
    epochs=report['h4_epochs']
    arm=report['h4_arm']
    if len(chunks)!=8 or len(epochs)!=2:
        raise RuntimeError('M470_H4_REPORT_CHUNKS')
    combined={}
    pending={}
    for j,ep in enumerate(epochs):
        counts={}
        for k,row in enumerate(chunks[j*4:j*4+4]):
            n=j*4+k+1
            if row['chunk']!=n or row['steps']!=n*5*TOTAL_STEPS_BATCH or row['passed'] is not True:
                raise RuntimeError('M470_H4_REPORT_CHUNK_SEQUENCE')
            gpu=row['gpu']
            if (gpu['total_mib']!=12288 or not 0<gpu['peak_used_mib']<.95*12288 or gpu['sampler_errors']):
                raise RuntimeError('M470_H4_REPORT_CHUNK_GPU')
            if (sum(row['counts'][f'm463_h4_class_{ci}_steps_per_rollout'] for ci in range(4))!=5*TOTAL_STEPS_BATCH
                    or row['counts']['m463_h4_bad_per_rollout']!=0):
                raise RuntimeError('M470_H4_REPORT_CHUNK_INVARIANT')
            for key,v in row['counts'].items():
                if type(v) is not int or not 0<=v<=5*TOTAL_STEPS_BATCH:
                    raise RuntimeError('M470_H4_REPORT_CHUNK_COUNTS')
                counts[key]=counts.get(key,0)+v
                combined[key]=combined.get(key,0)+v
        if counts!=ep['counts']:
            raise RuntimeError('M470_H4_REPORT_EPOCH_COUNTS')
        q=ep['pending_at_reset']
        result=h4_epoch_audit(counts,q,arm)
        if j==1:
            result['checks']={k:v for k,v in result['checks'].items()
                if not k.startswith(('stand_','turn_'))}
            result['passed']=all(result['checks'].values())
        if result!=ep['audit'] or not result['passed']:
            raise RuntimeError('M470_H4_REPORT_EPOCH_AUDIT')
        for key,v in q.items():
            pending[key]=pending.get(key,0)+v
    result=h4_epoch_audit(combined,pending,arm,final=True)
    if (combined!=report['h4_final_counts'] or pending!=report['h4_final_pending']
            or result!=report['h4_final_exposure'] or not result['passed']):
        raise RuntimeError('M470_H4_REPORT_FINAL_LEDGER')
    return combined


def h4_legacy_audit(history):
    # Same original repeat-side requirements and exact tolerance.
    a=audit_coverage(history,20)
    c=a['counts']
    t,m,f,left,right,repeat=(c[PREFIX+k+'_per_rollout'] for k in (
        'repeat_side_targeted','repeat_side_matched','repeat_side_fallback',
        'repeat_side_target_left','repeat_side_target_right','repeat'))
    a['checks'].update({
        'repeat_side_targeted_observed':t>0,
        'repeat_side_matched_observed':m>0,
        'repeat_side_both_directions_observed':left>0 and right>0,
        'repeat_side_counts_reconcile':abs(t-m-f)<.1 and abs(t-left-right)<.1 and t<=repeat})
    a['passed']=all(a['checks'].values())
    return a


def h4_final_checks(report,audit,control):
    counts=h4_verify_ledger(report)
    if report['h4_arm']=='candidate' and report['h4_control_report_sha256']!=(control or {}).get('sha256'):
        raise RuntimeError('M470_H4_MATCHED_CONTROL_BINDING')
    legacy=h4_legacy_audit(report['report_history'])
    if legacy!=audit or not legacy['passed']:
        raise RuntimeError('M470_H4_LEGACY_COVERAGE')
    n=counts['m463_h4_first_events_per_rollout']+counts['m463_h4_repeat_events_per_rollout']
    if abs(legacy['push_events_observed']-n)>=.1:
        raise RuntimeError('M470_H4_LEGACY_EVENT_RECONCILE')
    if [row['additional_steps'] for row in report['report_history']]!=[0,*H4_EDGES]:
        raise RuntimeError('M470_H4_EVAL_EDGES')
    for k in ('checkpoints','fullstate_checkpoints'):
        if [v['steps'] for v in report[k]]!=list(H4_EDGES):
            raise RuntimeError('M470_H4_CHECKPOINT_EDGES')
    if report['h4_arm']=='candidate':
        if control is None or not h4_matched_nominal_check(control['counts'],counts):
            raise RuntimeError('M470_H4_MATCHED_NOMINAL')
        report['h4_matched_nominal_pass']=True
    report['h4_legacy_reconciled']=True


def h4_approved_control_sha():
    """A later Rules-only review may pin the control without source drift."""
    import re
    rules=Path(__file__).with_name('G1_RESEARCH_RULES.md')
    hits=re.findall(r'^H4_APPROVED_CONTROL_REPORT_SHA256: ([0-9a-f]{64})$',rules.read_text('utf8'),re.M)
    if len(hits)!=1:
        raise RuntimeError('M470_H4_CONTROL_NOT_APPROVED')
    return hits[0]


def h4_read_control(path,expected_sha,contract):
    """Verify the separately reviewed, immutable matched-control artifact.

    No historical OLD10M, resume, arbitrary CLI digest or partial control.
    """
    if not isinstance(expected_sha,str) or len(expected_sha)!=64 or any(x not in '0123456789abcdef' for x in expected_sha):
        raise RuntimeError('M470_H4_CONTROL_NOT_APPROVED')
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('M470_H4_CONTROL_FILE')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise RuntimeError('M470_H4_CONTROL_SHA')
    r=json.loads(raw)
    expected={'status':'M463_COVERAGE_LONGTRAIN_PASS','h4_arm':'control',
        'h4_common_contract':contract,'h4_spec':H4_SPEC,'training_performed':True,
        'source_commit':PIN,'source_checkpoint_sha256':SEED1_SHA,'orbax_sha256':H4_ORBAX_SHA,
        'trained_from':'C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE','training_seed':1,
        'experiment_design':'H4_MATCHED_C401M_SINGLE_PROCESS','stage':'easy_repeat_support','mode':'longtrain',
        'hardware_scaled_config':contract['config'],'train_reward_config':contract['reward'],
        'runner_sha256':contract['source_digests']['m463_push_ppo_train.py'],
        'steps_observed':10485760,'run_steps_observed':10485760,'rollouts_target':40,
        'rollouts_per_epoch':20,'cumulative_steps_start':0,'resume_fullstate_sha256':None,
        'coverage_gate_pass':True,'memory_gate_pass':True,'weights_finite':True,
        'callback_vs_final_equal':True,'gpu_total_mib':12288,'h4_legacy_reconciled':True,
        'h4_resume_forbidden':True,'promotion_authorized':False,'h4_control_report_sha256':None,
        'reward_PD_termination_source_unchanged':True,'h1_m1_enabled':False,'h1_m2_enabled':False}
    for key,v in expected.items():
        if key not in r or r[key]!=v:
            raise RuntimeError('M470_H4_CONTROL_CONTRACT '+key)
    if r.get('h3_m1_enabled',False) or not 0<r['gpu_peak_mib']<.95*12288 or r['gpu_monitor_errors']:
        raise RuntimeError('M470_H4_CONTROL_SAFETY')
    counts=h4_verify_ledger(r)
    h4_final_checks(r,r['coverage'],None)
    def artifact(name,sha):
        if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
            raise RuntimeError('M470_H4_CONTROL_ARTIFACT_PATH')
        q=path.parent/name
        if q.is_symlink() or not q.is_file() or digest(q)!=sha:
            raise RuntimeError('M470_H4_CONTROL_ARTIFACT_SHA '+name)
        return q
    im=r['instrumented_brax']
    if im['original_brax_sha256']!=contract['original_brax_sha256'] or im['instrumented_sha256']!=contract['generated_brax_sha256']:
        raise RuntimeError('M470_H4_CONTROL_GENERATED_PIN')
    artifact('m463_brax_ppo_instrumented.py',contract['generated_brax_sha256'])
    for step,policy,full in zip(H4_EDGES,r['checkpoints'],r['fullstate_checkpoints']):
        if (policy['file']!=f'm463_easy_repeat_support_longtrain_step_{step:09d}.msgpack' or
                full['file']!=f'm470_fullstate_step_{step:09d}.msgpack' or
                full['manifest']!=f'm470_fullstate_step_{step:09d}.json'):
            raise RuntimeError('M470_H4_CONTROL_ARTIFACT_NAMES')
        artifact(policy['file'],policy['sha256'])
        artifact(full['file'],full['sha256'])
        manifest=json.loads(artifact(full['manifest'],full['manifest_sha256']).read_bytes())
        for key,value in {'h4_arm':'control','h4_resume_forbidden':True,
            'h4_common_contract':contract,'cumulative_steps':step,
            'fullstate_sha256':full['sha256'],'parent_resume_sha256':None,
            'reset_edge':True,'optimizer_state_included':True,'training_rng_included':True,
            'evaluator_rng_included':True}.items():
            if key not in manifest or manifest[key]!=value:
                raise RuntimeError('M470_H4_CONTROL_MANIFEST '+key)
    return {'sha256':expected_sha,'counts':counts}


def h4_static_ledger_fixture():
    """No JAX, no simulation. Synthetic count/report/artifact contract tests."""
    import tempfile
    import struct
    def f32(x):
        return struct.unpack('f',struct.pack('f',x))[0]
    # All possible actual count sums for five 8192x32 rollouts.
    for n in range(5*TOTAL_STEPS_BATCH+1):
        if h4_integer_count(f32(n/5))!=n:
            raise RuntimeError('M470_H4_COUNT_RECONSTRUCTION')
    def counts(arm):
        c={}
        def setc(name,value=0):
            c['m463_h4_'+name+'_per_rollout']=value
        for name in ('bad','assigned','repeat_assigned','first_events','repeat_events',
                     'first_closed','repeat_closed','first_due','repeat_due','event_terminal',
                     'close_repeat','close_terminal','close_complete','close_boundary',
                     'broad_events','repeat_matched','repeat_fallback'):
            setc(name)
        for ci in range(4):
            for name in ('steps','assigned','repeat_assigned','first_due','repeat_due',
                         'first_closed','repeat_closed','first_events','repeat_events'):
                setc(f'class_{ci}_{name}')
        for kind in (1,2):
            for lo,hi in ((0,32),(33,239),(240,308),(309,348),(349,639)):
                setc(f'age_{kind}_{lo}_{hi}')
            for side in range(4):
                setc(f'support_{kind}_{side}')
            for w in H4_SPEC['tail_windows']:
                for v in ('executed','survived','same_command','same_unroll'):
                    setc(f'tail_{kind}_{w}_{v}')
        for lo,hi in ((0,29),(30,40),(41,639)):
            setc(f'gap_{lo}_{hi}')
        for j in range(6):
            for k in ('motion','direction'):
                setc(f'broad_{k}_{j}')
        for side in (1,2):
            setc(f'stand_{side}_events')
            for kind in (1,2):
                for w in H4_SPEC['tail_windows']:
                    setc(f'stand_{side}_tail_{kind}_{w}')
                    setc(f'stand_{side}_survived_{kind}_{w}')
            for motion in (3,4):
                setc(f'turn_{motion}_{side}_events')
            for motion in (0,3,4):
                for k in ('repeat_assigned','repeat_due','repeat_closed','repeat_events','matched','fallback'):
                    setc(f'target_{motion}_{side}_{k}')
        # 600 broad first + 120 broad repeat events per chunk.
        # Candidate has 6 target cells * 10 additional repeats.
        mature=60 if arm=='candidate' else 0
        setc('class_0_steps',TOTAL_STEPS_BATCH)
        setc('class_1_steps',4*TOTAL_STEPS_BATCH-(TOTAL_STEPS_BATCH if mature else 0))
        setc('class_3_steps',TOTAL_STEPS_BATCH if mature else 0)
        for ci,first,repeat in ((1,600,120),(3,mature,mature)):
            for k,n in (('assigned',first),('first_due',first),('first_events',first),
                        ('repeat_assigned',repeat),('repeat_due',repeat),('repeat_events',repeat)):
                setc(f'class_{ci}_{k}',n)
        for name,value in (('assigned',600+mature),('first_due',600+mature),
                           ('first_events',600+mature),('repeat_assigned',120+mature),
                           ('repeat_due',120+mature),('repeat_events',120+mature),
                           ('close_repeat',120+mature),('close_complete',600+mature),
                           ('broad_events',720),('repeat_matched',120+mature)):
            setc(name,value)
        setc('age_1_0_32',600)
        setc('age_1_240_308',mature)
        setc('age_2_33_239',120)
        setc('age_2_309_348',mature)
        setc('support_1_3',600+mature)
        setc('support_2_1',(120+mature)//2)
        setc('support_2_2',(120+mature)//2)
        setc('gap_30_40',120+mature)
        for j in range(6):
            for k in ('motion','direction'):
                setc(f'broad_{k}_{j}',120)
        for kind,total in ((1,600+mature),(2,120+mature)):
            for w in H4_SPEC['tail_windows']:
                n=total if kind==2 or w<=25 else 480
                setc(f'tail_{kind}_{w}_executed',n)
                setc(f'tail_{kind}_{w}_survived',n)
                setc(f'tail_{kind}_{w}_same_command',n)
        if mature:
            for side in (1,2):
                setc(f'stand_{side}_events',10)
                for w in H4_SPEC['tail_windows']:
                    setc(f'stand_{side}_tail_2_{w}',10)
                    setc(f'stand_{side}_survived_2_{w}',10)
                for motion in (0,3,4):
                    for k in ('repeat_assigned','repeat_due','repeat_events','matched'):
                        setc(f'target_{motion}_{side}_{k}',10)
                    if motion:
                        setc(f'turn_{motion}_{side}_events',10)
        return c
    q={'first_pending':0,'repeat_pending':0,'tail_pending':0}
    for ci in range(4):
        for k in ('first_pending','repeat_pending'):
            q[f'class_{ci}_{k}']=0
    for m in (0,3,4):
        for side in (1,2):
            q[f'target_{m}_{side}_pending']=0
    gpu={'total_mib':12288,'peak_used_mib':5000,'sampler_errors':[]}
    def build(arm):
        c=counts(arm)
        report={'h4_arm':arm,'h4_chunks':[],'h4_epochs':[]}
        for n in range(1,9):
            report['h4_chunks'].append({'chunk':n,'steps':n*5*TOTAL_STEPS_BATCH,
                 'passed':True,'gpu':dict(gpu),'counts':dict(c)})
            if n%4==0:
                ep={k:4*v for k,v in c.items()}
                res=h4_epoch_audit(ep,q,arm)
                if n==8:
                    res['checks']={k:v for k,v in res['checks'].items()
                        if not k.startswith(('stand_','turn_'))}
                    res['passed']=all(res['checks'].values())
                report['h4_epochs'].append({'counts':ep,'pending_at_reset':dict(q),'audit':res})
        total={k:8*v for k,v in c.items()}
        report['h4_final_counts']=total
        report['h4_final_pending']=dict(q)
        report['h4_final_exposure']=h4_epoch_audit(total,q,arm,final=True)
        if h4_verify_ledger(report)!=total:
            raise RuntimeError('M470_H4_FIXTURE_LEDGER')
        return report
    control=build('control')
    candidate=build('candidate')
    if not h4_matched_nominal_check(control['h4_final_counts'],candidate['h4_final_counts']):
        raise RuntimeError('M470_H4_FIXTURE_MATCHED_NOMINAL')
    def reject(fn):
        try:
            fn()
        except (RuntimeError,KeyError):
            return
        raise RuntimeError('M470_H4_FIXTURE_ACCEPTED_BAD_INPUT')
    for key in ('h4_first_events_per_rollout','h4_repeat_events_per_rollout',
                'h4_close_complete_per_rollout','h4_class_0_steps_per_rollout',
                'h4_stand_1_events_per_rollout'):
        bad=copy.deepcopy(candidate)
        bad['h4_chunks'][0]['counts']['m463_'+key]+=1
        reject(lambda:h4_verify_ledger(bad))
    # Fixture report has real bytes and hashes, but no actual trained weights.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        def write(name,data):
            p=root/name
            p.write_bytes(data)
            return digest(p)
        def jsonbytes(obj):
            return json.dumps(obj,sort_keys=True,allow_nan=False).encode('utf8')
        contract={'config':{'fixture':True},'reward':{'fixture':True},
                  'source_digests':{'m463_push_ppo_train.py':'fixture'},
                  'original_brax_sha256':H3_CAUSAL_ORIGINAL_BRAX_SHA,
                  'generated_brax_sha256':write('m463_brax_ppo_instrumented.py',b'fixture')}
        h=[]
        for step in (0,*H4_EDGES):
            metrics={}
            if step:
                metrics={PREFIX+'events_per_rollout':144.,
                    PREFIX+'push_magnitude_sum':60.,PREFIX+'phase_matched_per_rollout':100.,
                    PREFIX+'repeat_side_targeted_per_rollout':24.,
                    PREFIX+'repeat_side_matched_per_rollout':24.,
                    PREFIX+'repeat_side_fallback_per_rollout':0.,
                    PREFIX+'repeat_side_target_left_per_rollout':12.,
                    PREFIX+'repeat_side_target_right_per_rollout':12.,
                    PREFIX+'repeat_per_rollout':24.}
                for j in range(6):
                    metrics[PREFIX+f'direction_{j}_per_rollout']=24.
                    metrics[PREFIX+f'motion_{j}_per_rollout']=24.
                for j in range(3):
                    metrics[PREFIX+f'phase_target_{j}_per_rollout']=48.
            h.append({'additional_steps':step,'metrics':metrics})
        control.update(status='M463_COVERAGE_LONGTRAIN_PASS',h4_common_contract=contract,
            h4_spec=H4_SPEC,training_performed=True,source_commit=PIN,
            source_checkpoint_sha256=SEED1_SHA,orbax_sha256=H4_ORBAX_SHA,
            trained_from='C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE',training_seed=1,
            experiment_design='H4_MATCHED_C401M_SINGLE_PROCESS',stage='easy_repeat_support',
            mode='longtrain',hardware_scaled_config=contract['config'],
            train_reward_config=contract['reward'],runner_sha256='fixture',
            steps_observed=10485760,run_steps_observed=10485760,rollouts_target=40,
            rollouts_per_epoch=20,cumulative_steps_start=0,resume_fullstate_sha256=None,
            coverage_gate_pass=True,memory_gate_pass=True,weights_finite=True,
            callback_vs_final_equal=True,gpu_total_mib=12288,gpu_peak_mib=5000,gpu_monitor_errors=[],
            h4_legacy_reconciled=True,h4_resume_forbidden=True,promotion_authorized=False,
            h4_control_report_sha256=None,reward_PD_termination_source_unchanged=True,
            h1_m1_enabled=False,h1_m2_enabled=False,report_history=h,coverage=h4_legacy_audit(h),
            instrumented_brax={'original_brax_sha256':contract['original_brax_sha256'],
                               'instrumented_sha256':contract['generated_brax_sha256']},
            checkpoints=[],fullstate_checkpoints=[])
        for step in H4_EDGES:
            policy=f'm463_easy_repeat_support_longtrain_step_{step:09d}.msgpack'
            full=f'm470_fullstate_step_{step:09d}.msgpack'
            manifest=f'm470_fullstate_step_{step:09d}.json'
            ps=write(policy,b'fixture policy')
            fs=write(full,b'fixture fullstate')
            meta={'h4_arm':'control','h4_resume_forbidden':True,'h4_common_contract':contract,
                'cumulative_steps':step,'fullstate_sha256':fs,'parent_resume_sha256':None,
                'reset_edge':True,'optimizer_state_included':True,'training_rng_included':True,
                'evaluator_rng_included':True}
            ms=write(manifest,jsonbytes(meta))
            control['checkpoints'].append({'steps':step,'file':policy,'sha256':ps})
            control['fullstate_checkpoints'].append({'steps':step,'file':full,'sha256':fs,
                'manifest':manifest,'manifest_sha256':ms})
        path=root/'report.json'
        approved=write('report.json',jsonbytes(control))
        if h4_read_control(path,approved,contract)['counts']!=control['h4_final_counts']:
            raise RuntimeError('M470_H4_FIXTURE_CONTROL')
        reject(lambda:h4_read_control(path,None,contract))
        reject(lambda:h4_read_control(path,'0'*64,contract))
        reject(lambda:h4_read_control(path,approved,dict(contract,reward={'other':True})))
        bad=copy.deepcopy(control)
        bad['h4_arm']='candidate'
        wrong=write('report.json',jsonbytes(bad))
        reject(lambda:h4_read_control(path,wrong,contract))
        write('report.json',jsonbytes(control))
        write(policy,b'tampered')
        reject(lambda:h4_read_control(path,approved,contract))
    print('M470_H4_LEDGER_PROVENANCE_FIXTURE_PASS',flush=True)


def h4_static_regression(brax_train_path):
    """Offline: compile exact files and generated copies; do not import runtime."""
    import ast
    import subprocess
    import types
    root=Path(__file__).resolve().parent
    names=('m463_coverage_env.py','m463_instrument_brax.py','m463_push_ppo_train.py')
    for name in names:
        raw=(root/name).read_bytes()
        compile(raw,name,'exec')
        ast.parse(raw,filename=name)
    print('M470_H4_FULL_SOURCE_COMPILE_PASS',flush=True)
    import m463_instrument_brax as gen
    raw=Path(brax_train_path).read_bytes()
    if gen.sha256(raw)!=gen.PINNED_SHA or gen.PINNED_SHA!=H3_CAUSAL_ORIGINAL_BRAX_SHA:
        raise RuntimeError('M470_H4_STATIC_BRAX_PIN')
    historical=subprocess.check_output(['git','show',
        '3f4a92636adc94e758d7c4122e69fd2b1a7f77d5:m463_instrument_brax.py'],cwd=root)
    old=types.ModuleType('_h4_historical_generator')
    exec(compile(historical,'<historical generator>','exec'),old.__dict__)
    for resume in (False,True):
        for h3 in (False,True):
            before,sha0=old.make_patched_source(raw,enable_resume=resume,enable_h3=h3)
            after,sha1=gen.make_patched_source(raw,enable_resume=resume,enable_h3=h3)
            if before!=after or sha0!=sha1:
                raise RuntimeError('M470_H4_LEGACY_GENERATED_CHANGED')
            compile(after,'<legacy generated>','exec')
    print('M470_H4_LEGACY_GENERATED_BYTE_EQUAL_PASS 4/4',flush=True)
    legacy,_=gen.make_patched_source(raw,enable_resume=True)
    new,sha=gen.make_patched_source(raw,enable_resume=True,enable_h4=True)
    compile(new,'<H4 generated>','exec')
    tree=ast.parse(new)
    train=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='train')
    body=train.body
    if (not isinstance(body[0],ast.Expr) or not isinstance(body[1],ast.Raise)
            or 'M470_H4_EXECUTION_HOLD' not in ast.unparse(body[1])):
        raise RuntimeError('M470_H4_GENERATED_HOLD_MISSING')
    def span(s,start,end):
        a=s.index(start)
        return s[a:s.index(end,a)]
    source_step=span(legacy,'  def training_step(','  def training_epoch(')
    new_step=span(new,'  def training_step(','  def h4_chunk(')
    old_fields=''.join(', '+repr(k) for k in gen.EXT_FIELDS)
    all_fields=old_fields+''.join(', '+repr(k) for k in gen.H4_EXT_FIELDS)
    if new_step.count(all_fields)!=1 or new_step.count(gen.H4_METRICS_SOURCE)!=1:
        raise RuntimeError('M470_H4_GENERATED_STEP_ANCHOR')
    normalized=new_step.replace(all_fields,old_fields,1).replace(gen.H4_METRICS_SOURCE,'',1)
    if normalized!=source_step:
        raise RuntimeError('M470_H4_TRAINING_STEP_DRIFT')
    # Remove only H4 additions and replace only the H4 epoch wrapper; require
    # all remaining generated bytes to match the exact legacy generator.
    epoch_new=span(new,'  def h4_chunk(','  # Initialize model params and training state.')
    epoch_old=span(legacy,'  def training_epoch(','  # Initialize model params and training state.')
    normalized_all=new.replace(all_fields,old_fields,1).replace(gen.H4_METRICS_SOURCE,'',1)
    normalized_all=normalized_all.replace(epoch_new,epoch_old,1)
    normalized_all=normalized_all.replace('    h4_chunk_fn: Optional[Callable[..., None]] = None,\n','',1)
    normalized_all=normalized_all.replace("  raise RuntimeError('M470_H4_EXECUTION_HOLD')\n",'',1)
    if normalized_all!=legacy:
        raise RuntimeError('M470_H4_GENERATED_OUT_OF_SCOPE_CHANGE')
    if sha!=gen.PINNED_SHA:
        raise RuntimeError('M470_H4_GENERATED_SHA')
    print('M470_H4_GENERATED_COMPILE_AND_SCOPE_PASS',gen.sha256(new.encode('utf8')),flush=True)
    # Static checks on actual environment source, without importing it.
    env_tree=ast.parse((root/names[0]).read_bytes())
    fields=next(ast.literal_eval(x.value) for x in env_tree.body
                if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='H4_FIELDS' for t in x.targets))
    if tuple(fields)+('h4_actual_dx','h4_actual_dy')!=gen.H4_EXT_FIELDS:
        raise RuntimeError('M470_H4_EXTRA_SCHEMA')
    original_env=ast.parse(subprocess.check_output(['git','show',
        '3f4a92636adc94e758d7c4122e69fd2b1a7f77d5:m463_coverage_env.py'],cwd=root))
    def method(t,name):
        cls=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='CoverageJoystick')
        return ast.dump(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name))
    for name in ('_get_reward','sample_command','_cost_h1_m2_phase_foot_crossing'):
        if method(env_tree,name)!=method(original_env,name):
            raise RuntimeError('M470_H4_ENV_CONTROL_CHANGED '+name)
    print('M470_H4_ENV_STATIC_SCHEMA_PASS',len(fields)+2,flush=True)
    # Exercise the actual CLI on the exact checkout; no runtime flags can dispatch.
    from unittest.mock import patch
    from types import SimpleNamespace
    base=['--checkpoint','unused','--restore-report','unused','--output-dir','unused',
          '--stage','easy_repeat_support']
    cases=[]
    for arm in ('control','candidate'):
        cases.append((['--mode','plan','--h4-arm',arm],0))
        for mode in ('smoke','probe2','smoke3','pilot','longtrain'):
            cases.append((['--mode',mode,'--h4-arm',arm],2))
        for flags in (['--h1-m1'],['--h1-m2'],['--h3-m1'],['--h3-diagnostic-only'],
                      ['--resume-state','unused'],['--rollouts','40'],['--num-evals','3'],
                      ['--independent-budget','10M'],['--smoke-report','unused'],
                      ['--h4-control-report','unused'],['--h3-causal-report','unused']):
            cases.append((['--mode','plan','--h4-arm',arm]+flags,2))
    cases.append((['--mode','plan','--h4-control-report','unused'],2))
    for flags,code in cases:
        p=subprocess.run([sys.executable,'-B',str(root/names[2]),*base,*flags],
                         capture_output=True,text=True,cwd=root)
        if p.returncode!=code or (code==0 and 'M470_H4_PLAN_NO_GPU' not in p.stdout):
            raise RuntimeError('M470_H4_STATIC_CLI '+repr(flags)+' '+p.stderr)
    args=SimpleNamespace(h4_arm='candidate',stage='easy_repeat_support',mode='longtrain',
        h1_m1=False,h1_m2=False,h3_m1=False,h3_diagnostic_only=False,
        h3_causal_report=None,h3_old10m_report=None,resume_state=None,smoke_report=None,
        independent_budget=None,rollouts=None,num_evals=None,h4_control_report=None)
    try:
        run(args)
    except RuntimeError as exc:
        if str(exc)!='M470_H4_EXECUTION_HOLD':
            raise
    else:
        raise RuntimeError('M470_H4_DIRECT_RUN_NOT_BLOCKED')
    print('M470_H4_CLI_INTERLOCK_PASS',len(cases),'plus direct run',flush=True)
    h4_static_ledger_fixture()
    if 'jax' in sys.modules or 'brax' in sys.modules or 'mujoco' in sys.modules:
        raise RuntimeError('M470_H4_STATIC_RUNTIME_IMPORTED')
    print('M470_H4_STATIC_REGRESSION_PASS NO_JAX NO_SIMULATION NO_PPO',flush=True)



# H4-M1-1B separate diagnostic; production and readiness remain locked.
H4R_EXECUTION_AUTHORIZED = False
H4R_SPEC = {'num_envs':16,'batch_size':2,'num_minibatches':8,
            'unroll_length':32,'num_updates_per_batch':5,'seed':1,
            'rollouts_per_path':20,'rollouts_per_chunk':5,'ticks':640,
            'paths':['R0','R1','S'],'transitions_per_path':10240,
            'max_transitions':30720,'max_optimizer_updates':2400,
            'eval_transitions':0,'float_cap':1e-5,'eps_multiplier':8}
H4R_REWARD_SHA='84c285315e109ca8716476b0f527873046b441d35e3fe963570e5f2019fe3530'
H4R_LOCAL_SHA={
    'm462_ppo_warmstart_run':'2b8846723287b6999db3cc27fbf50e556ce6b6bfd19822ff2bfd7a84a53d5328',
    'm462_typed_orbax_stage':'2dc81e3c3f515745954d189eeaec53bc014fe0e399b98dbe9a3219586dfb85cc',
    'nvcc_compat':'c92cd0f79e04acc409c4d0d52c7b1cc2c5b71822aafbd1e4d957a949f5044e8f',
    'm463_coverage_env':'5a995942b87ba9514d9187fc7b864f33ee6eafa4d4994662f1faeb5c96e74e27'}
H4R_CONFIG_SHA='36ab984a162e053a58d6f4313369a95dde77fd84a4db61b54d42ab856fadd19b'


def h4r_require(ok,label):
    if not bool(ok):
        raise RuntimeError('M470_H4R_'+label)


def validate_h4r_cli(a):
    requested=getattr(a,'h4_readiness',False)
    if not requested:
        h4r_require(a.mode!='readiness' and getattr(a,'h4_readiness_reward_report',None) is None,'FLAG_REQUIRED')
        return
    h4r_require(a.mode in ('plan','readiness') and a.stage=='easy_repeat_support','MODE_STAGE')
    h4r_require(not any(getattr(a,k,False) for k in (
        'h1_m1','h1_m2','h3_m1','h3_diagnostic_only')) and all(
        getattr(a,k,None) is None for k in ('h4_arm','h4_control_report','h3_causal_report',
        'h3_old10m_report','smoke_report','resume_state','independent_budget','rollouts','num_evals')),
        'CONFLICT')
    h4r_require(getattr(a,'h4_readiness_reward_report',None) is not None,'REWARD_REFERENCE_REQUIRED')
    if a.mode!='plan':
        h4r_require(H4R_EXECUTION_AUTHORIZED is True and H4_EXECUTION_AUTHORIZED is False,
                     'EXECUTION_HOLD')


def h4r_distance(np,a,b,exact=False):
    a,b=np.asarray(a),np.asarray(b)
    h4r_require(a.shape==b.shape and a.dtype==b.dtype,'LEAF_SCHEMA')
    h4r_require(a.dtype.kind in 'biuf' and np.isfinite(a).all() and np.isfinite(b).all(),'LEAF_NONFINITE')
    if exact or a.dtype.kind!='f':
        return (0. if np.array_equal(a,b) else 1.),0.
    u,v=a.astype(np.float64),b.astype(np.float64)
    d=float(np.max(np.abs(u-v)/np.maximum(1.,np.maximum(np.abs(u),np.abs(v))),initial=0.))
    return d,float(np.finfo(a.dtype).eps)


def h4r_compare(np,reference,other,repeatability=None,exact_paths=()):
    h4r_require(set(reference)==set(other),'PATH_SCHEMA')
    if repeatability is not None:
        h4r_require(set(reference)==set(repeatability),'REPEATABILITY_SCHEMA')
    rows={}
    for k,a in reference.items():
        d,eps=h4r_distance(np,a,other[k],k in exact_paths)
        limit=0. if eps==0 else H4R_SPEC['float_cap']
        if repeatability is not None and eps:
            limit=min(limit,max(repeatability[k]['distance'],H4R_SPEC['eps_multiplier']*eps))
        rows[k]={'distance':d,'limit':limit,'passed':d<=limit}
    return {'paths':rows,'passed':all(v['passed'] for v in rows.values())}


def h4r_counts(metrics,rollouts):
    result={}
    for k,v in metrics.items():
        # Weighted reward sums are floating metrics, not counts.
        if k.endswith('_per_rollout') and not k.endswith('_weighted_sum_per_rollout'):
            x=float(v)*rollouts
            h4r_require(math.isfinite(x) and 0<=x<=512*rollouts and abs(x-round(x))<=.125,
                        'COUNT '+k)
            result[k]=int(round(x))
    h4r_require(result['h4r_nonfinite_per_rollout']==0 and
                 result['m463_h4_bad_per_rollout']==0,'ROLLOUT_BAD')
    h4r_require(sum(result[f'm463_h4_class_{c}_steps_per_rollout'] for c in range(4))
                ==512*rollouts,'PARTITION')
    return result


def h4r_pending(np,info):
    names=('h4_first_pending','h4_repeat_pending','h4_tail_kind','h4_class',
           'h4_repeat_origin_motion','m463_direction')
    a={k:np.asarray(info[k]).reshape(-1) for k in names}
    h4r_require(all(v.size==16 for v in a.values()),'PENDING_SHAPE')
    for k,allowed in (('h4_first_pending',(0,1)),('h4_repeat_pending',(0,1)),
                      ('h4_tail_kind',(0,1,2)),('h4_class',(0,1,2,3))):
        h4r_require(np.isin(a[k],allowed).all(),'PENDING_VALUE '+k)
    fp,rp,cls=a['h4_first_pending']>0,a['h4_repeat_pending']>0,a['h4_class']
    q={'first_pending':int(fp.sum()),'repeat_pending':int(rp.sum()),
       'tail_pending':int((a['h4_tail_kind']>0).sum())}
    for ci in range(4):
        q[f'class_{ci}_first_pending']=int((fp&(cls==ci)).sum())
        q[f'class_{ci}_repeat_pending']=int((rp&(cls==ci)).sum())
    for m in (0,3,4):
        for side in (1,2):
            q[f'target_{m}_{side}_pending']=int((rp&(cls==3)&
                (a['h4_repeat_origin_motion']==m)&(a['m463_direction']==side+1)).sum())
    return q


def h4r_ledger(np,counts,env_info):
    pending=h4r_pending(np,env_info)
    checks=h4_epoch_audit(counts,pending,'control')['checks']
    excluded=('step_partition','nominal_fraction')
    h4r_require(all(k in checks for k in excluded),'LEDGER_SCHEMA')
    checks={k:v for k,v in checks.items() if k not in excluded}
    checks['readiness_step_partition']=sum(counts[f'm463_h4_class_{c}_steps_per_rollout']
                                          for c in range(4))==10240
    checks['legacy_event_reconcile']=(counts['m463_events_per_rollout']==
        counts['m463_h4_first_events_per_rollout']+counts['m463_h4_repeat_events_per_rollout'])
    checks['legacy_repeat_reconcile']=(counts['m463_repeat_per_rollout']==
                                      counts['m463_h4_repeat_events_per_rollout'])
    return {'checks':checks,'pending':pending,'excluded_training_scale_gates':list(excluded),
            'passed':all(checks.values())}


def h4r_audit_callback(api,args,report,persist,gpu_check):
    """Host only: compare full actual carries, never write a weight artifact."""
    jax,np=api['jax'],api['numpy']
    from flax import serialization
    import optax
    from m462_typed_orbax_stage import checkpoint_parts,compare_exact
    expected_parts=checkpoint_parts(serialization.msgpack_restore(args.checkpoint.read_bytes()))
    schedule=[('initial',0),('R0_before',20),('R0',20),('R1_before',20),('R1',20)]
    for n in range(1,5):
        schedule.extend([('S_before',n),('S_chunk',n)])
    schedule.append(('S',20))
    initial_hash=None
    initial_count=None
    expected_keys=[]
    ref=None
    ref_counts=None
    repeatability=None
    split_counts={}
    prior_split_hash=None
    index=0
    def snapshot(tree):
        paths,structure=jax.tree_util.tree_flatten_with_path(tree)
        values={}
        for p,v in paths:
            name=jax.tree_util.keystr(p)
            h4r_require(name not in values,'DUPLICATE_PATH')
            a=np.array(jax.device_get(v),copy=True)
            h4r_require(a.dtype.kind in 'biuf' and np.isfinite(a).all(),'NONFINITE '+name)
            values[name]=a
        return values,str(structure)
    def digest_snapshot(values,structure):
        h=hashlib.sha256(structure.encode())
        for name,v in sorted(values.items()):
            h.update(json.dumps([name,str(v.dtype),list(v.shape)]).encode()+b'\0')
            h.update(v.tobytes())
        return h.hexdigest()
    def exact_paths(values):
        return tuple(k for k in values if k.endswith(('.done','.count',"['command']","['truncation']","['episode_done']")))
    def state_checks(carry,rollouts):
        state,env,key=carry
        h4r_require(np.asarray(state.env_steps).shape==(1,) and int(state.env_steps[0])==rollouts*512,'ENV_STEPS')
        adam=jax.tree_util.tree_leaves(state.optimizer_state,
            is_leaf=lambda x:isinstance(x,optax.ScaleByAdamState))
        adam=[v for v in adam if isinstance(v,optax.ScaleByAdamState)]
        h4r_require(len(adam)==1 and int(adam[0].count[0])==rollouts*40,'ADAM_COUNT')
        if rollouts==0:
            h4r_require(all(not np.any(np.asarray(x)) for x in
                jax.tree_util.tree_leaves((adam[0].mu,adam[0].nu))),'ADAM_FRESH')
        cnt=np.asarray(state.normalizer_params.count)
        h4r_require(cnt.shape==(1,),'NORMALIZER_COUNT_SHAPE')
        expected=initial_count.copy()
        for _ in range(rollouts):
            expected=np.add(expected,np.asarray(512,dtype=expected.dtype),dtype=expected.dtype)
        h4r_require(np.array_equal(cnt,expected),'NORMALIZER_COUNT')
        h4r_require(np.array_equal(np.asarray(key),expected_keys[rollouts]),'RNG_KEY')
        return {'env_steps':rollouts*512,'adam_count':rollouts*40,
                'normalizer_count':cnt.item(),'rng_key':np.asarray(key).tolist()}
    def callback(label,n,carry,metrics):
        nonlocal initial_hash,initial_count,index,ref,ref_counts,repeatability,prior_split_hash
        h4r_require(index<len(schedule) and schedule[index]==(label,n),'CALL_SEQUENCE')
        index+=1
        gpu_check()
        values,structure=snapshot(carry)
        state_hash=digest_snapshot(values,structure)
        if label=='initial':
            initial_hash=state_hash
            ts=carry[0]
            initial_count=np.array(ts.normalizer_params.count,copy=True)
            key=carry[2]
            for j in range(21):
                expected_keys.append(np.array(key,copy=True))
                if j<20:
                    key=jax.vmap(lambda k:jax.random.split(k,3)[2])(key)
            restored=tuple(serialization.to_state_dict(jax.tree_util.tree_map(lambda x:x[0],v))
                           for v in (ts.normalizer_params,ts.params.policy,ts.params.value))
            h4r_require(compare_exact(expected_parts,restored,jax,np,'H4R_INITIAL_RESTORE')==25,
                         'RESTORED_TENSOR_COUNT')
            report['initial']={'carry_sha256':initial_hash,'restored_weights_exact':True,
                               **state_checks(carry,0)}
            persist()
            return
        if label.endswith('_before'):
            rollouts=(n-1)*5 if label=='S_before' else 0
            h4r_require(state_hash==(initial_hash if rollouts==0 else prior_split_hash),'CARRY_BOUNDARY')
            state_checks(carry,rollouts)
            charge=5 if label=='S_before' else 20
            report['attempted_transitions']+=charge*512
            report['attempted_optimizer_updates']+=charge*40
            h4r_require(report['attempted_transitions']<=30720 and
                         report['attempted_optimizer_updates']<=2400,'BUDGET')
            report['dispatches'].append({'label':label,'index':n,'rollouts':charge,
                'input_sha256':state_hash})
            persist()
            return
        rollouts=n*5 if label=='S_chunk' else 20
        checks=state_checks(carry,rollouts)
        m={k:np.array(jax.device_get(v),copy=True) for k,v in metrics.items()}
        h4r_require(all(v.ndim==0 and np.isfinite(v).all() for v in m.values()),'METRIC_SCHEMA')
        counts=h4r_counts(m,5 if label=='S_chunk' else 20)
        if label!='S':
            completed=5 if label=='S_chunk' else 20
            report['recorded_transitions']+=completed*512
            report['recorded_optimizer_updates']+=completed*40
        row={'label':label,'index':n,'carry_sha256':state_hash,**checks,
             'metrics':{k:v.item() for k,v in m.items()},'counts':counts}
        report['boundaries'].append(row)
        if label=='S_chunk':
            prior_split_hash=state_hash
            for k,v in counts.items():
                split_counts[k]=split_counts.get(k,0)+v
            persist()
            return
        row['ledger']=h4r_ledger(np,counts,carry[1].info)
        persist()
        h4r_require(row['ledger']['passed'],'LEDGER')
        # Compare count metrics as integer totals, not floating reductions.
        for k,v in m.items():
            if k not in counts:
                values['metric:'+k]=v
        if label=='R0':
            ref=(values,structure)
            ref_counts=counts
        else:
            h4r_require(structure==ref[1] and counts==ref_counts,'REFERENCE_DISCRETE_METRICS')
            if label=='S':
                h4r_require(counts==split_counts,'CHUNK_REDUCTION')
            result=h4r_compare(np,ref[0],values,
                repeatability if label=='S' else None,exact_paths(ref[0]))
            report['comparisons'][label]=result
            persist()
            h4r_require(result['passed'],'REFERENCE_REPEATABILITY' if label=='R1' else 'SPLIT_EQUIVALENCE')
            if label=='R1':
                repeatability=result['paths']
        persist()
    return callback


def h4r_bootstrap_pins(args,*,h4cc=False,h4m6=None):
    from importlib.machinery import PathFinder
    root=Path(__file__).resolve().parent
    compat=root.parent/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'
    paths={}
    local_pins=dict(H4R_LOCAL_SHA)
    if h4cc:
        local_pins['m463_coverage_env']='d131e0bd14f6f1c8730eef5aaac32a6ef11e30ca19b7efb0a0751d1822991d01'
    if h4m6 is not None:
        h4m6_require(h4cc and h4m6['id']==H4M6_ID,'BOOTSTRAP_MODE')
        identity=h4m6_source_identity(root)
        h4m6_require(identity['sha256']==h4m6['source_sha256'],'BOOTSTRAP_SOURCE')
        local_pins['m463_coverage_env']=identity['files']['m463_coverage_env.py']
    for name,expected in local_pins.items():
        search=[str(compat),*sys.path] if name=='nvcc_compat' else sys.path
        spec=PathFinder.find_spec(name,search)
        h4r_require(spec is not None and spec.origin is not None,'MODULE_MISSING '+name)
        actual=Path(spec.origin).resolve()
        intended=(compat if name=='nvcc_compat' else root)/(name+'.py')
        h4r_require(actual==intended.resolve() and digest(actual)==expected,'MODULE_PIN '+name)
        paths[name]=str(actual)
    for name,blob in (('train','b36b69a38d2ebb1a218342b3f34386d47624383e'),
                      ('paper_config','6dc390de5eec3cd0b036b2d766c71e3164e1b17f')):
        spec=PathFinder.find_spec(name,sys.path)
        h4r_require(spec is not None and spec.origin is not None,'MODULE_MISSING '+name)
        actual=Path(spec.origin).resolve()
        raw=actual.read_bytes()
        actual_blob=hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
        h4r_require(actual==root/(name+'.py') and actual_blob==blob,'MODULE_PIN '+name)
        paths[name]=str(actual)
    import subprocess
    for name in ('m463_instrument_brax','m463_push_ppo_train'):
        spec=PathFinder.find_spec(name,sys.path)
        h4r_require(spec is not None and spec.origin is not None,'MODULE_MISSING '+name)
        actual=Path(spec.origin).resolve()
        h4r_require(actual==root/(name+'.py'),'MODULE_ORIGIN '+name)
        tracked=subprocess.check_output(['git','show','HEAD:'+name+'.py'],cwd=root)
        h4r_require(actual.read_bytes()==tracked,'TRACKED_SOURCE '+name)
        if name=='m463_instrument_brax':
            paths[name]=str(actual)
    h4r_require(digest(root/'paper_config_preflight.json')==H4R_CONFIG_SHA,'PAPER_CONFIG_PIN')
    for path,sha in ((args.checkpoint,SEED1_SHA),(args.restore_report,H3_CAUSAL_RESTORE_SHA),
                     (args.h4_readiness_reward_report,H4R_REWARD_SHA)):
        h4r_require(path.is_file() and not path.is_symlink() and digest(path)==sha,'INPUT_PIN '+str(path))
    return paths


def h4r_run(args):
    validate_h4r_cli(args)
    h4r_require(args.mode=='readiness' and H4R_EXECUTION_AUTHORIZED is True and
                 H4_EXECUTION_AUTHORIZED is False,'EXECUTION_HOLD')
    import shutil
    import traceback
    h4r_require(not args.output_dir.exists() and args.output_dir.parent.is_dir(),'OUTPUT')
    h4r_require(shutil.disk_usage(args.output_dir.parent).free>=2*1024**3,'DISK')
    args.output_dir.mkdir(exist_ok=False)
    report={'status':'M470_H4R_IN_PROGRESS','spec':H4R_SPEC,'attempted_transitions':0,
            'attempted_optimizer_updates':0,'recorded_transitions':0,'recorded_optimizer_updates':0,
            'evaluation_transitions':0,'dispatches':[],'boundaries':[],'comparisons':{},
            'production_authorized':False,'promotion_authorized':False,'weights_retained':False,
            'runner_sha256':digest(Path(__file__))}
    def persist():
        path=args.output_dir/'report.json'
        tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report,allow_nan=False,indent=2)+'\n',encoding='utf8')
        tmp.replace(path)
    persist()
    monitor=None
    try:
        pins=h4r_bootstrap_pins(args)
        report['resolved_modules']=pins
        from m462_ppo_warmstart_run import read_restore_gate,check_source_and_config,GPUMonitor,sample_gpu
        from m463_instrument_brax import build_h4_readiness_ppo
        gate,orbax_dir=read_restore_gate(args.restore_report,args.checkpoint)
        h4r_require(gate['orbax_dir_sha256']==H4_ORBAX_SHA,'ORBAX_PIN')
        report['frozen_inputs']={'checkpoint_sha256':SEED1_SHA,'restore_report_sha256':H3_CAUSAL_RESTORE_SHA,
                                'orbax_sha256':H4_ORBAX_SHA,'reward_reference_sha256':H4R_REWARD_SHA}
        monitor=GPUMonitor()
        def gpu_check():
            h4r_require(shutil.disk_usage(args.output_dir).free>=2*1024**3,'DISK')
            g=sample_gpu()
            snap=monitor.snapshot()
            h4r_require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288 and
                not snap['sampler_errors'] and (snap['peak_used_mib'] is None or
                snap['peak_used_mib']<.95*12288),'GPU')
            report['gpu']=snap
        with monitor:
            gpu_check()
            # Sole bootstrap invocation; it applies NVCC before importing JAX.
            api=check_source_and_config('A')
            jax=api['jax']
            h4r_require(jax.__version__=='0.4.35' and jax.default_backend()=='gpu' and
                         jax.process_count()==jax.local_device_count()==1,'JAX_RUNTIME')
            from m463_coverage_env import CoverageJoystick
            for name,path in pins.items():
                mod=sys.modules.get(name)
                h4r_require(mod is not None and Path(mod.__file__).resolve()==Path(path),'IMPORTED_ORIGIN '+name)
            h4r_require(h4r_bootstrap_pins(args)==pins,'BOOTSTRAP_IMMUTABILITY')
            scaled=copy.deepcopy(api['scaled'])
            h4r_require(m470_fingerprint(scaled.to_dict())==H3_FP,'PRODUCTION_CONFIG')
            scaled.num_envs=16
            scaled.batch_size=2
            scaled.num_timesteps=10240
            scaled.num_evals=1
            cfg=scaled.to_dict()
            report['scaled_config']=copy.deepcopy(cfg)
            source_cfg=api['source_cfg']
            env_cfg=copy.deepcopy(source_cfg)
            reward=json.loads(args.h4_readiness_reward_report.read_bytes())['train_reward_config']
            h4r_require(source_cfg.to_dict()['reward_config']==reward and
                        bool(source_cfg.push_config.enable),'ORIGINAL_REWARD')
            env_cfg.push_config.enable=False
            h4r_require(env_cfg.to_dict()['reward_config']==reward,'REWARD_UNCHANGED')
            env=CoverageJoystick(config=env_cfg,stage='easy_repeat_support',h4_arm='control')
            h4r_require(env.action_size==29 and float(env.dt)==.02,'ENV_CONFIG')
            mod,meta=build_h4_readiness_ppo(api['ppo'],args.output_dir/'h4_readiness_generated.py')
            report['generated']=meta
            factory=functools.partial(api['networks'].make_ppo_networks,**cfg.pop('network_factory'))
            audit=h4r_audit_callback(api,args,report,persist,gpu_check)
            report['bootstrap_pass']=True
            persist()
            print('M470_H4R_BOOTSTRAP_PASS',flush=True)
            result=mod.readiness_train(environment=env,wrap_env_fn=api['wrapper'].wrap_for_brax_training,
                 network_factory=factory,seed=1,randomization_fn=api['randomizer'],
                 restore_checkpoint_path=str(orbax_dir.resolve()),h4r_audit_fn=audit,**cfg)
            h4r_require(result is None,'NO_PARAMETER_RETURN')
            h4r_require(report['attempted_transitions']==report['recorded_transitions']==30720 and
                report['attempted_optimizer_updates']==report['recorded_optimizer_updates']==2400 and
                len(report['dispatches'])==6 and len(report['boundaries'])==7 and
                all(report['comparisons'][k]['passed'] for k in ('R1','S')),'FINAL')
            h4r_require(h4r_bootstrap_pins(args)==pins,'INPUT_IMMUTABILITY')
            read_restore_gate(args.restore_report,args.checkpoint)
            gpu_check()
        report['gpu']=monitor.snapshot()
        h4r_require(report['gpu']['samples']>0 and not report['gpu']['sampler_errors'] and
                     0<report['gpu']['peak_used_mib']<.95*12288,'FINAL_GPU')
        report['status']='M470_H4R_NUMERICAL_PASS'
        persist()
    except BaseException as exc:
        report['status']='M470_H4R_STOPPED'
        report['error']=type(exc).__name__+': '+str(exc)
        report['traceback']=traceback.format_exc()
        if monitor is not None:
            report['gpu']=monitor.snapshot()
        persist()
        raise



def h4r_static_regression(brax_train_path,technical_report):
    """Exact checkout compile and focused non-runtime readiness regression."""
    import ast
    import subprocess
    import types
    import numpy as np
    root=Path(__file__).resolve().parent
    names=('m463_push_ppo_train.py','m463_instrument_brax.py')
    for name in names:
        raw=(root/name).read_bytes()
        compile(raw,name,'exec')
        ast.parse(raw)
    import m463_instrument_brax as gen
    raw=Path(brax_train_path).read_bytes()
    h4r_require(gen.sha256(raw)==gen.PINNED_SHA==H3_CAUSAL_ORIGINAL_BRAX_SHA,'STATIC_BRAX')
    base='7ae983b6b2ce4f889b0e1d5a7eb4ef1b4d70d212'
    old_source=subprocess.check_output(['git','show',base+':m463_instrument_brax.py'],cwd=root)
    old=types.ModuleType('_h4r_old_generator')
    exec(compile(old_source,'<old generator>','exec'),old.__dict__)
    for resume,h3,h4 in [(False,False,False),(False,True,False),(True,False,False),
                          (True,True,False),(True,False,True)]:
        flags=dict(enable_resume=resume,enable_h3=h3,enable_h4=h4)
        before=old.make_patched_source(raw,**flags)
        after=gen.make_patched_source(raw,**flags)
        h4r_require(before==after,'LEGACY_GENERATED_DRIFT')
        compile(after[0],'<unchanged generated>','exec')
    print('M470_H4R_LEGACY_GENERATED_BYTE_EQUAL_PASS 5/5',flush=True)
    production,_=gen.make_patched_source(raw,enable_resume=True,enable_h4=True)
    diagnostic,sha=gen.make_h4_readiness_source(raw)
    compile(diagnostic,'<readiness generated>','exec')
    def function(source,name):
        return next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
    train=function(diagnostic,'readiness_train')
    prod=function(production,'train')
    h4r_require(isinstance(train.body[1],ast.Raise) and
                 'M470_H4_READINESS_EXECUTION_HOLD' in ast.unparse(train.body[1]),'GENERATED_HOLD')
    h4r_require(isinstance(prod.body[1],ast.Raise) and
                 'M470_H4_EXECUTION_HOLD' in ast.unparse(prod.body[1]),'PRODUCTION_HOLD')
    h4r_require(not any(isinstance(x,ast.FunctionDef) and x.name=='train'
                    for x in ast.parse(diagnostic).body),'NO_PRODUCTION_ENTRY')
    h4r_require('acting.Evaluator(' not in diagnostic and 'policy_params_fn(current_step' not in diagnostic
                 and 'm470_state_checkpoint_fn(current_step' not in diagnostic
                 and 'for it in range(num_evals_after_init)' not in diagnostic,'NO_EVAL_OR_SAVE')
    normalized=diagnostic.replace(gen.H4R_FINITE_SOURCE,'',1)
    for name in ('training_step','h4_chunk','sgd_step','minibatch_step'):
        def nested(s,entry):
            f=function(s,entry)
            return ast.dump(next(x for x in f.body if isinstance(x,ast.FunctionDef) and x.name==name))
        h4r_require(nested(normalized,'readiness_train')==nested(production,'train'),'PPO_DRIFT '+name)
    h4r_require(sha==gen.PINNED_SHA and diagnostic.count(gen.H4R_FINITE_SOURCE)==1 and
                diagnostic.count(gen.H4R_CONTRACT)==1 and diagnostic.endswith(gen.H4R_EXIT),'GENERATOR_SCOPE')
    # The diagnostic is a separate entry with one fixed dispatch schedule.
    h4r_require(diagnostic.count('h4r_reference(*initial)')==1 and
                diagnostic.count('h4_chunk(*carry)')==1 and
                "for label in ('R0', 'R1'):" in diagnostic and
                'for idx in range(4):' in gen.H4R_EXIT and
                'length=20)' in gen.H4R_EXIT,'DISPATCH_SCOPE')
    print('M470_H4R_GENERATED_COMPILE_SCOPE_PASS',gen.sha256(diagnostic.encode()),flush=True)
    h4r_require(not H4_EXECUTION_AUTHORIZED and not H4R_EXECUTION_AUTHORIZED,'HOLD')
    base_cli=[sys.executable,'-B',str(root/names[0]),'--checkpoint','unused',
        '--restore-report','unused','--output-dir','unused','--stage','easy_repeat_support']
    valid=['--h4-readiness','--h4-readiness-reward-report','unused']
    cases=[(['--mode','plan']+valid,0),
           (['--mode','plan','--h4-readiness'],2),
           (['--mode','plan','--h4-readiness-reward-report','unused'],2),
           (['--mode','readiness'],2)]
    for mode in ('readiness','smoke','probe2','smoke3','pilot','longtrain'):
        cases.append((['--mode',mode]+valid,2))
    for flag in (['--h4-arm','control'],['--h4-control-report','unused'],['--h1-m1'],
                 ['--h1-m2'],['--h3-m1'],['--h3-diagnostic-only'],
                 ['--h3-causal-report','unused'],['--h3-old10m-report','unused'],
                 ['--smoke-report','unused'],['--resume-state','unused'],
                 ['--independent-budget','10M'],['--rollouts','20'],['--num-evals','1']):
        cases.append((['--mode','plan']+valid+flag,2))
    for flags,code in cases:
        p=subprocess.run(base_cli+flags,capture_output=True,text=True,cwd=root)
        h4r_require(p.returncode==code and (code!=0 or 'M470_H4R_PLAN_NO_GPU' in p.stdout),
                    'STATIC_CLI '+repr(flags)+' '+p.stderr)
    args=types.SimpleNamespace(mode='readiness',stage='easy_repeat_support',
                h4_readiness=True,h4_readiness_reward_report=Path('unused'))
    for fn in (run,h4r_run):
        try:
            fn(args)
        except RuntimeError as exc:
            h4r_require(str(exc)=='M470_H4R_EXECUTION_HOLD','DIRECT_HOLD')
        else:
            raise RuntimeError('M470_H4R_DIRECT_ENTRY_NOT_BLOCKED')
    print('M470_H4R_CLI_HOLD_PASS',len(cases),'plus two direct entry guards',flush=True)
    def reject(fn):
        try:
            fn()
        except RuntimeError:
            return
        raise RuntimeError('M470_H4R_FIXTURE_ACCEPTED_INVALID')
    # Comparison and count arithmetic, actual helper functions, no JAX.
    a={'weights':np.array([1.],np.float32),'key':np.array([1,2],np.uint32)}
    same=h4r_compare(np,a,a)
    h4r_require(same['passed'],'FIXTURE_EQUAL')
    other=dict(a,weights=np.array([1.+2e-6],np.float32))
    h4r_require(h4r_compare(np,a,other)['passed'] and
                 not h4r_compare(np,a,other,same['paths'])['passed'],'FIXTURE_EPS')
    h4r_require(h4r_compare(np,a,other,h4r_compare(np,a,other)['paths'])['passed'],'FIXTURE_REPEAT')
    h4r_require(not h4r_compare(np,a,dict(a,weights=np.array([1.0001],np.float32)))['passed'],'FIXTURE_CAP')
    h4r_require(not h4r_compare(np,a,dict(a,key=np.array([1,3],np.uint32)))['passed'],'FIXTURE_DISCRETE')
    reject(lambda:h4r_compare(np,a,dict(a,weights=np.array([np.nan],np.float32))))
    reject(lambda:h4r_compare(np,a,dict(a,weights=np.array([1.,2.],np.float32))))
    reject(lambda:h4r_compare(np,a,dict(a,weights=np.array([1.],np.float64))))
    for n in (5,20):
        for count in range(512*n+1):
            v=np.float32(count/n)
            h4r_require(abs(float(v)*n-count)<=.125,'COUNT_ARITHMETIC')
    # Previously accepted natural control report is ONLY a synthetic fixture:
    # tile its four lanes fourfold, not a new physical experiment.
    path=Path(technical_report)
    h4r_require(digest(path)=='b355a69d0e8d11e4f72c5242e7c7f2c5ef982e30acb1bdd493ed122cd58e1f9a',
                 'FIXTURE_REPORT_PIN')
    r=json.loads(path.read_bytes())
    control=next(v for v in r['natural'] if v['arm']=='control')
    m={}
    for row in control['rollout_counts']:
        for k,v in row.items():
            if k.endswith('_per_rollout'):
                m[k]=m.get(k,0)+4*int(v)
    m={k:np.float32(v/20) for k,v in m.items()}
    m['h4r_nonfinite_per_rollout']=0.
    m['m463_events_per_rollout']=m['m463_h4_first_events_per_rollout']+m['m463_h4_repeat_events_per_rollout']
    m['m463_repeat_per_rollout']=m['m463_h4_repeat_events_per_rollout']
    counts=h4r_counts(m,20)
    info={k:np.tile(v,4) for k,v in control['pending_at_end'].items()}
    ledger=h4r_ledger(np,counts,info)
    h4r_require(ledger['passed'],'FIXTURE_LEDGER')
    bad=dict(counts)
    bad['m463_h4_first_events_per_rollout']+=1
    h4r_require(not h4r_ledger(np,bad,info)['passed'],'FIXTURE_BAD_LEDGER')
    reject(lambda:h4r_counts(dict(m,h4r_nonfinite_per_rollout=.05),20))
    h4r_require('jax' not in sys.modules and 'brax' not in sys.modules and 'mujoco' not in sys.modules,
                 'STATIC_RUNTIME_IMPORTED')
    print('M470_H4R_NUMERICAL_LEDGER_FIXTURE_PASS',len(ledger['checks']),flush=True)
    print('M470_H4R_STATIC_PASS NO_SIMULATION NO_PPO',flush=True)



# H4-M1-1C isolated causal diagnostic, separate from both prior entries.
H4C_EXECUTION_AUTHORIZED = False
H4C_AUTHORIZED_RUN = None
H4C_SPEC = {'paths':['A0','A1','B0','B1'], 'num_envs':16,'batch_size':2,
            'unroll_length':32,'num_minibatches':8,'num_updates_per_batch':5,
            'seed':1,'max_physical_transitions':1024,'max_optimizer_updates':80,
            'eval_transitions':0,'float_cap':1e-5}
H4C_FAILURE_SHA = 'c2d07bb0f270329b27ad6927d367027731ea0ecbd985070d8d8fea03934fd5ff'
H4C_BRAX_BLOBS = {
    'brax.training.acting':'ec6f81d3e3c48e96a22f7df292fb1a98b63f938d',
    'brax.training.agents.ppo.losses':'93d0b1e39a610d55a5a8ed5d5027fc8684945417',
    'brax.training.gradients':'185419fb4ea35d232be50ebf3561e850f74ea692',
    'brax.training.acme.running_statistics':'4361e185b49917543dfa759feb521be14aeaa487',
    'brax.training.agents.ppo.networks':'1922bad3499988bc664c3024f9640332d66982d5'}


def h4c_require(ok,label):
    if not bool(ok):
        raise RuntimeError('M470_H4C_'+label)


def validate_h4c_cli(a):
    requested=getattr(a,'h4_causal',False)
    if not requested:
        h4c_require(a.mode!='causal' and getattr(a,'h4_causal_failure_report',None) is None,
                     'FLAG_REQUIRED')
        return
    h4c_require(a.mode in ('plan','causal') and a.stage=='easy_repeat_support','MODE_STAGE')
    h4c_require(not any(getattr(a,k,False) for k in
        ('h1_m1','h1_m2','h3_m1','h3_diagnostic_only','h4_readiness')) and
        all(getattr(a,k,None) is None for k in
            ('h4_arm','h4_control_report','h3_causal_report','h3_old10m_report',
             'smoke_report','resume_state','independent_budget','rollouts','num_evals')),
        'CONFLICT')
    h4c_require(getattr(a,'h4_causal_failure_report',None) is not None and
                getattr(a,'h4_readiness_reward_report',None) is not None,'REFERENCE_REQUIRED')
    if a.mode!='plan':
        h4c_require(H4C_EXECUTION_AUTHORIZED is True and H4R_EXECUTION_AUTHORIZED is False
                    and H4_EXECUTION_AUTHORIZED is False,'EXECUTION_HOLD')


class H4CBudget:
    """Host-only strict sequence and pre-dispatch accounting."""
    schedule=('initial','A0_before','A0','A1_before','A1',
              'B0_before','B0','B1_before','B1')

    def __init__(self,report):
        self.report=report
        self.index=0

    def advance(self,label):
        h4c_require(self.index<len(self.schedule) and self.schedule[self.index]==label,'SEQUENCE')
        self.index+=1
        r=self.report
        if label.endswith('_before'):
            phys,updates=(512,0) if label.startswith('A') else (0,40)
            h4c_require(r['attempted_physical_transitions']+phys<=1024 and
                        r['attempted_optimizer_updates']+updates<=80,'BUDGET')
            r['attempted_physical_transitions']+=phys
            r['attempted_optimizer_updates']+=updates
            r['dispatches'].append(label[:-7])
        elif label!='initial':
            r['recorded_physical_transitions']+=512 if label.startswith('A') else 0
            r['recorded_optimizer_updates']+=40 if label.startswith('B') else 0
            r['completed'].append(label)

    def complete(self):
        r=self.report
        h4c_require(self.index==len(self.schedule) and
            r['dispatches']==r['completed']==list(H4C_SPEC['paths']) and
            r['attempted_physical_transitions']==r['recorded_physical_transitions']==1024 and
            r['attempted_optimizer_updates']==r['recorded_optimizer_updates']==80,'FINAL_BUDGET')


def h4c_report():
    return {'status':'M470_H4C_IN_PROGRESS','spec':copy.deepcopy(H4C_SPEC),
            'attempted_physical_transitions':0,'recorded_physical_transitions':0,
            'attempted_optimizer_updates':0,'recorded_optimizer_updates':0,
            'evaluation_transitions':0,'initialization_resets':0,'initialization_lanes':0,
            'initialization_counted_as_physics':False,'dispatches':[],'completed':[],
            'phase_verdicts':{'A':'NOT_RUN','B':'NOT_RUN'},'comparisons':{},
            'boundaries':{},'production_authorized':False,'promotion_authorized':False,
            'weights_retained':False,'batch_retained':False,
            'post_terminal_physics':'wrapped reset pose, not pre-reset fall pose'}


def h4c_ledger(np,counts,env_info):
    # Preserve old h4r_ledger and production thresholds byte-for-byte.
    pending=h4r_pending(np,env_info)
    checks=h4_epoch_audit(counts,pending,'control')['checks']
    excluded=('step_partition','nominal_fraction')
    h4c_require(all(k in checks for k in excluded),'LEDGER_SCHEMA')
    checks={k:v for k,v in checks.items() if k not in excluded}
    checks['readiness_step_partition']=sum(counts[f'm463_h4_class_{c}_steps_per_rollout']
                                           for c in range(4))==512
    checks['legacy_event_reconcile']=(counts['m463_events_per_rollout']==
        counts['m463_h4_first_events_per_rollout']+counts['m463_h4_repeat_events_per_rollout'])
    checks['legacy_repeat_reconcile']=(counts['m463_repeat_per_rollout']==
                                      counts['m463_h4_repeat_events_per_rollout'])
    h4c_require(len(checks)==67,'LEDGER_LENGTH')
    return {'checks':checks,'pending':pending,'passed':all(checks.values()),
            'excluded_training_scale_gates':list(excluded)}


def h4c_snapshot(jax,np,tree,*,finite=True):
    paths,structure=jax.tree_util.tree_flatten_with_path(tree)
    values={}
    for p,v in paths:
        name=jax.tree_util.keystr(p)
        h4c_require(name not in values,'DUPLICATE_PATH')
        a=np.array(jax.device_get(v),copy=True)
        h4c_require(a.dtype.kind in 'biuf','LEAF_DTYPE '+name)
        if finite:
            h4c_require(np.isfinite(a).all(),'NONFINITE '+name)
        a.setflags(write=False)
        values[name]=a
    return values,str(structure)


def h4c_digest(snapshot):
    values,structure=snapshot
    h=hashlib.sha256(structure.encode())
    for name,v in sorted(values.items()):
        h.update(json.dumps([name,str(v.dtype),list(v.shape)]).encode()+b'\0')
        h.update(v.tobytes())
    return h.hexdigest()


def h4c_exact_path(name):
    # Floating-coded flags and command vectors require exact equality.
    return (name.endswith(('.done','.discount','.count',"['done']","['discount']",
              "['command']","['truncation']","['episode_done']")) or
            "['h4c_finite']" in name or
            (name.startswith("['metrics']") and name.endswith("_per_rollout']")
             and not name.endswith("_weighted_sum_per_rollout']")))


def h4c_compare(np,left,right,phase):
    """Record all differing paths, first bit/cap violation, and max residual."""
    a,sa=left
    b,sb=right
    h4c_require(sa==sb and set(a)==set(b),'COMPARE_SCHEMA')
    rows={}
    first_bits=None
    first_fail=None
    def event(name,idx,kind):
        # A trace [device,outer,time,lane,...]; batch [device,lane,time,...].
        # B metric [device,pass,minibatch,...] or [device,pass,...].
        stage='final'
        tick=lane=sgd_pass=minibatch=None
        if phase=='A':
            if name.startswith("['trace']") and len(idx)>=3:
                tick=idx[2]
                if "['pre']" in name:
                    stage='pre'
                elif "['post']" in name:
                    stage='post'
                else:
                    stage='key'
                lane=idx[3] if stage in ('pre','post') and len(idx)>=4 else None
            elif name.startswith("['batch']") and len(idx)>=3:
                lane,tick=idx[1:3]
                stage=('action' if ('.action' in name or "['policy_extras']" in name)
                       else 'pre' if '.observation' in name else 'post')
        elif name.startswith("['metrics']") and len(idx)>=2:
            sgd_pass=idx[1]
            if any(k in name for k in ("['h4c_perm_key']","['h4c_grad_key']",
                         "['h4c_sgd_key']","['h4c_minibatch_final_key']")):
                stage='sgd_key'
            else:
                stage='minibatch'
                minibatch=idx[2] if len(idx)>=3 else None
        return {'path':name,'index':list(idx),'tick':tick,'lane':lane,
                'sgd_pass':sgd_pass,'minibatch':minibatch,'stage':stage,'kind':kind}
    def rank(e):
        order={'key':0,'pre':1,'action':2,'post':3,'sgd_key':0,'minibatch':1,'final':9}
        if phase=='A':
            return (e['tick'] if e['tick'] is not None else 999,order[e['stage']],
                    e['lane'] if e['lane'] is not None else -1,e['path'],e['index'])
        return (e['sgd_pass'] if e['sgd_pass'] is not None else 999,order[e['stage']],
                e['minibatch'] if e['minibatch'] is not None else -1,e['path'],e['index'])
    for name,x in a.items():
        y=b[name]
        d,eps=h4r_distance(np,x,y,h4c_exact_path(name))
        limit=0. if eps==0 else H4C_SPEC['float_cap']
        # Compare each element's actual bytes, including signed zero.
        raw_x=np.ascontiguousarray(x).reshape(-1).view(np.uint8).reshape(x.size,x.dtype.itemsize)
        raw_y=np.ascontiguousarray(y).reshape(-1).view(np.uint8).reshape(y.size,y.dtype.itemsize)
        bits=np.any(raw_x!=raw_y,axis=1).reshape(x.shape)
        if not bits.any():
            continue
        u,v=x.astype(np.float64),y.astype(np.float64)
        residual=np.abs(u-v)/np.maximum(1.,np.maximum(np.abs(u),np.abs(v)))
        failures=(x!=y) if eps==0 else residual>limit
        def first(mask,kind):
            if not mask.any():
                return None
            ordered=phase=='A' and name.startswith("['batch']") and mask.ndim>=3
            search=np.swapaxes(mask,1,2) if ordered else mask
            idx=tuple(int(i) for i in np.unravel_index(int(np.flatnonzero(search)[0]),search.shape))
            if ordered:
                idx=(idx[0],idx[2],idx[1],*idx[3:])
            result=event(name,idx,kind)
            result.update(left=x[idx].item(),right=y[idx].item())
            return result
        bit=first(bits,'BITWISE')
        fail=first(failures,'ACCEPTANCE')
        if first_bits is None or rank(bit)<rank(first_bits):
            first_bits=bit
        if fail is not None and (first_fail is None or rank(fail)<rank(first_fail)):
            first_fail=fail
        rows[name]={'distance':d,'limit':limit,'passed':d<=limit,
                    'max_abs':float(np.max(np.abs(u-v),initial=0.)),
                    'bitwise_mismatch_count':int(bits.sum()),
                    'first_bitwise':bit,'first_acceptance':fail}
    return {'passed':all(r['passed'] for r in rows.values()),'bitwise_equal':not rows,
            'compared_leaves':len(a),'different_leaves':len(rows),'paths':rows,
            'first_observed_bitwise':first_bits,'first_observed_acceptance':first_fail}


def h4c_trace_batch_layout(np,x):
    x=np.asarray(x)
    h4c_require(x.shape[:4]==(1,1,32,16),'TRACE_LAYOUT')
    return np.swapaxes(x[:,0],1,2)


def h4c_expected_keys(jax,np,key):
    ks,kg,kn=jax.random.split(key,3)
    current,outer=jax.random.split(kg)
    actor_keys=[]; next_keys=[]
    for _ in range(32):
        actor,current=jax.random.split(current)
        actor_keys.append(np.asarray(actor))
        next_keys.append(np.asarray(current))
    expected={'h4c_perm_key':[],'h4c_grad_key':[],'h4c_sgd_key':[],
              'h4c_minibatch_final_key':[],'h4c_loss_key':[],
              'h4c_next_minibatch_key':[]}
    sk=ks
    for _ in range(5):
        sk,perm,grad=jax.random.split(sk,3)
        expected['h4c_perm_key'].append(np.asarray(perm))
        expected['h4c_grad_key'].append(np.asarray(grad))
        expected['h4c_sgd_key'].append(np.asarray(sk))
        mk=grad; loss_keys=[]; minibatch_keys=[]
        for _ in range(8):
            mk,loss=jax.random.split(mk)
            loss_keys.append(np.asarray(loss)); minibatch_keys.append(np.asarray(mk))
        expected['h4c_loss_key'].append(loss_keys)
        expected['h4c_next_minibatch_key'].append(minibatch_keys)
        expected['h4c_minibatch_final_key'].append(np.asarray(mk))
    return (np.asarray(ks)[None,:],np.asarray(kn)[None,:],
            np.asarray(outer)[None,:],np.asarray(current)[None,None,:],
            np.asarray(actor_keys)[None,None,:,:],np.asarray(next_keys)[None,None,:,:],
            {k:np.asarray(v)[None,...] for k,v in expected.items()},np.asarray(sk)[None,:])

def h4c_audit_callback(api,args,report,persist,gpu_check):
    jax,np=api['jax'],api['numpy']
    from flax import serialization
    import optax
    from m462_typed_orbax_stage import checkpoint_parts,compare_exact
    expected_parts=checkpoint_parts(serialization.msgpack_restore(args.checkpoint.read_bytes()))
    budget=H4CBudget(report)
    initial_hash=dr_hash=initial_count=None
    fixed_hash=None
    a_ref=b_ref=None
    a0_batch_hash=a0_sgd_hash=None
    expected_sgd=expected_new=expected_outer=expected_inner=None
    expected_actor_keys=expected_next_keys=None
    expected_sgd_metrics=None
    expected_sgd_final=None
    def snap(tree,finite=True):
        return h4c_snapshot(jax,np,tree,finite=finite)
    def digest_tree(tree,finite=True):
        return h4c_digest(snap(tree,finite))
    def model_digest(model,axes):
        # Model constants are provenance, not dynamic/learned tensor validity.
        # Hash all numeric leaves and treedef including static metadata exactly.
        return digest_tree((model,axes),False)
    def same(x,y,label):
        x,y=np.asarray(x),np.asarray(y)
        h4c_require(x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y),
                     'KEY_OR_VALUE '+label)
    def state_check(ts,updates):
        h4c_require(np.asarray(ts.env_steps).shape==(1,) and
                     int(ts.env_steps[0])==(512 if updates else 0),'LOGICAL_ENV_STEPS')
        adam=jax.tree_util.tree_leaves(ts.optimizer_state,
            is_leaf=lambda x:isinstance(x,optax.ScaleByAdamState))
        adam=[v for v in adam if isinstance(v,optax.ScaleByAdamState)]
        h4c_require(len(adam)==1 and np.asarray(adam[0].count).shape==(1,) and
                     int(adam[0].count[0])==updates,'ADAM_COUNT')
        if not updates:
            h4c_require(all(not np.any(np.asarray(x)) for x in
                jax.tree_util.tree_leaves((adam[0].mu,adam[0].nu))),'ADAM_FRESH')
        count=np.asarray(ts.normalizer_params.count)
        h4c_require(count.shape==(1,),'NORMALIZER_SHAPE')
        target=np.add(initial_count,np.asarray(512 if updates else 0,dtype=initial_count.dtype),
                      dtype=initial_count.dtype)
        same(count,target,'NORMALIZER_COUNT')
        return {'logical_env_steps':int(ts.env_steps[0]),'adam_count':updates,
                'normalizer_count':count.item()}
    def callback(label,value,aux):
        nonlocal initial_hash,dr_hash,initial_count,fixed_hash,a_ref,b_ref
        nonlocal a0_batch_hash,a0_sgd_hash
        nonlocal expected_sgd,expected_new,expected_outer,expected_inner
        nonlocal expected_actor_keys,expected_next_keys,expected_sgd_metrics,expected_sgd_final
        report['active_boundary']=label
        gpu_check()
        h4c_require(budget.index<len(budget.schedule) and budget.schedule[budget.index]==label,'SEQUENCE')
        if label=='initial':
            initial_hash=digest_tree(value)
            dr_hash=model_digest(*aux)
            ts=value[0]
            initial_count=np.array(ts.normalizer_params.count,copy=True)
            restored=tuple(serialization.to_state_dict(jax.tree_util.tree_map(lambda x:x[0],v))
                           for v in (ts.normalizer_params,ts.params.policy,ts.params.value))
            h4c_require(compare_exact(expected_parts,restored,jax,np,'H4C_RESTORE')==25,'RESTORE')
            report['initialization_resets']=1
            report['initialization_lanes']=16
            report['initial']={'carry_sha256':initial_hash,'dr_sha256':dr_hash,
                               'restored_weights_exact':True,**state_check(ts,0)}
            (expected_sgd,expected_new,expected_outer,expected_inner,expected_actor_keys,
             expected_next_keys,expected_sgd_metrics,expected_sgd_final)=h4c_expected_keys(jax,np,value[2][0])
        elif label.startswith('A') and label.endswith('_before'):
            h4c_require(digest_tree(value)==initial_hash and model_digest(*aux)==dr_hash,'A_INPUT')
            state_check(value[0],0)
        elif label in ('A0','A1'):
            env,batch,ks,kn,trace,inner,outer,metrics=value
            h4c_require(digest_tree(aux[0])==initial_hash and model_digest(aux[1],aux[2])==dr_hash,
                         'A_INPUT_MUTATED')
            same(ks,expected_sgd,'SGD');same(kn,expected_new,'NEW')
            same(inner,expected_inner,'INNER');same(outer,expected_outer,'OUTER')
            same(trace['actor_key'],expected_actor_keys,'ACTOR')
            same(trace['next_key'],expected_next_keys,'NEXT_ACTOR')
            b=snap(batch)
            h4c_require(all(x.shape[:3]==(1,16,32) for x in b[0].values()),'BATCH_LAYOUT')
            for side in ('pre','post'):
                h4c_require(all(x.shape[:4]==(1,1,32,16)
                                 for x in snap(trace[side])[0].values()),'TRACE_LAYOUT')
            h4c_require(np.isin(np.asarray(batch.discount),(0,1)).all(),'DISCOUNT')
            h4c_require(np.asarray(batch.action).shape==(1,16,32,29),'ACTION_LAYOUT')
            # Sidecar observations and flags must describe the very same
            # actor_step transitions, including AutoReset semantics.
            def batch_layout(x):
                return h4c_trace_batch_layout(np,x)
            for side,obs in (('pre',batch.observation),('post',batch.next_observation)):
                left=jax.tree_util.tree_leaves(trace[side]['obs'])
                right=jax.tree_util.tree_leaves(obs)
                h4c_require(jax.tree_util.tree_structure(trace[side]['obs'])==
                             jax.tree_util.tree_structure(obs),'OBS_SCHEMA')
                for x,y in zip(left,right):
                    same(batch_layout(x),y,'TRACE_OBS_'+side)
            same(batch_layout(trace['post']['done']),1-np.asarray(batch.discount),'TRACE_DONE')
            ex=batch.extras['state_extras']
            for k in ('truncation','h4_source_terminal','m463_event','m463_repeat',
                      'm463_direction','m463_support_side','h4_class','h4_first_pending',
                      'h4_repeat_pending','h4_tail_kind','h4_tail_age'):
                same(batch_layout(trace['post']['info'][k]),ex[k],'TRACE_EXTRA '+k)
            h4c_require(np.isin(np.asarray(ex['truncation']),(0,1)).all() and
                         np.isin(np.asarray(ex['h4_source_terminal']),(0,1)).all(),'TERMINAL_SCHEMA')
            for side in ('pre','post'):
                h4c_require(np.isin(np.asarray(trace[side]['done']),(0,1)).all(),'DONE_SCHEMA')
            for k in ('m463_event','m463_repeat','h4_first_pending','h4_repeat_pending','h4_bad',
                      'h4_boundary','h4_source_terminal','h4_tick_survived','m463_no_push'):
                h4c_require(np.isin(np.asarray(ex[k]),(0,1)).all(),'FLAG_SCHEMA '+k)
            for k,allowed in (('h4_class',(0,1)),('h4_tail_kind',(0,1,2)),
                              ('m463_direction',tuple(range(6))),('m463_support_side',tuple(range(4)))):
                h4c_require(np.isin(np.asarray(ex[k]),allowed).all(),'CATEGORY_SCHEMA '+k)
            m={k:np.asarray(v) for k,v in metrics.items()}
            h4c_require(all(v.shape==(1,) and np.isfinite(v).all() for v in m.values()),'METRICS')
            counts=h4r_counts({k:v[0] for k,v in m.items()},1)
            ledger=h4c_ledger(np,counts,env.info)
            report['boundaries'][label]={'ledger':ledger,'counts':counts}
            h4c_require(ledger['passed'],'LEDGER')
            observed=snap({'env':env,'batch':batch,'trace':trace,'metrics':metrics,
                           'key_sgd':ks,'new_key':kn,'inner_key':inner,'outer_key':outer})
            row={'batch_sha256':h4c_digest(b),'output_sha256':h4c_digest(observed),
                 'input_sha256':initial_hash,'ledger':ledger,'counts':counts,
                 'metrics':{k:v.item() for k,v in m.items()},'physical_transitions':512,
                 'optimizer_updates':0}
            report['boundaries'][label]=row
            if label=='A0':
                a_ref=observed
                a0_batch_hash=h4c_digest(b);a0_sgd_hash=digest_tree(ks)
            else:
                result=h4c_compare(np,a_ref,observed,'A')
                report['comparisons']['A']=result
                report['phase_verdicts']['A']='PASS' if result['passed'] else 'DIVERGENT'
                a_ref=None
        elif label.startswith('B') and label.endswith('_before'):
            ts,batch,ks=value
            state_check(ts,0)
            h4c_require(digest_tree(batch)==a0_batch_hash and digest_tree(ks)==a0_sgd_hash,
                         'FIXED_BATCH')
            # A source carry digest includes the original environment/key; B
            # must also match the exact initial training state recorded below.
            h4c_require(digest_tree(ts)==report['initial']['training_state_sha256'],'B_INITIAL_STATE')
            d=digest_tree(value)
            if label=='B0_before':
                fixed_hash=d
            h4c_require(d==fixed_hash,'B_INPUT')
        elif label in ('B0','B1'):
            ts,key,metrics=value
            h4c_require(digest_tree(aux)==fixed_hash,'B_INPUT_MUTATED')
            checks=state_check(ts,40)
            same(key,expected_sgd_final,'SGD_FINAL')
            h4c_require(set(expected_sgd_metrics)<=set(metrics),'SGD_KEY_SCHEMA')
            for k,expected in expected_sgd_metrics.items():
                same(metrics[k],expected,k)
            finite=np.asarray(metrics['h4c_finite'])
            h4c_require(finite.shape==(1,5,8) and finite.dtype.kind=='b' and finite.all(),'MINIBATCH_FINITE')
            for k in ('total_loss','policy_loss','v_loss','entropy_loss'):
                h4c_require(np.asarray(metrics[k]).shape==(1,5,8),'LOSS_LAYOUT')
            observed=snap({'training_state':ts,'key':key,'metrics':metrics})
            report['boundaries'][label]={'input_sha256':fixed_hash,'output_sha256':h4c_digest(observed),
                'physical_transitions':0,'optimizer_updates':40,**checks,
                'minibatch_metrics':{k:np.asarray(v).tolist() for k,v in metrics.items()}}
            if label=='B0':
                b_ref=observed
            else:
                result=h4c_compare(np,b_ref,observed,'B')
                report['comparisons']['B']=result
                report['phase_verdicts']['B']='PASS' if result['passed'] else 'DIVERGENT'
                b_ref=None
        else:
            raise RuntimeError('M470_H4C_CALLBACK_LABEL')
        if label=='initial':
            report['initial']['training_state_sha256']=digest_tree(value[0])
        budget.advance(label)
        persist()
    return callback,budget


def h4c_dependency_pins(api):
    import inspect
    import importlib
    from m463_instrument_brax import h4c_git_blob
    ppo_path=Path(inspect.getfile(api['ppo'].train)).resolve()
    # .../brax/training/agents/ppo/train.py -> .../brax
    root=ppo_path.parents[3]
    paths={}
    for name,blob in H4C_BRAX_BLOBS.items():
        mod=importlib.import_module(name)
        actual=Path(inspect.getfile(mod)).resolve()
        intended=root.joinpath(*name.split('.')[1:]).with_suffix('.py')
        h4c_require(actual==intended and h4c_git_blob(actual.read_bytes())==blob,
                     'DEPENDENCY_PIN '+name)
        paths[name]={'path':str(actual),'blob':blob}
    return paths


def h4c_failure_pin(path):
    h4c_require(path is not None and path.is_file() and not path.is_symlink(),'FAILURE_REPORT_PATH')
    h4c_require(digest(path)==H4C_FAILURE_SHA,'FAILURE_REPORT_SHA')
    r=json.loads(path.read_bytes())
    h4c_require(r['status']=='M470_H4R_STOPPED' and
                r['attempted_transitions']==r['recorded_transitions']==20480 and
                r['attempted_optimizer_updates']==r['recorded_optimizer_updates']==1600 and
                len(r['dispatches'])==len(r['boundaries'])==2 and
                r['error'].endswith('M470_H4R_REFERENCE_DISCRETE_METRICS') and
                r['comparisons']=={} and r['production_authorized'] is False,'PRIOR_FAILURE_CONTRACT')
    return H4C_FAILURE_SHA


def h4c_run(args):
    validate_h4c_cli(args)
    h4c_require(args.mode=='causal' and H4C_EXECUTION_AUTHORIZED is True and
                H4R_EXECUTION_AUTHORIZED is False and H4_EXECUTION_AUTHORIZED is False,'EXECUTION_HOLD')
    h4c_require(isinstance(H4C_AUTHORIZED_RUN,str),'AUTHORIZATION_OUTPUT_HOLD')
    h4c_require(args.output_dir.resolve()==(Path(__file__).resolve().parent/H4C_AUTHORIZED_RUN).resolve()
                and not args.output_dir.is_symlink(),'ONE_RUN_PATH')
    import shutil
    import traceback
    h4c_require(not args.output_dir.exists() and args.output_dir.parent.is_dir(),'OUTPUT')
    h4c_require(shutil.disk_usage(args.output_dir.parent).free>=2*1024**3,'DISK')
    args.output_dir.mkdir(exist_ok=False)
    report=h4c_report()
    report['runner_sha256']=digest(Path(__file__))
    def persist():
        path=args.output_dir/'report.json'
        tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report,allow_nan=False,indent=2)+'\n',encoding='utf8')
        tmp.replace(path)
    persist()
    monitor=None
    try:
        failure_pin=h4c_failure_pin(args.h4_causal_failure_report)
        report['prior_failure_sha256']=failure_pin
        pins=h4r_bootstrap_pins(args)
        report['resolved_modules']=pins
        from m462_ppo_warmstart_run import read_restore_gate,check_source_and_config,GPUMonitor,sample_gpu
        from m463_instrument_brax import build_h4_causal_ppo
        gate,orbax_dir=read_restore_gate(args.restore_report,args.checkpoint)
        h4c_require(gate['orbax_dir_sha256']==H4_ORBAX_SHA,'ORBAX_PIN')
        report['frozen_inputs']={'checkpoint_sha256':SEED1_SHA,'restore_report_sha256':H3_CAUSAL_RESTORE_SHA,
                                'orbax_sha256':H4_ORBAX_SHA,'reward_reference_sha256':H4R_REWARD_SHA}
        monitor=GPUMonitor()
        def gpu_check():
            h4c_require(shutil.disk_usage(args.output_dir).free>=2*1024**3,'DISK')
            g=sample_gpu()
            snap=monitor.snapshot()
            h4c_require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288 and
                not snap['sampler_errors'] and (snap['peak_used_mib'] is None or
                snap['peak_used_mib']<.95*12288),'GPU')
            report['gpu']=snap
        with monitor:
            gpu_check()
            # Sole bootstrap invocation; it applies NVCC before importing JAX.
            api=check_source_and_config('A')
            jax=api['jax']
            h4c_require(jax.__version__=='0.4.35' and jax.default_backend()=='gpu' and
                         jax.process_count()==jax.local_device_count()==1,'JAX_RUNTIME')
            from m463_coverage_env import CoverageJoystick
            for name,path in pins.items():
                mod=sys.modules.get(name)
                h4c_require(mod is not None and Path(mod.__file__).resolve()==Path(path),'IMPORTED_ORIGIN '+name)
            h4c_require(h4r_bootstrap_pins(args)==pins,'BOOTSTRAP_IMMUTABILITY')
            report['brax_dependencies']=h4c_dependency_pins(api)
            scaled=copy.deepcopy(api['scaled'])
            h4c_require(m470_fingerprint(scaled.to_dict())==H3_FP,'PRODUCTION_CONFIG')
            scaled.num_envs=16
            scaled.batch_size=2
            scaled.num_timesteps=512
            scaled.num_evals=1
            cfg=scaled.to_dict()
            report['scaled_config']=copy.deepcopy(cfg)
            source_cfg=api['source_cfg']
            env_cfg=copy.deepcopy(source_cfg)
            reward=json.loads(args.h4_readiness_reward_report.read_bytes())['train_reward_config']
            h4c_require(source_cfg.to_dict()['reward_config']==reward and
                        bool(source_cfg.push_config.enable),'ORIGINAL_REWARD')
            env_cfg.push_config.enable=False
            h4c_require(env_cfg.to_dict()['reward_config']==reward,'REWARD_UNCHANGED')
            env=CoverageJoystick(config=env_cfg,stage='easy_repeat_support',h4_arm='control')
            h4c_require(env.action_size==29 and float(env.dt)==.02,'ENV_CONFIG')
            mod,meta=build_h4_causal_ppo(api['ppo'],args.output_dir/'h4_causal_generated.py')
            report['generated']=meta
            factory=functools.partial(api['networks'].make_ppo_networks,**cfg.pop('network_factory'))
            audit,budget=h4c_audit_callback(api,args,report,persist,gpu_check)
            report['bootstrap_pass']=True
            persist()
            print('M470_H4C_BOOTSTRAP_PASS',flush=True)
            result=mod.causal_train(environment=env,wrap_env_fn=api['wrapper'].wrap_for_brax_training,
                 network_factory=factory,seed=1,randomization_fn=api['randomizer'],
                 restore_checkpoint_path=str(orbax_dir.resolve()),h4c_audit_fn=audit,**cfg)
            h4c_require(result is None,'NO_PARAMETER_RETURN')
            budget.complete()
            h4c_require(all(report['phase_verdicts'][k] in ('PASS','DIVERGENT') for k in ('A','B')),
                         'VERDICTS')
            h4c_require(h4c_failure_pin(args.h4_causal_failure_report)==failure_pin and
                        h4c_dependency_pins(api)==report['brax_dependencies'],'EVIDENCE_IMMUTABILITY')
            h4c_require(h4r_bootstrap_pins(args)==pins,'INPUT_IMMUTABILITY')
            read_restore_gate(args.restore_report,args.checkpoint)
            gpu_check()
        report['gpu']=monitor.snapshot()
        h4c_require(report['gpu']['samples']>0 and not report['gpu']['sampler_errors'] and
                     0<report['gpu']['peak_used_mib']<.95*12288,'FINAL_GPU')
        report['status']='M470_H4C_DIAGNOSTIC_COMPLETE'
        persist()
    except BaseException as exc:
        report['status']='M470_H4C_STOPPED'
        active=report.get('active_boundary','')
        if active[:1] in ('A','B') and report['phase_verdicts'][active[0]]=='NOT_RUN':
            report['phase_verdicts'][active[0]]='INVALID'
        report['error']=type(exc).__name__+': '+str(exc)
        report['traceback']=traceback.format_exc()
        if monitor is not None:
            report['gpu']=monitor.snapshot()
        persist()
        raise




def h4c_static_regression(brax_train_path,acting_path,failure_report):
    """Focused exact-checkout static validation; no runtime imports or steps."""
    import ast
    import subprocess
    import types
    import numpy as np
    from types import SimpleNamespace
    import m463_instrument_brax as gen
    root=Path(__file__).resolve().parent
    names=('m463_push_ppo_train.py','m463_instrument_brax.py')
    for name in names:
        data=(root/name).read_bytes()
        compile(data,name,'exec')
        ast.parse(data)
    h4c_require(not any((H4_EXECUTION_AUTHORIZED,H4R_EXECUTION_AUTHORIZED,
                        H4C_EXECUTION_AUTHORIZED)) and H4C_AUTHORIZED_RUN is None,'STATIC_HOLDS')
    raw=Path(brax_train_path).read_bytes()
    acting_raw=Path(acting_path).read_bytes()
    h4c_require(gen.sha256(raw)==gen.PINNED_SHA and gen.h4c_git_blob(acting_raw)==gen.H4C_ACTING_BLOB,
                 'STATIC_SOURCE_PIN')
    previous='87364f9d23ef09b305a0d39c8c7be25b5c39680a'
    old_raw=subprocess.check_output(['git','show',previous+':m463_instrument_brax.py'],cwd=root)
    h4c_require((root/names[1]).read_bytes().startswith(old_raw),'GENERATOR_PREFIX')
    old=types.ModuleType('_h4c_previous_generator')
    exec(compile(old_raw,'<previous generator>','exec'),old.__dict__)
    held,sha=gen.make_h4_readiness_source(raw)
    h4c_require((held,sha)==old.make_h4_readiness_source(raw) and
                 gen.sha256(held.encode())==gen.H4C_HELD_SHA,'PREVIOUS_READINESS')
    h4c_require(gen.make_patched_source(raw,enable_resume=True,enable_h4=True)==
                 old.make_patched_source(raw,enable_resume=True,enable_h4=True),'PRODUCTION')
    causal,sha=gen.make_h4_causal_source(raw,acting_raw)
    compile(causal,'<causal generated>','exec')
    tree=ast.parse(causal)
    entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='causal_train')
    h4c_require(isinstance(entry.body[1],ast.Raise) and
                'M470_H4C_EXECUTION_HOLD' in ast.unparse(entry.body[1]),'GENERATED_HOLD')
    h4c_require(not any(isinstance(n,ast.FunctionDef) and n.name in ('train','readiness_train')
                        for n in tree.body),'ISOLATED_ENTRY')
    nested={n.name:n for n in entry.body if isinstance(n,ast.FunctionDef)}
    h4c_require('h4c_collect' in nested and 'h4c_optimize' in nested and
                not {'training_step','h4_chunk','training_epoch_with_timing'}&set(nested),
                'NO_LEGACY_DISPATCH')
    h4c_require('acting.Evaluator(' not in causal and 'policy_params_fn(current_step' not in causal
                and 'm470_state_checkpoint_fn(current_step' not in causal
                and 'for it in range(num_evals_after_init)' not in causal,'NO_EVAL_SAVE')
    collect,opt,reducer,construct=gen.h4c_regions(held)
    collection=causal[causal.index('  def h4c_collect('):causal.index('  def h4c_optimize(')]
    normalized=collection.replace('next_state, data, trace, inner_key = h4c_generate_unroll(',
                                  'next_state, data = acting.generate_unroll(',1)
    normalized=normalized.replace('(next_state, next_key), (data, trace, inner_key)',
                                  '(next_state, next_key), data',1)
    normalized=normalized.replace('(state, outer_key), (data, trace, inner_key) = jax.lax.scan(',
                                  '(state, _), data = jax.lax.scan(',1)
    h4c_require(collect in normalized and reducer in collection,'COLLECTION_REDUCER_DRIFT')
    optimizer=ast.get_source_segment(causal,nested['h4c_optimize'])
    h4c_require(opt in optimizer.replace('(optimizer_state, params, sgd_final_key), metrics =',
                        '(optimizer_state, params, _), metrics =',1) and construct in optimizer,
                'OPTIMIZER_DRIFT')
    h4c_require(not any(isinstance(n,ast.Name) and n.id in ('env','state','acting','h4c_collect')
                        for n in ast.walk(nested['h4c_optimize'])),'OPTIMIZER_ENV_DEPENDENCY')
    lo=causal.index('  def minibatch_step(')
    hi=causal.index('  def h4c_collect(',lo)
    norm=causal[lo:hi].replace(gen.H4C_MINIBATCH_TRACE,'',1)
    norm=norm.replace('(optimizer_state, params, minibatch_final_key), metrics =',
                      '(optimizer_state, params, _), metrics =',1)
    trace=("    metrics = {**metrics, 'h4c_perm_key': key_perm,\n"
           "               'h4c_grad_key': key_grad, 'h4c_sgd_key': key,\n"
           "               'h4c_minibatch_final_key': minibatch_final_key}\n")
    h4c_require(norm.count(trace)==1,'SGD_TRACE')
    norm=norm.replace(trace,'',1)
    h4c_require(norm==held[held.index('  def minibatch_step('):held.index('  def training_step(')],
                'SGD_MINIBATCH_DRIFT')
    trace_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and
                  n.name=='h4c_generate_unroll')
    trace_src=ast.get_source_segment(causal,trace_fn)
    old_act=acting_raw.decode()
    old_fn=next(n for n in ast.parse(old_act).body if isinstance(n,ast.FunctionDef)
                and n.name=='generate_unroll')
    old_unroll=ast.get_source_segment(old_act,old_fn)
    body=trace_src[trace_src.index('  @jax.jit\n'):]
    body=body.replace('    pre = h4c_trace_state(state)\n','',1)
    body=body.replace('nstate, transition = acting.actor_step(','nstate, transition = actor_step(',1)
    body=body.replace("    trace = {'pre': pre, 'post': h4c_trace_state(nstate),\n"
                      "             'actor_key': current_key, 'next_key': next_key}\n",'',1)
    body=body.replace('return (nstate, next_key), (transition, trace)',
                      'return (nstate, next_key), transition',1)
    body=body.replace('(final_state, final_key), (data, trace) =',
                      '(final_state, _), data =',1)
    body=body.replace('return final_state, data, trace, final_key','return final_state, data',1)
    h4c_require(body==old_unroll[old_unroll.index('  @jax.jit\n'):],
                'ACTING_BODY_DRIFT')
    h4c_require(trace_src.count('acting.actor_step(')==1 and
                'env.step(' not in trace_src,'EXTRA_PHYSICS')
    print('M470_H4C_GENERATED_SOURCE_SCOPE_PASS',gen.sha256(causal.encode()),flush=True)
    def reject(fn,marker=None):
        try:
            fn()
        except RuntimeError as exc:
            h4c_require(marker is None or marker in str(exc),'STATIC_WRONG_EXCEPTION')
        else:
            raise RuntimeError('M470_H4C_STATIC_ACCEPTED_INVALID')
    base=[sys.executable,'-B',str(root/names[0]),'--checkpoint','unused',
          '--restore-report','unused','--output-dir','unused','--stage','easy_repeat_support']
    flags=['--h4-causal','--h4-causal-failure-report','unused',
           '--h4-readiness-reward-report','unused']
    cases=[(['--mode','plan']+flags,0),
           (['--mode','causal']+flags,2),
           (['--mode','causal'],2),
           (['--mode','plan','--h4-causal'],2),
           (['--mode','plan','--h4-causal-failure-report','unused'],2)]
    for mode in ('readiness','smoke','probe2','smoke3','pilot','longtrain'):
        cases.append((['--mode',mode]+flags,2))
    for flag in (['--h4-arm','control'],['--h4-control-report','unused'],
                 ['--h4-readiness'],['--h1-m1'],['--h1-m2'],['--h3-m1'],
                 ['--h3-diagnostic-only'],['--h3-causal-report','unused'],
                 ['--h3-old10m-report','unused'],['--smoke-report','unused'],
                 ['--resume-state','unused'],['--independent-budget','10M'],
                 ['--rollouts','1'],['--num-evals','1']):
        cases.append((['--mode','plan']+flags+flag,2))
    for extra,code in cases:
        p=subprocess.run(base+extra,capture_output=True,text=True,cwd=root)
        h4c_require(p.returncode==code and (code!=0 or 'M470_H4C_PLAN_NO_GPU' in p.stdout),
                    'STATIC_CLI '+repr(extra)+' '+p.stderr)
    args=SimpleNamespace(mode='causal',stage='easy_repeat_support',h4_causal=True,
                h4_causal_failure_report=Path('unused'),h4_readiness_reward_report=Path('unused'))
    for fn in (run,h4c_run):
        reject(lambda:fn(args),'M470_H4C_EXECUTION_HOLD')
    print('M470_H4C_CLI_HOLD_PASS',len(cases),'plus two direct guards',flush=True)
    # Actual generated host dispatch loop with pure Python mocks.
    suffix=gen.H4C_EXIT[gen.H4C_EXIT.index("  h4c_audit_fn('initial'"):]
    ns={}
    exec(compile('def dispatch(initial,model,model_axes,dr,collect,optimize,h4c_audit_fn):\n'+suffix,
                 '<causal host dispatch fixture>','exec'),ns)
    initial=(object(),object(),object())
    model,axes=object(),object()
    dr=SimpleNamespace(_mjx_model_v=model,_in_axes=axes)
    for fail_label in (None,'A0','A1','B0'):
        report=h4c_report();budget=H4CBudget(report)
        calls=[];batches=[];keys=[]
        def collect(value):
            h4c_require(value is initial,'FIXTURE_A_INPUT')
            calls.append('A');batches.append(object());keys.append(object())
            return (object(),batches[-1],keys[-1])
        def optimize(ts,batch,key):
            h4c_require(ts is initial[0] and batch is batches[0] and key is keys[0],
                         'FIXTURE_FIXED_BATCH')
            calls.append('B')
            return object()
        def audit(label,value,aux):
            if label==fail_label:
                raise RuntimeError('FIXTURE_STOP')
            budget.advance(label)
        if fail_label is None:
            h4c_require(ns['dispatch'](initial,model,axes,dr,collect,optimize,audit) is None,
                        'FIXTURE_RETURN')
            budget.complete()
            h4c_require(calls==['A','A','B','B'],'FIXTURE_CALLS')
            reject(lambda:budget.advance('B1_before'),'SEQUENCE')
        else:
            reject(lambda:ns['dispatch'](initial,model,axes,dr,collect,optimize,audit),'FIXTURE_STOP')
            expected={'A0':['A'],'A1':['A','A'],'B0':['A','A','B']}[fail_label]
            h4c_require(calls==expected,'FIXTURE_STOP_SEQUENCE')
    r=h4c_report();budget=H4CBudget(r);budget.advance('initial')
    r['attempted_physical_transitions']=1024
    reject(lambda:budget.advance('A0_before'),'BUDGET')
    r=h4c_report();budget=H4CBudget(r)
    for label in budget.schedule[:5]:
        budget.advance(label)
    r['attempted_optimizer_updates']=80
    reject(lambda:budget.advance('B0_before'),'BUDGET')
    print('M470_H4C_DISPATCH_BUDGET_FIXTURE_PASS',flush=True)
    name="['batch'].action"
    x=np.zeros((1,16,32,29),np.float32);y=x.copy()
    y[0,0,30,0]=1.;y[0,1,2,1]=2.
    compare=lambda x,y,phase='A':h4c_compare(np,(x,'fixture'),(y,'fixture'),phase)
    c=compare({name:x},{name:y})
    h4c_require(not c['passed'] and c['first_observed_acceptance']['tick']==2 and
                c['first_observed_acceptance']['lane']==1,'CHRONOLOGY')
    y=x.copy();y[0,0,30,0]=1e-6
    c=compare({name:x},{name:y})
    h4c_require(c['passed'] and not c['bitwise_equal'] and
                c['first_observed_acceptance'] is None,'SMALL_FLOAT')
    z=np.array(0.,np.float32);w=np.array(-0.,np.float32)
    h4c_require(not compare({'scalar':z},{'scalar':w})['bitwise_equal'],'SIGNED_ZERO')
    flag="['trace']['post']['done']"
    x=np.zeros((1,1,32,16),np.float32);y=x.copy();y[0,0,2,5]=1e-8
    h4c_require(not compare({flag:x},{flag:y})['passed'],'DISCRETE_FLOAT')
    name="['metrics']['policy_loss']"
    x=np.zeros((1,5,8),np.float32);y=x.copy();y[0,1,2]=.01
    c=compare({name:x},{name:y},'B')
    h4c_require(c['first_observed_acceptance']['sgd_pass']==1 and
                c['first_observed_acceptance']['minibatch']==2,'MINIBATCH')
    for bad in (np.array([np.nan],np.float32),np.array([np.inf],np.float32),
                np.array([1.],np.float64),np.array([1.,2.],np.float32)):
        reject(lambda:compare({'x':np.array([1.],np.float32)},{'x':bad}))
    # Synthetic layout and deterministic mock key splitter validate host
    # accounting, not JAX's PRNG implementation.
    x=np.arange(32*16,dtype=np.int32).reshape(1,1,32,16)
    y=h4c_trace_batch_layout(np,x)
    h4c_require(y.shape==(1,16,32) and y[0,5,7]==7*16+5,'LAYOUT_FIXTURE')
    reject(lambda:h4c_trace_batch_layout(np,x.reshape(1,1,16,32)))
    # Bound mock integers and make every split output distinguishable.
    class BoundedRandom:
        @staticmethod
        def split(key,n=2):
            a,b=map(int,key)
            return np.array([[(a*7+i+1)%2**32,(b*3+i+1)%2**32]
                             for i in range(n)],dtype=np.uint32)
    rng=SimpleNamespace(random=BoundedRandom())
    key=np.array([11,23],np.uint32)
    ks,kn,outer,inner,actors,nexts,sgd,final=h4c_expected_keys(rng,np,key)
    expected=rng.random.split(key,3)
    h4c_require(np.array_equal(ks[0],expected[0]) and np.array_equal(kn[0],expected[2]),
                 'MOCK_ROOT_KEYS')
    h4c_require(actors.shape==nexts.shape==(1,1,32,2) and inner.shape==(1,1,2) and
                 sgd['h4c_loss_key'].shape==(1,5,8,2) and final.shape==(1,2),'MOCK_KEY_LAYOUT')
    h4c_require(np.array_equal(inner[0,0],nexts[0,0,-1]) and
                 np.array_equal(final[0],sgd['h4c_sgd_key'][0,-1]),'MOCK_FINAL_KEYS')
    print('M470_H4C_LAYOUT_KEY_FIXTURE_PASS',flush=True)
    f=Path(failure_report)
    h4c_failure_pin(f)
    prior=json.loads(f.read_bytes())
    counts={k:0 for k in prior['boundaries'][0]['counts']}
    counts['m463_h4_class_0_steps_per_rollout']=512
    info={k:np.zeros((1,16),np.int32) for k in ('h4_first_pending','h4_repeat_pending',
        'h4_tail_kind','h4_class','h4_repeat_origin_motion','m463_direction')}
    ledger=h4c_ledger(np,counts,info)
    h4c_require(ledger['passed'] and len(ledger['checks'])==67,'LEDGER_FIXTURE')
    bad=dict(counts);bad['m463_h4_first_events_per_rollout']=1
    h4c_require(not h4c_ledger(np,bad,info)['passed'],'LEDGER_BAD_FIXTURE')
    h4c_require('jax' not in sys.modules and 'brax' not in sys.modules and
                'mujoco' not in sys.modules,'STATIC_RUNTIME_IMPORTED')
    print('M470_H4C_COMPARE_LEDGER_FIXTURE_PASS 67',flush=True)
    print('M470_H4C_STATIC_PASS NO_SIMULATION NO_PPO',flush=True)



# H4-M1-2R: a separate exploratory question, not a waiver of H4R FAIL.
H4X_ID = 'H4-M1-2R-V1'
H4X_CAUSAL_SHA = '8ff1d568e28a20842f05ef52a1107f5911980d16e128bf8b1881484091c805c5'
H4X_PATHS = {arm:'m463_runs/m470_h4_m1_2r_'+arm+'_v1' for arm in ('control','candidate')}
H4X_MIB = 1024**2
H4X_SPEC = {'id':H4X_ID,'training':10485760,'optimizer':1600,'evaluation':6000,
            'train_reset_lanes':24576,'eval_reset_lanes':6,'chunks':8,
            'chunk_training':1310720,'chunk_optimizer':200,'eval_calls':3,
            'train_reset_calls':3,'total_physical':10491760,
            'output_cap':512*H4X_MIB,'policy_cap':64*H4X_MIB,
            'fullstate_cap':128*H4X_MIB,'other_cap':128*H4X_MIB,
            'disk_reserve':2*1024**3,'external_log_reserve':16*H4X_MIB}
H4X_COUNTERS = {'training':0,'optimizer':0,'evaluation':0,
                'train_reset_lanes':0,'eval_reset_lanes':0}


def h4x_require(ok,label):
    if not bool(ok):
        raise RuntimeError('M470_H4X_'+label)


def h4x_source_digest(root):
    paths=(*H4_FILES,'m463_gate3_cp_eval.py','paper_config_preflight.json')
    payload={name:digest(root/name) for name in paths}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def h4x_parse_marker(rules):
    import re
    h4x_require(not any(line.startswith(('H4CC_ACTIVE_GO:','H4CC_EVAL_ACTIVE_GO:','H4CC_TECH_ACTIVE_GO:'))
                        for line in rules.splitlines()),'CCRC_AUTHORIZATION_CONFLICT')
    lines=[line for line in rules.splitlines() if line.startswith('H4X_ACTIVE_GO:')]
    h4x_require(len(lines)==1,'EXECUTION_HOLD' if not lines else 'MULTIPLE_AUTHORIZATIONS')
    match=re.fullmatch(r'H4X_ACTIVE_GO: (H4-M1-2R-V1) (control|candidate) '
                       r'(m463_runs/[a-z0-9_]+) ([0-9a-f]{64}) ([0-9a-f]{64}) ([0-9a-f]{64})',lines[0])
    h4x_require(match is not None,'AUTHORIZATION_SCHEMA')
    contract,arm,path,source,generated,evidence=match.groups()
    h4x_require(path==H4X_PATHS[arm],'AUTHORIZATION_PATH')
    return dict(id=contract,arm=arm,output=path,source_sha256=source,
                generated_sha256=generated,evidence_sha256=evidence)


def h4x_authorization():
    h4x_require(not any((H4_EXECUTION_AUTHORIZED,H4R_EXECUTION_AUTHORIZED,H4C_EXECUTION_AUTHORIZED)),
                'LEGACY_HOLD_REQUIRED')
    root=Path(__file__).resolve().parent
    auth=h4x_parse_marker((root/'G1_RESEARCH_RULES.md').read_text('utf8'))
    h4x_require(h4x_source_digest(root)==auth['source_sha256'],'SOURCE_CONTRACT')
    if auth['arm']=='control':
        h4x_require(auth['evidence_sha256']==H4X_CAUSAL_SHA,'CAUSAL_AUTHORIZATION')
    else:
        h4x_require(auth['evidence_sha256']==h4_approved_control_sha(),'CONTROL_AUTHORIZATION')
    return auth


def h4x_file_pin(path,sha):
    h4x_require(path is not None and path.is_file() and not path.is_symlink()
                and digest(path)==sha,'INPUT_PIN '+str(path))


def h4x_validate_cli(a):
    enabled=getattr(a,'h4_exploratory',False)
    evidence=(getattr(a,'h4_exploratory_causal_report',None),
              getattr(a,'h4_exploratory_reward_report',None))
    if not enabled:
        h4x_require(all(v is None for v in evidence),'FLAG_REQUIRED')
        return None
    h4x_require(a.mode in ('plan','longtrain') and a.stage=='easy_repeat_support'
                and getattr(a,'h4_arm',None) in H4X_PATHS,'MODE_ARM')
    h4x_require(not any(getattr(a,k,False) for k in
                        ('h1_m1','h1_m2','h3_m1','h3_diagnostic_only','h4_readiness','h4_causal'))
                and all(getattr(a,k,None) is None for k in
                    ('h3_causal_report','h3_old10m_report','h4_readiness_reward_report',
                     'h4_causal_failure_report','resume_state','smoke_report','independent_budget',
                     'rollouts','num_evals')),'CONFLICT')
    if a.mode=='plan':
        h4x_require(getattr(a,'h4_control_report',None) is None,'PLAN_CONTROL')
        return None
    auth=h4x_authorization()
    root=Path(__file__).resolve().parent
    h4x_require(a.h4_arm==auth['arm'] and a.output_dir.resolve()==root/auth['output']
                and not a.output_dir.is_symlink() and not a.output_dir.parent.is_symlink(),
                'ARM_OR_OUTPUT')
    h4x_require((a.h4_arm=='candidate')==(getattr(a,'h4_control_report',None) is not None),
                'CONTROL_REQUIRED')
    h4x_file_pin(evidence[0],H4X_CAUSAL_SHA)
    prior=json.loads(evidence[0].read_bytes())
    h4x_require(prior['status']=='M470_H4C_DIAGNOSTIC_COMPLETE'
                and prior['phase_verdicts']=={'A':'DIVERGENT','B':'PASS'}
                and prior['comparisons']['B']['bitwise_equal'] is True
                and prior['recorded_physical_transitions']==1024
                and prior['recorded_optimizer_updates']==80,'CAUSAL_EVIDENCE')
    h4x_file_pin(evidence[1],H4R_REWARD_SHA)
    h4x_file_pin(a.checkpoint,SEED1_SHA)
    h4x_file_pin(a.restore_report,H3_CAUSAL_RESTORE_SHA)
    if a.h4_arm=='candidate':
        h4x_file_pin(a.h4_control_report,auth['evidence_sha256'])
        h4x_require(a.h4_control_report.resolve()==root/H4X_PATHS['control']/'report.json','CONTROL_PATH')
        prior_control=json.loads(a.h4_control_report.read_bytes())
        prior_auth=prior_control['h4x_authorization']
        h4x_require(prior_auth=={**auth,'arm':'control','output':H4X_PATHS['control'],
                    'evidence_sha256':H4X_CAUSAL_SHA},'CONTROL_AUTHORIZATION_CONTRACT')
        for name,value in (('attempt.json',{'authorization':prior_auth,'reserved':True}),
                           ('dispatch.started.json',{'authorization':prior_auth}),
                           ('attempt_result.json',{'status':'COMPLETE','authorization':prior_auth})):
            marker=a.h4_control_report.parent/name
            h4x_require(marker.is_file() and not marker.is_symlink() and
                        json.loads(marker.read_bytes())==value,'CONTROL_ATTEMPT '+name)
        h4x_verify_operational(prior_control)
        h4_read_control(a.h4_control_report,auth['evidence_sha256'],prior_control['h4_common_contract'])
    return auth


def h4x_storage(out,extra=0,kind='other',initial=False):
    import shutil
    h4x_require(type(extra) is int and extra>=0 and kind in ('policy','fullstate','other'),'STORAGE_SCHEMA')
    free=shutil.disk_usage(out.parent if initial else out).free
    if initial:
        h4x_require(free>=H4X_SPEC['disk_reserve']+H4X_SPEC['output_cap'],'DISK_INITIAL')
        return
    sizes={'policy':0,'fullstate':0,'other':0}
    for p in out.rglob('*'):
        h4x_require(not p.is_symlink(),'STORAGE_SYMLINK')
        if p.is_dir():
            continue
        h4x_require(p.is_file(),'STORAGE_FILE')
        k=('policy' if p.name.startswith('m463_easy_repeat_support_longtrain_step_')
             and p.suffix=='.msgpack' else 'fullstate' if p.name.startswith('m470_fullstate_step_')
             and p.suffix=='.msgpack' else 'other')
        size=p.stat().st_size
        if k!='other':
            h4x_require(size<=H4X_SPEC[k+'_cap'],'ARTIFACT_SIZE')
        sizes[k]+=size
    sizes[kind]+=extra
    # Leave bounded space for the launcher log outside the exclusive output.
    # The future GO launcher must enforce this external cap.
    sizes['other']+=H4X_SPEC['external_log_reserve']
    h4x_require(extra<=H4X_SPEC[kind+'_cap'] and
                sizes['policy']<=2*H4X_SPEC['policy_cap'] and
                sizes['fullstate']<=2*H4X_SPEC['fullstate_cap'] and
                sizes['other']<=H4X_SPEC['other_cap'] and
                sum(sizes.values())<=H4X_SPEC['output_cap'],'STORAGE_CAP')
    h4x_require(free>=H4X_SPEC['disk_reserve']+extra+H4X_SPEC['external_log_reserve'],'DISK_RESERVE')


def h4x_write_json(path,value,exclusive=False):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    h4x_storage(path.parent,len(raw))
    if exclusive:
        with path.open('xb') as f:
            f.write(raw)
    else:
        tmp=path.with_suffix('.json.tmp')
        with tmp.open('xb') as f:
            f.write(raw)
        tmp.replace(path)


def h4x_generated_guard(audit,generated_file):
    # Called before the first upstream runtime operation. No active marker,
    # no valid reservation, or repeated generated entry => hard stop.
    auth=h4x_authorization()
    root=Path(__file__).resolve().parent
    out=root/auth['output']
    generated=Path(generated_file)
    h4x_require(audit is not None and callable(audit) and
                getattr(audit,'h4x_authorization',None)==auth,'AUDIT_REQUIRED')
    h4x_require(generated.resolve()==out/'m463_brax_ppo_instrumented.py'
                and not generated.is_symlink() and digest(generated)==auth['generated_sha256'],
                'GENERATED_PIN')
    reservation=out/'attempt.json'
    h4x_require(reservation.is_file() and not reservation.is_symlink() and
                json.loads(reservation.read_bytes())=={'authorization':auth,'reserved':True},
                'RESERVATION')
    h4x_write_json(out/'dispatch.started.json',{'authorization':auth},exclusive=True)


def h4x_bootstrap(args,*,h4cc=False,h4m6=None):
    from types import SimpleNamespace
    proxy=SimpleNamespace(checkpoint=args.checkpoint,restore_report=args.restore_report,
                          h4_readiness_reward_report=args.h4_exploratory_reward_report)
    pins=h4r_bootstrap_pins(proxy,h4cc=h4cc,h4m6=h4m6)
    # Exact launcher overlay resolution before JAX is imported.
    from importlib import metadata
    from importlib.machinery import PathFinder
    root=Path(__file__).resolve().parent
    base=root.parent
    expected=[
        base/'G1_M3_11E_PPO_CLOUDPICKLE_OVERLAY_V1/overlays/cloudpickle_20261107_111524_3962615',
        base/'G1_M3_11D_PPO_GYM_ISOLATED_REPAIR_V1/overlays/wheels_20261107_111000_3952924',
        base/'G1_M3_11C_PPO_MANIFEST_REPAIR_V1/overlays/wheels_20261107_110316_3940764',
        base/'G1_M3_7_G1_MJX_GPU_ONE_STEP_V1/source/mujoco_playground_2025',root]
    paths=[Path(p).resolve() for p in os.environ.get('PYTHONPATH','').split(os.pathsep)]
    h4x_require(paths==[p.resolve() for p in expected],'LAUNCHER_PATH')
    venv=base/'G1_M3_7_G1_MJX_GPU_ONE_STEP_V1/.venv_m37_jax0435_mjx327'
    h4x_require(Path(sys.executable)==venv/'bin/python' and
                Path(sys.prefix).resolve()==venv.resolve(),'LAUNCHER_PYTHON')
    spec=PathFinder.find_spec('dm_env',sys.path)
    origin=(expected[2]/'dm_env/__init__.py').resolve()
    dist=metadata.distribution('dm-env')
    h4x_require(spec is not None and spec.origin is not None and
                Path(spec.origin).resolve()==origin and dist.version=='1.6' and
                Path(dist.locate_file('dm_env/__init__.py')).resolve()==origin,'DM_ENV')
    return pins


def h4x_schedule():
    rows=['reset_before','reset_after','initial','eval_before','eval_after']
    for _ in range(2):
        rows.append('epoch')
        for _ in range(4):
            rows.extend(('chunk_before','chunk_after','chunk_ledger'))
        rows.extend(('reset_before','reset_after','eval_before','eval_after'))
    return rows+['finish']


class H4XBudget:
    def __init__(self,report):
        self.report=report
        report.update(spec=copy.deepcopy(H4X_SPEC),attempted=dict(H4X_COUNTERS),
                      recorded=dict(H4X_COUNTERS),events=[],complete=False)

    def check(self,event):
        i=len(self.report['events'])
        schedule=h4x_schedule()
        h4x_require(i<len(schedule) and schedule[i]==event,'SEQUENCE '+event)

    def advance(self,event,detail):
        self.check(event)
        r=self.report
        charges={'reset_before':('attempted',{'train_reset_lanes':8192}),
                 'reset_after':('recorded',{'train_reset_lanes':8192}),
                 'eval_before':('attempted',{'evaluation':2000,'eval_reset_lanes':2}),
                 'eval_after':('recorded',{'evaluation':2000,'eval_reset_lanes':2}),
                 'chunk_before':('attempted',{'training':1310720,'optimizer':200}),
                 'chunk_ledger':('recorded',{'training':1310720,'optimizer':200})}
        if event in charges:
            target,counts=charges[event]
            new=dict(r[target])
            for k,v in counts.items():
                new[k]+=v
                h4x_require(new[k]<=H4X_SPEC[k],'BUDGET '+k)
                if target=='recorded':
                    h4x_require(new[k]<=r['attempted'][k],'UNATTEMPTED '+k)
            r[target]=new
        r['events'].append({'event':event,**detail})
        if event=='finish':
            expected={k:H4X_SPEC[k] for k in H4X_COUNTERS}
            h4x_require(r['attempted']==r['recorded']==expected,'FINAL_BUDGET')
            r['complete']=True


def h4x_normalizer_expected(np,value,dtype,rollouts):
    h4x_require(type(rollouts) is int and 0<=rollouts<=40,'NORMALIZER_ROLLOUTS')
    target=np.asarray(value,dtype=np.dtype(dtype))
    h4x_require(target.dtype.kind in 'fiu' and np.isfinite(target).all(),'NORMALIZER_DTYPE')
    for _ in range(rollouts):
        target=np.add(target,np.asarray(262144,dtype=target.dtype),dtype=target.dtype)
    return target


def h4x_verify_operational(report, *, h4cc=False,h4m6=False):
    import numpy as np
    h4x_require(report['h4x_exploratory'] is True,'EXPLORATORY_REPORT')
    auth=report['h4x_authorization']
    if h4m6:
        h4m6_require(h4cc and auth['id']==H4M6_ID and auth['phase']==report['h4m6_arm']
            and report['h4_arm']=='candidate' and report['h4cc_arm']=='ref'
            and auth['output']==H4M6_PATHS[auth['phase']]
            and auth['generated_sha256']==report['h4_common_contract']['generated_brax_sha256'],'OPERATIONAL_AUTH')
    elif h4cc:
        h4x_require(auth['id']==H4CC_ID and auth['arm']==report['h4cc_arm']
                    and report['h4_arm']=='candidate'
                    and auth['output']==str(H4CC_PATHS[auth['arm']])
                    and auth['generated_sha256']==report['h4_common_contract']['generated_brax_sha256'],
                    'CCRC_REPORT_AUTHORIZATION')
    else:
        h4x_require(auth['id']==H4X_ID and auth['arm']==report['h4_arm'] and
                    auth['output']==H4X_PATHS[auth['arm']] and
                    auth['generated_sha256']==report['h4_common_contract']['generated_brax_sha256'],
                    'REPORT_AUTHORIZATION')
    r=report['h4x_operational']
    h4x_require(r['spec']==H4X_SPEC and r['complete'] is True,'OPERATIONAL_CONTRACT')
    replay={}
    budget=H4XBudget(replay)
    chunk=0
    after=None
    previous=None
    evals=[]
    resets=[]
    epochs=[]
    pending_eval=None
    previous_eval=None
    initial=next(row for row in r['events'] if row['event']=='initial')
    dtype=initial['normalizer_dtype']
    def counters(row,n):
        h4x_require(row['steps']==n*1310720 and row['adam']==n*200 and
                    row['normalizer_dtype']==dtype and row['training_schema']==initial['training_schema'] and
                    row['environment_schema']==initial['environment_schema'] and
                    row['normalizer']==h4x_normalizer_expected(np,initial['normalizer'],dtype,n*5).item(),
                    'REPORT_COUNTER')
    for row in r['events']:
        event=row['event']
        budget.advance(event,{k:v for k,v in row.items() if k!='event'})
        if event=='reset_before':
            h4x_require(row['dr_sha256']==initial['dr_sha256'] and len(row['keys_sha256'])==64,'REPORT_RESET')
            resets.append(row['keys_sha256'])
        if event=='reset_after':
            h4x_require(row['finite'] is True and row['environment_schema']==initial['environment_schema'],
                        'REPORT_RESET_FINITE')
        if event=='epoch':
            counters(row,chunk)
            epochs.append(row['key'])
            previous=None
        if event=='eval_before':
            h4x_require(row['step']==chunk*1310720 and
                        (previous_eval is None or row['key']==previous_eval),'REPORT_EVAL_PRE')
            evals.append(row['step'])
            pending_eval=row
        if event=='eval_after':
            h4x_require(pending_eval is not None and row['step']==pending_eval['step'] and
                        row['key']==pending_eval['expected_next_key'] and row['finite_metrics'] is True,
                        'REPORT_EVAL_POST')
            previous_eval=row['key']
        if event=='chunk_ledger':
            h4x_require(row['chunk']==chunk and row['passed'] is True,'REPORT_LEDGER')
        if event=='finish':
            counters(row,8)
        if event=='chunk_before':
            counters(row,chunk)
            h4x_require(row['expected_normalizer']==h4x_normalizer_expected(np,initial['normalizer'],dtype,(chunk+1)*5).item(),'REPORT_EXPECTED_NORMALIZER')
            h4x_require(row['chunk']==chunk+1 and row['steps']==chunk*1310720
                        and row['adam']==chunk*200,'REPORT_PRE_COUNTER')
            if previous is not None:
                h4x_require(row['key']==previous['key'] and row['normalizer']==previous['normalizer']
                            and row['same_carry'] is True,'REPORT_CARRY')
            else:
                h4x_require(row['key']==epochs[-1],'REPORT_EPOCH_KEY')
            after=row
        if event=='chunk_after':
            chunk+=1
            counters(row,chunk)
            h4x_require(after is not None and row['chunk']==chunk and row['finite'] is True
                        and row['steps']==chunk*1310720 and row['adam']==chunk*200
                        and row['key']==after['expected_next_key']
                        and row['normalizer']==after['expected_normalizer'],'REPORT_POST_COUNTER')
            previous=row
    h4x_require(replay==r and chunk==8 and len(resets)==3 and len(epochs)==2
                and evals==[0,*H4_EDGES],'REPORT_OPERATIONAL_REPLAY')
    counters(initial,0)
    h4x_require(initial['restored_leaves']==25 and initial['adam']==0
                and initial['steps']==0 and len(initial['dr_sha256'])==64,'REPORT_INITIAL')
    return initial['dr_sha256']


def h4x_tree_schema(jax,np,tree):
    # Read only pytree structure/array metadata, not dynamic state values.
    leaves,structure=jax.tree_util.tree_flatten(tree)
    fields=[]
    for leaf in leaves:
        if hasattr(leaf,'shape') and hasattr(leaf,'dtype'):
            shape,dtype=list(leaf.shape),np.dtype(leaf.dtype)
        else:
            a=np.asarray(leaf)
            shape,dtype=list(a.shape),a.dtype
        h4x_require(dtype.kind in 'biuf','TREE_DTYPE')
        fields.append([shape,str(dtype)])
    return hashlib.sha256(json.dumps([str(structure),fields],separators=(',',':')).encode()).hexdigest()


def h4x_model_digest(jax,np,model,axes):
    # Hash actual randomized model, not just the seed. Transfer small slices
    # of static model leaves, never the full batched dynamic environment.
    leaves,structure=jax.tree_util.tree_flatten_with_path((model,axes))
    h=hashlib.sha256(str(structure).encode())
    for path,v in leaves:
        shape=tuple(v.shape) if hasattr(v,'shape') else ()
        dtype=np.dtype(v.dtype) if hasattr(v,'dtype') else np.asarray(v).dtype
        h4x_require(dtype.kind in 'biuf','MODEL_DTYPE')
        h.update(json.dumps([jax.tree_util.keystr(path),list(shape),str(dtype)]).encode()+b'\0')
        if not shape:
            h.update(np.asarray(jax.device_get(v)).tobytes())
        else:
            row_bytes=max(1,math.prod(shape[1:])*dtype.itemsize)
            stride=max(1,(4*1024**2)//row_bytes)
            for i in range(0,shape[0],stride):
                h.update(np.asarray(jax.device_get(v[i:i+stride])).tobytes())
    return h.hexdigest()


def h4x_audit_callback(jax,np,monitor,report,persist,args,auth,control_dr):
    import optax
    from flax import serialization
    from m462_typed_orbax_stage import checkpoint_parts,compare_exact
    from m462_ppo_warmstart_run import sample_gpu
    expected_parts=checkpoint_parts(serialization.msgpack_restore(args.checkpoint.read_bytes()))
    op={}
    report['h4x_operational']=op
    budget=H4XBudget(op)
    _,local=jax.random.split(jax.random.PRNGKey(1))
    local=jax.random.fold_in(local,0)
    local,key_env,eval_key=jax.random.split(local,3)
    env_keys=jax.random.split(key_env,8192).reshape(1,8192,2)
    epoch_keys=None
    last_reset=None
    last_carry=None
    model_ref=axes_ref=dr_sha=None
    initial_count=None
    training_schema=environment_schema=None
    pending=None
    def array(v):
        return np.asarray(jax.device_get(v))
    def same(x,y,label):
        a,b=array(x),array(y)
        h4x_require(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),'KEY '+label)
    def key_list(v):
        a=array(v)
        h4x_require(a.dtype==np.dtype('uint32') and a.shape in ((2,),(1,2)),'KEY_SCHEMA')
        return a.tolist()
    def key_hash(v):
        a=array(v)
        h4x_require(a.dtype==np.dtype('uint32'),'KEY_DTYPE')
        return hashlib.sha256(a.tobytes()).hexdigest()
    def env_schema(state):
        value=h4x_tree_schema(jax,np,state)
        h4x_require(value==environment_schema,'ENV_SCHEMA')
        return value
    def state_check(ts,chunk):
        h4x_require(h4x_tree_schema(jax,np,ts)==training_schema,'TRAINING_SCHEMA')
        steps=array(ts.env_steps)
        count=array(ts.normalizer_params.count)
        adam=[x for x in jax.tree_util.tree_leaves(ts.optimizer_state,
                    is_leaf=lambda x:isinstance(x,optax.ScaleByAdamState))
              if isinstance(x,optax.ScaleByAdamState)]
        h4x_require(len(adam)==1 and steps.shape==count.shape==(1,) and
                    steps.dtype.kind in 'iu' and int(steps[0])==chunk*1310720,'STATE_STEPS')
        ac=array(adam[0].count)
        h4x_require(ac.shape==(1,) and ac.dtype.kind in 'iu' and int(ac[0])==chunk*200,'ADAM_COUNT')
        target=h4x_normalizer_expected(np,initial_count,initial_count.dtype,chunk*5)
        same(count,target,'NORMALIZER_COUNT')
        if chunk==0:
            h4x_require(all(not np.any(array(v)) for v in
                     jax.tree_util.tree_leaves((adam[0].mu,adam[0].nu))),'ADAM_FRESH')
        return {'steps':int(steps[0]),'adam':int(ac[0]),'normalizer':count.item(),
                'normalizer_dtype':str(count.dtype),'training_schema':training_schema}
    def safe():
        h4x_storage(args.output_dir)
        g=sample_gpu()
        snap=monitor.snapshot()
        h4x_require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288 and
                    not snap['sampler_errors'] and (snap['peak_used_mib'] is None or
                    snap['peak_used_mib']<.95*12288),'GPU')
    def callback(event,**v):
        nonlocal local,eval_key,env_keys,epoch_keys,last_reset,last_carry,model_ref,axes_ref,dr_sha
        nonlocal initial_count,pending,training_schema,environment_schema
        budget.check(event)
        if event.endswith('_before'):
            rules=Path(__file__).with_name('G1_RESEARCH_RULES.md').read_text('utf8')
            current=(h4m6_authorization(phase=auth['phase']) if auth['id']==H4M6_ID else
                     h4cc_authorization() if auth['id']==H4CC_ID else h4x_parse_marker(rules))
            h4x_require(current==auth,'AUTHORIZATION_CHANGED')
        safe()
        chunk=op['recorded']['optimizer']//200
        row={}
        if event=='reset_before':
            same(v['local_key'],local,'LOCAL_RESET')
            same(v['eval_key'],eval_key,'EVAL_RESET')
            if op['recorded']['train_reset_lanes']:
                env_keys=jax.random.split(env_keys[0,0],8192).reshape(1,8192,2)
                h4x_require(v['model'] is model_ref and v['axes'] is axes_ref,'DR_MUTATION')
            else:
                model_ref,axes_ref=v['model'],v['axes']
                dr_sha=h4x_model_digest(jax,np,model_ref,axes_ref)
                h4x_require(control_dr is None or dr_sha==control_dr,'MATCHED_DR')
            same(v['keys'],env_keys,'ENV_RESET')
            row={'keys_sha256':key_hash(v['keys']),'local_key':key_list(local),
                 'eval_key':key_list(eval_key),'dr_sha256':dr_sha}
            last_carry=None
        elif event=='reset_after':
            finite=array(v['finite'])
            h4x_require(finite.shape==() and finite.dtype.kind=='b' and bool(finite),'RESET_NONFINITE')
            last_reset=v['env_state']
            h4x_require(array(last_reset.done).shape==(1,8192),'RESET_LAYOUT')
            if environment_schema is None:
                environment_schema=h4x_tree_schema(jax,np,last_reset)
            row={'finite':True,'environment_schema':env_schema(last_reset)}
        elif event=='initial':
            ts=v['training_state']
            finite=array(v['finite'])
            h4x_require(v['env_state'] is last_reset and finite.shape==() and
                        finite.dtype.kind=='b' and bool(finite),'INITIAL_CARRY')
            training_schema=h4x_tree_schema(jax,np,ts)
            initial_count=array(ts.normalizer_params.count).copy()
            restored=tuple(serialization.to_state_dict(jax.tree_util.tree_map(lambda x:x[0],p))
                       for p in (ts.normalizer_params,ts.params.policy,ts.params.value))
            h4x_require(compare_exact(expected_parts,restored,jax,np,'H4X_RESTORE')==25,'RESTORE')
            row={**state_check(ts,0),'restored_leaves':25,'dr_sha256':dr_sha,
                 'environment_schema':env_schema(v['env_state'])}
        elif event=='epoch':
            h4x_require(v['env_state'] is last_reset,'EPOCH_RESET_STATE')
            epoch,local=jax.random.split(local)
            epoch_keys=jax.random.split(epoch,1)
            same(v['keys'],epoch_keys,'EPOCH_KEY')
            same(v['local_key'],local,'EPOCH_LOCAL')
            row={**state_check(v['training_state'],chunk),'key':key_list(epoch_keys),
                 'local_key':key_list(local),'environment_schema':env_schema(v['env_state'])}
            last_reset=None
        elif event=='chunk_before':
            carry=v['carry']
            h4x_require(v['index']==chunk%4+1,'CHUNK_INDEX')
            same_carry=last_carry is carry
            if v['index']==1:
                same(carry[2],epoch_keys,'CHUNK_ROOT')
            else:
                h4x_require(same_carry,'CARRY_REPLACED')
            key=carry[2][0]
            for _ in range(5):
                key=jax.random.split(key,3)[2]
            target=h4x_normalizer_expected(np,initial_count,initial_count.dtype,(chunk+1)*5)
            pending={'key':key[None,:],'normalizer':target}
            row={**state_check(carry[0],chunk),'chunk':chunk+1,'key':key_list(carry[2]),
                 'expected_next_key':key_list(pending['key']),
                 'expected_normalizer':target.item(),'same_carry':same_carry,
                 'environment_schema':env_schema(carry[1])}
        elif event=='chunk_after':
            carry=v['carry']
            h4x_require(v['index']==chunk%4+1 and pending is not None,'CHUNK_COMPLETION')
            same(carry[2],pending['key'],'CHUNK_NEXT')
            finite=array(v['metrics']['h4x_finite'])
            h4x_require(finite.shape==() and finite.dtype==np.dtype('float32') and
                        finite.item()==1.0,'ROLLOUT_NONFINITE')
            row={**state_check(carry[0],chunk+1),'chunk':chunk+1,
                 'key':key_list(carry[2]),'finite':True,'environment_schema':env_schema(carry[1])}
            last_carry=carry
            pending=None
        elif event=='chunk_ledger':
            h4x_require(v['index']==chunk%4+1 and len(report['h4_chunks'])==chunk+1
                        and report['h4_chunks'][-1]['passed'] is True,'CHUNK_LEDGER')
            row={'chunk':chunk+1,'passed':True}
        elif event=='eval_before':
            same(v['key'],eval_key,'EVAL_BEFORE')
            next_eval,unroll_key=jax.random.split(eval_key)
            row={'step':v['step'],'key':key_list(eval_key),'expected_next_key':key_list(next_eval),
                 'derived_reset_keys_sha256':key_hash(jax.random.split(unroll_key,2))}
        elif event=='eval_after':
            eval_key=jax.random.split(eval_key)[0]
            same(v['key'],eval_key,'EVAL_AFTER')
            h4x_require(all(np.isfinite(array(x)).all() for x in v['metrics'].values()),'EVAL_NONFINITE')
            row={'step':v['step'],'key':key_list(eval_key),'finite_metrics':True}
        elif event=='finish':
            h4x_require(v['env_state'] is last_reset and chunk==8,'FINAL_CARRY')
            row={**state_check(v['training_state'],8),'environment_schema':env_schema(v['env_state'])}
        else:
            raise RuntimeError('M470_H4X_EVENT')
        if event.startswith('eval_'):
            h4x_require(v['step']==chunk*1310720 and v['step'] in (0,*H4_EDGES),'EVAL_EDGE')
        budget.advance(event,row)
        persist()
    callback.h4x_authorization=auth
    return callback


def h4x_run(args):
    # Reserve once before any bootstrap/import of the actual runtime.
    auth=h4x_validate_cli(args)
    h4x_require(auth is not None,'EXECUTION_HOLD')
    out=args.output_dir
    h4x_require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'OUTPUT_CONSUMED')
    h4x_storage(out,initial=True)
    out.mkdir(exist_ok=False)
    h4x_write_json(out/'attempt.json',{'authorization':auth,'reserved':True},exclusive=True)
    try:
        pins=h4x_bootstrap(args)
        _run_impl(args,h4x={'authorization':auth,'pins':pins})
        h4x_write_json(out/'attempt_result.json',{'status':'COMPLETE','authorization':auth},exclusive=True)
    except BaseException as exc:
        try:
            h4x_write_json(out/'attempt_result.json',{'status':'STOPPED','error':repr(exc),
                           'authorization':auth},exclusive=True)
        except Exception as write_error:
            print('M470_H4X_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise




def h4x_static_regression(brax_train_path,acting_path,causal_report):
    """Exact checkout, focused on the new exploratory variant. No runtime."""
    import ast
    import subprocess
    import tempfile
    import types
    import numpy as np
    from unittest.mock import patch
    from types import SimpleNamespace
    import m463_instrument_brax as gen
    root=Path(__file__).resolve().parent
    module=sys.modules[__name__]
    def reject(fn,marker=None):
        try:
            fn()
        except (RuntimeError,FileExistsError) as exc:
            h4x_require(marker is None or marker in str(exc),'STATIC_WRONG_ERROR '+str(exc))
        else:
            raise RuntimeError('M470_H4X_STATIC_ACCEPTED_INVALID')
    for name in ('m463_push_ppo_train.py','m463_instrument_brax.py'):
        raw=(root/name).read_bytes()
        compile(raw,name,'exec')
        ast.parse(raw)
    h4x_require(not any((H4_EXECUTION_AUTHORIZED,H4R_EXECUTION_AUTHORIZED,H4C_EXECUTION_AUTHORIZED)),
                'STATIC_LEGACY_HOLDS')
    reject(h4x_authorization,'EXECUTION_HOLD')
    reject(lambda:h4x_generated_guard(None,'unused'),'EXECUTION_HOLD')
    h4x_file_pin(Path(causal_report),H4X_CAUSAL_SHA)
    raw=Path(brax_train_path).read_bytes()
    acting_raw=Path(acting_path).read_bytes()
    old=types.ModuleType('_h4x_previous_generator')
    old_raw=subprocess.check_output(['git','show','239a6749325bc79ec9924958593112ea644c4019:m463_instrument_brax.py'],cwd=root)
    exec(compile(old_raw,'<previous generator>','exec'),old.__dict__)
    # Only compare outputs of variants affected by the builder/generator change.
    held,original=gen.make_patched_source(raw,enable_resume=True,enable_h4=True)
    h4x_require((held,original)==old.make_patched_source(raw,enable_resume=True,enable_h4=True),'OLD_H4')
    h4x_require(gen.make_patched_source(raw)==old.make_patched_source(raw),'OLD_LEGACY')
    h4x_require(gen.make_h4_readiness_source(raw)==old.make_h4_readiness_source(raw),'OLD_H4R')
    h4x_require(gen.make_h4_causal_source(raw,acting_raw)==old.make_h4_causal_source(raw,acting_raw),'OLD_H4C')
    generated,sha=gen.make_h4_exploratory_source(raw)
    h4x_require(sha==original==gen.PINNED_SHA and gen.normalize_h4_exploratory_source(generated)==held,
                'GENERATED_SCOPE')
    compile(generated,'<exploratory generated>','exec')
    entry=next(n for n in ast.parse(generated).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    h4x_require(isinstance(entry.body[1],ast.ImportFrom) and isinstance(entry.body[2],ast.Expr)
                and 'h4x_generated_guard' in ast.unparse(entry.body[2]),'EARLY_GENERATED_GUARD')
    h4x_require(generated.count("h4x_audit_fn('chunk_before'")==1 and
                generated.count("h4x_audit_fn('chunk_after'")==1 and
                generated.count("h4x_audit_fn('eval_before'")==2 and
                generated.count("h4x_audit_fn('reset_before'")==2 and
                gen.H4X_FINITE_SOURCE in generated,'GENERATED_HOOKS')
    print('M470_H4X_GENERATED_SCOPE_PASS',gen.sha256(generated.encode()),flush=True)
    # Exact CLI on current checkout, with no real execution.
    cli=[sys.executable,'-B',str(root/'m463_push_ppo_train.py'),
         '--stage','easy_repeat_support','--checkpoint','unused','--restore-report','unused',
         '--output-dir','unused']
    cases=[]
    for arm in ('control','candidate'):
        flags=['--h4-exploratory','--h4-arm',arm]
        cases.extend([(['--mode','plan']+flags,0),(['--mode','longtrain']+flags,2)])
        for extra in (['--h4-readiness'],['--h4-causal'],['--h3-m1'],['--resume-state','unused'],
                      ['--rollouts','40'],['--num-evals','3'],['--independent-budget','10M'],
                      ['--h4-control-report','unused']):
            cases.append((['--mode','plan']+flags+extra,2))
    cases.extend([(['--mode','plan','--h4-exploratory'],2),
                  (['--mode','plan','--h4-exploratory-causal-report','unused'],2),
                  (['--mode','longtrain','--h4-arm','control'],2)])
    for extra,rc in cases:
        proc=subprocess.run(cli+extra,capture_output=True,text=True,cwd=root)
        h4x_require(proc.returncode==rc and (rc!=0 or 'M470_H4X_PLAN_NO_GPU' in proc.stdout),
                    'STATIC_CLI '+repr(extra)+' '+proc.stderr)
    args=SimpleNamespace(h4_exploratory=True,mode='longtrain',stage='easy_repeat_support',h4_arm='control')
    reject(lambda:run(args),'EXECUTION_HOLD')
    reject(lambda:_run_impl(args),'RUN_RESERVATION')
    print('M470_H4X_CLI_HOLD_PASS',len(cases),flush=True)
    # Marker parser and one-use generated entry fixture, no actual Rules edit.
    marker='H4X_ACTIVE_GO: '+H4X_ID+' control '+H4X_PATHS['control']+' '+'a'*64+' '+'b'*64+' '+H4X_CAUSAL_SHA
    auth=h4x_parse_marker(marker)
    for bad in ('',marker+'\n'+marker,marker.replace(H4X_PATHS['control'],H4X_PATHS['candidate']),
                marker.replace('a'*64,'a'*63)):
        reject(lambda:h4x_parse_marker(bad))
    for arm,path in (('candidate',H4X_PATHS['control']),('control',H4X_PATHS['candidate'])):
        bad=SimpleNamespace(h4_exploratory=True,mode='longtrain',stage='easy_repeat_support',
                            h4_arm=arm,output_dir=root/path)
        with patch.object(module,'h4x_authorization',return_value=auth):
            reject(lambda:h4x_validate_cli(bad),'ARM_OR_OUTPUT')
    with tempfile.TemporaryDirectory(prefix='h4x-static-') as temp:
        temp=Path(temp)
        out=temp/H4X_PATHS['control']
        out.mkdir(parents=True)
        artifact=out/'m463_brax_ppo_instrumented.py'
        artifact.write_text('offline fixture',encoding='utf8')
        authorization={**auth,'generated_sha256':digest(artifact)}
        def audit(*a,**kw):
            raise AssertionError('fixture should not invoke training')
        audit.h4x_authorization=authorization
        with patch.object(module,'__file__',str(temp/'runner.py')), \
             patch.object(module,'h4x_authorization',return_value=authorization), \
             patch.object(module,'h4x_storage',return_value=None):
            reject(lambda:h4x_generated_guard(audit,artifact),'RESERVATION')
            h4x_write_json(out/'attempt.json',{'authorization':authorization,'reserved':True},exclusive=True)
            reject(lambda:h4x_generated_guard(None,artifact),'AUDIT_REQUIRED')
            h4x_generated_guard(audit,artifact)
            reject(lambda:h4x_generated_guard(audit,artifact))
            artifact.write_text('changed',encoding='utf8')
            reject(lambda:h4x_generated_guard(audit,artifact),'GENERATED_PIN')
        with patch.object(module,'h4x_validate_cli',return_value=auth):
            reject(lambda:h4x_run(SimpleNamespace(output_dir=out)),'OUTPUT_CONSUMED')
    print('M470_H4X_AUTHORIZATION_FIXTURE_PASS',flush=True)
    # Storage fixture uses sparse files, not real checkpoint data.
    with tempfile.TemporaryDirectory(prefix='h4x-storage-') as temp:
        out=Path(temp)
        import shutil
        disk=SimpleNamespace(free=8*1024**3)
        with patch.object(shutil,'disk_usage',return_value=disk):
            h4x_storage(out)
            reject(lambda:h4x_storage(out,H4X_SPEC['policy_cap']+1,'policy'),'STORAGE_CAP')
            p=out/'m470_fullstate_step_005242880.msgpack'
            with p.open('wb') as f:
                f.truncate(H4X_SPEC['fullstate_cap']+1)
            reject(lambda:h4x_storage(out),'ARTIFACT_SIZE')
            p.unlink()
            reject(lambda:h4x_storage(out,H4X_SPEC['other_cap']),'STORAGE_CAP')
            disk.free=H4X_SPEC['disk_reserve']
            reject(lambda:h4x_storage(out,1),'DISK_RESERVE')
    print('M470_H4X_STORAGE_FIXTURE_PASS',flush=True)
    # Actual callback with pure NumPy/mock JAX and fake restore/monitor modules.
    # The mock validates host logic only, not real JAX PRNG or simulator.
    class Adam:
        def __init__(self,n):
            self.count=np.array([n],np.int32)
            self.mu=self.nu=np.zeros((1,1),np.float32)
    class Random:
        @staticmethod
        def PRNGKey(seed):
            return np.array([0,seed],np.uint32)
        @staticmethod
        def split(key,n=2):
            a,b=map(int,key)
            i=np.arange(1,n+1,dtype=np.uint64)
            return np.stack(((a*7+i)%2**32,(b*3+i)%2**32),axis=-1).astype(np.uint32)
        @staticmethod
        def fold_in(key,i):
            return Random.split(key,i+1)[-1]
    def leaves(v,is_leaf=None):
        if is_leaf and is_leaf(v):
            return [v]
        if isinstance(v,(tuple,list)):
            return [x for a in v for x in leaves(a,is_leaf)]
        if isinstance(v,dict):
            return [x for a in v.values() for x in leaves(a,is_leaf)]
        if isinstance(v,SimpleNamespace):
            return leaves(vars(v),is_leaf)
        return [v]
    def flatten(v):
        if isinstance(v,Adam):
            return [v.count,v.mu,v.nu],'Adam'
        if isinstance(v,SimpleNamespace):
            v=vars(v)
        if isinstance(v,dict):
            pairs=[(k,flatten(a)) for k,a in sorted(v.items())]
            return [leaf for _,(ls,_) in pairs for leaf in ls],str([(k,t) for k,(_,t) in pairs])
        if isinstance(v,(tuple,list)):
            pairs=[flatten(a) for a in v]
            return [leaf for ls,_ in pairs for leaf in ls],str([t for _,t in pairs])
        return [v],'leaf'
    def tree_map(fn,v):
        if isinstance(v,SimpleNamespace):
            return SimpleNamespace(**{k:tree_map(fn,a) for k,a in vars(v).items()})
        return fn(v)
    fake_jax=SimpleNamespace(random=Random(),device_get=lambda x:x,
        tree_util=SimpleNamespace(tree_leaves=leaves,tree_map=tree_map,tree_flatten=flatten))
    gpu={'total_mib':12288,'used_mib':100,'peak_used_mib':100,'sampler_errors':[]}
    fake_modules={
        'optax':SimpleNamespace(ScaleByAdamState=Adam),
        'flax':SimpleNamespace(serialization=SimpleNamespace(msgpack_restore=lambda b:None,to_state_dict=lambda x:x)),
        'm462_typed_orbax_stage':SimpleNamespace(checkpoint_parts=lambda x:None,compare_exact=lambda *a:25),
        'm462_ppo_warmstart_run':SimpleNamespace(sample_gpu=lambda:gpu)}
    def ts(n,adam_delta=0,count_delta=0):
        count=h4x_normalizer_expected(np,401080320,'float32',n*5).item()+count_delta
        return SimpleNamespace(env_steps=np.array([n*1310720],np.int32),
            normalizer_params=SimpleNamespace(count=np.array([count],np.float32)),
            optimizer_state=(Adam(n*200+adam_delta),),
            params=SimpleNamespace(policy=np.zeros((1,1),np.float32),value=np.zeros((1,1),np.float32)))
    def execute(failure=None):
        report={'h4_chunks':[],'h4x_exploratory':True,'h4x_authorization':auth,'h4_arm':'control',
                'h4_common_contract':{'generated_brax_sha256':auth['generated_sha256']}}
        model,axes=object(),object()
        monitor=SimpleNamespace(snapshot=lambda:gpu)
        args=SimpleNamespace(checkpoint=Path(causal_report),output_dir=Path('unused'))
        with patch.dict(sys.modules,fake_modules), \
             patch.object(module,'h4x_model_digest',return_value='d'*64), \
             patch.object(module,'h4x_storage',return_value=None), \
             patch.object(module,'h4x_parse_marker',return_value=auth):
            cb=h4x_audit_callback(fake_jax,np,monitor,report,lambda:None,args,auth,
                                 'e'*64 if failure=='dr' else None)
            _,local=Random.split(Random.PRNGKey(1))
            local=Random.fold_in(local,0)
            local,key_env,eval_key=Random.split(local,3)
            env_keys=Random.split(key_env,8192).reshape(1,8192,2)
            def reset():
                cb('reset_before',keys=env_keys,local_key=local,eval_key=eval_key,model=model,axes=axes)
                state=SimpleNamespace(done=np.zeros((1,8192),np.float32))
                cb('reset_after',env_state=state,finite=np.array(failure!='reset_finite'))
                return state
            def evaluate(step):
                nonlocal eval_key
                cb('eval_before',step=step,key=eval_key)
                eval_key=Random.split(eval_key)[0]
                cb('eval_after',step=step,key=eval_key if failure!='eval_key' else np.zeros((2,),np.uint32),
                   metrics={'reward':np.array(1.)})
            state=reset()
            training=ts(0)
            cb('initial',training_state=training,env_state=state,finite=np.array(True))
            evaluate(0)
            n=0
            for _ in range(2):
                epoch,local=Random.split(local)
                keys=Random.split(epoch,1)
                cb('epoch',keys=keys,local_key=local,training_state=training,env_state=state)
                carry=(training,state,keys)
                for index in range(1,5):
                    if failure=='carry' and index==2:
                        carry=tuple(list(carry))
                    cb('chunk_before',index=index,carry=carry)
                    key=carry[2][0]
                    for _ in range(5):
                        key=Random.split(key,3)[2]
                    n+=1
                    training=ts(n,int(failure=='adam'),64*int(failure=='normalizer'))
                    if failure=='training_schema':
                        training.params.policy=np.zeros((1,2),np.float32)
                    env=SimpleNamespace(done=np.zeros((1,8191 if failure=='env_schema' else 8192),np.float32))
                    carry=(training,env,key[None,:] if failure!='key' else np.zeros((1,2),np.uint32))
                    cb('chunk_after',index=index,carry=carry,metrics={
                       'h4x_finite':np.array(0. if failure=='finite' else 1.,np.float32)})
                    report['h4_chunks'].append({'passed':failure!='ledger'})
                    cb('chunk_ledger',index=index)
                env_keys=Random.split(env_keys[0,0],8192).reshape(1,8192,2)
                state=reset()
                evaluate(n*1310720)
            cb('finish',training_state=training,env_state=state)
        return report
    report=execute()
    h4x_require(h4x_verify_operational(report)=='d'*64,'MOCK_OPERATIONAL')
    for failure,marker in (('dr','MATCHED_DR'),('reset_finite','RESET_NONFINITE'),
                           ('eval_key','KEY EVAL_AFTER'),('carry','CARRY_REPLACED'),
                           ('adam','ADAM_COUNT'),('normalizer','KEY NORMALIZER_COUNT'),
                           ('key','KEY CHUNK_NEXT'),('finite','ROLLOUT_NONFINITE'),('ledger','CHUNK_LEDGER'),
                           ('training_schema','TRAINING_SCHEMA'),('env_schema','ENV_SCHEMA')):
        reject(lambda:execute(failure),marker)
    bad=copy.deepcopy(report)
    next(r for r in bad['h4x_operational']['events'] if r['event']=='chunk_after')['adam']+=1
    reject(lambda:h4x_verify_operational(bad),'REPORT_COUNTER')
    bad=copy.deepcopy(report)
    bad['h4x_operational']['recorded']['evaluation']-=2000
    reject(lambda:h4x_verify_operational(bad),'REPORT_OPERATIONAL_REPLAY')
    budget=H4XBudget({})
    reject(lambda:budget.advance('chunk_before',{}),'SEQUENCE')
    budget.report['attempted']['train_reset_lanes']=24576
    reject(lambda:budget.advance('reset_before',{}),'BUDGET')
    h4x_require('jax' not in sys.modules and 'brax' not in sys.modules and 'mujoco' not in sys.modules,
                'STATIC_RUNTIME_IMPORTED')
    print('M470_H4X_OPERATIONAL_FIXTURE_PASS 1 positive 11 negative',flush=True)
    print('M470_H4X_STATIC_PASS NO_SIMULATION NO_PPO',flush=True)


def validate_h4_cli(a):
    if getattr(a,'h4_exploratory',False):
        h4x_validate_cli(a)
        return
    h4x_validate_cli(a)
    arm=getattr(a,'h4_arm',None)
    if arm is None:
        if getattr(a,'h4_control_report',None) is not None:
            raise RuntimeError('M470_H4_EVIDENCE_REQUIRES_ARM')
        return
    if arm not in H4_SPEC['arms'] or a.stage!='easy_repeat_support':
        raise RuntimeError('M470_H4_ARM_STAGE')
    if (a.h1_m1 or a.h1_m2 or a.h3_m1 or a.h3_diagnostic_only or
            any(v is not None for v in (a.h3_causal_report,a.h3_old10m_report,
                a.resume_state,a.smoke_report,a.independent_budget,a.rollouts,a.num_evals))):
        raise RuntimeError('M470_H4_CONFLICT')
    if a.mode=='plan':
        if getattr(a,'h4_control_report',None) is not None:
            raise RuntimeError('M470_H4_PLAN_EVIDENCE_NOT_ALLOWED')
        return
    if not H4_EXECUTION_AUTHORIZED:
        raise RuntimeError('M470_H4_EXECUTION_HOLD')
    if a.mode!='longtrain' or ((arm=='candidate') != (a.h4_control_report is not None)):
        raise RuntimeError('M470_H4_RUNTIME_FLAGS')


def validate_h3_cli(a):
    diagnostic = bool(getattr(a, 'h3_diagnostic_only', False))
    causal = getattr(a, 'h3_causal_report', None)
    comparator = getattr(a, 'h3_old10m_report', None)
    if not getattr(a, 'h3_m1', False):
        if diagnostic:
            raise RuntimeError('M470_H3_DIAGNOSTIC_REQUIRES_H3')
        if causal is not None or comparator is not None:
            raise RuntimeError('M470_H3_EVIDENCE_REQUIRES_H3')
        return
    if diagnostic:
        raise RuntimeError('M470_H3_DIAGNOSTIC_CLOSED')
    if a.h1_m1 or a.h1_m2 or a.stage != 'easy_repeat_support' or a.resume_state is not None:
        raise RuntimeError('M470_H3_M1_MODE_NOT_APPROVED')
    if a.mode == 'plan':
        if any(v is not None for v in (causal, comparator, a.smoke_report,
                                      a.independent_budget, a.rollouts, a.num_evals)):
            raise RuntimeError('M470_H3_PLAN_FLAGS_NOT_APPROVED')
        return
    if a.mode != 'longtrain' or a.independent_budget != '10M':
        raise RuntimeError('M470_H3_ONLY_INDEPENDENT_10M_APPROVED')
    if (a.rollouts, a.num_evals) != (40, 6):
        raise RuntimeError('M470_H3_LONGTRAIN_BUDGET_MISMATCH')
    if any(v is None for v in (causal, comparator, a.smoke_report)):
        raise RuntimeError('M470_H3_LONGTRAIN_EVIDENCE_REQUIRED')


def read_h3_causal_report(path):
    """Accept only the reviewed diagnostic bytes; never reclassify exact physics."""
    if path is None or path.is_symlink() or not path.is_file():
        raise RuntimeError('M470_H3_CAUSAL_REPORT_PATH_INVALID')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != H3_CAUSAL_SHA:
        raise RuntimeError('M470_H3_CAUSAL_REPORT_SHA_MISMATCH')
    r = json.loads(raw)
    def require(ok, label):
        if not ok:
            raise RuntimeError('M470_H3_CAUSAL_'+label)
    expected = {
        'status': 'M470_H3_DIAGNOSTIC_COMPLETE_PPO_HOLD',
        'stage': 'easy_repeat_support', 'mode': 'smoke3',
        'experiment_class': 'H3_CAUSAL_VALIDATION_ONLY_NO_PPO',
        'source_commit': PIN, 'source_checkpoint_sha256': SEED1_SHA,
        'orbax_sha256': '5acfabc1376e91a2a9a52b47ac821e296f0e84f5545b646b07c9482ddc462bf3',
        'training_seed': 1, 'train_steps_target': 0, 'rollouts_target': 0,
        'steps_observed': 0, 'run_steps_observed': 0,
        'training_performed': False, 'report_history': [], 'checkpoints': [],
        'h3_diagnostic_only': True, 'h3_m1_enabled': True,
        'h1_m1_enabled': False, 'h1_m2_enabled': False,
        'h3_m1_spec': H3_SPEC, 'h3_ppo_fingerprint': H3_FP,
        'h3_source_digests': H3_CAUSAL_SOURCE_DIGESTS,
        'runner_sha256': H3_CAUSAL_SOURCE_DIGESTS['m463_push_ppo_train.py'],
        'h3_static_reward_config_unchanged': True, 'future_smoke_config_steps': 786432,
    }
    for key, value in expected.items():
        require(r.get(key) == value, 'REPORT_FIELD_'+key)
    require(m470_fingerprint(r['hardware_scaled_config']) == H3_FP,
            'CONFIG_FINGERPRINT')
    require(r['hardware_scaled_config']['num_timesteps'] == 786432
            and r['hardware_scaled_config']['num_evals'] == 2, 'CONFIG_BUDGET')
    require(r['gpu_total_mib'] == 12288 and 0 < r['gpu_peak_mib'] < .95*r['gpu_total_mib']
            and r['gpu_samples'] > 0 and r['gpu_monitor_errors'] == [], 'DIAGNOSTIC_GPU')
    im = r['instrumented_brax']
    require(im['original_brax_sha256'] == H3_CAUSAL_ORIGINAL_BRAX_SHA
            and im['instrumented_sha256'] == H3_CAUSAL_BRAX_SHA, 'DIAGNOSTIC_BRAX')
    g = r['h3_m1_technical_gate']
    require(g['historical_exact_gate'] == 'FAILED_85891a1'
            and g['passed'] is False and g['training_authorized'] is False
            and g['diagnostic_complete'] is True and g['shared_snapshot_passed'] is True
            and g['cross_executable_exact_equal'] is False, 'HISTORICAL_GATE')
    require(g['max_control_transitions'] == g['diagnostic_control_transitions'] == 54
            and g['direct_control_transitions'] == 36, 'DIAGNOSTIC_BUDGET')
    require(set(g['numerical_controls']) == {'off_off','on_on','off_on'}
            and all(v['exact_equal'] is False for v in g['numerical_controls'].values()),
            'NUMERICAL_CONTROLS')
    b,c = g['same_postphysics_reward'],g['wrapped_validation']
    require(b['passed'] is True and b['transitions'] == 9
            and b['unchanged_23_exact'] is True and b['input_dicts_unchanged'] is True,
            'SAME_POSTPHYSICS')
    require(c['passed'] is True and c['transitions'] == 9
            and c['autoreset_verified_not_recovery'] is True
            and c['audit']['passed'] is True, 'WRAPPED')
    audit = h3_audit_counts({PREFIX+k:v for k,v in c['audit']['counts'].items()})
    require(audit['passed'] and audit == c['audit'], 'WRAPPED_AUDIT')
    # The current runner intentionally differs; all production implementation
    # inputs and the evaluator must match the independently reviewed evidence.
    for name in ('m463_coverage_env.py','m463_instrument_brax.py'):
        require(digest(Path(__file__).with_name(name)) == H3_CAUSAL_SOURCE_DIGESTS[name],
                'CURRENT_SOURCE_'+name)
    require(digest(Path(__file__).with_name('m463_gate3_cp_eval.py'))
            == H3_CAUSAL_EVALUATOR_SHA, 'CURRENT_EVALUATOR')
    return {
        'basis': 'V1_2_65_CAUSAL_B_C_ACCEPTED_EXPLORATORY_SMOKE3',
        'diagnostic_report_sha256': H3_CAUSAL_SHA,
        'diagnostic_report_path': str(path.resolve()),
        'diagnostic_runner_sha256': H3_CAUSAL_SOURCE_DIGESTS['m463_push_ppo_train.py'],
        'historical_exact_gate': 'FAILED_85891a1',
        'independent_executable_exact_equal': False,
        'same_postphysics_reward_passed': True, 'wrapped_validation_passed': True,
        'diagnostic_control_transitions_reexecuted': 0,
        'ppo_steps_authorized': 786432, 'optimizer_updates_target': 120,
    }


def read_h3_independent_evidence(args, causal_basis):
    """Offline immutable evidence and comparator byte validation; no JAX."""
    def require(ok, label):
        if not ok:
            raise RuntimeError('M470_H3_10M_'+label)
    def read(path, sha, label):
        require(path is not None and path.is_file() and not path.is_symlink(), label+'_PATH')
        raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==sha, label+'_SHA')
        return json.loads(raw)
    sr=read(args.smoke_report,H3_SMOKE_SHA,'SMOKE')
    old=read(args.h3_old10m_report,H3_OLD10M_SHA,'OLD10M')
    for r,steps,rollouts,evals in ((sr,786432,3,2),(old,10485760,40,6)):
        require(r['source_commit']==PIN and r['source_checkpoint_sha256']==SEED1_SHA
                and r['orbax_sha256']=='5acfabc1376e91a2a9a52b47ac821e296f0e84f5545b646b07c9482ddc462bf3'
                and r['stage']=='easy_repeat_support' and r['training_seed']==1
                and r['train_steps_target']==r['steps_observed']==r['run_steps_observed']==steps
                and r['rollouts_target']==rollouts and r['hardware_scaled_config']['num_evals']==evals
                and r['hardware_scaled_config']['num_timesteps']==steps
                and m470_fingerprint(r['hardware_scaled_config'])==H3_FP
                and r['training_performed'] is True and r['weights_finite'] is True
                and r['callback_vs_final_equal'] is True and r['coverage_gate_pass'] is True
                and r['memory_gate_pass'] is True and not r['gpu_monitor_errors']
                and 0<r['gpu_peak_mib']<.95*r['gpu_total_mib']
                and r['gpu_total_mib']==12288
                and r['instrumented_brax']['original_brax_sha256']==H3_CAUSAL_ORIGINAL_BRAX_SHA,
                'COMMON_REPORT')
    require(sr['status']=='M463_COVERAGE_SMOKE3_PASS' and sr['h3_m1_spec']==H3_SPEC
            and sr['h3_m1_enabled'] is True and sr['h3_diagnostic_only'] is False
            and sr['h1_m1_enabled'] is False and sr['h1_m2_enabled'] is False
            and sr['h3_ppo_fingerprint']==H3_FP
            and sr['runner_sha256']==H3_SMOKE_RUNNER_SHA
            and sr['h3_source_digests']=={**H3_CAUSAL_SOURCE_DIGESTS,
                       'm463_push_ppo_train.py':H3_SMOKE_RUNNER_SHA}
            and sr['h3_smoke_acceptance_basis']==causal_basis
            and not sr.get('h3_m1_technical_gate',{}).get('passed',False)
            and sr['instrumented_brax']['instrumented_sha256']==H3_CAUSAL_BRAX_SHA,
            'SMOKE_CONTRACT')
    audit=sr['h3_m1_training_transition_audit']
    require(audit['passed'] and h3_audit_counts(sr['coverage']['counts'])==audit,
            'SMOKE_AUDIT')
    require(old['status']=='M463_COVERAGE_LONGTRAIN_PASS'
            and old['runner_sha256']==H3_OLD10M_RUNNER_SHA
            and old['instrumented_brax']['instrumented_sha256']==H3_OLD10M_BRAX_SHA
            and old['experiment_design']=='INDEPENDENT_C401M_SINGLE_PROCESS'
            and old['independent_budget']=='10M' and old['rollouts_per_epoch']==8
            and old['cumulative_steps_start']==0 and old['resume_fullstate_sha256'] is None
            and old['trained_from']=='C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE'
            and old['reward_PD_termination_source_unchanged'] is True
            and old['config_fingerprint']==H3_FP, 'OLD10M_CONTRACT')
    require([x['additional_steps'] for x in old['report_history']]==list(range(0,10485761,2097152))
            and [x['steps'] for x in old['checkpoints']]==list(range(2097152,10485761,2097152)),
            'OLD10M_HISTORY')
    ck=old['checkpoints'][-1]
    require(ck['sha256']==H3_OLD10M_POLICY_SHA
            and ck['file']=='m463_easy_repeat_support_longtrain_step_010485760.msgpack',
            'OLD10M_FINAL_METADATA')
    policy=args.h3_old10m_report.parent/ck['file']
    generated=args.h3_old10m_report.parent/'m463_brax_ppo_instrumented.py'
    for p,sha in ((policy,H3_OLD10M_POLICY_SHA),(generated,H3_OLD10M_BRAX_SHA)):
        require(p.is_file() and not p.is_symlink() and digest(p)==sha,'OLD10M_ARTIFACT_'+p.name)
    return {**causal_basis,
        'basis':'V1_2_67_SMOKE_TECH_PASS_OLD10M_SOURCE_COMPARATOR',
        'smoke_report_sha256':H3_SMOKE_SHA,'old10m_report_sha256':H3_OLD10M_SHA,
        'old10m_policy_sha256':H3_OLD10M_POLICY_SHA,
        'old10m_original_generated_brax_sha256':H3_OLD10M_BRAX_SHA,
        'ppo_steps_authorized':10485760,'optimizer_updates_target':1600,
        'source_comparison_not_bitwise_runtime_proof':True}


def verify_h3_longtrain_brax(api, instrument_meta):
    """Prove generated-code equivalence except the reviewed telemetry additions."""
    import inspect
    import subprocess
    from m463_instrument_brax import EXT_FIELDS,H3_EXT_FIELDS,H3_METRICS_SOURCE
    raw=Path(inspect.getfile(api['ppo'].train)).read_bytes()
    def require(ok,label):
        if not ok:
            raise RuntimeError('M470_H3_10M_BRAX_'+label)
    def historical(name):
        return subprocess.check_output(['git','show',H3_OLD10M_COMMIT+':'+name],
                                      cwd=Path(__file__).resolve().parent)
    require(hashlib.sha256(historical('m463_push_ppo_train.py')).hexdigest()==H3_OLD10M_RUNNER_SHA,
            'HISTORICAL_RUNNER')
    ns={'__name__':'h3_old10m_instrumenter'}
    exec(compile(historical('m463_instrument_brax.py'),'<OLD10M instrumenter>','exec'),ns)
    legacy,sha=ns['make_patched_source'](raw,enable_resume=True)
    require(sha==H3_CAUSAL_ORIGINAL_BRAX_SHA and
            hashlib.sha256(legacy.encode('utf8')).hexdigest()==H3_OLD10M_BRAX_SHA,
            'HISTORICAL_GENERATED')
    p=Path(instrument_meta['instrumented_file'])
    require(digest(p)==instrument_meta['instrumented_sha256'],'CURRENT_GENERATED_SHA')
    generated=p.read_text('utf8')
    current_fields=EXT_FIELDS+H3_EXT_FIELDS
    field_text=lambda fields: "extra_fields=('truncation', "+', '.join(repr(k) for k in fields)+'),'
    require(generated.count(field_text(current_fields))==1,'EXTRAS')
    normalized=generated.replace(field_text(current_fields),field_text(ns['EXT_FIELDS']),1)
    require(normalized.count(H3_METRICS_SOURCE)==1,'H3_TELEMETRY')
    normalized=normalized.replace(H3_METRICS_SOURCE,'',1)
    start='    # H1-M1: the transitions consumed by PPO, not the two-env eval reward.'
    end='    metrics = {**metrics, **coverage, **m464_tail}'
    require(normalized.count(start)==1 and normalized.count(end)==1,'H1_TELEMETRY')
    lo,hi=normalized.index(start),normalized.index(end)
    require(lo<hi,'TELEMETRY_ORDER')
    normalized=normalized[:lo]+normalized[hi:]
    require(normalized==legacy,'NONTELEMETRY_SOURCE_DIFF')
    return {'old10m_generated_sha256':H3_OLD10M_BRAX_SHA,
            'current_generated_sha256':instrument_meta['instrumented_sha256'],
            'normalized_generated_matches_old10m':True,
            'bitwise_runtime_equivalence_claimed':False}


def h3_audit_counts(counts):
    vals = {k: float(counts[PREFIX+k]) for k in H3_METRIC_KEYS}
    if not all(math.isfinite(v) for v in vals.values()):
        raise RuntimeError('M470_H3_NONFINITE_AUDIT')
    checks = {k: vals[k] == 0 for k in H3_BAD_KEYS}
    checks.update({k: vals[k] > 0 for k in H3_POS_KEYS})
    checks['h3_signed_sums'] = (vals['h3_base_sum_pre_dt'] < 0
        and vals['h3_base_sum_pre_dt'] < vals['h3_modified_sum_pre_dt'] <= 0)
    checks['h3_terminal_count_nonnegative'] = vals['h3_terminal_steps_per_rollout'] >= 0
    checks['h3_post_active_le_active'] = vals['h3_post_active_per_rollout'] <= vals['h3_active_per_rollout']
    checks['h3_post_active_le_post_steps'] = vals['h3_post_active_per_rollout'] <= vals['h3_stand_post_steps_per_rollout']
    # Float32 reductions across many samples: descriptive aggregate reconciliation.
    predicted = 0.02*(vals['h3_modified_sum_pre_dt']-vals['h3_base_sum_pre_dt'])
    observed = vals['h3_delta_sum_dt']
    checks['h3_delta_sum_reconciles'] = abs(predicted-observed) <= max(0.01, 1e-4*abs(predicted))
    return {'counts': vals, 'checks': checks, 'passed': all(checks.values())}


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



def m470_fingerprint(scaled_config):
    """Hash all training semantics, excluding chunk size and eval cadence."""
    fixed = copy.deepcopy(scaled_config)
    fixed.pop('num_timesteps', None)
    fixed.pop('num_evals', None)
    return hashlib.sha256(json.dumps(
        fixed, sort_keys=True, separators=(',', ':'), default=str
    ).encode('utf8')).hexdigest()


def m470_resume_manifest(path: Path, fingerprint, instrument_sha, batches_per_eval):
    if path.is_symlink() or not path.is_file() or path.suffix != '.msgpack':
        raise RuntimeError('M470_RESUME_PATH_INVALID')
    manifest_path = path.with_suffix('.json')
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError('M470_RESUME_MANIFEST_MISSING')
    meta = json.loads(manifest_path.read_text('utf8'))
    if meta.get('h4_arm') is not None or meta.get('h4_resume_forbidden',False):
        raise RuntimeError('M470_H4_RESUME_FORBIDDEN')
    if meta.get('h3_m1_enabled', False):
        raise RuntimeError('M470_H3_RESUME_FORBIDDEN')
    expected = {
        'format': 'M470_PPO_FULLSTATE_RESET_EDGE_V1',
        'stage': 'easy_repeat_support',
        'source_checkpoint_sha256': SEED1_SHA,
        'runner_sha256': digest(Path(__file__)),
        'config_fingerprint': fingerprint,
        'instrumented_brax_sha256': instrument_sha,
        'rollouts_per_epoch': batches_per_eval,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError('M470_RESUME_MANIFEST_MISMATCH '+key)
    if meta.get('fullstate_sha256') != digest(path):
        raise RuntimeError('M470_RESUME_FULLSTATE_SHA_MISMATCH')
    steps = meta.get('cumulative_steps')
    if not isinstance(steps, int) or steps <= 0 or steps % TOTAL_STEPS_BATCH:
        raise RuntimeError('M470_RESUME_STEPS_INVALID')
    return meta


def h3_technical_gate(api, train_cfg, stage, instrument_meta, result, persist):
    """Causal diagnostic ONLY; never authorizes PPO."""
    import inspect
    import subprocess
    import textwrap
    from types import SimpleNamespace
    from brax.training import acting
    import jax.numpy as jp
    from m463_coverage_env import CoverageJoystick
    from m463_instrument_brax import (
        EXT_FIELDS, H3_EXT_FIELDS, H3_METRICS_SOURCE, make_patched_source,
    )
    jax, np = api['jax'], api['numpy']

    def require(ok, label):
        if not bool(ok):
            raise RuntimeError('M470_H3_TECH_' + label)

    require(jax.default_backend() == 'gpu' and jax.process_count() == 1
            and jax.local_device_count() == 1, 'GPU_SINGLE_PROCESS')
    raw = Path(inspect.getfile(api['ppo'].train)).read_bytes()
    old = {'__name__': 'h3_legacy_instrumenter'}
    exec(compile(subprocess.check_output([
        'git', 'show', 'defe4268ee96a39064b41fa72c8f0b2597f6e1a1:m463_instrument_brax.py'
    ]), '<legacy instrumenter>', 'exec'), old)
    for resume in (False, True):
        legacy, sha = old['make_patched_source'](raw, enable_resume=resume)
        current, sha2 = make_patched_source(raw, enable_resume=resume, enable_h3=False)
        require(legacy == current and sha == sha2, 'LEGACY_BYTES')
        generated, _ = make_patched_source(raw, enable_resume=resume, enable_h3=True)
        compile(generated, '<actual H3 Brax source>', 'exec')
        require(generated.count("'m463_h3_active_per_rollout'") == 1, 'GENERATED_METRICS')
        if not resume:
            require(Path(instrument_meta['instrumented_file']).read_text('utf8') == generated,
                    'ACTUAL_INSTRUMENTED_FILE')
    print('M470_H3_TECH_LEGACY_BYTES_PASS', flush=True)

    from mujoco_playground._src import mjx_env
    from m463_coverage_env import joystick, geoms_colliding
    labels = ('prepush', 'first', 'post', 'repeat', 'moving', 'terminal',
              'boundary', 'nominal', 'truncation')
    expected_n = np.array([0,100,99,100,100,100,0,0,0])
    expected_event = np.array([0,1,0,1,1,1,0,0,0])
    expected_stand = np.array([1,1,1,1,0,1,1,1,1])
    expected_done = np.array([0,0,0,0,0,1,0,0,0])
    ncase = len(labels)
    keys = jax.random.split(jax.random.PRNGKey(913701), ncase)
    action = jp.zeros((ncase,29), dtype=jp.float32)
    scales = train_cfg.to_dict()['reward_config']['scales']
    slots = tuple(sorted(scales))
    require(len(slots)==24 and scales['stand_still']==-1.0, 'SLOT_SCHEMA')
    stand_index = slots.index('stand_still')
    diag_fields = ('m470_diag_base_raw','m470_diag_h3_raw','m470_diag_base_total',
                   'm470_diag_h3_total','m470_diag_inputs_unchanged',
                   'm470_diag_command','m470_diag_contact','m470_diag_first_contact',
                   'm470_diag_done')

    def host(x):
        return np.asarray(jax.device_get(x))

    def exact(a,b,name):
        require(jax.tree_util.tree_structure(a)==jax.tree_util.tree_structure(b),name+'_TREE')
        for (path,x),(_,y) in zip(jax.tree_util.tree_flatten_with_path(a)[0],
                                 jax.tree_util.tree_flatten_with_path(b)[0]):
            np.testing.assert_array_equal(host(x),host(y),
                err_msg=name+' '+jax.tree_util.keystr(path))

    def finite_tree(x,name):
        for path,v in jax.tree_util.tree_flatten_with_path(x)[0]:
            arr=host(v)
            require(arr.dtype.kind in 'biuf' and np.all(np.isfinite(arr)),
                    'NONFINITE_'+name+' '+jax.tree_util.keystr(path))

    # Check physical quantities used by the actual policy, rewards and dynamics.
    # Other MJX leaves (including possible padding/sentinels) are reported by
    # the exact comparison, not asserted to be physical measurements.
    critical = ('qpos','qvel','qacc','ctrl','actuator_force','sensordata',
                'xpos','xquat','xmat','site_xpos','site_xmat')
    def finite_state(st,name):
        finite_tree({k:getattr(st.data,k) for k in critical},name+'_DATA')
        finite_tree(st.obs,name+'_OBS')
        finite_tree(st.reward,name+'_REWARD')
        finite_tree(st.done,name+'_DONE')
        finite_tree(st.metrics,name+'_METRICS')
        finite_tree({k:v for k,v in st.info.items() if k!='first_state'},name+'_INFO')
        require(set(st.obs)=={'state','privileged_state'} and
                st.obs['state'].shape==(ncase,103),name+'_OBS_SCHEMA')

    def make(enabled,cls=CoverageJoystick):
        return cls(stage=stage,config=copy.deepcopy(train_cfg),h3_m1=enabled)

    fixtures = {
        'steps': [10,24,25,40,24,24,0,10,999],
        'm463_age': [10,24,25,40,24,24,0,10,999],
        'm463_next': [24,0,9999,0,0,0,0,0,9999],
        'm463_push_count': [0,0,2,1,0,0,1,0,2],
        'm463_recent': [0,0,100,65,0,0,88,0,0],
        'm463_no_push': [0,0,0,0,0,0,0,1,0],
        'm463_direction': [0]*ncase,
        'm463_phase_target': [0]*ncase,
        'm463_last_event_age': [-1,-1,24,24,-1,-1,5,-1,24],
        'step': [10,24,25,40,24,24,0,10,499],
    }

    def prepare(env,state,model,axes):
        # All nine diagnostic poses come from the source knees_bent keyframe,
        # with a fixed 0.02 rad first-joint offset. Rebuild data via actual
        # mjx_env.init/forward under the exact model (including each DR model).
        # No stale FK/contact/sensor values from editing only qpos.
        def one(m,st,index):
            info=dict(st.info)
            for k,values in fixtures.items():
                dtype=info[k].dtype if k in info else jp.int32
                info[k]=jp.asarray(values,dtype=dtype)[index]
            info['command']=jp.array([jp.where(index==4,.6,0.),0.,0.],dtype=jp.float32)
            q=env._init_q.at[7].add(.02)
            fallen=q.at[2].set(1.5).at[3:7].set(jp.array([0.,1.,0.,0.],dtype=q.dtype))
            q=jp.where(index==5,fallen,q)
            data=mjx_env.init(m,qpos=q,qvel=jp.zeros_like(st.data.qvel),ctrl=q[7:])
            contact=jp.array([geoms_colliding(data,g,env._floor_geom_id)
                              for g in env._feet_geom_id])
            info['last_contact']=contact
            old_model=env._mjx_model
            try:
                env._mjx_model=m
                obs=env._get_obs(data,info,contact)
                pre_done=env._get_termination(data)
            finally:
                env._mjx_model=old_model
            return st.replace(data=data,obs=obs,info=info),pre_done
        st,pre_done=jax.jit(jax.vmap(one,in_axes=(axes,0,0)))(
            model,state,jp.arange(ncase,dtype=jp.int32))
        np.testing.assert_array_equal(host(pre_done),expected_done,
                                      err_msg='COHERENT_FIXTURE_PRE_DONE')
        finite_state(st,'FIXTURE')
        return st

    def fork(st,enabled):
        return st.replace(info={k:v for k,v in st.info.items()
                                if enabled or k not in H3_EXT_FIELDS},
                          metrics=dict(st.metrics),obs=dict(st.obs))

    def comparison(x,y,cross):
        # Exact equality, not a new acceptance tolerance. Store every differing
        # leaf and its per-case finite differences (mixed units remain separate).
        def common(st):
            return {'data':st.data,'obs':st.obs,'done':st.done,
                    'info':{k:v for k,v in st.info.items()
                            if not cross or k not in H3_EXT_FIELDS},
                    'metrics':{k:v for k,v in st.metrics.items()
                               if not cross or k!='reward/stand_still'},
                    **({} if cross else {'reward':st.reward})}
        u,v=common(x),common(y)
        require(jax.tree_util.tree_structure(u)==jax.tree_util.tree_structure(v),
                'COMPARE_TREE')
        differences={}
        for (path,a),(_,b) in zip(jax.tree_util.tree_flatten_with_path(u)[0],
                                 jax.tree_util.tree_flatten_with_path(v)[0]):
            a,b=host(a),host(b)
            require(a.shape==b.shape and a.dtype==b.dtype,'COMPARE_SHAPE_DTYPE')
            if a.dtype.kind not in 'biuf':
                require(np.array_equal(a,b),'COMPARE_NONNUMERIC')
                continue
            equal=np.equal(a,b)
            nonfinite=~(np.isfinite(a)&np.isfinite(b))
            if bool(np.all(equal)) and not bool(np.any(nonfinite)):
                continue
            batched=a.ndim>0 and a.shape[0]==ncase
            rows={}
            for i,name in enumerate(labels if batched else ('global',)):
                aa,bb,ee,nf=(a[i],b[i],equal[i],nonfinite[i]) if batched else (a,b,equal,nonfinite)
                aa,bb=np.asarray(aa),np.asarray(bb)
                valid=~np.asarray(nf)
                av,bv=aa[valid].astype(np.float64),bb[valid].astype(np.float64)
                delta=np.abs(av-bv)
                rows[name]={'elements':int(aa.size),'mismatch':int(np.sum(~np.asarray(ee))),
                    'nonfinite_either':int(np.sum(nf)),
                    'max_abs_finite':float(np.max(delta)) if delta.size else None,
                    'max_rel_finite':float(np.max(delta/np.maximum(np.abs(av),1e-30))) if delta.size else None}
            differences[jax.tree_util.keystr(path)]=rows
        return {'exact_equal':not any(row['mismatch'] for rows in differences.values()
                                     for row in rows.values()),
                'different_or_nonfinite_leaves':differences,
                'relative_denominator_floor':1e-30}

    off,on=make(False),make(True)
    require(off._config.to_dict()==on._config.to_dict(),'DIRECT_CONFIG')
    exact(off._default_pose,on._default_pose,'DIRECT_DEFAULT_POSE')
    # Diagnostic direct arms share the actual same MJX model object.
    model=on.mjx_model
    off._mjx_model=model
    shared=prepare(on,jax.jit(jax.vmap(on.reset))(keys),model,None)
    require(set(H3_EXT_FIELDS).issubset(shared.info),'H3_RESET_SCHEMA')
    a,b=fork(shared,False),fork(shared,True)
    require(set(b.info)-set(a.info)==set(H3_EXT_FIELDS),'FORK_SCHEMA')
    exact(a.data,b.data,'INITIAL_DATA')
    exact(a.obs,b.obs,'INITIAL_OBS')
    exact(a.reward,b.reward,'INITIAL_REWARD')
    exact(a.done,b.done,'INITIAL_DONE')
    exact(a.metrics,b.metrics,'INITIAL_METRICS')
    exact(a.info,{k:b.info[k] for k in a.info},'INITIAL_INFO')
    result['shared_snapshot_passed']=True
    persist()
    print('M470_H3_DIAG_SHARED_SNAPSHOT_PASS',flush=True)

    # Separate jitted call sites for the two same-treatment controls and
    # cross-treatment pair. No cross-executable mismatch is silently waived.
    def direct(env,enabled):
        fn=jax.jit(jax.vmap(lambda st,act: env.step(st,act)))
        out=fn(fork(shared,enabled),action)
        finite_state(out,'DIRECT')
        return out
    a1=direct(off,False)
    a2=direct(off,False)
    b1=direct(on,True)
    b2=direct(on,True)
    result['numerical_controls']={
        'off_off':comparison(a1,a2,False),
        'on_on':comparison(b1,b2,False),
        'off_on':comparison(a1,b1,True),
    }
    result['direct_control_transitions']=4*ncase
    result['cross_executable_exact_equal']=result['numerical_controls']['off_on']['exact_equal']
    result['direct_done']={k:host(v.done).tolist() for k,v in
                           [('off1',a1),('off2',a2),('on1',b1),('on2',b2)]}
    result['direct_support_side']={k:host(v.info['m463_support_side']).tolist() for k,v in
                                   [('off1',a1),('off2',a2),('on1',b1),('on2',b2)]}
    persist()
    print('M470_H3_DIAG_NUMERICAL_CONTROLS_RECORDED',
          {k:v['exact_equal'] for k,v in result['numerical_controls'].items()},flush=True)

    # B: capture both ACTUAL reward functions at the same reward-call boundary.
    # This local diagnostic subclass does not enter production PPO/wrapper C.
    class RewardAuditJoystick(CoverageJoystick):
        def _get_reward(self,data,action,info,metrics,done,first_contact,contact):
            bi,hi,bm,hm=dict(info),dict(info),dict(metrics),dict(metrics)
            base=joystick.Joystick._get_reward(self,data,action,bi,bm,done,first_contact,contact)
            shaped=CoverageJoystick._get_reward(self,data,action,hi,hm,done,first_contact,contact)
            if set(base)!=set(slots) or set(shaped)!=set(slots):
                raise RuntimeError('M470_H3_DIAG_REWARD_KEYS')
            def unchanged(before,after):
                if jax.tree_util.tree_structure(before)!=jax.tree_util.tree_structure(after):
                    return jp.asarray(False)
                return jp.all(jp.stack([jp.array_equal(x,y) for x,y in zip(
                    jax.tree_util.tree_leaves(before),jax.tree_util.tree_leaves(after))]))
            same=unchanged(info,bi)&unchanged(info,hi)&unchanged(metrics,bm)&unchanged(metrics,hm)
            info.update(m470_diag_base_raw=jp.stack([base[k] for k in slots]),
                        m470_diag_h3_raw=jp.stack([shaped[k] for k in slots]),
                        m470_diag_base_total=sum(base[k]*scales[k] for k in base)*self.dt,
                        m470_diag_h3_total=sum(shaped[k]*scales[k] for k in shaped)*self.dt,
                        m470_diag_inputs_unchanged=same,
                        m470_diag_command=info['command'],
                        m470_diag_contact=contact,
                        m470_diag_first_contact=first_contact,
                        m470_diag_done=done)
            return shaped

    def expected_context(info):
        for k,expected in [('m470_h3_n',expected_n),('m463_event',expected_event),
                           ('m470_h3_stand',expected_stand),('m470_h3_done',expected_done)]:
            np.testing.assert_array_equal(host(info[k]),expected,err_msg=k)
        require(host(info['m463_repeat'])[3]==1,'REPEAT_EVENT')

    audit_env=make(True,RewardAuditJoystick)
    audit_env._mjx_model=model
    captured=jax.jit(jax.vmap(audit_env.step))(fork(shared,True),action)
    finite_state(captured,'CAPTURED')
    ci=captured.info
    require(set(ci)-set(b1.info)==set(diag_fields),'CAPTURE_SCHEMA')
    expected_context(ci)
    np.testing.assert_array_equal(host(captured.done),expected_done)
    np.testing.assert_array_equal(host(ci['m470_diag_done']),expected_done)
    np.testing.assert_array_equal(host(ci['m470_diag_inputs_unchanged']),np.ones(ncase,bool))
    np.testing.assert_array_equal(np.linalg.norm(host(ci['m470_diag_command']),axis=1)<.01,
                                  expected_stand.astype(bool))
    raw,shaped=host(ci['m470_diag_base_raw']),host(ci['m470_diag_h3_raw'])
    require(raw.shape==shaped.shape==(ncase,24),'CAPTURE_REWARD_SHAPE')
    other=[i for i in range(24) if i!=stand_index]
    np.testing.assert_array_equal(raw[:,other],shaped[:,other],
                                  err_msg='SAME_POSTPHYSICS_UNCHANGED23')
    base=-raw[:,stand_index]
    weighted=-shaped[:,stand_index]
    g=np.where(expected_done,1.,1.-.5*expected_n/100)
    np.testing.assert_allclose(weighted,g*base,rtol=2e-6,atol=2e-6)
    np.testing.assert_array_equal(weighted[[0,4,5,6,7,8]],base[[0,4,5,6,7,8]])
    require(np.all(weighted[[1,2,3]]>base[[1,2,3]]),'POSITIVE_ACTIVATION')
    np.testing.assert_allclose(host(ci['m470_h3_base']),base,rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(host(ci['m470_h3_weighted']),weighted,rtol=2e-6,atol=2e-6)
    for i,k in enumerate(slots):
        np.testing.assert_allclose(host(captured.metrics['reward/'+k]),
            shaped[:,i]*np.asarray(scales[k],dtype=shaped.dtype),rtol=2e-6,atol=2e-6,
            err_msg='ACTUAL_WEIGHTED_'+k)
    bt,ht=host(ci['m470_diag_base_total']),host(ci['m470_diag_h3_total'])
    np.testing.assert_allclose(ht-bt,.02*(weighted-base),rtol=2e-6,atol=1e-5)
    np.testing.assert_allclose(host(captured.reward),ht,rtol=0,atol=1e-5)
    result['same_postphysics_reward']={'passed':True,'slots':list(slots),
        'base_raw':raw.tolist(),'h3_raw':shaped.tolist(),
        'base_total':bt.tolist(),'h3_total':ht.tolist(),
        'command':host(ci['m470_diag_command']).tolist(),
        'contact':host(ci['m470_diag_contact']).tolist(),
        'first_contact':host(ci['m470_diag_first_contact']).tolist(),
        'transitions':ncase,'unchanged_23_exact':True,'input_dicts_unchanged':True}
    persist()
    print('M470_H3_DIAG_SAME_POSTPHYSICS_REWARD_PASS',flush=True)

    # C: actual production H3 class (not the audit subclass), actual DR and
    # Episode/AutoReset wrapper, actual actor_step and reducer. Use each
    # vectorized randomized model to construct coherent fixtures.
    wrap_env=make(True)
    wrapped=api['wrapper'].wrap_for_brax_training(
        wrap_env,episode_length=1000,action_repeat=1,
        randomization_fn=functools.partial(api['randomizer'],
            rng=jax.random.split(jax.random.PRNGKey(913702),ncase)))
    dr=wrapped.env.env
    require(isinstance(dr,api['wrapper'].BraxDomainRandomizationVmapWrapper)
            and dr.env is wrap_env,'WRAPPED_DR_STRUCTURE')
    ws=prepare(on,jax.jit(wrapped.reset)(keys),dr._mjx_model_v,dr._in_axes)
    def policy(obs,key):
        return action,{}
    def collect(st):
        return acting.actor_step(wrapped,st,policy,jax.random.PRNGKey(913703),
                                 extra_fields=('truncation',)+EXT_FIELDS+H3_EXT_FIELDS)
    wn,transition=jax.jit(collect)(ws)
    finite_state(wn,'WRAPPED')
    finite_tree(transition,'TRANSITION')
    extras=transition.extras['state_extras']
    require(transition.reward.shape==(ncase,),'WRAPPED_REWARD_SHAPE')
    require(set(extras)==set(('truncation',)+EXT_FIELDS+H3_EXT_FIELDS),'WRAPPED_EXTRA_SCHEMA')
    for k in ('truncation',)+EXT_FIELDS+H3_EXT_FIELDS:
        require(extras[k].shape==transition.reward.shape,'EXTRA_SHAPE_'+k)
        exact(extras[k],wn.info[k],'WRAPPED_EXTRA_'+k)
    expected_context(extras)
    np.testing.assert_array_equal(host(extras['truncation']),[0,0,0,0,0,0,0,0,1])
    np.testing.assert_array_equal(host(wn.done),[0,0,0,0,0,1,0,0,1])
    np.testing.assert_array_equal(host(transition.discount),1-host(wn.done))
    exact(transition.observation,ws.obs,'ACTOR_INPUT_OBS')
    exact(transition.action,action,'FIXED_ACTION')
    exact(transition.next_observation,wn.obs,'NEXT_OBS')
    # AutoReset is not recovery; explicitly check terminal/truncated reset qpos.
    np.testing.assert_array_equal(host(wn.data.qpos)[[5,8]],
                                  host(ws.info['first_state'].qpos)[[5,8]])
    np.testing.assert_allclose(host(transition.reward),
        host(extras['m470_h3_reward_sum']),rtol=0,atol=1e-5)
    coverage={}
    exec(compile(textwrap.dedent(H3_METRICS_SOURCE),'<actual H3 reducer>','exec'),{
        'extras':extras,'data':transition,'jnp':jp,'coverage':coverage,
        'recent':extras['m463_recent']})
    counts={'training/'+k:float(host(v)) for k,v in coverage.items()}
    audit=h3_audit_counts(counts)
    result['wrapped_validation']={'passed':audit['passed'],'transitions':ncase,
        'audit':audit,'actual_extras':{k:host(v).tolist() for k,v in extras.items()},
        'reward':host(transition.reward).tolist(),'done':host(wn.done).tolist(),
        'autoreset_verified_not_recovery':True}
    persist()
    require(audit['passed'],'WRAPPED_AUDIT')
    result.update(diagnostic_complete=True,diagnostic_control_transitions=6*ncase,
                  passed=False,training_authorized=False)
    persist()
    print('M470_H3_DIAG_WRAPPED_PASS',flush=True)
    print('M470_H3_DIAGNOSTIC_COMPLETE_PPO_HOLD',flush=True)
    return result


def run(args):
    if getattr(args,'h4m6_arm',None) is not None:
        return h4m6_run(args)
    if getattr(args,'h4cc_arm',None) is not None:
        return h4cc_run(args)
    if getattr(args,'h4_exploratory',False):
        return h4x_run(args)
    h4x_validate_cli(args)
    return _run_impl(args)


def _run_impl(args,h4x=None):
    m6_arm=getattr(args,'h4m6_arm',None)
    if m6_arm is not None:
        h4m6_train_gate(m6_arm)
    cc_arm='ref' if m6_arm is not None else getattr(args,'h4cc_arm',None)
    if cc_arm is not None or getattr(args,'h4_exploratory',False):
        auth=(h4m6_authorization(phase=m6_arm) if m6_arm is not None else
              h4cc_validate_cli(args) if cc_arm is not None else h4x_validate_cli(args))
        if m6_arm is not None:
            h4m6_require(h4x is not None and h4x['m6_evidence']==h4m6_train_inputs(args,auth),'RUN_EVIDENCE')
        h4x_require(h4x is not None and h4x['authorization']==auth,'RUN_RESERVATION')
        h4x_require(json.loads((args.output_dir/'attempt.json').read_bytes())==
                    {'authorization':h4x['authorization'],'reserved':True},'RUN_RESERVATION')
    else:
        h4x_require(h4x is None,'UNEXPECTED_RESERVATION')
    validate_h4c_cli(args)
    if getattr(args,'h4_causal',False):
        return h4c_run(args)
    validate_h4r_cli(args)
    if getattr(args,'h4_readiness',False):
        return h4r_run(args)
    if cc_arm is None:
        validate_h4_cli(args)
    h4_arm='candidate' if cc_arm is not None else getattr(args,'h4_arm',None)
    if h4_arm is not None:
        if (not H4_EXECUTION_AUTHORIZED and h4x is None) or args.mode!='longtrain':
            raise RuntimeError('M470_H4_EXECUTION_HOLD')
        args.rollouts,args.num_evals=40,3
    validate_h3_cli(args)
    h3_m1 = bool(getattr(args, 'h3_m1', False))
    h3_diagnostic = bool(getattr(args, 'h3_diagnostic_only', False))
    h3_acceptance = read_h3_causal_report(args.h3_causal_report) if h3_m1 and not h3_diagnostic else None
    if h3_acceptance is not None:
        h3_acceptance=read_h3_independent_evidence(args,h3_acceptance)
        import shutil
        if digest(args.restore_report) != H3_CAUSAL_RESTORE_SHA:
            raise RuntimeError('M470_H3_CAUSAL_RESTORE_REPORT_SHA')
        # Operational reserve, not a reward/physics criterion.
        if shutil.disk_usage(args.output_dir.parent).free < 2*1024**3:
            raise RuntimeError('M470_H3_SMOKE_DISK_RESERVE')
    if h4_arm is not None:
        import shutil
        if shutil.disk_usage(args.output_dir.parent).free<2*1024**3:
            raise RuntimeError('M470_H4_DISK_RESERVE')
    if h4_arm=='candidate' and cc_arm is None:
        approved_sha=h4_approved_control_sha()
        if args.h4_control_report.is_symlink() or not args.h4_control_report.is_file() or digest(args.h4_control_report)!=approved_sha:
            raise RuntimeError('M470_H4_CONTROL_SHA')
    if args.output_dir.exists() and h4x is None:
        raise FileExistsError('M463_REFUSE_TO_OVERWRITE_OUTPUT_DIRECTORY')
    # Import from PRIOR, already proven checkpoint validator and GPU monitor.
    from m462_ppo_warmstart_run import (read_restore_gate, sample_gpu,
                                        GPUMonitor, check_source_and_config)
    from m463_instrument_brax import build_instrumented_ppo

    report_gate,orbax_dir=read_restore_gate(args.restore_report,args.checkpoint)
    if h4_arm is not None and (digest(args.restore_report)!=H3_CAUSAL_RESTORE_SHA or
            digest(args.checkpoint)!=SEED1_SHA or report_gate['orbax_dir_sha256']!=H4_ORBAX_SHA):
        raise RuntimeError('M470_H4_FROZEN_RESTORE_MISMATCH')
    if h3_m1 and (digest(args.checkpoint) != SEED1_SHA or
            report_gate['orbax_dir_sha256'] !=
            '5acfabc1376e91a2a9a52b47ac821e296f0e84f5545b646b07c9482ddc462bf3'):
        raise RuntimeError('M470_H3_FROZEN_RESTORE_MISMATCH')
    if args.mode in ('probe2', 'pilot'):
        ensure_smoke_gate(args.smoke_report,args.stage,report_gate['orbax_dir_sha256'])
    longtrain = args.mode == 'longtrain'
    independent_budget = getattr(args, 'independent_budget', None)
    independent = independent_budget is not None
    h1_m1 = bool(args.h1_m1)
    h1_m2 = bool(args.h1_m2)
    if h1_m1 and h1_m2:
        raise RuntimeError('M470_H1_M1_M2_MUTUALLY_EXCLUSIVE')
    if h1_m2 and (
        args.stage != 'easy_repeat_support'
        or args.mode not in ('smoke3', 'longtrain')
        or (args.mode == 'longtrain' and independent_budget != '10M')
        or args.resume_state is not None
    ):
        raise RuntimeError('M470_H1M2_MODE_NOT_APPROVED')
    if h1_m1 and (
        args.stage != 'easy_repeat_support'
        or args.mode not in ('smoke3', 'longtrain')
        or (args.mode == 'longtrain' and independent_budget != '10M')
        or args.resume_state is not None
    ):
        raise RuntimeError('M470_H1M1_MODE_NOT_APPROVED')
    if independent and (not longtrain or args.resume_state is not None):
        raise RuntimeError('M470_INDEPENDENT_BUDGET_REQUIRES_FRESH_C401M')
    if longtrain:
        if independent:
            locked = {'10M': (40, 6), '50M': (192, 25),
                      '100M': (384, 49)}
            if (args.rollouts, args.num_evals) != locked[independent_budget]:
                raise RuntimeError('M470_INDEPENDENT_BUDGET_CONFIG_MISMATCH')
        if args.stage != 'easy_repeat_support':
            raise RuntimeError('M470_LONGTRAIN_STAGE_LOCKED')
        if (args.rollouts is None or args.rollouts < 1 or
                args.num_evals is None or args.num_evals < 2 or
                args.num_evals > (49 if independent else 6) or
                args.rollouts > (384 if independent else 40) or
                args.rollouts % (args.num_evals - 1) != 0):
            raise RuntimeError('M470_LONGTRAIN_BUDGET_NOT_DIVISIBLE')
        batches = args.rollouts
    else:
        if args.rollouts is not None or args.num_evals is not None or args.resume_state is not None:
            raise RuntimeError('M470_LONGTRAIN_FLAGS_REQUIRE_LONGTRAIN_MODE')
        batches = BATCHES[args.mode]
    gpu_initial=sample_gpu()
    # This checks source pin, GPU, Brax source SHA, config, domain randomizer.
    api=check_source_and_config('A')
    jax,np=api['jax'],api['numpy']
    from flax import serialization
    from m463_coverage_env import CoverageJoystick, STAGES
    if h4x is not None:
        h4x_require(jax.__version__=='0.4.35','JAX_VERSION')
        h4x['brax_dependencies']=h4c_dependency_pins(api)
        import inspect
        from brax.envs.wrappers import training as brax_wrapper
        from m463_instrument_brax import h4c_git_blob
        wrapper_file=Path(inspect.getfile(brax_wrapper)).resolve()
        ppo_root=Path(inspect.getfile(api['ppo'].train)).resolve().parents[3]
        h4x_require(wrapper_file==ppo_root/'envs/wrappers/training.py' and
                    h4c_git_blob(wrapper_file.read_bytes())=='2c09c8268a29d14c832dd07050575e4541ee98af','EPISODE_WRAPPER_PIN')
        for name,path in h4x['pins'].items():
            module=sys.modules.get(name)
            h4x_require(module is not None and Path(module.__file__).resolve()==Path(path),'IMPORTED_ORIGIN '+name)
        h4x_require(h4x_bootstrap(args,h4cc=cc_arm is not None,**({'h4m6':auth} if m6_arm is not None else {}))==h4x['pins'],'BOOTSTRAP_IMMUTABILITY')
    if jax.default_backend()!='gpu':raise RuntimeError('M463_GPU_REQUIRED')
    if (h3_acceptance is not None or h4_arm is not None) and (jax.process_count() != 1 or jax.local_device_count() != 1
            or gpu_initial['total_mib'] != 12288
            or not 0 <= gpu_initial['used_mib'] < .95*gpu_initial['total_mib']):
        raise RuntimeError('M470_H3_SMOKE_GPU_PREFLIGHT')
    if args.stage not in STAGES:raise RuntimeError('M463_STAGE_UNKNOWN')
    source_cfg=api['source_cfg']
    train_cfg=copy.deepcopy(source_cfg)
    train_cfg.push_config.enable=False  # avoid source push + explicit push stacking
    if not bool(source_cfg.push_config.enable):
        raise RuntimeError('M463_ORIGINAL_SOURCE_PUSH_WAS_NOT_ENABLED')
    original_reward = source_cfg.to_dict()['reward_config']
    if h3_m1 and (float(original_reward['scales']['stand_still']) != -1.0
                  or float(original_reward['scales']['termination']) != -100.0
                  or len(original_reward['scales']) != 24):
        raise RuntimeError('M470_H3_SOURCE_REWARD_MISMATCH')
    if h1_m1:
        if float(original_reward['scales']['lin_vel_z']) != 0.0:
            raise RuntimeError('M470_H1M1_SOURCE_LIN_VEL_Z_NONZERO')
        expected_reward = copy.deepcopy(original_reward)
        expected_reward['scales']['lin_vel_z'] = -0.25
        train_cfg.reward_config.scales.lin_vel_z = -0.25
        if train_cfg.to_dict()['reward_config'] != expected_reward:
            raise RuntimeError('M470_H1M1_REWARD_DIFF_NOT_ISOLATED')
    elif h1_m2:
        if float(original_reward['scales']['feet_clearance']) != 0.0:
            raise RuntimeError('M470_H1M2_SOURCE_FEET_CLEARANCE_NONZERO')
        expected_reward = copy.deepcopy(original_reward)
        expected_reward['scales']['feet_clearance'] = -0.25
        train_cfg.reward_config.scales.feet_clearance = -0.25
        if train_cfg.to_dict()['reward_config'] != expected_reward:
            raise RuntimeError('M470_H1M2_REWARD_DIFF_NOT_ISOLATED')
    elif original_reward != train_cfg.to_dict()['reward_config']:
        raise RuntimeError('M463_REWARD_CHANGED')
    if h4x is not None:
        expected_reward=json.loads(args.h4_exploratory_reward_report.read_bytes())['train_reward_config']
        h4x_require(original_reward==expected_reward and train_cfg.to_dict()['reward_config']==expected_reward,'REWARD_PIN')
    train_env=CoverageJoystick(config=train_cfg,stage=args.stage,
                               h1_m1=h1_m1,h1_m2=h1_m2,h3_m1=h3_m1,h4_arm=h4_arm,h4cc_arm=cc_arm,
                               **({'h4m6_arm':m6_arm} if m6_arm is not None else {}))
    # Original source-driven randomized evaluation; only 2 envs as previous.
    eval_env=api['locomotion'].load(api['env_name'],config=source_cfg)
    scaled=api['scaled']
    scaled.num_timesteps=batches*TOTAL_STEPS_BATCH
    scaled.num_evals = args.num_evals if longtrain else (2 if args.mode in ('smoke', 'probe2', 'smoke3') else 6)
    assert scaled.num_envs==8192 and scaled.num_minibatches==8
    assert scaled.unroll_length==32 and scaled.batch_size==1024
    assert scaled.num_updates_per_batch==5 and scaled.action_repeat==1
    cfg=scaled.to_dict()
    if args.mode in ('smoke3', 'longtrain') and cfg.get('num_resets_per_eval') != 1:
        raise RuntimeError('M470_SMOKE3_REQUIRES_CONTINUOUS_ENV_STATE')
    fp = m470_fingerprint(scaled.to_dict()) if longtrain else None
    h3_fp = m470_fingerprint(scaled.to_dict()) if h3_m1 else None
    if h4_arm is not None and (fp!=H3_FP or batches!=40 or scaled.num_evals!=3 or
            cfg.get('episode_length')!=1000 or cfg.get('num_resets_per_eval')!=1):
        raise RuntimeError('M470_H4_PPO_CONFIG_MISMATCH')
    if h3_m1 and h3_fp != H3_FP:
        raise RuntimeError('M470_H3_PPO_FINGERPRINT_MISMATCH')
    if h3_acceptance is not None and (not longtrain or batches != 40
            or scaled.num_timesteps != 10485760 or scaled.num_evals != 6):
        raise RuntimeError('M470_H3_LONGTRAIN_BUDGET_MISMATCH')
    if h3_acceptance is not None:
        sr=json.loads(args.smoke_report.read_bytes())
        if original_reward!=sr['train_reward_config']:
            raise RuntimeError('M470_H3_LONGTRAIN_REWARD_CONFIG_MISMATCH')
    h3_source_digests = ({name: digest(Path(__file__).with_name(name))
        for name in ('m463_coverage_env.py', 'm463_instrument_brax.py',
                     'm463_push_ppo_train.py')} if h3_m1 else None)
    net_cfg=cfg.pop('network_factory')
    cfg.update(seed=1,randomization_fn=api['randomizer'],
               restore_checkpoint_path=str(orbax_dir.resolve()))
    out=args.output_dir
    if h4x is None:
        out.mkdir(parents=True,exist_ok=False)
    else:
        h4x_storage(out)
        h4x_require(cfg.get('num_eval_envs')==2 and train_env.action_size==29 and float(train_env.dt)==.02,'EXPLORATORY_CONFIG')
    instrumented_ppo,instrument_meta=build_instrumented_ppo(
        api['ppo'],out/'m463_brax_ppo_instrumented.py',
        enable_resume=longtrain, enable_h3=h3_m1,enable_h4=h4_arm is not None,
        enable_h4x=h4x is not None,enable_h4cc=cc_arm is not None,
        **({'enable_h4m6':True} if m6_arm is not None else {}))
    if h4x is not None:
        h4x_require(instrument_meta['instrumented_sha256']==h4x['authorization']['generated_sha256'],'GENERATED_PIN')
        h4x_storage(out)
    h4_contract=None
    h4_control=None
    if h4_arm is not None:
        h4_contract=h4_common_contract(scaled.to_dict(),original_reward,instrument_meta)
        if cc_arm is not None:
            historical_path=Path(__file__).resolve().parent/H4X_PATHS['control']/'report.json'
            h4_control=h4cc_read_historical_control(historical_path,h4_contract)
            old_report=json.loads(historical_path.read_bytes())
            h4x_require(h4x['brax_dependencies']==old_report['h4x_brax_dependencies'],'CCRC_HISTORICAL_DEPENDENCIES')
            if m6_arm=='rss':
                ref=h4x['m6_evidence']['ref']
                h4m6_require(ref['h4_common_contract']==h4_contract
                    and ref['h4x_brax_dependencies']==h4x['brax_dependencies'],'MATCHED_REF_CONTRACT')
            elif cc_arm=='ccrc':
                ref=h4cc_read_final_report(args.h4cc_ref_report,h4x['authorization']['evidence_sha256'],'ref',h4_contract,contract_arm='ccrc')
                h4x_require(ref['h4x_brax_dependencies']==h4x['brax_dependencies'],'CCRC_FRESH_DEPENDENCIES')
        elif h4_arm=='candidate':
            h4_control=h4_read_control(args.h4_control_report,h4_approved_control_sha(),h4_contract)
    if h3_acceptance is not None:
        h3_acceptance['longtrain_brax_comparison']=verify_h3_longtrain_brax(api,instrument_meta)
        print('M470_H3_OLD10M_GENERATED_SOURCE_PASS',flush=True)
    prior_step = 0
    resume_meta = None
    if longtrain:
        if jax.process_count() != 1 or jax.local_device_count() != 1:
            raise RuntimeError('M470_FULLSTATE_REQUIRES_ONE_GPU_PROCESS')
        if args.resume_state is not None:
            resume_meta = m470_resume_manifest(
                args.resume_state, fp, instrument_meta['instrumented_sha256'],
                batches // (scaled.num_evals - 1))
            prior_step = resume_meta['cumulative_steps']
            cfg['restore_checkpoint_path'] = None
            cfg['m470_resume_state_path'] = str(args.resume_state.resolve())
            cfg['m470_resume_expected_steps'] = prior_step
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
        'rollouts_target':batches,'hardware_scaled_config':scaled.to_dict(),
        'source_push_disabled_in_custom_train_env':True,
        'source_random_push_enabled_in_eval':True,
        'reward_PD_termination_source_unchanged':not (h1_m1 or h1_m2 or h3_m1),
        'PD_termination_source_unchanged':True,
        'h1_m1_enabled':h1_m1,
        'h1_m2_enabled':h1_m2,
        'h1_m2_reward_slot':('feet_clearance' if h1_m2 else None),
        'h1_m2_reward_scale':(-0.25 if h1_m2 else 0.0),
        'h1_m2_formula':(
            'xor_contact*phase_swing*approach*longitudinal_adjacency*'
            'square(predicted_lateral_order_risk);horizon=0.12s'
            if h1_m2 else None),
        'h1_m1_reward_slot':('lin_vel_z' if h1_m1 else None),
        'h1_m1_reward_scale':(-0.25 if h1_m1 else 0.0),
        'h1_m1_formula':('clip((0.75-z)/0.15,0,1)*clip((-vz-0.25)/0.75,0,1)^2'
                          if h1_m1 else None),
        'train_reward_config':train_cfg.to_dict()['reward_config'],
        'eval_source_reward_unchanged':True,
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
    if h4_arm is not None:
        report.update(h4_arm=h4_arm,h4_spec=H4_SPEC,h4_common_contract=h4_contract,
                      h4_control_report_sha256=(h4_control['sha256'] if h4_control else None),
                      h4_resume_forbidden=True,promotion_authorized=False)
    if cc_arm is not None:
        report.update(h4cc_arm=cc_arm,h4cc_spec=H4CC_SPEC,h4cc_historical_control_sha256=H4CC_HISTORICAL_CONTROL_SHA)
    if h4x is not None:
        report.update(h4x_authorization=h4x['authorization'],h4x_bootstrap=h4x['pins'],
                      h4x_brax_dependencies=h4x['brax_dependencies'],h4x_exploratory=True)
    if m6_arm is not None:
        report.update(h4m6_arm=m6_arm,h4m6_spec=H4M6_SPEC,
            h4m6_technical_sha256=auth['technical_sha256'],
            h4m6_ref_sha256=auth['ref_sha256'],
            h4m6_external_sha256=auth['external_sha256'],
            reward_PD_termination_source_unchanged=(m6_arm=='ref'))
    if h3_m1:
        report.update(h3_m1_enabled=True, h3_m1_spec=H3_SPEC,
                      h3_ppo_fingerprint=h3_fp,
                      h3_source_digests=h3_source_digests,
                      h3_smoke_report_sha256=(digest(args.smoke_report) if longtrain else None),
                      h3_static_reward_config_unchanged=True)
    if h3_acceptance is not None:
        report['h3_independent_acceptance_basis'] = h3_acceptance
        report['h3_diagnostic_only'] = False
    if h3_diagnostic:
        report.update(experiment_class='H3_CAUSAL_VALIDATION_ONLY_NO_PPO',
                      train_steps_target=0,rollouts_target=0,
                      future_smoke_config_steps=scaled.num_timesteps,
                      h3_diagnostic_only=True)
    if longtrain:
        report.update(
            independent_budget=independent_budget,
            experiment_design=('H4_MATCHED_C401M_SINGLE_PROCESS' if h4_arm is not None else
                               'INDEPENDENT_C401M_SINGLE_PROCESS' if independent else 'LEGACY_LONGTRAIN'),
            rollouts_per_epoch=batches // (scaled.num_evals - 1),
            trained_from=('C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE'
                          if resume_meta is None else 'FULL_OPTIMIZER_STATE_RESUME'),
            config_fingerprint=fp,
            cumulative_steps_start=prior_step,
            cumulative_steps_target=prior_step + scaled.num_timesteps,
            resume_fullstate_sha256=(resume_meta['fullstate_sha256']
                                      if resume_meta is not None else None),
            fullstate_checkpoints=[],
            checkpoint_semantics='optimizer+model+normalizer+env_steps+training_rng+eval_rng; deterministic reset epoch boundary',
        )
    def persist():
        if h4x is not None:
            h4x_write_json(report_path,report)
            return
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
        if h3_m1 and int(step) > 0:
            m = row['metrics']
            if any(PREFIX+k not in m for k in H3_METRIC_KEYS):
                raise RuntimeError('M470_H3_MISSING_LIVE_TELEMETRY')
            if any(float(m[PREFIX+k]) != 0 for k in H3_BAD_KEYS):
                raise RuntimeError('M470_H3_LIVE_INVARIANT_VIOLATION')
            if h3_acceptance is not None:
                live_gpu=monitor.snapshot()
                if (live_gpu['peak_used_mib'] is None or live_gpu['total_mib'] is None
                        or live_gpu['total_mib'] != 12288
                        or live_gpu['peak_used_mib'] >= .95*live_gpu['total_mib']
                        or live_gpu['sampler_errors']):
                    raise RuntimeError('M470_H3_SMOKE_LIVE_VRAM_GATE')
        print('M463_PROGRESS',args.stage,'steps',step,
              'reward',row['metrics'].get('eval/episode_reward'),
              'train_push_events_mean_batch',row['metrics'].get(PREFIX+'events_per_rollout'),flush=True)
    def save_policy(step,_make_policy,params):
        if int(step)==0:return
        name=f'm463_{args.stage}_{args.mode}_step_{int(step):09d}.msgpack'
        p=out/name
        raw=serialization.to_bytes(params)
        if h4x is not None:
            h4x_require(int(step) in H4_EDGES and int(step)==H4_EDGES[len(checkpoints)],'POLICY_EDGE')
            h4x_storage(out,len(raw),'policy')
        with p.open('xb') as f:f.write(raw)
        checkpoints.append({'steps':int(step),'sha256':hashlib.sha256(raw).hexdigest(),
                            'file':name})
        persist()
        print('M463_CHECKPOINT',int(step),name,flush=True)

    def save_fullstate(step, bundle):
        # Host callback is called only AFTER the epoch's reset and evaluation.
        if not longtrain:
            raise RuntimeError('M470_UNEXPECTED_FULLSTATE_CALLBACK')
        if step <= prior_step or step > prior_step + scaled.num_timesteps:
            raise RuntimeError('M470_FULLSTATE_STEP_OUT_OF_RANGE')
        bundle = jax.device_get(bundle)
        restored_step = int(bundle['training_state'].env_steps)
        if restored_step != int(step):
            raise RuntimeError('M470_FULLSTATE_STEP_MISMATCH')
        for leaf in jax.tree_util.tree_leaves(bundle['training_state']):
            if not np.all(np.isfinite(np.asarray(leaf))):
                raise RuntimeError('M470_NONFINITE_FULLSTATE')
        raw = serialization.to_bytes(bundle)
        roundtrip = serialization.from_bytes(bundle, raw)
        if jax.tree_util.tree_structure(roundtrip) != jax.tree_util.tree_structure(bundle):
            raise RuntimeError('M470_FULLSTATE_ROUNDTRIP_STRUCTURE')
        for original, decoded in zip(
                jax.tree_util.tree_leaves(bundle),
                jax.tree_util.tree_leaves(roundtrip)):
            np.testing.assert_array_equal(np.asarray(original), np.asarray(decoded))
        file = out / f'm470_fullstate_step_{step:09d}.msgpack'
        meta_file = file.with_suffix('.json')
        info = {
            'format':'M470_PPO_FULLSTATE_RESET_EDGE_V1',
            'stage':args.stage,
            'source_checkpoint_sha256':SEED1_SHA,
            'runner_sha256':digest(Path(__file__)),
            'config_fingerprint':fp,
            'instrumented_brax_sha256':instrument_meta['instrumented_sha256'],
            'rollouts_per_epoch':batches // (scaled.num_evals - 1),
            'cumulative_steps':int(step),
            'fullstate_sha256':hashlib.sha256(raw).hexdigest(),
            'parent_resume_sha256':(resume_meta['fullstate_sha256']
                                     if resume_meta is not None else None),
            'reset_edge':True,
            'optimizer_state_included':True,
            'training_rng_included':True,
            'evaluator_rng_included':True,
        }
        if h4_arm is not None:
            info.update(h4_arm=h4_arm,h4_resume_forbidden=True,h4_common_contract=h4_contract)
        if cc_arm is not None:
            info.update(h4cc_arm=cc_arm,h4cc_authorization=h4x['authorization'])
        if m6_arm is not None:
            info.update(h4m6_arm=m6_arm,h4m6_authorization=auth)
        if h3_m1:
            info['h3_m1_enabled'] = True
            info['h3_m1_spec'] = H3_SPEC
        if h4x is not None:
            h4x_require(step in H4_EDGES and step==H4_EDGES[len(report['fullstate_checkpoints'])],'FULLSTATE_EDGE')
            h4x_storage(out,len(raw),'fullstate')
        with file.open('xb') as stream:
            stream.write(raw)
        if h4x is not None:
            h4x_storage(out,len((json.dumps(info,indent=2,allow_nan=False)+'\n').encode()))
        with meta_file.open('x', encoding='utf8') as stream:
            json.dump(info, stream, indent=2, allow_nan=False)
            stream.write('\n')
        report['fullstate_checkpoints'].append({
            'steps':int(step), 'file':file.name,
            'sha256':info['fullstate_sha256'],
            'manifest':meta_file.name, **({'manifest_sha256':digest(meta_file)} if h4_arm is not None else {})})
        persist()
        print('M470_FULLSTATE_CHECKPOINT',int(step),file.name,
              info['fullstate_sha256'],flush=True)

    monitor=GPUMonitor()
    if h4_arm is not None:
        cfg['h4_chunk_fn']=h4_chunk_callback(jax,np,monitor,report,persist,h4_arm,h4cc_arm=cc_arm,
            **({'h4m6_arm':m6_arm} if m6_arm is not None else {}))
        if h4x is not None:
            control_dr=(H4CC_MATCHED_DR_SHA if cc_arm is not None else
                        h4x_verify_operational(json.loads(args.h4_control_report.read_bytes()))
                        if h4_arm=='candidate' else None)
            cfg['h4x_audit_fn']=h4x_audit_callback(jax,np,monitor,report,persist,args,
                                                h4x['authorization'],control_dr)
        persist()
    try:
        with monitor:
            if h3_diagnostic:
                report['h3_m1_technical_gate']={
                    'passed':False,'diagnostic_complete':False,
                    'training_authorized':False,'max_control_transitions':54,
                    'historical_exact_gate':'FAILED_85891a1',
                    'cases':['prepush','first','post','repeat','moving','terminal',
                             'boundary','nominal','truncation'],
                }
                persist()
                h3_technical_gate(api,train_cfg,args.stage,instrument_meta,
                                  report['h3_m1_technical_gate'],persist)
                tech_gpu=monitor.snapshot()
                if (tech_gpu['peak_used_mib'] is None or tech_gpu['total_mib'] is None
                        or tech_gpu['peak_used_mib'] >= .95*tech_gpu['total_mib']
                        or tech_gpu['sampler_errors']):
                    raise RuntimeError('M470_H3_TECH_VRAM_GATE')
            else:
                # Separate approved acceptance basis; historical exact gate stays FAIL.
                if h3_m1:
                    if (h3_acceptance is None or args.mode != 'longtrain'
                            or args.independent_budget != '10M'
                            or not h3_acceptance['longtrain_brax_comparison']['normalized_generated_matches_old10m']
                            or digest(args.h3_causal_report) != H3_CAUSAL_SHA
                            or digest(args.smoke_report) != H3_SMOKE_SHA
                            or digest(args.h3_old10m_report) != H3_OLD10M_SHA
                            or args.resume_state is not None):
                        raise RuntimeError('M470_H3_PPO_HOLD_CAUSAL_VALIDATION')
                    print('M470_H3_INDEPENDENT_10M_READY',json.dumps(h3_acceptance,sort_keys=True),
                          flush=True)
                if h4_arm is not None and not H4_EXECUTION_AUTHORIZED and h4x is None:
                    raise RuntimeError('M470_H4_EXECUTION_HOLD')
                if h4x is not None:
                    h4x_require(h4m6_current_auth(h4x['authorization'],cc_arm)==h4x['authorization'],'FINAL_AUTHORIZATION')
                report['training_performed']=True
                if h3_m1 or h4_arm is not None:
                    persist()
                _,params,_=instrumented_ppo.train(
                    environment=train_env,eval_env=eval_env,
                    wrap_env_fn=api['wrapper'].wrap_for_brax_training,
                    network_factory=functools.partial(api['networks'].make_ppo_networks,**net_cfg),
                    progress_fn=progress,policy_params_fn=save_policy,
                    **({'m470_state_checkpoint_fn': save_fullstate} if longtrain else {}),
                    **cfg)
        gpu=monitor.snapshot()
        report.update(gpu_peak_mib=gpu['peak_used_mib'],gpu_total_mib=gpu['total_mib'],
                      gpu_samples=gpu['samples'],gpu_monitor_errors=gpu['sampler_errors'])
        if h3_diagnostic:
            if (gpu['peak_used_mib'] is None or gpu['total_mib'] is None
                    or gpu['peak_used_mib'] >= .95*gpu['total_mib'] or gpu['sampler_errors']):
                raise RuntimeError('M470_H3_TECH_VRAM_GATE')
            report.update(status='M470_H3_DIAGNOSTIC_COMPLETE_PPO_HOLD',
                          steps_observed=0,run_steps_observed=0,
                          training_performed=False)
            persist()
            return
        expected=prior_step+scaled.num_timesteps if longtrain else scaled.num_timesteps
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
        if longtrain and (not report['fullstate_checkpoints'] or
                report['fullstate_checkpoints'][-1]['steps'] != expected):
            raise RuntimeError('M470_FINAL_FULLSTATE_CHECKPOINT_MISSING')
        batches_per_eval=batches//(scaled.num_evals-1)
        if batches_per_eval*(scaled.num_evals-1)!=batches:
            raise RuntimeError('M463_INVALID_EVAL_BATCH_COUNT')
        audit=audit_coverage(history,batches_per_eval)
        if args.mode == 'smoke3' or (longtrain and batches >= 3):
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
        if h1_m1:
            c = audit['counts']
            active = float(c.get(PREFIX+'m1_active_per_rollout',0.0))
            penalty = float(c.get(PREFIX+'m1_weighted_sum_per_rollout',0.0))
            nonfinite = float(c.get(PREFIX+'m1_nonfinite_per_rollout',0.0))
            report['h1_m1_training_transition_audit'] = {
                'active_steps':active,
                'sum_weighted_pre_dt':penalty,
                'nonfinite_steps':nonfinite,
            }
            audit['checks'].update({
                'm1_actually_active_in_training_transitions':active>0,
                'm1_negative_weighted_contribution':penalty<0,
                'm1_finite_training_transitions':nonfinite==0,
            })
            audit['passed']=all(audit['checks'].values())
        if h1_m2:
            c = audit['counts']
            active = float(c.get(PREFIX+'m2_active_per_rollout',0.0))
            penalty = float(c.get(PREFIX+'m2_weighted_sum_per_rollout',0.0))
            nonfinite = float(c.get(PREFIX+'m2_nonfinite_per_rollout',0.0))
            report['h1_m2_training_transition_audit'] = {
                'active_steps':active,
                'sum_weighted_pre_dt':penalty,
                'nonfinite_steps':nonfinite,
            }
            audit['checks'].update({
                'm2_actually_active_in_training_transitions':active>0,
                'm2_negative_weighted_contribution':penalty<0,
                'm2_finite_training_transitions':nonfinite==0,
            })
            audit['passed']=all(audit['checks'].values())
        if h3_m1:
            h3_result = h3_audit_counts(audit['counts'])
            report['h3_m1_training_transition_audit'] = h3_result
            audit['checks'].update(h3_result['checks'])
            audit['passed'] = all(audit['checks'].values())
        if h4_arm is not None:
            h4_final_checks(report,audit,h4_control)
        if h4x is not None:
            dr=h4x_verify_operational(report,h4cc=cc_arm is not None,**({'h4m6':True} if m6_arm is not None else {}))
            if cc_arm is not None:
                h4x_require(dr==H4CC_MATCHED_DR_SHA,'CCRC_FINAL_DR')
                h4x_require(h4cc_read_historical_control(historical_path,h4_contract)==h4_control,'CCRC_FINAL_HISTORICAL')
                h4cc_verify_ledger(report)
                if cc_arm=='ccrc':
                    ref=h4cc_read_final_report(args.h4cc_ref_report,h4x['authorization']['evidence_sha256'],'ref',h4_contract,contract_arm='ccrc')
                    h4x_require(h4cc_matched_nominal(ref['h4_final_counts'],report['h4_final_counts']),'CCRC_MATCHED_NOMINAL')
                    report['h4cc_exposure_decision']=h4cc_exposure_decision(ref['h4cc_final_counts'],report['h4cc_final_counts'])
            if m6_arm is not None:
                final=h4m6_replay_ledger(report,m6_arm)
                h4m6_require(final==report['h4m6_final'] and final['h4_counts']==report['h4_final_counts'],'FINAL_REPLAY')
                if m6_arm=='ref':
                    h4m6_require(final['audit']['exposure_adequate'] is True,'REF_EXPOSURE')
                else:
                    report['h4m6_exposure_comparison']=h4m6_exposure_comparison(ref['h4m6_final'],final)
                h4m6_post_train_inputs(args,auth,h4x['m6_evidence'])
            h4x_require(h4m6_current_auth(h4x['authorization'],cc_arm)==h4x['authorization'] and h4x_bootstrap(args,h4cc=cc_arm is not None,**({'h4m6':auth} if m6_arm is not None else {}))==h4x['pins']
                        and h4c_dependency_pins(api)==h4x['brax_dependencies'],'FINAL_IMMUTABILITY')
            h4x_file_pin(args.h4_exploratory_causal_report,H4X_CAUSAL_SHA)
            h4x_require(h4c_git_blob(wrapper_file.read_bytes())=='2c09c8268a29d14c832dd07050575e4541ee98af','FINAL_WRAPPER_PIN')
            read_restore_gate(args.restore_report,args.checkpoint)
            if m6_arm is not None:
                h4m6_require((args.rollouts,args.num_evals)==(40,3),'FINAL_BUDGET')
            elif cc_arm is not None:
                h4cc_validate_cli(args,final=True)
            elif h4_arm=='candidate':
                h4x_file_pin(args.h4_control_report,h4x['authorization']['evidence_sha256'])
            h4x_storage(out)
        memory_ok=(gpu['peak_used_mib'] is not None and gpu['total_mib'] is not None
                   and gpu['peak_used_mib']<gpu['total_mib']*0.95)
        report.update(steps_observed=expected,run_steps_observed=scaled.num_timesteps,
                      resume_proof_only=bool(longtrain and batches < 3),
                      final_tensor_count=n,weights_finite=True,
                      callback_vs_final_equal=True,coverage=audit,
                      coverage_gate_pass=audit['passed'],memory_gate_pass=memory_ok,
                      finished_at_unix=time.time())
        ok=audit['passed'] and memory_ok and not gpu['sampler_errors']
        report['status'] = ('M470_FULLSTATE_RESUME_PROOF_ONLY' if longtrain and batches < 3 and ok
                            else 'M463_COVERAGE_'+args.mode.upper()+'_PASS' if ok
                            else 'M463_COVERAGE_GATE_FAIL')
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


# H4-M2 distinct held contract. Requires separate owner GO and pins.
H4CC_EXECUTION_AUTHORIZED = False
H4CC_ID = 'H4-M2-CCRC-V1'
H4CC_PATHS = {a:Path('m463_runs/m470_h4_m2_'+a+'_v1') for a in ('ref','ccrc')}
H4CC_SPEC = {**H4X_SPEC,'id':H4CC_ID,'hold_ticks':250,
             'technical_cap':2016,'dev_cap':89616,'nominal_min':1867669}
H4CC_COUNTERS = ('start','event_terminal','tail_terminal','completed','refresh',
    'sample_same','sample_different','deferred','published','terminal_restored','bad',
    'qualified_1','qualified_2') + tuple(
    f'intersection_{side}_{w}' for side in (1,2) for w in (10,25,50,100,250))

def h4cc_count_key(name):
    return 'm463_h4cc_'+name+'_per_rollout'

def h4cc_epoch_audit(counts, pending, arm):
    """Pending snapshot at explicit reset is censored, not a completed tail."""
    if arm not in ('ref','ccrc'):
        raise RuntimeError('M470_H4CC_ARM')
    def c(k):
        return counts[h4cc_count_key(k)]
    def h(k):
        return counts['m463_h4_'+k+'_per_rollout']
    if any(type(c(k)) is not int or c(k)<0 for k in H4CC_COUNTERS):
        raise RuntimeError('M470_H4CC_COUNTER_SCHEMA')
    if set(pending)!= {'active','pending_valid'} or any(
            type(v) is not int or v<0 for v in pending.values()):
        raise RuntimeError('M470_H4CC_PENDING_SCHEMA')
    checks = {
        'no_bad':c('bad')==0,
        'tail_conservation':c('start')==c('event_terminal')+c('tail_terminal')+
            c('completed')+pending['active'],
        'refresh_partition':c('refresh')==c('sample_same')+c('sample_different'),
        'pending_conservation':c('deferred')==c('published')+
            c('terminal_restored')+pending['pending_valid'],
        'pending_bounds':pending['pending_valid']<=pending['active'],
        'qualified_bound':c('qualified_1')+c('qualified_2')<=c('start'),
        'arm_scope': (c('deferred')==c('published')==c('terminal_restored')==
                      pending['pending_valid']==0 if arm=='ref' else
                      c('deferred')<=c('refresh')),
    }
    for side in (1,2):
        q=c(f'qualified_{side}')
        checks[f'qualified_{side}']=q==h(f'stand_{side}_events')
        prev=q
        for w in (10,25,50,100,250):
            n=c(f'intersection_{side}_{w}')
            surv=h(f'stand_{side}_survived_2_{w}')
            executed=h(f'stand_{side}_tail_2_{w}')
            checks[f'window_{side}_{w}']=0<=n<=min(prev,surv) and surv<=executed<=q
            if arm=='ccrc':
                checks[f'exact_{side}_{w}']=n==surv
            prev=n
    return {'checks':checks,'passed':all(checks.values())}

def h4cc_chunk_ledger(jax,np,report,arm):
    """Read-only host audit at the existing five-rollout chunk boundary."""
    if arm not in ('ref','ccrc'):
        raise RuntimeError('M470_H4CC_ARM')
    chunks=[]
    epochs=[]
    epoch_cc={}
    epoch_h4={}
    all_cc={}
    all_h4={}
    all_pending={'active':0,'pending_valid':0}
    report['h4cc_chunks']=chunks
    report['h4cc_epochs']=epochs
    def add(dst,src):
        for k,v in src.items():
            dst[k]=dst.get(k,0)+v
    def callback(index,env_state,metrics,h4_counts):
        if len(chunks)>=8 or index!=len(chunks)%4+1:
            raise RuntimeError('M470_H4CC_CHUNK_SEQUENCE')
        cc={}
        for key in map(h4cc_count_key,H4CC_COUNTERS):
            arr=np.asarray(jax.device_get(metrics[key]))
            if arr.ndim!=0 or not np.isfinite(arr).all():
                raise RuntimeError('M470_H4CC_METRIC_SCHEMA '+key)
            cc[key]=h4_integer_count(arr)
        if cc[h4cc_count_key('bad')]:
            raise RuntimeError('M470_H4CC_BAD')
        add(epoch_cc,cc);add(all_cc,cc)
        add(epoch_h4,h4_counts);add(all_h4,h4_counts)
        chunks.append({'chunk':len(chunks)+1,'counts':cc})
        if index==4:
            pending={}
            for name in all_pending:
                arr=np.asarray(jax.device_get(env_state.info['h4cc_'+name]))
                if arr.size!=8192 or not np.isin(arr,(0,1)).all():
                    raise RuntimeError('M470_H4CC_PENDING_SCHEMA '+name)
                pending[name]=int(np.count_nonzero(arr))
            audit=h4cc_epoch_audit({**epoch_h4,**epoch_cc},pending,arm)
            epochs.append({'counts':dict(epoch_cc),'pending_at_reset':pending,'audit':audit})
            if not audit['passed']:
                raise RuntimeError('M470_H4CC_EPOCH_LEDGER')
            add(all_pending,pending)
            epoch_cc.clear();epoch_h4.clear()
            if len(chunks)==8:
                final=h4cc_epoch_audit({**all_h4,**all_cc},all_pending,arm)
                report['h4cc_final_counts']=dict(all_cc)
                report['h4cc_final_censored']=dict(all_pending)
                report['h4cc_final_audit']=final
                if not final['passed']:
                    raise RuntimeError('M470_H4CC_FINAL_LEDGER')
    return callback

H4CC_HISTORICAL_CONTROL_SHA = '2ab06f3cc92d546f49eb5b40551a1a8238398106afc547822d0ac2b73529a059'
H4CC_HISTORICAL_GENERATED_SHA = '8f30e5c695211b21815375c37b1bda67fcda98b2ee259a703e87b182c3402062'
H4CC_MATCHED_DR_SHA = '60a4e6999c4fbad089439fabe9de951ba063bc68db1380b46ad47cb2c2f96ebf'

def h4cc_read_historical_control(path,new_contract):
    """Separate exact archive acceptance; do not relax legacy same-source reader."""
    root=Path(__file__).resolve().parent
    path=Path(path)
    h4x_require(path.resolve()==root/H4X_PATHS['control']/'report.json',
                'CCRC_HISTORICAL_CONTROL_PATH')
    h4x_file_pin(path,H4CC_HISTORICAL_CONTROL_SHA)
    old=json.loads(path.read_bytes())
    contract=old['h4_common_contract']
    h4x_require(contract['generated_brax_sha256']==H4CC_HISTORICAL_GENERATED_SHA,
                'CCRC_HISTORICAL_GENERATED')
    # Archived source and generated bytes are validated under their own contract.
    result=h4_read_control(path,H4CC_HISTORICAL_CONTROL_SHA,contract)
    for name,expected in (
        ('attempt.json','4ea87a9695dbea17bc9e14052223f86f5126aca90a6b422e5e5959e8253c0bd2'),
        ('dispatch.started.json','a067569b1dc7f281fd4147402f3fc3d0907027f7c21a251119dbe13765b88aa1'),
        ('attempt_result.json','ceaa02e61c04dda2f35a8b4a1f020fac74d1bb9f60673ac3d2d1d2ecfae8e334')):
        h4x_file_pin(path.parent/name,expected)
    for key in ('source_commit','checkpoint_sha256','restore_report_sha256',
                'orbax_sha256','seed','config','reward','original_brax_sha256'):
        h4x_require(new_contract[key]==contract[key],'CCRC_ARCHIVE_CONTRACT '+key)
    h4x_require(h4x_verify_operational(old)==H4CC_MATCHED_DR_SHA,
                'CCRC_HISTORICAL_DR')
    h4x_require(result['counts']['m463_h4_class_0_steps_per_rollout']==2334586,
                'CCRC_HISTORICAL_NOMINAL')
    return result


def h4cc_matched_nominal(ref_counts,treatment_counts):
    k='m463_h4_class_0_steps_per_rollout'
    r,t=ref_counts[k],treatment_counts[k]
    if any(type(v) is not int or not 0<=v<=10485760 for v in (r,t)):
        raise RuntimeError('M470_H4CC_NOMINAL_SCHEMA')
    return r>=1867669 and t>=1867669 and 5*t>=4*r


def h4cc_exposure_decision(ref, treatment):
    """Only after separately accepted original and new ledger gates."""
    rows={}
    for side in (1,2):
        q=[r[h4cc_count_key(f'qualified_{side}')] for r in (ref,treatment)]
        n=[r[h4cc_count_key(f'intersection_{side}_250')] for r in (ref,treatment)]
        if any(type(v) is not int for v in q+n) or any(not 0<=i<=d for i,d in zip(n,q)):
            raise RuntimeError('M470_H4CC_INTERSECTION_SCHEMA')
        rows[str(side)]={'qualified':q,'intersection':n,
            'ceiling':not all(q) or n[0]==q[0],
            'improved':n[1]>n[0] and n[1]*q[0]>n[0]*q[1]}
    decision=('INCONCLUSIVE' if any(v['ceiling'] for v in rows.values()) else
              'PASS' if all(v['improved'] for v in rows.values()) else 'NO_GO')
    return {'decision':decision,'sides':rows,'execution_authorized':False}

def h4cc_approved_sha(rules,label):
    import re
    hits=re.findall(r'^H4CC_APPROVED_'+label+r'_SHA256: ([0-9a-f]{64})$',rules,re.M)
    h4x_require(len(hits)==1,'CCRC_APPROVED_'+label)
    return hits[0]

# The accepted REF/technical bytes are immutable. Only these two verifier
# files may differ; both current raw pins and the archived nine-file digest
# must match. Never use the archived digest as current execution identity.
H4CC_ARCHIVED_SOURCE_SHA = '4a43f092d9a9d65822cc990249d0c17a621197cad9268fc5c617481456ff225d'
H4CC_ARCHIVED_VERIFIER_FILES = {
    'm463_push_ppo_train.py':'36daf64779bc1252d521e4eb226b2cffb3563f5476d1f7f7943c41b69228c21f',
    'm463_gate3_cp_eval.py':'29de11a6c32abab1fb4e4012f885c91479e23ff6a80c188e3cfa3a5ebce8ce7a'}


def h4cc_source_identity(root):
    root=Path(root).resolve()
    rules=(root/'G1_RESEARCH_RULES.md').read_text('utf8')
    paths=(*H4_FILES,'m463_gate3_cp_eval.py','paper_config_preflight.json')
    current={}
    for name in paths:
        p=root/name
        h4x_require(p.is_file() and not p.is_symlink(),'CCRC_VERIFIER_FILE '+name)
        current[name]=digest(p)
    for name,label in (('m463_push_ppo_train.py','VERIFIER_TRAINER'),
                       ('m463_gate3_cp_eval.py','VERIFIER_EVALUATOR')):
        h4x_require(current[name]==h4cc_approved_sha(rules,label)
                    and current[name]!=H4CC_ARCHIVED_VERIFIER_FILES[name],
                    'CCRC_VERIFIER_RAW '+name)
    archive={**current,**H4CC_ARCHIVED_VERIFIER_FILES}
    def common(m):
        return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    current_sha,archived_sha=common(current),common(archive)
    h4x_require(current_sha==h4cc_approved_sha(rules,'VERIFIER_COMMON')
                and current_sha==h4x_source_digest(root)
                and archived_sha==H4CC_ARCHIVED_SOURCE_SHA and current_sha!=archived_sha,
                'CCRC_VERIFIER_COMMON')
    return {'current_sha256':current_sha,'archived_sha256':archived_sha,
            'current_files':current,'archived_files':archive}


def h4cc_contract_bridge(contract,from_arm,to_arm,identity):
    # Construct an expected comparison contract only. Never mutate a report.
    h4x_require(from_arm in ('ref','ccrc') and to_arm in ('ref','ccrc'),
                'CCRC_VERIFIER_ARM')
    maps={'ref':identity['archived_files'],'ccrc':identity['current_files']}
    expected={k:maps[from_arm][k] for k in H4_FILES}
    h4x_require(contract['source_digests']==expected,'CCRC_VERIFIER_CONTRACT_SOURCE')
    return {**contract,'source_digests':{k:maps[to_arm][k] for k in H4_FILES}}


def h4cc_matched_ref_authorization(auth,identity):
    h4x_require(auth['arm']=='ccrc' and auth['source_sha256']==identity['current_sha256'],
                'CCRC_VERIFIER_AUTH_SOURCE')
    return {**auth,'arm':'ref','output':str(H4CC_PATHS['ref']),
            'source_sha256':identity['archived_sha256'],
            'evidence_sha256':auth['technical_sha256']}


def h4cc_parse_marker(rules):
    import re
    h4x_require(not any(x.startswith(('H4X_ACTIVE_GO:','H4CC_EVAL_ACTIVE_GO:','H4CC_TECH_ACTIVE_GO:')) for x in rules.splitlines()),'CCRC_LEGACY_CONFLICT')
    hits=[x for x in rules.splitlines() if x.startswith('H4CC_ACTIVE_GO:')]
    h4x_require(len(hits)==1,'CCRC_EXECUTION_HOLD')
    m=re.fullmatch(r'H4CC_ACTIVE_GO: (H4-M2-CCRC-V1) (ref|ccrc) '
                   r'(m463_runs/[a-z0-9_]+) ([0-9a-f]{64}) ([0-9a-f]{64}) '
                   r'([0-9a-f]{64}) ([0-9a-f]{64})',hits[0])
    h4x_require(m is not None,'CCRC_AUTHORIZATION_SCHEMA')
    ident,arm,out,src,gen,tech,evidence=m.groups()
    h4x_require(out==str(H4CC_PATHS[arm]) and tech==h4cc_approved_sha(rules,'TECH_REPORT')
                and evidence==(tech if arm=='ref' else h4cc_approved_sha(rules,'REF_REPORT')),
                'CCRC_AUTHORIZATION_BINDING')
    return dict(id=ident,arm=arm,output=out,source_sha256=src,generated_sha256=gen,
                technical_sha256=tech,evidence_sha256=evidence)

def h4cc_authorization():
    # Separate Rules marker, never a broad source-flag bypass.
    h4x_require(not any((H4CC_EXECUTION_AUTHORIZED,H4_EXECUTION_AUTHORIZED,
                        H4R_EXECUTION_AUTHORIZED,H4C_EXECUTION_AUTHORIZED)),
                'CCRC_LEGACY_HOLD_REQUIRED')
    root=Path(__file__).resolve().parent
    auth=h4cc_parse_marker((root/'G1_RESEARCH_RULES.md').read_text('utf8'))
    identity=h4cc_source_identity(root)
    h4x_require(auth['arm']=='ccrc' and identity['current_sha256']==auth['source_sha256'],
                'CCRC_SOURCE_CONTRACT')
    return auth

def h4cc_generated_guard(audit,generated_file,environment):
    auth=h4cc_authorization()
    root=Path(__file__).resolve().parent
    out=root/auth['output']; generated=Path(generated_file)
    h4x_require(callable(audit) and getattr(audit,'h4x_authorization',None)==auth
                and getattr(environment,'_h4cc_arm',None)==auth['arm']
                and getattr(environment,'_h4_arm',None)=='candidate','CCRC_AUDIT_ENV')
    h4x_require(generated.resolve()==out/'m463_brax_ppo_instrumented.py'
                and not generated.is_symlink() and digest(generated)==auth['generated_sha256'],'CCRC_GENERATED_PIN')
    reservation=out/'attempt.json'
    h4x_require(reservation.is_file() and not reservation.is_symlink() and
        json.loads(reservation.read_bytes())=={'authorization':auth,'reserved':True},'CCRC_RESERVATION')
    h4x_write_json(out/'dispatch.started.json',{'authorization':auth},exclusive=True)

def h4cc_verify_ledger(report):
    arm=report['h4cc_arm']; chunks=report['h4cc_chunks']; epochs=report['h4cc_epochs']
    h4x_require(len(chunks)==8 and len(epochs)==2,'CCRC_LEDGER_LENGTH')
    total={}; pending={'active':0,'pending_valid':0}
    keys=set(map(h4cc_count_key,H4CC_COUNTERS))
    for j,ep in enumerate(epochs):
        counts={}
        for i,row in enumerate(chunks[4*j:4*j+4]):
            h4x_require(row['chunk']==4*j+i+1 and set(row['counts'])==keys
                and all(type(v) is int and 0<=v<=5*TOTAL_STEPS_BATCH for v in row['counts'].values()),
                'CCRC_CHUNK_COUNTS')
            for k,v in row['counts'].items():
                counts[k]=counts.get(k,0)+v; total[k]=total.get(k,0)+v
        q=ep['pending_at_reset']
        h4x_require(all(type(v) is int and 0<=v<=8192 for v in q.values()),'CCRC_CENSOR_COUNTS')
        check=h4cc_epoch_audit({**report['h4_epochs'][j]['counts'],**counts},q,arm)
        h4x_require(counts==ep['counts'] and check==ep['audit'] and check['passed'],'CCRC_EPOCH_REPLAY')
        for k in pending:pending[k]+=q[k]
    check=h4cc_epoch_audit({**report['h4_final_counts'],**total},pending,arm)
    h4x_require(total==report['h4cc_final_counts'] and pending==report['h4cc_final_censored']
                and check==report['h4cc_final_audit'] and check['passed'],'CCRC_FINAL_REPLAY')
    return total

def h4cc_read_final_report(path,expected_sha,arm,contract=None,*,contract_arm=None):
    root=Path(__file__).resolve().parent
    h4x_require(path is not None and arm in H4CC_PATHS,'CCRC_REPORT_REQUIRED')
    path=Path(path)
    h4x_require(path.resolve()==root/H4CC_PATHS[arm]/'report.json'
                and not path.parent.is_symlink(),'CCRC_REPORT_PATH')
    identity=h4cc_source_identity(root)
    rules=(root/'G1_RESEARCH_RULES.md').read_text('utf8')
    h4x_require(arm!='ref' or expected_sha==h4cc_approved_sha(rules,'REF_REPORT'),
                'CCRC_APPROVED_ARCHIVED_REF')
    h4x_file_pin(path,expected_sha)
    r=json.loads(path.read_bytes()); a=r['h4x_authorization']; c=r['h4_common_contract']
    h4x_require(r['status']=='M463_COVERAGE_LONGTRAIN_PASS' and r['h4cc_arm']==arm
                and r['h4cc_spec']==H4CC_SPEC and r['h4_arm']=='candidate'
                and r['h4cc_historical_control_sha256']==H4CC_HISTORICAL_CONTROL_SHA
                and r['h4_control_report_sha256']==H4CC_HISTORICAL_CONTROL_SHA
                and r['source_commit']==PIN and r['source_checkpoint_sha256']==SEED1_SHA
                and r['training_seed']==1 and r['trained_from']=='C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE'
                and r['resume_fullstate_sha256'] is None and r['cumulative_steps_start']==0
                and r['steps_observed']==r['run_steps_observed']==10485760
                and r['reward_PD_termination_source_unchanged'] is True
                and r['training_performed'] is True and r['promotion_authorized'] is False
                and r['coverage_gate_pass'] is True and r['memory_gate_pass'] is True
                and r['weights_finite'] is True and r['callback_vs_final_equal'] is True
                and r['final_tensor_count']==25 and r['gpu_total_mib']==12288
                and 0<r['gpu_peak_mib']<.95*12288 and r['gpu_monitor_errors']==[]
                and r['h4_matched_nominal_pass'] is True and r['h4_legacy_reconciled'] is True,
                'CCRC_FINAL_REPORT')
    h4x_require(a['id']==H4CC_ID and a['arm']==arm and a['output']==str(H4CC_PATHS[arm])
                and a['generated_sha256']==c['generated_brax_sha256']
                and a['source_sha256']==identity['archived_sha256' if arm=='ref' else 'current_sha256']
                and r['runner_sha256']==c['source_digests']['m463_push_ppo_train.py']
                and (contract is None or c==h4cc_contract_bridge(contract,contract_arm or arm,arm,identity))
                and c==h4cc_contract_bridge(h4_common_contract(r['hardware_scaled_config'],
                    r['train_reward_config'],r['instrumented_brax']),'ccrc',arm,identity),
                'CCRC_FINAL_SOURCE')
    h4x_require(a['technical_sha256']==h4cc_approved_sha(rules,'TECH_REPORT')
                and a['evidence_sha256']==(a['technical_sha256'] if arm=='ref' else h4cc_approved_sha(rules,'REF_REPORT')),
                'CCRC_FINAL_EVIDENCE')
    from m463_gate3_cp_eval import h4cc_tech_read_report,H4CC_TECH_OUTPUT
    h4cc_tech_read_report(root/H4CC_TECH_OUTPUT,a['technical_sha256'],
                          identity['archived_sha256'],a['generated_sha256'])
    for name,value in (('attempt.json',{'authorization':a,'reserved':True}),
                       ('dispatch.started.json',{'authorization':a}),
                       ('attempt_result.json',{'status':'COMPLETE','authorization':a})):
        p=path.parent/name
        h4x_require(p.is_file() and not p.is_symlink() and json.loads(p.read_bytes())==value,'CCRC_ATTEMPT '+name)
    h4x_file_pin(path.parent/'m463_brax_ppo_instrumented.py',a['generated_sha256'])
    h4x_require([v['steps'] for v in r['checkpoints']]==list(H4_EDGES)
                and [v['steps'] for v in r['fullstate_checkpoints']]==list(H4_EDGES),'CCRC_ARTIFACT_EDGES')
    for step,pol,full in zip(H4_EDGES,r['checkpoints'],r['fullstate_checkpoints']):
        h4x_require(pol['file']==f'm463_easy_repeat_support_longtrain_step_{step:09d}.msgpack'
                    and full['file']==f'm470_fullstate_step_{step:09d}.msgpack'
                    and full['manifest']==f'm470_fullstate_step_{step:09d}.json','CCRC_ARTIFACT_NAMES')
        h4x_file_pin(path.parent/pol['file'],pol['sha256'])
        h4x_file_pin(path.parent/full['file'],full['sha256'])
        h4x_file_pin(path.parent/full['manifest'],full['manifest_sha256'])
        m=json.loads((path.parent/full['manifest']).read_bytes())
        h4x_require(m['h4cc_arm']==arm and m['h4cc_authorization']==a
                    and m['h4_common_contract']==c and m['fullstate_sha256']==full['sha256']
                    and m['cumulative_steps']==step and m['parent_resume_sha256'] is None
                    and all(m[k] is True for k in ('reset_edge','optimizer_state_included',
                                                 'training_rng_included','evaluator_rng_included')),
                    'CCRC_MANIFEST')
    ctrl=h4cc_read_historical_control(root/H4X_PATHS['control']/'report.json',c)
    h4_final_checks(r,r['coverage'],ctrl)
    h4cc_verify_ledger(r)
    h4x_require(h4x_verify_operational(r,h4cc=True)==H4CC_MATCHED_DR_SHA,'CCRC_REPORT_DR')
    return r

def h4cc_validate_cli(a,*,final=False):
    auth=h4cc_authorization()
    root=Path(__file__).resolve().parent
    h4x_require(a.mode=='longtrain' and a.stage=='easy_repeat_support'
                and a.h4cc_arm==auth['arm'] and a.output_dir.resolve()==root/auth['output']
                and not a.output_dir.is_symlink() and not a.output_dir.parent.is_symlink(),'CCRC_MODE_ARM')
    h4x_require(not any(getattr(a,k,None) for k in ('h4_arm','h4_exploratory','h4_readiness',
                'h4_causal','h3_m1','h3_diagnostic_only','h1_m1','h1_m2','resume_state',
                'independent_budget','h4_control_report','h4_causal_failure_report',
                'h4_readiness_reward_report','h3_causal_report','h3_old10m_report',
                'smoke_report')) and ((a.rollouts,a.num_evals)==(40,3) if final else
                (a.rollouts,a.num_evals)==(None,None)), 'CCRC_CONFLICT')
    h4x_file_pin(a.checkpoint,SEED1_SHA)
    h4x_file_pin(a.restore_report,H3_CAUSAL_RESTORE_SHA)
    h4x_file_pin(a.h4_exploratory_causal_report,H4X_CAUSAL_SHA)
    prior=json.loads(a.h4_exploratory_causal_report.read_bytes())
    h4x_require(prior['status']=='M470_H4C_DIAGNOSTIC_COMPLETE'
                and prior['phase_verdicts']=={'A':'DIVERGENT','B':'PASS'}
                and prior['comparisons']['B']['bitwise_equal'] is True
                and prior['recorded_physical_transitions']==1024
                and prior['recorded_optimizer_updates']==80,'CCRC_CAUSAL_EVIDENCE')
    h4x_file_pin(a.h4_exploratory_reward_report,H4R_REWARD_SHA)
    from m463_gate3_cp_eval import h4cc_tech_read_report
    identity=h4cc_source_identity(root)
    h4cc_tech_read_report(a.h4cc_technical_report,auth['technical_sha256'],
                          identity['archived_sha256'],auth['generated_sha256'])
    if a.h4cc_arm=='ref':
        h4x_require(a.h4cc_ref_report is None,'CCRC_REF_CONFLICT')
    else:
        ref=h4cc_read_final_report(a.h4cc_ref_report,auth['evidence_sha256'],'ref')
        old=ref['h4x_authorization']
        h4x_require(old==h4cc_matched_ref_authorization(auth,identity),'CCRC_FRESH_MATCH')
    return auth

def h4cc_run(args):
    auth=h4cc_validate_cli(args)
    out=args.output_dir
    h4x_require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'CCRC_OUTPUT_CONSUMED')
    h4x_storage(out,initial=True)
    out.mkdir(exist_ok=False)
    h4x_write_json(out/'attempt.json',{'authorization':auth,'reserved':True},exclusive=True)
    try:
        pins=h4x_bootstrap(args,h4cc=True)
        _run_impl(args,h4x={'authorization':auth,'pins':pins})
        h4x_write_json(out/'attempt_result.json',{'status':'COMPLETE','authorization':auth},exclusive=True)
    except BaseException as exc:
        try:
            h4x_write_json(out/'attempt_result.json',{'status':'STOPPED','error':repr(exc),
                           'authorization':auth},exclusive=True)
        except Exception as write_error:
            print('M470_H4CC_PARTIAL_WRITE_FAILED',repr(write_error),flush=True)
        raise


def h4cc_validate_held_plan(a):
    if a.mode!='plan':
        raise RuntimeError('M470_H4CC_EXECUTION_HOLD')
    if (a.stage!='easy_repeat_support' or a.output_dir!=H4CC_PATHS[a.h4cc_arm]
            or any(getattr(a,k,None) for k in ('h4_arm','h4_exploratory','h4_readiness',
                'h4_causal','h3_m1','h3_diagnostic_only','h1_m1','h1_m2','resume_state',
                'independent_budget','h4_control_report','h4_exploratory_causal_report',
                'h4_exploratory_reward_report','h4_causal_failure_report',
                'h4_readiness_reward_report','h3_causal_report','h3_old10m_report',
                'smoke_report','rollouts','num_evals','h4cc_technical_report','h4cc_ref_report'))):
        raise RuntimeError('M470_H4CC_PLAN_CONTRACT')



# H4-M6 held core. No legacy authorization or runtime entry is reused.
H4M6_ID = 'H4-M6-RSS-V1'
H4M6_PATHS = {p: 'm463_runs/m470_h4_m6_'+p+'_v1' for p in ('ref','rss','dev')}
H4M6_PATHS['tech'] = 'm463_runs/m470_h4_m6_tech_v3'
H4M6_SPEC = {'id':H4M6_ID,'window':50,'peak_relaxation':.5,
    'event_unchanged':True,'terminal_unchanged':True,'seed':1,
    'checkpoint_sha256':SEED1_SHA,'restore_sha256':H3_CAUSAL_RESTORE_SHA,
    'orbax_sha256':H4_ORBAX_SHA,'config_fingerprint':H3_FP,
    'tech_physical':416,'tech_reset_lanes':8,'tech_max_lane_steps':52,
    'training_per_arm':10485760,'optimizer_per_arm':1600,'source_eval_per_arm':6000,
    'chunks_per_arm':8,'train_reset_lanes_per_arm':24576,'eval_reset_lanes_per_arm':6,
    'dev_physical':89616,'dev_logical':120,'dev_group_resets':24,
    'mechanism_records':416,'mechanism_record_cap':65536,'mechanism_report_cap':32*1024**2,
    'tech_report_cap':16*1024**2,'log_cap':16*1024**2,
    'total_future_physical':21073552,'execution_authorized':False}

def h4m6_require(ok,label):
    if not bool(ok):
        raise RuntimeError('M470_H4M6_'+label)

def h4m6_spec_digest():
    return hashlib.sha256(json.dumps(H4M6_SPEC,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def h4m6_plan(phase):
    h4m6_require(phase in H4M6_PATHS,'PHASE')
    return {'spec':dict(H4M6_SPEC),'spec_sha256':h4m6_spec_digest(),
            'phase':phase,'output':H4M6_PATHS[phase],
            'training_performed':False,'execution_authorized':False}

def h4m6_parse_marker(rules,source,spec):
    """Schema only; authorization is not execution and held dispatch never consumes it."""
    import re
    active=re.findall(r'^[A-Z0-9_]*ACTIVE_GO:.*$',rules,re.M)
    h4m6_require(len(active)==1 and active[0].startswith('H4M6_ACTIVE_GO: '),'EXECUTION_HOLD' if not active else 'AUTH_CONFLICT')
    tokens=active[0].split()
    h4m6_require(len(tokens)==12 and tokens[1]==H4M6_ID,'AUTH_SCHEMA')
    _,ident,phase,path,src,sp,gen,ext,st,tech,ref,rss=tokens
    sha=lambda x:re.fullmatch(r'[0-9a-f]{64}',x) is not None
    h4m6_require(phase in H4M6_PATHS and path==H4M6_PATHS[phase] and
        all(sha(x) for x in (src,sp,gen,ext,st)) and src==source and sp==spec,'AUTH_PIN')
    expected={'tech':(False,False,False),'ref':(True,False,False),
              'rss':(True,True,False),'dev':(True,True,True)}[phase]
    h4m6_require(all(sha(x) if need else x=='-' for x,need in zip((tech,ref,rss),expected)),'AUTH_EVIDENCE')
    approved=re.findall(r'^H4M6_STATIC_APPROVED: (.+)$',rules,re.M)
    h4m6_require(len(approved)==1,'STATIC_REQUIRED')
    fields=approved[0].split()
    h4m6_require(len(fields)==6 and fields[:5]==[src,sp,gen,ext,st] and
        re.fullmatch(r'm463_runs/m470_h4_m6_2r2_full_static_[0-9_]+\.log',fields[5]) is not None,'STATIC_BINDING')
    for tag,need,val in zip(('TECH','REF','RSS'),expected,(tech,ref,rss)):
        pins=re.findall(r'^H4M6_APPROVED_'+tag+r'_SHA256: ([0-9a-f]{64})$',rules,re.M)
        h4m6_require((pins==[val]) if need else (not pins),'EVIDENCE_BINDING '+tag)
    return dict(id=ident,phase=phase,output=path,source_sha256=src,spec_sha256=sp,
        generated_sha256=gen,external_sha256=ext,static_sha256=st,static_log=fields[5],
        technical_sha256=tech,ref_sha256=ref,rss_sha256=rss)


def h4m6_train_gate(phase=None):
    auth=h4m6_authorization(phase=phase)
    h4m6_require(auth['phase'] in ('ref','rss'),'TRAIN_PHASE')
    return auth

def h4m6_train_inputs(args,auth):
    """Pre-import evidence replay under a distinct H4M6 phase."""
    from m463_gate3_cp_eval import h4m6_read_technical,h4m6_external_sources,h4m3_digest,H4M6_TECH_OUTPUT
    root=Path(__file__).resolve().parent
    h4m6_require(auth['phase'] in ('ref','rss') and args.h4m6_arm==auth['phase']
        and args.mode=='longtrain' and args.stage=='easy_repeat_support'
        and args.output_dir.resolve()==root/auth['output'],'TRAIN_PHASE_PATH')
    h4m6_require(not any(getattr(args,k,None) for k in (
        'h4cc_arm','h4cc_technical_report','h4cc_ref_report','h4_arm','h4_control_report',
        'h4_exploratory','h4_readiness','h4_readiness_reward_report','h4_causal',
        'h4_causal_failure_report','h3_m1','h3_diagnostic_only','h3_causal_report',
        'h3_old10m_report','h1_m1','h1_m2','smoke_report','resume_state','independent_budget')),
        'TRAIN_CONFLICT')
    h4m6_require((args.rollouts,args.num_evals)==(None,None),'TRAIN_BUDGET_FLAGS')
    h4x_file_pin(args.checkpoint,SEED1_SHA)
    h4x_file_pin(args.restore_report,H3_CAUSAL_RESTORE_SHA)
    h4x_file_pin(args.h4_exploratory_causal_report,H4X_CAUSAL_SHA)
    h4x_file_pin(args.h4_exploratory_reward_report,H4R_REWARD_SHA)
    prior=json.loads(args.h4_exploratory_causal_report.read_bytes())
    h4m6_require(prior['status']=='M470_H4C_DIAGNOSTIC_COMPLETE'
        and prior['phase_verdicts']=={'A':'DIVERGENT','B':'PASS'}
        and prior['comparisons']['B']['bitwise_equal'] is True
        and prior['recorded_physical_transitions']==1024
        and prior['recorded_optimizer_updates']==80,'TRAIN_CAUSAL_EVIDENCE')
    h4m6_require(args.h4m6_technical_report is not None
        and args.h4m6_technical_report.resolve()==root/H4M6_TECH_OUTPUT,'TRAIN_TECH_PATH')
    tech=h4m6_read_technical(args.h4m6_technical_report,auth['technical_sha256'],auth)
    external=h4m6_external_sources(root)
    h4m6_require(h4m3_digest(external)==auth['external_sha256']
        and external==tech['provenance']['external'],'TRAIN_EXTERNAL')
    ref=None
    if auth['phase']=='ref':
        h4m6_require(args.h4m6_ref_report is None,'TRAIN_REF_CONFLICT')
    else:
        ref=h4m6_read_final_report(args.h4m6_ref_report,auth['ref_sha256'],'ref',auth)
    return {'pins':h4x_bootstrap(args,h4cc=True,h4m6=auth),
        'external':external,'tech_sha256':auth['technical_sha256'],'ref':ref,
        'input_paths':{k:str(getattr(args,k).resolve()) for k in (
          'checkpoint','restore_report','h4_exploratory_causal_report',
          'h4_exploratory_reward_report','h4m6_technical_report','h4m6_ref_report')
          if getattr(args,k) is not None}}

def h4m6_check_attempt(out,auth,*,complete=False):
    out=Path(out)
    req=h4m6_require
    req(out.resolve()==out.absolute() and out.is_dir() and not out.is_symlink(),'ATTEMPT_PATH')
    items=[('attempt.json',{'authorization':auth,'reserved':True})]
    if complete:
        items.extend((('dispatch.started.json',{'authorization':auth}),
                      ('attempt_result.json',{'status':'COMPLETE','authorization':auth})))
    for name,expected in items:
        p=out/name
        req(p.is_file() and not p.is_symlink() and json.loads(p.read_bytes())==expected,'ATTEMPT '+name)

def h4m6_read_final_report(path,expected_sha,arm,auth):
    """Strict final report replay; historical H4CC readers are not modified."""
    req=h4m6_require
    root=Path(__file__).resolve().parent
    req(arm in ('ref','rss') and path is not None,'REPORT_ARM')
    path=Path(path)
    req(path.resolve()==root/H4M6_PATHS[arm]/'report.json' and not path.parent.is_symlink(),'REPORT_PATH')
    h4x_file_pin(path,expected_sha)
    r=json.loads(path.read_bytes()); a=r['h4x_authorization']; c=r['h4_common_contract']
    common=('id','source_sha256','spec_sha256','generated_sha256','external_sha256','static_sha256','static_log','technical_sha256')
    req(all(a[k]==auth[k] for k in common) and a['phase']==arm
        and a['output']==H4M6_PATHS[arm] and a['rss_sha256']=='-'
        and auth[arm+'_sha256']==expected_sha
        and a['ref_sha256']==('-' if arm=='ref' else auth['ref_sha256']),'REPORT_AUTH')
    req(r['status']=='M463_COVERAGE_LONGTRAIN_PASS' and r['h4m6_arm']==arm
        and r['h4m6_spec']==H4M6_SPEC and r['h4_arm']=='candidate'
        and r['h4m6_technical_sha256']==a['technical_sha256']
        and r['h4m6_ref_sha256']==a['ref_sha256']
        and r['h4m6_external_sha256']==a['external_sha256']
        and r['config_fingerprint']==H3_FP and r['gpu_samples']>0
        and r['h4cc_arm']=='ref' and r['h4cc_spec']==H4CC_SPEC
        and r['source_commit']==PIN and r['source_checkpoint_sha256']==SEED1_SHA
        and r['orbax_sha256']==H4_ORBAX_SHA and r['training_seed']==1
        and r['trained_from']=='C401M_WEIGHTS_ONLY_ADAM_STARTED_ONCE'
        and r['resume_fullstate_sha256'] is None and r['cumulative_steps_start']==0
        and r['steps_observed']==r['run_steps_observed']==10485760
        and r['reward_PD_termination_source_unchanged']==(arm=='ref')
        and r['PD_termination_source_unchanged'] is True and r['eval_source_reward_unchanged'] is True
        and r['training_performed'] is True and r['promotion_authorized'] is False
        and r['coverage_gate_pass'] is True and r['memory_gate_pass'] is True
        and r['weights_finite'] is True and r['callback_vs_final_equal'] is True
        and r['final_tensor_count']==25 and r['gpu_total_mib']==12288
        and 0<r['gpu_peak_mib']<.95*12288 and not r['gpu_monitor_errors']
        and r['h4_matched_nominal_pass'] is True and r['h4_legacy_reconciled'] is True,'REPORT_CONTRACT')
    req(c==h4_common_contract(r['hardware_scaled_config'],r['train_reward_config'],r['instrumented_brax'])
        and r['runner_sha256']==c['source_digests']['m463_push_ppo_train.py']
        and a['generated_sha256']==c['generated_brax_sha256']
        and a['source_sha256']==h4m6_source_identity(root)['sha256'],'REPORT_SOURCE')
    h4m6_check_attempt(path.parent,a,complete=True)
    h4x_file_pin(path.parent/'m463_brax_ppo_instrumented.py',a['generated_sha256'])
    req([v['steps'] for v in r['checkpoints']]==[v['steps'] for v in r['fullstate_checkpoints']]==list(H4_EDGES),'ARTIFACT_EDGES')
    for step,pol,full in zip(H4_EDGES,r['checkpoints'],r['fullstate_checkpoints']):
        req(pol['file']==f'm463_easy_repeat_support_longtrain_step_{step:09d}.msgpack'
            and full['file']==f'm470_fullstate_step_{step:09d}.msgpack'
            and full['manifest']==f'm470_fullstate_step_{step:09d}.json','ARTIFACT_NAMES')
        for name,sha,cap in ((pol['file'],pol['sha256'],H4X_SPEC['policy_cap']),
                             (full['file'],full['sha256'],H4X_SPEC['fullstate_cap']),
                             (full['manifest'],full['manifest_sha256'],H4X_SPEC['other_cap'])):
            p=path.parent/name;h4x_file_pin(p,sha);req(0<p.stat().st_size<=cap,'ARTIFACT_SIZE')
        m=json.loads((path.parent/full['manifest']).read_bytes())
        req(m['h4m6_arm']==arm and m['h4m6_authorization']==a
            and m['h4_common_contract']==c and m['fullstate_sha256']==full['sha256']
            and m['cumulative_steps']==step and m['parent_resume_sha256'] is None
            and all(m[k] is True for k in ('reset_edge','optimizer_state_included',
                'training_rng_included','evaluator_rng_included')),'MANIFEST')
    ctrl=h4cc_read_historical_control(root/H4X_PATHS['control']/'report.json',c)
    h4_final_checks(r,r['coverage'],ctrl)
    h4cc_verify_ledger(r)
    req(h4x_verify_operational(r,h4cc=True,h4m6=True)==H4CC_MATCHED_DR_SHA,'REPORT_DR')
    final=h4m6_replay_ledger(r,arm)
    req(final==r['h4m6_final'] and final['audit']['passed']
        and final['h4_counts']==r['h4_final_counts'],'REPORT_LEDGER')
    if arm=='ref':
        req(final['audit']['exposure_adequate'] is True,'REF_EXPOSURE')
    return r

def h4m6_run(args):
    auth=h4m6_train_gate(args.h4m6_arm)
    evidence=h4m6_train_inputs(args,auth)
    out=args.output_dir.resolve()
    h4m6_require(out.parent.is_dir() and not out.parent.is_symlink()
        and not out.exists() and not out.is_symlink(),'TRAIN_OUTPUT_CONSUMED')
    h4x_storage(out,initial=True)
    out.mkdir(exist_ok=False)
    h4x_write_json(out/'attempt.json',{'authorization':auth,'reserved':True},exclusive=True)
    try:
        _run_impl(args,h4x={'authorization':auth,'pins':evidence['pins'],'m6_evidence':evidence})
        h4x_write_json(out/'attempt_result.json',{'status':'COMPLETE','authorization':auth},exclusive=True)
    except BaseException as exc:
        try:h4x_write_json(out/'attempt_result.json',{'status':'STOPPED','authorization':auth,'error':repr(exc)[:4096]},exclusive=True)
        except BaseException as err:print('M470_H4M6_TRAIN_PARTIAL_WRITE_FAILED',repr(err),flush=True)
        raise

def h4m6_current_auth(auth,cc_arm):
    return (h4m6_authorization(phase=auth['phase']) if auth['id']==H4M6_ID
            else h4cc_authorization() if cc_arm is not None else h4x_authorization())

def h4m6_post_train_inputs(args,auth,evidence):
    """Re-open the exact pre-import evidence in a clean process, no runtime import."""
    import subprocess
    root=Path(__file__).resolve().parent
    command=[sys.executable,'-B','-c',
        'import json,sys;from pathlib import Path;from types import SimpleNamespace;'
        'import m463_push_ppo_train as t;p=json.loads(sys.argv[1]);'
        'a=SimpleNamespace(**{k:Path(v) for k,v in p.items()});'
        'a.h4m6_arm=sys.argv[2];a.mode="longtrain";a.stage="easy_repeat_support";'
        'a.output_dir=Path(t.H4M6_PATHS[a.h4m6_arm]);a.rollouts=a.num_evals=None;'
        'a.h4m6_ref_report=getattr(a,"h4m6_ref_report",None);'
        'auth=t.h4m6_authorization(phase=a.h4m6_arm);'
        'print(json.dumps([auth,t.h4m6_train_inputs(a,auth)],sort_keys=True))',
        json.dumps(evidence['input_paths'],sort_keys=True),args.h4m6_arm]
    h4m6_require(json.loads(subprocess.check_output(command,cwd=root,text=True))==[auth,evidence],'TRAIN_POST_INPUTS')

def h4m6_generated_guard(audit,generated_file,environment):
    auth=h4m6_train_gate(getattr(environment,'_h4m6_arm',None))
    root=Path(__file__).resolve().parent
    out=root/auth['output']; generated=Path(generated_file)
    h4m6_require(auth['phase'] in ('ref','rss') and callable(audit)
        and getattr(audit,'h4x_authorization',None)==auth
        and getattr(environment,'_h4m6_arm',None)==auth['phase']
        and getattr(environment,'_h4cc_arm',None)=='ref'
        and getattr(environment,'_h4_arm',None)=='candidate','GENERATED_ENV')
    h4m6_require(generated.resolve()==out/'m463_brax_ppo_instrumented.py'
        and not generated.is_symlink() and digest(generated)==auth['generated_sha256'],'GENERATED_PIN')
    h4m6_check_attempt(out,auth)
    h4x_write_json(out/'dispatch.started.json',{'authorization':auth},exclusive=True)

def h4m6_validate_held_plan(a):
    h4m6_require(a.mode=='plan','EXECUTION_HOLD')
    phase=getattr(a,'h4m6_arm',None)
    h4m6_require(phase in ('ref','rss') and a.stage=='easy_repeat_support'
        and a.output_dir==Path(H4M6_PATHS[phase]),'PLAN_PATH')
    h4m6_require(not any(getattr(a,k,None) for k in (
        'h4cc_arm','h4cc_technical_report','h4cc_ref_report','h4_arm','h4_control_report',
        'h4_exploratory','h4_exploratory_causal_report','h4_exploratory_reward_report',
        'h4_readiness','h4_readiness_reward_report','h4_causal','h4_causal_failure_report',
        'h3_m1','h3_diagnostic_only','h3_causal_report','h3_old10m_report',
        'h1_m1','h1_m2','smoke_report','rollouts','num_evals','resume_state',
        'independent_budget')),'PLAN_CONFLICT')
    return h4m6_plan(phase)

def h4m6_count_keys():
    keys=['m463_h4m6_bad_per_rollout']
    for side in (1,2):
        names=('qualified','event_terminal','start','eligible','positive_start',
            'completed','close_terminal','close_truncation','close_boundary',
            'close_command','close_event','opportunity','attempt')
        names+=tuple(f'{k}_{w}' for w in (1,10,25,50) for k in ('reached','same_unroll'))
        names+=tuple(f'offset_{off}' for off in range(32))
        keys.extend(f'm463_h4m6_{side}_{name}_per_rollout' for name in names)
    return tuple(keys)

def h4m6_sum_keys():
    keys=['m463_h4m6_'+k+'_sum' for k in ('base_reward','expected_reward',
        'shadow_relief','delivered_relief','transition_reward','ppo_scaled_relief')]
    keys.extend(f'm463_h4m6_{side}_{name}{suffix}_sum' for side in (1,2)
        for name in ('shadow_relief','delivered_relief') for suffix in ('','_same_unroll'))
    return tuple(keys)

def h4m6_epoch_audit(counts,sums,pending,h4_counts,arm,*,final=False):
    """Conservation and scope gates; dose sufficiency is distinct from technical PASS."""
    import math
    h4m6_require(arm in ('ref','rss') and set(counts)==set(h4m6_count_keys()) and
        set(sums)==set(h4m6_sum_keys()) and set(pending)=={'1','2'},'LEDGER_SCHEMA')
    cap=10485760 if final else 5242880
    h4m6_require(all(type(v) is int and 0<=v<=cap for v in counts.values()) and
        all(type(v) is int and 0<=v<=(16384 if final else 8192) for v in pending.values()) and
        all(type(v) in (int,float) and math.isfinite(v) for v in sums.values()),'LEDGER_VALUES')
    c=lambda side,name:counts[f'm463_h4m6_{side}_{name}_per_rollout']
    f=lambda name:sums['m463_h4m6_'+name+'_sum']
    checks={'no_bad':counts['m463_h4m6_bad_per_rollout']==0,
        'actual_reward':f('expected_reward')==f('transition_reward'),
        'scope_global':0<=f('delivered_relief')<=f('shadow_relief'),
        'arm_global':f('delivered_relief')==(f('shadow_relief') if arm=='rss' else 0)}
    floors={}
    for side in (1,2):
        q=c(side,'qualified'); n=c(side,'start'); e=c(side,'eligible')
        closed=sum(c(side,k) for k in ('completed','close_terminal','close_truncation',
                                      'close_boundary','close_command','close_event'))
        checks[f'qualified_{side}']=q==h4_counts[f'm463_h4_stand_{side}_events_per_rollout'] and q==n+c(side,'event_terminal')
        checks[f'closure_{side}']=n==closed+pending[str(side)]
        checks[f'no_impossible_{side}']=c(side,'close_boundary')==c(side,'close_event')==0
        checks[f'opportunity_{side}']=q<=c(side,'attempt')<=c(side,'opportunity')
        checks[f'dose_{side}']=0<=e<=50*n and c(side,'positive_start')<=n and sum(c(side,f'offset_{i}') for i in range(32))==e
        prev=n
        for w in (1,10,25,50):
            a=c(side,f'reached_{w}'); b=c(side,f'same_unroll_{w}')
            checks[f'window_{side}_{w}']=0<=b<=a<=prev and (w<32 or b==0)
            prev=a
        checks[f'complete_{side}']=c(side,'completed')==c(side,'reached_50')
        for suffix in ('','_same_unroll'):
            sh=f(f'{side}_shadow_relief{suffix}'); de=f(f'{side}_delivered_relief{suffix}')
            checks[f'relief_{side}{suffix}']=0<=de<=sh and de==(sh if arm=='rss' else 0)
        floors[f'qualified_{side}']=q>=32
        floors[f'reached25_{side}']=c(side,'reached_25')>=16
        floors[f'positive_{side}']=c(side,'positive_start')>=32
    return {'checks':checks,'passed':all(checks.values()),'dose_floors':floors,
            'exposure_adequate':all(floors.values()) if final else None}

def h4m6_exposure_comparison(ref,rss):
    """Post-training opportunity comparability; never equate survival outcomes."""
    rows={}
    for side in (1,2):
        for name in ('qualified','opportunity'):
            key=f'm463_h4m6_{side}_{name}_per_rollout'
            rows[key]=[ref['counts'][key],rss['counts'][key]]
    for name in ('assigned','repeat_events'):
        key=f'm463_h4_class_3_{name}_per_rollout'
        rows[key]=[ref['h4_counts'][key],rss['h4_counts'][key]]
    nominal='m463_h4_class_0_steps_per_rollout'
    r,t=ref['h4_counts'][nominal],rss['h4_counts'][nominal]
    h4m6_require(all(type(v) is int and 0<=v<=10485760
        for pair in rows.values() for v in pair) and all(type(v) is int and 0<=v<=10485760 for v in (r,t)),'MATCH_COUNTS')
    checks={k:a>0 and 4*a<=5*b and 4*b<=5*a for k,(a,b) in rows.items()}
    checks['matched_nominal']=r>=1867669 and t>=1867669 and 5*t>=4*r
    adequate=(ref['audit']['passed'] is True and rss['audit']['passed'] is True and
        ref['audit']['exposure_adequate'] is True and rss['audit']['exposure_adequate'] is True and all(checks.values()))
    return {'counts':rows,'checks':checks,'decision':'ADEQUATE' if adequate else 'INCONCLUSIVE_EXPOSURE'}


def h4m6_chunk_ledger(jax,np,report,arm):
    """Compose with the existing H4/CCRC host callback, read-only carry audit."""
    h4m6_require(arm in ('ref','rss'),'ARM')
    chunks=[]; epochs=[]; counts={}; sums={}; h4_counts={}
    report['h4m6_chunks']=chunks
    report['h4m6_epochs']=epochs
    def merge(dst,src):
        for k,v in src.items():
            dst[k]=dst.get(k,0)+v
    def callback(index,env_state,metrics,old_counts):
        h4m6_require(len(chunks)<8 and index==len(chunks)%4+1,'CHUNK_SEQUENCE')
        def scalar(k):
            a=np.asarray(jax.device_get(metrics[k]))
            h4m6_require(a.shape==() and a.dtype==np.float32 and np.isfinite(a),'METRIC '+k)
            return a.item()
        c={k:h4_integer_count(scalar(k)) for k in h4m6_count_keys()}
        z={k:5*scalar(k) for k in h4m6_sum_keys()}
        h4m6_require(c['m463_h4m6_bad_per_rollout']==0,'BAD_TRANSITION')
        merge(counts,c); merge(sums,z); merge(h4_counts,old_counts)
        chunks.append({'chunk':len(chunks)+1,'counts':c,'sums':z,'h4_counts':dict(old_counts)})
        if index==4:
            info=env_state.info
            active,age,side,positive,consumed=[np.asarray(jax.device_get(info['h4m6_'+k])) for k in ('active','age','side','positive','consumed')]
            h4m6_require(all(x.shape==(1,8192) and x.dtype==np.int32 for x in (active,age,side,positive,consumed))
                and np.isin(active,(0,1)).all() and np.isin(positive,(0,1)).all()
                and np.isin(consumed,(0,1)).all() and np.all(active<=consumed)
                and np.all(np.where(active>0,(age>=0)&(age<50)&np.isin(side,(1,2)),(age==0)&(side==0)&(positive==0))),
                'PENDING_SCHEMA')
            pending={str(i):int(np.count_nonzero((active>0)&(side==i))) for i in (1,2)}
            audit=h4m6_epoch_audit(counts,sums,pending,h4_counts,arm)
            epochs.append({'counts':dict(counts),'sums':dict(sums),'h4_counts':dict(h4_counts),
                           'pending':pending,'audit':audit})
            h4m6_require(audit['passed'],'EPOCH_LEDGER')
            counts.clear(); sums.clear(); h4_counts.clear()
            if len(chunks)==8:
                report['h4m6_final']=h4m6_replay_ledger(report,arm)
    return callback

def h4m6_replay_ledger(report,arm):
    """Replay independently from eight observed chunks; no accepted-summary shortcut."""
    chunks=report['h4m6_chunks']; epochs=report['h4m6_epochs']
    h4m6_require(len(chunks)==8 and len(epochs)==2,'FINAL_LENGTH')
    c_all={}; s_all={}; h_all={}; p_all={'1':0,'2':0}
    def merge(dst,src):
        for k,v in src.items():
            dst[k]=dst.get(k,0)+v
    for ep in range(2):
        counts={}; sums={}; h4={}
        for n,row in enumerate(chunks[ep*4:ep*4+4],ep*4+1):
            h4m6_require(row['chunk']==n and set(row['counts'])==set(h4m6_count_keys())
                and set(row['sums'])==set(h4m6_sum_keys())
                and all(type(v) is int and 0<=v<=1310720 for v in row['counts'].values())
                and all(type(v) is int and 0<=v<=1310720 for v in row['h4_counts'].values())
                and all(type(v) in (int,float) and math.isfinite(v) for v in row['sums'].values()),'FINAL_CHUNK')
            merge(counts,row['counts']); merge(sums,row['sums']); merge(h4,row['h4_counts'])
        saved=epochs[ep]; pending=saved['pending']
        audit=h4m6_epoch_audit(counts,sums,pending,h4,arm)
        h4m6_require(saved=={'counts':counts,'sums':sums,'h4_counts':h4,'pending':pending,'audit':audit}
            and audit['passed'],'FINAL_EPOCH_REPLAY')
        merge(c_all,counts); merge(s_all,sums); merge(h_all,h4); merge(p_all,pending)
    audit=h4m6_epoch_audit(c_all,s_all,p_all,h_all,arm,final=True)
    h4m6_require(audit['passed'],'FINAL_LEDGER')
    return {'counts':c_all,'sums':s_all,'h4_counts':h_all,'pending':p_all,'audit':audit}

def h4m6_source_identity(root):
    """Separate current-source map. Historical H4CC source pins remain untouched."""
    root=Path(root).resolve()
    names=(*H4_FILES,'m463_gate3_cp_eval.py','paper_config_preflight.json')
    result={}
    for name in names:
        p=root/name
        h4m6_require(p.is_file() and not p.is_symlink(),'SOURCE_FILE '+name)
        result[name]=digest(p)
    common=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    h4m6_require(common==h4x_source_digest(root),'COMMON_SOURCE')
    return {'files':result,'sha256':common}


def h4m6_authorization(root=None,phase=None):
    root=Path(__file__).resolve().parent if root is None else Path(root).resolve()
    rules=(root/'G1_RESEARCH_RULES.md').read_text('utf8')
    if not (any(x.startswith('H4M6_ACTIVE_GO:') for x in rules.splitlines())
        and any(x.startswith('H4M6_STATIC_APPROVED:') for x in rules.splitlines())):
        raise RuntimeError('M470_H4M6_EXECUTION_HOLD')
    auth=h4m6_parse_marker(rules,h4m6_source_identity(root)['sha256'],h4m6_spec_digest())
    h4m6_require(phase is None or auth['phase']==phase,'AUTH_PHASE')
    import m463_gate3_cp_eval as ev
    h4m6_require(not any((H4_EXECUTION_AUTHORIZED,H4R_EXECUTION_AUTHORIZED,
                           H4C_EXECUTION_AUTHORIZED,H4CC_EXECUTION_AUTHORIZED,
                           ev.H4_EVAL_EXECUTION_AUTHORIZED,ev.H4CC_EVAL_EXECUTION_AUTHORIZED,
                           ev.H4CC_TECH_EXECUTION_AUTHORIZED)),'LEGACY_HOLD_REQUIRED')
    h4m6_validate_static(root,auth)
    return auth

def h4m6_validate_static(root,auth):
    root=Path(root).resolve()
    h4m6_require(h4m6_source_identity(root)['sha256']==auth['source_sha256']
        and h4m6_spec_digest()==auth['spec_sha256'],'STATIC_SOURCE')
    log=root/auth['static_log']
    h4m6_require(log.is_file() and not log.is_symlink() and 0<log.stat().st_size<=H4M6_SPEC['log_cap']
        and digest(log)==auth['static_sha256'],'STATIC_LOG')
    rows=log.read_text('utf8').splitlines()
    for marker,value in (
        ('H4_M6_2R2_FULL_STATIC_PASS',auth['source_sha256']),
        ('H4_M6_2R2_SPEC_SHA256',auth['spec_sha256']),
        ('H4_M6_2R2_GENERATED_SHA256',auth['generated_sha256']),
        ('H4_M6_2R2_EXTERNAL_SHA256',auth['external_sha256']),
        ('H4_M6_2R2_ARITH_REPORT_SHA256','77af67e65ef8757ba580a42e84b86c0501fd3e912cc74fea93441700e8e9903c')):
        h4m6_require([x for x in rows if x.startswith(marker+' ')]==[marker+' '+value],'STATIC_MARKER '+marker)
    return True

def h4m6_fixed_gae(np,reward,values,bootstrap,discount,truncation):
    """Host arithmetic fixture of pinned losses.py; no actor/critic/optimizer call."""
    reward=np.asarray(reward,dtype=np.float32)
    values=np.asarray(values,dtype=np.float32)
    bootstrap=np.asarray(bootstrap,dtype=np.float32)
    discount=np.asarray(discount,dtype=np.float32)
    truncation=np.asarray(truncation,dtype=np.float32)
    h4m6_require(reward.shape==values.shape==discount.shape==truncation.shape and
        reward.ndim==2 and reward.shape[0]==32 and bootstrap.shape==reward.shape[1:] and
        all(np.isfinite(x).all() for x in (reward,values,bootstrap,discount,truncation)) and
        np.isin(discount,(0,1)).all() and np.isin(truncation,(0,1)).all(),'GAE_SHAPE')
    gamma=np.float32(.98); lam=np.float32(.95)
    scaled=reward*np.float32(.1)
    mask=1-truncation
    term=(1-discount)*mask
    next_values=np.concatenate((values[1:],bootstrap[None]),axis=0)
    delta=(scaled+gamma*(1-term)*next_values-values)*mask
    acc=np.zeros_like(bootstrap); accum=np.empty_like(values)
    for k in range(31,-1,-1):
        acc=delta[k]+gamma*(1-term[k])*mask[k]*lam*acc
        accum[k]=acc
    vs=values+accum
    next_vs=np.concatenate((vs[1:],bootstrap[None]),axis=0)
    adv=(scaled+gamma*(1-term)*next_vs-values)*mask
    normalized=(adv-adv.mean())/(adv.std()+np.float32(1e-8))
    return {'vs':vs,'adv':adv,'normalized':normalized}

# H4-M8-CAM-V1 held core. No executable dispatcher or runtime authorization.
H4M8_EXECUTION_AUTHORIZED = False
H4M8_SPEC = {'id':'H4-M8-CAM-V1','length_m':.793,'dv_ref_mps':.35,
    'alpha':.25,'dt':.02,'window_ticks':50,'delay_ticks':7,'ramp_ticks':8,
    'weight8_total':288,'max_window_penalty':.18,'reward_scaling':.1,
    'checkpoint_sha256':SEED1_SHA,'restore_sha256':H3_CAUSAL_RESTORE_SHA,
    'orbax_sha256':H4_ORBAX_SHA,'config_fingerprint':H3_FP,
    'arms':('cam-ref','cam'),'training_budget_authorized':False}
H4M8_PATHS = {'cam-ref':'m463_runs/m470_h4_m8_cam_ref_v1',
               'cam':'m463_runs/m470_h4_m8_cam_v1'}
H4M8_BASE_HEAD = '72940da43157a1190554827806e04de4a1c2d970'


def h4m8_held_plan(a):
    # Plan-only and fail closed. No implicit mapping to any old runtime.
    if a.mode!='plan':
        raise RuntimeError('M470_H4M8_EXECUTION_HOLD')
    if a.stage!='easy_repeat_support' or a.h4m8_arm not in H4M8_PATHS:
        raise RuntimeError('M470_H4M8_PLAN_SCOPE')
    allowed={'mode','stage','checkpoint','restore_report','output_dir','h4m8_arm'}
    if any(v is not None and v is not False for k,v in vars(a).items() if k not in allowed):
        raise RuntimeError('M470_H4M8_CONFLICTING_FLAG')
    if a.output_dir.resolve()!=Path(H4M8_PATHS[a.h4m8_arm]).resolve():
        raise RuntimeError('M470_H4M8_PLAN_NAMESPACE')
    if a.output_dir.exists() or a.output_dir.is_symlink():
        raise RuntimeError('M470_H4M8_PLAN_OUTPUT_EXISTS')
    return {'arm':a.h4m8_arm,'spec':H4M8_SPEC,'execution_authorized':False,
            'training_performed':False,'implementation_complete':False,
            'checkpoint_verified':False,'restore_verified':False}


def h4m8_window_ledger(counts,pending_before,pending_after):
    """Per-side integer conservation; pending includes host epoch reset disposition."""
    keys={'qualified','start','event_terminal','eligible','positive_start',
          'completed','close_terminal','close_truncation','close_boundary',
          'close_command','close_event','weight8','reach25','opportunities',
          'attempts','host_reset'}
    if set(counts)!=keys or any(type(v) is not int or v<0 for v in
            (*counts.values(),pending_before,pending_after)):
        raise RuntimeError('M470_H4M8_LEDGER_SCHEMA')
    c=counts
    closes=sum(c[k] for k in ('completed','close_terminal','close_truncation',
                               'close_boundary','close_command','close_event','host_reset'))
    if (c['qualified']!=c['start']+c['event_terminal'] or
            pending_before+c['start']!=closes+pending_after or
            c['positive_start']>pending_before+c['start'] or
            c['reach25']>pending_before+c['start'] or
            c['weight8']>8*c['eligible'] or c['weight8']>288*(pending_before+c['start'])):
        raise RuntimeError('M470_H4M8_LEDGER_CONSERVATION')
    return True


def h4m8_static_regression(root,brax_train_path):
    """Actual source AST/compile and NumPy fixture only. No native imports."""
    import ast
    import subprocess
    import numpy as np
    import m463_instrument_brax as gen
    root=Path(root).resolve()
    def require(ok,label):
        if not ok:
            raise RuntimeError('M470_H4M8_STATIC_'+label)
    def old(name):
        return subprocess.check_output(['git','show',H4M8_BASE_HEAD+':'+name],cwd=root)
    def bits(a,b):
        return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    files=('m463_coverage_env.py','m463_instrument_brax.py','m463_push_ppo_train.py','m463_gate3_cp_eval.py')
    src={name:(root/name).read_bytes() for name in files}
    for name,raw in src.items():
        compile(raw,name,'exec')
    require(src[files[3]]==old(files[3]),'EVALUATOR_IDENTITY')
    require(src[files[1]].startswith(old(files[1])),'LEGACY_INSTRUMENT_PREFIX')
    tree=ast.parse(src[files[0]])
    base_tree=ast.parse(old(files[0]))
    def method(tree,name):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CoverageJoystick')
        return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    require(ast.dump(method(tree,'_get_reward'))==ast.dump(method(base_tree,'_get_reward')),'NATIVE_REWARD')
    for name in ('reset','step'):
        n=copy.deepcopy(method(tree,name))
        n.body=[x for x in n.body if not (isinstance(x,ast.If) and
                '_h4m8_common' in ast.unparse(x.test))]
        require(ast.dump(n)==ast.dump(method(base_tree,name)),'LEGACY_'+name.upper())
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name.startswith('h4m8_')) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('H4M8_') for t in n.targets))]
    ns={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<CAM pure helpers>','exec'),ns)
    pre,post,momentum=(ns[k] for k in ('h4m8_pre','h4m8_post','h4m8_momentum'))
    require(tuple('h4m8_'+k for k in ns['H4M8_FIELDS'])==gen.H4M8_EXT_FIELDS,'EXTRA_FIELDS')
    # Analytic rigid-body spin fixture: zero orbital term; mass variation changes normalization.
    m=np.ones(31,np.float32);m[0]=0
    I=np.ones((31,3),np.float32);I[0]=0
    r=np.zeros((31,3),np.float32)
    R=np.repeat(np.eye(3,dtype=np.float32)[None],31,axis=0)
    cv=np.zeros((31,6),np.float32);cv[1:,0]=1
    ids=np.r_[0,np.ones(30,dtype=np.int32)].astype(np.int32)
    M,h2,c,good=momentum(np,m,I,r,R,cv,r,ids)
    require(bool(good) and M==30 and h2==900 and 0<c<1,'MOMENTUM_ANALYTIC')
    m2=m.copy();m2[16]+=np.float32(.5)
    M2,h22,c2,good2=momentum(np,m2,I,r,R,cv,r,ids)
    require(bool(good2) and M2==30.5 and h22==h2 and c2<c,'MASS_NORMALIZATION')
    badids=ids.copy();badids[3]=0
    require(not bool(momentum(np,m,I,r,R,cv,r,badids)[3]),'ROOT_MASK_NEGATIVE')
    zero=np.zeros(3,np.float32)
    def initial():
        return {'h4m8_'+k:np.int32(0) for k in ns['H4M8_STATE']}
    def step(old,side,k,*,event=False,done=False,boundary=False,command=None,clock=None,wrapper=None):
        age=335+k
        cmd=zero if command is None else command
        q=pre(np,old,event=np.bool_(event),repeat=np.bool_(event),cls=np.int32(3),
            first_age=np.int32(300),first_mag=np.float32(.5),gap=np.int32(35),
            magnitude=np.float32(.35),first_motion=np.int32(0),pre_motion=np.int32(0),
            actor_motion=np.int32(0),direction=np.int32(side+1),support_side=np.int32(side),
            boundary=np.bool_(boundary),age=np.int32(age),clock=np.int32(age if clock is None else clock),
            wrapper_steps=np.float32(age if wrapper is None else wrapper),
            command=cmd,actor_command=cmd,critic_command=cmd)
        return post(np,old,q,done=np.bool_(done),post_zero=np.bool_(np.all(cmd==0)),
            post_coherent=np.bool_(True),native_mature=np.bool_(event),native_side=np.int32(side if event else 0),
            mass=M,h2=h2,cost=c,momentum_good=good)
    rows=[]
    for side in (1,2):
        x=step(initial(),side,0,event=True)
        require(x['h4m8_start']==1 and x['h4m8_weight8']==0 and x['h4m8_bad']==0,'START')
        data=[]
        for k in range(1,51):
            x=step(x,side,k)
            require(x['h4m8_tick_age']==k and x['h4m8_bad']==0,'AGE')
            data.append(x)
        require(sum(int(x['h4m8_weight8']) for x in data)==288 and
            sum(x['h4m8_weight8']>0 for x in data)==43 and x['h4m8_completed']==1 and
            x['h4m8_active']==0 and x['h4m8_consumed']==1,'WINDOW')
        require(sum(x['h4m8_positive_start'] for x in data)==1,'POSITIVE_START')
        rows.extend(data)
        for reason,kwargs in [('terminal',{'done':True}),('truncation',{'wrapper':999}),
                              ('command',{'command':np.ones(3,np.float32)})]:
            x=step(initial(),side,0,event=True)
            x=step(x,side,1,**kwargs)
            require(x['h4m8_close_'+reason]==1 and x['h4m8_shadow']==0 and x['h4m8_active']==0,'CLOSE_'+reason)
        x=step(initial(),side,0,event=True,done=True)
        require(x['h4m8_event_terminal']==1 and x['h4m8_start']==0,'EVENT_TERMINAL')
        x=step(initial(),side,0,event=True)
        x=step(x,side,1,boundary=True)
        require(x['h4m8_close_boundary']==1 and x['h4m8_bad']==1 and x['h4m8_active']==0,'BOUNDARY_NEGATIVE')
    # Preserve exact raw reward bits on all ineligible rows, including -0.
    shape=(5,32)
    ex={key:np.zeros(shape,np.float32 if key in ('h4m8_mass','h4m8_h2','h4m8_cost','h4m8_shadow') else np.int32)
        for key in gen.H4M8_EXT_FIELDS}
    for key,val in [('h4m8_mass',M),('h4m8_h2',h2),('h4m8_cost',c)]:
        ex[key].fill(val)
    for i,row in enumerate(rows):
        for key in ex:
            ex[key].flat[i]=row[key]
    native=np.full(shape,-0.,np.float32);native[0,12]=np.float32(-.2)
    discount=np.ones(shape,np.float32);trunc=np.zeros(shape,np.float32)
    discount[4,30:]=0;trunc[4,31]=1
    ref,desired,actual,bad=gen.h4m8_learning_reward(np,native,discount,trunc,ex,np.bool_(False))
    require(bits(ref,native) and not bad.any() and not actual.any(),'SHAM')
    cam,desired,actual,bad=gen.h4m8_learning_reward(np,native,discount,trunc,ex,np.bool_(True))
    mask=desired>0
    require(not bad.any() and mask.sum()==86 and bits(cam[~mask],native[~mask]) and
        bits(cam[mask],(native-desired)[mask]) and np.all(actual>=0),'CAM_REWARD')
    corrupt={key:value.copy() for key,value in ex.items()}
    corrupt['h4m8_weight8'][0,8]=9
    require(gen.h4m8_learning_reward(np,native,discount,trunc,corrupt,np.bool_(True))[3].any(),'NEGATIVE_WEIGHT')
    # Historical fixed-V arithmetic helper; no JAX, no optimizer.
    v=np.zeros(shape[::-1],np.float32);boot=np.zeros(5,np.float32)
    ga=lambda z:h4m6_fixed_gae(np,z.T,v,boot,discount.T,trunc.T)
    g0,g1,g2=ga(native),ga(ref),ga(cam)
    require(all(bits(g0[k],g1[k]) for k in g0) and g2['adv'][-1,4]==0 and
        np.any(g2['vs']!=g0['vs']),'GAE')
    counts=dict(qualified=1,start=1,event_terminal=0,eligible=50,positive_start=1,
        completed=1,close_terminal=0,close_truncation=0,close_boundary=0,close_command=0,
        close_event=0,weight8=288,reach25=1,opportunities=1,attempts=1,host_reset=0)
    require(h4m8_window_ledger(counts,0,0),'LEDGER')
    try:
        h4m8_window_ledger(dict(counts,completed=0),0,0)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('M470_H4M8_STATIC_LEDGER_NEGATIVE')
    raw=Path(brax_train_path).read_bytes()
    prev={'__name__':'_h4m8_previous_instrumenter'}
    exec(compile(old(files[1]),'<previous instrumenter>','exec'),prev)
    for fn,kw in [('make_h4_exploratory_source',{}),
                  ('make_h4_exploratory_source',{'enable_h4cc':True}),('make_h4m6_source',{})]:
        require(getattr(gen,fn)(raw,**kw)==prev[fn](raw,**kw),'LEGACY_GENERATOR')
    generated,original=gen.make_h4m8_source(raw)
    require(original==H3_CAUSAL_ORIGINAL_BRAX_SHA and
        generated.count("raise RuntimeError('M470_H4M8_GENERATED_EXECUTION_HOLD')")==1 and
        generated.count('sgd_step, data=learning_data, normalizer_params=normalizer_params')==1,'GENERATED_HOLD')
    compile(generated,'<CAM generated>','exec')
    require(not H4M8_EXECUTION_AUTHORIZED and not any(x in sys.modules for x in
        ('jax','mujoco','brax.training.agents.ppo.train')),'NO_NATIVE_IMPORT')
    return {'decision':'HELD_CORE_STATIC_PASS','implementation_complete':False,
        'original_brax_sha256':original,'generated_sha256':hashlib.sha256(generated.encode()).hexdigest(),
        'files_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in src.items()},
        'window_rows':len(rows),'nonzero_reward_rows':int(mask.sum()),
        'physical_transitions':0,'resets':0,'optimizer_updates':0,
        'runtime_authorized':False,'full_integration_pending':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','smoke','probe2','smoke3','pilot','longtrain','readiness','causal'],required=True)
    p.add_argument('--stage',choices=['easy','easy_repeat','easy_repeat_support','medium','hard'],default='easy')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--restore-report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke-report',type=Path)
    p.add_argument('--rollouts',type=int)
    p.add_argument('--num-evals',type=int)
    p.add_argument('--resume-state',type=Path)
    p.add_argument('--independent-budget',choices=['10M','50M','100M'])
    p.add_argument('--h1-m1',action='store_true',help='H1-M1 experimental reward: smoke3 or independent 10M only')
    p.add_argument('--h1-m2',action='store_true',help='H1-M2 phase/contact foot-crossing reward: smoke3 or independent 10M only; separate owner GO required')
    p.add_argument('--h3-m1',action='store_true',help='H3-M1 independent 10M only with immutable diagnostic/smoke/comparator evidence')
    p.add_argument('--h3-diagnostic-only',action='store_true',
                   help='Historical diagnostic mode (closed; no rerun)')
    p.add_argument('--h3-causal-report',type=Path,
                   help='Exact reviewed causal diagnostic report')
    p.add_argument('--h3-old10m-report',type=Path,
                   help='Exact OLD10M comparator report and adjacent frozen artifacts')
    p.add_argument('--h4-arm',choices=['control','candidate'],help='H4 design plan only; execution HOLD')
    p.add_argument('--h4-control-report',type=Path,help='Future pinned matched control; execution HOLD')
    p.add_argument('--h4-readiness',action='store_true',help='Isolated readiness plan; execution HOLD')
    p.add_argument('--h4-readiness-reward-report',type=Path)
    p.add_argument('--h4-causal',action='store_true',help='Isolated causal plan; execution HOLD')
    p.add_argument('--h4-causal-failure-report',type=Path)
    p.add_argument('--h4-exploratory',action='store_true',help='H4-M1-2R separate Rules-only authorization')
    p.add_argument('--h4-exploratory-causal-report',type=Path)
    p.add_argument('--h4-exploratory-reward-report',type=Path)
    p.add_argument('--h4cc-arm',choices=['ref','ccrc'],help='CCRC separate held authorization')
    p.add_argument('--h4cc-technical-report',type=Path)
    p.add_argument('--h4cc-ref-report',type=Path)
    p.add_argument('--h4m6-arm',choices=['ref','rss'],help='H4-M6 held plan only')
    p.add_argument('--h4m6-technical-report',type=Path)
    p.add_argument('--h4m6-ref-report',type=Path)
    p.add_argument('--h4m8-arm',choices=['cam-ref','cam'],help='H4-M8 held plan only; runtime unavailable')
    a=p.parse_args()
    if a.h4m8_arm is not None:
        try:
            plan=h4m8_held_plan(a)
        except RuntimeError as exc:
            p.error(str(exc))
        print('M470_H4M8_PLAN_NO_GPU',json.dumps(plan,sort_keys=True))
        return
    if a.h4m6_arm is not None:
        try:
            if a.mode!='plan':
                return h4m6_run(a)
            h4m6_require(a.h4m6_technical_report is None and a.h4m6_ref_report is None,'PLAN_EVIDENCE')
            plan=h4m6_validate_held_plan(a)
        except RuntimeError as exc:
            p.error(str(exc))
        print('M470_H4M6_PLAN_NO_GPU',json.dumps(plan,sort_keys=True))
        return
    if a.h4m6_technical_report is not None or a.h4m6_ref_report is not None:
        p.error('M470_H4M6_FLAG_REQUIRED')
    if a.h4cc_arm is not None:
        if a.mode!='plan':
            try:
                h4cc_validate_cli(a)
            except RuntimeError as exc:
                p.error(str(exc))
            return h4cc_run(a)
        try:
            h4cc_validate_held_plan(a)
        except RuntimeError as exc:
            p.error(str(exc))
        print('M470_H4CC_PLAN_NO_GPU',json.dumps({'arm':a.h4cc_arm,'spec':H4CC_SPEC,
              'execution_authorized':False,'training_performed':False},sort_keys=True))
        return
    if a.h4cc_technical_report is not None or a.h4cc_ref_report is not None:
        p.error('M470_H4CC_FLAG_REQUIRED')
    try:
        validate_h4c_cli(a)
        if not a.h4_causal:
            validate_h4r_cli(a)
        validate_h4_cli(a)
    except RuntimeError as exc:
        p.error(str(exc))
    if a.h4_causal:
        if a.mode=='plan':
            print('M470_H4C_PLAN_NO_GPU',json.dumps({'spec':H4C_SPEC,'execution_authorized':False,
                'production_authorized':False},sort_keys=True))
            return
        return h4c_run(a)
    if a.h4_readiness:
        if a.mode=='plan':
            print('M470_H4R_PLAN_NO_GPU',json.dumps({'spec':H4R_SPEC,'execution_authorized':False,
                 'production_authorized':False},sort_keys=True))
            return
        return h4r_run(a)
    if a.h4_exploratory and a.mode=='plan':
        print('M470_H4X_PLAN_NO_GPU',json.dumps({'arm':a.h4_arm,'spec':H4X_SPEC,
              'execution_authorized':False,'training_performed':False},sort_keys=True))
        return
    if a.h4_arm is not None and a.mode=='plan':
        print('M470_H4_PLAN_NO_GPU',json.dumps({'arm':a.h4_arm,'spec':H4_SPEC,
              'execution_authorized':False,'training_performed':False},sort_keys=True))
        return
    if a.independent_budget is not None:
        if (a.mode != 'longtrain' or a.resume_state is not None or
                a.rollouts is not None or a.num_evals is not None):
            p.error('--independent-budget requires longtrain without '
                    '--resume-state/--rollouts/--num-evals')
        a.rollouts, a.num_evals = {
            '10M': (40, 6),
            '50M': (192, 25),
            '100M': (384, 49),
        }[a.independent_budget]
    try:
        validate_h3_cli(a)
    except RuntimeError as exc:
        p.error(str(exc))
    if a.mode=='plan':
        if a.h3_m1:
            print('M470_H3_M1_PLAN_NO_GPU', json.dumps({
                'spec': H3_SPEC, 'stage': a.stage,
                'smoke_steps': 786432, 'smoke_rollouts': 3,
                'independent_steps': 10485760, 'independent_rollouts': 40,
                'independent_evals': 6, 'num_envs': 8192,
                'unroll_length': 32, 'num_minibatches': 8,
                'batch_size': 1024, 'updates_per_batch': 5,
                'fingerprint': H3_FP, 'training_performed': False,
                'execution_authorized': False,
                'ppo_hold': True, 'diagnostic_closed': True,
                'causal_report_flag': '--h3-causal-report',
                'causal_report_sha256': H3_CAUSAL_SHA,
                'diagnostic_max_control_transitions': 54,
            }, sort_keys=True))
            return
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
