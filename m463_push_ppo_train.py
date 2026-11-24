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


def h4_chunk_callback(jax,np,monitor,report,persist,arm):
    """Prepared callback; no H4 runtime is authorized by this source revision.

    The caller must bind the same monitor and report as the future H4 runner.
    """
    chunks=[]
    epoch_counts={}
    all_counts={}
    all_pending={}
    report['h4_chunks']=chunks
    report['h4_epochs']=[]
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


def validate_h4_cli(a):
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
    validate_h4_cli(args)
    h4_arm=getattr(args,'h4_arm',None)
    if h4_arm is not None:
        if not H4_EXECUTION_AUTHORIZED or args.mode!='longtrain':
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
    if h4_arm=='candidate':
        approved_sha=h4_approved_control_sha()
        if args.h4_control_report.is_symlink() or not args.h4_control_report.is_file() or digest(args.h4_control_report)!=approved_sha:
            raise RuntimeError('M470_H4_CONTROL_SHA')
    if args.output_dir.exists():
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
    train_env=CoverageJoystick(config=train_cfg,stage=args.stage,
                               h1_m1=h1_m1,h1_m2=h1_m2,h3_m1=h3_m1,h4_arm=h4_arm)
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
    out.mkdir(parents=True,exist_ok=False)
    instrumented_ppo,instrument_meta=build_instrumented_ppo(
        api['ppo'],out/'m463_brax_ppo_instrumented.py',
        enable_resume=longtrain, enable_h3=h3_m1,enable_h4=h4_arm is not None)
    h4_contract=None
    h4_control=None
    if h4_arm is not None:
        h4_contract=h4_common_contract(scaled.to_dict(),original_reward,instrument_meta)
        if h4_arm=='candidate':
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
        if h3_m1:
            info['h3_m1_enabled'] = True
            info['h3_m1_spec'] = H3_SPEC
        with file.open('xb') as stream:
            stream.write(raw)
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
        cfg['h4_chunk_fn']=h4_chunk_callback(jax,np,monitor,report,persist,h4_arm)
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
                if h4_arm is not None and not H4_EXECUTION_AUTHORIZED:
                    raise RuntimeError('M470_H4_EXECUTION_HOLD')
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','smoke','probe2','smoke3','pilot','longtrain'],required=True)
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
    a=p.parse_args()
    try:
        validate_h4_cli(a)
    except RuntimeError as exc:
        p.error(str(exc))
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
