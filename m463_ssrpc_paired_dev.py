#!/usr/bin/env python3
"""H4-M28-8: held SSRPC paired DEV host contracts.

The historical evaluator is part of the training source digest. Keep it and
the trainer byte-identical; do not monkeypatch historical provenance readers.
Static/native supervisor entry requires distinct root-only Owner markers.
No fixtures or native execution were performed during held preparation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import selectors
import sys
import time
import types
import subprocess
from pathlib import Path

ID = 'H4-M28-SSRPC-PAIRED-DEV-V1'
BASE = '421cd06e6a4b6d088e76085d18ca94cbbe44e4ba'
BRANCH = 'experiment/b1-hardware-scaled'
CLOSED_ROOT = 'e09c35a82f06a97fe5406ba36bb3fd9664913583'
FROZEN = {
    'm463_gate3_cp_eval.py': 'dd0edd9f38dc52f383db2aef2c951fefbc682bbd',
    'm463_push_ppo_train.py': 'c9fe53c2262f1a7c9a4647973f4ab7fec710a4db',
    'm463_coverage_env.py': '3961ec7bc4d7f646a8c7515a0889427826f56f6c',
    'm463_instrument_brax.py': 'e38d4788bca4a7f965cab2f1a0f20f17aff632cc',
}
ARMS = ('ctrl', 'cand')
LABELS = {'ctrl': 'SSRPC_CTRL', 'cand': 'SSRPC_CAND'}
TRAIN = {a: Path('m463_runs/m470_h4_m28_ssrpc_' + a + '_v2') for a in ARMS}
PAIR = Path('m463_runs/m470_h4_m28_ssrpc_paired_dev_v1')
STATIC = Path('m463_runs/m470_h4_m28_ssrpc_paired_dev_static_v1')
OUTER = Path('m463_runs/m470_h4_m28_ssrpc_paired_dev_v1_outer.log')
STATIC_OUTER = Path('m463_runs/m470_h4_m28_ssrpc_paired_dev_static_v1_outer.log')
POLICY = 'm463_easy_repeat_support_longtrain_step_015728640.msgpack'
FULL = 'm470_fullstate_step_015728640.msgpack'
META = 'm470_fullstate_step_015728640.json'
OFFICIAL = '32bf04d64bafd7e80b250b862493b54a7c8db6af1bcf1010bf0df984a904f40c'
M12 = 'eb44c364983078fc35b89152d7a42eae310ed31caba6d4a9b4abf335027826b1'
PARENT = '9b19312da08d71513ad8fa6638c632401bec20e09399fdf4a9f6f6e2b2619c80'
STATIC_TRAIN_SHA = '2cb566ab9a9c58f0aef95f3752a43a073098cdf8e417875c9fc557001cc22069'
PINS = {
    'ctrl': {
        'attempt.json': 'bb501e3c4f2640fae94b0f6a03c9be53fe4f79b82f70f1ce8abf59e0888fb9a2',
        'dispatch.started.json': '1ec12369f89baa35822ec545e84ac21c1bb4b2d6d68c47ebe2975b5880ff9a2b',
        'worker.started.json': '6c7049c8c30c3b17ac843ae5c618c3a97bd0c0956137f22b6a00e19f3d2ce1af',
        'worker.log': '1260b68cce1b637a5ab9997cae5f5dbe273d3897494c9acfbb2e97c37cf0d284',
        'report.json': '93f84d08c3244560ae75a293c5bd321c4b78c8c609f31964a2c79fe0e74f8cf7',
        'attempt_result.json': '1f0f7f93ea59b6cc11d0c9e90f870fb77ea4447cce5fe687def36f9e02a646b5',
        'supervisor.terminal.json': '585189f2c4e10ddab845ab942708a0f2554fc7bec96a2ddea7fcec3c1af36e21',
        META: '3d02bf8fe7cd441f08333b90b328607b1416e5c9dba791902ab855142b9c596c',
        FULL: '4b541169784ebdc3126515b3e0ab7e07dd68f1e461ce177931037a90738b0f71',
        POLICY: '6ed43e4d6d05853622ba82b740c9c28a58930b1a4ec137be4b660f7a95034c05',
        'm463_brax_ppo_instrumented.py': '688cfd439596e08b8ba9d32f40080bb5cea9151df8ee4dee72157a453c6f42dc',
    },
    'cand': {
        'attempt.json': 'b359d989c90abab206a4647e1c71f9d42402bac45964abe3df454f854b046360',
        'dispatch.started.json': '4abfd0fe034bf4e0e6ba6453c412a7867af9ac88d21a08c0eb34bce40926f10b',
        'worker.started.json': '82682aa50296acd711648ffeac299dfdba0926cff74b5a4ff3a51475fbf11e02',
        'worker.log': '1260b68cce1b637a5ab9997cae5f5dbe273d3897494c9acfbb2e97c37cf0d284',
        'report.json': 'f1f2f64bb8b1687b7ae9a566009829c2d30f295ceae2476d1f3ae99fc7d760e2',
        'attempt_result.json': 'a0c0d2acfe8c2322233bbd9bf4787f6a19015cd2551a3de3d9320a925e947aec',
        'supervisor.terminal.json': 'd6c979a8be0c7d9ca0b695164ecd4dee5a974c157a97e878ed9f3f150a6a5aff',
        META: 'ce0848c433f2c6666d35b331e4f716b4c9cf7c3b5ccaf4d2b2429cf3642c99a2',
        FULL: '9184c190237585be3cf01f8b52950fb9c7b0b97730985f6a784d18ec23a5a953',
        POLICY: 'b6feb836049df777c6a40fa411f29365035d567d623d02b90c06f6c0a156634b',
        'm463_brax_ppo_instrumented.py': 'df7b511e01b442744fb45425435bb1b81bfcfd508911e8c86ddedf142d14f10f',
    },
}
OUTER_PINS = {
    'ctrl': 'c629707a02bed136d93c14f4ed49cbb417b0ba1a4999363680d171633e92ee10',
    'cand': 'db26b6594a6acc9b59341f2a35d2b14e60c129acc795d6ed2ba586d69f17a6d2',
}
SPEC = dict(groups=24, logical=120, physical=89616, resets=24,
    per_arm_groups=12, per_arm_logical=60, per_arm_physical=44808, per_arm_resets=12,
    seeds=[7331,7357,7601,7613], profiles=['stand','forward_fast','turn_left'],
    directions=['left','right'], push_step=300, push_magnitude_mps=.50,
    second_push_step=335, second_push_magnitude_mps=.35, max_steps=1000, dt_s=.02,
    deterministic_actor=True, domain_randomization_enabled=False,
    random_source_push_enabled=False, original_reward_PD_termination=True,
    optimizer_updates=0, training_performed=False, promotion_authorized=False,
    holdout_authorized=False)
REPORT_CAP = 128*1024**2  # each historical single-policy report cap
LOG_CAP = 16*1024**2
RESERVE = 2*1024**3
PAIR_STORAGE_CAP = 512*1024**2
STATIC_STORAGE_CAP = 32*1024**2
WATCHDOG = 3600
STATIC_WATCHDOG = 3600
ROOT_RULES = 'G1_RESEARCH_RULES.md'
STATIC_MARKER = 'H4SSRPC_PAIR_STATIC_GO:'
NATIVE_MARKER = 'H4SSRPC_PAIR_ACTIVE_GO:'


def require(ok, label):
    if not ok:
        raise RuntimeError('H4SSRPC_PAIRED_DEV_' + label)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, timeout=30)


def blob(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def pinned_file(root, rel, expected, trainer):
    p = root/rel
    require(p.absolute()==p.resolve() and p.is_file() and not p.is_symlink(),
            'INPUT_PATH_'+str(rel))
    trainer.h4x_file_pin(p, expected)
    return expected


def prepare_inputs(root):
    """Future host-only admission. Strict historical replay, not native work."""
    root = Path(root).resolve()
    require(root == Path(__file__).resolve().parent and git(root,'branch','--show-current').decode().strip()==BRANCH
            and not git(root,'status','--porcelain','--untracked-files=no').strip(), 'TRACKED_STATE')
    pins = {}
    for name, expected in FROZEN.items():
        p = root/name
        raw = p.read_bytes()
        require(p.resolve()==p.absolute() and blob(raw)==expected
                and raw==git(root,'show',BASE+':'+name)
                and raw==git(root,'show','HEAD:'+name), 'FROZEN_'+name)
        pins['source/'+name] = hashlib.sha256(raw).hexdigest()
    self_path = Path(__file__).resolve()
    require(self_path.parent==root and self_path.read_bytes()==git(root,'show','HEAD:'+self_path.name),
            'WRAPPER_TRACKED')
    pins['source/'+self_path.name] = hashlib.sha256(self_path.read_bytes()).hexdigest()
    import m463_gate3_cp_eval as e
    import m463_push_ppo_train as t
    require(Path(e.__file__).resolve()==root/'m463_gate3_cp_eval.py'
            and Path(t.__file__).resolve()==root/'m463_push_ppo_train.py', 'MODULE_ORIGINS')
    e.h4m3_no_runtime_imports()
    pins['historical_bootstrap'] = t.ssrpc_bootstrap_identity(root)
    require(root==e.H4_EVAL_ROOT, 'EVALUATOR_ROOT')
    reports = {}
    for arm in ARMS:
        out = TRAIN[arm]
        for name, digest in PINS[arm].items():
            pins[arm+'/'+name] = pinned_file(root, out/name, digest, t)
        pins[arm+'/outer.log'] = pinned_file(root, Path(str(out)+'_outer.log'), OUTER_PINS[arm], t)
        auth = t.ssrpc_read_json(root/out/'attempt.json',65536)['authorization']
        require(auth['arm']==arm and auth['output']==str(out)
                and auth['static_sha256']==STATIC_TRAIN_SHA, 'TRAIN_AUTH_'+arm)
        # No new evaluator edits or bypasses: existing historical reader enforces
        # source ancestry, original static receipt, parent, packed replay and terminal.
        r = t.ssrpc_read_final(root, PINS[arm]['report.json'], auth)
        expected = 'MATCHED_CONTROL_COMPLETE' if arm=='ctrl' else 'EXPLORATORY_EXPOSURE_PASS'
        require(r['status']=='TECHNICAL_PASS' and r['ssrpc_ledger']['decision']==expected
                and r['ssrpc_ledger']['common_floors'] is True
                and r['promotion_authorized'] is False
                and r['artifacts'][POLICY]==PINS[arm][POLICY]
                and r['artifacts'][FULL]==PINS[arm][FULL]
                and r['parent']['parent_pins']['m470_fullstate_step_010485760.msgpack']==PARENT,
                'TRAIN_FINAL_'+arm)
        reports[arm] = r
    c, a = (reports[k] for k in ARMS)
    require(c['parent']==a['parent'] and c['hardware_scaled_config']==a['hardware_scaled_config']
            and c['reward']==a['reward'] and c['dependencies']==a['dependencies']
            and c['budget']==a['budget']
            and a['authorization']['ctrl_report_sha256']==PINS['ctrl']['report.json']
            and a['authorization']['ctrl_terminal_sha256']==PINS['ctrl']['supervisor.terminal.json'],
            'MATCHED_TRAINING')
    require(e.H4CC_ARCHIVE_SHA==OFFICIAL and e.H4RED_EVAL_M12_PINS['report.json']==M12,
            'ARCHIVE_CONSTANTS')
    archive = e.h4cc_read_archive(root/e.H4_EVAL_OUTPUT)
    pins['archive/report.json'] = OFFICIAL
    pins.update(e.h4red_eval_archive_pins(root))
    secondary = e.h4red_eval_secondary(root, archive)
    for k in e.H4M12_PROTOCOL:
        if k in SPEC:
            require(archive[k]==SPEC[k], 'ARCHIVE_PROTOCOL_'+k)
    require(tuple(archive['frame_columns'])==e.H4M3_COLUMNS
            and archive['source_commit']==e.PIN and archive['brax_train_sha256']==e.TRAIN_SHA
            and archive['jax_version']=='0.4.35', 'ARCHIVE_SCHEMA')
    for name in ('train.py','paper_config.py','m462_ppo_warmstart_run.py','m462_typed_orbax_stage.py'):
        pins['source/'+name] = pinned_file(root, Path(name), archive['input_sha256']['source/'+name], t)
    # Check module origins before any evaluator runtime imports; local files are pinned by archive.
    from importlib.util import find_spec
    for name in ('train','paper_config','m462_ppo_warmstart_run','m462_typed_orbax_stage'):
        origin=find_spec(name)
        require(origin is not None and origin.origin is not None and Path(origin.origin).resolve()==root/(name+'.py'), 'LOCAL_MODULE_'+name)
    pins['source/paper_config_preflight.json'] = pinned_file(root, Path('paper_config_preflight.json'), t.H4R_CONFIG_SHA, t)
    train = Path(archive['brax_source_verification']['path'])
    t.h4x_file_pin(train,e.TRAIN_SHA)
    pins['brax_train']=e.TRAIN_SHA
    br=train.resolve().parents[3]
    for name,entry in c['dependencies'].items():
        p=Path(entry['path'])
        require(p.resolve()==br.joinpath(*name.split('.')[1:]).with_suffix('.py')
                and p.is_file() and e.h4m3_git_blob(p)==entry['blob'], 'DEP_'+name)
        pins['dependency/'+name]=e.sha256(p)
    wrapper=br/'envs/wrappers/training.py'
    require(wrapper.is_file() and e.h4m3_git_blob(wrapper)=='2c09c8268a29d14c832dd07050575e4541ee98af','EPISODE_WRAPPER')
    pins['dependency/episode_wrapper']=e.sha256(wrapper)
    external=e.h4m3_external_sources(root)
    e.h4m3_no_runtime_imports()
    require(t.ssrpc_static_no_native(),'HOST_NATIVE_IMPORT')
    return pins, archive, secondary, external


def arm_spec(arm):
    require(arm in ARMS, 'ARM')
    return dict(SPEC, id=ID, arm=arm, policy=LABELS[arm],
                output=str(PAIR/arm/'report.json'))


def compatibility_view(e, arm, fresh):
    """Adapt only report labels, not metrics, trajectories or scientific gates."""
    require(arm in ARMS and fresh['spec']==arm_spec(arm)
            and all(g['policy']==LABELS[arm] for g in fresh['groups']), 'ARM_IDENTITY')
    view=dict(fresh,groups=[dict(g,policy='M12') for g in fresh['groups']])
    ledger=e.h4m12_ledger(view)
    require(ledger['complete_groups']==12 and ledger['logical']==60
            and ledger['reset_lanes']==12 and ledger['physical']<=44808, 'ARM_LEDGER')
    require(tuple(fresh['frame_columns'])==e.H4M3_COLUMNS, 'FRAME_SCHEMA')
    for g in fresh['groups']:
        e.h4m12_validate_group(g,fresh['frame_columns'])
    return view,ledger


def paired_gate(official_cand, pair):
    """Only the prospective incremental gates; never replace official h3_decide."""
    primary=pair['primary_stand_double']
    values=[primary[m][a] for m in ('survival','recovery5') for a in ('CTRL','CAND')]
    require(all(type(v) is int and 0<=v<=8 for v in values), 'PRIMARY_RANGE')
    nominal=pair['nominal_survivors']
    require(set(nominal)=={'CTRL','CAND'} and all(type(v) is int and 0<=v<=12 for v in nominal.values()),
            'NOMINAL_RANGE')
    moving=pair['moving_double_recovery5']
    require(set(moving)=={'forward_fast','turn_left'}
            and all(set(v)=={'CTRL','CAND'} and all(type(n) is int and 0<=n<=8 for n in v.values())
                    for v in moving.values()), 'MOVING_RANGE')
    require(official_cand in ('DEV_PASS','NO_GO','INCONCLUSIVE'), 'OFFICIAL_DECISION')
    gates=dict(nominal=nominal=={'CTRL':12,'CAND':12},
               primary_nonlower=all(primary[m]['CAND']>=primary[m]['CTRL'] for m in ('survival','recovery5')),
               primary_strict_gain=any(primary[m]['CAND']>primary[m]['CTRL'] for m in ('survival','recovery5')),
               moving_nonloss=all(v['CAND']>=v['CTRL'] for v in moving.values()))
    ceiling=all(v==8 for v in values)
    if official_cand=='NO_GO' or not all(gates[k] for k in ('nominal','primary_nonlower','moving_nonloss')):
        decision='NO_GO'
    elif official_cand=='INCONCLUSIVE' or ceiling:
        decision='INCONCLUSIVE'
    else:
        decision='DEV_PASS' if gates['primary_strict_gain'] else 'NO_GO'
    return dict(decision=decision, gates=gates, ceiling=ceiling,
                promotion_authorized=False, holdout_authorized=False, further_training_authorized=False)


def decide_pair(archive, secondary, ctrl, cand):
    """Future offline reader computation. Caller must verify technical receipts."""
    import m463_gate3_cp_eval as e
    views, ledgers, official, descriptive = {}, {}, {}, {}
    for arm,fresh in zip(ARMS,(ctrl,cand)):
        view,ledger=compatibility_view(e,arm,fresh)
        require(all(archive[k]==fresh[k] for k in e.H4M12_PROTOCOL), 'PAIR_PROTOCOL_'+arm)
        views[arm],ledgers[arm]=view,ledger
        official[arm]=e.h4m12_decide(archive,view)
        official[arm]['comparison_roles']['CAND']=LABELS[arm]
        descriptive[arm]=e.h4red_eval_descriptive(secondary,view,red=False)
        descriptive[arm]['comparison_roles']['CAND']=LABELS[arm]
    total=sum(v['physical'] for v in ledgers.values())
    require(total<=SPEC['physical'] and sum(v['logical'] for v in ledgers.values())==120
            and sum(v['reset_lanes'] for v in ledgers.values())==24, 'PAIR_LEDGER')
    groups=[dict(g,policy=label) for arm,label in zip(ARMS,('CTRL','CAND')) for g in views[arm]['groups']]
    paired_view=dict(cand,groups=groups,logical_trajectories_completed=120,
                     physical_transitions_attempted=total,physical_transitions_recorded=total)
    paired=e.h3_decide(paired_view,policies=e.H4_EVAL_POLICIES,comparator='CTRL',candidate='CAND',
                       logical_cap=120,physical_cap=89616)
    result=paired_gate(official['cand']['decision'],paired)
    return dict(result, id=ID, official=official, secondary_descriptive=descriptive,
                matched_pair_metrics=paired, matched_pair_roles=dict(CTRL=LABELS['ctrl'],CAND=LABELS['cand']),
                matched_pair_h3_verdict_descriptive=True, arm_ledgers=ledgers,
                physical=total, logical=120, resets=24, optimizer_updates=0)



# Held future receipt schema. These readers do NOT issue an authorization.
# A separately reviewed supervisor must first validate the root-only marker,
# fresh static receipt, exact source and local provenance before reservation.
RECEIPT_CAP = 65536
PAIR_REPORT_CAP = 16*1024**2
AUTH_KEYS = {'id','phase','source_head','source_blob','source_sha256','provenance_sha256',
             'static_sha256','static_terminal_sha256','authorization_head'}
STATIC_KEYS = {'id','phase','source_head','source_blob','source_sha256','authorization_head'}


def read_bounded(root, rel, cap, trainer):
    p=root/rel
    require(p.absolute()==p.resolve() and p.is_file() and not p.is_symlink()
            and 0<p.stat().st_size<=cap, 'RECEIPT_PATH_'+str(rel))
    return trainer.ssrpc_read_json(p,cap)


def check_auth_identity(auth, pins, external):
    import m463_gate3_cp_eval as e
    require(type(auth) is dict and set(auth)==AUTH_KEYS and auth['id']==ID and auth['phase']=='native'
            and all(type(v) is str for v in auth.values()), 'AUTH_SCHEMA')
    require(all(re.fullmatch('[0-9a-f]{64}',auth[k]) for k in
                ('source_sha256','provenance_sha256','static_sha256','static_terminal_sha256'))
            and all(re.fullmatch('[0-9a-f]{40}',auth[k]) for k in
                ('source_head','source_blob','authorization_head')), 'AUTH_DIGESTS')
    require(auth['source_sha256']==pins['source/'+Path(__file__).name]
            and auth['source_blob']==blob(Path(__file__).read_bytes())
            and auth['provenance_sha256']==e.h4m3_digest([pins,external]), 'AUTH_BINDING')


def read_arm(root, arm, auth, pins, external, archive, *, complete, terminal_required=True):
    """Held strict final/partial receipt reader; no native execution."""
    import m463_gate3_cp_eval as e
    import m463_push_ppo_train as t
    root=Path(root).resolve()
    require(arm in ARMS and type(complete) is bool, 'READ_ARM')
    check_auth_identity(auth,pins,external)
    base=PAIR/arm
    out=root/base
    require(out.absolute()==out.resolve() and out.is_dir() and not out.is_symlink(), 'ARM_NAMESPACE')
    identity=dict(authorization=auth,arm=arm)
    attempt=read_bounded(root,base/'attempt.json',RECEIPT_CAP,t)
    require(set(attempt)==set(identity)|{'reserved','parent_pid'}
            and all(attempt[k]==v for k,v in identity.items())
            and attempt['reserved'] is True and type(attempt['parent_pid']) is int
            and attempt['parent_pid']>0,'ATTEMPT')
    worker=read_bounded(root,base/'worker.started.json',RECEIPT_CAP,t)
    require(set(worker)==set(identity)|{'pid','parent_pid'} and all(worker[k]==v for k,v in identity.items())
            and type(worker['pid']) is int and worker['pid']>0 and worker['pid']!=attempt['parent_pid']
            and worker['parent_pid']==attempt['parent_pid'], 'WORKER')
    report=read_bounded(root,base/'report.json',REPORT_CAP,t)
    require(report['authorization']==auth and report['spec']==arm_spec(arm)
            and report['input_sha256']==pins
            and report['evaluator_sha256']==pins['source/m463_gate3_cp_eval.py']
            and report['wrapper_sha256']==pins['source/'+Path(__file__).name]
            and report['promotion_authorized'] is False and report['further_training_authorized'] is False
            and report['holdout_authorized'] is False and report['training_performed'] is False
            and all(g['policy']==LABELS[arm] for g in report['groups']), 'REPORT_IDENTITY')
    dispatch=root/base/'dispatch.started.json'
    if complete or dispatch.exists() or dispatch.is_symlink():
        require(read_bounded(root,base/'dispatch.started.json',RECEIPT_CAP,t)==identity, 'DISPATCH')
    result_path=root/base/'attempt_result.json'
    terminal_path=root/base/'supervisor.terminal.json'
    failure_path=root/base/'supervisor.failed.json'
    log=root/base/'worker.log'
    if complete:
        require(report['status']=='COMPLETE' and not failure_path.exists()
                and not failure_path.is_symlink(), 'COMPLETE_STATUS')
        view, ledger=compatibility_view(e,arm,report)
        require(report['ledger']==ledger and all(archive[k]==report[k] for k in e.H4M12_PROTOCOL)
                and report['decision']==e.h4m12_decide(archive,view)
                and report['brax_source_verification']==archive['brax_source_verification']
                and report['mjx_sources']==external['mjx']
                and set(report['playground_runtime'])=={'joystick','mjx_env'},
                'COMPLETE_REPLAY')
        gpu=report['gpu']
        require(gpu['total_mib']==12288 and 0<gpu['peak_used_mib']<.95*12288
                and type(gpu['samples']) is int and gpu['samples']>0 and not gpu['sampler_errors'],
                'GPU')
        digest=e.sha256(out/'report.json')
        expected=dict(identity,status='COMPLETE',report_sha256=digest,producer_rc=0)
        require(exact_json(read_bounded(root,base/'attempt_result.json',RECEIPT_CAP,t),expected),'RESULT')
        require(log.absolute()==log.resolve() and log.is_file() and not log.is_symlink()
                and log.stat().st_size<=LOG_CAP,'LOG')
        expected=dict(expected,reader_rc=0,worker_pid=worker['pid'],parent_pid=attempt['parent_pid'],
            worker_log_sha256=e.sha256(log),stored_log_bytes=log.stat().st_size)
        if terminal_required:
            require(exact_json(read_bounded(root,base/'supervisor.terminal.json',RECEIPT_CAP,t),expected),'TERMINAL')
        else:
            require(unused(terminal_path),'PRETERMINAL')
    else:
        require(report['status'] in ('IN_PROGRESS','STOP') and not terminal_path.exists()
                and not terminal_path.is_symlink(), 'PARTIAL_STATUS')
        # Reuse the historical partial row/flag/ledger checker on a temporary
        # compatibility view; never write it to disk or change the source report.
        view=dict(report,groups=[dict(g,policy='M12') for g in report['groups']],
                  status='M470_H4M12_DEV_'+report['status'],spec=e.H4M12_EVAL_SPEC)
        ledger=e.h4red_eval_partial_check(view,auth,pins,red=False)
        if result_path.exists() or result_path.is_symlink():
            require(read_bounded(root,base/'attempt_result.json',RECEIPT_CAP,t)==
                    dict(identity,status='STOP'), 'PARTIAL_RESULT')
    e.h4m3_no_runtime_imports()
    return report,ledger


def read_pair(root, auth, pins, archive, secondary, external):
    """Complete pair only. Recompute all decisions; no single-arm promotion."""
    import m463_gate3_cp_eval as e
    import m463_push_ppo_train as t
    root=Path(root).resolve()
    check_auth_identity(auth,pins,external)
    c,_=read_arm(root,'ctrl',auth,pins,external,archive,complete=True)
    a,_=read_arm(root,'cand',auth,pins,external,archive,complete=True)
    result=decide_pair(archive,secondary,c,a)
    expected=dict(authorization=auth, status='COMPLETE',
        arm_report_sha256={arm:e.sha256(root/PAIR/arm/'report.json') for arm in ARMS},
        arm_terminal_sha256={arm:e.sha256(root/PAIR/arm/'supervisor.terminal.json') for arm in ARMS},
        result=result)
    require(exact_json(read_bounded(root,PAIR/'report.json',PAIR_REPORT_CAP,t),expected),'PAIR_FINAL_REPLAY')
    require(not (root/PAIR/'supervisor.failed.json').exists()
            and not (root/PAIR/'supervisor.failed.json').is_symlink(), 'PAIR_FAILURE')
    pair_attempt=read_bounded(root,PAIR/'attempt.json',RECEIPT_CAP,t)
    require(set(pair_attempt)=={'authorization','reserved','parent_pid'} and pair_attempt['authorization']==auth
            and pair_attempt['reserved'] is True and type(pair_attempt['parent_pid']) is int
            and pair_attempt['parent_pid']>0,'PAIR_ATTEMPT')
    workers=[]
    for arm in ARMS:
        require(read_bounded(root,PAIR/arm/'attempt.json',RECEIPT_CAP,t)['parent_pid']==pair_attempt['parent_pid'], 'PAIR_PARENT')
        workers.append(read_bounded(root,PAIR/arm/'worker.started.json',RECEIPT_CAP,t)['pid'])
    require(len(set(workers))==2,'PAIR_FRESH_WORKERS')
    terminal=dict(authorization=auth,status='COMPLETE',report_sha256=e.sha256(root/PAIR/'report.json'),
                  producer_rc=0,reader_rc=0,arm_terminal_sha256=expected['arm_terminal_sha256'])
    require(exact_json(read_bounded(root,PAIR/'attempt_result.json',RECEIPT_CAP,t),{k:v for k,v in terminal.items() if k!='reader_rc'}), 'PAIR_RESULT')
    require(exact_json(read_bounded(root,PAIR/'supervisor.terminal.json',RECEIPT_CAP,t),terminal),'PAIR_TERMINAL')
    return expected


def gate_fixture_cases():
    """Prospective focused fixture definitions; not executed in H4-M28-8."""
    def row(cs,cr,ts,tr,nominal=12,moving=8):
        return dict(primary_stand_double={'survival':{'CTRL':cs,'CAND':ts},
                    'recovery5':{'CTRL':cr,'CAND':tr}},
                    nominal_survivors={'CTRL':12,'CAND':nominal},
                    moving_double_recovery5={'forward_fast':{'CTRL':8,'CAND':moving},
                                            'turn_left':{'CTRL':8,'CAND':8}})
    return [
        ('strict_gain','DEV_PASS',row(6,6,8,8),'DEV_PASS'),
        ('one_metric_gain','DEV_PASS',row(8,7,8,8),'DEV_PASS'),
        ('both_ceiling','DEV_PASS',row(8,8,8,8),'INCONCLUSIVE'),
        ('official_failure','NO_GO',row(6,6,8,8),'NO_GO'),
        ('official_missing_window','INCONCLUSIVE',row(6,6,8,8),'INCONCLUSIVE'),
        ('nominal_loss','DEV_PASS',row(6,6,8,8,nominal=11),'NO_GO'),
        ('moving_loss','DEV_PASS',row(6,6,8,8,moving=7),'NO_GO'),
        ('primary_loss','DEV_PASS',row(8,8,7,8),'NO_GO'),
        ('no_increment','DEV_PASS',row(7,7,7,7),'NO_GO'),
    ]



def exact_json(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def unused(path):
    return not path.exists() and not path.is_symlink()


def parse_marker(raw):
    """Exact single active marker; the HEAD itself is derived from Git."""
    lines=[v for v in raw.decode('utf8').splitlines() if re.match(r'^[A-Z0-9_]+_GO:',v)]
    require(len(lines)==1,'MARKER_COUNT')
    m=re.fullmatch(r'H4SSRPC_PAIR_(STATIC|ACTIVE)_GO: ([0-9a-f]{40}) ([0-9a-f]{40}) ([0-9a-f]{64})(?: ([0-9a-f]{64}) ([0-9a-f]{64}) ([0-9a-f]{64}))?',lines[0])
    require(m is not None,'MARKER_SCHEMA')
    phase,head,source_blob,source_sha,prov,static_sha,terminal_sha=m.groups()
    require((phase=='STATIC' and prov is None) or (phase=='ACTIVE' and prov is not None),'MARKER_PHASE')
    a=dict(id=ID,phase='static' if phase=='STATIC' else 'native',source_head=head,
           source_blob=source_blob,source_sha256=source_sha)
    if phase=='ACTIVE':
        a.update(provenance_sha256=prov,static_sha256=static_sha,static_terminal_sha256=terminal_sha)
    return a


def authorization(root,phase,*,historical=None):
    head=historical or git(root,'rev-parse','HEAD').decode().strip()
    raw=git(root,'show',head+':'+ROOT_RULES)
    auth=dict(parse_marker(raw),authorization_head=head)
    require(auth['phase']==phase and set(auth)==(STATIC_KEYS if phase=='static' else AUTH_KEYS),'AUTH_PHASE')
    prior=auth['source_head']
    require(git(root,'rev-list','--parents','-n','1',head).split()==[head.encode(),prior.encode()]
            and git(root,'diff','--name-only',prior,head).splitlines()==[ROOT_RULES.encode()], 'ROOT_ONLY')
    old=git(root,'show',prior+':'+ROOT_RULES)
    marker=[v for v in raw.splitlines(keepends=True) if re.match(rb'^[A-Z0-9_]+_GO:',v)]
    require(blob(old)==CLOSED_ROOT and len(marker)==1 and raw.replace(marker[0],b'',1)==old,'ROOT_EXACT')
    name=Path(__file__).name
    current=(root/name).read_bytes()
    require(blob(current)==auth['source_blob'] and hashlib.sha256(current).hexdigest()==auth['source_sha256']
            and current==git(root,'show',head+':'+name)
            and current==git(root,'show','HEAD:'+name),'SOURCE_AUTH')
    require(subprocess.run(['git','merge-base','--is-ancestor',head,'HEAD'],cwd=root,
                           timeout=30).returncode==0,'AUTH_ANCESTRY')
    if historical is None:
        require((root/ROOT_RULES).read_bytes()==raw
                and git(root,'branch','--show-current').decode().strip()==BRANCH
                and not git(root,'status','--porcelain','--untracked-files=no').strip(),'AUTH_LIVE')
    return auth


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')


def phase_dir(phase):
    require(phase in ('static','native'),'PHASE')
    return STATIC if phase=='static' else PAIR


def storage(root, phase, extra=0, initial=False):
    require(type(extra) is int and extra>=0,'STORAGE_EXTRA')
    directory=root/phase_dir(phase)
    outer=root/(STATIC_OUTER if phase=='static' else OUTER)
    cap=STATIC_STORAGE_CAP if phase=='static' else PAIR_STORAGE_CAP
    parent=directory.parent
    require(parent.is_dir() and parent.absolute()==parent.resolve(),'STORAGE_PARENT')
    total=0
    if not initial:
        require(directory.is_dir() and directory.absolute()==directory.resolve(),'STORAGE_ROOT')
        for p in directory.rglob('*'):
            require(not p.is_symlink(),'STORAGE_SYMLINK')
            if p.is_dir():continue
            require(p.is_file(),'STORAGE_FILE')
            total+=p.stat().st_size
        if outer.exists():
            require(outer.is_file() and not outer.is_symlink() and outer.stat().st_size<=RECEIPT_CAP,'OUTER_CAP')
            total+=outer.stat().st_size
    require(total+extra<=cap and shutil.disk_usage(parent).free>=RESERVE+(cap if initial else extra),'STORAGE_BOUND')


def write_json(root,phase,path,value,*,exclusive=True,cap=RECEIPT_CAP):
    path=Path(path)
    raw=json_bytes(value)
    require(path.absolute()==path.resolve() and path.parent.is_dir() and
            path.is_relative_to(root/phase_dir(phase)) and 0<len(raw)<=cap,'WRITE_PATH_CAP')
    storage(root,phase,len(raw))
    if exclusive:
        target=path
    else:
        require(path.is_file() and not path.is_symlink(),'WRITE_BASE')
        target=path.with_suffix(path.suffix+'.tmp')
    with target.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    if not exclusive:os.replace(target,path)


def report_view(session,arm):
    return dict(session,groups=[dict(g,policy=LABELS[arm]) for g in session['groups']])


def isolated_graph(e, *, enter, live_auth, train_dir, writer):
    """Reuse original code objects; only private routing/receipt globals differ."""
    ns=dict(e.run.__globals__)
    original=e.run.__globals__
    require(original is e.h3_group.__globals__,'GRAPH_GLOBALS')
    def clone(fn):
        copy=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        copy.__kwdefaults__=fn.__kwdefaults__
        return copy
    ns.update(h4m12_enter=enter,h4m12_eval_auth=live_auth,
              h4r11_dev_train_dir=train_dir,h4m12_write=writer,H4M12_POLICY=POLICY)
    ns['h3_group']=clone(e.h3_group)
    run=clone(e.run)
    allowed={'h4m12_enter','h4m12_eval_auth','h4r11_dev_train_dir','h4m12_write','H4M12_POLICY','h3_group'}
    require(set(ns)==set(original) and all(ns[k] is original[k] for k in original if k not in allowed)
            and run.__code__ is e.run.__code__ and ns['h3_group'].__code__ is e.h3_group.__code__
            and ns['h3_decide'] is e.h3_decide and ns['h4_shared_frozen_loader'] is e.h4_shared_frozen_loader,
            'GRAPH_IDENTITY')
    return run


def native_worker(root,auth,arm,pins,archive,external):
    import m463_gate3_cp_eval as e
    import m463_push_ppo_train as t
    from m462_ppo_warmstart_run import GPUMonitor,sample_gpu
    base=root/PAIR/arm
    identity=dict(authorization=auth,arm=arm)
    session={k:SPEC[k] for k in e.H4M12_PROTOCOL if k in SPEC}
    session.update(status='IN_PROGRESS',authorization=auth,spec=arm_spec(arm),
        source_commit=e.PIN,brax_train_sha256=e.TRAIN_SHA,
        evaluator_sha256=pins['source/m463_gate3_cp_eval.py'],
        wrapper_sha256=pins['source/'+Path(__file__).name],input_sha256=pins,
        expected_logical_trajectories=60,max_physical_transitions=44808,
        physical_transitions_attempted=0,physical_transitions_recorded=0,
        logical_trajectories_completed=0,reset_lanes_attempted=0,reset_lanes_recorded=0,
        frame_columns=None,groups=[],promotion_authorized=False,further_training_authorized=False,
        holdout_authorized=False,training_performed=False)
    args=types.SimpleNamespace(h4m12_eval=True,h4red_eval=False,h2a=False,h2b=False,
        output=base/'report.json',seeds=list(SPEC['seeds']),profiles=list(SPEC['profiles']),
        directions=list(SPEC['directions']),push_step=300,push_magnitude=.50,
        second_push_step=335,second_push_magnitude=.35,max_steps=1000,max_cases=60,
        h3_session=session)
    def persist(path,obj,initial=False):
        require(path==args.output and obj is session,'WORKER_WRITE_ROUTE')
        write_json(root,'native',path,report_view(obj,arm),exclusive=initial,cap=REPORT_CAP)
    def live_auth(where,*,red=False):
        require(Path(where)==root and red is False and authorization(root,'native')==auth,'LIVE_AUTH')
        return auth
    def train_dir(red):
        require(red is False,'POLICY_ROUTE')
        pinned_file(root,TRAIN[arm]/POLICY,PINS[arm][POLICY],t)
        return TRAIN[arm]
    def enter(a,*,red=False):
        require(a is args and live_auth(root,red=red)==auth and not session['groups']
                and all(session[k]==0 for k in ('physical_transitions_attempted','physical_transitions_recorded',
                    'logical_trajectories_completed','reset_lanes_attempted','reset_lanes_recorded'))
                and exact_json(read_bounded(root,PAIR/arm/'report.json',REPORT_CAP,t),report_view(session,arm))
                and unused(base/'dispatch.started.json'),'ENTRY')
        e.h4m3_no_runtime_imports()
        a.h4m12_archive=archive;a.h4m12_external=external
        write_json(root,'native',base/'dispatch.started.json',identity)
    monitor=None
    try:
        e.h4m3_no_runtime_imports()
        g=sample_gpu()
        require(g['total_mib']==12288 and 0<=g['used_mib']<.95*12288,'GPU_INITIAL')
        session['gpu_initial']=g
        persist(args.output,session,initial=True)
        run=isolated_graph(e,enter=enter,live_auth=live_auth,train_dir=train_dir,writer=persist)
        monitor=GPUMonitor();args.h3_monitor=monitor
        with monitor:run(args)
        session['gpu']=e.h3_gpu_check(monitor)
        session['ledger']=e.h4m12_ledger(session)
        require(session['decision']==e.h4m12_decide(archive,session),'NATIVE_DECISION_REPLAY')
        require(all(archive[k]==session[k] for k in e.H4M12_PROTOCOL),'NATIVE_PROTOCOL')
        session['status']='COMPLETE'
        persist(args.output,session)
        result=dict(identity,status='COMPLETE',report_sha256=e.sha256(args.output),producer_rc=0)
        write_json(root,'native',base/'attempt_result.json',result)
    except BaseException as exc:
        session['status']='STOP';session['error']=repr(exc)[:4096]
        if monitor is not None:session['gpu']=monitor.snapshot()
        try:
            if not unused(args.output):
                session['partial_ledger']=e.h4m12_ledger(session,partial=True)
                persist(args.output,session)
            write_json(root,'native',base/'attempt_result.json',dict(identity,status='STOP'))
        except BaseException as err:
            print('H4SSRPC_PAIR_PARTIAL_WRITE_FAILED',repr(err),flush=True)
        raise


def fixture_ids():
    return tuple('gate/'+row[0] for row in gate_fixture_cases())+(
        'marker/static','marker/native','marker/none','marker/duplicate','marker/foreign',
        'marker/static_extra','marker/native_missing','marker/bad_hex','marker/bad_spacing',
        'identity/json_bool','identity/json_order','identity/nonfinite','identity/duplicate_json',
        'identity/source_graph','identity/frozen_source','identity/historical_inputs','identity/archives',
        'protocol/budget','protocol/seed_order','protocol/arm_namespace','protocol/arm_policy',
        'gate/invalid_primary','gate/invalid_nominal','gate/invalid_moving','gate/invalid_official',
        'resource/atomic_bound','resource/static_bound','resource/log_bound','resource/disk_reserve',
        'supervisor/ctrl_technical_stop','supervisor/ctrl_scientific_no_go','supervisor/cand_complete',
        'supervisor/receipt_binding','supervisor/exclusive_path','supervisor/worker_parent',
        'host/compile_wrapper','host/compile_frozen','host/strict_read','host/strict_duplicate',
        'host/strict_nonfinite','host/strict_cap','host/exclusive_write','host/atomic_replace',
        'host/symlink_reject','pipe/success','pipe/nonzero','pipe/log_cap','pipe/watchdog',
    )


def expect_reject(fn):
    try:fn()
    except (RuntimeError,ValueError,TypeError,KeyError,FileExistsError):return True
    raise RuntimeError('FIXTURE_EXPECTED_REJECTION')


def next_arm(completed, technical_ok):
    require(tuple(completed) in ((),('ctrl',),('ctrl','cand')) and type(technical_ok) is bool,'SEQUENCE')
    if not technical_ok or len(completed)==2:return None
    return ARMS[len(completed)]


def static_fixtures(root,pins,archive,secondary,external):
    """Execute only under future static GO. No graph call, JAX/MJX imports or PPO."""
    import m463_gate3_cp_eval as e
    import m463_push_ppo_train as t
    results=[]
    def case(name,ok):
        require(ok is True,'FIXTURE_'+name);results.append(name)
    for name,official,metrics,expected in gate_fixture_cases():
        case('gate/'+name,paired_gate(official,metrics)['decision']==expected)
    a='a'*40;b='b'*40;c='c'*64;d='d'*64
    st=(STATIC_MARKER+' '+a+' '+b+' '+c+'\n').encode()
    nt=(NATIVE_MARKER+' '+a+' '+b+' '+c+' '+d+' '+d+' '+d+'\n').encode()
    case('marker/static',parse_marker(st)['phase']=='static')
    case('marker/native',parse_marker(nt)['static_terminal_sha256']==d)
    for name,raw in [('none',b''),('duplicate',st+nt),('foreign',b'OTHER_ACTIVE_GO: x\n'),
        ('static_extra',st.rstrip()+b' '+d.encode()+b' '+d.encode()+b' '+d.encode()),
        ('native_missing',nt.rsplit(b' ',1)[0]),('bad_hex',st.replace(b'a'*40,b'g'*40)),
        ('bad_spacing',st.replace(b': ',b':  '))]:
        case('marker/'+name,expect_reject(lambda:parse_marker(raw)))
    case('identity/json_bool',not exact_json({'x':True},{'x':1}))
    case('identity/json_order',exact_json({'x':1,'y':2},{'y':2,'x':1}))
    case('identity/nonfinite',expect_reject(lambda:json_bytes({'x':float('nan')})))
    def no_dup(pairs):
        out={}
        for key,val in pairs:
            require(key not in out,'DUPLICATE');out[key]=val
        return out
    case('identity/duplicate_json',expect_reject(lambda:json.loads('{"x":1,"x":2}',object_pairs_hook=no_dup)))
    before=dict(e.run.__globals__)
    dummy=lambda *args,**kw:None
    graph=isolated_graph(e,enter=dummy,live_auth=dummy,train_dir=dummy,writer=dummy)
    case('identity/source_graph',graph.__code__ is e.run.__code__ and all(e.run.__globals__[k] is v for k,v in before.items())
         and graph.__globals__['h3_group'].__code__ is e.h3_group.__code__)
    case('identity/frozen_source',all(blob((root/k).read_bytes())==v for k,v in FROZEN.items()))
    case('identity/historical_inputs',all(pins[a+'/'+k]==v for a in ARMS for k,v in PINS[a].items()))
    case('identity/archives',pins['archive/report.json']==OFFICIAL and pins['secondary/report.json']==M12
         and archive['status']=='M470_H4_EVAL_COMPLETED' and secondary['status']=='M470_H4M12_DEV_COMPLETE')
    case('protocol/budget',SPEC['physical']==2*12*3734 and SPEC['logical']==2*12*5 and SPEC['resets']==24 and SPEC['optimizer_updates']==0)
    case('protocol/seed_order',tuple(SPEC['seeds'])==e.H3_EVAL_SEEDS and tuple(SPEC['profiles'])==e.H3_EVAL_PROFILES)
    case('protocol/arm_namespace',arm_spec('ctrl')['output']!=arm_spec('cand')['output'] and STATIC!=PAIR)
    case('protocol/arm_policy',PINS['ctrl'][POLICY]!=PINS['cand'][POLICY] and arm_spec('ctrl')['policy']!=arm_spec('cand')['policy'])
    import copy
    official,metrics=gate_fixture_cases()[0][1:3]
    for label,modify in [('invalid_primary',lambda q:q['primary_stand_double']['survival'].update(CAND=9)),
        ('invalid_nominal',lambda q:q['nominal_survivors'].update(CTRL=True)),
        ('invalid_moving',lambda q:q['moving_double_recovery5']['turn_left'].update(CAND=-1))]:
        q=copy.deepcopy(metrics);modify(q)
        case('gate/'+label,expect_reject(lambda:paired_gate(official,q)))
    case('gate/invalid_official',expect_reject(lambda:paired_gate('PASS',metrics)))
    case('resource/atomic_bound',3*REPORT_CAP+2*LOG_CAP+2*PAIR_REPORT_CAP+32*RECEIPT_CAP<=PAIR_STORAGE_CAP)
    case('resource/static_bound',LOG_CAP+8*RECEIPT_CAP<STATIC_STORAGE_CAP)
    case('resource/log_bound',LOG_CAP==16*1024**2 and WATCHDOG==3600)
    case('resource/disk_reserve',RESERVE==2*1024**3 and shutil.disk_usage(root/PAIR.parent).free>=RESERVE)
    case('supervisor/ctrl_technical_stop',next_arm(('ctrl',),False) is None)
    case('supervisor/ctrl_scientific_no_go',next_arm(('ctrl',),True)=='cand')
    case('supervisor/cand_complete',next_arm(ARMS,True) is None)
    case('supervisor/receipt_binding',not exact_json({'producer_rc':0},{'producer_rc':False}))
    case('supervisor/exclusive_path',unused(root/PAIR) and unused(root/OUTER))
    case('supervisor/worker_parent',expect_reject(lambda:next_arm(('cand',),True)))
    # Real host-only parser/writer/supervisor cases under the separate static GO.
    case('host/compile_wrapper',compile((root/Path(__file__).name).read_bytes(),Path(__file__).name,'exec') is not None)
    case('host/compile_frozen',all(compile((root/name).read_bytes(),name,'exec') is not None for name in FROZEN))
    import tempfile
    with tempfile.TemporaryDirectory(prefix='fixture_',dir=root/STATIC) as tmp:
        d=Path(tmp);rel=d.relative_to(root);p=d/'receipt.json'
        write_json(root,'static',p,{'a':1})
        case('host/strict_read',exact_json(read_bounded(root,rel/'receipt.json',RECEIPT_CAP,t),{'a':1}))
        q=d/'invalid.json';q.write_bytes(b'{"a":1,"a":2}')
        case('host/strict_duplicate',expect_reject(lambda:read_bounded(root,rel/'invalid.json',RECEIPT_CAP,t)))
        q.write_bytes(b'{"a":NaN}')
        case('host/strict_nonfinite',expect_reject(lambda:read_bounded(root,rel/'invalid.json',RECEIPT_CAP,t)))
        case('host/strict_cap',expect_reject(lambda:read_bounded(root,rel/'receipt.json',1,t)))
        case('host/exclusive_write',expect_reject(lambda:write_json(root,'static',p,{'a':2})))
        write_json(root,'static',p,{'a':3},exclusive=False)
        case('host/atomic_replace',exact_json(read_bounded(root,rel/'receipt.json',RECEIPT_CAP,t),{'a':3})
            and unused(p.with_suffix('.json.tmp')))
        link=d/'alias.json';link.symlink_to(p)
        case('host/symlink_reject',expect_reject(lambda:read_bounded(root,rel/'alias.json',RECEIPT_CAP,t)))
        link.unlink()
        def pipe(name,code,seconds=15,cap=1024):
            return bounded_process(root,'static',[sys.executable,'-B','-c',code],d/(name+'.log'),seconds,cap)
        case('pipe/success',type(pipe('success','print("OK")')) is int and (d/'success.log').read_bytes()==b'OK\\n')
        case('pipe/nonzero',expect_reject(lambda:pipe('nonzero','raise SystemExit(7)')))
        case('pipe/log_cap',expect_reject(lambda:pipe('cap','print("x"*2048)',cap=32)) and (d/'cap.log').stat().st_size<=32)
        case('pipe/watchdog',expect_reject(lambda:pipe('watchdog','import time; time.sleep(30)',seconds=.25)))
    require(tuple(results)==fixture_ids(),'FIXTURE_MANIFEST')
    e.h4m3_no_runtime_imports()
    require(t.ssrpc_static_no_native(),'STATIC_NATIVE_IMPORT')
    return results


def static_worker(root,auth,pins,archive,secondary,external):
    import m463_gate3_cp_eval as e
    names=static_fixtures(root,pins,archive,secondary,external)
    value=dict(authorization=auth,status='FULL_HOST_STATIC_PASS',fixtures=names,count=len(names),
        pins=pins,external=external,provenance_sha256=e.h4m3_digest([pins,external]),
        native_execution=0,optimizer_updates=0,promotion_authorized=False)
    write_json(root,'static',root/STATIC/'report.json',value)
    digest=e.sha256(root/STATIC/'report.json')
    write_json(root,'static',root/STATIC/'attempt_result.json',dict(authorization=auth,status='COMPLETE',report_sha256=digest,producer_rc=0))
    print('H4SSRPC_PAIR_FULL_STATIC_PASS '+digest,flush=True)


def read_static(root,pins,external,*,expected=None,terminal_required=True):
    import m463_gate3_cp_eval as e
    import m463_push_ppo_train as t
    out=root/STATIC
    require(unused(out/'supervisor.failed.json'),'STATIC_FAILED')
    report=read_bounded(root,STATIC/'report.json',RECEIPT_CAP,t)
    auth=report['authorization']
    require(auth==authorization(root,'static',historical=auth['authorization_head']),'STATIC_AUTH')
    value=dict(authorization=auth,status='FULL_HOST_STATIC_PASS',fixtures=list(fixture_ids()),count=len(fixture_ids()),
        pins=pins,external=external,provenance_sha256=e.h4m3_digest([pins,external]),
        native_execution=0,optimizer_updates=0,promotion_authorized=False)
    require(exact_json(value,report),'STATIC_REPORT')
    attempt=read_bounded(root,STATIC/'attempt.json',RECEIPT_CAP,t)
    worker=read_bounded(root,STATIC/'worker.started.json',RECEIPT_CAP,t)
    require(set(attempt)=={'authorization','reserved','parent_pid'} and attempt['authorization']==auth
        and attempt['reserved'] is True and type(attempt['parent_pid']) is int and attempt['parent_pid']>0
        and set(worker)=={'authorization','pid','parent_pid'} and worker['authorization']==auth
        and type(worker['pid']) is int and worker['pid']>0 and worker['pid']!=attempt['parent_pid']
        and worker['parent_pid']==attempt['parent_pid'],'STATIC_WORKER')
    digest=e.sha256(out/'report.json')
    result=dict(authorization=auth,status='COMPLETE',report_sha256=digest,producer_rc=0)
    require(exact_json(read_bounded(root,STATIC/'attempt_result.json',RECEIPT_CAP,t),result),'STATIC_RESULT')
    log=out/'worker.log'
    require(log.absolute()==log.resolve() and log.is_file() and 0<log.stat().st_size<=LOG_CAP
        and log.read_bytes().splitlines().count(('H4SSRPC_PAIR_FULL_STATIC_PASS '+digest).encode())==1,'STATIC_LOG')
    terminal=dict(result,reader_rc=0,worker_pid=worker['pid'],parent_pid=attempt['parent_pid'],
        worker_log_sha256=e.sha256(log),stored_log_bytes=log.stat().st_size)
    if terminal_required:
        require(exact_json(read_bounded(root,STATIC/'supervisor.terminal.json',RECEIPT_CAP,t),terminal),'STATIC_TERMINAL')
    if expected is not None:
        require(digest==expected['static_sha256'] and e.sha256(out/'supervisor.terminal.json')==expected['static_terminal_sha256']
                and report['provenance_sha256']==expected['provenance_sha256']
                and auth['source_blob']==expected['source_blob'] and auth['source_sha256']==expected['source_sha256'],'STATIC_NATIVE_BINDING')
    return terminal


def worker_entry(root,phase,arm):
    import m463_push_ppo_train as t
    auth=authorization(root,phase)
    base=phase_dir(phase)/(arm if phase=='native' else '')
    require((phase=='native' and arm in ARMS) or (phase=='static' and arm=='static'),'WORKER_PHASE')
    identity=dict(authorization=auth,**({'arm':arm} if phase=='native' else {}))
    receipt=read_bounded(root,base/'attempt.json',RECEIPT_CAP,t)
    require(exact_json(receipt,dict(identity,reserved=True,parent_pid=os.getppid()))
        and unused(root/base/'worker.started.json'),'WORKER_ADMISSION')
    if phase=='native':
        pair=read_bounded(root,PAIR/'attempt.json',RECEIPT_CAP,t)
        require(exact_json(pair,dict(authorization=auth,reserved=True,parent_pid=os.getppid())),'WORKER_PAIR')
    write_json(root,phase,root/base/'worker.started.json',dict(identity,pid=os.getpid(),parent_pid=os.getppid()))
    pins,archive,secondary,external=prepare_inputs(root)
    if phase=='static':
        static_worker(root,auth,pins,archive,secondary,external)
    else:
        check_auth_identity(auth,pins,external)
        read_static(root,pins,external,expected=auth)
        if arm=='cand':read_arm(root,'ctrl',auth,pins,external,archive,complete=True)
        native_worker(root,auth,arm,pins,archive,external)


def bounded_process(root,phase,argv,log,seconds,log_cap):
    """Shared actual bounded pipe; static fixtures use only stdlib child commands."""
    require(type(seconds) in (int,float) and 0<seconds<=WATCHDOG and type(log_cap) is int and 0<log_cap<=LOG_CAP,
            'CHILD_LIMITS')
    deadline=time.monotonic()+seconds
    with log.open('xb') as stream:
        proc=subprocess.Popen(argv,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        count=0
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(proc.stdout,selectors.EVENT_READ)
                while sel.get_map():
                    require(time.monotonic()<deadline,'WATCHDOG')
                    storage(root,phase)
                    for key,_ in sel.select(timeout=.25):
                        chunk=os.read(key.fd,65536)
                        if not chunk:sel.unregister(key.fileobj);continue
                        require(count+len(chunk)<=log_cap,'LOG_CAP')
                        storage(root,phase,len(chunk))
                        stream.write(chunk);stream.flush();count+=len(chunk)
                rc=proc.wait(timeout=max(.001,deadline-time.monotonic()))
                os.fsync(stream.fileno())
                require(rc==0,'WORKER_EXIT_'+str(rc))
        except BaseException:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=10)
            raise
        finally:proc.stdout.close()
    return proc.pid


def bounded_child(root,phase,arm,base):
    pid=bounded_process(root,phase,[sys.executable,'-B',str(Path(__file__).resolve()),'_worker',phase,arm],
            base/'worker.log',STATIC_WATCHDOG if phase=='static' else WATCHDOG,LOG_CAP)
    import m463_push_ppo_train as t
    rel=phase_dir(phase)/(arm if phase=='native' else '')/'worker.started.json'
    receipt=read_bounded(root,rel,RECEIPT_CAP,t)
    require(type(receipt['pid']) is int and receipt['pid']==pid and receipt['parent_pid']==os.getpid(),'CHILD_PID_BINDING')
    return pid


def supervisor(root,phase):
    import m463_gate3_cp_eval as e
    import m463_push_ppo_train as t
    auth=authorization(root,phase)
    base=root/phase_dir(phase)
    outer=root/(STATIC_OUTER if phase=='static' else OUTER)
    require(unused(base) and unused(outer),'ONE_USE')
    storage(root,phase,initial=True)
    # Reserve outer before host replay; no retry even if later admission fails.
    with outer.open('xb') as f:
        f.write(('AUTHORIZATION_HEAD='+auth['authorization_head']+'\nSTATUS=INVOKED\n').encode())
        f.flush();os.fsync(f.fileno())
        try:
            base.mkdir(exist_ok=False)
            write_json(root,phase,base/'attempt.json',dict(authorization=auth,reserved=True,parent_pid=os.getpid()))
            pins,archive,secondary,external=prepare_inputs(root)
            if phase=='static':
                worker_pid=bounded_child(root,phase,'static',base)
                # Independent host admission and strict result reader, no second fixture execution.
                p,_,_,x=prepare_inputs(root)
                require(exact_json([pins,external],[p,x]),'STATIC_POST_INPUTS')
                terminal=read_static(root,pins,external,terminal_required=False)
                require(terminal['worker_pid']==worker_pid,'STATIC_SUPERVISOR_PID')
                write_json(root,phase,base/'supervisor.terminal.json',terminal)
                read_static(root,pins,external)
            else:
                check_auth_identity(auth,pins,external)
                read_static(root,pins,external,expected=auth)
                completed=[]
                for arm in ARMS:
                    require(next_arm(completed,True)==arm and authorization(root,'native')==auth,'ARM_ORDER')
                    out=base/arm;out.mkdir(exist_ok=False)
                    write_json(root,phase,out/'attempt.json',dict(authorization=auth,arm=arm,reserved=True,parent_pid=os.getpid()))
                    try:
                        worker_pid=bounded_child(root,phase,arm,out)
                    except BaseException as exc:
                        partial_validated=False
                        try:
                            read_arm(root,arm,auth,pins,external,archive,complete=False)
                            partial_validated=True
                        except BaseException as err:
                            print('PAIR_PARTIAL_UNVERIFIED',repr(err),flush=True)
                        write_json(root,phase,out/'supervisor.failed.json',dict(authorization=auth,arm=arm,
                            status='STOP',partial_validated=partial_validated,error=repr(exc)[:4096]))
                        raise
                    p,_,_,x=prepare_inputs(root)
                    require(exact_json([pins,external],[p,x]),'POST_INPUTS')
                    read_arm(root,arm,auth,pins,external,archive,complete=True,terminal_required=False)
                    result=read_bounded(root,PAIR/arm/'attempt_result.json',RECEIPT_CAP,t)
                    log=out/'worker.log'
                    terminal=dict(result,reader_rc=0,worker_pid=worker_pid,parent_pid=os.getpid(),
                        worker_log_sha256=e.sha256(log),stored_log_bytes=log.stat().st_size)
                    write_json(root,phase,out/'supervisor.terminal.json',terminal)
                    read_arm(root,arm,auth,pins,external,archive,complete=True)
                    completed.append(arm)
                c,_=read_arm(root,'ctrl',auth,pins,external,archive,complete=True)
                a,_=read_arm(root,'cand',auth,pins,external,archive,complete=True)
                expected=dict(authorization=auth,status='COMPLETE',
                    arm_report_sha256={arm:e.sha256(root/PAIR/arm/'report.json') for arm in ARMS},
                    arm_terminal_sha256={arm:e.sha256(root/PAIR/arm/'supervisor.terminal.json') for arm in ARMS},
                    result=decide_pair(archive,secondary,c,a))
                write_json(root,phase,base/'report.json',expected,cap=PAIR_REPORT_CAP)
                terminal=dict(authorization=auth,status='COMPLETE',report_sha256=e.sha256(base/'report.json'),
                    producer_rc=0,reader_rc=0,arm_terminal_sha256=expected['arm_terminal_sha256'])
                write_json(root,phase,base/'attempt_result.json',{k:v for k,v in terminal.items() if k!='reader_rc'})
                write_json(root,phase,base/'supervisor.terminal.json',terminal)
                read_pair(root,auth,pins,archive,secondary,external)
            f.write(b'EXIT_CODE=0\n');f.flush();os.fsync(f.fileno())
        except BaseException as exc:
            # Never erase partial evidence or re-dispatch either arm.
            try:
                write_json(root,phase,base/'supervisor.failed.json',
                    dict(authorization=auth,status='STOP',error=repr(exc)[:4096]))
            except BaseException as err:print('PAIR_FAILURE_RECEIPT_ERROR',repr(err),flush=True)
            f.write(b'EXIT_CODE=1\n');f.flush();os.fsync(f.fileno())
            raise


def main():
    root=Path(__file__).resolve().parent
    if sys.argv[1:] in (['static'],['native']):
        supervisor(root,sys.argv[1])
    elif len(sys.argv)==4 and sys.argv[1]=='_worker':
        worker_entry(root,sys.argv[2],sys.argv[3])
    else:
        raise SystemExit('H4SSRPC_PAIR_HELD: distinct Owner marker required')


if __name__=='__main__':
    main()
