#!/usr/bin/env python3
"""H4-M28-8: held SSRPC paired DEV host contracts.

The historical evaluator is part of the training source digest. Keep it and
the trainer byte-identical; do not monkeypatch historical provenance readers.
This module has NO static/native launcher. Direct invocation always HOLDs.
The functions below are source preparation, not executed fixture evidence.
"""
from __future__ import annotations

import hashlib
import json
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
# Proposed watchdogs/aggregate accounting need review before native integration.
EXECUTION_AUTHORIZED = False


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
AUTH_KEYS = {'id','source_blob','source_sha256','provenance_sha256',
             'static_sha256','authorization_head'}


def read_bounded(root, rel, cap, trainer):
    p=root/rel
    require(p.absolute()==p.resolve() and p.is_file() and not p.is_symlink()
            and 0<p.stat().st_size<=cap, 'RECEIPT_PATH_'+str(rel))
    return trainer.ssrpc_read_json(p,cap)


def check_auth_identity(auth, pins, external):
    import re
    import m463_gate3_cp_eval as e
    require(type(auth) is dict and set(auth)==AUTH_KEYS and auth['id']==ID
            and all(type(v) is str for v in auth.values()), 'AUTH_SCHEMA')
    require(all(re.fullmatch('[0-9a-f]{64}',auth[k]) for k in
                ('source_sha256','provenance_sha256','static_sha256'))
            and all(re.fullmatch('[0-9a-f]{40}',auth[k]) for k in
                ('source_blob','authorization_head')), 'AUTH_DIGESTS')
    require(auth['source_sha256']==pins['source/'+Path(__file__).name]
            and auth['source_blob']==blob(Path(__file__).read_bytes())
            and auth['provenance_sha256']==e.h4m3_digest([pins,external]), 'AUTH_BINDING')


def read_arm(root, arm, auth, pins, external, archive, *, complete):
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
        _, ledger=compatibility_view(e,arm,report)
        require(report['ledger']==ledger and all(archive[k]==report[k] for k in e.H4M12_PROTOCOL),
                'COMPLETE_REPLAY')
        gpu=report['gpu']
        require(gpu['total_mib']==12288 and 0<gpu['peak_used_mib']<.95*12288
                and type(gpu['samples']) is int and gpu['samples']>0 and not gpu['sampler_errors'],
                'GPU')
        digest=e.sha256(out/'report.json')
        expected=dict(identity,status='COMPLETE',report_sha256=digest,producer_rc=0)
        require(read_bounded(root,base/'attempt_result.json',RECEIPT_CAP,t)==expected,'RESULT')
        require(log.absolute()==log.resolve() and log.is_file() and not log.is_symlink()
                and log.stat().st_size<=LOG_CAP,'LOG')
        expected=dict(expected,reader_rc=0,worker_log_sha256=e.sha256(log),stored_log_bytes=log.stat().st_size)
        require(read_bounded(root,base/'supervisor.terminal.json',RECEIPT_CAP,t)==expected,'TERMINAL')
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
    require(read_bounded(root,PAIR/'report.json',PAIR_REPORT_CAP,t)==expected,'PAIR_FINAL_REPLAY')
    require(not (root/PAIR/'supervisor.failed.json').exists()
            and not (root/PAIR/'supervisor.failed.json').is_symlink(), 'PAIR_FAILURE')
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


def held():
    raise SystemExit('H4SSRPC_PAIRED_DEV_HELD: no static or native authorization/launcher')


if __name__=='__main__':
    held()
