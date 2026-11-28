#!/usr/bin/env python3
"""R8C one-use host-only adapter. Frozen nine-file source is not edited."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BRANCH='experiment/b1-hardware-scaled'
BASE='62f5ddc50ac784eb662bc237ec5d9dfb092c22a8'
CLOSED='e09c35a82f06a97fe5406ba36bb3fd9664913583'
MARKER='H4PTR_R8C_STATIC_GO'
ID='H4-M29-R12-PTR-R8C-FOCUSED-HOST-V1'
OUT=ROOT/'m463_runs/m470_h4_m29_ptr_r8c_static_v1'
ROLE='PTR_R8C_CHILD_ROLE'
BOOT='afd9a7f1f1cc82040267b9099112b3fbb015e401ecc8677db1f03ccbfdf9d7c1'
SOURCE='c19f094a57dfd1c19e95c3b2a1bf4a405944ff837c16a75cb0320f77a3416a10'
GENERATED='20ab3e2c58097d643e14ac0c3e3615e81e56532bc3b31399fe1f2c5e9567bcb4'
MANIFEST='a779a0dc06d136464c1bc3a388fdac73129cab818e092aac7d6837ce6fc201d5'
RELEASE_TRAINER='08cd160c7b2fb5b76b5aa53c58d8ac855e9c7cf33d425b55014e8e2c18bac9b6'
RELEASE='1e5b44786e2f32b85e5230696203a22f29d477fd94c2fa272f7abc3fc93578a9'
CASES=('historical_and_source_pins','local_generated_identity_and_compile',
       'exact_release_projection','native_schema_and_closed_gate','audit_log_prefix',
       'partial_receipt_envelope')
FILES=('m462_ppo_warmstart_run.py','m462_typed_orbax_stage.py','m463_coverage_env.py',
       'm463_gate3_cp_eval.py','m463_instrument_brax.py','m463_push_ppo_train.py',
       'paper_config.py','paper_config_preflight.json','train.py')
PASS='PTR_R8C_INDEPENDENT_READER_PASS'
PREFIX='PTR_R8C_CASE_PASS='
ARTIFACTS=('attempt.json','worker.started.json','worker.log','report.json','attempt_result.json')
CAP=16*1024**2

def require(ok,label):
    if not ok:raise RuntimeError('PTR_R8C_'+label)

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def exact(a,b,label):
    require(canonical(a)==canonical(b),label)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,timeout=30)

def no_native():
    return not any(k.split('.')[0] in ('jax','jaxlib','mujoco','brax','mujoco_playground','torch','tensorflow')
                   for k in sys.modules)

def regular(p):
    require(p.is_file() and not p.is_symlink(),'FILE_'+p.name)
    return p.read_bytes()

def auth():
    require(no_native() and Path.cwd()==ROOT and sys.flags.dont_write_bytecode,'ENTRY')
    head=git('rev-parse','HEAD').decode().strip()
    require(git('symbolic-ref','--short','HEAD').decode().strip()==BRANCH and
        git('rev-parse','refs/remotes/origin/'+BRANCH).decode().strip()==head and
        not git('status','--porcelain','--untracked-files=no'),'BRANCH')
    rp=ROOT/'G1_RESEARCH_RULES.md';raw=regular(rp)
    lines=re.findall(rb'(?m)^[A-Z0-9_]+_GO:.*$',raw)
    prefix=(MARKER+': ').encode()
    require(len(lines)==1 and lines[0].startswith(prefix),'ONE_GO')
    a=json.loads(lines[0][len(prefix):])
    parent=git('rev-parse','HEAD^').decode().strip()
    wrapper=Path(__file__).resolve();selfraw=regular(wrapper)
    expected=dict(id=ID,source_head=parent,source_sha256=SOURCE,generated_sha256=GENERATED,
        fixture_manifest_sha256=MANIFEST,wrapper_sha256=sha(selfraw),
        projected_source_sha256=RELEASE)
    exact(a,expected,'AUTH_SCHEMA')
    prior=git('show',parent+':G1_RESEARCH_RULES.md')
    require(hashlib.sha1(b'blob '+str(len(prior)).encode()+b'\0'+prior).hexdigest()==CLOSED and
        raw==git('show','HEAD:G1_RESEARCH_RULES.md')==prior+prefix+canonical(a).encode()+b'\n' and
        git('rev-list','--parents','-n','1',head).decode().split()==[head,parent] and
        git('diff','--name-only',parent,head).splitlines()==[b'G1_RESEARCH_RULES.md'],'ROOT_ONLY')
    require(subprocess.run(['git','merge-base','--is-ancestor',BASE,parent],cwd=ROOT,
        timeout=30).returncode==0,'LINEAGE')
    require(wrapper==ROOT/'m470_ptr_r8c_host.py' and not (ROOT/wrapper.name).is_symlink() and
        selfraw==git('show',parent+':'+wrapper.name)==git('show','HEAD:'+wrapper.name),'WRAPPER_PIN')
    pins={}
    for name in FILES:
        value=regular(ROOT/name)
        require(value==git('show',BASE+':'+name)==git('show',parent+':'+name)==
            git('show','HEAD:'+name),'SOURCE_'+name)
        pins[name]=sha(value)
    require(sha(canonical(pins).encode())==SOURCE and sha(canonical(list(CASES)).encode())==MANIFEST,
        'AGGREGATE_OR_MANIFEST')
    require(no_native(),'AUTH_NATIVE')
    return a,head,pins

def attempt(a,head,pid):
    require(type(pid) is int and pid>0,'PARENT_PID')
    return dict(authorization=a,authorization_head=head,bootstrap_sha256=BOOT,parent_pid=pid)

def optional(name):
    p=OUT/name
    if not p.exists() and not p.is_symlink():return None
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'ARTIFACT_'+name)
    return sha(p.read_bytes())

def receipt(t,name):
    return t.ssrpc_read_json(OUT/name,65536)

def worker(t,att):
    if optional('worker.started.json') is None:return None
    w=receipt(t,'worker.started.json');pid=w['worker_pid']
    require(type(pid) is int and pid>0 and pid!=att['parent_pid'],'WORKER_PID')
    exact(w,dict(att,worker_pid=pid),'WORKER_BINDING')
    return pid

def result(a,head,rc):
    require(rc is None or type(rc) is int,'PRODUCER_RC_TYPE')
    rs=optional('report.json')
    return dict(status='PRODUCED' if rc==0 and rs is not None else 'STOP',
        authorization=a,authorization_head=head,producer_rc=rc,report_sha256=rs)

def log_cases(complete,reader=False):
    raw=regular(OUT/'worker.log')
    require(len(raw)<=CAP,'LOG_CAP')
    lines=raw.splitlines()
    seen=[line[len(PREFIX):].decode('ascii') for line in lines if line.startswith(PREFIX.encode())]
    require(tuple(seen)==CASES[:len(seen)],'CASE_ORDER')
    if complete:
        expected=[(PREFIX+k).encode() for k in CASES]+([PASS.encode()] if reader else [])
        require(lines==expected,'COMPLETE_LOG')
    return seen

def expected_report(t,a,head,pins):
    projected=t.ptr_r8b_release_projection(regular(ROOT/'m463_push_ppo_train.py'))
    rp=dict(pins,**{'m463_push_ppo_train.py':sha(projected)})
    require(rp['m463_push_ppo_train.py']==RELEASE_TRAINER and sha(canonical(rp).encode())==RELEASE,
        'PROJECTED_AGGREGATE')
    return dict(status='MINIMAL_FOCUSED_HOST_PASS',authorization=a,authorization_head=head,
        bootstrap_sha256=BOOT,source_files=pins,source_sha256=SOURCE,
        manifest_sha256=MANIFEST,generated_sha256=GENERATED,projected_trainer_sha256=RELEASE_TRAINER,
        projected_source_sha256=RELEASE,fixtures=[dict(id=k,status='PASS') for k in CASES],
        zero_native_execution=True,zero_optimizer_updates=True,zero_normalizer_updates=True,
        promotion_authorized=False)

def validate_payload(t,a,head,pins,att,reader=False):
    require(worker(t,att) is not None,'WORKER_MISSING')
    exact(receipt(t,'attempt.json'),att,'ATTEMPT')
    log_cases(True,reader)
    exact(receipt(t,'report.json'),expected_report(t,a,head,pins),'REPORT')
    exact(receipt(t,'attempt_result.json'),result(a,head,0),'RESULT')
    require(no_native(),'PAYLOAD_NATIVE')

def terminal(t,a,head,att,rc,rr,error):
    pid=worker(t,att)
    hashes={name:optional(name) for name in ARTIFACTS}
    return dict(status='COMPLETE' if error is None and rc==rr==0 else 'STOP',
        authorization=a,authorization_head=head,bootstrap_sha256=BOOT,
        parent_pid=att['parent_pid'],worker_pid=pid,producer_rc=rc,reader_rc=rr,error=error,
        stage='REPORT_WRITTEN' if hashes['report.json'] else 'WORKER_STARTED' if pid else 'RESERVATION_ONLY',
        artifact_sha256=hashes,stored_log_bytes=(OUT/'worker.log').stat().st_size if hashes['worker.log'] else 0,
        promotion_authorized=False)

def validate_terminal(t,a,head,pins,att,complete):
    name='supervisor.terminal.json' if complete else 'supervisor.failed.json'
    other='supervisor.failed.json' if complete else 'supervisor.terminal.json'
    require(optional(other) is None,'TERMINAL_CONFLICT')
    x=receipt(t,name);rc,rr,err=(x[k] for k in ('producer_rc','reader_rc','error'))
    require(all(v is None or type(v) is int for v in (rc,rr)) and (rr is None or rc==0),'TERMINAL_RC')
    exact(x,terminal(t,a,head,att,rc,rr,err),'TERMINAL')
    exact(receipt(t,'attempt_result.json'),result(a,head,rc),'TERMINAL_RESULT')
    exact(receipt(t,'attempt.json'),att,'TERMINAL_ATTEMPT')
    entries={p.name for p in OUT.iterdir()}
    require(entries<=set(ARTIFACTS)|{name} and all(p.is_file() and not p.is_symlink() for p in OUT.iterdir()),
        'TERMINAL_INVENTORY')
    if complete:
        require(type(rc) is type(rr) is int and rc==rr==0 and err is None and
            entries==set(ARTIFACTS)|{name},'TERMINAL_COMPLETE')
        validate_payload(t,a,head,pins,att,reader=True)
    else:
        require(type(err) is str and 0<len(err)<=2048,'TERMINAL_STOP')
        if optional('worker.log') is not None:log_cases(False)
        if optional('report.json') is not None:
            require(worker(t,att) is not None,'STOP_REPORT_WORKER')
            exact(receipt(t,'report.json'),expected_report(t,a,head,pins),'STOP_REPORT')
    require(no_native(),'TERMINAL_NATIVE')

def child(t,args,a,head,pins,role):
    require(role in ('producer','reader') and OUT.is_dir() and not OUT.is_symlink()
        and not OUT.parent.is_symlink(),'CHILD_ROLE')
    att=attempt(a,head,os.getppid())
    exact(receipt(t,'attempt.json'),att,'CHILD_ATTEMPT')
    if role=='reader':
        require(worker(t,att)!=os.getpid(),'READER_PID')
        validate_payload(t,a,head,pins,att)
        # Re-read provenance and bootstrap independently, without executing any case again.
        t.ptr_r8a_historical(ROOT)
        t.ptr_bootstrap_held(args)
        require(t.ssrpc_bootstrap_identity(ROOT)==BOOT and auth()==(a,head,pins),'READER_PROVENANCE')
        print(PASS,flush=True)
        return
    t.ssrpc_static_write(OUT/'worker.started.json',dict(att,worker_pid=os.getpid()))
    seen=[]
    def record(name):
        require(len(seen)<len(CASES) and name==CASES[len(seen)] and no_native(),'PRODUCER_ORDER')
        seen.append(name);print(PREFIX+name,flush=True)
    observed=t.ptr_r8b_focus_cases(ROOT,args,record)
    require(tuple(seen)==CASES,'PRODUCER_CASES')
    expected=dict(cases=list(CASES),manifest_sha256=MANIFEST,source_files=pins,source_sha256=SOURCE,
        generated_sha256=GENERATED,projected_trainer_sha256=RELEASE_TRAINER,
        bootstrap_sha256=BOOT,zero_native_execution=True,promotion_authorized=False)
    exact(observed,expected,'PRODUCER_OBSERVED')
    require(auth()==(a,head,pins) and t.ssrpc_bootstrap_identity(ROOT)==BOOT,'PRODUCER_FINAL')
    t.ssrpc_static_write(OUT/'report.json',expected_report(t,a,head,pins))

def supervisor(t,argv,a,head,pins):
    t.ptr_r4b_unused(OUT);t.h4x_storage(OUT,initial=True)
    OUT.mkdir(exist_ok=False)
    att=attempt(a,head,os.getpid())
    t.ssrpc_static_write(OUT/'attempt.json',att)
    rc=rr=None;proc=None;role=None;producer_pid=None;total=0
    deadline=time.monotonic()+3600
    def drain(log,which):
        nonlocal proc,role,total,producer_pid
        role=which
        proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),*argv],
            cwd=ROOT,env=dict(os.environ,**{ROLE:which}),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True)
        if which=='producer':producer_pid=proc.pid
        with selectors.DefaultSelector() as sel:
            sel.register(proc.stdout,selectors.EVENT_READ)
            while True:
                require(time.monotonic()<deadline,'WATCHDOG')
                if not sel.select(timeout=1):continue
                data=os.read(proc.stdout.fileno(),65536)
                if not data:break
                require(total+len(data)<=CAP,'LOG_CAP')
                log.write(data);log.flush();total+=len(data)
        value=proc.wait(timeout=max(.001,deadline-time.monotonic()))
        proc.stdout.close();proc=None;log.flush();os.fsync(log.fileno())
        return value
    try:
        with (OUT/'worker.log').open('xb') as log:
            rc=drain(log,'producer')
            t.ssrpc_static_write(OUT/'attempt_result.json',result(a,head,rc))
            require(rc==0 and worker(t,att)==producer_pid,'PRODUCER_RC_PID')
            validate_payload(t,a,head,pins,att)
            rr=drain(log,'reader');require(rr==0,'READER_RC')
        t.h4x_storage(OUT)
        require(auth()==(a,head,pins),'SUPERVISOR_FINAL')
        validate_payload(t,a,head,pins,att,reader=True)
        require({p.name for p in OUT.iterdir()}==set(ARTIFACTS),'PRETERMINAL_INVENTORY')
        t.ssrpc_static_write(OUT/'supervisor.terminal.json',terminal(t,a,head,att,rc,rr,None))
        validate_terminal(t,a,head,pins,att,True)
    except BaseException as exc:
        if proc is not None:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            value=proc.wait()
            if role=='producer':rc=value
            else:rr=value
            if proc.stdout is not None:proc.stdout.close()
        if optional('attempt_result.json') is None:
            t.ssrpc_static_write(OUT/'attempt_result.json',result(a,head,rc))
        if optional('supervisor.terminal.json') is None:
            t.ssrpc_static_write(OUT/'supervisor.failed.json',terminal(t,a,head,att,rc,rr,repr(exc)[:2048]))
            try:validate_terminal(t,a,head,pins,att,False)
            except Exception as error:print('PTR_R8C_PARTIAL_READER_STOP',repr(error),flush=True)
        raise

def main(argv):
    p=argparse.ArgumentParser(description='R8C one-use host-only verification')
    for name in ('checkpoint','restore-report','reward-report'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv)
    a,head,pins=auth()
    import m463_push_ppo_train as t
    require(Path(t.__file__).resolve()==ROOT/'m463_push_ppo_train.py' and no_native() and
        t.PTR_R8B_CASES==CASES and t.ptr_r8b_manifest()==MANIFEST,'TRAINER_MODULE')
    role=os.environ.get(ROLE)
    if role is None:supervisor(t,argv,a,head,pins)
    else:child(t,args,a,head,pins,role)

if __name__=='__main__':main(sys.argv[1:])
