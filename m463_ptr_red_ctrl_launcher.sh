#!/usr/bin/env bash
# H4-M29 R12-PTR-RED CTRL-only one-use outer capture. HELD until separate Owner GO.
set -u -o pipefail
if [[ $# != 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
  printf 'STOP: exact authorized HEAD required\n' >&2; exit 64
fi
expected=$1
cd -- "$(dirname -- "$(readlink -f -- "$0")")" || exit 65
out=m463_runs/m470_h4_m29_ptr_red_ctrl_v1
log=${out}_outer.log
receipt=${out}_outer.receipt.json
if [[ ! -d m463_runs || -L m463_runs || -e "$out" || -L "$out" || -e "$log" || -L "$log" || -e "$receipt" || -L "$receipt" ]]; then
  printf 'STOP: namespace not unused\n' >&2; exit 66
fi
set -C
exec 9>"$log" || exit 67
exec 8>"$receipt" || exit 68
set +C
(
  set -e
  finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf '===== ARTIFACT INVENTORY =====\n'
    python3 - "$out" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists() and not p.is_symlink():
    print('NO_OUTPUT_DIRECTORY'); raise SystemExit(0)
if not p.is_dir() or p.is_symlink(): raise SystemExit(1)
total=0
for f in sorted(p.iterdir()):
    if not f.is_file() or f.is_symlink(): raise SystemExit(1)
    n=f.stat().st_size; total+=n
    if total>512*1024**2: raise SystemExit(1)
    h=hashlib.sha256()
    with f.open('rb') as s:
        for b in iter(lambda:s.read(1024**2),b''): h.update(b)
    print('ARTIFACT_BYTES',n,h.hexdigest(),f)
for name in ('supervisor.terminal.json','supervisor.failed.json'):
    f=p/name
    if f.is_file() and not f.is_symlink():
        if f.stat().st_size>65536: raise SystemExit(1)
        print('===== '+name+' ====='); print(f.read_text())
PY
    local irc=$?
    if (( rc == 0 && irc != 0 )); then rc=$irc; fi
    printf 'CTRL_RED_OUTER_BODY_RC=%s\n' "$rc"
    exit "$rc"
  }
  trap finish EXIT
  printf '===== CTRL-RED PREFLIGHT =====\n'
  [[ $(git symbolic-ref --short HEAD) == experiment/b1-hardware-scaled ]]
  [[ -z $(git status --porcelain --untracked-files=no) ]]
  git fetch origin experiment/b1-hardware-scaled
  [[ $(git rev-parse refs/remotes/origin/experiment/b1-hardware-scaled) == "$expected" ]]
  git merge --ff-only "$expected"
  [[ $(git rev-parse HEAD) == "$expected" ]]
  [[ -z $(git status --porcelain --untracked-files=no) ]]
  [[ $(git diff --name-only HEAD^ HEAD) == G1_RESEARCH_RULES.md ]]
  [[ ! -e "$out" && ! -L "$out" ]]
  root=$(pwd -P); base=$(dirname "$root")
  python3 - "$root" <<'PY'
import hashlib,json,re,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); raw=(root/'G1_RESEARCH_RULES.md').read_bytes()
lines=re.findall(rb'(?m)^[A-Z0-9_]+_GO:.*$',raw)
prefix=b'H4PTR_RED_CTRL_V1_GO: '
if len(lines)!=1 or not lines[0].startswith(prefix): raise SystemExit('STOP: CTRL-RED runtime marker absent')
a=json.loads(lines[0][len(prefix):]); p=root/'m463_ptr_red_ctrl_launcher.sh'
if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=a.get('launcher_sha256'):
    raise SystemExit('STOP: launcher identity')
if shutil.disk_usage(root/'m463_runs').free < (2*1024**3+512*1024**2+16*1024**2):
    raise SystemExit('STOP: disk reserve')
print('CTRL_RED_MARKER_PREFLIGHT_PASS')
PY
  export JAX_PLATFORMS=cuda JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONNOUSERSITE=1
  unset LD_LIBRARY_PATH PYTHONHOME
  export PYTHONPATH="$base/G1_M3_11E_PPO_CLOUDPICKLE_OVERLAY_V1/overlays/cloudpickle_20261107_111524_3962615:$base/G1_M3_11D_PPO_GYM_ISOLATED_REPAIR_V1/overlays/wheels_20261107_111000_3952924:$base/G1_M3_11C_PPO_MANIFEST_REPAIR_V1/overlays/wheels_20261107_110316_3940764:$base/G1_M3_7_G1_MJX_GPU_ONE_STEP_V1/source/mujoco_playground_2025:$root"
  "$base/G1_M3_7_G1_MJX_GPU_ONE_STEP_V1/.venv_m37_jax0435_mjx327/bin/python" -B -u -c 'import sys;import m463_push_ppo_train as t;t.ptr_r12_red_runtime_main(sys.argv[1:])' \
    --checkpoint m430_seed1/checkpoint_pilot_20261114_120515_1722937_step_401080320.msgpack \
    --restore-report m462_runs/typed_restore_20261120_162901.json \
    --reward-report m463_runs/m470_h3_m1_2_independent_10M_20261124_183319_225465810/report.json
  printf 'CTRL_RED_SUPERVISOR_EXIT_OK_PENDING_SA_REVIEW\n'
) 2>&1 8>&- 9>&- | (ulimit -f 16384 && exec tee -a /proc/self/fd/9) 8>&-
rcs=("${PIPESTATUS[@]}")
exec 9>&-
python3 - "$log" "$expected" "${rcs[0]}" "${rcs[1]}" >&8 <<'PY'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); rc,tee=map(int,sys.argv[3:]); n=p.stat().st_size
data=p.read_bytes() if n<=16*1024**2 else b''
ok=(rc==tee==0 and n>0 and bool(data) and
    data.splitlines().count(b'CTRL_RED_SUPERVISOR_EXIT_OK_PENDING_SA_REVIEW')==1 and
    data.splitlines().count(b'CTRL_RED_OUTER_BODY_RC=0')==1 and
    b'===== ARTIFACT INVENTORY =====' in data and b'===== supervisor.terminal.json =====' in data)
h=hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
v=dict(status='CAPTURE_COMPLETE' if ok else 'STOP',authorization_head=sys.argv[2],outer_rc=rc,tee_rc=tee,
       stored_log_bytes=n,outer_log_sha256=h.hexdigest(),promotion_authorized=False)
print(json.dumps(v,sort_keys=True),flush=True); os.fsync(sys.stdout.fileno())
raise SystemExit(0 if ok else 1)
PY
receipt_rc=$?
exec 8>&-
cat -- "$receipt"
printf 'CTRL_RED_OUTER_EXIT_CODE=%s\nCTRL_RED_TEE_EXIT_CODE=%s\nCTRL_RED_RECEIPT_EXIT_CODE=%s\n' "${rcs[0]}" "${rcs[1]}" "$receipt_rc"
if (( rcs[0] != 0 || rcs[1] != 0 || receipt_rc != 0 )); then exit 1; fi
