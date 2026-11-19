#!/usr/bin/env python3
"""M4.32 -- independent confirmation on 32 genuinely new evaluation seeds.

This is EVALUATION ONLY. It imports the unchanged M4.26/M4.27 engines and
M4.31's checkpoint/config provenance validator. Neither PPO nor environment
source, reward, termination or checkpoints are modified. Output is written
into a distinct m432_new_holdout/ directory, never into baseline or M4.31.

Modes: inspect | det-stoch | dr | compare
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys

WORK = Path(__file__).resolve().parent
OUTDIR = WORK / "m432_new_holdout"
SEEDS = [5001 + 37 * i for i in range(32)]
PREVIOUS_SEEDS = [3001 + 37 * i for i in range(32)]
FINAL_STEP = 401_080_320
TARGETS = {
    "det-stoch": [
        ("seed0", OUTDIR / "seed0_det_stoch.json"),
        ("seed1", OUTDIR / "seed1_det_stoch.json"),
    ],
    "dr": [
        ("seed0", OUTDIR / "seed0_brax_nominal_dr.json"),
        ("seed1", OUTDIR / "seed1_brax_nominal_dr.json"),
    ],
}
COMPARE_OUT = OUTDIR / "m432_seed0_vs_seed1_new32.json"


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"REQUIRED_FILE_MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_evaluation_data(data: dict, group: str) -> None:
    if data.get("seeds") != SEEDS or len(set(data["seeds"])) != 32:
        raise RuntimeError(f"EVAL_SEED_SET_MISMATCH: {group}")
    if data.get("checkpoint_step", FINAL_STEP) != FINAL_STEP:
        raise RuntimeError(f"EVAL_CHECKPOINT_STEP_MISMATCH: {group}")
    if data.get("training_repeated") is not False:
        raise RuntimeError(f"RETRAINING_UNEXPECTED: {group}")
    if group == "det-stoch":
        if (data.get("status") != "M426_EVALUATION_COMPLETED"
                or data.get("max_steps") != 1000
                or data.get("policy_modes") != ["deterministic", "stochastic"]
                or data.get("evaluation_domain_randomization") is not False):
            raise RuntimeError("M432_M426_PROTOCOL_MISMATCH")
        if sorted(x["policy_mode"] for x in data["results"]) != ["deterministic", "stochastic"]:
            raise RuntimeError("M432_INVALID_POLICY_MODES")
    else:
        if (data.get("status") != "M427_EVALUATION_COMPLETED"
                or data.get("steps_max") != 1000
                or data.get("num_envs") != 32
                or data.get("policy_mode") != "stochastic"
                or data.get("source_randomizer_applied") is not True
                or data.get("same_reset_keys_between_conditions") is not True):
            raise RuntimeError("M432_M427_PROTOCOL_MISMATCH")
        if sorted(x["mode"] for x in data["results"]) != ["nominal", "source_domain_randomization"]:
            raise RuntimeError("M432_INVALID_DR_MODES")
    for r in data["results"]:
        rows = r["rollouts"]
        if sorted(x["seed"] for x in rows) != SEEDS:
            raise RuntimeError("M432_ROLLOUT_SEEDS_INVALID")
        if any(not (1 <= int(x["steps_survived"]) <= 1000) for x in rows):
            raise RuntimeError("M432_ROLLOUT_STEPS_INVALID")


def preflight() -> dict:
    # Validate all M4.30/M4.31 sources, checkpoint sha256, and identical PPO
    # configs before any imported evaluation code is allowed to run.
    import m431_seed1_matched_eval as m431
    ctx = m431.validate()
    prior = load(m431.OUT)
    if (prior.get("status") != "M431_SEED0_SEED1_COMPARISON_COMPLETED"
            or prior.get("matched_heldout_seeds") != PREVIOUS_SEEDS
            or not prior.get("same_scaled_training_config")
            or prior.get("checkpoint_step") != FINAL_STEP):
        raise RuntimeError("M431_PREVIOUS_COMPARISON_INVALID")
    if set(SEEDS) & set(PREVIOUS_SEEDS):
        raise RuntimeError("NEW_EVALUATION_SEEDS_NOT_DISJOINT")
    if len(set(SEEDS)) != 32:
        raise RuntimeError("DUPLICATE_SEED")
    if (ctx["seed0_sha256"] != prior["seed0_checkpoint_sha256"]
            or ctx["seed1_sha256"] != prior["seed1_checkpoint_sha256"]):
        raise RuntimeError("M431_COMPARISON_CHECKPOINT_HASH_MISMATCH")
    return {**ctx, "prior": prior}


@contextmanager
def run_mode():
    saved = sys.argv
    sys.argv = [saved[0], "--mode", "run"]
    try:
        yield
    finally:
        sys.argv = saved


def eval_group(group: str, ctx: dict) -> None:
    import m426_heldout_det_stoch as m426
    import m427_domain_randomization_eval as m427
    if (m426.SEEDS != PREVIOUS_SEEDS or m427.SEEDS != PREVIOUS_SEEDS
            or m426.STEPS != 1000 or m427.STEPS != 1000
            or m426.SELECTED_STEPS != [FINAL_STEP]
            or m427.CHECKPOINT_STEP != FINAL_STEP):
        raise RuntimeError("REFERENCE_EVALUATORS_MODIFIED")
    if group == "dr" and m427.N != 32:
        raise RuntimeError("REFERENCE_BATCh_SIZE_MODIFIED")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    m426.SEEDS = list(SEEDS)
    m427.SEEDS = list(SEEDS)
    m427.N = len(SEEDS)
    for tag, dest in TARGETS[group]:
        if dest.exists():
            check_evaluation_data(load(dest), group)
            print(f"M432_EXISTING_VALID_RESULT_SKIP {dest}", flush=True)
            continue
        # Python's imported 'read_entries' in M4.27 uses m426.WORK, so the
        # two module globals must point to the same checkpoint directory.
        checkpoint_dir = WORK if tag == "seed0" else WORK / "m430_seed1"
        report_path = (WORK / "pilot_20261113_170501_977563_report.json"
                       if tag == "seed0" else ctx["report"])
        m426.WORK = checkpoint_dir
        m426.REPORT = report_path
        m426.OUT = dest
        m427.WORK = checkpoint_dir
        m427.REPORT = report_path
        m427.OUT = dest
        print(f"M432_EVAL_START {group} {tag} new_seeds={SEEDS[0]}..{SEEDS[-1]}", flush=True)
        with run_mode():
            if group == "det-stoch":
                m426.main()
            else:
                m427.main()
        check_evaluation_data(load(dest), group)
        print(f"M432_EVAL_DONE {group} {tag} {dest}", flush=True)
    print(f"M432_{group.upper().replace('-', '_')}_COMPLETED", flush=True)


def paired(old: list[dict], new: list[dict], type_: str, mode: str) -> dict:
    by_old = {r["seed"]:r for r in old}
    by_new = {r["seed"]:r for r in new}
    if set(by_old) != set(SEEDS) or set(by_new) != set(SEEDS):
        raise RuntimeError("UNMATCHED_SEEDS")
    o = [int(by_old[s]["steps_survived"]) for s in SEEDS]
    n = [int(by_new[s]["steps_survived"]) for s in SEEDS]
    diff = [b-a for a,b in zip(o,n)]
    if type_ == "det-stoch":
        suc = lambda rows: sum((not x["terminated"]) and x["steps_survived"] == 1000 for x in rows)
    else:
        suc = lambda rows: sum(x["completed_1000_steps"] for x in rows)
    avg = statistics.mean(diff)
    se = statistics.stdev(diff) / len(diff)**.5
    data = {
        "mode":mode, "n":len(SEEDS),
        "seed0_mean_steps":statistics.mean(o),
        "seed1_mean_steps":statistics.mean(n),
        "mean_delta_seed1_minus_seed0":avg,
        "approx_paired_95ci": [avg-2.04*se, avg+2.04*se],
        "seed0_1000step_completions":suc(old),
        "seed1_1000step_completions":suc(new),
        "seed1_better":sum(d>0 for d in diff),
        "seed1_worse":sum(d<0 for d in diff),
        "ties":sum(d==0 for d in diff),
        "paired_survival": [{"seed":s,"seed0_steps":a,"seed1_steps":b,"delta":b-a}
                            for s,a,b in zip(SEEDS,o,n)],
    }
    if type_ == "det-stoch":
        for metric in ("xy_rmse", "yaw_rmse"):
            data["seed0_mean_"+metric] = statistics.mean(float(x[metric]) for x in old)
            data["seed1_mean_"+metric] = statistics.mean(float(x[metric]) for x in new)
    return data


def compare(ctx: dict) -> None:
    if COMPARE_OUT.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE_EXISTING_COMPARISON: {COMPARE_OUT}")
    files = {group: [load(p) for _,p in targets] for group,targets in TARGETS.items()}
    for group, runs in files.items():
        for data in runs:
            check_evaluation_data(data, group)
    summary = []
    for mode in ("deterministic", "stochastic"):
        old, new = (next(x for x in data["results"] if x["policy_mode"] == mode)
                    for data in files["det-stoch"])
        summary.append(paired(old["rollouts"],new["rollouts"],"det-stoch",mode))
    for mode in ("nominal", "source_domain_randomization"):
        old, new = (next(x for x in data["results"] if x["mode"] == mode)
                    for data in files["dr"])
        summary.append(paired(old["rollouts"],new["rollouts"],"dr","brax_"+mode))

    output={
        "status":"M432_NEW_HELDOUT_COMPARISON_COMPLETED",
        "training_repeated":False,
        "checkpoint_step":FINAL_STEP,
        "seed0_checkpoint_sha256":ctx["seed0_sha256"],
        "seed1_checkpoint_sha256":ctx["seed1_sha256"],
        "new_heldout_seeds":SEEDS,
        "prior_m431_seeds":PREVIOUS_SEEDS,
        "evaluation_source_sha256":{
            name:checksum(WORK / name) for name in (
                "m426_heldout_det_stoch.py",
                "m427_domain_randomization_eval.py",
                "m431_seed1_matched_eval.py"
            )},
        "metrics_by_protocol":summary,
        "limitations":[
            "New held-out seed list is independent of M4.31's selection set.",
            "Approximate t(31) CI uses t critical ~2.04; diagnostic, not a proof of equivalence.",
            "M426 per-seed evaluation and M427 batched Brax evaluation differ; compare checkpoint pairs WITHIN protocol only.",
            "Brax DR uses the same source randomizer and fixed DR RNG for both policies; it is not paper-exact evaluator.",
            "1000-step completion is not proof of natural gait or real-world robustness.",
        ],
    }
    with COMPARE_OUT.open("x",encoding="utf-8") as f:
        json.dump(output,f,indent=2,ensure_ascii=False)
        f.write("\n")
    for r in summary:
        print(f"M432_COMPARE {r['mode']}: seed0={r['seed0_mean_steps']:.2f} "
              f"seed1={r['seed1_mean_steps']:.2f} delta={r['mean_delta_seed1_minus_seed0']:+.2f} "
              f"success={r['seed0_1000step_completions']}/32->"
              f"{r['seed1_1000step_completions']}/32",flush=True)
    print(f"M432_NEW_HELDOUT_COMPARISON_COMPLETED {COMPARE_OUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("inspect","det-stoch","dr","compare"))
    args=parser.parse_args()
    ctx=preflight()
    print("M432_CHECKPOINTS_AND_CONFIG_VERIFIED",flush=True)
    print(f"M432_NEW_SEEDS first={SEEDS[0]} last={SEEDS[-1]} count={len(SEEDS)}",flush=True)
    if args.mode=="inspect":
        print("M432_INSPECT_PASS NO_EVALUATION NO_TRAINING",flush=True)
    elif args.mode=="compare":
        compare(ctx)
    else:
        eval_group(args.mode,ctx)


if __name__=="__main__":
    main()
