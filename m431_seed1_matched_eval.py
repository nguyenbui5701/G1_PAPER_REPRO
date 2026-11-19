#!/usr/bin/env python3
"""M4.31 — strictly matched held-out evaluation of independent M4.30 seed=1.

Reuse M4.26 / M4.27 evaluation functions, RNG, randomizer, metrics and seeds
UNCHANGED by redirecting only their working paths to m430_seed1/. Never
train, modify baseline files, or overwrite existing evaluation results.

Modes: inspect | det-stoch | dr | compare
Run from G1_PAPER_REPRO, alongside m426_heldout_det_stoch.py and
m427_domain_randomization_eval.py, with m430_seed1/ training outputs present.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys

WORK = Path(__file__).resolve().parent
SEED1 = WORK / "m430_seed1"
MANIFEST = SEED1 / "m430_manifest.json"
FINISHED = SEED1 / "m430_run_finished.json"
BASELINE_REPORT = WORK / "pilot_20261113_170501_977563_report.json"
BASELINE_DET_STOCH = WORK / "m426_401m_heldout_det_stoch_32seeds.json"
BASELINE_DR = WORK / "m427_401m_dr_vs_nominal_32seeds.json"
SEED1_DET_STOCH = SEED1 / "m431_seed1_det_stoch_32seeds.json"
SEED1_DR = SEED1 / "m431_seed1_dr_nominal_32seeds.json"
OUT = SEED1 / "m431_seed0_vs_seed1_comparison.json"
FINAL_STEP = 401_080_320
SEEDS = [3001 + 37*i for i in range(32)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"REQUIRED_FILE_MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def report_final(report: dict, root: Path) -> tuple[dict, Path]:
    matches = [x for x in report.get("checkpoints", [])
               if int(x["step"]) == FINAL_STEP]
    if len(matches) != 1:
        raise RuntimeError(f"FINAL_CHECKPOINT_ENTRY_MISSING_OR_DUPLICATED: {root}")
    entry = matches[0]
    name = Path(entry["file"]).name
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        raise RuntimeError(f"UNSAFE_OR_MISSING_CHECKPOINT: {path}")
    if sha256(path) != entry["sha256"]:
        raise RuntimeError(f"CHECKPOINT_SHA256_MISMATCH: {path}")
    return entry, path


def validate() -> dict:
    for name in ("m426_heldout_det_stoch.py", "m427_domain_randomization_eval.py",
                 "paper_config.py", "train.py"):
        if not (WORK / name).is_file():
            raise RuntimeError(f"REQUIRED_EVALUATION_SOURCE_MISSING: {name}")
    mf = load_json(MANIFEST)
    finished = load_json(FINISHED)
    baseline = load_json(BASELINE_REPORT)
    reports = sorted(SEED1.glob("pilot_*_report.json"))
    if len(reports) != 1:
        raise RuntimeError(f"EXPECTED_ONE_SEED1_TRAINING_REPORT: found {len(reports)}")
    rpath = reports[0]
    seed1 = load_json(rpath)
    if (mf.get("seed") != 1 or finished.get("seed") != 1
            or finished.get("status") != "M430_TRAINING_COMPLETED"
            or finished.get("final_step") != FINAL_STEP):
        raise RuntimeError("M430_MANIFEST_OR_FINISHED_INVALID")
    if Path(finished["report"]).name != rpath.name:
        raise RuntimeError("M430_FINISHED_REPORT_MISMATCH")
    if sha256(BASELINE_REPORT) != mf.get("baseline_report_sha256"):
        raise RuntimeError("BASELINE_REPORT_CHANGED_SINCE_M430")
    source = SEED1 / "pilot_seed1.py"
    if not source.is_file() or sha256(source) != mf.get("seed1_source_sha256"):
        raise RuntimeError("M430_TRAINER_SOURCE_HASH_MISMATCH")
    for label, rep in (("seed0", baseline), ("seed1", seed1)):
        if rep.get("status") != "PILOT_COMPLETED" or rep.get("environment") != "G1JoystickFlatTerrain":
            raise RuntimeError(f"{label.upper()}_REPORT_INVALID")
        if not rep.get("domain_randomization_enabled"):
            raise RuntimeError(f"{label.upper()}_TRAINING_DOMAIN_RANDOMIZATION_OFF")
        if len(rep.get("checkpoints", [])) != 15:
            raise RuntimeError(f"{label.upper()}_CHECKPOINT_COUNT_INVALID")
    if baseline["scaled_config"] != seed1["scaled_config"]:
        raise RuntimeError("TRAINING_CONFIGS_DIFFER")
    if seed1["scaled_config"] != mf.get("baseline_scaled_config"):
        raise RuntimeError("M430_SEED1_CONFIG_MISMATCH")
    b_entry, _ = report_final(baseline, WORK)
    s_entry, spath = report_final(seed1, SEED1)
    if b_entry["sha256"] != mf.get("baseline_final_checkpoint_sha256"):
        raise RuntimeError("BASELINE_CHECKPOINT_CHANGED_SINCE_M430")
    if b_entry["sha256"] == s_entry["sha256"]:
        raise RuntimeError("SEED1_CHECKPOINT_IDENTICAL_TO_BASELINE")
    if seed1["final_checkpoint_comparison"].get("tensor_values_equal") is not True:
        raise RuntimeError("M430_RETURNED_FINAL_TENSORS_MISMATCH")
    for label, f in (("M426_BASELINE", BASELINE_DET_STOCH), ("M427_BASELINE", BASELINE_DR)):
        if not f.is_file():
            raise RuntimeError(f"{label}_RESULTS_MISSING: {f}")
    b426 = load_json(BASELINE_DET_STOCH)
    b427 = load_json(BASELINE_DR)
    if (b426["status"] != "M426_EVALUATION_COMPLETED"
            or b426["seeds"] != SEEDS or b426["max_steps"] != 1000
            or b426["policy_modes"] != ["deterministic", "stochastic"]
            or b427["status"] != "M427_EVALUATION_COMPLETED"
            or b427["seeds"] != SEEDS or b427["steps_max"] != 1000
            or not b427["source_randomizer_applied"]):
        raise RuntimeError("REFERENCE_EVALUATOR_PROTOCOL_MISMATCH")
    print("M431_INPUTS_VERIFIED", flush=True)
    print("SEED1_FINAL", spath.name, s_entry["sha256"], flush=True)
    print("SEED0_FINAL_SHA256", b_entry["sha256"], flush=True)
    print("SCALED_CONFIG_IDENTICAL=YES; FRESH_SEED_1=YES; NO_TRAINING", flush=True)
    return {"report": rpath, "baseline": baseline, "seed1": seed1,
            "seed0_sha256": b_entry["sha256"], "seed1_sha256": s_entry["sha256"]}


@contextmanager
def evaluator_args():
    saved = sys.argv
    sys.argv = [saved[0], "--mode", "run"]
    try:
        yield
    finally:
        sys.argv = saved


def run_evaluation(which: str, context: dict) -> None:
    import m426_heldout_det_stoch as m426
    import m427_domain_randomization_eval as m427
    if m426.SEEDS != SEEDS or m427.SEEDS != SEEDS or m426.STEPS != 1000 or m427.STEPS != 1000:
        raise RuntimeError("REFERENCE_EVALUATOR_SEEDS_OR_STEPS_CHANGED")
    if m426.SELECTED_STEPS != [FINAL_STEP] or m427.CHECKPOINT_STEP != FINAL_STEP:
        raise RuntimeError("REFERENCE_EVALUATOR_FINAL_STEP_CHANGED")

    # Existing M426 function read_entries resolves its file paths against its
    # module global WORK. M427 imports the exact SAME function object.
    m426.WORK = SEED1
    m426.REPORT = context["report"]
    m426.OUT = SEED1_DET_STOCH
    m427.WORK = SEED1
    m427.REPORT = context["report"]
    m427.OUT = SEED1_DR
    target = SEED1_DET_STOCH if which == "det-stoch" else SEED1_DR
    if target.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE_EXISTING_RESULT: {target}")
    print(f"M431_REUSE_EXACT_REFERENCE_EVALUATOR mode={which}", flush=True)
    with evaluator_args():
        if which == "det-stoch":
            m426.main()
        else:
            m427.main()
    if not target.is_file():
        raise RuntimeError("EVALUATOR_RETURNED_WITHOUT_OUTPUT")
    print(f"M431_{which.upper().replace('-', '_')}_COMPLETED {target}", flush=True)


def paired_summary(old: list[dict], new: list[dict], success_field: str, mode: str) -> dict:
    a = {int(x["seed"]): x for x in old}
    b = {int(x["seed"]): x for x in new}
    if set(a) != set(SEEDS) or set(b) != set(SEEDS):
        raise RuntimeError(f"SEED_SET_MISMATCH mode={mode}")
    pairs = [(a[s], b[s]) for s in SEEDS]
    o = [int(x["steps_survived"]) for x, _ in pairs]
    n = [int(x["steps_survived"]) for _, x in pairs]
    delta = [v-u for u,v in zip(o,n)]
    def successes(rs):
        if success_field == "terminated":
            return sum(int((not r["terminated"]) and r["steps_survived"] == 1000) for r in rs)
        return sum(int(r["completed_1000_steps"]) for r in rs)
    res = {
        "mode": mode, "n": len(SEEDS),
        "seed0_mean_steps": statistics.mean(o),
        "seed1_mean_steps": statistics.mean(n),
        "mean_delta_seed1_minus_seed0": statistics.mean(delta),
        "seed0_median_steps": statistics.median(o),
        "seed1_median_steps": statistics.median(n),
        "seed0_successes": successes([x for x,_ in pairs]),
        "seed1_successes": successes([x for _,x in pairs]),
        "seed1_better_seeds": sum(d>0 for d in delta),
        "seed1_worse_seeds": sum(d<0 for d in delta),
        "ties": sum(d==0 for d in delta),
        "paired_survival": [{"seed":s,"seed0_steps":u,"seed1_steps":v,"delta":v-u}
                            for s,u,v in zip(SEEDS,o,n)],
    }
    if success_field == "terminated":
        for k, rows in (("seed0",old),("seed1",new)):
            reasons = Counter(reason for r in rows
                              for reason in (r.get("terminal_diagnosis") or {}).get("triggered",[]))
            res[k+"_termination_reasons"] = dict(reasons)
            for metric in ("xy_rmse", "yaw_rmse"):
                res[k+"_mean_"+metric] = statistics.mean(float(r[metric]) for r in rows)
    return res


def compare(context: dict) -> None:
    if OUT.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE_EXISTING_RESULT: {OUT}")
    b426 = load_json(BASELINE_DET_STOCH)
    s426 = load_json(SEED1_DET_STOCH)
    b427 = load_json(BASELINE_DR)
    s427 = load_json(SEED1_DR)
    if (s426.get("status") != "M426_EVALUATION_COMPLETED" or
            s427.get("status") != "M427_EVALUATION_COMPLETED" or
            s426.get("seeds") != SEEDS or s427.get("seeds") != SEEDS or
            s426.get("max_steps") != 1000 or s427.get("steps_max") != 1000 or
            s426.get("evaluation_domain_randomization") is not False or
            not s427.get("source_randomizer_applied") or
            s427.get("evaluation_type") != b427.get("evaluation_type")):
        raise RuntimeError("M431_EVALUATION_PROTOCOL_MISMATCH")
    comparisons = []
    for mode in ("deterministic", "stochastic"):
        old = next(r for r in b426["results"] if r["policy_mode"]==mode)
        new = next(r for r in s426["results"] if r["policy_mode"]==mode)
        comparisons.append(paired_summary(old["rollouts"],new["rollouts"],"terminated", mode))
    for mode in ("nominal", "source_domain_randomization"):
        old = next(r for r in b427["results"] if r["mode"]==mode)
        new = next(r for r in s427["results"] if r["mode"]==mode)
        comparisons.append(paired_summary(old["rollouts"],new["rollouts"],"completed", "brax_"+mode))

    output = {
        "status":"M431_SEED0_SEED1_COMPARISON_COMPLETED",
        "training_repeated":False,"checkpoint_step":FINAL_STEP,
        "seed0_training_seed":0,"seed1_training_seed":1,
        "seed0_checkpoint_sha256":context["seed0_sha256"],
        "seed1_checkpoint_sha256":context["seed1_sha256"],
        "same_scaled_training_config":True,
        "matched_heldout_seeds":SEEDS,
        "metrics_by_protocol":comparisons,
        "caveats": [
            "No additional training: these are evaluations of already-trained checkpoints.",
            "M426 deterministic/stochastic per-seed CPU loop and M427 batched Brax nominal/DR are distinct evaluation protocols; compare seed0 vs seed1 WITHIN each protocol only.",
            "DR evaluation reuses source randomizer and fixed M427 randomization RNG; 32 seeds give limited statistical power.",
            "Surviving 1000 steps does not imply natural gait or paper-exact reproduction.",
        ],
    }
    with OUT.open("x",encoding="utf-8") as f:
        json.dump(output,f,indent=2,ensure_ascii=False)
        f.write("\n")
    for x in comparisons:
        print(f"M431_COMPARE {x['mode']}: seed0={x['seed0_mean_steps']:.2f} "
              f"seed1={x['seed1_mean_steps']:.2f} diff={x['mean_delta_seed1_minus_seed0']:+.2f} "
              f"success={x['seed0_successes']}/32 -> {x['seed1_successes']}/32",flush=True)
    print("M431_COMPARISON_COMPLETED",OUT,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("inspect","det-stoch","dr","compare"),required=True)
    args=parser.parse_args()
    ctx=validate()
    if args.mode == "inspect":
        print("M431_INSPECT_PASS NO_TRAINING NO_EVALUATION",flush=True)
    elif args.mode == "compare":
        compare(ctx)
    else:
        run_evaluation(args.mode,ctx)


if __name__=="__main__":
    main()
