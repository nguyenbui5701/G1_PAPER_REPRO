#!/usr/bin/env python3
"""M4.50: reuse unchanged M4.26 with only source push disabled."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = [5001, 5186, 5445, 5667, 5889, 6148]

SOURCES = {
    "baseline0": (
        "pilot_20261113_170501_977563_report.json",
        "396f1bb7121418d3bb2ad0e7bd7aabe095d97fc5f72208bc0913e1bee3edc9aa"
    ),
    "b1": (
        "pilot_20261115_210149_3046829_report.json",
        "21ad713f61ef8ac22b72de2bec6aec9b45ca7de27101afdec21a043fe17ef173"
    ),
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", choices=SOURCES, required=True)
    p.add_argument("--mode", choices=("inspect", "run"), required=True)
    args = p.parse_args()

    report_name, expected_sha = SOURCES[args.label]
    report_path = ROOT / report_name
    report = json.loads(report_path.read_text())

    reference = json.loads(
        (ROOT / "m432_new_holdout/m432_seed0_vs_seed1_new32.json").read_text()
    )
    source_sha = reference["evaluation_source_sha256"][
        "m426_heldout_det_stoch.py"
    ]
    if sha256(ROOT / "m426_heldout_det_stoch.py") != source_sha:
        raise RuntimeError("REFERENCE_EVALUATOR_CHANGED")

    if report["status"] != "PILOT_COMPLETED":
        raise RuntimeError("INVALID_TRAINING_REPORT")

    entries = [
        x for x in report["checkpoints"]
        if int(x["step"]) == 401080320
    ]
    if len(entries) != 1 or entries[0]["sha256"] != expected_sha:
        raise RuntimeError("CHECKPOINT_PROVENANCE_MISMATCH")

    checkpoint = ROOT / Path(entries[0]["file"]).name
    if (checkpoint.is_symlink() or not checkpoint.is_file()
            or checkpoint.resolve().parent != ROOT.resolve()
            or sha256(checkpoint) != expected_sha):
        raise RuntimeError("CHECKPOINT_FILE_INVALID")

    if not set(SEEDS).issubset(reference["new_heldout_seeds"]):
        raise RuntimeError("INVALID_EVALUATION_SEEDS")

    output = ROOT / f"m450_{args.label}_push_off_6seeds.json"
    manifest = ROOT / f"m450_{args.label}_push_off_manifest.json"

    if args.mode == "run" and (output.exists() or manifest.exists()):
        raise RuntimeError("REFUSE_OVERWRITE_EXISTING_RESULTS")

    from paper_config import BASE, ENV

    compat_dir = BASE / "G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1"
    sys.path.insert(0, str(compat_dir))
    from nvcc_compat import apply
    print("NVCC_COMPAT:", apply(), flush=True)

    from mujoco_playground import locomotion
    import m426_heldout_det_stoch as m426

    original_get_config = locomotion.get_default_config
    if not bool(original_get_config(ENV).push_config.enable):
        raise RuntimeError("SOURCE_DEFAULT_PUSH_NOT_ENABLED")

    calls = [0]

    def get_config_push_off(name):
        cfg = original_get_config(name)
        if name == ENV:
            if not bool(cfg.push_config.enable):
                raise RuntimeError("UNEXPECTED_DEFAULT_PUSH_CONFIG")
            cfg.push_config.enable = False
            calls[0] += 1
        return cfg

    locomotion.get_default_config = get_config_push_off

    m426.WORK = ROOT
    m426.REPORT = report_path
    m426.OUT = output
    m426.SEEDS = list(SEEDS)
    m426.STEPS = 1000
    m426.SELECTED_STEPS = [401080320]

    sys.argv = ["m426_heldout_det_stoch.py", "--mode", args.mode]

    print("M450_REUSE_ORIGINAL_M426", flush=True)
    print("LABEL:", args.label, flush=True)
    print("PUSH_ENABLED: FALSE", flush=True)
    print("CHECKPOINT_SHA256:", expected_sha, flush=True)

    from unittest.mock import patch

    # Real nvcc_compat.apply() succeeded earlier in this adapter.
    # Prevent the unchanged M4.26 from calling the non-idempotent shim twice.
    with patch("nvcc_compat.apply", return_value="M450_ALREADY_APPLIED"):
        m426.main()

    if args.mode == "inspect":
        print("M450_INSPECT_PASS_NO_ROLLOUT", flush=True)
        return

    if calls[0] == 0:
        raise RuntimeError("PUSH_OFF_CONFIG_NOT_USED")

    result = json.loads(output.read_text())
    if (result["status"] != "M426_EVALUATION_COMPLETED"
            or result["seeds"] != SEEDS
            or result["max_steps"] != 1000
            or len(result["results"]) != 2
            or any(len(r["rollouts"]) != 6 for r in result["results"])):
        raise RuntimeError("EVALUATION_RESULT_INVALID")

    data = {
        "status": "M450_PUSH_OFF_COMPLETED",
        "label": args.label,
        "no_training": True,
        "checkpoint_step": 401080320,
        "checkpoint_sha256": expected_sha,
        "training_report": report_name,
        "evaluator": "m426_heldout_det_stoch.py",
        "evaluator_sha256": source_sha,
        "reset_seeds": SEEDS,
        "policy_modes": ["deterministic", "stochastic"],
        "domain_randomization_enabled": False,
        "push_enabled": False,
        "only_environment_change": "push_config.enable=False",
        "source_default_config_intercept_count": calls[0],
        "result_file": output.name,
        "result_sha256": sha256(output)
    }
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    print("M450_PUSH_OFF_PASS", args.label, flush=True)

if __name__ == "__main__":
    main()
