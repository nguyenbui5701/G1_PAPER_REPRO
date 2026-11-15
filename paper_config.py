
#!/usr/bin/env python3
"""Prepare the Table 20 Brax PPO configuration.
Read-only with respect to the MuJoCo Playground repository.
"""

import json
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/labrobot/nguyenbui/humanoid")
WORK = Path(__file__).resolve().parent

REPO = BASE / (
    "G1_M3_7_G1_MJX_GPU_ONE_STEP_V1/"
    "source/mujoco_playground_2025"
)

PIN = "9bcbe736177472ba35d60773e3030a36e37962b7"
ENV = "G1JoystickFlatTerrain"

PAPER = {
    "num_timesteps": 400_000_000,
    "num_evals": 16,
    "num_resets_per_eval": 1,
    "reward_scaling": 0.1,
    "unroll_length": 32,
    "num_updates_per_batch": 5,
    "discounting": 0.98,
    "learning_rate": 1e-4,
    "entropy_cost": 0.0,
    "num_envs": 32768,
    "batch_size": 1024,
    "clipping_epsilon": 0.2,
}

PAPER_NETWORK = {
    "policy_hidden_layer_sizes": (512, 256, 64),
    "value_hidden_layer_sizes": (256, 256, 256, 256),
    "value_obs_key": "privileged_state",
}


def main():
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True
    ).strip()

    if head != PIN:
        raise RuntimeError("Unexpected MuJoCo Playground commit")

    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        text=True
    ).strip()

    if status:
        raise RuntimeError("Source worktree is not clean")

    # Apply once, before any JAX/Playground import.
    m40 = BASE / "G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1"
    sys.path.insert(0, str(m40))

    from nvcc_compat import apply
    print("NVCC_COMPAT:", apply(), flush=True)

    from mujoco_playground.config import locomotion_params
    from mujoco_playground import locomotion
    from mujoco_playground import registry

    original = locomotion_params.brax_ppo_config(ENV)
    cfg = locomotion_params.brax_ppo_config(ENV)

    for key, value in PAPER.items():
        if key not in cfg:
            raise KeyError(f"Unknown PPO parameter: {key}")
        cfg[key] = value

    for key, value in PAPER_NETWORK.items():
        if key not in cfg.network_factory:
            raise KeyError(f"Unknown network parameter: {key}")
        cfg.network_factory[key] = value

    actual = cfg.to_dict()
    network = actual["network_factory"]

    for key, value in PAPER.items():
        assert actual[key] == value, key

    for key, value in PAPER_NETWORK.items():
        assert tuple(network[key]) == tuple(value) if isinstance(
            value, tuple
        ) else network[key] == value, key

    env_cfg = locomotion.get_default_config(ENV)
    randomizer = registry.get_domain_randomizer(ENV)

    report = {
        "status": "CONFIG_PREFLIGHT_OK",
        "source_commit": head,
        "environment": ENV,
        "paper_reference": "Appendix F Table 20",
        "original_ppo_config": original.to_dict(),
        "candidate_ppo_config": actual,
        "reward_base_height_scale": float(
            env_cfg.reward_config.scales.base_height
        ),
        "domain_randomizer_available": randomizer is not None,
        "domain_randomizer_enabled_in_training": "NOT_YET_VERIFIED",
        "warning": (
            "Configuration comparison only. The exact paper "
            "environment and training schedule remain to be verified."
        ),
        "training_performed": False,
    }

    output = WORK / "paper_config_preflight.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("CONFIG_PREFLIGHT_OK")
    print("Environment:", ENV)
    print("Reward base_height:", report["reward_base_height_scale"])
    print("Randomizer available:",
          report["domain_randomizer_available"])
    print("Report:", output)
    print("NO_TRAINING_PERFORMED")


if __name__ == "__main__":
    main()
