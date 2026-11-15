
#!/usr/bin/env python3
"""Unitree G1 PPO: paper config verification and GPU smoke."""

import argparse
import functools
import hashlib
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from paper_config import BASE, REPO, ENV, PIN, PAPER, PAPER_NETWORK

WORK = Path(__file__).resolve().parent


def check_source():
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        text=True,
    ).strip()

    if head != PIN or status:
        raise RuntimeError("Pinned source changed or worktree dirty")


def build_paper_config():
    from mujoco_playground.config import locomotion_params

    cfg = locomotion_params.brax_ppo_config(ENV)

    for key, value in PAPER.items():
        cfg[key] = value

    for key, value in PAPER_NETWORK.items():
        cfg.network_factory[key] = value

    report = json.loads(
        (WORK / "paper_config_preflight.json").read_text()
    )

    if report["status"] != "CONFIG_PREFLIGHT_OK":
        raise RuntimeError("Paper config preflight not passed")

        # Normalize tuple/list representation before comparison.
    actual = json.loads(json.dumps(cfg.to_dict()))
    expected = report["candidate_ppo_config"]

    if actual != expected:
        print("\nCONFIGURATION DIFFERENCES:")

        for key in sorted(set(actual) | set(expected)):
            if actual.get(key) != expected.get(key):
                print(f"\nParameter: {key}")
                print(f"  Actual:   {actual.get(key)!r}")
                print(f"  Expected: {expected.get(key)!r}")

        raise RuntimeError("Paper configuration mismatch")

    print("PAPER_CONFIG_VERIFIED=YES", flush=True)

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("inspect", "smoke"),
        required=True,
    )
    args = parser.parse_args()

    check_source()

    # Apply CUDA compatibility fix before importing JAX.
    m40 = BASE / "G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1"
    sys.path.insert(0, str(m40))

    from nvcc_compat import apply
    print("NVCC_COMPAT:", apply(), flush=True)

    import jax
    from flax import serialization
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks
    from mujoco_playground import locomotion, wrapper, registry

    paper = build_paper_config()

    print("PAPER_CONFIG_OK", flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("DEVICES:", jax.devices(), flush=True)

    if args.mode == "inspect":
        print("NO_TRAINING_PERFORMED", flush=True)
        return

    if jax.default_backend() != "gpu":
        raise RuntimeError("CUDA GPU backend required")

    # Hardware-scaled smoke only.
    # These are NOT the paper's sampling parameters.
    scaled = copy.deepcopy(paper)
    scaled.num_envs = 16
    scaled.unroll_length = 2
    scaled.num_minibatches = 1
    scaled.batch_size = 32
    scaled.num_updates_per_batch = 1
    scaled.episode_length = 16
    scaled.num_eval_envs = 2
    scaled.num_timesteps = 1152
    scaled.num_evals = 3
    # Verify hardware scaling has not modified paper settings.
    assert paper.num_envs == 32768
    assert paper.unroll_length == 32
    assert paper.batch_size == 1024

    assert scaled.num_envs == 16
    assert scaled.unroll_length == 2
    assert scaled.batch_size == 32

    print("PAPER_CONFIG_UNCHANGED=YES", flush=True)
    print("GPU_SCALED_CONFIG_OK", flush=True)  
    
     

    randomizer = registry.get_domain_randomizer(ENV)

    if randomizer is None:
        raise RuntimeError("G1 domain randomizer missing")

    # Training environment
    env_cfg = locomotion.get_default_config(ENV)
    env = locomotion.load(ENV, config=env_cfg)

    # Independent evaluation environment
    eval_env_cfg = locomotion.get_default_config(ENV)
    eval_env = locomotion.load(ENV, config=eval_env_cfg)

    if eval_env is env:
        raise RuntimeError("Training and evaluation environments must differ")

    print("SEPARATE_EVAL_ENV=YES", flush=True)

    if float(env._config.reward_config.scales.base_height) != 0.0:
        raise RuntimeError("Original reward has changed")

    network_cfg = dict(scaled.network_factory)

    train_cfg = dict(scaled)
    del train_cfg["network_factory"]

    train_cfg["seed"] = 0
    train_cfg["randomization_fn"] = randomizer

    network_factory = functools.partial(
        networks.make_ppo_networks,
        **network_cfg,
    )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = WORK / f"checkpoint_smoke_{tag}.msgpack"
    metadata = WORK / f"smoke_metrics_{tag}.json"

    if checkpoint.exists() or metadata.exists():
        raise FileExistsError("Output already exists")

    observed_steps = []

    def progress(step, metrics):
        observed_steps.append(int(step))
        print(
            f"PPO_PROGRESS step={int(step)} "
            f"reward={metrics.get('eval/episode_reward', 'n/a')}",
            flush=True,
        )

    print("SMOKE_ONLY=YES", flush=True)
    print("PAPER_REPRODUCED=NO", flush=True)

    _, params, metrics = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        progress_fn=progress,
        **train_cfg,
    )

    raw = serialization.to_bytes(params)
    restored = serialization.from_bytes(params, raw)

    if serialization.to_bytes(restored) != raw:
        raise RuntimeError("Checkpoint verification failed")

    with checkpoint.open("xb") as file:
        file.write(raw)

    result = {
        "status": "GPU_SMOKE_COMPLETED",
        "source_commit": PIN,
        "environment": ENV,
        "jax_backend": jax.default_backend(),
        "paper_reproduced": False,
        "domain_randomization_enabled": True,
        "training_steps_requested": 1152,
        "progress_steps_observed": observed_steps,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "paper_config": paper.to_dict(),
        "scaled_config": scaled.to_dict(),
    }

    with metadata.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("GPU_SMOKE_COMPLETED", flush=True)
    print("CHECKPOINT:", checkpoint, flush=True)
    print("METADATA:", metadata, flush=True)


if __name__ == "__main__":
    main()
