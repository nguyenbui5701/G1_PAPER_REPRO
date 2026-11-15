
#!/usr/bin/env python3
"""G1 PPO hardware-scaled pilot. Not full paper reproduction."""
import argparse
import copy
import functools
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from train import build_paper_config, check_source
from paper_config import BASE, ENV

WORK = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("inspect", "run"), required=True
    )
    args = parser.parse_args()

    check_source()

    # Apply existing NVCC fix before importing JAX.
    m40 = BASE / "G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1"
    sys.path.insert(0, str(m40))

    from nvcc_compat import apply
    print("NVCC_COMPAT:", apply(), flush=True)

    import jax
    import numpy as np
    from flax import serialization
    from brax.training.agents.ppo import networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import locomotion, registry, wrapper

    paper = build_paper_config()
    scaled = copy.deepcopy(paper)

    # RTX 3060 pilot: sampling reduced, paper reward/network retained.
    scaled.num_envs = 8192
    scaled.num_eval_envs = 2

    scaled.batch_size = 1024
    scaled.num_minibatches = 8
    scaled.unroll_length = 32
    scaled.num_updates_per_batch = 5

    scaled.episode_length = 1000
    scaled.num_timesteps = 262_144
    scaled.num_evals = 2

    steps_per_rollout = (
        scaled.batch_size
        * scaled.num_minibatches
        * scaled.unroll_length
        * scaled.action_repeat
    )

    assert scaled.batch_size * scaled.num_minibatches == scaled.num_envs
    assert steps_per_rollout == 262_144

    num_rollout_batches = (
        scaled.num_timesteps // steps_per_rollout
    )

    optimizer_updates = (
        num_rollout_batches
        * scaled.num_updates_per_batch
        * scaled.num_minibatches
    )

    eval_intervals = scaled.num_evals - 1

    assert scaled.num_timesteps % steps_per_rollout == 0
    assert num_rollout_batches == 1
    assert optimizer_updates == 40
    assert scaled.num_evals == 2
    assert num_rollout_batches % eval_intervals == 0

    print("M420C_NUM_ENVS:", scaled.num_envs)
    print("M420C_NUM_EVALS:", scaled.num_evals)
    print("M420C_TRAIN_STEPS:", scaled.num_timesteps)
    print("M420C_STEPS_PER_ROLLOUT:", steps_per_rollout)
    print("M420C_ROLLOUT_BATCHES:", num_rollout_batches)
    print("M420C_OPTIMIZER_UPDATES:", optimizer_updates)

    if args.mode == "inspect":
        print("NO_TRAINING_PERFORMED")
        return

    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU_BACKEND_REQUIRED")

    randomizer = registry.get_domain_randomizer(ENV)
    if randomizer is None:
        raise RuntimeError("MISSING_G1_RANDOMIZER")

    env = locomotion.load(
        ENV, config=locomotion.get_default_config(ENV)
    )
    eval_env = locomotion.load(
        ENV, config=locomotion.get_default_config(ENV)
    )

    if env is eval_env:
        raise RuntimeError("TRAIN_AND_EVAL_ENV_MUST_DIFFER")

    for e in (env, eval_env):
        if float(e._config.reward_config.scales.base_height) != 0.0:
            raise RuntimeError("REWARD_CONFIG_CHANGED")

    cfg = scaled.to_dict()
    net_cfg = cfg.pop("network_factory")
    cfg.update(seed=0, randomization_fn=randomizer)

    network_factory = functools.partial(
        networks.make_ppo_networks, **net_cfg
    )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag += f"_{os.getpid()}"

    metadata = WORK / f"pilot_{tag}_report.json"
    progress_history = []
    checkpoints = []

    report = {
        "status": "STARTED",
        "paper_reproduced": False,
        "environment": ENV,
        "domain_randomization_enabled": True,
        "paper_config": paper.to_dict(),
        "scaled_config": scaled.to_dict(),
        "progress": progress_history,
        "checkpoints": checkpoints,
    }

    def persist():
        tmp = metadata.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        tmp.replace(metadata)

    if metadata.exists():
        raise FileExistsError(str(metadata))

    persist()

    def progress(step, metrics):
        value = metrics.get("eval/episode_reward")
        reward = None if value is None else float(jax.device_get(value))

        if reward is not None and not math.isfinite(reward):
            raise RuntimeError("NONFINITE_EVAL_REWARD")

        progress_history.append({
            "step": int(step),
            "eval_reward": reward,
        })
        persist()
        print(
            f"PILOT_PROGRESS step={step} reward={reward}",
            flush=True,
        )

    def save_policy(step, _make_policy, params):
        if int(step) == 0:
            return

        raw = serialization.to_bytes(params)
        output = WORK / (
            f"checkpoint_pilot_{tag}_step_{int(step):09d}.msgpack"
        )

        with output.open("xb") as f:
            f.write(raw)

        digest = hashlib.sha256(raw).hexdigest()

        checkpoints.append({
            "step": int(step),
            "file": str(output),
            "sha256": digest,
        })
        persist()
        print("PILOT_CHECKPOINT:", output.name, flush=True)

    try:
        _, params, _ = ppo.train(
            environment=env,
            eval_env=eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            network_factory=network_factory,
            progress_fn=progress,
            policy_params_fn=save_policy,
            **cfg,
        )

        expected = scaled.num_timesteps

        if (
            not progress_history
            or progress_history[-1]["step"] != expected
        ):
            raise RuntimeError("STEP_BUDGET_NOT_COMPLETED")

        if (
            not checkpoints
            or checkpoints[-1]["step"] != expected
        ):
            raise RuntimeError("FINAL_CHECKPOINT_NOT_SAVED")

        # Preserve the final parameters returned by PPO.
        final = serialization.to_bytes(params)
        final_path = WORK / f"checkpoint_pilot_{tag}_returned_final.msgpack"

        with final_path.open("xb") as f:
            f.write(final)

        # Restore the callback checkpoint into the final parameter structure.
        callback_path = Path(checkpoints[-1]["file"])
        callback_raw = callback_path.read_bytes()

        # Callback has (normalizer, {"policy": actor, "value": critic}).
        # PPO final return has (normalizer, actor, critic).     

        if len(params) != 3:
            raise RuntimeError("UNEXPECTED_PPO_RETURN_SCHEMA")      

        callback_target = (
            params[0],
            {
                "policy": params[1],
                "value": params[2],
            },
        )       

        restored_callback = serialization.from_bytes(
            callback_target,
            callback_raw,
        )       

        # Convert restored callback into the same 3-part layout.
        callback_params = (
            restored_callback[0],
            restored_callback[1]["policy"],
            restored_callback[1]["value"],
        )

        final_leaves = jax.tree_util.tree_leaves(params)
        callback_leaves = jax.tree_util.tree_leaves(callback_params)

        if len(final_leaves) != len(callback_leaves):
            raise RuntimeError("FINAL_PARAMETER_TREE_LENGTH_MISMATCH")

        mismatched_leaves = []

        for i, (x, y) in enumerate(zip(final_leaves, callback_leaves)):
            a = np.asarray(jax.device_get(x))
            b = np.asarray(jax.device_get(y))

            if not np.array_equal(a, b):
                mismatched_leaves.append(i)

        report["final_checkpoint_comparison"] = {
            "returned_final_file": str(final_path),
            "returned_final_sha256": hashlib.sha256(final).hexdigest(),
            "callback_sha256": checkpoints[-1]["sha256"],
            "serialized_bytes_equal": (
                hashlib.sha256(final).hexdigest()
                == checkpoints[-1]["sha256"]
            ),
            "tensor_values_equal": len(mismatched_leaves) == 0,
            "mismatched_leaf_indices": mismatched_leaves,
        }

        print(
            "FINAL_TENSOR_VALUES_EQUAL=",
            len(mismatched_leaves) == 0,
            flush=True,
        )

        for leaf in jax.tree_util.tree_leaves(params):
            value = np.asarray(jax.device_get(leaf))
            if (
                np.issubdtype(value.dtype, np.number)
                and not np.all(np.isfinite(value))
            ):
                raise RuntimeError("NONFINITE_CHECKPOINT_PARAMETERS")

        if mismatched_leaves:
            report["status"] = "PILOT_FINISHED_FINAL_MISMATCH"
        else:
            report["status"] = "PILOT_COMPLETED"

        persist()
        print(report["status"], flush=True)

    except Exception as exc:
        report["status"] = "PILOT_FAILED"
        report["error"] = repr(exc)
        persist()
        raise


if __name__ == "__main__":
    main()
