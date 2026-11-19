
#!/usr/bin/env python3
"""Evaluate existing G1 PPO pilot checkpoints. No training."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from paper_config import BASE, ENV
from train import check_source

WORK = Path(__file__).resolve().parent
REPORT = WORK / "pilot_20261113_170501_977563_report.json"
OUT = WORK / "m426_401m_heldout_det_stoch_32seeds.json"

SEEDS = [3001 + 37 * i for i in range(32)]
STEPS = 1000
SELECTED_STEPS = [401_080_320]


def read_entries(report):
    entries = {
        int(x["step"]): x for x in report["checkpoints"]
    }

    selected = []

    for step in SELECTED_STEPS:
        entry = entries[step]
        path = WORK / Path(entry["file"]).name

        if path.is_symlink() or path.resolve().parent != WORK.resolve():
            raise RuntimeError("INVALID_CHECKPOINT_PATH")

        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()

        if digest != entry["sha256"]:
            raise RuntimeError(f"INVALID_SHA256 step={step}")

        selected.append((step, path, raw))

    return selected


def to_jax_tree(x, jax, jnp):
    return jax.tree_util.tree_map(jnp.asarray, x)


def restore_params(raw, serialization, running_statistics, jax, jnp):
    state = serialization.msgpack_restore(raw)

    if isinstance(state, dict):
        try:
            indexed = {int(k): v for k, v in state.items()}
        except (ValueError, TypeError):
            raise RuntimeError(
                f"UNEXPECTED_ROOT_KEYS: {list(state)}"
            )

        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise RuntimeError("NONCONTIGUOUS_PARAMETER_KEYS")

        parts = [indexed[i] for i in expected]

    elif isinstance(state, (tuple, list)):
        parts = list(state)

    else:
        raise RuntimeError("INVALID_CHECKPOINT_ROOT")

    if len(parts) not in (2, 3):
        raise RuntimeError(
            f"UNEXPECTED_PARAMETER_COUNT={len(parts)}"
        )

    normalizer_state = parts[0]

    if not isinstance(normalizer_state, dict):
        raise RuntimeError("INVALID_NORMALIZER_STRUCTURE")

    if not {"mean", "std"}.issubset(normalizer_state):
        raise RuntimeError(
            f"MISSING_NORMALIZER_FIELDS: {list(normalizer_state)}"
        )

    normalizer = running_statistics.NestedMeanStd(
        mean=to_jax_tree(normalizer_state["mean"], jax, jnp),
        std=to_jax_tree(normalizer_state["std"], jax, jnp),
    )

    # Inspect the actual checkpoint parameter container.
    actor_state = parts[1]

    if not isinstance(actor_state, dict):
        raise RuntimeError(
            f"INVALID_ACTOR_CONTAINER: {type(actor_state).__name__}"
        )

    print(
        "CHECKPOINT_PARAMETER_KEYS:",
        sorted(str(k) for k in actor_state.keys()),
        flush=True,
    )

    # Some checkpoint formats store actor and critic together.
    # Extract only the actor for deterministic inference.
    if "policy" in actor_state:
        actor_state = actor_state["policy"]
        print("ACTOR_EXTRACTED_FROM_PPO_CONTAINER", flush=True)

    if not isinstance(actor_state, dict):
        raise RuntimeError("INVALID_ACTOR_STATE")

    # Flax expects {'params': {'hidden_0': ..., ...}}.
    if "params" not in actor_state:
        if "hidden_0" in actor_state:
            actor_state = {"params": actor_state}
            print("FLAX_PARAMS_WRAPPER_RESTORED", flush=True)
        else:
            raise RuntimeError(
                "ACTOR_PARAMS_NOT_FOUND: "
                f"{list(actor_state.keys())}"
            )

    if (
        not isinstance(actor_state["params"], dict)
        or "hidden_0" not in actor_state["params"]
    ):
        raise RuntimeError(
            "INVALID_FLAX_ACTOR_LAYOUT: "
            f"{list(actor_state['params'].keys())}"
        )

    actor_params = to_jax_tree(actor_state, jax, jnp)

    print("ACTOR_PARAMETER_LAYOUT_OK", flush=True)

    return (normalizer, actor_params)


def rmse(errors):
    if not errors:
        return None
    return math.sqrt(sum(errors) / len(errors))


def classify_done(env, data):
    """Reproduce G1 pinned-source termination predicates."""
    import jax.numpy as jnp
    from mujoco_playground._src.locomotion.g1 import (
        joystick as g1_joystick
    )

    collision = g1_joystick.collision

    gravity_z = float(
        env.get_gravity(data, "torso")[-1]
    )

    reasons = {
        "fall_torso": gravity_z < 0.0,

        "right_foot_left_foot": bool(
            collision.geoms_colliding(
                data,
                env._right_foot_geom_id,
                env._left_foot_geom_id,
            )
        ),

        "left_foot_right_shin": bool(
            collision.geoms_colliding(
                data,
                env._left_foot_geom_id,
                env._right_shin_geom_id,
            )
        ),

        "right_foot_left_shin": bool(
            collision.geoms_colliding(
                data,
                env._right_foot_geom_id,
                env._left_shin_geom_id,
            )
        ),

        "qpos_nan": bool(jnp.isnan(data.qpos).any()),
        "qvel_nan": bool(jnp.isnan(data.qvel).any()),
    }

    return {
        "torso_gravity_z": gravity_z,
        "reasons": reasons,
        "triggered": [
            key for key, active in reasons.items()
            if active
        ],
        "recomputed_done": any(reasons.values()),
    }



def rollout(env, reset_fn, step_fn, policy_fn,
            params, seed, jax, jnp, deterministic=True):
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)

    policy = jax.jit(policy_fn(params, deterministic=deterministic))

    samples = []
    terminated = False
    termination_diagnosis = None

    for index in range(STEPS):
        key, action_key = jax.random.split(key)

        if "command" not in state.info:
            raise RuntimeError("MISSING_COMMAND_IN_ENV_INFO")

        command = jnp.asarray(state.info["command"])

        if command.shape != (3,):
            raise RuntimeError(
                f"UNEXPECTED_COMMAND_SHAPE: {command.shape}"
            )

        # Body-frame velocities before the transition.
        linvel = env.get_local_linvel(state.data, "pelvis")
        gyro = env.get_gyro(state.data, "pelvis")

        action, _ = policy(state.obs, action_key)

        if not bool(jnp.all(jnp.isfinite(action))):
            raise RuntimeError(
                f"NONFINITE_ACTION seed={seed} step={index}"
            )

        next_state = step_fn(state, action)

        # Record environment-defined termination indicators.
        termination_metrics = {}

        for name, value in next_state.metrics.items():
            if str(name).startswith("termination/"):
                metric = jnp.asarray(value)

                if metric.size != 1:
                    raise RuntimeError(
                        f"UNEXPECTED_TERMINATION_METRIC_SHAPE: {name}"
                    )

                termination_metrics[str(name)] = float(
                    jnp.ravel(metric)[0]
                )

        qpos_has_nan = bool(jnp.isnan(next_state.data.qpos).any())
        qvel_has_nan = bool(jnp.isnan(next_state.data.qvel).any())

        cmd = [float(v) for v in command]
        vel = [float(v) for v in linvel]
        angular = [float(v) for v in gyro]

        xy_error2 = (
            (vel[0] - cmd[0]) ** 2
            + (vel[1] - cmd[1]) ** 2
        )
        yaw_error2 = (angular[2] - cmd[2]) ** 2

        height = float(next_state.data.qpos[2])
        reward = float(next_state.reward)
        done = bool(next_state.done)

        sample = {
            "step": index + 1,
            "command": cmd,
            "actual_xy": vel[:2],
            "actual_yaw": angular[2],
            "root_height": height,
            "reward": reward,
            "done": done,
            "termination_metrics": termination_metrics,
            "qpos_has_nan": qpos_has_nan,
            "qvel_has_nan": qvel_has_nan,
            "xy_error2": xy_error2,
            "yaw_error2": yaw_error2,
        }

        if "last_contact" in next_state.info:
            sample["foot_contact"] = [
                bool(x) for x in
                jnp.ravel(next_state.info["last_contact"])
            ]

        if not all(math.isfinite(x) for x in (
            height, reward, xy_error2, yaw_error2
        )):
            raise RuntimeError(
                f"NONFINITE_ROLLOUT seed={seed} step={index}"
            )

        samples.append(sample)
        state = next_state

        if done:
            terminated = True

            termination_diagnosis = classify_done(
                env, next_state.data
            )

            print(
                f"TERMINATION seed={seed} "
                f"step={index + 1} "
                f"reasons={termination_diagnosis['triggered']} "
                f"gravity_z={termination_diagnosis['torso_gravity_z']:.5f}",
                flush=True,
            )

            if not termination_diagnosis["recomputed_done"]:
                raise RuntimeError(
                    f"DONE_REASON_MISMATCH seed={seed} step={index + 1}"
                )

            break

    first18 = samples[:18]

    return {
        "seed": seed,
        "steps_survived": len(samples),
        "terminated": terminated,
        "terminal_metrics": samples[-1]["termination_metrics"],
        "terminal_qpos_has_nan": samples[-1]["qpos_has_nan"],
        "terminal_qvel_has_nan": samples[-1]["qvel_has_nan"],
        "final_height": samples[-1]["root_height"],
        "mean_height": sum(
            x["root_height"] for x in samples
        ) / len(samples),
        "xy_rmse": rmse([
            x["xy_error2"] for x in samples
        ]),
        "yaw_rmse": rmse([
            x["yaw_error2"] for x in samples
        ]),
        "first18_coverage": len(first18),
        "first18_xy_rmse": rmse([
            x["xy_error2"] for x in first18
        ]),
        "first18_yaw_rmse": rmse([
            x["yaw_error2"] for x in first18
        ]),
        "samples": samples,
        "terminal_diagnosis": termination_diagnosis,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True,
        choices=["inspect", "run"]
    )
    args = parser.parse_args()

    check_source()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    entries = read_entries(report)

    print("CHECKPOINT_SHA256_OK", flush=True)

    m40 = BASE / "G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1"
    sys.path.insert(0, str(m40))

    from nvcc_compat import apply
    print("NVCC_COMPAT:", apply(), flush=True)

    import jax
    import jax.numpy as jnp
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    restored = {}

    for step, _, raw in entries:
        restored[step] = restore_params(
            raw, serialization, running_statistics,
            jax, jnp
        )
        print(
            f"RESTORE_OK step={step} "
            f"parts={len(restored[step])}",
            flush=True
        )

    if args.mode == "inspect":
        print("EVAL_INSPECT_OK")
        print("NO_ROLLOUT_PERFORMED")
        return

    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU_BACKEND_REQUIRED")

    from mujoco_playground import locomotion

    env = locomotion.load(
        ENV,
        config=locomotion.get_default_config(ENV),
    )

    if not all(hasattr(env, name) for name in (
        "get_local_linvel", "get_gyro"
    )):
        raise RuntimeError("MISSING_VELOCITY_ACCESSORS")

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    sample = reset_fn(jax.random.PRNGKey(999))
    observation_size = jax.tree_util.tree_map(
        lambda x: x.shape, sample.obs
    )

    net_cfg = report["scaled_config"]["network_factory"]

    net = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **net_cfg,
    )

    # Validate actor parameter keys and shapes against the network.
    from flax import traverse_util

    reference_actor = net.policy_network.init(
        jax.random.PRNGKey(0)
    )

    reference_tree = serialization.to_state_dict(reference_actor)
    reference_flat = traverse_util.flatten_dict(reference_tree)

    for checkpoint_step, restored_params in restored.items():
        actor_tree = serialization.to_state_dict(
            restored_params[1]
        )
        actor_flat = traverse_util.flatten_dict(actor_tree)

        expected_keys = set(reference_flat)
        actual_keys = set(actor_flat)

        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)

            raise RuntimeError(
                f"ACTOR_PARAMETER_KEYS_MISMATCH "
                f"step={checkpoint_step} "
                f"missing={missing} unexpected={unexpected}"
            )

        for key in reference_flat:
            expected_shape = reference_flat[key].shape
            actual_shape = actor_flat[key].shape

            if expected_shape != actual_shape:
                raise RuntimeError(
                    f"ACTOR_SHAPE_MISMATCH "
                    f"step={checkpoint_step} "
                    f"layer={key} "
                    f"expected={expected_shape} "
                    f"actual={actual_shape}"
                )

        print(
            f"ACTOR_NETWORK_MATCH_OK "
            f"step={checkpoint_step} "
            f"tensors={len(actor_flat)}",
            flush=True,
        )

    make_policy = ppo_networks.make_inference_fn(net)

    results = []

    for deterministic in (True, False):
        mode_name = "deterministic" if deterministic else "stochastic"
        for step in SELECTED_STEPS:
            runs = []
            for seed in SEEDS:
                run = rollout(
                    env, reset_fn, step_fn,
                    make_policy, restored[step], seed,
                    jax, jnp, deterministic=deterministic
                )
                # Keep diagnostics but avoid duplicating full long trajectories.
                runs.append({k: v for k, v in run.items() if k != "samples"})
                print(
                    f"M426_EVAL mode={mode_name} step={step} seed={seed} "
                    f"survived={run['steps_survived']} "
                    f"xy_rmse={run['xy_rmse']:.4f} "
                    f"yaw_rmse={run['yaw_rmse']:.4f}",
                    flush=True,
                )

            results.append({
                "policy_mode": mode_name,
                "checkpoint_step": step,
                "mean_steps_survived": sum(
                    x["steps_survived"] for x in runs
                ) / len(runs),
                "episode_successes": sum(
                    not x["terminated"] and x["steps_survived"] == STEPS
                    for x in runs
                ),
                "terminated_count": sum(int(x["terminated"]) for x in runs),
                "rollouts": runs,
            })

    output = {
        "status": "M426_EVALUATION_COMPLETED",
        "experiment_class": "hardware_scaled_8192",
        "environment": ENV,
        "training_repeated": False,
        "evaluation_domain_randomization": False,
        "evaluation_observation_noise": "environment_default",
        "policy_modes": ["deterministic", "stochastic"],
        "heldout_seeds": True,
        "seeds": SEEDS,
        "max_steps": STEPS,
        "results": results,
    }
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("M426_EVALUATION_COMPLETED")
    print("OUTPUT:", OUT)


if __name__ == "__main__":
    main()
