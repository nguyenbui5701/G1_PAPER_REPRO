#!/usr/bin/env python3
"""M4.27 G1 PPO no-train randomized-vs-nominal rollout study.

Both arms use the pinned MuJoCo Playground Brax wrapper, stochastic policy,
identical reset seeds, episode length, default environment, and checkpoint.
Only the source G1 domain randomizer differs. Batched 32-environment run;
JAX GPU required. This is NOT a paper-exact evaluation reproduction.

Must run alongside m426_heldout_det_stoch.py, train.py, paper_config.py and
pilot_20261113_170501_977563_report.json in G1_PAPER_REPRO.
"""
import argparse
import functools
import json
import math
import sys
from pathlib import Path

from paper_config import BASE, ENV
from train import check_source
from m426_heldout_det_stoch import read_entries, restore_params

WORK = Path(__file__).resolve().parent
REPORT = WORK / "pilot_20261113_170501_977563_report.json"
OUT = WORK / "m427_401m_dr_vs_nominal_32seeds.json"
SEEDS = [3001 + 37 * i for i in range(32)]
N = len(SEEDS)
STEPS = 1000
CHECKPOINT_STEP = 401_080_320


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("inspect", "run"))
    args = parser.parse_args()
    check_source()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PILOT_COMPLETED"
    entries = read_entries(report)
    if [step for step, _, _ in entries] != [CHECKPOINT_STEP]:
        raise RuntimeError("CHECKPOINT_SELECTION_MISMATCH")
    print("M427_CHECKPOINT_SHA256_OK", flush=True)

    if args.mode == "inspect":
        print("M427_INSPECT_OK NO_ROLLOUT_PERFORMED", flush=True)
        return

    compat_dir = BASE / "G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1"
    sys.path.insert(0, str(compat_dir))
    from nvcc_compat import apply
    print("NVCC_COMPAT:", apply(), flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import locomotion, wrapper

    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU_BACKEND_REQUIRED")
    params = restore_params(
        entries[0][2], serialization, running_statistics, jax, jnp
    )
    base_env = locomotion.load(
        ENV, config=locomotion.get_default_config(ENV)
    )
    probe = jax.jit(base_env.reset)(jax.random.PRNGKey(999))
    observation_size = jax.tree_util.tree_map(
        lambda x: x.shape, probe.obs
    )
    network = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=base_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **report["scaled_config"]["network_factory"],
    )
    make_policy = ppo_networks.make_inference_fn(network)
    policy = jax.jit(make_policy(params, deterministic=False))

    randomizer = locomotion.get_domain_randomizer(ENV)
    if randomizer is None:
        raise RuntimeError("G1_SOURCE_RANDOMIZER_MISSING")
    dr_rng = jax.random.split(jax.random.PRNGKey(42700), N)
    randomized_model, axes = randomizer(base_env.mjx_model, rng=dr_rng)

    # Physical-domain randomization gate. Prevent silently evaluating nominal
    # environments twice due to incorrect wrapper setup.
    baseline_mass = jnp.asarray(base_env.mjx_model.body_mass)
    randomized_mass = jnp.asarray(randomized_model.body_mass)
    mass_delta = float(jnp.max(jnp.abs(randomized_mass - baseline_mass)))
    if randomized_mass.shape[0] != N or mass_delta <= 0.0:
        raise RuntimeError("DOMAIN_RANDOMIZATION_NOT_APPLIED")
    print(f"M427_DOMAIN_RANDOMIZATION_GATE=PASS max_mass_delta={mass_delta:.6g}", flush=True)
    print("M427_POLICY=STOCHASTIC", flush=True)

    # M4.26 uses split(reset_seed) as the environment reset key.
    reset_keys = jnp.stack([
        jax.random.split(jax.random.PRNGKey(seed))[1]
        for seed in SEEDS
    ])
    results = []
    for mode in ("nominal", "source_domain_randomization"):
        environment = locomotion.load(
            ENV, config=locomotion.get_default_config(ENV)
        )
        kwargs = {}
        if mode == "source_domain_randomization":
            kwargs["randomization_fn"] = functools.partial(
                randomizer, rng=dr_rng
            )

        # This is the same wrapper used by Brax PPO training/evaluation;
        # it batches 32 envs and applies the source physics randomizer.
        wrapped = wrapper.wrap_for_brax_training(
            environment,
            episode_length=STEPS,
            action_repeat=1,
            **kwargs,
        )
        batched_reset = jax.jit(wrapped.reset)
        batched_step = jax.jit(wrapped.step)
        state = batched_reset(reset_keys)
        if jnp.asarray(state.done).shape != (N,):
            raise RuntimeError(f"UNEXPECTED_DONE_SHAPE={state.done.shape}")
        print(f"M427_BEGIN mode={mode} envs={N}", flush=True)

        first_done = np.zeros(N, dtype=np.int32)
        cum_reward = np.zeros(N, dtype=np.float64)
        # Use a dedicated identical sequence of stochastic policy RNG keys in
        # both arms. Actions diverge with different physical dynamics.
        policy_rng = jax.random.PRNGKey(42701)
        for i in range(STEPS):
            policy_rng, action_key = jax.random.split(policy_rng)
            action, _ = policy(state.obs, action_key)
            if not bool(jnp.all(jnp.isfinite(action))):
                raise RuntimeError(f"NONFINITE_ACTION mode={mode} step={i}")
            next_state = batched_step(state, action)
            step_done = np.asarray(jax.device_get(next_state.done)).reshape(N)
            reward = np.asarray(jax.device_get(next_state.reward)).reshape(N)
            if not np.isfinite(reward).all():
                raise RuntimeError(f"NONFINITE_REWARD mode={mode} step={i}")
            active = first_done == 0
            cum_reward[active] += reward[active]
            newly_finished = active & (step_done > 0.5)
            first_done[newly_finished] = i + 1
            state = next_state
            if (i + 1) % 100 == 0:
                alive = int(np.count_nonzero(first_done == 0))
                print(f"M427_PROGRESS mode={mode} step={i+1} not_finished={alive}", flush=True)
            if not (first_done == 0).any():
                break

        # The configured episode limit is exactly 1000 steps; a done on step
        # 1000 can be a time-limit truncation, not a physics termination.
        survival = np.where(first_done == 0, STEPS, first_done)
        successes = int(np.count_nonzero(survival >= STEPS))
        if np.any(survival < 1) or np.any(survival > STEPS):
            raise RuntimeError("INVALID_SURVIVAL_VALUES")
        records = [{
            "seed": int(seed),
            "steps_survived": int(survival[k]),
            "completed_1000_steps": bool(survival[k] >= STEPS),
            "episode_reward_until_first_done": float(cum_reward[k]),
        } for k, seed in enumerate(SEEDS)]
        summary = {
            "mode": mode,
            "mean_steps_survived": float(np.mean(survival)),
            "median_steps_survived": float(np.median(survival)),
            "episode_successes": successes,
            "rollouts": records,
        }
        results.append(summary)
        print(
            f"M427_RESULT mode={mode} mean_steps={summary['mean_steps_survived']:.3f} "
            f"successes={successes}/{N}", flush=True
        )

    data = {
        "status": "M427_EVALUATION_COMPLETED",
        "evaluation_type": "matched_brax_wrapper_stochastic_nominal_vs_dr",
        "environment": ENV,
        "checkpoint_step": CHECKPOINT_STEP,
        "training_repeated": False,
        "policy_mode": "stochastic",
        "num_envs": N,
        "steps_max": STEPS,
        "seeds": SEEDS,
        "same_reset_keys_between_conditions": True,
        "source_randomizer_applied": True,
        "max_body_mass_delta": mass_delta,
        "results": results,
        "limitations": (
            "Batched Brax wrapper with source domain randomization, not an "
            "identical reproduction of Brax PPO evaluator's metrics or RNG. "
            "Success here means 1000 environment steps, not necessarily good gait. "
            "No termination-cause breakdown is produced in this batched run."
        ),
    }
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("M427_EVALUATION_COMPLETED", flush=True)
    print("OUTPUT:", OUT, flush=True)


if __name__ == "__main__":
    main()
