#!/usr/bin/env python3
"""M4.33 | Read-only interactive MuJoCo viewer for G1 checkpoint comparison.

Two training seeds (0/1), one holdout evaluation seed at a time.
Physics and observations are advanced by the SAME MJX environment and PPO
inference as M4.26. A separate CPU MuJoCo MjData is used for drawing only.
No training, reward changes, reset overrides or source patches.

Usage:
  python m433_visual_validation.py --mode inspect --seed 5186 --train-seed 1
  python m433_visual_validation.py --mode run --seed 5186 --train-seed 1

Viewer keys: Space pause/resume; R restart episode; Q quit. GUI mouse for camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parent
FINAL_STEP = 401_080_320
HOLDOUT_REPORT = WORK / "m432_new_holdout" / "m432_seed0_vs_seed1_new32.json"
SUGGESTED = {5186, 5149, 5667, 5593, 5112, 5889}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def preflight(train_seed: int, eval_seed: int, mode: str) -> dict:
    import m431_seed1_matched_eval as m431
    ctx = m431.validate()  # source/checkpoint SHA and identical PPO config
    data = json.loads(HOLDOUT_REPORT.read_text(encoding="utf-8"))
    if (data.get("status") != "M432_NEW_HELDOUT_COMPARISON_COMPLETED"
            or data.get("checkpoint_step") != FINAL_STEP
            or data.get("training_repeated") is not False
            or len(data.get("new_heldout_seeds", [])) != 32
            or set(data.get("new_heldout_seeds", [])) & set(data.get("prior_m431_seeds", []))):
        raise RuntimeError("M432_HOLDOUT_REPORT_INVALID")
    if (data.get("seed0_checkpoint_sha256") != ctx["seed0_sha256"]
            or data.get("seed1_checkpoint_sha256") != ctx["seed1_sha256"]):
        raise RuntimeError("M432_CHECKPOINT_PROVENANCE_MISMATCH")
    if eval_seed not in data["new_heldout_seeds"]:
        raise RuntimeError(f"EVALUATION_SEED_NOT_IN_M432_HOLDOUT: {eval_seed}")
    by_mode = {entry["mode"]: entry for entry in data["metrics_by_protocol"]}
    key = "deterministic" if mode == "deterministic" else "stochastic"
    baseline = by_mode[key]
    pair = next((x for x in baseline["paired_survival"] if x["seed"] == eval_seed), None)
    if pair is None:
        raise RuntimeError("NO_MATCHED_SEED_IN_M432")
    expected = int(pair[f"seed{train_seed}_steps"])
    if train_seed == 0:
        report_path = m431.BASELINE_REPORT
        checkpoint_sha = ctx["seed0_sha256"]
        root = WORK
    else:
        report_path = ctx["report"]
        checkpoint_sha = ctx["seed1_sha256"]
        root = m431.SEED1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entries = [x for x in report["checkpoints"] if int(x["step"]) == FINAL_STEP]
    if len(entries) != 1:
        raise RuntimeError("FINAL_STEP_CHECKPOINT_ENTRY_INVALID")
    entry = entries[0]
    path = root / Path(entry["file"]).name
    if (path.is_symlink() or not path.is_file()
            or path.resolve().parent != root.resolve()
            or entry["sha256"] != checkpoint_sha or _sha(path) != checkpoint_sha):
        raise RuntimeError("FINAL_CHECKPOINT_HASH_OR_PATH_INVALID")
    print(f"M433_PREFLIGHT_OK train_seed={train_seed} eval_seed={eval_seed} "
          f"policy={mode} expected_steps={expected} checkpoint_sha={checkpoint_sha[:16]}",
          flush=True)
    return {"checkpoint_path": path, "report": report,
            "expected_steps": expected, "checkpoint_sha256": checkpoint_sha}


def run(args, ctx):
    # This must happen before JAX is imported in the current process.
    from paper_config import BASE, ENV
    from train import check_source
    check_source()
    sys.path.insert(0, str(BASE / "G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1"))
    from nvcc_compat import apply
    print("NVCC_COMPAT:", apply(), flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import locomotion
    import mujoco
    import mujoco.viewer
    from m426_heldout_det_stoch import restore_params

    if jax.default_backend() != "gpu":
        raise RuntimeError("M433_GPU_BACKEND_REQUIRED")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("M433_GRAPHICAL_DESKTOP_REQUIRED: use Ubuntu GUI, not SSH-only")

    env = locomotion.load(ENV, config=locomotion.get_default_config(ENV))
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    probe = reset_fn(jax.random.PRNGKey(999))
    observation_size = jax.tree_util.tree_map(lambda x: x.shape, probe.obs)
    net = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ctx["report"]["scaled_config"]["network_factory"],
    )
    params = restore_params(ctx["checkpoint_path"].read_bytes(),
                            serialization, running_statistics, jax, jnp)
    make_policy = ppo_networks.make_inference_fn(net)
    policy = jax.jit(make_policy(params, deterministic=args.policy == "deterministic"))

    # The CPU model is never stepped; it is a pose mirror of the MJX trajectory.
    model = env.mj_model
    if not isinstance(model, mujoco.MjModel):
        raise RuntimeError("M433_ENV_MJ_MODEL_UNAVAILABLE")
    view_data = mujoco.MjData(model)
    dt = float(env.dt)
    if not 0 < dt <= 0.1:
        raise RuntimeError(f"INVALID_ENV_DT: {dt}")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def reset_episode():
        key = jax.random.PRNGKey(args.seed)
        key, reset_key = jax.random.split(key)
        return reset_fn(reset_key), key, 0

    def mirror(state):
        np.copyto(view_data.qpos, np.asarray(jax.device_get(state.data.qpos)))
        np.copyto(view_data.qvel, np.asarray(jax.device_get(state.data.qvel)))
        if getattr(state.data, "time", None) is not None:
            view_data.time = float(jax.device_get(state.data.time))
        mujoco.mj_forward(model, view_data)

    control = {"paused": False, "restart": False, "quit": False}

    def key_callback(keycode):
        if keycode == 32:  # space
            control["paused"] = not control["paused"]
        elif keycode in (ord("r"), ord("R")):
            control["restart"] = True
        elif keycode in (ord("q"), ord("Q")):
            control["quit"] = True

    state, key, steps = reset_episode()
    mirror(state)
    print("M433_VIEWER_START: Space=pause/resume, R=reset, Q=quit; "
          "mouse=rotate/pan/zoom. CPU model is visualization only.", flush=True)

    with mujoco.viewer.launch_passive(model, view_data, key_callback=key_callback,
                                     show_left_ui=True, show_right_ui=False) as viewer:
        if pelvis_id >= 0:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = pelvis_id
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -15
        viewer.sync()
        episode_done = False
        while viewer.is_running() and not control["quit"]:
            if control["restart"]:
                state, key, steps = reset_episode()
                episode_done = False
                control["restart"] = False
                with viewer.lock():
                    mirror(state)
                viewer.sync()
                print("M433_EPISODE_RESTART", flush=True)
            if episode_done or control["paused"]:
                viewer.sync()
                time.sleep(0.04)
                continue

            start = time.perf_counter()
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            if not bool(jnp.all(jnp.isfinite(action))):
                raise RuntimeError(f"NONFINITE_POLICY_ACTION: step={steps + 1}")
            state = step_fn(state, action)
            steps += 1
            with viewer.lock():
                mirror(state)
            viewer.sync()
            if steps % 50 == 0:
                z = float(jax.device_get(state.data.qpos[2]))
                print(f"M433_PROGRESS step={steps} sim_t={steps*dt:.2f} "
                      f"root_z={z:.3f}", flush=True)
            if bool(state.done) or steps >= 1000:
                terminated = bool(state.done)
                verdict = "MATCH" if steps == ctx["expected_steps"] else "MISMATCH"
                print(f"M433_EPISODE_END train_seed={args.train_seed} seed={args.seed} "
                      f"policy={args.policy} steps={steps} terminated={terminated} "
                      f"expected={ctx['expected_steps']} replay={verdict}", flush=True)
                episode_done = True
                if verdict == "MISMATCH":
                    print("M433_WARNING: different episode length than M4.32. "
                          "Do not interpret visuals as a matched replay.", flush=True)
            delay = dt / args.speed - (time.perf_counter() - start)
            if delay > 0:
                time.sleep(delay)
        print("M433_VIEWER_CLOSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "run"), required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=5186)
    parser.add_argument("--policy", choices=("deterministic", "stochastic"),
                        default="deterministic")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="target simulation playback speed multiplier")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0.1 <= args.speed <= 5.0:
        parser.error("--speed must be within [0.1, 5.0]")
    ctx = preflight(args.train_seed, args.seed, args.policy)
    if args.mode == "inspect":
        print("M433_INSPECT_PASS NO_SIMULATION NO_TRAINING", flush=True)
        return
    run(args, ctx)


if __name__ == "__main__":
    main()
