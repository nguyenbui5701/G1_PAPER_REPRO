# M4.45 — Paper Fidelity Audit: Unitree G1 PPO

**Scope:** read-only evidence audit; no new seeds, no training, no source or checkpoint change.

## Verdict

**Hardware-scaled, partially paper-aligned PPO reproduction. NOT paper-exact.**

Training is complete and the seed-1 checkpoint at step 401,080,320 was verified, but the actual PPO learning schedule differs from the paper-target schedule because the rollout batch and environment count were reduced by four.

## Source hierarchy

- **Paper:** RSS 2025, Appendix F.2 Table XV (default locomotion PPO) and Table XX (G1Joystick), https://www.roboticsproceedings.org/rss21/p020.pdf
- **Actual train run:** m430_seed1 pilot report, manifest, completed marker and code embedded in the supplied `m430_seed1.tar.gz`.
- **Source:** pinned MuJoCo Playground Git commit `9bcbe736177472ba35d60773e3030a36e37962b7` (directly verified in local checkout).

## PPO configuration

| Parameter | Paper target | Actual scaled run | Conclusion |
|---|---:|---:|---|
| `episode_length` | 1000 | 1000 | MATCH |
| `num_timesteps` | 400000000 | 400000000 | MATCH |
| `num_evals` | 16 | 16 | MATCH |
| `num_resets_per_eval` | 1 | 1 | MATCH |
| `reward_scaling` | 0.1 | 0.1 | MATCH |
| `normalize_observations` | True | True | MATCH |
| `action_repeat` | 1 | 1 | MATCH |
| `unroll_length` | 32 | 32 | MATCH |
| `num_minibatches` | 32 | 32 | MATCH |
| `num_updates_per_batch` | 5 | 5 | MATCH |
| `discounting` | 0.98 | 0.98 | MATCH |
| `learning_rate` | 0.0001 | 0.0001 | MATCH |
| `entropy_cost` | 0.0 | 0.0 | MATCH |
| `num_envs` | 32768 | 8192 | **DIFF (hardware-scaled)** |
| `batch_size` | 1024 | 256 | **DIFF (hardware-scaled)** |
| `clipping_epsilon` | 0.2 | 0.2 | MATCH |
| `max_grad_norm` | 1.0 | 1.0 | MATCH |

Actor network `(512,256,64)`, critic `(256,256,256,256)`, policy observation key `state` and critic observation key `privileged_state` match target (Table XX + Table XV inheritance).

## PPO training schedule implications

| Metric | Paper config schedule (same Brax implementation) | Actual M4.30 run |
|---|---:|---:|
| Steps / rollout batch | 1,048,576 | 262,144 |
| Rollout batches | 390 | 1,530 |
| Scheduled environment steps | 408,944,640 | 401,080,320 |
| Optimizer minibatch updates | 62,400 | 244,800 |

The scaled training uses **3.923×** as many optimizer minibatch updates for approximately the same requested 400M environment steps. This is a verified *algorithmic schedule difference*, not a proven cause of gait or robustness behavior.

## Environment, reward, randomization and evaluation

- Actual environment: `G1JoystickFlatTerrain` (registered 2025 code). Paper Table XX labels it `G1Joystick`; exact branch-equivalence for every experiment is **NOT established**.
- Training script creates environments from `locomotion.get_default_config(ENV)` and passes `registry.get_domain_randomizer(ENV)` as `randomization_fn`; the report flags domain randomization enabled. There is no evidence of changing reward scales in that training script; this does not independently prove every source reward/observation function matches the 2025 published experiment.
- Pinned G1 source previously audited: reward categories include linear/angular tracking, feet phase/airtime/slip, pose/joint terms, contact and termination; gait phase continues to advance at zero command; default init pose is `knees_bent`. This is intentionally source behavior and was not changed.
- Pinned source includes timed random world-XY velocity pushes (`0.1–2.0 m/s` source default); diagnostic fixed commands/magnitudes in M4.35–M4.44 are **not** paper evaluation.
- Actual `num_eval_envs=2` is a hardware-driven choice. **Table XX does not specify this value**; `2` means internal training eval has high statistical uncertainty. Earlier full configuration preflight showed training-runner default `128`; this needs proper evaluation-equivalence classification, not automatic Table XX mismatch.

### Resource limitation already demonstrated

M4.22E attempted 32,768 environments on **RTX 3060 12 GB** and failed with `RESOURCE_EXHAUSTED: Out of memory` before completing training. Increasing the environment count to 32,768 without a new memory plan or larger GPU is not currently viable.

## Priorities and decision gate

1. **M4.46 (no training, no new seed):** audit exact source reward, command sampling, G1 randomizer and observation semantics against paper Section C; summarize existing gait evidence and videos. Identify which apparent gait deviations are actually source-defined versus learned-policy behavior.
2. **M4.47 (design decision):** pick either (A) a higher-memory GPU to attempt full 32,768-env Paper PPO, or (B) an explicitly labeled memory-scaled reproduction. Avoid treating changed rollout batch / optimizer update schedule as identical PPO.
3. **M4.48 (only when justified):** train a controlled new run using the committed protocol, compare to the retained 401M baseline, and report both natural gait and push robustness. Do not adjust reward, PD, termination or phase without separate experimental objective and paper justification.

## Audit execution

- Report and manifest decoded, seed and source-file hashes checked.
- Callback final checkpoint SHA-256: `1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8` (verified against checkpoint bytes in supplied archive).
- Runtime local pinned-source verification: PASS.
- No JAX/MJX simulation or seed runs performed.
