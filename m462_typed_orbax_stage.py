#!/usr/bin/env python3
"""M4.62: safe Flax callback .msgpack -> typed Brax 0.12.1 Orbax checkpoint.

Stages weights-only; NEVER resumes Adam optimizer. Validates all normalizer,
actor and critic tensor leaves with an actual ppo.train(num_timesteps=0) restore.

This program is for the user's PINNED 2025 MuJoCo Playground + Brax 0.12.1.
No package installs, source edits, PPO rollout, or changes to baseline 401M.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys

CHECKPOINT_SHA = '1ccf579ea35d4518713b00ea20d77db8463bc469b7de903b8f10f457b3dc90b8'
PPO_TRAIN_SHA = 'e20613d42bf0868a13003e0312998816cc8d3593bb5ccaf9f1d070fd09c44bd7'
SOURCE_COMMIT = '9bcbe736177472ba35d60773e3030a36e37962b7'
BASELINE_STEP = 401_080_320


def checkpoint_parts(obj):
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        norm, models = obj
    elif isinstance(obj, dict) and set(obj) == {'0', '1'}:
        norm, models = obj['0'], obj['1']
    else:
        raise ValueError('M462_UNEXPECTED_CALLBACK_SCHEMA')
    if not isinstance(norm, dict) or set(norm) != {'mean', 'std', 'count', 'summed_variance'}:
        raise ValueError('M462_NORMALIZER_SCHEMA_CHANGED')
    if not isinstance(models, dict) or set(models) != {'policy', 'value'}:
        raise ValueError('M462_ACTOR_CRITIC_SCHEMA_CHANGED')
    return norm, models['policy'], models['value']


def orbax_directory_digest(path):
    path=Path(path)
    files=sorted(x for x in path.rglob('*') if x.is_file())
    if not files:
        raise RuntimeError('M462_EMPTY_ORBAX_CHECKPOINT_DIR')
    h=hashlib.sha256()
    for x in files:
        rel=x.relative_to(path).as_posix()
        h.update(rel.encode('utf-8')+b'\0')
        with x.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                h.update(chunk)
    return h.hexdigest(),len(files)


def shape_dtype_tree(tree, jax):
    return [(tuple(v.shape), str(v.dtype)) for v in jax.tree_util.tree_leaves(tree)]


def compare_exact(expected, obtained, jax, np, label):
    a = jax.tree_util.tree_leaves(expected)
    b = jax.tree_util.tree_leaves(obtained)
    if len(a) != len(b):
        raise RuntimeError(f'M462_{label}_LEAF_COUNT_MISMATCH expected={len(a)} got={len(b)}')
    if jax.tree_util.tree_structure(expected) != jax.tree_util.tree_structure(obtained):
        raise RuntimeError(f'M462_{label}_TREE_STRUCTURE_MISMATCH')
    for i, (x, y) in enumerate(zip(a, b)):
        u, v = np.asarray(jax.device_get(x)), np.asarray(jax.device_get(y))
        if u.shape != v.shape or u.dtype != v.dtype or not np.array_equal(u, v):
            raise RuntimeError(f'M462_{label}_TENSOR_MISMATCH index={i} shape={u.shape} dtype={u.dtype}')
    return len(a)


def do_stage(args):
    from train import check_source, build_paper_config
    from paper_config import BASE, REPO, PIN, ENV

    check_source()
    if PIN != SOURCE_COMMIT:
        raise RuntimeError('M462_SOURCE_PIN_CHANGED')

    sys.path.insert(0, str(BASE/'G1_M4_0_G1_PPO_ONE_UPDATE_SMOKE_V1'))
    from nvcc_compat import apply
    print('M462_NVCC_COMPAT', apply(), flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import losses, networks, train as ppo
    from mujoco_playground import locomotion, registry, wrapper
    from orbax import checkpoint as ocp

    if jax.default_backend() != 'gpu':
        raise RuntimeError('M462_GPU_BACKEND_REQUIRED')
    source_py = Path(inspect.getfile(ppo.train)).resolve()
    actual_train_sha = hashlib.sha256(source_py.read_bytes()).hexdigest()
    if actual_train_sha != PPO_TRAIN_SHA:
        raise RuntimeError('M462_BRAX_PPO_SOURCE_SHA_MISMATCH '+actual_train_sha)
    if 'restore_checkpoint_path' not in inspect.signature(ppo.train).parameters:
        raise RuntimeError('M462_BRAX_RESTORE_PATH_API_MISSING')
    if (args.out_dir.exists() or args.report.exists()):
        raise FileExistsError('M462_REFUSE_STAGE_OR_REPORT_OVERWRITE')

    raw = args.checkpoint.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA:
        raise RuntimeError('M462_BASELINE_CHECKPOINT_SHA_MISMATCH')
    paper = build_paper_config()
    ck_normalizer, ck_actor, ck_critic = checkpoint_parts(serialization.msgpack_restore(raw))

    mean = ck_normalizer['mean']
    obs_shapes = {k:tuple(np.asarray(v).shape) for k,v in mean.items()}
    if set(obs_shapes) != {'state','privileged_state'} or obs_shapes['state'] != (103,) or obs_shapes['privileged_state'] != (216,):
        raise RuntimeError('M462_OBSERVATION_DIMENSION_CHANGED '+str(obs_shapes))

    train_env = locomotion.load(ENV, config=locomotion.get_default_config(ENV))
    if float(train_env._config.reward_config.scales.base_height) != 0.0:
        raise RuntimeError('M462_REWARD_OR_SOURCE_CHANGED')
    net_cfg = dict(paper.network_factory)
    nets = functools.partial(networks.make_ppo_networks, **net_cfg)(
        obs_shapes, train_env.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    # Rebuild types that the installed Brax PPO expects at its restore site.
    template_norm = running_statistics.init_state({
        key:jnp.zeros(shape, dtype=jnp.float32) for key,shape in obs_shapes.items()})
    template_actor = nets.policy_network.init(jax.random.PRNGKey(11))
    template_critic = nets.value_network.init(jax.random.PRNGKey(17))
    typed_norm = serialization.from_state_dict(template_norm, ck_normalizer)
    typed_actor = serialization.from_state_dict(template_actor, ck_actor)
    typed_critic = serialization.from_state_dict(template_critic, ck_critic)
    typed = (typed_norm, losses.PPONetworkParams(policy=typed_actor, value=typed_critic))
    source_parts = (typed_norm, typed_actor, typed_critic)
    sizes = {name: len(jax.tree_util.tree_leaves(x)) for name,x in zip(('normalizer','actor','critic'),source_parts)}
    if sizes != {'normalizer':7,'actor':8,'critic':10}:
        raise RuntimeError('M462_CHECKPOINT_LEAF_COUNTS_DIFFER '+str(sizes))
    if any(not np.all(np.isfinite(np.asarray(jax.device_get(v)))) for v in jax.tree_util.tree_leaves(typed)):
        raise RuntimeError('M462_NONFINITE_CHECKPOINT_TENSOR')

    # Orbax save/restore. No writes whatsoever to the original .msgpack or source.
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt = ocp.PyTreeCheckpointer()
    ckpt.save(str(args.out_dir), item=typed)
    staged = ckpt.restore(str(args.out_dir), item=typed)
    n_stage = compare_exact(typed, staged, jax, np, 'ORBAX_STAGE')
    staged_sha, staged_file_count = orbax_directory_digest(args.out_dir)
    print('M462_ORBAX_STAGE_EXACT_PASS', n_stage, flush=True)

    # Strong gate: exercise Brax's actual restore_checkpoint_path at zero steps.
    # Tiny one-env initialization to limit GPU RAM; no PPO training/rollout.
    restore_cfg = copy.deepcopy(paper)
    restore_cfg.num_envs = 1
    restore_cfg.batch_size = 1
    restore_cfg.num_minibatches = 1
    restore_cfg.num_eval_envs = 1
    restore_cfg.num_evals = 1
    restore_cfg.num_timesteps = 0
    restore_cfg.num_updates_per_batch = 1
    restore_cfg.episode_length = 1000
    restore_cfg.unroll_length = 32
    cfg = restore_cfg.to_dict()
    cfg.pop('network_factory')
    cfg.update(seed=1, randomization_fn=registry.get_domain_randomizer(ENV))
    if cfg['randomization_fn'] is None:
        raise RuntimeError('M462_SOURCE_DOMAIN_RANDOMIZER_MISSING')
    _, restored_parts, _ = ppo.train(
        environment=train_env,
        eval_env=locomotion.load(ENV, config=locomotion.get_default_config(ENV)),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=functools.partial(networks.make_ppo_networks, **net_cfg),
        restore_checkpoint_path=str(args.out_dir.resolve()),
        **cfg)
    if len(restored_parts) != 3:
        raise RuntimeError('M462_BRAX_ZERO_STEP_RETURN_SCHEMA_CHANGED')
    sizes_restored = {name:compare_exact(e,o,jax,np,name.upper()) for name,e,o in zip(
        ('normalizer','actor','critic'),source_parts,restored_parts)}

    # The actor must also produce the exact same deterministic action before training.
    obs = {k:jnp.zeros(shape,dtype=jnp.float32) for k,shape in obs_shapes.items()}
    policy = networks.make_inference_fn(nets)
    a,_ = policy((typed_norm,typed_actor),deterministic=True)(obs,jax.random.PRNGKey(0))
    b,_ = policy((restored_parts[0],restored_parts[1]),deterministic=True)(obs,jax.random.PRNGKey(0))
    if not np.array_equal(np.asarray(jax.device_get(a)),np.asarray(jax.device_get(b))):
        raise RuntimeError('M462_RESTORED_DETERMINISTIC_ACTION_MISMATCH')

    result = {
        'status':'M462_TYPED_ORBAX_RESTORE_PASS_NO_TRAINING',
        'checkpoint_sha256':CHECKPOINT_SHA, 'checkpoint_step':BASELINE_STEP,
        'source_commit':PIN,'source_clean':True,
        'brax_ppo_train_sha256':actual_train_sha, 'jax_backend':jax.default_backend(),
        'orbax_dir':str(args.out_dir.resolve()),
        'orbax_dir_sha256':staged_sha,'orbax_file_count':staged_file_count,
        'source_checkpoint_layout':'Flax callback (normalizer, dict(policy,value))',
        'orbax_layout':'(RunningStatisticsState, PPONetworkParams(policy,value))',
        'tensor_leaf_counts':sizes_restored,'orbax_staged_leaf_count':n_stage,
        'normalizer_shape_by_key':{k:list(v) for k,v in obs_shapes.items()},
        'action_size':int(train_env.action_size),
        'all_restored_tensor_values_bitwise_equal':True,
        'deterministic_action_equal':True,
        'weights_only_not_optimizer_resume':True,
        'ppo_train_zero_steps':True, 'training_performed':False,'rollout_performed':False,
        'notes':['Original 401M .msgpack not modified.','Brax optimizer is reinitialized on warm-start.',
                 'This gate must pass before the A/B one-rollout GPU smoke.']
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print('M462_TYPED_RESTORE_PASS_NO_TRAINING',flush=True)
    print('M462_RESTORE_REPORT',args.report,flush=True)
    print('M462_ORBAX_DIRECTORY',args.out_dir,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('plan','stage-verify'),required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    if args.mode=='plan':
        print('M462_STAGE_PLAN no training, 1-env zero-step typed-restore validation')
        print('ckpt',args.checkpoint,'orbax_out',args.out_dir,'report',args.report)
        return
    if not args.checkpoint.is_file():
        ap.error('--checkpoint must be an existing file')
    do_stage(args)

if __name__=='__main__':
    main()
