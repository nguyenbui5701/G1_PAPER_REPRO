#!/usr/bin/env python3
"""Create a VERIFIED read-only-source copy of Brax PPO 0.12.1 with rollout exposure metrics.

Does NOT change site-packages or pinned MuJoCo Playground. Call once per run.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

PINNED_SHA = 'e20613d42bf0868a13003e0312998816cc8d3593bb5ccaf9f1d070fd09c44bd7'
EXT_FIELDS = (
    'm463_event', 'm463_motion', 'm463_direction', 'm463_phase_target',
    'm463_phase_actual', 'm463_phase_matched', 'm463_fallback',
    'm463_repeat', 'm463_support_side', 'm463_repeat_gap',
    'm463_magnitude', 'm463_term_recent', 'm463_no_push', 'm463_recent',
    'm463_repeat_side_target',
    'm463_repeat_side_matched',
    'm463_repeat_side_fallback',
    'm470_m1_weighted',
    'm470_m2_weighted',
)


H3_EXT_FIELDS = (
    'm470_h3_n', 'm470_h3_prior', 'm470_h3_boundary',
    'm470_h3_stand', 'm470_h3_done', 'm470_h3_base',
    'm470_h3_weighted', 'm470_h3_reward_sum',
)

H3_METRICS_SOURCE = '''    # H3-M1 opt-in, exact sampled PPO transitions.
    hn = extras['m470_h3_n']
    hp = extras['m470_h3_prior']
    hb = extras['m470_h3_boundary']
    hs = extras['m470_h3_stand']
    hd = extras['m470_h3_done']
    hbase = extras['m470_h3_base']
    hw = extras['m470_h3_weighted']
    htotal = extras['m470_h3_reward_sum']
    he = extras['m463_event']
    hnp = extras['m463_no_push']
    ht = extras['truncation']
    expected_n = jnp.where(he == 1, 100,
                          jnp.maximum(jnp.where(hb == 1, 0, hp) - 1, 0))
    hg = jnp.where(hd == 1, 1.0, 1.0 - 0.5 * hn.astype(jnp.float32) / 100.0)
    hdelta = hw - hbase
    allowed = (hs == 1) & (hd == 0) & (hn > 0) & (hnp == 0)
    hfinite = (jnp.isfinite(hbase) & jnp.isfinite(hw) &
               jnp.isfinite(htotal) & jnp.isfinite(data.reward))
    bad_binary = ((hb != 0) & (hb != 1)) | ((hs != 0) & (hs != 1)) | ((hd != 0) & (hd != 1)) | ((he != 0) & (he != 1)) | ((hnp != 0) & (hnp != 1))
    bad_timing = (bad_binary | (hp < 0) | (hp > 100) | (hn < 0) |
                  (hn > 100) | (hn != expected_n) | (hn != recent) |
                  ((hnp == 1) & ((he != 0) | (hn != 0))))
    bad_formula = ((hbase > 0) | (hw > 0) | (hdelta < 0) |
                   ((hs == 0) & (hbase != 0)) |
                   (jnp.abs(hw - hg * hbase) > 2e-6 * (1.0 + jnp.abs(hbase))))
    bad_forbidden = (~allowed) & (hdelta != 0)
    bad_truncation = (ht != 0) & (hdelta != 0)
    bad_total = jnp.abs(htotal - data.reward) > 1e-5
    hcount = lambda mask: jnp.sum(mask.astype(jnp.float32))
    coverage.update({
        'm463_h3_active_per_rollout': hcount(allowed & (hdelta > 0)),
        'm463_h3_post_active_per_rollout': hcount(allowed & (hn < 100) & (hdelta > 0)),
        'm463_h3_stand_events_per_rollout': hcount((he == 1) & (hs == 1)),
        'm463_h3_stand_post_steps_per_rollout': hcount((hs == 1) & (hn > 0) & (hn < 100)),
        'm463_h3_nominal_steps_per_rollout': hcount(hnp == 1),
        'm463_h3_nonstand_steps_per_rollout': hcount(hs == 0),
        'm463_h3_terminal_steps_per_rollout': hcount(hd == 1),
        'm463_h3_timing_bad_per_rollout': hcount(bad_timing),
        'm463_h3_formula_bad_per_rollout': hcount(bad_formula),
        'm463_h3_forbidden_bad_per_rollout': hcount(bad_forbidden),
        'm463_h3_truncation_bad_per_rollout': hcount(bad_truncation),
        'm463_h3_reward_sum_bad_per_rollout': hcount(bad_total),
        'm463_h3_nonfinite_per_rollout': hcount(~hfinite),
        'm463_h3_base_sum_pre_dt': jnp.sum(hbase),
        'm463_h3_modified_sum_pre_dt': jnp.sum(hw),
        'm463_h3_delta_sum_dt': jnp.sum(hdelta * 0.02),
    })
'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()



def _add_m470_fullstate_hooks(s):
    """Opt-in additions to the exact pinned Brax source, never site-packages."""
    def one(old, new):
        nonlocal s
        if s.count(old) != 1:
            raise RuntimeError('M470_FULLSTATE_ANCHOR_NOT_UNIQUE ' + old[:80])
        s = s.replace(old, new, 1)

    one(
        '    augment_pixels: bool = False,\n):',
        '    augment_pixels: bool = False,\n'
        '    m470_resume_state_path: Optional[str] = None,\n'
        '    m470_resume_expected_steps: Optional[int] = None,\n'
        '    m470_state_checkpoint_fn: Callable[..., None] = lambda *args: None,\n'
        '):',
    )
    one(
        '  if num_timesteps == 0:\n',
        '''  # M4.70: resume only at host reset/evaluation boundaries.
  m470_resumed = m470_resume_state_path is not None
  if m470_resumed:
    if restore_checkpoint_path is not None:
      raise RuntimeError('M470_RESUME_AND_WEIGHT_RESTORE_CONFLICT')
    from pathlib import Path as _m470_Path
    from flax import serialization as _m470_serialization
    m470_file = _m470_Path(m470_resume_state_path)
    if not m470_file.is_file() or m470_file.is_symlink():
      raise RuntimeError('M470_FULLSTATE_FILE_INVALID')
    m470_template = {
        'training_state': training_state,
        'local_key': local_key,
        'key_envs': key_envs,
        'eval_key': eval_key,
    }
    m470_payload = _m470_serialization.from_bytes(
        m470_template, m470_file.read_bytes())
    restored = m470_payload['training_state']
    if (jax.tree_util.tree_structure(restored) !=
        jax.tree_util.tree_structure(training_state)):
      raise RuntimeError('M470_FULLSTATE_TREE_CHANGED')
    if int(jax.device_get(restored.env_steps)) <= 0:
      raise RuntimeError('M470_INVALID_RESTORED_STEPS')
    if (m470_resume_expected_steps is None or
        int(jax.device_get(restored.env_steps)) !=
        int(m470_resume_expected_steps)):
      raise RuntimeError('M470_RESUME_EXPECTED_STEPS_MISMATCH')
    if int(jax.device_get(restored.env_steps)) % env_step_per_training_step:
      raise RuntimeError('M470_RESTORED_STEPS_UNALIGNED')
    if m470_payload['key_envs'].shape != key_envs.shape:
      raise RuntimeError('M470_RESTORED_ENV_RNG_SHAPE_MISMATCH')
    training_state = restored
    local_key = m470_payload['local_key']
    key_envs = m470_payload['key_envs']
    m470_eval_rng = m470_payload['eval_key']
    # Last checkpoint was taken AFTER reset_fn(key_envs) at the epoch edge.
    # Replay that exact deterministic reset, not a fresh random reset.
    env_state = reset_fn(key_envs)

  if num_timesteps == 0:
''',
    )
    one(
        '  # Run initial eval\n',
        '''  if m470_resumed:
    # Retain original eval-env randomization from the training seed;
    # restore only the evaluator's advancing PRNG state.
    evaluator._key = m470_eval_rng

  # Run initial eval
''',
    )
    one(
        '  if process_id == 0 and num_evals > 1:\n',
        '  if process_id == 0 and num_evals > 1 and not m470_resumed:\n',
    )
    one(
        '  current_step = 0\n',
        '''  current_step = (
      int(_unpmap(training_state.env_steps)) if m470_resumed else 0
  )
''',
    )
    one(
        '      policy_params_fn(current_step, make_policy, params)\n',
        '''      policy_params_fn(current_step, make_policy, params)
      # Evaluator RNG and environment RNG have now advanced to the
      # exact next-epoch boundary. Saved optimizer is NOT reinitialized.
      m470_state_checkpoint_fn(current_step, {
          'training_state': _unpmap(training_state),
          'local_key': local_key,
          'key_envs': key_envs,
          'eval_key': evaluator._key,
      })
''',
    )
    return s

H4_EXT_FIELDS = ('h4_class', 'h4_boundary', 'h4_age', 'h4_target', 'h4_motion', 'h4_actor_motion', 'h4_refresh', 'h4_changed', 'h4_assigned', 'h4_first_pending', 'h4_repeat_pending', 'h4_first_due', 'h4_repeat_due', 'h4_first_closed', 'h4_repeat_closed', 'h4_event_qual', 'h4_mature_repeat', 'h4_repeat_assigned', 'h4_event_terminal', 'h4_tail_kind', 'h4_tail_cell', 'h4_tail_age', 'h4_tail_same', 'h4_tick_kind', 'h4_tick_cell', 'h4_tick_age', 'h4_tick_same', 'h4_tick_survived', 'h4_close_repeat', 'h4_close_terminal', 'h4_close_complete', 'h4_close_boundary', 'h4_bad', 'h4_assign_motion', 'h4_repeat_origin_motion', 'h4_class_event', 'h4_source_terminal', 'h4_command_mismatch','h4_actual_dx','h4_actual_dy',)

H4_METRICS_SOURCE = '''    # H4: only PPO-consumed scalar extras, not configured event counts.
    h4_count = lambda mask: jnp.sum(mask.astype(jnp.float32))
    hc = extras['h4_class']
    he = extras['m463_event'] > 0
    hr = extras['m463_repeat'] > 0
    ht = extras['h4_tick_age']
    hk = extras['h4_tick_kind']
    hs = extras['h4_tick_survived'] > 0
    # H4 collection cap 640 < wrapper episode_length 1000; any truncation
    # signals a contract violation. The technical gate must verify this.
    ha = extras['h4_age']
    hm = jnp.sqrt(extras['h4_actual_dx']**2 + extras['h4_actual_dy']**2)
    coverage['m463_h4_bad_per_rollout'] = h4_count(
        (extras['h4_bad'] != 0) | (extras['truncation'] != 0) |
        (~jnp.isfinite(hm)) | (jnp.abs(hm - extras['m463_magnitude']) > 3e-6) |
        (ha < 0) | (ha >= 640) | (hc < 0) | (hc > 3))
    coverage['m463_h4_magnitude_sum'] = jnp.sum(hm)
    for kind in (1,2):
        mask = he & (hr if kind == 2 else ~hr)
        for lo,hi in ((0,32),(33,239),(240,308),(309,348),(349,639)):
            coverage[f'm463_h4_age_{kind}_{lo}_{hi}_per_rollout'] = h4_count(
                mask & (ha >= lo) & (ha <= hi))
        for side in range(4):
            coverage[f'm463_h4_support_{kind}_{side}_per_rollout'] = h4_count(
                mask & (extras['m463_support_side'] == side))
    for lo,hi in ((0,29),(30,40),(41,639)):
        coverage[f'm463_h4_gap_{lo}_{hi}_per_rollout'] = h4_count(
            hr & (extras['m463_repeat_gap'] >= lo) & (extras['m463_repeat_gap'] <= hi))
    coverage['m463_h4_repeat_matched_per_rollout'] = h4_count(
        extras['m463_repeat_side_matched'] > 0)
    coverage['m463_h4_repeat_fallback_per_rollout'] = h4_count(
        extras['m463_repeat_side_fallback'] > 0)
    for name in ('source_terminal','command_mismatch'):
        coverage[f'm463_h4_{name}_per_rollout'] = h4_count(extras['h4_'+name]>0)
    for ci in range(4):
        cm = hc == ci
        for name in ('assigned','repeat_assigned','first_due','repeat_due',
                     'first_closed','repeat_closed'):
            coverage[f'm463_h4_class_{ci}_{name}_per_rollout'] = h4_count(
                cm & (extras['h4_'+name]>0))
        coverage[f'm463_h4_class_{ci}_first_events_per_rollout'] = h4_count(cm & he & (~hr))
        coverage[f'm463_h4_class_{ci}_repeat_events_per_rollout'] = h4_count(cm & hr)
        coverage[f'm463_h4_class_{ci}_steps_per_rollout'] = h4_count(hc == ci)
        coverage[f'm463_h4_class_{ci}_assign_per_rollout'] = h4_count(
            (hc == ci) & (extras['h4_boundary'] > 0))
    for name in ('assigned','repeat_assigned','first_closed','repeat_closed',
                 'first_due','repeat_due','refresh','changed','event_terminal',
                 'close_repeat','close_terminal','close_complete','close_boundary'):
        coverage[f'm463_h4_{name}_per_rollout'] = h4_count(extras['h4_'+name] > 0)
    coverage['m463_h4_first_events_per_rollout'] = h4_count(he & (~hr))
    coverage['m463_h4_repeat_events_per_rollout'] = h4_count(he & hr)
    coverage['m463_h4_broad_events_per_rollout'] = h4_count(he & (hc == 1))
    for j in range(6):
        coverage[f'm463_h4_broad_motion_{j}_per_rollout'] = h4_count(
            he & (hc == 1) & (extras['h4_motion'] == j))
        coverage[f'm463_h4_broad_direction_{j}_per_rollout'] = h4_count(
            he & (hc == 1) & (extras['m463_direction'] == j))
    # Joint target ledger: origin command at first actual push, not
    # the possibly refreshed post-step motion tag.
    for motion in (0,3,4):
        for side in (1,2):
            cell = (hc == 3) & (extras['h4_repeat_origin_motion'] == motion) & (
                extras['m463_direction'] == side+1)
            for name in ('repeat_assigned','repeat_due','repeat_closed'):
                coverage[f'm463_h4_target_{motion}_{side}_{name}_per_rollout'] = h4_count(
                    cell & (extras['h4_'+name]>0))
            coverage[f'm463_h4_target_{motion}_{side}_repeat_events_per_rollout'] = h4_count(cell & hr)
            coverage[f'm463_h4_target_{motion}_{side}_matched_per_rollout'] = h4_count(
                cell & (extras['m463_repeat_side_matched']>0))
            coverage[f'm463_h4_target_{motion}_{side}_fallback_per_rollout'] = h4_count(
                cell & (extras['m463_repeat_side_fallback']>0))
    for side in (1,2):
        coverage[f'm463_h4_stand_{side}_events_per_rollout'] = h4_count(extras['h4_event_qual'] == side)
        for motion in (3,4):
            coverage[f'm463_h4_turn_{motion}_{side}_events_per_rollout'] = h4_count(
                (extras['h4_mature_repeat'] > 0) & (extras['h4_motion'] == motion) &
                (extras['m463_direction'] == side+1))
    for kind in (1,2):
        for w in (10,25,50,100,250):
            mask = (ht == w) & (hk == kind)
            for label,flag in (
                ('executed',mask), ('survived',mask & hs),
                ('same_command',mask & (extras['h4_tick_same'] > 0)),
                ('same_unroll',mask & (jnp.arange(unroll_length)[None,:] >= w))):
                coverage[f'm463_h4_tail_{kind}_{w}_{label}_per_rollout'] = h4_count(flag)
            for side in (1,2):
                cell = mask & (extras['h4_tick_cell'] == side)
                coverage[f'm463_h4_stand_{side}_tail_{kind}_{w}_per_rollout'] = h4_count(cell)
                coverage[f'm463_h4_stand_{side}_survived_{kind}_{w}_per_rollout'] = h4_count(cell & hs)
'''


# CCRC opt-in metrics: same generated bytes for both arms, never loss inputs.
H4CC_EXT_FIELDS = tuple('h4cc_'+k for k in (
    'start','event_terminal','tick','tick_age','tick_side','tick_same',
    'tick_survived','tail_terminal','completed','refresh','sample_same',
    'sample_different','deferred','published','terminal_restored','bad',
    'episode','event_age','event_motion','event_side'))
H4CC_METRICS_SOURCE = '''    for name in ('start','event_terminal','tail_terminal','completed','refresh',
                 'sample_same','sample_different','deferred','published','terminal_restored','bad'):
        coverage[f'm463_h4cc_{name}_per_rollout'] = h4_count(extras['h4cc_'+name]>0)
    for side in (1,2):
        coverage[f'm463_h4cc_qualified_{side}_per_rollout'] = h4_count(
            (extras['h4cc_start']>0) & (extras['h4cc_event_side']==side))
        for w in (10,25,50,100,250):
            cell = ((extras['h4cc_tick']>0) & (extras['h4cc_tick_side']==side) &
                    (extras['h4cc_tick_age']==w))
            coverage[f'm463_h4cc_intersection_{side}_{w}_per_rollout'] = h4_count(
                cell & (extras['h4cc_tick_same']>0) & (extras['h4cc_tick_survived']>0))
'''


def _add_h4_chunk_hooks(s):
    """H4 only: preserve the actual scan carry across four host chunks."""
    def one(old,new):
        nonlocal s
        if s.count(old) != 1:
            raise RuntimeError('M470_H4_ANCHOR_NOT_UNIQUE '+old[:80])
        s=s.replace(old,new,1)
    one('    m470_resume_state_path: Optional[str] = None,\n',
        '    h4_chunk_fn: Optional[Callable[..., None]] = None,\n'
        '    m470_resume_state_path: Optional[str] = None,\n')
    one('  # Initialize model params and training state.\n',
        '''  if (num_training_steps_per_epoch != 20 or num_evals != 3
      or num_resets_per_eval != 1 or num_timesteps != 10485760
      or num_envs != 8192 or unroll_length != 32 or action_repeat != 1
      or h4_chunk_fn is None or m470_resume_state_path is not None):
    raise RuntimeError('M470_H4_CHUNK_CONTRACT')

  # Initialize model params and training state.
''')
    # Source revision is explicitly offline-only. Even an imported
    # generated module must fail before the first runtime operation.
    one('  if madrona_backend:\n',
        "  raise RuntimeError('M470_H4_EXECUTION_HOLD')\n"
        '  if madrona_backend:\n')
    start=s.index('  def training_epoch(')
    end=s.index('  # Initialize model params and training state.',start)
    # Keep the contract above outside the replacement.
    cut=s.index('  if (num_training_steps_per_epoch != 20',start)
    if not start < cut < end:
        raise RuntimeError('M470_H4_EPOCH_ANCHOR')
    s=s[:start]+'''  def h4_chunk(training_state, state, key):
    carry, metrics = jax.lax.scan(
        training_step, (training_state, state, key), (), length=5)
    return carry, jax.tree_util.tree_map(jnp.mean, metrics)

  h4_chunk = jax.pmap(h4_chunk, axis_name=_PMAP_AXIS_NAME)

  def training_epoch_with_timing(training_state, env_state, key):
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    carry = (training_state, env_state, key)
    chunks = []
    for idx in range(4):
      carry, metrics = h4_chunk(*carry)
      metrics = jax.tree_util.tree_map(jnp.mean, metrics)
      jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
      h4_chunk_fn(idx+1, _unpmap(carry[0]), carry[1], metrics)
      chunks.append(metrics)
    training_state, env_state, _ = _strip_weak_type(carry)
    metrics = jax.tree_util.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *chunks)
    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (num_training_steps_per_epoch * env_step_per_training_step
           * max(num_resets_per_eval, 1)) / epoch_training_time
    return training_state, env_state, {
        'training/sps': sps, 'training/walltime': training_walltime,
        **{f'training/{name}': value for name,value in metrics.items()},
    }

'''+s[cut:]
    return s


def make_patched_source(raw, *, require_pin=True, enable_resume=False, enable_h3=False, enable_h4=False, enable_h4cc=False):
    if isinstance(raw, bytes):
        actual_sha=sha256(raw)
        s=raw.decode('utf8')
    else:
        s=raw
        actual_sha=sha256(s.encode('utf8'))
    if require_pin and actual_sha != PINNED_SHA:
        raise RuntimeError('M463_BRAX_SOURCE_SHA_UNEXPECTED '+actual_sha)
    old="extra_fields=('truncation',),"
    if s.count(old) != 1:
        raise RuntimeError('M463_BRAX_UNROLL_EXTRA_FIELDS_PATCH_ANCHOR_NOT_UNIQUE')
    if enable_h4 and (enable_h3 or not enable_resume):
        raise RuntimeError('M470_H4_GENERATOR_CONTRACT')
    if enable_h4cc and not enable_h4:
        raise RuntimeError('M470_H4CC_GENERATOR_CONTRACT')
    fields = EXT_FIELDS + (H3_EXT_FIELDS if enable_h3 else ()) + (H4_EXT_FIELDS if enable_h4 else ()) + (H4CC_EXT_FIELDS if enable_h4cc else ())
    new="extra_fields=('truncation', " + ', '.join(repr(k) for k in fields) + '),'
    s=s.replace(old,new,1)
    # Only one target: training_step, immediately before new TrainingState.
    anchor='    new_training_state = TrainingState(\n'
    if s.count(anchor)!=1:
        raise RuntimeError('M463_BRAX_TRAINING_METRICS_ANCHOR_NOT_UNIQUE')
    instrument='''    # M4.63: analyze the EXACT transitions just collected for SGD.
    # No host callback or JAX debugger; this is reduction inside the PPO scan.
    extras = data.extras['state_extras']
    event = extras['m463_event'].astype(jnp.float32)
    event_count = jnp.sum(event)
    coverage = {
        'm463_events_per_rollout': event_count,
        'm463_events_per_1000_steps': event_count * (1000.0 / env_step_per_training_step),
        'm463_episodes_no_push_fraction_per_step': jnp.mean(extras['m463_no_push'].astype(jnp.float32)),
        'm463_phase_matched_per_rollout': jnp.sum(extras['m463_phase_matched'].astype(jnp.float32)),
        'm463_phase_fallback_per_rollout': jnp.sum(extras['m463_fallback'].astype(jnp.float32)),
        'm463_repeat_per_rollout': jnp.sum(extras['m463_repeat'].astype(jnp.float32)),
        'm463_term_recent_per_rollout': jnp.sum(extras['m463_term_recent'].astype(jnp.float32)),
        'm463_push_magnitude_sum': jnp.sum(extras['m463_magnitude'].astype(jnp.float32)),
    }

    # M4.70-F: observed joint repeat exposure in PPO transitions.
    repeat_event = extras['m463_repeat'].astype(jnp.float32)

    right_support = extras['m463_support_side'] == 2

    gap_near_35 = (
        (extras['m463_repeat_gap'] >= 30)
        & (extras['m463_repeat_gap'] <= 40)
    )

    stand_right = (
        (extras['m463_motion'] == 0)
        & (extras['m463_direction'] == 3)
    )

    coverage.update({
        'm463_repeat_right_support_per_rollout':
            jnp.sum(repeat_event * right_support),

        'm463_repeat_gap_30_40_per_rollout':
            jnp.sum(repeat_event * gap_near_35),

        'm463_repeat_right_support_gap_30_40_per_rollout':
            jnp.sum(repeat_event * right_support * gap_near_35),

        'm463_repeat_stand_right_support_gap_30_40_per_rollout':
            jnp.sum(
                repeat_event
                * right_support
                * gap_near_35
                * stand_right
            ),
        'm463_repeat_side_targeted_per_rollout': jnp.sum(
            (extras['m463_repeat_side_target'] > 0).astype(jnp.float32)
        ),
        
        'm463_repeat_side_matched_per_rollout': jnp.sum(
            extras['m463_repeat_side_matched'].astype(jnp.float32)
        ),
        
        'm463_repeat_side_fallback_per_rollout': jnp.sum(
            extras['m463_repeat_side_fallback'].astype(jnp.float32)
        ),
        
        'm463_repeat_side_target_left_per_rollout': jnp.sum(
            (extras['m463_repeat_side_target'] == 1).astype(jnp.float32)
        ),
        
        'm463_repeat_side_target_right_per_rollout': jnp.sum(
            (extras['m463_repeat_side_target'] == 2).astype(jnp.float32)
        ),
    })
    
    for idx in range(6):
        coverage[f'm463_direction_{idx}_per_rollout'] = jnp.sum(
            event * (extras['m463_direction'] == idx))
        coverage[f'm463_motion_{idx}_per_rollout'] = jnp.sum(
            event * (extras['m463_motion'] == idx))
    for idx in range(3):
        coverage[f'm463_phase_target_{idx}_per_rollout'] = jnp.sum(
            event * (extras['m463_phase_target'] == idx))
        coverage[f'm463_phase_actual_{idx}_per_rollout'] = jnp.sum(
            event * (extras['m463_phase_actual'] == idx))
    # Metrics are averaged by Brax across epochs; runner multiplies by
    # number of rollout batches for exact (fp32-integer) exposure totals.
    # M4.64: nstate.info['m463_recent'] is 100 on push transition,
    # 99 on FIRST *later policy action*, then counts down to 1.
    # Source PPO generated THESE actual transitions before any SGD update.
    recent = extras['m463_recent'].astype(jnp.int32)
    m464_tail = {
        'm463_postpush_action_steps_per_rollout': jnp.sum(
            ((recent >= 1) & (recent <= 99)).astype(jnp.float32)),
        'm463_postpush_steps_1_9_per_rollout': jnp.sum(
            ((recent >= 91) & (recent <= 99)).astype(jnp.float32)),
        'm463_postpush_steps_10_24_per_rollout': jnp.sum(
            ((recent >= 76) & (recent <= 90)).astype(jnp.float32)),
        'm463_postpush_steps_25_49_per_rollout': jnp.sum(
            ((recent >= 51) & (recent <= 75)).astype(jnp.float32)),
        'm463_postpush_steps_50_99_per_rollout': jnp.sum(
            ((recent >= 1) & (recent <= 50)).astype(jnp.float32)),
        'm463_events_with_1_post_action_per_rollout': jnp.sum(
            (recent == 99).astype(jnp.float32)),
        'm463_events_with_10_post_actions_per_rollout': jnp.sum(
            (recent == 90).astype(jnp.float32)),
        'm463_events_with_25_post_actions_per_rollout': jnp.sum(
            (recent == 75).astype(jnp.float32)),
        'm463_events_with_50_post_actions_per_rollout': jnp.sum(
            (recent == 50).astype(jnp.float32)),
        'm463_events_with_99_post_actions_per_rollout': jnp.sum(
            (recent == 1).astype(jnp.float32)),
    }
    # H1-M1: the transitions consumed by PPO, not the two-env eval reward.
    m1w = extras['m470_m1_weighted'].astype(jnp.float32)
    coverage['m463_m1_active_per_rollout'] = jnp.sum(
        (m1w < -1e-8).astype(jnp.float32))
    coverage['m463_m1_weighted_sum_per_rollout'] = jnp.sum(m1w)
    coverage['m463_m1_nonfinite_per_rollout'] = jnp.sum(
        (~jnp.isfinite(m1w)).astype(jnp.float32))
    # H1-M2: one-factor signed crossing risk, same live PPO transitions.
    m2w = extras['m470_m2_weighted'].astype(jnp.float32)
    coverage['m463_m2_active_per_rollout'] = jnp.sum(
        (m2w < -1e-8).astype(jnp.float32))
    coverage['m463_m2_weighted_sum_per_rollout'] = jnp.sum(m2w)
    coverage['m463_m2_nonfinite_per_rollout'] = jnp.sum(
        (~jnp.isfinite(m2w)).astype(jnp.float32))
    metrics = {**metrics, **coverage, **m464_tail}
'''
    if enable_h3:
        marker = '    metrics = {**metrics, **coverage, **m464_tail}\n'
        if instrument.count(marker) != 1:
            raise RuntimeError('M470_H3_INSTRUMENT_ANCHOR_NOT_UNIQUE')
        instrument = instrument.replace(marker, H3_METRICS_SOURCE + marker, 1)
    if enable_h4:
        marker = '    metrics = {**metrics, **coverage, **m464_tail}\n'
        if instrument.count(marker) != 1:
            raise RuntimeError('M470_H4_METRICS_ANCHOR')
        instrument = instrument.replace(marker, H4_METRICS_SOURCE + (H4CC_METRICS_SOURCE if enable_h4cc else '') + marker, 1)
    s = s.replace(anchor, instrument+anchor, 1)
    if enable_resume:
        s = _add_m470_fullstate_hooks(s)
    if enable_h4:
        s = _add_h4_chunk_hooks(s)
    return s, actual_sha


def build_instrumented_ppo(pinned_ppo_module, output:Path, *, enable_resume=False, enable_h3=False, enable_h4=False, enable_h4x=False, enable_h4cc=False, enable_h4m6=False, enable_h4m8=False):
    if enable_h4m8:
        # Module-loading builder cannot open a held CAM source.
        raise RuntimeError('M470_H4M8_BUILDER_EXECUTION_HOLD')
    import inspect
    p=Path(inspect.getfile(pinned_ppo_module.train))
    if enable_h4cc and not enable_h4x:
        raise RuntimeError('M470_H4CC_BUILDER_CONTRACT')
    if enable_h4m6 and not (enable_h4cc and enable_h4x and enable_h4 and enable_resume and not enable_h3):
        raise RuntimeError('M470_H4M6_BUILDER_CONTRACT')
    if enable_h4x:
        if not enable_resume or not enable_h4 or enable_h3:
            raise RuntimeError('M470_H4X_BUILDER_CONTRACT')
        patched,original_sha=(make_h4m6_source(p.read_bytes()) if enable_h4m6 else
            make_h4_exploratory_source(p.read_bytes(),enable_h4cc=enable_h4cc))
    else:
        patched,original_sha=make_patched_source(p.read_bytes(),
            enable_resume=enable_resume, enable_h3=enable_h3, enable_h4=enable_h4)
    if enable_h4x:
        from m463_push_ppo_train import h4x_storage
        if output.exists() or output.is_symlink():
            raise RuntimeError('M470_H4X_GENERATED_EXISTS')
        h4x_storage(output.parent,len(patched.encode('utf8')))
    if output.exists():
        if output.read_text('utf8')!=patched:
            raise RuntimeError('M463_REFUSE_TO_OVERWRITE_DIFFERENT_PATCH')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf8') as f:f.write(patched)
    spec=importlib.util.spec_from_file_location('m463_instrumented_brax_ppo',str(output))
    if spec is None or spec.loader is None:raise RuntimeError('M463_MODULE_SPEC_FAIL')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.train.__module__!='m463_instrumented_brax_ppo':
        raise RuntimeError('M463_MODULE_NOT_INSTRUMENTED')
    return mod,{'original_brax_sha256':original_sha,
                'instrumented_file':str(output.resolve()),
                'instrumented_sha256':sha256(patched.encode('utf8')),
                'extra_fields':list(EXT_FIELDS + (H3_EXT_FIELDS if enable_h3 else ()) + (H4_EXT_FIELDS if enable_h4 else ()) + (H4CC_EXT_FIELDS if enable_h4cc else ()) + (H4M6_EXT_FIELDS if enable_h4m6 else ()))}


# Readiness-only source. Never change the production H4 generator or hold.
H4R_FINITE_SOURCE = '''    # Read-only diagnostic of actual sampled PPO tensors and updated carry.
    h4r_leaves = jax.tree_util.tree_leaves((
        data, new_training_state, state, metrics))
    h4r_finite = jnp.asarray(True)
    for h4r_leaf in h4r_leaves:
      h4r_finite = h4r_finite & jnp.all(jnp.isfinite(h4r_leaf))
    metrics['h4r_nonfinite_per_rollout'] = (~h4r_finite).astype(jnp.float32)
'''

H4R_CONTRACT = '''  if (num_training_steps_per_epoch != 20 or num_evals != 1
      or num_resets_per_eval != 1 or num_timesteps != 10240
      or num_envs != 16 or batch_size != 2 or num_minibatches != 8
      or num_updates_per_batch != 5 or unroll_length != 32
      or action_repeat != 1 or episode_length != 1000 or seed != 1
      or device_count != 1 or process_count != 1 or local_device_count != 1
      or not wrap_env or wrap_env_fn is None or randomization_fn is None
      or restore_checkpoint_path is None or m470_resume_state_path is not None
      or m470_resume_expected_steps is not None or h4_chunk_fn is not None
      or madrona_backend or augment_pixels or eval_env is not None
      or not normalize_observations or not normalize_advantage
      or learning_rate != 1e-4 or entropy_cost != 0.
      or discounting != .98 or reward_scaling != .1
      or clipping_epsilon != .2 or gae_lambda != .95 or max_grad_norm != 1.
      or h4r_audit_fn is None):
    raise RuntimeError('M470_H4_READINESS_CONFIG')
'''

H4R_EXIT = '''  # No evaluator, normal training loop, checkpoint or parameter return.
  training_state, env_state = _strip_weak_type((training_state, env_state))
  epoch_key, local_key = jax.random.split(local_key)
  initial = (training_state, env_state,
             jax.random.split(epoch_key, local_devices_to_use))

  def h4r_reference(training_state, state, key):
    carry, metrics = jax.lax.scan(
        training_step, (training_state, state, key), (), length=20)
    return carry, jax.tree_util.tree_map(jnp.mean, metrics)

  h4r_reference = jax.pmap(h4r_reference, axis_name=_PMAP_AXIS_NAME)
  h4r_audit_fn('initial', 0, initial, None)
  for label in ('R0', 'R1'):
    h4r_audit_fn(label+'_before', 20, initial, None)
    result, metrics = h4r_reference(*initial)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    h4r_audit_fn(label, 20, result, metrics)
    del result, metrics
  carry = initial
  chunks = []
  for idx in range(4):
    h4r_audit_fn('S_before', idx+1, carry, None)
    carry, metrics = h4_chunk(*carry)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    h4r_audit_fn('S_chunk', idx+1, carry, metrics)
    chunks.append(metrics)
  metrics = jax.tree_util.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *chunks)
  h4r_audit_fn('S', 20, carry, metrics)
  return None
'''


def make_h4_readiness_source(raw):
    """Generate an isolated, hard-held diagnostic from exact production H4.

    The only callable entry is readiness_train; no path to evaluation or
    production training survives. This source revision cannot execute PPO.
    """
    s, original_sha = make_patched_source(raw, enable_resume=True, enable_h4=True)
    def one(old,new):
        nonlocal s
        if s.count(old)!=1:
            raise RuntimeError('M470_H4R_ANCHOR '+old[:80])
        s=s.replace(old,new,1)
    one('def train(\n','def readiness_train(\n')
    one('    h4_chunk_fn: Optional[Callable[..., None]] = None,\n',
        '    h4r_audit_fn: Optional[Callable[..., None]] = None,\n'
        '    h4_chunk_fn: Optional[Callable[..., None]] = None,\n')
    one("  raise RuntimeError('M470_H4_EXECUTION_HOLD')\n",
        "  raise RuntimeError('M470_H4_READINESS_EXECUTION_HOLD')\n")
    # Validate the frozen diagnostic parameters before reset/initialization.
    start=s.index('  if (num_training_steps_per_epoch != 20')
    end=s.index('  # Initialize model params and training state.',start)
    s=s[:start]+s[end:]
    one('  key = jax.random.PRNGKey(seed)\n',
        H4R_CONTRACT+'\n  key = jax.random.PRNGKey(seed)\n')
    one('    return (new_training_state, state, new_key), metrics\n',
        H4R_FINITE_SOURCE+'    return (new_training_state, state, new_key), metrics\n')
    # No resume or zero-step return in this diagnostic entry.
    start=s.index('  # M4.70: resume only at host reset/evaluation boundaries.')
    end=s.index('  training_state = jax.device_put_replicated(',start)
    s=s[:start]+s[end:]
    # Drop the unused production epoch timing wrapper (same actual h4_chunk).
    start=s.index('  def training_epoch_with_timing(')
    end=s.index('  # Initialize model params and training state.',start)
    s=s[:start]+s[end:]
    marker='  if not eval_env:\n'
    if s.count(marker)!=1:
        raise RuntimeError('M470_H4R_EVAL_ANCHOR')
    s=s[:s.index(marker)]+H4R_EXIT
    return s, original_sha


def build_h4_readiness_ppo(pinned_ppo_module, output):
    import inspect
    raw=Path(inspect.getfile(pinned_ppo_module.train)).read_bytes()
    s, original_sha=make_h4_readiness_source(raw)
    compile(s,str(output),'exec')
    with Path(output).open('x',encoding='utf8') as f:
        f.write(s)
    spec=importlib.util.spec_from_file_location('m463_h4_readiness_ppo',str(output))
    if spec is None or spec.loader is None:
        raise RuntimeError('M470_H4R_MODULE_SPEC')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod,'train') or mod.readiness_train.__module__!='m463_h4_readiness_ppo':
        raise RuntimeError('M470_H4R_MODULE_ENTRY')
    return mod, {'original_brax_sha256':original_sha,
                 'generated_sha256':sha256(s.encode('utf8')),
                 'generated_file':str(Path(output).resolve())}


# H4-M1-1C: source-only causal variant, all runtime entry points held.
H4C_ACTING_BLOB = 'ec6f81d3e3c48e96a22f7df292fb1a98b63f938d'
H4C_HELD_SHA = '32ee163079334c965bc12ff517f00adbf232c92722fb5f43ac582c4ef3cec538'

H4C_TRACE_SOURCE = '''def h4c_trace_state(state):
  # Snapshot leaves, not mutable info/metrics dictionaries. Post-step data
  # may be cached reset pose: done/source_terminal must accompany the trace.
  fields = ('rng', 'command', 'phase', 'last_contact', 'steps', 'step',
            'truncation', 'm463_age', 'm463_next', 'm463_push_count',
            'm463_event', 'm463_repeat', 'm463_direction',
            'm463_phase_actual', 'm463_support_side', 'm463_magnitude',
            'h4_class', 'h4_first_pending', 'h4_repeat_pending',
            'h4_tail_kind', 'h4_tail_age', 'h4_source_terminal')
  return {'obs': jax.tree_util.tree_map(lambda x: x, state.obs),
          'qpos': state.data.qpos, 'qvel': state.data.qvel,
          'ctrl': state.data.ctrl, 'done': state.done,
          'info': {k: state.info[k] for k in fields}}

'''

H4C_MINIBATCH_TRACE = '''    metrics = {**metrics,
               'h4c_loss_key': key_loss,
               'h4c_next_minibatch_key': key}
    leaves = jax.tree_util.tree_leaves((params, optimizer_state, metrics))
    finite = jnp.asarray(True)
    for leaf in leaves:
      finite = finite & jnp.all(jnp.isfinite(leaf))
    metrics['h4c_finite'] = finite
'''

H4C_EXIT = '''  training_state, env_state = _strip_weak_type((training_state, env_state))
  epoch_key, local_key = jax.random.split(local_key)
  initial = (training_state, env_state,
             jax.random.split(epoch_key, local_devices_to_use))
  # DR model is part of the input even though it is not in the scan carry.
  dr = env.env.env
  if type(dr).__name__ != 'BraxDomainRandomizationVmapWrapper':
    raise RuntimeError('M470_H4C_DR_WRAPPER')
  model = dr._mjx_model_v
  model_axes = dr._in_axes
  collect = jax.pmap(h4c_collect, axis_name=_PMAP_AXIS_NAME)
  optimize = jax.pmap(h4c_optimize, axis_name=_PMAP_AXIS_NAME)
  h4c_audit_fn('initial', initial, (model, model_axes))
  for label in ('A0', 'A1'):
    h4c_audit_fn(label+'_before', initial, (dr._mjx_model_v, dr._in_axes))
    result = collect(initial)
    h4c_audit_fn(label, result, (initial, dr._mjx_model_v, dr._in_axes))
    if label == 'A0':
      # Keep the exact immutable device batch, not a regenerated copy.
      batch, key_sgd = result[1], result[2]
    del result
  fixed = (initial[0], batch, key_sgd)
  for label in ('B0', 'B1'):
    h4c_audit_fn(label+'_before', fixed, None)
    result = optimize(*fixed)
    h4c_audit_fn(label, result, fixed)
    del result
  return None
'''


def h4c_git_blob(raw):
    return hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()


def h4c_once(source, old, new):
    if source.count(old)!=1:
        raise RuntimeError('M470_H4C_ANCHOR '+old[:90])
    return source.replace(old,new,1)


def h4c_regions(held):
    start=held.index('  def training_step(')
    end=held.index('  def h4_chunk(',start)
    step=held[start:end]
    collect_start=step.index('    training_state, state, key = carry\n')
    opt_start=step.index('    # Update normalization params')
    reducer_start=step.index('    # M4.63: analyze the EXACT transitions')
    state_start=step.index('    new_training_state = TrainingState(')
    finite_start=step.index(H4R_FINITE_SOURCE)
    if not collect_start<opt_start<reducer_start<state_start<finite_start:
        raise RuntimeError('M470_H4C_REGION_ORDER')
    return (step[collect_start:opt_start],step[opt_start:reducer_start],
            step[reducer_start:state_start],step[state_start:finite_start])


def make_h4_causal_source(raw, acting_raw):
    """Generate diagnostic from accepted held source, never authorize runtime."""
    import ast
    if h4c_git_blob(acting_raw)!=H4C_ACTING_BLOB:
        raise RuntimeError('M470_H4C_ACTING_PIN')
    held, original=make_h4_readiness_source(raw)
    if sha256(held.encode())!=H4C_HELD_SHA:
        raise RuntimeError('M470_H4C_READINESS_PIN')
    acting_source=acting_raw.decode('utf8')
    fn=next(n for n in ast.parse(acting_source).body
            if isinstance(n,ast.FunctionDef) and n.name=='generate_unroll')
    unroll=ast.get_source_segment(acting_source,fn)
    # Retain original body and jit/scan, add only read-only outputs.
    body=unroll[unroll.index('  @jax.jit\n'):]
    body=h4c_once(body,'    nstate, transition = actor_step(',
                  '    pre = h4c_trace_state(state)\n'
                  '    nstate, transition = acting.actor_step(')
    body=h4c_once(body,'    return (nstate, next_key), transition',
                  "    trace = {'pre': pre, 'post': h4c_trace_state(nstate),\n"
                  "             'actor_key': current_key, 'next_key': next_key}\n"
                  '    return (nstate, next_key), (transition, trace)')
    body=h4c_once(body,'  (final_state, _), data = jax.lax.scan(',
                  '  (final_state, final_key), (data, trace) = jax.lax.scan(')
    body=h4c_once(body,'  return final_state, data',
                  '  return final_state, data, trace, final_key')
    trace_fn=('def h4c_generate_unroll(env, env_state, policy, key, '
              'unroll_length, extra_fields=()):\n'+body+'\n\n')
    collect,opt,reducer,construct=h4c_regions(held)
    collect=h4c_once(collect,'      next_state, data = acting.generate_unroll(',
                     '      next_state, data, trace, inner_key = h4c_generate_unroll(')
    collect=h4c_once(collect,'      return (next_state, next_key), data',
                     '      return (next_state, next_key), (data, trace, inner_key)')
    collect=h4c_once(collect,'    (state, _), data = jax.lax.scan(',
                     '    (state, outer_key), (data, trace, inner_key) = jax.lax.scan(')
    opt=h4c_once(opt,'    (optimizer_state, params, _), metrics = jax.lax.scan(',
                 '    (optimizer_state, params, sgd_final_key), metrics = jax.lax.scan(')
    split=('  def h4c_collect(carry):\n'+collect+
           '    metrics = {}\n'+reducer+
           '    finite = jnp.asarray(True)\n'
           '    for leaf in jax.tree_util.tree_leaves((state,data,trace,metrics)):\n'
           '      finite = finite & jnp.all(jnp.isfinite(leaf))\n'
           "    metrics['h4r_nonfinite_per_rollout'] = (~finite).astype(jnp.float32)\n"
           '    return state, data, key_sgd, new_key, trace, inner_key, outer_key, metrics\n\n'
           '  def h4c_optimize(training_state, data, key_sgd):\n'+opt+construct+
           '    return new_training_state, sgd_final_key, metrics\n\n')
    s=held
    start=s.index('  def training_step(')
    end=s.index('  # Initialize model params and training state.',start)
    s=s[:start]+split+s[end:]
    s=h4c_once(s,'    return (optimizer_state, params, key), metrics\n\n  def sgd_step(',
               H4C_MINIBATCH_TRACE+
               '    return (optimizer_state, params, key), metrics\n\n  def sgd_step(')
    # Return actual permutation and gradient-scan seed at each SGD pass.
    s=h4c_once(s,'    (optimizer_state, params, _), metrics = jax.lax.scan(\n'
                 '        functools.partial(minibatch_step, normalizer_params=normalizer_params),',
               '    (optimizer_state, params, minibatch_final_key), metrics = jax.lax.scan(\n'
               '        functools.partial(minibatch_step, normalizer_params=normalizer_params),')
    s=h4c_once(s,'    return (optimizer_state, params, key), metrics\n\n  def h4c_collect(',
               "    metrics = {**metrics, 'h4c_perm_key': key_perm,\n"
               "               'h4c_grad_key': key_grad, 'h4c_sgd_key': key,\n"
               "               'h4c_minibatch_final_key': minibatch_final_key}\n"
               '    return (optimizer_state, params, key), metrics\n\n  def h4c_collect(')
    s=h4c_once(s,'def readiness_train(\n',H4C_TRACE_SOURCE+trace_fn+'def causal_train(\n')
    s=h4c_once(s,"  raise RuntimeError('M470_H4_READINESS_EXECUTION_HOLD')\n",
               "  raise RuntimeError('M470_H4C_EXECUTION_HOLD')\n")
    contract=H4R_CONTRACT.replace('num_training_steps_per_epoch != 20',
                                 'num_training_steps_per_epoch != 1')
    contract=contract.replace('num_timesteps != 10240','num_timesteps != 512')
    contract=contract.replace('or num_envs != 16','or env_step_per_training_step != 512\n      or num_envs != 16')
    s=h4c_once(s,H4R_CONTRACT,contract)
    s=h4c_once(s,H4R_EXIT,H4C_EXIT)
    s=s.replace('h4r_audit_fn','h4c_audit_fn').replace('M470_H4_READINESS_CONFIG','M470_H4C_CONFIG')
    compile(s,'<held causal source>','exec')
    return s,original


def build_h4_causal_ppo(pinned_ppo_module, output):
    import inspect
    raw=Path(inspect.getfile(pinned_ppo_module.train)).read_bytes()
    from brax.training import acting
    acting_raw=Path(inspect.getfile(acting)).read_bytes()
    s,original=make_h4_causal_source(raw,acting_raw)
    with Path(output).open('x',encoding='utf8') as f:
        f.write(s)
    spec=importlib.util.spec_from_file_location('m463_h4_causal_ppo',str(output))
    if spec is None or spec.loader is None:
        raise RuntimeError('M470_H4C_MODULE_SPEC')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod,'train') or hasattr(mod,'readiness_train'):
        raise RuntimeError('M470_H4C_MODULE_ENTRY')
    return mod,{'original_brax_sha256':original,'acting_blob':H4C_ACTING_BLOB,
                'generated_sha256':sha256(s.encode()),
                'generated_file':str(Path(output).resolve())}


# H4-M1-2R: prospective matched exploratory contract, opt-in only.
# The default H4/H4R/H4C generated variants remain byte identical.
H4X_FINITE_SOURCE = '''    h4x_finite = jnp.asarray(True)
    # Raw SGD/minibatch metrics are still present, before host chunk means.
    for h4x_leaf in jax.tree_util.tree_leaves((data,new_training_state,state,metrics)):
      h4x_finite = h4x_finite & jnp.all(jnp.isfinite(h4x_leaf))
    metrics['h4x_finite'] = h4x_finite.astype(jnp.float32)
'''

H4X_CONFIG_SOURCE = '''  if (num_timesteps != 10485760 or num_evals != 3
      or num_training_steps_per_epoch != 20 or num_resets_per_eval != 1
      or num_envs != 8192 or num_eval_envs != 2 or batch_size != 1024
      or num_minibatches != 8 or num_updates_per_batch != 5
      or unroll_length != 32 or episode_length != 1000 or action_repeat != 1
      or process_count != 1 or local_device_count != 1 or local_devices_to_use != 1
      or seed != 1 or m470_resume_state_path is not None
      or restore_checkpoint_path is None or randomization_fn is None
      or not wrap_env or h4_chunk_fn is None):
    raise RuntimeError('M470_H4X_CONFIG')
'''

H4X_RESET_HELPER = '''  @jax.jit
  def h4x_dynamic_finite(tree):
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
      finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite

'''

# Unique source anchors, reversible for the focused source-equivalence test.
H4X_PATCHES = (
    ('    h4_chunk_fn: Optional[Callable[..., None]] = None,\n',
     '    h4x_audit_fn: Optional[Callable[..., None]] = None,\n'
     '    h4_chunk_fn: Optional[Callable[..., None]] = None,\n'),
    ("  raise RuntimeError('M470_H4_EXECUTION_HOLD')\n",
     "  from m463_push_ppo_train import h4x_generated_guard\n"
     "  h4x_generated_guard(h4x_audit_fn, __file__)\n"),
    ('  key = jax.random.PRNGKey(seed)\n',
     H4X_CONFIG_SOURCE+'  key = jax.random.PRNGKey(seed)\n'),
    ('  env_state = reset_fn(key_envs)\n  # Discard',
     H4X_RESET_HELPER+
     "  dr = env.env.env\n"
     "  if type(dr).__name__ != 'BraxDomainRandomizationVmapWrapper':\n"
     "    raise RuntimeError('M470_H4X_DR_WRAPPER')\n"
     "  h4x_audit_fn('reset_before', keys=key_envs, local_key=local_key,\n"
     "                eval_key=eval_key, model=dr._mjx_model_v, axes=dr._in_axes)\n"
     "  env_state = reset_fn(key_envs)\n"
     "  h4x_audit_fn('reset_after', env_state=env_state, finite=h4x_dynamic_finite(env_state))\n"
     '  # Discard'),
    ('    return (new_training_state, state, new_key), metrics\n',
     H4X_FINITE_SOURCE+'    return (new_training_state, state, new_key), metrics\n'),
    ('      carry, metrics = h4_chunk(*carry)\n',
     "      h4x_audit_fn('chunk_before', index=idx+1, carry=carry)\n"
     '      carry, metrics = h4_chunk(*carry)\n'),
    ('      h4_chunk_fn(idx+1, _unpmap(carry[0]), carry[1], metrics)\n',
     "      h4x_audit_fn('chunk_after', index=idx+1, carry=carry, metrics=metrics)\n"
     '      h4_chunk_fn(idx+1, _unpmap(carry[0]), carry[1], metrics)\n'
     "      h4x_audit_fn('chunk_ledger', index=idx+1)\n"),
    ('  if not eval_env:\n',
     "  h4x_audit_fn('initial', training_state=training_state, env_state=env_state,\n"
     "                finite=h4x_dynamic_finite(training_state))\n"
     '  if not eval_env:\n'),
    ('    metrics = evaluator.run_evaluation(\n',
     "    h4x_audit_fn('eval_before', step=0, key=evaluator._key)\n"
     '    metrics = evaluator.run_evaluation(\n'),
    ('    progress_fn(0, metrics)\n',
     "    h4x_audit_fn('eval_after', step=0, key=evaluator._key, metrics=metrics)\n"
     '    progress_fn(0, metrics)\n'),
    ('      epoch_key, local_key = jax.random.split(local_key)\n'
     '      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)\n',
     '      epoch_key, local_key = jax.random.split(local_key)\n'
     '      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)\n'
     "      h4x_audit_fn('epoch', keys=epoch_keys, local_key=local_key,\n"
     "                    training_state=training_state, env_state=env_state)\n"),
    ('      env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state\n',
     "      h4x_audit_fn('reset_before', keys=key_envs, local_key=local_key,\n"
     "                    eval_key=evaluator._key, model=dr._mjx_model_v, axes=dr._in_axes)\n"
     '      env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state\n'
     "      h4x_audit_fn('reset_after', env_state=env_state, finite=h4x_dynamic_finite(env_state))\n"),
    ('      metrics = evaluator.run_evaluation(\n',
     "      h4x_audit_fn('eval_before', step=current_step, key=evaluator._key)\n"
     '      metrics = evaluator.run_evaluation(\n'),
    ('      progress_fn(current_step, metrics)\n',
     "      h4x_audit_fn('eval_after', step=current_step, key=evaluator._key, metrics=metrics)\n"
     '      progress_fn(current_step, metrics)\n'),
    ('  total_steps = current_step\n',
     "  h4x_audit_fn('finish', training_state=training_state, env_state=env_state)\n"
     '  total_steps = current_step\n'),
)


def make_h4_exploratory_source(raw, *, enable_h4cc=False):
    """Same held H4 base; only read-only instrumentation and dispatch guards."""
    held, original = make_patched_source(raw, enable_resume=True, enable_h4=True, enable_h4cc=enable_h4cc)
    s = held
    # Initial and epoch evaluation anchors differ by indentation; match
    # complete lines, not substrings of longer indentation.
    for old,new in H4X_PATCHES:
        if old.startswith('    metrics = evaluator.run_evaluation('):
            old = '\n'+old
            new = '\n'+new
        s = h4c_once(s,old,new)
    if enable_h4cc:
        # Separate dispatch guard; H4X authorization can never open CCRC.
        s = h4c_once(s,
            '  from m463_push_ppo_train import h4x_generated_guard\n'
            '  h4x_generated_guard(h4x_audit_fn, __file__)\n',
            '  from m463_push_ppo_train import h4cc_generated_guard\n'
            '  h4cc_generated_guard(h4x_audit_fn, __file__, environment)\n')
    compile(s,'<H4 exploratory generated>','exec')
    return s,original


def normalize_h4_exploratory_source(s):
    for old,new in reversed(H4X_PATCHES):
        if old.startswith('    metrics = evaluator.run_evaluation('):
            old = '\n'+old
            new = '\n'+new
        s = h4c_once(s,new,old)
    return s


def h4cc_static_generated_regression(raw):
    """Offline source generation only: no JAX/Brax imports, no file writes."""
    import ast
    legacy,original=make_h4_exploratory_source(raw)
    cc,cc_original=make_h4_exploratory_source(raw,enable_h4cc=True)
    if original!=PINNED_SHA or cc_original!=original:
        raise RuntimeError('M470_H4CC_STATIC_ORIGINAL_PIN')
    legacy_sha=sha256(legacy.encode('utf8'))
    if legacy_sha!='8f30e5c695211b21815375c37b1bda67fcda98b2ee259a703e87b182c3402062':
        raise RuntimeError('M470_H4CC_STATIC_LEGACY_GENERATED_SHA '+legacy_sha)
    fields=lambda items:"extra_fields=('truncation', " + ', '.join(repr(k) for k in items) + '),'
    old=EXT_FIELDS+H4_EXT_FIELDS
    normalized=h4c_once(cc,fields(old+H4CC_EXT_FIELDS),fields(old))
    normalized=h4c_once(normalized,H4CC_METRICS_SOURCE,'')
    normalized=h4c_once(normalized,
        '  from m463_push_ppo_train import h4cc_generated_guard\n'
        '  h4cc_generated_guard(h4x_audit_fn, __file__, environment)\n',
        '  from m463_push_ppo_train import h4x_generated_guard\n'
        '  h4x_generated_guard(h4x_audit_fn, __file__)\n')
    if normalized!=legacy or ast.dump(ast.parse(normalized))!=ast.dump(ast.parse(legacy)):
        raise RuntimeError('M470_H4CC_STATIC_GENERATED_DIFF')
    compile(cc,'<CCRC generated static>','exec')
    compile(legacy,'<H4 legacy generated static>','exec')
    return {'original_sha256':original,'legacy_sha256':legacy_sha,
            'ccrc_sha256':sha256(cc.encode('utf8')),
            'normalized_byte_equal':True,'normalized_ast_equal':True,
            'generated_compile':True}

# H4-M6 only. A separate generated source, shared by REF6/RSS. No legacy edits.
H4M6_EXT_FIELDS = tuple('h4m6_'+k for k in (
    'qualified','event_terminal','start','event_side','eligible','tick_age',
    'tick_side','positive_start','completed','close_terminal','close_truncation',
    'close_boundary','close_command','close_event','bad','opportunity_side',
    'attempt_side','factor','raw_stand','shadow_stand','actual_stand',
    'shadow_relief','delivered_relief','base_reward','expected_reward'))
H4M6_METRICS_SOURCE = '''    # H4-M6: actual PPO data, never a second rollout or loss replacement.
    m6 = lambda key: extras['h4m6_'+key]
    elig = m6('eligible') > 0
    m6k = m6('tick_age')
    m6offset = jnp.arange(unroll_length)[None,:]
    m6bad = ((m6('bad')>0) | (data.reward != m6('expected_reward')) |
             (~jnp.isfinite(data.reward)) |
             (elig & ((m6k<1) | (m6k>50) | (data.discount!=1) |
                      (extras['truncation']!=0))) |
             ((~elig) & ((m6('factor')!=1) | (m6('shadow_relief')!=0) |
                         (m6('delivered_relief')!=0))) |
             (m6('shadow_relief')<0) | (m6('delivered_relief')<0) |
             (m6('delivered_relief')>m6('shadow_relief')) |
             (m6('shadow_stand')>m6('raw_stand')) |
             (m6('actual_stand')>m6('raw_stand')))
    for name in ('factor','raw_stand','shadow_stand','actual_stand','shadow_relief',
                 'delivered_relief','base_reward','expected_reward'):
        m6bad = m6bad | (~jnp.isfinite(m6(name)))
    coverage['m463_h4m6_bad_per_rollout'] = h4_count(m6bad)
    for side in (1,2):
        ev = m6('event_side')==side
        tick = m6('tick_side')==side
        for name in ('qualified','event_terminal','start'):
            coverage[f'm463_h4m6_{side}_{name}_per_rollout'] = h4_count(ev & (m6(name)>0))
        for name in ('eligible','positive_start','completed','close_terminal',
                     'close_truncation','close_boundary','close_command','close_event'):
            coverage[f'm463_h4m6_{side}_{name}_per_rollout'] = h4_count(tick & (m6(name)>0))
        for name in ('opportunity','attempt'):
            coverage[f'm463_h4m6_{side}_{name}_per_rollout'] = h4_count(m6(name+'_side')==side)
        for w in (1,10,25,50):
            cell = tick & elig & (m6k==w)
            coverage[f'm463_h4m6_{side}_reached_{w}_per_rollout'] = h4_count(cell)
            coverage[f'm463_h4m6_{side}_same_unroll_{w}_per_rollout'] = h4_count(cell & (m6offset>=w))
        for off in range(32):
            coverage[f'm463_h4m6_{side}_offset_{off}_per_rollout'] = h4_count(tick & elig & (m6offset==off))
        for name in ('shadow_relief','delivered_relief'):
            coverage[f'm463_h4m6_{side}_{name}_sum'] = jnp.sum(jnp.where(tick,m6(name),0.))
            coverage[f'm463_h4m6_{side}_{name}_same_unroll_sum'] = jnp.sum(jnp.where(tick & elig & (m6offset>=m6k),m6(name),0.))
    for name in ('base_reward','expected_reward','shadow_relief','delivered_relief'):
        coverage[f'm463_h4m6_{name}_sum'] = jnp.sum(m6(name))
    coverage['m463_h4m6_transition_reward_sum'] = jnp.sum(data.reward)
    coverage['m463_h4m6_ppo_scaled_relief_sum'] = jnp.sum(m6('delivered_relief')*reward_scaling)
'''

def make_h4m6_source(raw):
    """Held source generator; do not import/execute generated train."""
    base,original = make_h4_exploratory_source(raw,enable_h4cc=True)
    if original!=PINNED_SHA:
        raise RuntimeError('M470_H4M6_ORIGINAL_PIN')
    old_fields = EXT_FIELDS+H4_EXT_FIELDS+H4CC_EXT_FIELDS
    fmt = lambda fields: "extra_fields=('truncation', " + ', '.join(repr(k) for k in fields) + '),'
    old_guard = ('  from m463_push_ppo_train import h4cc_generated_guard\n'
                 '  h4cc_generated_guard(h4x_audit_fn, __file__, environment)\n')
    new_guard = ('  from m463_push_ppo_train import h4m6_generated_guard\n'
                 '  h4m6_generated_guard(h4x_audit_fn, __file__, environment)\n')
    s = h4c_once(base,fmt(old_fields),fmt(old_fields+H4M6_EXT_FIELDS))
    s = h4c_once(s,H4CC_METRICS_SOURCE,H4CC_METRICS_SOURCE+H4M6_METRICS_SOURCE)
    s = h4c_once(s,old_guard,new_guard)
    reverse = h4c_once(s,new_guard,old_guard)
    reverse = h4c_once(reverse,H4CC_METRICS_SOURCE+H4M6_METRICS_SOURCE,H4CC_METRICS_SOURCE)
    reverse = h4c_once(reverse,fmt(old_fields+H4M6_EXT_FIELDS),fmt(old_fields))
    if reverse!=base:
        raise RuntimeError('M470_H4M6_GENERATED_REVERSE')
    compile(s,'<H4M6 held generated>','exec')
    return s,original


# H4-M8: staged held generator. Not reachable from the production builder.
H4M8_EXT_FIELDS = tuple('h4m8_'+k for k in (
    'qualified','event_terminal','start','event_side','eligible','tick_age',
    'tick_side','weight8','positive_start','completed','close_terminal',
    'close_truncation','close_boundary','close_command','close_event',
    'opportunity_side','attempt_side','bad','mass','h2','cost','shadow'))


def h4m8_learning_reward(xp, native, discount, truncation, extras, treatment):
    """Pure batch arithmetic. No new loss, physics, or policy computation."""
    treatment=xp.asarray(treatment)
    if treatment.shape!=() or treatment.dtype!=xp.bool_:
        raise RuntimeError('M470_H4M8_TREATMENT_SCHEMA')
    if native.ndim!=2 or native.shape[1]!=32 or native.dtype!=xp.float32:
        raise RuntimeError('M470_H4M8_REWARD_SCHEMA')
    if discount.shape!=native.shape or truncation.shape!=native.shape:
        raise RuntimeError('M470_H4M8_MASK_SHAPE')
    for key in H4M8_EXT_FIELDS:
        x=extras[key]
        dtype=xp.float32 if key in ('h4m8_mass','h4m8_h2','h4m8_cost','h4m8_shadow') else xp.int32
        if x.shape!=native.shape or x.dtype!=dtype:
            raise RuntimeError('M470_H4M8_EXTRA_SCHEMA '+key)
    e=extras['h4m8_eligible']; k=extras['h4m8_tick_age']; w=extras['h4m8_weight8']
    c=extras['h4m8_cost']; shadow=extras['h4m8_shadow']
    mass=extras['h4m8_mass']; h2=extras['h4m8_h2']
    expected_w=xp.where(e==1,xp.minimum(8,xp.minimum(xp.maximum(0,k-7),xp.maximum(0,51-k))),0)
    allowed=(e==1)&(discount==1)&(truncation==0)
    desired=xp.where(allowed,shadow,xp.asarray(0.,xp.float32))
    # Selecting original leaf preserves negative zero and unchanged reward bits.
    learning=xp.where(treatment & allowed & (desired>0),native-desired,native)
    actual=xp.where(treatment & allowed & (desired>0),native-learning,xp.asarray(0.,xp.float32))
    bad=((extras['h4m8_bad']!=0)|(~xp.isfinite(native))|(~xp.isfinite(learning))|
         (~xp.isfinite(mass))|(~xp.isfinite(h2))|(~xp.isfinite(c))|(~xp.isfinite(shadow))|
         (mass<=0)|(h2<0)|(c<0)|(c>1)|(shadow<0)|(shadow>xp.asarray(.005,xp.float32))|
         ((e!=0)&(e!=1))|((discount!=0)&(discount!=1))|((truncation!=0)&(truncation!=1))|
         ((e==1)&((k<1)|(k>50)))|((e==0)&(k!=0))|(w!=expected_w)|
         ((w==0)&(shadow!=0))|((w>0)&((discount!=1)|(truncation!=0)))|
         (actual<0))
    href=mass*xp.asarray(.793,xp.float32)*xp.asarray(.35,xp.float32)
    expected_c=h2/(href*href+h2)
    expected_shadow=xp.where(w>0,xp.asarray(.25,xp.float32)*xp.asarray(.02,xp.float32)*
                    (w.astype(xp.float32)/xp.asarray(8.,xp.float32))*c,xp.asarray(0.,xp.float32))
    # Exact arithmetic comparison is a conservative technical guard; a compiled
    # context mismatch is STOP, not permission to adjust a tolerance.
    bad=bad | (expected_c!=c) | (expected_shadow!=shadow)
    for key in ('qualified','event_terminal','start','eligible','positive_start','completed',
                'close_terminal','close_truncation','close_boundary','close_command','close_event','bad'):
        x=extras['h4m8_'+key]
        bad=bad | ((x!=0)&(x!=1))
    for key in ('event_side','tick_side','opportunity_side','attempt_side'):
        x=extras['h4m8_'+key]
        bad=bad | (x<0)|(x>2)
    bad=bad | ((extras['h4m8_start']+extras['h4m8_event_terminal'])!=extras['h4m8_qualified'])
    bad=bad | ((e==1)&(extras['h4m8_tick_side']==0))
    return learning,desired,actual,bad


H4M8_TRANSFORM_SOURCE = '''    # Common collected native data is preserved for coverage and reporting.
    from m463_instrument_brax import h4m8_learning_reward
    h8_reward,h8_desired,h8_actual,h8_bad = h4m8_learning_reward(
        jnp,data.reward,data.discount,data.extras['state_extras'],h4m8_treatment)
    learning_data = data._replace(reward=jax.lax.stop_gradient(h8_reward))
'''

H4M8_METRICS_SOURCE = '''    # Exact collected batch; no extra rollout or optimizer invocation.
    coverage['h4m8_bad_per_rollout'] = jnp.sum(h8_bad.astype(jnp.float32))
    coverage['h4m8_desired_sum'] = jnp.sum(h8_desired)
    coverage['h4m8_actual_sum'] = jnp.sum(h8_actual)
    coverage['h4m8_native_reward_sum'] = jnp.sum(data.reward)
    coverage['h4m8_learning_reward_sum'] = jnp.sum(learning_data.reward)
    coverage['h4m8_round_to_identity_count'] = jnp.sum(
        ((h8_desired>0)&(h8_actual==0)&h4m8_treatment).astype(jnp.float32))
    h8 = data.extras['state_extras']
    for h8side in (1,2):
      event_side=h8['h4m8_event_side']==h8side
      tick_side=h8['h4m8_tick_side']==h8side
      prefix='h4m8_side_'+str(h8side)+'_'
      for name in ('qualified','start','event_terminal'):
        coverage[prefix+name]=jnp.sum((event_side&(h8['h4m8_'+name]==1)).astype(jnp.float32))
      for name in ('eligible','positive_start','completed','close_terminal',
                   'close_truncation','close_boundary','close_command','close_event'):
        coverage[prefix+name]=jnp.sum((tick_side&(h8['h4m8_'+name]==1)).astype(jnp.float32))
      coverage[prefix+'weight8']=jnp.sum(jnp.where(tick_side,h8['h4m8_weight8'],0))
      coverage[prefix+'reach25']=jnp.sum((tick_side&(h8['h4m8_tick_age']==25)).astype(jnp.float32))
      coverage[prefix+'opportunities']=jnp.sum((h8['h4m8_opportunity_side']==h8side).astype(jnp.float32))
      coverage[prefix+'attempts']=jnp.sum((h8['h4m8_attempt_side']==h8side).astype(jnp.float32))
      coverage[prefix+'desired']=jnp.sum(jnp.where(tick_side,h8_desired,0.))
      coverage[prefix+'actual']=jnp.sum(jnp.where(tick_side,h8_actual,0.))
'''


def make_h4m8_source(raw):
    """Held only; separate generator, byte-reversible to accepted CCRC base."""
    base,original=make_h4_exploratory_source(raw,enable_h4cc=True)
    old=EXT_FIELDS+H4_EXT_FIELDS+H4CC_EXT_FIELDS
    fmt=lambda fields: "extra_fields=('truncation', " + ', '.join(repr(k) for k in fields) + '),'
    patches=(
        (fmt(old),fmt(old+H4M8_EXT_FIELDS)),
        ('    h4x_audit_fn: Optional[Callable[..., None]] = None,\n',
         '    h4m8_treatment=None,\n    h4x_audit_fn: Optional[Callable[..., None]] = None,\n'),
        ('  from m463_push_ppo_train import h4cc_generated_guard\n'
         '  h4cc_generated_guard(h4x_audit_fn, __file__, environment)\n',
         "  raise RuntimeError('M470_H4M8_GENERATED_EXECUTION_HOLD')\n"),
        ('    assert data.discount.shape[1:] == (unroll_length,)\n',
         '    assert data.discount.shape[1:] == (unroll_length,)\n'+H4M8_TRANSFORM_SOURCE),
        ('sgd_step, data=data, normalizer_params=normalizer_params',
         'sgd_step, data=learning_data, normalizer_params=normalizer_params'),
        (H4CC_METRICS_SOURCE,H4CC_METRICS_SOURCE+H4M8_METRICS_SOURCE),
    )
    s=base
    for a,b in patches:
        s=h4c_once(s,a,b)
    reverse=s
    for a,b in reversed(patches):
        reverse=h4c_once(reverse,b,a)
    if reverse!=base or original!=PINNED_SHA:
        raise RuntimeError('M470_H4M8_GENERATED_REVERSE')
    compile(s,'<H4M8 held generated>','exec')
    return s,original


# H4-M8-2B: integrated held generation, distinct from accepted core generator.
# This module never imports or executes generated CAM training while HOLD.
def h4m8_batch_contract(xp, extras, before, after):
    """Check per-world latch continuity, native qualification and exact ledgers."""
    x=lambda k:extras['h4m8_'+k]
    e=x('eligible')==1
    start=x('start')==1
    complete=x('completed')==1
    age=x('tick_age')
    side=x('tick_side')
    q=x('qualified')==1
    close=sum(x('close_'+k) for k in ('terminal','truncation','boundary','command','event'))
    n=e.shape[0]
    for z in (before,after):
        for k in ('active','age','side','positive','consumed'):
            if z['h4m8_'+k].shape!=(n,) or z['h4m8_'+k].dtype!=xp.int32:
                raise RuntimeError('M470_H4M8_CARRY_SCHEMA '+k)
    pending=start | (e & (~complete))
    next_age=xp.where(e & (~complete),age,0)
    next_side=xp.where(pending,xp.where(start,x('event_side'),side),0)
    prev=xp.concatenate((before['h4m8_active'][:,None]>0,pending[:,:-1]),axis=1)
    prev_age=xp.concatenate((before['h4m8_age'][:,None],next_age[:,:-1]),axis=1)
    prev_side=xp.concatenate((before['h4m8_side'][:,None],next_side[:,:-1]),axis=1)
    bad=(extras['h4_bad']!=0)|(extras['h4cc_bad']!=0)
    bad=bad | (q!=(extras['h4_event_qual']>0)) | (x('event_side')!=extras['h4_event_qual'])
    bad=bad | (x('event_terminal')!=(q & (extras['h4_source_terminal']>0)).astype(xp.int32))
    bad=bad | (close<0)|(close>1)|(e.astype(xp.int32)+close!=prev.astype(xp.int32))
    bad=bad | (start & prev) | (side!=xp.where(prev,prev_side,0))
    bad=bad | (e & (age!=prev_age+1)) | (complete!=(e & (age==50)))
    bad=bad | ((x('positive_start')>0)&((~e)|(x('shadow')<=0)))
    bad=bad | (x('close_terminal')>0)&(extras['h4_source_terminal']==0)
    for k,last in (('active',pending),('age',next_age),('side',next_side)):
        bad=bad | (after['h4m8_'+k][:,None]!=last[:,-1:])
    prev_positive=before['h4m8_positive']>0
    consumed=before['h4m8_consumed']>0
    for k in range(e.shape[1]):
        positive=e[:,k] & (x('shadow')[:,k]>0)
        expected=positive & (~prev_positive)
        # A scalar mismatch is enough to reject the entire batch.
        bad=bad | xp.any((x('positive_start')[:,k]>0)!=expected)
        consumed=consumed & (extras['h4_boundary'][:,k]==0)
        bad=bad | xp.any(q[:,k]&consumed)
        consumed=consumed|q[:,k]
        prev_positive=pending[:,k]&(~start[:,k])&(prev_positive|positive)
    bad=bad | xp.any(after['h4m8_positive']!=(prev_positive.astype(xp.int32)))
    bad=bad | xp.any(after['h4m8_consumed']!=(consumed.astype(xp.int32)))
    return bad


H4M8_UPDATE_BLOCK = '''    # Update normalization params and normalize observations.
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        _remove_pixels(data.observation),
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=learning_data, normalizer_params=normalizer_params
        ),
        (training_state.optimizer_state, training_state.params, key_sgd),
        (),
        length=num_updates_per_batch,
    )
'''

H4M8_GATED_UPDATE_BLOCK = '''    # Validity is an input to the conditional, not a post-SGD metric.
    from m463_instrument_brax import h4m8_batch_contract
    h8_bad = h8_bad | h4m8_batch_contract(
        jnp,data.extras['state_extras'],h8_before,state.info)
    h8_valid = ~jnp.any(h8_bad)
    for h8_leaf in jax.tree_util.tree_leaves((data,training_state,state,learning_data)):
      h8_valid = h8_valid & jnp.all(jnp.isfinite(h8_leaf))
    # Single device is frozen in H4X_CONFIG_SOURCE; all replicas agree anyway.
    h8_valid = jax.lax.pmin(h8_valid.astype(jnp.int32), _PMAP_AXIS_NAME) == 1
    def h8_update(_):
      norm = running_statistics.update(
          training_state.normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
      )
      (opt, par, _), met = jax.lax.scan(
          functools.partial(sgd_step, data=learning_data, normalizer_params=norm),
          (training_state.optimizer_state, training_state.params, key_sgd),
          (), length=num_updates_per_batch,
      )
      return opt,par,norm,met
    def h8_reject(_):
      # eval_shape traces shape metadata only; this branch performs no update.
      meta = jax.eval_shape(h8_update,None)[3]
      met = jax.tree_util.tree_map(lambda x:jnp.zeros(x.shape,x.dtype),meta)
      return training_state.optimizer_state,training_state.params,training_state.normalizer_params,met
    optimizer_state,params,normalizer_params,metrics = jax.lax.cond(
        h8_valid,h8_update,h8_reject,None)
    # Keep actual consumed cost separate from desired on a rejected batch.
    h8_actual = jnp.where(h8_valid,h8_actual,jnp.zeros_like(h8_actual))
'''

H4M8_CHUNK_OLD = '''  def h4_chunk(training_state, state, key):
    carry, metrics = jax.lax.scan(
        training_step, (training_state, state, key), (), length=5)
    return carry, jax.tree_util.tree_map(jnp.mean, metrics)
'''

H4M8_CHUNK_HELD = '''  def h4_chunk(training_state, state, key):
    # A failed batch latches STOP for all remaining scan slots. Do not
    # collect more transitions or invoke optimizer/normalizer in those slots.
    def h8_scan(carry, unused):
      original, failed = carry
      def collect(c):
        out, met = training_step(c,None)
        stop = (met['h4m8_batch_rejected']>0) | (met['h4x_finite']!=1)
        return (out,stop),met
      def skip(c):
        meta = jax.eval_shape(training_step,c,None)[1]
        met = jax.tree_util.tree_map(lambda x:jnp.zeros(x.shape,x.dtype),meta)
        return (c,jnp.asarray(True)),met
      return jax.lax.cond(failed,skip,collect,original)
    (carry,failed), metrics = jax.lax.scan(
        h8_scan, ((training_state,state,key),jnp.asarray(False)),(),length=5)
    reduced = jax.tree_util.tree_map(jnp.mean,metrics)
    for k,v in metrics.items():
      if k.startswith('h4m8_'):
        reduced[k] = jnp.sum(v,axis=0)
    metrics = reduced
    metrics['h4m8_chunk_failed'] = failed.astype(jnp.float32)
    return carry, metrics
'''

def make_h4m8_integrated_source(raw):
    """Generate, reverse-check and compile only; unconditional train HOLD remains."""
    base,original=make_h4m8_source(raw)
    old_step='        env_steps=training_state.env_steps + env_step_per_training_step,\n'
    new_step='        env_steps=jax.lax.cond(h8_valid,lambda _:training_state.env_steps+env_step_per_training_step,lambda _:training_state.env_steps,None),\n'
    host_anchor="      h4x_audit_fn('chunk_after', index=idx+1, carry=carry, metrics=metrics)\n"
    patches=(
        (H4M8_UPDATE_BLOCK,H4M8_GATED_UPDATE_BLOCK),
        (old_step,new_step),
        ('    training_state, state, key = carry\n',
         "    training_state, state, key = carry\n"
         "    h8_before = {k:state.info[k] for k in ('h4m8_active','h4m8_age','h4m8_side','h4m8_positive','h4m8_consumed')}\n"),
        (H4M8_METRICS_SOURCE,H4M8_METRICS_SOURCE+
         "    coverage['h4m8_batch_collected'] = jnp.asarray(1.,jnp.float32)\n"
         "    coverage['h4m8_batch_rejected'] = (~h8_valid).astype(jnp.float32)\n"
         "    coverage['h4m8_batch_committed'] = h8_valid.astype(jnp.float32)\n"
         "    coverage['h4m8_consumed_learning_sum'] = jnp.where(h8_valid,jnp.sum(learning_data.reward),0.)\n"
         "    for h8side in (1,2):\n"
         "      h8prefix='h4m8_side_'+str(h8side)+'_'\n"
         "      coverage[h8prefix+'pending_before']=jnp.sum(((h8_before['h4m8_active']>0)&(h8_before['h4m8_side']==h8side)).astype(jnp.float32))\n"
         "      h8positive=(h8['h4m8_tick_side']==h8side)&(h8_desired>0)\n"
         "      coverage[h8prefix+'positive_rows']=jnp.sum(h8positive.astype(jnp.float32))\n"
         "      coverage[h8prefix+'delivered_rows']=jnp.sum((h8positive&(h8_actual>0)&h8_valid).astype(jnp.float32))\n"
         "      coverage[h8prefix+'consumed_desired']=jnp.where(h8_valid & h4m8_treatment,jnp.sum(jnp.where(h8positive,h8_desired,0.)),0.)\n"
         "      coverage[h8prefix+'pending_after']=jnp.sum(((state.info['h4m8_active']>0)&(state.info['h4m8_side']==h8side)).astype(jnp.float32))\n"
         "      for h8k in range(1,51):\n"
         "        coverage[h8prefix+'age_'+str(h8k)]=jnp.sum(((h8['h4m8_tick_side']==h8side)&(h8['h4m8_tick_age']==h8k)).astype(jnp.float32))\n"
         "      for h8k in range(unroll_length):\n"
         "        coverage[h8prefix+'offset_'+str(h8k)]=jnp.sum(((h8['h4m8_tick_side'][:,h8k]==h8side)&(h8['h4m8_shadow'][:,h8k]>0)).astype(jnp.float32))\n"),
        (H4M8_CHUNK_OLD,H4M8_CHUNK_HELD),
        ('((h8_desired>0)&(h8_actual==0)&h4m8_treatment).astype(jnp.float32)',
         '((h8_desired>0)&(h8_actual==0)&h4m8_treatment&h8_valid).astype(jnp.float32)'),
        ('    h4m8_treatment=None,\n','    h4m8_treatment=None,\n    h4m8_abort_fn=None,\n'),
        (host_anchor,
         "      if float(metrics['h4m8_chunk_failed']) != 0.:\n"
         "        if h4m8_abort_fn is None:\n"
         "          raise RuntimeError('M470_H4M8_ABORT_CALLBACK_MISSING')\n"
         "        h4m8_abort_fn(idx+1,carry,metrics)\n"
         "        raise RuntimeError('M470_H4M8_PRE_SGD_OR_FINITE_STOP')\n"+host_anchor),
    )
    s=base
    for old,new in patches:
        s=h4c_once(s,old,new)
    back=s
    for old,new in reversed(patches):
        back=h4c_once(back,new,old)
    if back!=base or s.count("raise RuntimeError('M470_H4M8_GENERATED_EXECUTION_HOLD')")!=1:
        raise RuntimeError('M470_H4M8_INTEGRATION_REVERSE_OR_HOLD')
    compile(s,'<CAM integrated held>','exec')
    return s,original


def h4m8_build_held(raw):
    """Common generated source for either arm; no module import/write/dispatch."""
    s,original=make_h4m8_integrated_source(raw)
    return s,{'original_brax_sha256':original,
              'instrumented_sha256':sha256(s.encode('utf8')),
              'extra_fields':list(EXT_FIELDS+H4_EXT_FIELDS+H4CC_EXT_FIELDS+H4M8_EXT_FIELDS),
              'runtime_authorized':False,'generated_execution_hold':True}


# H4-M12-1: held reset-edge continuation core. No executable builder entry.
H4M12_HISTORICAL_GENERATED_SHA = '8f30e5c695211b21815375c37b1bda67fcda98b2ee259a703e87b182c3402062'


def make_h4m12_held_source(raw):
    """One original CTRL epoch; reversible host-boundary changes only.

    The generated train function is unconditionally held. A future reviewed
    integration must supply actual callbacks, authorization and budget guards.
    No module import, native call, checkpoint read or write here.
    """
    base, original = make_h4_exploratory_source(raw)
    if original != PINNED_SHA or sha256(base.encode('utf8')) != H4M12_HISTORICAL_GENERATED_SHA:
        raise RuntimeError('M470_H4M12_HISTORICAL_GENERATOR_PIN')
    config = H4X_CONFIG_SOURCE.replace('num_timesteps != 10485760 or num_evals != 3',
                                       'num_timesteps != 5242880 or num_evals != 2')
    config = config.replace('seed != 1 or m470_resume_state_path is not None',
                            'seed != 1 or m470_resume_state_path is None\n'
                            '      or m470_resume_expected_steps != 10485760')
    config = config.replace('restore_checkpoint_path is None or randomization_fn is None',
                            'restore_checkpoint_path is not None or randomization_fn is None')
    config = config.replace('M470_H4X_CONFIG', 'M470_H4M12_CONFIG')
    old_chunk = '''  if (num_training_steps_per_epoch != 20 or num_evals != 3
      or num_resets_per_eval != 1 or num_timesteps != 10485760
      or num_envs != 8192 or unroll_length != 32 or action_repeat != 1
      or h4_chunk_fn is None or m470_resume_state_path is not None):
    raise RuntimeError('M470_H4_CHUNK_CONTRACT')
'''
    new_chunk = old_chunk.replace('num_evals != 3','num_evals != 2').replace(
        'num_timesteps != 10485760','num_timesteps != 5242880').replace(
        'm470_resume_state_path is not None','m470_resume_state_path is None').replace(
        'M470_H4_CHUNK_CONTRACT','M470_H4M12_CHUNK_CONTRACT')
    patches = (
        ("  from m463_push_ppo_train import h4x_generated_guard\n"
         "  h4x_generated_guard(h4x_audit_fn, __file__)\n",
         "  raise RuntimeError('M470_H4M12_GENERATED_EXECUTION_HOLD')\n"),
        (H4X_CONFIG_SOURCE, config),
        (old_chunk, new_chunk),
        ("  h4x_audit_fn('reset_before', keys=key_envs, local_key=local_key,\n"
         "                eval_key=eval_key, model=dr._mjx_model_v, axes=dr._in_axes)\n",
         "  h4x_audit_fn('template_reset_before', keys=key_envs, local_key=local_key,\n"
         "                eval_key=eval_key, model=dr._mjx_model_v, axes=dr._in_axes)\n"),
        ("  h4x_audit_fn('reset_after', env_state=env_state, finite=h4x_dynamic_finite(env_state))\n"
         "  # Discard",
         "  h4x_audit_fn('template_reset_after', env_state=env_state, finite=h4x_dynamic_finite(env_state))\n"
         "  # Discard"),
        ("    restored = m470_payload['training_state']\n",
         """    if (jax.tree_util.tree_structure(m470_payload) !=
        jax.tree_util.tree_structure(m470_template)):
      raise RuntimeError('M470_H4M12_RESTORE_TREE')
    for ref_leaf, restored_leaf in zip(jax.tree_util.tree_leaves(m470_template),
                                       jax.tree_util.tree_leaves(m470_payload)):
      ra, rb = np.asarray(jax.device_get(ref_leaf)), np.asarray(jax.device_get(restored_leaf))
      if ra.shape != rb.shape or ra.dtype != rb.dtype or not np.isfinite(rb).all():
        raise RuntimeError('M470_H4M12_RESTORE_LEAF')
    h4x_audit_fn('restore_payload', template=m470_template, payload=m470_payload)
    restored = m470_payload['training_state']
"""),
        ("    env_state = reset_fn(key_envs)\n\n  if num_timesteps == 0:\n",
         """    h4x_audit_fn('restore_reset_before', keys=key_envs, local_key=local_key,
                  eval_key=m470_eval_rng, model=dr._mjx_model_v, axes=dr._in_axes)
    env_state = reset_fn(key_envs)
    h4x_audit_fn('restore_reset_after', env_state=env_state,
                  finite=h4x_dynamic_finite(env_state))

  if num_timesteps == 0:
"""),
        ("  assert total_steps >= num_timesteps\n",
         "  if total_steps != 15728640:\n"
         "    raise RuntimeError('M470_H4M12_FINAL_STEPS')\n"),
    )
    s = base
    for old, new in patches:
        s = h4c_once(s, old, new)
    back = s
    for old, new in reversed(patches):
        back = h4c_once(back, new, old)
    if back != base or s.count("raise RuntimeError('M470_H4M12_GENERATED_EXECUTION_HOLD')") != 1:
        raise RuntimeError('M470_H4M12_REVERSE_OR_HOLD')
    compile(s, '<H4-M12 held generated>', 'exec')
    return s, original


def make_h4m12_integrated_source(raw):
    """Same held core; distinct fail-closed entry, parent byte check, no loss edits."""
    base, original = make_h4m12_held_source(raw)
    patches = (
        # Only the deserialization template gets JAX's scalar dtype. The
        # upstream fresh Python env_steps=0 is not an ndarray until pmap.
        ("    m470_template = {\n"
         "        'training_state': training_state,\n",
         "    m470_template = {\n"
         "        'training_state': training_state.replace(\n"
         "            env_steps=jnp.asarray(training_state.env_steps)),\n"),
        ("  raise RuntimeError('M470_H4M12_GENERATED_EXECUTION_HOLD')\n",
         "  from m463_push_ppo_train import h4m12_generated_guard\n"
         "  h4m12_generated_guard(h4x_audit_fn, __file__, environment, m470_resume_state_path)\n"),
        ("    m470_payload = _m470_serialization.from_bytes(\n"
         "        m470_template, m470_file.read_bytes())\n",
         "    from hashlib import sha256 as _m470_sha\n"
         "    m470_raw = m470_file.read_bytes()\n"
         "    if _m470_sha(m470_raw).hexdigest() != '9b19312da08d71513ad8fa6638c632401bec20e09399fdf4a9f6f6e2b2619c80':\n"
         "      raise RuntimeError('M470_H4M12_PARENT_CHANGED')\n"
         "    m470_payload = _m470_serialization.from_bytes(m470_template, m470_raw)\n"),
    )
    s = base
    for old,new in patches:
        s=h4c_once(s,old,new)
    back=s
    for old,new in reversed(patches):
        back=h4c_once(back,new,old)
    if back!=base:
        raise RuntimeError('M470_H4M12_INTEGRATION_PROJECTION')
    compile(s,'<H4-M12 integrated generated>','exec')
    return s,original
