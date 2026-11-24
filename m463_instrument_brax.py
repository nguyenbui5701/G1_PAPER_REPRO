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
    coverage['m463_h4_bad_per_rollout'] = h4_count(
        (extras['h4_bad'] != 0) | (extras['truncation'] != 0))
    ha = extras['h4_age']
    hm = jnp.sqrt(extras['h4_actual_dx']**2 + extras['h4_actual_dy']**2)
    coverage['m463_h4_bad_per_rollout'] += h4_count(
        (~jnp.isfinite(hm)) | (jnp.abs(hm - extras['m463_magnitude']) > 3e-6))
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
    one('  if madrona_backend:\\n',
        "  raise RuntimeError('M470_H4_EXECUTION_HOLD')\\n"
        '  if madrona_backend:\\n')
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


def make_patched_source(raw, *, require_pin=True, enable_resume=False, enable_h3=False, enable_h4=False):
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
    fields = EXT_FIELDS + (H3_EXT_FIELDS if enable_h3 else ()) + (H4_EXT_FIELDS if enable_h4 else ())
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
        instrument = instrument.replace(marker, H4_METRICS_SOURCE + marker, 1)
    s = s.replace(anchor, instrument+anchor, 1)
    if enable_resume:
        s = _add_m470_fullstate_hooks(s)
    if enable_h4:
        s = _add_h4_chunk_hooks(s)
    return s, actual_sha


def build_instrumented_ppo(pinned_ppo_module, output:Path, *, enable_resume=False, enable_h3=False, enable_h4=False):
    import inspect
    p=Path(inspect.getfile(pinned_ppo_module.train))
    patched,original_sha=make_patched_source(p.read_bytes(),
        enable_resume=enable_resume, enable_h3=enable_h3, enable_h4=enable_h4)
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
                'extra_fields':list(EXT_FIELDS + (H3_EXT_FIELDS if enable_h3 else ()) + (H4_EXT_FIELDS if enable_h4 else ()))}
