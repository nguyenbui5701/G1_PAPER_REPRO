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
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def make_patched_source(raw, *, require_pin=True):
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
    new="extra_fields=('truncation', " + ', '.join(repr(k) for k in EXT_FIELDS) + '),'
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
    metrics = {**metrics, **coverage, **m464_tail}
'''
    return s.replace(anchor,instrument+anchor,1),actual_sha


def build_instrumented_ppo(pinned_ppo_module, output:Path):
    import inspect
    p=Path(inspect.getfile(pinned_ppo_module.train))
    patched,original_sha=make_patched_source(p.read_bytes())
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
                'extra_fields':list(EXT_FIELDS)}
