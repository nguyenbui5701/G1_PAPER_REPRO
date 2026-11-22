#!/usr/bin/env python3
"""M4.63 external (source-preserving) G1 push coverage environment.

Only this separate experiment file modifies the environment logic; the pinned
Playground 2025 source, PD, reward and termination code are not edited.
Every telemetry field is a JAX array captured after the transition.
Push is an instantaneous world-XY root qvel delta in m/s, NOT a force in N.
"""
from __future__ import annotations

# These imports execute only in the user's pinned venv.
import jax
import jax.numpy as jp
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.g1 import joystick


# Balanced motion commands: stand, forward slow, forward fast,
# forward+left turn, forward+right turn, lateral strafe.
COMMANDS = jp.asarray([
    [0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0],
    [0.6, 0.0, 0.0],
    [0.4, 0.0, 0.8],
    [0.4, 0.0, -0.8],
    [0.0, 0.35, 0.0],
], dtype=jp.float32)

# forward/back/left/right/forward-left/forward-right in BODY coordinates.
PUSH_UNIT_BODY = jp.asarray([
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [0.7071067811865476, 0.7071067811865476],
    [0.7071067811865476, -0.7071067811865476],
], dtype=jp.float32)

STAGES = {
    # Magnitudes form curriculum examples, not paper parameters.
    # Target after initial 12..24 steps, then at most 8 steps to find phase.
    'easy': (0.20, 0.60, 0.00),
    'easy_repeat': (0.20, 0.60, 0.25),
    'medium': (0.30, 0.90, 0.25),
    'hard': (0.40, 1.20, 0.50),
    'easy_repeat_delayed': (0.20, 0.60, 0.25),
    'easy_repeat_support': (0.20, 0.60, 0.25),
    'easy_mixed_time': (0.20, 0.60, 0.00),
}

EXTRA_FIELDS = (
    'm463_event',
    'm463_motion',
    'm463_direction',
    'm463_phase_target',
    'm463_phase_actual',
    'm463_phase_matched',
    'm463_fallback',
    'm463_repeat',
    'm463_support_side', 'm463_repeat_gap',
    'm463_repeat_side_target',
    'm463_repeat_side_matched',
    'm463_repeat_side_fallback',
    'm463_magnitude',
    'm463_term_recent',
    'm463_no_push',
)


def heading_from_wxyz(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class CoverageJoystick(joystick.Joystick):
    """JAX-compatible source G1 subclass, independent of pinned checkout.

    BraxAutoResetWrapper restores initial physical state after termination but
    leaves info alive; `info['steps']==0` is therefore used as the episode
    boundary for re-sampling push targets and counters.  Tracking `age` also
    supports a direct/unwrapped one-env diagnostic.
    """
    def __init__(self, *, stage: str, config, task='flat_terrain'):
        if stage not in STAGES:
            raise ValueError('M463_BAD_STAGE')
        if bool(config.push_config.enable):
            raise ValueError('M463_SOURCE_PUSH_MUST_BE_OFF_TO_PREVENT_DOUBLE_PUSH')
        self._m463_stage = stage
        self._m463_min_mag, self._m463_max_mag, self._m463_repeat_probability = STAGES[stage]
        super().__init__(task=task, config=config)

    def sample_command(self, rng):
        # Source calls this method in reset and after its regular command refresh.
        idx = jax.random.randint(rng, (), 0, int(COMMANDS.shape[0]))
        return COMMANDS[idx]

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        # Step will re-sample at any episode boundary (info['steps']==0),
        # but keys and shapes must exist at reset for scan/vmap.
        zeros = jp.asarray(0, dtype=jp.int32)
        info.update(
            m463_age=zeros,
            m463_next=zeros,
            m463_push_count=zeros,
            m463_recent=zeros,
            m463_no_push=zeros,
            m463_direction=zeros,
            m463_phase_target=zeros,
            m463_event=zeros,
            m463_motion=zeros,
            m463_phase_actual=zeros,
            m463_phase_matched=zeros,
            m463_fallback=zeros,
            m463_repeat=zeros,
            m463_support_side=zeros,
            m463_repeat_gap=zeros,
            m463_repeat_side_target=zeros,
            m463_repeat_side_matched=zeros,
            m463_repeat_side_fallback=zeros,
            m463_last_event_age=jp.asarray(-1, dtype=jp.int32),
            m463_magnitude=jp.asarray(0.0, dtype=jp.float32),
            m463_term_recent=zeros,
        )
        return state.replace(info=info)

    def step(self, state, action):
        info = dict(state.info)
        rng, k_start, k_dir, k_phase, k_time, k_mag, k_gap, k_repeat = jax.random.split(
            info['rng'], 8)
        info['rng'] = rng
        # This wrapper resets `steps` to 0 on the next transition after done.
        boundary = (info['steps'] == 0) if 'steps' in info else (info['m463_age'] == 0)
        age = jp.where(boundary, jp.asarray(0, jp.int32), info['m463_age'])
        count = jp.where(boundary, jp.asarray(0, jp.int32), info['m463_push_count'])
        recent = jp.where(boundary, jp.asarray(0, jp.int32), info['m463_recent'])
        last_event_age = jp.where(
	        boundary, jp.asarray(-1, jp.int32), info['m463_last_event_age']
        )
        if self._m463_stage == 'easy_mixed_time':
            k_pick, k_early, k_late = jax.random.split(k_time, 3)

            early = jax.random.randint(k_early, (), 12, 25)
            late = jax.random.randint(k_late, (), 200, 220)

            target_first = jp.where(
                jax.random.bernoulli(k_pick, p=0.5),
                late,
                early
            )
        else:
            target_first = jax.random.randint(k_time, (), 12, 25)
        next_step = jp.where(boundary, target_first, info['m463_next'])
        disabled = jp.where(boundary, jax.random.bernoulli(k_start, p=0.20),
                            info['m463_no_push'] > 0)
        direction = jp.where(boundary, jax.random.randint(k_dir, (), 0, 6),
                             info['m463_direction'])
        # 0 = any, 1 = one foot contact, 2 = both foot contacts.
        phase_target = jp.where(boundary, jax.random.randint(k_phase, (), 0, 3),
                                info['m463_phase_target'])

        contact = jp.asarray([
            geoms_colliding(state.data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])
        n_contact = jp.sum(contact.astype(jp.int32))
        phase_actual = jp.where(n_contact == 1, 1, jp.where(n_contact == 2, 2, 0))
        # Pre-push: 0=air, 1=left, 2=right, 3=both.
        support_side = (
        	contact[0].astype(jp.int32)
        	+ 2 * contact[1].astype(jp.int32)
        )
        # Giữ nguyên ý nghĩa phase-matching của stage cũ.
        matched = (phase_target == 0) | (phase_target == phase_actual)

        # Chỉ áp dụng cho repeat-push sang trái/phải.
        # 1 = chỉ chân trái chống; 2 = chỉ chân phải chống.
        side_target = jp.where(
            direction == 2, 1,
            jp.where(direction == 3, 2, 0)
        )

        side_gate = (
            (self._m463_stage == 'easy_repeat_support')
            & (count > 0)
            & (side_target > 0)
        )

        side_matched = support_side == side_target
        effective_match = jp.where(side_gate, side_matched, matched)

        deadline = age >= next_step + 8
        ready = (age >= next_step) & (effective_match | deadline)
        # After first exposure, optional repeat 15..30 steps later.
        max_count = (
            1 if self._m463_stage in ('easy', 'easy_mixed_time')
            else 2
        )
        event = (~disabled) & ready & (count < max_count)
        # Magnitude is sampled only logically at event time. JAX still computes
        # branches without Python dynamic condition (fully JIT-compatible).
        magnitude = jax.random.uniform(k_mag, (), minval=self._m463_min_mag,
                                       maxval=self._m463_max_mag)
        body_delta = PUSH_UNIT_BODY[direction] * magnitude
        yaw = heading_from_wxyz(state.data.qpos[3:7])
        c, s = jp.cos(yaw), jp.sin(yaw)
        world_delta = jp.asarray([
            c * body_delta[0] - s * body_delta[1],
            s * body_delta[0] + c * body_delta[1],
        ]) * event.astype(jp.float32)
        updated_qvel = state.data.qvel.at[:2].add(world_delta)
        state = state.replace(data=state.data.replace(qvel=updated_qvel), info=info)
        # The pinned G1 source runs the SAME PD, physics, reward, termination,
        # observations, and RNG progression. Built-in random push is OFF.
        after = super().step(state, action)
        out = dict(after.info)
        repeat = event & (count > 0)
        side_event = repeat & side_gate
        actual_gap = jp.where(
	        repeat, age - last_event_age, jp.asarray(0, jp.int32)
        )
        new_count = count + event.astype(jp.int32)
        # A new repeat push is scheduled even for ineligible episodes, then
        # suppressed by max_count, so no conditional Python/JAX shape branches.
        gap_min, gap_max = (
            (90, 116)
            if self._m463_stage == 'easy_repeat_delayed'
            else (15, 31)
        )
        gap = jax.random.randint(k_gap, (), gap_min, gap_max)
        repeat_allowed = jax.random.bernoulli(k_repeat, p=self._m463_repeat_probability)
        new_next = jp.where(event & repeat_allowed, age + gap,
                            jp.where(event, jp.asarray(2**30, jp.int32), next_step))
        # If no repeat was sampled, do not allow a second event.
        new_count = jp.where(event & (~repeat_allowed), max_count, new_count)
        motion_id = jp.argmin(jp.sum(jp.abs(COMMANDS - out['command']), axis=-1))
        recent_after = jp.where(event, 100, jp.maximum(recent - 1, 0))
        term_recent = (after.done > 0) & (recent_after > 0)
        out.update(
            m463_age=age+1,
            m463_next=new_next,
            m463_push_count=new_count,
            m463_recent=recent_after,
            m463_no_push=disabled.astype(jp.int32),
            m463_direction=direction.astype(jp.int32),
            m463_phase_target=phase_target.astype(jp.int32),
            m463_event=event.astype(jp.int32),
            m463_motion=motion_id.astype(jp.int32),
            m463_phase_actual=phase_actual.astype(jp.int32),
            m463_phase_matched=(event & matched).astype(jp.int32),
            m463_fallback=(event & (~matched)).astype(jp.int32),
            m463_repeat=repeat.astype(jp.int32),
            m463_magnitude=jp.where(event, magnitude, 0.0),
            m463_term_recent=term_recent.astype(jp.int32),
            m463_support_side=support_side,
            m463_repeat_gap=actual_gap,
            m463_repeat_side_target=jp.where(
                side_event, side_target, 0
            ).astype(jp.int32),
            
            m463_repeat_side_matched=(
                side_event & side_matched
            ).astype(jp.int32),
            
            m463_repeat_side_fallback=(
                side_event & (~side_matched)
            ).astype(jp.int32),
            m463_last_event_age=jp.where(event, age, last_event_age),
        )
        return after.replace(info=out)
