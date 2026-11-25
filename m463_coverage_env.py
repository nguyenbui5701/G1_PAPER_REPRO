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
from mujoco_playground._src import gait
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


# H4: scalar info only, never included in actor/critic observations.
H4_FIELDS = ('h4_class', 'h4_boundary', 'h4_age', 'h4_target', 'h4_motion', 'h4_actor_motion', 'h4_refresh', 'h4_changed', 'h4_assigned', 'h4_first_pending', 'h4_repeat_pending', 'h4_first_due', 'h4_repeat_due', 'h4_first_closed', 'h4_repeat_closed', 'h4_event_qual', 'h4_mature_repeat', 'h4_repeat_assigned', 'h4_event_terminal', 'h4_tail_kind', 'h4_tail_cell', 'h4_tail_age', 'h4_tail_same', 'h4_tick_kind', 'h4_tick_cell', 'h4_tick_age', 'h4_tick_same', 'h4_tick_survived', 'h4_close_repeat', 'h4_close_terminal', 'h4_close_complete', 'h4_close_boundary', 'h4_bad', 'h4_assign_motion', 'h4_repeat_origin_motion', 'h4_class_event', 'h4_source_terminal', 'h4_command_mismatch',)
H4_WINDOWS = (10, 25, 50, 100, 250)


# H4-M2: metadata-only shadow in REF, exact command delivery in CCRC.
H4CC_FIELDS = ('start','event_terminal','tick','tick_age','tick_side','tick_same',
    'tick_survived','tail_terminal','completed','refresh','sample_same',
    'sample_different','deferred','published','terminal_restored','bad','episode',
    'event_age','event_motion','event_side')
H4CC_STATE = ('active','age','same','pending_valid')

def h4cc_transition(xp, old, *, treatment, event, mature_repeat, side, motion,
                     age, boundary, terminal, refresh, pre, actor, critic,
                     native, next_actor, next_critic):
    """Pure array transition; source action/reward and RNG have already run."""
    treatment = xp.asarray(treatment, dtype=bool)
    active = old['h4cc_active'] > 0
    prior = active & (~boundary)
    start = mature_repeat
    tick = prior & (~event)
    n = xp.where(tick, old['h4cc_age'] + 1, 0)
    held = xp.where(start, pre, old['h4cc_held_command'])
    exact = xp.all(actor == held) & xp.all(critic == held) & xp.all(pre == held)
    same = exact & (start | (old['h4cc_same'] > 0))
    scope = start | tick
    entered = start & (~terminal)
    keep = (entered | (tick & (n < 250))) & (~terminal)
    complete = tick & (n == 250) & (~terminal)
    old_pending = (old['h4cc_pending_valid'] > 0) & prior
    native_refresh = scope & refresh
    defer = treatment & native_refresh & (~terminal)
    pending = xp.where(defer, native, old['h4cc_pending_command'])
    valid = old_pending | defer
    publish = treatment & complete & valid
    restore = treatment & tick & terminal & old_pending
    delivered = xp.where(treatment & keep, held, native)
    delivered = xp.where(publish, pending, delivered)
    delivered = xp.where(restore & (~refresh), pending, delivered)
    # On terminal AutoReset owns next obs. On release sync existing slices only.
    patch = treatment & (keep | publish)
    ao = xp.where(patch, delivered, next_actor)
    co = xp.where(patch, delivered, next_critic)
    finite = xp.all(xp.isfinite(held)) & xp.all(xp.isfinite(native))
    bad = ((active & boundary) | (prior & event) | (tick & (n > 250)) |
           (old_pending & (~treatment)) | (old_pending & native_refresh) |
           (scope & (~finite)) |
           (treatment & scope & (~same)) |
           (patch & (~(xp.all(ao == delivered) & xp.all(co == delivered)))))
    vals = dict(start=start,event_terminal=start & terminal,tick=tick,
        tick_age=n,tick_side=xp.where(tick,old['h4cc_event_side'],0),
        tick_same=tick & same,tick_survived=tick & (~terminal),
        tail_terminal=tick & terminal,completed=complete,refresh=native_refresh,
        sample_same=native_refresh & xp.all(native == held),
        sample_different=native_refresh & xp.any(native != held),
        deferred=defer,published=publish,terminal_restored=restore,bad=bad,
        episode=old['h4cc_episode'] + boundary.astype(xp.int32),
        event_age=xp.where(start,age,old['h4cc_event_age']),
        event_motion=xp.where(start,motion,old['h4cc_event_motion']),
        event_side=xp.where(start,side,old['h4cc_event_side']),
        active=keep,age=xp.where(keep,n,0),same=keep & same,
        pending_valid=treatment & keep & valid)
    out = {'h4cc_'+k:xp.asarray(v,xp.int32) for k,v in vals.items()}
    out['h4cc_held_command'] = xp.where(keep,held,xp.zeros_like(held))
    out['h4cc_pending_command'] = xp.where(out['h4cc_pending_valid']>0,
                                         pending,xp.zeros_like(pending))
    return delivered, ao, co, out


# H4-M6-RSS-V1: event metadata only; never part of observations.
H4M6_REWARD_NAMES = ('tracking_lin_vel','tracking_ang_vel','lin_vel_z','ang_vel_xy',
    'orientation','base_height','torques','action_rate','energy','dof_acc',
    'feet_slip','feet_clearance','feet_height','feet_air_time','feet_phase',
    'alive','termination','stand_still','collision','contact_force',
    'joint_deviation_hip','joint_deviation_knee','dof_pos_limits','pose')
H4M6_STATE = ('active','age','side','positive','consumed')
H4M6_INT_FIELDS = ('qualified','event_terminal','start','event_side','eligible',
    'tick_age','tick_side','positive_start','completed','close_terminal',
    'close_truncation','close_boundary','close_command','close_event','bad',
    'opportunity_side','attempt_side')
H4M6_FLOAT_FIELDS = ('factor','raw_stand','shadow_stand','actual_stand',
    'shadow_relief','delivered_relief','base_reward','expected_reward')
H4M6_FIELDS = H4M6_INT_FIELDS + H4M6_FLOAT_FIELDS

def h4m6_pre(xp, old, *, event, repeat, cls, first_age, first_mag, gap,
              magnitude, first_motion, pre_motion, actor_motion, direction,
              support_side, boundary, age, clock, wrapper_steps,
              command, actor_command, critic_command):
    """Pure prospective metadata; called before original source step."""
    active = old['h4m6_active'] > 0
    old_age = old['h4m6_age']
    consumed = (old['h4m6_consumed'] > 0) & (~boundary)
    side_target = xp.where(direction == 2, 1, xp.where(direction == 3, 2, 0))
    mature = (event & repeat & (cls == 3) & (first_age >= 240) &
              (first_age <= 308) & (first_mag >= .45) & (first_mag <= .55) &
              (gap >= 30) & (gap <= 40) & (magnitude >= .30) &
              (magnitude <= .40) & (first_motion == pre_motion) &
              (actor_motion == pre_motion))
    native_qual = mature & (pre_motion == 0) & (side_target > 0) & (support_side == side_target)
    coherent = xp.all(command == actor_command) & xp.all(command == critic_command)
    zero = xp.all(command == 0) & coherent
    clock_ok = (clock == age) & (wrapper_steps == age) & (clock < 500)
    truncation = wrapper_steps + 1 >= 1000
    qualified = native_qual & zero & clock_ok & (~boundary) & (~truncation) & (~active) & (~consumed)
    opportunity = xp.where(boundary & (cls == 3) & (pre_motion == 0),side_target,0)
    attempt = xp.where(repeat & event & (cls == 3) & (first_motion == 0) &
                       (pre_motion == 0),side_target,0)
    k = old_age + 1
    valid_old = (old_age >= 0) & (old_age < 50) & ((old['h4m6_side']==1) | (old['h4m6_side']==2))
    close_boundary = active & boundary
    close_event = active & (~boundary) & event
    close_truncation = active & (~boundary) & (~event) & truncation
    close_command = active & (~boundary) & (~event) & (~truncation) & ((~zero) | (clock >= 500))
    eligible_pre = active & valid_old & (~boundary) & (~event) & (~truncation) & zero & clock_ok
    bad = (native_qual & (~qualified)) | (active & (~valid_old)) | close_boundary | close_event
    bad = bad | (active & (~coherent)) | (active & (~boundary) & (~truncation) &
              (clock < 500) & ((clock != age) | (wrapper_steps != age)))
    bad = bad | ((~active) & ((old_age != 0) | (old['h4m6_side'] != 0) | (old['h4m6_positive'] != 0))) | (active & (~consumed) & (~boundary))
    return dict(qualified=qualified,native_qual=native_qual,mature=mature,consumed=consumed,
                side=side_target,active=active,k=k,eligible_pre=eligible_pre,
                close_boundary=close_boundary,close_event=close_event,
                close_truncation=close_truncation,close_command=close_command,
                opportunity_side=opportunity,attempt_side=attempt,bad=bad)

def h4m6_reward(xp, raw, eligible_pre, k, done, treatment):
    """Identity outside scope; scalar float32 shadow and delivered raw cost."""
    eligible = eligible_pre & (~done)
    one = xp.asarray(1.,dtype=raw.dtype)
    factor = xp.where(eligible,one-xp.asarray(.5,dtype=raw.dtype) *
                      (xp.asarray(51,dtype=raw.dtype)-k.astype(raw.dtype)) /
                      xp.asarray(50,dtype=raw.dtype),one)
    shadow = xp.where(eligible,factor*raw,raw)
    delivered = xp.where(eligible & treatment,shadow,raw)
    return eligible,factor,shadow,delivered

def h4m6_post(xp, old, pre, *, done, eligible, raw, shadow):
    """Finalize exactly one window; close priority boundary/event/trunc/command/terminal."""
    qualified = pre['qualified']
    start = qualified & (~done)
    close_terminal = pre['active'] & done & (~pre['close_boundary']) & (~pre['close_event']) & (~pre['close_truncation']) & (~pre['close_command'])
    complete = eligible & (pre['k'] == 50)
    keep = start | (eligible & (~complete))
    positive = eligible & (raw > shadow)
    positive_start = positive & (~(old['h4m6_positive'] > 0))
    bad = pre['bad'] | (eligible != (pre['eligible_pre'] & (~done))) | (raw < 0) | (shadow < 0) | (shadow > raw)
    closed = pre['close_boundary'] | pre['close_event'] | pre['close_truncation'] | pre['close_command'] | close_terminal | complete
    bad = bad | (pre['active'] & (~(closed | eligible))) | (closed & keep)
    side = old['h4m6_side']
    vals = dict(qualified=qualified,event_terminal=qualified & done,start=start,
        event_side=xp.where(qualified,pre['side'],0),eligible=eligible,
        tick_age=xp.where(eligible,pre['k'],0),tick_side=xp.where(pre['active'],side,0),
        positive_start=positive_start,completed=complete,close_terminal=close_terminal,
        close_truncation=pre['close_truncation'],close_boundary=pre['close_boundary'],
        close_command=pre['close_command'],close_event=pre['close_event'],
        opportunity_side=pre['opportunity_side'],attempt_side=pre['attempt_side'],bad=bad,
        consumed=pre['consumed'] | qualified,
        active=keep,age=xp.where(keep & (~start),pre['k'],0),
        side=xp.where(keep,xp.where(start,pre['side'],side),0),
        positive=keep & (~start) & ((old['h4m6_positive'] > 0) | positive))
    return {'h4m6_'+k:xp.asarray(v,xp.int32) for k,v in vals.items()}


def heading_from_wxyz(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def h3_countdown(event, prior_recent, boundary):
    prior = jp.where(boundary, 0, prior_recent)
    return jp.where(event, 100, jp.maximum(prior - 1, 0)).astype(jp.int32)


def h3_stand_factor(countdown, done):
    n = jp.asarray(countdown, dtype=jp.float32)
    return jp.where(done, 1.0, 1.0 - 0.5 * n / 100.0)


class CoverageJoystick(joystick.Joystick):
    """JAX-compatible source G1 subclass, independent of pinned checkout.

    BraxAutoResetWrapper restores initial physical state after termination but
    leaves info alive; `info['steps']==0` is therefore used as the episode
    boundary for re-sampling push targets and counters.  Tracking `age` also
    supports a direct/unwrapped one-env diagnostic.
    """
    def __init__(self, *, stage: str, config, task='flat_terrain', h1_m1=False, h1_m2=False, h3_m1=False, h4_arm=None, h4cc_arm=None, h4m6_arm=None, h4m6_tech_capture=False):
        if stage not in STAGES:
            raise ValueError('M463_BAD_STAGE')
        if bool(config.push_config.enable):
            raise ValueError('M463_SOURCE_PUSH_MUST_BE_OFF_TO_PREVENT_DOUBLE_PUSH')
        # Experimental reward must be opt-in; baseline is source-exact.
        self._h1_m1 = bool(h1_m1)
        self._h1_m2 = bool(h1_m2)
        if self._h1_m1 and self._h1_m2:
            raise ValueError('M470_H1_M1_M2_MUTUALLY_EXCLUSIVE')
        self._h3_m1 = bool(h3_m1)
        if self._h3_m1 and (self._h1_m1 or self._h1_m2):
            raise ValueError('M470_H3_M1_MUTUALLY_EXCLUSIVE')
        if self._h3_m1 and (
            stage != 'easy_repeat_support'
            or float(config.reward_config.scales.stand_still) != -1.0
            or float(config.reward_config.scales.termination) != -100.0
            or float(config.ctrl_dt) != 0.02
            or int(config.action_repeat) != 1
        ):
            raise ValueError('M470_H3_M1_ENV_CONFIG_MISMATCH')
        if h4_arm not in (None, 'control', 'candidate'):
            raise ValueError('M470_H4_ARM')
        if h4_arm is not None and (stage != 'easy_repeat_support' or
                self._h1_m1 or self._h1_m2 or self._h3_m1 or
                float(config.ctrl_dt) != .02 or int(config.action_repeat) != 1):
            raise ValueError('M470_H4_ENV_CONTRACT')
        if h4cc_arm not in (None, 'ref', 'ccrc') or (h4cc_arm is not None and h4_arm != 'candidate'):
            raise ValueError('M470_H4CC_ENV_CONTRACT')
        if h4m6_arm not in (None,'ref','rss') or (h4m6_arm is not None and
                (h4_arm != 'candidate' or h4cc_arm != 'ref' or
                 float(config.reward_config.scales.stand_still) != -1.0 or
                 float(config.reward_config.scales.termination) != -100.0 or
                 len(config.reward_config.scales) != 24)):
            raise ValueError('M470_H4M6_ENV_CONTRACT')
        if h4m6_tech_capture and h4m6_arm is None:
            raise ValueError('M470_H4M6_CAPTURE_CONTRACT')
        self._h4m6_tech_capture = bool(h4m6_tech_capture)
        self._h4m6_arm = h4m6_arm
        self._h4cc_arm = h4cc_arm
        self._h4_arm = h4_arm
        self._m463_stage = stage
        self._m463_min_mag, self._m463_max_mag, self._m463_repeat_probability = STAGES[stage]
        super().__init__(task=task, config=config)

    def sample_command(self, rng):
        # Source calls this method in reset and after its regular command refresh.
        idx = jax.random.randint(rng, (), 0, int(COMMANDS.shape[0]))
        return COMMANDS[idx]

    def _cost_h1_m2_phase_foot_crossing(self, data, info, contact):
        """One-step risk proxy: approaching left/right foot lateral crossing.

        Only one-foot support plus desired swing; no global site-gap threshold,
        no contact-specific collision term, no mutation to the pinned source.
        JAX-safe scalar with bounded [0,1] magnitude.
        """
        feet = data.site_xpos[self._feet_site_id]  # left, right
        feet_world_vel = data.sensordata[self._foot_linvel_sensor_adr]
        rotation = data.site_xmat[self._pelvis_imu_site_id]
        delta_local = rotation.T @ (feet[0] - feet[1])
        omega_local = rotation.T @ self.get_global_angvel(data, "pelvis")
        relative_vel_local = (
            rotation.T @ (feet_world_vel[0] - feet_world_vel[1])
            - jp.cross(omega_local, delta_local)
        )
        tau = 0.12  # seconds, predeclared short crossing prediction horizon
        predicted = delta_local + tau * jp.clip(
            relative_vel_local, -1.5, 1.5
        )
        # lateral ordering: left foot should remain on left side of right foot.
        lateral_risk = jp.clip((0.09 - predicted[1]) / 0.09, 0.0, 1.0)
        # Longitudinal adjacency is needed: staggered feet can pass safely.
        adjacent = jp.clip((0.22 - jp.abs(predicted[0])) / 0.12, 0.0, 1.0)
        approaching = jp.clip(
            (-relative_vel_local[1] - 0.05) / 0.25, 0.0, 1.0
        )
        # The pinned phase reward uses the same desired-foot-height function.
        expected_height = gait.get_rz(
            info["phase"],
            swing_height=self._config.reward_config.max_foot_height,
        )
        swing_weight = jp.clip(
            (expected_height - 0.02) / 0.08, 0.0, 1.0
        )
        swing_weight = jp.max(
            swing_weight * (~contact).astype(jp.float32)
        )
        single_support = jp.logical_xor(contact[0], contact[1])
        return (
            single_support.astype(jp.float32)
            * swing_weight * approaching * adjacent * jp.square(lateral_risk)
        )

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        """Opt-in H1 or H3: change exactly one source reward term."""
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        if self._h1_m1:
            pelvis_vz = self.get_global_linvel(data, "pelvis")[2]
            height_risk = jp.clip((0.75 - data.qpos[2]) / 0.15, 0.0, 1.0)
            downward_risk = jp.clip((-pelvis_vz - 0.25) / 0.75, 0.0, 1.0)
            rewards["lin_vel_z"] = height_risk * jp.square(downward_risk)
        if self._h1_m2:
            rewards["feet_clearance"] = self._cost_h1_m2_phase_foot_crossing(
                data, info, contact
            )
        if self._h3_m1:
            g = h3_stand_factor(info['m470_h3_n'], done)
            rewards['stand_still'] = g * rewards['stand_still']
        if getattr(self,'_h4m6_arm',None) is not None:
            if tuple(rewards) != H4M6_REWARD_NAMES:
                raise RuntimeError('M470_H4M6_REWARD_KEYS')
            raw = rewards['stand_still']
            eligible,factor,shadow,delivered = h4m6_reward(
                jp,raw,info['h4m6_pre_eligible']>0,info['h4m6_pre_k'],
                done>0,(info['h4m6_tech_treatment']>0 if self._h4m6_tech_capture else self._h4m6_arm=='rss'))
            base = sum(v*self._config.reward_config.scales[k] for k,v in rewards.items())*self.dt
            rewards['stand_still'] = delivered
            if self._h4m6_tech_capture:
                info['h4m6_raw_rewards'] = jp.stack([rewards[k] for k in H4M6_REWARD_NAMES])
            expected = sum(v*self._config.reward_config.scales[k] for k,v in rewards.items())*self.dt
            info.update(h4m6_eligible=eligible.astype(jp.int32),
                h4m6_factor=factor,h4m6_raw_stand=raw,h4m6_shadow_stand=shadow,
                h4m6_actual_stand=delivered,h4m6_shadow_relief=(raw-shadow)*self.dt,
                h4m6_delivered_relief=(raw-delivered)*self.dt,
                h4m6_base_reward=base,h4m6_expected_reward=expected)
        return rewards

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
            m470_m1_weighted=jp.asarray(0.0, dtype=jp.float32),
            m470_m2_weighted=jp.asarray(0.0, dtype=jp.float32),
        )
        if self._h3_m1:
            info.update({
                k: jp.asarray(0, dtype=jp.int32)
                for k in ('m470_h3_n', 'm470_h3_prior', 'm470_h3_boundary',
                          'm470_h3_stand', 'm470_h3_done')
            })
            info.update({
                k: jp.asarray(0.0, dtype=jp.float32)
                for k in ('m470_h3_base', 'm470_h3_weighted',
                          'm470_h3_reward_sum')
            })
        if self._h4_arm is not None:
            info.update({k: zeros for k in H4_FIELDS})
            info.update(h4_actual_dx=jp.asarray(0.,jp.float32),
                        h4_actual_dy=jp.asarray(0.,jp.float32),
                        h4_first_age=jp.asarray(-1, jp.int32),
                        h4_first_mag=jp.asarray(0., jp.float32),
                        h4_first_motion=jp.asarray(-1, jp.int32),
                        h4_tail_motion=jp.asarray(-1, jp.int32))
        if getattr(self, '_h4cc_arm', None) is not None:
            info.update({'h4cc_'+k:zeros for k in (*H4CC_FIELDS,*H4CC_STATE)})
            info.update(h4cc_held_command=jp.zeros_like(info['command']),
                        h4cc_pending_command=jp.zeros_like(info['command']))
        if getattr(self,'_h4m6_arm',None) is not None:
            info.update({'h4m6_'+k:zeros for k in H4M6_STATE+H4M6_INT_FIELDS})
            info.update({'h4m6_'+k:jp.asarray(1. if k=='factor' else 0.,jp.float32)
                         for k in H4M6_FLOAT_FIELDS})
            info.update(h4m6_pre_eligible=zeros,h4m6_pre_k=zeros)
            if self._h4m6_tech_capture:
                info['h4m6_raw_rewards'] = jp.zeros((24,),jp.float32)
                info['h4m6_tech_treatment'] = zeros
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

        if self._h4_arm is not None:
            # Preserve original scheduler key split, including control draws.
            h4_class = jp.where(disabled, 0, 1)
            if self._h4_arm == 'candidate':
                pick = jax.random.uniform(jax.random.fold_in(k_start, 41), ())
                h4_class = jp.where(disabled, 0,
                    jp.where(pick < .5, 1, jp.where(pick < .7, 2, 3)))
            h4_class = jp.where(boundary, h4_class, info['h4_class'])
            mature = h4_class >= 2
            if self._h4_arm == 'candidate':
                late = jax.random.randint(jax.random.fold_in(k_time, 42), (), 240, 301)
                lateral = jax.random.randint(jax.random.fold_in(k_dir, 43), (), 2, 4)
                next_step = jp.where(boundary & mature, late, next_step)
                direction = jp.where(boundary & mature, lateral, direction)
            pre_motion = jp.argmin(jp.sum(jp.abs(COMMANDS - info['command']), axis=-1))
            actor_motion = jp.argmin(jp.sum(jp.abs(COMMANDS - state.obs['state'][9:12]), axis=-1))
            pre_command = info['command']
            pre_source_step = info['step']

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
        if self._h4_arm == 'candidate':
            # Mature single can never repeat, including deadline fallback.
            event = event & ((h4_class != 2) | (count == 0))
        # Magnitude is sampled only logically at event time. JAX still computes
        # branches without Python dynamic condition (fully JIT-compatible).
        magnitude = jax.random.uniform(k_mag, (), minval=self._m463_min_mag,
                                       maxval=self._m463_max_mag)
        if self._h4_arm == 'candidate':
            lo = jp.where(count == 0, .45, .30)
            hi = jp.where(count == 0, .55, .40)
            mature_mag = jax.random.uniform(jax.random.fold_in(k_mag, 44), (),
                                           minval=lo, maxval=hi)
            magnitude = jp.where(mature, mature_mag, magnitude)
        body_delta = PUSH_UNIT_BODY[direction] * magnitude
        yaw = heading_from_wxyz(state.data.qpos[3:7])
        c, s = jp.cos(yaw), jp.sin(yaw)
        world_delta = jp.asarray([
            c * body_delta[0] - s * body_delta[1],
            s * body_delta[0] + c * body_delta[1],
        ]) * event.astype(jp.float32)
        if self._h3_m1:
            h3_prior = info['m463_recent']
            h3_n = h3_countdown(event, h3_prior, boundary)
            h3_pre_command = info['command']
            h3_pre_stand = (jp.linalg.norm(h3_pre_command) < 0.01)
            info['m470_h3_n'] = h3_n
        if getattr(self,'_h4m6_arm',None) is not None:
            m6 = h4m6_pre(jp,info,event=event,repeat=event & (count>0),cls=h4_class,
                first_age=jp.where(boundary,-1,info['h4_first_age']),
                first_mag=jp.where(boundary,0.,info['h4_first_mag']),
                gap=age-last_event_age,magnitude=magnitude,
                first_motion=jp.where(boundary,-1,info['h4_first_motion']),
                pre_motion=pre_motion,actor_motion=actor_motion,direction=direction,
                support_side=support_side,boundary=boundary,age=age,clock=pre_source_step,
                wrapper_steps=info['steps'],command=pre_command,
                actor_command=state.obs['state'][9:12],
                critic_command=state.obs['privileged_state'][9:12])
            info['h4m6_pre_eligible'] = m6['eligible_pre'].astype(jp.int32)
            info['h4m6_pre_k'] = m6['k'].astype(jp.int32)
        updated_qvel = state.data.qvel.at[:2].add(world_delta)
        if self._h4_arm is not None:
            h4_actual_delta = updated_qvel[:2] - state.data.qvel[:2]
            injection_bad = (~jp.isfinite(updated_qvel).all() |
                (jp.max(jp.abs((updated_qvel[:2] - state.data.qvel[:2]) -
                               world_delta)) > 2e-6))
        state = state.replace(data=state.data.replace(qvel=updated_qvel), info=info)
        # The pinned G1 source runs the same PD, physics, termination,
        # observations, and RNG progression. Only opt-in reward hooks differ.
        # Built-in random push is OFF.
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
        if self._h4_arm == 'candidate':
            short_gap = jax.random.randint(jax.random.fold_in(k_gap, 45), (), 30, 33)
            gap = jp.where(mature, short_gap, gap)
            repeat_allowed = jp.where(mature, h4_class == 3, repeat_allowed)
        new_next = jp.where(event & repeat_allowed, age + gap,
                            jp.where(event, jp.asarray(2**30, jp.int32), next_step))
        # If no repeat was sampled, do not allow a second event.
        new_count = jp.where(event & (~repeat_allowed), max_count, new_count)
        if getattr(self, '_h4cc_arm', None) is not None:
            # Exact existing mature-repeat predicate, evaluated before post-motion.
            cc_first_age = jp.where(boundary, -1, info['h4_first_age'])
            cc_first_mag = jp.where(boundary, 0., info['h4_first_mag'])
            cc_first_motion = jp.where(boundary, -1, info['h4_first_motion'])
            cc_mature = (repeat & (h4_class == 3) &
                (cc_first_age >= 240) & (cc_first_age <= 308) &
                (cc_first_mag >= .45) & (cc_first_mag <= .55) &
                (actual_gap >= 30) & (actual_gap <= 40) &
                (magnitude >= .30) & (magnitude <= .40) &
                (cc_first_motion == pre_motion) & (actor_motion == pre_motion))
            cc_side = jp.where(cc_mature & (pre_motion == 0) & side_matched,side_target,0)
            command, ao, co, cc = h4cc_transition(
                jp,info,treatment=self._h4cc_arm=='ccrc',event=event,
                mature_repeat=cc_mature,side=cc_side,motion=pre_motion,age=age,
                boundary=boundary,terminal=after.done>0,refresh=pre_source_step>=500,
                pre=pre_command,actor=state.obs['state'][9:12],
                critic=state.obs['privileged_state'][9:12],native=out['command'],
                next_actor=after.obs['state'][9:12],
                next_critic=after.obs['privileged_state'][9:12])
            out.update(cc)
            if self._h4cc_arm=='ccrc':
                out['command'] = command
                obs = dict(after.obs)
                obs['state'] = obs['state'].at[9:12].set(ao)
                obs['privileged_state'] = obs['privileged_state'].at[9:12].set(co)
                after = after.replace(obs=obs)
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
            # This is source-scaled but pre-dt, from actual training steps.
            m470_m1_weighted=after.metrics['reward/lin_vel_z'],
            m470_m2_weighted=after.metrics['reward/feet_clearance'],
        )
        if self._h3_m1:
            base = -self._cost_stand_still(h3_pre_command,
                                          after.data.qpos[7:])
            total = sum(after.metrics['reward/' + k]
                        for k in self._config.reward_config.scales) * self.dt
            out.update(
                m470_h3_n=h3_n,
                m470_h3_prior=h3_prior.astype(jp.int32),
                m470_h3_boundary=jp.asarray(boundary, dtype=jp.int32),
                m470_h3_stand=h3_pre_stand.astype(jp.int32),
                m470_h3_done=(after.done > 0).astype(jp.int32),
                m470_h3_base=base.astype(jp.float32),
                m470_h3_weighted=after.metrics['reward/stand_still'].astype(jp.float32),
                m470_h3_reward_sum=jp.asarray(total, dtype=jp.float32),
            )
        if self._h4_arm is not None:
            # Actual event and prior-event tail, before AutoReset alters data/obs.
            # Stop a tail on repeat; the repeat action itself is not counted.
            first = event & (~repeat)
            terminal = after.done > 0
            old_kind = info['h4_tail_kind']
            old_cell = info['h4_tail_cell']
            old_age = info['h4_tail_age']
            prior = (old_kind > 0) & (~boundary)
            tick = prior & (~event)
            tick_age = jp.where(tick, old_age + 1, 0)
            same = ((info['h4_tail_same'] > 0) &
                    (pre_motion == info['h4_tail_motion']) &
                    (actor_motion == info['h4_tail_motion']))
            complete = tick & (tick_age == 250)
            # A terminal at tick 250 counts executed but not survived.
            close_terminal = tick & terminal
            close_complete = complete & (~terminal)
            keep = tick & (~terminal) & (~complete)
            first_age = jp.where(boundary, -1, info['h4_first_age'])
            first_mag = jp.where(boundary, 0., info['h4_first_mag'])
            first_motion = jp.where(boundary, -1, info['h4_first_motion'])
            assign_motion = jp.where(boundary, pre_motion, info['h4_assign_motion'])
            mature_repeat = (repeat & (h4_class == 3) &
                (first_age >= 240) & (first_age <= 308) &
                (first_mag >= .45) & (first_mag <= .55) &
                (actual_gap >= 30) & (actual_gap <= 40) &
                (magnitude >= .30) & (magnitude <= .40) &
                (first_motion == pre_motion) & (actor_motion == pre_motion))
            qualify = mature_repeat & (pre_motion == 0) & side_matched
            cell = jp.where(qualify, side_target, 0)
            # Track assigned first/repeat targets, including unreached terminal.
            fp = jp.where(boundary, ~disabled, info['h4_first_pending'] > 0)
            rp = (~boundary) & (info['h4_repeat_pending'] > 0)
            f_due = fp & (age == next_step)
            r_due = rp & (age == next_step)
            next_fp = fp & (~first) & (~terminal)
            next_rp = ((rp & (~repeat)) | (first & repeat_allowed)) & (~terminal)
            exact_pre = jp.all(COMMANDS[pre_motion] == pre_command)
            exact_actor = jp.all(COMMANDS[actor_motion] == state.obs['state'][9:12])
            bad = (injection_bad | (~exact_pre) | (~exact_actor) |
                   (~jp.isfinite(after.data.qpos).all()) |
                   (~jp.isfinite(after.data.qvel).all()) |
                   (~jp.isfinite(after.reward)) |
                   (event & disabled) |
                   (event & (~jp.isfinite(magnitude))) |
                   (repeat & (~rp)) | (first & (~fp)) |
                   ((h4_class == 2) & repeat) |
                   (first & mature & ((age < 240) | (age > 308) |
                                      (magnitude < .45) | (magnitude > .55))) |
                   (repeat & mature & ((magnitude < .30) | (magnitude > .40))) |
                   ((h4_class == 3) & repeat & ((actual_gap < 30) | (actual_gap > 40))))
            vals = dict(
                h4_class=h4_class, h4_boundary=boundary, h4_age=age,
                h4_assign_motion=assign_motion,
                h4_repeat_origin_motion=jp.where(first,pre_motion,first_motion),
                h4_class_event=jp.where(event,h4_class,-1),
                h4_source_terminal=terminal,
                h4_command_mismatch=actor_motion != pre_motion,
                h4_target=next_step, h4_motion=pre_motion,
                h4_actor_motion=actor_motion, h4_refresh=pre_source_step >= 500,
                h4_changed=jp.any(pre_command != out['command']),
                h4_assigned=boundary & (~disabled),
                h4_first_pending=next_fp, h4_repeat_pending=next_rp,
                h4_first_due=f_due, h4_repeat_due=r_due,
                h4_first_closed=fp & terminal & (~first),
                h4_repeat_closed=terminal & ((rp & (~repeat)) | (first & repeat_allowed)),
                h4_event_qual=cell, h4_mature_repeat=mature_repeat,
                h4_repeat_assigned=first & repeat_allowed, h4_event_terminal=event & terminal,
                h4_tail_kind=jp.where(event & (~terminal), jp.where(repeat,2,1),
                                     jp.where(keep,old_kind,0)),
                h4_tail_cell=jp.where(event & (~terminal),cell,jp.where(keep,old_cell,0)),
                h4_tail_age=jp.where(keep,tick_age,0),
                h4_tail_same=jp.where(event & (~terminal),actor_motion==pre_motion,keep & same),
                h4_tick_kind=jp.where(tick,old_kind,0),
                h4_tick_cell=jp.where(tick,old_cell,0),
                h4_tick_age=tick_age, h4_tick_same=tick & same,
                h4_tick_survived=tick & (~terminal),
                h4_close_repeat=prior & event,
                h4_close_terminal=close_terminal,
                h4_close_complete=close_complete,
                h4_close_boundary=boundary & (old_kind>0),
                h4_bad=bad)
            if getattr(self, '_h4cc_arm', None) is not None:
                vals['h4_bad'] = vals['h4_bad'] | (out['h4cc_bad'] > 0) | (cc_mature != mature_repeat)
            out.update({k: jp.asarray(v, jp.int32) for k,v in vals.items()})
            out.update(
                h4_actual_dx=h4_actual_delta[0], h4_actual_dy=h4_actual_delta[1],
                h4_first_age=jp.where(first,age,first_age),
                h4_first_mag=jp.where(first,magnitude,first_mag),
                h4_first_motion=jp.where(first,pre_motion,first_motion),
                h4_tail_motion=jp.where(event,pre_motion,info['h4_tail_motion']))
        if getattr(self,'_h4m6_arm',None) is not None:
            m6out = h4m6_post(jp,info,m6,done=after.done>0,
                eligible=out['h4m6_eligible']>0,raw=out['h4m6_raw_stand'],
                shadow=out['h4m6_shadow_stand'])
            finite = jp.all(jp.isfinite(jp.stack([out['h4m6_'+k] for k in H4M6_FLOAT_FIELDS])))
            bad = ((m6out['h4m6_bad']>0) | (~finite) |
                   (m6['mature'] != (out['h4_mature_repeat']>0)) |
                   (m6['native_qual'] != (out['h4_event_qual']>0)) |
                   (out['h4m6_expected_reward'] != after.reward) |
                   (after.metrics['reward/stand_still'] != -out['h4m6_actual_stand']) |
                   ((out['h4m6_eligible']==0) & (out['h4m6_delivered_relief']!=0)) |
                   ((~(out['h4m6_tech_treatment']>0) if self._h4m6_tech_capture else self._h4m6_arm=='ref')
                     & (out['h4m6_delivered_relief']!=0)))
            if self._h4m6_tech_capture:
                bad = bad | ((out['h4m6_tech_treatment']!=0)&(out['h4m6_tech_treatment']!=1))
            m6out['h4m6_bad'] = bad.astype(jp.int32)
            out.update(m6out)
            out['h4_bad'] = jp.maximum(out['h4_bad'],m6out['h4m6_bad'])
        return after.replace(info=out)
