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
H4M6_BOOL_REWARD_NAMES = ('termination','collision')
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


# H4-M8-CAM-V1: common read-only telemetry. No native reward hook.
H4M8_STATE = ('active','age','side','consumed','positive')
H4M8_INT_FIELDS = ('qualified','event_terminal','start','event_side','eligible',
    'tick_age','tick_side','weight8','positive_start','completed','close_terminal',
    'close_truncation','close_boundary','close_command','close_event',
    'opportunity_side','attempt_side','bad')
H4M8_FLOAT_FIELDS = ('mass','h2','cost','shadow')
H4M8_FIELDS = H4M8_INT_FIELDS + H4M8_FLOAT_FIELDS


def h4m8_momentum(xp, mass, inertia, xipos, ximat, cvel, subtree_com, rootid):
    """Same last-forward epoch; actual per-world model, no extra forward."""
    arrays = tuple(xp.asarray(x) for x in (mass,inertia,xipos,ximat,cvel,subtree_com))
    if tuple(x.shape for x in arrays) != ((31,),(31,3),(31,3),(31,3,3),(31,6),(31,3)):
        raise RuntimeError('M470_H4M8_MOMENTUM_SHAPE')
    if any(x.dtype != xp.float32 for x in arrays) or rootid.shape != (31,):
        raise RuntimeError('M470_H4M8_MOMENTUM_DTYPE')
    mass,inertia,xipos,ximat,cvel,subtree_com = arrays
    m,I,r,R,cv = mass[1:],inertia[1:],xipos[1:],ximat[1:],cvel[1:]
    M = xp.sum(m)
    omega = cv[:,:3]
    v = cv[:,3:] + xp.cross(omega,r-subtree_com[1])
    center = xp.sum(m[:,None]*r,axis=0)/M
    vc = xp.sum(m[:,None]*v,axis=0)/M
    iw = (R*I[:,None,:]) @ xp.swapaxes(R,1,2)
    H = xp.sum((iw@omega[:,:,None])[:,:,0] +
               xp.cross(r-center,m[:,None]*(v-vc)),axis=0)
    h2 = xp.sum(H*H)
    href = M*xp.asarray(.793,xp.float32)*xp.asarray(.35,xp.float32)
    cost = h2/(href*href+h2)
    good = (xp.all(rootid[1:]==1) & (rootid[0]==0) & (mass[0]==0) &
            xp.all(m>0) & xp.all(I>0) & (M>0) & (h2>=0) & (cost>=0) & (cost<=1))
    for x in (*arrays,H,center,vc,M,h2,cost):
        good = good & xp.all(xp.isfinite(x))
    return M,h2,cost,good


def h4m8_pre(xp, old, *, event, repeat, cls, first_age, first_mag, gap,
              magnitude, first_motion, pre_motion, actor_motion, direction,
              support_side, boundary, age, clock, wrapper_steps,
              command, actor_command, critic_command):
    active = old['h4m8_active'] > 0
    old_age = old['h4m8_age']
    consumed = (old['h4m8_consumed'] > 0) & (~boundary)
    side = xp.where(direction==2,1,xp.where(direction==3,2,0))
    mature = (event & repeat & (cls==3) & (first_age>=240) & (first_age<=308) &
              (first_mag>=.45) & (first_mag<=.55) & (gap>=30) & (gap<=40) &
              (magnitude>=.30) & (magnitude<=.40) & (first_motion==pre_motion) &
              (actor_motion==pre_motion))
    native_qual = mature & (pre_motion==0) & (side>0) & (support_side==side)
    coherent = xp.all(command==actor_command) & xp.all(command==critic_command)
    zero = xp.all(command==0) & coherent
    clock_ok = (clock==age) & (wrapper_steps==age) & (clock<500)
    trunc = wrapper_steps+1>=1000
    qualified = native_qual & zero & clock_ok & (~boundary) & (~trunc) & (~active) & (~consumed)
    valid_old = (old_age>=0) & (old_age<50) & ((old['h4m8_side']==1)|(old['h4m8_side']==2))
    cb = active & boundary
    ce = active & (~boundary) & event
    ct = active & (~boundary) & (~event) & trunc
    cc = active & (~boundary) & (~event) & (~trunc) & ((~zero)|(clock>=500))
    eligible = active & valid_old & (~boundary) & (~event) & (~trunc) & zero & clock_ok
    bad = (native_qual & (~qualified)) | (active & (~valid_old)) | cb | ce
    bad = bad | (active & (~coherent)) | (active & (~boundary) & (~trunc) &
              (clock<500) & ((clock!=age)|(wrapper_steps!=age)))
    bad = bad | ((~active) & ((old_age!=0)|(old['h4m8_side']!=0)|(old['h4m8_positive']!=0)))
    bad = bad | (active & (~consumed) & (~boundary))
    for key in ('active','consumed','positive'):
        x=old['h4m8_'+key]
        bad=bad | ((x!=0)&(x!=1))
    return dict(active=active,consumed=consumed,k=old_age+1,side=side,
        mature=mature,native_qual=native_qual,qualified=qualified,eligible=eligible,
        close_boundary=cb,close_event=ce,close_truncation=ct,close_command=cc,
        opportunity_side=xp.where(boundary & (cls==3) & (pre_motion==0),side,0),
        attempt_side=xp.where(event & repeat & (cls==3) & (first_motion==0) & (pre_motion==0),side,0),bad=bad)


def h4m8_post(xp, old, pre, *, done, post_zero, post_coherent, native_mature,
               native_side, mass, h2, cost, momentum_good):
    # Source terminal is known before EpisodeWrapper and AutoReset.
    qualified=pre['qualified']
    prior_close=pre['close_boundary']|pre['close_event']|pre['close_truncation']|pre['close_command']
    close_terminal=pre['active'] & (~prior_close) & done
    late_command=pre['eligible'] & (~done) & (~(post_zero & post_coherent))
    close_command=pre['close_command']|late_command
    start=qualified & (~done) & post_zero & post_coherent
    eligible=pre['eligible'] & (~done) & post_zero & post_coherent
    k=pre['k']
    weight8=xp.where(eligible,xp.minimum(8,xp.minimum(xp.maximum(0,k-7),xp.maximum(0,51-k))),0)
    shadow=xp.where(weight8>0,xp.asarray(.25,xp.float32)*xp.asarray(.02,xp.float32)*
                    (weight8.astype(xp.float32)/xp.asarray(8.,xp.float32))*cost,
                    xp.asarray(0.,xp.float32))
    complete=eligible & (k==50)
    keep=start | (eligible & (~complete))
    positive=eligible & (shadow>0)
    old_positive=old['h4m8_positive']>0
    closed=prior_close|close_terminal|late_command|complete
    bad=pre['bad'] | (~momentum_good) | (~xp.isfinite(shadow)) | (shadow<0) | (shadow>xp.asarray(.005,xp.float32))
    bad=bad | (pre['mature']!=native_mature) | (xp.where(pre['native_qual'],pre['side'],0)!=native_side)
    bad=bad | (qualified & (~done) & (~start)) | (pre['eligible'] & (~done) & (~post_coherent))
    bad=bad | (pre['active'] & (~(closed|eligible))) | (closed & keep)
    vals=dict(qualified=qualified,event_terminal=qualified & done,start=start,
        event_side=xp.where(qualified,pre['side'],0),eligible=eligible,
        tick_age=xp.where(eligible,k,0),tick_side=xp.where(pre['active'],old['h4m8_side'],0),
        weight8=weight8,positive_start=positive & (~old_positive),completed=complete,
        close_terminal=close_terminal,close_truncation=pre['close_truncation'],
        close_boundary=pre['close_boundary'],close_command=close_command,close_event=pre['close_event'],
        opportunity_side=pre['opportunity_side'],attempt_side=pre['attempt_side'],bad=bad,
        active=keep,age=xp.where(keep & (~start),k,0),side=xp.where(keep,xp.where(start,pre['side'],old['h4m8_side']),0),
        consumed=pre['consumed']|qualified,positive=keep & (~start) & (old_positive|positive))
    result={'h4m8_'+key:xp.asarray(value,xp.int32) for key,value in vals.items()}
    result.update(h4m8_mass=xp.asarray(mass,xp.float32),h4m8_h2=xp.asarray(h2,xp.float32),
                  h4m8_cost=xp.asarray(cost,xp.float32),h4m8_shadow=xp.asarray(shadow,xp.float32))
    return result


def heading_from_wxyz(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def h3_countdown(event, prior_recent, boundary):
    prior = jp.where(boundary, 0, prior_recent)
    return jp.where(event, 100, jp.maximum(prior - 1, 0)).astype(jp.int32)


def h3_stand_factor(countdown, done):
    n = jp.asarray(countdown, dtype=jp.float32)
    return jp.where(done, 1.0, 1.0 - 0.5 * n / 100.0)



# H4R11_CORE_BEGIN
# Held R11-NSTR-V1. No native import or extra forward in these helpers.
R11_SCALARS = ('bad','open','open_side','close','touch','transfer','rejected','side',
               'before_active','after_active','before_used','after_used',
               'native','score','desired','actual')
R11_FIELDS = tuple('r11_'+k for k in R11_SCALARS)


def r11_init(xp):
    out={k:xp.asarray(0,xp.int32) for k in R11_FIELDS[:-4]}
    out.update({k:xp.asarray(0.,xp.float32) for k in R11_FIELDS[-4:]})
    out.update(r11_stage=xp.asarray(0,xp.int32),r11_age=xp.asarray(0,xp.int32),
        r11_role=xp.asarray(0,xp.int32),r11_used=xp.asarray(0,xp.int32),
        r11_touch_n=xp.asarray(0,xp.int32),r11_transfer_n=xp.asarray(0,xp.int32),
        r11_anchor=xp.zeros((8,),xp.float32))
    return out


def r11_rect(xp, pos, rot):
    # Enclosing projected rectangle of the native foot box, not a CoP hull.
    a=rot[:2,0]
    n=xp.linalg.norm(a)
    a=a/xp.maximum(n,xp.asarray(1e-6,xp.float32))
    b=xp.stack((-a[1],a[0]))
    axes=xp.stack((a,b))
    ext=xp.abs(axes@rot[:2,:])@xp.asarray([.09,.03,.008],xp.float32)
    rect=xp.concatenate((pos[:2],axes.reshape((4,)),ext))
    good=(n>1e-6)&xp.all(xp.isfinite(rect))&xp.all(ext>0)
    good=good&xp.all(xp.abs(rot.T@rot-xp.eye(3,dtype=xp.float32))<1e-4)
    good=good&(xp.linalg.det(rot)>0)
    return rect,good


def r11_distance(xp, xi, rect):
    return xp.linalg.norm(xp.maximum(xp.abs(rect[2:6].reshape((2,2))@(xi-rect[:2]))-rect[6:],0.))


def r11_quality(xp, old_d, new_d, speed):
    return xp.clip((old_d-new_d)/.10,0.,1.)/(1.+(new_d/.10)**2+(speed/.35)**2)


def r11_sample(xp, m, d, feet, floor):
    # Actual per-world randomized mass; all COM, geometry and loads from
    # the retained last-forward/solve epoch. qpos separately for position gate.
    mass=xp.asarray(m.body_mass)
    if (mass.shape!=(31,) or d.xipos.shape!=(31,3) or d.cvel.shape!=(31,6)
            or d.subtree_com.shape!=(31,3) or d.qpos.shape!=(36,)
            or d.ctrl.shape!=(29,) or d.actuator_force.shape!=(29,)
            or int(m.opt.cone)!=0):
        raise RuntimeError('M470_R11_NATIVE_SCHEMA')
    M=xp.sum(mass[1:])
    den=xp.maximum(M,xp.asarray(1e-6,xp.float32))
    vel=d.cvel[1:,3:]+xp.cross(d.cvel[1:,:3],d.xipos[1:]-d.subtree_com[1])
    c=xp.sum(mass[1:,None]*d.xipos[1:],axis=0)/den
    v=xp.sum(mass[1:,None]*vel,axis=0)/den
    g=xp.asarray(m.opt.gravity)
    xi=c[:2]+v[:2]/xp.sqrt(xp.asarray(9.81/.70,xp.float32))
    bw=xp.maximum(M*xp.abs(g[2]),xp.asarray(1e-6,xp.float32))
    good=(M>0)&(mass[0]==0)&xp.all(mass[1:]>0)&xp.all(xp.asarray(m.body_rootid)[1:]==1)
    good=good&(g[0]==0)&(g[1]==0)&(g[2]<0)
    for x in (mass,c,v,xi,g):
        good=good&xp.all(xp.isfinite(x))
    co=d.contact
    geom=xp.asarray(co.geom); adr=xp.asarray(co.efc_address); dim=xp.asarray(co.dim)
    n=geom.shape[0]
    if (geom.shape!=(n,2) or adr.shape!=(n,) or dim.shape!=(n,)
            or co.dist.shape!=(n,) or co.includemargin.shape!=(n,)
            or d.efc_force.ndim!=1 or d.efc_force.shape[0]<4):
        raise RuntimeError('M470_R11_CONTACT_SCHEMA')
    frame=co.frame.reshape((n,3,3))
    # Safe gather even for inactive slots. A live invalid address fails closed.
    idx=xp.clip(adr,0,d.efc_force.shape[0]-4)
    edges=d.efc_force[idx[:,None]+xp.arange(4)]
    normal=xp.sum(edges,axis=1)
    loads=[]; rects=[]
    for foot in feet:
        fwd=(geom[:,0]==floor)&(geom[:,1]==foot)
        rev=(geom[:,1]==floor)&(geom[:,0]==foot)
        pair=fwd|rev
        active=pair&(adr>=0)&(co.dist<=co.includemargin)
        up=xp.where(fwd,frame[:,0,2],-frame[:,0,2])
        valid=(dim==3)&(adr<=d.efc_force.shape[0]-4)&(up>.99)&(normal>=0)
        valid=valid&xp.isfinite(normal)&xp.all(xp.isfinite(edges)&(edges>=0),axis=1)
        valid=valid&xp.all(xp.isfinite(frame),axis=(1,2))
        valid=valid&xp.all(xp.abs(frame@xp.swapaxes(frame,1,2)-xp.eye(3,dtype=xp.float32))<1e-4,axis=(1,2))
        good=good&xp.all((~pair)|(xp.isfinite(co.dist)&xp.isfinite(co.includemargin)))
        good=good&xp.all((~active)|valid)
        loads.append(xp.sum(xp.where(active&valid,normal,0.))/bw)
        rect,rect_good=r11_rect(xp,d.geom_xpos[foot],d.geom_xmat[foot].reshape((3,3)))
        rects.append(rect)
        good=good&rect_good&xp.all(xp.asarray(m.geom_size)[foot]==xp.asarray([.09,.03,.008],xp.float32))
    loads=xp.stack(loads); rects=xp.stack(rects)
    good=good&xp.all(xp.isfinite(loads))&xp.all(loads>=0)
    good=good&xp.all(xp.asarray(m.actuator_trnid)[:,0]==xp.arange(1,30))
    good=good&xp.all(xp.asarray(m.jnt_qposadr)[1:]==xp.arange(7,36))
    safe=xp.asarray(True)
    for values,ranges,limited in (
            (d.qpos[7:],xp.asarray(m.jnt_range)[1:],xp.asarray(m.jnt_limited)[1:]),
            (d.ctrl,xp.asarray(m.actuator_ctrlrange),xp.asarray(m.actuator_ctrllimited)),
            (d.actuator_force,xp.asarray(m.actuator_forcerange),xp.asarray(m.actuator_forcelimited))):
        if values.shape!=(29,) or ranges.shape!=(29,2) or limited.shape!=(29,):
            raise RuntimeError('M470_R11_RANGE_SCHEMA')
        good=good&xp.all(xp.isfinite(values))&xp.all(xp.isfinite(ranges))
        good=good&xp.all((limited==0)|(limited==1))&xp.all((limited==0)|(ranges[:,0]<=ranges[:,1]))
        safe=safe&xp.all((limited==0)|((values>=ranges[:,0])&(values<=ranges[:,1])))
    return loads,rects,xi,xp.linalg.norm(v[:2]),safe,good


def r11_transition(xp, old, pre, post, *, event, boundary, coherent, terminal, native):
    # Every condition is causal; no future labels or random draws.
    pl,pr,_,_,_,pg=pre
    nl,nr,xi,speed,safe,ng=post
    old_stage=old['r11_stage']; was=old_stage>0
    role=old['r11_role']; other=1-role
    age=old['r11_age']+1; used=old['r11_used']
    tn=old['r11_touch_n']; fn=old['r11_transfer_n']
    anchor=old['r11_anchor']
    finite=xp.isfinite(native)
    valid=pg&ng&finite
    bad=~valid
    allowed=coherent&(~boundary)&(~terminal)&valid
    active=was&(~event)&allowed&(age<=100)
    pre_sole=(pl>=.05)&(pl[::-1]<.05)
    post_sole=(nl>=.05)&(nl[::-1]<.05)
    start_role=xp.argmax(pre_sole.astype(xp.int32)).astype(xp.int32)
    start=event&allowed&xp.any(pre_sole&post_sole)&(r11_distance(xp,xi,nr[start_role])>.03)
    # Stage 1: retain old support through touchdown onset and confirmation.
    waiting=active&(old_stage==1)
    new_loaded=nl[other]>=.05
    retention=(nl[role]>=.05)
    # After first onset old support need not remain for later confirmation.
    keep_wait=waiting&(retention|((tn>0)&new_loaded))
    onset=keep_wait&new_loaded&(tn==0)&retention
    anchor=xp.where(onset,pr[role],anchor)
    next_tn=xp.where(keep_wait&new_loaded,tn+1,0)
    touch=keep_wait&(next_tn==3)
    # Stage 2: three separate subsequent samples of majority/new-only load.
    transfer_phase=active&(old_stage==2)&new_loaded
    qualifies=(nl[other]>=.50)&(nl[role]<.05)
    next_fn=xp.where(transfer_phase&qualifies,fn+1,0)
    transfer=transfer_phase&(next_fn==3)&(used<2)
    d_old=r11_distance(xp,xi,anchor)
    d_new=r11_distance(xp,xi,nr[other])
    quality=r11_quality(xp,d_old,d_new,speed)
    reject=transfer&(~safe)
    score=xp.where(transfer&safe,quality,0.).astype(xp.float32)
    desired=(xp.asarray(.04,xp.float32)*score).astype(xp.float32)
    rewarded=xp.where(desired>0,native+desired,native)
    actual=(rewarded-native).astype(xp.float32)
    bad=bad|(~xp.isfinite(rewarded))|(~xp.isfinite(actual))
    # Consume a credit even when quality is zero, never rearm after rejection.
    used_next=used+transfer.astype(xp.int32)
    rearm=transfer&safe&(used_next<2)&(d_new>.03)&(age<100)
    stay=((keep_wait&(~touch))|touch|(transfer_phase&(~transfer)))&(age<100)
    keep=stay|rearm
    stage=xp.where(rearm,1,xp.where(touch|transfer_phase,2,1))
    next_stage=xp.where(start,1,xp.where(keep,stage,0))
    next_role=xp.where(start,start_role,xp.where(rearm,other,role))
    next_used=xp.where(start,0,xp.where(keep,used_next,0))
    after_active=next_stage>0
    closed=was&(~keep)
    o=r11_init(xp)
    ints=dict(stage=next_stage,age=xp.where(start,0,xp.where(keep,age,0)),
        role=xp.where(after_active,next_role,0),used=next_used,
        touch_n=xp.where(keep&(~touch)&(~rearm)&(old_stage==1),next_tn,0),
        transfer_n=xp.where(keep&(~rearm)&(old_stage==2),next_fn,0),
        bad=bad,open=start,open_side=xp.where(start,start_role+1,0),close=closed,touch=touch,transfer=transfer,rejected=reject,
        side=xp.where(transfer,other+1,0),before_active=was,after_active=after_active,
        before_used=used,after_used=next_used)
    o.update({'r11_'+k:xp.asarray(z,xp.int32) for k,z in ints.items()})
    o['r11_anchor']=xp.where(after_active&(~start)&(~rearm),anchor,xp.zeros_like(anchor))
    o.update(r11_native=xp.asarray(native,xp.float32),r11_score=score,
             r11_desired=desired,r11_actual=actual)
    return o,rewarded

# H4R11_CORE_END

# H4R12_CORE_BEGIN
R12_INTS=('bad','before_active','after_active','before_age','after_age','before_side','after_side',
      'before_kind','after_kind','open','close','close_reason','scored','support_mode',
      'coherent','terminal','timeout','event_side','joint_excess','ctrl_excess','force_excess','foot_foot')
R12_FLOATS=('native','cost','desired','delivered','delta')
R12_FIELDS=tuple('r12_'+k for k in R12_INTS+R12_FLOATS)
R12_CARRY=('active','age','side','kind')
R12_SOURCE_FLAGS=('m463_event','m463_repeat','m463_no_push','h4_boundary','h4_class','h4_motion',
              'h4_actor_motion','m463_direction','h4_refresh','h4_changed','h4_bad','h4_source_terminal','truncation')


R12_PAIRS = tuple((i,j) for i in range(8) for j in range(i+1,8))


def r12_hull_distance(xp, xi, rects):
    f=xp.float32
    signs=xp.asarray([[-1,-1],[-1,1],[1,-1],[1,1]],f)
    corners=(rects[:,None,:2]+(signs[None,:,:]*rects[:,None,6:]) @ rects[:,2:6].reshape((2,2,2))).reshape((8,2))
    pairs=xp.asarray(R12_PAIRS,xp.int32)
    a,b=corners[pairs[:,0]],corners[pairs[:,1]]
    e=b-a
    length2=xp.sum(e*e,axis=1)
    tol=xp.asarray(1e-6,f)
    nd=length2>xp.asarray(1e-12,f)
    diff=corners[None,:,:]-a[:,None,:]
    cross=e[:,None,0]*diff[:,:,1]-e[:,None,1]*diff[:,:,0]
    pos=xp.all(cross>=-tol,axis=1)
    neg=xp.all(cross<=tol,axis=1)
    # Reject all-collinear pairs, for which neither orientation is unique.
    support=nd & (pos ^ neg)
    q=xi-a
    side=e[:,0]*q[:,1]-e[:,1]*q[:,0]
    inside=xp.all((~support)|xp.where(pos,side>=-tol,side<=tol))
    t=xp.clip(xp.sum(q*e,axis=1)/xp.where(nd,length2,xp.ones_like(length2)),0.,1.)
    d=xp.linalg.norm(q-t[:,None]*e,axis=1)
    dmin=xp.min(xp.where(support,d,xp.asarray(float('inf'),f)))
    # Require nondegenerate, finite hull. A single edge cannot enclose area.
    good=(xp.sum(support.astype(xp.int32))>=3)&xp.all(xp.isfinite(corners))&xp.all(xp.isfinite(xi))&xp.all(xp.isfinite(cross))&xp.all(xp.isfinite(d))&xp.isfinite(dmin)
    dist=xp.where(inside,xp.asarray(0.,f),dmin)
    good=good&xp.isfinite(dist)&(dist>=0)
    return dist.astype(f),good


def r12_cost(xp, sample):
    loads,rects,xi,speed,safe,valid=sample
    f=xp.float32
    loaded=loads>=xp.asarray(.05,f)
    mode=loaded[0].astype(xp.int32)+2*loaded[1].astype(xp.int32)
    hull,hull_ok=r12_hull_distance(xp,xi,rects)
    dl=r11_distance(xp,xi,rects[0]);dr=r11_distance(xp,xi,rects[1])
    distance=xp.where(mode==3,hull,xp.where(mode==1,dl,xp.where(mode==2,dr,xp.asarray(0.,f))))
    good=valid&((mode!=3)|hull_ok)&xp.isfinite(distance)&(distance>=0)
    # Stable D^2/(D^2+.1^2); no invalid division when D=0.
    ref=xp.asarray(.10,f);scale=xp.maximum(distance,ref)
    a=distance/scale;b=ref/scale
    cost=xp.where(mode==0,xp.asarray(1.,f),a*a/(a*a+b*b)).astype(f)
    good=good&xp.isfinite(cost)&(cost>=0)&(cost<=1)
    return cost,mode,good


def r12_latch(xp, was, old_age, pre, post, *, event, boundary, coherent, terminal, truncated, nominal, native):
    # Actual event and coherence flags supplied by the native step.
    f=xp.float32
    cost,mode,good=r12_cost(xp,post)
    valid=pre[-1]&good&xp.isfinite(native)&((~was)|( (old_age>=0)&(old_age<100)))
    allowed=valid&coherent&(~boundary)&(~terminal)&(~truncated)&(~nominal)
    age=old_age+1
    active=was&(~event)&allowed&(age<=100)
    start=event&allowed
    keep=active&(age<100)
    closed=was&(~keep)
    next_active=start|keep
    next_age=xp.where(keep,age,0).astype(xp.int32)
    desired=xp.where(active,-xp.asarray(.04,f)*xp.asarray(.02,f)*cost,xp.asarray(0.,f)).astype(f)
    delivered=xp.where(desired<0,native+desired,native).astype(f)
    delta=(delivered-native).astype(f)
    bad=(~valid)|(~xp.isfinite(delivered))|(~xp.isfinite(delta))|(delta>0)
    # 1 repeat, 2 reset, 3 terminal, 4 truncation, 5 coherence/nominal, 6 invalid, 7 expiry.
    reason=xp.where(closed,xp.where(~valid,6,xp.where(boundary,2,xp.where(terminal,3,xp.where(truncated,4,xp.where(event,1,xp.where((~coherent)|nominal,5,7)))))),0).astype(xp.int32)
    return dict(active=next_active,age=next_age,open=start,close=closed,close_reason=reason,
                scored=active,support_mode=mode,cost=cost,native=native,desired=desired,
                delivered=delivered,delta=delta,bad=bad),delivered

def r12_init(xp):
    d={f'r12_{k}':xp.asarray(0,xp.int32) for k in R12_INTS+R12_CARRY}
    d.update({f'r12_{k}':xp.asarray(0.,xp.float32) for k in R12_FLOATS})
    return d


def r12_env_transition(xp,old,pre,post,*,event,repeat,direction,boundary,coherent,terminal,timeout,nominal,native,joint_excess,ctrl_excess,force_excess,foot_foot):
    """After original super.step; no physics/forward/inference calls."""
    x=lambda k:old['r12_'+k]
    z,reward=r12_latch(xp,x('active')==1,x('age'),pre,post,event=event,boundary=boundary,coherent=coherent,terminal=terminal,truncated=timeout,nominal=nominal,native=native)
    out=r12_init(xp)
    op=z['open'];keep=z['active']&(~op)
    event_side=xp.where(event,direction+1,0)
    side=xp.where(op,event_side,xp.where(keep,x('side'),0))
    kind=xp.where(op,xp.where(repeat,2,1),xp.where(keep,x('kind'),0))
    values=dict(active=z['active'],age=z['age'],side=side,kind=kind,before_active=x('active'),before_age=x('age'),before_side=x('side'),before_kind=x('kind'),after_active=z['active'],after_age=z['age'],after_side=side,after_kind=kind,open=op,close=z['close'],close_reason=z['close_reason'],scored=z['scored'],support_mode=z['support_mode'],coherent=coherent,terminal=terminal,timeout=timeout,event_side=event_side,joint_excess=joint_excess,ctrl_excess=ctrl_excess,force_excess=force_excess,foot_foot=foot_foot,bad=z['bad'])
    out.update({f'r12_{k}':xp.asarray(v,xp.int32) for k,v in values.items()})
    out.update({f'r12_{k}':xp.asarray(z[k],xp.float32) for k in R12_FLOATS})
    return out,reward


def r12_range_flags(xp,m,d):
    # Same 29 fields/ranges as r11_sample; r11_sample separately validates
    # finite values, range metadata and index mapping. No new safety penalty.
    flags=[]
    for values,ranges,limited in (
        (d.qpos[7:],xp.asarray(m.jnt_range)[1:],xp.asarray(m.jnt_limited)[1:]),
        (d.ctrl,xp.asarray(m.actuator_ctrlrange),xp.asarray(m.actuator_ctrllimited)),
        (d.actuator_force,xp.asarray(m.actuator_forcerange),xp.asarray(m.actuator_forcelimited))):
        flags.append(xp.any((limited!=0)&((values<ranges[:,0])|(values>ranges[:,1]))))
    return tuple(flags)

# H4R12_CORE_END

# H4FFE_CORE_BEGIN
# HELD HOST CORE ONLY: not called by CoverageJoystick or any trainer.
# No runtime authorization. The inputs are the actual pre-injection sample
# and existing scheduler draws at future integration; never binary contacts.
FFE_STATE_FIELDS=('k','hard','band','cell','due','phase','repeat_due','block')
FFE_FIRST=1
FFE_REPEAT=2
FFE_CLOSED=3
FFE_MISS_BOUNDARY=1
FFE_MISS_NOMINAL=2
FFE_MISS_EXPIRED=3
FFE_MISS_COHERENCE=4
FFE_MISS_TERMINAL=5
FFE_MISS_INVALID=6

def ffe_init(xp):
    # Unassigned until the first real transition at the restored host edge.
    return {name:xp.asarray(0 if name in ('k','hard','phase','block') else -1,
                            xp.int32) for name in FFE_STATE_FIELDS}

def ffe_pre(xp,old,*,hard_draw,band_draw,cell_draw,due_draw,boundary,nominal,
            count,coherent,source_step,mode,cost,geometry_valid):
    """Pure pre-injection candidate decision; no RNG/physics/reward calls.

    Caller must use independent fold_in subkeys and preserve original draws.
    coherent means exact zero native/actor/critic command and live prestate.
    mode is actual r12_cost loaded support: 0/1/2/3, not binary contact.
    All input shapes/dtypes and actual transition conservation need a separate
    batch guard. This function alone is NOT a production scheduler.
    """
    i=xp.int32
    k=old['k'];initial=k==0
    hard=xp.where(initial,hard_draw,old['hard']).astype(i)
    band=xp.where(initial,band_draw,old['band']).astype(i)
    cell=xp.where(initial,cell_draw,old['cell']).astype(i)
    due=xp.where(initial,due_draw,old['due']).astype(i)
    phase=xp.where(initial,FFE_FIRST,old['phase']).astype(i)
    repeat_due=old['repeat_due']
    target_mode=xp.where(cell<2,cell+1,3).astype(i)
    direction=(2+cell%2).astype(i)
    # Hard assignment remains fixed; original broad can return only at a
    # natural boundary strictly after closure of this opportunity.
    block=(hard>0)&xp.where(initial,True,xp.where(
        boundary&(phase==FFE_CLOSED),False,old['block']>0))
    active=block&(phase!=FFE_CLOSED)
    first=active&(phase==FFE_FIRST)
    wait=active&(phase==FFE_REPEAT)
    at_due=k>=due
    geometry_ok=geometry_valid&xp.isfinite(cost)&(cost>=0)&(cost<=1)&(mode>=0)&(mode<=3)
    assignment_ok=((hard==0)|(hard==1))&(band>=0)&(band<4)&(cell>=0)&(cell<4)&(
        due>=40+160*band)&(due<=80+160*band)
    bad=(k<0)|(k>=640)|(~assignment_ok)|(
        active&(~geometry_ok)&((first&at_due)|wait))
    # Reset before T retains slot; reset at/after T or during repeat closes.
    boundary_miss=active&boundary&((first&at_due)|wait)
    nominal_miss=active&nominal&((first&at_due)|wait)
    coherence_miss=wait&(~coherent | (source_step>=500))
    expired=(first&(k>due+8))|(wait&(k>repeat_due+8))
    eligible=active&(~boundary)&(~nominal)&coherent&geometry_ok&(source_step<=459)&(~bad)
    first_event=first&at_due&(k<=due+8)&eligible&(count==0)&(mode>0)
    # Repeat clock source limit is the command refresh boundary, not the
    # first-event <=459 guard; at most 40 ticks after the first event.
    repeat_event=wait&(k>=repeat_due)&(k<=repeat_due+8)&(~boundary)&(~nominal)&(
        coherent)&(source_step<500)&geometry_ok&(~bad)&(count==1)&(
        mode==target_mode)&(cost>=xp.asarray(.5,xp.float32))
    reason=xp.where(bad,FFE_MISS_INVALID,xp.where(boundary_miss,FFE_MISS_BOUNDARY,
        xp.where(nominal_miss,FFE_MISS_NOMINAL,xp.where(coherence_miss,FFE_MISS_COHERENCE,
        xp.where(expired,FFE_MISS_EXPIRED,0))))).astype(i)
    first_event=first_event&(reason==0)
    repeat_event=repeat_event&(reason==0)
    # On the final allowed tick, a miss closes immediately, not a tick late.
    deadline_miss=(first&(k==due+8)&(~first_event))|(wait&(k==repeat_due+8)&(~repeat_event))
    reason=xp.where((reason==0)&deadline_miss,FFE_MISS_EXPIRED,reason).astype(i)
    phase=xp.where((reason>0)|repeat_event,FFE_CLOSED,phase).astype(i)
    state=dict(k=k,hard=hard,band=band,cell=cell,due=due,phase=phase,
               repeat_due=repeat_due,block=block.astype(i))
    return state,dict(first=first_event,repeat=repeat_event,direction=direction,
                      block=block,bad=bad,reason=reason,target_mode=target_mode)

def ffe_post(xp,state,pre,*,gap,post_coherent,terminal,refresh):
    """Advance once per actual action; preserve actual event even if terminal."""
    i=xp.int32
    k=state['k'];first=pre['first'];repeat=pre['repeat']
    first_or_wait=first|(state['phase']==FFE_REPEAT)
    gap_bad=first&((gap<30)|(gap>32))
    closure=first_or_wait&(terminal|refresh|(~post_coherent))
    # A terminal first event still counts but never opens a repeat.
    reason=xp.where(gap_bad,FFE_MISS_INVALID,xp.where(closure,
        xp.where(terminal,FFE_MISS_TERMINAL,FFE_MISS_COHERENCE),pre['reason'])).astype(i)
    phase=xp.where(closure|gap_bad|repeat,FFE_CLOSED,
        xp.where(first,FFE_REPEAT,state['phase'])).astype(i)
    # If a hard slot is due but terminal occurred on its non-event row, close.
    due_terminal=(state['hard']>0)&(phase==FFE_FIRST)&(k>=state['due'])&terminal
    reason=xp.where(due_terminal,FFE_MISS_TERMINAL,reason).astype(i)
    phase=xp.where(due_terminal,FFE_CLOSED,phase).astype(i)
    out=dict(state,k=(k+1).astype(i),phase=phase,
             repeat_due=xp.where(first,k+gap,state['repeat_due']).astype(i))
    return out,dict(reason=reason,bad=pre['bad']|gap_bad)


def ffe_exposure_gate(chunk_cell,nominal,ordinary,archived):
    """Host engineering screen only; inputs must later be ledger-replayed.

    chunk_cell: 4 chronological chunks x 4 mutually exclusive qualified
    actual repeat cells. ordinary/archived: motion IDs 1..5, no hard events.
    No legacy audit, reward or scientific gate is replaced by this helper.
    """
    def integer(x):
        if type(x) is not int or x<0:
            raise RuntimeError('M470_FFE_EXPOSURE_INTEGER')
        return x
    if (len(chunk_cell)!=4 or any(len(r)!=4 for r in chunk_cell) or
            len(ordinary)!=5 or len(archived)!=5):
        raise RuntimeError('M470_FFE_EXPOSURE_SHAPE')
    rows=[[integer(x) for x in row] for row in chunk_cell]
    nominal=integer(nominal)
    broad=[integer(x) for x in ordinary]
    old=[integer(x) for x in archived]
    if nominal>5242880 or sum(map(sum,rows))>5242880 or any(x<=0 for x in old):
        raise RuntimeError('M470_FFE_EXPOSURE_RANGE')
    half=[[rows[2*h][j]+rows[2*h+1][j] for j in range(4)] for h in range(2)]
    checks=dict(half_cell=all(x>=3 for row in half for x in row),
                chunk=all(sum(row)>=1 for row in rows),
                nominal=nominal>=916743 and 100*nominal>=15*5242880,
                ordinary=all(5*n>=4*o for n,o in zip(broad,old)))
    return dict(checks=checks,half_cell=half,qualified=sum(map(sum,rows)),
                decision='EXPOSURE_PASS' if all(checks.values()) else 'EXPOSURE_INCONCLUSIVE')

def ffe_core_fixtures(np):
    """Synthetic scalar host fixtures, no native/JAX import or calls."""
    def require(ok,label):
        if not bool(ok):
            raise RuntimeError('M470_FFE_CORE_FIXTURE '+label)
    def pre(s,**kw):
        a=dict(hard_draw=1,band_draw=0,cell_draw=0,due_draw=40,
               boundary=False,nominal=False,count=0,coherent=True,
               source_step=40,mode=1,cost=np.float32(.5),geometry_valid=True)
        a.update(kw)
        return ffe_pre(np,s,**a)
    def post(s,p,**kw):
        a=dict(gap=30,post_coherent=True,terminal=False,refresh=False)
        a.update(kw)
        return ffe_post(np,s,p,**a)
    tested=0
    for band in range(4):
        for cell in range(4):
            due=40+160*band
            s=ffe_init(np)
            for k in range(640):
                require(int(s['k'])==k,'CLOCK')
                repeat=due+30
                # A natural episode boundary before due and after closure;
                # pass different later draws to prove slot is not resampled.
                q,p=pre(s,band_draw=band if k==0 else (band+1)%4,
                        cell_draw=cell if k==0 else (cell+1)%4,
                        due_draw=due if k==0 else 40,
                        boundary=k in (0,10,repeat+2),
                        count=int(due<k<=repeat),
                        mode=cell+1 if cell<2 else 3)
                require(int(q['band'])==band and int(q['cell'])==cell and
                        int(q['due'])==due,'FIXED_ASSIGNMENT')
                require(bool(p['first'])==(k==due),'FIRST_TIME')
                require(bool(p['repeat'])==(k==repeat),'REPEAT_TIME')
                require(int(p['direction'])==2+cell%2,'DIRECTION')
                require(bool(p['block'])==(k<repeat+2),'BROAD_BLOCK')
                require(not p['bad'],'VALID_ROW')
                s,z=post(q,p)
                require(not z['bad'],'VALID_POST')
            require(int(s['k'])==640 and int(s['phase'])==FFE_CLOSED,'FINAL_CLOCK')
            tested+=1
    # Fixture helper starts from a known assigned phase; no production
    # state synthesis or actual transition is performed.
    q,p=pre(ffe_init(np),boundary=True)
    s,_=post(q,p)
    s=dict(s,k=np.int32(40))
    for change,reason in [
        (dict(boundary=True),FFE_MISS_BOUNDARY),
        (dict(nominal=True),FFE_MISS_NOMINAL),
        (dict(geometry_valid=False),FFE_MISS_INVALID),
    ]:
        q,p=pre(s,**change)
        require(not p['first'] and not p['repeat'] and int(p['reason'])==reason,'FIRST_REJECT')
        tested+=1
    q,p=pre(dict(s,k=np.int32(48)),coherent=False)
    require(not p['first'] and p['reason']==FFE_MISS_EXPIRED,'FIRST_DEADLINE');tested+=1
    q,p=pre(s)
    require(p['first'],'FIRST_EVENT')
    for change,reason in [
        (dict(terminal=True),FFE_MISS_TERMINAL),
        (dict(refresh=True),FFE_MISS_COHERENCE),
        (dict(post_coherent=False),FFE_MISS_COHERENCE),
        (dict(gap=29),FFE_MISS_INVALID),
    ]:
        z,a=post(q,p,**change)
        require(z['phase']==FFE_CLOSED and a['reason']==reason,'POST_REJECT')
        tested+=1
    z,a=post(q,p,gap=32)
    require(z['repeat_due']==72,'GAP32');tested+=1
    for change in (dict(mode=3),dict(mode=0),dict(cost=np.float32(.499)),
                   dict(coherent=False),dict(source_step=500),dict(nominal=True),
                   dict(boundary=True),dict(geometry_valid=False)):
        q,p=pre(dict(z,k=np.int32(72)),count=1,**change)
        require(not p['repeat'],'REPEAT_REJECT');tested+=1
    q,p=pre(dict(z,k=np.int32(80)),count=1)
    require(p['repeat'],'GAP40');tested+=1
    q,p=pre(dict(z,k=np.int32(80)),count=1,mode=2)
    require(not p['repeat'] and p['reason']==FFE_MISS_EXPIRED,'NO_FALLBACK');tested+=1
    q,p=pre(ffe_init(np),hard_draw=0,boundary=True)
    require(not p['block'] and not p['first'] and not p['repeat'],'ORDINARY_UNTOUCHED');tested+=1
    rows=[[1]*4,[2]*4,[1]*4,[2]*4]
    good=ffe_exposure_gate(rows,916743,[8]*5,[10]*5)
    require(good['decision']=='EXPOSURE_PASS' and good['qualified']==24,'EXPOSURE_PASS');tested+=1
    for rr,n,b in (([[0]*4,[3]*4,[1]*4,[2]*4],916743,[8]*5),
                   (rows,916742,[8]*5),(rows,916743,[7,8,8,8,8]),
                   ([[1]*4,[1]*4,[2]*4,[2]*4],916743,[8]*5)):
        require(ffe_exposure_gate(rr,n,b,[10]*5)['decision']=='EXPOSURE_INCONCLUSIVE','EXPOSURE_REJECT');tested+=1
    return dict(fixtures=tested,physical_transitions=0,resets=0,
                optimizer_updates=0,native_inference=0)

# H4FFE_CORE_END

# H4FFE_ENV_WIRING_BEGIN
# Only used by opt-in FFE; legacy branches remain byte-projected.
FFE_ROW_INTS=('hard_draw','band_draw','cell_draw','due_draw','count','after_count','source_step','after_source_step',
    'pre_coherent','post_coherent','geometry_valid','mode','gap','refresh',
    'terminal','first','repeat','block','reason','bad','stratum','ordinary_event',
    'tail_stratum','tick_stratum','tail_same','tick_same','joint_excess','ctrl_excess','force_excess','foot_foot')
FFE_ROW_FLOATS=('cost','native')
FFE_EXTRA_FIELDS=tuple('ffe_'+k for k in (
    tuple('before_'+k for k in FFE_STATE_FIELDS)+
    tuple('after_'+k for k in FFE_STATE_FIELDS)+FFE_ROW_INTS+FFE_ROW_FLOATS))
FFE_CARRY_FIELDS=tuple('ffe_'+k for k in FFE_STATE_FIELDS)+('ffe_tail_stratum','ffe_tail_same')

def ffe_env_init(xp):
    row={'ffe_'+k:xp.asarray(0,xp.int32) for k in FFE_ROW_INTS}
    row.update({'ffe_'+k:xp.asarray(0.,xp.float32) for k in FFE_ROW_FLOATS})
    old=ffe_init(xp)
    row.update({'ffe_'+k:v for k,v in old.items()})
    for prefix in ('before_','after_'):
        row.update({'ffe_'+prefix+k:v for k,v in old.items()})
    return row

# H4FFE_ENV_WIRING_END

class CoverageJoystick(joystick.Joystick):
    """JAX-compatible source G1 subclass, independent of pinned checkout.

    BraxAutoResetWrapper restores initial physical state after termination but
    leaves info alive; `info['steps']==0` is therefore used as the episode
    boundary for re-sampling push targets and counters.  Tracking `age` also
    supports a direct/unwrapped one-env diagnostic.
    """
    def __init__(self, *, stage: str, config, task='flat_terrain', h1_m1=False, h1_m2=False, h3_m1=False, h4_arm=None, h4cc_arm=None, h4m6_arm=None, h4m6_tech_capture=False, h4m8_common=False, h4red=False, r11=False, r12=False, ffe=False, ffe2=False):
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
        if type(h4m8_common) is not bool or (h4m8_common and (
                h4_arm!='candidate' or h4cc_arm!='ref' or h4m6_arm is not None or h4m6_tech_capture or
                float(config.reward_config.scales.stand_still)!=-1.0 or
                float(config.reward_config.scales.termination)!=-100.0 or
                len(config.reward_config.scales)!=24 or int(config.episode_length)!=1000)):
            raise ValueError('M470_H4M8_ENV_CONTRACT')
        if type(h4red) is not bool or (h4red and (
                stage != 'easy_repeat_support' or h4_arm != 'control' or
                h4cc_arm is not None or h4m6_arm is not None or
                h4m6_tech_capture or h4m8_common)):
            raise ValueError('M470_H4RED_ENV_CONTRACT')
        if type(r11) is not bool or (r11 and (stage!='easy_repeat_support' or h4_arm!='control' or h4red or h1_m1 or h1_m2 or h3_m1 or h4cc_arm is not None or h4m6_arm is not None or h4m6_tech_capture or h4m8_common)):
            raise ValueError('M470_R11_ENV_CONTRACT')
        if type(r12) is not bool or (r12 and (stage!='easy_repeat_support' or h4_arm!='control' or r11 or h4red or h1_m1 or h1_m2 or h3_m1 or h4cc_arm is not None or h4m6_arm is not None or h4m6_tech_capture or h4m8_common)):
            raise ValueError('M470_R12_ENV_CONTRACT')
        if type(ffe) is not bool or type(ffe2) is not bool or (ffe2 and ffe):
            raise ValueError('M470_FFE2_MODE_CONFLICT')
        self._ffe2 = ffe2
        ffe = ffe or ffe2
        if type(ffe) is not bool or (ffe and (stage!='easy_repeat_support' or
                h4_arm!='control' or h4red or r11 or r12 or h1_m1 or h1_m2 or h3_m1 or
                h4cc_arm is not None or h4m6_arm is not None or h4m6_tech_capture or h4m8_common or
                len(config.reward_config.scales)!=24 or int(config.episode_length)!=1000)):
            raise ValueError('M470_FFE_ENV_CONTRACT')
        self._ffe = ffe
        self._r12 = r12
        self._r11 = r11
        self._h4red = h4red
        self._h4m8_common = h4m8_common
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
                # Validate native scalar dtype before the heterogeneous telemetry stack.
                for k in H4M6_REWARD_NAMES:
                    term=jp.asarray(rewards[k])
                    expected_dtype=jp.bool_ if k in H4M6_BOOL_REWARD_NAMES else jp.float32
                    if term.shape != () or term.dtype != expected_dtype:
                        raise RuntimeError('M470_H4M6_TECH_NATIVE_REWARD_TYPE '+k)
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
        if getattr(self,'_h4m8_common',False):
            info.update({'h4m8_'+k:zeros for k in H4M8_STATE+H4M8_INT_FIELDS})
            info.update({'h4m8_'+k:jp.asarray(0.,jp.float32) for k in H4M8_FLOAT_FIELDS})
        if self._r11:
            info.update(r11_init(jp))
        if self._r12:
            info.update(r12_init(jp))
        if self._ffe:
            info.update(ffe_env_init(jp))
            if self._ffe2:
                info.update(ffe2_env_init(jp))
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
            target_first = jax.random.randint(k_time, (), 12, 344 if self._h4red else 25)
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

        if self._ffe:
            if 'steps' not in info:
                raise RuntimeError('M470_FFE_WRAPPED_CLOCK_REQUIRED')
            ffe_old={k:info['ffe_'+k] for k in FFE_STATE_FIELDS}
            # Fixed tags on the existing scheduler subkeys. Original split,
            # original Bernoulli no-push draw and ordinary draws are unchanged.
            ffe_hard=jax.random.bernoulli(jax.random.fold_in(k_start,18101),p=.125).astype(jp.int32)
            ffe_band=jax.random.randint(jax.random.fold_in(k_time,18102),(),0,4,dtype=jp.int32)
            ffe_cell=jax.random.randint(jax.random.fold_in(k_dir,18103),(),0,4,dtype=jp.int32)
            ffe_due=40+160*ffe_band+jax.random.randint(jax.random.fold_in(k_time,18104),(),0,41,dtype=jp.int32)
            ffe_gap=jax.random.randint(jax.random.fold_in(k_gap,18105),(),30,33,dtype=jp.int32)
            ffe_cost,ffe_mode,ffe_geometry=r12_cost(jp,r11_sample(
                jp,self.mjx_model,state.data,self._feet_geom_id,self._floor_geom_id))
            ffe_coherent=(jp.all(pre_command==0)&
                jp.all(state.obs['state'][9:12]==pre_command)&
                jp.all(state.obs['privileged_state'][9:12]==pre_command)&(state.done==0))
            # SSRPC_DISPATCH_PRE_BEGIN
            ffe_pre_fn=(ssrpc_pre_cand if self.ssrpc_arm=='cand' else ssrpc_pre_ctrl) if isinstance(self,SSRPCCoverageJoystick) else (ffe2_pre if self._ffe2 else ffe_pre)
            # SSRPC_DISPATCH_PRE_END
            ffe_prestate,ffe_pick=ffe_pre_fn(jp,ffe_old,hard_draw=ffe_hard,
                band_draw=ffe_band,cell_draw=ffe_cell,due_draw=ffe_due,
                boundary=boundary,nominal=disabled,count=count,coherent=ffe_coherent,
                source_step=pre_source_step,mode=ffe_mode,cost=ffe_cost,geometry_valid=ffe_geometry)
            ffe_block=ffe_pick['block']&(~disabled)
            # Legacy due is an episode-relative view of the frozen absolute
            # opportunity. Keep true pending entries on expiry until native
            # terminal or host reset. Do not invent an actual event or close.
            ffe_phase=jp.where(ffe_old['k']==0,FFE_FIRST,ffe_old['phase'])
            ffe_target=jp.where(ffe_phase==FFE_REPEAT,ffe_old['repeat_due'],ffe_prestate['due'])
            ffe_target=jp.where(ffe_phase==FFE_CLOSED,2**30,age+ffe_target-ffe_old['k'])
            next_step=jp.where(ffe_block,ffe_target,next_step).astype(jp.int32)
            direction=jp.where(ffe_block,ffe_pick['direction'],direction).astype(jp.int32)

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
        if self._ffe:
            event=jp.where(ffe_pick['block'],ffe_pick['first']|ffe_pick['repeat'],event)
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
        if getattr(self,'_h4m8_common',False):
            m8 = h4m8_pre(jp,info,event=event,repeat=event & (count>0),cls=h4_class,
                first_age=jp.where(boundary,-1,info['h4_first_age']),
                first_mag=jp.where(boundary,0.,info['h4_first_mag']),
                gap=age-last_event_age,magnitude=magnitude,
                first_motion=jp.where(boundary,-1,info['h4_first_motion']),
                pre_motion=pre_motion,actor_motion=actor_motion,direction=direction,
                support_side=support_side,boundary=boundary,age=age,clock=pre_source_step,
                wrapper_steps=info['steps'],command=pre_command,
                actor_command=state.obs['state'][9:12],critic_command=state.obs['privileged_state'][9:12])
        if self._r11:
            r11_pre=r11_sample(jp,self.mjx_model,state.data,self._feet_geom_id,self._floor_geom_id)
            r11_old={k:info[k] for k in r11_init(jp)}
            r11_coherent=(jp.all(pre_command==0)&jp.all(state.obs['state'][9:12]==pre_command)&jp.all(state.obs['privileged_state'][9:12]==pre_command)&(pre_source_step<500)&(~disabled))
            r11_timeout=info['steps']>=int(self._config.episode_length)-1
        if self._r12:
            r12_pre=r11_sample(jp,self.mjx_model,state.data,self._feet_geom_id,self._floor_geom_id)
            r12_old={k:info[k] for k in r12_init(jp)}
            r12_coherent=(jp.all(pre_command==0)&jp.all(state.obs['state'][9:12]==pre_command)&jp.all(state.obs['privileged_state'][9:12]==pre_command)&(pre_source_step<500)&(~disabled))
            r12_timeout=info['steps']>=int(self._config.episode_length)-1
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
        if self._ffe:
            ffe_post_coherent=(jp.all(out['command']==0)&
                jp.all(after.obs['state'][9:12]==out['command'])&
                jp.all(after.obs['privileged_state'][9:12]==out['command']))
            # SSRPC_DISPATCH_POST_BEGIN
            ffe_post_fn=ssrpc_post if isinstance(self,SSRPCCoverageJoystick) else (ffe2_post if self._ffe2 else ffe_post)
            # SSRPC_DISPATCH_POST_END
            ffe_new,ffe_result=ffe_post_fn(jp,ffe_prestate,ffe_pick,gap=ffe_gap,
                post_coherent=ffe_post_coherent,terminal=after.done>0,
                refresh=pre_source_step>=500)
            # Only an actual first with a coherent, live continuation assigns
            # a hard repeat. A terminal first remains an actual first event.
            # SSRPC_DISPATCH_GAP_BEGIN
            gap=jp.where(ffe_block,35 if isinstance(self,SSRPCCoverageJoystick) else (30 if self._ffe2 else ffe_gap),gap)
            # SSRPC_DISPATCH_GAP_END
            repeat_allowed=jp.where(ffe_block,
                ffe_pick['first']&(ffe_new['phase']==FFE_REPEAT),repeat_allowed)
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
        if getattr(self,'_h4m8_common',False):
            model,d=self.mjx_model,after.data
            M,h2,cost,good=h4m8_momentum(jp,model.body_mass,model.body_inertia,
                d.xipos,d.ximat,d.cvel,d.subtree_com,model.body_rootid)
            post_command=out['command']
            post_coherent=(jp.all(post_command==after.obs['state'][9:12]) &
                           jp.all(post_command==after.obs['privileged_state'][9:12]))
            m8out=h4m8_post(jp,info,m8,done=after.done>0,
                post_zero=jp.all(post_command==0),post_coherent=post_coherent,
                native_mature=out['h4_mature_repeat']>0,native_side=out['h4_event_qual'],
                mass=M,h2=h2,cost=cost,momentum_good=good)
            out.update(m8out)
            # CAM validity stays in its own new telemetry; old leaves are untouched.
        if self._r11:
            r11_post=r11_sample(jp,self.mjx_model,after.data,self._feet_geom_id,self._floor_geom_id)
            coherent=r11_coherent&jp.all(out['command']==pre_command)&jp.all(after.obs['state'][9:12]==pre_command)&jp.all(after.obs['privileged_state'][9:12]==pre_command)
            extra,reward=r11_transition(jp,r11_old,r11_pre,r11_post,event=event,boundary=boundary,coherent=coherent,terminal=(after.done>0)|r11_timeout,native=after.reward)
            out.update(extra)
            after=after.replace(reward=reward)
        if self._r12:
            r12_post=r11_sample(jp,self.mjx_model,after.data,self._feet_geom_id,self._floor_geom_id)
            coherent=r12_coherent&jp.all(out['command']==pre_command)&jp.all(after.obs['state'][9:12]==pre_command)&jp.all(after.obs['privileged_state'][9:12]==pre_command)
            joint,ctrl,force=r12_range_flags(jp,self.mjx_model,after.data)
            foot_foot=geoms_colliding(after.data,self._right_foot_geom_id,self._left_foot_geom_id)
            extra,reward=r12_env_transition(jp,r12_old,r12_pre,r12_post,event=event,repeat=repeat,direction=direction,boundary=boundary,coherent=coherent,terminal=after.done>0,timeout=r12_timeout,nominal=disabled,native=after.reward,joint_excess=joint,ctrl_excess=ctrl,force_excess=force,foot_foot=foot_foot)
            out.update(extra)
            after=after.replace(reward=reward)
        if self._ffe:
            ffe_stratum=jp.where(disabled,0,jp.where(ffe_block,2,1))
            ffe_tail_stratum=jp.where(event & (after.done==0),ffe_stratum,
                jp.where(out['h4_tail_kind']>0,info['ffe_tail_stratum'],0))
            ffe_tick_stratum=jp.where(out['h4_tick_kind']>0,info['ffe_tail_stratum'],0)
            ffe_tick_same=(out['h4_tick_same']>0)&(info['ffe_tail_same']>0)&(
                (ffe_tick_stratum!=2)|ffe_coherent)
            ffe_tail_same=jp.where(event & (after.done==0),
                (actor_motion==pre_motion)&((ffe_stratum!=2)|ffe_post_coherent),
                (out['h4_tail_kind']>0)&ffe_tick_same)
            joint,ctrl,force=r12_range_flags(jp,self.mjx_model,after.data)
            foot=geoms_colliding(after.data,self._right_foot_geom_id,self._left_foot_geom_id)
            ffe_row=dict(hard_draw=ffe_hard,band_draw=ffe_band,cell_draw=ffe_cell,
                due_draw=ffe_due,count=count,after_count=new_count,source_step=pre_source_step,
                after_source_step=out['step'],
                pre_coherent=ffe_coherent,post_coherent=ffe_post_coherent,
                geometry_valid=ffe_geometry,mode=ffe_mode,gap=ffe_gap,
                refresh=pre_source_step>=500,terminal=after.done>0,
                first=ffe_pick['first'],repeat=ffe_pick['repeat'],block=ffe_pick['block'],
                reason=ffe_result['reason'],bad=ffe_result['bad']|injection_bad,
                stratum=ffe_stratum,ordinary_event=event&(ffe_stratum==1),
                tail_stratum=ffe_tail_stratum,tick_stratum=ffe_tick_stratum,
                tail_same=ffe_tail_same,tick_same=ffe_tick_same,
                joint_excess=joint,ctrl_excess=ctrl,force_excess=force,foot_foot=foot)
            out.update({'ffe_'+k:jp.asarray(v,jp.int32) for k,v in ffe_row.items()})
            for prefix,values in (('',ffe_new),('before_',ffe_old),('after_',ffe_new)):
                out.update({'ffe_'+prefix+k:jp.asarray(v,jp.int32) for k,v in values.items()})
            out.update(ffe_native=jp.asarray(after.reward,jp.float32),
                       ffe_cost=jp.asarray(ffe_cost,jp.float32))
            if self._ffe2:
                shadow=info['ffe2_shadow_gap']
                shadow_next=jp.where(ffe_pick['first'],ffe_gap,shadow).astype(jp.int32)
                # SSRPC_DISPATCH_LABEL_BEGIN
                label_fn=(ssrpc_labels_cand if self.ssrpc_arm=='cand' else ssrpc_labels_ctrl) if isinstance(self,SSRPCCoverageJoystick) else ffe2_risk_labels
                labels=label_fn(jp,ffe_old,ffe_prestate,ffe_pick,ffe_new,ffe_result,
                # SSRPC_DISPATCH_LABEL_END
                    boundary=boundary,nominal=disabled,count=count,coherent=ffe_coherent,
                    source_step=pre_source_step,mode=ffe_mode,cost=ffe_cost,geometry_valid=ffe_geometry,
                    shadow_gap=shadow,terminal=after.done>0,refresh=pre_source_step>=500,
                    post_coherent=ffe_post_coherent)
                out.update(ffe2_shadow_gap=shadow_next,ffe2_shadow_before=shadow,
                           ffe2_shadow_after=shadow_next)
                out.update({'ffe2_'+k:jp.asarray(v,jp.int32) for k,v in labels.items()})
        return after.replace(info=out)

# H4FFE2_HELD_CORE_BEGIN
# Pure V2 scheduler, called only by opt-in CoverageJoystick; runtime remains HOLD.
# Do not remove execution HOLD until separate full integration review.
def ffe2_pre(xp,old,*,hard_draw,band_draw,cell_draw,due_draw,boundary,nominal,
            count,coherent,source_step,mode,cost,geometry_valid):
    """Pure pre-injection candidate decision; no RNG/physics/reward calls.

    Caller must use independent fold_in subkeys and preserve original draws.
    coherent means exact zero native/actor/critic command and live prestate.
    mode is actual r12_cost loaded support: 0/1/2/3, not binary contact.
    All input shapes/dtypes and actual transition conservation need a separate
    batch guard. This function alone is NOT a production scheduler.
    """
    i=xp.int32
    k=old['k'];initial=k==0
    hard=xp.where(initial,hard_draw,old['hard']).astype(i)
    band=xp.where(initial,band_draw,old['band']).astype(i)
    cell=xp.where(initial,cell_draw,old['cell']).astype(i)
    due=xp.where(initial,due_draw,old['due']).astype(i)
    phase=xp.where(initial,FFE_FIRST,old['phase']).astype(i)
    repeat_due=old['repeat_due']
    target_mode=xp.where(cell<2,cell+1,3).astype(i)
    direction=(2+cell%2).astype(i)
    # Hard assignment remains fixed; original broad can return only at a
    # natural boundary strictly after closure of this opportunity.
    block=(hard>0)&xp.where(initial,True,xp.where(
        boundary&(phase==FFE_CLOSED),False,old['block']>0))
    active=block&(phase!=FFE_CLOSED)
    first=active&(phase==FFE_FIRST)
    wait=active&(phase==FFE_REPEAT)
    at_due=k>=due
    geometry_ok=geometry_valid&xp.isfinite(cost)&(cost>=0)&(cost<=1)&(mode>=0)&(mode<=3)
    assignment_ok=((hard==0)|(hard==1))&(band>=0)&(band<4)&(cell>=0)&(cell<4)&(
        due>=40+160*band)&(due<=80+160*band)
    bad=(k<0)|(k>=640)|(~assignment_ok)|(
        active&(~geometry_ok)&((first&at_due)|wait))
    # Reset before T retains slot; reset at/after T or during repeat closes.
    boundary_miss=active&boundary&((first&at_due)|wait)
    nominal_miss=active&nominal&((first&at_due)|wait)
    coherence_miss=wait&(~coherent | (source_step>=500))
    first_end=112+160*band
    repeat_end=repeat_due+10
    expired=(first&(k>first_end))|(wait&(k>repeat_end))
    eligible=active&(~boundary)&(~nominal)&coherent&geometry_ok&(source_step<=459)&(~bad)
    first_event=first&at_due&(k<=first_end)&eligible&(count==0)&(mode>0)
    # Repeat clock source limit is the command refresh boundary, not the
    # first-event <=459 guard; at most 40 ticks after the first event.
    repeat_event=wait&(k>=repeat_due)&(k<=repeat_end)&(~boundary)&(~nominal)&(
        coherent)&(source_step<500)&geometry_ok&(~bad)&(count==1)&(
        mode==target_mode)&(cost>=xp.asarray(.5,xp.float32))
    reason=xp.where(bad,FFE_MISS_INVALID,xp.where(boundary_miss,FFE_MISS_BOUNDARY,
        xp.where(nominal_miss,FFE_MISS_NOMINAL,xp.where(coherence_miss,FFE_MISS_COHERENCE,
        xp.where(expired,FFE_MISS_EXPIRED,0))))).astype(i)
    first_event=first_event&(reason==0)
    repeat_event=repeat_event&(reason==0)
    # On the final allowed tick, a miss closes immediately, not a tick late.
    deadline_miss=(first&(k==first_end)&(~first_event))|(wait&(k==repeat_end)&(~repeat_event))
    reason=xp.where((reason==0)&deadline_miss,FFE_MISS_EXPIRED,reason).astype(i)
    phase=xp.where((reason>0)|repeat_event,FFE_CLOSED,phase).astype(i)
    state=dict(k=k,hard=hard,band=band,cell=cell,due=due,phase=phase,
               repeat_due=repeat_due,block=block.astype(i))
    return state,dict(first=first_event,repeat=repeat_event,direction=direction,
                      block=block,bad=bad,reason=reason,target_mode=target_mode)

def ffe2_post(xp,state,pre,*,gap,post_coherent,terminal,refresh):
    """Advance once per actual action; preserve actual event even if terminal."""
    i=xp.int32
    k=state['k'];first=pre['first'];repeat=pre['repeat']
    first_or_wait=first|(state['phase']==FFE_REPEAT)
    gap_bad=first&((gap<30)|(gap>32))
    closure=first_or_wait&(terminal|refresh|(~post_coherent))
    # A terminal first event still counts but never opens a repeat.
    reason=xp.where(gap_bad,FFE_MISS_INVALID,xp.where(closure,
        xp.where(terminal,FFE_MISS_TERMINAL,FFE_MISS_COHERENCE),pre['reason'])).astype(i)
    phase=xp.where(closure|gap_bad|repeat,FFE_CLOSED,
        xp.where(first,FFE_REPEAT,state['phase'])).astype(i)
    # If a hard slot is due but terminal occurred on its non-event row, close.
    due_terminal=(state['hard']>0)&(phase==FFE_FIRST)&(k>=state['due'])&terminal
    reason=xp.where(due_terminal,FFE_MISS_TERMINAL,reason).astype(i)
    phase=xp.where(due_terminal,FFE_CLOSED,phase).astype(i)
    out=dict(state,k=(k+1).astype(i),phase=phase,
             repeat_due=xp.where(first,k+30,state['repeat_due']).astype(i))
    return out,dict(reason=reason,bad=pre['bad']|gap_bad)

def ffe2_risk_labels(xp,old,pre,pick,nxt,result,*,boundary,nominal,count,
                     coherent,source_step,mode,cost,geometry_valid,
                     shadow_gap,terminal,refresh,post_coherent):
    """Passive labels on an actual pre/post transition; shadow_gap is the
    sampled V1 gap latched at the actual first event (not current RNG draw).
    No state mutation, no policy feedback and no extra native calls.
    """
    k=old['k']; initial=k==0
    phase=xp.where(initial,FFE_FIRST,old['phase'])
    active=pick['block']&(phase!=FFE_CLOSED)
    first=active&(phase==FFE_FIRST)
    wait=active&(phase==FFE_REPEAT)
    due=pre['due']; first_end=112+160*pre['band']
    actual_gap=k-(old['repeat_due']-30)
    first_risk=first&(k>=due)&(k<=first_end)
    repeat_risk=wait&(actual_gap>=30)&(actual_gap<=40)
    geometry=geometry_valid&xp.isfinite(cost)&(cost>=0)&(cost<=1)&(mode>=0)&(mode<=3)
    def bits(predicates):
        out=xp.zeros_like(k,dtype=xp.int32)
        for index,passed in enumerate(predicates):
            out=out|((~passed).astype(xp.int32)<<index)
        return out
    common=(~boundary,~nominal,coherent)
    first_bits=bits(common+(source_step<=459,geometry,count==0,mode>0))
    repeat_bits=bits(common+(source_step<500,geometry,count==1,
                              mode==pick['target_mode'],cost>=xp.asarray(.5,xp.float32)))
    first_event=pick['first']; repeat_event=pick['repeat']
    first_eligible=first_risk&(first_bits==0)&(~pick['bad'])
    repeat_eligible=repeat_risk&(repeat_bits==0)&(~pick['bad'])
    closed=nxt['phase']==FFE_CLOSED
    first_shadow=first_risk&(k<=due+8)
    repeat_shadow=repeat_risk&(actual_gap>=shadow_gap)&(actual_gap<=shadow_gap+8)
    return dict(
        assigned=(pre['hard']==1)&initial,
        first_due=first&(k==due),first_risk=first_risk,
        first_eligible=first_eligible,first_event=first_event,
        first_terminal=first_event&terminal,
        first_miss=first&closed&(~first_event),
        repeat_opened=first_event&(nxt['phase']==FFE_REPEAT),
        repeat_wait=wait,repeat_risk=repeat_risk,
        repeat_eligible=repeat_eligible,repeat_event=repeat_event,
        repeat_miss=wait&closed&(~repeat_event),
        repeat_terminal=repeat_event&terminal,
        first_mask=xp.where(first_risk,first_bits,-1),
        repeat_mask=xp.where(repeat_risk,repeat_bits,-1),
        first_shadow=first_shadow,first_extended=first_risk&(~first_shadow),
        repeat_shadow=repeat_shadow,repeat_extended=repeat_risk&(~repeat_shadow),
        gap=xp.where(repeat_risk,actual_gap,-1),
        mode_cost=xp.where(repeat_risk&geometry,
            2*mode+(cost>=xp.asarray(.5,xp.float32)).astype(xp.int32),-1),
        pre_reason=pick['reason'],post_reason=result['reason'],
        terminal=terminal,refresh=refresh,post_coherent=post_coherent,
        bad=result['bad']|(first_event!=first_eligible)|(repeat_event!=repeat_eligible)|
            (wait&((shadow_gap<30)|(shadow_gap>32))))
# H4FFE2_HELD_CORE_END

# H4FFE2_ENV_SCHEMA_BEGIN
# V2-only fields. Stable per-lane carry through natural episode boundaries.
FFE2_LABEL_KEYS=('assigned','first_due','first_risk','first_eligible','first_event',
    'first_terminal','first_miss','repeat_opened','repeat_wait','repeat_risk',
    'repeat_eligible','repeat_event','repeat_miss','repeat_terminal','first_mask',
    'repeat_mask','first_shadow','first_extended','repeat_shadow','repeat_extended',
    'gap','mode_cost','pre_reason','post_reason','terminal','refresh','post_coherent','bad')
FFE2_ROW_FIELDS=tuple('ffe2_'+k for k in FFE2_LABEL_KEYS)+(
    'ffe2_shadow_before','ffe2_shadow_after')
FFE2_CARRY_FIELDS=('ffe2_shadow_gap',)

def ffe2_env_init(xp):
    negative={'first_mask','repeat_mask','gap','mode_cost'}
    row={'ffe2_'+k:xp.asarray(-1 if k in negative else 0,xp.int32)
         for k in FFE2_LABEL_KEYS}
    row.update({k:xp.asarray(-1,xp.int32) for k in (
        'ffe2_shadow_gap','ffe2_shadow_before','ffe2_shadow_after')})
    return row
# H4FFE2_ENV_SCHEMA_END

# H4SSRPC_HELD_ENV_BEGIN
# Fixed-gap derivation of FFE2; original functions are retained.
def ssrpc_pre(xp,old,*,inclusive,hard_draw,band_draw,cell_draw,due_draw,boundary,nominal,
            count,coherent,source_step,mode,cost,geometry_valid):
    """Pure pre-injection candidate decision; no RNG/physics/reward calls.

    Caller must use independent fold_in subkeys and preserve original draws.
    coherent means exact zero native/actor/critic command and live prestate.
    mode is actual r12_cost loaded support: 0/1/2/3, not binary contact.
    All input shapes/dtypes and actual transition conservation need a separate
    batch guard. This function alone is NOT a production scheduler.
    """
    i=xp.int32
    k=old['k'];initial=k==0
    hard=xp.where(initial,hard_draw,old['hard']).astype(i)
    band=xp.where(initial,band_draw,old['band']).astype(i)
    cell=xp.where(initial,cell_draw,old['cell']).astype(i)
    due=xp.where(initial,due_draw,old['due']).astype(i)
    phase=xp.where(initial,FFE_FIRST,old['phase']).astype(i)
    repeat_due=old['repeat_due']
    target_mode=xp.where(cell<2,cell+1,3).astype(i)
    direction=(2+cell%2).astype(i)
    # Hard assignment remains fixed; original broad can return only at a
    # natural boundary strictly after closure of this opportunity.
    block=(hard>0)&xp.where(initial,True,xp.where(
        boundary&(phase==FFE_CLOSED),False,old['block']>0))
    active=block&(phase!=FFE_CLOSED)
    first=active&(phase==FFE_FIRST)
    wait=active&(phase==FFE_REPEAT)
    at_due=k>=due
    geometry_ok=geometry_valid&xp.isfinite(cost)&(cost>=0)&(cost<=1)&(mode>=0)&(mode<=3)
    assignment_ok=((hard==0)|(hard==1))&(band>=0)&(band<4)&(cell>=0)&(cell<4)&(
        due>=40+160*band)&(due<=80+160*band)
    bad=(k<0)|(k>=640)|(~assignment_ok)|(
        active&(~geometry_ok)&((first&at_due)|wait))
    # Reset before T retains slot; reset at/after T or during repeat closes.
    boundary_miss=active&boundary&((first&at_due)|wait)
    nominal_miss=active&nominal&((first&at_due)|wait)
    coherence_miss=wait&(~coherent | (source_step>=500))
    first_end=112+160*band
    repeat_end=repeat_due
    expired=(first&(k>first_end))|(wait&(k>repeat_end))
    eligible=active&(~boundary)&(~nominal)&coherent&geometry_ok&(source_step<=459)&(~bad)
    first_event=first&at_due&(k<=first_end)&eligible&(count==0)&(mode>0)
    # Repeat clock source limit is the command refresh boundary, not the
    # first-event <=459 guard; at most 40 ticks after the first event.
    repeat_event=wait&(k==repeat_due)&(~boundary)&(~nominal)&(
        coherent)&(source_step<500)&geometry_ok&(~bad)&(count==1)&(
        mode==target_mode)&(inclusive | (cost>=xp.asarray(.5,xp.float32)))
    reason=xp.where(bad,FFE_MISS_INVALID,xp.where(boundary_miss,FFE_MISS_BOUNDARY,
        xp.where(nominal_miss,FFE_MISS_NOMINAL,xp.where(coherence_miss,FFE_MISS_COHERENCE,
        xp.where(expired,FFE_MISS_EXPIRED,0))))).astype(i)
    first_event=first_event&(reason==0)
    repeat_event=repeat_event&(reason==0)
    # On the final allowed tick, a miss closes immediately, not a tick late.
    deadline_miss=(first&(k==first_end)&(~first_event))|(wait&(k==repeat_end)&(~repeat_event))
    reason=xp.where((reason==0)&deadline_miss,FFE_MISS_EXPIRED,reason).astype(i)
    phase=xp.where((reason>0)|repeat_event,FFE_CLOSED,phase).astype(i)
    state=dict(k=k,hard=hard,band=band,cell=cell,due=due,phase=phase,
               repeat_due=repeat_due,block=block.astype(i))
    return state,dict(first=first_event,repeat=repeat_event,direction=direction,
                      block=block,bad=bad,reason=reason,target_mode=target_mode)

def ssrpc_post(xp,state,pre,*,gap,post_coherent,terminal,refresh):
    """Advance once per actual action; preserve actual event even if terminal."""
    i=xp.int32
    k=state['k'];first=pre['first'];repeat=pre['repeat']
    first_or_wait=first|(state['phase']==FFE_REPEAT)
    gap_bad=first&((gap<30)|(gap>32))
    closure=first_or_wait&(terminal|refresh|(~post_coherent))
    # A terminal first event still counts but never opens a repeat.
    reason=xp.where(gap_bad,FFE_MISS_INVALID,xp.where(closure,
        xp.where(terminal,FFE_MISS_TERMINAL,FFE_MISS_COHERENCE),pre['reason'])).astype(i)
    phase=xp.where(closure|gap_bad|repeat,FFE_CLOSED,
        xp.where(first,FFE_REPEAT,state['phase'])).astype(i)
    # If a hard slot is due but terminal occurred on its non-event row, close.
    due_terminal=(state['hard']>0)&(phase==FFE_FIRST)&(k>=state['due'])&terminal
    reason=xp.where(due_terminal,FFE_MISS_TERMINAL,reason).astype(i)
    phase=xp.where(due_terminal,FFE_CLOSED,phase).astype(i)
    out=dict(state,k=(k+1).astype(i),phase=phase,
             repeat_due=xp.where(first,k+35,state['repeat_due']).astype(i))
    return out,dict(reason=reason,bad=pre['bad']|gap_bad)

def ssrpc_risk_labels(xp,old,pre,pick,nxt,result,*,inclusive,boundary,nominal,count,
                     coherent,source_step,mode,cost,geometry_valid,
                     shadow_gap,terminal,refresh,post_coherent):
    """Passive labels on an actual pre/post transition; shadow_gap is the
    sampled V1 gap latched at the actual first event (not current RNG draw).
    No state mutation, no policy feedback and no extra native calls.
    """
    k=old['k']; initial=k==0
    phase=xp.where(initial,FFE_FIRST,old['phase'])
    active=pick['block']&(phase!=FFE_CLOSED)
    first=active&(phase==FFE_FIRST)
    wait=active&(phase==FFE_REPEAT)
    due=pre['due']; first_end=112+160*pre['band']
    actual_gap=k-(old['repeat_due']-35)
    first_risk=first&(k>=due)&(k<=first_end)
    repeat_risk=wait&(actual_gap==35)
    geometry=geometry_valid&xp.isfinite(cost)&(cost>=0)&(cost<=1)&(mode>=0)&(mode<=3)
    def bits(predicates):
        out=xp.zeros_like(k,dtype=xp.int32)
        for index,passed in enumerate(predicates):
            out=out|((~passed).astype(xp.int32)<<index)
        return out
    common=(~boundary,~nominal,coherent)
    first_bits=bits(common+(source_step<=459,geometry,count==0,mode>0))
    repeat_bits=bits(common+(source_step<500,geometry,count==1,
                              mode==pick['target_mode'],inclusive | (cost>=xp.asarray(.5,xp.float32))))
    first_event=pick['first']; repeat_event=pick['repeat']
    first_eligible=first_risk&(first_bits==0)&(~pick['bad'])
    repeat_eligible=repeat_risk&(repeat_bits==0)&(~pick['bad'])
    closed=nxt['phase']==FFE_CLOSED
    first_shadow=first_risk&(k<=due+8)
    repeat_shadow=repeat_risk&(actual_gap>=shadow_gap)&(actual_gap<=shadow_gap+8)
    return dict(
        assigned=(pre['hard']==1)&initial,
        first_due=first&(k==due),first_risk=first_risk,
        first_eligible=first_eligible,first_event=first_event,
        first_terminal=first_event&terminal,
        first_miss=first&closed&(~first_event),
        repeat_opened=first_event&(nxt['phase']==FFE_REPEAT),
        repeat_wait=wait,repeat_risk=repeat_risk,
        repeat_eligible=repeat_eligible,repeat_event=repeat_event,
        repeat_miss=wait&closed&(~repeat_event),
        repeat_terminal=repeat_event&terminal,
        first_mask=xp.where(first_risk,first_bits,-1),
        repeat_mask=xp.where(repeat_risk,repeat_bits,-1),
        first_shadow=first_shadow,first_extended=first_risk&(~first_shadow),
        repeat_shadow=repeat_shadow,repeat_extended=repeat_risk&(~repeat_shadow),
        gap=xp.where(repeat_risk,actual_gap,-1),
        mode_cost=xp.where(repeat_risk&geometry,
            2*mode+(cost>=xp.asarray(.5,xp.float32)).astype(xp.int32),-1),
        pre_reason=pick['reason'],post_reason=result['reason'],
        terminal=terminal,refresh=refresh,post_coherent=post_coherent,
        bad=result['bad']|(first_event!=first_eligible)|(repeat_event!=repeat_eligible)|
            (wait&((shadow_gap<30)|(shadow_gap>32))))

def ssrpc_pre_ctrl(xp,old,**kw):
    return ssrpc_pre(xp,old,inclusive=False,**kw)

def ssrpc_pre_cand(xp,old,**kw):
    return ssrpc_pre(xp,old,inclusive=True,**kw)

def ssrpc_labels_ctrl(xp,old,pre,pick,nxt,result,**kw):
    return ssrpc_risk_labels(xp,old,pre,pick,nxt,result,inclusive=False,**kw)

def ssrpc_labels_cand(xp,old,pre,pick,nxt,result,**kw):
    return ssrpc_risk_labels(xp,old,pre,pick,nxt,result,inclusive=True,**kw)

class SSRPCCoverageJoystick(CoverageJoystick):
    """Opt-in matched arm; historical FFE authorizations do not apply."""
    def __init__(self, *, arm, config):
        if type(arm) is not str or arm not in ('ctrl','cand'):
            raise ValueError('SSRPC_ARM')
        self.__ssrpc_arm=arm
        super().__init__(stage='easy_repeat_support',config=config,h4_arm='control',ffe2=True)

    @property
    def ssrpc_arm(self):
        return self.__ssrpc_arm
# H4SSRPC_HELD_ENV_END
