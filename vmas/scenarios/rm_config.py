#  Copyright (c) 2026.
#  RM 3v3 simplified rules v1.2 — single source of truth for all constants.
#
#  规则书：rm_simplified_rules_v1.1.md（内容为 v1.2 实现修订版）
#  本文件只放常量，不放逻辑；改规则先改这里，再同步规则书。
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


def _rect(x0: float, y0: float, x1: float, y1: float) -> Tuple[Tuple[float, float], ...]:
    """CCW rectangle vertices."""
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


@dataclass
class RmConfig:
    # ------------------------------------------------------------------
    # 时间 / 场地
    # ------------------------------------------------------------------
    dt: float = 0.1
    episode_time: float = 180.0          # 1800 frames
    field_hx: float = 6.0                # wall at x = ±6.0
    field_hy: float = 4.0                # wall at y = ±4.0
    bound_x: float = 5.7                 # robot center legal range
    bound_y: float = 3.7

    # ------------------------------------------------------------------
    # 机器人
    # ------------------------------------------------------------------
    n_agents: int = 6
    n_per_team: int = 3
    robot_radius: float = 0.30           # collision radius
    occlusion_radius: float = 0.35       # line-of-sight radius
    mass: float = 1.0
    v_max: float = 3.0
    a_max: float = 5.0
    hp_max: float = 325.0
    # v1.21: interpret the first two action dims as body/gimbal-frame velocity
    # (+x forward, +y left) instead of canonical world velocity.  The chassis
    # has no separate yaw in this simulator; s.yaw is the single heading, so
    # this also makes "chassis heading == gimbal heading" explicit.  Default
    # False keeps old checkpoints / acceptance demos on world-frame actions.
    relative_actions: bool = False
    # v1.22: one agent per team starts near its own spawn; the other two are
    # random.  The existing 50/50 symmetric/mixed random sampler is reused.
    spawn_anchor_init: bool = True
    spawn_anchor_jitter: float = 0.45

    # Training-only randomisation of the "official fixed" start.  Defaults
    # are exact/no-op so evaluation and legacy runs keep the strict fixed
    # slots.  Enable these on the training CLI for domain randomisation.
    fixed_slot_permute: bool = False
    fixed_jitter: float = 0.0
    fixed_yaw_jitter: float = 0.0

    # ------------------------------------------------------------------
    # 云台 / 小陀螺
    # ------------------------------------------------------------------
    yaw_rate_max: float = math.radians(270.0)   # manual yaw rate, rad/s
    yaw_step_max: float = math.radians(27.0)    # auto-track step per frame
    spin_time: float = 1.0                      # 0 -> 1 time
    spin_follow_rate: float = 1.0               # s per second
    spin_speed_coef: float = 0.5                # v/a reduction at s=1
    spin_cap_coef: float = 0.1                  # p_cap = 1 - 0.1*s
    spin_incoming_coef: float = 0.4             # m_spin = 1 - 0.4*s_target
    spin_tau_coef: float = 0.5                  # tau_eff = 0.4*(1+0.5*s_target)

    # ------------------------------------------------------------------
    # 感知 / 命中
    # ------------------------------------------------------------------
    fov_half: float = math.radians(45.0)        # 90° total
    fov_half_narrow: float = math.radians(40.0) # 80° total when locked
    lock_range: float = 12.0                    # p_dist = 0 at >= 12 m
    vis_samples_obs: int = 16                   # coarse p_vis for observation/lock
    vis_samples_fire: int = 64                  # exact p_vis for firing
    lidar_rays: int = 128
    lidar_range: float = 16.0
    lidar_fov_half: float = math.radians(90.0)  # front 180°
    obstacle_rays: int = 8                     # fixed-map replacement for lidar
    hp_noise_base: float = 10.0
    hp_noise_spin: float = 5.0
    yaw_noise_deg: float = 5.0
    hit_cue_noise_deg: float = 45.0
    tau_base: float = 0.4
    p_floor: float = 0.02

    # ------------------------------------------------------------------
    # 武器 / 热量
    # ------------------------------------------------------------------
    heat_max: float = 160.0
    heat_per_frame: float = 15.0
    cool_per_frame: float = 2.0
    home_cool_mult: float = 2.0          # own-spawn (heal zone) heat cooling multiplier
    damage_per_frame: float = 30.0              # 30 * p_final
    fire_angle: float = math.radians(1.0)

    # ------------------------------------------------------------------
    # 死亡 / 复活 / 回血
    # ------------------------------------------------------------------
    respawn_time: float = 5.0
    respawn_invuln: bool = True        # 复活后 3 s 无敌
    respawn_heat_lock: bool = True     # 无敌结束时把枪口热量置于过热门锁
    invuln_time: float = 3.0
    hp_revive: float = 65.0
    heal_per_frame: float = 8.125               # 81.25/s
    spawn_damage_per_frame: float = 8.125       # enemy spawn damage

    # ------------------------------------------------------------------
    # 碰撞
    # ------------------------------------------------------------------
    wall_foul_speed: float = 2.5
    robot_foul_speed: float = 2.0
    foul_tie_tol: float = 0.1
    corpse_decel: float = 0.5                   # m/s per frame
    corpse_stop_eps: float = 0.01

    # ------------------------------------------------------------------
    # 占点 / 比分
    # ------------------------------------------------------------------
    center_radius: float = 1.5
    score_per_frame: float = 0.1
    kill_score: float = 10.0
    target_score: float = 90.0

    # ------------------------------------------------------------------
    # 固定出生位
    # ------------------------------------------------------------------
    red_slots: Tuple[Tuple[float, float], ...] = (
        (-5.35, 3.30), (-5.35, 2.20), (-4.50, 2.75),
    )
    blue_slots: Tuple[Tuple[float, float], ...] = (
        (5.35, -3.30), (5.35, -2.20), (4.50, -2.75),
    )

    # ------------------------------------------------------------------
    # 掩体 / 启动区（CCW polygons）
    # ------------------------------------------------------------------
    red_l: Tuple[Tuple[float, float], ...] = (
        (-3.50, -1.00), (-3.30, -1.00), (-3.30, 2.86),
        (-1.50, 2.86), (-1.50, 3.06), (-3.50, 3.06),
    )
    blue_l: Tuple[Tuple[float, float], ...] = (
        (3.50, 1.00), (3.30, 1.00), (3.30, -2.86),
        (1.50, -2.86), (1.50, -3.06), (3.50, -3.06),
    )
    red_short: Tuple[Tuple[float, float], ...] = _rect(-1.65, -0.10, -1.50, 1.70)
    blue_short: Tuple[Tuple[float, float], ...] = _rect(1.50, -1.70, 1.65, 0.10)
    red_spawn: Tuple[Tuple[float, float], ...] = _rect(-6.00, 1.50, -3.50, 4.00)
    blue_spawn: Tuple[Tuple[float, float], ...] = _rect(3.50, -4.00, 6.00, -1.50)

    @property
    def covers(self) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
        return (self.red_l, self.blue_l, self.red_short, self.blue_short)

    # ------------------------------------------------------------------
    # 奖励（训练用；裁判规则只看第 13 节）
    # ------------------------------------------------------------------
    k_score_reward: float = 0.01          # 1 score point -> 0.01 reward
    terminal_win: float = 6.0            # normal win base; early win gets time bonus
    terminal_loss: float = -6.0          # normal loss is fixed
    terminal_foul: float = -10.0         # collision-foul base, at-fault agent only
    terminal_foul_win: float = 0.0       # opponent no longer gets a collision bonus
    k_foul_intensity: float = 3.0        # extra penalty * normalized impact severity
    terminal_push_foul: float = -10.0    # body/wall push foul (single-sided)
    terminal_draw: float = 0.0
    terminal_both_200: float = 2.0
    terminal_both_foul: float = -10.0
    k_terminal_time: float = 3.0         # normal win: +3 * remaining_time / episode_time
    k_terminal_margin: float = 3.0       # normal win: +3 * score_margin/target_score
    k_hit_reward: float = 0.05            # full 325 damage -> 0.05
    k_center_reward: float = 0.003         # approach-center shaping, per frame
    k_center_progress: float = 0.02        # potential shaping per metre moved toward centre
    center_sigma: float = 1.5
    # v1.23: anti-camping / time pressure.  Low-HP retreat is exempt.
    k_camp_penalty: float = 0.0005         # base per frame while stationary outside centre
    k_camp_ramp: float = 0.0015            # additional per-frame penalty by end of episode
    k_wall_camp_penalty: float = 0.0005    # legacy, unused
    k_wall_camp_ramp: float = 0.0020       # legacy, unused
    camp_grace_time: float = 8.0           # seconds before generic camping penalty starts
    camp_speed: float = 0.25               # m/s; below this counts as stationary
    wall_camp_speed: float = 0.30
    wall_camp_margin: float = 0.30
    # v1.24 contact-duration penalty: grow while the robot stays in contact
    # with a wall/cover, regardless of episode time.  This ramp is unbounded;
    # a sustained intentional push still terminates after wall_push_time.
    k_wall_contact_penalty: float = 0.001
    k_wall_contact_ramp: float = 0.004
    wall_contact_ramp_time: float = 2.0
    # Body pushing is never useful for teammates, opponents, or corpses:
    # unlike wall contact, this ramp is intentionally NOT capped.  The timer
    # only advances while a living agent is actually moving into another
    # robot/corpse (v_pre toward it); the pushed body does not accumulate the
    # pusher's timer.
    k_body_push_penalty: float = 0.001
    k_body_push_ramp: float = 0.004
    body_push_ramp_time: float = 2.0
    body_push_min: float = 0.3          # only a real push counts
    body_push_time: float = 0.5         # sustained push -> game-ending foul
    body_push_teammate_penalty: float = 0.5
    # Proximity pressure: both sides pay, 60% active / 40% passive.
    proximity_radius: float = 0.65
    # 2026-10: 原 0.0005/0.0005，v59 曾整体 ×10；现按用户要求减半回 0.0025。
    k_proximity_penalty: float = 0.0025
    k_proximity_ramp: float = 0.0025
    proximity_ramp_time: float = 2.0
    proximity_active_share: float = 0.6
    proximity_passive_share: float = 0.4
    # v1.23/1.24: empty game ending.
    stagnation_time: float = 10.0
    stagnation_penalty: float = 5.0        # negative terminal reward for both teams
    stagnation_clearance_discount: float = 0.5  # farther from obstacles -> less negative
    stagnation_clearance_ref: float = 4.0  # metres used to normalize min clearance
    # Empty-game speed relief: the EMA of normalized speed during the empty
    # window; a moving agent keeps up to 50% of the penalty.
    stagnation_speed_discount: float = 0.5
    stagnation_speed_ema_time: float = 2.0
    # Empty-game penalty: the team whose alive centroid is closer to the map
    # centre loses less.  0.5 means the closer side keeps at most 50% more.
    stagnation_center_discount: float = 0.5
    # Per-agent empty-game penalty relief: an agent that left its spawn is
    # punished less than one that stayed at home.  Distance is measured from
    # the centre of its own spawn rectangle and normalised by this reference.
    stagnation_spawn_discount: float = 0.5
    stagnation_spawn_ref: float = 6.0
    # v1.25: dense actual-velocity exit shaping while full HP inside own spawn.
    k_outward_speed_reward: float = 0.002
    # v1.26: sustained pushing into a wall/cover is a game-ending foul.
    wall_push_threshold: float = 0.05
    wall_push_time: float = 1.0
    k_push_foul: float = 1.0
    wall_push_teammate_penalty: float = 0.5

    # survival / low-HP return-home shaping (v1.9)
    k_survival_reward: float = 0.00005      # per alive agent per frame
    k_death_penalty: float = 0.2           # instant dense penalty on the death frame
    k_wall_collision_penalty: float = 0.003   # legacy, unused by v1.22
    k_body_collision_penalty: float = 0.003   # legacy, unused by v1.22
    # v1.22 event contact shaping: impact once, then a small push penalty.
    k_wall_impact: float = 0.02
    k_wall_push: float = 0.001
    k_agent_impact: float = 0.3
    wall_push_min: float = 0.2
    k_fire_heat_penalty: float = 0.001
    k_occupy_reward: float = 0.002            # constant per-frame reward inside centre
    k_kill_reward_pool: float = 0.3           # per kill, split by cumulative damage credit
    k_low_hp_home_reward: float = 0.0005   # per low-HP alive agent per frame, Gaussian near own spawn
    low_hp_threshold: float = 0.5          # fraction of hp_max
    home_sigma: float = 2.0                # Gaussian sigma around own spawn center, metres

    # ------------------------------------------------------------------
    # 训练课程（不是比赛规则）
    # ------------------------------------------------------------------
    init_mode: str = "fixed"              # fixed | symmetric | asymmetric | mixed | curriculum
    obs_fp16: bool = False                # store actor obs / critic state in fp16
    rsi_symmetric_prob: float = 0.5       # mixed: 50% symmetric, 50% asymmetric
    # curriculum: per-reset probability of official fixed start rises from
    # fixed_prob_start to fixed_prob_end over curriculum_steps env steps.
    fixed_prob_start: float = 0.0
    fixed_prob_end: float = 1.0
    curriculum_steps: int = 50_000_000
    # Keep each env's fixed/random role between resets.  The batch-wide
    # quota is still an exact count (e.g. 720/800), but as fixed_prob changes
    # with the curriculum clock the quota is rebalanced: individual envs are
    # promoted/demoted and use the new role from their next reset onward.
    # Sampling the role independently at every reset is the legacy behavior;
    # it makes short fixed episodes contribute far less wall-clock time than
    # their nominal probability.
    curriculum_persistent_roles: bool = True
    rsi_min_sep: float = 1.0              # min center distance between robots
    rsi_time_min: float = 30.0
    rsi_time_max: float = 180.0
    # Late-game RSI: keep episode_time as the full 180 s horizon, but start
    # random episodes with only `rsi_remaining_*` seconds left on the clock.
    # This avoids changing the time-scale of the value function / GAE.
    rsi_late_game: bool = False
    rsi_remaining_min: float = 30.0
    rsi_remaining_max: float = 60.0
    rsi_hp_min: float = 65.0
    rsi_hp_max: float = 325.0
    rsi_spin_max: float = 0.5
    rsi_heat_max: float = 80.0

    def team(self, i: int) -> int:
        return 0 if i < self.n_per_team else 1

    def enemy_indices(self, i: int) -> Tuple[int, int, int]:
        if i < self.n_per_team:
            return (3, 4, 5)
        return (0, 1, 2)

    def teammate_indices(self, i: int) -> Tuple[int, int]:
        if i < self.n_per_team:
            return tuple(j for j in range(3) if j != i)  # type: ignore[return-value]
        return tuple(j for j in range(3, 6) if j != i)    # type: ignore[return-value]

    def own_spawn(self, i: int) -> Tuple[Tuple[float, float], ...]:
        return self.red_spawn if self.team(i) == 0 else self.blue_spawn

    def enemy_spawn(self, i: int) -> Tuple[Tuple[float, float], ...]:
        return self.blue_spawn if self.team(i) == 0 else self.red_spawn


DEFAULT_CONFIG = RmConfig()
