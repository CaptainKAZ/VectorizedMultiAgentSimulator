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

    # ------------------------------------------------------------------
    # 云台 / 小陀螺
    # ------------------------------------------------------------------
    yaw_rate_max: float = math.radians(270.0)   # manual yaw rate, rad/s
    yaw_step_max: float = math.radians(27.0)    # auto-track step per frame
    spin_time: float = 0.5                      # 0 -> 1 time
    spin_follow_rate: float = 1.0 / 0.5         # s per second (2.0/s)
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
    damage_per_frame: float = 30.0              # 30 * p_final
    fire_angle: float = math.radians(1.0)

    # ------------------------------------------------------------------
    # 死亡 / 复活 / 回血
    # ------------------------------------------------------------------
    respawn_time: float = 5.0
    invuln_time: float = 5.0
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
    kill_score: float = 20.0
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
    terminal_win: float = 2.0
    terminal_loss: float = -2.0
    terminal_foul: float = -10.0
    terminal_draw: float = 0.0
    terminal_both_200: float = 2.0
    terminal_both_foul: float = -10.0
    k_hit_reward: float = 0.02            # full 325 damage -> 0.02
    k_center_reward: float = 0.0015        # approach-center shaping, per frame
    center_sigma: float = 1.5
    # survival / low-HP return-home shaping (v1.9)
    k_survival_reward: float = 0.0002      # per alive agent per frame
    k_death_penalty: float = 0.5           # instant dense penalty on the death frame
    k_low_hp_home_reward: float = 0.0005   # per low-HP alive agent per frame, Gaussian near own spawn
    low_hp_threshold: float = 0.5          # fraction of hp_max
    home_sigma: float = 2.0                # Gaussian sigma around own spawn center, metres

    # ------------------------------------------------------------------
    # 训练课程（不是比赛规则）
    # ------------------------------------------------------------------
    init_mode: str = "fixed"              # fixed | symmetric | asymmetric | mixed | curriculum
    rsi_symmetric_prob: float = 0.5       # mixed: 50% symmetric, 50% asymmetric
    # curriculum: per-reset probability of official fixed start rises from
    # fixed_prob_start to fixed_prob_end over curriculum_steps env steps.
    fixed_prob_start: float = 0.0
    fixed_prob_end: float = 1.0
    curriculum_steps: int = 50_000_000
    rsi_min_sep: float = 1.0              # min center distance between robots
    rsi_time_min: float = 30.0
    rsi_time_max: float = 180.0
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
