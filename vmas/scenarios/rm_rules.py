#  Copyright (c) 2026.
#  RM 3v3 rule engine (v1.2).  Pure torch functions; no VMAS imports.
#
#  Frame order follows rules §5:
#    process_action -> spin update -> velocity target -> VMAS integration
#    -> post_step: collision -> visibility/lock -> turret -> fire/heat/damage
#       -> death -> respawn/heal -> occupation/score -> time -> observation
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from vmas.scenarios.rm_config import RmConfig
from vmas.scenarios import rm_geometry as G

Tensor = torch.Tensor
PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1e-6

# ---------------------------------------------------------------------------
# outcome codes (for logs / liveview / eval)
# ---------------------------------------------------------------------------
OUTCOME_ONGOING = 0
OUTCOME_RED_200 = 1
OUTCOME_BLUE_200 = 2
OUTCOME_RED_TIME = 3
OUTCOME_BLUE_TIME = 4
OUTCOME_RED_DMG = 5
OUTCOME_BLUE_DMG = 6
OUTCOME_RED_HP = 7
OUTCOME_BLUE_HP = 8
OUTCOME_RED_FOUL = 9          # blue committed the collision foul
OUTCOME_BLUE_FOUL = 10        # red committed the collision foul
OUTCOME_DRAW_TIME = 20
OUTCOME_DRAW_200 = 21
OUTCOME_DRAW_FOUL = 22

OUTCOME_NAMES = {
    0: "进行中",
    1: "红方先到目标分",
    2: "蓝方先到目标分",
    3: "红方时间胜",
    4: "蓝方时间胜",
    5: "红方伤害tiebreak",
    6: "蓝方伤害tiebreak",
    7: "红方血量tiebreak",
    8: "蓝方血量tiebreak",
    9: "红方胜(蓝方碰撞犯规)",
    10: "蓝方胜(红方碰撞犯规)",
    20: "平局(时间tiebreak)",
    21: "平局(双方同帧200)",
    22: "平局(双方同帧犯规)",
}

# observation layout
SELF_DIM = 24
MATE_DIM = 18          # rel/abs pos + respawn_timer (dead teammate corpse info)
ENEMY_DIM = 17         # rel/abs pos + respawn_timer (visible dead enemy corpse info)
CUE_DIM = 3
LIDAR_DIM = 128
HUD_DIM = 10
OBS_DIM = SELF_DIM + 2 * MATE_DIM + 3 * ENEMY_DIM + CUE_DIM + LIDAR_DIM + HUD_DIM

# global state layout (critic)
G_AGENT_DIM = 22
G_GLOBAL_DIM = 11
GLOBAL_DIM = 6 * G_AGENT_DIM + G_GLOBAL_DIM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def clip_norm(x: Tensor, max_norm: Tensor | float) -> Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    m = torch.as_tensor(max_norm, device=x.device, dtype=x.dtype)
    if m.ndim == 0:
        m = m.expand_as(n)
    else:
        m = m.unsqueeze(-1) if m.ndim == n.ndim - 1 else m
    scale = torch.clamp(m / n.clamp_min(EPS), max=1.0)
    return x * scale


def point_in_polygon_closed(points: Tensor, poly: Tensor) -> Tensor:
    inside = G.point_in_polygon(points, poly)
    _, dist, _, _ = G.nearest_point_on_polygon(points, poly)
    return inside | (dist <= 1e-6)


def _as_idx(idx) -> Optional[List[int]]:
    if idx is None:
        return None
    if isinstance(idx, int):
        return [idx]
    if isinstance(idx, torch.Tensor):
        return idx.reshape(-1).tolist()
    return list(idx)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
@dataclass
class RmState:
    cfg: RmConfig
    device: torch.device
    B: int

    pos: Tensor = field(init=False)
    vel: Tensor = field(init=False)
    v_pre: Tensor = field(init=False)
    yaw: Tensor = field(init=False)
    yaw_rate: Tensor = field(init=False)
    yaw_cmd: Tensor = field(init=False)
    hp: Tensor = field(init=False)
    heat: Tensor = field(init=False)
    overheated: Tensor = field(init=False)
    spin: Tensor = field(init=False)
    spin_cmd: Tensor = field(init=False)
    alive: Tensor = field(init=False)
    invuln: Tensor = field(init=False)
    invuln_timer: Tensor = field(init=False)
    respawn_timer: Tensor = field(init=False)
    target: Tensor = field(init=False)
    target_cmd: Tensor = field(init=False)
    fire_cmd: Tensor = field(init=False)
    p_t: Tensor = field(init=False)
    tau_eff: Tensor = field(init=False)
    last_fired: Tensor = field(init=False)
    hit_cue: Tensor = field(init=False)       # world angle (rad)
    hit_cue_valid: Tensor = field(init=False)

    score: Tensor = field(init=False)
    time: Tensor = field(init=False)
    total_damage: Tensor = field(init=False)
    cum_damage: Tensor = field(init=False)    # (B,6,6) attacker -> victim
    occ: Tensor = field(init=False)
    outcome: Tensor = field(init=False)
    winner: Tensor = field(init=False)        # -1 ongoing, 0 draw, 1 red, 2 blue
    foul_loser: Tensor = field(init=False)    # -1 none, 0 red, 1 blue
    double_foul: Tensor = field(init=False)
    both_200: Tensor = field(init=False)
    done: Tensor = field(init=False)
    just_died: Tensor = field(init=False)
    step_count: int = 0                    # env steps since env creation (curriculum clock)

    obs: Tensor = field(init=False)
    action_mask: Tensor = field(init=False)
    global_state: Tensor = field(init=False)
    reward: Tensor = field(init=False)
    reward_dense: Tensor = field(init=False)
    reward_terminal: Tensor = field(init=False)
    last_hit_damage: Tensor = field(init=False)
    last_damage: Tensor = field(init=False)
    last_p_final: Tensor = field(init=False)    # env p_final of the last fired shot (0 if none)

    # static maps
    team: Tensor = field(init=False)
    enemy_idx: Tensor = field(init=False)     # (6,3)
    mate_idx: Tensor = field(init=False)      # (6,2)
    covers: List[Tensor] = field(default_factory=list)
    cover_edges: Tensor = field(init=False)
    own_spawns: List[Tensor] = field(default_factory=list)
    enemy_spawns: List[Tensor] = field(default_factory=list)
    spawn_pool_red: Optional[Tensor] = field(default=None, init=False)
    spawn_pool_blue: Optional[Tensor] = field(default=None, init=False)

    def __post_init__(self):
        cfg = self.cfg
        d = self.device
        B, N = self.B, cfg.n_agents
        f = lambda *shape: torch.zeros(shape, device=d, dtype=torch.float32)
        b = lambda *shape: torch.zeros(shape, device=d, dtype=torch.bool)
        l = lambda *shape: torch.zeros(shape, device=d, dtype=torch.long)
        self.pos = f(B, N, 2)
        self.vel = f(B, N, 2)
        self.v_pre = f(B, N, 2)
        self.yaw = f(B, N)
        self.yaw_rate = f(B, N)
        self.yaw_cmd = f(B, N)
        self.hp = f(B, N)
        self.heat = f(B, N)
        self.overheated = b(B, N)
        self.spin = f(B, N)
        self.spin_cmd = f(B, N)
        self.alive = b(B, N)
        self.invuln = b(B, N)
        self.invuln_timer = f(B, N)
        self.respawn_timer = f(B, N)
        self.target = l(B, N)
        self.target_cmd = l(B, N)
        self.fire_cmd = b(B, N)
        self.p_t = f(B, N)
        self.tau_eff = f(B, N)
        self.last_fired = b(B, N)
        self.hit_cue = f(B, N)
        self.hit_cue_valid = b(B, N)
        self.score = f(B, 2)
        self.time = f(B)
        self.total_damage = f(B, 2)
        self.cum_damage = f(B, N, N)
        self.occ = f(B, 2)
        self.outcome = l(B)
        self.winner = torch.full((B,), -1, device=d, dtype=torch.long)
        self.foul_loser = torch.full((B,), -1, device=d, dtype=torch.long)
        self.double_foul = b(B)
        self.both_200 = b(B)
        self.done = b(B)
        self.just_died = b(B, N)
        self.obs = f(B, N, OBS_DIM)
        self.action_mask = b(B, N, 4)
        self.global_state = f(B, GLOBAL_DIM)
        self.reward = f(B, N)
        self.reward_dense = f(B, N)
        self.reward_terminal = f(B, N)
        self.last_hit_damage = f(B, N)
        self.last_damage = f(B, N, N)
        self.last_p_final = f(B, N)

        self.team = torch.tensor(
            [cfg.team(i) for i in range(N)], device=d, dtype=torch.long
        )
        self.enemy_idx = torch.tensor(
            [cfg.enemy_indices(i) for i in range(N)], device=d, dtype=torch.long
        )
        self.mate_idx = torch.tensor(
            [cfg.teammate_indices(i) for i in range(N)], device=d, dtype=torch.long
        )
        self.covers = [G.polygon_tensor(p, d) for p in cfg.covers]
        edges = []
        for poly in self.covers:
            n = poly.shape[0]
            for k in range(n):
                a = poly[k]
                b = poly[(k + 1) % n]
                edges.append(torch.stack([a[0], a[1], b[0], b[1]]))
        self.cover_edges = torch.stack(edges, dim=0)
        self.own_spawns = [
            G.polygon_tensor(cfg.own_spawn(i), d) for i in range(N)
        ]
        self.enemy_spawns = [
            G.polygon_tensor(cfg.enemy_spawn(i), d) for i in range(N)
        ]


def make_state(B: int, device: torch.device, cfg: RmConfig) -> RmState:
    return RmState(cfg=cfg, device=device, B=B)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------
def _fixed_reset_rows(s: RmState, rows: List[int]) -> None:
    cfg = s.cfg
    for b in rows:
        for i in range(cfg.n_agents):
            if i < cfg.n_per_team:
                p = cfg.red_slots[i]
            else:
                p = cfg.blue_slots[i - cfg.n_per_team]
            s.pos[b, i] = torch.tensor(p, device=s.device)
            s.yaw[b, i] = math.atan2(-p[1], -p[0])
        s.vel[b].zero_()
        s.v_pre[b].zero_()
        s.yaw_rate[b].zero_()
        s.yaw_cmd[b].zero_()
        s.hp[b].fill_(cfg.hp_max)
        s.heat[b].zero_()
        s.overheated[b].zero_()
        s.spin[b].zero_()
        s.spin_cmd[b].zero_()
        s.alive[b].fill_(True)
        s.invuln[b].zero_()
        s.invuln_timer[b].zero_()
        s.respawn_timer[b].zero_()
        s.target[b].zero_()
        s.target_cmd[b].zero_()
        s.fire_cmd[b].zero_()
        s.p_t[b].zero_()
        s.tau_eff[b].fill_(cfg.tau_base)
        s.last_fired[b].zero_()
        s.hit_cue[b].zero_()
        s.hit_cue_valid[b].zero_()
        s.score[b].zero_()
        s.time[b] = cfg.episode_time
        s.total_damage[b].zero_()
        s.cum_damage[b].zero_()
        s.last_p_final[b].zero_()
        s.occ[b].zero_()
        s.outcome[b] = OUTCOME_ONGOING
        s.winner[b] = -1
        s.foul_loser[b] = -1
        s.double_foul[b] = False
        s.both_200[b] = False
        s.done[b] = False
        s.just_died[b].zero_()


def _build_spawn_pool(s: RmState, team: int, step: float = 0.2) -> Tensor:
    """Grid of legal spawn centres for one team (covers + enemy spawn excluded)."""
    cfg = s.cfg
    d = s.device
    xs = torch.arange(-cfg.bound_x + 0.1, cfg.bound_x - 0.1 + 1e-6, step, device=d)
    ys = torch.arange(-cfg.bound_y + 0.1, cfg.bound_y - 0.1 + 1e-6, step, device=d)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    valid = torch.ones(grid.shape[0], dtype=torch.bool, device=d)
    for poly in s.covers:
        _, dist, _, inside = G.nearest_point_on_polygon(grid, poly)
        valid &= (~inside) & (dist >= cfg.robot_radius + 0.05)
    enemy_spawn = s.enemy_spawns[0 if team == 0 else cfg.n_per_team]
    valid &= ~point_in_polygon_closed(grid, enemy_spawn)
    return grid[valid]


def _spawn_pools(s: RmState) -> Tuple[Tensor, Tensor]:
    if s.spawn_pool_red is None or s.spawn_pool_blue is None:
        s.spawn_pool_red = _build_spawn_pool(s, 0)
        s.spawn_pool_blue = _build_spawn_pool(s, 1)
    return s.spawn_pool_red, s.spawn_pool_blue


def _sample_valid_points(
    s: RmState,
    n: int,
    sym: Tensor,
    gen: torch.Generator,
    max_rounds: int = 100,
) -> Tuple[Tensor, Tensor]:
    """Vectorised rejection sampling for n spawn rows.

    Returns red/blue centres, shape (n, 3, 2).  Valid rows are kept and only the
    invalid ones are resampled, so the loop terminates quickly even at B=8192.
    """
    cfg = s.cfg
    d = s.device
    red_pool, blue_pool = _spawn_pools(s)
    if red_pool.shape[0] == 0 or blue_pool.shape[0] == 0:
        raise RuntimeError("RM spawn pool is empty")
    n_team = cfg.n_per_team
    n_pts = 2 * n_team
    red = torch.zeros(n, n_team, 2, device=d)
    blue = torch.zeros(n, n_team, 2, device=d)
    valid = torch.zeros(n, dtype=torch.bool, device=d)
    eye = torch.eye(n_pts, device=d) * 1e9
    for _ in range(max_rounds):
        need = ~valid
        k = int(need.sum())
        if k == 0:
            break
        ri = torch.randint(red_pool.shape[0], (k, n_team), generator=gen, device=d)
        bi = torch.randint(blue_pool.shape[0], (k, n_team), generator=gen, device=d)
        r = red_pool[ri]
        b = blue_pool[bi]
        b = torch.where(sym[need].view(k, 1, 1), -r, b)
        pts = torch.cat((r, b), dim=1)
        dist = torch.cdist(pts, pts) + eye
        ok = dist.amin(dim=(1, 2)) >= cfg.rsi_min_sep
        if bool(ok.any()):
            idx = need.nonzero(as_tuple=False).squeeze(1)[ok]
            red[idx] = r[ok]
            blue[idx] = b[ok]
            valid[idx] = True
    if not bool(valid.all()):
        # Practically unreachable; keep the reset legal by using the fixed slots.
        bad = (~valid).nonzero(as_tuple=False).squeeze(1)
        red[bad] = torch.tensor(cfg.red_slots, device=d)
        blue[bad] = torch.tensor(cfg.blue_slots, device=d)
    return red, blue


def _random_reset_rows(s: RmState, rows: List[int], mode: str, seed: int = 0) -> None:
    cfg = s.cfg
    n = len(rows)
    if n == 0:
        return
    d = s.device
    gen = torch.Generator(device=d)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)

    if mode == "symmetric":
        sym = torch.ones(n, dtype=torch.bool, device=d)
    elif mode == "asymmetric":
        sym = torch.zeros(n, dtype=torch.bool, device=d)
    elif mode == "mixed":
        sym = torch.rand(n, generator=gen, device=d) < cfg.rsi_symmetric_prob
    else:
        raise ValueError(f"unknown init mode {mode!r}")

    red, blue = _sample_valid_points(s, n, sym, gen)
    pts = torch.cat((red, blue), dim=1)
    yaw_red = torch.rand(n, 3, generator=gen, device=d) * TWO_PI - PI
    yaw_blue_free = torch.rand(n, 3, generator=gen, device=d) * TWO_PI - PI
    yaw_blue = torch.where(sym.view(n, 1), G.wrap_angle(yaw_red + PI), yaw_blue_free)
    yaw = torch.cat((yaw_red, yaw_blue), dim=1)

    hp_red = torch.rand(n, 3, generator=gen, device=d) * (cfg.rsi_hp_max - cfg.rsi_hp_min) + cfg.rsi_hp_min
    hp_blue_free = torch.rand(n, 3, generator=gen, device=d) * (cfg.rsi_hp_max - cfg.rsi_hp_min) + cfg.rsi_hp_min
    hp = torch.cat((hp_red, torch.where(sym.view(n, 1), hp_red, hp_blue_free)), dim=1)

    spin_red = torch.rand(n, 3, generator=gen, device=d) * cfg.rsi_spin_max
    spin_blue_free = torch.rand(n, 3, generator=gen, device=d) * cfg.rsi_spin_max
    spin = torch.cat((spin_red, torch.where(sym.view(n, 1), spin_red, spin_blue_free)), dim=1)

    heat_red = torch.rand(n, 3, generator=gen, device=d) * cfg.rsi_heat_max
    heat_blue_free = torch.rand(n, 3, generator=gen, device=d) * cfg.rsi_heat_max
    heat = torch.cat((heat_red, torch.where(sym.view(n, 1), heat_red, heat_blue_free)), dim=1)

    t = torch.rand(n, generator=gen, device=d) * (cfg.rsi_time_max - cfg.rsi_time_min) + cfg.rsi_time_min

    idx = torch.as_tensor(rows, device=d, dtype=torch.long)
    s.pos[idx] = pts
    s.yaw[idx] = yaw
    s.hp[idx] = hp
    s.spin[idx] = spin
    s.heat[idx] = heat
    s.vel[idx] = 0.0
    s.v_pre[idx] = 0.0
    s.yaw_rate[idx] = 0.0
    s.yaw_cmd[idx] = 0.0
    s.overheated[idx] = False
    s.spin_cmd[idx] = spin
    s.alive[idx] = True
    s.invuln[idx] = False
    s.invuln_timer[idx] = 0.0
    s.respawn_timer[idx] = 0.0
    s.target[idx] = 0
    s.target_cmd[idx] = 0
    s.fire_cmd[idx] = False
    s.p_t[idx] = 0.0
    s.tau_eff[idx] = cfg.tau_base
    s.last_fired[idx] = False
    s.hit_cue[idx] = 0.0
    s.hit_cue_valid[idx] = False
    s.score[idx] = 0.0
    s.time[idx] = t
    s.total_damage[idx] = 0.0
    s.cum_damage[idx] = 0.0
    s.last_p_final[idx] = 0.0
    s.occ[idx] = 0.0
    s.outcome[idx] = OUTCOME_ONGOING
    s.winner[idx] = -1
    s.foul_loser[idx] = -1
    s.double_foul[idx] = False
    s.both_200[idx] = False
    s.done[idx] = False
    s.just_died[idx] = False


def _fixed_prob(s: RmState) -> float:
    """Probability of official fixed start at the current curriculum clock."""
    cfg = s.cfg
    if cfg.curriculum_steps <= 0:
        return float(cfg.fixed_prob_end)
    t = min(1.0, max(0.0, s.step_count / float(cfg.curriculum_steps)))
    return float(
        cfg.fixed_prob_start
        + (cfg.fixed_prob_end - cfg.fixed_prob_start) * t
    )


def reset_state(
    s: RmState,
    idx=None,
    mode: Optional[str] = None,
    seed: int = 0,
) -> None:
    rows = _as_idx(idx)
    if rows is None:
        rows = list(range(s.B))
    mode = mode or s.cfg.init_mode
    if mode == "fixed":
        _fixed_reset_rows(s, rows)
    elif mode == "curriculum":
        # Per-reset choice: official fixed start vs RSI random start.
        # The fixed probability rises linearly from fixed_prob_start to
        # fixed_prob_end over cfg.curriculum_steps env steps.
        gen = torch.Generator(device=s.device)
        gen.manual_seed(int(seed) & 0x7FFFFFFF)
        p = _fixed_prob(s)
        pick = torch.rand(len(rows), generator=gen, device=s.device) < p
        flags = pick.reshape(-1).tolist()
        fixed_rows = [r for r, f in zip(rows, flags) if f]
        random_rows = [r for r, f in zip(rows, flags) if not f]
        if fixed_rows:
            _fixed_reset_rows(s, fixed_rows)
        if random_rows:
            _random_reset_rows(
                s,
                random_rows,
                "mixed",
                seed=seed + (random_rows[0] if random_rows else 0),
            )
    else:
        _random_reset_rows(s, rows, mode, seed=seed + (rows[0] if rows else 0))
    build_observation(s)


# ---------------------------------------------------------------------------
# action decode (called from Scenario.process_action, before VMAS integration)
# ---------------------------------------------------------------------------
def decode_agent_action(
    s: RmState, i: int, raw_u: Tensor, current_vel: Tensor
) -> Tensor:
    """Update spin/target/yaw commands and return the velocity target.

    raw_u: (B,5) [vx, vy, yaw, spin, target]
    current_vel: (B,2)
    """
    cfg = s.cfg
    B = s.B
    dev = s.device
    raw_u = raw_u.to(dev)
    spin_target = torch.clamp(raw_u[:, 3], 0.0, 1.0)
    s.spin_cmd[:, i] = spin_target
    s.target_cmd[:, i] = torch.round(torch.clamp(raw_u[:, 4], 0.0, 3.0)).long()
    s.fire_cmd[:, i] = raw_u[:, 5] > 0.5
    s.yaw_cmd[:, i] = torch.clamp(raw_u[:, 2], -1.0, 1.0)

    alive = s.alive[:, i]
    # v1.2: continuous slew-rate follow of the spin target
    ds = torch.clamp(
        spin_target - s.spin[:, i],
        -cfg.spin_follow_rate * cfg.dt,
        cfg.spin_follow_rate * cfg.dt,
    )
    s.spin[:, i] = torch.clamp(s.spin[:, i] + ds, 0.0, 1.0)
    s.spin[:, i] = torch.where(alive, s.spin[:, i], torch.zeros_like(s.spin[:, i]))

    vmax = cfg.v_max * (1.0 - cfg.spin_speed_coef * s.spin[:, i])
    amax = cfg.a_max * (1.0 - cfg.spin_speed_coef * s.spin[:, i])

    # v1.14: observations are 180-degree canonicalised for blue, so the
    # 2-D velocity intent is canonical too.  Rotate it back to world for blue.
    team_sign = 1.0 if int(s.team[i]) == 0 else -1.0
    v_des = clip_norm(raw_u[:, :2], 1.0) * (team_sign * vmax).unsqueeze(-1)
    v_next = clip_norm(
        current_vel + clip_norm(v_des - current_vel, (amax * cfg.dt).unsqueeze(-1)),
        vmax.unsqueeze(-1),
    )

    # corpse: no control, 5.0 m/s^2 deceleration (0.5 m/s per frame)
    speed = torch.linalg.vector_norm(current_vel, dim=-1, keepdim=True)
    new_speed = torch.clamp(speed - cfg.corpse_decel, min=0.0)
    v_corpse = current_vel * (new_speed / speed.clamp_min(EPS))
    v_corpse = torch.where(new_speed < cfg.corpse_stop_eps, torch.zeros_like(v_corpse), v_corpse)

    v_target = torch.where(alive.unsqueeze(-1), v_next, v_corpse)
    s.v_pre[:, i] = v_target
    s.yaw_cmd[:, i] = torch.where(alive, s.yaw_cmd[:, i], torch.zeros_like(s.yaw_cmd[:, i]))
    s.target_cmd[:, i] = torch.where(alive, s.target_cmd[:, i], torch.zeros_like(s.target_cmd[:, i]))
    return v_target


# ---------------------------------------------------------------------------
# collision
# ---------------------------------------------------------------------------
def _foul_add(foul_team: Tensor, team: int, mask: Tensor) -> None:
    if mask.ndim == 2:
        mask = mask.any(dim=1)
    if mask.any():
        foul_team[:, team] = foul_team[:, team] | mask


def check_fouls(s: RmState) -> Tensor:
    """Check collision fouls from the v_pre snapshot.

    Returns a (B,) bool mask.  This function does not move anything; the
    caller snapshots fouled rows and then calls resolve_contacts() so that
    non-fouled rows in the same batch still get their separation.
    """
    cfg = s.cfg
    dev = s.device
    B, N = s.B, cfg.n_agents
    pos = s.pos
    vel = s.vel
    vpre = s.v_pre
    r = cfg.robot_radius

    foul_team = torch.zeros((B, 2), device=dev, dtype=torch.bool)

    # --- 1. wall fouls -------------------------------------------------
    wall_foul = torch.zeros((B, N), device=dev, dtype=torch.bool)
    for axis, lo, hi in ((0, -cfg.bound_x, cfg.bound_x), (1, -cfg.bound_y, cfg.bound_y)):
        at_lo = pos[..., axis] <= lo + 1e-6
        at_hi = pos[..., axis] >= hi - 1e-6
        imp_lo = torch.clamp(-vpre[..., axis], min=0.0)
        imp_hi = torch.clamp(vpre[..., axis], min=0.0)
        wall_foul = wall_foul | (at_lo & (imp_lo >= cfg.wall_foul_speed))
        wall_foul = wall_foul | (at_hi & (imp_hi >= cfg.wall_foul_speed))
    for t in (0, 1):
        _foul_add(foul_team, t, wall_foul & (s.team == t).unsqueeze(0))

    # --- 2. cover fouls ------------------------------------------------
    cover_foul = torch.zeros((B, N), device=dev, dtype=torch.bool)
    for poly in s.covers:
        q, dist, normal, inside = G.nearest_point_on_polygon(
            pos.reshape(B * N, 2), poly
        )
        q = q.view(B, N, 2)
        dist = dist.view(B, N)
        normal = normal.view(B, N, 2)
        inside = inside.view(B, N)
        overlap = inside | (dist < r)
        impact = torch.clamp(-torch.sum(vpre * normal, dim=-1), min=0.0)
        cover_foul = cover_foul | (overlap & (impact >= cfg.wall_foul_speed))
    for t in (0, 1):
        _foul_add(foul_team, t, cover_foul & (s.team == t).unsqueeze(0))

    # --- 3. robot / corpse fouls ---------------------------------------
    for i in range(N):
        for j in range(i + 1, N):
            rel = pos[:, j] - pos[:, i]
            d = torch.linalg.vector_norm(rel, dim=-1)
            n = rel / d.clamp_min(EPS).unsqueeze(-1)
            n = torch.where((d > EPS).unsqueeze(-1), n, torch.tensor([1.0, 0.0], device=dev))
            v_imp = torch.sum((vpre[:, i] - vpre[:, j]) * n, dim=-1)
            impact = torch.clamp(v_imp, min=0.0)
            overlap = d < 2.0 * r
            ai = s.alive[:, i]
            aj = s.alive[:, j]
            both_alive = ai & aj
            one_alive = ai ^ aj
            if (overlap & both_alive & (impact >= cfg.robot_foul_speed)).any():
                c_i = torch.clamp(torch.sum(vpre[:, i] * n, dim=-1), min=0.0)
                c_j = torch.clamp(torch.sum(-vpre[:, j] * n, dim=-1), min=0.0)
                tie = torch.abs(c_i - c_j) <= cfg.foul_tie_tol
                i_active = (~tie) & (c_i > c_j)
                j_active = (~tie) & (c_j > c_i)
                same = s.team[i] == s.team[j]
                foul_i = overlap & both_alive & (impact >= cfg.robot_foul_speed) & i_active
                foul_j = overlap & both_alive & (impact >= cfg.robot_foul_speed) & j_active
                foul_tie = overlap & both_alive & (impact >= cfg.robot_foul_speed) & tie
                ti, tj = int(s.team[i]), int(s.team[j])
                _foul_add(foul_team, ti, foul_i)
                _foul_add(foul_team, tj, foul_j)
                if same:
                    _foul_add(foul_team, ti, foul_tie)
                else:
                    _foul_add(foul_team, ti, foul_tie)
                    _foul_add(foul_team, tj, foul_tie)
            # one alive vs corpse: only the alive robot's own normal component
            if (overlap & one_alive).any():
                alive_is_i = ai & (~aj)
                alive_is_j = aj & (~ai)
                imp_i = torch.clamp(torch.sum(vpre[:, i] * n, dim=-1), min=0.0)
                imp_j = torch.clamp(torch.sum(-vpre[:, j] * n, dim=-1), min=0.0)
                foul_i = overlap & alive_is_i & (imp_i >= cfg.robot_foul_speed)
                foul_j = overlap & alive_is_j & (imp_j >= cfg.robot_foul_speed)
                _foul_add(foul_team, int(s.team[i]), foul_i)
                _foul_add(foul_team, int(s.team[j]), foul_j)

    # --- 4. outcomes ---------------------------------------------------
    foul_any = foul_team.any(dim=1)
    both = foul_team[:, 0] & foul_team[:, 1]
    red_only = foul_team[:, 0] & (~foul_team[:, 1])
    blue_only = foul_team[:, 1] & (~foul_team[:, 0])
    s.done = s.done | foul_any
    s.double_foul = s.double_foul | both
    s.outcome = torch.where(both, torch.full_like(s.outcome, OUTCOME_DRAW_FOUL), s.outcome)
    s.winner = torch.where(both, torch.zeros_like(s.winner), s.winner)
    s.outcome = torch.where(red_only, torch.full_like(s.outcome, OUTCOME_BLUE_FOUL), s.outcome)
    s.winner = torch.where(red_only, torch.full_like(s.winner, 2), s.winner)
    s.outcome = torch.where(blue_only, torch.full_like(s.outcome, OUTCOME_RED_FOUL), s.outcome)
    s.winner = torch.where(blue_only, torch.full_like(s.winner, 1), s.winner)
    s.foul_loser = torch.where(
        both, torch.full_like(s.foul_loser, -1),
        torch.where(red_only, torch.zeros_like(s.foul_loser),
                    torch.where(blue_only, torch.ones_like(s.foul_loser), s.foul_loser)),
    )
    return foul_any


def resolve_contacts(s: RmState) -> None:
    """Two rounds of non-foul separation (round 1 also corrects velocity)."""
    cfg = s.cfg
    dev = s.device
    B, N = s.B, cfg.n_agents
    pos = s.pos
    vel = s.vel
    r = cfg.robot_radius

    for round_idx in (0, 1):
        # walls: VMAS already clamped positions; zero the inward normal velocity
        if round_idx == 0:
            vx = vel[..., 0]
            vy = vel[..., 1]
            left = pos[..., 0] <= -cfg.bound_x + 1e-6
            right = pos[..., 0] >= cfg.bound_x - 1e-6
            bottom = pos[..., 1] <= -cfg.bound_y + 1e-6
            top = pos[..., 1] >= cfg.bound_y - 1e-6
            vx = torch.where(left, torch.clamp(vx, min=0.0), vx)
            vx = torch.where(right, torch.clamp(vx, max=0.0), vx)
            vy = torch.where(bottom, torch.clamp(vy, min=0.0), vy)
            vy = torch.where(top, torch.clamp(vy, max=0.0), vy)
            vel[..., 0] = vx
            vel[..., 1] = vy
        # covers
        for poly in s.covers:
            q, dist, normal, inside = G.nearest_point_on_polygon(
                pos.reshape(B * N, 2), poly
            )
            q = q.view(B, N, 2)
            dist = dist.view(B, N)
            normal = normal.view(B, N, 2)
            inside = inside.view(B, N)
            overlap = inside | (dist < r)
            if overlap.any():
                target_pos = q + r * normal
                pos[:] = torch.where(overlap.unsqueeze(-1), target_pos, pos)
                if round_idx == 0:
                    vn = torch.sum(vel * normal, dim=-1, keepdim=True)
                    vel[:] = torch.where(
                        overlap.unsqueeze(-1) & (vn < 0),
                        vel - vn * normal,
                        vel,
                    )
        # robots
        for i in range(N):
            for j in range(i + 1, N):
                rel = pos[:, j] - pos[:, i]
                d = torch.linalg.vector_norm(rel, dim=-1)
                n = rel / d.clamp_min(EPS).unsqueeze(-1)
                n = torch.where((d > EPS).unsqueeze(-1), n, torch.tensor([1.0, 0.0], device=dev))
                overlap = d < 2.0 * r
                if overlap.any():
                    pen = (2.0 * r - d).clamp_min(0.0)
                    pos[:, i] = pos[:, i] - 0.5 * pen.unsqueeze(-1) * n
                    pos[:, j] = pos[:, j] + 0.5 * pen.unsqueeze(-1) * n
                    if round_idx == 0:
                        v_imp = torch.sum((vel[:, i] - vel[:, j]) * n, dim=-1)
                        corr = torch.clamp(v_imp, min=0.0)
                        vel[:, i] = vel[:, i] - 0.5 * corr.unsqueeze(-1) * n
                        vel[:, j] = vel[:, j] + 0.5 * corr.unsqueeze(-1) * n
        pos[..., 0] = torch.clamp(pos[..., 0], -cfg.bound_x, cfg.bound_x)
        pos[..., 1] = torch.clamp(pos[..., 1], -cfg.bound_y, cfg.bound_y)


def resolve_collisions(s: RmState) -> Tensor:
    """Backward-compatible helper: check fouls and resolve contacts if clean."""
    foul = check_fouls(s)
    if not foul.any():
        resolve_contacts(s)
    return foul


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------
def visibility_matrix(
    s: RmState, yaw: Tensor, n_samples: int, locked_target: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (p_vis, in_fov, visible_geom) with shapes (B,6,3).

    visible_geom = in_fov & p_vis > 0, independent of alive/invuln; callers
    combine with alive/invuln for target selection.
    """
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    p_vis = torch.zeros((B, N, 3), device=dev)
    in_fov = torch.zeros((B, N, 3), device=dev, dtype=torch.bool)

    for i in range(N):
        for k in range(3):
            j = int(s.enemy_idx[i, k])
            rel = s.pos[:, j] - s.pos[:, i]
            d = torch.linalg.vector_norm(rel, dim=-1)
            bearing = torch.atan2(rel[:, 1], rel[:, 0])
            err = G.wrap_angle(bearing - yaw[:, i])
            prev = locked_target[:, i]
            half = torch.where(
                (prev == 0) | (prev == k + 1),
                torch.full_like(err, cfg.fov_half),
                torch.full_like(err, cfg.fov_half_narrow),
            )
            in_fov[:, i, k] = err.abs() <= half

            # occluders: every other robot except self and target
            ids = [m for m in range(N) if m != i and m != j]
            circles = torch.stack(
                [torch.cat([s.pos[:, m], torch.full((B, 1), cfg.occlusion_radius, device=dev)], dim=-1)
                 for m in ids],
                dim=1,
            ) if ids else torch.zeros((B, 0, 3), device=dev)
            pv = G.visibility_arc(
                s.pos[:, i], yaw[:, i], s.pos[:, j], cfg.robot_radius,
                circles, s.cover_edges, n_samples,
            )
            p_vis[:, i, k] = torch.where(in_fov[:, i, k], pv, torch.zeros_like(pv))
    visible_geom = in_fov & (p_vis > 0)
    return p_vis, in_fov, visible_geom


# ---------------------------------------------------------------------------
# main post-step
# ---------------------------------------------------------------------------
def _snapshot(s: RmState) -> Dict[str, Tensor]:
    """Clone all per-env state tensors (shape[0] == B)."""
    snap = {}
    for name, value in vars(s).items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == s.B:
            snap[name] = value.clone()
    return snap


def _restore_rows(s: RmState, snap: Dict[str, Tensor], mask: Tensor) -> None:
    if not bool(mask.any()):
        return
    for name, value in snap.items():
        cur = getattr(s, name)
        if torch.is_tensor(cur) and cur.shape == value.shape:
            m = mask.reshape((s.B,) + (1,) * (cur.ndim - 1))
            setattr(s, name, torch.where(m, value, cur))


def _finish(s: RmState, score_before: Tensor, snap, foul_mask, gen) -> None:
    if snap is not None:
        _restore_rows(s, snap, foul_mask)
    _finalize_rewards(s, score_before)
    build_observation(s, generator=gen)


def step_post(s: RmState, generator: Optional[torch.Generator] = None) -> None:
    cfg = s.cfg
    s.step_count += 1
    B, N = s.B, cfg.n_agents
    dev = s.device
    gen = generator

    score_before = s.score.clone()
    s.reward_dense.zero_()
    s.reward_terminal.zero_()
    s.last_fired.zero_()
    s.hit_cue_valid.zero_()
    s.last_hit_damage.zero_()
    s.last_damage.zero_()
    s.last_p_final.zero_()
    s.just_died.zero_()
    s.foul_loser.fill_(-1)
    s.double_foul.zero_()
    s.both_200.zero_()

    # ---- 5.4 collision / foul ---------------------------------------
    # Check fouls for all rows, snapshot the fouled rows, then resolve
    # contacts for the non-fouled rows.  This keeps vectorised rows independent
    # even when only one row ends by a collision foul.
    foul_any = check_fouls(s)
    snap = _snapshot(s) if bool(foul_any.any()) else None
    resolve_contacts(s)
    if bool(foul_any.all()):
        _finish(s, score_before, snap, foul_any, gen)
        return

    # ---- 5.5/5.6/5.7 visibility, lock, turret ------------------------
    prev_target = s.target.clone()
    p_vis_pre, in_fov_pre, visible_pre = visibility_matrix(
        s, s.yaw, cfg.vis_samples_obs, prev_target
    )

    new_target = torch.zeros_like(s.target)
    for k in range(1, 4):
        # v1.13: invulnerable enemies may be selected and pre-aimed; dead ones not.
        alive_k = s.alive[:, s.enemy_idx[:, k - 1]]
        valid = (
            (s.target_cmd == k)
            & visible_pre[:, :, k - 1]
            & alive_k
        )
        new_target = torch.where(valid, torch.full_like(new_target, k), new_target)
    lost = (new_target == 0) | (new_target != prev_target)
    s.p_t = torch.where(lost, torch.zeros_like(s.p_t), s.p_t)
    s.target = new_target

    # turret: auto-track if locked, otherwise manual
    yaw_old = s.yaw.clone()
    bearing_to_target = torch.zeros_like(s.yaw)
    for i in range(N):
        for k in range(3):
            j = int(s.enemy_idx[i, k])
            m = s.target[:, i] == k + 1
            rel = s.pos[:, j] - s.pos[:, i]
            bearing_to_target[:, i] = torch.where(
                m,
                torch.atan2(rel[:, 1], rel[:, 0]),
                bearing_to_target[:, i],
            )
        locked = s.target[:, i] != 0
        err = G.wrap_angle(bearing_to_target[:, i] - s.yaw[:, i])
        step = torch.clamp(err, -cfg.yaw_step_max, cfg.yaw_step_max)
        yaw_auto = G.wrap_angle(s.yaw[:, i] + step)
        yaw_manual = G.wrap_angle(s.yaw[:, i] + s.yaw_cmd[:, i] * cfg.yaw_rate_max * cfg.dt)
        yaw_next = torch.where(locked & s.alive[:, i], yaw_auto, yaw_manual)
        yaw_next = torch.where(s.alive[:, i], yaw_next, s.yaw[:, i])
        s.yaw[:, i] = yaw_next
    s.yaw_rate = G.wrap_angle(s.yaw - yaw_old) / cfg.dt

    # post-turn check: lock only survives if the target is still visible
    _, _, visible_post = visibility_matrix(
        s, s.yaw, cfg.vis_samples_obs, s.target
    )
    lock_alive = torch.zeros((B, N), device=dev, dtype=torch.bool)
    for i in range(N):
        alive_i = torch.zeros((B,), device=dev, dtype=torch.bool)
        for k in range(3):
            j = int(s.enemy_idx[i, k])
            # v1.13: invuln keeps the lock (pre-aim), death drops it.
            alive_i = alive_i | (
                (s.target[:, i] == k + 1)
                & visible_post[:, i, k]
                & s.alive[:, j]
            )
        lock_alive[:, i] = alive_i
    s.target = torch.where(lock_alive, s.target, torch.zeros_like(s.target))
    s.p_t = torch.where(lock_alive, s.p_t, torch.zeros_like(s.p_t))

    # ---- 5.8 heat / fire / damage -----------------------------------
    # cool all alive, then unlock overheat
    s.heat = torch.where(s.alive, torch.clamp(s.heat - cfg.cool_per_frame, min=0.0), s.heat)
    s.overheated = s.overheated & ~(s.alive & (s.heat <= 0.0))

    damage = torch.zeros((B, N, N), device=dev)
    for i in range(N):
        locked = s.target[:, i] != 0
        # compute p_ideal for every locked shooter
        for k in range(3):
            m = s.target[:, i] == k + 1
            if not m.any():
                continue
            j = int(s.enemy_idx[i, k])
            rel = s.pos[:, j] - s.pos[:, i]
            d = torch.linalg.vector_norm(rel, dim=-1)
            d_safe = torch.clamp(d, min=2.0 * cfg.robot_radius)
            # full 64-point visibility for firing
            ids = [q for q in range(N) if q != i and q != j]
            circles = torch.stack(
                [torch.cat([s.pos[:, q], torch.full((B, 1), cfg.occlusion_radius, device=dev)], dim=-1)
                 for q in ids], dim=1)
            pv = G.visibility_arc(
                s.pos[:, i], s.yaw[:, i], s.pos[:, j], cfg.robot_radius,
                circles, s.cover_edges, cfg.vis_samples_fire,
            )
            p_dist = torch.where(
                d <= 1.0, torch.ones_like(d),
                torch.where(d < cfg.lock_range, 1.0 - ((d - 1.0) / (cfg.lock_range - 1.0)) ** 2, torch.zeros_like(d)),
            )
            p_self = 1.0 - 0.25 * torch.clamp(
                torch.linalg.vector_norm(s.vel[:, i], dim=-1) / cfg.v_max, 0.0, 1.0
            )
            v_rel = s.vel[:, j] - s.vel[:, i]
            cross = rel[:, 0] * v_rel[:, 1] - rel[:, 1] * v_rel[:, 0]
            omega = torch.abs(cross) / (d_safe ** 2)
            p_motion = 1.0 / (1.0 + (omega / 0.75) ** 2)
            p_base = p_dist * p_self * p_motion * pv
            p_base = torch.where(
                (d >= cfg.lock_range) | (pv <= 0.0),
                torch.zeros_like(p_base),
                torch.clamp(p_base, min=cfg.p_floor),
            )
            p_cap = 1.0 - cfg.spin_cap_coef * s.spin[:, i]
            p_ideal = torch.minimum(p_base, p_cap)
            tau_eff = cfg.tau_base * (1.0 + cfg.spin_tau_coef * s.spin[:, j])
            alpha = 1.0 - torch.exp(-cfg.dt / tau_eff)
            s.tau_eff[:, i] = torch.where(m, tau_eff, s.tau_eff[:, i])
            p_t_old = s.p_t[:, i]
            p_t_new = torch.where(m, p_t_old + alpha * (p_ideal - p_t_old), p_t_old)
            s.p_t[:, i] = p_t_new
            p_final = p_t_new * (1.0 - cfg.spin_incoming_coef * s.spin[:, j])

            bearing = torch.atan2(rel[:, 1], rel[:, 0])
            angle_err = G.wrap_angle(bearing - s.yaw[:, i])
            fire = (
                m
                & s.alive[:, i]
                & (~s.invuln[:, i])
                & (~s.overheated[:, i])
                & (s.heat[:, i] < cfg.heat_max)
                & s.alive[:, j]
                & (~s.invuln[:, j])
                & (pv > 0.0)
                & (d < cfg.lock_range)
                & (angle_err.abs() < cfg.fire_angle)
                & s.fire_cmd[:, i]
            )
            if fire.any():
                s.heat[:, i] = torch.where(fire, torch.clamp(s.heat[:, i] + cfg.heat_per_frame, max=cfg.heat_max), s.heat[:, i])
                s.overheated[:, i] = s.overheated[:, i] | (fire & (s.heat[:, i] >= cfg.heat_max))
                dmg = torch.where(fire, cfg.damage_per_frame * p_final, torch.zeros_like(p_final))
                damage[:, i, j] = damage[:, i, j] + dmg
                s.last_fired[:, i] = s.last_fired[:, i] | fire
                s.last_p_final[:, i] = torch.where(fire, p_final, s.last_p_final[:, i])

    # spawn damage: enemy spawn, alive, not invuln; no killer, no cue
    spawn_in = torch.zeros((B, N), device=dev, dtype=torch.bool)
    for i in range(N):
        spawn_in[:, i] = point_in_polygon_closed(s.pos[:, i], s.enemy_spawns[i])
    spawn_dmg = spawn_in & s.alive & (~s.invuln)
    s.hp = torch.where(spawn_dmg, torch.clamp(s.hp - cfg.spawn_damage_per_frame, min=0.0), s.hp)
    s.last_hit_damage = damage.sum(dim=2)
    s.last_damage = damage.clone()

    # hit direction cue: strongest attacker of this frame, +/-45 deg, 1 frame
    total_dmg = damage.sum(dim=1)
    for j in range(N):
        m = total_dmg[:, j] > 1e-9
        if not m.any():
            continue
        col = damage[:, :, j]
        atk = torch.argmax(col, dim=1)
        best = col.gather(1, atk.unsqueeze(1)).squeeze(1)
        valid = m & (best > 1e-9)
        atk_pos = s.pos.gather(1, atk.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
        vic = s.pos[:, j]
        bearing = torch.atan2(atk_pos[:, 1] - vic[:, 1], atk_pos[:, 0] - vic[:, 0])
        noise = (torch.rand((B,), device=dev, generator=gen) * 2.0 - 1.0) * math.radians(45.0)
        s.hit_cue[:, j] = torch.where(valid, G.wrap_angle(bearing + noise), s.hit_cue[:, j])
        s.hit_cue_valid[:, j] = valid

    # ---- 5.9 death / kills ------------------------------------------
    hp_before = s.hp.clone()
    total_dmg = damage.sum(dim=1)  # (B,6)
    actual = torch.minimum(total_dmg, hp_before)
    s.hp = torch.clamp(hp_before - total_dmg, min=0.0)
    died = s.alive & (s.hp <= 0.0)
    just_died = died.clone()

    kill_add = torch.zeros((B, 2), device=dev)
    for j in range(N):
        if not (total_dmg[:, j] > 0).any():
            continue
        col = damage[:, :, j]  # attackers x batch
        killer = torch.argmax(col, dim=1)
        best = col.gather(1, killer.unsqueeze(1)).squeeze(1)
        kill = died[:, j] & (best > 0)
        # tie -> smaller index: argmax already returns the first max
        kt = s.team[killer]
        for t in (0, 1):
            kill_add[:, t] = kill_add[:, t] + (kill & (kt == t)).float() * cfg.kill_score
        # actual damage credit, proportional to raw damage
        share = col / total_dmg[:, j].clamp_min(EPS).unsqueeze(1)
        credit = share * actual[:, j].unsqueeze(1)
        for t in (0, 1):
            s.total_damage[:, t] = s.total_damage[:, t] + credit[:, (s.team == t)].sum(dim=1)
    s.cum_damage = s.cum_damage + damage
    s.score = s.score + kill_add

    # death state
    s.just_died = just_died
    if died.any():
        s.alive = s.alive & (~died)
        s.hp = torch.where(died, torch.zeros_like(s.hp), s.hp)
        s.heat = torch.where(died, torch.zeros_like(s.heat), s.heat)
        s.overheated = s.overheated & (~died)
        s.spin = torch.where(died, torch.zeros_like(s.spin), s.spin)
        s.p_t = torch.where(died, torch.zeros_like(s.p_t), s.p_t)
        s.respawn_timer = torch.where(died, torch.full_like(s.respawn_timer, cfg.respawn_time), s.respawn_timer)
        s.target = torch.where(died, torch.zeros_like(s.target), s.target)
        # clear all locks that pointed at a dead target (including the dead
        # agent's own lock) and reset p_t for those shooters
        target_died = died.clone()
        for i in range(N):
            for k in range(3):
                j = int(s.enemy_idx[i, k])
                target_died[:, i] = target_died[:, i] | (
                    (s.target[:, i] == k + 1) & died[:, j]
                )
        s.target = torch.where(target_died, torch.zeros_like(s.target), s.target)
        s.p_t = torch.where(target_died, torch.zeros_like(s.p_t), s.p_t)
        s.hit_cue_valid = s.hit_cue_valid & (~died)
        s.hit_cue = torch.where(died, torch.zeros_like(s.hit_cue), s.hit_cue)
        s.last_fired = s.last_fired & (~died)

    # immediate 200 check after kills
    reached = s.score >= cfg.target_score - 1e-6
    if reached.any():
        both = reached[:, 0] & reached[:, 1]
        red = reached[:, 0] & (~both)
        blue = reached[:, 1] & (~both)
        s.done = s.done | reached.any(dim=1)
        s.both_200 = s.both_200 | both
        s.outcome = torch.where(both, torch.full_like(s.outcome, OUTCOME_DRAW_200), s.outcome)
        s.winner = torch.where(both, torch.zeros_like(s.winner), s.winner)
        s.outcome = torch.where(red, torch.full_like(s.outcome, OUTCOME_RED_200), s.outcome)
        s.winner = torch.where(red, torch.ones_like(s.winner), s.winner)
        s.outcome = torch.where(blue, torch.full_like(s.outcome, OUTCOME_BLUE_200), s.outcome)
        s.winner = torch.where(blue, torch.full_like(s.winner, 2), s.winner)
        _finish(s, score_before, snap, foul_any, gen)
        return

    # ---- 5.10 respawn / invuln / heal -------------------------------
    dead = ~s.alive
    s.respawn_timer = torch.where(
        dead & (~just_died),
        torch.clamp(s.respawn_timer - cfg.dt, min=0.0),
        s.respawn_timer,
    )
    revive = dead & (~just_died) & (s.respawn_timer <= 0.0)
    just_revived = revive.clone()
    if revive.any():
        # same entity, current corpse position, no search/no displacement
        s.alive = s.alive | revive
        s.hp = torch.where(revive, torch.full_like(s.hp, cfg.hp_revive), s.hp)
        s.heat = torch.where(revive, torch.zeros_like(s.heat), s.heat)
        s.overheated = s.overheated & (~revive)
        s.spin = torch.where(revive, torch.zeros_like(s.spin), s.spin)
        s.p_t = torch.where(revive, torch.zeros_like(s.p_t), s.p_t)
        s.target = torch.where(revive, torch.zeros_like(s.target), s.target)
        s.hit_cue_valid = s.hit_cue_valid & (~revive)
        s.vel = torch.where(revive.unsqueeze(-1), torch.zeros_like(s.vel), s.vel)
        # v1.3: inherit the gimbal yaw from the death frame.  A dead agent's yaw
        # is frozen by the turret update above, so s.yaw already holds it.
        s.invuln = s.invuln | revive
        s.invuln_timer = torch.where(revive, torch.full_like(s.invuln_timer, cfg.invuln_time), s.invuln_timer)
        s.respawn_timer = torch.where(revive, torch.zeros_like(s.respawn_timer), s.respawn_timer)

    s.invuln_timer = torch.where(
        s.invuln & (~just_revived),
        torch.clamp(s.invuln_timer - cfg.dt, min=0.0),
        s.invuln_timer,
    )
    expired = s.invuln & (~just_revived) & (s.invuln_timer <= 0.0)

    heal_in = torch.zeros((B, N), device=dev, dtype=torch.bool)
    for i in range(N):
        heal_in[:, i] = point_in_polygon_closed(s.pos[:, i], s.own_spawns[i])
    heal = s.alive & (~just_revived) & heal_in
    s.hp = torch.where(heal, torch.clamp(s.hp + cfg.heal_per_frame, max=cfg.hp_max), s.hp)

    # ---- 5.11 occupation / score -----------------------------------
    in_center = torch.linalg.vector_norm(s.pos, dim=-1) <= cfg.center_radius
    occ_red = (s.alive & (~s.invuln) & in_center & (s.team == 0).unsqueeze(0)).sum(dim=1).float()
    occ_blue = (s.alive & (~s.invuln) & in_center & (s.team == 1).unsqueeze(0)).sum(dim=1).float()
    s.occ = torch.stack((occ_red, occ_blue), dim=1)
    score_add = torch.zeros((B, 2), device=dev)
    score_add[:, 0] = torch.where(occ_red > occ_blue, torch.full_like(occ_red, cfg.score_per_frame), torch.zeros_like(occ_red))
    score_add[:, 1] = torch.where(occ_blue > occ_red, torch.full_like(occ_blue, cfg.score_per_frame), torch.zeros_like(occ_blue))
    s.score = s.score + score_add

    reached = s.score >= cfg.target_score - 1e-6
    if reached.any():
        both = reached[:, 0] & reached[:, 1]
        red = reached[:, 0] & (~both)
        blue = reached[:, 1] & (~both)
        s.done = s.done | reached.any(dim=1)
        s.both_200 = s.both_200 | both
        s.outcome = torch.where(both, torch.full_like(s.outcome, OUTCOME_DRAW_200), s.outcome)
        s.winner = torch.where(both, torch.zeros_like(s.winner), s.winner)
        s.outcome = torch.where(red, torch.full_like(s.outcome, OUTCOME_RED_200), s.outcome)
        s.winner = torch.where(red, torch.ones_like(s.winner), s.winner)
        s.outcome = torch.where(blue, torch.full_like(s.outcome, OUTCOME_BLUE_200), s.outcome)
        s.winner = torch.where(blue, torch.full_like(s.winner, 2), s.winner)
        _finish(s, score_before, snap, foul_any, gen)
        return

    s.invuln = s.invuln & (~expired)

    # ---- 5.12 time / tiebreak --------------------------------------
    s.time = torch.clamp(s.time - cfg.dt, min=0.0)
    timeout = (~s.done) & (s.time <= 0.0)
    if timeout.any():
        rd = s.score[:, 0] > s.score[:, 1]
        bd = s.score[:, 1] > s.score[:, 0]
        eq = ~(rd | bd)
        dmg_rd = eq & (s.total_damage[:, 0] > s.total_damage[:, 1])
        dmg_bd = eq & (s.total_damage[:, 1] > s.total_damage[:, 0])
        dmg_eq = eq & (s.total_damage[:, 0] == s.total_damage[:, 1])
        total_hp = torch.stack(
            (torch.where(s.alive & (s.team == 0).unsqueeze(0), s.hp, torch.zeros_like(s.hp)).sum(dim=1),
             torch.where(s.alive & (s.team == 1).unsqueeze(0), s.hp, torch.zeros_like(s.hp)).sum(dim=1)),
            dim=1,
        )
        hp_rd = dmg_eq & (total_hp[:, 0] > total_hp[:, 1])
        hp_bd = dmg_eq & (total_hp[:, 1] > total_hp[:, 0])
        hp_eq = dmg_eq & (total_hp[:, 0] == total_hp[:, 1])
        s.done = s.done | timeout
        s.outcome = torch.where(timeout & rd, torch.full_like(s.outcome, OUTCOME_RED_TIME), s.outcome)
        s.winner = torch.where(timeout & rd, torch.ones_like(s.winner), s.winner)
        s.outcome = torch.where(timeout & bd, torch.full_like(s.outcome, OUTCOME_BLUE_TIME), s.outcome)
        s.winner = torch.where(timeout & bd, torch.full_like(s.winner, 2), s.winner)
        s.outcome = torch.where(timeout & dmg_rd, torch.full_like(s.outcome, OUTCOME_RED_DMG), s.outcome)
        s.winner = torch.where(timeout & dmg_rd, torch.ones_like(s.winner), s.winner)
        s.outcome = torch.where(timeout & dmg_bd, torch.full_like(s.outcome, OUTCOME_BLUE_DMG), s.outcome)
        s.winner = torch.where(timeout & dmg_bd, torch.full_like(s.winner, 2), s.winner)
        s.outcome = torch.where(timeout & hp_rd, torch.full_like(s.outcome, OUTCOME_RED_HP), s.outcome)
        s.winner = torch.where(timeout & hp_rd, torch.ones_like(s.winner), s.winner)
        s.outcome = torch.where(timeout & hp_bd, torch.full_like(s.outcome, OUTCOME_BLUE_HP), s.outcome)
        s.winner = torch.where(timeout & hp_bd, torch.full_like(s.winner, 2), s.winner)
        s.outcome = torch.where(timeout & hp_eq, torch.full_like(s.outcome, OUTCOME_DRAW_TIME), s.outcome)
        s.winner = torch.where(timeout & hp_eq, torch.zeros_like(s.winner), s.winner)

    _finish(s, score_before, snap, foul_any, gen)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------
def _finalize_rewards(s: RmState, score_before: Tensor) -> None:
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    team = s.team.unsqueeze(0).expand(B, N)

    score_delta = s.score - score_before
    dense = cfg.k_score_reward * score_delta.gather(1, team)
    # small per-attacker hit shaping (actual/raw damage dealt to enemies)
    hit = cfg.k_hit_reward * (s.last_hit_damage / cfg.hp_max)

    center = torch.linalg.vector_norm(s.pos, dim=-1)
    center_r = cfg.k_center_reward * torch.exp(
        -(center ** 2) / (2.0 * cfg.center_sigma ** 2)
    ) * (center > cfg.center_radius).float() * s.alive.float()

    # v1.9 shaping: stay alive; when low HP, return to own spawn (heal zone)
    alive_f = s.alive.float()
    # survival reward scales with HP, so low-HP agents have a clear incentive
    # to heal / go home rather than trade at 65 HP.
    survival = cfg.k_survival_reward * (s.hp / cfg.hp_max) * alive_f
    own_centers = torch.stack([p.mean(dim=0) for p in s.own_spawns], dim=0)  # (N,2)
    d_home = torch.linalg.vector_norm(
        s.pos - own_centers.unsqueeze(0), dim=-1
    )  # (B,N)
    low = (s.hp < cfg.low_hp_threshold * cfg.hp_max).float()
    home = (
        cfg.k_low_hp_home_reward
        * torch.exp(-(d_home ** 2) / (2.0 * cfg.home_sigma ** 2))
        * low
        * alive_f
    )
    # v1.12 collision/contact penalty: robot-robot, robot-corpse, wall, cover.
    # The resolver pushes bodies apart; use a small tolerance so the contact
    # frame is still penalized.
    contact = torch.zeros((B, N), device=dev, dtype=torch.float32)
    for i in range(N):
        d_other = torch.linalg.vector_norm(
            s.pos[:, i].unsqueeze(1) - s.pos, dim=-1
        )
        near_robot = d_other <= 2.0 * cfg.robot_radius + 1e-3
        near_robot[:, i] = False
        robot_hits = near_robot.sum(dim=1).float()
        wall = (
            (s.pos[:, i, 0].abs() >= cfg.bound_x - cfg.robot_radius - 1e-3)
            | (s.pos[:, i, 1].abs() >= cfg.bound_y - cfg.robot_radius - 1e-3)
        ).float()
        cover_hits = torch.zeros((B,), device=dev)
        for poly in s.covers:
            _, dist, _, inside = G.nearest_point_on_polygon(s.pos[:, i], poly)
            near = (dist <= cfg.robot_radius + 1e-3) | inside
            cover_hits = torch.maximum(cover_hits, near.float())
        contact[:, i] = robot_hits + wall + cover_hits

    dense = dense + center_r + survival + home - cfg.k_collision_penalty * contact

    terminal = torch.zeros((B, N), device=dev)
    done = s.done
    win_team = s.winner  # 0 draw, 1 red, 2 blue
    # normal win/loss
    for t in (0, 1):
        is_team = (team == t)
        if t == 0:
            win = done & (win_team == 1)
            lose = done & (win_team == 2)
        else:
            win = done & (win_team == 2)
            lose = done & (win_team == 1)
        terminal = torch.where(is_team & win.unsqueeze(1), torch.full_like(terminal, cfg.terminal_win), terminal)
        terminal = torch.where(is_team & lose.unsqueeze(1), torch.full_like(terminal, cfg.terminal_loss), terminal)
        # collision foul: loser gets -10; winner keeps +2
        foul_lose = lose & (s.foul_loser == t)
        terminal = torch.where(is_team & foul_lose.unsqueeze(1), torch.full_like(terminal, cfg.terminal_foul), terminal)
    terminal = torch.where(done.unsqueeze(1) & (win_team == 0).unsqueeze(1), torch.full_like(terminal, cfg.terminal_draw), terminal)
    terminal = torch.where(done.unsqueeze(1) & s.both_200.unsqueeze(1), torch.full_like(terminal, cfg.terminal_both_200), terminal)
    terminal = torch.where(done.unsqueeze(1) & s.double_foul.unsqueeze(1), torch.full_like(terminal, cfg.terminal_both_foul), terminal)

    reward_dense = dense + hit
    just_died = s.just_died.bool()
    dead = ~s.alive
    # death frame: instant penalty; subsequent dead frames: zero dense reward
    reward_dense = torch.where(
        just_died,
        torch.full_like(reward_dense, -cfg.k_death_penalty),
        reward_dense,
    )
    reward_dense = torch.where(
        dead & (~just_died),
        torch.zeros_like(reward_dense),
        reward_dense,
    )
    s.reward_dense = reward_dense
    s.reward_terminal = terminal
    s.reward = s.reward_dense + s.reward_terminal


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------
def _canon_sign(s: RmState) -> Tensor:
    # +1 for red, -1 for blue; (B,6,1)
    return torch.where(s.team == 0, 1.0, -1.0).view(1, -1, 1)


def _canon_yaw(s: RmState) -> Tensor:
    # rotate blue by pi
    off = torch.where(s.team == 0, 0.0, PI)
    return G.wrap_angle(s.yaw + off.view(1, -1))


def build_observation(s: RmState, generator: Optional[torch.Generator] = None) -> None:
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    gen = generator
    sign = _canon_sign(s)
    yaw_c = _canon_yaw(s)

    p_vis, in_fov, visible_geom = visibility_matrix(
        s, s.yaw, cfg.vis_samples_obs, s.target
    )

    obs = torch.zeros((B, N, OBS_DIM), device=dev)
    mask = torch.zeros((B, N, 4), device=dev, dtype=torch.bool)
    mask[:, :, 0] = True

    for i in range(N):
        o = []
        # self (24)
        o += [sign[:, i, 0] * s.pos[:, i, 0] / cfg.field_hx,
              sign[:, i, 0] * s.pos[:, i, 1] / cfg.field_hy]
        o += [sign[:, i, 0] * s.vel[:, i, 0] / cfg.v_max,
              sign[:, i, 0] * s.vel[:, i, 1] / cfg.v_max]
        o += [torch.sin(yaw_c[:, i]), torch.cos(yaw_c[:, i])]
        o += [s.yaw_rate[:, i] / cfg.yaw_rate_max]
        o += [s.hp[:, i] / cfg.hp_max]
        o += [s.heat[:, i] / cfg.heat_max]
        o += [s.overheated[:, i].float()]
        o += [s.spin[:, i]]
        o += [s.alive[:, i].float()]
        o += [s.invuln[:, i].float()]
        o += [s.invuln_timer[:, i] / cfg.invuln_time]
        o += [s.respawn_timer[:, i] / cfg.respawn_time]
        o += [(s.target[:, i] != 0).float()]
        for k in range(4):
            o += [(s.target[:, i] == k).float()]
        o += [point_in_polygon_closed(s.pos[:, i], s.own_spawns[i]).float()]
        o += [(torch.linalg.vector_norm(s.pos[:, i], dim=-1) <= cfg.center_radius).float()]
        o += [point_in_polygon_closed(s.pos[:, i], s.enemy_spawns[i]).float()]
        o += [torch.linalg.vector_norm(s.pos[:, i], dim=-1) / cfg.lock_range]
        # teammates (2 * 18)
        for j in s.mate_idx[i].tolist():
            alive_j = s.alive[:, j]
            rel = sign[:, i, 0].unsqueeze(-1) * (s.pos[:, j] - s.pos[:, i])
            abs_pos = sign[:, i, 0].unsqueeze(-1) * s.pos[:, j]
            rvel = sign[:, i, 0].unsqueeze(-1) * (s.vel[:, j] - s.vel[:, i])
            dyaw = G.wrap_angle(s.yaw[:, j] - s.yaw[:, i])
            o += [rel[:, 0] / cfg.lock_range, rel[:, 1] / cfg.lock_range]
            o += [abs_pos[:, 0] / cfg.field_hx, abs_pos[:, 1] / cfg.field_hy]
            o += [torch.where(alive_j, rvel[:, 0], torch.zeros_like(rvel[:, 0])) / (2 * cfg.v_max),
                  torch.where(alive_j, rvel[:, 1], torch.zeros_like(rvel[:, 1])) / (2 * cfg.v_max)]
            o += [torch.where(alive_j, s.hp[:, j], torch.zeros_like(s.hp[:, j])) / cfg.hp_max,
                  alive_j.float(),
                  torch.where(alive_j, s.invuln[:, j].float(), torch.zeros_like(s.hp[:, j]))]
            o += [torch.sin(dyaw), torch.cos(dyaw)]
            o += [torch.where(alive_j, s.heat[:, j], torch.zeros_like(s.heat[:, j])) / cfg.heat_max,
                  torch.where(alive_j, s.spin[:, j], torch.zeros_like(s.spin[:, j]))]
            for k in range(4):
                o += [(alive_j & (s.target[:, j] == k)).float()]
            o += [s.respawn_timer[:, j] / cfg.respawn_time]
        # enemies (3 * 17)
        for k in range(3):
            j = int(s.enemy_idx[i, k])
            vis = visible_geom[:, i, k]
            alive_j = s.alive[:, j]
            rel = sign[:, i, 0].unsqueeze(-1) * (s.pos[:, j] - s.pos[:, i])
            abs_pos = sign[:, i, 0].unsqueeze(-1) * s.pos[:, j]
            rvel = sign[:, i, 0].unsqueeze(-1) * (s.vel[:, j] - s.vel[:, i])
            hp_sigma = cfg.hp_noise_base + cfg.hp_noise_spin * s.spin[:, j]
            noise = torch.randn((B,), device=dev, generator=gen) * hp_sigma
            hp_obs = torch.round(torch.clamp(s.hp[:, j] + noise, 0.0, cfg.hp_max)) / cfg.hp_max
            dyaw = G.wrap_angle(s.yaw[:, j] - s.yaw[:, i])
            # observed yaw noise: +/-5 deg, exact on the frame the target fires
            noise_deg = (torch.rand((B,), device=dev, generator=gen) * 2.0 - 1.0) * cfg.yaw_noise_deg
            dyaw_obs = G.wrap_angle(
                dyaw + torch.where(s.last_fired[:, j], torch.zeros_like(dyaw), torch.deg2rad(noise_deg))
            )
            o += [vis.float()]
            o += [torch.where(vis, rel[:, 0], torch.zeros_like(rel[:, 0])) / cfg.lock_range,
                  torch.where(vis, rel[:, 1], torch.zeros_like(rel[:, 1])) / cfg.lock_range]
            o += [torch.where(vis, abs_pos[:, 0], torch.zeros_like(abs_pos[:, 0])) / cfg.field_hx,
                  torch.where(vis, abs_pos[:, 1], torch.zeros_like(abs_pos[:, 1])) / cfg.field_hy]
            o += [torch.where(vis & alive_j, rvel[:, 0], torch.zeros_like(rvel[:, 0])) / (2 * cfg.v_max),
                  torch.where(vis & alive_j, rvel[:, 1], torch.zeros_like(rvel[:, 1])) / (2 * cfg.v_max)]
            o += [torch.where(vis & alive_j, hp_obs, torch.zeros_like(hp_obs))]
            o += [torch.where(vis & alive_j, s.invuln[:, j].float(), torch.zeros_like(s.hp[:, j]))]
            o += [torch.where(vis, torch.sin(dyaw_obs), torch.zeros_like(dyaw_obs)),
                  torch.where(vis, torch.cos(dyaw_obs), torch.zeros_like(dyaw_obs))]
            o += [torch.where(vis & alive_j, s.spin[:, j], torch.zeros_like(s.spin[:, j]))]
            o += [torch.where(vis & alive_j, s.last_fired[:, j].float(), torch.zeros_like(s.spin[:, j]))]
            o += [torch.where(vis, p_vis[:, i, k], torch.zeros_like(p_vis[:, i, k]))]
            o += [torch.where(vis, alive_j.float(), torch.zeros_like(s.hp[:, j]))]
            o += [((s.target[:, i] == k + 1) & vis & alive_j).float()]
            o += [torch.where(vis, s.respawn_timer[:, j], torch.zeros_like(s.respawn_timer[:, j])) / cfg.respawn_time]
            # v1.13: invuln enemies are targetable for pre-aim; fire still blocked
            mask[:, i, k + 1] = vis & alive_j
        # hit cue (3)
        cue_off = 0.0 if int(s.team[i]) == 0 else PI
        cue = G.wrap_angle(s.hit_cue[:, i] + cue_off)
        o += [torch.where(s.hit_cue_valid[:, i], torch.sin(cue), torch.zeros_like(cue)),
              torch.where(s.hit_cue_valid[:, i], torch.cos(cue), torch.zeros_like(cue)),
              s.hit_cue_valid[:, i].float()]
        # HUD (10)
        t = int(s.team[i])
        own_score = s.score[:, t] / cfg.target_score
        enemy_score = s.score[:, 1 - t] / cfg.target_score
        own_alive = (s.alive & (s.team == t).unsqueeze(0)).sum(dim=1).float() / 3.0
        enemy_alive = (s.alive & (s.team == 1 - t).unsqueeze(0)).sum(dim=1).float() / 3.0
        own_hp = torch.where(s.alive & (s.team == t).unsqueeze(0), s.hp, torch.zeros_like(s.hp)).sum(dim=1) / (3 * cfg.hp_max)
        enemy_hp = torch.where(s.alive & (s.team == 1 - t).unsqueeze(0), s.hp, torch.zeros_like(s.hp)).sum(dim=1) / (3 * cfg.hp_max)
        o += [own_score, enemy_score, (s.score[:, t] - s.score[:, 1 - t]) / cfg.target_score,
              s.time / cfg.episode_time,
              s.occ[:, t] / 3.0, s.occ[:, 1 - t] / 3.0,
              own_alive, enemy_alive, own_hp, enemy_hp]
        # lidar (128), appended at the end of the vector
        ids = [m for m in range(N) if m != i]
        circles = torch.stack(
            [torch.cat([s.pos[:, m], torch.full((B, 1), cfg.occlusion_radius, device=dev)], dim=-1)
             for m in ids], dim=1)
        lid = G.lidar_ranges(
            s.pos[:, i], s.yaw[:, i], s.cover_edges, circles,
            cfg.lidar_rays, cfg.lidar_range, cfg.lidar_fov_half,
            cfg.field_hx, cfg.field_hy,
        )
        # scalar part is OBS_DIM - LIDAR_DIM, lidar is the last block
        obs[:, i] = torch.cat([torch.stack(o, dim=-1), lid], dim=-1)

    # dead agents: only "none" target
    dead = ~s.alive
    mask = torch.where(dead.unsqueeze(-1), torch.zeros_like(mask), mask)
    mask[:, :, 0] = True
    s.obs = obs
    s.action_mask = mask
    s.global_state = _build_global_state(s)


def _build_global_state(s: RmState) -> Tensor:
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    parts = []
    for i in range(N):
        parts += [
            s.pos[:, i, 0] / cfg.field_hx, s.pos[:, i, 1] / cfg.field_hy,
            s.vel[:, i, 0] / cfg.v_max, s.vel[:, i, 1] / cfg.v_max,
            torch.sin(s.yaw[:, i]), torch.cos(s.yaw[:, i]),
            s.hp[:, i] / cfg.hp_max, s.heat[:, i] / cfg.heat_max,
            s.overheated[:, i].float(), s.spin[:, i],
            s.alive[:, i].float(), s.invuln[:, i].float(),
            s.invuln_timer[:, i] / cfg.invuln_time,
            s.respawn_timer[:, i] / cfg.respawn_time,
            s.target[:, i].float() / 3.0, s.p_t[:, i], s.tau_eff[:, i] / 0.6,
            torch.full((B,), float(s.team[i].item()), device=dev),
        ]
        for k in range(3):
            j = int(s.enemy_idx[i, k])
            parts.append(s.cum_damage[:, i, j] / cfg.hp_max)
        # actual hit rate directly from the environment: p_final of the last
        # fired shot (p_t already lives in the per-agent block above)
        parts.append(s.last_p_final[:, i])
    total_hp = torch.stack(
        (torch.where(s.alive & (s.team == 0).unsqueeze(0), s.hp, torch.zeros_like(s.hp)).sum(dim=1),
         torch.where(s.alive & (s.team == 1).unsqueeze(0), s.hp, torch.zeros_like(s.hp)).sum(dim=1)),
        dim=1,
    )
    parts += [
        s.score[:, 0] / cfg.target_score, s.score[:, 1] / cfg.target_score,
        s.time / cfg.episode_time,
        s.occ[:, 0] / 3.0, s.occ[:, 1] / 3.0,
        (s.alive & (s.team == 0).unsqueeze(0)).sum(dim=1).float() / 3.0,
        (s.alive & (s.team == 1).unsqueeze(0)).sum(dim=1).float() / 3.0,
        total_hp[:, 0] / (3 * cfg.hp_max), total_hp[:, 1] / (3 * cfg.hp_max),
        s.total_damage[:, 0] / 1000.0, s.total_damage[:, 1] / 1000.0,
    ]
    return torch.stack(parts, dim=-1)
