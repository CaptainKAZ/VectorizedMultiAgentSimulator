#  Copyright (c) 2026.
#  RM 3v3 rule engine (v1.2).  Pure torch functions; no VMAS imports.
#
#  Frame order follows rules §5:
#    process_action -> spin update -> velocity target -> VMAS integration
#    -> post_step: collision -> visibility/lock -> turret -> fire/heat/damage
#       -> death -> respawn/heal -> occupation/score -> time -> observation
from __future__ import annotations

import os

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
OUTCOME_STAGNATION = 23       # no fire/damage/occupation within grace period
OUTCOME_RED_WALL_PUSH = 24    # blue pushed wall/cover -> red wins
OUTCOME_BLUE_WALL_PUSH = 25   # red pushed wall/cover -> blue wins
OUTCOME_BOTH_WALL_PUSH = 26   # both teams pushed -> draw
OUTCOME_RED_BODY_PUSH = 27    # blue sustained robot push -> red wins
OUTCOME_BLUE_BODY_PUSH = 28   # red sustained robot push -> blue wins
OUTCOME_BOTH_BODY_PUSH = 29   # both teams sustained robot push -> draw

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
    23: "空局(10秒无事件)",
    24: "红方胜(蓝方推墙/掩体犯规)",
    25: "蓝方胜(红方推墙/掩体犯规)",
    26: "平局(双方推墙/掩体犯规)",
    27: "红方胜(蓝方推人犯规)",
    28: "蓝方胜(红方推人犯规)",
    29: "平局(双方推人犯规)",
}

# observation layout
SELF_DIM = 24
SPAWN_DIM = 4          # own spawn centre relative + distance + is_in_spawn
MATE_DIM = 18          # rel/abs pos + respawn_timer (dead teammate corpse info)
ENEMY_DIM = 17         # rel/abs pos + respawn_timer (visible dead enemy corpse info)
CUE_DIM = 3
OBSTACLE_DIM = 8
CENTER_DIM = 4
HUD_DIM = 10
SCALAR_DIM = (
    SELF_DIM + SPAWN_DIM + CENTER_DIM
    + 2 * MATE_DIM + 3 * ENEMY_DIM + CUE_DIM + HUD_DIM
)
OBS_DIM = SCALAR_DIM + OBSTACLE_DIM

# global state layout (critic)
G_AGENT_DIM = 22 + OBSTACLE_DIM
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
    # Spawn areas are axis-aligned rectangles.  The generic path runs a
    # ray-cast plus a nearest-edge search for every point; for a rectangle the
    # same closed-region test is just a clamped coordinate distance.  Keep the
    # generic path for any non-rectangular polygon.
    if poly.shape[0] == 4:
        px, py = points[..., 0], points[..., 1]
        xs, ys = poly[:, 0], poly[:, 1]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        rect = bool(
            torch.all((xs == xmin) | (xs == xmax))
            and torch.all((ys == ymin) | (ys == ymax))
        )
        if rect:
            dx = torch.clamp(torch.maximum(xmin - px, px - xmax), min=0.0)
            dy = torch.clamp(torch.maximum(ymin - py, py - ymax), min=0.0)
            inside = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
            near = torch.sqrt(dx * dx + dy * dy) <= 1e-6
            return inside | near
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
    center_dist_prev: Tensor = field(init=False)
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
    foul_agent: Tensor = field(init=False)    # (B,N) actual collision culprit(s)
    foul_intensity: Tensor = field(init=False)  # (B,N) normalized 0..1 impact severity
    double_foul: Tensor = field(init=False)
    both_200: Tensor = field(init=False)
    done: Tensor = field(init=False)
    just_died: Tensor = field(init=False)
    step_count: int = 0                    # env steps since env creation (curriculum clock)
    curriculum_role: Tensor = field(init=False)  # (B,) -1 unset, 0 random, 1 fixed

    obs: Tensor = field(init=False)
    action_mask: Tensor = field(init=False)
    fire_mask: Tensor = field(init=False)   # (B,N,2): class 0 always, class 1 only when locked
    global_state: Tensor = field(init=False)
    reward: Tensor = field(init=False)
    reward_dense: Tensor = field(init=False)
    reward_terminal: Tensor = field(init=False)
    last_hit_damage: Tensor = field(init=False)
    last_damage: Tensor = field(init=False)
    last_p_final: Tensor = field(init=False)    # env p_final of the last fired shot (0 if none)
    prev_wall_contact: Tensor = field(init=False)
    prev_pair_contact: Tensor = field(init=False)
    obstacle_dist: Tensor = field(init=False)   # actor/critic 8-way clearances
    stagnation_had_combat: Tensor = field(init=False)  # (B,)
    stagnation_had_occ: Tensor = field(init=False)     # (B,)
    stagnation_had_heal: Tensor = field(init=False)    # (B,)
    stagnation_last_event: Tensor = field(init=False) # (B,) episode_elapsed of last event
    stagnation_speed: Tensor = field(init=False)       # (B,N) EMA of normalized speed since event
    episode_elapsed: Tensor = field(init=False)        # (B,) seconds since reset
    episode_start_elapsed: Tensor = field(init=False)  # (B,) late-game clock offset
    wall_contact_time: Tensor = field(init=False)      # (B,N) seconds in wall/cover contact
    body_push_time: Tensor = field(init=False)     # (B,N) seconds pushing any other robot
    body_push_foul: Tensor = field(init=False)     # (B,N) pushers at body-push terminal
    proximity_time: Tensor = field(init=False)     # (B,N,N) seconds within proximity radius
    wall_push_time: Tensor = field(init=False)          # (B,N) continuous push timer
    wall_push_foul_intent: Tensor = field(init=False)   # (B,N) pusher intent at foul

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
        self.center_dist_prev = f(B, N)
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
        self.kill_damage = f(B, N, N)
        self.last_kill_credit = f(B, N)
        self.last_kill_score = f(B, 2)
        self.occ = f(B, 2)
        self.outcome = l(B)
        self.winner = torch.full((B,), -1, device=d, dtype=torch.long)
        self.foul_loser = torch.full((B,), -1, device=d, dtype=torch.long)
        self.foul_agent = b(B, N)
        self.foul_intensity = f(B, N)
        self.double_foul = b(B)
        self.both_200 = b(B)
        self.done = b(B)
        self.just_died = b(B, N)
        self.curriculum_role = torch.full(
            (B,), -1, device=d, dtype=torch.long
        )
        self.obs = f(B, N, OBS_DIM)
        self.action_mask = b(B, N, 4)
        self.fire_mask = b(B, N, 2)
        self.global_state = f(B, GLOBAL_DIM)
        self.reward = f(B, N)
        self.reward_dense = f(B, N)
        self.reward_terminal = f(B, N)
        self.last_hit_damage = f(B, N)
        self.last_damage = f(B, N, N)
        self.last_p_final = f(B, N)
        self.prev_wall_contact = b(B, N)
        self.prev_pair_contact = b(B, N, N)
        self.obstacle_dist = f(B, N, OBSTACLE_DIM)
        self.stagnation_had_combat = b(B)
        self.stagnation_had_occ = b(B)
        self.stagnation_had_heal = b(B)
        self.stagnation_last_event = f(B)
        self.stagnation_speed = f(B, N)
        self.episode_elapsed = f(B)
        self.episode_start_elapsed = f(B)
        self.wall_contact_time = f(B, N)
        self.body_push_time = f(B, N)
        self.body_push_foul = b(B, N)
        self.proximity_time = f(B, N, N)
        self.wall_push_time = f(B, N)
        self.wall_push_foul_intent = f(B, N)


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
def _fixed_reset_rows(s: RmState, rows: List[int], seed: int = 0) -> None:
    cfg = s.cfg
    d = s.device
    n = len(rows)
    if n == 0:
        return
    gen = torch.Generator(device=d)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    n_team = cfg.n_per_team
    slots_r = torch.tensor(cfg.red_slots, device=d)
    slots_b = torch.tensor(cfg.blue_slots, device=d)
    if cfg.fixed_slot_permute:
        perm_r = torch.argsort(torch.rand(n, n_team, generator=gen, device=d), dim=1)
        perm_b = torch.argsort(torch.rand(n, n_team, generator=gen, device=d), dim=1)
    else:
        ar = torch.arange(n_team, device=d)
        perm_r = ar.unsqueeze(0).expand(n, -1)
        perm_b = ar.unsqueeze(0).expand(n, -1)
    pr = slots_r[perm_r]
    pb = slots_b[perm_b]
    if cfg.fixed_jitter > 0.0:
        # Rejection-sample small individual jitters, keeping the six centres
        # at least rsi_min_sep apart.  Invalid rows fall back to the exact
        # permuted slots, so this can never create an illegal reset.
        eye = torch.eye(2 * n_team, device=d) * 1e9
        base_r, base_b = pr.clone(), pb.clone()
        accepted = torch.zeros(n, dtype=torch.bool, device=d)
        best_r, best_b = pr.clone(), pb.clone()
        lo = torch.tensor([-cfg.bound_x + 0.1, -cfg.bound_y + 0.1], device=d)
        hi = torch.tensor([cfg.bound_x - 0.1, cfg.bound_y - 0.1], device=d)
        for _ in range(8):
            jr = (torch.rand(n, n_team, 2, generator=gen, device=d) * 2.0 - 1.0) * cfg.fixed_jitter
            jb = (torch.rand(n, n_team, 2, generator=gen, device=d) * 2.0 - 1.0) * cfg.fixed_jitter
            cr = torch.maximum(torch.minimum(base_r + jr, hi), lo)
            cb = torch.maximum(torch.minimum(base_b + jb, hi), lo)
            pts = torch.cat((cr, cb), dim=1)
            sep = torch.cdist(pts, pts) + eye
            ok = sep.amin(dim=(1, 2)) >= cfg.rsi_min_sep
            take = (~accepted) & ok
            if bool(take.any()):
                best_r[take] = cr[take]
                best_b[take] = cb[take]
                accepted |= take
            if bool(accepted.all()):
                break
        pr, pb = best_r, best_b
    yaw_r = torch.atan2(-pr[..., 1], -pr[..., 0])
    yaw_b = torch.atan2(-pb[..., 1], -pb[..., 0])
    if cfg.fixed_yaw_jitter > 0.0:
        yaw_r = G.wrap_angle(
            yaw_r + (torch.rand(n, n_team, generator=gen, device=d) * 2.0 - 1.0) * cfg.fixed_yaw_jitter
        )
        yaw_b = G.wrap_angle(
            yaw_b + (torch.rand(n, n_team, generator=gen, device=d) * 2.0 - 1.0) * cfg.fixed_yaw_jitter
        )
    for k, b in enumerate(rows):
        s.pos[b, :n_team] = pr[k]
        s.pos[b, n_team:] = pb[k]
        s.yaw[b, :n_team] = yaw_r[k]
        s.yaw[b, n_team:] = yaw_b[k]
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
        s.episode_start_elapsed[b] = 0.0
        s.total_damage[b].zero_()
        s.cum_damage[b].zero_()
        s.kill_damage[b].zero_()
        s.last_kill_credit[b].zero_()
        s.last_kill_score[b].zero_()
        s.last_p_final[b].zero_()
        s.occ[b].zero_()
        s.outcome[b] = OUTCOME_ONGOING
        s.winner[b] = -1
        s.foul_loser[b] = -1
        s.foul_agent[b].zero_()
        s.foul_intensity[b].zero_()
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
    red_anchor_c = torch.stack([p.mean(dim=0) for p in s.own_spawns[:3]], dim=0).mean(dim=0)
    blue_anchor_c = torch.stack([p.mean(dim=0) for p in s.own_spawns[3:]], dim=0).mean(dim=0)
    for _ in range(max_rounds):
        need = ~valid
        k = int(need.sum())
        if k == 0:
            break
        ri = torch.randint(red_pool.shape[0], (k, n_team), generator=gen, device=d)
        bi = torch.randint(blue_pool.shape[0], (k, n_team), generator=gen, device=d)
        r = red_pool[ri]
        b = blue_pool[bi]
        if cfg.spawn_anchor_init:
            jr = (torch.rand((k, 2), generator=gen, device=d) * 2.0 - 1.0) * cfg.spawn_anchor_jitter
            jb = (torch.rand((k, 2), generator=gen, device=d) * 2.0 - 1.0) * cfg.spawn_anchor_jitter
            r[:, 0] = red_anchor_c + jr
            b[:, 0] = blue_anchor_c + jb
        b = torch.where(sym[need].view(k, 1, 1), -r, b)
        if cfg.spawn_anchor_init:
            # symmetric rows must mirror the anchor as well
            b[:, 0] = torch.where(
                sym[need].view(k, 1), -r[:, 0], blue_anchor_c + jb
            )
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
    if cfg.spawn_anchor_init:
        yaw_red[:, 0] = torch.atan2(-red[:, 0, 1], -red[:, 0, 0])
        yaw_blue[:, 0] = torch.where(
            sym,
            G.wrap_angle(yaw_red[:, 0] + PI),
            torch.atan2(-blue[:, 0, 1], -blue[:, 0, 0]),
        )
        # Randomise which agent index carries the spawn anchor.  The actor is
        # shared and has no explicit agent id, but the critic has slot
        # embeddings; keeping the anchor on a fixed slot biases those slots.
        perm_r = torch.argsort(
            torch.rand(n, 3, generator=gen, device=d), dim=1
        )
        perm_b_free = torch.argsort(
            torch.rand(n, 3, generator=gen, device=d), dim=1
        )
        perm_b = torch.where(sym.view(n, 1), perm_r, perm_b_free)
        gather2 = perm_r.unsqueeze(-1).expand(-1, -1, 2)
        red = torch.gather(red, 1, gather2)
        blue = torch.gather(blue, 1, perm_b.unsqueeze(-1).expand(-1, -1, 2))
        yaw_red = torch.gather(yaw_red, 1, perm_r)
        yaw_blue = torch.gather(yaw_blue, 1, perm_b)
        pts = torch.cat((red, blue), dim=1)
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

    idx = torch.as_tensor(rows, device=d, dtype=torch.long)
    if cfg.rsi_late_game:
        # Full 180 s episode: randomise the amount of time still on the
        # clock, not the episode horizon.  s.time is the remaining clock.
        remaining = (
            torch.rand(n, generator=gen, device=d)
            * (cfg.rsi_remaining_max - cfg.rsi_remaining_min)
            + cfg.rsi_remaining_min
        ).clamp(0.0, cfg.episode_time)
        start_elapsed = cfg.episode_time - remaining
        s.time[idx] = remaining
        s.episode_start_elapsed[idx] = start_elapsed
    else:
        t = (
            torch.rand(n, generator=gen, device=d)
            * (cfg.rsi_time_max - cfg.rsi_time_min)
            + cfg.rsi_time_min
        )
        s.time[idx] = t
        s.episode_start_elapsed[idx] = 0.0
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
    s.total_damage[idx] = 0.0
    s.cum_damage[idx] = 0.0
    s.last_p_final[idx] = 0.0
    s.occ[idx] = 0.0
    s.outcome[idx] = OUTCOME_ONGOING
    s.winner[idx] = -1
    s.foul_loser[idx] = -1
    s.foul_agent[idx] = False
    s.foul_intensity[idx] = 0.0
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
    build_obs: bool = True,
) -> None:
    rows = _as_idx(idx)
    if rows is None:
        rows = list(range(s.B))
    mode = mode or s.cfg.init_mode
    cfg = s.cfg
    if mode == "fixed":
        _fixed_reset_rows(s, rows, seed=seed)
    elif mode == "curriculum":
        # Official fixed start vs RSI random start.
        # With persistent roles each environment keeps its role across
        # resets; a short fixed episode therefore cannot silently turn into
        # a long random episode and shrink the fixed share of the batch.
        # The initial allocation is an exact quota.
        gen = torch.Generator(device=s.device)
        gen.manual_seed(int(seed) & 0x7FFFFFFF)
        p = _fixed_prob(s)
        if cfg.curriculum_persistent_roles:
            # Exact quota at the current curriculum probability.  Roles are
            # persistent between resets, but the quota is rebalanced when
            # fixed_prob moves; otherwise an env allocated at the start
            # would retain its role for the whole 50M-step curriculum.
            target_fixed = int(round(s.B * p))
            unknown = s.curriculum_role < 0
            if bool(unknown.any()):
                order = torch.randperm(
                    s.B, generator=gen, device=s.device
                )
                s.curriculum_role.zero_()
                s.curriculum_role[order[:target_fixed]] = 1
            current_fixed = int(s.curriculum_role.sum().item())
            if current_fixed != target_fixed:
                if target_fixed > current_fixed:
                    pool = (s.curriculum_role == 0).nonzero(
                        as_tuple=False
                    ).reshape(-1)
                    n = target_fixed - current_fixed
                    pick = pool[
                        torch.randperm(
                            pool.numel(), generator=gen, device=s.device
                        )[:n]
                    ]
                    s.curriculum_role[pick] = 1
                else:
                    pool = (s.curriculum_role == 1).nonzero(
                        as_tuple=False
                    ).reshape(-1)
                    n = current_fixed - target_fixed
                    pick = pool[
                        torch.randperm(
                            pool.numel(), generator=gen, device=s.device
                        )[:n]
                    ]
                    s.curriculum_role[pick] = 0
            fixed_rows = [
                r for r in rows if int(s.curriculum_role[r]) == 1
            ]
            random_rows = [
                r for r in rows if int(s.curriculum_role[r]) == 0
            ]
        else:
            # Legacy independent per-reset choice.
            pick = torch.rand(len(rows), generator=gen, device=s.device) < p
            flags = pick.reshape(-1).tolist()
            fixed_rows = [r for r, f in zip(rows, flags) if f]
            random_rows = [r for r, f in zip(rows, flags) if not f]
        if fixed_rows:
            _fixed_reset_rows(s, fixed_rows, seed=seed)
        if random_rows:
            _random_reset_rows(
                s,
                random_rows,
                "mixed",
                seed=seed + (random_rows[0] if random_rows else 0),
            )
    else:
        _random_reset_rows(s, rows, mode, seed=seed + (rows[0] if rows else 0))
    s.center_dist_prev[rows] = torch.linalg.vector_norm(s.pos[rows], dim=-1)
    s.prev_wall_contact[rows] = False
    s.prev_pair_contact[rows] = False
    s.stagnation_had_combat[rows] = False
    s.stagnation_had_occ[rows] = False
    s.stagnation_had_heal[rows] = False
    start_elapsed = s.episode_start_elapsed[rows]
    s.stagnation_last_event[rows] = start_elapsed
    s.episode_elapsed[rows] = start_elapsed
    s.stagnation_speed[rows] = 0.0
    s.wall_contact_time[rows] = 0.0
    s.body_push_time[rows] = 0.0
    s.body_push_foul[rows] = False
    s.proximity_time[rows] = 0.0
    s.wall_push_time[rows] = 0.0
    s.wall_push_foul_intent[rows] = 0.0
    s.kill_damage[rows] = 0.0
    s.last_kill_credit[rows] = 0.0
    s.last_kill_score[rows] = 0.0
    if build_obs:
        build_observation(s)


# ---------------------------------------------------------------------------
# action decode (called from Scenario.process_action, before VMAS integration)
# ---------------------------------------------------------------------------
def decode_agent_action(
    s: RmState, i: int, raw_u: Tensor, current_vel: Tensor
) -> Tensor:
    """Update spin/target/yaw commands and return the velocity target.

    raw_u: (B,6) [vx, vy, yaw, spin, target, fire]
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

    v_body = clip_norm(raw_u[:, :2], 1.0)
    if bool(getattr(cfg, "relative_actions", False)):
        # v1.21: body/gimbal-frame velocity.  +x = forward along s.yaw,
        # +y = left.  For blue the observation is 180-degree canonicalised,
        # but the body frame itself is shared: R(yaw_actual) is correct for
        # both teams (no team_sign here).  This removes the policy's need to
        # rotate world-frame velocity commands by its own heading.
        yaw = s.yaw[:, i]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        v_world = torch.stack(
            [
                v_body[:, 0] * cos_yaw - v_body[:, 1] * sin_yaw,
                v_body[:, 0] * sin_yaw + v_body[:, 1] * cos_yaw,
            ],
            dim=-1,
        )
        v_des = v_world * vmax.unsqueeze(-1)
    else:
        # v1.14: observations are 180-degree canonicalised for blue, so the
        # 2-D velocity intent is canonical too.  Rotate it back to world for blue.
        team_sign = 1.0 if int(s.team[i]) == 0 else -1.0
        v_des = v_body * (team_sign * vmax).unsqueeze(-1)

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
    foul_agent = torch.zeros((B, N), device=dev, dtype=torch.bool)
    foul_intensity = torch.zeros((B, N), device=dev, dtype=torch.float32)
    robot_denom = max(2.0 * cfg.v_max - cfg.robot_foul_speed, EPS)

    def _mark_agent(i: int, mask: Tensor, impact: Tensor) -> None:
        """Mark agent i as an actual culprit and keep its strongest severity."""
        nonlocal foul_agent, foul_intensity
        if not bool(mask.any()):
            return
        foul_agent[:, i] = foul_agent[:, i] | mask
        sev = torch.clamp(
            (impact - cfg.robot_foul_speed) / robot_denom, 0.0, 1.0
        )
        foul_intensity[:, i] = torch.maximum(
            foul_intensity[:, i], torch.where(mask, sev, torch.zeros_like(sev))
        )

    # --- 1. wall fouls -------------------------------------------------
    wall_foul = torch.zeros((B, N), device=dev, dtype=torch.bool)
    for axis, lo, hi in ((0, -cfg.bound_x, cfg.bound_x), (1, -cfg.bound_y, cfg.bound_y)):
        at_lo = pos[..., axis] <= lo + 1e-6
        at_hi = pos[..., axis] >= hi - 1e-6
        imp_lo = torch.clamp(-vpre[..., axis], min=0.0)
        imp_hi = torch.clamp(vpre[..., axis], min=0.0)
        wall_foul = wall_foul | (at_lo & (imp_lo >= cfg.wall_foul_speed))
        wall_foul = wall_foul | (at_hi & (imp_hi >= cfg.wall_foul_speed))
    foul_agent = foul_agent | wall_foul
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
    foul_agent = foul_agent | cover_foul
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
                _mark_agent(i, foul_i, impact)
                _mark_agent(j, foul_j, impact)
                if same:
                    _foul_add(foul_team, ti, foul_tie)
                    _mark_agent(i, foul_tie, impact)
                    _mark_agent(j, foul_tie, impact)
                else:
                    _foul_add(foul_team, ti, foul_tie)
                    _foul_add(foul_team, tj, foul_tie)
                    _mark_agent(i, foul_tie, impact)
                    _mark_agent(j, foul_tie, impact)
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
                _mark_agent(i, foul_i, imp_i)
                _mark_agent(j, foul_j, imp_j)

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
    s.foul_agent = foul_agent
    s.foul_intensity = foul_intensity
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

    # The same step computes visibility up to three times: before target
    # selection, after the turret turn, and again when building the final
    # observation.  The last two calls are usually identical.  Memoise on the
    # exact inputs (step, positions, yaw, target and sample count) so this is a
    # pure reuse, not an approximation.
    cache = getattr(s, "_vis_cache", None)
    if cache is not None:
        c_step, c_samples, c_pos, c_yaw, c_target, c_out = cache
        if (
            c_step == s.step_count
            and c_samples == n_samples
            and torch.equal(c_pos, s.pos)
            and torch.equal(c_yaw, yaw)
            and torch.equal(c_target, locked_target)
        ):
            return c_out

    # Build all (observer, target) pairs at once.  The original loop called
    # visibility_arc 18 times with B rows; flattening to B*18 rows preserves
    # every per-pair operation exactly but removes the Python loop and gives
    # PyTorch a much larger, better-utilised batch.
    pair_i = torch.arange(N, device=dev).view(N, 1).expand(N, 3).reshape(-1)
    pair_k = torch.arange(3, device=dev).view(1, 3).expand(N, 3).reshape(-1)
    pair_j = s.enemy_idx.reshape(-1)
    P = pair_i.numel()

    obs_pos = s.pos[:, pair_i]                  # (B, P, 2)
    target_pos = s.pos[:, pair_j]               # (B, P, 2)
    rel = target_pos - obs_pos
    d = torch.linalg.vector_norm(rel, dim=-1)
    bearing = torch.atan2(rel[..., 1], rel[..., 0])
    yaw_pair = yaw[:, pair_i]                   # (B, P)
    err = G.wrap_angle(bearing - yaw_pair)
    prev = locked_target[:, pair_i]             # (B, P)
    half = torch.where(
        (prev == 0) | (prev == pair_k + 1),
        torch.full_like(err, cfg.fov_half),
        torch.full_like(err, cfg.fov_half_narrow),
    )
    in_fov_flat = err.abs() <= half

    # occluders: every other robot except self and target; 4 ids per pair.
    occ_ids = []
    for i in range(N):
        for k in range(3):
            j = int(s.enemy_idx[i, k])
            occ_ids.append([m for m in range(N) if m != i and m != j])
    occ_idx = torch.tensor(occ_ids, device=dev, dtype=torch.long)  # (P, 4)
    circles = torch.empty((B, P, occ_idx.shape[1], 3), device=dev)
    circles[..., :2] = s.pos[:, occ_idx]
    circles[..., 2] = cfg.occlusion_radius

    vis_args = (
        obs_pos.reshape(B * P, 2),
        yaw_pair.reshape(B * P),
        target_pos.reshape(B * P, 2),
        cfg.robot_radius,
        circles.reshape(B * P, occ_idx.shape[1], 3),
        s.cover_edges,
        n_samples,
    )
    if os.environ.get("RM_JIT", "0") == "1" and getattr(G, "VISIBILITY_ARC_JIT", None) is not None:
        pv = G.VISIBILITY_ARC_JIT(*vis_args, 1e-9)
    else:
        pv = G.visibility_arc(*vis_args)
    in_fov = in_fov_flat.reshape(B, N, 3)
    pv3 = pv.reshape(B, N, 3)
    p_vis = torch.where(in_fov, pv3, torch.zeros_like(pv3))
    visible_geom = in_fov & (p_vis > 0)
    s._vis_cache = (
        s.step_count,
        n_samples,
        s.pos.clone(),
        yaw.clone(),
        locked_target.clone(),
        (p_vis, in_fov, visible_geom),
    )
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
    s.step_count += s.B  # curriculum clock counts env steps, not vectorised steps
    s.episode_elapsed = s.episode_elapsed + cfg.dt
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
    s.last_kill_credit.zero_()
    s.last_kill_score.zero_()
    s.foul_loser.fill_(-1)
    s.foul_agent.zero_()
    s.foul_intensity.zero_()
    s.body_push_foul.zero_()
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
        j_k = s.enemy_idx[:, k - 1]
        alive_k = s.alive[:, j_k]
        dist_k = torch.linalg.vector_norm(
            s.pos[:, j_k] - s.pos, dim=-1
        )
        valid = (
            (s.target_cmd == k)
            & visible_pre[:, :, k - 1]
            & alive_k
            & (dist_k < cfg.lock_range)
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
                & (torch.linalg.vector_norm(s.pos[:, j] - s.pos[:, i], dim=-1) < cfg.lock_range)
            )
        lock_alive[:, i] = alive_i
    s.target = torch.where(lock_alive, s.target, torch.zeros_like(s.target))
    s.p_t = torch.where(lock_alive, s.p_t, torch.zeros_like(s.p_t))

    # ---- 5.8 heat / fire / damage -----------------------------------
    # cool all alive; own spawn (heal zone) cools at home_cool_mult times rate
    home_cool = torch.zeros((B, N), device=dev, dtype=torch.bool)
    for i in range(N):
        home_cool[:, i] = point_in_polygon_closed(s.pos[:, i], s.own_spawns[i])
    cool = torch.where(
        s.alive & home_cool,
        torch.full_like(s.heat, cfg.cool_per_frame * cfg.home_cool_mult),
        torch.full_like(s.heat, cfg.cool_per_frame),
    )
    s.heat = torch.where(s.alive, torch.clamp(s.heat - cool, min=0.0), s.heat)
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
    combat_event = (
        s.last_fired.any(dim=1) | (s.last_damage.sum(dim=(1, 2)) > 0.0)
    )
    s.stagnation_had_combat = s.stagnation_had_combat | combat_event
    s.stagnation_last_event = torch.where(
        combat_event, s.episode_elapsed, s.stagnation_last_event
    )

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
    s.kill_damage = s.kill_damage + damage
    s.last_kill_score = kill_add

    # individual kill credit: split a fixed pool by cumulative damage to the
    # victim; reset the victim's damage column for its next life.
    for j in range(N):
        died_j = died[:, j] & (total_dmg[:, j] > 0)
        if not bool(died_j.any()):
            s.kill_damage[:, :, j] = torch.where(
                died[:, j].unsqueeze(1),
                torch.zeros_like(s.kill_damage[:, :, j]),
                s.kill_damage[:, :, j],
            )
            continue
        col = s.kill_damage[:, :, j]
        share = col / col.sum(dim=1, keepdim=True).clamp_min(EPS)
        pool = share * cfg.k_kill_reward_pool
        s.last_kill_credit = s.last_kill_credit + pool * died_j.unsqueeze(1)
        s.kill_damage[:, :, j] = torch.where(
            died_j.unsqueeze(1),
            torch.zeros_like(col),
            col,
        )
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
        # 复活先保持无敌；热量清零，等无敌结束时再进入过热锁。
        s.heat = torch.where(revive, torch.zeros_like(s.heat), s.heat)
        s.overheated = s.overheated & (~revive)
        s.spin = torch.where(revive, torch.zeros_like(s.spin), s.spin)
        s.p_t = torch.where(revive, torch.zeros_like(s.p_t), s.p_t)
        s.target = torch.where(revive, torch.zeros_like(s.target), s.target)
        s.hit_cue_valid = s.hit_cue_valid & (~revive)
        s.vel = torch.where(revive.unsqueeze(-1), torch.zeros_like(s.vel), s.vel)
        # v1.3: inherit the gimbal yaw from the death frame.  A dead agent's yaw
        # is frozen by the turret update above, so s.yaw already holds it.
        if cfg.respawn_invuln:
            s.invuln = s.invuln | revive
            s.invuln_timer = torch.where(
                revive, torch.full_like(s.invuln_timer, cfg.invuln_time), s.invuln_timer
            )
        else:
            s.invuln = s.invuln & (~revive)
            s.invuln_timer = torch.where(revive, torch.zeros_like(s.invuln_timer), s.invuln_timer)
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
    hp_before_heal = s.hp.clone()
    s.hp = torch.where(heal, torch.clamp(s.hp + cfg.heal_per_frame, max=cfg.hp_max), s.hp)
    heal_event = (heal & (s.hp > hp_before_heal)).any(dim=1)
    s.stagnation_had_heal = s.stagnation_had_heal | heal_event
    s.stagnation_last_event = torch.where(
        heal_event, s.episode_elapsed, s.stagnation_last_event
    )

    # ---- 5.11 occupation / score -----------------------------------
    in_center = torch.linalg.vector_norm(s.pos, dim=-1) <= cfg.center_radius
    occ_red = (s.alive & (~s.invuln) & in_center & (s.team == 0).unsqueeze(0)).sum(dim=1).float()
    occ_blue = (s.alive & (~s.invuln) & in_center & (s.team == 1).unsqueeze(0)).sum(dim=1).float()
    s.occ = torch.stack((occ_red, occ_blue), dim=1)
    occ_event = s.occ.sum(dim=1) > 0.0
    s.stagnation_had_occ = s.stagnation_had_occ | occ_event
    s.stagnation_last_event = torch.where(
        occ_event, s.episode_elapsed, s.stagnation_last_event
    )
    score_add = torch.zeros((B, 2), device=dev)
    # Net occupation: only the surplus units over the opponent score.  With
    # 2 red + 1 blue inside, red gains one unit-share (0.1/frame), not 0.2.
    net_occ = occ_red - occ_blue
    score_add[:, 0] = cfg.score_per_frame * torch.clamp(net_occ, min=0.0)
    score_add[:, 1] = cfg.score_per_frame * torch.clamp(-net_occ, min=0.0)
    s.score = s.score + score_add

    # Empty-game speed relief: keep an EMA of actual speed since the last
    # event.  An agent that keeps moving during the 10 s window loses less
    # than one that parks, without rewarding a single high-speed terminal
    # frame.  Reset the EMA on any event that refreshes the empty-game timer.
    event_now = combat_event | heal_event | occ_event
    speed_now = torch.linalg.vector_norm(s.vel, dim=-1) / max(cfg.v_max, EPS)
    speed_now = speed_now.clamp(0.0, 1.0) * s.alive.float()
    alpha = min(cfg.dt / max(cfg.stagnation_speed_ema_time, EPS), 1.0)
    s.stagnation_speed = torch.where(
        event_now.unsqueeze(1),
        speed_now,
        (1.0 - alpha) * s.stagnation_speed + alpha * speed_now,
    )

    # v1.23: an empty first 20 s is terminated immediately with a negative
    # terminal reward for both teams instead of simulating a dead episode.
    inactive = (s.episode_elapsed - s.stagnation_last_event) >= cfg.stagnation_time
    stagnant = (~s.done) & inactive
    if stagnant.any():
        s.done = s.done | stagnant
        s.outcome = torch.where(
            stagnant, torch.full_like(s.outcome, OUTCOME_STAGNATION), s.outcome
        )
        s.winner = torch.where(stagnant, torch.zeros_like(s.winner), s.winner)
        s.foul_loser = torch.where(stagnant, torch.full_like(s.foul_loser, -1), s.foul_loser)
        _finish(s, score_before, snap, foul_any, gen)
        return

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

    # 无敌结束的同一帧进入枪口过热锁：不能靠无敌结束立刻开火，
    # 必须先冷却；复活瞬间仍然保持 3 s 无敌。
    if cfg.respawn_heat_lock and bool(expired.any()):
        s.heat = torch.where(expired, torch.full_like(s.heat, cfg.heat_max), s.heat)
        s.overheated = s.overheated | expired
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

    # v1.26: sustained intentional pushing into a wall/cover is a game-ending
    # illegal action.  It is an individual foul: the pusher gets the extra
    # intent-scaled penalty, teammates only a smaller loss.
    push_intent, push_contact = _wall_cover_push(s)
    pushing = (
        push_contact
        & (push_intent > cfg.wall_push_threshold)
        & s.alive
        & (~s.done.unsqueeze(-1))
    )
    s.wall_push_time = torch.where(
        pushing,
        s.wall_push_time + cfg.dt,
        torch.zeros_like(s.wall_push_time),
    )
    trigger = s.wall_push_time >= cfg.wall_push_time
    if trigger.any():
        current = pushing
        s.wall_push_foul_intent = torch.where(
            trigger,
            torch.where(current, push_intent, torch.zeros_like(push_intent)),
            torch.zeros_like(push_intent),
        )
        red_any = (current & (s.team == 0).unsqueeze(0)).any(dim=1)
        blue_any = (current & (s.team == 1).unsqueeze(0)).any(dim=1)
        trigger_row = trigger.any(dim=1)
        red_only = trigger_row & red_any & (~blue_any)
        blue_only = trigger_row & blue_any & (~red_any)
        both_push = trigger_row & red_any & blue_any
        s.done = s.done | trigger_row
        s.outcome = torch.where(
            red_only, torch.full_like(s.outcome, OUTCOME_BLUE_WALL_PUSH), s.outcome
        )
        s.winner = torch.where(red_only, torch.full_like(s.winner, 2), s.winner)
        s.outcome = torch.where(
            blue_only, torch.full_like(s.outcome, OUTCOME_RED_WALL_PUSH), s.outcome
        )
        s.winner = torch.where(blue_only, torch.ones_like(s.winner), s.winner)
        s.outcome = torch.where(
            both_push, torch.full_like(s.outcome, OUTCOME_BOTH_WALL_PUSH), s.outcome
        )
        s.winner = torch.where(both_push, torch.zeros_like(s.winner), s.winner)
        s.foul_loser = torch.where(
            trigger_row, torch.full_like(s.foul_loser, -1), s.foul_loser
        )

    _finish(s, score_before, snap, foul_any, gen)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------
def _wall_cover_push(s: RmState) -> Tuple[Tensor, Tensor]:
    """Return (push_intent [B,N], contact [B,N]) for walls and covers.

    push_intent uses the commanded velocity `v_pre` and the surface outward
    normal, so being pushed into a wall (actual velocity) does not count.
    """
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    x = s.pos[..., 0]
    y = s.pos[..., 1]
    wall_contact = (
        (x >= cfg.bound_x - cfg.robot_radius - 1e-3)
        | (x <= -cfg.bound_x + cfg.robot_radius + 1e-3)
        | (y >= cfg.bound_y - cfg.robot_radius - 1e-3)
        | (y <= -cfg.bound_y + cfg.robot_radius + 1e-3)
    )
    wall_nx = (
        (x >= cfg.bound_x - cfg.robot_radius - 1e-3).float()
        - (x <= -cfg.bound_x + cfg.robot_radius + 1e-3).float()
    )
    wall_ny = (
        (y >= cfg.bound_y - cfg.robot_radius - 1e-3).float()
        - (y <= -cfg.bound_y + cfg.robot_radius + 1e-3).float()
    )
    wall_norm = torch.linalg.vector_norm(
        torch.stack((wall_nx, wall_ny), dim=-1), dim=-1
    ).clamp_min(EPS)
    wall_normal = torch.stack((wall_nx, wall_ny), dim=-1) / wall_norm.unsqueeze(-1)
    wall_push = torch.clamp(
        (s.v_pre * wall_normal).sum(dim=-1), min=0.0
    ) * wall_contact.float()

    cover_contact = torch.zeros((B, N), device=dev, dtype=torch.bool)
    cover_push = torch.zeros((B, N), device=dev, dtype=torch.float32)
    for i in range(N):
        for poly in s.covers:
            _, dist, normal, inside = G.nearest_point_on_polygon(s.pos[:, i], poly)
            near = (dist <= cfg.robot_radius + 1e-3) | inside
            cover_contact[:, i] = cover_contact[:, i] | near
            push_i = torch.clamp(
                -(s.v_pre[:, i] * normal).sum(dim=-1), min=0.0
            )
            cover_push[:, i] = torch.maximum(
                cover_push[:, i], torch.where(near, push_i, torch.zeros_like(push_i))
            )
    contact = wall_contact | cover_contact
    intent = torch.clamp(
        (wall_push + cover_push) / cfg.v_max, 0.0, 1.0
    )
    return intent, contact


def _proximity_penalty(
    s: RmState, dmat: Tensor, pair_normal: Tensor
) -> Tensor:
    """Proximity pressure: both agents pay, 60% active / 40% passive.

    A pair is "close" inside cfg.proximity_radius (0.65 m).  The active side
    is the one whose pre-contact velocity points into the other robot harder;
    if neither (or both equally) move in, the split is 50/50.  The penalty
    grows with time in range (up to 2x) and with closeness.  This applies to
    teammates and corpses too; dead agents are masked out of dense reward
    later, but the living side still pays its share.
    """
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    diag = torch.arange(N, device=dev)
    near = dmat <= cfg.proximity_radius
    near[:, diag, diag] = False
    s.proximity_time = torch.where(
        near,
        s.proximity_time + cfg.dt,
        torch.zeros_like(s.proximity_time),
    )
    pen = torch.zeros((B, N), device=dev, dtype=s.pos.dtype)
    width = max(cfg.proximity_radius - 2.0 * cfg.robot_radius, EPS)
    for i in range(N):
        for j in range(i + 1, N):
            m = near[:, i, j]
            d = dmat[:, i, j]
            closeness = torch.clamp(
                (cfg.proximity_radius - d) / width, 0.0, 1.0
            )
            time_frac = torch.clamp(
                s.proximity_time[:, i, j] / max(cfg.proximity_ramp_time, EPS),
                0.0,
                1.0,
            )
            total = (
                (cfg.k_proximity_penalty + cfg.k_proximity_ramp * time_frac)
                * (0.2 + 0.8 * closeness)
                * m.float()
            )
            n_ij = pair_normal[:, i, j]
            v_in_i = torch.clamp(
                (s.v_pre[:, i] * n_ij).sum(dim=-1), min=0.0
            )
            v_in_j = torch.clamp(
                (s.v_pre[:, j] * (-n_ij)).sum(dim=-1), min=0.0
            )
            tie = (v_in_i == v_in_j)
            share_i = torch.where(
                v_in_i > v_in_j,
                torch.full_like(total, cfg.proximity_active_share),
                torch.full_like(total, cfg.proximity_passive_share),
            )
            share_i = torch.where(tie, torch.full_like(total, 0.5), share_i)
            pen[:, i] = pen[:, i] + total * share_i
            pen[:, j] = pen[:, j] + total * (1.0 - share_i)
    return pen


def _finalize_rewards(s: RmState, score_before: Tensor) -> None:
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    team = s.team.unsqueeze(0).expand(B, N)

    # Kill score is paid individually below; the team score reward only covers
    # the remaining score change (occupation / any other team score).
    score_delta = s.score - score_before - s.last_kill_score
    dense = cfg.k_score_reward * score_delta.gather(1, team)
    # small per-attacker hit shaping (actual/raw damage dealt to enemies)
    hit = cfg.k_hit_reward * (s.last_hit_damage / cfg.hp_max)
    kill_credit = s.last_kill_credit

    center = torch.linalg.vector_norm(s.pos, dim=-1)
    center_r = cfg.k_center_reward * torch.exp(
        -(center ** 2) / (2.0 * cfg.center_sigma ** 2)
    ) * (center > cfg.center_radius).float() * s.alive.float()
    # v1.15 potential shaping: reward reduction of distance to centre, so the
    # gradient exists from the spawn distance (the Gaussian above is tiny there).
    center_progress = (
        cfg.k_center_progress * (s.center_dist_prev - center) * s.alive.float()
    )
    in_point = (center <= cfg.center_radius) & s.alive & (~s.invuln)
    occupy = cfg.k_occupy_reward * in_point.float()

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
    # v1.23 anti-camping: after a grace period, stationary agents outside the
    # centre (especially in their own spawn) pay a small per-frame penalty.
    # Low-HP retreat and agents actively holding a target are exempt.
    x = s.pos[..., 0]
    y = s.pos[..., 1]
    in_own_spawn = torch.zeros_like(s.alive)
    for i in range(N):
        in_own_spawn[:, i] = point_in_polygon_closed(s.pos[:, i], s.own_spawns[i])
    # v1.25: while full HP inside own spawn, reward actual velocity on the two
    # valid exit axes, taking the larger component (not a sum).  Red exits
    # right (+x) or down (-y); blue exits left (-x) or up (+y).  Inward motion
    # keeps its negative sign.
    vx = s.vel[..., 0]
    vy = s.vel[..., 1]
    red_exit = torch.maximum(vx, -vy)
    blue_exit = torch.maximum(-vx, vy)
    outward_speed = torch.where(
        s.team.unsqueeze(0) == 0, red_exit, blue_exit
    ) / cfg.v_max
    outward_speed = torch.clamp(outward_speed, -1.0, 1.0)
    outward_reward = (
        cfg.k_outward_speed_reward
        * outward_speed
        * (s.hp >= cfg.hp_max - 1e-6).float()
        * in_own_spawn.float()
        * s.alive.float()
    )
    speed_now = torch.linalg.vector_norm(s.vel, dim=-1)
    elapsed = s.episode_elapsed
    camp = (
        in_own_spawn | (center > cfg.center_radius)
    ) & (low < 0.5) & (elapsed.unsqueeze(-1) > cfg.camp_grace_time) & (
        speed_now < cfg.camp_speed
    ) & (s.target == 0) & s.alive
    ramp = torch.clamp(
        (elapsed - cfg.camp_grace_time)
        / max(cfg.episode_time - cfg.camp_grace_time, 1e-6),
        0.0,
        1.0,
    )
    camp_weight = (
        cfg.k_camp_penalty + cfg.k_camp_ramp * ramp
    ).unsqueeze(-1)
    camp_pen = camp_weight * camp.float()

    # v1.22 contact shaping: wall/cover impact once, then a small penalty only
    # while the commanded velocity still pushes into the surface.  Agent-agent
    # contact is one event per pair (split equally between the two agents).
    contact_penalty = torch.zeros((B, N), device=dev, dtype=torch.float32)

    x = s.pos[..., 0]
    y = s.pos[..., 1]
    wall_x = (
        (x >= cfg.bound_x - cfg.robot_radius - 1e-3)
        | (x <= -cfg.bound_x + cfg.robot_radius + 1e-3)
    )
    wall_y = (
        (y >= cfg.bound_y - cfg.robot_radius - 1e-3)
        | (y <= -cfg.bound_y + cfg.robot_radius + 1e-3)
    )
    wall_now = wall_x | wall_y
    wall_nx = (
        (x >= cfg.bound_x - cfg.robot_radius - 1e-3).float()
        - (x <= -cfg.bound_x + cfg.robot_radius + 1e-3).float()
    )
    wall_ny = (
        (y >= cfg.bound_y - cfg.robot_radius - 1e-3).float()
        - (y <= -cfg.bound_y + cfg.robot_radius + 1e-3).float()
    )
    wall_norm = torch.linalg.vector_norm(
        torch.stack((wall_nx, wall_ny), dim=-1), dim=-1
    ).clamp_min(EPS)
    wall_normal = torch.stack((wall_nx, wall_ny), dim=-1) / wall_norm.unsqueeze(-1)
    wall_push = torch.clamp(
        (s.v_pre * wall_normal).sum(dim=-1), min=0.0
    ) * wall_now.float()

    cover_now = torch.zeros((B, N), device=dev, dtype=torch.bool)
    cover_push = torch.zeros((B, N), device=dev, dtype=torch.float32)
    for i in range(N):
        for poly in s.covers:
            _, dist, normal, inside = G.nearest_point_on_polygon(s.pos[:, i], poly)
            near = (dist <= cfg.robot_radius + 1e-3) | inside
            cover_now[:, i] = cover_now[:, i] | near
            # normal points outward; commanded motion into the cover is -normal
            push_i = torch.clamp(
                -(s.v_pre[:, i] * normal).sum(dim=-1), min=0.0
            )
            cover_push[:, i] = torch.maximum(
                cover_push[:, i], torch.where(near, push_i, torch.zeros_like(push_i))
            )

    push = wall_push + cover_push
    surface_now = wall_now | cover_now
    new_wall = surface_now & (~s.prev_wall_contact)
    wall_impact = cfg.k_wall_impact * new_wall.float() * torch.clamp(
        push / cfg.v_max, 0.0, 1.0
    )
    wall_push_pen = cfg.k_wall_push * surface_now.float() * (
        push > cfg.wall_push_min
    ).float() * torch.clamp(push / cfg.v_max, 0.0, 1.0)
    # Grow the wall/cover penalty with continuous contact duration, not with
    # the episode clock.  Sliding tangentially is cheaper than pushing in.
    s.wall_contact_time = torch.where(
        surface_now,
        s.wall_contact_time + cfg.dt,
        torch.zeros_like(s.wall_contact_time),
    )
    contact_frac = s.wall_contact_time / cfg.wall_contact_ramp_time
    push_frac = torch.clamp(push / cfg.v_max, 0.0, 1.0)
    wall_contact_pen = (
        (cfg.k_wall_contact_penalty + cfg.k_wall_contact_ramp * contact_frac)
        * (0.2 + 0.8 * push_frac)
        * surface_now.float()
    )
    contact_penalty = contact_penalty + wall_impact + wall_push_pen + wall_contact_pen

    dmat = torch.cdist(s.pos, s.pos)
    pair_now = dmat <= (2.0 * cfg.robot_radius + 1e-3)
    diag = torch.arange(N, device=dev)
    pair_now[:, diag, diag] = False
    pair_normal = (s.pos[:, None, :, :] - s.pos[:, :, None, :]) / dmat.clamp_min(EPS).unsqueeze(-1)
    rel_v = s.v_pre[:, :, None, :] - s.v_pre[:, None, :, :]
    approach = torch.clamp(
        (rel_v * pair_normal).sum(dim=-1), min=0.0
    )
    new_pair = pair_now & (~s.prev_pair_contact)
    pair_pen = (
        0.5
        * cfg.k_agent_impact
        * new_pair.float()
        * torch.clamp(approach / (2.0 * cfg.v_max), 0.0, 1.0)
    ).sum(dim=-1)
    contact_penalty = contact_penalty + pair_pen

    # Proximity pressure: both sides pay inside 0.65 m.  This is separate
    # from the one-shot impact and the sustained body-push foul.
    contact_penalty = contact_penalty + _proximity_penalty(
        s, dmat, pair_normal
    )

    # Body pushing: the one-time pair impact above is not enough, because a
    # sustained shove can be repeated every frame.  This applies to BOTH
    # teammates and opponents, and also to robots that are already dead (a
    # corpse is still a physical body).  Only a living agent whose own
    # pre-contact velocity points into another robot accumulates time; the
    # pushed robot/corpse is not penalised.  The ramp is deliberately unbounded.
    pusher_alive = s.alive.unsqueeze(2)
    other_present = s.alive.unsqueeze(1) | (s.respawn_timer > 0.0).unsqueeze(1)
    v_into_other = (s.v_pre.unsqueeze(2) * pair_normal).sum(dim=-1)
    body_push = (
        pair_now
        & pusher_alive
        & other_present
        & (v_into_other > cfg.body_push_min)
    )
    body_push_any = body_push.any(dim=-1)
    body_push_strength = torch.where(
        body_push, v_into_other, torch.zeros_like(v_into_other)
    ).amax(dim=-1)
    s.body_push_time = torch.where(
        body_push_any,
        s.body_push_time + cfg.dt,
        torch.zeros_like(s.body_push_time),
    )
    # A sustained robot-on-robot push is a game-ending foul at 0.5 s.
    body_trigger = (
        body_push_any
        & (s.body_push_time >= cfg.body_push_time)
        & (~s.done.unsqueeze(1))
    )
    if body_trigger.any():
        red_any = (body_trigger & (s.team == 0).unsqueeze(0)).any(dim=1)
        blue_any = (body_trigger & (s.team == 1).unsqueeze(0)).any(dim=1)
        trigger_row = body_trigger.any(dim=1)
        red_only = trigger_row & red_any & (~blue_any)
        blue_only = trigger_row & blue_any & (~red_any)
        both_push = trigger_row & red_any & blue_any
        s.done = s.done | trigger_row
        s.outcome = torch.where(
            red_only,
            torch.full_like(s.outcome, OUTCOME_BLUE_BODY_PUSH),
            s.outcome,
        )
        s.winner = torch.where(
            red_only, torch.full_like(s.winner, 2), s.winner
        )
        s.outcome = torch.where(
            blue_only,
            torch.full_like(s.outcome, OUTCOME_RED_BODY_PUSH),
            s.outcome,
        )
        s.winner = torch.where(
            blue_only, torch.ones_like(s.winner), s.winner
        )
        s.outcome = torch.where(
            both_push,
            torch.full_like(s.outcome, OUTCOME_BOTH_BODY_PUSH),
            s.outcome,
        )
        s.winner = torch.where(
            both_push, torch.zeros_like(s.winner), s.winner
        )
        s.foul_loser = torch.where(
            trigger_row, torch.full_like(s.foul_loser, -1), s.foul_loser
        )
        s.body_push_foul = body_trigger
    push_frac = s.body_push_time / cfg.body_push_ramp_time
    push_speed = torch.clamp(
        body_push_strength / (2.0 * cfg.v_max), 0.0, 1.0
    )
    body_push_pen = (
        (cfg.k_body_push_penalty + cfg.k_body_push_ramp * push_frac)
        * (0.2 + 0.8 * push_speed)
        * body_push_any.float()
    )
    contact_penalty = contact_penalty + body_push_pen

    # Actual contact (not the 0.65 m proximity ring) clears spin.  Both
    # robots are cleared on a robot-robot hit; the contacting agent is
    # cleared on a wall/cover hit.  Once separated and re-contacted, it clears
    # again; staying in contact does not keep clearing every frame.
    contact_now = new_pair.any(dim=-1) | new_wall
    s.spin = torch.where(contact_now, torch.zeros_like(s.spin), s.spin)

    s.prev_wall_contact = surface_now.detach()
    s.prev_pair_contact = pair_now.detach()

    fire_heat_pen = cfg.k_fire_heat_penalty * s.last_fired.float()

    dense = (
        dense
        + center_r
        + center_progress
        + occupy
        + survival
        + home
        + outward_reward
        + kill_credit
        - contact_penalty
        - fire_heat_pen
        - camp_pen
    )

    terminal = torch.zeros((B, N), device=dev)
    done = s.done
    win_team = s.winner  # 0 draw, 1 red, 2 blue

    # Normal win: +6 base, plus up to +3 for winning early.  Normal loss is
    # fixed -6, independent of the score margin.  The special push fouls are
    # applied below and override this block.
    normal = done & (win_team != 0) & (s.foul_loser < 0)
    win_side = team == (win_team.unsqueeze(1) - 1)
    time_frac = torch.clamp(
        s.time / max(cfg.episode_time, EPS), 0.0, 1.0
    )
    winner_score = torch.where(win_team == 1, s.score[:, 0], s.score[:, 1])
    loser_score = torch.where(win_team == 1, s.score[:, 1], s.score[:, 0])
    margin_frac = torch.clamp(
        (winner_score - loser_score) / max(cfg.target_score, EPS), 0.0, 1.0
    )
    win_score = (
        cfg.terminal_win
        + cfg.k_terminal_time * time_frac
        + cfg.k_terminal_margin * margin_frac
    )
    base = torch.where(
        win_side,
        win_score.unsqueeze(1).expand_as(terminal),
        torch.full_like(terminal, cfg.terminal_loss),
    )
    terminal = torch.where(normal.unsqueeze(1), base, terminal)

    terminal = torch.where(done.unsqueeze(1) & (win_team == 0).unsqueeze(1), torch.full_like(terminal, cfg.terminal_draw), terminal)
    terminal = torch.where(done.unsqueeze(1) & s.both_200.unsqueeze(1), torch.full_like(terminal, cfg.terminal_both_200), terminal)

    # Collision foul: only the actual culprit(s) pay.  Base -10 plus
    # 3 * normalized collision intensity.  Opponents get no positive reward.
    culprit = done.unsqueeze(1) & s.foul_agent
    foul_pen = cfg.terminal_foul - cfg.k_foul_intensity * s.foul_intensity
    terminal = torch.where(culprit, foul_pen, terminal)
    stagnant = done & (s.outcome == OUTCOME_STAGNATION)
    # Use the nearest obstacle, not the 8-ray mean: a corner has long
    # cross-field rays but a very small minimum clearance.
    clearance = (
        s.obstacle_dist.min(dim=-1).values
        * (cfg.lidar_range / cfg.stagnation_clearance_ref)
    ).clamp(0.0, 1.0)
    stagnation_pen = cfg.stagnation_penalty * (
        1.0 - cfg.stagnation_clearance_discount * clearance
    )
    # Per-agent spawn-distance relief: an empty-game terminal should not
    # punish an agent that actually left its spawn as much as one that stayed
    # parked there.  The multiplier is per-agent and composes with the
    # obstacle-clearance and team-centre discounts below.
    spawn_centers = torch.stack(
        [p.mean(dim=0) for p in s.own_spawns], dim=0
    )  # (N,2)
    spawn_far = (
        torch.linalg.vector_norm(s.pos - spawn_centers.unsqueeze(0), dim=-1)
        / max(float(cfg.stagnation_spawn_ref), EPS)
    ).clamp(0.0, 1.0)
    stagnation_pen = stagnation_pen * (
        1.0 - cfg.stagnation_spawn_discount * spawn_far
    )
    # The side closer to the centre (alive centroid) keeps a larger part of
    # its terminal reward.  With equal distance both keep the full penalty.
    alive_f = s.alive.float()
    team_center_dist = []
    for t in (0, 1):
        team_mask = (team == t).float() * alive_f
        count = team_mask.sum(dim=1)
        centroid = (s.pos * team_mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1.0).unsqueeze(-1)
        dist = torch.linalg.vector_norm(centroid, dim=-1)
        # A wiped team is treated as maximally far, so the surviving side gets
        # the near-centre discount when applicable.
        team_center_dist.append(
            torch.where(count > 0.5, dist, torch.full_like(dist, 1e3))
        )
    d_red, d_blue = team_center_dist
    denom = (d_red + d_blue).clamp_min(EPS)
    red_adv = ((d_blue - d_red) / denom).clamp(0.0, 1.0)
    blue_adv = ((d_red - d_blue) / denom).clamp(0.0, 1.0)
    team_factor = 1.0 - cfg.stagnation_center_discount * torch.stack(
        (red_adv, blue_adv), dim=1
    )
    # stagnation_pen is per-agent (it includes that agent's own obstacle
    # clearance); the centre discount is per-team.
    team_factor = torch.where(team == 0, team_factor[:, 0:1], team_factor[:, 1:2])
    stagnation_pen = stagnation_pen * team_factor
    # Per-agent speed relief: moving during the empty window is less
    # punishable than parking.  This uses the EMA maintained in step_post,
    # so a single fast frame cannot game the terminal penalty.
    speed_factor = 1.0 - cfg.stagnation_speed_discount * s.stagnation_speed.clamp(0.0, 1.0)
    stagnation_pen = stagnation_pen * speed_factor
    terminal = torch.where(
        stagnant.unsqueeze(1),
        -stagnation_pen,
        terminal,
    )

    # Wall/cover push foul: the pusher gets -10; teammates -0.5; the
    # opponent gets no positive reward.  If both teams push, both pushers get
    # -10 and the non-pushers on both teams get -0.5 (a draw).
    wall_foul = done & (
        (s.outcome == OUTCOME_RED_WALL_PUSH)
        | (s.outcome == OUTCOME_BLUE_WALL_PUSH)
        | (s.outcome == OUTCOME_BOTH_WALL_PUSH)
    )
    if wall_foul.any():
        pusher = s.wall_push_foul_intent > 0.0
        red_has = (pusher & (team == 0)).any(dim=1)
        blue_has = (pusher & (team == 1)).any(dim=1)
        team_has = torch.where(
            team == 0,
            red_has.unsqueeze(1),
            blue_has.unsqueeze(1),
        )
        wall_term = torch.zeros_like(terminal)
        wall_term = torch.where(
            pusher,
            torch.full_like(wall_term, cfg.terminal_push_foul),
            wall_term,
        )
        wall_term = torch.where(
            (~pusher) & team_has,
            torch.full_like(wall_term, -cfg.wall_push_teammate_penalty),
            wall_term,
        )
        terminal = torch.where(wall_foul.unsqueeze(1), wall_term, terminal)

    # Sustained robot-on-robot push foul: same economics as wall push.
    body_foul = done & (
        (s.outcome == OUTCOME_RED_BODY_PUSH)
        | (s.outcome == OUTCOME_BLUE_BODY_PUSH)
        | (s.outcome == OUTCOME_BOTH_BODY_PUSH)
    )
    if body_foul.any():
        pusher = s.body_push_foul
        red_has = (pusher & (team == 0)).any(dim=1)
        blue_has = (pusher & (team == 1)).any(dim=1)
        team_has = torch.where(
            team == 0,
            red_has.unsqueeze(1),
            blue_has.unsqueeze(1),
        )
        body_term = torch.zeros_like(terminal)
        body_term = torch.where(
            pusher,
            torch.full_like(body_term, cfg.terminal_push_foul),
            body_term,
        )
        body_term = torch.where(
            (~pusher) & team_has,
            torch.full_like(body_term, -cfg.body_push_teammate_penalty),
            body_term,
        )
        terminal = torch.where(body_foul.unsqueeze(1), body_term, terminal)

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
    s.center_dist_prev = center.detach()
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


def obstacle_distances_for_state(s: RmState) -> Tensor:
    """[B,N,8] world-frame clearance to static walls/covers."""
    cfg = s.cfg
    # One batched ray cast over all agents instead of one call per agent.
    # The ray geometry is independent per row, so this is numerically the
    # same result with less dispatch overhead.
    origin = s.pos.reshape(s.B * cfg.n_agents, 2)
    out = G.obstacle_distances(
        origin, s.cover_edges,
        cfg.lidar_range, cfg.field_hx, cfg.field_hy,
        cfg.obstacle_rays,
    )
    return out.reshape(s.B, cfg.n_agents, cfg.obstacle_rays)


def canonicalize_obstacles(obst: Tensor, s: RmState) -> Tensor:
    """Rotate world-frame obstacle clearances by 180° for blue.

    The fixed 8-ray order is E,NE,N,NW,W,SW,S,SE; a 180° rotation maps
    index i -> (i+4)%8.
    """
    blue = (s.team == 1).view(1, -1, 1)
    rotated = obst[..., [4, 5, 6, 7, 0, 1, 2, 3]]
    return torch.where(blue, rotated, obst)


class _StateRows:
    """Lightweight row-subset view used to rebuild only reset environments."""

    def __init__(self, base: RmState, rows):
        object.__setattr__(self, "_base", base)
        object.__setattr__(
            self,
            "_rows",
            torch.as_tensor(rows, device=base.device, dtype=torch.long).reshape(-1),
        )
        local = {
            "B": int(len(rows)),
            "cfg": base.cfg,
            "device": base.device,
            "team": base.team,
            "enemy_idx": base.enemy_idx,
            "mate_idx": base.mate_idx,
            "covers": base.covers,
            "cover_edges": base.cover_edges,
            "own_spawns": base.own_spawns,
            "enemy_spawns": base.enemy_spawns,
        }
        object.__setattr__(self, "_local", local)

    def __getattr__(self, name):
        local = object.__getattribute__(self, "_local")
        if name in local:
            return local[name]
        base = object.__getattribute__(self, "_base")
        value = getattr(base, name)
        rows = object.__getattribute__(self, "_rows")
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == base.B:
            return value[rows]
        return value

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._local[name] = value


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
    own_centers = torch.stack([p.mean(dim=0) for p in s.own_spawns], dim=0)  # (N,2)
    obstacle = obstacle_distances_for_state(s)
    s.obstacle_dist = obstacle
    obstacle_c = canonicalize_obstacles(obstacle, s)

    for i in range(N):
        o = []
        own_in = point_in_polygon_closed(s.pos[:, i], s.own_spawns[i])
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
        o += [own_in.float()]
        o += [(torch.linalg.vector_norm(s.pos[:, i], dim=-1) <= cfg.center_radius).float()]
        o += [point_in_polygon_closed(s.pos[:, i], s.enemy_spawns[i]).float()]
        o += [torch.linalg.vector_norm(s.pos[:, i], dim=-1) / cfg.lock_range]
        # own spawn centre relative to the agent (3); canonical for blue
        rel_spawn = sign[:, i, 0].unsqueeze(-1) * (
            own_centers[i].view(1, 2) - s.pos[:, i]
        )
        o += [rel_spawn[:, 0] / cfg.field_hx,
              rel_spawn[:, 1] / cfg.field_hy,
              torch.linalg.vector_norm(rel_spawn, dim=-1) / cfg.lock_range,
              own_in.float()]
        # centre token (4), same shape as the spawn token
        rel_center = sign[:, i, 0].unsqueeze(-1) * (-s.pos[:, i])
        center_dist = torch.linalg.vector_norm(s.pos[:, i], dim=-1)
        o += [rel_center[:, 0] / cfg.field_hx,
              rel_center[:, 1] / cfg.field_hy,
              center_dist / cfg.lock_range,
              (center_dist <= cfg.center_radius).float()]
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
            # v1.22: beyond lock_range they remain visible but cannot be locked.
            mask[:, i, k + 1] = (
                vis & alive_j
                & (torch.linalg.vector_norm(rel, dim=-1) < cfg.lock_range)
            )
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
        # 8 world-frame obstacle clearances, canonicalised for blue
        o += [obstacle_c[:, i, k] for k in range(OBSTACLE_DIM)]
        obs[:, i] = torch.stack(o, dim=-1)

    # dead agents: only "none" target
    dead = ~s.alive
    mask = torch.where(dead.unsqueeze(-1), torch.zeros_like(mask), mask)
    mask[:, :, 0] = True
    if bool(getattr(cfg, "obs_fp16", False)):
        s.obs = obs.to(torch.float16)
    else:
        s.obs = obs
    s.action_mask = mask
    # Fire is a binary trigger: 0 (release) is always legal, 1 (press) is
    # legal only when a target is actually locked.  This prevents a policy
    # with target=0 from wasting heat/ammo and makes deployment honest.
    s.fire_mask = torch.stack(
        (torch.ones_like(s.target, dtype=torch.bool), s.target != 0), dim=-1
    )
    gs = _build_global_state(s)
    s.global_state = gs.to(torch.float16) if bool(getattr(cfg, "obs_fp16", False)) else gs


def build_observation_rows(s: RmState, rows) -> None:
    """Rebuild observation/global-state/action-mask only for reset rows.

    TorchRL calls ``reset_at`` once per resetting environment.  Rebuilding all
    ``B`` rows on each such call is O(num_resets * B) and was the dominant
    reset cost.  ``_StateRows`` exposes a row view with the same field names;
    all observation logic is per-row, so the result is identical for those
    rows.  Other rows keep the observation produced by the last full step.
    """
    rows_t = torch.as_tensor(rows, device=s.device, dtype=torch.long).reshape(-1)
    if rows_t.numel() == 0:
        return
    view = _StateRows(s, rows_t)
    build_observation(view)
    s.obs[rows_t] = view.obs
    s.action_mask[rows_t] = view.action_mask
    s.fire_mask[rows_t] = view.fire_mask
    s.global_state[rows_t] = view.global_state
    s.obstacle_dist[rows_t] = view.obstacle_dist


def _build_global_state(s: RmState) -> Tensor:
    cfg = s.cfg
    B, N = s.B, cfg.n_agents
    dev = s.device
    parts = []
    # build_observation has just computed the same obstacle tensor for the
    # actor; reuse it rather than casting every ray twice per step.
    obstacle = s.obstacle_dist
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
        parts += [obstacle[:, i, k] for k in range(OBSTACLE_DIM)]
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
