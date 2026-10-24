#  Copyright (c) 2026.
#  RM 3v3 scenario wrapper for VMAS.
#
#  The rule engine lives in rm_rules.py; this file only adapts VMAS agents,
#  actions and rendering.  It does not depend on BenchMARL and can be used
#  standalone for the acceptance simulator.
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

from vmas.scenarios.rm_config import RmConfig, DEFAULT_CONFIG
from vmas.scenarios import rm_rules as R
from vmas.scenarios import rm_geometry as G


class Scenario(BaseScenario):
    """RM 3v3 simplified rules v1.2."""

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        cfg = kwargs.pop("cfg", None) or RmConfig(**{
            k: v for k, v in kwargs.items() if hasattr(DEFAULT_CONFIG, k)
        })
        self.cfg = cfg
        self.seed = int(kwargs.pop("seed", 0))
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(self.seed)

        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=cfg.dt,
            substeps=1,
            drag=0.0,
            linear_friction=0.0,
            angular_friction=0.0,
            x_semidim=cfg.bound_x,
            y_semidim=cfg.bound_y,
            collision_force=0.0,   # collision is handled by rm_rules
        )
        world.visualize_semidims = True

        self.agent_names: List[str] = []
        self.name_to_index: Dict[str, int] = {}
        for i in range(cfg.n_agents):
            team = cfg.team(i)
            slot = i % cfg.n_per_team + 1
            name = f"{'R' if team == 0 else 'B'}{slot}"
            color = Color.RED if team == 0 else Color.BLUE
            agent = Agent(
                name=name,
                collide=False,
                movable=True,
                rotatable=False,
                shape=Sphere(radius=cfg.robot_radius),
                dynamics=Holonomic(),
                u_range=[1.0, 1.0, 1.0, 1.0, 3.0, 1.0],
                action_size=6,
                mass=cfg.mass,
                drag=0.0,
                linear_friction=0.0,
                max_speed=cfg.v_max,
                color=color,
            )
            # P controller with gain 1/dt: force = (v_target - v)/dt, so VMAS
            # integrates exactly to v_target in one step (mass = 1, no drag).
            agent.controller = VelocityController(
                agent, world, [1.0 / cfg.dt, 0.0, 0.0], "standard"
            )
            world.add_agent(agent)
            self.agent_names.append(name)
            self.name_to_index[name] = i

        self._add_landmarks(world, cfg)
        self.s = R.make_state(batch_dim, device, cfg)
        self.step_count = 0
        # reset_at() is called once per resetting env by torchrl.  Building the
        # full observation every time is O(num_resets * B); defer it until the
        # next observation/global-state access.
        self._obs_dirty = False
        self._obs_dirty_rows = None
        return world

    def _add_landmarks(self, world: World, cfg: RmConfig):
        def box(name, cx, cy, lx, ly, color=Color.GRAY):
            lm = Landmark(
                name=name,
                collide=False,
                movable=False,
                shape=Box(length=lx, width=ly),
                color=color,
            )
            world.add_landmark(lm)
            lm.state.pos = torch.tensor(
                [cx, cy], device=world.device, dtype=torch.float32
            ).repeat(world.batch_dim, 1)
            return lm

        # red L: vertical + horizontal bar (visual only; collision uses polygon)
        box("red_l_v", -3.40, 1.03, 0.20, 4.06, Color.RED)
        box("red_l_h", -2.50, 2.96, 2.00, 0.20, Color.RED)
        # blue L
        box("blue_l_v", 3.40, -1.03, 0.20, 4.06, Color.BLUE)
        box("blue_l_h", 2.50, -2.96, 2.00, 0.20, Color.BLUE)
        # short covers
        box("red_short", -1.575, 0.80, 0.15, 1.80, Color.GRAY)
        box("blue_short", 1.575, -0.80, 0.15, 1.80, Color.GRAY)

    # ------------------------------------------------------------------
    # VMAS hooks
    # ------------------------------------------------------------------
    def reset_world_at(self, env_index: Optional[int] = None):
        rows = None
        if env_index is not None:
            rows = env_index.reshape(-1).tolist() if torch.is_tensor(env_index) else [env_index]
        R.reset_state(
            self.s, rows, mode=self.cfg.init_mode, seed=self.seed,
            build_obs=False,
        )
        if rows is None:
            self._obs_dirty = True
            self._obs_dirty_rows = None
        else:
            if not self._obs_dirty:
                self._obs_dirty_rows = set()
            if self._obs_dirty_rows is not None:
                self._obs_dirty_rows.update(rows)
            self._obs_dirty = True
        for i, agent in enumerate(self.world.agents):
            agent.state.pos = self.s.pos[:, i].clone()
            agent.state.vel = self.s.vel[:, i].clone()
            if rows is not None:
                agent.controller.reset(torch.tensor(rows, device=self.s.device, dtype=torch.long))
            else:
                agent.controller.reset()
        self.step_count = 0

    def process_action(self, agent: Agent):
        i = self.name_to_index[agent.name]
        raw_u = agent.action.u
        if raw_u.shape[-1] < 6:
            # controller from the previous frame may have replaced action.u; pad
            pad = torch.zeros((raw_u.shape[0], 6 - raw_u.shape[-1]), device=raw_u.device)
            raw_u = torch.cat([raw_u, pad], dim=-1)
        v_target = R.decode_agent_action(self.s, i, raw_u, agent.state.vel)
        agent.action.u = torch.cat(
            [v_target, torch.zeros((v_target.shape[0], 4), device=v_target.device)], dim=-1
        )
        agent.controller.process_force()

    def post_step(self):
        for i, agent in enumerate(self.world.agents):
            self.s.pos[:, i] = agent.state.pos
            self.s.vel[:, i] = agent.state.vel
        R.step_post(self.s, generator=self.gen)
        for i, agent in enumerate(self.world.agents):
            agent.state.pos = self.s.pos[:, i].clone()
            agent.state.vel = self.s.vel[:, i].clone()
        self.step_count += 1

    def _ensure_obs(self):
        if self._obs_dirty:
            if self._obs_dirty_rows is None:
                R.build_observation(self.s)
            else:
                R.build_observation_rows(self.s, sorted(self._obs_dirty_rows))
            self._obs_dirty = False
            self._obs_dirty_rows = None

    def observation(self, agent: Agent):
        self._ensure_obs()
        i = self.name_to_index[agent.name]
        return self.s.obs[:, i]

    def reward(self, agent: Agent):
        i = self.name_to_index[agent.name]
        return self.s.reward[:, i]

    def done(self):
        return self.s.done

    def info(self, agent: Agent):
        return {
            "outcome": self.s.outcome,
            "winner": self.s.winner,
            "score_red": self.s.score[:, 0],
            "score_blue": self.s.score[:, 1],
            "time": self.s.time,
            "occ_red": self.s.occ[:, 0],
            "occ_blue": self.s.occ[:, 1],
            "hp": self.s.hp[:, self.name_to_index[agent.name]],
            "heat": self.s.heat[:, self.name_to_index[agent.name]],
            "spin": self.s.spin[:, self.name_to_index[agent.name]],
        }

    def get_action_mask(self) -> torch.Tensor:
        return self.s.action_mask

    def get_fire_mask(self) -> torch.Tensor:
        return self.s.fire_mask

    def get_global_state(self) -> torch.Tensor:
        self._ensure_obs()
        return self.s.global_state

    # ------------------------------------------------------------------
    # debug / simulator helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Plain tensor snapshot for the browser simulator."""
        s = self.s
        return {
            "pos": s.pos.detach().cpu(),
            "vel": s.vel.detach().cpu(),
            "yaw": s.yaw.detach().cpu(),
            "hp": s.hp.detach().cpu(),
            "heat": s.heat.detach().cpu(),
            "overheated": s.overheated.detach().cpu(),
            "spin": s.spin.detach().cpu(),
            "alive": s.alive.detach().cpu(),
            "invuln": s.invuln.detach().cpu(),
            "invuln_timer": s.invuln_timer.detach().cpu(),
            "respawn_timer": s.respawn_timer.detach().cpu(),
            "target": s.target.detach().cpu(),
            "p_t": s.p_t.detach().cpu(),
            "last_fired": s.last_fired.detach().cpu(),
            "last_damage": s.last_damage.detach().cpu(),
            "score": s.score.detach().cpu(),
            "time": s.time.detach().cpu(),
            "occ": s.occ.detach().cpu(),
            "outcome": s.outcome.detach().cpu(),
            "winner": s.winner.detach().cpu(),
            "action_mask": s.action_mask.detach().cpu(),
            "team": s.team.detach().cpu(),
        }

    def extra_render(self, env_index: int = 0):
        # Keep VMAS's own renderer minimal; the acceptance simulator draws its own HUD.
        return []
