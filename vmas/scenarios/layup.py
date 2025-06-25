import torch
from vmas import render_interactively
from vmas.simulator.core import World, Agent, Landmark, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.utils import Color
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils, X
from vmas.simulator.controllers.velocity_controller import VelocityController

class Scenario(BaseScenario):
    """
    "飞身上篮"简化版2v2投篮强化学习环境
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # ----------------- 超参数设定 (Hyperparameters) -----------------
        # 场地尺寸
        self.W = kwargs.get("W", 8.0)
        self.L = kwargs.get("L", 15.0)
        self.spawn_area_depth = kwargs.get("spawn_area_depth", 1.0)
        # 投篮区域半径
        self.R_spot = kwargs.get("R_spot", 1.5)
        # 进攻时限 (秒)
        self.t_limit = kwargs.get("t_limit", 20.0)
        # 物理步长
        self.dt = kwargs.get("dt", 0.1)
        # 智能体物理属性
        self.agent_radius = kwargs.get("agent_radius", 0.3)
        self.a_max = kwargs.get("a_max", 3.0) # 最大加速度
        self.v_max = kwargs.get("v_max", 5.0) # 最大速度
        
        # 终止条件阈值
        self.v_shot_threshold = kwargs.get("v_shot_threshold", 0.1) # 投篮速度阈值
        self.a_shot_threshold = kwargs.get("a_shot_threshold", 0.1) # 投篮加速度阈值
        self.v_foul_threshold = kwargs.get("v_foul_threshold", 2.0) # 碰撞犯规速度阈值

        # 奖励系数
        self.max_score = kwargs.get("max_score", 100.0)
        self.R_foul = kwargs.get("R_foul", 50.0)
        self.k_a1_approach = kwargs.get("k_a1_approach", 2.0)
        self.k_a1_in_spot_reward = kwargs.get("k_a1_in_spot_reward", 2.0)
        self.k_def_block = kwargs.get("k_def_block", 5.0)
        self.block_sigma = kwargs.get("block_sigma", 2.5)
        self.k_coll_active = kwargs.get("k_coll_active", 2.0)
        self.k_coll_passive = kwargs.get("k_coll_passive", 1.0)
        self.oob_penalty = kwargs.get("oob_penalty", -5.0) # 出界惩罚值
        self.def_proximity_threshold = kwargs.get("def_proximity_threshold", 1.0) # 防守奖励生效的最大距离
        self.overextend_penalty = kwargs.get("overextend_penalty", -0.1) # 防守方越过半场的惩罚
        self.k_def_spot_approach = kwargs.get("k_def_spot_approach", 0.5) # 防守方接近隐藏投篮点的奖励系数
        self.k_a1_blocked_penalty = kwargs.get("k_a1_blocked_penalty", -2.0) # A1被封堵的惩罚系数
        self.k_a2_screen = kwargs.get("k_a2_screen", 3.0) # A2执行掩护的奖励系数
        self.k_a2_crowding_penalty = kwargs.get("k_a2_crowding_penalty", 0.01) # A2因过于靠近A1而受到的扎堆惩罚系数
        self.screen_sigma = kwargs.get("screen_sigma", 0.5)

        # ----------------- 环境构建 (World Setup) -----------------
        self.max_steps = int(self.t_limit / self.dt)
        self.n_agents = 4
        self.n_attackers = 2
        self.n_defenders = 2

        world = World(batch_dim, device, dt=self.dt, substeps=4,
                        x_semidim=self.W / 2, y_semidim=self.L / 2)

        # 智能体
        self.attackers = []
        self.defenders = []
        for i in range(self.n_agents):
            is_attacker = i < self.n_attackers
            team_name = "attacker" if is_attacker else "defender"
            agent_id = i + 1 if is_attacker else i - self.n_attackers + 1
            agent = Agent(
                name=f"{team_name}_{agent_id}",
                collide=True,
                movable=True,
                rotatable=False,
                u_range = self.v_max,
                drag = 0.01,
                shape=Sphere(radius=self.agent_radius),
                dynamics=Holonomic(),
                render_action=True,
                color=Color.RED if is_attacker and agent_id == 1 else Color.BLUE if not is_attacker else Color.PINK
            )
            agent.is_attacker = is_attacker
            agent.controller = VelocityController(
            agent, world, [6,0,0.01], "parallel"
        )
            world.add_agent(agent)
            if is_attacker: self.attackers.append(agent)
            else: self.defenders.append(agent)

        # 地标
        self.basket = Landmark(name="basket", collide=False, shape=Sphere(radius=0.1), color=Color.ORANGE)
        self.spot_center = Landmark(name="spot_center", collide=False, shape=Sphere(radius=0.05), color=Color.GREEN)
        self.shooting_area_vis = Landmark(name="shooting_area_vis", collide=False, shape=Sphere(radius=self.R_spot), color=Color.LIGHT_GREEN)
        center_line = Landmark(name="center_line",collide=False,shape=Line(length=self.W),color=Color.GRAY
        )
        world.add_landmark(center_line)
        world.add_landmark(self.basket)
        world.add_landmark(self.spot_center)
        world.add_landmark(self.shooting_area_vis)

        # 内部状态
        self.t_remaining = torch.zeros(batch_dim, 1, device=device)
        self.prev_dist_to_spot = torch.zeros(batch_dim, self.n_agents, device=device)
        self.terminal_rewards = torch.zeros(batch_dim, self.n_agents, device=device)
        self.dones = None
        self.p_vels = torch.zeros((batch_dim, self.n_agents, 2), device=device)

        return world

    def reset_world_at(self, env_index: int | None = None):
        if env_index is None:
            batch_dim = self.world.batch_dim
            batch_range = slice(None)
        else:
            batch_dim = 1
            batch_range = env_index

        self.t_remaining[batch_range] = self.t_limit
        
        basket_pos = torch.zeros(batch_dim, 2, device=self.world.device)
        basket_pos[:, 1] = self.L / 2 - 0.6

        self.basket.set_pos(basket_pos, batch_index=env_index)

        spot_x = (torch.rand(batch_dim, 1, device=self.world.device) - 0.5) * self.W
        spot_y = torch.rand(batch_dim, 1, device=self.world.device) * (self.L / 4)
        spot_pos = torch.cat([spot_x, spot_y], dim=1)
        self.spot_center.set_pos(spot_pos, batch_index=env_index)
        self.shooting_area_vis.set_pos(spot_pos, batch_index=env_index)

        # --- 矢量化拒绝采样 ---
        min_dist = self.agent_radius * 2
        
        # 存储所有智能体的最终生成位置
        agent_positions = torch.zeros(batch_dim, self.n_agents, 2, device=self.world.device)
        
        # 1. 分层放置每个智能体
        for i in range(self.n_agents):
            agent = self.world.agents[i]
            
            # 生成当前智能体的候选位置
            def sample_positions():
                x_pos = (torch.rand(batch_dim, 1, device=self.world.device) - 0.5) * self.W
                if agent.is_attacker:
                    y_pos = -self.L / 2 + torch.rand(batch_dim, 1, device=self.world.device) * self.spawn_area_depth
                else:
                    y_pos = self.L / 2 - torch.rand(batch_dim, 1, device=self.world.device) * self.spawn_area_depth
                return torch.cat([x_pos, y_pos], dim=1)

            candidate_positions = sample_positions()
            
            # 2. 对于已放置的智能体，进行碰撞检测和重采样
            if i > 0:
                # 设置最大重试次数以避免死循环
                for _ in range(10): 
                    # (batch_dim, i, 2)
                    previous_positions = agent_positions[:, :i, :]
                    # (batch_dim, 1, 2)
                    current_positions_exp = candidate_positions.unsqueeze(1)
                    
                    # 计算与所有已放置智能体的距离
                    distances = torch.linalg.norm(current_positions_exp - previous_positions, dim=2)
                    
                    # 找到任何一个小于最小距离的碰撞
                    collisions = torch.any(distances < min_dist, dim=1)
                    
                    # 如果没有碰撞了，就跳出重试循环
                    if not torch.any(collisions):
                        break
                    
                    # 3. 仅为发生碰撞的环境重新生成位置
                    num_collisions = collisions.sum()
                    if num_collisions > 0:
                        new_samples = sample_positions()
                        candidate_positions[collisions] = new_samples[collisions]

            # 将合格的位置存入最终位置张量
            agent_positions[:, i, :] = candidate_positions

        # 4. 批量设置所有智能体的位置和速度
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(agent_positions[:, i, :], batch_index=env_index)
            agent.set_vel(torch.zeros(batch_dim, 2, device=self.world.device), batch_index=env_index)

        # 批量重置所有智能体到投篮点的距离
        for i, agent in enumerate(self.world.agents):
            agent_pos = agent_positions[:, i, :]
            dist = torch.linalg.norm(agent_pos - spot_pos, dim=1)
            self.prev_dist_to_spot[batch_range, i] = dist
        self.terminal_rewards.zero_()
        # 重置时也要清零上一帧速度
        self.p_vels[batch_range].zero_()

    def process_action(self, agent: Agent):
        # [MODIFIED] 1. 全向加速度限制
        # 策略输出的 agent.action.u 被视为期望速度

        # 首先，将期望速度限制在智能体的最大速度范围内
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)

        # 忽略过小的动作输入 (死区)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.08] = 0.0

        # 计算在单个时间步内达到期望速度所需的加速度
        requested_a = (agent.action.u - agent.state.vel) / self.world.dt

        # 将所需加速度限制在智能体的最大物理能力 a_max 内
        # clamp_with_norm 确保了加速度限制在各个方向上是均匀的 (圆形限制)
        achievable_a = TorchUtils.clamp_with_norm(requested_a, self.a_max)

        # 根据可行的加速度，计算出本帧智能体能达到的最终速度
        # 这将作为速度控制器的目标
        agent.action.u = agent.state.vel + achievable_a * self.world.dt


        # 调用底层速度控制器以应用计算出的最终速度
        agent.controller.process_force()


    def check_done(self):
        
        # 检查所有 batch 是否有任一终止条件被触发
        dones = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)

        a1 = self.attackers[0]
        
        # -- 条件1: 尝试投篮 (Shot Attempt) --
        dist_to_spot = torch.linalg.norm(a1.state.pos - self.spot_center.state.pos, dim=1)
        in_area = (dist_to_spot <= self.R_spot) & (a1.state.pos[:,1] > 0)
        stopped = torch.linalg.norm(a1.state.vel, dim=1) < self.v_shot_threshold
        if a1.action.u is not None:
            not_accelerating = torch.linalg.norm(a1.action.u, dim=1) < self.a_shot_threshold
        else:
            not_accelerating = torch.tensor([False] * self.world.batch_dim, device=self.world.device)
        shot_attempted = in_area & stopped & not_accelerating & ~dones
        
        if torch.any(shot_attempted):
            final_score = self.max_score * (1 - dist_to_spot[shot_attempted] / self.R_spot)
            self.terminal_rewards[shot_attempted, :self.n_attackers] = final_score.unsqueeze(-1)
            self.terminal_rewards[shot_attempted, self.n_attackers:] = -final_score.unsqueeze(-1)
            dones |= shot_attempted

        # -- 条件2: 时间耗尽 (Time Up) --
        time_up = (self.t_remaining.squeeze(-1) <= 0) & ~dones
        if torch.any(time_up):
            # 进攻方失败，受到惩罚
            self.terminal_rewards[time_up, :self.n_attackers] = -self.max_score
            # 防守方成功，获得奖励
            self.terminal_rewards[time_up, self.n_attackers:] = self.max_score
            dones |= time_up

        # --- 碰撞犯规检查 ---
        for i, agent_i in enumerate(self.world.agents):
            for j, agent_j in enumerate(self.world.agents):
                if i >= j: continue
                are_colliding = self.world.is_overlapping(agent_i, agent_j)
                are_colliding &= ~dones
                if torch.any(are_colliding):
                    v_rel = torch.linalg.norm(agent_i.state.vel - agent_j.state.vel, dim=1)
                    is_foul = v_rel > self.v_foul_threshold
                    foul_collision = are_colliding & is_foul
                    if torch.any(foul_collision):
                        # --- 判断主动方 (使用上一帧的速度) ---
                        pos_rel = agent_j.state.pos - agent_i.state.pos
                        # 从我们自己维护的张量中获取碰撞前的速度
                        agent_i_p_vel = self.p_vels[:, i]
                        vel_rel_on_pos = torch.einsum("bd,bd->b", agent_i_p_vel, pos_rel)
                        i_is_active = vel_rel_on_pos > 0
                        
                        i_fouled_mask = i_is_active & foul_collision
                        j_fouled_mask = ~i_is_active & foul_collision

                        is_friendly_fire = agent_i.is_attacker == agent_j.is_attacker
                        if is_friendly_fire:
                            if agent_i.is_attacker:
                                self.terminal_rewards[foul_collision, :self.n_attackers] = -self.R_foul
                            else:
                                self.terminal_rewards[foul_collision, self.n_attackers:] = -self.R_foul
                        else:
                            if agent_i.is_attacker:
                                self.terminal_rewards[i_fouled_mask, :self.n_attackers] = -self.R_foul
                                self.terminal_rewards[i_fouled_mask, self.n_attackers:] = self.R_foul
                                self.terminal_rewards[j_fouled_mask, self.n_attackers:] = -self.R_foul
                                self.terminal_rewards[j_fouled_mask, :self.n_attackers] = self.R_foul
                            else:
                                self.terminal_rewards[i_fouled_mask, self.n_attackers:] = -self.R_foul
                                self.terminal_rewards[i_fouled_mask, :self.n_attackers] = self.R_foul
                                self.terminal_rewards[j_fouled_mask, :self.n_attackers] = -self.R_foul
                                self.terminal_rewards[j_fouled_mask, self.n_attackers:] = self.R_foul

                        dones |= foul_collision

        if torch.any(dones):
                for i in range(self.n_agents):
                    agent_pos = self.world.agents[i].state.pos
                    dist = torch.linalg.norm(agent_pos - self.spot_center.state.pos, dim=1)
                    self.prev_dist_to_spot[:, i][dones] = dist[dones]
        return dones
    
    def done(self):
        return self.dones if self.dones is not None else torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)

    def reward(self, agent: Agent):
        is_first_agent = agent == self.world.agents[0]
        if is_first_agent:
            self.t_remaining -= self.world.dt
            self.dones = self.check_done()
            all_current_vels = torch.stack([a.state.vel for a in self.world.agents], dim=1)
            self.p_vels.copy_(all_current_vels)
        agent_idx = self.world.agents.index(agent)

        # 稠密奖励
        dense_reward = torch.zeros(self.world.batch_dim, device=self.world.device)

        # 出界惩罚
        pos = agent.state.pos
        is_oob_x = torch.abs(pos[:, 0]) > (0.99 * self.W / 2)
        is_oob_y = torch.abs(pos[:, 1]) > (0.99 * self.L / 2)
        is_oob = is_oob_x | is_oob_y
        dense_reward[is_oob] += self.oob_penalty

        a1 = self.attackers[0]
        a2 = self.attackers[1]
        
        if agent == a1:
            # 接近投篮点奖励
            current_dist = torch.linalg.norm(pos - self.spot_center.state.pos, dim=1)
            dense_reward += (self.prev_dist_to_spot[:, agent_idx] - current_dist) * self.k_a1_approach
            self.prev_dist_to_spot[:, agent_idx] = current_dist

            if self.k_a1_in_spot_reward > 0:
                is_in_spot = (current_dist <= self.R_spot) & (agent.state.pos[:,1] > 0)
                reward_magnitude = (1 - current_dist / self.R_spot)
                spot_reward = self.k_a1_in_spot_reward * reward_magnitude * is_in_spot.float()
                dense_reward += spot_reward
            
            # 被封堵惩罚
            total_block_factor = torch.zeros_like(dense_reward)
            for defender in self.defenders:
                p_d = defender.state.pos
                ap = self.basket.state.pos - pos
                ad = p_d - pos
                proj_len = torch.einsum("bd,bd->b", ad, ap) / (torch.linalg.norm(ap, dim=1).pow(2) + 1e-6)
                proj_len = torch.clamp(proj_len, 0, 1)
                closest_point_on_line = pos + proj_len.unsqueeze(-1) * ap
                dist_perp = torch.linalg.norm(p_d - closest_point_on_line, dim=1)
                # 只考虑近距离的封堵
                dist_to_def = torch.linalg.norm(pos - p_d, dim=1)
                close_enough_mask = dist_to_def < self.def_proximity_threshold
                block_factor = torch.exp(-dist_perp.pow(2) / (2 * self.block_sigma ** 2))
                total_block_factor += block_factor * close_enough_mask
            dense_reward += total_block_factor * self.k_a1_blocked_penalty

        elif agent == a2:
            # --- A2 (掩护) 奖励 ---
            total_screen_reward = torch.zeros_like(dense_reward)
            p_a1 = a1.state.pos
            p_a2 = pos

            for defender in self.defenders:
                p_d = defender.state.pos
                da1 = p_a1 - p_d
                da2 = p_a2 - p_d
                da1_norm_sq = torch.sum(da1**2, dim=1) + 1e-6
                proj_len_ratio = torch.einsum("bd,bd->b", da2, da1) / da1_norm_sq
                is_between = (proj_len_ratio > 0.05) & (proj_len_ratio < 0.95)
                dist_perp_sq = torch.sum((da2 - proj_len_ratio.unsqueeze(-1) * da1)**2, dim=1)
                screen_factor = torch.exp(-dist_perp_sq / (2 * self.screen_sigma ** 2))
                dist_a2_d_sq = torch.sum((p_a2 - p_d)**2, dim=1)
                dist_a2_a1_sq = torch.sum((p_a2 - p_a1)**2, dim=1)
                is_closer_to_defender = dist_a2_d_sq < dist_a2_a1_sq
                good_screen_mask = is_between & is_closer_to_defender
                total_screen_reward += screen_factor * good_screen_mask.float()
            
            dense_reward += total_screen_reward * self.k_a2_screen

            # --- A2 (扎堆) 惩罚 ---
            dist_a2_a1 = torch.linalg.norm(p_a2 - p_a1, dim=1)
            crowding_penalty_factor = torch.exp(-dist_a2_a1.pow(2) / (2 * self.screen_sigma ** 2))
            dense_reward -= self.k_a2_crowding_penalty * crowding_penalty_factor

        elif not agent.is_attacker:
            p_basket = self.basket.state.pos
            p_d = pos
            in_defensive_half = p_d[:, 1] > 0
            overextend_penalty_tensor = torch.where(in_defensive_half, torch.zeros_like(dense_reward), torch.full_like(dense_reward, self.overextend_penalty))
            dense_reward += overextend_penalty_tensor

            if torch.any(in_defensive_half):
                current_dist_to_spot = torch.linalg.norm(p_d - self.spot_center.state.pos, dim=1)
                spot_approach_reward = (self.prev_dist_to_spot[:, agent_idx] - current_dist_to_spot) * self.k_def_spot_approach
                self.prev_dist_to_spot[:, agent_idx] = current_dist_to_spot
                dense_reward += spot_approach_reward

                p_a1 = a1.state.pos
                dist_to_a1 = torch.linalg.norm(p_d - p_a1, dim=1)
                proximity_factor = torch.clamp(1.0 - (dist_to_a1 / self.def_proximity_threshold), min=0.0)
                ap = p_basket - p_a1
                ad = p_d - p_a1
                proj_len = torch.einsum("bd,bd->b", ad, ap) / (torch.linalg.norm(ap, dim=1).pow(2) + 1e-6)
                proj_len = torch.clamp(proj_len, 0, 1)
                closest_point_on_line = p_a1 + proj_len.unsqueeze(-1) * ap
                dist_perp = torch.linalg.norm(p_d - closest_point_on_line, dim=1)
                block_factor = torch.exp(-dist_perp.pow(2) / (2 * self.block_sigma ** 2))
                block_reward = self.k_def_block * block_factor * proximity_factor
                dense_reward += torch.where(in_defensive_half, block_reward, torch.zeros_like(block_reward))

        # [MODIFIED] 碰撞惩罚
        for other in self.world.agents:
            if agent == other: continue
            
            # self.world.collides 应返回一个形状为 (batch_dim,) 的布尔张量
            is_colliding = self.world.is_overlapping(agent, other)
            
            # 为避免if语句带来的问题，我们直接进行张量运算。
            # 先计算出所有环境下可能发生的惩罚值。
            pos_rel = other.state.pos - agent.state.pos
            v_rel = agent.state.vel - other.state.vel
            v_rel_norm = torch.linalg.norm(v_rel, dim=1)
            
            vel_proj = torch.einsum("bd,bd->b", agent.state.vel, pos_rel)
            is_active = vel_proj > 0

            active_penalty = -self.k_coll_active * v_rel_norm
            passive_penalty = -self.k_coll_passive * v_rel_norm
            
            penalty = torch.where(is_active, active_penalty, passive_penalty)
            
            # 使用 is_colliding 张量作为掩码(mask)来施加惩罚。
            # .float() 将布尔张量 (True/False) 转换为 (1.0/0.0)。
            # 在没有碰撞的环境中，惩罚为 penalty * 0.0 = 0。
            dense_reward += penalty * is_colliding.float()
        
        rew = dense_reward + self.terminal_rewards[:, agent_idx]
        return rew

    def observation(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        
        # 1-4: 自身和队友信息
        if agent.is_attacker:
            teammate = self.attackers[1 - agent_idx]
        else:
            teammate_local_idx = agent_idx - self.n_attackers
            teammate = self.defenders[1 - teammate_local_idx]

        # 5-8: 对手信息
        if agent.is_attacker:
            opp1, opp2 = self.defenders[0], self.defenders[1]
        else: # is defender
            opp1, opp2 = self.attackers[0], self.attackers[1] # A1, A2

        # 9: 关键目标信息
        if agent.is_attacker:
            key_info_rel = self.spot_center.state.pos - agent.state.pos
        else: # is defender
            a1 = self.attackers[0]
            key_info_rel = self.basket.state.pos - a1.state.pos
            
        obs = torch.cat([
            agent.state.pos,
            agent.state.vel,
            teammate.state.pos - agent.state.pos,
            teammate.state.vel - agent.state.vel,
            opp1.state.pos - agent.state.pos,
            opp1.state.vel - agent.state.vel,
            opp2.state.pos - agent.state.pos,
            opp2.state.vel - agent.state.vel,
            key_info_rel,
            self.t_remaining / self.t_limit,
        ], dim=-1)

        return obs


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )