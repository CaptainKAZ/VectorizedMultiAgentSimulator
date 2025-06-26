import torch
from vmas import render_interactively
from vmas.simulator.core import World, Agent, Landmark, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

class Scenario(BaseScenario):
    """
    "飞身上篮"简化版2v2投篮强化学习环境 (已优化和修复版本)
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
        self.a_max = kwargs.get("a_max", 3.0)  # 最大加速度
        self.v_max = kwargs.get("v_max", 5.0)  # 最大速度

        # 终止条件阈值
        self.v_shot_threshold = kwargs.get("v_shot_threshold", 0.2)  # 投篮速度阈值
        self.a_shot_threshold = kwargs.get("a_shot_threshold", 1.0)  # 投篮加速度阈值
        self.v_foul_threshold = kwargs.get("v_foul_threshold", 2.0)  # 碰撞犯规速度阈值

        # 奖励系数
        self.max_score = kwargs.get("max_score", 120.0)
        self.R_foul = kwargs.get("R_foul", 160.0)
        self.k_a1_approach = kwargs.get("k_a1_approach", 2.0)
        self.k_a1_in_spot_reward = kwargs.get("k_a1_in_spot_reward", 2.0)
        self.k_def_block = kwargs.get("k_def_block", 5.0)
        self.block_sigma = kwargs.get("block_sigma", 2.5)
        self.k_coll_active = kwargs.get("k_coll_active", 5.0)
        self.k_coll_passive = kwargs.get("k_coll_passive", 0.1)
        self.oob_penalty = kwargs.get("oob_penalty", -10.0)  # 出界惩罚值
        self.def_proximity_threshold = kwargs.get("def_proximity_threshold", 0.9)  # 防守奖励生效的最大距离
        self.overextend_penalty = kwargs.get("overextend_penalty", -6.0)  # 防守方越过半场的惩罚
        self.k_def_spot_approach = kwargs.get("k_def_spot_approach", 0.01)  # 防守方接近隐藏投篮点的奖励系数
        self.k_a1_blocked_penalty = kwargs.get("k_a1_blocked_penalty", -1.0)  # A1被封堵的惩罚系数
        self.k_velocity_penalty = kwargs.get("k_velocity_penalty", 0.2)
        self.k_a2_screen = kwargs.get("k_a2_screen", 3.0)  # A2执行掩护的奖励系数
        self.k_a2_crowding_penalty = kwargs.get("k_a2_crowding_penalty", 8)  # A2因过于靠近A1而受到的扎堆惩罚系数
        self.screen_sigma = kwargs.get("screen_sigma", 0.9)
        self.gaussian_sigma = kwargs.get("gaussian_sigma", self.R_spot * 0.8) # 高斯引导奖励的sigma
        self.gaussian_scale = kwargs.get("gaussian_scale", self.k_a1_approach * 5.0) # 高斯引导奖励的缩放系数
        self.low_velocity_threshold = kwargs.get("low_velocity_threshold", 1.0) # 低速推挤判定阈值
        self.k_push_penalty = kwargs.get("k_push_penalty", 5.0) # 主动推挤惩罚系数
        self.k_u_penalty_general = kwargs.get("k_u_penalty_general", 0.01) # 全局控制量惩罚系数
        self.k_u_penalty_a1_in_spot = kwargs.get("k_u_penalty_a1_in_spot", 0.005) # A1在投篮点的额外控制量惩罚
        self.proximity_threshold = kwargs.get("proximity_threshold", self.agent_radius * 2.2) # 近距离惩罚阈值 (直径的1.1倍)
        self.proximity_penalty_margin = kwargs.get("proximity_penalty_margin", 0.15)      # 近距离惩罚曲线的软度
        self.k_proximity_penalty = kwargs.get("k_proximity_penalty", 5)      # 近距离惩罚系数
        self.k_overextend_penalty = kwargs.get("k_overextend_penalty", 10.0)     # 增强的越界惩罚系数
        self.k_pressure = kwargs.get("k_pressure", 0.5)                   # 压迫驱离奖励系数
        self.k_positioning = kwargs.get("k_positioning", 0.2)                # 有效站位奖励系数
        self.k_contest = kwargs.get("k_contest", 20.0)                     # 关键干扰奖励系数
        self.k_a1_restriction = kwargs.get("k_a1_restriction", 0.5)         # 限制A1移动空间奖励系数
        self.k_spot_control = kwargs.get("k_spot_control", 1.0)             # 投篮区域控制奖励系数


        # ----------------- 环境构建 (World Setup) -----------------
        self.max_steps = int(self.t_limit / self.dt)
        self.n_agents = 4
        self.n_attackers = 2
        self.n_defenders = 2

        world = World(batch_dim, device, dt=self.dt, substeps=4,
                      x_semidim=self.W / 2, y_semidim=self.L / 2)

        # 创建智能体
        for i in range(self.n_agents):
            is_attacker = i < self.n_attackers
            team_name = "attacker" if is_attacker else "defender"
            agent_id = i + 1 if is_attacker else i - self.n_attackers + 1
            agent = Agent(
                name=f"{team_name}_{agent_id}",
                collide=True,
                movable=True,
                rotatable=False,
                u_range=self.v_max,
                drag=0.01,
                shape=Sphere(radius=self.agent_radius),
                dynamics=Holonomic(),
                render_action=True,
                color=Color.RED if is_attacker and agent_id == 1 else Color.BLUE if not is_attacker else Color.PINK
            )
            # 添加自定义属性以区分队伍
            agent.is_attacker = is_attacker
            agent.controller = VelocityController(agent, world, [6,0,0.01], "parallel")

            world.add_agent(agent)

        # 将智能体列表分开存储，方便后续使用
        self.attackers = world.agents[:self.n_attackers]
        self.defenders = world.agents[self.n_attackers:]
        self.a1 = self.attackers[0]
        self.a2 = self.attackers[1]

        # 创建地标
        self.basket = Landmark(name="basket", collide=False, shape=Sphere(radius=0.1), color=Color.ORANGE)
        self.spot_center = Landmark(name="spot_center", collide=False, shape=Sphere(radius=0.05), color=Color.GREEN)
        self.shooting_area_vis = Landmark(name="shooting_area_vis", collide=False, shape=Sphere(radius=self.R_spot), color=Color.LIGHT_GREEN)
        center_line = Landmark(name="center_line", collide=False, shape=Line(length=self.W), color=Color.GRAY)
        world.add_landmark(center_line)
        world.add_landmark(self.basket)
        world.add_landmark(self.spot_center)
        world.add_landmark(self.shooting_area_vis)

        # 内部状态变量
        self.t_remaining = torch.zeros(batch_dim, 1, device=device)
        self.prev_dist_to_spot = torch.zeros(batch_dim, self.n_agents, device=device)
        self.terminal_rewards = torch.zeros(batch_dim, self.n_agents, device=device)
        self.dones = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.p_vels = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.raw_actions = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.prev_dist_a1_to_basket = torch.zeros(batch_dim, device=device)
        self.contest_reward_this_step = torch.zeros(batch_dim, self.n_defenders, device=device)

        return world

    def reset_world_at(self, env_index: int | None = None):
        """
        使用矢量化拒绝采样重置智能体位置，以避免初始重叠。
        """
        # 如果是部分重置，获取对应的批次范围
        if env_index is None:
            batch_range = slice(None)
            batch_dim = self.world.batch_dim
        else:
            batch_range = env_index
            batch_dim = 1 if isinstance(env_index, int) else len(env_index)
        
        # 重置时间和奖励
        self.t_remaining[batch_range] = self.t_limit
        self.terminal_rewards[batch_range] = 0.0
        self.p_vels[batch_range] = 0.0
        self.contest_reward_this_step[batch_range] = 0.0

        # 设置篮筐和投篮点位置
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
        agent_positions = torch.zeros(batch_dim, self.n_agents, 2, device=self.world.device)
        needs_resampling = torch.ones(batch_dim, device=self.world.device, dtype=torch.bool)

        # 在一个循环中处理所有需要重采样的环境，直到没有环境存在碰撞
        for _ in range(10): # 设置最大重试次数以避免死循环
            if not torch.any(needs_resampling):
                break

            num_resample = needs_resampling.sum()

            # 为需要重采样的环境生成新位置
            # 进攻方位置
            att_x = (torch.rand(num_resample, self.n_attackers, 1, device=self.world.device) - 0.5) * self.W
            att_y = -self.L / 2 + torch.rand(num_resample, self.n_attackers, 1, device=self.world.device) * self.spawn_area_depth
            # 防守方位置
            def_x = (torch.rand(num_resample, self.n_defenders, 1, device=self.world.device) - 0.5) * self.W
            def_y = self.L / 2 - torch.rand(num_resample, self.n_defenders, 1, device=self.world.device) * self.spawn_area_depth

            # 合并成一个张量
            new_pos = torch.cat([
                torch.cat([att_x, att_y], dim=-1),
                torch.cat([def_x, def_y], dim=-1)
            ], dim=1)
            
            # 将新生成的位置更新到总位置张量中
            agent_positions[needs_resampling] = new_pos

            # 矢量化计算距离矩阵
            pos_diff = agent_positions.unsqueeze(2) - agent_positions.unsqueeze(1)
            dist_matrix = torch.linalg.norm(pos_diff, dim=-1)
            
            # 为了避免检查对角线（自己和自己的距离），填充一个大于min_dist的值
            dist_matrix.diagonal(dim1=-2, dim2=-1).fill_(min_dist + 1)
            
            # 检查每个环境中是否存在任何小于最小距离的碰撞
            collisions = torch.any(dist_matrix < min_dist, dim=(1, 2))
            needs_resampling = collisions
        
        # 批量设置所有智能体的位置和速度
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(agent_positions[:, i, :], batch_index=env_index)
            agent.set_vel(torch.zeros(batch_dim, 2, device=self.world.device), batch_index=env_index)

        # 批量重置所有智能体到投篮点的距离
        dists = torch.linalg.norm(agent_positions - spot_pos.unsqueeze(1), dim=-1)
        self.prev_dist_to_spot[batch_range] = dists

        # 初始化A1到篮筐的距离
        dist_a1_basket = torch.linalg.norm(agent_positions[:, 0, :] - basket_pos, dim=-1)
        self.prev_dist_a1_to_basket[batch_range] = dist_a1_basket

    def process_action(self, agent: Agent):
        # 保存模型输出的原始期望速度
        agent_idx = self.world.agents.index(agent)
        self.raw_actions[:, agent_idx, :] = agent.action.u.clone()

        # 忽略过小的动作输入 (死区)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.1] = 0.0

        # 将期望速度限制在最大速度范围内 (圆形限制)
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)
        
        # 计算达到期望速度所需的加速度
        requested_a = (agent.action.u - agent.state.vel) / self.world.dt

        # 将所需加速度限制在最大物理能力范围内 (圆形限制)
        achievable_a = TorchUtils.clamp_with_norm(requested_a, self.a_max)

        # 根据可行的加速度，计算出本帧智能体能达到的最终速度
        agent.action.u = agent.state.vel + achievable_a * self.world.dt
        # 调用底层速度控制器以应用计算出的最终速度
        agent.controller.process_force()

    def pre_step(self):
        """
        在每个物理步长开始前执行。
        用于集中计算所有智能体间的交互信息，避免重复计算。
        """
        # 更新剩余时间
        self.t_remaining -= self.world.dt

        # --- 矢量化计算所有智能体间的交互信息 ---
        # 1. 整合所有智能体的位置和速度到一个大张量中
        self.all_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1) # (B, N, 2)
        self.all_vel = torch.stack([a.state.vel for a in self.world.agents], dim=1) # (B, N, 2)
        
        # 2. 使用广播计算成对(pairwise)的相对位置和距离
        #    (B, N, 1, 2) - (B, 1, N, 2) -> (B, N, N, 2)
        self.pos_diffs = self.all_pos.unsqueeze(2) - self.all_pos.unsqueeze(1)
        self.dist_matrix = torch.linalg.norm(self.pos_diffs, dim=-1) # (B, N, N)

        # 3. 计算碰撞矩阵
        self.collision_matrix = self.dist_matrix < (self.agent_radius * 2)
        # 忽略对角线（自己和自己）
        self.collision_matrix.diagonal(dim1=-2, dim2=-1).fill_(False)

        # 4. 计算成对相对速度
        self.vel_diffs = self.all_vel.unsqueeze(2) - self.all_vel.unsqueeze(1) # (B, N, N, 2)
        self.vel_diffs_norm = torch.linalg.norm(self.vel_diffs, dim=-1) # (B, N, N)

        # 5. 检查并设置 done 标志
        self.dones = self.check_done()

    def post_step(self):
        """
        在物理步长结束后执行，用于记录状态以备下一帧使用。
        """
        # 记录当前帧的速度，作为下一帧的“上一帧速度”
        self.p_vels.copy_(self.all_vel)

        # 更新A1到篮筐的距离
        self.prev_dist_a1_to_basket.copy_(torch.linalg.norm(self.a1.state.pos - self.basket.state.pos, dim=-1))

        # 更新A1到投篮点的距离
        self.prev_dist_a1_to_spot.copy_(torch.linalg.norm(self.a1.state.pos - self.spot_center.state.pos, dim=-1))

        # 如果任何环境已结束，需要更新 prev_dist_to_spot 的最终值
        if torch.any(self.dones):
            dists = torch.linalg.norm(self.all_pos - self.spot_center.state.pos.unsqueeze(1), dim=-1)
            self.prev_dist_to_spot[self.dones] = dists[self.dones]
    
    def check_done(self):
        # 初始化一个全为False的done张量
        dones = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        
        # -- 条件1: 尝试投篮 (Shot Attempt) --
        dist_to_spot = torch.linalg.norm(self.a1.state.pos - self.spot_center.state.pos, dim=1)
        in_area = (dist_to_spot <= self.R_spot) & (self.a1.state.pos[:, 1] > 0)
        stopped = torch.linalg.norm(self.a1.state.vel, dim=1) < self.v_shot_threshold
        # 使用原始期望速度判断是否“意图”停止
        not_accelerating = torch.linalg.norm(self.raw_actions[:, 0, :], dim=1) < self.a_shot_threshold
        
        # 仅在未结束的环境中检查此条件
        shot_attempted = in_area & stopped & not_accelerating & ~dones
        if torch.any(shot_attempted):
            # 1. 计算基础分
            final_score = self.max_score * (1 - dist_to_spot[shot_attempted] / self.R_spot)

            # 2. 【新增】计算视野遮挡系数
            shot_b_idx = shot_attempted.nonzero(as_tuple=True)[0]
            a1_pos = self.a1.state.pos[shot_b_idx]
            basket_pos = self.basket.state.pos[shot_b_idx]
            
            # 获取所有其他智能体（A2, D1, D2）的位置
            other_agent_indices = [i for i in range(self.n_agents) if i != 0]
            blocker_pos = self.all_pos[shot_b_idx][:, other_agent_indices, :]
            
            # 从A1到篮筐的向量
            shot_vector = basket_pos - a1_pos
            shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
            
            # 从A1到潜在遮挡者的向量
            blocker_vector = blocker_pos - a1_pos.unsqueeze(1)
            
            # ############# 代码修复处 #############
            # 将遮挡者向量投影到投篮向量上，判断其相对位置
            # 使用显式广播和求和代替einsum，避免维度匹配错误
            # blocker_vector: (n_shots, 3, 2), shot_vector.unsqueeze(1): (n_shots, 1, 2)
            # dot_product 结果: (n_shots, 3)
            dot_product = torch.sum(blocker_vector * shot_vector.unsqueeze(1), dim=-1)
            # shot_vector_norm_sq 形状为 (n_shots, 1)，可直接用于广播除法
            proj_len_ratio = dot_product / shot_vector_norm_sq
            # ######################################
            
            # 条件A: 遮挡者必须在A1和篮筐之间
            is_between = (proj_len_ratio > 0) & (proj_len_ratio < 1)
            
            # 计算遮挡者到投篮路线的垂直距离
            projection = proj_len_ratio.unsqueeze(-1) * shot_vector.unsqueeze(1)
            dist_perp_sq = torch.sum((blocker_vector - projection)**2, dim=-1)
            
            # 条件B: 遮挡者的身体必须与投篮路线相交
            block_sigma = self.agent_radius  # 使用智能体半径作为遮挡判定的标准差
            intersects_line = dist_perp_sq < block_sigma**2
            
            # 条件C: 遮挡者必须离A1足够近，形成有效干扰
            dist_to_a1_sq = torch.sum(blocker_vector**2, dim=-1)
            is_close = dist_to_a1_sq < (3 * self.agent_radius)**2 # 定义“近”为3倍半径范围内
            
            # 结合所有条件
            is_blocker = is_between & intersects_line & is_close
            
            # 基于垂直距离计算每个有效遮挡者的贡献值
            block_contribution = torch.exp(-dist_perp_sq / (2 * block_sigma**2))
            
            # 将所有有效遮挡者的贡献值相加，得到总遮挡系数
            total_block_factor = (block_contribution * is_blocker.float()).sum(dim=1)
            
            # 将系数限制在[0, 1]范围内
            total_block_factor = torch.clamp(total_block_factor, 0, 1)

            # 3. 根据遮挡系数修正最终得分
            final_score_modified = final_score * (1 - total_block_factor)
            
            # 4. 分配修正后的奖励
            self.terminal_rewards[shot_b_idx, :self.n_attackers] = final_score_modified.unsqueeze(-1)
            self.terminal_rewards[shot_b_idx, self.n_attackers:] = -final_score_modified.unsqueeze(-1)

            # 5. 分配关键干扰奖励给防守方
            for i, d_idx in enumerate(self.defenders):
                self.contest_reward_this_step[shot_b_idx, i] = self.k_contest * total_block_factor
            
            dones |= shot_attempted

        # -- 条件2: 时间耗尽 (Time Up) --
        time_up = (self.t_remaining.squeeze(-1) <= 0) & ~dones
        if torch.any(time_up):
            self.terminal_rewards[time_up, :self.n_attackers] = -self.max_score
            self.terminal_rewards[time_up, self.n_attackers:] = self.max_score
            dones |= time_up

        # -- 条件3: 碰撞犯规 (Foul) --
        # 犯规条件: 正在碰撞 & 相对速度超过阈值 & 环境尚未结束
        is_foul = self.collision_matrix & (self.vel_diffs_norm > self.v_foul_threshold) & ~dones.view(-1, 1, 1)

        # 获取所有犯规碰撞的索引 (只考虑上三角矩阵避免重复计算)
        foul_indices = torch.triu(is_foul, diagonal=1).nonzero(as_tuple=True)
        if foul_indices[0].numel() > 0:
            b_idx, i_idx, j_idx = foul_indices

            # 确定每次犯规中的主动方
            agent_i_p_vel = self.p_vels[b_idx, i_idx]
            pos_rel = self.all_pos[b_idx, j_idx] - self.all_pos[b_idx, i_idx]
            vel_rel_on_pos = torch.einsum("bd,bd->b", agent_i_p_vel, pos_rel)
            i_is_active = vel_rel_on_pos > 0

            # 确定犯规中每个智能体的队伍
            i_is_attacker = i_idx < self.n_attackers
            j_is_attacker = j_idx < self.n_attackers

            # 确定主动方和被动方的队伍
            active_fouler_is_attacker = torch.where(i_is_active, i_is_attacker, j_is_attacker)
            passive_agent_is_attacker = torch.where(i_is_active, j_is_attacker, i_is_attacker)
            
            # 判断是否是友军犯规
            is_friendly_fire = (active_fouler_is_attacker == passive_agent_is_attacker)
            
            # --- 情况1: 对手犯规 ---
            opponent_foul_mask = ~is_friendly_fire
            if torch.any(opponent_foul_mask):
                # 筛选出对手犯规的相关信息
                opp_b = b_idx[opponent_foul_mask]
                opp_active_is_attacker = active_fouler_is_attacker[opponent_foul_mask]

                # 子情况1a: 进攻方是主动犯规方 (惩罚进攻队, 奖励防守队)
                attacker_foul_mask = opp_active_is_attacker
                b_att_foul = opp_b[attacker_foul_mask]
                if b_att_foul.numel() > 0:
                    self.terminal_rewards[b_att_foul, :self.n_attackers] = -self.R_foul
                    self.terminal_rewards[b_att_foul, self.n_attackers:] = self.R_foul

                # 子情况1b: 防守方是主动犯规方 (惩罚防守队, 奖励进攻队)
                defender_foul_mask = ~opp_active_is_attacker
                b_def_foul = opp_b[defender_foul_mask]
                if b_def_foul.numel() > 0:
                    self.terminal_rewards[b_def_foul, self.n_attackers:] = -self.R_foul
                    self.terminal_rewards[b_def_foul, :self.n_attackers] = self.R_foul
            
            # --- 情况2: 友军犯规 ---
            friendly_foul_mask = is_friendly_fire
            if torch.any(friendly_foul_mask):
                # 筛选出友军犯规的相关信息
                ff_b = b_idx[friendly_foul_mask]
                ff_active_is_attacker = active_fouler_is_attacker[friendly_foul_mask]

                # 子情况2a: 进攻队内部犯规 (惩罚进攻队)
                ff_attacker_mask = ff_active_is_attacker
                b_ff_att_foul = ff_b[ff_attacker_mask]
                if b_ff_att_foul.numel() > 0:
                    self.terminal_rewards[b_ff_att_foul, :self.n_attackers] = -self.R_foul
                
                # 子情况2b: 防守队内部犯规 (惩罚防守队)
                ff_defender_mask = ~ff_active_is_attacker
                b_ff_def_foul = ff_b[ff_defender_mask]
                if b_ff_def_foul.numel() > 0:
                    self.terminal_rewards[b_ff_def_foul, self.n_attackers:] = -self.R_foul

            # 在所有发生犯规的环境中设置done为True
            dones[b_idx] = True

        return dones
    
    def done(self):
        return self.dones

    def reward(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        dense_reward = torch.zeros(self.world.batch_dim, device=self.world.device)

        # --- 通用奖励/惩罚 ---
        # 出界惩罚
        pos = agent.state.pos
        is_oob = (torch.abs(pos[:, 0]) > (0.99 * self.W / 2)) | \
                 (torch.abs(pos[:, 1]) > (0.99 * self.L / 2))
        dense_reward[is_oob] += self.oob_penalty * (torch.linalg.norm(agent.state.vel[is_oob], dim=1)+1)
        agent.state.vel[is_oob] = 0.0

        # 使用原始期望速度计算通用控制量惩罚
        raw_u_norm = torch.linalg.vector_norm(self.raw_actions[:, agent_idx, :], dim=1)
        dense_reward -= self.k_u_penalty_general * raw_u_norm

        # 新增：基于物理引擎的平滑近距离惩罚，避免扎堆
        agent_dists = self.dist_matrix[:, agent_idx, :]
        # 排除自己与自己的距离，避免不必要的计算
        agent_dists[:, agent_idx] = self.proximity_threshold

        # 找出所有小于阈值的距离，但不包括已经碰撞的
        collision_dist = self.agent_radius * 2
        is_too_close = (agent_dists < self.proximity_threshold) & (agent_dists > collision_dist)
        if torch.any(is_too_close):
            # 使用 logaddexp (Softplus) 函数计算平滑的侵入深度
            k = self.proximity_penalty_margin
            penetration = torch.logaddexp(
                torch.tensor(0.0, device=self.world.device),
                (self.proximity_threshold - agent_dists) / k,
            ) * k

            # 计算最终惩罚
            proximity_penalty = -self.k_proximity_penalty * penetration

            # 只对过于接近的智能体施加惩罚
            total_proximity_penalty = (proximity_penalty * is_too_close.float()).sum(dim=1)
            dense_reward += total_proximity_penalty

        # 碰撞惩罚 (使用 pre_step 中计算好的矩阵)
        agent_collisions = self.collision_matrix[:, agent_idx, :]
        if torch.any(agent_collisions):
            # 1. 高速碰撞惩罚 (基于相对速度)
            # 判断主动/被动碰撞
            pos_rel = self.all_pos - pos.unsqueeze(1) # (B, N, 2)
            vel_proj = torch.einsum("bd,bnd->bn", agent.state.vel, pos_rel)
            is_active = vel_proj > 0

            # 计算惩罚值
            active_penalty = -self.k_coll_active * self.vel_diffs_norm[:, agent_idx, :]
            passive_penalty = -self.k_coll_passive * self.vel_diffs_norm[:, agent_idx, :]
            
            # 根据主动/被动选择惩罚
            penalty = torch.where(is_active, active_penalty, passive_penalty)
            
            # 只在碰撞时施加惩罚
            total_collision_penalty = (penalty * agent_collisions.float()).sum(dim=1)
            dense_reward += total_collision_penalty

            # 2. 低速推挤惩罚 (基于原始期望速度)
            is_low_speed_collision = agent_collisions & (self.vel_diffs_norm[:, agent_idx, :] < self.low_velocity_threshold)
            if torch.any(is_low_speed_collision):
                # 获取智能体的原始期望速度
                raw_action_force = self.raw_actions[:, agent_idx, :]

                # 计算控制力在相对位置向量上的投影
                pos_diffs_agent_centric = self.pos_diffs[:, agent_idx, :, :]
                raw_action_force_expanded = raw_action_force.unsqueeze(1).expand(-1, self.n_agents, -1)

                pos_diffs_norm = torch.linalg.norm(pos_diffs_agent_centric, dim=-1, keepdim=True) + 1e-6
                proj_vector = (pos_diffs_agent_centric / pos_diffs_norm)
                
                push_force_magnitude = torch.einsum('bnd,bnd->bn', raw_action_force_expanded, proj_vector)

                # 只惩罚正向推力 (即推向对方)
                push_penalty = -self.k_push_penalty * torch.clamp(push_force_magnitude, min=0.0)
                
                # 只在低速碰撞时施加惩罚
                total_push_penalty = (push_penalty * is_low_speed_collision.float()).sum(dim=1)
                dense_reward += total_push_penalty


        # --- 分角色奖励/惩罚 ---
        if agent == self.a1:
            # A1 (持球进攻者)
            current_dist = torch.linalg.norm(pos - self.spot_center.state.pos, dim=1) # (B,)
            
            # 使用高斯函数引导A1到达目标点
            gaussian_reward = self.gaussian_scale * torch.exp(- (current_dist**2) / (2 * self.gaussian_sigma**2))
            dense_reward += gaussian_reward


            # 2. 在投篮区域内的奖励 (保持不变)
            is_in_spot = (current_dist <= self.R_spot) & (pos[:, 1] > 0)
            spot_reward = self.k_a1_in_spot_reward * (1 - current_dist / self.R_spot) * is_in_spot.float()
            dense_reward += spot_reward
            
            # 3. 【新增】在投篮区内时，惩罚高速移动，鼓励刹车
            velocity_norm = torch.linalg.norm(agent.state.vel, dim=1)
            velocity_penalty = -self.k_velocity_penalty * velocity_norm * is_in_spot.float()
            dense_reward += velocity_penalty
            # ######################################
            
            # 新增：在投篮区内时，施加更大的控制量惩罚以鼓励静止 (基于原始期望速度)
            dense_reward -= self.k_u_penalty_a1_in_spot * raw_u_norm * is_in_spot.float()

            # 4. 被封堵惩罚 (保持不变)
            def_pos = torch.stack([d.state.pos for d in self.defenders], dim=1) # (B, D, 2)
            ap = self.basket.state.pos.unsqueeze(1) - pos.unsqueeze(1)  # (B, 1, 2)
            ad = def_pos - pos.unsqueeze(1) # (B, D, 2)
            
            proj_len = torch.einsum("bnd,bmd->bnm", ad, ap).squeeze(-1) / (torch.linalg.norm(ap, dim=-1).pow(2) + 1e-6) # (B, D)
            proj_len = torch.clamp(proj_len, 0, 1)
            closest_point_on_line = pos.unsqueeze(1) + proj_len.unsqueeze(-1) * ap
            dist_perp = torch.linalg.norm(def_pos - closest_point_on_line, dim=-1) # (B, D)
            
            dist_to_def = torch.linalg.norm(pos.unsqueeze(1) - def_pos, dim=-1) # (B, D)
            close_enough_mask = dist_to_def < self.def_proximity_threshold
            block_factor = torch.exp(-dist_perp.pow(2) / (2 * self.block_sigma ** 2))
            total_block_factor = (block_factor * close_enough_mask.float()).sum(dim=1) # (B,)
            dense_reward += total_block_factor * self.k_a1_blocked_penalty

        elif agent == self.a2:
            # A2 (无球/掩护者)
            p_a1 = self.a1.state.pos
            p_a2 = pos
            
            # 1. 掩护奖励 (矢量化计算)
            def_pos = torch.stack([d.state.pos for d in self.defenders], dim=1) # (B, D, 2)
            da1 = p_a1.unsqueeze(1) - def_pos # (B, D, 2)
            da2 = p_a2.unsqueeze(1) - def_pos # (B, D, 2)

            da1_norm_sq = torch.sum(da1**2, dim=-1) + 1e-6 # (B, D)
            proj_len_ratio = torch.einsum("bnd,bnd->bn", da2, da1) / da1_norm_sq # (B, D)
            is_between = (proj_len_ratio > 0.05) & (proj_len_ratio < 0.95)
            
            dist_perp_sq = torch.sum((da2 - proj_len_ratio.unsqueeze(-1) * da1)**2, dim=-1) # (B, D)
            screen_factor = torch.exp(-dist_perp_sq / (2 * self.screen_sigma ** 2))
            
            dist_a2_d_sq = torch.sum((p_a2.unsqueeze(1) - def_pos)**2, dim=-1)
            dist_a2_a1_sq = torch.sum((p_a2 - p_a1)**2, dim=-1).unsqueeze(-1)
            is_closer_to_defender = dist_a2_d_sq < dist_a2_a1_sq
            
            good_screen_mask = is_between & is_closer_to_defender
            total_screen_reward = (screen_factor * good_screen_mask.float()).sum(dim=1) # (B,)
            dense_reward += total_screen_reward * self.k_a2_screen

            # 2. 扎堆惩罚
            dist_a2_a1 = torch.linalg.norm(p_a2 - p_a1, dim=1)
            crowding_penalty_factor = torch.exp(-dist_a2_a1.pow(2) / (2 * self.screen_sigma ** 2))
            dense_reward -= self.k_a2_crowding_penalty * crowding_penalty_factor

        else: # 防守方
            p_d = pos
            agent_d_idx = self.defenders.index(agent)
            in_defensive_half = p_d[:, 1] > 0
            
            # 1. 越过半场惩罚 (与越界深度成正比)
            # 只有当Y坐标小于0时才计算越界深度
            overextend_depth = torch.where(p_d[:, 1] < 0, torch.abs(p_d[:, 1]), torch.zeros_like(p_d[:, 1]))
            dense_reward -= self.k_overextend_penalty * overextend_depth

            # 2. 压迫驱离奖励 (只在防守半场生效)
            # 计算A1当前到篮筐的距离
            current_dist_a1_to_basket = torch.linalg.norm(self.a1.state.pos - self.basket.state.pos, dim=-1)
            # 如果A1被驱离篮筐，则奖励防守方
            pressure_reward = self.k_pressure * (current_dist_a1_to_basket - self.prev_dist_a1_to_basket)
            dense_reward += pressure_reward * in_defensive_half.float()

            # 3. 有效站位奖励 (只在防守半场生效)
            # 计算防守者到A1的距离
            dist_d_a1 = torch.linalg.norm(p_d - self.a1.state.pos, dim=1)
            
            # 计算A1到篮筐的向量
            a1_to_basket_vec = self.basket.state.pos - self.a1.state.pos
            a1_to_basket_norm_sq = torch.sum(a1_to_basket_vec**2, dim=-1, keepdim=True) + 1e-6

            # 计算防守者到A1的向量
            d_to_a1_vec = p_d - self.a1.state.pos

            # 计算防守者向量在A1到篮筐向量上的投影长度比例
            proj_len_ratio = torch.sum(d_to_a1_vec * a1_to_basket_vec, dim=-1) / a1_to_basket_norm_sq.squeeze(-1)
            
            # 判断防守者是否在A1和篮筐之间 (0 < ratio < 1)
            is_between_a1_basket = (proj_len_ratio > 0) & (proj_len_ratio < 1)

            # 计算防守者到A1-篮筐连线的垂直距离
            projection = proj_len_ratio.unsqueeze(-1) * a1_to_basket_vec
            dist_perp = torch.linalg.norm(d_to_a1_vec - projection, dim=-1)

            # 站位奖励：距离A1越近，且越靠近A1-篮筐连线，奖励越高
            # 使用高斯衰减，距离越远，奖励越小
            positioning_reward = self.k_positioning * torch.exp(-dist_d_a1.pow(2) / (2 * self.agent_radius**2)) * torch.exp(-dist_perp.pow(2) / (2 * self.agent_radius**2))
            
            # 只有当防守者在A1和篮筐之间时才给予站位奖励
            dense_reward += positioning_reward * is_between_a1_basket.float() * in_defensive_half.float()

            # 4. 关键干扰奖励 (事件驱动，只在防守半场生效)
            dense_reward += self.contest_reward_this_step[:, agent_d_idx] * in_defensive_half.float()

            # 重置关键干扰奖励，确保只在投篮发生的那一帧生效
            self.contest_reward_this_step[:, agent_d_idx] = 0.0

            # 5. 限制A1移动空间奖励 (只在防守半场生效)
            dist_d_a1 = torch.linalg.norm(p_d - self.a1.state.pos, dim=1)
            # 奖励防守者在一定范围内贴近A1，但不进入碰撞区域
            # 理想距离范围：2*agent_radius (碰撞) 到 4*agent_radius
            ideal_min_dist = self.agent_radius * 2.0
            ideal_max_dist = self.agent_radius * 4.0

            # 计算一个奖励因子，当距离在理想范围内时为正，否则为负或零
            restriction_factor = torch.where(
                (dist_d_a1 > ideal_min_dist) & (dist_d_a1 < ideal_max_dist),
                1.0 - (dist_d_a1 - ideal_min_dist) / (ideal_max_dist - ideal_min_dist), # 距离越近，奖励越大
                torch.zeros_like(dist_d_a1)
            )
            dense_reward += self.k_a1_restriction * restriction_factor * in_defensive_half.float()

            # 6. 投篮区域控制奖励 (只在防守半场生效)
            current_dist_a1_to_spot = torch.linalg.norm(self.a1.state.pos - self.spot_center.state.pos, dim=1)
            
            # 如果A1在投篮区域之外
            is_a1_outside_spot = current_dist_a1_to_spot > self.R_spot
            # 奖励防守方将A1推离投篮区域
            spot_control_reward_outside = self.k_spot_control * (current_dist_a1_to_spot - self.prev_dist_a1_to_spot) * is_a1_outside_spot.float()
            
            # 如果A1在投篮区域之内
            is_a1_inside_spot = ~is_a1_outside_spot
            # 奖励防守方将A1推向区域边缘 (距离增加)，惩罚A1深入区域 (距离减少)
            spot_control_reward_inside = self.k_spot_control * (current_dist_a1_to_spot - self.prev_dist_a1_to_spot) * is_a1_inside_spot.float()

            dense_reward += (spot_control_reward_outside + spot_control_reward_inside) * in_defensive_half.float()
        
        # 返回稠密奖励和回合结束时的稀疏奖励之和
        return dense_reward + self.terminal_rewards[:, agent_idx]

    def observation(self, agent: Agent):
        # 确定智能体索引和队伍信息
        agent_idx = self.world.agents.index(agent)
        is_attacker = agent_idx < self.n_attackers

        if is_attacker:
            teammate = self.attackers[1 - agent_idx]
            opp1, opp2 = self.defenders[0], self.defenders[1]
            # 进攻方关注投篮点
            key_info_rel = self.spot_center.state.pos - agent.state.pos
        else:
            teammate = self.defenders[1 - (agent_idx - self.n_attackers)]
            opp1, opp2 = self.attackers[0], self.attackers[1]
            # 防守方关注A1到篮筐的路径
            key_info_rel = self.basket.state.pos - self.a1.state.pos

        # 将所有信息拼接成一个观测向量
        obs = torch.cat([
            agent.state.pos,                              # 自身位置
            agent.state.vel,                              # 自身速度
            teammate.state.pos - agent.state.pos,         # 队友相对位置
            teammate.state.vel - agent.state.vel,         # 队友相对速度
            opp1.state.pos - agent.state.pos,             # 对手1相对位置
            opp1.state.vel - agent.state.vel,             # 对手1相对速度
            opp2.state.pos - agent.state.pos,             # 对手2相对位置
            opp2.state.vel - agent.state.vel,             # 对手2相对速度
            key_info_rel,                                 # 关键目标信息
            self.t_remaining / self.t_limit,              # 归一化的剩余时间
        ], dim=-1)

        return obs


if __name__ == "__main__":
    # 使用此脚本可以交互式地运行和测试环境
    render_interactively(
        __file__,
        control_two_agents=True, # 允许手动控制两个智能体进行测试
    )