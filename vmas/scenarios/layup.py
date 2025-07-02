from traceback import print_tb
import torch
from vmas import render_interactively
from vmas.simulator.core import World, Agent, Landmark, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController
import matplotlib
matplotlib.use('Agg') # Crucial for non-interactive plotting
import matplotlib.pyplot as plt
import io
import pyglet
import numpy as np

class Scenario(BaseScenario):
    """
    "飞身上篮"简化版2v2投篮强化学习环境 (已优化和修复版本)
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.viewer_zoom = 3.0
        self.viewer_size = [1400,700]
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
        self.a_shot_threshold = kwargs.get("a_shot_threshold", 0.8)  # 投篮加速度阈值
        self.v_foul_threshold = kwargs.get("v_foul_threshold", 0.6)  # 碰撞犯规速度阈值

        # 奖励系数
        self.max_score = kwargs.get("max_score", 1000.0)
        self.R_foul = kwargs.get("R_foul", 800.0)
        self.k_a1_approach = kwargs.get("k_a1_approach", 20.0)
        self.k_a1_in_spot_reward = kwargs.get("k_a1_in_spot_reward", 2.0)
        self.block_sigma = kwargs.get("block_sigma", 2.5)
        self.k_coll_active = kwargs.get("k_coll_active", 5.0)
        self.k_coll_passive = kwargs.get("k_coll_passive", 0.1)
        self.oob_margin = kwargs.get("oob_margin", 0.05)
        self.oob_penalty = kwargs.get("oob_penalty", -60.0)  # 出界惩罚值
        self.def_proximity_threshold = kwargs.get("def_proximity_threshold", 0.9)  # 防守奖励生效的最大距离
        self.k_a1_blocked_penalty = kwargs.get("k_a1_blocked_penalty", -5.0)  # A1被封堵的惩罚系数
        self.k_velocity_penalty = kwargs.get("k_velocity_penalty", 0.2)
        self.k_def_proximity_penalty = kwargs.get("k_def_proximity_penalty", 10.0) # 防守方近距离惩罚系数
        self.k_def_push_penalty = kwargs.get("k_def_push_penalty", 200.0) # 防守方主动推挤惩罚系数
        self.proximity_penalty_reduction_in_spot = kwargs.get("proximity_penalty_reduction_in_spot", 0.2) # 在投篮圈内近距离惩罚的减少比例
        self.gaussian_sigma = kwargs.get("gaussian_sigma", self.R_spot * 0.5) # 高斯引导奖励的sigma
        self.gaussian_scale = kwargs.get("gaussian_scale", self.k_a1_approach * 3.0) # 高斯引导奖励的缩放系数
        self.low_velocity_threshold = kwargs.get("low_velocity_threshold", self.v_foul_threshold) # 低速推挤判定阈值
        self.k_push_penalty = kwargs.get("k_push_penalty", 200.0) # 主动推挤惩罚系数
        self.k_u_penalty_general = kwargs.get("k_u_penalty_general", 0.01) # 全局控制量惩罚系数
        self.k_u_penalty_a1_in_spot = kwargs.get("k_u_penalty_a1_in_spot", 0.05) # A1在投篮点的额外控制量惩罚
        self.proximity_threshold = kwargs.get("proximity_threshold", self.agent_radius * 2.2) # 近距离惩罚阈值 (直径的1.1倍)
        self.proximity_penalty_margin = kwargs.get("proximity_penalty_margin", 0.15)      # 近距离惩罚曲线的软度
        self.k_proximity_penalty = kwargs.get("k_proximity_penalty", 20)      # 近距离惩罚系数
        self.k_overextend_penalty = kwargs.get("k_overextend_penalty", 10.0)     # 增强的越界惩罚系数
        self.k_positioning = kwargs.get("k_positioning", 60.0)                # 有效站位奖励系数
        self.k_contest = kwargs.get("k_contest", 60.0)                     # 关键干扰奖励系数
        self.k_spot_control = kwargs.get("k_spot_control", 3.0)             # 投篮区域控制奖励系数
        self.k_def_gaussian_spot = kwargs.get("k_def_gaussian_spot", 30)   # 防守方高斯奖励系数
        self.def_gaussian_spot_sigma = kwargs.get("def_gaussian_spot_sigma", self.R_spot) # 防守方高斯奖励宽度
        self.def_pos_offset = kwargs.get("def_pos_offset", self.agent_radius * 2.5) # 防守卡位的前置距离
        self.def_pos_sigma = kwargs.get("def_pos_sigma", 0.9)   
        # A2 掩护奖励新参数                                                                                                                                                                                                               
        self.k_ideal_screen_pos = kwargs.get("k_ideal_screen_pos", 9.0)
        self.screen_pos_offset = kwargs.get("screen_pos_offset", self.agent_radius * 3)
        self.screen_pos_sigma = kwargs.get("screen_pos_sigma", self.R_spot)
        self.k_screen_gate = kwargs.get("k_screen_gate", 5.0)
        self.k_repulsion_reward = kwargs.get("k_repulsion_reward", 9.0) # A2驱离奖励系数  
        self.repulsion_proximity_threshold = kwargs.get("repulsion_proximity_threshold", self.R_spot) # A2驱离奖励生效距离
        self.k_a2_relative_distance_penalty = kwargs.get("k_a2_relative_distance_penalty", 2.0) # A2相对距离惩罚系数  
        # 新增：延迟启动
        self.start_delay_frames = kwargs.get("start_delay_frames", 20) # A1延迟启动的帧数

        # 新增：投篮前静止
        self.shot_still_frames = kwargs.get("shot_still_frames", 4) # 投篮前需要静止的帧数
        self.k_a1_stillness_reward = kwargs.get("k_a1_stillness_reward", 20) # 在投篮点保持静止的奖励
        self.k_a2_shot_line_penalty = kwargs.get("k_a2_shot_line_penalty", 2) # A2在投篮线上的惩罚




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
        self.terminal_rewards = torch.zeros(batch_dim, self.n_agents, device=device)
        self.dones = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.p_vels = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.raw_actions = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.prev_dist_a1_to_basket = torch.zeros(batch_dim, device=device)
        self.contest_reward_this_step = torch.zeros(batch_dim, self.n_defenders, device=device)
        self.prev_dist_a1_to_spot = torch.zeros(batch_dim, device=device)
        self.delay_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.a1_still_frames_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)

        self.rewards_history = [
            [torch.empty((0,), device=world.device, dtype=torch.float32)
             for _ in range(batch_dim)]
            for _ in world.agents
        ]

        self.plot_artists = []
        for i, agent in enumerate(world.agents):
            # 为每个智能体创建一个图表对象
            fig, ax = plt.subplots(figsize=(5, 3), dpi=80) # 可以适当调整分辨率
            fig.tight_layout(pad=1)
            
            # 创建一个空的Line对象作为占位符，之后我们会更新它的数据
            # 'r-' 表示红色的实线
            line, = ax.plot([], [], 'r-') 
            
            ax.set_title(f"Agent {agent.name}", fontsize=6)
            # 你可以在这里设置固定的Y轴范围，或者让它自动调整
            # ax.set_ylim(-1, 1) 

            # 将所有需要用到的对象打包存储起来
            artist_dict = {
                'fig': fig,
                'ax': ax,
                'line': line,
            }
            self.plot_artists.append(artist_dict)

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
        self.delay_counter[batch_range] = self.start_delay_frames
        self.a1_still_frames_counter[batch_range] = 0

        # 设置篮筐和投篮点位置
        basket_pos = torch.zeros(batch_dim, 2, device=self.world.device)
        basket_pos[:, 1] = self.L / 2 - 0.6
        self.basket.set_pos(basket_pos, batch_index=env_index)

        spot_x = (torch.rand(batch_dim, 1, device=self.world.device) - 0.5) * self.W
        spot_y = torch.rand(batch_dim, 1, device=self.world.device) * (self.L / 4) + (self.R_spot/2)
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
            att_x = (torch.rand(num_resample, self.n_attackers, 1, device=self.world.device) - 0.5) * (self.W - 2 * self.agent_radius)
            att_y = (-self.spawn_area_depth) + torch.rand(num_resample, self.n_attackers, 1, device=self.world.device) * self.spawn_area_depth
            # a1放在出生区域右下角
            fixed_x_a1 = -self.W / 2 + 2 * self.agent_radius 
            fixed_y_a1 = -self.L / 2 + 2 * self.agent_radius
            
            att_x[:, 0, :] = fixed_x_a1
            att_y[:, 0, :] = fixed_y_a1

            # 防守方位置
            def_x = (torch.rand(num_resample, self.n_defenders, 1, device=self.world.device) - 0.5) * (self.W - 2 * self.agent_radius)
            def_y = (self.spawn_area_depth) - torch.rand(num_resample, self.n_defenders, 1, device=self.world.device) * self.spawn_area_depth

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

        # 初始化A1到篮筐的距离
        dist_a1_basket = torch.linalg.norm(agent_positions[:, 0, :] - basket_pos, dim=-1)
        self.prev_dist_a1_to_basket[batch_range] = dist_a1_basket

        # 初始化A1到投篮点的距离
        dist_a1_spot = torch.linalg.norm(agent_positions[:, 0, :] - spot_pos, dim=-1)
        self.prev_dist_a1_to_spot[batch_range] = dist_a1_spot

        if env_index is None: # 重置所有环境
            for i in range(len(self.world.agents)):
                for j in range(self.world.batch_dim):
                    self.rewards_history[i][j] = torch.empty((0,), device=self.world.device, dtype=torch.float32)
        else: # 重置单个环境
            for i in range(len(self.world.agents)):
                self.rewards_history[i][env_index] = torch.empty((0,), device=self.world.device, dtype=torch.float32)


    def process_action(self, agent: Agent):
        # 保存模型输出的原始期望速度
        agent_idx = self.world.agents.index(agent)
        self.raw_actions[:, agent_idx, :] = agent.action.u.clone()

        # 如果在延迟期内，A1动作置零
        if agent == self.a1:
            is_delayed = self.delay_counter > 0
            agent.action.u[is_delayed] = 0.0

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

        # 更新延迟计时器
        self.delay_counter = torch.clamp(self.delay_counter - 1, min=0)

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

        
    
    def check_done(self):
        # 初始化一个全为False的done张量
        dones = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        
        # -- 条件1: 尝试投篮 (Shot Attempt) --
        dist_to_spot = torch.linalg.norm(self.a1.state.pos - self.spot_center.state.pos, dim=1)
        in_area = (dist_to_spot <= self.R_spot) & (self.a1.state.pos[:, 1] > 0)
        is_still = torch.linalg.norm(self.a1.state.vel, dim=1) < self.v_shot_threshold
        # 使用原始期望速度判断是否“意图”停止
        not_accelerating = torch.linalg.norm(self.raw_actions[:, 0, :], dim=1) < self.a_shot_threshold

        # 更新静止计时器
        is_ready_to_shoot = in_area & is_still & not_accelerating
        # 如果满足静止条件，计时器+1
        self.a1_still_frames_counter[is_ready_to_shoot] += 1
        # 如果不满足，计时器清零
        self.a1_still_frames_counter[~is_ready_to_shoot] = 0

        # 检查是否达到投篮所需的静止帧数
        shot_attempted = (self.a1_still_frames_counter >= self.shot_still_frames) & ~dones
        if torch.any(shot_attempted):
            # 1. 计算基础分
            final_score = self.max_score * (1 - dist_to_spot[shot_attempted] / self.R_spot)

            # 2. 【新增】计算视野遮挡系数
            shot_b_idx = shot_attempted.nonzero(as_tuple=True)[0]
            a1_pos = self.a1.state.pos[shot_b_idx]
            basket_pos = self.basket.state.pos[shot_b_idx]
            
            # 获取所有防守智能体 (D1, D2) 的位置
            defender_indices = [self.world.agents.index(d) for d in self.defenders]
            blocker_pos = self.all_pos[shot_b_idx][:, defender_indices, :]

            # 从A1到篮筐的向量
            shot_vector = basket_pos - a1_pos
            shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
            
            # 从A1到潜在遮挡者的向量
            blocker_vector = blocker_pos - a1_pos.unsqueeze(1)
            
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
            self.terminal_rewards[shot_b_idx, :self.n_attackers] += final_score_modified.unsqueeze(-1)
            self.terminal_rewards[shot_b_idx, self.n_attackers:] = -final_score_modified.unsqueeze(-1)

            # 检查A2是否离防守方过远 (使用平均距离)
            a1_pos_shot = self.a1.state.pos[shot_b_idx]
            a2_pos_shot = self.a2.state.pos[shot_b_idx]
            def_pos_shot = torch.stack([d.state.pos[shot_b_idx] for d in self.defenders], dim=1) # (num_shots, D, 2)

            # 计算A1到每个防守方的距离，并求平均
            dist_a1_to_def_all = torch.linalg.norm(a1_pos_shot.unsqueeze(1) - def_pos_shot, dim=-1) # (num_shots, D)
            avg_dist_a1_to_def = torch.mean(dist_a1_to_def_all, dim=1) # (num_shots,)

            # 计算A2到每个防守方的距离，并求平均
            dist_a2_to_def_all = torch.linalg.norm(a2_pos_shot.unsqueeze(1) - def_pos_shot, dim=-1) # (num_shots, D)
            avg_dist_a2_to_def = torch.mean(dist_a2_to_def_all, dim=1) # (num_shots,)

            # 如果A2的平均距离大于A1的平均距离，则施加惩罚
            a2_too_far_mask = avg_dist_a2_to_def > avg_dist_a1_to_def
            if torch.any(a2_too_far_mask):
                penalty_b_idx = shot_b_idx[a2_too_far_mask]
                self.terminal_rewards[penalty_b_idx, self.attackers.index(self.a2)] -= self.k_a2_relative_distance_penalty * (avg_dist_a2_to_def[a2_too_far_mask] - avg_dist_a1_to_def[a2_too_far_mask])

            # 5. 分配关键干扰奖励给防守方
            for i, d_idx in enumerate(self.defenders):
                self.contest_reward_this_step[shot_b_idx, i] = self.k_contest * total_block_factor
            
            dones |= shot_attempted

        # -- 条件2: 时间耗尽 (Time Up) --
        time_up = (self.t_remaining.squeeze(-1) <= 0) & ~dones
        if torch.any(time_up):
            self.terminal_rewards[time_up, :self.n_attackers] = -self.R_foul
            self.terminal_rewards[time_up, self.n_attackers:] = self.R_foul
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
        # 1. 计算智能体在x和y轴上超出“安全区域”的平滑深度
        # 我们使用 Softplus (logaddexp) 函数来创建一个平滑的、非负的“越界深度”
        # 当智能体在边界内时，深度约等于0；越过边界后，深度会平滑地线性增长。
        safe_x = self.W / 2 - (self.agent_radius / 2)
        safe_y = self.L / 2 - (self.agent_radius / 2)
        
        # 计算x和y方向的越界深度
        oob_depth_x = torch.logaddexp(torch.tensor(0.0, device=self.world.device), 
                                     (torch.abs(pos[:, 0]) - safe_x) / self.oob_margin)
        oob_depth_y = torch.logaddexp(torch.tensor(0.0, device=self.world.device), 
                                     (torch.abs(pos[:, 1]) - safe_y) / self.oob_margin)
        
        # 2. 将总的越界深度转化为惩罚值
        # oob_penalty 是一个负数，例如 -10.0
        # 我们保留了您之前设计的与速度相关的部分，这会更严厉地惩罚快速出界的行为
        velocity_norm = torch.linalg.norm(agent.state.vel, dim=1) + 1.0
        oob_reward = self.oob_penalty * self.oob_margin * (oob_depth_x + oob_depth_y) * velocity_norm
        dense_reward += oob_reward
        
        # 3. 物理惩罚：当智能体真正出界时，将其速度置零
        # 这里的判断仍然是一个“硬”边界，因为物理效应（比如撞墙）是瞬时的。
        # 我们只对奖励进行平滑，物理效应保持不变。
        is_hard_oob = (torch.abs(pos[:, 0]) > (0.999 * self.W / 2)) | (torch.abs(pos[:, 1]) > (0.999 * self.L / 2))
        agent.state.vel[is_hard_oob] = 0.0

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
            if agent.is_attacker:
                proximity_penalty = -self.k_proximity_penalty * penetration
            else:
                # 防守方近距离惩罚全场生效，但在投篮圈内减少惩罚
                dist_d_to_spot = torch.linalg.norm(pos - self.spot_center.state.pos, dim=-1)
                is_in_spot_area = (dist_d_to_spot <= self.R_spot)
                
                # 根据是否在投篮圈内调整惩罚系数
                adjusted_k_def_proximity_penalty = torch.where(
                    is_in_spot_area,
                    self.k_def_proximity_penalty * (1 - self.proximity_penalty_reduction_in_spot),
                    self.k_def_proximity_penalty
                ).unsqueeze(1)
                proximity_penalty = -adjusted_k_def_proximity_penalty * penetration

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
                if not agent.is_attacker:
                    push_penalty = -self.k_def_push_penalty * torch.clamp(push_force_magnitude, min=0.0)
                
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

            # 新增：对成功保持静止以准备投篮的行为给予奖励
            # a1_still_frames_counter > 0 意味着当前帧满足了静止条件
            is_charging_shot = self.a1_still_frames_counter > 0
            dense_reward += self.k_a1_stillness_reward * is_charging_shot.float()
            
            # 新增：在投篮区内时，施加更大的控制量惩罚以鼓励静止 (基于原始期望速度)
            dense_reward -= self.k_u_penalty_a1_in_spot * raw_u_norm * is_in_spot.float()

            # 4. 被封堵惩罚 (保持不变)
            def_pos = torch.stack([d.state.pos for d in self.defenders], dim=1) # (B, D, 2)
            ap = self.basket.state.pos.unsqueeze(1) - pos.unsqueeze(1)  # (B, 1, 2)
            ad = def_pos - pos.unsqueeze(1) # (B, D, 2)
            
            # 计算投影长度比例，判断防守者是否在A1和篮筐之间
            proj_len_ratio = torch.einsum("bnd,bmd->bnm", ad, ap).squeeze(-1) / (torch.linalg.norm(ap, dim=-1).pow(2) + 1e-6) # (B, D)
            is_between_mask = (proj_len_ratio > 0) & (proj_len_ratio < 1)

            # 计算到投篮路线的垂直距离
            closest_point_on_line = pos.unsqueeze(1) + proj_len_ratio.unsqueeze(-1) * ap
            dist_perp = torch.linalg.norm(def_pos - closest_point_on_line, dim=-1) # (B, D)
            
            # 检查距离是否足够近构成威胁
            dist_to_def = torch.linalg.norm(pos.unsqueeze(1) - def_pos, dim=-1) # (B, D)
            close_enough_mask = dist_to_def < self.def_proximity_threshold

            # 结合所有条件计算最终的封堵因子
            block_factor = torch.exp(-dist_perp.pow(2) / (2 * self.block_sigma ** 2))
            final_block_mask = is_between_mask & close_enough_mask
            total_block_factor = (block_factor * final_block_mask.float()).sum(dim=1) # (B,)
            dense_reward += total_block_factor * self.k_a1_blocked_penalty

        elif agent == self.a2:
            # A2 (无球/掩护者) - 最终版平滑奖励逻辑
            p_a1 = self.a1.state.pos
            p_a2 = pos
            def_pos = torch.stack([d.state.pos for d in self.defenders], dim=1) # (B, D, 2)

            # 1. 找到对 A1 威胁最大的防守者 (离 A1 最近的)
            dist_a1_to_defs = torch.linalg.norm(p_a1.unsqueeze(1) - def_pos, dim=-1) # (B, D)
            _, closest_def_indices = torch.min(dist_a1_to_defs, dim=1) # (B,)
            batch_indices = torch.arange(self.world.batch_dim, device=self.world.device).unsqueeze(1)
            p_closest_def = def_pos[batch_indices, closest_def_indices.unsqueeze(1)].squeeze(1) # (B, 2)

            # 2. 定义动态的“理想掩护点” P_ideal
            # 该点位于 A1 和最近防守者之间，并靠近防守者
            def_to_a1_vec = p_a1 - p_closest_def
            def_to_a1_norm = torch.linalg.norm(def_to_a1_vec, dim=-1, keepdim=True) + 1e-6
            def_to_a1_unit_vec = def_to_a1_vec / def_to_a1_norm
            ideal_screen_pos = p_closest_def + self.screen_pos_offset * def_to_a1_unit_vec

            # 3. 计算核心高斯奖励：奖励靠近“理想掩护点”
            dist_a2_to_ideal = torch.linalg.norm(p_a2 - ideal_screen_pos, dim=-1)
            ideal_screen_reward = self.k_ideal_screen_pos * torch.exp(-dist_a2_to_ideal.pow(2) / (2 * self.screen_pos_sigma**2))

            # 4. 【关键】计算 Sigmoid 门控因子，确保 A2 位于防守者和 A1 之间
            # 从 A2 指向防守者的向量
            vec_a2_to_def = p_closest_def - p_a2
            # 从 A2 指向 A1 的向量
            vec_a2_to_a1 = p_a1 - p_a2
            
            # 计算两个向量的点积。如果 A2 在二者之间，点积为负
            dot_product = torch.sum(vec_a2_to_def * vec_a2_to_a1, dim=-1)
            
            # 使用 Sigmoid 函数将点积转化为一个 0 到 1 之间的平滑因子
            # 当点积为负时 (正确位置), -k * dot_product 为正, sigmoid 输出接近 1
            # 当点积为正时 (错误位置), -k * dot_product 为负, sigmoid 输出接近 0
            soft_gate_factor = torch.sigmoid(-self.k_screen_gate * dot_product)

            # 5. 将核心奖励与门控因子相乘，得到最终的掩护奖励
            final_screen_reward = ideal_screen_reward * soft_gate_factor
            dense_reward += final_screen_reward

            # 7. 新增：驱离奖励 (结果导向)
            # 获取最近防守者的速度
            # all_vel 的形状是 (B, N, 2), 我们需要用正确的索引来获取速度
            defender_indices = torch.tensor([self.world.agents.index(d) for d in self.defenders], device=self.world.device)
            closest_def_agent_indices = defender_indices[closest_def_indices]
            v_closest_def = self.all_vel[batch_indices, closest_def_agent_indices.unsqueeze(1)].squeeze(1)

            # 计算从A1指向最近防守者的单位向量 (驱离方向)
            a1_to_def_vec = p_closest_def - p_a1
            a1_to_def_norm = torch.linalg.norm(a1_to_def_vec, dim=-1, keepdim=True) + 1e-6
            repulsion_direction = a1_to_def_vec / a1_to_def_norm

            # 将防守者速度投影到驱离方向上
            repulsion_speed = torch.sum(v_closest_def * repulsion_direction, dim=-1)

            # 门控条件：只有当A2离该防守者足够近时，才认为这个驱离是A2的功劳
            dist_a2_to_closest_def = torch.linalg.norm(p_a2 - p_closest_def, dim=-1)
            is_a2_responsible = dist_a2_to_closest_def < self.repulsion_proximity_threshold

            # 计算最终奖励：只奖励正向的驱离速度，并且A2要负责
            repulsion_reward = self.k_repulsion_reward * torch.clamp(repulsion_speed, min=0.0) * is_a2_responsible.float()
            dense_reward += repulsion_reward

            # 6. (保留) A2在A1投篮线上的惩罚，防止帮倒忙
            basket_pos = self.basket.state.pos
            shot_vector = basket_pos - p_a1
            shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
            a2_vector = p_a2 - p_a1
            dot_product_shotline = torch.sum(a2_vector * shot_vector, dim=-1)
            proj_len_ratio = dot_product_shotline / shot_vector_norm_sq.squeeze(-1)
            is_between = (proj_len_ratio > 0) & (proj_len_ratio < 1)
            projection = proj_len_ratio.unsqueeze(-1) * shot_vector
            dist_perp_sq = torch.sum((a2_vector - projection)**2, dim=-1)
            a2_line_block_sigma = self.agent_radius * 0.5
            intersects_line = dist_perp_sq < a2_line_block_sigma**2
            is_a2_on_shot_line = is_between & intersects_line
            dense_reward -= self.k_a2_shot_line_penalty * is_a2_on_shot_line.float()

        else: # 防守方
            p_d = pos
            agent_d_idx = self.defenders.index(agent)
            in_defensive_half = p_d[:, 1] > 0
            
            # 1. 越过半场惩罚 (与越界深度成正比)
            # 只有当Y坐标小于0时才计算越界深度
            overextend_depth = torch.where(p_d[:, 1] < 0, torch.abs(p_d[:, 1]) + 1 , torch.zeros_like(p_d[:, 1]))
            dense_reward -= self.k_overextend_penalty * overextend_depth

            # 2. 有效站位奖励 (改进版：基于理想防守点，更平滑)
            # 这个奖励引导防守者主动去封堵A1前往篮筐的路线
            
            # 2.1. 计算从A1指向篮筐的单位向量
            a1_to_basket_vec = self.basket.state.pos - self.a1.state.pos
            a1_to_basket_norm = torch.linalg.norm(a1_to_basket_vec, dim=-1, keepdim=True) + 1e-6
            a1_to_basket_unit_vec = a1_to_basket_vec / a1_to_basket_norm

            # 2.2. 定义动态的“理想防守点” P_ideal
            ideal_pos = self.a1.state.pos + self.def_pos_offset * a1_to_basket_unit_vec

            # 2.3. 计算到理想点的距离，并生成基础高斯奖励
            dist_to_ideal = torch.linalg.norm(p_d - ideal_pos, dim=-1)
            base_positioning_reward = self.k_positioning * torch.exp(-dist_to_ideal.pow(2) / (2 * self.def_pos_sigma**2))

            # 2.4. 创建一个“软门控”因子，判断防守者是否在A1身前
            # 我们使用投射的点积来判断前后关系
            d_to_a1_vec = p_d - self.a1.state.pos
            # 点积 > 0 表示在身前，< 0 表示在身后
            proj_dot_product = torch.sum(d_to_a1_vec * a1_to_basket_unit_vec, dim=-1)
            
            # 使用Sigmoid函数将点积转化为一个0到1之间的平滑因子
            # k值控制过渡带的宽度，k越大，过渡越快，越接近硬截止
            k_smooth = 5.0
            soft_gate_factor = torch.sigmoid(k_smooth * proj_dot_product)
            
            # 2.5. 将基础奖励与软门控相乘，得到最终奖励
            final_positioning_reward = base_positioning_reward * soft_gate_factor
            
            # 2.6. 应用奖励
            dense_reward += final_positioning_reward * in_defensive_half.float()

            # 3. 关键干扰奖励 (事件驱动，只在防守半场生效)
            dense_reward += self.contest_reward_this_step[:, agent_d_idx] * in_defensive_half.float()

            # 重置关键干扰奖励，确保只在投篮发生的那一帧生效
            self.contest_reward_this_step[:, agent_d_idx] = 0.0

            # 4. 投篮区域控制奖励 (只在防守半场生效)
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

            # 5. 防守方高斯引导奖励 (只在防守半场生效)
            # 鼓励防守方靠近投篮点
            dist_d_to_spot = torch.linalg.norm(p_d - self.spot_center.state.pos, dim=-1)
            def_gaussian_reward = self.k_def_gaussian_spot * torch.exp(- (dist_d_to_spot**2) / (2 * self.def_gaussian_spot_sigma**2))
            dense_reward += def_gaussian_reward * in_defensive_half.float()
            
        # 返回稠密奖励和回合结束时的稀疏奖励之和
        rew = dense_reward + self.terminal_rewards[:, agent_idx]

        # 如果是A1且在延迟期内，将对应环境的奖励置为0
        if agent == self.a1:
            is_delayed = self.delay_counter > 0
            rew = torch.where(is_delayed, 0.0, rew)
    
        for i in range(self.world.batch_dim):
            # 提取单个环境的奖励，并保持为Tensor
            reward_for_env = rew[i].unsqueeze(0)
            # torch.cat 可以在GPU上高效执行
            self.rewards_history[agent_idx][i] = torch.cat(
                (self.rewards_history[agent_idx][i], reward_for_env)
            )
        return rew

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
    
    def extra_render(self, env_index: int):
        geoms = []

        from vmas.simulator.rendering import Geom
        import io
        
        # SpriteGeom 类保持不变，它已经很好了
        class SpriteGeom(Geom):
            def __init__(self, image, x, y, target_width, target_height):
                super().__init__()
                # 1. 获取原始图像的纹理
                texture = image.get_texture()
                
                # 2. 【关键】使用内置方法获取一个垂直翻转的纹理视图
                # 这不会移动任何像素数据，只是改变了渲染方式
                flipped_texture = texture.get_transform(flip_y=True)
                self.sprite = pyglet.sprite.Sprite(img=flipped_texture, x=x, y=y)
                # 自动计算缩放比例
                if self.sprite.width > 0: self.sprite.scale_x = target_width / self.sprite.width
                if self.sprite.height > 0: self.sprite.scale_y = target_height / self.sprite.height
                
                self.sprite.blend_src = pyglet.gl.GL_SRC_ALPHA
                self.sprite.blend_dest = pyglet.gl.GL_ONE_MINUS_SRC_ALPHA

            def render1(self):
                self.sprite.draw()

        # --- 绘图区域定义 ---
        plot_width = 10  # 期望的显示宽度
        plot_height = 6  # 期望的显示高度
        # 固定的位置列表
        pose_list = [(-14, 0), (4, 0), (-14, -6), (4, -6)] 

        for i, agent in enumerate(self.world.agents):
            history_tensor = self.rewards_history[i][env_index]
            
            if history_tensor.numel() <= 1: # 至少需要两个点才能画线
                continue

            history_np = history_tensor.cpu().numpy()

            # --- 高效的艺术家更新 ---
            # 1. 从初始化好的列表中获取艺术家对象
            artists = self.plot_artists[i]
            fig = artists['fig']
            ax = artists['ax']
            line = artists['line']

            # 2. 只更新线条的数据
            # 创建x轴数据（0, 1, 2, ...）
            x_data = range(len(history_np))
            line.set_data(x_data, history_np)

            # 3. 自动调整坐标轴范围以适应新数据
            ax.relim()
            ax.autoscale_view()

            # 4. 将更新后的画布绘制到内存缓冲区
            # 这是比 savefig 更快的做法
            with io.BytesIO() as buf:
                fig.canvas.draw()
                # 从canvas直接获取RGBA像素数据
                image_data = fig.canvas.buffer_rgba().tobytes()
                # 将其转换为pyglet图像
                plot_image = pyglet.image.ImageData(
                    fig.canvas.get_width_height()[0],
                    fig.canvas.get_width_height()[1],
                    'RGBA',
                    image_data
                )
            
                # --- 渲染逻辑 ---
                if i < len(pose_list):
                    x, y = pose_list[i]
                    # 注意：这里的坐标单位是世界坐标，您可能需要根据场景调整
                    # 如果希望它们固定在屏幕上，可能需要用到我们之前讨论的HUD方法
                    img_geom = SpriteGeom(plot_image, x, y+6, plot_width, plot_height)
                    geoms.append(img_geom)

        return geoms

if __name__ == "__main__":
    # 使用此脚本可以交互式地运行和测试环境
    render_interactively(
        __file__,
        control_two_agents=True, # 允许手动控制两个智能体进行测试
    )