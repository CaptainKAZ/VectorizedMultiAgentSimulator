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
        """
        This function needs to be implemented when creating a scenario.
        In this function the user should instantiate the world and insert agents and landmarks in it.

        Args:
            batch_dim (int): the number of vecotrized environments.
            device (Union[str, int, torch.device], optional): the device of the environmemnt.
            kwargs (dict, optional): named arguments passed from environment creation

        Returns:
            :class:`~vmas.simulator.core.World` : the :class:`~vmas.simulator.core.World`
            instance which is automatically set in :class:`~world`.

        Examples:
            >>> from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def make_world(self, batch_dim: int, device: torch.device, **kwargs):
            ...         # Pass any kwargs you desire when creating the environment
            ...         n_agents = kwargs.get("n_agents", 5)
            ...
            ...         # Create world
            ...         world = World(batch_dim, device, dt=0.1, drag=0.25, dim_c=0)
            ...         # Add agents
            ...         for i in range(n_agents):
            ...             agent = Agent(
            ...                 name=f"agent {i}",
            ...                 collide=True,
            ...                 mass=1.0,
            ...                 shape=Sphere(radius=0.04),
            ...                 max_speed=None,
            ...                 color=Color.BLUE,
            ...                 u_range=1.0,
            ...             )
            ...             world.add_agent(agent)
            ...         # Add landmarks
            ...         for i in range(5):
            ...             landmark = Landmark(
            ...                 name=f"landmark {i}",
            ...                 collide=True,
            ...                 movable=False,
            ...                 shape=Box(length=0.3,width=0.1),
            ...                 color=Color.RED,
            ...             )
            ...             world.add_landmark(landmark)
            ...         return world
        """

        self.viewer_zoom = 3.0
        self.viewer_size = [1400,700]
        # ----------------- 超参数设定 (Hyperparameters) -----------------
       # ========== 1. 场地与物理属性 (Arena & Physics) ==========
        # 场地宽度 (x-axis)
        self.W = kwargs.get("W", 8.0)
        # 场地长度 (y-axis)
        self.L = kwargs.get("L", 15.0)
        # 智能体出生区域的深度
        self.spawn_area_depth = kwargs.get("spawn_area_depth", 1.0)
        # 投篮区域的半径
        self.R_spot = kwargs.get("R_spot", 1.5)
        # 每回合的进攻时限 (秒)
        self.t_limit = kwargs.get("t_limit", 20.0)
        # 物理引擎的步长 (秒)
        self.dt = kwargs.get("dt", 0.1)
        # 智能体的半径，用于碰撞检测
        self.agent_radius = kwargs.get("agent_radius", 0.3)
        # 智能体的最大加速度
        self.a_max = kwargs.get("a_max", 3.0)
        # 智能体的最大速度
        self.v_max = kwargs.get("v_max", 5.0)

        # ========== 2. 终止条件阈值 (Termination Thresholds) ==========
        # A1被认为“静止”以尝试投篮的最大速度阈值
        self.v_shot_threshold = kwargs.get("v_shot_threshold", 0.2)
        # A1被认为“无加速意图”以尝试投篮的最大动作(加速度)阈值
        self.a_shot_threshold = kwargs.get("a_shot_threshold", 0.8)
        # 两个智能体之间的碰撞被判定为“犯规”的最小相对速度阈值
        self.v_foul_threshold = kwargs.get("v_foul_threshold", 0.6)
        
        # ========== 3. 稠密奖励/惩罚系数 (Dense Reward/Penalty Coefficients) ==========
        self.dense_reward_factor = kwargs.get("dense_reward_factor", 0.1)
        # --- 3.1 A1 (持球进攻者) 奖励 ---
        # A1成功进入投篮区域的基础奖励系数。奖励大小与离区域中心距离成反比
        self.k_a1_in_spot_reward = kwargs.get("k_a1_in_spot_reward", 5.0)
        # 速度朝向spot奖励
        self.k_a1_speed_spot_reward = kwargs.get("k_a1_speed_spot_reward", 20.0)
        # A1在投篮区域内移动的速度惩罚系数，鼓励其减速准备投篮
        self.k_velocity_penalty = kwargs.get("k_velocity_penalty", 0.6)
        # A1在投篮点附近保持静止以“蓄力”投篮的奖励系数
        self.k_a1_stillness_reward = kwargs.get("k_a1_stillness_reward", 30)
        # A1的投篮线路被防守者封堵时的惩罚系数
        self.k_a1_blocked_penalty = kwargs.get("k_a1_blocked_penalty", -5.0)
        # 用于引导A1到投篮点的高斯奖励的缩放(峰值)系数
        self.gaussian_scale = kwargs.get("gaussian_scale", 30.0)
        # 用于引导A1到投篮点的高斯奖励的宽度(sigma)
        self.gaussian_sigma = kwargs.get("gaussian_sigma", 0.5 * self.R_spot)
        # 低速阈值：在投篮区内，当速度低于此值时，其控制量会受到惩罚
        self.low_u_threshold = kwargs.get("low_u_threshold", 0.4)
        # 当A1在投篮区域内且低速时，施加的额外控制量惩罚，以鼓励其静止
        self.k_control_penalty_at_low_speed = kwargs.get("k_control_penalty_at_low_speed", 3)
        # 当A1在投篮区域外且低速时，施加惩罚鼓励其移动位置进攻
        self.hesitate_speed_threshold = kwargs.get("hesitate_speed_threshold", self.low_u_threshold)
        # 惩罚量
        self.k_hesitation_penalty = kwargs.get("k_hesitation_penalty", 10)

        # A1的排斥力场范围，可以设得比别人更大
        self.a1_proximity_threshold = kwargs.get("a1_proximity_threshold", self.agent_radius * 3.0) 
        # A1的边界软度，值越大，惩罚从0到最大值的过渡越平缓（越软）
        self.a1_proximity_penalty_margin = kwargs.get("a1_proximity_penalty_margin", 0.1) 
        # A1的近距离惩罚系数
        self.k_a1_proximity_penalty = kwargs.get("k_a1_proximity_penalty", 20)

        # --- 3.2 A2 (掩护进攻者) 奖励 ---
        # 奖励A2移动到理想掩护位置的系数 (基于高斯函数)
        self.k_ideal_screen_pos = kwargs.get("k_ideal_screen_pos", 10.0)
        # A2驱离防守者(使其远离A1)的奖励系数，这是一个结果导向的奖励
        self.k_repulsion_reward = kwargs.get("k_repulsion_reward", 10.0)
        # 判定A2驱离奖励是否生效的最大距离 (A2必须离被驱离的防守者足够近)
        self.repulsion_proximity_threshold = kwargs.get("repulsion_proximity_threshold", self.R_spot)
        # 惩罚A2挡住A1投篮线路的系数，防止“帮倒忙”
        self.k_a2_shot_line_penalty = kwargs.get("k_a2_shot_line_penalty", 5)
        # 定义理想掩护点在“防守者-A1”连线上的偏移量 (靠近防守者一侧)
        self.screen_pos_offset = kwargs.get("screen_pos_offset", self.agent_radius * 3)
        # A2掩护位置高斯奖励的宽度(sigma)，决定了奖励区域的大小
        self.screen_pos_sigma = kwargs.get("screen_pos_sigma", self.R_spot)
        # 用于判断A2是否在A1和防守者之间的Sigmoid门控函数的陡峭程度
        self.k_screen_gate = kwargs.get("k_screen_gate", 7.0)

        # 时间紧迫性惩罚
        self.time_penalty_grace_period = kwargs.get("time_penalty_grace_period",10)
        self.k_time_penalty = kwargs.get("k_time_penalty", 1)

        # --- 3.3 防守方奖励 ---
        # 奖励防守者占据“A1-篮筐”之间的有利位置的系数
        self.k_positioning = kwargs.get("k_positioning", 30.0)
        # 奖励防守者成功将A1“推离”投篮点的系数
        self.k_spot_control = kwargs.get("k_spot_control", 3.0)
        # 惩罚防守者越过半场的系数，鼓励其保持防守阵型
        self.k_overextend_penalty = kwargs.get("k_overextend_penalty", 40.0)
        # 吸引防守者向投篮点移动的高斯奖励的系数
        self.k_def_gaussian_spot = kwargs.get("k_def_gaussian_spot", 10)
        # 防守者投篮点高斯吸引奖励的宽度(sigma)
        self.def_gaussian_spot_sigma = kwargs.get("def_gaussian_spot_sigma", 0.6 * self.R_spot)
        # 定义理想防守位置在A1前方的偏移距离
        self.def_pos_offset = kwargs.get("def_pos_offset", self.agent_radius * 2.5)
        # 理想防守位置高斯奖励的宽度(sigma)，决定了对站位精确度的要求
        self.def_pos_sigma = kwargs.get("def_pos_sigma", 0.9)   

        # --- 3.4 通用惩罚 (对所有智能体生效) ---
        # 出界的基础惩罚值 (负数)
        self.oob_penalty = kwargs.get("oob_penalty", -90.0)
        # 用于平滑计算出界惩罚的边界宽度，值越小，惩罚在边界处变化越剧烈
        self.oob_margin = kwargs.get("oob_margin", 0.05)
        # 主动碰撞者受到的惩罚系数
        self.k_coll_active = kwargs.get("k_coll_active", 5.0)
        # 被动碰撞者受到的惩罚系数
        self.k_coll_passive = kwargs.get("k_coll_passive", 0.1)
        # 全局控制量(action)惩罚系数，鼓励智能体使用更小的力，节省能量
        self.k_u_penalty_general = kwargs.get("k_u_penalty_general", 0.01)
        # 智能体之间触发近距离惩罚的距离阈值
        self.proximity_threshold = kwargs.get("proximity_threshold", self.agent_radius * 2.2)
        # 平滑近距离惩罚曲线的软度/宽度
        self.proximity_penalty_margin = kwargs.get("proximity_penalty_margin", 0.15)      
        # 通用近距离惩罚系数 (针对进攻方)
        self.k_proximity_penalty = kwargs.get("k_proximity_penalty", 60)
        # 防守方之间的近距离惩罚系数，防止防守者扎堆
        self.k_def_proximity_penalty = kwargs.get("k_def_proximity_penalty", 10.0)
        # 当防守方在投篮圈内时，对其近距离扎堆惩罚的减免比例，允许在关键区域紧逼
        self.proximity_penalty_reduction_in_spot = kwargs.get("proximity_penalty_reduction_in_spot", 0.2)
        # 判断低速“推挤”行为的速度上限
        self.low_velocity_threshold = kwargs.get("low_velocity_threshold", self.v_foul_threshold)
        # 进攻方低速推挤行为的惩罚系数
        self.k_push_penalty = kwargs.get("k_push_penalty", 100.0)
        # 防守方低速推挤行为的惩罚系数 (通常比进攻方更高)
        self.k_def_push_penalty = kwargs.get("k_def_push_penalty", 100.0)
        # 判断为站位的速度阈值
        self.stand_still_threshold = kwargs.get("stand_still_threshold", 0.6)
        # 成功站定“造撞人”的奖励系数
        self.k_stand_still_reward = kwargs.get("k_stand_still_reward", 10.0)
        # “造撞人”奖励生效的最大距离
        self.charge_drawing_range = kwargs.get("charge_drawing_range", self.agent_radius * 6.0)

        # ========== 4. 终局奖励/惩罚系数 (Terminal Reward/Penalty Coefficients) ==========
        # --- 4.1 进攻方终局奖励 ---
        # 在投篮点中心成功投篮可获得的最大基础分
        self.max_score = kwargs.get("max_score", 2000.0)
        # A1投篮时，根据与防守者的距离(空间)获得的额外奖励系数
        self.k_spacing_bonus = kwargs.get("k_spacing_bonus", 10.0)
        # A2在A1成功投篮时，因处于良好掩护位置而获得的额外奖励系数
        self.k_a2_screen_bonus = kwargs.get("k_a2_screen_bonus", 400.0)
        # 用于判断A2掩护位置是否成功的高斯宽度(sigma)
        self.a2_screen_sigma = kwargs.get("a2_screen_sigma", 4 * self.agent_radius)
        # 时间耗尽时，对进攻方的基础惩罚
        self.R_foul = kwargs.get("R_foul", 800.0)
        # 时间耗尽时，根据A1离投篮点的距离施加的额外惩罚系数
        self.k_timeout_dist_penalty = kwargs.get("k_timeout_dist_penalty", 50.0)
        
        # --- 4.2 防守方终局奖励 ---
        # 防守方因成功干扰(封盖)投篮而获得的奖励系数
        self.k_def_block_reward = kwargs.get("k_def_block_reward", 800.0)
        # 防守方因成功逼迫A1在远离投篮点中心的位置出手而获得的奖励系数
        self.k_def_force_reward = kwargs.get("k_def_force_reward", 100.0)
        # 防守方因在投篮瞬间占据有利防守位置而获得的奖励系数
        self.k_def_pos_reward = kwargs.get("k_def_pos_reward", 400.0)
        # 防守方因在投篮瞬间有效控制了投篮区域而获得的奖励系数
        self.k_def_area_reward = kwargs.get("k_def_area_reward", 100.0)
        # 防守方放任投篮的固有惩罚
        self.k_def_shot_penalty = kwargs.get("k_def_shot_penalty", 300.0)
        
        # --- 4.3 犯规相关终局奖惩 ---
        # 犯规惩罚会根据碰撞的相对速度动态增加，此为增加的系数
        self.k_foul_vel_penalty = kwargs.get("k_foul_vel_penalty", 800.0)
        # 犯规发生时，犯规者和受害者的队友所受到的奖惩影响比例 (0到1之间)
        self.foul_teammate_factor = kwargs.get("foul_teammate_factor", 0.25)

        self.wall_collision_frames = kwargs.get("wall_collision_frames", 10)
        self.R_wall_collision_penalty = kwargs.get("R_wall_collision_penalty", -800.0) # 巨大的负奖励


        # ========== 5. 其他行为控制参数 (Other Behavior Control) ==========
        # 回合开始时，A1需要保持静止的帧数，给与防守方反应时间
        self.start_delay_frames = kwargs.get("start_delay_frames", 20)
        # A1在投篮区域内需要保持静止以触发投篮的连续帧数
        self.shot_still_frames = kwargs.get("shot_still_frames", 4)
        # 判断防守者是否构成封堵威胁的距离阈值
        self.def_proximity_threshold = kwargs.get("def_proximity_threshold", 0.9)
        # 用于计算封堵程度的高斯宽度
        self.block_sigma = kwargs.get("block_sigma", self.agent_radius * 1.5)

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
        self.prev_dist_a1_to_spot = torch.zeros(batch_dim, device=device)
        self.delay_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.a1_still_frames_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.wall_collision_counters = torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32)
        self.shots_this_step = torch.zeros(batch_dim, device=device, dtype=torch.bool)

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
        """Resets the world at the specified env_index.

        When a ``None`` index is passed, the world should make a vectorized (batched) reset.
        The ``entity.set_x()`` methods already have this logic integrated and will perform
        batched operations when index is ``None``.

        When this function is called, all entities have already had their state reset to zeros according to the ``env_index``.
        In this function you shoud change the values of the reset states according to your task.
        For example, some functions you might want to use are:

        - ``entity.set_pos()``,
        - ``entity.set_vel()``,
        - ``entity.set_rot()``,
        - ``entity.set_ang_vel()``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            env_index (int, otpional): index of the environment to reset. If ``None`` a vectorized reset should be performed.

        Spawning at fixed positions

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def reset_world_at(self, env_index)
            ...        for i, agent in enumerate(self.world.agents):
            ...            agent.set_pos(
            ...                torch.tensor(
            ...                     [-0.2 + 0.1 * i, 1.0],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...        for i, landmark in enumerate(self.world.landmarks):
            ...            landmark.set_pos(
            ...                torch.tensor(
            ...                     [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...            landmark.set_rot(
            ...                torch.tensor(
            ...                     [torch.pi / 4 if i % 2 else -torch.pi / 4],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )

        Spawning at random positions

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import ScenarioUtils
            >>> class Scenario(BaseScenario):
            >>>     def reset_world_at(self, env_index)
            >>>         ScenarioUtils.spawn_entities_randomly(
            ...             self.world.agents + self.world.landmarks,
            ...             self.world,
            ...             env_index,
            ...             min_dist_between_entities=0.02,
            ...             x_bounds=(-1.0,1.0),
            ...             y_bounds=(-1.0,1.0),
            ...         )

        """
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
        self.delay_counter[batch_range] = self.start_delay_frames
        self.a1_still_frames_counter[batch_range] = 0
        self.wall_collision_counters[batch_range] = 0

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
        """This function can be overridden to process the agent actions before the simulation step.

        It has access to the world through the :class:`world` attribute

        For example here you can manage additional actions before passing them to the dynamics.

        Args:
            agent (Agent): the agent process the action of

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import TorchUtils
            >>> class Scenario(BaseScenario):
            >>>     def process_action(self, agent):
            >>>         # Clamp square to circle
            >>>         agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)
            >>>         # Can use a PID controller to turn velocity actions into forces
            >>>         # (e.g., from vmas.simulator.controllers.velocity_controller)
            >>>         agent.controller.process_force()
            >>>         return
        """
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
        """This function can be overridden to perform any computation that has to happen before the simulation step.
        Its intended use is for computation that has to happen only once before the simulation step has accured.

        For example, you can store temporal data before letting the world step.

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> class Scenario(BaseScenario):
            >>>     def pre_step(self):
            >>>         for agent in self.world.agents:
            >>>             agent.prev_state = agent.state
            >>>         return
        """
        """
        在每个物理步长开始前执行。
        用于集中计算所有智能体间的交互信息，避免重复计算。
        """
        # 为当前时间步重置投篮标志位
        self.shots_this_step.zero_()
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

         # 定义墙的边界
        wall_x = self.world.x_semidim * 0.999
        wall_y = self.world.y_semidim * 0.999

        # 判断每个智能体是否正在“推墙”
        # 条件：1. 紧贴墙边 2. 速度方向朝向墙外
        is_pushing_wall_x = ((self.all_pos[..., 0] > wall_x)) | \
                            ((self.all_pos[..., 0] < -wall_x))
        is_pushing_wall_y = ((self.all_pos[..., 1] > wall_y)) | \
                            ((self.all_pos[..., 1] < -wall_y))

        is_pushing_wall = is_pushing_wall_x | is_pushing_wall_y # (B, N)

        # 更新计数器
        # 如果正在推墙，计数器+1
        self.wall_collision_counters[is_pushing_wall] += 1
        # 如果没有推墙，计数器清零 (这是实现“连续”的关键)
        self.wall_collision_counters[~is_pushing_wall] = 0

        # 5. 检查并设置 done 标志
        self.dones = self.check_done()

    def post_step(self):
        """This function can be overridden to perform any computation that has to happen after the simulation step.
        Its intended use is for computation that has to happen only once after the simulation step has accured.

        For example, you can store temporal sensor data in this function.

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> class Scenario(BaseScenario):
            >>>     def post_step(self):
            >>>         for agent in self.world.agents:
            >>>             # Let the sensor take a measurement
            >>>             measurements = agent.sensors[0].measure()
            >>>             # Store sensor data in agent.sensor_history
            >>>             agent.sensor_history.append(measurements)
            >>>         return
        """
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
        # (此部分被完全重构以实现新的奖励逻辑)
        dist_to_spot = torch.linalg.norm(self.a1.state.pos - self.spot_center.state.pos, dim=1)
        in_area = (dist_to_spot <= self.R_spot) & (self.a1.state.pos[:, 1] > 0)
        is_still = torch.linalg.norm(self.a1.state.vel, dim=1) < self.v_shot_threshold
        not_accelerating = torch.linalg.norm(self.raw_actions[:, 0, :], dim=1) < self.a_shot_threshold

        is_ready_to_shoot = in_area & is_still & not_accelerating
        self.a1_still_frames_counter[is_ready_to_shoot] += 1
        self.a1_still_frames_counter[~is_ready_to_shoot] = 0

        shot_attempted = (self.a1_still_frames_counter >= self.shot_still_frames) & ~dones
        if torch.any(shot_attempted):
            self.shots_this_step = shot_attempted
            print(f"got {torch.sum(shot_attempted).item()} shot attempt")
            shot_b_idx = shot_attempted.nonzero(as_tuple=True)[0]
            # --- 提取投篮瞬间的关键状态 ---
            a1_pos = self.a1.state.pos[shot_b_idx]
            a2_pos = self.a2.state.pos[shot_b_idx]
            spot_pos = self.spot_center.state.pos[shot_b_idx]
            basket_pos = self.basket.state.pos[shot_b_idx]
            
            defender_indices = [self.world.agents.index(d) for d in self.defenders]
            defender_pos = self.all_pos[shot_b_idx][:, defender_indices, :]

            # --- 通用计算：封堵干扰系数 ---
            shot_vector = basket_pos - a1_pos
            blocker_vector = defender_pos - a1_pos.unsqueeze(1)
            shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
            dot_product = torch.sum(blocker_vector * shot_vector.unsqueeze(1), dim=-1)
            proj_len_ratio = dot_product / shot_vector_norm_sq
            is_between = (proj_len_ratio > 0) & (proj_len_ratio < 1)
            projection = proj_len_ratio.unsqueeze(-1) * shot_vector.unsqueeze(1)
            dist_perp_sq = torch.sum((blocker_vector - projection)**2, dim=-1)
            is_blocker_per_defender = is_between & (dist_perp_sq < (self.proximity_threshold)**2)
            block_contribution = torch.exp(-dist_perp_sq / (2 * self.block_sigma**2)) * is_blocker_per_defender.float()
            total_block_factor = torch.clamp(block_contribution.sum(dim=1), 0, 1)

            # --- 进攻方终局奖励计算 ---
            # 1. A1的奖励
            base_score = self.max_score * (1 - dist_to_spot[shot_b_idx] / self.R_spot)
            final_score_modified = base_score * (1 - total_block_factor)
            dist_a1_to_defs = torch.linalg.norm(blocker_vector, dim=-1)
            avg_dist_to_defs = torch.mean(dist_a1_to_defs, dim=1)
            spacing_bonus = self.k_spacing_bonus * avg_dist_to_defs
            a1_reward = final_score_modified + spacing_bonus
            self.terminal_rewards[shot_b_idx, 0] += a1_reward

            # 2. A2的奖励 (包含团队得分 + 个人掩护奖励)
            # 2.1 识别关键防守者 (离A1最近的)
            dist_a1_to_defs_sq = torch.sum(blocker_vector**2, dim=-1)
            _, closest_def_indices = torch.min(dist_a1_to_defs_sq, dim=1)
            batch_indices = torch.arange(len(shot_b_idx), device=self.world.device)
            p_closest_def = defender_pos[batch_indices, closest_def_indices]
            
            # 2.2 计算理想掩护位置 (在A1和关键防守者之间，靠近防守者)
            def_to_a1_vec = a1_pos - p_closest_def
            def_to_a1_norm = torch.linalg.norm(def_to_a1_vec, dim=-1, keepdim=True) + 1e-6
            def_to_a1_unit_vec = def_to_a1_vec / def_to_a1_norm
            ideal_screen_pos = p_closest_def + self.screen_pos_offset * def_to_a1_unit_vec
            
            # 2.3 计算A2的个人掩护奖励
            dist_a2_to_ideal_sq = torch.sum((a2_pos - ideal_screen_pos)**2, dim=-1)
            # 使用门控，确保A2在A1和防守者之间
            vec_a2_to_def = p_closest_def - a2_pos
            vec_a2_to_a1 = a1_pos - a2_pos
            dot_product_gate = torch.sum(vec_a2_to_def * vec_a2_to_a1, dim=-1)
            screen_gate = torch.sigmoid(-self.k_screen_gate * dot_product_gate) # 在中间时，gate接近1
            
            screen_bonus = self.k_a2_screen_bonus * torch.exp(-dist_a2_to_ideal_sq / (2 * self.a2_screen_sigma**2)) * screen_gate
            
            # A2总奖励 = 基础团队分 + 个人掩护绩效
            a2_reward = final_score_modified + screen_bonus + spacing_bonus
            self.terminal_rewards[shot_b_idx, 1] += a2_reward

            # --- 防守方终局奖励计算 (独立的、累加的奖励体系) ---
            # 对每个防守队员独立计算奖励
            for i in range(self.n_defenders):
                # 奖励1: 成功干扰奖励 (基于个人对投篮的直接贡献)
                R_block = self.k_def_block_reward * block_contribution[:, i]

                # 奖励2: 成功逼迫奖励 (基于A1被迫远离投篮点)
                # 这个奖励对两个防守队员是相同的，因为这是团队努力的结果
                dist_a1_to_spot = dist_to_spot[shot_b_idx]
                R_force = self.k_def_force_reward * (dist_a1_to_spot / self.R_spot)

                # 奖励3: 成功卡位奖励 (基于锥形高斯逻辑)
                a1_to_spot_unit_vec = (basket_pos - a1_pos) / (torch.linalg.norm(basket_pos - a1_pos, dim=-1, keepdim=True) + 1e-6)
                d_from_a1_vec = defender_pos[:, i, :] - a1_pos
                proj_dot = torch.sum(d_from_a1_vec * a1_to_spot_unit_vec, dim=-1)
                pos_gate = torch.sigmoid(5.0 * proj_dot) # 在前方时gate接近1
                
                ideal_pos = a1_pos + self.def_pos_offset * a1_to_spot_unit_vec
                dist_to_ideal_sq = torch.sum((defender_pos[:, i, :] - ideal_pos)**2, dim=-1)
                positioning_reward_factor = torch.exp(-dist_to_ideal_sq / (2 * self.def_pos_sigma**2))
                R_positioning = self.k_def_pos_reward * positioning_reward_factor * pos_gate

                # 奖励4: 区域控制奖励 (基于对关键区域的控制)
                dist_def_to_spot_sq = torch.sum((defender_pos[:, i, :] - spot_pos)**2, dim=-1)
                R_area_control = self.k_def_area_reward * torch.exp(-dist_def_to_spot_sq / (2 * self.def_gaussian_spot_sigma**2))

                # 累加所有奖励项
                total_def_reward = R_block + R_force + R_positioning + R_area_control - self.k_def_shot_penalty
                self.terminal_rewards[shot_b_idx, self.n_attackers + i] += total_def_reward
            
            dones |= shot_attempted

        # -- 条件2: 时间耗尽 (Time Up) --
        time_up = (self.t_remaining.squeeze(-1) <= 0) & ~dones
        if torch.any(time_up):
            a1_pos_timeout = self.a1.state.pos[time_up]
            dist_a1_to_spot_timeout = torch.linalg.norm(a1_pos_timeout - self.spot_center.state.pos[time_up], dim=-1)
            attacker_penalty = -self.R_foul - self.k_timeout_dist_penalty * dist_a1_to_spot_timeout
            self.terminal_rewards[time_up, :self.n_attackers] = attacker_penalty.unsqueeze(-1)
            self.terminal_rewards[time_up, self.n_attackers:] = -attacker_penalty.unsqueeze(-1)
            dones |= time_up

        # -- 条件3: 碰撞犯规 (Foul) --
        is_foul = self.collision_matrix & (self.vel_diffs_norm > self.v_foul_threshold) & ~dones.view(-1, 1, 1)
        if torch.triu(is_foul, diagonal=1).any():
            b_idx, i_idx, j_idx = torch.triu(is_foul, diagonal=1).nonzero(as_tuple=True)
            relative_speeds = self.vel_diffs_norm[b_idx, i_idx, j_idx]
            dynamic_foul_magnitude = self.R_foul + self.k_foul_vel_penalty * relative_speeds
            agent_i_p_vel = self.p_vels[b_idx, i_idx]
            pos_rel = self.all_pos[b_idx, j_idx] - self.all_pos[b_idx, i_idx]
            vel_rel_on_pos = torch.einsum("bd,bd->b", agent_i_p_vel, pos_rel)
            i_is_active = vel_rel_on_pos > 0
            active_indices = torch.where(i_is_active, i_idx, j_idx)
            passive_indices = torch.where(i_is_active, j_idx, i_idx)
            active_is_attacker = active_indices < self.n_attackers
            passive_is_attacker = passive_indices < self.n_attackers
            is_friendly_fire = (active_is_attacker == passive_is_attacker)
            foul_rewards = torch.zeros_like(self.terminal_rewards)
            opp_foul_mask = ~is_friendly_fire
            if torch.any(opp_foul_mask):
                opp_b = b_idx[opp_foul_mask]
                opp_active = active_indices[opp_foul_mask]
                opp_passive = passive_indices[opp_foul_mask]
                opp_magnitude = dynamic_foul_magnitude[opp_foul_mask]
                
                teammate_map = torch.tensor([1, 0, 3, 2], device=self.world.device, dtype=torch.long)
                opp_active_teammate = teammate_map[opp_active]
                opp_passive_teammate = teammate_map[opp_passive]
                
                num_opp_fouls = opp_b.shape[0]
                opp_rewards_to_add = torch.zeros(num_opp_fouls, self.n_agents, device=self.world.device)
                opp_row_indices = torch.arange(num_opp_fouls, device=self.world.device)

                opp_rewards_to_add[opp_row_indices, opp_active] = -opp_magnitude
                opp_rewards_to_add[opp_row_indices, opp_passive] = opp_magnitude
                opp_rewards_to_add[opp_row_indices, opp_active_teammate] = -opp_magnitude * self.foul_teammate_factor
                opp_rewards_to_add[opp_row_indices, opp_passive_teammate] = opp_magnitude * self.foul_teammate_factor
                
                foul_rewards.index_add_(0, opp_b, opp_rewards_to_add)

            # Handle friendly-fire fouls
            ff_foul_mask = is_friendly_fire
            if torch.any(ff_foul_mask):
                ff_b = b_idx[ff_foul_mask]
                ff_active = active_indices[ff_foul_mask]
                ff_passive = passive_indices[ff_foul_mask]
                ff_magnitude = dynamic_foul_magnitude[ff_foul_mask]

                num_ff_fouls = ff_b.shape[0]
                ff_rewards_to_add = torch.zeros(num_ff_fouls, self.n_agents, device=self.world.device)
                ff_row_indices = torch.arange(num_ff_fouls, device=self.world.device)

                ff_rewards_to_add[ff_row_indices, ff_active] = -ff_magnitude
                ff_rewards_to_add[ff_row_indices, ff_passive] = -ff_magnitude
                
                foul_rewards.index_add_(0, ff_b, ff_rewards_to_add)

            self.terminal_rewards += foul_rewards
            dones[b_idx] = True

        # -- 条件4: 撞墙超时 (Wall Collision Timeout) --
        # 检查是否有任何智能体的撞墙计数器达到了阈值
        is_wall_timeout = (self.wall_collision_counters >= self.wall_collision_frames) & ~dones.view(-1, 1)

        if torch.any(is_wall_timeout):
            # 找到是哪个环境(b_idx)的哪个智能体(agent_idx)触发了超时
            b_idx, agent_idx = is_wall_timeout.nonzero(as_tuple=True)

            # 对触发超时的智能体施加巨大负奖励
            # 注意：这里我们只惩罚犯错的智能体。如果想惩罚整个队，逻辑会更复杂。
            self.terminal_rewards[b_idx, agent_idx] += self.R_wall_collision_penalty
            
            # 将对应环境的done标志设为True
            dones[b_idx] = True
        return dones
    
    def info(self, agent: Agent):
        """This function computes the info dict for ``agent`` in a vectorized way.

        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape ``(n_envs, info_size)``

        By default this function returns an empty dictionary.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the info for

        Returns:
             Union[torch.Tensor, Dict[str, torch.Tensor]]: the info
        """
        # 返回在当前批次（batch）的这一步中发生的事件数量。
        # 训练脚本可以累积这些值。
        # 我们将值进行扩展，以匹配 (n_envs, 1) 的期望形状。
        return {
            "shots_in_step":self.shots_this_step.clone().float().unsqueeze(-1),
        }

    def done(self):
        """This function computes the done flag for each env in a vectorized way.

        The returned tensor should contain the ``done`` for all envs and should have
        shape ``(n_envs)`` and dtype ``torch.bool``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        By default, this function returns all ``False`` s.

        The scenario can still be done if ``max_steps`` has been set at envirtonment construction.

        Returns:
            torch.Tensor: done tensor of shape ``(self.world.batch_dim)``

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def done(self):
            ...         # retrun done when all agents have battery level lower than a threshold
            ...         return torch.stack([a.battery_level < threshold for a in self.world.agents], dim=-1).all(-1)
        """
        return self.dones

    def reward(self, agent: Agent):
        """This function computes the reward for ``agent`` in a vectorized way.

        The returned tensor should contain the reward for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim)`` and dtype ``torch.float``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the reward for

        Returns:
             torch.Tensor: reward tensor of shape ``(self.world.batch_dim)``

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def reward(self, agent):
            ...         # reward every agent proportionally to distance from first landmark
            ...         rew = -torch.linalg.vector_norm(agent.state.pos - self.world.landmarks[0].state.pos, dim=-1)
            ...         return rew
        """
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
        # 排除自己与自己的距离
        agent_dists[:, agent_idx] = float('inf') 

        # 物理碰撞的距离
        collision_dist = self.agent_radius * 2

        # 根据智能体身份，使用不同的惩罚参数
        if agent == self.a1:
            # --- A1 使用专属参数 ---
            is_too_close = (agent_dists < self.a1_proximity_threshold) & (agent_dists > collision_dist)
            if torch.any(is_too_close):
                # 使用 a1 专属的“软度”参数
                k = self.a1_proximity_penalty_margin
                penetration = torch.logaddexp(
                    torch.tensor(0.0, device=self.world.device),
                    (self.a1_proximity_threshold - agent_dists) / k,
                ) * k
                # 使用 a1 专属的惩罚系数
                proximity_penalty = -self.k_a1_proximity_penalty * penetration
                total_proximity_penalty = (proximity_penalty * is_too_close.float()).sum(dim=1)
                dense_reward += total_proximity_penalty

        elif agent.is_attacker: # a2 会进入这个分支
            # --- A2 使用通用的进攻方参数 ---
            is_too_close = (agent_dists < self.proximity_threshold) & (agent_dists > collision_dist)
            if torch.any(is_too_close):
                k = self.proximity_penalty_margin
                penetration = torch.logaddexp(
                    torch.tensor(0.0, device=self.world.device),
                    (self.proximity_threshold - agent_dists) / k,
                ) * k
                proximity_penalty = -self.k_proximity_penalty * penetration
                total_proximity_penalty = (proximity_penalty * is_too_close.float()).sum(dim=1)
                dense_reward += total_proximity_penalty
        
        else: # 防守方进入这个分支
            # --- 防守方使用独立的防守参数 (逻辑不变) ---
            is_too_close = (agent_dists < self.proximity_threshold) & (agent_dists > collision_dist)
            if torch.any(is_too_close):
                k = self.proximity_penalty_margin
                dist_d_to_spot = torch.linalg.norm(pos - self.spot_center.state.pos, dim=-1)
                is_in_spot_area = (dist_d_to_spot <= self.R_spot)
                
                adjusted_k_def_proximity_penalty = torch.where(
                    is_in_spot_area,
                    self.k_def_proximity_penalty * (1 - self.proximity_penalty_reduction_in_spot),
                    self.k_def_proximity_penalty
                ).unsqueeze(1)
                
                penetration = torch.logaddexp(
                    torch.tensor(0.0, device=self.world.device),
                    (self.proximity_threshold - agent_dists) / k,
                ) * k
                proximity_penalty = -adjusted_k_def_proximity_penalty * penetration
                
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
                # 修正：确保向量从当前智能体指向其他智能体
                pos_diffs_agent_centric = self.all_pos - agent.state.pos.unsqueeze(1)
                raw_action_force_expanded = raw_action_force.unsqueeze(1).expand(-1, self.n_agents, -1)

                pos_diffs_norm = torch.linalg.norm(pos_diffs_agent_centric, dim=-1, keepdim=True) + 1e-6
                proj_vector = (pos_diffs_agent_centric / pos_diffs_norm)
                push_force_magnitude = torch.einsum('bnd,bnd->bn', raw_action_force_expanded, proj_vector)

                # 优化：根据智能体角色选择惩罚系数
                if not agent.is_attacker:
                    penalty_coeff = self.k_def_push_penalty
                else:
                    penalty_coeff = self.k_push_penalty
                push_penalty = -penalty_coeff * torch.clamp(push_force_magnitude, min=0.0)

                # 只在低速碰撞时施加惩罚
                total_push_penalty = (push_penalty * is_low_speed_collision.float()).sum(dim=1)
                dense_reward += total_push_penalty

        # 别人向自己撞过来的时候如果站定有奖励
        # 1. 判断当前智能体是否“站定”
        is_standing_still = torch.linalg.norm(agent.state.vel, dim=1) < self.stand_still_threshold

        # 初始化本帧的奖惩值
        charge_drawing_reward = torch.zeros_like(dense_reward)

        # 2. 遍历所有对手，计算其接近速度
        for i, other_agent in enumerate(self.world.agents):
            if i == agent_idx or agent.is_attacker == other_agent.is_attacker:
                continue # 跳过自己和队友

            # 3. 计算“接近速度”作为奖励/惩罚的梯度
            # 从对手指向当前智能体的相对位置向量
            relative_pos_to_agent = agent.state.pos - other_agent.state.pos
            # 两个智能体之间的距离
            relative_dist = torch.linalg.norm(relative_pos_to_agent, dim=-1)

            # 新增：判断对手是否在“造撞人”的有效范围内
            is_within_charge_range = relative_dist < self.charge_drawing_range

            # 对手速度向量在相对位置向量上的标量投影，即“接近速度”
            # 公式为: proj = (V_opponent · P_relative) / |P_relative|
            dot_product = torch.sum(other_agent.state.vel * relative_pos_to_agent, dim=-1)
            speed_of_approach = dot_product / (relative_dist + 1e-6)

            # 4. 我们只关心正向的接近速度（即对手确实在靠近）。这个值就是我们的梯度。
            # 使用clamp函数过滤掉负值（即远离的情况）。
            approach_gradient = torch.clamp(speed_of_approach, min=0)

            # 5. 根据梯度计算奖惩
            # 奖励条件：[站定] * [对手接近速度梯度] * [奖励系数] * [在范围内]
            # 对手冲得越快 (approach_gradient越大)，站定获得的奖励就越高。
            reward_for_this_opponent = self.k_stand_still_reward * approach_gradient * is_standing_still.float() * is_within_charge_range.float()

            charge_drawing_reward += reward_for_this_opponent

        # 将计算出的总奖惩加入到稠密奖励中
        dense_reward += charge_drawing_reward


        # --- 分角色奖励/惩罚 ---
        if agent == self.a1:
            # A1 (持球进攻者)
            current_dist = torch.linalg.norm(pos - self.spot_center.state.pos, dim=1) # (B,)
            
            # 使用高斯函数引导A1到达目标点
            gaussian_reward = self.gaussian_scale * torch.exp(- (current_dist**2) / (2 * self.gaussian_sigma**2))
            dense_reward += gaussian_reward

            # 新增：A1速度指向spot的奖励，远离spot的惩罚 (主导性质)
            # 计算从A1当前位置指向spot的向量
            vector_to_spot = self.spot_center.state.pos - pos
            # 归一化向量
            vector_to_spot_norm = vector_to_spot / (torch.linalg.norm(vector_to_spot, dim=1, keepdim=True) + 1e-6)
            
            # 计算A1速度向量在指向spot向量上的投影
            speed_projection = torch.sum(agent.state.vel * vector_to_spot_norm, dim=1)
            
            # 奖励：速度指向spot (投影为正，且距离小于一定阈值)
            # 惩罚：速度背离spot (投影为负，或距离大于一定阈值)
            alignment_reward = self.k_a1_speed_spot_reward * speed_projection
            dense_reward += alignment_reward


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
            
            # 2. 条件性控制惩罚 (新核心逻辑):
            # 只有当速度低于我们定义的“低速阈值”时，才惩罚其控制量的大小。
            # 这旨在惩罚“蠕动”或“犹豫不决”的低速微调行为。
            is_at_low_speed = raw_u_norm < self.low_u_threshold
            control_penalty_at_low_speed = -self.k_control_penalty_at_low_speed * raw_u_norm

            # 结合条件：当 [在投篮区内] 且 [处于低速状态] 时，施加控制惩罚
            dense_reward += control_penalty_at_low_speed * is_in_spot.float() * is_at_low_speed.float()

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

            # 惩罚在投篮区域外的犹豫
            hesitation_factor = torch.clamp((1.0 - (velocity_norm / self.hesitate_speed_threshold)), min=0.0)**2
            is_not_charging_shot = ~is_in_spot # 不在篮框外犹豫

            # 2. 当所有条件都满足时，施加惩罚
            hesitation_penalty = -self.k_hesitation_penalty * hesitation_factor * is_not_charging_shot.float()
            dense_reward += hesitation_penalty

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

            # 6.  A2在A1投篮线上的惩罚，防止帮倒忙 (修改版：与距离平滑关联)
            basket_pos = self.basket.state.pos
            shot_vector = basket_pos - p_a1
            shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
            a2_vector = p_a2 - p_a1
            dot_product_shotline = torch.sum(a2_vector * shot_vector, dim=-1)
            proj_len_ratio = dot_product_shotline / shot_vector_norm_sq.squeeze(-1)  # (B,)
            is_between = (proj_len_ratio > 0) & (proj_len_ratio < 1)
            projection = proj_len_ratio.unsqueeze(-1) * shot_vector
            dist_perp_sq = torch.sum((a2_vector - projection)**2, dim=-1)
            dist_perp = torch.sqrt(dist_perp_sq)  # 实际垂直距离

            # 新增：距离A1的距离因素 (越近惩罚越强)
            dist_a2_to_a1 = torch.linalg.norm(a2_vector, dim=-1)
            proximity_factor = torch.exp(-dist_a2_to_a1.pow(2) / (2 * (2 * self.agent_radius)**2))

            # 修改：使用距离和高斯函数平滑计算惩罚 (无需硬阈值)
            line_block_factor = is_between.float() * torch.exp(-dist_perp_sq / (2 * (0.5 * self.agent_radius)**2))
            dense_reward -= self.k_a2_shot_line_penalty * line_block_factor * proximity_factor

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

            # 4. 投篮区域控制奖励 (只在防守半场生效)
            current_dist_a1_to_spot = torch.linalg.norm(self.a1.state.pos - self.spot_center.state.pos, dim=1)
            
            # 奖励防守方将A1推离投篮区域
            spot_control_reward = self.k_spot_control * (current_dist_a1_to_spot - self.prev_dist_a1_to_spot)

            dense_reward += (spot_control_reward) * in_defensive_half.float()

            # 5. 防守方高斯引导奖励 (只在防守半场生效)
            # 鼓励防守方靠近投篮点
            dist_d_to_spot = torch.linalg.norm(p_d - self.spot_center.state.pos, dim=-1)
            def_gaussian_reward = self.k_def_gaussian_spot * torch.exp(- (dist_d_to_spot**2) / (2 * self.def_gaussian_spot_sigma**2))
            dense_reward += def_gaussian_reward * in_defensive_half.float()

        # --- 新增时变惩罚逻辑 (对所有进攻方生效) ---
        if agent.is_attacker:
            # 1. 计算已经流逝的时间
            elapsed_time = self.t_limit - self.t_remaining.squeeze(-1)

            # 2. 只有在宽限期之后才开始计算惩罚
            apply_penalty_mask = elapsed_time > self.time_penalty_grace_period

            if torch.any(apply_penalty_mask):
                # 3. 计算惩罚因子，我们使用二次方关系，让惩罚后期增长更快
                # 这样，在比赛快结束时，紧迫感会急剧增强
                time_factor = (elapsed_time - self.time_penalty_grace_period)**2
                
                # 4. 计算最终的时间惩罚值
                time_penalty = -self.k_time_penalty * time_factor

                # 5. 将惩罚施加给在宽限期之后的环境实例
                dense_reward[apply_penalty_mask] += time_penalty[apply_penalty_mask]

            
        # 返回稠密奖励和回合结束时的稀疏奖励之和
        rew = self.dense_reward_factor * dense_reward + self.terminal_rewards[:, agent_idx]

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
        """This function computes the observations for ``agent`` in a vectorized way.

        The returned tensor should contain the observations for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim, n_agent_obs)``, or be a dict with leaves following that shape.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the observations for

        Returns:
             Union[torch.Tensor, Dict[str, torch.Tensor]]: the observation

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def observation(self, agent):
            ...         # get positions of all landmarks in this agent's reference frame
            ...         landmark_rel_poses = []
            ...         for landmark in self.world.landmarks:
            ...             landmark_rel_poses.append(landmark.state.pos - agent.state.pos)
            ...         return torch.cat([agent.state.pos, agent.state.vel, *landmark_rel_poses], dim=-1)

        You can also return observations in a dictionary

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def observation(self, agent):
            ...         return {"pos": agent.state.pos, "vel": agent.state.vel}

        """
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
        """
        This function facilitates additional user/scenario-level rendering for a specific environment index.

        The returned list is a list of geometries. It is the user's responsibility to set attributes such as color,
        position and rotation.

        Args:
            env_index (int, optional): index of the environment to render. Defaults to ``0``.

        Returns: A list of geometries to render for the current time step.

        Examples:
            >>> from vmas.simulator.utils import Color
            >>> from vmas.simulator.scenario import BaseScenario
            >>> class Scenario(BaseScenario):
            >>>     def extra_render(self, env_index):
            >>>         from vmas.simulator import rendering
            >>>         color = Color.BLACK.value
            >>>         line = rendering.Line(
            ...            (self.world.agents[0].state.pos[env_index]),
            ...            (self.world.agents[1].state.pos[env_index]),
            ...            width=1,
            ...         )
            >>>         xform = rendering.Transform()
            >>>         line.add_attr(xform)
            >>>         line.set_color(*color)
            >>>         return [line]
        """
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