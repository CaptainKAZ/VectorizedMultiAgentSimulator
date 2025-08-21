# 导入必要的库
from traceback import print_tb
import torch
from typing import Dict, Tuple
from vmas import render_interactively
from vmas.simulator.core import World, Agent, Landmark, Sphere, Line
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController
import matplotlib
matplotlib.use('Agg') # 对于非交互式绘图至关重要
import matplotlib.pyplot as plt
import io
import pyglet
import numpy as np

from vmas.scenarios.layup_jit import calculate_rewards_and_dones_jit
import time
from functools import partial, update_wrapper

# class timer:
#     """
#     一个通用的、用于计算函数或方法平均运行时间的类装饰器。
#     它实现了描述器协议，可以正确处理类实例的 `self` 参数。
#     """
#     def __init__(self, func):
#         update_wrapper(self, func)
#         self.func = func
#         self.run_times = []
#         self.last_print_time = time.perf_counter()

#     def __get__(self, instance, owner):
#         """实现描述器协议，使其能作为方法装饰器。"""
#         if instance is None:
#             # 如果通过类来访问，例如 MyClass.my_method，则返回装饰器实例本身
#             return self
#         # 如果通过实例来访问，例如 my_instance.my_method
#         # 使用 partial 将实例(instance)和 __call__ 方法绑定在一起
#         # 这就模拟了 Python 的绑定方法机制
#         return partial(self.__call__, instance)

#     def __call__(self, *args, **kwargs):
#         # 注意：如果作为方法装饰器，由于 __get__ 的作用，
#         # 这里的第一个参数 `*args[0]` 将会是类的实例 `self`。
        
#         # 1. 执行函数并计时
#         start_time = time.perf_counter()
#         result = self.func(*args, **kwargs)
#         end_time = time.perf_counter()
        
#         # 2. 存储本次运行时间
#         self.run_times.append(end_time - start_time)
        
#         # 3. 检查是否需要打印平均时间
#         current_time = time.perf_counter()
#         if current_time - self.last_print_time >= 1.0:
#             num_runs = len(self.run_times)
#             avg_time = sum(self.run_times) / num_runs
            
#             # 打印信息
#             # print(f"'{self.func.__name__}' {num_runs} {avg_time * 1e6:.3f} μs")
            
#             # 4. 重置状态
#             self.run_times = []
#             self.last_print_time = current_time
            
#         return result

class Scenario(BaseScenario):
    """
    "飞身上篮"简化版2v2投篮强化学习环境 (已优化和修复版本)
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.viewer_zoom = 3.0
        self.viewer_size = [1400,700]
        # ----------------- 超参数设定 (Hyperparameters) -----------------
        # 创建一个字典来存储所有超参数，以便统一传递给JIT编译的函数
        self.h_params = {}

        # =================================================================================
        # 1. 基础物理与环境设定 (Basic Physics & Environment)
        # =================================================================================
        # --- 场地属性 ---
        self.h_params["W"] = kwargs.get("W", 8.0)  # 场地宽度 (x-axis)
        self.h_params["L"] = kwargs.get("L", 15.0) # 场地长度 (y-axis)
        self.h_params["R_spot"] = kwargs.get("R_spot", 1.2) # 篮下可投篮的圆形区域半径

        # --- 游戏规则 ---
        self.h_params["t_limit"] = kwargs.get("t_limit", 15.0) # 每回合最大时长（秒）
        self.dt = kwargs.get("dt", 0.1) # 物理仿真的时间步长
        self.spawn_area_depth = kwargs.get("spawn_area_depth", 1.0) # 防守方和a2生成位置的宽度
        self.start_delay_frames = kwargs.get("start_delay_frames", 10) # 回合开始时，智能体需要等待的帧数，期间不响应动作

        # --- 智能体物理属性 ---
        self.h_params["agent_radius"] = kwargs.get("agent_radius", 0.3) # 智能体半径，用于碰撞检测
        self.h_params["a_max"] = kwargs.get("a_max", 3.0) # 智能体的最大加速度
        self.h_params["v_max"] = kwargs.get("v_max", 5.0) # 智能体的最大速度


        # =================================================================================
        # 2. 回合终止条件 (Episode Termination Conditions)
        # =================================================================================
        # --- 2.1 投篮判定 ---
        self.h_params["v_shot_threshold"] = kwargs.get("v_shot_threshold", 0.1) # 触发投篮所允许的最大速度
        self.h_params["a_shot_threshold"] = kwargs.get("a_shot_threshold", 0.4)  # 触发投篮所允许的最大动作指令模长
        self.h_params["shot_still_frames"] = kwargs.get("shot_still_frames", 10)   # 触发投篮需要在投篮区内保持静止的帧数

        # --- 2.2 犯规判定 ---
        self.h_params["v_foul_threshold"] = kwargs.get("v_foul_threshold", 0.4)        # 判定为碰撞犯规的最小相对速度
        self.h_params["wall_collision_frames"] = kwargs.get("wall_collision_frames", 20.0) # 持续撞墙导致回合结束的帧数阈值
        self.h_params["max_time_over_midline"] = kwargs.get("max_time_over_midline", 20.0) # 防守方允许越过中线的最大帧数

        # --- 2.3 胜负判定 ---
        self.h_params["win_condition_block_threshold"] = kwargs.get("win_condition_block_threshold", 0.5) # 判定投篮被成功封盖的封盖因子阈值，大于此值则投篮失败


        # =================================================================================
        # 3. 终局奖励设定 (Terminal Rewards)
        # =================================================================================
        # --- 3.1 投篮成功 ---
        self.h_params["max_score"] = kwargs.get("max_score", 6000.0)    # 投篮得分的基础分，离篮筐越近得分越高
        self.h_params["shoot_score"] = kwargs.get("shoot_score", 4000.0)  # 成功出手投篮的固定额外奖励
        self.h_params["k_time_bonus"] = kwargs.get("k_time_bonus", 4000.0) # 投篮时间奖励系数，剩余时间越多奖励越高
        self.h_params["k_spacing_bonus"] = kwargs.get("k_spacing_bonus", 1000.0) # A1投篮时，与防守方平均距离的奖励系数
        self.h_params['k_shot_stillness_vel_bonus'] = kwargs.get("k_shot_stillness_vel_bonus", 1000.0) # A1投篮时速度够慢的额外奖励
        self.h_params['k_shot_stillness_act_bonus'] = kwargs.get("k_shot_stillness_act_bonus", 1000.0) # A1投篮时动作指令够小的额外奖励
        self.h_params["k_a2_screen_bonus"] = kwargs.get("k_a2_screen_bonus", 2000.0) # A1投篮时，A2成功掩护的额外奖励
        self.h_params["a2_screen_sigma"] = kwargs.get("a2_screen_sigma", 4 * self.h_params["agent_radius"]) # A2掩护奖励高斯函数的标准差

        # --- 3.2 进攻超时 ---
        self.h_params["defender_timeout_reward"] = kwargs.get("defender_timeout_reward", 9000.0) # 进攻超时，防守方获得的奖励
        self.h_params["attacker_timeout_reward_max"] = kwargs.get("attacker_timeout_reward_max", 2000) # 进攻超时，进攻方惩罚/奖励的绝对值上限
        self.h_params["k_timeout_move_vel_penalty"] = kwargs.get("k_timeout_move_vel_penalty", 200.0) # 超时瞬间，A1因速度过大受到的惩罚系数
        self.h_params["k_timeout_move_act_penalty"] = kwargs.get("k_timeout_move_act_penalty", 200.0) # 超时瞬间，A1因动作指令过大受到的惩罚系数
        self.h_params["k_timeout_dist_reward_factor"] = kwargs.get("k_timeout_dist_reward_factor", 100.0) # 超时瞬间，A1在圈外时，根据距离远近受到的惩罚系数
        self.h_params["attacker_timeout_base_reward_out_spot"] = kwargs.get("attacker_timeout_base_reward_out_spot", -100.0) # 超时瞬间，A1在圈外的基础惩罚
        self.h_params["attacker_timeout_reward_in_spot"] = kwargs.get("attacker_timeout_reward_in_spot", 100.0)    # 超时瞬间，A1在圈内的基础奖励/惩罚

        # --- 3.3 犯规 ---
        self.h_params["R_foul"] = kwargs.get("R_foul", 4000.0) # 碰撞犯规的基础奖励/惩罚值
        self.h_params["k_foul_vel_penalty"] = kwargs.get("k_foul_vel_penalty", 100.0) # 碰撞犯规时，根据相对速度大小调整惩罚的系数
        self.h_params["foul_teammate_factor"] = kwargs.get("foul_teammate_factor", 0.05) # 犯规发生时，被犯规方队友获得的奖励比例
        self.h_params["R_wall_collision_penalty"] = kwargs.get("R_wall_collision_penalty", -11000.0) # 因持续撞墙导致回合结束的惩罚
        self.h_params["R_midline_foul"] = kwargs.get("R_midline_foul", 10000.0) # 防守方因持续越线导致回合结束的惩罚

        # --- 3.4 投篮失败 (防守方终局奖励) ---
        self.h_params["k_def_block_reward"] = kwargs.get("k_def_block_reward", 3000.0) # 防守方因封盖贡献获得的奖励系数
        self.h_params["k_def_force_reward"] = kwargs.get("k_def_force_reward", 2000.0) # 防守方因迫使A1远离篮筐投篮获得的奖励系数
        self.h_params["k_def_pos_reward"] = kwargs.get("k_def_pos_reward", 100.0)   # 防守方因占据理想防守位置获得的奖励系数
        self.h_params["k_def_area_reward"] = kwargs.get("k_def_area_reward", 150.0)  # 防守方因控制投篮区域获得的奖励系数
        self.h_params["k_def_shot_penalty"] = kwargs.get("k_def_shot_penalty", 300.0)  # 对方投篮时，防守方受到的基础小额惩罚（鼓励积极防守）


        # =================================================================================
        # 4. 稠密奖励与行为塑造 (Dense Rewards & Behavior Shaping)
        # =================================================================================

        # --- 4.1 通用项 (General for All Agents) ---
        self.dense_reward_factor = kwargs.get("dense_reward_factor", 0.1) # 稠密奖励整体缩放系数
        self.h_params["oob_penalty"] = kwargs.get("oob_penalty", -3000.0) # 出界惩罚系数
        self.h_params["oob_margin"] = kwargs.get("oob_margin", 0.05) # 出界惩罚的平滑边界宽度
        self.h_params["k_u_penalty_general"] = kwargs.get("k_u_penalty_general", 0.1) # 动作指令大小的基础惩罚系数
        self.h_params["k_action_access_max_penalty"] = kwargs.get("k_action_access_max_penalty", 20) # 动作指令超过阈值时的额外惩罚系数
        self.h_params["k_action_access_max_threshold"] = kwargs.get("k_action_access_max_threshold", 0.95) # 触发额外动作惩罚的阈值（v_max的百分比）
        self.h_params["k_brake_usage_penalty"] = kwargs.get("k_brake_usage_penalty", 0.1) # 使用刹车的惩罚系数
        self.h_params["k_conflicting_action_penalty"] = kwargs.get("k_conflicting_action_penalty", 10) # 同时输出方向和刹车指令的矛盾惩罚系数
        self.h_params["k_excess_acceleration_penalty"] = kwargs.get("k_excess_acceleration_penalty", 0.001) # 请求加速度超过物理极限的惩罚系数
        self.h_params["k_action_jerk_penalty"] = kwargs.get("k_action_jerk_penalty", 0.) # 动作指令变化率（Jerk）的惩罚系数，鼓励平滑动作
        self.h_params["k_coll_active"] = kwargs.get("k_coll_active", 5.0) # 作为主动碰撞方受到的惩罚系数
        self.h_params["k_coll_passive"] = kwargs.get("k_coll_passive", 0.1) # 作为被动碰撞方受到的惩罚系数
        self.h_params["proximity_threshold"] = kwargs.get("proximity_threshold", self.h_params["agent_radius"] * 2.3) # 智能体间的安全距离，小于此距离将触发近距离惩罚
        self.h_params["proximity_penalty_margin"] = kwargs.get("proximity_penalty_margin", 0.10) # 近距离惩罚的平滑边界宽度
        self.h_params["k_proximity_penalty"] = kwargs.get("k_proximity_penalty", 60.0) # 通用近距离惩罚系数
        self.h_params["low_velocity_threshold"] = kwargs.get("low_velocity_threshold", self.h_params['v_foul_threshold']) # 区分高速碰撞和低速推挤的阈值
        self.h_params["k_push_penalty"] = kwargs.get("k_push_penalty", 120.0) # 进攻方在低速碰撞中推挤对方的惩罚系数
        self.h_params["stand_still_threshold"] = kwargs.get("stand_still_threshold", self.h_params['v_foul_threshold']) # 判定为“站定”状态的最大速度
        self.h_params["k_stand_still_reward"] = kwargs.get("k_stand_still_reward", 10.0) # 站定不动时，对正在冲过来的对手“造犯规”的奖励系数
        self.h_params["charge_drawing_range"] = kwargs.get("charge_drawing_range", self.h_params["agent_radius"] * 6.0) # “造犯规”的有效距离

        # --- 4.2 进攻方 - A1 (持球人) ---
        self.h_params["k_a1_speed_spot_reward"] = kwargs.get("k_a1_speed_spot_reward", 1500.0) # 吸引A1到投篮点的路程总奖励
        self.h_params["gaussian_scale"] = kwargs.get("gaussian_scale", 300.0) # 吸引A1到投篮点的高斯奖励的峰值大小
        self.h_params["gaussian_sigma"] = kwargs.get("gaussian_sigma", 0.5 * self.h_params["R_spot"]) # 高斯奖励的宽度，决定了吸引力的范围
        self.h_params["k_a1_in_spot_reward"] = kwargs.get("k_a1_in_spot_reward", 3.0) # A1在投篮区域内时，每步获得的持续性奖励系数
        self.h_params["k_a1_ready_to_shoot_reward"] = kwargs.get("k_a1_ready_to_shoot_reward", 50.0) # A1处于“准备投篮”状态时的奖励系数
        self.h_params["k_a1_velocity_stillness_reward"] = kwargs.get("k_a1_velocity_stillness_reward", 10.0) # 在投篮区内，A1速度越慢奖励越高的系数
        self.h_params["velocity_stillness_sigma"] = kwargs.get("velocity_stillness_sigma", 0.4) # 速度静止奖励高斯函数的标准差
        self.h_params["k_a1_action_stillness_reward"] = kwargs.get("k_a1_action_stillness_reward", 10) # 在投篮区内，A1动作指令越小奖励越高的系数
        self.h_params["k_a1_brake_in_spot_reward"] = kwargs.get("k_a1_brake_in_spot_reward", 20) # 在投篮区内，A1刹车奖励
        self.h_params["action_stillness_sigma"] = kwargs.get("action_stillness_sigma", 0.3) # 动作静止奖励高斯函数的标准差
        self.h_params["low_u_threshold"] = kwargs.get("low_u_threshold", 0.9) # 判定A1有“停止意图”的动作指令模长阈值
        self.h_params["k_a1_separation_reward"] = kwargs.get("k_a1_separation_reward", 60.0) # A1被封锁时，奖励其向远离防守者的方向移动
        self.h_params["k_a1_tangential_reward"] = kwargs.get("k_a1_tangential_reward", 120.0) # A1在受压迫时，奖励其横向移动以摆脱防守
        self.h_params["a1_tangential_pressure_sigma"] = kwargs.get("a1_tangential_pressure_sigma", self.h_params["agent_radius"] * 6) # 计算横向移动奖励时，防守压力距离衰减的标准差
        self.h_params["k_a1_blocked_penalty"] = kwargs.get("k_a1_blocked_penalty", -70.0) # A1投篮路线被封锁时的惩罚系数
        self.h_params["hesitate_speed_threshold"] = kwargs.get("hesitate_speed_threshold", 1.5) # 在非投篮区，低于此速度被认为是“犹豫”，将受惩罚
        self.h_params["k_hesitation_penalty"] = kwargs.get("k_hesitation_penalty", 40) # A1犹豫不决的惩罚系数
        self.h_params["a1_proximity_threshold"] = kwargs.get("a1_proximity_threshold", self.h_params["agent_radius"] * 2.5) # 专门为A1设定的近距离惩罚触发距离
        self.h_params["a1_proximity_penalty_margin"] = kwargs.get("a1_proximity_penalty_margin", 0.01) # A1近距离惩罚的平滑边界宽度
        self.h_params["k_a1_proximity_penalty"] = kwargs.get("k_a1_proximity_penalty", 60) # A1的近距离惩罚系数

        # --- 4.3 进攻方 - A2 (无球人) ---
        self.h_params["k_ideal_screen_pos"] = kwargs.get("k_ideal_screen_pos", 60.0) # A2移动到最佳掩护位置的奖励系数
        self.h_params["k_a2_interference_reward"] = kwargs.get("k_a2_interference_reward", 40.0) # A2靠近并干扰防守者的奖励系数
        self.h_params["k_repulsion_reward"] = kwargs.get("k_repulsion_reward", 60.0) # A2迫使防守者远离A1的“排斥”奖励系数
        self.h_params["repulsion_proximity_threshold"] = kwargs.get("repulsion_proximity_threshold", self.h_params["R_spot"]) # 触发排斥奖励时，A2需要离防守者足够近的距离
        self.h_params["k_a2_shot_line_penalty"] = kwargs.get("k_a2_shot_line_penalty", 30) # A2阻挡A1投篮路线的惩罚系数
        self.h_params["screen_pos_offset"] = kwargs.get("screen_pos_offset", self.h_params["agent_radius"] * 3) # 定义“理想掩护位置”在防守者身后的距离
        self.h_params["screen_pos_sigma"] = kwargs.get("screen_pos_sigma", self.h_params["R_spot"]) # 掩护位置奖励高斯函数的标准差
        self.h_params["k_screen_gate"] = kwargs.get("k_screen_gate", 7.0) # A2掩护位置门控的Sigmoid函数斜率，判断A2是否在A1和防守者之间
        self.h_params["screen_spacing_gate_k"] = kwargs.get("screen_spacing_gate_k", 7.0) # A2掩护间距门控的Sigmoid函数斜率，判断A2是否离防守者比A1更近

        # --- 4.4 防守方 ---
        self.h_params["k_positioning"] = kwargs.get("k_positioning", 90.0) # 防守方占据理想防守位置（A1与篮筐之间）的奖励系数
        self.h_params["def_pos_offset"] = kwargs.get("def_pos_offset", self.h_params["agent_radius"] * 2.5) # 定义“理想防守位置”在A1身后的距离
        self.h_params["def_pos_sigma"] = kwargs.get("def_pos_sigma", 3 * self.h_params["agent_radius"]) # 防守位置奖励高斯函数的标准差
        self.h_params["k_def_pressure_reward"] = kwargs.get("k_def_pressure_reward", 30.0) # 防守方靠近A1施加压力的奖励系数
        self.h_params["def_pressure_range"] = kwargs.get("def_pressure_range", 6 * self.h_params["agent_radius"]) # 施加压力的有效最远距离
        self.h_params["k_spot_control_reward"] = kwargs.get("k_spot_control_reward", 30.0) # 防守方成功阻止A1向篮筐移动的奖励系数
        self.h_params["def_guard_threshold"] = kwargs.get("def_guard_threshold", self.h_params["agent_radius"] * 6.0) # 判定防守方正在“盯防”A1的最大距离
        self.h_params["k_def_gaussian_spot"] = kwargs.get("k_def_gaussian_spot", 30) # 吸引防守方占据投篮点中心区域的高斯奖励系数
        self.h_params["def_gaussian_spot_sigma"] = kwargs.get("def_gaussian_spot_sigma", 1.0 * self.h_params["R_spot"]) # 防守方高斯奖励的宽度
        self.h_params["k_def_a1_penetration_penalty"] = kwargs.get("k_def_a1_penetration_penalty", 5.0) # A1突破深入时，防守方受到的惩罚系数
        self.h_params["k_overextend_penalty"] = kwargs.get("k_overextend_penalty", 240.0) # 防守方越过中线太远的惩罚系数
        self.h_params["k_def_proximity_penalty"] = kwargs.get("k_def_proximity_penalty", 60.0) # 防守方的近距离惩罚系数
        self.h_params["proximity_penalty_reduction_in_spot"] = kwargs.get("proximity_penalty_reduction_in_spot", 0.2) # 在投篮区内，对防守方近距离惩罚的减免比例
        self.h_params["k_def_push_penalty"] = kwargs.get("k_def_push_penalty", 120.0) # 防守方在低速碰撞中推挤对方的惩罚系数

        # --- 4.5 时间压力 ---
        self.h_params["time_penalty_grace_period"] = kwargs.get("time_penalty_grace_period", 8) # 回合开始后，免除时间惩罚的宽限期（秒）
        self.h_params["k_attacker_time_penalty"] = kwargs.get("k_attacker_time_penalty", 2) # 宽限期后，若A1未进入投篮区，进攻方将受到时间惩罚
        self.h_params["k_defender_time_bonus"] = kwargs.get("k_defender_time_bonus", 2.0)   # 宽限期后，防守方将获得持续的时间奖励

        # --- 4.6 封盖相关参数 ---
        self.h_params["def_proximity_threshold"] = kwargs.get("def_proximity_threshold", 2.5*self.h_params["agent_radius"]) # 计算封盖时，判断防守者是否离A1足够近的距离阈值
        self.h_params["block_sigma"] = kwargs.get("block_sigma", 0.30) # 封盖因子高斯函数的标准差，影响封盖判定的严格程度
        self.h_params["block_gate_k"] = kwargs.get("block_gate_k", 25.0) # 封盖软门控Sigmoid函数的斜率
        
        # ----------------- 环境构建 (World Setup) -----------------
        self.max_steps = int(self.h_params["t_limit"] / self.dt)
        self.n_agents = 4
        self.n_attackers = 2
        self.n_defenders = 2

        world = World(batch_dim, device, dt=self.dt, substeps=4,
                      x_semidim=self.h_params["W"] / 2, y_semidim=self.h_params["L"] / 2)

        for i in range(self.n_agents):
            is_attacker = i < self.n_attackers
            team_name = "attacker" if is_attacker else "defender"
            agent_id = i + 1 if is_attacker else i - self.n_attackers + 1
            agent = Agent(
                name=f"{team_name}_{agent_id}",
                collide=True,
                movable=True,
                rotatable=False,
                u_range=self.h_params["v_max"],
                drag=0.01,
                shape=Sphere(radius=self.h_params["agent_radius"]),
                dynamics=Holonomic(),
                render_action=True,
                color=Color.RED if is_attacker and agent_id == 1 else Color.BLUE if not is_attacker else Color.PINK,
                action_size=3
            )
            agent.is_attacker = is_attacker
            agent.controller = VelocityController(agent, world, [6,0,0.01], "parallel")
            world.add_agent(agent)

        self.attackers = world.agents[:self.n_attackers]
        self.defenders = world.agents[self.n_attackers:]
        self.a1 = self.attackers[0]
        self.a2 = self.attackers[1]

        self.basket = Landmark(name="basket", collide=False, shape=Sphere(radius=0.1), color=Color.ORANGE)
        self.spot_center = Landmark(name="spot_center", collide=False, shape=Sphere(radius=0.05), color=Color.GREEN)
        self.shooting_area_vis = Landmark(name="shooting_area_vis", collide=False, shape=Sphere(radius=self.h_params["R_spot"]), color=Color.LIGHT_GREEN)
        center_line = Landmark(name="center_line", collide=False, shape=Line(length=self.h_params["W"]), color=Color.GRAY)
        world.add_landmark(center_line)
        world.add_landmark(self.basket)
        world.add_landmark(self.spot_center)
        world.add_landmark(self.shooting_area_vis)

        # 初始化内部状态变量
        self.t_remaining = torch.zeros(batch_dim, 1, device=device)
        self.step_dense_rewards = torch.zeros(batch_dim, self.n_agents, device=device) # 用于存储当前步的稠密奖励
        self.terminal_rewards = torch.zeros(batch_dim, self.n_agents, device=device)   # 用于存储终局奖励
        self.dones = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.p_vels = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.raw_actions = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.raw_breaks = torch.zeros((batch_dim, self.n_agents), device=device)
        self.delay_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.a1_still_frames_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.wall_collision_counters = torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32)
        self.defender_over_midline_counter = torch.zeros((batch_dim, self.n_defenders), device=device, dtype=torch.int32)
        self.win_this_step = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.dones_this_step = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.requested_accelerations = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.p_raw_actions = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.termination_reason_code = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.a1_normalized_speed_k = torch.zeros(batch_dim, device=device)


        # self.jitted_reward_calculator = torch.compile(calculate_rewards_and_dones_jit, mode="max-autotune")
        self.jitted_reward_calculator = calculate_rewards_and_dones_jit

        self.reward_hist = {}

        self.plot_artists = []
        for i, agent in enumerate(world.agents):
            fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
            fig.tight_layout(pad=1)
            line, = ax.plot([], [], 'r-') 
            ax.set_title(f"Agent {agent.name}", fontsize=6)
            artist_dict = {'fig': fig, 'ax': ax, 'line': line}
            self.plot_artists.append(artist_dict)

        return world

    # @timer
    def reset_world_at(self, env_index: int | None = None):
        """
        根据指定的生成规则，使用高效的并行计算和局部抖动网格重置环境。
        - a1: 位置固定。
        - a2: 在靠近中线的己方条带内随机生成。
        - d1, d2: 在靠近中线的己方条带内使用抖动网格生成，以避免碰撞。
        此方法通过构造保证了所有智能体无初始碰撞且满足边界条件。
        """
        if env_index is None:
            batch_range = slice(None)
            batch_dim = self.world.batch_dim
            self.reward_hist.clear()
        else:
            batch_range = env_index
            batch_dim = 1 if isinstance(env_index, int) else len(env_index)
            if isinstance(env_index, int):
                if env_index in self.reward_hist:
                    del self.reward_hist[env_index]
            else: # env_index is a list or tensor
                for idx in env_index:
                    if idx in self.reward_hist:
                        del self.reward_hist[idx]
        
        # 重置时间和内部状态
        self.t_remaining[batch_range] = self.h_params["t_limit"]
        self.terminal_rewards[batch_range] = 0.0
        self.p_vels[batch_range] = 0.0
        self.delay_counter[batch_range] = self.start_delay_frames
        self.a1_still_frames_counter[batch_range] = 0
        self.wall_collision_counters[batch_range] = 0
        self.defender_over_midline_counter[batch_range] = 0
        self.dones[batch_range] = 0
        self.p_raw_actions[batch_range] = 0.0
        self.termination_reason_code[batch_range] = 0

        # 随机化篮筐和投篮点位置
        basket_pos = torch.zeros(batch_dim, 2, device=self.world.device)
        basket_pos[:, 1] = self.h_params["L"] / 2 - 0.6
        self.basket.set_pos(basket_pos, batch_index=env_index)

        spot_x = (torch.rand(batch_dim, 1, device=self.world.device) - 0.5) * (self.h_params["W"]-self.h_params["R_spot"])
        spot_y = torch.rand(batch_dim, 1, device=self.world.device) * (self.h_params["L"] / 4) + (self.h_params["R_spot"])
        spot_pos = torch.cat([spot_x, spot_y], dim=1)
        self.spot_center.set_pos(spot_pos, batch_index=env_index)
        self.shooting_area_vis.set_pos(spot_pos, batch_index=env_index)

        # ---------- 高效并行化智能体放置 ----------
        # 1. 获取参数
        W, L = self.h_params["W"], self.h_params["L"]
        agent_radius = self.h_params["agent_radius"]
        spawn_area_depth = self.spawn_area_depth
        n_defenders = self.n_defenders
        device = self.world.device

        # --- 攻击方1 (a1): 固定位置 ---
        # 固定在场地左下角，并留出自身半径的边界
        pos_a1_x = -W / 2 + agent_radius * 2
        pos_a1_y = -L / 2 + agent_radius * 2
        pos_a1 = torch.tensor([[pos_a1_x, pos_a1_y]], device=device, dtype=torch.float32).expand(batch_dim, -1)

        # --- 攻击方2 (a2): 在己方条带内随机生成 ---
        # 生成区域: X轴在[-W/2, W/2]内，Y轴在[-spawn_area_depth, 0]内
        # 为避免在边缘生成，所有计算均考虑agent_radius的边界
        valid_width = W - 2 * agent_radius
        valid_depth = spawn_area_depth - agent_radius # 避免生成在y=0中线上

        pos_a2_x = (torch.rand(batch_dim, 1, device=device) - 0.5) * valid_width
        # 在Y轴 [-spawn_area_depth, -agent_radius] 区间内生成
        pos_a2_y = -agent_radius - torch.rand(batch_dim, 1, device=device) * valid_depth
        pos_a2 = torch.cat([pos_a2_x, pos_a2_y], dim=1)

        # --- 防守方 (d1, d2): 在己方条带内使用局部抖动网格 ---
        # 生成区域: X轴在[-W/2, W/2]内，Y轴在[0, spawn_area_depth]内
        # 使用1x2的网格放置2个防守方，以从结构上避免碰撞
        
        # 定义网格单元尺寸
        def_cell_w = valid_width / n_defenders
        
        # 计算抖动范围
        max_jitter_x = max(0.0, (def_cell_w / 2) - agent_radius)
        max_jitter_y = max(0.0, valid_depth / 2)

        # 生成随机抖动值
        def_jitter = (torch.rand(batch_dim, n_defenders, 2, device=device) - 0.5)
        def_jitter[:, :, 0] *= 2 * max_jitter_x
        def_jitter[:, :, 1] *= 2 * max_jitter_y

        # 计算网格单元中心点（基础位置），并随机分配智能体到单元
        def_indices = torch.rand(batch_dim, n_defenders, device=device).argsort(dim=1)
        def_base_x = -valid_width/2 + def_cell_w/2 + def_indices * def_cell_w
        def_base_y = torch.full_like(def_base_x, agent_radius + valid_depth / 2)
        def_base_pos = torch.stack([def_base_x, def_base_y], dim=-1)

        # 计算防守方最终位置
        pos_def = def_base_pos + def_jitter

        # --- 组合所有智能体位置 ---
        agent_positions = torch.cat([pos_a1.unsqueeze(1), pos_a2.unsqueeze(1), pos_def], dim=1)
        
        # 设置智能体状态
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(agent_positions[:, i, :], batch_index=env_index)
            agent.set_vel(torch.zeros(batch_dim, 2, device=self.world.device), batch_index=env_index)

        # 1. 计算初始距离
        initial_dist = torch.linalg.norm(pos_a1 - spot_pos, dim=1)
        # 2. 计算本回合专用的、归一化后的速度奖励系数 k' = k / D_initial
        normalized_k = self.h_params['k_a1_speed_spot_reward'] / (initial_dist + 1e-6)
        # 3. 将这个计算好的、恒定的系数存储起来
        self.a1_normalized_speed_k[batch_range] = normalized_k


    # @timer
    def process_action(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        
        # 1. 分离速度和刹车信号 (刹车信号范围现在是 [-5, 5])
        target_vel = agent.action.u[:, :2]
        brake_signal = agent.action.u[:, 2]

        # 2. 实现刹车逻辑，【关键修改点】
        # 当刹车信号 > 0 时，我们判定AI想要刹车
        is_braking = brake_signal > 0
        final_target_vel = torch.where(
            is_braking.unsqueeze(-1),
            torch.zeros_like(target_vel),
            target_vel
        )

        # 3. 保存原始动作
        self.raw_actions[:, agent_idx, :] = target_vel.clone()
        self.raw_breaks[:, agent_idx] = brake_signal.clone()

        # 4. 处理开局延迟
        if agent == self.a1:
            is_delayed = self.delay_counter > 0
            final_target_vel[is_delayed] = 0.0

        # 5. 实现动作死区
        action_norm = torch.linalg.vector_norm(final_target_vel, dim=1)
        final_target_vel[action_norm < 0.1] = 0.0
        
        # 6. 后续所有操作都基于我们最终计算出的 final_target_vel
        clamped_vel = TorchUtils.clamp_with_norm(final_target_vel, agent.u_range)
        
        requested_a = (clamped_vel - agent.state.vel) / self.world.dt
        self.requested_accelerations[:, agent_idx, :] = requested_a
        achievable_a = TorchUtils.clamp_with_norm(requested_a, self.h_params["a_max"])

        agent.action.u = agent.state.vel + achievable_a * self.world.dt
        
        agent.controller.process_force()

    # @timer
    def pre_step(self):
        """
        在每个物理步长开始前执行。
        这是执行JIT函数的最佳位置，因为它在所有智能体动作处理后、物理引擎步进前运行。
        """
        self.win_this_step.zero_()
        self.t_remaining -= self.world.dt
        self.delay_counter = torch.clamp(self.delay_counter - 1, min=0)

        # 1. 收集所有智能体的状态张量
        self.all_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        self.all_vel = torch.stack([a.state.vel for a in self.world.agents], dim=1)
        
        # 2. 预计算所有智能体间的交互信息，以供JIT函数使用
        self.pos_diffs = self.all_pos.unsqueeze(2) - self.all_pos.unsqueeze(1)
        self.dist_matrix = torch.linalg.norm(self.pos_diffs, dim=-1)
        self.collision_matrix = self.dist_matrix < (self.h_params["agent_radius"] * 2)
        self.collision_matrix.diagonal(dim1=-2, dim2=-1).fill_(False)

        self.vel_diffs = self.all_vel.unsqueeze(2) - self.all_vel.unsqueeze(1)
        self.vel_diffs_norm = torch.linalg.norm(self.vel_diffs, dim=-1)

        # 3. 更新撞墙计数器
        wall_x = self.world.x_semidim * 0.999
        wall_y = self.world.y_semidim * 0.999
        is_pushing_wall_x = (self.all_pos[..., 0] > wall_x) | (self.all_pos[..., 0] < -wall_x)
        is_pushing_wall_y = (self.all_pos[..., 1] > wall_y) | (self.all_pos[..., 1] < -wall_y)
        is_pushing_wall = is_pushing_wall_x | is_pushing_wall_y
        
        wall_counters_clone = self.wall_collision_counters.clone()
        wall_counters_clone[is_pushing_wall] += 1
        wall_counters_clone[~is_pushing_wall] = 0 # 没有推墙则清零，实现“连续”检测
        self.wall_collision_counters.copy_(wall_counters_clone)

        # 4. 调用核心JIT函数进行计算
        dense_rewards, terminal_rewards, dones, a1_still_frames_counter, wall_collision_counters, defender_over_midline_counter, win_this_step, updated_reason_code = \
            self.jitted_reward_calculator(
                self.h_params,
                self.all_pos,
                self.all_vel,
                self.p_vels,
                self.p_raw_actions,
                self.raw_actions,
                self.raw_breaks,
                self.basket.state.pos,
                self.spot_center.state.pos,
                self.t_remaining,
                self.a1_still_frames_counter.to(torch.int32), # 确保传入JIT的类型正确
                self.wall_collision_counters.to(torch.int32),
                self.defender_over_midline_counter.to(torch.int32),
                self.termination_reason_code.to(torch.int32),
                self.dones,
                self.dist_matrix,
                self.collision_matrix,
                self.vel_diffs_norm,
                self.requested_accelerations,
                self.a1_normalized_speed_k,
            )

        # 5. 根据JIT函数的输出更新场景状态
        self.step_dense_rewards = dense_rewards
        self.terminal_rewards = terminal_rewards
        self.dones = dones
        self.a1_still_frames_counter = a1_still_frames_counter.to(torch.int32)
        self.wall_collision_counters = wall_collision_counters.to(torch.int32)
        self.defender_over_midline_counter = defender_over_midline_counter.to(torch.int32)
        self.win_this_step = win_this_step
        self.termination_reason_code = updated_reason_code.to(torch.int32)

        # # 可以在这里处理JIT函数无法执行的操作，比如打印
        # if torch.any(self.win_this_step):
        #     print(f"got {torch.sum(self.win_this_step).item()} wins in this step")
        
        self.dones_this_step.copy_(self.dones)

    # @timer
    def post_step(self):
        """
        在物理步长结束后执行，用于记录状态以备下一帧使用。
        """
        # 记录当前帧的速度，作为下一帧的“上一帧速度”
        self.p_vels.copy_(self.all_vel)
        self.p_raw_actions.copy_(self.raw_actions)

        # 对物理出界的智能体，将其速度强制置零
        for agent in self.world.agents:
            pos = agent.state.pos
            is_hard_oob = (torch.abs(pos[:, 0]) > (0.999 * self.h_params['W'] / 2)) | (torch.abs(pos[:, 1]) > (0.999 * self.h_params['L'] / 2))
            agent.state.vel[is_hard_oob] = 0.0

    def info(self, agent: Agent):
        # 获取当前智能体的索引
        agent_idx = self.world.agents.index(agent)
        
        # 从预先计算好的奖励张量中，根据索引提取对应的值
        # .clone() 和 .unsqueeze(-1) 是为了保证格式正确
        dense_reward = self.dense_reward_factor * self.step_dense_rewards[:, agent_idx].clone().unsqueeze(-1)
        terminal_reward = self.terminal_rewards[:, agent_idx].clone().unsqueeze(-1)

        return {
            # 原有的信息
            "win_in_step": self.win_this_step.clone().float().unsqueeze(-1),
            "termination_reason": self.termination_reason_code.clone().float().unsqueeze(-1),
            
            # 新增的奖励信息
            "dense_reward": dense_reward,
            "terminal_reward": terminal_reward,
        }

    def done(self):
        # 直接返回在pre_step中由JIT函数计算好的dones标志
        return self.dones
    
    def get_global_state(self):
        """
        获取环境的全局状态，适配 Attention Critic 的输入格式。
        返回一个按照实体顺序组织特征的扁平化张量。
        
        返回:
            torch.Tensor: 全局状态张量，形状为 [B, D]，
                        其中 D 是全局状态的总维度。
        """
        # 1. 获取所有智能体的位置和速度
        # all_pos/all_vel 的形状为 [B, N, 2], N=4
        all_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        all_vel = torch.stack([a.state.vel for a in self.world.agents], dim=1)

        # 2. **【关键修改】** 将每个智能体的位置和速度拼接在一起
        # [B, N, 2] 和 [B, N, 2] -> [B, N, 4]
        # 这样，每个智能体的4个特征（pos_x, pos_y, vel_x, vel_y）就在一起了
        agent_states = torch.cat([all_pos, all_vel], dim=-1)

        # 3. 将智能体状态张量扁平化
        # [B, N, 4] -> [B, N * 4] = [B, 16]
        batch_dim = self.world.batch_dim
        flat_agent_states = agent_states.view(batch_dim, -1)

        # 4. 获取其他关键状态信息 (这部分不变)
        spot_pos = self.spot_center.state.pos      # 形状: [B, 2]
        basket_pos = self.basket.state.pos        # 形状: [B, 2]
        time_obs = self.t_remaining / self.h_params["t_limit"] # 形状: [B, 1]

        # 5. **【关键修改】** 按照 entity_configs 的顺序将所有信息拼接
        # 顺序: 4个agent, 1个spot, 1个basket, 1个time
        global_state = torch.cat([
            flat_agent_states,  # 16维
            spot_pos,           # 2维
            basket_pos,         # 2维
            time_obs,           # 1维
        ], dim=-1)

        # 返回的张量形状仍然是 [B, 21]，但内部的数据排列顺序已经改变
        return global_state
        

    def reward(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        
        # 核心计算已在pre_step中完成。这里只负责组合奖励并返回。
        # 最终奖励 = 稠密奖励 * 系数 + 终局奖励
        rew = self.dense_reward_factor * self.step_dense_rewards[:, agent_idx] + self.terminal_rewards[:, agent_idx]

        # 在开局延迟期内，A1的奖励为0
        if agent == self.a1:
            is_delayed = self.delay_counter > 0
            rew = torch.where(is_delayed, 0.0, rew)
        return rew

    # @timer
    def observation(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        is_attacker = agent_idx < self.n_attackers

        # --- 1. 为每个逻辑实体创建独立的、未填充的张量 ---
        self_pos = agent.state.pos
        self_vel = agent.state.vel
        
        # ... (获取队友和对手信息的代码不变) ...
        if is_attacker:
            teammate_idx = 1 - agent_idx
            opp1_idx, opp2_idx = self.n_attackers, self.n_attackers + 1
        else: 
            teammate_idx = 1 - (agent_idx - self.n_attackers) + self.n_attackers
            opp1_idx, opp2_idx = 0, 1
            
        teammate = self.world.agents[teammate_idx]
        opp1 = self.world.agents[opp1_idx]
        opp2 = self.world.agents[opp2_idx]

        self_obs = torch.cat([self_pos, self_vel], dim=-1)
        teammate_obs = torch.cat([teammate.state.pos - self_pos, self.p_vels[:, teammate_idx] - self_vel], dim=-1)
        opp1_obs = torch.cat([opp1.state.pos - self_pos, self.p_vels[:, opp1_idx] - self_vel], dim=-1)
        opp2_obs = torch.cat([opp2.state.pos - self_pos, self.p_vels[:, opp2_idx] - self_vel], dim=-1)
        
        spot_rel_pos = self.spot_center.state.pos - self_pos
        basket_rel_pos = self.basket.state.pos - self_pos
        time_obs = self.t_remaining / self.h_params["t_limit"]

        # --- 2. 对每个实体独立填充到4维 ---
        # F.pad(tensor, (左填充, 右填充))
        if is_attacker:
            spot_padded = spot_rel_pos
        else:
            spot_padded = torch.zeros_like(spot_rel_pos) # 防守方不知道投篮点，用全0填充

        basket_padded = basket_rel_pos
        time_padded = time_obs

        # --- 3. 将所有维度统一的实体拼接成一个扁平的28维向量 ---
        # 7个实体 * 每个4维 = 28维
        obs = torch.cat([
            self_obs,
            teammate_obs,
            opp1_obs,
            opp2_obs,
            spot_padded,
            basket_padded,
            time_padded,
        ], dim=-1)
    
        return obs
    
    def extra_render(self, env_index: int):
        # 此部分用于在渲染窗口中额外绘制调试信息（如奖励曲线图）
        geoms = []
        from vmas.simulator.rendering import Geom
        import io
        
        
        class SpriteGeom(Geom):
            def __init__(self, image, x, y, target_width, target_height):
                super().__init__()
                texture = image.get_texture()
                flipped_texture = texture.get_transform(flip_y=True)
                self.sprite = pyglet.sprite.Sprite(img=flipped_texture, x=x, y=y)
                if self.sprite.width > 0: self.sprite.scale_x = target_width / self.sprite.width
                if self.sprite.height > 0: self.sprite.scale_y = target_height / self.sprite.height
                self.sprite.blend_src = pyglet.gl.GL_SRC_ALPHA
                self.sprite.blend_dest = pyglet.gl.GL_ONE_MINUS_SRC_ALPHA
            def render1(self):
                self.sprite.draw()

        plot_width = 10
        plot_height = 6
        pose_list = [(-14, 0), (4, 0), (-14, -6), (4, -6)] 
        # 遍历每个智能体，更新其历史并绘图
        for i, agent in enumerate(self.world.agents):
            # 1. 计算当前步的奖励
            rew_tensor = self.reward(agent)
            # 2. 将当前环境的奖励值(标量)追加到历史记录中
            if env_index not in self.reward_hist:
                self.reward_hist[env_index] = {}
            if i not in self.reward_hist[env_index]:
                self.reward_hist[env_index][i] = []
            self.reward_hist[env_index][i].append(rew_tensor[env_index].item())

            # 3. 准备绘图
            history_list = self.reward_hist[env_index][i]
            artists = self.plot_artists[i]
            fig, ax, line = artists['fig'], artists['ax'], artists['line']
            
            x_data = range(len(history_list))
            line.set_data(x_data, history_list)
            ax.relim()
            ax.autoscale_view(tight=True)
            
            # 4. 将matplotlib图像转换为Pyglet可渲染对象
            with io.BytesIO() as buf:
                fig.canvas.draw()
                image_data = fig.canvas.buffer_rgba().tobytes()
                plot_image = pyglet.image.ImageData(
                    fig.canvas.get_width_height()[0],
                    fig.canvas.get_width_height()[1],
                    'RGBA',
                    image_data
                )
                if i < len(pose_list):
                    x, y = pose_list[i]
                    img_geom = SpriteGeom(plot_image, x, y + 6, plot_width, plot_height)
                    geoms.append(img_geom)
        return geoms

if __name__ == "__main__":
    # 使用此脚本可以交互式地运行和测试环境
    render_interactively(
        __file__,
        control_two_agents=True, # 允许手动控制两个智能体进行测试
    )