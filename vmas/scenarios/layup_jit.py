import torch
from typing import Dict, Tuple

def calculate_rewards_and_dones_jit(
    # 包含了所有超参数的字典
    h_params: Dict[str, float],
    # 包含世界状态的张量
    all_pos: torch.Tensor,
    all_vel: torch.Tensor,
    p_vels: torch.Tensor,
    p_raw_actions: torch.Tensor,
    raw_actions: torch.Tensor,
    raw_breaks: torch.Tensor,
    basket_pos: torch.Tensor,
    spot_center_pos: torch.Tensor,
    t_remaining: torch.Tensor,
    # 需要在函数内部更新并返回的状态张量
    a1_still_frames_counter: torch.Tensor,
    wall_collision_counters: torch.Tensor,
    defender_over_midline_counter: torch.Tensor,
    termination_reason_code: torch.Tensor,
    dones: torch.Tensor,
    # 预先计算好的交互张量
    dist_matrix: torch.Tensor,
    collision_matrix: torch.Tensor,
    vel_diffs_norm: torch.Tensor,
    requested_accelerations_tensor: torch.Tensor,
    a1_normalized_speed_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT兼容的全向量化函数，用于并行计算奖励和回合终止信号。
    此版本经过重构，提升了代码可读性，增加了详细注释，并减少了不必要的重复计算。

    返回:
        dense_reward (Tensor): 所有智能体在当前步的稠密奖励。
        terminal_rewards (Tensor): 所有智能体的终局奖励 (仅在回合结束时非零)。
        dones_out (Tensor): 所有环境的终止标志。
        a1_still_frames_counter (Tensor): 更新后的A1静止计数器。
        wall_collision_counters (Tensor): 更新后的撞墙计数器。
        defender_over_midline_counter (Tensor): 更新后的防守方越线计数器。
        attacker_win_this_step (Tensor): 标记哪些环境因进攻方获胜而结束。
        reason_code (Tensor): 记录每个环境的回合终止原因。
    """
    # =================================================================================
    # 0. 初始化与准备
    # =================================================================================
    
    # 从输入中提取维度和常量
    batch_dim, n_agents, _ = all_pos.shape
    device = all_pos.device
    n_attackers = 2
    n_defenders = 2
    
    # 初始化返回的张量
    terminal_rewards = torch.zeros(batch_dim, n_agents, device=device)
    dense_reward = torch.zeros(batch_dim, n_agents, device=device)
    dones_out = dones.clone()
    attacker_win_this_step = torch.zeros(batch_dim, device=device, dtype=torch.bool)
    reason_code = termination_reason_code.clone()

    # 提取关键智能体的位置和速度，方便后续使用
    a1_pos = all_pos[:, 0]
    a1_vel = all_vel[:, 0]
    a2_pos = all_pos[:, 1]
    defender_pos = all_pos[:, n_attackers:]
    is_braking = raw_breaks > 0

    # =================================================================================
    # 1. 回合终止条件检查 (Terminal Conditions)
    # =================================================================================

    # --- 条件1: 尝试投篮 (Shot Attempt) ---
    dist_a1_to_spot = torch.linalg.norm(a1_pos - spot_center_pos, dim=1)
    
    # 判断A1是否满足“准备投篮”的状态：在区域内、速度足够慢、没有加速意图、而且正在踩刹车
    in_area = (dist_a1_to_spot <= h_params['R_spot']) & (a1_pos[:, 1] > 0)
    is_still = torch.linalg.norm(a1_vel, dim=1) < h_params['v_shot_threshold']
    not_accelerating = (torch.linalg.norm(raw_actions[:, 0, :], dim=1) < h_params['a_shot_threshold']) | is_braking[:, 0]
    is_ready_to_shoot = in_area & is_still & not_accelerating
    
    # 更新A1静止计数器，如果满足准备条件则+1，否则清零
    prev_still_counter = a1_still_frames_counter
    curr_still_counter = torch.where(is_ready_to_shoot, prev_still_counter + 1, 0)
    
    # 如果静止帧数达到阈值，则触发投篮
    shot_attempted = (curr_still_counter >= h_params['shot_still_frames']) & ~dones_out
    if torch.any(shot_attempted):
        shot_b_idx = shot_attempted.nonzero().squeeze(-1) # 发生投篮的环境索引
        
        # 提取投篮瞬间的状态
        a1_pos_shot = a1_pos[shot_b_idx]
        a2_pos_shot = all_pos[shot_b_idx, 1]
        spot_pos_shot = spot_center_pos[shot_b_idx]
        basket_pos_shot = basket_pos[shot_b_idx]
        defender_pos_shot = all_pos[shot_b_idx][:, 2:]

        # --- 计算封盖因子 ---
        # 核心逻辑：判断防守球员是否在A1到篮筐的路径上，并根据距离计算封盖程度
        shot_vector = basket_pos_shot - a1_pos_shot
        blocker_vector = defender_pos_shot - a1_pos_shot.unsqueeze(1)
        
        shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
        dot_product = torch.sum(blocker_vector * shot_vector.unsqueeze(1), dim=-1)
        proj_len_ratio = dot_product / shot_vector_norm_sq
        
        # 硬门控：防守者必须在A1和篮筐之间
        is_between = (proj_len_ratio > 0) & (proj_len_ratio < 1)
        
        # 计算防守者到投篮路径的垂直距离
        projection = proj_len_ratio.unsqueeze(-1) * shot_vector.unsqueeze(1)
        dist_perp_sq = torch.sum((blocker_vector - projection)**2, dim=-1)
        
        # 软门控：防守者离A1越近，封盖贡献的权重越高 (Sigmoid函数实现软切换)
        dist_a1_to_def = torch.linalg.norm(blocker_vector, dim=-1)
        gate_input = h_params['def_proximity_threshold'] - dist_a1_to_def
        soft_proximity_gate = torch.sigmoid(h_params['block_gate_k'] * gate_input)
        
        is_blocker_per_defender = is_between & (dist_perp_sq < (h_params['proximity_threshold'])**2)
        
        # 综合计算每个防守者的封盖贡献
        block_contribution = (
            torch.exp(-dist_perp_sq / (2 * h_params['block_sigma']**2)) # 基于横向距离
            * is_blocker_per_defender.float()                          # 基于站位的硬门控
            * soft_proximity_gate                                      # 基于纵向距离的软门控
        )
        total_block_factor = torch.clamp(block_contribution.sum(dim=1), 0, 1)

        # --- 判断胜负并分配原因码 ---
        is_a_winning_shot = total_block_factor < h_params['win_condition_block_threshold']
        winning_shot_indices = shot_b_idx[is_a_winning_shot]
        losing_shot_indices = shot_b_idx[~is_a_winning_shot]

        if winning_shot_indices.numel() > 0:
            attacker_win_this_step[winning_shot_indices] = True
            reason_code[winning_shot_indices] = 1 # 原因码1: 投篮命中
        if losing_shot_indices.numel() > 0:
            reason_code[losing_shot_indices] = 11 # 原因码11: 投篮被盖
        
        # --- 计算进攻方奖励 (A1 & A2) ---
        base_score = h_params['max_score'] * (1 - dist_a1_to_spot[shot_b_idx] / h_params['R_spot'])
        final_score_modified = base_score * (1 - total_block_factor)
        time_bonus = h_params['k_time_bonus'] * (t_remaining[shot_b_idx].squeeze(-1) / h_params['t_limit']) * (1 - total_block_factor)
        
        dist_a1_to_defs_shot = torch.linalg.norm(blocker_vector, dim=-1)
        avg_dist_to_defs = torch.mean(dist_a1_to_defs_shot, dim=1)
        spacing_bonus = h_params['k_spacing_bonus'] * avg_dist_to_defs
        
        # A1 投篮静止奖励：速度越小、动作指令越小，奖励越高
        a1_speed_shot = torch.linalg.norm(a1_vel[shot_b_idx], dim=-1)
        a1_action_norm_shot = torch.linalg.norm(raw_actions[shot_b_idx, 0, :], dim=-1)
        vel_stillness_bonus = h_params['k_shot_stillness_vel_bonus'] * torch.exp(-a1_speed_shot)
        act_stillness_bonus = h_params['k_shot_stillness_act_bonus'] * torch.exp(-a1_action_norm_shot)
        
        a1_reward = final_score_modified + spacing_bonus + time_bonus + vel_stillness_bonus + act_stillness_bonus + h_params['shoot_score']
        terminal_rewards[shot_b_idx, 0] += a1_reward

        # A2 掩护奖励：判断A2是否为A1挡住了最关键的防守者
        _, closest_def_indices = torch.min(dist_a1_to_defs_shot**2, dim=1)
        batch_indices = torch.arange(len(shot_b_idx), device=device)
        p_closest_def = defender_pos_shot[batch_indices, closest_def_indices]
        
        def_to_a1_vec = a1_pos_shot - p_closest_def
        def_to_a1_unit_vec = def_to_a1_vec / (torch.linalg.norm(def_to_a1_vec, dim=-1, keepdim=True) + 1e-6)
        ideal_screen_pos = p_closest_def + h_params['screen_pos_offset'] * def_to_a1_unit_vec
        
        dist_a2_to_ideal_sq = torch.sum((a2_pos_shot - ideal_screen_pos)**2, dim=-1)
        
        # 位置门控: 确认A2在A1和防守者之间，形成有效掩护
        vec_a2_to_def = p_closest_def - a2_pos_shot
        vec_a2_to_a1 = a1_pos_shot - a2_pos_shot
        dot_product_gate = torch.sum(vec_a2_to_def * vec_a2_to_a1, dim=-1)
        screen_gate = torch.sigmoid(-h_params['k_screen_gate'] * dot_product_gate)
        
        screen_bonus = h_params['k_a2_screen_bonus'] * torch.exp(-dist_a2_to_ideal_sq / (2 * h_params['a2_screen_sigma']**2)) * screen_gate
        a2_reward = final_score_modified + screen_bonus + spacing_bonus + time_bonus
        terminal_rewards[shot_b_idx, 1] += a2_reward

        # --- 计算防守方奖励 (D1 & D2) ---
        for i in range(n_defenders):
            R_block = h_params['k_def_block_reward'] * block_contribution[:, i] # 封盖贡献奖励
            R_force = h_params['k_def_force_reward'] * (dist_a1_to_spot[shot_b_idx] / h_params['R_spot']) # 迫使远离投篮点奖励
            
            # 站位奖励：奖励防守者站在A1和篮筐之间
            a1_to_basket_unit_vec = (basket_pos_shot - a1_pos_shot) / (torch.linalg.norm(basket_pos_shot - a1_pos_shot, dim=-1, keepdim=True) + 1e-6)
            ideal_pos = a1_pos_shot + h_params['def_pos_offset'] * a1_to_basket_unit_vec
            dist_to_ideal_sq = torch.sum((defender_pos_shot[:, i, :] - ideal_pos)**2, dim=-1)
            
            # 位置门控：确保防守者在A1"身后"（朝向篮筐方向）
            d_from_a1_vec = defender_pos_shot[:, i, :] - a1_pos_shot
            proj_dot = torch.sum(d_from_a1_vec * a1_to_basket_unit_vec, dim=-1)
            pos_gate = torch.sigmoid(5.0 * proj_dot)
            
            positioning_reward_factor = torch.exp(-dist_to_ideal_sq / (2 * h_params['def_pos_sigma']**2))
            R_positioning = h_params['k_def_pos_reward'] * positioning_reward_factor * pos_gate
            
            # 区域控制奖励：奖励防守者靠近投篮点中心
            dist_def_to_spot_sq = torch.sum((defender_pos_shot[:, i, :] - spot_pos_shot)**2, dim=-1)
            R_area_control = h_params['k_def_area_reward'] * torch.exp(-dist_def_to_spot_sq / (2 * h_params['def_gaussian_spot_sigma']**2))
            
            total_def_reward = R_block + R_force + R_positioning + R_area_control - h_params['k_def_shot_penalty']
            terminal_rewards[shot_b_idx, n_attackers + i] += total_def_reward
        
        dones_out |= shot_attempted

    # --- 条件2: 时间耗尽 (Time Up) ---
    time_up = (t_remaining.squeeze(-1) <= 0) & ~dones_out
    if torch.any(time_up):
        # 提取超时瞬间的状态
        dist_a1_to_spot_timeout = dist_a1_to_spot[time_up] # 复用之前计算的距离
        is_in_spot = dist_a1_to_spot_timeout <= h_params['R_spot']

        # 移动惩罚：如果超时瞬间A1仍在高速移动，则施加惩罚
        a1_speed_timeout = torch.linalg.norm(a1_vel[time_up], dim=-1)
        a1_action_norm_timeout = torch.linalg.norm(raw_actions[time_up, 0, :], dim=-1)
        vel_penalty = h_params['k_timeout_move_vel_penalty'] * a1_speed_timeout
        act_penalty = h_params['k_timeout_move_act_penalty'] * a1_action_norm_timeout
        total_movement_penalty = vel_penalty + act_penalty

        # 根据A1是否在投篮圈内，计算不同的奖惩
        # 在圈内：获得基础奖励，但减去移动惩罚
        reward_in_spot = h_params['attacker_timeout_reward_in_spot'] - total_movement_penalty
        # 在圈外：根据离圈的距离获得惩罚
        reward_out_of_spot = h_params['attacker_timeout_base_reward_out_spot'] - h_params['k_timeout_dist_reward_factor'] * dist_a1_to_spot_timeout

        attacker_reward = torch.where(is_in_spot, reward_in_spot, reward_out_of_spot)
        attacker_reward_clamped = torch.clamp(
            attacker_reward,
            min=-h_params['attacker_timeout_reward_max'],
            max=h_params['attacker_timeout_reward_max']
        )
        
        # 分配奖惩
        terminal_rewards[time_up, 0] = attacker_reward_clamped
        terminal_rewards[time_up, 1] = h_params["foul_teammate_factor"] * attacker_reward_clamped
        terminal_rewards[time_up, n_attackers:] = h_params['defender_timeout_reward'] # 防守方获得固定奖励
        
        reason_code[time_up] = 12 # 原因码12: 进攻超时
        dones_out |= time_up
    
    # --- 条件3: 碰撞犯规 (Foul) ---
    # is_foul 条件: 发生碰撞 & 相对速度超过阈值 & 回合尚未结束
    is_foul = collision_matrix & (vel_diffs_norm > h_params['v_foul_threshold']) & ~dones_out.view(-1, 1, 1)
    if torch.triu(is_foul, diagonal=1).any():
        foul_indices = torch.triu(is_foul, diagonal=1).nonzero()
        b_idx, i_idx, j_idx = foul_indices.T # JIT 兼容的解包

        # 犯规惩罚大小与相对速度挂钩
        relative_speeds = vel_diffs_norm[b_idx, i_idx, j_idx]
        dynamic_foul_magnitude = h_params['R_foul'] + h_params['k_foul_vel_penalty'] * relative_speeds
        
        # 判断谁是主动撞人者 (active)
        agent_i_p_vel = p_vels[b_idx, i_idx]
        pos_rel = all_pos[b_idx, j_idx] - all_pos[b_idx, i_idx]
        vel_rel_on_pos = torch.einsum("bd,bd->b", agent_i_p_vel, pos_rel)
        i_is_active = vel_rel_on_pos > 0
        active_indices = torch.where(i_is_active, i_idx, j_idx)
        passive_indices = torch.where(i_is_active, j_idx, i_idx)
        
        active_is_attacker = active_indices < n_attackers
        passive_is_attacker = passive_indices < n_attackers
        is_friendly_fire = (active_is_attacker == passive_is_attacker)
        
        # 使用 index_add_ 高效地将犯规奖惩累加到对应环境中
        foul_rewards = torch.zeros_like(terminal_rewards)
        
        # 情况A: 敌对犯规
        opp_foul_mask = ~is_friendly_fire
        if torch.any(opp_foul_mask):
            opp_b = b_idx[opp_foul_mask]
            opp_active = active_indices[opp_foul_mask]
            opp_passive = passive_indices[opp_foul_mask]
            opp_magnitude = dynamic_foul_magnitude[opp_foul_mask]
            
            # 为犯规的智能体创建奖惩
            num_opp_fouls = opp_b.shape[0]
            opp_rewards_to_add = torch.zeros(num_opp_fouls, n_agents, device=device)
            opp_row_indices = torch.arange(num_opp_fouls, device=device)
            
            # 主动犯规者受罚，被犯规者得利
            opp_rewards_to_add[opp_row_indices, opp_active] = -opp_magnitude
            opp_rewards_to_add[opp_row_indices, opp_passive] = opp_magnitude * h_params['foul_teammate_factor']
            # 注意：此处原版代码中注释掉了对队友的奖惩，我们遵循当前有效逻辑
            
            foul_rewards.index_add_(0, opp_b, opp_rewards_to_add)
            
            # 根据犯规方判断胜负
            active_is_defender = opp_active >= n_attackers
            attacker_win_this_step[opp_b[active_is_defender]] = True
            reason_code[opp_b[active_is_defender]] = 2 # 原因码2: 对手犯规
            reason_code[opp_b[~active_is_defender]] = 13 # 原因码13: 己方犯规

        # 情况B: 友军误伤 (Friendly Fire)
        ff_foul_mask = is_friendly_fire
        if torch.any(ff_foul_mask):
            ff_b = b_idx[ff_foul_mask]
            ff_active = active_indices[ff_foul_mask]
            ff_passive = passive_indices[ff_foul_mask]
            ff_magnitude = dynamic_foul_magnitude[ff_foul_mask]
            
            num_ff_fouls = ff_b.shape[0]
            ff_rewards_to_add = torch.zeros(num_ff_fouls, n_agents, device=device)
            ff_row_indices = torch.arange(num_ff_fouls, device=device)

            # 友军误伤，双方都受罚
            ff_rewards_to_add[ff_row_indices, ff_active] = -ff_magnitude
            ff_rewards_to_add[ff_row_indices, ff_passive] = -ff_magnitude
            
            foul_rewards.index_add_(0, ff_b, ff_rewards_to_add)

            # 根据误伤方判断胜负
            ff_active_is_attacker = active_is_attacker[ff_foul_mask]
            attacker_win_this_step[ff_b[~ff_active_is_attacker]] = True # 防守方误伤 -> 进攻方胜利
            reason_code[ff_b[~ff_active_is_attacker]] = 5 # 原因码5: 对手友军误伤
            reason_code[ff_b[ff_active_is_attacker]] = 15 # 原因码15: 己方友军误伤
            
        terminal_rewards += foul_rewards
        dones_out[b_idx] = True

    # --- 条件4: 持续撞墙导致回合结束 (Wall Collision Timeout) ---
    is_wall_timeout_per_agent = (wall_collision_counters >= h_params['wall_collision_frames'])
    wall_timeout_triggered_in_env = is_wall_timeout_per_agent.any(dim=1) & ~dones_out
    if torch.any(wall_timeout_triggered_in_env):
        b_idx = wall_timeout_triggered_in_env.nonzero().squeeze(-1)
        
        # 判断是哪一方撞墙导致结束
        triggering_agents_in_env = is_wall_timeout_per_agent[b_idx]
        is_defender_triggered = triggering_agents_in_env[:, n_attackers:].any(dim=1)
        
        # 防守方撞墙 -> 进攻方赢
        b_idx_def_wall_coll = b_idx[is_defender_triggered]
        if b_idx_def_wall_coll.numel() > 0:
            attacker_win_this_step[b_idx_def_wall_coll] = True
            reason_code[b_idx_def_wall_coll] = 3 # 原因码3: 对手失误-撞墙
            
        # 进攻方撞墙 -> 进攻方输
        b_idx_att_wall_coll = b_idx[~is_defender_triggered]
        if b_idx_att_wall_coll.numel() > 0:
            reason_code[b_idx_att_wall_coll] = 14 # 原因码14: 己方失误-撞墙

        # 只惩罚当前正靠在墙边的智能体
        wall_x = h_params['W'] / 2 * 0.99
        wall_y = h_params['L'] / 2 * 0.99
        all_pos_in_env = all_pos[b_idx]
        is_at_wall_mask = (torch.abs(all_pos_in_env[..., 0]) > wall_x) | (torch.abs(all_pos_in_env[..., 1]) > wall_y)
        
        rewards_subset = terminal_rewards[b_idx]
        rewards_subset[is_at_wall_mask] += h_params['R_wall_collision_penalty']
        terminal_rewards[b_idx] = rewards_subset
        
        dones_out[b_idx] = True

    # --- 条件5: 防守方越过中线过久犯规 (Defender Over Midline Foul) ---
    is_over_midline = defender_pos[:, :, 1] < 0
    defender_over_midline_counter = torch.where(is_over_midline, defender_over_midline_counter + 1, 0)

    midline_foul_per_defender = (defender_over_midline_counter >= h_params['max_time_over_midline'])
    midline_foul_triggered_in_env = midline_foul_per_defender.any(dim=1) & ~dones_out
    if torch.any(midline_foul_triggered_in_env):
        b_idx = midline_foul_triggered_in_env.nonzero().squeeze(-1)
        
        attacker_win_this_step[b_idx] = True
        reason_code[b_idx] = 4 # 原因码4: 对手失误-越线
        
        # 只惩罚当前正处于越线状态的防守方
        offending_defenders_pos = defender_pos[b_idx]
        is_offending_defender = offending_defenders_pos[:, :, 1] < 0
        
        defender_rewards_subset = terminal_rewards[b_idx, n_attackers:]
        defender_rewards_subset[is_offending_defender] -= h_params['R_midline_foul']
        terminal_rewards[b_idx, n_attackers:] = defender_rewards_subset
        
        dones_out[b_idx] = True

    # =================================================================================
    # 2. 稠密奖励计算 (Dense Rewards)
    # =================================================================================
    
    # --- 2.1 创建角色掩码，方便后续分角色计算 ---
    agent_indices = torch.arange(n_agents, device=device)
    a1_mask = (agent_indices == 0).view(1, -1)
    a2_mask = (agent_indices == 1).view(1, -1)
    attacker_mask = (agent_indices < n_attackers).view(1, -1)
    defender_mask = (agent_indices >= n_attackers).view(1, -1)

    # --- 2.2 通用奖励/惩罚 (对所有智能体生效) ---

    # 2.2.1 出界惩罚 (OOB Penalty)
    # 使用 logaddexp 函数创建一个平滑的边界惩罚，越过边界越深，惩罚越大
    safe_x = h_params['W'] / 2 - (h_params['agent_radius'] / 2)
    safe_y = h_params['L'] / 2 - (h_params['agent_radius'] / 2)
    oob_depth_x = torch.logaddexp(torch.tensor(0.0, device=device), (torch.abs(all_pos[..., 0]) - safe_x) / h_params['oob_margin'])
    oob_depth_y = torch.logaddexp(torch.tensor(0.0, device=device), (torch.abs(all_pos[..., 1]) - safe_y) / h_params['oob_margin'])
    oob_penalty = h_params['oob_penalty'] * h_params['oob_margin'] * (oob_depth_x + oob_depth_y) * (torch.linalg.norm(all_vel, dim=-1) + 1.0)
    dense_reward += oob_penalty
    
    # 2.2.2 控制量惩罚 (Action Magnitude Penalty)
    raw_u_norm = torch.linalg.vector_norm(raw_actions, dim=-1)
    dense_reward -= h_params['k_u_penalty_general'] * raw_u_norm
    
    # 对超过最大动作阈值的行为进行额外惩罚
    penalty_threshold = h_params["v_max"] * h_params['k_action_access_max_threshold']
    excess_action_magnitude = torch.clamp(raw_u_norm - penalty_threshold, min=0.0)
    penalty_range = h_params["v_max"] * (1.0 - h_params['k_action_access_max_threshold'])
    action_limit_penalty = h_params['k_action_access_max_penalty'] * (excess_action_magnitude / (penalty_range + 1e-6))
    dense_reward -= action_limit_penalty

    # 2.2.3 刹车使用惩罚 (Brake Usage Penalty)
    excess_brake_magnitude = torch.clamp(raw_breaks - penalty_threshold, min=0.0)
    braking_limit_panlty = h_params['k_action_access_max_penalty'] * (excess_brake_magnitude / (penalty_range + 1e-6))
    dense_reward -= (h_params["k_brake_usage_penalty"] * is_braking.float() + braking_limit_panlty)
    
    # 2.2.4 矛盾动作惩罚 (Conflicting Action Penalty): 惩罚同时输出方向指令和刹车指令
    conflicting_action_penalty = h_params['k_conflicting_action_penalty'] * raw_u_norm * is_braking.float()
    dense_reward -= conflicting_action_penalty
    
    # 2.2.5 超出动力学极限惩罚 (Excess Acceleration Penalty)，刹车时豁免
    requested_a_norm = torch.linalg.norm(requested_accelerations_tensor, dim=-1)
    excess_acceleration = torch.clamp(requested_a_norm - h_params['a_max'], min=0.0)
    acceleration_penalty = -h_params['k_excess_acceleration_penalty'] * (excess_acceleration ** 2)
    dense_reward += torch.where(is_braking, 0.0, acceleration_penalty)

    # 2.2.6 动作平滑度惩罚 (Jerk Penalty)
    action_jerk = torch.linalg.norm(raw_actions - p_raw_actions, dim=-1)
    dense_reward -= h_params['k_action_jerk_penalty'] * action_jerk

    # --- 2.3 基于距离和碰撞的通用惩罚 ---

    # 2.3.1 近距离惩罚 (Proximity Penalty)
    dist_matrix_no_self = dist_matrix.clone()
    dist_matrix_no_self.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
    
    # 为不同角色的智能体设置不同的惩罚系数和触发距离
    k_def_proximity = torch.where(
        torch.linalg.norm(all_pos - spot_center_pos.unsqueeze(1), dim=-1) <= h_params['R_spot'], 
        h_params['k_def_proximity_penalty'] * (1 - h_params['proximity_penalty_reduction_in_spot']), 
        h_params['k_def_proximity_penalty']
    )
    k_prox = h_params['k_a1_proximity_penalty'] * a1_mask + \
             h_params['k_proximity_penalty'] * a2_mask + \
             k_def_proximity * defender_mask
             
    prox_threshold = torch.where(a1_mask, h_params['a1_proximity_threshold'], h_params['proximity_threshold'])
    k_margin_per_agent = torch.where(a1_mask, h_params['a1_proximity_penalty_margin'], h_params['proximity_penalty_margin'])

    is_too_close = dist_matrix_no_self < prox_threshold.unsqueeze(-1)
    if torch.any(is_too_close):
        # 再次使用 logaddexp 计算平滑的穿透深度
        penetration = torch.logaddexp(
            torch.tensor(0.0, device=device), 
            (prox_threshold.unsqueeze(-1) - dist_matrix_no_self) / k_margin_per_agent.unsqueeze(-1)
        ) * k_margin_per_agent.unsqueeze(-1)
        
        proximity_penalty = -k_prox.unsqueeze(-1) * penetration
        dense_reward += (proximity_penalty * is_too_close.float()).sum(dim=-1)

    # 2.3.2 碰撞惩罚 (Collision Penalty)
    if torch.any(collision_matrix):
        pos_rel = all_pos.unsqueeze(2) - all_pos.unsqueeze(1) # B,N,N,2
        
        # 高速碰撞惩罚
        vel_proj = torch.einsum("bnd,bnmd->bnm", all_vel, pos_rel)
        is_active = vel_proj > 0
        collision_penalty = torch.where(is_active, -h_params['k_coll_active'], -h_params['k_coll_passive']) * vel_diffs_norm
        dense_reward += (collision_penalty * collision_matrix.float()).sum(dim=-1)

        # 低速推挤惩罚
        is_low_speed_collision = collision_matrix & (vel_diffs_norm < h_params['low_velocity_threshold'])
        if torch.any(is_low_speed_collision):
            push_penalty_coeff = torch.where(attacker_mask, h_params['k_push_penalty'], h_params['k_def_push_penalty'])
            pos_diffs_norm = torch.linalg.norm(pos_rel, dim=-1, keepdim=True) + 1e-6
            proj_vector = -pos_rel / pos_diffs_norm
            push_force_magnitude = torch.einsum('bnd,bnmd->bnm', raw_actions, proj_vector)
            push_penalty = -push_penalty_coeff.unsqueeze(-1) * torch.clamp(push_force_magnitude, min=0.0) * (~is_braking).unsqueeze(-1).float()
            dense_reward += (push_penalty * is_low_speed_collision.float()).sum(dim=-1)
    
    # 2.3.3 造犯规奖励 (Charge Drawing Reward)
    # 奖励站定不动的智能体，如果对手正高速向它冲来
    is_standing_still = torch.linalg.norm(all_vel, dim=-1) < h_params['stand_still_threshold']
    is_to_stand = (raw_u_norm < h_params['stand_still_threshold']) | is_braking
    relative_pos_all = all_pos.unsqueeze(2) - all_pos.unsqueeze(1)
    relative_dist_all = torch.linalg.norm(relative_pos_all, dim=-1)
    is_within_charge_range = relative_dist_all < h_params['charge_drawing_range']
    
    dot_product = torch.sum(all_vel.unsqueeze(1) * relative_pos_all, dim=-1)
    speed_of_approach = torch.clamp(dot_product / (relative_dist_all + 1e-6), min=0)
    
    agent_is_attacker = attacker_mask.squeeze(0)
    is_opponent_matrix = agent_is_attacker.unsqueeze(1) != agent_is_attacker.unsqueeze(0)
    
    reward_for_opponents = h_params['k_stand_still_reward'] * speed_of_approach * \
                           is_standing_still.unsqueeze(-1).float() * is_to_stand.unsqueeze(-1).float() * \
                           is_within_charge_range.float() * is_opponent_matrix.float()
    dense_reward += reward_for_opponents.sum(dim=-1)

    # --- 2.4 分角色奖励/惩罚 ---
    
    # 预计算A1常用的向量和状态
    a1_speed = torch.linalg.norm(a1_vel, dim=1)
    is_in_spot_a1 = (dist_a1_to_spot <= h_params['R_spot']) & (a1_pos[:, 1] > 0)
    vec_a1_to_basket = basket_pos - a1_pos
    unit_vec_a1_to_basket = vec_a1_to_basket / (torch.linalg.norm(vec_a1_to_basket, dim=-1, keepdim=True) + 1e-6)
    vec_a1_to_defs = defender_pos - a1_pos.unsqueeze(1)
    dist_a1_to_defs = torch.linalg.norm(vec_a1_to_defs, dim=-1)

    # 2.4.1 A1 (持球进攻者) 奖励/惩罚
    
    # 引导奖励: 鼓励A1进入并停留在投篮点
    # a. 高斯吸引力奖励
    a1_gaussian_reward = h_params['gaussian_scale'] * torch.exp(- (dist_a1_to_spot**2) / (2 * h_params['gaussian_sigma']**2))
    # b. 朝着投篮点移动的速度奖励
    speed_to_spot_proj = torch.sum(a1_vel * (spot_center_pos - a1_pos) / (torch.linalg.norm(spot_center_pos - a1_pos, dim=1, keepdim=True) + 1e-6), dim=1)
    speed_spot_reward = a1_normalized_speed_k * speed_to_spot_proj
    # c. 在投篮区域内的存在奖励
    in_spot_reward = h_params['k_a1_in_spot_reward'] * (1.5 - dist_a1_to_spot / h_params['R_spot']) * is_in_spot_a1.float()

    # 被封盖惩罚: A1的投篮路线被防守者阻挡时受罚
    ap = vec_a1_to_basket.unsqueeze(1)
    ad = vec_a1_to_defs
    proj_ratio_blocked = torch.einsum("bnd,bmd->bnm", ad, ap).squeeze(-1) / (torch.linalg.norm(ap, dim=-1).pow(2) + 1e-6)
    is_between_mask = (proj_ratio_blocked > 0) & (proj_ratio_blocked < 1)
    closest_point = a1_pos.unsqueeze(1) + proj_ratio_blocked.unsqueeze(-1) * ap
    dist_perp = torch.linalg.norm(defender_pos - closest_point, dim=-1)
    soft_prox_gate_a1 = torch.sigmoid(h_params['block_gate_k'] * (h_params['def_proximity_threshold'] - dist_a1_to_defs))
    block_factor_a1 = torch.exp(-dist_perp.pow(2) / (2 * h_params['block_sigma'] ** 2)) * is_between_mask.float() * soft_prox_gate_a1
    total_block_factor_a1 = block_factor_a1.sum(dim=1)
    blocked_penalty = total_block_factor_a1 * h_params['k_a1_blocked_penalty']
    
    # 犹豫惩罚: 在非投篮区移动过慢时受罚
    hesitation_factor = torch.clamp(1.0 - (a1_speed / h_params['hesitate_speed_threshold']), min=0.0)**2
    hesitation_penalty = -h_params['k_hesitation_penalty'] * hesitation_factor * (~is_in_spot_a1).float()

    # 动态行为奖励: 根据被封盖程度，在“静止得分”和“移动摆脱”两种策略间动态切换
    # a. 静止奖励 (未被封锁时，鼓励在圈内静止准备投篮)
    raw_a1_u_norm = torch.linalg.vector_norm(raw_actions[:, 0, :], dim=-1)
    vel_still_reward = h_params['k_a1_velocity_stillness_reward'] * torch.exp(-(a1_speed**2) / (2 * h_params['velocity_stillness_sigma']**2))
    act_still_reward = h_params['k_a1_action_stillness_reward'] * torch.exp(-(raw_a1_u_norm**2) / (2 * h_params['action_stillness_sigma']**2))
    stillness_reward = (vel_still_reward + act_still_reward * (raw_a1_u_norm < h_params['low_u_threshold']).float()) * is_in_spot_a1.float()
    
    # b. 摆脱奖励 (被封锁时，鼓励向远离最近防守者的方向移动)
    dist_to_closest_def, closest_def_indices = torch.min(dist_a1_to_defs, dim=1)
    p_closest_def = defender_pos[torch.arange(batch_dim, device=device), closest_def_indices]
    unit_vec_away_from_def = (a1_pos - p_closest_def) / (dist_to_closest_def.unsqueeze(-1) + 1e-6)
    speed_of_separation = torch.sum(a1_vel * unit_vec_away_from_def, dim=1)
    separation_reward = h_params['k_a1_separation_reward'] * torch.clamp(speed_of_separation, min=0.0)

    # c. 动态加权
    dynamic_behavior_reward = (1.0 - total_block_factor_a1) * stillness_reward + total_block_factor_a1 * separation_reward

    # 横向机动奖励: 鼓励A1在受正面防守时横向移动
    pressure_gate_dist = torch.exp(-dist_to_closest_def.pow(2) / (2 * h_params['a1_tangential_pressure_sigma']**2))
    dot_prod_gate = torch.sum((p_closest_def - a1_pos) * vec_a1_to_basket, dim=-1)
    pressure_gate_pos = (dot_prod_gate > 0) & (dot_prod_gate < torch.sum(vec_a1_to_basket**2, dim=-1))
    pressure_gate = pressure_gate_dist * pressure_gate_pos.float()
    vel_parallel = torch.sum(a1_vel * unit_vec_a1_to_basket, dim=-1, keepdim=True) * unit_vec_a1_to_basket
    tangential_speed = torch.linalg.norm(a1_vel - vel_parallel, dim=-1)
    tangential_reward = h_params['k_a1_tangential_reward'] * tangential_speed * pressure_gate

    # 投篮蓄力奖励 & 放弃惩罚
    ready_to_shoot_reward = h_params['k_a1_ready_to_shoot_reward'] * is_ready_to_shoot.float()
    abandon_shot_penalty = -h_params['k_a1_ready_to_shoot_reward'] * ((prev_still_counter > 0) & (curr_still_counter == 0)).float()

    total_a1_reward = a1_gaussian_reward + speed_spot_reward + in_spot_reward + \
                      blocked_penalty + hesitation_penalty + dynamic_behavior_reward + \
                      tangential_reward + abandon_shot_penalty + ready_to_shoot_reward
    dense_reward[:, 0] += total_a1_reward

    # 2.4.2 A2 (无球掩护者) 奖励/惩罚
    p_a1_exp = a1_pos.unsqueeze(1)
    p_a2_exp = a2_pos.unsqueeze(1)
    
    # a. 掩护奖励 (Screening Reward)
    #    核心：找到对A1威胁最大的防守者，并移动到理想的掩护位置
    def_to_a1_vec = p_a1_exp - defender_pos
    ideal_screen_pos = defender_pos + h_params['screen_pos_offset'] * (def_to_a1_vec / (torch.linalg.norm(def_to_a1_vec, dim=-1, keepdim=True) + 1e-6))
    dist_a2_to_ideal_sq = torch.sum((p_a2_exp - ideal_screen_pos)**2, dim=-1)
    
    vec_a2_to_def = defender_pos - p_a2_exp
    vec_a2_to_a1 = p_a1_exp - p_a2_exp
    
    # 门控1: 位置门控，确保A2在A1和防守者之间
    dot_product_gate = torch.sum(vec_a2_to_def * vec_a2_to_a1, dim=-1)
    pos_gate_factor = torch.sigmoid(-h_params['k_screen_gate'] * dot_product_gate)
    # 门控2: 间距门控，确保A2离防守者比离A1更近
    spacing_gate_factor = torch.sigmoid(h_params['screen_spacing_gate_k'] * (torch.linalg.norm(vec_a2_to_a1, dim=-1) - torch.linalg.norm(vec_a2_to_def, dim=-1)))
    
    potential_screen_rewards = h_params['k_ideal_screen_pos'] * torch.exp(-dist_a2_to_ideal_sq / (2 * h_params['screen_pos_sigma']**2)) * pos_gate_factor * spacing_gate_factor
    screen_reward, _ = torch.max(potential_screen_rewards, dim=1) # 取所有可能掩护中的最大收益

    # b. 干扰和排斥奖励 (Interference & Repulsion)
    dist_a2_to_def = torch.linalg.norm(p_a2_exp - defender_pos, dim=-1)
    interference_reward, _ = torch.max(h_params['k_a2_interference_reward'] * torch.exp(-dist_a2_to_def.pow(2) / (2 * h_params['screen_pos_sigma']**2)), dim=1)
    
    repulsion_speed = torch.sum(all_vel[:, n_attackers:] * (-def_to_a1_vec / (torch.linalg.norm(def_to_a1_vec, dim=-1, keepdim=True) + 1e-6)), dim=-1)
    is_a2_responsible = dist_a2_to_def < h_params['repulsion_proximity_threshold']
    repulsion_reward, _ = torch.max(h_params['k_repulsion_reward'] * torch.clamp(repulsion_speed, min=0.0) * is_a2_responsible.float(), dim=1)

    # c. 阻挡A1投篮路线惩罚 (Line of Sight Penalty)
    shot_vec_a2 = vec_a1_to_basket
    a2_vec = a2_pos - a1_pos
    proj_ratio_a2 = torch.sum(a2_vec * shot_vec_a2, dim=-1) / (torch.sum(shot_vec_a2**2, dim=-1) + 1e-6)
    is_between_a2 = (proj_ratio_a2 > 0) & (proj_ratio_a2 < 1)
    dist_perp_sq_a2 = torch.sum((a2_vec - proj_ratio_a2.unsqueeze(-1) * shot_vec_a2)**2, dim=-1)
    proximity_factor_a2 = torch.exp(-torch.linalg.norm(a2_vec, dim=-1).pow(2) / (2 * (2 * h_params['agent_radius'])**2))
    line_block_factor = is_between_a2.float() * torch.exp(-dist_perp_sq_a2 / (2 * (0.5 * h_params['agent_radius'])**2))
    line_penalty = h_params['k_a2_shot_line_penalty'] * line_block_factor * proximity_factor_a2
    
    dense_reward[:, 1] += screen_reward + interference_reward + repulsion_reward - line_penalty

    # 2.4.3 防守者 (Defenders) 奖励/惩罚
    
    # 越过半场惩罚
    overextend_penalty = -h_params['k_overextend_penalty'] * torch.clamp(-defender_pos[..., 1], min=0.0)
    in_defensive_half = defender_pos[..., 1] >= 0
    
    # 理想站位奖励: 站在A1和篮筐之间
    ideal_pos = a1_pos.unsqueeze(1) + h_params['def_pos_offset'] * unit_vec_a1_to_basket.unsqueeze(1)
    dist_to_ideal = torch.linalg.norm(defender_pos - ideal_pos, dim=-1)
    base_pos_reward = h_params['k_positioning'] * torch.exp(-dist_to_ideal.pow(2) / (2 * h_params['def_pos_sigma']**2))
    soft_gate_def = torch.sigmoid(5.0 * torch.sum(vec_a1_to_defs * unit_vec_a1_to_basket.unsqueeze(1), dim=-1))
    positioning_reward = base_pos_reward * soft_gate_def * in_defensive_half.float()

    # 压迫奖励: 靠近A1施加压力
    pressure_factor = torch.clamp(1.0 - (dist_a1_to_defs / h_params['def_pressure_range']), min=0.0)
    pressure_reward = h_params['k_def_pressure_reward'] * (pressure_factor ** 2) * in_defensive_half.float() * soft_gate_def

    # A1突破惩罚: A1越深入禁区，防守方受罚越多
    penetration_penalty = -h_params['k_def_a1_penetration_penalty'] * (torch.clamp(a1_pos[:, 1], min=0.0) ** 2)
    
    # 区域控制奖励: 阻止A1向篮筐移动 & 占据投篮点附近
    is_guarding = in_defensive_half & (a1_pos[:, 1] > 0).unsqueeze(1) & (dist_a1_to_defs < h_params['def_guard_threshold'])
    radial_vel_to_spot = torch.sum(a1_vel.unsqueeze(1) * unit_vec_a1_to_basket.unsqueeze(1), dim=-1)
    spot_control_reward = h_params['k_spot_control_reward'] * (-torch.clamp(radial_vel_to_spot, max=0.0)) * is_guarding.float()
    
    dist_d_to_spot = torch.linalg.norm(defender_pos - spot_center_pos.unsqueeze(1), dim=-1)
    def_gaussian_reward = h_params['k_def_gaussian_spot'] * torch.exp(- (dist_d_to_spot**2) / (2 * h_params['def_gaussian_spot_sigma']**2)) * in_defensive_half.float()
    
    total_def_reward = overextend_penalty + positioning_reward + spot_control_reward + \
                       def_gaussian_reward + pressure_reward + penetration_penalty.unsqueeze(1)
    dense_reward[:, n_attackers:] += total_def_reward

    # --- 2.5 时间紧迫性惩罚/奖励 ---
    elapsed_time = h_params['t_limit'] - t_remaining.squeeze(-1)
    is_time_urgent = elapsed_time > h_params['time_penalty_grace_period']
    if torch.any(is_time_urgent):
        time_factor = (elapsed_time - h_params['time_penalty_grace_period'])**2
        
        # 进攻方惩罚 (如果时间紧迫但A1还未进入投篮区)
        is_stalling = is_time_urgent & ~is_in_spot_a1
        time_penalty_attackers = h_params['k_attacker_time_penalty'] * time_factor
        dense_reward[:, :n_attackers] -= time_penalty_attackers.unsqueeze(1) * is_stalling.unsqueeze(1)
        
        # 防守方奖励 (随着时间流逝，防守成功概率增加)
        time_bonus_defenders = h_params['k_defender_time_bonus'] * time_factor
        dense_reward[:, n_attackers:] += time_bonus_defenders.unsqueeze(1) * is_time_urgent.unsqueeze(1)

    return dense_reward, terminal_rewards, dones_out, curr_still_counter, wall_collision_counters, defender_over_midline_counter, attacker_win_this_step, reason_code