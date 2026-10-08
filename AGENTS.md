# Layup 场景奖励与判罚系统详解（当前实现）

> 本文描述 **`vmas/scenarios/layup.py` + `layup_jit.py` 当前代码**（2026-10-08 状态，commit `921f83f`）。
> 旧版文档（2026-03-06）描述的是「纯连续动作 + 双组课程 + 另一套奖励权重」，已完全不代表现状。
> **一切数值以代码为准**；本文每个数值后面都标注了它所在的键名，键名默认值集中在
> `layup.py` 第 80~677 行的 `self.h_params`（少数由任务 YAML `benchmarl/conf/task/vmas/layup.yaml` 覆盖）。
>
> 场景总览、观测布局、终局码经济账、训练侧结论见仓库根 `AGENTS.md`。

---

## 0. 读之前：奖励是如何被缩放与结算的

```python
# layup.py:1235  info() —— 日志口径
dense_reward    = 0.005 * dense_reward_factor * step_dense_rewards[agent]   # 0.005*0.1 = 5e-4
terminal_reward = 0.005 * terminal_rewards[agent]

# layup.py:1337  reward() —— 真正喂给算法的奖励
rew = 0.005 * (dense_reward_factor * step_dense_rewards[agent] + terminal_rewards[agent])
```

- **`dense_reward_factor = 0.1`**（`layup.py:355`），于是：
  - **稠密项最终权重 = 0.0005 × raw**（文中写作「×0.0005」）
  - **终局项最终权重 = 0.005 × raw**（文中写作「×0.005」，raw 值后面直接标出缩放后分数）
- 开局延迟期（`start_delay_frames = 10` 帧 = 1 s）内 **A1 奖励被强制为 0**（`layup.py:1344`），
  但 A2 / 防守方没有这个豁免。
- 奖励是 **per-agent** 的（`dense_reward[:, i]`），advantage 也是 per-agent；
  但 `termination_reason` 是 **全局码广播**给 4 个 agent（分析时必须取 `[..., 0, :]`）。

**一局的量级感**：整局 raw 稠密累计大约 ±几百，终局单项 raw 在 1000~18000（缩放后 ±5~±90）。
所以**终局奖励主导经济**，稠密项负责塑形与"别摆烂"。

---

## 1. 一局的时空与实体

| 项 | 值 | 位置 |
|---|---|---|
| 场地 | `W=8 m`（x）× `L=15 m`（y） | `layup.py:86-87` |
| 时间 | `t_limit=20 s`，`dt=0.1` ⇒ **200 步** | `layup.py:93-94` |
| 玩家半径 | `agent_radius=0.3 m` | `layup.py:107` |
| 动力 | `v_max=5 m/s`、`a_max=3 m/s²`（PID 速度控制器） | `layup.py:110-111` |
| 篮球区域 | `R_spot=0.9 m` 圆（**2026-10-03 由 1.2 缩小**） | `layup.py:90` |
| 篮筐 | `(0, L/2-0.6) = (0, 6.9)` | `layup.py:880-881` |
| 开局延迟 | 10 帧，A1 不动且不给奖励 | `layup.py:98` |

**出生点（`reset_world_at`，`layup.py:913-1000`）**

- **A1 固定**：`(-W/2+2r, -L/2+2r) = (-3.4, -6.9)`。
- **A2 随机**：x ∈ ±(W/2−r)，y ∈ `[-r-depth, -r]`（己方半场贴中线一带）。
- **D1/D2 随机**：x ∈ ±(W/2−1.5r)，y ∈ `[1.2r, depth-1.2r]`（对方半场贴中线一带，`depth=1.0`），
  并做最多 3 轮"两人最小间距 ≥ 2r×1.05"的重采样。
- `fixed_init=true` 时改为 A2 `(0,-0.65)`、D1 `(-2,0.5)`、D2 `(2,0.5)`（仅冒烟用，正式训练 `fixed_init=false`）。
- **投篮点随机**：三个 x 中心（`0, ±W/3.5 = ±2.29`）随机选一个，加 `N(0,0.6)` 抖动，
  y 中心 = `R_spot + L/8 = 2.775`，再夹到 `x∈±(W-R_spot)/2`、`y∈[R_spot, R_spot+L/4]`
  （`layup.py:884-905`，`fixed_spot=true` 才会固定到 `(0, 2.775)`）。

---

## 2. 动作空间：混合动作 + 投篮按键 + 掩码

**物理动作 3 通道**（`layup.py:703` `u_range=[v_max]*3`）：

| 通道 | 含义 | 谁用 |
|---|---|---|
| `u[0], u[1]` | 目标速度 x/y（m/s，**±5**） | 全员 |
| `u[2]` | **投篮按键**（>0.5 视为按下） | 仅 A1 有效，其余 agent 恒被忽略 |

**策略侧是复合动作空间**：`{continuous: [N,2], discrete: [N,1]}`（`BenchMARL/benchmarl/environments/layup/common.py`
的 `FlattenHybridAction._hybridize`）。连续叶子的**物理上下界 ±v_max 会从原始 spec 还原**
（`_continuous_bounds`），因此 `TanhNormal` 不是平凡 (−1,1) 边界 —— 这一点在 2026-10-04
曾因丢失边界导致"策略只能输出 1 m/s、log_prob 爆到 1e4"的 NaN 事故。

**`process_action`（`layup.py:1013-1074`）顺序**：

1. `target_vel = agent.action.u[:, :2]`（**线性映射，无 1.5 次方整形、无死区**；旧版有 `(mag/v_max)^0.5` 与 `norm<0.2` 死区，均已删除）。
2. **A1 按键分支**：`press = u[2] > 0.5`，`press_valid = press & is_in_spot_a1` ⇒ 本帧目标速度清零；
   `self.a1_press` 记录"本帧是否有效按下"（推进读条的唯一开关）。
3. 开局延迟期内 A1 目标速度强制 0。
4. `clamp_with_norm(·, v_max)` → 反解加速度 `(v_des − v)/dt` → `clamp_with_norm(·, a_max)` → `action.u = v + a·dt`。
5. **按下帧再把 `action.u` 置 0**（不直接改 `state.vel`）：制动力交给 `velocity_controller.process_force`
   的 PID 去实现"速度为 0"，保证物理一致性。

**动作掩码**（`layup.py:1076-1088` `get_action_mask()`）：返回 `[B, n_agents, 2]` bool，
`mask[:,:,0]=True`（"不按"恒可选）、`mask[:,0,1] = is_in_spot_a1`（**只有 A1 在圈内才允许按下**），
其余 agent 的离散位只能选"不按"。掩码在策略侧以 `logits.masked_fill(~mask, -1e9)` 注入
（`BenchMARL/benchmarl/algorithms/mappo.py`），且环境侧**必须同时写根键与 `("next", …)` 键**，
否则会被 `step_mdp()` 用旧值覆盖（`common.py::_write_action_mask`）。

**按键语义**：按住 = 开始/继续读条；**松开或出圈或不再满足静止条件 ⇒ 读条清零**；满 10 帧自动出手。

---

## 3. 读条与出手判定（`layup_jit.py:74-250`）

```python
in_area          = (dist_a1_to_spot <= R_spot) & (a1_pos.y > 0)
is_still         = |a1_vel| < v_shot_threshold                     # 课程阈值，训练 0.6→0.2 / 评测恒 0.2
not_accelerating = (|raw_action_A1| < a_shot_threshold=2.0) | is_braking
is_ready_to_shoot = in_area & is_still & not_accelerating
press_active      = a1_press & is_ready_to_shoot & ~done
curr_counter      = press_active ? prev_counter + 1 : 0             # 读条
shot_attempted    = curr_counter >= shot_still_frames (=10)
```

- **10 帧门槛不可降低**（用户硬约束：模拟机械准备过程，防守方有 10 帧补救窗口）。
- **课程**：训练环境从 `VMAS_INITIAL_SHOT_THRESHOLD` 起步（cold=1.2、cont=0.2），
  每累计 `batch_dim//10` 次成功就 `−0.05`，单调降到目标 `0.2`；**评测环境恒用 0.2**（`layup.py:118-153`）。

### 3.1 封盖因子与命中/被盖（终止码 1 / 11）

对每个防守者独立计算，再求和（`layup_jit.py:104-166`）：

```python
shot_vec   = basket − a1_pos
proj_ratio = ⟨def − a1_pos, shot_vec⟩ / |shot_vec|²
is_between = 0 < proj_ratio < 1                       # 硬门控：必须在 A1→篮筐之间
dist_perp  = |def − (a1_pos + proj_ratio·shot_vec)|   # 到投篮线的垂距
soft_prox  = sigmoid(block_gate_k(25) · (def_proximity_threshold(0.9) − |def − a1_pos|))
vel_gate   = sigmoid(block_vel_gate_k(10) · (v_block_threshold(0.5) − |v_def|))
block_i    = exp(−dist_perp²/(2·block_sigma(0.30)²)) · is_between · soft_prox · vel_gate
total_block = clamp(Σ_i block_i, 0, 1)
命中 ⟺ total_block < win_condition_block_threshold(0.5)
```

- **速度门控**（`v_block_threshold`/`block_vel_gate_k`）专门堵"读条期间冲刺扑盖"的奖励黑客。
- **几何干扰度 `contest_geom`** = 同一式子但**不含速度门控**：冲刺补防（good close-out）也算有效干扰，
  用于延误奖励打折与放投罚分减免（见 §4.2）。
- ⚠️ **封盖只统计防守者**：A2 挡在出手线路上**不会**降低命中率 —— 这正是" A2 卡位 + A1 背后出手 "
  必胜套路的结构性成因之一（见 §8）。

### 3.2 出手瞬间的奖励分配（raw 值）

**A1**（`layup_jit.py:175-190`）：

```
base_score  = max_score(8000) · (1 − d_spot/R_spot)                     # 离点越近越高
final_mod   = base_score · (1 − total_block)
time_bonus  = k_time_bonus(8000) · (t_remaining/t_limit) · (1 − total_block)
spacing     = k_spacing_bonus(1000) · mean(|def_i − a1|)
vel_still   = k_shot_stillness_vel_bonus(1000) · exp(−|v_a1|)
act_still   = k_shot_stillness_act_bonus(0) · exp(−|u_a1|)             # 当前为 0
A1_reward   = final_mod + spacing + time_bonus + vel_still + act_still
              + shoot_score(9000) − blocked_penalty
blocked_penalty = k_blocked_shot_penalty(12000) · (1 − 命中)           # 被盖 → −60
```

**A2**（`:100-102, :192-212`）：

```
A2_reward = (final_mod + screen_bonus + spacing + time_bonus + shoot_score) · 1[ A2 在对方半场 ]
            + k_a2_shot_pos_bonus(3000) · 1[ A2 在对方半场 ]            # 出手奖励
            − k_blocked_shot_penalty_a2(18000) · (1 − 命中)             # 被盖 → −90（比 A1 更重）
screen_bonus = k_a2_screen_bonus(3000) · exp(−|a2 − ideal_screen|²/(2·a2_screen_sigma²)) · gate
ideal_screen = 最近防守者 + screen_pos_offset(0.9) · unit(防守者→A1)
gate         = sigmoid(−k_screen_gate(7) · ⟨a2→最近防守, a2→A1⟩)        # A2 在 A1 与防守者之间
```

**防守方（逐人）**（`:214-248`）：

```
R_block       = k_def_block_reward(7000) · block_contribution_i         # 含速度门控的封盖贡献
R_force       = k_def_force_reward(2000) · (d_spot/R_spot)              # 迫使 A1 远离投篮点
R_positioning = k_def_pos_reward(100) · exp(−d_ideal²/(2·def_pos_sigma²)) · pos_gate   # 站 A1→篮筐连线后侧
R_area        = k_def_area_reward(150) · exp(−|def−spot|²/(2·def_gaussian_spot_sigma²))
R_delay_i     = k_def_delay_bonus(14000) · r^1.5 · (k_def_delay_floor(0.5) + 0.5·contest_i)
shot_penalty_i= k_def_shot_penalty(9000) · (1 − k_def_contest_discount_max(0.10)·contest_i)
合计 = R_block + R_force + R_positioning + R_area − shot_penalty_i + R_delay_i
其中 r = clamp((t_limit − t_remaining)/t_limit, 0, 1)
```

**设计要点**：
- **延误奖励用 `r^1.5` 而非 `r²`**：抬高"中后段把出手往后拖"的斜率。
- **延误奖励有 50% 保底**（`k_def_delay_floor`）："延误永远好过不延误"，即使没贴在盖帽位置也发一半；
  另一半按出手瞬间几何干扰度发放 ⇒ "退开看投"只拿保底。
- **放投基础罚分只允许被干扰度减免 ≤10%**（`k_def_contest_discount_max`）：保证防守整体仍是**净负**期望。
- 出手事件下防守方总账（实测）约为 **−45（基础罚）+ ≤+35（延误）+ ≤+35（封盖贡献）+ ≤+10（逼离）**，
  即从"看着对手投进"到"贴脸干扰"跨越约 40 分。

---

## 4. 终局码（权威表，`WIN_CODES = {1,2,3,4,5}`）

| 码 | 名称 | 触发（`layup_jit.py`） | 主要奖惩（raw） |
|---|---|---|---|
| **1** | 投篮命中（胜） | `total_block < 0.5`（`:162-164`） | A1≈+8000~+19000；A2 同向；防守 −9000± |
| **11** | 投篮被盖（负） | `total_block ≥ 0.5`（`:165-166`） | 同上公式 + A1 −12000、A2 −18000 额外扣 |
| **12** | 进攻超时（负） | `t_remaining ≤ 0`（`:252-298`） | A1∈[−10000, +10000]；防守恒 `+9000`；A2 若在己方半场再 `−2500·|y|` |
| **2** | 对手犯规（胜） | 见 §5 判责，责任在防守方（`:465-467`） | 防守主动方 −magnitude；被犯攻方 +magnitude；A2 近静止再 +6000 |
| **13** | 己方犯规（负） | 同上，责任在进攻方（`:468`） | 攻方主动方 −magnitude×**1.5**；队友再分摊 −0.3·magnitude |
| **5** | 对手友军误伤（胜） | 防守方两人互撞（`:491-492`） | 双方各 −magnitude×**2.0** |
| **15** | 己方友军误伤（负） | 进攻方两人互撞（`:493`） | 双方各 −magnitude×**2.0** |
| **3** | 对手失误-撞墙（胜） | 防守方贴墙（`:522-525`） | 贴墙者 −16000 |
| **14** | 己方失误-撞墙（负） | 进攻方贴墙（`:528-530`） | 贴墙者 −16000 |
| **4** | 对手失误-越线（胜） | 防守方越中线 ≥5 帧（`:548-554`） | 越线防守者 −18000 |

**超时（码 12）细节**（`:256-295`）：

```
vel_penalty      = k_timeout_move_vel_penalty(300) · |v_a1|
reward_in_spot   = attacker_timeout_reward_in_spot(−5000) − vel_penalty
reward_out_spot  = attacker_timeout_base_reward_out_spot(−7000) − k_timeout_dist_reward_factor(1000)·d_spot
A1 = clamp(上述其一, ±attacker_timeout_reward_max(10000))     # A2 同值
防守 = defender_timeout_reward(9000)（固定）
A2 若 y<0：再 − k_a2_stalling_penalty_timeup(2500) · |y_a2|
```

**撞墙（码 3/14）细节**（`:498-542`）：两条并行规则 ——
① 贴墙计数 `≥ wall_collision_frames(20)`；② **首帧贴墙且法向撞击速度 > `v_wall_crash_threshold(0.5)` 立即判负**
（堵"蹭墙免费急刹"）。罚分只给"当前正贴在墙上"的 agent（`R_wall_collision_penalty = −16000`）。

**越线（码 4）细节**（`:544-564`）：防守方 `y<0` 连续 `max_time_over_midline(5)` 帧 ⇒ 立即判负，
只罚"当前仍在越线"的防守者（`R_midline_foul = 18000`）。
⚠️ 另有 `:566-587` 的兜底规则：**任何回合结束时，若防守者此刻正处 y<0，其终局奖励被强制覆写为 −18000**
（不论本局因何结束）—— 这是"越线重罚"，不是笔误。

---

## 5. 判罚系统（六条规则，`layup_jit.py:300-496`）

### 规则 1：读条期碰 A1 ⇒ 防守犯规（码 2，无门槛）

```python
is_charging        = (curr_still_counter > 0) & ~shot_attempted
charging_foul      = is_charging & collision(A1, 任意防守者) & ~done
⇒ 立即终局、进攻方胜、码 2
⇒ 只罚"真正碰到 A1 的那位防守者"：其终局奖励 → −R_foul(=8000)
⇒ 攻方两人各 +max_score(=8000)
```

**用户硬约束：读条期间触碰绝对不允许判攻方**，且这里**没有接近分量门槛**（碰到就算）。

### 规则 2：接触判责（对称接近分量 + 篮球式豁免）

对每一对接触的 (i, j)（`torch.triu`，每对只判一次）：

```python
contact_dir = unit(p_j − p_i)                       # 沿接触连线
approach_i  = clamp(⟨v_i^prev, contact_dir⟩, 0)     # 用"接触前速度" p_vels
approach_j  = clamp(−⟨v_j^prev, contact_dir⟩, 0)
# ① 对称基线：接近分量大者主动；平手（差 < foul_approach_eps=1e-3）时整体速度大者主动
# ② 篮球式豁免（仅跨队）：violation_D = max(0, |v_D| − foul_legal_def_speed(0.4))
#                                        + max(0, approach_D − foul_legal_def_approach(0.4))
#    score_A = approach_A − k_legal_def_exempt(1.2) · violation_D
#    若 score_A ≤ approach_D ⇒ 责任改判给防守方（防守方"没站稳"，进攻方豁免）
# ③ 门槛：approach_max > foul_approach_threshold(0.35) 才算严重犯规；
#    横滑擦身（双方都不朝对方去）不判犯规，交给稠密层的近距/推挤惩罚
magnitude = R_foul(8000) + k_foul_vel_penalty(800) · approach_max
```

**发放账本**（敌对犯规，`:407-468`）：

| 角色 | 收/支 |
|---|---|
| 主动方 | `−magnitude × (攻方 1.5 / 守方 1.0)`（`k_attacker_active_foul_scale`） |
| 被动方 | `+magnitude × foul_teammate_factor(1.0)` |
| **主动方的队友** | `−magnitude × k_teammate_foul_share(0.3)`（跨智能体记账，防止"A2 犯规、A1 净赚"） |
| A2 作为被动方且 `|v| < foul_draw_speed_threshold(0.3)` | 再 `+k_foul_drawing_bonus(6000)`（站定造犯规） |
| 攻方主动犯规导致终局（防守方被动） | 全体防守方再 `+defender_fouled_bonus(2500)`（补偿被剥夺的拖延时间收益） |

- **友军误伤**（同队相撞）：**双方各 −magnitude×2.0**（`k_friendly_fire_scale`），比任何犯规都亏；
  且**与谁主动无关，双方同罚**（`:470-493`）。
- **判罚经济学（实测，iter450 buffer）**：造进攻犯规被撞者 `+54.5` > 逼出超时 `+45` > 出手事件 `≈−45~+26`。
  这也解释了为什么防守方"宁愿被撞也不愿放投"。

### 规则 3-6（已并入上表）

3. **撞墙**（码 3/14，`R_wall_collision_penalty`）；4. **越线**（码 4 / 终局覆写，`R_midline_foul`）；
5. **友军误伤**（码 5/15）；6. **超时**（码 12）。

---

## 6. 稠密奖励（每步，×0.0005 后生效）

### 6.1 通用项（全员，`layup_jit.py:600-719`）

| 项 | 公式 | 系数（`layup.py`） |
|---|---|---|
| 出界 | `oob_penalty · margin · (depth_x+depth_y) · (|v|+1)`，`logaddexp` 平滑 | `oob_penalty=−3000`，`oob_margin=0.05` |
| 动作幅度 | `−k_u_penalty_general · |u|` | `0.1` |
| 动作超限 | 超 `0.95·v_max` 的部分按比例罚 | `k_action_access_max_penalty=1` |
| 刹车 | `−k_brake_usage_penalty · 1[brake]`（+超限项） | `0.1` |
| 矛盾动作 | `−k_conflicting_action_penalty · |u| · 1[brake]` | `1.0` |
| 超动力极限 | `−k_excess_acceleration_penalty · max(0, |a_req|−a_max)`（刹车豁免） | `0.001` |
| 动作抖动 | `−k_action_jerk_penalty · |u − u_prev|` | `0.01` |
| 近距 | `−k_prox · penetration`，`k_prox` 分角色（A1 60 / A2 60 / 防守 60，圈内防守 ×0.8） | `k_a1_proximity_penalty=60`，`k_proximity_penalty=60` |
| 高速碰撞 | `vel_proj < 0` 方（**撞人方**）`−k_coll_active(5.0)·|Δv|`，被撞方 `−0.1·|Δv|` | `k_coll_active=5.0`，`k_coll_passive=0.1` |
| 低速推挤 | 攻方 `−k_push_penalty(120)`、守方 `−k_def_push_penalty(120)` × 朝向对方的速度分量 | |
| **造犯规（站定）** | `k_stand_still_reward(150) · role_scale · 对方接近速度 · 1[自己近静止] · 1[对方在 1.8 m 内]` | 攻方 `role_scale=2.0` |
| 加速度极限 | 见上 | |

> **碰撞符号**：`pos_rel[b,i,j] = p_i − p_j`，`vel_proj = v_i·(p_i−p_j)`；
> `vel_proj < 0` 表示 i 正沿接触线压向 j（撞人方）。2026-09-30 修过一次**写反**的 bug。

### 6.2 A1（持球人，`:731-854`）

| 项 | 公式 | 系数 |
|---|---|---|
| 高斯吸引 | `gaussian_scale · exp(−d_spot²/(2·gaussian_sigma²))` | `600`，`σ=0.5·R_spot=0.45` |
| 朝点速度 | `a1_normalized_speed_k · ⟨v, unit(spot−a1)⟩`，其中 `k = k_a1_speed_spot_reward(6000)/d_initial` | 每局按初始距离归一 |
| 圈内存在 | `k_a1_in_spot_reward(3.0) · (1.5 − d/R_spot)`（仅圈内） | `3.0` |
| 被封盖 | `−k_a1_blocked_penalty(90) · total_block_factor_a1` | `−90` |
| 犹豫 | `−k_hesitation_penalty(90) · clamp(1 − |v|/0.5, 0) · 1[圈外]` | `90` |
| **静止/摆脱动态加权** | `(1−bf)·stillness + bf·separation`，`bf=total_block_factor_a1` | 见下 |
| ↳ 静止（圈内） | `brake(20)·1[brake] + vel_still(20)·exp(−|v|²/2·0.4²) + act_still(50)·exp(−|u|²/2·0.3²)·1[|u|<0.9]` | `20/20/50` |
| ↳ 摆脱 | `k_a1_separation_reward(20) · max(0, ⟨v, unit(a1−最近防守)⟩)` | `20` |
| **走廊净空** | `k_a1_lane_clear_reward(600) · lane_clear · lane_threat · advance_norm` | `600` |
| 横移残量 | `k_a1_tangential_reward(150) · 切向速度 · lateral_gate(≤2) · advance_norm` | `150` |
| 读条奖（递增） | `k_a1_ready_to_shoot_reward(200) · (counter/10)` | `200` |
| 放弃罚 | `−0.5·200 · (prev_counter/10) · 1[本帧清零且上帧>0]` | 与进度等比 |

- **`advance_norm = max(0, ⟨v, unit(走廊)⟩)/v_max`**：原地摇摆 / 背向目标 ⇒ 横移与走廊奖励都为 0
  （这是 2026-10 治理"原地摇摆刷分"的关键门控）。
- 走廊基准会切换：**圈外走 A1→投篮点，圈内走 A1→篮筐**（`:782`）。

### 6.3 A2（掩护者，`:856-989`）

| 项 | 公式 | 系数 |
|---|---|---|
| **出手通道清空（主项）** | `k_a2_lane_clear(400) · Σ_i (1−block_i)·w(a2,D_i)·threat_i` | `400` |
| 理想掩护位 | `k_ideal_screen_pos(200) · exp(−d_ideal²/(2·σ²)) · pos_gate · spacing_gate`（取最大） | `200` |
| 干扰（硬门控） | `k_a2_interference_reward(40) · exp(−d²/…) · between_gate` | `40` |
| 卡位 | `k_a2_body_check(300) · block_share · exp(−d²) · between_gate · exp(−|v_a2|²/…)` | `300` |
| 排斥 | `k_repulsion_reward(200) · max(0, 防守远离 A1 的速度)`（仅 `d < R_spot`） | `200` |
| 挡线罚 | `−k_a2_shot_line_penalty(90) · line_block · proximity` | `−90`（**唯一反制，上限 −0.045/步**） |
| 拉开空间 | `k_a2_spread_reward(200) · max(0, 盯防者远离 A1 速度) + k_a2_spread_penalty(60)·clamp(负向, −1, 0)`（A1 过线后） | `200/60` |
| 走廊支援 | `k_a2_support_reward(150) · near_corridor · exp(−tb²/2·0.35²)`（A1 过线后） | `150` |
| 即时造撞 | `k_a2_draw_immediate(300) · min(防守朝 A2 速度, 3.0) · (1 − d/1.8)` | `300` |
| 给 A1 让路 | `−k_a2_friendly_clearance_penalty(1000) · max(0, 0.9 − d(a2,a1))` | `−1000`（**不参与 `k_a2_dense_scale` 缩放**） |
| 过中线门控 | `y_a2 ≥ 0` 才保留上述奖励；否则整体替换为 `−100.0 · |y_a2|` | `k_a2_stalling_penalty=100` |

- **`between_gate`（硬门控）**：A2 必须在"防守者→A1"连线上（垂距高斯 × 两侧 sigmoid），
  仅仅"靠近防守者"不再给钱。
- `k_a2_dense_scale = 1.0` 可整体缩放 A2 的角色特有项（通用物理惩罚与过线门控不受影响）。

### 6.4 防守方（`:991-1047`）

| 项 | 公式 | 系数 |
|---|---|---|
| 越线惩罚（稠密） | `−k_overextend_penalty(240) · max(0, −y)` | `240` |
| 站位 | `k_positioning(140) · exp(−d_ideal²/(2·def_pos_sigma²)) · soft_gate · 1[己方半场]` | `140` |
| 压迫 | `k_def_pressure_reward(30) · (1 − d/1.8)² · soft_gate · 1[己方半场]` | `30` |
| **A1 深入惩罚** | `−k_def_a1_penetration_penalty(30) · max(0, y_a1)²` | `30` |
| 阻止推进 | `k_spot_control_reward(100) · (−径向速度)`（仅 A1 过线后、且在盯防距离内） | `100` |
| 投篮点高斯 | `k_def_gaussian_spot(30) · exp(−d_def,spot²/(2·R_spot²)) · 1[己方半场]` | `30` |
| **读条期稳定站位** | `k_def_charging_stability(100) · position_quality · 1[A1 读条] · 1[|v_def|<0.5] · 1[己方半场]` | `100` |

- **理想站位会随 A1 位置切换**（`:997-1013`）：A1 过线 ⇒ `A1 + 0.75·unit(A1→篮筐)`（人球篮三点一线）；
  A1 未过线 ⇒ `(x_a1, r)` 对位不动。
- `soft_gate = sigmoid(5·⟨def−a1, unit(a1→basket)⟩)`：要求防守者站在 A1 **朝向篮筐的一侧**。
- 读条期稳定站位奖励（`k_def_charging_stability`）是给"提前占位、别冲刺扑盖"的正向引导，
  与出手判定的速度门控互为表里。

### 6.5 时间压力（`:1049-1062`）

```
elapsed = t_limit − t_remaining
1[elapsed > 8]:                                         # time_penalty_grace_period
  进攻方（A1 与 A2）− k_attacker_time_penalty(1.5) · (elapsed−8)² · 1[A1 不在圈内]
  防守方          + k_defender_time_bonus(0.5) · (elapsed−8)²
```

- 15 s 处 raw ≈ −0.5·… 累计整段"故意拖满"约 4.3 分（缩放后），**远小于终局 ±50**，
  因此它是"催促"而非"决定胜负"的量级；`1.5` 是 2026-10-03 从 0.7 调回的（画面"散步"）。

---

## 7. 感知噪声（模拟真实传感器，`layup.py:198-209, 795-838`）

- 对象：**队友 / 两名对手**的相对位置、相对速度、绝对位置、绝对速度。
- 标准差随距离线性增长（角度误差近似恒定）：
  ```
  σ_pos = perception_noise_floor(0.05) + k_perception_noise(0.02) · d      [m]
  σ_vel = perception_noise_vel_floor(0.02) + k_perception_noise_vel(0.01) · d  [m/s]
  ```
- **同一份含噪观测同时用于相对量与绝对量** ⇒ 两者一致，无法互相抵消还原真值。
- **不加噪**：自身本体感受、`spot`/`basket` 固定地标、critic 的全局 state。
- 实现：`_refresh_perception()` 在 `reset_world_at` 与 `post_step` 各刷新一次，
  `observation()` 只做切片（比 per-agent 采样省约 2/3 开销）；实测 **0.91 ms/步 ≈ 0.18 s/轮**（可忽略）。
- 关掉噪声需把四个系数同时置 0（`_noise_enabled` 由它们决定），否则 `randn` 分支照跑。

**观测布局（41 维，`layup.py:1420-1490`）**

| 段 | 维度 | 内容（归一化） | 备注 |
|---|---|---|---|
| self | 0:2 / 2:4 / 4:6 | 位置 `/±(W/2,L/2)`、速度 `/v_max`、**上一帧目标速度 `/v_max`** | 让策略能闭环"保持小动作" |
| A1 状态 | 6 / 7 / 8 | `is_in_spot_a1`、`读条进度 counter/10`、`t_remaining/t_limit` | 后两维对所有人可见 |
| teammate | 9:17 | 相对位置 `/±(W,L)`、相对速度 `/(2v_max)`、绝对位置、绝对速度 | 顺序：队友 |
| opp1 | 17:25 | 同上 | 顺序：对手 1 |
| opp2 | 25:33 | 同上 | 顺序：对手 2 |
| spot | 33:37 | 相对位置、绝对位置 | **防守方全 0** |
| basket | 37:41 | 相对位置、绝对位置 | |

- **观测与全局状态均以 fp16 存储**（省内存），模型入口转回 fp32。
- **critic 的全局 state = 23 维**（`layup.py:1259-1330`）：
  `A1(pos2+vel2+in_spot+progress=6) + A2(4) + D1(4) + D2(4) + spot(2) + basket(2) + time(1)`。

---

## 8. 已知问题：奖励/规则层的结构性漏洞

### 8.1 A2"占住封盖位 + A1 背后出手"（用户提出，尚未修）

- **现象**：A2 卡在防守者与 A1 之间的出手线路上，防守者被规则与物理同时锁死 ⇒ 攻方近乎无解。
- **代码成因**（三处不对称）：
  1. A2 占位收益 `lane_clear 400` + `body_check 300` + `ideal_screen_pos 200` + `draw_immediate 300`
     ≫ 唯一反制 `k_a2_shot_line_penalty = 90`（上限 −0.045/步） ⇒ **净赚**；
  2. **`block_factor` 只统计防守者**，A2 挡视线不降低命中率；
  3. 读条期触碰**无门槛**即判防守犯规，防守者不敢贴身；
  4. `def_proximity_threshold(0.9)` + `block_gate_k(25)` 让"站位不算近"的防守者完全没有封盖贡献。
- **候选修法**：
  - **F1** 提高挡视线罚（`k_a2_shot_line_penalty 90 → 3000~4000`，垂距 σ `0.15→0.35~0.45`，距 A1 σ `0.6→1.2`）；
  - **F2** 命中判定计入队友遮挡 / 判非法掩护（治根）；
  - **F3** 给守方活路（`def_proximity_threshold 0.9→1.2`、`block_gate_k 25→15`）；
  - **F4** 收紧 A2 的 `lane_clear` 权重。
  - （建议 F1+F3，**等用户拍板**。）

### 8.2 其他已知失衡

- **贴身防守的期望尚未真正为负**：实测 D 贴 A1 <1.2 m 时 advantage 仍为 `+0.15`。
  用户允许的软化上限是"少 10% 顶天"（`k_def_contest_discount_max=0.10` 即此约束的落地）。
- **造犯规可能压过逼超时**：实测被撞者 `+54.5` > 逼超时 `+45`，防守方可能偏向"站定挨撞"。

---

## 9. 调参索引（想改哪个行为，去改哪里）

| 想改… | 键（`layup.py` 行） | 影响面 |
|---|---|---|
| 投篮点难度 | `R_spot`（:90） | 读条区、按键掩码、高斯/封盖 σ、出手 base_score 全部联动 |
| 一局时长 | `t_limit`（:93） | 步数 = t_limit/dt，所有时间归一化随之变化 |
| 读条长度 | `shot_still_frames`（:157） | **用户硬约束：10 帧不可降** |
| 出手静止门槛 | `v_shot_threshold`（目标 0.2）、课程起点用 `VMAS_INITIAL_SHOT_THRESHOLD` | 训练难度曲线 |
| 命中判定 | `win_condition_block_threshold`(0.5)、`block_sigma`(0.30)、`block_gate_k`(25)、`def_proximity_threshold`(0.9)、`v_block_threshold`(0.5) | §3.1 |
| 判罚 | `foul_approach_threshold`(0.35)、`foul_legal_def_speed`(0.4)、`foul_legal_def_approach`(0.4)、`k_legal_def_exempt`(1.2)、`R_foul`(8000)、`k_foul_vel_penalty`(800)、`k_attacker_active_foul_scale`(1.5)、`k_teammate_foul_share`(0.3)、`k_friendly_fire_scale`(2.0) | §5 |
| 失误终局 | `wall_collision_frames`(20)、`v_wall_crash_threshold`(0.5)、`R_wall_collision_penalty`(−16000)、`max_time_over_midline`(5)、`R_midline_foul`(18000) | §4 |
| 防守经济 | `k_def_delay_bonus`(14000)、`k_def_delay_floor`(0.50)、`k_def_shot_penalty`(9000)、`k_def_contest_discount_max`(0.10)、`k_def_block_reward`(7000) | §3.2 |
| A1 塑形 | `k_a1_lane_clear_reward`(600)、`k_a1_tangential_reward`(150)、`k_a1_separation_reward`(20)、`k_hesitation_penalty`(90)、`gaussian_scale`(600)、`k_a1_speed_spot_reward`(6000) | §6.2 |
| A2 塑形 | `k_a2_lane_clear`(400)、`k_a2_body_check`(300)、`k_ideal_screen_pos`(200)、`k_a2_shot_line_penalty`(90)、`k_a2_spread_reward`(200)、`k_a2_support_reward`(150)、`k_a2_draw_immediate`(300)、`k_a2_dense_scale`(1.0) | §6.3 |
| 感知噪声 | `k_perception_noise`(0.02)、`perception_noise_floor`(0.05)、`k_perception_noise_vel`(0.01)、`perception_noise_vel_floor`(0.02) | §7 |
| 全局缩放 | `dense_reward_factor`(0.1) + `layup.py:1337` 的 `0.005` | 稠密 vs 终局权重比 |

> **改完记得同步**：仓库根 `AGENTS.md`（§3.5/§3.6 汇总）、本文件、以及 `benchmarl/conf/**` 中若有的覆盖值。
> **改了观测/state 维度 ⇒ 所有 checkpoint 不兼容，必须 cold 起。**

---

## 10. 历史沿革（为什么现在是这个样子）

| 时间 | 变更 | 动机 |
|---|---|---|
| 2026-09 下旬 | 抬高投篮类终局（`max_score` 6000→8000），超时惩罚从 −100 一路加重到 −7000 | 让"投进"显著优于"蹲点超时"，对齐胜率 |
| 2026-09-30 | 修复碰撞惩罚符号（`vel_proj < 0` 才是撞人方） | 原实现让追尾的被撞方吃重罚 |
| 2026-10-03 | `t_limit` 15→20 s；`R_spot` 1.2→0.9 m；被盖惩罚拆分（A1 −12000 / A2 −18000） | 更多博弈空间 / 提高难度 / 让掩护失职的 A2 担责 |
| 2026-10-04 | A2 结果导向化（`lane_clear` 主项 + 硬门控 `between_gate`）、`k_ideal_screen_pos` 30→200、`k_a2_draw_immediate` 新增 | "按手写点站位"≠有效掩护，改用"防守有没有挡住通道"作判据 |
| 2026-10-04 | 判罚重构：`|Δv|` 门槛 → **接近分量**门槛 + 篮球式豁免；门槛 0.35 | 旧门槛判的 86% 是"横滑擦身"，真正的追击型接触反而漏判 |
| 2026-10-04/06 | 读条期犯规改为**逐防守者掩码**；`k_legal_def_speed/approach` 0.3→0.4、`k_legal_def_exempt` 1.5→1.2 | 用户："读条期间碰到 A1 的防守方扣分，不是两个都扣"；减轻判责偏向守方 |
| 2026-10-04 | 延误奖励 `r²`→`r^1.5` + 50% 保底 + 干扰度打折；放投罚分最多减 10% | "延误永远好过不延误"，同时保证防守整体净负期望 |
| 2026-10-05/06 | `k_a1_tangential_reward` 900→150、新增 `k_a1_lane_clear_reward`（走廊净空）并乘 `advance_norm` | 治理视频里"原地摇摆刷分" |
| 2026-10-06 | 加感知噪声 | 让策略不能靠"读绝对真值"过拟合 |
| 2026-10-06 | 混合动作（A1 键 + 掩码 + 按下刹车 + 去死区线性映射 + `±v_max` 边界还原） | 用户定案：按住读条、圈内才可按、非 A1 只输出 2 维 |

**已判死、勿重试**：VMAS 端 `torch.compile` / `torch.jit.script`（奖励 87× 慢、`env.step` 24× 慢、
报 `Tensor cannot be used as a tuple`）；CUDA 采样（比 CPU 慢约 1.75×）。
