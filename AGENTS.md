# alphaRM — RM 3v3 规则、观测与奖励实现详解

**项目**：**alphaRM**
**权威规则书**：[`../rm_simplified_rules_v1.1.md`](../rm_simplified_rules_v1.1.md)
**适用范围**：`VectorizedMultiAgentSimulator/vmas/scenarios/rm*.py` 当前实现
**最后更新**：2026-10-24

> 本文件描述 alphaRM 的 **RM 3v3** 场景。
> 规则正文以根目录规则书为准；代码数值以 `rm_config.py` 为准。
> 本文解释“代码是怎么实现的、为什么这样写、哪里容易错”。

---

## 1. 文件职责

| 文件 | 职责 |
|---|---|
| `vmas/scenarios/rm.py` | 场景生命周期、`reset_world_at`、`step`、动作写入、渲染、fire mask 接口 |
| `vmas/scenarios/rm_rules.py` | **规则核心**：reset、动作解码、碰撞/推挤/伤害、观测 140、state 191、奖励、终局、可见性调用 |
| `vmas/scenarios/rm_config.py` | 全部规则/奖励/地图/出生点常量（`RmScenarioConfig`） |
| `vmas/scenarios/rm_geometry.py` | 射线、`ray_edges`、`visibility_arc`、障碍距离、几何工具 |
| `BenchMARL/benchmarl/environments/rm/` | TensorDict 包装、group spec、fire_mask spec |
| `BenchMARL/benchmarl/models/rm_encoder.py` | actor：10 token + Transformer + GRU |
| `BenchMARL/benchmarl/models/rm_critic.py` | critic：特权 state + 队视角 canonical + 共享 value head |

---

## 2. 一局的时空与实体

| 项 | 当前值 |
|---|---|
| 场地 | 12 m × 8 m，中心 `(0,0)`（`field_hx=6.0`、`field_hy=4.0`） |
| 队伍 | 红方 3 台（`team=0`，`-x` 半场），蓝方 3 台（`team=1`，`+x` 半场） |
| 时间 | `episode_time=180 s`，`dt=0.1` ⇒ 1800 步 |
| 目标分 | 90 |
| 机器人半径 | 0.3 m |
| 最大速度 | 3 m/s |
| 复活 | `respawn_time=5 s`；在尸体位置复活 |
| 复活无敌 | `invuln_time=3 s`；无敌结束后枪口进入过热门锁 |
| 出生区回血 | 自家出生区内 `heal_per_frame=8.125`（约 81 HP/s） |

红/蓝动作在世界系下镜像：蓝方内部做 180° canonicalization，网络看到的观测和动作
都是队视角一致的；环境执行时再转回世界系。

---

## 3. fixed / random / late-game reset

### 3.1 fixed

`_fixed_reset_rows(s, rows, seed)` 按固定 slot 放置：

- 红方 slots 约为 `(-5.35, 3.30), (-5.35, 2.20), (-4.50, 2.75)`；
- 蓝方是 180° 镜像；
- 朝向中心；
- `fixed_slot_permute=True` 时随机排列三个 slot；
- `fixed_jitter` / `fixed_yaw_jitter` 可加小幅位置/朝向抖动；
- 非法间距会用 `rsi_min_sep=1.0` 做拒绝采样，失败则回退精确 permuted slots。

> **fixed 初始互相不可见是设计**：中间 L 形掩体遮挡双方出生点。reset 时
> `target=none`、`fire=release` 是正常的；策略必须先移动绕过掩体，才能锁定/开火。
> 不要把这一点误判为观测 bug。

### 3.2 random / mixed

`_random_reset_rows(mode)` 支持：

- 每队一个 anchor 在本方出生区附近（`spawn_anchor_jitter=0.45`），其余两个随机；
- 50% 镜像 / 50% 非镜像；
- 随机 yaw、HP、剩余时间、热量；
- `rsi_min_sep=1.0` 最小间距；
- anchor 的 agent 编号随机化，避免“红 0 永远 anchor”。

### 3.3 late-game

`rsi_late_game=True` 时：

- `episode_time` 仍是 **180 s**；
- 随机剩余 `remaining ∈ [rsi_remaining_min, rsi_remaining_max] = [30, 60]`；
- `s.time = remaining`；
- `episode_start_elapsed = 180 - remaining`；
- `episode_elapsed` 和 `stagnation_last_event` 都从该 offset 开始。

**不要把 `rsi_time` 当成总时长 30–60 s**；那会改变时间尺度，和正式比赛不一致。

### 3.4 curriculum 与角色持久化

`init_mode=curriculum` 用 `fixed_prob` 控制 fixed/random 比例，并支持
**persistent roles**：

- `curriculum_persistent_roles=True`（默认）：首次 reset 按 `round(B*p)` 精确分配
  fixed/random 角色，之后每个 env 保持角色；
- 旧行为（每次 reset 独立抽样）可通过 `False` 恢复；
- 注意：fixed 局通常比 random 局短，如果每次独立抽样，**任意时刻 batch 里 fixed
  占比会被短局压低**，不要只看配置里的 90%。

---

## 4. 动作、target mask 与 fire mask

### 4.1 动作语义

策略输出 6 维：

| 维度 | 语义 | 备注 |
|---|---|---|
| 0,1 | `vx,vy` 世界系平移 | 蓝方内部 180° 反变换 |
| 2 | `yaw` 云台转向 | 锁定目标时 `yaw_auto` 可能覆盖 |
| 3 | `spin` 小陀螺强度/开关 | 1.0 s 上限 |
| 4 | `target` | 4 类：none/enemy1/enemy2/enemy3 |
| 5 | `fire` | 2 类：release/press |

环境最终 `u` 是 6 维；`decode_agent_action` 把连续参数映射到物理指令，
速度目标受 `a_max`/`v_max` 限制，底层 `VelocityController` 是 PID。

### 4.2 target mask

```
target 合法 = 目标存活
             & 可见
             & 距离 < lock_range (12 m)
```

不可见/超距目标不能锁定；fixed 开局可能 3 个全不合法，此时只能选 `none`。

### 4.3 fire mask（重要）

`RmState.fire_mask` 形状 `(B,N,2)`：

```
fire_mask[..., 0] = True          # release 永远允许
fire_mask[..., 1] = (target != 0) # 只有锁定目标才允许 press
```

- `build_observation` 和 `build_observation_rows` 都必须同步写入；
- TensorDict 侧 `(group,"fire_mask")` 由 `_write_action_mask` 写入 root/next；
- MAPPO 对 `fire` logits 做 `masked_fill(~fire_mask, -1e9)`；
- **没有目标时 fire=1 是非法动作**。如果看到 target=0 且 fire=1，先查 mask 注入，
  不要先怀疑观测。

---

## 5. 观测与全局 state

### 5.1 actor 观测（140 维）

| 段 | 维度 | 内容 |
|---|---:|---|
| self | 24 | 自身位置/速度/朝向 sin-cos/角速度/HP/热量/状态等 |
| spawn | 4 | 自家出生区信息 |
| center | 4 | 中心/占点信息 |
| mates | 2×18 | 两名队友的相对/绝对位置、速度、状态 |
| enemies | 3×17 | 三名敌人的相对/绝对位置、速度、最近开火等 |
| cue | 3 | 受击/威胁提示 |
| HUD | 10 | 时间、比分、目标分、存活数等 |
| obstacle | 8 | 8 个方向到墙/掩体的距离 |

- 蓝方观测做 180° canonicalization；
- **无 lidar、无地图 CNN**；
- 不可见敌人的 token 在 Transformer 中用 `src_key_padding_mask` 排除；
- cue 只保留当前帧最强威胁。

### 5.2 全局 state（191 维）

- `6 × 30 + 11`；
- 每 agent 的 30 = 22 个标量 + 8 个障碍距离；
- critic 使用红/蓝两个队视角 canonical state（蓝方旋转 180°、own/opponent 交换），
  共享 Transformer + 共享 value head，一次输出 6 个 value。

---

## 6. 规则与终局

### 6.1 终局码

| 码 | 含义 | 获胜方 |
|---|---|---|
| 1 / 2 | 红/蓝先到目标分 | 红/蓝 |
| 3 / 4 | 红/蓝清空对手 | 红/蓝 |
| 5 / 6 | 时间到，红/蓝总伤害高 | 红/蓝 |
| 7 / 8 | 伤害相同，红/蓝存活总 HP 高 | 红/蓝 |
| 9 / 10 | 红/蓝碰撞犯规 | 对方 |
| 11 / 12 | 红/蓝超时 | 对方 |
| 20 | 平局 | 无 |
| 23 | 空局（10 s 无事件） | 无 |
| 24 / 25 | 蓝/红推墙或掩体犯规 | 红/蓝 |
| 26 | 双方推墙，平局 | 无 |
| 27 / 28 | 蓝/红推人犯规 | 红/蓝 |
| 29 | 双方推人，平局 | 无 |

### 6.2 碰撞

- 实际肇事 agent 记录在 `s.foul_agent`；
- 碰撞烈度 `s.foul_intensity`：
  `clamp((impact - robot_foul_speed)/(2*v_max - robot_foul_speed), 0, 1)`，
  即相对速度 2→6 m/s 映射 0→1；
- 终局惩罚只作用 `foul_agent`：
  `terminal_foul - k_foul_intensity*foul_intensity`；
- 对手不再因为碰撞犯规加分；
- 双方碰撞则双方肇事 agent 都扣。

### 6.3 推墙 / 推人

- **推墙/掩体**：连续主动推 `wall_push_time=1.0 s` 触发；只算活着 agent；
- **推人**：连续主动推 `body_push_time=0.5 s`、法向速度 `>0.3 m/s` 触发；
  只给活着的主动推人者计时；队友、对手、尸体都算；
- 惩罚：
  - 主动方 `-10`；
  - 同队非主动方 `-0.5`；
  - 对手 `0`（单方面推时对手不加分）；
  - 双方都推：双方主动方各 `-10`，平局；
- 训练消融可设 `body_push_time=1e9`（`--no-body-push-terminal`），
  只关闭终局，dense 推人惩罚保留。

### 6.4 小陀螺

- 上限 `spin_time=1.0 s`；
- 实际接触瞬间清零：
  - 机器人中心距 ≤0.60 m（双方都清）；
  - 墙/掩体接触（该 agent）；
- 0.65 m 贴人圈不清零；
- 重新接触可以再次清零；一直贴着可以重新转起来。

### 6.5 空局

- `stagnation_time=10 s` 内没有开火/伤害/实际回血/中心占点事件 ⇒ `OUTCOME_STAGNATION=23`；
- 基础 `stagnation_penalty=5.0`；
- 逐 agent 折扣：
  - 障碍 clearance；
  - 离自家出生点距离（越远扣得越少）；
  - 队伍中心距离（近中心少扣）；
  - 速度 EMA（移动最多减半）；
- 这些折扣连乘；不要只看基础 `-5`。

---

## 7. 奖励结构（当前）

> 完整系数在 `rm_config.py`；这里是量级和方向。

- **正常赢**：`+6 + 3×剩余时间比例 + 3×分差/90`；
- **正常输**：`-6`；
- **碰撞犯规**：肇事者 `-10 - 3×烈度`；
- **推墙/推人**：主动 `-10`，队友 `-0.5`，对手 `0`；
- **占点**：净人数计分（红 2 蓝 1 ⇒ 红只加 1 份）；
- **击杀**：`kill_score=10`；击杀池按伤害分配；
- **贴人**：0.65 m 内主动 60% / 被动 40%，基础 `0.0025/帧` + ramp，
  2 s 封顶；队友/尸体也计；
- **撞击**：首次接触按法向相对速度一次性扣，`k_agent_impact=0.3`；
- **死亡**：`-0.2`；复活后 3 s 无敌，无敌结束过热门锁；
- **低血回家**：home shaping + 实际回血奖励；
- **空局**：见 6.5。

---

## 8. 可见性与几何

- `visibility_matrix(s, yaw, n_samples, locked_target)`：
  - 把所有 observer-target 对批量送入 `rm_geometry.visibility_arc`；
  - 计算 FOV、遮挡、可见性；
  - 带 exact memo cache（step/pos/yaw/target 完全相同才复用）；
- `ray_edges` / `segment_circle_blocked` 支持 `eps`；`RM_JIT=1` 可走
  `torch.jit.script(visibility_arc)`，默认 eager；
- 8 向障碍距离由 `obstacle_distances_for_state` 一次批量计算，并缓存到
  `s.obstacle_dist` 供观测/state 复用；
- 静态掩体/墙在 `rm_config.py` 定义，fixed/random 共用。

---

## 9. 局部 obs 重建

`RmScenario.reset_world_at` 不再让每个 env 的全量 reset 重建所有 B 行：

- 维护 `_obs_dirty_rows`；
- `_ensure_obs` 对部分 reset 调 `build_observation_rows(s, rows)`；
- `_StateRows` 把行切片成子 state，静态字段透传；
- `build_observation_rows` 只重建受影响行的 obs/global_state/action_mask/
  obstacle_dist，再写回原 state。

这属于性能优化；无噪声字段与 full build 已做逐位一致测试。

---

## 10. 自测与常见坑

```bash
docker exec -w /home/vscode/workspace/VectorizedMultiAgentSimulator robocon2025-marl bash -lc '
  OMP_NUM_THREADS=2 PYTHONPATH=. python /home/vscode/workspace/.opencode/rm_selftest.py'
```

当前应为 **43/43**。

常见坑：

1. **fixed 初始 target 全 none**：设计如此；不是观测损坏。
2. **fire mask 没写入**：target=0 仍 fire=1；检查 `get_fire_mask`、TensorDict spec、
   MAPPO 的 `fire` 分支。
3. **推挤终局太早**：0.5 s body push + -10 很重；先看终局占比和轨迹。
4. **碰撞判罚罚错人**：只罚 `foul_agent`；双方都撞则双方都罚。
5. **late-game 时间语义**：总时长 180 s，剩 30–60 s；不要设成总时长 30–60。
6. **checkpoint 恢复 NaN/meta**：先 cold smoke，再检查 buffer/optimizer/新增键；
   `RM_SKIP_BUFFER_RESTORE=1` 只是诊断。
7. **观测/动作维度改动**：旧 checkpoint 不兼容；不要静默 `strict=False` 后宣称能续训。
8. **性能**：不要只看 tqdm；用 CSV 原始耗时差分。

---

## 11. 相关文档

- [`../rm_simplified_rules_v1.1.md`](../rm_simplified_rules_v1.1.md) — 权威规则书
- [`../docs/rm/rm_impl_notes.md`](../docs/rm/rm_impl_notes.md) — 实现/性能/实验记录
- [`../AGENTS.md`](../AGENTS.md) — 项目级工作手册
- [`../README.md`](../README.md) — 项目门面
