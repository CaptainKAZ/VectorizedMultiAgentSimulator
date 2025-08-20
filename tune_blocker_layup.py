import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =============================================================================
# 1. 从 layup_jit.py 中提取的核心盖帽计算逻辑
#    这部分代码与您环境中的计算逻辑完全一致
# =============================================================================
def calculate_block_factor(a1_pos, basket_pos, defender_pos, h_params):
    """
    计算给定参数下的总盖帽系数。
    
    Args:
        a1_pos (torch.Tensor): 射手A1的位置, shape (1, 2)
        basket_pos (torch.Tensor): 篮筐的位置, shape (1, 2)
        defender_pos (torch.Tensor): 所有防守者的位置, shape (1, n_defenders, 2)
        h_params (dict): 包含所有相关超参数的字典
        
    Returns:
        float: 计算出的总盖帽系数
    """
    shot_vector = basket_pos - a1_pos
    blocker_vector = defender_pos - a1_pos.unsqueeze(1)
    
    shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
    dot_product = torch.sum(blocker_vector * shot_vector.unsqueeze(1), dim=-1)
    
    proj_len_ratio = dot_product / shot_vector_norm_sq
    
    is_between = (proj_len_ratio > 0) & (proj_len_ratio < 1)
    
    projection = proj_len_ratio.unsqueeze(-1) * shot_vector.unsqueeze(1)
    dist_perp_sq = torch.sum((blocker_vector - projection)**2, dim=-1)
    
    dist_a1_to_def = torch.linalg.norm(blocker_vector, dim=-1)
    gate_input = h_params['def_proximity_threshold'] - dist_a1_to_def
    soft_proximity_gate = torch.sigmoid(h_params['block_gate_k'] * gate_input)
    
    # 注意: layup_jit.py 中还有一个 proximity_threshold 硬门控，这里也一并实现
    is_blocker_per_defender = is_between & (dist_perp_sq < (h_params['proximity_threshold'])**2)
    
    block_contribution = (
        torch.exp(-dist_perp_sq / (2 * h_params['block_sigma']**2))
        * is_blocker_per_defender.float()
        * soft_proximity_gate
    )
    
    total_block_factor = torch.clamp(block_contribution.sum(dim=1), 0, 1)
    
    return total_block_factor.item()

# =============================================================================
# 2. Matplotlib 的可拖拽点对象
# =============================================================================
class DraggablePoint:
    def __init__(self, point, update_callback):
        self.point = point
        self.press = None
        self.update_callback = update_callback

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.point.axes: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = self.point.center, (event.xdata, event.ydata)

    def on_motion(self, event):
        'on motion we will move the point if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.point.axes: return
        point_pos, press_pos = self.press
        dx = event.xdata - press_pos[0]
        dy = event.ydata - press_pos[1]
        self.point.center = (point_pos[0] + dx, point_pos[1] + dy)
        self.update_callback() # 拖动时实时更新
        self.point.figure.canvas.draw_idle()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.update_callback() # 释放时也更新一次
        self.point.figure.canvas.draw_idle()

# =============================================================================
# 3. 主程序：创建交互式窗口
# =============================================================================
# --- 初始设置 ---
# 将位置定义为torch张量以匹配计算函数
a1_pos = torch.tensor([[5.0, 2.0]])
basket_pos = torch.tensor([[5.0, 10.0]])
d1_pos_init = [3.0, 6.0]
d2_pos_init = [7.0, 7.0]

# 初始超参数值
h_params = {
    'def_proximity_threshold': 1.2,
    'block_gate_k': 10.0,
    'proximity_threshold': 0.3 * 2.5, # agent_radius * 2.5
    'block_sigma': 0.3 * 1.5,       # agent_radius * 1.5
    'win_condition_block_threshold': 0.5,
}

# --- 创建画布和坐标系 ---
fig, ax = plt.subplots(figsize=(8, 10))
plt.subplots_adjust(bottom=0.35) # 为下方的滑块留出空间

ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("block factor tunner")

# --- 绘制固定对象 ---
a1_dot = ax.plot(a1_pos[0, 0], a1_pos[0, 1], 'bo', markersize=15, label='A1 (Shooter)')[0]
basket_dot = ax.plot(basket_pos[0, 0], basket_pos[0, 1], 'g^', markersize=15, label='Basket')[0]
shot_line = ax.plot([a1_pos[0, 0], basket_pos[0, 0]], [a1_pos[0, 1], basket_pos[0, 1]], 'k--', label='Shot Path')[0]

# --- 绘制可拖拽的防守者 ---
d1_patch = plt.Circle(d1_pos_init, 0.3, fc='r', alpha=0.8)
d2_patch = plt.Circle(d2_pos_init, 0.3, fc='r', alpha=0.8)
ax.add_patch(d1_patch)
ax.add_patch(d2_patch)

# --- 添加文本显示 ---
block_factor_text = ax.text(0.5, 1.1, '', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='blue')
result_text = ax.text(0.5, 1.05, '', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='red', weight='bold')

# --- 更新函数 ---
def update_plot(*args):
    # 1. 从滑块获取最新的超参数
    h_params['def_proximity_threshold'] = slider_def_prox.val
    h_params['block_gate_k'] = slider_gate_k.val
    h_params['proximity_threshold'] = slider_prox_thresh.val
    h_params['block_sigma'] = slider_sigma.val
    h_params['win_condition_block_threshold'] = slider_win_cond.val
    
    # 2. 从可拖拽点获取最新的防守者位置
    d1_center = d1_patch.center
    d2_center = d2_patch.center
    defender_pos = torch.tensor([[[d1_center[0], d1_center[1]], [d2_center[0], d2_center[1]]]])
    
    # 3. 重新计算盖帽系数
    block_factor = calculate_block_factor(a1_pos, basket_pos, defender_pos, h_params)
    
    # 4. 更新文本显示
    block_factor_text.set_text(f'Total Block Factor: {block_factor:.4f}')
    
    if block_factor >= h_params['win_condition_block_threshold']:
        result_text.set_text('BLOCKED!')
        result_text.set_color('red')
    else:
        result_text.set_text('SCORE!')
        result_text.set_color('green')
        
    fig.canvas.draw_idle()

# --- 创建滑块 ---
ax_def_prox = plt.axes([0.25, 0.25, 0.65, 0.03])
slider_def_prox = Slider(ax_def_prox, 'Def Proximity Thresh', 0.1, 5.0, valinit=h_params['def_proximity_threshold'])

ax_gate_k = plt.axes([0.25, 0.20, 0.65, 0.03])
slider_gate_k = Slider(ax_gate_k, 'Block Gate K', 1.0, 50.0, valinit=h_params['block_gate_k'])

ax_prox_thresh = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_prox_thresh = Slider(ax_prox_thresh, 'Proximity Thresh', 0.1, 2.0, valinit=h_params['proximity_threshold'])

ax_sigma = plt.axes([0.25, 0.10, 0.65, 0.03])
slider_sigma = Slider(ax_sigma, 'Block Sigma', 0.1, 2.0, valinit=h_params['block_sigma'])

ax_win_cond = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_win_cond = Slider(ax_win_cond, 'Win Condition Thresh', 0.0, 1.0, valinit=h_params['win_condition_block_threshold'])

# --- 绑定事件 ---
slider_def_prox.on_changed(update_plot)
slider_gate_k.on_changed(update_plot)
slider_prox_thresh.on_changed(update_plot)
slider_sigma.on_changed(update_plot)
slider_win_cond.on_changed(update_plot)

dp1 = DraggablePoint(d1_patch, update_plot)
dp1.connect()
dp2 = DraggablePoint(d2_patch, update_plot)
dp2.connect()

# --- 初始调用和显示 ---
update_plot()
ax.legend()
plt.show()