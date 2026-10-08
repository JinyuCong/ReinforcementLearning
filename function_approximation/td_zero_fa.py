import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from function_approximation.utils import state_to_index, index_to_state, get_features, NUM_FEATURES

env = GridWorld()

# ================================================================
# TD(0) with Linear Function Approximation —— 策略评估
# ================================================================
# 目标：给定一个固定策略，用线性函数近似估计 V(s)
#
# 用线性函数近似代替 V 表：
#   V(s; w) = φ(s)^T w
#   φ(s) ∈ R^d 是状态特征向量（多项式特征，d=6）
#   w ∈ R^d 是待学习的权重向量
#
# 与表格型 TD(0) 的对比：
#   表格型：直接更新 V[s]
#   函数近似：更新 w（参数共享，可泛化）
#
# TD 误差（和表格型完全一样）：
#   δ = r + γ * V(s'; w) - V(s; w)
#     = r + γ * φ(s')^T w - φ(s)^T w
#
# 参数更新（对 w 的梯度方向 = φ(s)）：
#   w += α * δ * φ(s)
#
# 算法思路：
#   for ep in range(num_episodes):
#     s = env.reset()，计算 φ(s)
#     while not done（最多 200 步）:
#       按策略选动作，执行 env.step 得到 r, s', done
#       计算 φ(s')
#       δ = r + γ * (1 - done) * φ(s')^T w - φ(s)^T w
#       w += α * δ * φ(s)
#       s ← s'，φ ← φ(s')
# ================================================================

def td_zero_fa(env, num_episodes=500, alpha=0.01, gamma=0.9):
    num_states = env.num_states
    num_actions = len(env.action_space)

    # 权重向量 w：shape (NUM_FEATURES,)，初始全 0
    w = np.zeros(NUM_FEATURES)

    # 固定的均匀随机策略（和表格型 TD(0) 一致）
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    # 你来写
    # 提示：
    #   get_features(state, env.env_size)  → φ(s)，shape (NUM_FEATURES,)
    #   V(s) = phi @ w                     → 标量
    #   策略选动作（和表格型 TD(0) 一样）：
    #       x, y = state
    #       state_idx = x + y * env.env_size[0]
    #       action_idx = np.random.choice(num_actions, p=policy_matrix[state_idx])

    for ep in range(num_episodes):
        state, _ = env.reset()
        state_idx = state_to_index(state, env_size=env.env_size)
        phi = get_features(state, env_size=env.env_size)
        
        done = False
        step = 0
        # 记录mean state value
        all_state_value = 0
        while not done and step < 200:
            pred_state_value = phi @ w
            # 随机选择一个action
            action_idx = np.random.choice(num_actions, p=policy_matrix[state_idx])
            action = env.action_space[action_idx]
            # 采取action走到下一个state
            next_state, reward, done, _ = env.step(action)
            # 提取下一个state的特征
            next_phi = get_features(next_state, env_size=env.env_size)
            # 计算下一个state的state value
            pred_next_state_value = next_phi @ w
            # 计算均方误差
            delta = reward + gamma * (1 - done) * pred_next_state_value - pred_state_value
            w += alpha * delta * phi
            
            state = next_state
            phi = next_phi            
            all_state_value += pred_state_value
            
            step += 1
        
        mean_state_value = all_state_value / step
        if ep % 10 == 0:
            print(f"Epsisode {ep} | mean state value : {mean_state_value:.3f}")

    return w


if __name__ == "__main__":
    w = td_zero_fa(env, num_episodes=500)

    # 从 w 重建每个状态的 V 值向量，用于可视化
    V = np.zeros(env.num_states)
    V_matrix = np.zeros((env.env_size[1], env.env_size[0]))

    for s in range(env.num_states):
        x = s % env.env_size[0]
        y = s // env.env_size[0]
        phi = get_features((x, y), env.env_size)
        V[s] = phi @ w
        V_matrix[y, x] = phi @ w

    state, _ = env.reset()
    env.render()
    env.add_state_values(V)
    env.render(animation_interval=30)

    # 3D surface plot
    import matplotlib.pyplot as plt
    plt.ioff()   # 关闭交互模式，让 plt.show() 变为阻塞

    rows = np.arange(1, env.env_size[1] + 1)
    cols = np.arange(1, env.env_size[0] + 1)
    C, R = np.meshgrid(cols, rows)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, C, V_matrix, cmap='viridis')
    ax.set_xlabel('row')
    ax.set_ylabel('column')
    ax.set_zlabel('V')
    ax.set_title('TD-Linear')
    plt.show()
