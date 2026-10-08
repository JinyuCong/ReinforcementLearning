import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from function_approximation.utils import get_features, epsilon_greedy_fa, NUM_FEATURES, state_to_index

env = GridWorld()

# ================================================================
# Q-learning with Linear Function Approximation —— Off-Policy TD Control
# ================================================================
# Q(s, a; W) = φ(s)^T W[:, a]
#
# 与 Sarsa-FA 的核心区别（和表格型完全一致）：
#   Sarsa-FA（on-policy）：  δ = r + γ * Q(s', a'; W) - Q(s, a; W)
#                             a' 是 ε-greedy 选出的动作
#   Q-learning-FA（off-policy）：δ = r + γ * max_a' Q(s', a'; W) - Q(s, a; W)
#                                 直接取 max，与执行策略无关
#
# 参数更新：
#   W[:, a] += α * δ * φ(s)
#
# 算法思路：
#   for ep in range(num_episodes):
#     s = env.reset()，计算 φ(s)
#     while not done（最多 200 步）:
#       用 ε-greedy 选 a，执行得到 r, s'，计算 φ(s')
#       δ = r + γ * max(φ(s')^T W) - φ(s)^T W[:, a]   ← 唯一区别
#       W[:, a] += α * δ * φ(s)
#       s ← s'，φ ← φ(s')
# ================================================================

def q_learning_fa(env, num_episodes=5000, alpha=0.01,
                  gamma=0.9, epsilon_start=1.0, epsilon_end=0.05):
    num_states = env.num_states
    num_actions = len(env.action_space)

    # 权重矩阵 W：shape (NUM_FEATURES, num_actions)，初始全 0
    W = np.zeros((NUM_FEATURES, num_actions))

    # 你来写
    # 提示：
    #   φ(s')^T W  得到 shape (num_actions,) 的 Q 值向量
    #   max_a' Q(s', a') = np.max(phi_next @ W)
    #   ε 线性衰减：epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)
    for ep in range(num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)

        state, _ = env.reset()
        # 初始state的特征
        phi = get_features(state, env_size=env.env_size)

        done = False
        step = 0
        while not done and step < 200:
            action_probs = epsilon_greedy_fa(epsilon, W, phi, num_actions)
            action_idx = np.random.choice(num_actions, p=action_probs)
            action = env.action_space[action_idx]

            next_state, reward, done, _ = env.step(action)
            next_phi = get_features(next_state, env.env_size)

            action_value = phi @ W[:, action_idx]
            next_action_value = np.max(next_phi @ W, axis=-1)

            delta = reward + gamma * next_action_value * (1 - done) - action_value
            W[:, action_idx] += alpha * delta * phi

            phi = next_phi

    return W


if __name__ == "__main__":
    W = q_learning_fa(env, num_episodes=5000)

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        phi = get_features(state, env.env_size)
        action_idx = np.argmax(phi @ W)
        state, reward, done, _ = env.step(env.action_space[action_idx])
        env.render()

    # 从 W 重建 policy_matrix 以便可视化
    num_states = env.num_states
    num_actions = len(env.action_space)
    policy_matrix = np.zeros((num_states, num_actions))
    for s in range(num_states):
        x = s % env.env_size[0]
        y = s // env.env_size[0]
        phi = get_features((x, y), env.env_size)
        best_a = np.argmax(phi @ W)
        policy_matrix[s, best_a] = 1.0
    env.add_policy(policy_matrix)
    env.render(animation_interval=30)
