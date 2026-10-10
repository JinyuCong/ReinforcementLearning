import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from td.utils import state_to_index, epsilon_greedy

env = GridWorld()

# ================================================================
# n-step Sarsa —— On-Policy TD Control
# ================================================================
# 与 1-step Sarsa 的区别：
#   1-step：TD target = r_{t+1} + γ * Q(s_{t+1}, a_{t+1})
#   n-step：TD target = r_{t+1} + γ*r_{t+2} + ... + γ^(n-1)*r_{t+n} + γ^n * Q(s_{t+n}, a_{t+n})
#
# 核心思路：
#   需要先缓存 n 步的 (s, a, r)，等攒够 n 步后再回头更新第 t 步的 Q
#
# 算法思路：
#   for ep in range(num_episodes):
#     用列表缓存本条 episode 的 states[], actions[], rewards[]
#     s = env.reset()，用 ε-greedy 选 a，存入缓存
#
#     T = inf（episode 终止时刻，初始设为无穷）
#     t = 0
#     loop:
#       if t < T:
#         执行 a_t，得到 r_{t+1}, s_{t+1}
#         存入 rewards, states
#         if done: T = t + 1
#         else: 用 ε-greedy 选 a_{t+1}，存入 actions
#
#       τ = t - n + 1         ← τ 是本次要更新的时刻
#       if τ >= 0:
#         G = Σ_{i=τ+1}^{min(τ+n, T)} γ^(i-τ-1) * r_i    ← n步累积回报
#         if τ + n < T:
#           G += γ^n * Q[s_{τ+n}, a_{τ+n}]                ← 加上 bootstrap
#         Q[s_τ, a_τ] -= alpha * (Q[s_τ, a_τ] - G)
#         用 ε-greedy 更新 policy_matrix[s_τ]
#
#       if τ == T - 1: break
#       t += 1
# ================================================================

def n_step_sarsa(env, n=3, num_episodes=5000, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05):
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    # 预计算 γ 的幂次，方便后面算 G
    gamma_powers = np.array([gamma ** i for i in range(n + 1)])

    # 你来写
    # 提示：
    #   用环形缓冲或普通列表存 states/actions/rewards
    #   rewards 下标从 1 开始（r_1 对应第一步的奖励），和教材保持一致
    #   G = sum(gamma_powers[i] * rewards[tau+1+i] for i in range(min(n, T-tau-1)))
    #   if tau + n < T: G += gamma_powers[n] * Q[states[tau+n], actions[tau+n]]
    for ep in range(num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)
        states, actions, rewards = [], [], []
        
        # 起始state，存入states
        state, _ = env.reset()
        state_idx = state_to_index(state, env_size=env.env_size)
        states.append(state_idx)
        
        # 用 ε-greedy 选 a，存入actions
        state_action_probs = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
        action_idx = np.random.choice(num_actions, p=state_action_probs)
        action = env.action_space[action_idx]
        actions.append(action_idx)
        
        T = np.inf
        t = 0
        while t < T:
            # 执行 a_t，得到 s_{t+1}, r_{t+1}
            state, reward, done, _ = env.step(action)
            state_idx = state_to_index(state, env.env_size)
            # 存入 rewards, states
            states.append(state_idx)
            rewards.append(reward)
            if done:
                T = t + 1
            else:
                state_action_probs = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
                action_idx = np.random.choice(num_actions, p=state_action_probs)
                action = env.action_space[action_idx]
                actions.append(action_idx)
            
            tau = t - n + 1
            if tau >= 0:
                # 累加 τ+1 到 min(τ+n, T) 的折扣奖励
                # rewards 列表下标从 0 开始，对应 r_1，所以 r_{i} = rewards[i-1]
                end = n if T == np.inf else min(n, int(T) - tau)
                G = sum([gamma_powers[i] * rewards[tau+i] for i in range(end)])
                # 如果 τ+n 还没到终止时刻，加上 bootstrap
                if tau + n < T:
                    G += gamma_powers[n] * Q[states[tau + n], actions[tau + n]]

                # 更新 Q
                Q[states[tau], actions[tau]] -= alpha * (Q[states[tau], actions[tau]] - G)

                # 更新策略
                policy_matrix[states[tau]] = epsilon / num_actions
                policy_matrix[states[tau], np.argmax(Q[states[tau]])] += 1 - epsilon

            if tau == T - 1:
                break
            t += 1
    
    return Q, policy_matrix


if __name__ == "__main__":
    Q, best_policy = n_step_sarsa(env, n=3, num_episodes=100)

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        s_idx = state_to_index(state, env.env_size)
        action = env.action_space[np.argmax(best_policy[s_idx])]
        state, reward, done, _ = env.step(action)
        env.render()

    env.add_policy(best_policy)
    env.render(animation_interval=3)
