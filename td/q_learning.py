import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from td.utils import state_to_index, epsilon_greedy

env = GridWorld()

# ================================================================
# Q-learning —— Off-Policy TD Control
# ================================================================
# 目标：学习最优策略，估计 Q(s,a)
#
# 与 Sarsa 的核心区别：
#   Sarsa（on-policy）：Q 更新用的是 ε-greedy 实际选出的 a'
#   Q-learning（off-policy）：Q 更新用的是 max Q(s',·)，与实际选的动作无关
#
# 更新公式（每步执行）：
#   Q(s,a) ← Q(s,a) - α * [Q(s,a) - (r + γ * max_a' Q(s',·))]
#
# 算法思路（和 Sarsa 几乎一样，只改 TD target）：
#   for ep in range(num_episodes):
#     s = env.reset()
#     while not done（最多200步）:
#       用 ε-greedy 选 a，执行得到 r, s'
#       Q[s,a] -= alpha * (Q[s,a] - (r + gamma * max(Q[s'])))  ← 唯一区别
#       用 ε-greedy 更新 policy_matrix[s]
#       s ← s'
# ================================================================

def q_learning(env, num_episodes=5000,
               alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05):
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    # 你来写
    # 和 Sarsa 代码几乎一样，只有 TD target 不同：
    #   Sarsa:      r + gamma * Q[s', a']
    #   Q-learning: r + gamma * np.max(Q[s'])
    for ep in range(num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)
        
        state, _ = env.reset()
        state_idx = state_to_index(state, env_size=env.env_size)
        
        step = 0
        done = False
        episode_reward = 0

        while not done and step < 200:
            state_action_probs = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
            action_idx = np.random.choice(num_actions, p=state_action_probs)
            action = env.action_space[action_idx]
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            next_state_idx = state_to_index(next_state, env_size=env.env_size)
            
            # Q(s,a) ← Q(s,a) - α * [Q(s,a) - (r + γ * max_a' Q(s',·))]
            Q[state_idx, action_idx] -= alpha * (Q[state_idx, action_idx] - (reward + gamma * np.max(Q[next_state_idx])))
            
            policy_matrix[state_idx] = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
            
            state_idx = next_state_idx
            step += 1

        print(f"episode {ep} | mean reward: {episode_reward / step}")

    return Q, policy_matrix


if __name__ == "__main__":
    Q, best_policy = q_learning(env)

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        s_idx = state_to_index(state, env.env_size)
        action = env.action_space[np.argmax(best_policy[s_idx])]
        state, reward, done, _ = env.step(action)
        env.render()

    env.add_policy(best_policy)
    env.render(animation_interval=300)
