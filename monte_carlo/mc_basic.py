import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from monte_carlo.utils import state_to_index, generate_episode, one_hot

env = GridWorld()

# ================================================================
# MC Basic —— 策略评估 + 策略改进（批量更新）
# ================================================================
# 算法思路（外层是策略迭代，内层是 MC 估计 Q）：
#   重复 num_iterations 次：
#     1. 【策略评估】用当前策略跑 num_episodes 条 episode，
#        估计 Q(s,a)（every-visit，从后往前算 G）
#     2. 【策略改进】对每个状态取 argmax_a Q(s,a)，
#        更新为确定性贪心策略（one-hot）
#
# 数据结构：
#   Q[s,a]          : 动作价值估计，shape (num_states, num_actions)
#   returns_sum[s,a]: 累积 G 之和，shape (num_states, num_actions)
#   counts[s,a]     : 访问次数，shape (num_states, num_actions)
#   policy_matrix   : 当前策略，改进后每行是 one-hot（确定性贪心）
# ================================================================

def mc_basic(env, num_iterations=1,
             num_episodes=1000, gamma=0.9):
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = None
    returns_sum = np.zeros((num_states, num_actions))
    counts = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    for it in range(num_iterations):
        for ep in range(num_episodes):
            episode = generate_episode(env, policy_matrix)
            G = 0
            for (state_index, action_index, reward) in reversed(episode):
                G += reward + gamma * G
                returns_sum[state_index, action_index] += G
                counts[state_index, action_index] += 1

            Q = returns_sum / (counts + 1e-8)
            best_action_indices = Q.argmax(axis=-1)
            policy_matrix = one_hot(best_action_indices, total_dim=num_actions)

    return policy_matrix, Q

if __name__ == "__main__":
    best_policy, Q = mc_basic(env)

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
