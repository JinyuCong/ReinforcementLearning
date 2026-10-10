import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from td.utils import state_to_index, epsilon_greedy

env = GridWorld()

# ================================================================
# Expected Sarsa —— On-Policy TD Control
# ================================================================
# 与 Sarsa 的唯一区别：
#   Sarsa：        TD target = r + γ * Q(s', a')        a' 是随机采样的
#   Expected Sarsa：TD target = r + γ * Σ_a π(a|s') * Q(s', a)  对 s' 下所有动作取期望
#
# 好处：消除了 a' 采样带来的方差，收敛更稳定
#
# 更新公式（每步执行）：
#   expected_q = Σ_a policy_matrix[s', a] * Q[s', a]   即 policy_matrix[s'] @ Q[s']
#   Q(s,a) ← Q(s,a) - α * [Q(s,a) - (r + γ * expected_q)]
#
# 算法思路（和 Sarsa 几乎一样，只改 TD target）：
#   for ep in range(num_episodes):
#     s = env.reset()
#     用 ε-greedy 在 s 上选 a
#     while not done（最多200步）:
#       执行 a，得到 r, s'
#       expected_q = policy_matrix[s'] @ Q[s']          ← 唯一区别，不需要采样 a'
#       Q[s,a] -= alpha * (Q[s,a] - (r + gamma * expected_q))
#       用 ε-greedy 更新 policy_matrix[s]
#       s ← s'，用 ε-greedy 选下一个 a
# ================================================================

def expected_sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05):
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    # 你来写
    # 提示：expected_q = policy_matrix[next_state_idx] @ Q[next_state_idx]
    #       其余结构和 sarsa.py 完全一样，只把 Q[s',a'] 替换成 expected_q
    for ep in range(num_episodes):
        # ε 线性衰减
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)
        
        state, _ = env.reset()
        state_idx = state_to_index(state, env_size=env.env_size)
        
        state_action_probs = epsilon_greedy(epsilon, Q ,policy_matrix, state_idx)
        
        action_idx = np.random.choice(num_actions, p=state_action_probs)
        action = env.action_space[action_idx]
        
        done = False
        step = 0
        while not done and step < 200:
            # 执行 a，得到 r, s'
            next_state, reward, done, _ = env.step(action)
            next_state_idx = state_to_index(next_state, env.env_size)
            
            # expected_q = policy_matrix[s'] @ Q[s'] 
            expected_q = policy_matrix[next_state_idx] @ Q[next_state_idx]
            
            # Q[s,a] -= alpha * (Q[s,a] - (r + gamma * expected_q))
            Q[state_idx, action_idx] -= alpha * (Q[state_idx, action_idx] - (reward + gamma * expected_q))
        
            # 用 ε-greedy 更新 policy_matrix[s]
            policy_matrix[state_idx] = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
            
            # s ← s'，用 ε-greedy 选下一个 a
            state_idx = next_state_idx
            action_idx = np.random.choice(num_actions, p=policy_matrix[state_idx])
            action = env.action_space[action_idx]
            
            step += 1
        
    return Q, policy_matrix


if __name__ == "__main__":
    Q, best_policy = expected_sarsa(env, num_episodes=500)

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
