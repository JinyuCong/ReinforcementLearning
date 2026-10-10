import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.grid_world import GridWorld
from td.utils import state_to_index, epsilon_greedy

env: GridWorld = GridWorld()

# ================================================================
# Sarsa(λ) —— On-Policy TD Control with Eligibility Traces
# ================================================================
# 目标：学习最优策略，估计 Q(s,a)
#
# 与 1-step Sarsa 的区别：
#   Sarsa    ：每步只更新当前这一个 Q(s,a)
#               终点的奖励要靠很多个 episode 才能一格一格传回起点
#   Sarsa(λ) ：给每个 (s,a) 维护一个资格迹 E(s,a)，记录"最近被访问的程度"
#               每步的 TD 误差 δ 按 E 的大小分摊给所有走过的 (s,a)
#               → 一个 episode 内就能把奖励沿轨迹往回传
#
# 与 n-step Sarsa 的区别：
#   n-step   ：要缓存 n 步数据，等攒够了再回头更新，n 只能取整数
#   Sarsa(λ) ：不需要缓存，每步立即更新，λ ∈ [0,1] 连续可调
#   λ = 0 → 退化为 1-step Sarsa；λ = 1 → 接近 MC
#
# 资格迹 E（和 Q 同形状，num_states × num_actions）：
#   每步先给当前 (s,a) 的迹加上去，再整体按 γλ 衰减
#   k 步前访问过的 (s,a)，迹约为 (γλ)^k → 离得越远，分到的误差越少
#
#   两种迹（二选一）：
#     累积迹：E[s,a] += 1
#     替换迹：E[s,:] = 0；E[s,a] = 1     ← 推荐，GridWorld 里撞墙原地不动时
#                                          累积迹会不断叠加，替换迹上限为 1
#
# 更新公式（每步执行）：
#   δ = r + γ * Q(s',a') - Q(s,a)              ← 和 Sarsa 一样的 TD 误差
#   E(s,a) 按上面的方式加上去
#   Q ← Q + α * δ * E                          ← 整张 Q 表一起更新（numpy 向量运算）
#   E ← γλ * E                                 ← 所有迹衰减
#
# 算法思路：
#   for ep in range(num_episodes):
#     E = 0（整张表清零）                     ← 关键：每个 episode 开头都要清零
#     s = env.reset()
#     用 ε-greedy 在 s 上选 a
#     while not done（最多200步）:
#       执行 a，得到 r, s'
#       用 ε-greedy 在 s' 上选 a'
#       δ = r + gamma * Q[s',a'] - Q[s,a]
#       E[s,a] 更新（累积迹或替换迹）
#       Q += alpha * δ * E
#       E *= gamma * lam
#       用 ε-greedy 更新 policy_matrix                ← Q 整表都变了，
#                                                      至少更新 E > 0 的那些状态
#                                                      （或者直接全部状态都更新）
#       s ← s'，a ← a'
#
# 注意点：
#   - λ 取 0.8~0.9；λ 越接近 1 方差越大
#   - α 要比 1-step Sarsa 小一些（每步更新很多格子，总更新量变大）
#   - 终点 s' 的 Q 从未被更新过，一直为 0，所以 δ 不乘 (1-done) 也没问题，
#     但写成 r + γ*Q[s',a']*(1-done) 更保险
#   - 可以和 sarsa.py 用相同 episode 数对比每个 episode 的步数曲线，
#     在 10x10 迷宫上 Sarsa(λ) 应明显更快学会到达终点
#
# 数据结构：
#   Q             : (num_states, num_actions) 动作价值表
#   E             : (num_states, num_actions) 资格迹表
#   policy_matrix : (num_states, num_actions) ε-greedy 策略
# ================================================================

def sarsa_lambda(env, num_episodes=5000, alpha=0.05, gamma=0.9, lam=0.9,
                 epsilon_start=1.0, epsilon_end=0.05):
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    # 你来写
    for ep in range(num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)

        E = np.zeros((num_states, num_actions))
        state, _ = env.reset()

        done = False
        step = 0
        ep_reward = 0
        while not done and step < 200:
            state_idx =  state_to_index(state, env_size=env.env_size)
            action_probs = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
            action_idx = np.random.choice(num_actions, p=action_probs)
            action = env.action_space[action_idx]

            # 执行 a，得到 r, s'
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            next_state_idx = state_to_index(next_state, env_size=env.env_size)

            # 用 ε-greedy 在 s' 上选 a'
            next_action_probs = epsilon_greedy(epsilon, Q, policy_matrix, next_state_idx)
            next_action_idx = np.random.choice(num_actions, p=next_action_probs)

            delta = reward + gamma * Q[next_state_idx, next_action_idx] - Q[state_idx, action_idx]

            # E[s,a] 替换迹更新
            E[state_idx, :] = 0
            E[state_idx, action_idx] = 1

            Q += alpha * delta * E
            E *= gamma * lam

            # 用 ε-greedy 更新 policy_matrix
            # Q 整表都变了，所以更新整个policy表
            for i in range(num_states):
                action_probs_new = epsilon_greedy(epsilon, Q, policy_matrix, i)
                policy_matrix[i] = action_probs_new

            state = next_state
            step += 1

        print(f"Episode {ep + 1} mean reward: {ep_reward / (step + 1e-8)}")

    return Q, policy_matrix


if __name__ == "__main__":
    Q, best_policy = sarsa_lambda(env, num_episodes=5000)

    state, _ = env.reset()
    env.render()
    done = False
    step = 0
    while not done and step < 200:  # 防止策略原地打转导致死循环
        s_idx = state_to_index(state, env.env_size)
        action = env.action_space[np.argmax(best_policy[s_idx])]
        state, reward, done, _ = env.step(action)
        env.render()
        step += 1

    env.add_policy(best_policy)
    env.render(animation_interval=30)
