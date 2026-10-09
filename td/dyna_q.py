import sys
import os
from typing import DefaultDict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import random
from src.grid_world import GridWorld
from td.utils import state_to_index, epsilon_greedy

env = GridWorld()

class ReplayBuffer:
    """
    经验池
    """
    def __init__(self):
        self.buffer = DefaultDict(tuple)

    def __len__(self):
        return len(self.buffer)

    def push(self, state_idx, action_idx, reward, next_state_idx):
        """
        buffer记住(s, a, r, s')
        :param state_idx: 这个state的idx
        :param action_idx: 在这个state选择的action的idx
        :param reward: immediate reward
        :param next_state_idx: 做出动作后到的下一个state的idx
        :return:
        """
        self.buffer[(state_idx, action_idx)] = (reward, next_state_idx)

    def sample(self, sampling_num: int):
        """
        采样
        :param sampling_num: 采样的个数
        :return:
        """
        sampling_num = min(len(self), sampling_num)
        sampled_keys = random.sample(list(self.buffer.keys()), sampling_num)
        return [k + self.buffer[k] for k in sampled_keys]

def dyna_q(env: GridWorld, num_episodes=5000, alpha=0.1, gamma=0.9,
           epsilon_start=1.0, epsilon_end=0.05):
    """
    每走一步真实环境，先做一次 Q-learning 更新，并把 (s,a) → (r,s') 记进模型；
    再从模型里随机取 n 条经验回放、做 n 次额外的 Q-learning 更新。
    :param env: 环境
    :param num_episodes:
    :param alpha: 学习率
    :param gamma: 回报衰减
    :param epsilon_start: ϵ起始值
    :param epsilon_end: ϵ结束值
    :return:
    """
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    policy_matrix = np.ones((num_states, num_actions)) / num_actions

    buffer = ReplayBuffer()

    for ep in range(num_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / (num_episodes - 1)
        state, _ = env.reset()
        step = 0
        done = False
        episode_reward = 0

        while not done and step < 200:
            # 根据policy选这个状态的动作
            state_idx = state_to_index(state, env_size=env.env_size)
            state_action_probs = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
            action_idx = np.random.choice(num_actions, p=state_action_probs)
            action = env.action_space[action_idx]

            # 采取动作
            next_state, reward, done, _ = env.step(action)
            next_state_idx = state_to_index(next_state, env_size=env.env_size)

            # 更新一次Q和policy
            Q[state_idx, action_idx] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action_idx])
            policy_matrix[state_idx] = epsilon_greedy(epsilon, Q, policy_matrix, state_idx)
            episode_reward += reward

            # buffer记住(s, a, r, s')，随机取n条经验回放做n次Q更新
            buffer.push(state_idx, action_idx, reward, next_state_idx)
            samples = buffer.sample(50)
            for (s, a, r, next_s) in samples:
                Q[s, a] += alpha * (r + gamma * np.max(Q[next_s]) - Q[s, a])
                policy_matrix[s] = epsilon_greedy(epsilon, Q, policy_matrix, s)

            state = next_state
            step += 1

        print(f"episode {ep} | mean reward: {episode_reward / step}")

    return Q, policy_matrix


if __name__ == "__main__":
    Q, policy_matrix = dyna_q(env)

    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        s_idx = state_to_index(state, env.env_size)
        action = env.action_space[np.argmax(policy_matrix[s_idx])]
        state, reward, done, _ = env.step(action)
        env.render()

    env.add_policy(policy_matrix)
    env.render(animation_interval=300)