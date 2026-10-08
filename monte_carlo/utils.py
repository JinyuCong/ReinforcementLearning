import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def state_to_index(state, env_size):
    '''

    :param state:
    :param env_size:
    :return:
    '''
    x, y = state
    return x + y * env_size[0]

def one_hot(indices: np.ndarray, total_dim: int):
    '''
    给一个1维的向量做one hot
    :param indices: 下标向量
    :param total_dim: 第二维总维度
    :return: one hot 矩阵
    '''
    n = indices.shape[0]
    one_hot_matrix = np.zeros((n, total_dim))

    for i in range(n):
        one_hot_matrix[i, indices[i]] = 1

    return one_hot_matrix



def generate_episode(env, policy_matrix):
    episode = []
    state, _ = env.reset()
    max_steps = 200

    for _ in range(max_steps):
        state_index = state_to_index(state, env_size=env.env_size)
        action_index = np.random.choice(len(env.action_space), p=policy_matrix[state_index])
        action = env.action_space[action_index]
        next_state, reward, done, _ = env.step(action)
        episode.append((state_index, action_index, reward))
        state = next_state
        if done:
            break

    return episode
