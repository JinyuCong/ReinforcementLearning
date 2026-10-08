import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ale_py
import gymnasium as gym
import cv2
from collections import deque

from policy_gradient.utils import LinearActor, LinearCritic

gym.register_envs(ale_py)
env = gym.make("CartPole-v1", render_mode="human")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# PPO —— Proximal Policy Optimization（Clip 版本）
# ================================================================
# Actor-Critic 每步更新一次，数据效率低，且更新步长难以控制：
#   步长太大 → 策略崩塌（新旧策略差异太大，训练不稳定）
#   步长太小 → 收敛慢
#
# PPO 的核心思想：收集一批数据后，对同一批数据做多次梯度更新，
# 但用 Clip 限制新旧策略的比率，防止更新过大：
#
#   r_t(θ) = π(a_t|s_t; θ) / π(a_t|s_t; θ_old)   ← 重要性采样比率
#
#   L_CLIP = E[ min(r_t·A_t,  clip(r_t, 1-ε, 1+ε)·A_t) ]
#
#   其中 A_t = δ_t = r + γV(s') - V(s) 是优势函数估计
#   ε（clip_eps）通常取 0.2
#
# 算法流程：
#   for update in range(num_updates):
#     用旧策略 π_old 收集 steps_per_update 步数据：
#       存储 (s, a, r, done, log_prob_old, V(s))
#     计算每步的优势 A_t 和 return G_t
#     对收集的数据做 K 轮（ppo_epochs）mini-batch 更新：
#       r_t = exp(log_π_new(a|s) - log_π_old(a|s))
#       L_clip  = min(r_t·A, clip(r_t, 1-ε, 1+ε)·A)
#       L_value = (V(s) - G_t)²
#       loss = -L_clip + c·L_value
#       反向传播更新
#
# 与 Actor-Critic 的区别：
#   A-C  : 每步更新一次，on-policy，数据用一次就丢
#   PPO  : 收集一批数据后多次更新（ppo_epochs），clip 保证稳定性
#
# 数据结构：
#   actor  : 策略网络（softmax 输出）
#   critic : 值函数网络（scalar 输出）
# ================================================================

    
class UpdateDataset(Dataset):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return {
            'state': self.buffer[index][0],
            'action': self.buffer[index][1],
            'reward': self.buffer[index][2],
            'done': self.buffer[index][3],
            'log_action_p': self.buffer[index][4],
            'state_value': self.buffer[index][5],
            'G': self.buffer[index][6],
            'A': self.buffer[index][7]
        }


def ppo(env, num_updates=500, steps_per_update=256,
        lr=3e-4, gamma=0.99, clip_eps=0.2,
        ppo_epochs=4, batch_size=64, value_coef=0.5):
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    actor_net = LinearActor(state_dim, num_actions).to(device)
    critic_net = LinearCritic(state_dim).to(device)
    actor_optim = torch.optim.Adam(actor_net.parameters(), lr=lr)
    critic_optim = torch.optim.Adam(critic_net.parameters(), lr=lr)

    state, _ = env.reset()

    for update in range(num_updates):
        buffer = []

        # 用旧策略 π_old 收集 steps_per_update 步数据
        for step in range(steps_per_update):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                state_value = critic_net(state_t)
                log_probs = actor_net(state_t)

            action_probs = torch.exp(log_probs)
            action = torch.multinomial(action_probs, 1).item()
            log_action_p = log_probs[0, action]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.append((state_t.squeeze(0), action, reward, done, log_action_p.squeeze(0), state_value))
            state = next_state

            if done:
                state, _ = env.reset()

        G = 0
        # 将TD target和advantage添加到buffer每个元素的末尾
        # 这时变为(s, a, r, done, log_prob_old, V(s), G, A)
        for i, (_, _, reward, done, _, state_value) in enumerate(reversed(buffer)):
            G = reward + gamma * G * (1 - done)
            advantage = G - state_value.detach().item()
            buffer[len(buffer) - i - 1] += (G, advantage)

        buffer_dataset = UpdateDataset(buffer)
        buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=False)

        update_loss = 0
        for epoch in range(ppo_epochs):
            for batch in buffer_loader:
                state_t = batch["state"].to(device)
                action = batch["action"].view(-1, 1).to(device)
                log_prob_old = batch["log_action_p"].view(-1, 1).to(device)
                G = batch["G"].view(-1, 1).to(device)
                advantage = batch["A"].view(-1, 1).to(device)

                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                new_log_probs = actor_net(state_t)
                log_prob_new = new_log_probs.gather(1, action)
                ratio = torch.exp(log_prob_new - log_prob_old)

                clip_loss = torch.min(ratio * advantage, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage).mean()
                V_new = critic_net(state_t)
                value_loss = ((V_new - G) ** 2).mean()

                all_loss = -clip_loss + value_coef * value_loss

                actor_optim.zero_grad()
                critic_optim.zero_grad()
                all_loss.backward()
                actor_optim.step()
                critic_optim.step()

                update_loss += all_loss.item()

        update_loss /= (ppo_epochs * len(buffer_loader))
        print(f"Update {update + 1} | Mean loss : {update_loss}")

    return actor_net
if __name__ == "__main__":
    actor = ppo(env, num_updates=500, steps_per_update=256)