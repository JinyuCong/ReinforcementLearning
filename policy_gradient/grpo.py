import sys
import os

from sympy import false

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym

from policy_gradient.utils import LinearActor

env = gym.make("CartPole-v1")  # 训练时不渲染，评估时再开 render_mode="human"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# GRPO —— Group Relative Policy Optimization
# ================================================================
# PPO 需要一个 Critic 网络 V(s) 来估计优势 A_t = G_t - V(s)：
#   - 多训练一个网络，显存/计算翻倍
#   - V(s) 本身估得不准时，优势就有偏差，反过来拖累 Actor
#
# GRPO 的核心思想：去掉 Critic，用"组内相对比较"代替 V(s) 当 baseline：
#   从同一个起点 s_0 出发，用旧策略 π_old 采样一组（group）G 条轨迹，
#   每条轨迹得到一个总回报 R_i，然后在组内做标准化：
#
#   A_i = (R_i - mean(R_1..R_G)) / (std(R_1..R_G) + eps)
#
#   这条轨迹上每一步 t 都共享同一个优势 A_i（outcome supervision）
#   → 比组内平均好的轨迹被鼓励，比平均差的被抑制
#
# 目标函数（沿用 PPO 的 Clip，再加一个对参考策略的 KL 惩罚）：
#
#   r_{i,t}(θ) = π_θ(a_{i,t}|s_{i,t}) / π_old(a_{i,t}|s_{i,t})
#
#   L_CLIP = 1/G Σ_i  1/|T_i| Σ_t  min(r_{i,t}·A_i, clip(r_{i,t}, 1-ε, 1+ε)·A_i)
#
#   KL 用无偏、恒非负的 k3 估计（逐步计算）：
#     ρ = π_ref(a|s) / π_θ(a|s)
#     D_KL ≈ ρ - log(ρ) - 1
#
#   loss = -( L_CLIP - β·D_KL )
#
#   ε（clip_eps）通常取 0.2，β（kl_coef）取 0.01~0.04
#
# 三个策略的区别：
#   π_θ   : 正在训练的策略
#   π_old : 采样这一批数据时的策略（每次 update 开始前从 π_θ 拷贝，不求梯度）
#           → 用于重要性采样比率 r_t
#   π_ref : 参考策略（如初始策略，或每隔若干 update 同步一次），不求梯度
#           → 用于 KL 惩罚，防止策略整体漂得太远
#
# 算法流程：
#   π_ref ← 拷贝初始 π_θ
#   for update in range(num_updates):
#     π_old ← 拷贝当前 π_θ
#     for q in range(num_starts):                  # 一次 update 用多个起点
#       seed = 随机数
#       for i in range(group_size):                # 同一起点采 G 条轨迹
#         s = env.reset(seed=seed)                 # ← 相同 seed 保证起点相同
#         用 π_old 跑完一整条 episode：
#           存储 (s_t, a_t, log_π_old(a_t|s_t))
#         R_i = 这条轨迹的总回报
#       组内标准化得到 A_1..A_G，把 A_i 分给轨迹 i 的每一步
#     对收集的数据做 K 轮（grpo_epochs）更新：
#       log_π_new = π_θ(s) 中 a 对应的 log 概率
#       log_π_ref = π_ref(s) 中 a 对应的 log 概率（no_grad）
#       r_t   = exp(log_π_new - log_π_old)
#       L_clip = min(r_t·A, clip(r_t, 1-ε, 1+ε)·A)
#       KL     = exp(log_π_ref - log_π_new) - (log_π_ref - log_π_new) - 1
#       每条轨迹内先对 t 求平均，再对组内 G 条轨迹求平均
#       loss = -(L_clip - β·KL)
#       反向传播更新 π_θ
#     （可选）每隔 ref_update_freq 次 update：π_ref ← 拷贝 π_θ
#
# 与 PPO 的区别：
#   PPO  : 需要 Critic，A_t = G_t - V(s_t)，每一步的优势不同
#   GRPO : 不需要 Critic，A_i 由组内回报相对比较得到，同一条轨迹每步共享
#          额外加 KL(π_θ || π_ref) 惩罚（PPO 原版没有）
#
# 在 CartPole 上的注意点：
#   - 每步奖励都是 +1，R_i 就是坚持的步数
#   - 若组内 G 条轨迹回报完全相同，std = 0 → A_i 全为 0，
#     这组数据不提供梯度（所以 eps 必须加，且 group_size 不宜太小）
#   - 轨迹长短不一，1/|T_i| 的归一化保证长轨迹不会主导梯度
#
# 数据结构：
#   actor     : 策略网络 π_θ（log_softmax 输出），复用 LinearActor
#   old_actor : π_old（deepcopy，不求梯度）
#   ref_actor : π_ref（deepcopy，不求梯度）
#   没有 critic
# ================================================================


def grpo(env, num_updates=300, num_starts=4, group_size=8,
         lr=3e-4, clip_eps=0.2, kl_coef=0.04,
         grpo_epochs=4, ref_update_freq=20):
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    pi_theta = LinearActor(state_dim, num_actions).to(device)
    pi_old = LinearActor(state_dim, num_actions).to(device)
    pi_ref = LinearActor(state_dim, num_actions).to(device)

    pi_ref.load_state_dict(pi_theta.state_dict())

    optimizer = torch.optim.Adam(pi_theta.parameters(), lr=lr)

    for update in range(num_updates):
        pi_old.load_state_dict(pi_theta.state_dict())

        all_trajs = []  # 所有组的 (轨迹, 优势)
        all_rewards = []  # 所有轨迹的回报，用于日志
        for q in range(num_starts):
            seed = np.random.randint(0, 2**31 - 1)
            # 同一起点采 G 条轨迹
            trajectory_group = []  # 这一组的轨迹
            group_rewards = []  # 这一组的reward
            for i in range(group_size):
                trajectory = []  # 采集一条轨迹
                g_reward = 0

                state, _ = env.reset(seed=seed)
                done = False
                while not done:
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_dim)
                    with torch.no_grad():
                        log_probs_old = pi_old(state_t)  # (1, num_actions)
                    action = torch.multinomial(log_probs_old.exp(), 1).item()
                    log_pi_old = log_probs_old[0, action]

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    g_reward += reward
                    trajectory.append((state_t.squeeze(0), action, log_pi_old))

                    state = next_state

                trajectory_group.append(trajectory)
                group_rewards.append(g_reward)

            group_rewards_t = torch.FloatTensor(group_rewards)
            group_advantage = (group_rewards_t - group_rewards_t.mean()) / (group_rewards_t.std() + 1e-8)

            # 优势在组内算好，轨迹先存起来，等所有组采完再统一更新
            all_trajs.extend(zip(trajectory_group, group_advantage))
            all_rewards.extend(group_rewards)

        # 对收集的数据做 K 轮（grpo_epochs）更新：
        for k in range(grpo_epochs):
            # 一条轨迹和一个reward对应
            all_loss = 0
            for traj, A in all_trajs:
                # 对每一个轨迹中的采样点计算 L_clip 和 KL
                states = torch.stack([s for s, _, _ in traj])
                actions = torch.tensor([a for _, a, _ in traj], device=device).view(-1, 1)
                log_pi_old = torch.stack([lp for _, _, lp in traj]).view(-1, 1)

                # log_π_new = π_θ(s) 中 a 对应的 log 概率
                log_probs_new: torch.Tensor = pi_theta(states)
                log_pi_new = log_probs_new.gather(1, actions)

                # log_π_ref = π_ref(s) 中 a 对应的 log 概率（no_grad）
                with torch.no_grad():
                    log_probs_ref: torch.Tensor = pi_ref(states)
                    log_pi_ref = log_probs_ref.gather(1, actions)

                ratio = torch.exp(log_pi_new - log_pi_old)
                clip_loss = torch.min(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A).mean()
                kl_loss = (torch.exp(log_pi_ref - log_pi_new) - (log_pi_ref - log_pi_new) - 1).mean()

                loss = -(clip_loss - kl_coef * kl_loss)

                all_loss += loss
            # 再对所有轨迹（num_starts × group_size 条）求平均
            all_loss /= len(all_trajs)

            # 反向传播更新 π_θ
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        print(f"Update {update + 1} | Mean return: {np.mean(all_rewards):.1f}")

        if update % ref_update_freq == 0:
            pi_ref.load_state_dict(pi_theta.state_dict())

    return pi_theta


if __name__ == "__main__":
    actor = grpo(env)
