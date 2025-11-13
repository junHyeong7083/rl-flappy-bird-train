import os
import math
import random
from collections import deque
import argparse
import time

import gymnasium as gym
import flappy_bird_gymnasium  # env 등록용
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ======================
# 하이퍼파라미터
# ======================
ENV_ID = "FlappyBird-v0"

STATE_DIM = 4        # [새 y, 수직속도, 다음파이프까지 x, 다음파이프틈 y]
ACTION_DIM = 2       # 0: NOOP, 1: FLAP

GAMMA = 0.99
LR = 5e-5
BATCH_SIZE = 64
REPLAY_SIZE = 100_000
MIN_REPLAY_SIZE = 2_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 80_000

TARGET_UPDATE_FREQ = 1_000
MAX_EPISODES = 3_000
MAX_STEPS_PER_EP = 2_000

SAVE_PATH = "dqn_flappy.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        action="store_true",
        help="학습 중 플래피버드 화면 렌더링 (느려짐)"
    )
    return parser.parse_args()


# ======================
# 상태 전처리 (12차원 → 4차원)
# ======================
def process_obs(obs: np.ndarray) -> np.ndarray:
    """
    flappy_bird_gymnasium 의 use_lidar=False 일 때 obs 형태 (길이 12):

    0: last pipe horizontal position
    1: last top pipe vertical position
    2: last bottom pipe vertical position
    3: next pipe horizontal position
    4: next top pipe vertical position
    5: next bottom pipe vertical position
    6: next next pipe horizontal position
    7: next next top pipe vertical position
    8: next next bottom pipe vertical position
    9: player vertical position
    10: player vertical velocity
    11: player rotation

    여기서 기획서의 상태 정의에 맞게 4개만 뽑는다.
    [새 y, 수직속도, 다음 파이프까지 x, 다음 파이프 틈 y]
    """
    last_pipe_x, last_top_y, last_bottom_y, \
    next_pipe_x, next_top_y, next_bottom_y, \
    next2_pipe_x, next2_top_y, next2_bottom_y, \
    player_y, player_v, player_rot = obs

    gap_y = (next_top_y + next_bottom_y) / 2.0
    horiz_dist = next_pipe_x  # 플레이어 x는 거의 고정이라, x좌표 자체가 사실상 거리

    state = np.array(
        [player_y, player_v, horiz_dist, gap_y],
        dtype=np.float32
    )

    return state


# ======================
# DQN 신경망
# ======================
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================
# 리플레이 버퍼
# ======================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ======================
# Epsilon-greedy 정책
# ======================
def get_epsilon(step: int) -> float:
    # 지수 감소
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)


def select_action(policy_net: DQN, state: np.ndarray, step: int) -> int:
    eps = get_epsilon(step)
    if random.random() < eps:
        return random.randrange(ACTION_DIM)
    else:
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = policy_net(s)
            return int(q_values.argmax(dim=1).item())


# ======================
# 학습 스텝
# ======================
def optimize_model(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer
):
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.from_numpy(states).to(device)
    actions = torch.from_numpy(actions).long().to(device)
    rewards = torch.from_numpy(rewards).to(device)
    next_states = torch.from_numpy(next_states).to(device)
    dones = torch.from_numpy(dones).to(device)

    # Q(s,a) 계산
    q_values = policy_net(states)
    # (batch, action_dim) -> (batch,)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s',a')
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q_values, _ = next_q_values.max(dim=1)
        target_values = rewards + GAMMA * max_next_q_values * (1.0 - dones)

    loss = nn.MSELoss()(state_action_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()


# ======================
# 메인 학습 루프
# ======================
def main(render: bool):
    print(f"[INFO] render = {render}")

    env = gym.make(
        ENV_ID,
        render_mode="human" if render else None,
        use_lidar=False
    )

    policy_net = DQN(STATE_DIM, ACTION_DIM).to(device)
    target_net = DQN(STATE_DIM, ACTION_DIM).to(device)

    if os.path.exists(SAVE_PATH):
        print(f"[INFO] Found existion checkpoint: {SAVE_PATH}")
        state_dict = torch.load(SAVE_PATH, map_location=device)
        policy_net.load_state_dict(state_dict)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    #target_net.load_state_dict(policy_net.state_dict())
    #target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    global_step = 0

    # 초기 리플레이 채우기 (랜덤 정책)
    print("Filling replay buffer with random policy...")
    while len(replay_buffer) < MIN_REPLAY_SIZE:
        obs, _ = env.reset()
        state = process_obs(obs)
        done = False

        while not done:
            action = random.randrange(ACTION_DIM)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = process_obs(next_obs)

            # 보상 shaping: gap 중심에서 멀어질수록 살짝 패널티
            player_y, player_v, horiz_dist, gap_y = next_state
            dist_to_gap = abs(player_y - gap_y)
            shaped_reward = reward - 0.001 * dist_to_gap

            replay_buffer.push(state, action, shaped_reward, next_state, done)
            state = next_state
            global_step += 1

            if render:
                env.render()
                time.sleep(0.01)

            if len(replay_buffer) >= MIN_REPLAY_SIZE:
                break

    print("Replay buffer filled. Start training...")

    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset()
        state = process_obs(obs)
        done = False

        episode_reward = 0.0
        step_in_ep = 0

        while not done and step_in_ep < MAX_STEPS_PER_EP:
            action = select_action(policy_net, state, global_step)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = process_obs(next_obs)

            # 보상 shaping
            player_y, player_v, horiz_dist, gap_y = next_state
            dist_to_gap = abs(player_y - gap_y)
            shaped_reward = reward - 0.001 * dist_to_gap

            replay_buffer.push(state, action, shaped_reward, next_state, done)
            episode_reward += shaped_reward
            state = next_state

            optimize_model(policy_net, target_net, optimizer, replay_buffer)

            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            global_step += 1
            step_in_ep += 1

            if render:
                env.render()
                time.sleep(0.01)

        print(
            f"[Episode {episode:4d}] "
            f"steps={step_in_ep:4d}  "
            f"reward={episode_reward:7.3f}  "
            f"eps={get_epsilon(global_step):.3f}"
        )

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"Model saved to {SAVE_PATH}")

    env.close()
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print("Training finished. Final model saved.")


if __name__ == "__main__":
    args = parse_args()
    main(render=args.render)
