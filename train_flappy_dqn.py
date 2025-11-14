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

STATE_DIM = 4   # [ny, nv, nx, ng] (정규화된 새 y, 속도, 파이프거리, 틈 y)
ACTION_DIM = 2  # 0: NOOP, 1: FLAP

GAMMA = 0.99
LR = 5e-5
BATCH_SIZE = 64
REPLAY_SIZE = 100_000
MIN_REPLAY_SIZE = 2_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20_000   # 2000 에피소드 기준, 탐색 좀 빠르게 줄이기

TARGET_UPDATE_FREQ = 1_000
MAX_EPISODES = 2_000        # 에피소드 수 (필요하면 여기 늘리면 됨)
MAX_STEPS_PER_EP = 2_000

SAVE_PATH = "dqn_flappy.pth"

# === 모방학습(Behavior Cloning) 관련 하이퍼파라미터 ===
BC_STEPS = 5_000         # 휴리스틱 정책 샘플 수집 스텝 수
BC_EPOCHS = 5            # 모방학습 에폭 수
BC_BATCH_SIZE = 256      # 모방학습 배치 크기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
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
# 상태 전처리 (12차원 → 4차원 + 정규화)
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
    """
    last_pipe_x, last_top_y, last_bottom_y, \
    next_pipe_x, next_top_y, next_bottom_y, \
    next2_pipe_x, next2_top_y, next2_bottom_y, \
    player_y, player_v, player_rot = obs

    gap_y = (next_top_y + next_bottom_y) / 2.0
    horiz_dist = next_pipe_x

    # 대충 스케일 가정해서 [-1,1] 근처로 정규화
    y_scale = 300.0
    v_scale = 10.0
    x_scale = 300.0

    ny = (player_y - y_scale / 2.0) / y_scale    # ~[-0.5, 0.5]
    nv = player_v / v_scale                      # ~[-1, 1]
    nx = horiz_dist / x_scale                    # [0, 1] 근처
    ng = (gap_y - y_scale / 2.0) / y_scale       # ~[-0.5, 0.5]

    state = np.array(
        [ny, nv, nx, ng],
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

    q_values = policy_net(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

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
# 보상 shaping 함수
# ======================
def shape_reward(raw_reward: float) -> float:
    """
    규칙:
    - 매 타임스텝: -0.1
    - 파이프 통과(raw_reward > 0.5로 가정): +50
    """
    r = -0.1
    if raw_reward > 0.5:
        r += 50.0
    return r


# ======================
# 휴리스틱 정책 (모방학습용)
# ======================
def heuristic_policy(state: np.ndarray) -> int:
    """
    state = [ny, nv, nx, ng]
    - ny: 새의 정규화된 y
    - nv: 수직 속도
    - nx: 다음 파이프까지 거리
    - ng: 파이프 틈 정규화된 y

    대충:
    - 새가 구멍보다 많이 아래 → 점프(1)
    - 많이 위 → 점프 안 함(0)
    - 중간대 → 속도 보고 결정
    """
    ny, nv, nx, ng = state

    # 구멍보다 너무 아래에 있으면 점프
    if ny < ng - 0.05:
        return 1  # FLAP
    # 구멍보다 너무 위에 있으면 점프하지 않기
    if ny > ng + 0.05:
        return 0  # NOOP

    # 구멍 근처에서는 천천히 떨어지는 쪽을 선호
    return 1 if nv > 0.1 else 0


# ======================
# 모방학습용 데이터 수집
# ======================
def collect_bc_dataset(env, steps: int):
    print(f"[BC] Collecting heuristic dataset for {steps} steps...")
    states = []
    actions = []

    obs, _ = env.reset()
    state = process_obs(obs)

    for t in range(steps):
        action = heuristic_policy(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        state = process_obs(obs)

        if (t + 1) % 1000 == 0:
            print(f"[BC] Collected {t+1} transitions")

    states = np.stack(states).astype(np.float32)
    actions = np.array(actions, dtype=np.int64)
    print(f"[BC] Dataset shape: states={states.shape}, actions={actions.shape}")
    return states, actions


# ======================
# 모방학습(Behavior Cloning)
# ======================
def behavior_cloning_train(model: DQN, states: np.ndarray, actions: np.ndarray):
    print(f"[BC] Start behavior cloning: epochs={BC_EPOCHS}, batch_size={BC_BATCH_SIZE}")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    all_states = torch.from_numpy(states).to(device)
    all_actions = torch.from_numpy(actions).to(device)

    dataset_size = all_states.size(0)
    indices = np.arange(dataset_size)

    for epoch in range(1, BC_EPOCHS + 1):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, dataset_size, BC_BATCH_SIZE):
            end = start + BC_BATCH_SIZE
            batch_idx = indices[start:end]

            batch_states = all_states[batch_idx]
            batch_actions = all_actions[batch_idx]

            logits = model(batch_states)    # (B, ACTION_DIM)
            loss = loss_fn(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"[BC] Epoch {epoch}/{BC_EPOCHS} - loss={avg_loss:.4f}")

    print("[BC] Behavior cloning finished.")


# ======================
# (선택) 정책 평가 함수 - 필요하면 직접 호출해서 테스트
# ======================
def eval_policy(policy_net: DQN, episodes: int = 10, render: bool = True):
    env = gym.make(ENV_ID, render_mode="human" if render else None, use_lidar=False)
    policy_net.eval()

    total_pipes = 0
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = process_obs(obs)
        done = False
        pipes = 0
        steps = 0

        while not done and steps < MAX_STEPS_PER_EP:
            with torch.no_grad():
                s = torch.from_numpy(state).unsqueeze(0).to(device)
                action = int(policy_net(s).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = process_obs(next_obs)
            steps += 1

            if reward > 0.5:
                pipes += 1

        total_pipes += pipes
        print(f"[EVAL ep {ep:3d}] steps={steps:4d}  pipes={pipes}")

    env.close()
    print(f"[EVAL] avg pipes = {total_pipes / episodes:.2f}")


# ======================
# 메인 학습 루프
# ======================
def main(render: bool):
    print(f"[INFO] render = {render}")

    # 모방학습/강화학습 공용 env (처음엔 렌더 없이)
    env = gym.make(
        ENV_ID,
        render_mode=None,
        use_lidar=False
    )

    policy_net = DQN(STATE_DIM, ACTION_DIM).to(device)
    target_net = DQN(STATE_DIM, ACTION_DIM).to(device)

    global_step = 0

    # 1) 체크포인트 있으면 로드, 없으면 모방학습 → 초기화
    if os.path.exists(SAVE_PATH):
        print(f"[INFO] Found existing checkpoint: {SAVE_PATH}")
        state_dict = torch.load(SAVE_PATH, map_location=device)
        policy_net.load_state_dict(state_dict)
        print("[INFO] Checkpoint loaded into policy_net (skip BC).")
    else:
        # (1) 휴리스틱 정책으로 데이터 수집
        bc_states, bc_actions = collect_bc_dataset(env, BC_STEPS)
        # (2) 모방학습으로 DQN 초기 가중치 학습
        behavior_cloning_train(policy_net, bc_states, bc_actions)
        # (3) 초기 모델 저장
        torch.save(policy_net.state_dict(), SAVE_PATH)
        print(f"[BC] Initial model saved to {SAVE_PATH}")

    # 이제 target_net 초기화
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    # 2) 본격 DQN 학습 - 리플레이 버퍼 채우기 (이제 랜덤 X, BC policy로 채우기)
    print("Filling replay buffer with BC policy (greedy)...")
    policy_net.eval()
    while len(replay_buffer) < MIN_REPLAY_SIZE:
        obs, _ = env.reset()
        state = process_obs(obs)
        done = False

        while not done:
            with torch.no_grad():
                s = torch.from_numpy(state).unsqueeze(0).to(device)
                action = int(policy_net(s).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = process_obs(next_obs)

            shaped_reward = shape_reward(reward)

            replay_buffer.push(state, action, shaped_reward, next_state, done)

            state = next_state
            global_step += 1

            if len(replay_buffer) >= MIN_REPLAY_SIZE:
                break

    print("Replay buffer filled. Start DQN training...")

    # 렌더링 옵션 반영 위해 env 다시 생성
    env.close()
    env = gym.make(
        ENV_ID,
        render_mode="human" if render else None,
        use_lidar=False
    )

    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset()
        state = process_obs(obs)
        done = False

        episode_reward = 0.0
        step_in_ep = 0
        pipes_cleared = 0

        while not done and step_in_ep < MAX_STEPS_PER_EP:
            action = select_action(policy_net, state, global_step)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = process_obs(next_obs)

            # 파이프 통과 감지 (env 보상 > 0.5로 가정)
            if reward > 0.5:
                pipes_cleared += 1

            shaped_reward = shape_reward(reward)

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
            f"[Episode {episode:5d}] "
            f"steps={step_in_ep:4d}  "
            f"reward={episode_reward:7.3f}  "
            f"pipes={pipes_cleared:2d}  "
            f"eps={get_epsilon(global_step):.3f}"
        )

        if episode % 200 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"Model saved to {SAVE_PATH}")

    env.close()
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print("Training finished. Final model saved.")


if __name__ == "__main__":
    args = parse_args()
    main(render=args.render)
