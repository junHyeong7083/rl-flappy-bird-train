"""
Flappy Bird PPO Training (SB3 스타일 직접 구현)
"""
import os
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import torch.optim as optim

from config import (
    ENV_ID, STATE_DIM, ACTION_DIM, LR,
    N_STEPS, MAX_TIMESTEPS, SAVE_PATH,
    device, print_device_info
)
from agent import ActorCritic, RolloutBuffer, ppo_update


def train():
    print_device_info()

    # 로그 파일 설정
    log_file = open("training_log.txt", "w")
    log_file.write("episode,timestep,reward,length,score,pg_loss,v_loss\n")

    # 환경 생성
    env = gym.make(ENV_ID, use_lidar=False)
    print(f"Environment: {ENV_ID}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")

    # 모델 생성
    model = ActorCritic(STATE_DIM, ACTION_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # 체크포인트 로드
    if os.path.exists(SAVE_PATH):
        print(f"Loading checkpoint: {SAVE_PATH}")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    buffer = RolloutBuffer()

    # 학습 변수
    obs, _ = env.reset()
    state = obs.astype(np.float32)  # 원본 observation 사용

    total_timesteps = 0
    episode = 0
    episode_reward = 0
    episode_length = 0
    best_reward = -float('inf')
    last_pg_loss, last_v_loss = 0.0, 0.0

    print(f"\nStarting training for {MAX_TIMESTEPS} timesteps...")
    print("=" * 60)

    while total_timesteps < MAX_TIMESTEPS:
        # 롤아웃 수집
        buffer.reset()

        for step in range(N_STEPS):
            action, log_prob, value = model.get_action_and_value(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, reward, done, log_prob, value)

            episode_reward += reward
            episode_length += 1
            total_timesteps += 1

            if done:
                episode += 1

                # 에피소드 로깅
                score = info.get("score", 0)
                log_file.write(f"{episode},{total_timesteps},{episode_reward:.2f},{episode_length},{score},{last_pg_loss:.4f},{last_v_loss:.4f}\n")
                log_file.flush()

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(model.state_dict(), SAVE_PATH)
                    print(f"[Ep {episode:5d}] reward={episode_reward:7.2f} len={episode_length:4d} score={score:2d} ★ BEST")
                elif episode % 10 == 0:
                    print(f"[Ep {episode:5d}] reward={episode_reward:7.2f} len={episode_length:4d} score={score:2d}")

                # 리셋
                obs, _ = env.reset()
                state = obs.astype(np.float32)
                episode_reward = 0
                episode_length = 0
            else:
                state = next_obs.astype(np.float32)

        # PPO 업데이트
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(device)
            _, last_value = model(s)
            last_value = last_value.squeeze().item()

        pg_loss, v_loss, ent_loss = ppo_update(model, optimizer, buffer, last_value)
        last_pg_loss, last_v_loss = pg_loss, v_loss

        # 업데이트 로깅
        print(f"  [Update @ {total_timesteps:7d}] pg_loss={pg_loss:.4f} v_loss={v_loss:.4f}")

    # 최종 저장
    torch.save(model.state_dict(), SAVE_PATH)
    log_file.close()
    print("=" * 60)
    print(f"Training finished. Best reward: {best_reward:.2f}")
    print(f"Model saved to {SAVE_PATH}")
    print(f"Log saved to training_log.txt")

    env.close()


def play(render=True):
    """학습된 모델 테스트"""
    env = gym.make(ENV_ID, render_mode="human" if render else None, use_lidar=False)

    model = ActorCritic(STATE_DIM, ACTION_DIM).to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {SAVE_PATH}")

    for ep in range(10):
        obs, _ = env.reset()
        state = obs.astype(np.float32)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _, _ = model.get_action_and_value(state, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_obs.astype(np.float32)
            total_reward += reward
            steps += 1

        score = info.get("score", 0)
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, score={score}")

    env.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play()
    else:
        train()
