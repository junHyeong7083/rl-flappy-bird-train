"""
학습된 PPO 모델 테스트
"""
import argparse

import gymnasium as gym
import flappy_bird_gymnasium
import torch

from config import ENV_ID, STATE_DIM, ACTION_DIM, SAVE_PATH, device, process_obs
from agent import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description="학습된 모델 테스트")
    parser.add_argument("--model", type=str, default=SAVE_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def play(model_path: str, episodes: int, render: bool):
    model = ActorCritic(STATE_DIM, ACTION_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Model loaded: {model_path}")

    env = gym.make(
        ENV_ID,
        render_mode="human" if render else None,
        use_lidar=False
    )

    total_pipes = 0
    best_pipes = 0

    print(f"\n{'='*50}")
    print(f"{'Episode':^10} {'Steps':^10} {'Pipes':^10}")
    print(f"{'='*50}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = process_obs(obs)
        done = False
        pipes = 0
        steps = 0

        while not done:
            with torch.no_grad():
                s = torch.from_numpy(state).float().unsqueeze(0).to(device)
                logits, _ = model(s)
                action = logits.argmax(dim=-1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = process_obs(next_obs)
            steps += 1

            if reward > 0.5:
                pipes += 1

        total_pipes += pipes
        best_pipes = max(best_pipes, pipes)
        print(f"{ep:^10} {steps:^10} {pipes:^10}")

    env.close()

    print(f"{'='*50}")
    print(f"\n[결과]")
    print(f"  평균 파이프: {total_pipes / episodes:.2f}")
    print(f"  최고 파이프: {best_pipes}")


if __name__ == "__main__":
    args = parse_args()
    play(args.model, args.episodes, not args.no_render)
