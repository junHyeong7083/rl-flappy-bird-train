"""
Flappy Bird PPO Configuration (SB3 스타일)
"""
import torch
import numpy as np

# ======================
# 환경 설정
# ======================
ENV_ID = "FlappyBird-v0"
STATE_DIM = 12   # 원본 observation 사용 (전처리 안 함)
ACTION_DIM = 2

# ======================
# PPO 하이퍼파라미터 (SB3 기본값)
# ======================
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
BATCH_SIZE = 64

PPO_EPOCHS = 10
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01  # 탐색 유지
VALUE_COEF = 0.5

# 학습 설정
N_STEPS = 2048       # 업데이트 전 수집할 스텝 수
MAX_TIMESTEPS = 5_000_000

# 저장 경로
SAVE_PATH = "ppo_flappy.pth"

# 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    print("Using device:", device)
