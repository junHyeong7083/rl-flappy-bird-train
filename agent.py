"""
Flappy Bird PPO Agent (SB3 스타일 직접 구현)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from config import (
    STATE_DIM, ACTION_DIM, GAMMA, LR, BATCH_SIZE,
    PPO_EPOCHS, PPO_CLIP, GAE_LAMBDA, ENTROPY_COEF, VALUE_COEF,
    device
)


class ActorCritic(nn.Module):
    """Actor-Critic 네트워크 (SB3 MlpPolicy 스타일)"""
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()

        # Policy network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(64, action_dim)

        # Value network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(64, 1)

        # 가중치 초기화 (SB3 스타일)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # Policy
        policy_latent = self.policy_net(x)
        logits = self.action_head(policy_latent)

        # Value
        value_latent = self.value_net(x)
        value = self.value_head(value_latent)

        return logits, value

    def get_action_and_value(self, state: np.ndarray, deterministic: bool = False):
        """상태에서 액션, log_prob, value 반환"""
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(device)
            logits, value = self(s)

            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def evaluate_actions(self, states, actions):
        """배치 평가 - log_prob, entropy, value 반환"""
        logits, values = self(states)

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)


class RolloutBuffer:
    """경험 저장 버퍼"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value: float):
        """GAE 계산 (SB3 스타일)"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        # GAE 계산
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]

        return advantages, returns

    def get_samples(self, advantages, returns):
        """학습용 데이터 반환"""
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)

        return states, actions, old_log_probs, advantages, returns

    def __len__(self):
        return len(self.states)


def ppo_update(model, optimizer, buffer, last_value):
    """PPO 업데이트 (SB3 스타일)"""
    # GAE 계산
    advantages, returns = buffer.compute_returns_and_advantages(last_value)

    # 데이터 가져오기
    states, actions, old_log_probs, advantages, returns = buffer.get_samples(advantages, returns)

    # 텐서 변환
    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).long().to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    # Advantage 정규화 (SB3 스타일)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # 여러 에폭 학습
    n_samples = len(buffer)
    indices = np.arange(n_samples)

    total_pg_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    n_updates = 0

    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(indices)

        for start in range(0, n_samples, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_indices = indices[start:end]

            # 배치 데이터
            batch_states = states_t[batch_indices]
            batch_actions = actions_t[batch_indices]
            batch_old_log_probs = old_log_probs_t[batch_indices]
            batch_advantages = advantages_t[batch_indices]
            batch_returns = returns_t[batch_indices]

            # 현재 정책 평가
            new_log_probs, entropy, values = model.evaluate_actions(batch_states, batch_actions)

            # Ratio 계산
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # PPO 클리핑
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipping 없이 단순 MSE)
            value_loss = nn.functional.mse_loss(values, batch_returns)

            # Entropy loss (탐색 장려)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            # 업데이트
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_pg_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            n_updates += 1

    return total_pg_loss / n_updates, total_value_loss / n_updates, total_entropy_loss / n_updates
