"""학습 로그 시각화"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.txt")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Reward
axes[0, 0].plot(df["episode"], df["reward"], alpha=0.3)
axes[0, 0].plot(df["episode"], df["reward"].rolling(100).mean(), color="red", label="MA100")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].set_title("Episode Reward")
axes[0, 0].legend()

# 2. Score (파이프 통과)
axes[0, 1].plot(df["episode"], df["score"], alpha=0.3)
axes[0, 1].plot(df["episode"], df["score"].rolling(100).mean(), color="red", label="MA100")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Score")
axes[0, 1].set_title("Pipes Passed")
axes[0, 1].legend()

# 3. Policy Loss
axes[1, 0].plot(df["episode"], df["pg_loss"])
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Policy Loss")
axes[1, 0].set_title("Policy Loss")

# 4. Value Loss
axes[1, 1].plot(df["episode"], df["v_loss"])
axes[1, 1].set_xlabel("Episode")
axes[1, 1].set_ylabel("Value Loss")
axes[1, 1].set_title("Value Loss")

plt.tight_layout()
plt.savefig("training_plot.png", dpi=150)
plt.show()

print("Saved to training_plot.png")
