import gymnasium as gym
import flappy_bird_gymnasium  # 환경 등록용

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
# use_lidar=True: 숫자 상태(180개 레이저) / False면 RGB 화면

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 0: 아무것도 안 함, 1: 점프
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
