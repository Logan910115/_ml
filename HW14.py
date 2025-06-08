import gymnasium as gym
import numpy as np

# 創建 CartPole 環境
env = gym.make("CartPole-v1", render_mode="human")  # 可視化模擬

# 初始化記錄
episode_steps = []  # 記錄每個 episode 的步數
max_steps = 1000  # 最大步數限制

# 運行多個 episode
for episode in range(10):  # 測試 10 個 episode
    observation, info = env.reset(seed=42)
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < max_steps:
        env.render()
        # 簡單行動策略：根據桿角度決定
        angle = observation[2]  # 桿角度
        action = 1 if angle < 0 else 0  # 角度 < 0 推右，否則推左
        
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'Episode {episode + 1}, Step {steps + 1}: observation={observation}, action={action}')
        
        steps += 1
        
        if terminated or truncated:
            print(f'Episode {episode + 1} done at step {steps}')
            episode_steps.append(steps)
            observation, info = env.reset()
            break

# 統計與輸出
avg_steps = np.mean(episode_steps)
max_steps_episode = np.max(episode_steps)
print(f'\n統計結果：')
print(f'平均持續步數: {avg_steps:.2f}')
print(f'最長持續步數: {max_steps_episode}')
print(f'每個 episode 持續步數: {episode_steps}')

env.close()