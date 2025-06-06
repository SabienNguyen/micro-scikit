import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# 1. Create the CartPole environment
env = make_vec_env("CartPole-v1", n_envs=1)

# 2. Create and train the DQN agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    train_freq=4,
    target_update_interval=100,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./cartpole_tensorboard/"
)

# 3. Train the model
model.learn(total_timesteps=10000)
model.save("dqn_cartpole")

# 4. Test the trained model
test_env = gym.make("CartPole-v1", render_mode="human")
obs, _ = test_env.reset()

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated

test_env.close()
