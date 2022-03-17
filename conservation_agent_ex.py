import gym
import gym_conservation

from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Parallel environments

# Conservation V5
env = make_vec_env("conservation-v5", n_envs=4)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("conservation-v5-A2C-sarah")
eval_env = Monitor(gym.make("conservation-v5"))
score_A2C_5 = evaluate_policy(model, Monitor(eval_env), n_eval_episodes=10)
score_A2C_5
