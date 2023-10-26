import gymnasium
from stable_baselines3 import PPO
import goldenruleenv
from singleagentwrapper import SingleAgentWrapper

env = SingleAgentWrapper(gymnasium.make('GoldenRuleEnv'))


# Создание модели
model = PPO("MlpPolicy", env, verbose=1)

# Обучение модели
model.learn(total_timesteps=100000)

model.save('basic_ppo')

obs = env.reset()
