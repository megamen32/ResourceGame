import os.path

import gymnasium
from stable_baselines3 import PPO
import goldenruleenv
from singleagentwrapper import SingleAgentWrapper



env = SingleAgentWrapper(gymnasium.make('GoldenRuleEnv'))

# Настройка архитектуры сети
policy_kwargs = dict(
    net_arch=[256, 256]  # два скрытых слоя по 256 нейронов
)
PATH = f'basic_ppo_{policy_kwargs}'
# Создание модели
if os.path.exists('%s.zip' % PATH):
    model = PPO.load('%s' % PATH, env)
else:
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

# Обучение модели
model.learn(total_timesteps=100000)

model.save('%s' % PATH)

obs = env.reset()
