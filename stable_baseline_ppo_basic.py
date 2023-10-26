import os.path
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import goldenruleenv
from singleagentwrapper import SingleAgentWrapper

num_cpu = 4  # Меняйте на количество ядер вашего процессора


def make_env():
    def _init():
        return SingleAgentWrapper(gymnasium.make('GoldenRuleEnv'))

    return _init



# Настройка архитектуры сети
policy_kwargs = dict(
    net_arch=[64, 64]  # два скрытых слоя по 256 нейронов
)
PATH = f'basic_ppo_{policy_kwargs["net_arch"]}'
if __name__ == '__main__':
    # Создание параллельных сред
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    # Создание модели
    if os.path.exists('%s.zip' % PATH):
        model = PPO.load('%s' % PATH, env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    # Обучение модели
    model.learn(total_timesteps=500000)

    model.save('%s' % PATH)

    obs = env.reset()
