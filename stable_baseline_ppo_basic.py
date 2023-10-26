import os.path
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import goldenruleenv
from singleagentwrapper import SingleAgentWrapper
import atexit
import time

num_cpu = 4  # Меняйте на количество ядер вашего процессора

import glob

import glob


def load_latest_checkpoint(model_name, env):
    """
    Загрузите последний чекпоинт или сохраненную модель, в зависимости от того, какой из них новее.
    Если не найдено ни одного, создайте новую модель.
    """
    list_of_files = glob.glob('./checkpoints/*.zip')
    model_path = f'{model_name}.zip'
    model_path_exit = f'{model_name}_exit_save.zip'

    # Соберем все возможные пути в один список и найдем самый новый файл
    all_files = list_of_files + ([model_path] if os.path.exists(model_path) else [])+(
        [model_path_exit] if os.path.exists(model_path_exit) else [])

    if not all_files:  # Если нет ни чекпоинтов, ни сохраненной модели
        print('Creating new model', model_path)
        return PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    # Выберем самый новый файл
    latest_file = max(all_files, key=os.path.getctime)

    print(f"Loading model from {latest_file}")
    return PPO.load(latest_file, env)


def make_env():
    def _init():
        return SingleAgentWrapper(gymnasium.make('GoldenRuleEnv'))

    return _init


# Настройка архитектуры сети
policy_kwargs = dict(
    net_arch=[256, 256,256]  # два скрытых слоя по 64 нейронов
)

PATH = f'basic_ppo_{policy_kwargs["net_arch"]}'

if __name__ == '__main__':
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])
    model = load_latest_checkpoint(PATH, env)
    # Сохраняется каждые 10 минут
    save_freq =10000  # Предположим, что каждый шаг занимает 2 секунды на одном ядре

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./checkpoints/', name_prefix=f'rl_model{policy_kwargs["net_arch"]}')


    def save_on_exit():
        print("Saving model before exiting...", PATH)
        model.save('%s' % PATH)


    atexit.register(save_on_exit)

    model.learn(total_timesteps=5000000, log_interval=1, callback=checkpoint_callback)

    model.save('%s' % PATH)
    print(policy_kwargs)
