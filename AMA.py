from stable_baselines3 import PPO1

import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.vec_env import DummyVecEnv
from goldenruleenv import GoldenRuleEnv

class PredictorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PredictorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


env = DummyVecEnv([lambda: GoldenRuleEnv().parralel])
num_agents = 10

agents = [PPO("MlpPolicy", env, verbose=1) for _ in range(num_agents)]
total_actions = sum(env.action_space.nvec)
predictors = [PredictorNetwork(env.observation_space.shape[0], total_actions) for _ in range(num_agents)]

predictor_optimizers = [optim.Adam(pred.parameters(), lr=0.003) for pred in predictors]
obs,_=env.reset()
for _ in range(training_iterations):
    # Для каждого агента выполняем действие
    actions = [agent.predict(obs) for agent in agents]

    # Передаем действия в среду и получаем новые наблюдения и награды
    obs, rewards, dones, truncated,infos = env.step(actions)

    # Обновляем actor-сети каждого агента
    for agent in agents:
        agent.learn(total_timesteps=1)

    # Обновляем predictor-сети каждого агента на основе действий других агентов
    for i, predictor in enumerate(predictors):
        # Выбор действия другого агента как целевого значения
        target_action = actions[1 - i]
        predicted_action = predictor(torch.tensor(obs))

        loss = nn.CrossEntropyLoss()(predicted_action, torch.tensor(target_action))

        predictor_optimizers[i].zero_grad()
        loss.backward()
        predictor_optimizers[i].step()
