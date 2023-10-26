import gymnasium as gym
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)
        self.observation_space = env.observation_space  # Берем пространство наблюдений для одного агента

    def reset(self, **kwargs):
        obs,_ = self.env.reset(**kwargs)
        return obs[0],_  # Возвращаем наблюдение для одного агента

    def step(self, action):
        actions = [action] + [[0, 0, 0, 0] for _ in
                              range(len(self.env.agents) - 1)]  # Действие для одного агента, остальные не активны
        obs, reward, done, truncated,info = self.env.step(actions)
        return obs[0], reward[0], done, truncated,info
