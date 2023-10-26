import gymnasium
import gymnasium as gym
import numpy
import pygame as pygame
from gymnasium import spaces
import numpy as np
import random

from gymnasium.spaces import MultiDiscrete

from agent import Agent
from resources import Resource






# Основной класс среды
class GoldenRuleEnv(gym.Env):
    def __init__(self, world_size=500, vision_radius=50,render_mode='human'):
        super(GoldenRuleEnv, self).__init__()

        self.world_size = world_size  # Размер мира
        self.vision_radius = vision_radius  # Радиус видимости агента
        self.agents = []  # Список агентов
        self.resources = []  # Список ресурсов
        self.spawn_rate=0.001
        self.init_agents=10
        self.init_resourses=20

        # Пространство действий: dx, dy (-1, 0, 1), атака/не атаковать (0, 1), сеьсть яблоко/не есть(0, 1)
        self.action_space = MultiDiscrete([3, 3, 2, 2])

        # Пространство наблюдений: пока что просто координаты всех агентов и ресурсов в радиусе видимости
        self.observation_space = spaces.Box(low=0, high=self.world_size, shape=(34, ), dtype=np.float32)
        self.reset()
        #if render_mode=='human':
        pygame.init()
        self.screen = pygame.display.set_mode((self.world_size, self.world_size))
        pygame.display.set_caption("Golden Rule Environment")


    def reset(self,seed=None,options=None):
        # Сброс среды
        self.agents = [Agent(random.randint(0, self.world_size), random.randint(0, self.world_size),env=self) for _ in range(self.init_agents)]  # Примерное количество агентов
        self.resources = [Resource(random.randint(0, self.world_size), random.randint(0, self.world_size)) for _ in range(self.init_resourses)]  # Примерное количество ресурсов
        return self._get_observation(),{}

    def _get_observation(self):
        max_nearby_agents = 5

        max_nearby_resources = 5
        agent_data_length = 4
        nearby_agent_data_length = 3
        nearby_resource_data_length = 3

        total_observation_length = agent_data_length + \
                                   max_nearby_agents * nearby_agent_data_length + \
                                   max_nearby_resources * nearby_resource_data_length

        observations = []

        for agent in self.agents:
            nearby_agents = sorted(
                [other.get_visible_state() for other in self.agents if
                 (other.x - agent.x) ** 2 + (other.y - agent.y) ** 2 <= self.vision_radius ** 2 and other != agent],
                key=lambda other: (other[0] - agent.x) ** 2 + (other[1] - agent.y) ** 2
            )[:max_nearby_agents]

            nearby_resources = sorted(
                [res.get_visible_state() for res in self.resources if
                 (res.x - agent.x) ** 2 + (res.y - agent.y) ** 2 <= self.vision_radius ** 2],
                key=lambda res: (res[0] - agent.x) ** 2 + (res[1] - agent.y) ** 2
            )[:max_nearby_resources]

            # Дополняем списки до максимального размера нулевыми векторами
            while len(nearby_agents) < max_nearby_agents:
                nearby_agents.append([0, 0, 0])

            while len(nearby_resources) < max_nearby_resources:
                nearby_resources.append([0, 0])

            observation = [agent.x, agent.y, agent.health, agent.resources]
            for other in nearby_agents:
                observation.extend(other)
            for res in nearby_resources:
                observation.extend(res)

            # Убедимся, что наблюдение имеет правильную длину, и дополним его нулями, если это необходимо
            while len(observation) < total_observation_length:
                observation.append(0)

            observations.append(observation)

        return numpy.array(observations)

    def step(self, actions):
        rewards = []

        # Для каждого агента
        for i, agent in enumerate(self.agents):
            action = actions[i]
            dx, dy, attack,eat = action  # Разбираем действие на компоненты
            attack=attack==1
            eat=eat==1
            # Двигаем агента
            agent.move(dx, dy)

            nearby_food=self._find_nearest_food(agent)
            if nearby_food and agent.resources<agent.max_resources:
                agent.resources+=nearby_food.amount
                self.resources.remove(nearby_food)

            if eat and agent.resources>0 and agent.health<agent.max_health:
                agent.resources-=1
                agent.health+=1

            # Если агент решил атаковать
            if attack:
                # Найдем ближайшего агента в радиусе атаки и атакуем его
                target = self._find_nearest_agent(agent)
                if target:
                    agent.attack(target)

            # Вычисляем награду для агента
            reward = self._compute_reward(agent)
            rewards.append(reward)
            if agent.health<=0:
                self.agents.remove(agent)

        # Обновляем ресурсы и другие аспекты мира
        self._update_world()

        observations = self._get_observation()
        done = len(self.agents)<=1  # условие завершения эпизода
        trunctated=max(agent.resources for agent in self.agents)<=0
        info = {}  # дополнительная информация
        self.render()
        return observations, rewards, done,trunctated, info


    def render(self, mode='human'):
       self.screen.fill((255, 255, 255))  # Очистка экрана (белый цвет)

       # Отображение ресурсов (яблок) зелеными кругами
       for res in self.resources:
           pygame.draw.circle(self.screen, (0, 255, 0), (res.x, res.y), 5)

       # Отображение агентов красными кругами
       for agent in self.agents:
           pygame.draw.circle(self.screen, (255, 0, 0), (agent.x, agent.y), 5)

       pygame.display.flip()  # Обновление экрана

    def close(self):
      pygame.quit()
    def _find_nearest_agent(self, agent):
        min_distance = float('inf')
        nearest_agent = None
        for other in self.agents:
            if other != agent:
                distance = (other.x - agent.x)**2 + (other.y - agent.y)**2
                if distance < min_distance and distance <= agent.attack_radius**2:  # Предположим, радиус атаки = радиусу видимости
                    min_distance = distance
                    nearest_agent = other
        return nearest_agent
    def _find_nearest_food(self, agent: Agent) -> Resource:
            min_distance = float('inf')
            nearest_food = None
            for food in self.resources:
                distance = (food.x - agent.x) ** 2 + (food.y - agent.y) ** 2
                if distance < min_distance and distance <= 5:  # Предположим, радиус атаки = радиусу видимости
                    min_distance = distance
                    nearest_food = food
            return nearest_food

    def _compute_reward(self, agent):
        # Награда или штраф на основе действий и состояния агента
        # Пример:
        reward = 0
        if agent.prev_resources > agent.resources:
            reward += 1
        if agent.health<agent.prev_health:
            reward-=2
        if agent.health==0:
            reward-=10
        if agent.resources==0:
            reward-=0.5
        return reward

    def _update_world(self):
        # Обновляем ресурсы и другие аспекты мира
        # Пример: спавн новых ресурсов
        if random.random() < self.spawn_rate:  # Предположим, у нас есть некий коэффициент спавна ресурсов
            self.resources.append(Resource(random.randint(0, self.world_size), random.randint(0, self.world_size)))


gymnasium.register('GoldenRuleEnv','goldenruleenv:GoldenRuleEnv')
