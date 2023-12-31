import math

import gymnasium
import gymnasium as gym
import numpy
import pygame as pygame
from gymnasium import spaces
import numpy as np
import random

from gymnasium.spaces import MultiDiscrete

from agent import Agent, AttackAnimation
from resources import Resource






# Основной класс среды
class GoldenRuleEnv(gym.Env):
    def __init__(self, world_size=500, vision_radius=200,render_mode='human'):
        super(GoldenRuleEnv, self).__init__()

        self.starvation = 1/500/2
        self.world_size = world_size  # Размер мира
        self.vision_radius = vision_radius  # Радиус видимости агента
        self.agents = []  # Список агентов
        self.resources = []  # Список ресурсов
        self.spawn_rate=0.1
        self.init_agents=10
        self.init_resourses=20

        # Пространство действий: dx, dy (-1, 0, 1), атака/не атаковать (0, 1), сеьсть яблоко/не есть(0, 1)
        self.action_space = MultiDiscrete([3, 3, 2, 2])

        # Пространство наблюдений: пока что просто координаты всех агентов и ресурсов в радиусе видимости
        self.observation_space = spaces.Box(low=-self.world_size, high=self.world_size, shape=(32, ), dtype=np.float64)
        self.reset()
        #if render_mode=='human':
        pygame.init()
        self.screen = pygame.display.set_mode((self.world_size, self.world_size))
        pygame.display.set_caption("Golden Rule Environment")


    def reset(self,seed=None,options=None):
        # Сброс среды
        self.agents = [Agent(random.randint(0, self.world_size), random.randint(0, self.world_size),env=self) for _ in range(self.init_agents)]  # Примерное количество агентов
        self.resources = [Resource(random.randint(0, self.world_size), random.randint(0, self.world_size),random.randint(1, 5)) for _ in range(self.init_resourses)]  # Примерное количество ресурсов
        self.attack_animations = []

        return self._get_observation(),{}

    def _get_observation(self):
        max_nearby_agents = 5

        max_nearby_resources = 5
        agent_data_length = 2
        nearby_agent_data_length = 3
        nearby_resource_data_length = 3

        total_observation_length = agent_data_length + \
                                   max_nearby_agents * nearby_agent_data_length + \
                                   max_nearby_resources * nearby_resource_data_length

        observations = []

        for agent in self.agents:
            nearby_agents = sorted(
                [(other.x - agent.x, other.y - agent.y, other.health) for other in self.agents if
                 (other.x - agent.x) ** 2 + (other.y - agent.y) ** 2 <= self.vision_radius ** 2 and other != agent],
                key=lambda other: other[0] ** 2 + other[1] ** 2
            )[:max_nearby_agents]

            nearby_resources = sorted(
                [(res.x - agent.x, res.y - agent.y) for res in self.resources if
                 (res.x - agent.x) ** 2 + (res.y - agent.y) ** 2 <= self.vision_radius ** 2],
                key=lambda res: res[0] ** 2 + res[1] ** 2
            )[:max_nearby_resources]

            # Дополняем списки до максимального размера нулевыми векторами
            while len(nearby_agents) < max_nearby_agents:
                nearby_agents.append([0, 0, 0])

            while len(nearby_resources) < max_nearby_resources:
                nearby_resources.append([0, 0,0])

            # Поскольку мы теперь используем относительные координаты, начальные координаты агента будут [0, 0]
            observation = [agent.health, agent.resources]
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
        dones=[False for _ in range((self.init_agents))]
        truncate=[False for _ in range((self.init_agents))]
        if not self.agents:
            dones= [True for _ in range((self.init_agents))]

        for i, agent in enumerate(self.agents):
            action = actions[i]
            dx, dy, attack,eat = action  # Разбираем действие на компоненты
            dx = dx - 1  # Переводим [0, 1, 2] в [-1, 0, 1]
            dy = dy - 1  # То же самое
            attack=attack==1
            eat=eat==1
            # Двигаем агента
            agent.move(dx, dy)


            nearby_food=self._find_nearest_food(agent)
            agent.is_eating -=1
            if nearby_food and agent.resources<agent.max_resources:
                agent.resources+=nearby_food.amount
                agent.resources=min(agent.max_resources,agent.resources)
                self.resources.remove(nearby_food)
                agent.is_eating=2

            if eat and agent.resources>=1 and agent.health<agent.max_health:
                agent.resources-=1
                agent.health+=1

            # Если агент решил атаковать
            #agent.is_attacking=False
            agent.attack_idle-=1
            if attack and agent.attack_idle<=0:
                agent.is_attacking = True
                agent.attack_idle=agent.time_between_attacks
                agent.resources -= 1/2
                # Найдем ближайшего агента в радиусе атаки и атакуем его
                target = self._find_nearest_agent(agent)
                if target:
                    agent.attack(target)

            # Вычисляем награду для агента
            reward = self._compute_reward(agent)
            rewards.append(reward)
            if agent.health<=0 or agent.resources <= 0:
                dones[i]=True
                self.agents.remove(agent)
            #if agent.resources<=0:
                #truncate[i]=True

        # Обновляем ресурсы и другие аспекты мира
        self._update_world()

        observations = self._get_observation()
        #done = len(self.agents)<=1  # условие завершения эпизода
        #trunctated=max(agent.resources for agent in self.agents)<=0
        info = {}  # дополнительная информация
        self.render()
        if  len(rewards)==0  or len(observations)==0 :
            dones = [True for _ in range((self.init_agents))]
            truncate = [False for _ in range((self.init_agents))]
            observations = [[0 for i in range(self.observation_space.shape[0])] for _ in range((self.init_agents))]
            rewards = [0 for _ in range((self.init_agents))]
        if len(self.agents)==1:
            dones = [True for _ in range((self.init_agents))]
        return observations, rewards, dones,truncate, info


    def render(self, mode='human'):
       self.screen.fill((255, 255, 255))  # Очистка экрана (белый цвет)

       # Отображение ресурсов (яблок) зелеными кругами
       for res in self.resources:
           pygame.draw.circle(self.screen, (0, 255, 0), (res.x, res.y), 2*res.amount)

       # Отображение агентов красными кругами
       for i, agent in enumerate(self.agents):
            i1 = max(0, 255 * agent.resources / agent.max_resources)
            color = (i1, 0, 0 if i != 0 else i1)
            if i == 0:
                # Рисуем квадрат для нулевого агента
                size = math.log2(max(1,min( agent.health*4,agent.max_health*4)))*2
                pygame.draw.rect(self.screen, color, (agent.x - size , agent.y - size, size*2, size*2))
            else:
                pygame.draw.circle(self.screen, color, (agent.x, agent.y), agent.health)

            if agent.is_attacking:
                self.attack_animations.append(AttackAnimation(agent.x, agent.y,agent.attack_radius))
                agent.is_attacking=False

        # Отображаем и обновляем атакующие анимации
       for animation in self.attack_animations:
            attack_ring_radius = (animation.frame / animation.animation_time) * animation.max_radius
            pygame.draw.circle(self.screen, (255, 0, 0), (animation.x, animation.y), int(attack_ring_radius), 2)
            animation.update()
            if animation.is_finished():
                self.attack_animations.remove(animation)

       pygame.event.pump()
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
                if distance < min_distance and distance <= 5**2:  # Предположим, радиус атаки = радиусу видимости
                    min_distance = distance
                    nearest_food = food
            return nearest_food

    def _compute_reward(self, agent):

        # Награда или штраф на основе действий и состояния агента
        reward = 0

        # Расстояние до ближайшего ресурса
        def distance_to_nearest_resource(agent):
            if not self.resources:  # Если нет ресурсов
                return float('inf')
            distances = [np.linalg.norm(np.array([agent.x, agent.y]) - np.array([resource.x, resource.y])) for resource in
                         self.resources]
            return min(distances)

        previous_distance = agent.previous_food_distance

        # После действия агента обновите его состояние и найдите новое расстояние
        # (Вызовите эту функцию после действия агента в вашем главном цикле)
        new_distance = distance_to_nearest_resource(agent)


        agent.previous_food_distance=new_distance
        if agent.prev_resources < agent.resources:
            reward += 5*(agent.resources- agent.prev_resources)
        elif agent.prev_resources > agent.resources:
            reward -= (agent.prev_resources - agent.resources) * 2
            if new_distance < previous_distance:
                reward += 0.2  # Награда за приближение к ресурсу
            elif  agent.is_eating<=0:
                reward-=0.3

        agent.prev_resources = agent.resources

        if agent.health < agent.prev_health:
            reward -= 2

        agent.prev_health = agent.prev_health

        if agent.health <= 0:
            reward -= 100
        if agent.resources <= 0:
            reward -= 100

        return reward

    def _update_world(self):
        # Обновляем ресурсы и другие аспекты мира
        # Пример: спавн новых ресурсов
        if random.random() < self.spawn_rate:  # Предположим, у нас есть некий коэффициент спавна ресурсов
            self.resources.append(Resource(random.randint(0, self.world_size), random.randint(0, self.world_size),random.randint(1,3)))


gymnasium.register('GoldenRuleEnv','goldenruleenv:GoldenRuleEnv')
