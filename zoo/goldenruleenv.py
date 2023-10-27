import math


import numpy
import pygame as pygame
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete
import numpy as np
import random
from gymnasium.utils import EzPickle, seeding
from pettingzoo import ParallelEnv,AECEnv
from pettingzoo.utils import agent_selector, wrappers

from agent import Agent, AttackAnimation
from resources import Resource
import functools
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.conversions import parallel_wrapper_fn

def env(**kwargs):
    env = GoldenRuleEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


# Основной класс среды
class GoldenRuleEnv(AECEnv,EzPickle):
    metadata={"render_modes": ["human",'rgb_array'], "name": "GoldenRuleEnv", "is_parallelizable": True}
    def __init__(self, world_size=500, vision_radius=200,render_mode='rgb_array'):
        super(GoldenRuleEnv, self).__init__()
        EzPickle.__init__(
            self,
            world_size=world_size,
            vision_radius=vision_radius,
            render_mode=render_mode,
        )
        self.vector_state = False
        self.starvation = 1/500/2
        self.world_size = world_size  # Размер мира
        self.vision_radius = vision_radius  # Радиус видимости агента
        self.agents_list = []  # Список агентов
        self.resources = []  # Список ресурсов
        self.spawn_rate=0.1
        self.init_agents=10
        self.init_resourses=20
        self.render_mode=render_mode


        self.agents=[f'player_{i}' for i in range(self.init_agents)]
        #self.possible_agents = [Agent(random.randint(0, self.world_size), random.randint(0, self.world_size), env=self) for _ in range(self.init_agents)]  # Примерное количество агентов
        self.possible_agents = self.agents
        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.possible_agents))))
        )
        self.agent_selection = self.possible_agents[0]  # Текущий агент

        # Пространство действий: dx, dy (-1, 0, 1), атака/не атаковать (0, 1), сеьсть яблоко/не есть(0, 1)
        self._action_space = MultiDiscrete([3, 3, 2, 2])
        self.action_spaces = dict(
            zip(self.agents, [self._action_space for _ in enumerate(self.agents)])
        )

        # Пространство наблюдений: пока что просто координаты всех агентов и ресурсов в радиусе видимости
        self._observation_space = spaces.Box(low=-self.world_size, high=self.world_size, shape=(32, ), dtype=np.float64)
        self.observation_spaces = dict(
            zip(
                self.agents,
                [self._observation_space for _ in enumerate(self.agents)],
            )
        )
        self.state_space = spaces.Box(
            low=-self.world_size,
            high=self.world_size,
            shape=[self.init_agents,self._observation_space.shape[0]],
            dtype=np.float64,
        )
        self._agent_selector = agent_selector(self.possible_agents)
        self.reset()

        if self.render_mode=='human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.world_size, self.world_size))
            pygame.display.set_caption("Golden Rule Environment")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Ваше определение пространства наблюдений
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Ваше определение пространства действий
        return self.action_spaces[agent]

    def reset(self,seed=None,options=None):
        # Сброс среды
        self.attack_animations = []
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards =  {a: 0 for a in self.agents}
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        #reinit
        self.agents_list = [Agent(random.randint(0, self.world_size), random.randint(0, self.world_size), env=self) for _ in range(self.init_agents)]  # Примерное количество агентов
        self.resources = [Resource(random.randint(0, self.world_size), random.randint(0, self.world_size), random.randint(1, 5)) for _ in range(self.init_resourses)]  # Примерное количество ресурсов
        for agent in self.agents_list:
            agent.reset()

    def _get_observation(self,agent):
        max_nearby_agents = 5

        max_nearby_resources = 5
        agent_data_length = 2
        nearby_agent_data_length = 3
        nearby_resource_data_length = 3

        total_observation_length = agent_data_length + \
                                   max_nearby_agents * nearby_agent_data_length + \
                                   max_nearby_resources * nearby_resource_data_length

        observations = []


        nearby_agents = sorted(
            [(other.x - agent.x, other.y - agent.y, other.health) for other in self.agents_list if
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

        return observation


    def state(self):
        state=[self._get_observation(agent) for agent in self.agents_list]
        return state

    def step(self, action):

        rewards = []
        self._cumulative_rewards[self.agent_selection] = 0
        dones =False
        truncate = False
        agent = self.agents_list[self.agent_name_mapping[self.agent_selection]]
        self._cumulative_rewards[self.agent_selection] = 0

        if  action is None or agent.dead:

            action=[1,1,0,0]
            self.rewards[agent_selector]=0

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

        if agent.health<=0 or agent.resources <= 0:
            self.terminations[self.agent_selection]=True
            agent.dead=True
        if self._agent_selector.is_last():
            # Обновляем ресурсы и другие аспекты мира
            self._update_world()
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()
            print(self.agent_selection,self._agent_selector)


        self._clear_rewards()
        next_agent = self.agents_list[self.agent_name_mapping[self.agent_selection]]
        self.rewards[self.agent_selection] = self._compute_reward(next_agent)

        self._accumulate_rewards()
        if self.render_mode=='human':
            self.render()



    def _advance_agent_selection(self):
        cur_idx = self.agents_list.index(self.agent_selection)
        if cur_idx == len(self.agents_list) - 1:
            self.agent_selection = self.agents_list[0]
        else:
            self.agent_selection = self.agents_list[cur_idx + 1]

    def observe(self, agent):
        # Вернуть наблюдение для данного агента
        return self._get_observation(self.agents_list[self.agent_name_mapping[agent]])



    def reward(self):
        agent_id = self.agent_selection
        agent = self.agents_list[agent_id]
        return self._compute_reward(agent)


    def render(self, mode='human'):
       self.screen.fill((255, 255, 255))  # Очистка экрана (белый цвет)

       # Отображение ресурсов (яблок) зелеными кругами
       for res in self.resources:
           pygame.draw.circle(self.screen, (0, 255, 0), (res.x, res.y), 2*res.amount)

       # Отображение агентов красными кругами
       for i, agent in enumerate(self.agents_list):
            if agent.dead:
               continue
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
        for other in self.agents_list:
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


