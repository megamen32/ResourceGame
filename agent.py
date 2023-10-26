# Определение класса агента
import random

from resources import Resource


class Agent:
    def __init__(self, x, y,env):
        self.x = x
        self.y = y

        self.max_health=10.0
        self.health =1  # Здоровье
        self.attack_radius=10
        self.max_resources=10
        self.resources = self.max_resources  # Начальное количество ресурсов (яблоко)
        ###cache
        from goldenruleenv import GoldenRuleEnv
        self.env:GoldenRuleEnv=env
        self.is_attacking = False
        self.attack_idle=0
        self.time_between_attacks=20
        ###cache for states
        self.prev_health=self.health
        self.prev_resources=self.resources
        self.previous_food_distance= float('inf')
        self.normal_health=random.uniform(0.2,1)
        self.is_eating=0



        # Метод для движения агента
    def move(self, dx, dy):
        moved=False
        if self.resources<=0:
            return
        if self.x+dx<self.env.world_size and self.x+dx>0 and abs(dx)>0:
            self.x += dx
            moved=True
        if self.y+dy<self.env.world_size and self.y+dy>0 and abs(dy)>0:
            self.y += dy
            moved=True
        if moved:
            self.resources -= abs(dx) / 500  # Расход ресурсов за движение
            self.resources -= abs(dy) / 500
        else:
            self.resources -= self.env.starvation

    # Метод для атаки другого агента
    def attack(self, other):
        other.health -= 1

        # Если здоровье другого агента становится <= 0, забираем его ресурсы
        if other.health <= 0:
            self.env.resources.append(Resource(other.x,other.y, other.resources ))
    def get_visible_state(agent):
        return [agent.x, agent.y, agent.health]


class AttackAnimation:
    def __init__(self, x, y,max_radius=10):
        self.x = x
        self.y = y
        self.frame = 0
        self.max_radius=max_radius
        self.animation_time=10

    def update(self):
        self.frame += 1

    def is_finished(self):
        return self.frame >= self.animation_time

