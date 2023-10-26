# Определение класса агента
from resources import Resource


class Agent:
    def __init__(self, x, y,env):
        self.x = x
        self.y = y
        self.resources = 10.0  # Начальное количество ресурсов (яблоко)
        self.max_health=3.0
        self.health = self.max_health  # Здоровье
        self.attack_radius=5
        self.max_resources=10
        ###cache
        from goldenruleenv import GoldenRuleEnv
        self.env:GoldenRuleEnv=env
        ###cache for states
        self.prev_health=self.health
        self.prev_resources=self.resources

    # Метод для движения агента
    def move(self, dx, dy):
        moved=False
        if self.resources<=0:
            return
        if self.x+dx<self.env.world_size and self.x+dx>0:
            self.x += dx
            moved=True
        if self.y+dy<self.env.world_size and self.y+dy>0:
            self.y += dy
            moved=True
        if moved:
            self.resources -= abs(dx) / 50  # Расход ресурсов за движение
            self.resources -= abs(dy) / 50

    # Метод для атаки другого агента
    def attack(self, other):
        if self.resources >= 10/50:
            other.health -= 1
            self.resources -= 10/50
            # Если здоровье другого агента становится <= 0, забираем его ресурсы
            if other.health <= 0:
                self.env.resources.append(Resource(other.x,other.y, other.resources * 0.5))
    def get_visible_state(agent):
        return [agent.x, agent.y, agent.health]

