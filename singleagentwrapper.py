import gymnasium as gym
import numpy as np

smart=True
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env, radius=50,agent_idx=0):
        super(SingleAgentWrapper, self).__init__(env)
        self.observation_space = env.observation_space
        self.radius = radius
        self.agent_idx=agent_idx

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs[self.agent_idx], _

    def step(self, action):
        boids_actions = [self.boids_algorithm(agent) for agent in self.env.agents[1:]] if smart else [[1,1,0,0] for agent in self.env.agents[1:]]
        actions = [action] + boids_actions
        obs, reward, done, truncated, info = self.env.step(actions)
        return obs[self.agent_idx], reward[self.agent_idx], done[self.agent_idx], truncated[self.agent_idx], info

    def boids_algorithm(self, agent):

        # 1. Находить соседей
        neighbors = [other for other in self.env.agents if other != agent and np.linalg.norm(
            np.array([agent.x, agent.y]) - np.array([other.x, other.y])) <= self.radius]

        eat = 0

        if agent.health < agent.max_health // 2:
            eat = 1

        attack = 0
        for neighbor in neighbors:
            if np.linalg.norm(np.array([agent.x, agent.y]) - np.array(
                    [neighbor.x, neighbor.y])) <= agent.attack_radius and agent.health >= 2:
                attack = 1
                break
        # Находить ближайший ресурс
        nearest_resource = None
        min_distance_to_resource =30
        for resource in self.env.resources:
            distance = np.linalg.norm(np.array([agent.x, agent.y]) - np.array([resource.x, resource.y]))
            if distance < min_distance_to_resource:
                min_distance_to_resource = distance
                nearest_resource = resource

        if agent.resources < agent.max_resources / 2 and nearest_resource:
            # Если у агента меньше половины ресурсов, двигаться к ближайшему ресурсу
            target_direction = np.array([nearest_resource.x, nearest_resource.y]) - np.array([agent.x, agent.y])
        elif neighbors:
            # Если у агента более половины ресурсов, двигаться к ближайшему агенту
            avg_x, avg_y = np.mean([[neighbor.x, neighbor.y] for neighbor in neighbors], axis=0)
            target_direction = np.array([avg_x, avg_y]) - np.array([agent.x, agent.y])
        elif nearest_resource:
            target_direction = np.array([nearest_resource.x, nearest_resource.y]) - np.array([agent.x, agent.y])
        else:
            dx, dy = np.random.randint(0, 3, 2)  # Если нет соседей, действовать случайно
            return [dx, dy, attack, eat]

        # Преобразовать целевое направление в действие
        if target_direction[0] > 0:
            dx = 2
        elif target_direction[0] < 0:
            dx = 0
        else:
            dx = 1

        if target_direction[1] > 0:
            dy = 2
        elif target_direction[1] < 0:
            dy = 0
        else:
            dy =1



        return [dx, dy, attack, eat]

