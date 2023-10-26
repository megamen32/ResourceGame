import gymnasium
import pygame
import goldenruleenv

def handle_human_input(env, active_keys):
    # Обработка действий пользователя
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            return "QUIT", active_keys
        elif event.type == pygame.KEYDOWN:

            if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                active_keys.add(event.key)
            elif event.key == pygame.K_SPACE:
                return "ATTACK", active_keys
            elif event.key == pygame.K_e:
                return "EAT", active_keys
        elif event.type == pygame.KEYUP:
            if event.key in active_keys:
                active_keys.remove(event.key)



    if pygame.K_LEFT in active_keys:
        return "MOVE_LEFT", active_keys
    elif pygame.K_RIGHT in active_keys:
        return "MOVE_RIGHT", active_keys
    elif pygame.K_UP in active_keys:
        return "MOVE_UP", active_keys
    elif pygame.K_DOWN in active_keys:
        return "MOVE_DOWN", active_keys
    elif pygame.K_SPACE in active_keys:
        return "ATTACK", active_keys
    elif pygame.K_e in active_keys:
        return "EAT", active_keys

    return None, active_keys

def action_from_input(input_str):
    action_map = {
        "MOVE_LEFT": (-1, 0, 0, 0),
        "MOVE_RIGHT": (1, 0, 0, 0),
        "MOVE_UP": (0, -1, 0, 0),
        "MOVE_DOWN": (0, 1, 0, 0),
        "ATTACK": (0, 0, 1, 0),
        "EAT": (0, 0, 0, 1)
    }
    return list(action_map.get(input_str, (0, 0, 0, 0)))

env = gymnasium.make('GoldenRuleEnv')
observation,_ = env.reset()
human_input = None
active_keys = set()

while True:
    new_human_input, active_keys = handle_human_input(env, active_keys)
    if new_human_input == "QUIT":
        break


    action = action_from_input(new_human_input)
    nulls_actions = [[0,0,0,0] for _ in range(env.init_agents-1)]
    actions = [action, *nulls_actions]
    observation, reward, done, truncated, info = env.step(actions)
    env.render()

    if done:
        break
