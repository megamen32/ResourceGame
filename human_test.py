import gymnasium
import numpy as np
import pygame
import goldenruleenv
from singleagentwrapper import SingleAgentWrapper


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
        "MOVE_LEFT": (0, 1, 0, 0),  # -1 становится 0
        "MOVE_RIGHT": (2, 1, 0, 0),  # 1 становится 2
        "MOVE_UP": (1, 0, 0, 0),  # -1 становится 0
        "MOVE_DOWN": (1, 2, 0, 0),  # 1 становится 2
        "ATTACK": (1, 1, 1, 0),
        "EAT": (1, 1, 0, 1)
    }

    return list(action_map.get(input_str, (1, 1, 0, 0)))

env = SingleAgentWrapper(gymnasium.make('GoldenRuleEnv'))
observation,_ = env.reset()
human_input = None
active_keys = set()
cum_reward=0
step=0
while True:
    step+=1
    new_human_input, active_keys = handle_human_input(env, active_keys)
    if new_human_input == "QUIT":
        break


    action = action_from_input(new_human_input)


    observation, reward, done, truncated, info = env.step(action)
    cum_reward+=reward
    env.render()
    pygame.time.wait(50)
    if step%50==0:
        print('cummulitive_reward',cum_reward,'reward',reward,'state\n',env.agents[0].get_visible_state())
    if done or truncated:
        env.reset()
        cum_reward = 0

