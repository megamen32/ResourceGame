import pygame.time

import goldenruleenv

env = goldenruleenv.GoldenRuleEnv()

env.reset()
while True:
    # this is where you would insert your policy
    actions = {agent:env.action_space(agent).sample() if not agent.dead else None for agent in env.agents_list}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    pygame.time.wait(50)
    if all(terminations.values()) or all(truncations.values()):
        env.reset()
env.close()