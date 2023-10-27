import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo.policies import MlpPolicy

import stable_baseline_ppo_basic
from goldenruleenv import GoldenRuleEnv


def transfer_weights(source_model, target_model):
    """
    Transfer weights from source_model to target_model.
    Assumes that target_model has at least as many layers/parameters as source_model.
    """
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()

    # Transfer weights from source to target where possible
    for name, param in source_state.items():
        if name in target_state and param.size() == target_state[name].size():
            target_state[name].copy_(param)

    # Update the target model state
    target_model.policy.load_state_dict(target_state)
old_model = PPO.load('basic_ppo_[256, 256, 256].zip')
policy_kwargs = dict(
    net_arch=[32,32],

)
if __name__ == '__main__':

    env= SubprocVecEnv([stable_baseline_ppo_basic.make_env() for _ in range(stable_baseline_ppo_basic.num_cpu)])

    new_model = PPO(MlpPolicy, env, policy_kwargs=policy_kwargs)  # Create a new model with the updated observation space
    transfer_weights(old_model, new_model)
    stable_baseline_ppo_basic.main(policy_kwargs,model=new_model)
    # Transfer weights from old_model to new_model
    #