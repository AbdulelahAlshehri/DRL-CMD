# Imports
from __future__ import absolute_import, division, print_function

import os
import numpy as np
from stable_baselines3 import DDPG, DQN, A2C
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import MaskablePPO

from config.run_settings import RunSettings, suppress_warnings
from util.parse import ParseData
from structs.case import CaseSuite
from structs.dataset import DataSet
from env.environment import MolecularSearchEnv

total_timesteps = 1000


def main():
    suppress_warnings()
    group_data = DataSet.instance()
    parse_data = ParseData()
    settings = RunSettings(parse_data)
    cs = CaseSuite(parse_data, group_data)
    run(cs, settings)


def run(cs, settings):
    for c in cs.cases:
        run_case(c, settings)


def run_case(case, settings):

    model_list = create_agents(settings.algorithms, case)
    for model in model_list:
        model.learn(total_timesteps=total_timesteps)


def create_agents(agents, case):
    result = [create_agent(a, MolecularSearchEnv(case)) for a in agents]
    return result


def create_agent(agent, env):

    if agent == "rf":
        return NotImplementedError()

    # if agent == "trpo":
    #     return TRPO("MlpPolicy", env, verbose=1)

    elif agent == "ppo":
        return MaskablePPO("MlpPolicy", env, verbose=1)

    elif agent == 'dqn':
        return DQN("MlpPolicy", env, verbose=1)

    elif agent == 'ddpg':
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(
            n_actions), sigma=0.1 * np.ones(n_actions))
        return DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

    elif agent == 'a2c':
        return A2C("MlpPolicy", env, verbose=1)

    else:
        return Exception('not a valid algorithm to use')


if __name__ == '__main__':
    main()

    # if (level == 1):
    #     train_env = tf_py_environment.TFPyEnvironment(MolecularSearchEnv(
    #         constraints, util.RewardFunction.L1_EQUAL_WEIGHT, building_blocks, molecule_set, hist, p))
    #     eval_env = tf_py_environment.TFPyEnvironment(MolecularSearchEnv(
    #         constraints, util.RewardFunction.L1_EQUAL_WEIGHT, building_blocks, molecule_set, hist, p))
    #     actor_net_inner = actor_distribution_network.ActorDistributionNetwork(
    #         train_env.observation_spec()['observation'],
    #         train_env.action_spec(),
    #         fc_layer_params=fc_layer_params)

    #     actor_net = mask_splitter_network.MaskSplitterNetwork(
    #         splitter_fn=observation_and_action_constraint_splitter,
    #         wrapped_network=actor_net_inner,
    #         passthrough_mask=True,
    #     )

    #     actor_net2 = actor_distribution_network.ActorDistributionNetwork(
    #         train_env.observation_spec()['observation'],
    #         train_env.action_spec(),
    #         fc_layer_params=fc_layer_params)

    #     value_net = value_network.ValueNetwork(
    #         train_env.observation_spec()['observation'],
    #         preprocessing_layers=None,
    #         preprocessing_combiner=None,
    #         conv_layer_params=None,
    #         fc_layer_params=(64, 64),
    #         dropout_layer_params=None,
    #     )
    # if (level == 2):
    #     train_env = tf_py_environment.TFPyEnvironment(MolecularSearchEnv2(
    #         constraints, util.RewardFunction.L1_EQUAL_WEIGHT, building_blocks, molecule_set, hist, p))
    #     eval_env = tf_py_environment.TFPyEnvironment(MolecularSearchEnv2(
    #         constraints, util.RewardFunction.L1_EQUAL_WEIGHT, building_blocks, molecule_set, hist, p))

    #     actor_net_inner = actor_distribution_network.ActorDistributionNetwork(
    #         train_env.observation_spec()['observation'],
    #         train_env.action_spec(),
    #         fc_layer_params=fc_layer_params)

    #     actor_net = mask_splitter_network.MaskSplitterNetwork(
    #         splitter_fn=observation_and_action_constraint_splitter,
    #         wrapped_network=actor_net_inner,
    #         passthrough_mask=True,
    #     )

    #     actor_net2 = actor_distribution_network.ActorDistributionNetwork(
    #         train_env.observation_spec()['observation'],
    #         train_env.action_spec(),
    #         fc_layer_params=fc_layer_params)

    #     value_net = value_network.ValueNetwork(
    #         train_env.observation_spec()['observation'],
    #         preprocessing_layers=None,
    #         preprocessing_combiner=None,
    #         conv_layer_params=None,
    #         fc_layer_params=(64, 64),
    #         dropout_layer_params=None,
    #     )

    # Define agent for each algorithm
