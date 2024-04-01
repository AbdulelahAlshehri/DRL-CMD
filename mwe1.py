import os
import warnings

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO, TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import (load_results,
                                                      plot_results, ts2xy)

from config.run_settings import RunSettings
from env.environment import MolecularSearchEnv, mask_fn
from structs.case import Case, CaseInstance, CaseSuite
from structs.dataset import DataSet
from util.parse import ParseData


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record(
            'reward', self.training_env.get_attr('total_reward')[0])
        return True


warnings.filterwarnings('ignore')

log_dir = "mwe_DCM/"

parse = ParseData('-c DCM -r 2 -sp'.split())
cs = CaseSuite(parse, DataSet.instance())
rs = RunSettings()
case = CaseInstance(Case(cs.load_case_data()['DCM'], DataSet.instance()), rs)
env = Monitor(ActionMasker(MolecularSearchEnv(case), mask_fn), log_dir)


model = MaskablePPO('MultiInputPolicy', env, verbose=1,
                    tensorboard_log=log_dir, n_steps=10).learn(10000, tb_log_name="mbt_run")
# env.state.show()
