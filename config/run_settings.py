from util.parse import create_parser
from util.dataload import open_config
import warnings
import os


class RunSettings():
    """Represents the settings for a given run of the program.

    Attributes:
        hp: Dictionary of hyperparameters from the external config file.
        logdir: Directory to store logs.
        total_groups: Dict of cardinalities of each functional group level.
        algorithms: Dict of flags for the algorithms to use.
        rings: Number of rings desired for the generated molecules.
    """

    def __init__(self, testargs=None):
        init_default_config(self)
        init_user_options(self, testargs)


def suppress_warnings():
    # Mute warnings

    def warn(*args, **kwargs):
        pass

    warnings.warn = warn


def init_default_config(self):
    config = open_config()
    self.hp = config['hp']
    self.logdir = config['log']['root']
    self.total_groups = config['total_groups']
    self.algorithms = config['alg_defaults']
    self.reward = config['scoring']['reward']
    self.rings = None


def init_user_options(self, testargs):

    parser = create_parser()
    if testargs is not None:
        args = parser.parse_known_args(testargs)[0]
        algorithms = args.algorithms[0]
        if 'rf' not in algorithms:
            self.algorithms['enable_rf'] = False
        if 'trpo' in algorithms:
            self.algorithms['enable_trpo'] = True
        if 'ppo' in algorithms:
            self.algorithms['enable_ppo'] = True
        if 'dqn' in algorithms:
            self.algorithms['enable_dqn'] = True
        if 'ddpg' in algorithms:
            self.algorithms['enable_ddpg'] = True
        if 'a2c' in algorithms:
            self.algorithms['enable_a2c'] = True
        if 'soft' in algorithms:
            self.algorithms['enable_soft'] = True
        if args.show_prop:
            self.show_prop = args.show_prop
    else:
        args = parser.parse_known_args()[0]
    if args.rings is not None:
        self.rings = int(args.rings)
