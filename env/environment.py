from typing import List

import colorama
import gym
import numpy as np
from colorama import Back, Fore, Style
from gym import spaces

from config.run_settings import RunSettings
from env.reward import (L1ScaledRewardCalculator,
                        L1WithEqualWeightRewardCalculator,
                        L2ScaledRewardCalculator,
                        L2WithEqualWeightRewardCalculator)
from structs.case import Case
from structs.dataset import DataSet
from structs.state import RLState
from env.validity import ValidityChecker
from env.masking import mask, MaskParams
import pprint


colorama.init(autoreset=True)

MINIMUM_REWARD = -999


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


class MolecularSearchEnv(gym.Env):
    """The environment defining the molecular search space with level-2 estimation.

    The observation space consists of the counts of the functional groups,
        adjacency matrix and all node features. The state attribute holds
        a more informative RLState object which holds the NetworkX representation
        and many helper methods.

    To ensure a permutation invariant representation of each unique graph for the
    observation space, graphs are embeddeded using the normalized laplacian and
    stored in a cache.

    Furthermore, graphs with the same feature counts are stored
    in buckets, and a graph isomorphism test is performed if the former check
    fails to distinguish a unique graph.

    If the number of counts of each functional group is the same, the
    augmented adjacency matrix is the same, and the [?~?]

    Observation space: The observation space consists of an a) n x f matrix,
        where n is the number of nodes in the current graph and f is a feature
        vector corresponding to each node; and b) a graph embedding.
        Features:
        - total valency
        - v_a
        - v_b
        - v_c

    Action space: The action space is a Discrete Gym space, representing a discrete 5-dimensional box
        (new_group, new_group_bond_type, tail_group, tail_group_bond_type, tail_enumeration)

    Constraints in the observation space are enforced through action masking, and
    disallow transitions to states where valency checks or other GC heuristics
    are violated.
    """

    def __init__(self, case: Case, run_settings: RunSettings = None, verbose=True):
        super(MolecularSearchEnv, self).__init__()
        # set attributes
        self.data = DataSet.instance()
        self.case = case
        self.verbose = verbose
        self.num_bb = len(case.bb)
        self.p = 2
        self.action = None
        # self.run_settings = run_settings

        max_per_group = 10  # case.group_constraints  # max number of nodes fix this
        self.max_per_group = 10
        self.bond_types = int(3)

        self.n = np.prod(
            [self.num_bb,
             self.bond_types,
             self.num_bb,
             self.bond_types,
             max_per_group])

        self.action_dims = (self.num_bb,
                            self.bond_types,
                            self.num_bb,
                            self.bond_types,
                            max_per_group)
        # Action Space
        self.action_space = spaces.Discrete(self.n)

        # Observation Space
        self.observation_space = spaces.Dict({
            'group_counts': spaces.MultiDiscrete([max_per_group] * self.num_bb),
            'vtotal': spaces.MultiDiscrete([6] * max_per_group * self.num_bb),
            'va': spaces.MultiDiscrete([6] * max_per_group * self.num_bb),
            'vb': spaces.MultiDiscrete([6] * max_per_group * self.num_bb),
            'vc': spaces.MultiDiscrete([6] * max_per_group * self.num_bb)})

        
        # self.observation_space = spaces.Box(low = 0, high=5, shape=([
        #         self.num_bb, 5,4,4,4]), dtype=int)

        # Initialize environment attributes
        self.invalids = dict()
        self.histogram = dict()

        self._latest_reward = -999

        # The initial state is an empty molecule, i.e. a molecule with no functional groups.
        # Adjacency matrix is initialized to 0
        # valencies are all initialized to 0
        self._initial_state = RLState(case)
        self.state = self._initial_state

    def reward(self):
        """Returns the function used to calculate the reward.
        """
        reward_func = self.case.reward_type
        if reward_func == 'l1':
            return L1WithEqualWeightRewardCalculator
        if reward_func == 'l1scaled':
            return L1ScaledRewardCalculator
        if reward_func == 'l2':
            return L2WithEqualWeightRewardCalculator
        if reward_func == 'l2scaled':
            return L2ScaledRewardCalculator

    def step(self, action):
        done = False

        self.action = action

        params = np.unravel_index(
            int(action), self.action_dims)

        group, bond_type_h, tail, bond_type_t, enum = self.action_to_str(params)

        print("Action: ", int(action), " | ",
              "Group: ", group, " | ",
              "Bond type (head): ", bond_type_h, " | ",
              "Bond type (tail): ", bond_type_t, " | ",
              "Tail: ", tail, " | ",
              "Enumeration number: ", enum)
        print(self.state)
        print("Current nodes in graph (beginning): ", self.state.graph.nodes(data=True))
        print("Attempting to add group...")

        self._new_state = self.state.add_group(params)
        # print(self._new_state.graph.nodes(data=True))

        if self._new_state is None:
            print("Invalid")
            done = True
            self.constraint_setter(action)
            print(Fore.RED + "Terminated! Total episode reward: " +
                    str(self._latest_reward))
            done = True
        else:
            vc = ValidityChecker(self._new_state, self.case, level=2)

            if vc.is_frag_valid():
                self.state = self._new_state
                self.state.property_vals = vc.prop_vals
                reward_calc = self.reward()
                self._latest_reward = reward_calc(
                    self.state, self.case).compute_reward()
            else:
                print("Invalid")
                self.constraint_setter(action)
                done = True
                print(Fore.RED + "Terminated! Total episode reward: " +
                      str(self._latest_reward))
        # check termination

            if (not np.any(self.state.valency.total)) and vc.is_frag_valid():
                print(Fore.CYAN + "Found molecule! Total episode reward: " +
                      str(self._latest_reward))
                done = True
        print("Current nodes in graph (end): ", pprint.PrettyPrinter(indent=2).pformat(self.state.graph.nodes(data=True)))
        print("Valence left: ", int(np.sum(self.state.valency.total)))
        print(Fore.BLUE + "#############################")
        return self.state.obs, self._latest_reward, done, {}

    def reset(self):

        num_invalids = np.sum([np.count_nonzero(x==0) for x in self.invalids.values()])
        print(Fore.GREEN +
              " \n==========================NEW RUN==========================")
        print("Number of encounters of invalid states",
              num_invalids)
        self.state = self._initial_state
        self._latest_reward = MINIMUM_REWARD
        return self.state.obs

    def render(self, mode='human'):
        pass

    def constraint_setter(self, action, mask=None):
        key = self.state.key()
        if not self.invalids.get(key) is not None:
                #  or self.histogram.get(key) is not None)
                # and action.tobytes() in self.invalids[key]):
            self.invalids[key] = np.ones(self.n)

        self.invalids[key][action] = False

    def get_action_mask(self):
        params = MaskParams(self)
        result = mask(params)
        print('Number of masked actions: ', np.count_nonzero(result==0),"/",self.n)
        print(np.where(result))
        return result


    def action_to_str(self, params):
        group, bond_type_h, tail, bond_type_t, enum = params
        group_idx = self.state.bb[group]
        tail_idx = self.state.bb[tail]
        group= group_idx_to_name(group_idx)
        bond_type_h = get_v(bond_type_h)
        tail = group_idx_to_name(tail_idx)
        bond_type_t = get_v(bond_type_t)

        return group, bond_type_h, tail, bond_type_t, enum



def group_idx_to_name(num):
    return DataSet.instance().first_order_group_names[num - 1]

def get_v(type):
    if type == 0:
        return "a"
    if type == 1:
        return "b"
    if type == 2:
        return "c"

def init_vars(num_rings):
    hist = dict()
    p = 1 - num_rings
    # Initialize empty set for all found valid molecules
    molecule_set = set()

    return hist, p, molecule_set
