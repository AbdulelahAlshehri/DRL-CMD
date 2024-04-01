from dataclasses import dataclass
# from rdkit.Chem import rdMolDescriptors
# from rdkit.Chem import DataStructs
from structs.state import RLState
from structs.case import Case
import numpy as np

MW_COST_MULTIPLIER = 2
PROPERTY_VIOLATION_RATIO = 0.75


class PropScoreParams:
    """Representation of property-related parameters required to compute the RL score.

    Attributes:
        min: lower bound of all property constraints.
        center: median of property constraint range.
        max: upper bound of all property constraints.
        vals: property scores.
        cx: full property constraints matrix.
    """

    def __init__(self, mol: RLState, case: Case):

        self.cx = case.property_constraints
        self.min = self.cx[:, 1]
        self.max = self.cx[:, 2]
        self.center = np.reshape(
            (np.add(self.min, self.max) / 2), (-1, 1))
        self.vals = np.reshape(mol.property_vals[:, 0], (-1, 1))
        self.mw_range = self.cx[0]
        self.mw_val = self.vals[0]
        # prop_max = np.reshape(prop_max, (-1, 1))
        self.max = self.max.reshape(-1, 1)


class GroupScoreParams:
    """Representation of property-related parameters required to compute the RL score.

    Attributes:
        min: lower bound of all group constraints.
        vals: property scores.
        cx: full group constraints matrix.
    """

    def __init__(self, case: Case):
        self.cx = case.group_constraints


class RewardCalculator:
    def __init__(self, mol: RLState, case: Case):
        self.mol = mol
        self.case = case

    def compute_reward(self) -> float:
        raise Exception("must be handled by subclass")

    # TODO
    def preprocess(self):
        self.p_params = PropScoreParams(self.mol, self.case)
        self.g_params = GroupScoreParams(self.case)

        return self.p_params, self.g_params

    def __repr__(self):
        msg = (
            f"{'Min:': <17}{self.p_params.min}\n"
            f"{'Max:': <17}{self.p_params.max.flatten()}\n"
            f"{'Center:': <17}{self.p_params.center.flatten()}\n"
            f"{'Polled values:': <17}{self.p_params.vals.flatten()}\n"
        )
        if hasattr(self, "p_vio"):
            msg += f"{'Prop. vio pen.': <17}{self.p_vio[0]}\n"
        if hasattr(self, 'm_vio'):
            msg += f"{'MW vio pen.': <17}{self.m_vio}\n"
        if hasattr(self, 'p_vio') and hasattr(self, 'm_vio'):
            msg += f"{'Total addl. pen.': <17}{self.m_vio + self.p_vio[0]}\n"

        msg += f"{'Reward:': <17}{self.reward}"
        return msg


class GaussianRewardCalculator(RewardCalculator):
    def __init__(self, mol: RLState, case: Case):
        super().__init__(mol, case)

    def compute_reward(self, mol: RLState, case: Case):
        sigma = 0.5
        mu = 0.5
        x = ...
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


class L1WithEqualWeightRewardCalculator(RewardCalculator):
    def __init__(self, mol: RLState, case: Case):
        super().__init__(mol, case)

    def compute_reward(self):
        p_params, g_params = self.preprocess()
        r_p = -np.linalg.norm((np.divide(p_params.vals - p_params.center,
                                         p_params.max - p_params.center)),  ord=1)
        r_g = 0
        self.reward = r_p + r_g

        # print(p_params.vals - p_params.center)
        # print(p_params.max - p_params.center)
        return r_p + r_g


class L1ScaledRewardCalculator(L1WithEqualWeightRewardCalculator):
    def __init__(self, mol: RLState, case: Case):
        super().__init__(mol, case)

    def compute_reward(self):
        p_params, g_params = self.preprocess()
        reward = super().compute_reward()
        r, p_vio, m_vio = scale(reward, p_params, g_params)
        self.p_vio = p_vio
        self.m_vio = m_vio
        self.reward = r

        return r


class L2WithEqualWeightRewardCalculator(RewardCalculator):
    def __init__(self, mol: RLState, case: Case):
        super().__init__(mol, case)

    def compute_reward(self):
        p_params, g_params = self.preprocess()
        r = - np.linalg.norm((np.divide(p_params.vals - p_params.center,
                                        p_params.max - p_params.center)))
        return r


class L2ScaledRewardCalculator(RewardCalculator):
    def __init__(self, mol: RLState, case: Case):
        super().__init__(mol, case)

    def compute_reward(self):
        reward = super().compute_reward(self.mol, self.case)
        return scale(reward)


class ZeroOneRewardCalculator(RewardCalculator):
    def __init__(mol: RLState, case: Case):
        super().__init__(mol, case)


def scale(reward, p_params, g_params):
    # multiply cost by factor `MW_COST_MULTIPLIER` before target MW constraint is satisfied, to encourage finding solutions
    # in the appropriate regime. This is so that the agent still gets a small reward when beginning
    # to construct the solution (preventing early termination) but is
    # more incentivized to find solutions in the correct MW range
    # allows priority violations

    m_vio = 0
    # Multiplicative cost
    if (p_params.mw_val < p_params.mw_range[0]):
        m_vio = reward*MW_COST_MULTIPLIER - reward
        reward *= MW_COST_MULTIPLIER

    # Additive cost
    if ((p_params.min > p_params.vals).any() or (p_params.max < p_params.vals).any()):
        min_constraint = np.reshape(p_params.min, (-1, 1))
        max_constraint = np.reshape(p_params.max, (-1, 1))
        i_violated = np.where(np.logical_or(min_constraint > p_params.vals,
                                            max_constraint < p_params.vals) == True)

        p_vio = np.maximum(
            (p_params.vals[i_violated] - max_constraint[i_violated]),
            (min_constraint[i_violated] - p_params.vals[i_violated]))

        p_vio = np.divide(
            p_vio, p_params.center[i_violated])

        reward -= np.sum(PROPERTY_VIOLATION_RATIO *
                         p_vio)

    return reward, -p_vio, m_vio
