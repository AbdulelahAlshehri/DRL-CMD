
import numpy as np
# from tqdm import tqdm
from dataclasses import dataclass, field, InitVar
from structs.state import RLState
from structs.case import Case
from typing import Union, Callable, Tuple, List

from util.dataload import load_prop_models


@dataclass
class ValidityChecker:
    """Entry point for all validity checking internals.

    Attributes:
        self.frag_valence_valid
        self.full_valence_valid
        self.prop_valid_vec
        self.prop_vals
        self.group_valid

    Methods:
        is_frag_valid
        is_mol_valid
        is_prop_valid
        is_group_valid

    """
    mol: RLState
    case: Case
    level: int

    def __post_init__(self):
        self.check()

    def check(self):
        pls = load_prop_models(self.case, self.level)
        self.frag_valence_valid, self.full_valence_valid = ValenceCxChecker(
            self.mol, self.case).check()
        self.prop_valid_vec, self.prop_vals = PropertyCxChecker(
            self.mol, self.case, pls).check()
        self.group_valid = GroupCxChecker(self.mol, self.case).check()

        return self.prop_valid_vec, self.prop_vals, self.frag_valence_valid, \
            self.full_valence_valid

    def is_frag_valid(self):
        print('Fragment valid: ', self.frag_valence_valid[0], '\nProperties valid: ', self.prop_valid_vec)
        return self.prop_valid_vec and self.frag_valence_valid

    def is_mol_valid(self):
        return self.prop_valid_vec and self.full_valence_valid and self.is_prop_valid()

    def is_prop_valid(self):
        return self.prop_valid_vec

    def is_group_valid(self):
        if np.sum(self.mol.group_count) == 0:
            return True
        return self.group_valid


@dataclass
class Params:
    mol: RLState
    case: InitVar[Case]
    state: np.ndarray = field(init=False)

    def __post_init__(self, case: Case) -> None:
        self.state = self.mol.to_full_vec()
        self.p = case.p
        self.c = case.c

    def values(self):
        pass


@dataclass
class ValenceParams(Params):
    """Internal class for the parameters to the valence validity checker.

    Attributes:
        mol: RLState representation of molecular fragment.
        case: representation of the corresponding case
        state: vector of group counts
        valency: 220-long vector of group valencies
        p: as specified in OptCAMD paper
        c: number of cycles

    """
    valency: np.ndarray = field(init=False)

    def __post_init__(self, case: Case) -> None:
        super().__post_init__(case)
        self.valency = case.group_valencies

    def values(self):
        """Returns (state, constraints, p, c)."""
        return self.state, self.valency, self.p, self.c


@dataclass
class GroupParams(Params):
    """Internal class for the parameters to the group constraints checker.

    Attributes:
        mol: RLState representation of molecular fragment.
        case: Representation of the corresponding case.
        state: Vector of group counts.
        constraints: All case constraints in a matrix.
        p: As specified in OptCAMD paper.
        c: Number of cycles.
    """
    constraints: np.ndarray = None

    def __post_init__(self,  case: Case) -> None:
        super().__post_init__(case)
        self.constraints = case.constraints[:3, :]

    def values(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Returns (state, constraints, p, c)."""
        return self.state, self.constraints, self.p, self.c


@dataclass
class PropertyParams(Params):
    """Internal class for the parameters to the property constraints checker.

    Attributes:
        mol: RLState representation of molecular fragment.
        case: Representation of the corresponding case.
        state: Vector of group counts.
        constraints: All property constraints in a matrix.
        p: As specified in OptCAMD paper.
        c: Number of cycles.
        group_weights: Vector of the molecular weights of the building block
            groups.
    """
    constraints: np.ndarray = None
    group_weights: np.ndarray = None

    def __post_init__(self, case: Case) -> None:
        self.constraints = case.constraints[3:, :]
        self.group_weights = self.mol.group_data.weights

    def values(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Returns (state, constraints, p, c)."""
        return self.state, self.constraints, self.p, self.c, self.group_weights


@dataclass
class Checker:
    mol: RLState
    case: InitVar[Case]
    params: Params

    def check(self):
        pass


@dataclass
class PropertyCxChecker(Checker):
    params: PropertyParams = field(init=False)
    prop_models: List
    incl_std_dev: bool = True

    def __post_init__(self, case: Case) -> None:
        self.params = PropertyParams(self.mol, case)

    def check(self) -> Tuple[bool, bool]:
        """Checks if all user-defined property constraints are satisfied.

        Args:
            state: the state vector from the RL exploration.
            pls: An n-dimensional object list containing all the relevant 
                property models.
            property_constraints: an nx2 numpy matrix where the first row 
                corresponds to the molecular weight constraint.           
            incl_std_dev: boolean indicating whether to consider the 
                   uncertainty in the constraint satisfaction.

        Returns: a tuple containing a boolean indicating whether all property 
            constraints are satisfied, and an nx2-dimensional vector of 
            the calculated properties from the property model and 
            their uncertainties.
        """
        property_constraints = self.params.constraints
        n = np.shape(property_constraints)[0]
        property_vals = np.zeros((n, 2))
        mw = None
        state = self.mol.to_full_vec()
        prop_values = np.empty((0, 2))
        for i, cnst in enumerate(property_constraints[:, 0]):
            # print(property_constraints[0])
            # For molecular weight, calculate it directly without a property model.
            if int(cnst) == 4:
                mw = self.poll_mw().reshape(1, -1)
                # print(prop_values)
                # print(prop_values.shape)
                # print(mw.shape)
                prop_values = np.vstack(
                    (prop_values, mw))
            else:
                prop = self.poll_prop(int(cnst)).flatten()
                # print(property_constraints)
                prop_values = np.vstack((prop_values, prop))

        # first check if molecular weight is less than max, then check if the rest of the property constraints
        # are in their  respective ranges.

        augmented_constraints = np.copy(property_constraints[:, 1:])
        augmented_constraints[1:, 0] -= prop_values[1:, 1]
        augmented_constraints[1:, 1] += prop_values[1:, 1]
        # print("Minimums: " + str(augmented_constraints[1:, 0]))
        # print("Maximums: " + str(augmented_constraints[1:, 1]))
        # print("Molecular weight check: " + str(prop_values[0][0]))
        # print(prop_values)
        property_check = check_bounds(
            prop_values[1:, 0], augmented_constraints[1:, :])
        # print(property_vals[0])
        result = ((np.less_equal(prop_values[0][0], property_constraints[0, 2]) and
                   np.all(property_check)),
                  prop_values)
        # print(augmented_constraints)
        # print(property_check)
        return result

    def poll_mw(self) -> np.ndarray:
        state = self.mol.to_full_vec()
        poll_vec = np.zeros((424))
        # poll_vec[:220] = state
        return np.array([np.dot(self.params.group_weights, state[:220]), 0])

    def poll_prop(self, i) -> np.ndarray:
        # print(self.pls[i].X_train_.shape)
        poll_vec = self.mol.to_full_vec()
        # poll_vec[:220] = self.mol.to_full_vec()

        y_pred, std = self.prop_models[i].predict(
            poll_vec.reshape(1, -1), return_std=True)
        property_val = np.array(
            [y_pred.flatten().item(), std.flatten().item()])
        # print(property_val)
        if not self.incl_std_dev:
            property_val[1] = 0
        return property_val


@dataclass
class ValenceCxChecker(Checker):
    params: ValenceParams = None

    def __post_init__(self, case: Case) -> None:
        self.params = ValenceParams(self.mol, case)

    def check(self) -> Tuple[float, bool]:
        """
        Selected group combination must be able to connect into an 
        integral zero-valency structure. When considering fragments, the result
        of (2-vj)*nj must be <= 2p, as this indicates there are available free 
        attachments.

        Returns: tuple (fragment, complete) indicating if valency check passes
            for the fragment, and if the molecule connects into a zero 
            integral valency structure (indicating the molecule
            is complete)
        """

        fragment_check = self.sufficient_valence_check()
        complete_check = self.total_valence_check()

        return fragment_check, complete_check

    def is_complete_molecule(self) -> bool:
        return self.total_valence_check() \
            and self.cyclic_check() \
            and self.aromatic_check()

    def sufficient_valence_check(self) -> Tuple[bool, bool]:
        dotprod, c = self._dotprod_valency(), self.params.c
        return (dotprod >= 2.0 * c, (dotprod == -2) | (dotprod == 0) | (dotprod == 2))

    def total_valence_check(self) -> bool:
        dotprod, c = self._dotprod_valency(), self.params.c
        # print(dotprod, 2.0*c)
        return dotprod == 2.0 * c

    def _dotprod_valency(self) -> float:
        state, valency, _, _ = self.params.values()
        return np.dot((valency - 2), state[:220]) + 2

    def valid_v_score(res) -> bool:
        return ((res == -2) | (res == 0) | (res == 2))

    def cyclic_check(self) -> bool:
        # TODO
        """Checks if the molecule in question is cyclic.
        Two cases:
            - if molecule is acyclic, perform check on number of attachments only
            - if molecule is mono/bicyclic, check if there exists >= min 
              required groups with valency >= 2 as well as attachment check

        Args:
            state: a vector of size m
            p: None or Int in range [-1,1]

        Returns: True if p != 1 and molecule is cyclic, False otherwise"""
        if self.params.p is None:
            return True

        if self.setting_mono():
            return self.is_valid_monocyclic()
        if self.setting_bi():
            return self.is_valid_bicyclic()
        if self.setting_acyc():
            return self.is_valid_acyclic()
        else:
            return self.is_valid_polycyclic()

    def setting_mono(self) -> bool:
        return self.params.p == 0

    def setting_bi(self) -> bool:
        return self.params.p == -1

    def setting_acyc(self) -> bool:
        return self.params.p == 1

    def is_valid_acyclic(self) -> bool:
        return self.has_valid_free_attachments()

    def is_valid_monocyclic(self) -> bool:
        return self.is_valid_ncyclic(num=3)

    def is_valid_bicyclic(self) -> bool:
        return self.is_valid_ncyclic(num=4)

    def is_valid_polycyclic(self) -> bool:
        return self.is_valid_ncyclic(num=(1 + self.params.p))

    def is_valid_ncyclic(self, num: int) -> bool:
        state, valency, _, _ = self.params
        valid_ncyclic = np.all(
            np.sum(state[np.where(valency >= 2)]) >= num)
        free_attachment_check = self.has_valid_free_attachments()

        return valid_ncyclic and free_attachment_check

    def has_valid_free_attachments(self) -> bool:
        state, valency, _, _ = self.params
        group_sum = np.sum(state)

        return np.all(group_sum >= valency[state.astype(int)] + 1)

    def aromatic_check(self) -> bool:
        # TODO
        """Checks the aromatic constraints.

        Args:
            state: vector of size m
            p: None or -1, 0, indication of monocyclic (0) or bicyclic (-1)

        Returns: boolean indicating constraints satisfied """
        return self.aro_comp_check() and \
            self.aro_existence_check() and \
            self.aro_ring_check()

    def aro_comp_check(self) -> bool:
        _, _, _, p = self.params
        mol = self.mol
        aro_comp_check = (mol.n_G_a(3) >=
                          (1.01 * (p - 1) + 0.0001 * mol.n_G_na)).all()
        aro_comp_check2 = (3 * mol.n_G_a(3) - 1) + mol.n_G_a(4) + \
            mol.n_G_na(2) >= (3 - 100 * (p + 1))

        return aro_comp_check and aro_comp_check2

    def aro_existence_check(self) -> bool:
        mol = self.mol
        existence_check = np.all(mol.rings >= (mol.aromatics > 0))
        existence_check2 = mol.rings <= np.count_nonzero(mol.aromatics)

        return existence_check and existence_check2

    def aro_ring_check(self) -> bool:
        mol = self.mol
        return mol.rings <= self.params.c


@dataclass
class GroupCxChecker(Checker):
    params: PropertyParams = field(init=False)

    def __post_init__(self, case) -> None:
        self.params = GroupParams(self.mol, case).values()

    def check(self):
        """Checks whether group constraints are satisfied.
        "Functional" groups are defined as all groups except CH3, CH2, CH, and C(no. 0-3)

        Args:
            state: the state vector from the RL exploration.
            constraints: an nx2 numpy matrix where the first row corresponds 
                to the total group number constraint,
                and the second corresponds to the total functional 
                group number constraint.

        Returns: boolean indicating constraints satisfied"""
        state, constraints, _, _ = self.params
        vals = self.get_sums(state)
        user_defined_group_constraints = check_bounds(
            vals, constraints[:3, 1:])
        return np.all(user_defined_group_constraints)

    def group_constraints_check_initial(self, state, constraints):
        """Checks whether the max group constraints
        (NUM_GROUPS, NUM_FUNC_GROUPS, NUM_REPEAT_GROUPS) are satisfied.
        functional groups are all groups except CH3, CH2, CH, and C(no. 0-3)

        Args:
            state: the state vector from the RL exploration.
            constraints: an nx2 numpy matrix where the first row corresponds to 
                the total group number constraint, and the second corresponds to the 
                total functional group number constraint.

        Returns: boolean indicating max constraints satisfied"""
        vals = self.get_sums(state)
        user_defined_group_constraints = check_max(vals, constraints[:3, 1:])
        return np.all(user_defined_group_constraints)

    def get_sums(self, state):
        group_sum = np.sum(state)
        func_group_sum = np.sum(state[4:])
        repeat_group_sum = np.sum(state[np.where(state > 1)] - 1)
        return np.array([group_sum, repeat_group_sum, func_group_sum])


# General helper functions


def check_bounds(n: int, bound_arr: np.ndarray) -> bool:
    return np.less_equal(n, bound_arr[:, 1]) & \
        np.greater_equal(n, bound_arr[:, 0])


def check_max(n: int, bound_arr: np.ndarray) -> bool:
    return np.less_equal(n, bound_arr[:, 1])
