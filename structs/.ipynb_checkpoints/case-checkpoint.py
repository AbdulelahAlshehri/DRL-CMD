from dataclasses import InitVar, dataclass, field
from typing import List

import numpy as np
import pandas as pd
from util.dataload import open_cases, open_dataset

from structs.dataset import DataSet
from structs.smarts import GroupSMARTS

constraint_idx_map = {
    # Mapping of property/constraint parameter in human readable format to index
    "NUM_GROUPS": 1,  # total number of groups
    "NUM_REPEAT_GROUPS": 2,  # total number of repeat groups
    "NUM_FUNC_GROUPS": 3,  # total number of functional groups
    "MOLECULAR_WEIGHT": 4,  # molecular weight
    "FLASH_POINT": 5,  # - flash point
    "MELTING_POINT":  6,  # normal melting point (Tm)
    "BOILING_POINT": 7,  # normal boiling point (Tb)
    "CRITICAL_TEMP":  8,  # critical temperature (Tc)
    "CRITICAL_PRESSURE":  9,  # critical pressure (Pc)
    "CRITICAL_VOLUME": 10,  # critical volume (Vc)
    "GIBBS":  11,  # standard Gibbs free energy, 298K
    "ENTHALPY_FORMATION": 12,  # standard Enthalpy of formation, 298K
    "ENTHALPY_VAPORIZATION": 13,  # enthalpy of vaporization, 298K
    "ENTHALPY_FUS": 14,  # enthalpy of fusion, 298K
    "ENTHALPY_VAP_BP": 15,  # enthalpy of vaporization at Tb
    "AUTOIG_TEMP": 16,  # autoignition temp
    "PKA": 17,  # pKa
    "ENTHALPY_SOL": 18,  # heat of solution at P, hsol
    "LC50": 19,  # LC50
    "LOGP": 20,  # logP
    "LOGWS": 21,  # logWs
    "LD50": 22,  # ld50
    "OSHA_TWA": 23,  # osha-twa
    "HSP": 24,  # hild solubility param
    "BCF": 25,
    "VAPOR_PRESSURE": 26,
    "VISCOSITY": 27
}


group_constraint_idxs = np.array([1, 2, 3])
prop_constraint_idxs = np.arange(4, 28, 1)


class CaseSuite:
    def __init__(self, parse_data, dataset):
        self.cases = []
        self.group_data = dataset
        self.case_data = self.load_case_data()
        self.case_names = parse_data.cases
        self.generate_cases(self.case_data)

    def load_case_data(self):
        return open_cases()

    def generate_cases(self, case_choices):
        for case in case_choices:
            self.generate_case(self.case_data[case])

    def generate_case(self, case_data):
        self.cases.append(Case(case_data, self.group_data))


@dataclass
class Case:
    raw_data: pd.DataFrame
    group_data: DataSet

    def __post_init__(self):
        self.convert_constraints()

    def convert_constraints(self):
        constraints_matrix = np.empty((0, 3))

        for k, v in self.raw_data['constraints'].items():
            constraints_matrix = np.vstack(
                ((constraints_matrix), self.convert_constraint(k, v)))
        self.constr = constraints_matrix

    def convert_constraint(self, k, v):
        idx = constraint_idx_map[k]
        min = 0 if 'min' not in v.keys() else v['min']
        max = v['max'] if 'max' in v.keys() else 999
        return np.array([idx, min, max])

    @property
    def constraints(self):
        return self.constr

    @property
    def bb(self):
        return np.array(self.raw_data['building_blocks'])

    @property
    def property_constraints(self):
        return self.constraints[3:, :]

    @property
    def aromatics(self):
        mask = np.isin(self.bb - 1, self.group_data.aromatics)
        return self.bb[mask]

    @property
    def nonaromatics(self):
        mask = np.isin(self.bb - 1, self.group_data.nonaromatics)
        return self.bb[mask]

    @property
    def cyclics(self):
        mask = np.isin(self.bb - 1, self.group_data.cyclics)
        return self.bb[mask]

    @property
    def noncyclics(self):
        mask = np.isin(self.bb-1, self.group_data.noncyclics)
        return self.bb[mask]

    @property
    def objective(self):
        return self.raw_data['objective']

    @property
    def group_constraints(self):
        return self.constraints[group_constraint_idxs]

    @property
    def reward_type(self):
        return self.raw_data['scoring']['reward_type']

    @property
    def name(self):
        return self.raw_data['name']

    @property
    def group_valencies(self):
        return self.group_data.valences[0]


class CaseInstance(Case):
    """Case object augmented with run settings data.

    """

    def __init__(self, case, run_settings):
        super().__init__(case.raw_data, case.group_data)
        # maybe add q here?
        self.c = run_settings.rings
        self.p = 1 - run_settings.rings
        self.reward = run_settings.reward

    @property
    def cyclicity_unspecified(self):
        return True if self.p == None else False

    @property
    def reward_type(self):
        return self.reward
