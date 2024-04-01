from structs.case import CaseSuite, Case
from structs.dataset import DataSet
from structs.state import RLState
from config.run_settings import RunSettings
from util.parse import ParseData
import pytest
import numpy as np
import pprint
import networkx as nx
from rdkit import Chem


def test_CaseSuite_load(CaseSuite_obj, dataset_obj):
    yml_data = CaseSuite_obj.load_case_data()
    assert yml_data['surfactant']['constraints']['NUM_GROUPS']['max'] == 30
    assert yml_data['refrigerant']['objective'] == [
        "max", "ENTHALPY_VAPORIZATION"]


def test_Case_nonaramatics(surfactant_Case_obj):
    assert np.array_equal(
        surfactant_Case_obj.nonaromatics, np.arange(1, 11))


def test_Case_aromatics(surfactant_Case_obj):
    assert np.array_equal(surfactant_Case_obj.aromatics,
                          np.array([15, 20, 21, 22]))


# def test_CaseSuite_generate_case(CaseSuite_obj):
#     pass


def test_Case_objective(absorbent_Case_obj):
    assert absorbent_Case_obj.objective == ["min", "LC_50"]


# def test_Case_suite_generate_cases(CaseSuite_obj):
#     assert CaseSuite_obj.cases == ["hello"]


def test_Case_convert_constraint(surfactant_Case_obj):
    constr = {'min': 3, 'max': 5}
    assert np.array_equal(surfactant_Case_obj.convert_constraint(
        'NUM_GROUPS', constr), np.array([1, 3, 5]))


def test_Case_convert_constraints(surfactant_Case_obj):
    arr = surfactant_Case_obj.constraints
    # print(arr)
    assert np.array_equal(arr,
                          np.array([[1, 20, 30],
                                    [2, 0, 15],
                                    [3, 6, 10],
                                    [4, 300, 999],
                                    [6, 300, 999],
                                    [7, 400, 999],
                                    [19, 0, 4]]))
