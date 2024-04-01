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


def dataset_smarts_valid_tests(dataset):
    results = []
    for s in dataset.smarts:
        if s.smart_str[0][0] != '[placeholder]':
            results.append(s)
    return results


smarts_valid_tests = dataset_smarts_valid_tests(DataSet.instance())


def test_dataset_type_cardinalities(dataset_obj):
    assert len(dataset_obj.urea_amides[0]) == 17
    assert len(dataset_obj.urea_amide_subgroups[0]) == 10
    assert len(dataset_obj.aromatics[0]) == 50


def test_DataSet_smarts_exists(dataset_obj):
    assert dataset_obj.smarts != None


@pytest.mark.parametrize("smarts", smarts_valid_tests)
def test_DataSet_smarts_valid(smarts):
    for smart in smarts.smart_str:
        for s in smart:
            pprint.pprint(s)
            assert Chem.MolFromSmarts(s)


def test_DataSet_group_nums(dataset_obj):
    assert np.array_equal(dataset_obj.group_nums, np.arange(1, 425))
