# import model
from env.environment import MolecularSearchEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import pytest
from config.run_settings import RunSettings
from structs.case import CaseSuite, Case
from structs.dataset import DataSet
from structs.state import RLState
from util.parse import ParseData
import pytest
import numpy as np
import pprint
import networkx as nx
from rdkit import Chem


@pytest.fixture
def dataset_obj():
    return DataSet.instance()


@pytest.fixture
def parse_data_obj():
    return ParseData('-c mbt surfactant -r 2 -sp'.split())


@pytest.fixture
def CaseSuite_obj(parse_data_obj, dataset_obj):
    return CaseSuite(parse_data_obj, dataset_obj)


@pytest.fixture
def mbt_Case_obj(CaseSuite_obj):
    yml_data = CaseSuite_obj.load_case_data()
    return Case(yml_data['mbt'], DataSet.instance())


@pytest.fixture
def RLState_obj(mbt_Case_obj):
    return RLState(mbt_Case_obj)


def test_env_check(mbt_Case_obj):
    e = MolecularSearchEnv(mbt_Case_obj, RunSettings())
    check_env(e)
