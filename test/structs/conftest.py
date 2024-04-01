import pytest
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
def absorbent_Case_obj(CaseSuite_obj):
    yml_data = CaseSuite_obj.load_case_data()
    return Case(yml_data['absorbent'], DataSet.instance())


@pytest.fixture
def RLState_obj(mbt_Case_obj):
    return RLState(mbt_Case_obj)


@pytest.fixture
def surfactant_Case_obj(CaseSuite_obj, dataset_obj):
    yml_data = CaseSuite_obj.load_case_data()
    return Case(yml_data['surfactant'], dataset_obj)
    return Case(yml_data['surfactant'], dataset_obj)
