# from validity import valence_check, cyclic_check, aromatic_check
import pytest
import pandas as pd
import pathlib
import env.validity as validity
from unittest.mock import patch, MagicMock, PropertyMock
from structs.state import RLState
from structs.case import Case


@pytest.fixture
def mbt_Case_obj(CaseSuite_obj):
    yml_data = CaseSuite_obj.load_case_data()
    return Case(yml_data['mbt'])


valence_tests = []
df = pd.read_excel("test/csv/valence_validity_small.xlsx")

data = df.iloc[:, 6:226]
valences = pd.to_numeric(pd.read_excel(
    "data/valence.xlsx").iloc[:220, 8]).fillna(0).to_numpy()

for i in df.index:
    valence_tests.append(
        (pd.to_numeric(pd.Series(data.iloc[i])).to_numpy(), True))

monocyclics_tests = []

monocyclic_p_test_data = pd.read_csv("test/csv/monocyclic.csv")
df2 = monocyclic_p_test_data.iloc[:, 2:222]

for i in df2.index:
    monocyclics_tests.append(
        (pd.to_numeric(pd.Series(df2.iloc[i])).to_numpy(), True))


bicyclics_tests = []

bicyclic_p_test_data = pd.read_excel("test/csv/bicyclic.xlsx")
df3 = bicyclic_p_test_data.iloc[:, 2:222]

for i in df3.index:
    bicyclics_tests.append(
        (pd.to_numeric(pd.Series(df3.iloc[i])).to_numpy(), True))

tricyclics_tests = []

tricyclic_p_test_data = pd.read_excel("test/csv/tricyclics.csv")
df4 = tricyclic_p_test_data.iloc[:, 2:222]

for i in df4.index:
    tricyclics_tests.append(
        (pd.to_numeric(pd.Series(df4.iloc[i])).to_numpy(), True))


# aro_f_test_data = pd.read_excel("test/csv/aromatics_fails.csv")
# aro_p_test_data = pd.read_excel("5est/csv/aromatics_passes.csv")


# @patch('structs.repr.Case')
# @patch('validity.ValenceParams')
# @patch('structs.repr.RLState')
# @pytest.mark.parametrize("data, expected", valence_tests)
# def test_total_valence_check(x, y, case, data, expected):
#     y = validity.ValenceParams()
#     y.values.return_value = (data, data, 0, 0)
#     vx = validity.ValenceCxChecker(x, y, case)
#     # vx.state = data
#     assert vx.total_valence_check()

# ValenceChecker Tests

@patch('structs.case.Case')
@patch('env.validity.ValenceParams')
@patch('structs.state.RLState')
@pytest.mark.parametrize("data, expected", valence_tests)
def test_sufficient_valence_check(x, y, case, data, expected):
    y = validity.ValenceParams()
    y.values.return_value = (data, valences, 0, 0)
    y.c = 0
    vx = validity.ValenceCxChecker(x, y, case)
    # vx.state = data
    assert vx.sufficient_valence_check()[0]


# @patch('structs.repr.Case')
# @patch('validity.ValenceParams')
# @patch('structs.repr.RLState')
# @pytest.mark.parametrize("data, expected", valence_tests)
# def test_total_valence_check(x, y, case, data, expected):
#     y = validity.ValenceParams()
#     # print(valences)
#     y.values.return_value = (data, valences, 2, 0)
#     # y.valency = valences
#     y.c = 0
#     vx = validity.ValenceCxChecker(x, y, case)
#     assert vx.total_valence_check()
#     # vx.state = data


@patch('structs.case.Case')
@patch('env.validity.ValenceParams')
@patch('structs.state.RLState')
@pytest.mark.parametrize("data, expected", monocyclics_tests)
def test_monocyclics_passes(x, y, case, data, expected):
    y = validity.ValenceParams()
    # print(valences)
    y.values.return_value = (data, valences, 0, 0)
    # y.valency = valences
    y.c = 1
    vx = validity.ValenceCxChecker(x, y, case)
    assert vx.total_valence_check()


@patch('structs.case.Case')
@patch('env.validity.ValenceParams')
@patch('structs.state.RLState')
@pytest.mark.parametrize("data, expected", bicyclics_tests)
def test_bicyclics_passes(x, y, case, data, expected):
    y = validity.ValenceParams()
    # print(valences)
    y.values.return_value = (data, valences, 0, 0)
    # y.valency = valences
    y.c = 2
    vx = validity.ValenceCxChecker(x, y, case)
    assert vx.total_valence_check()


@patch('structs.case.Case')
@patch('env.validity.ValenceParams')
@patch('structs.state.RLState')
@pytest.mark.parametrize("data, expected", tricyclics_tests)
def test_tricyclics_passes(x, y, case, data, expected):
    y = validity.ValenceParams()
    # print(valences)
    y.values.return_value = (data, valences, 0, 0)
    # y.valency = valences
    y.c = 3
    vx = validity.ValenceCxChecker(x, y, case)
    assert vx.total_valence_check()

# assert c.total_valence_check() == expected

# def test_cyclic_check(smiles_list):
#     pass

# def test_aromatic_check(smiles_list):
#     pass

# def test_group_constraints_check(smiles_list):
#     pass

# def test_property_constraints_check(smiles_list):
#     pass
