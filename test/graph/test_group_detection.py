from graph import GraphDataLoader, GraphQueryDataLoader, QueryGraph, Fragmentation, find_matches, frag_to_nx_graph
from graph import find_fragmentation, get_rdmol
from graph import smarts
from rdkit.Chem import MolFromSmarts
from structs.dataset import DataSet
from structs.smarts import GroupSMARTS
import pandas as pd
import pytest
import pprint
# from conftest import _slim, _slim_reason
import numpy as np


@pytest.fixture
def dataset():
    return DataSet.instance()


@pytest.fixture
def fragmentation(dataset):
    mol = get_rdmol("Clc1ccccc1")
    return find_fragmentation(mol, dataset.smarts)


def test_fragmentation_repr(dataset):
    mol = get_rdmol("Clc1ccccc1")
    frag = find_fragmentation(mol, dataset.smarts)
    print(frag)
    pass


single_tests = []
df_single = pd.read_excel("test/csv/single_group_matching.xlsx")
start = 3
stop = 180
names = df_single.iloc[start:stop, 0]
smiles = df_single.iloc[start:stop, 1]
labels = df_single.iloc[start:stop, 2]
id = []

# 106, 118

for i in range(start, stop):
    id.append(i+1)

    single_tests.append(
        (smiles.at[i], labels.at[i], names.at[i], i+1))

full_fragmentation_tests = []
df_full = pd.read_excel("test/csv/fragmenting_tests_labeled.xlsx")


# single_tests_full = []
# df_single_full = df_single
# start_sf = 0
# stop_sf = -1
# names_sf = names = df_single.iloc[start:stop, 1]
# smiles_sf = df_single.iloc[start:stop, 1]
# labels_sf = df_single.iloc[start:stop, 3:-1]
# id_sf = []

# for i in range(start_sf, stop_sf):
#     id_sf.append(i)
#     single_tests_full.append(
#         (smiles_sf.at[i-1], labels_sf.at[i-1], names_sf.at[i-1], i))


# start_full = 0
# stop_full = 20
# names = df_full.iloc[start_full:stop_full, 0]
# smiles = df_full.iloc[start_full:stop_full, 1]
# labels = df_full.iloc[start_full:stop_full, 2:-1]
id_full = []

# for i in range(start_full, stop_full):
#     id_full.append(i)
#     # print(labels)
#     full_fragmentation_tests.append(
#         (smiles.at[i], pd.to_numeric(pd.Series(labels.iloc[i, :])).to_numpy(), names.at[i], i))


with open('test/txt/fragmentation_long.txt') as f:
    long_fragmentation_tests = f.readlines()


def test_replace_substruct():
    return NotImplementedError()


def test_get_rd_mol():
    """Basic test to check if a rdkit.Chem.rdchem.Mol object is returned.
    """
    assert get_rdmol('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC') is not None
    pass

# SMARTS SECTION

# @pytest.mark.parametrize("smile", short_fragmentation_tests)


def test_find_fragmentation_brief(dataset, fragmentation):
    """Test fragmentation of a simple SMILES string.
    """
    mol = get_rdmol("Clc1ccccc1")
    frag = find_fragmentation(mol, dataset.smarts)
    assert frag.matches == fragmentation.matches
    assert frag.vec[14] == 5
    assert frag.vec[122] == 1


@pytest.mark.parametrize("smile", long_fragmentation_tests)
def test_find_fragmentation_long(smile):
    """Test if fragmentation of long smiles hangs.
    """
    mol = get_rdmol(smile)
    print(find_fragmentation(mol, smarts))

# @pytest.mark.xfail() 35, 80, 95


@ pytest.mark.parametrize("smile, expected,name,number", single_tests, ids=id)
def test_correct_smarts(smile, expected, name, number, dataset):
    """Test if SMARTS substructures are properly (individually) matched.
    """
    # if number in set([150, 151, 155, 157, 158, 160, 162, 164, 175]):
    #     pytest.skip()
    mol = get_rdmol(smile)
    # print(dataset.smarts[number-1].smart_str[0])
    smart = mol.GetSubstructMatches(
        MolFromSmarts(dataset.smarts[number-1].smart_str[0][0]))
    assert len(smart) != 0 or (len(smart) == 0 and expected == 0)


@ pytest.mark.timeout(120)
@ pytest.mark.parametrize("smile, expected,name,number", single_tests, ids=id)
def test_group_matching_single(smile, expected, name, number, dataset):
    """Test if a fragmentation is found and the (single) target group is
    matched correctly.
    """
    mol = get_rdmol(smile)
    pprint.pprint(smile)
    frag = find_fragmentation(mol, dataset.smarts)
    # pprint.pprint(frag.groups_idx)
    # pprint.pprint(frag.matches)
    # pprint.pprint
    assert frag.vec[number-1] == expected


@ pytest.mark.parametrize("smile, expected,name,number", full_fragmentation_tests, ids=id_full)
def test_group_matching_full_fragmentation(smile, expected, name, number, dataset):
    """Test if a fragmentation is found and matches the ground truth based on
    the vectors generated from ProPred.
    """
    mol = get_rdmol(smile)
    generated = find_fragmentation(mol, dataset.smarts).vec
    np.testing.assert_array_equal(generated, expected)

# TODO


# @ pytest.mark.timeout(120)
# @ pytest.mark.parametrize("smile, expected,name,number", single_tests_full, ids=id_sf)
# def test_group_matching_single_full(smile, expected, name, number, dataset):
#     mol = get_rdmol(smile)
#     assert find_fragmentation(mol, dataset.smarts).vec == expected

# GNN SECTION

# @ pytest.mark.parametrize("smile, expected", single_tests, ids=id)
# def test_smart_matching_single(smile, expected):
#     print(smile)
#     mol = get_rdmol(smile)
#     print(find_fragmentation(mol, smarts, smarts))


# groups in test cases (priority): 1,2,3,4,5,6,7,8,9,10,15,22,28,29,33,34,38,42,
# 50,41,44,48,49,50,51,54,55,56,57,58,59,60,61,120,121,122,123

restricted_tests = []
df_restricted = pd.read_csv("test/csv/restricted_groups_test_small.csv")
start = 0
stop = 200
names = df_restricted.iloc[:, 0]
smiles = df_restricted.iloc[:, 1]
labels = df_restricted.iloc[:, 2: 222]
id_restricted = []

for i in range(start, stop):
    id_restricted.append(i+1)

    restricted_tests.append(
        (smiles.at[i], np.array(labels.iloc[i, :], dtype=np.float32), names.at[i], i+1))


@ pytest.mark.parametrize("smile,expected,name,number", restricted_tests, ids=id_restricted)
def test_restricted_group_matching(smile, expected, name, number, dataset):
    """Test whether fragmentations are correct when smarts queries are restricted
    to the groups used in the case studies
    """
    mol = get_rdmol(smile)
    idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 28, 29, 33, 34, 38, 42,
            50, 41, 44, 48, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61, 120,
            121, 122, 123]
    smart_restricted = []
    for smart in dataset.smarts:
        if smart.group_num in idxs:
            smart_restricted.append(smart)
    generated = find_fragmentation(mol, smart_restricted).vec
    np.testing.assert_array_equal(generated, expected)


def test_frag_to_nx_graph(dataset):
    """Test generating a single NetworkX graph from a Fragmentation object
    """
    mol = get_rdmol("C=C(C)C(=O)OC(C)(C)C")
    idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 28, 29, 33, 34, 38, 42,
            50, 41, 44, 48, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61, 120,
            121, 122, 123]
    smart_restricted = []
    for smart in dataset.smarts:
        if smart.group_num in idxs:
            smart_restricted.append(smart)
    frag = find_fragmentation(mol, smart_restricted)
    frag_to_nx_graph(frag, mol)
