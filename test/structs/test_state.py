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


def test_RLState_add_group(RLState_obj):
    assert len(RLState_obj.graph.nodes()) == 0
    new_state = RLState_obj.add_group((4, 0, 0, 0, 1))
    assert len(new_state.graph.nodes()) == 1
    assert RLState_obj.n_G == 0
    np.testing.assert_array_equal(new_state.n_G, RLState_obj.n_G + 1)
    np.testing.assert_array_equal(new_state.group_count[4], 1)
    # print(RLState_obj.graph.nodes())
    # print(new_state.graph.nodes())
    assert new_state.graph.nodes[40]["va"] == 2
    assert new_state.graph.nodes[40]["vb"] == 0
    assert new_state.graph.nodes[40]["vc"] == 0
    print(new_state)
    new_state = new_state.add_group((4, 0, 4, 0, 0))
    assert new_state.graph.nodes[41]["valency"] == 1
    assert new_state.graph.nodes[41]["group"] == 15


def test_RLState_add_group_to_nonempty_basic(RLState_obj):
    new_state = RLState_obj.add_group((4, 0, 0, 0, 0))
    new_state = new_state.add_group((4, 0, 4, 0, 0))
    assert len(new_state.graph.nodes()) == 2
    np.testing.assert_array_equal(new_state.group_count[4], 2)
    assert new_state.graph.nodes[40]["va"] == 1
    assert new_state.graph.nodes[40]["vb"] == 0
    assert new_state.graph.nodes[40]["vc"] == 0
    assert new_state.graph.nodes[40]["valency"] == 1


# def test_RLState_getters_default(RLState_obj):
#     # assert RLState_obj.get_valency() ==
#     np.testing.assert_array_equal(RLState_obj.noncyclics, np.zeros(9))
#     np.testing.assert_array_equal(RLState_obj.cyclics, np.array([]))
#     np.testing.assert_array_equal(RLState_obj.aromatics, np.zeros((3)))
#     np.testing.assert_array_equal(RLState_obj.bb, np.array(
#         [1, 2, 3, 4, 15, 22, 29, 42, 123]))
#     np.testing.assert_array_equal(
#         RLState_obj.valency.total, np.zeros((RLState_obj.num_bb)))
#     np.testing.assert_array_equal(
#         RLState_obj.valency.va, np.zeros((RLState_obj.num_bb)))
#     np.testing.assert_array_equal(
#         RLState_obj.valency.vb, np.zeros((RLState_obj.num_bb)))
#     np.testing.assert_array_equal(
#         RLState_obj.valency.vc, np.zeros((RLState_obj.num_bb)))
#     np.testing.assert_array_equal(RLState_obj.i_G_a, np.array([15, 22, 123]))
#     np.testing.assert_array_equal(
#         RLState_obj.i_G_na, np.array([1, 2, 3, 4, 29, 42]))
#     np.testing.assert_array_equal(RLState_obj.i_G_c, np.array([]))
#     np.testing.assert_array_equal(
#         RLState_obj.i_G_nc, np.array([1, 2, 3, 4, 15, 22, 29, 42, 123]))


# def test_RLState_is_isomorphic(RLState_obj):
#     pass


# def test_RLState_to_cmp_vec(RLState_obj):
#     pass


# def test_RLState_to_full_vec(RLState_obj):
#     pass
