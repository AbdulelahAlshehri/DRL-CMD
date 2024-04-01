from structs.valency import Valency
import pytest
import networkx as nx
import numpy as np


@pytest.fixture
def Valency_obj():
    G = nx.Graph()
    G.add_node(2, bb_idx=2, inner_idx=0, va="2", vb="1")
    return Valency(G, 2, 3)


def test_val_to_arr(Valency_obj):
    expected = np.zeros(2 * 3)
    expected[4] = 2
    np.testing.assert_equal(Valency_obj.val_to_arr('va'), expected)
