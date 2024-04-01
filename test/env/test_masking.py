import env.masking as masking
from env.masking import MaskParams
from structs.dataset import DataSet
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


@pytest.fixture
def maskParams_mock_obj():
    with patch('env.masking.MaskParams') as mock:
        with patch('structs.state.RLState') as statemock:
            instance = mock.return_value
            instance.p = 1
            instance.num_bb = 4
            instance.bond_types = 3
            instance.max_per_group = 2
            instance.bb_idx = np.array([1, 2, 94, 101])
            instance.action_dims = (4, 3, 4, 3, 2)
            statemock.valency.va = np.array([0, 0, 0, 1, 0, 0, 0, 2]).reshape(instance.action_dims[0], instance.action_dims[-1])
            statemock.valency.vb = np.array([0, 0, 0, 0, 0, 0, 0, 2]).reshape(instance.action_dims[0], instance.action_dims[-1])
            statemock.valency.vc = np.array([0, 1, 0, 1, 0, 0, 1, 1]).reshape(instance.action_dims[0], instance.action_dims[-1])
            instance.state = statemock
            instance.state.group_count = np.array([1, 0, 0, 1])
            return instance


@pytest.fixture
def maskParams_mock_obj_full():
    with patch('env.masking.MaskParams') as mock:
        with patch('structs.state.RLState') as statemock:
            instance = mock.return_value
            instance.bb_idx = np.array(list(range(0, 220)))
            return instance


def test_adj_mask(maskParams_mock_obj):
    mask = masking.adj_mask(maskParams_mock_obj)
    expected = np.zeros((maskParams_mock_obj.action_dims))
    expected[:, :, 0, :, 0] = 1
    expected[:, :, 3, :, 0] = 1
    expected[:, :, 1, :, 0] = 0
    expected[:, :, 2, :, 0] = 0

    np.testing.assert_array_equal(mask, expected)

# def test_aromatics_mask(maskParams_mock_obj):
#     mask = masking.aromatics_mask(maskParams_mock_obj)
#     expected = np.array([True, True, True, True, False, False, True, True])
#     expected = np.logical_and(expected, np.ones((4, 3, 4, 3, 2), dtype=bool))
#     np.testing.assert_array_equal(mask, expected)


# def test_valency_mask(maskParams_mock_obj):
#     expected = np.zeros((3, 8))
#     expected[0, :] = np.array([0, 0, 0, 1, 0, 0, 0, 2])
#     expected[1, :] = np.array([0, 0, 0, 0, 0, 0, 0, 2])
#     expected[2, :] = np.array([0, 1, 0, 1, 0, 0, 1, 1])
#     expected = np.broadcast_to(expected, (4, 3, 8))

#     np.testing.assert_array_equal(
#         masking.valency_head_mask(maskParams_mock_obj), expected.astype(bool))


# def test_invalids_mask():
#     pass


# def test_forbidden_bonds_mask():
#     pass


# def test_forbidden_pairs_gen(maskParams_mock_obj_full):
#     A_nonA = masking.gen_A_nonA(maskParams_mock_obj_full)
#     UA_SUAS = masking.gen_UA_SUAS(maskParams_mock_obj_full)
#     UAS_UAS = masking.gen_UAS_UAS(maskParams_mock_obj_full)
#     print(len(A_nonA))
#     print(len(UA_SUAS))
#     print(len(UAS_UAS))


# def test_mbt_gen():
#     params = MagicMock()
#     params.bb_idx = np.array([1, 2, 3, 4, 15, 22, 29, 42, 123])
#     print(masking.gen_A_nonA(params))
#     print(masking.gen_UA_SUAS(params))
#     print(masking.gen_UAS_UAS(params))
