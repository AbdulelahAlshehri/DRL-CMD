import numpy as np
import util
from sklearn.gaussian_process import *

import pandas as pd


class TestUtilFactoryMethods(unittest.TestCase):
    def setUp(self):
        # Sets up example formulations to be used in the following tests.
        ex = [("NUM_GROUPS", 3, 8),
              ("NUM_REPEAT_GROUPS", 0, 7),
              ("NUM_FUNC_GROUPS", 1, 6),
              ("MOLECULAR_WEIGHT", 80, 200),
              ("MELTING_POINT", 173, 310),
              ("BOILING_POINT", 373, 600),
              ("FLASH_POINT", 273, 393)]

        ex_simple = [("NUM_GROUPS", 400, 600),
                     ("NUM_REPEAT_GROUPS", 0, 7),
                     ("NUM_FUNC_GROUPS", 1, 6),
                     ("MELTING_POINT", 173, 310),
                     ("BOILING_POINT", 373, 600)]
        self.constraints = util.constraints_gen(ex_simple)
        aromatics_fails = pd.read_csv("test/aromatics_fails.csv")
        aromatics_fails = aromatics_fails.to_numpy()[:, 6:]

        np.random.seed(2815)
        self.zero_state = util.state_gen(220, 1)
        self.valency_ex1 = [2, 1, 0]
        self.valency_ex2 = [0, 8, 0, 0]
        self.valency_ex3 = [0, 0, 0, 0, 0]

    def test_spec_bounds(self):
        np.testing.assert_equal(util.spec_bounds(
            200, 500, 2), ([200, 200], [500, 500]))

    def test_constraints_gen(self):
        np.testing.assert_almost_equal(self.constraints,
                                       [[1., 400., 600.],
                                        [2.,   0., 7.],
                                        [3.,   1., 6.],
                                        [6., 173., 310.],
                                        [7., 373., 600.]])

    def test_state_gen1(self):
        np.testing.assert_equal(util.state_gen(424, 0), np.zeros((424,)))

    def test_state_gen2(self):
        np.testing.assert_equal(util.state_gen(10, 8) <= 8, True)

    def test_cyclic_check_passes_validation(self):
        cyclics_passes = (pd.read_csv(
            "test/cyclics_passes.csv")).to_numpy()[:, 2:]

        np.testing.assert_equal(np.all(util.cyclic_check(
            cyclics_passes, util.valency, None)), True)

    def test_cyclic_check_fails_validation(self):
        cyclics_fails = pd.read_csv("test/cyclics_fails.csv").to_numpy()[:, 2:]

        np.testing.assert_equal(util.cyclic_check(
            cyclics_fails, util.valency, -1), False)

        np.testing.assert_equal(util.cyclic_check(
            cyclics_fails, util.valency, 0), False)

    def test_aromatic_check_passes_validation(self):
        # TODO
        return
    #     aromatics_passes = (pd.read_csv(
    #         "test/aromatics_passes.csv")).to_numpy()[:, 2:]

    #     np.testing.assert_equal(np.all(util.aromatic_check(
    #         aromatics_passes, util.valency, None)), True)

    def test_aromatic_check_fails_validation(self):
        # TODO
        return
    #     aromatics_fails = (pd.read_csv(
    #         "test/aromatics_fails.csv")).to_numpy()[:, 6:]

    #     np.testing.assert_equal(
    #         np.all(util.aromatic_check(aromatics_fails), True))
    #     #

    def test_group_constraints_check(self):
        # TODO
        return

    def test_valence_check_nonzero(self):
        np.testing.assert_equal(util.valence_check(
            self.valency_ex1, util.valency[:3], 1), False)
        np.testing.assert_equal(util.valence_check(
            self.valency_ex1, util.valency[:3], 0), False)
        np.testing.assert_equal(util.valence_check(
            self.valency_ex1, util.valency[:3], -1), True)

        np.testing.assert_equal(util.valence_check(
            self.valency_ex2, util.valency[:4], 1), False)
        np.testing.assert_equal(util.valence_check(
            self.valency_ex2, util.valency[:4], 0), True)
        np.testing.assert_equal(util.valence_check(
            self.valency_ex2, util.valency[:4], -1), False)

    def test_valency_check_p_is_none(self):
        np.testing.assert_equal(util.valence_check(
            self.valency_ex1, util.valency[:3], None), True)
        np.testing.assert_equal(util.valence_check(
            self.valency_ex2, util.valency[:4], None), True)
        np.testing.assert_equal(util.valence_check(
            self.valency_ex3, util.valency[:5], None), True)

    def test_valency_check_fails(self):
        np.testing.assert_equal(util.valence_check(
            [0, 0, 3], util.valency[:3], None), False)
        np.testing.assert_equal(util.valence_check(
            [1, 2, 4, 5, 3], util.valency[:5], None), False)
        np.testing.assert_equal(util.valence_check(
            [10, 0, 0], util.valency[:3], None), False)

    def test_group_constraints_check(self):
        # TODO
        return

    def test_group_constraints_check(self):
        # TODO
        return

    def test_is_valid(self):
        # TODO
        is_valid, explored_vals = util.is_valid(
            self.zero_state, util.valency, None, self.constraints)
        np.testing.assert_equal(is_valid, False)
        np.testing.assert_almost_equal(
            explored_vals[1], [463.6104469, 128.0688435])

    # def test_is_valid_zeros(self):
    #     np.testing.assert_equal(util.valence_check(
    #         self.valency_ex3, util.valency[:5], -1), False)
    #     np.testing.assert_equal(util.valence_check(
    #         self.valency_ex3, util.valency[:5], 0), True)
    #     np.testing.assert_equal(util.valence_check(
    #         self.valency_ex3, util.valency[:5], 1), False)

    # def test_reward_function_is_zero_one_passes(self):
    #     property_vals = [4, 6,  5, 125, 200,  400,  300]
    #     assert(util.rewardFunction(
    #         util.RewardFunction.ZERO_ONE, self.constraints, property_vals) == 1.0, 1.0)

    # def test_reward_function_is_zero_one_fails(self):
    #     property_vals = [4, 6,  5, 1000, 200,  400,  300]
    #     assert(util.rewardFunction(
    #         util.RewardFunction.ZERO_ONE, self.constraints, property_vals) == 0, 1.0)

    def test_reward_function_is_l1_passes(self):
        # TODO
        property_vals = [4, 6,  5, 125, 200,  400,  300]
        self.constraints = np.array([[1, 2, 4], [2, 5, 7], [3, 4, 6], [4, 124, 126], [
            5, 199, 201], [6, 399, 401], [7, 299, 303]])
        assert(util.rewardFunction(
            util.RewardFunction.L1_EQUAL_WEIGHT, self.constraints, property_vals) == (-2.0, 1.0))

    # def test_reward_function_is_l1_fails(self):
    #     # TODO
    #     return

    # def test_reward_function_is_l2_passes(self):
    #     # TODO
    #     return

    # def test_reward_function_is_l2_fails(self):
    #     # TODO
    #     return

    # def test_reward_function_is_gaussian_passes(self):
    #     # TODO
    #     return

    # def test_reward_function_is_gaussian_fails(self):
    #     # TODO
    #     return

    # def tearDown(self):
    #     # to be implementedz
    #     return

    # def test_adjacency_gen():
    #     # TODO
    #     # to be implemented
    #     return

    def test_reward_scaling(self):
        # TODO
        return

    def test_graph_gen(self):
        graphs = []
        # not sure why indexing is off by 1
        print(util.second_order_graph_gen(221)[0])

    def test_enumerate_first_order(self):
        building_blocks = np.array([1, 2, 3, 4, 20, 22, 29,
                                    42, 123]) - 1
        group_constraints = np.array([[3, 8], [0, 7]])
        constraints = [("NUM_GROUPS", 3, 8),
                       ("NUM_REPEAT_GROUPS", 0, 7),
                       ("NUM_FUNC_GROUPS", 1, 6),
                       ("MOLECULAR_WEIGHT", 80, 200),
                       ("MELTING_POINT", 173, 310),
                       ("BOILING_POINT", 373, 600),
                       ("FLASH_POINT", 273, 393)
                       ]

        constraints = util.constraints_gen(constraints)
        gen = util.enumerate_first_order_sets(9, group_constraints)
        # print(util.enumerate_first_order_sets(5, group_constraints))
        print(util.filter_enumerated_sets(gen, constraints, building_blocks))


if __name__ == '__main__':
    unittest.main()
