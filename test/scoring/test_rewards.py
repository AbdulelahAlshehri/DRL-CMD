from env.reward import *
import pytest
from util.parse import ParseData
from structs.dataset import DataSet
from structs.case import CaseSuite, Case
from structs.state import RLState


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
    state = RLState(mbt_Case_obj)
    state.property_vals = np.array([[1.0, 0],
                                    [334, 0],
                                    [242.5, 0],
                                    [487.5, 0],
                                    [3.4, 0],
                                    [20.5, 0]])
    return state


@pytest.fixture
def gp(mbt_Case_obj):
    return GroupScoreParams(mbt_Case_obj)


@pytest.fixture
def pp(RLState_obj, mbt_Case_obj):
    return PropScoreParams(RLState_obj, mbt_Case_obj)


def test_ParameterObjects(gp, pp):
    print(pp.min)
    print(pp.max)
    print(pp.cx)
    print(pp.mw_range)
    print(pp.mw_val)
    print(pp.vals)


def test_L1(RLState_obj, mbt_Case_obj):
    calc = L1WithEqualWeightRewardCalculator(RLState_obj, mbt_Case_obj)
    r = calc.compute_reward()
    print(calc)


def test_L1scaled(RLState_obj, mbt_Case_obj):
    calc = L1ScaledRewardCalculator(RLState_obj, mbt_Case_obj)
    r = calc.compute_reward()
    print(calc)

    # def test_GaussianRewardCalculator():
    #     calc = GaussianRewardCalculator(RewardCalculator)
    #     pass

    # def test_L1ScaledRewardCalculator():
    #     calc = L1ScaledRewardCalculator(RewardCalculator)
    #     pass

    # def test_L1WithEqualWeightRewardCalculator():
    #     calc = L1WithEqualWeightRewardCalculator(RewardCalculator)
    #     pass

    # def test_L2ScaledRewardCalculator():
    #     calc = L2ScaledRewardCalculator(RewardCalculator)
    #     pass

    # def test_L2WithEqualWeightRewardCalculator():
    #     calc = L2WithEqualWeightRewardCalculator(RewardCalculator)
    #     pass

    # def test_ZeroOneRewardCalculator():
    #     calc = ZeroOneRewardCalculator(RewardCalculator)
    #     pass
