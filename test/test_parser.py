from util.parse import create_parser, ParseData
import pytest


@pytest.fixture
def multialg_args():
    return '-r 1 -a rf ppo dqn'.split()


@pytest.fixture
def surfactant_args():
    return '-c mbt surfactant -r 2 -sp'.split()


def test_create_parser_default():
    parser = create_parser()
    expected = {'cases': None,
                'rings': '1',
                'verbose': False,
                'show_prop': False,
                'algorithms': [],
                'save_settings': None}
    args = parser.parse_args('-r 1'.split())
    assert vars(args) == expected


def test_create_parser_modified_with_algs():
    parser = create_parser()
    expected = {'cases': None,
                'rings': '1',
                'verbose': False,
                'show_prop': False,
                'algorithms': [['rf', 'ppo', 'dqn']],
                'save_settings': None}
    args = parser.parse_args('-r 1 -a rf ppo dqn'.split())
    assert vars(args) == expected


def test_create_parser_modified(surfactant_args):
    parser = create_parser()
    expected = {'cases': [['mbt', 'surfactant']],
                'rings': '2', 'show_prop': True,
                'verbose': False,
                'show_prop': True,
                'algorithms': [],
                'save_settings': None}
    args = parser.parse_args(surfactant_args)
    assert vars(args) == expected


def test_ParseData_cases(surfactant_args):
    pd = ParseData(surfactant_args)
    assert pd.cases == ['mbt', 'surfactant']


def test_ParseData_rings(multialg_args):
    pd = ParseData(multialg_args)
    assert pd.rings == '1'


def test_ParseData_verbose(multialg_args):
    pd = ParseData(multialg_args)
    assert pd.verbose == False


def test_ParseData_algorithms(multialg_args):
    pd = ParseData(multialg_args)
    assert pd.algorithms == ['rf', 'ppo', 'dqn']


def test_ParseData_show_prop(multialg_args):
    pd = ParseData(multialg_args)
    assert pd.show_prop == False
