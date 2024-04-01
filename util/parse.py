
import argparse


class ParseData():
    def __init__(self, testargs=None):
        self.data = None
        self.get_parse_data(testargs)

    def get_parse_data(self, testargs):
        if self.data is None:
            parser = create_parser()
            if testargs is None:
                self.data = parser.parse_known_args()[0]
                print(self.data)
            else:
                self.data = parser.parse_known_args(testargs)[0]
                print(self.data)

    @property
    def cases(self):
        return self.data.cases[0]

    @property
    def rings(self):
        return self.data.rings

    @property
    def verbose(self):
        return self.data.verbose

    @property
    def algorithms(self):
        return self.data.algorithms

    @property
    def show_prop(self):
        return self.data.show_prop

    @property
    def algorithms(self):
        return self.data.algorithms[0]

    # TODO
    @property
    def save(self):
        pass


def create_parser():
    parser = argparse.ArgumentParser(description='Run RL-CAMD.')

    # Case study options
    parser.add_argument('-c', dest='cases',
                        action='append',
                        required=False,
                        nargs='*',
                        help='run user specified cases')

    # Functional group granularity options

    # Cyclic / Aromaticity options
    parser.add_argument('-r', dest='rings',
                        help='set the number of desired rings',
                        required=False, default='1')

    # Verbose
    parser.add_argument('-v', dest='verbose',
                        action='store_true',
                        help='enable verbose output',
                        required=False,
                        default=False)
    parser.add_argument('-sp', dest='show_prop',
                        action='store_true',
                        help='show property values of found solutions during a run',
                        required=False,
                        default=False)

    # Options for algorithms to use
    parser.add_argument('-a', dest='algorithms',
                        help='algorithm(s) to run',
                        action='append',
                        nargs='*',
                        default=[],
                        required=False)

    parser.add_argument('--save', dest='save_settings',
                        help='write settings to file',
                        action='store',
                        required=False)

    return parser
