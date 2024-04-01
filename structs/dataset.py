import numpy as np
from util import util
import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic

from dataclasses import dataclass, field, InitVar
# import graph
from typing import List

import pandas as pd

from util.dataload import open_cases, open_dataset
from structs.smarts import GroupSMARTS


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class DataSet:
    """Representation of main dataset.

    Attributes:
        valences:
        first_order_group_names:
        weights:
        group_names:
        group_nums:
        group_class:
        cyclics:
        noncyclics:
        aromatics:
        nonaromatics:
        smarts_table:
        parents:
        smart:
        add_smart_exclusions:
        _has_parents:
    """

    def __init__(self):
        pd_data, dataset = open_dataset()
        self.data: np.ndarray = dataset
        self.pd_data: pd.DataFrame = pd_data

    @property
    def valences(self):
        va, vb, vc = self.data[:220,
                               2], self.data[:220, 4], self.data[:220, 6]
        total = va + vb + vc
        
        return total, va, vb, vc

    @property
    def first_order_group_names(self):
        return self.data[:220, 1]

    @property
    def weights(self):
        return self.data[:220, 12]

    @property
    def group_names(self):
        return self.data[:424, 1]

    @property
    def group_nums(self):
        return self.data[:424, 0]

    @property
    def group_class(self):
        return self.data[:350, 11]

    def get_class(self, group_num):
        return self.data[group_num - 1, 11]

    def get_cyclic(self, group_num):
        return self.data[group_num - 1, 16]

    @property
    def aromatics(self):
        return np.where(self.data[:220, 11] == 'A')

    @property
    def smarts_table(self):
        # return self.pd_data.iloc[0:185, 5].to_list()
        return self.pd_data.iloc[0:220, 5].str.split('|')

    @property
    def parents(self):
        return self.pd_data.iloc[0:220, 15]

    @property
    def smarts(self):
        results = []
        table = self.smarts_table
        for i, smart_list in table.items():
            original_smarts = []
            modified_smarts = []
            for smart in smart_list:
                original_smarts.append(smart)
                if self._has_parents(i):
                    # indicates multiple expressions
                    if smart.count('[') > 1 or smart[-1] != ']':
                        # wrap with component grouping
                        # leave out outermost right bracket
                        smart = "[$(" + smart + ")"
                    else:
                        smart = smart[:-1]  # take out right bracket
                    smart = self.add_smart_exclusions(smart, i)
                modified_smarts.append([smart])
            results.append(GroupSMARTS(original_smarts,
                                       modified_smarts, i + 1, self.weights[i]))
        return results

    def _has_parents(self, smart_i):
        return str(self.parents.iat[smart_i]) != '-' and pd.notna(self.parents[smart_i])

    def add_smart_exclusions(self, smart, i):
        # print(parents)
        parents = self.parents[i].split(',')
        parents = [int(p) for p in parents]

        for p in parents:
            # print(p)
            # print(self.smarts_table.iat[p-1][0])
            parent_smart = self.smarts_table.iat[p-1][0]
            smart = smart + "&!$(" + parent_smart + ")"
        smart = smart + "]"
        return smart

    @ property
    # TODO
    def groups_with_perms(self):
        return self.pd_data.iloc[:, 1].str.extract(
            r'((\((([a-z]((,*([a-z]*([><=,])+\s*))|[a-z])*( in |[><=])*([0-9]..[0-9]|[0-9]))[;,\s]*)*\)))').dropna(thresh=2)

    @ property
    def cyclics(self):
        return np.where(self.data[:220, 16] == 'C')

    @ property
    def nonaromatics(self):
        return np.where(self.data[:220, 11] != 'A')

    @ property
    def noncyclics(self):
        return np.where(self.data[:220, 16] != 'C')

    @property
    def urea_amides(self):
        return np.where(self.data[:220, 11] == 'UA')

    @property
    def urea_amide_subgroups(self):
        return np.where(self.data[:220, 11] == 'UAS')

    @property
    def standards(self):
        return np.where(self.data[:220, 11] == 'S')

    def __str__(self):
        return "Placeholder text"


def vec_to_plaintext(vec: np.ndarray) -> List:
    """Converts a group count vector to a human-readable list of group counts

    Args:
        vec: The group count vector.
    """

    group_idxs = np.nonzero(vec)
    names = DataSet.instance().first_order_group_names
    fun = np.vectorize(lambda x: str(int(vec[x])) + " " + names[x])
    return fun(group_idxs).flatten().tolist()
