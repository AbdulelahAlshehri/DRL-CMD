
import heapq
import itertools


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from util.dataload import load_prop_models

# Verbose flags
verbose = 0
show_prop = False
incl_std_dev = True

# Initialize global prop_model list
prop_models = None


constraint_idx_map = {
    # Mapping of property/constraint parameter in human readable format to index
    "NUM_GROUPS": 1,  # total number of groups
    "NUM_REPEAT_GROUPS": 2,  # total number of repeat groups
    "NUM_FUNC_GROUPS": 3,  # total number of functional groups
    "MOLECULAR_WEIGHT": 4,  # molecular weight
    "FLASH_POINT": 5,  # - flash point
    "MELTING_POINT":  6,  # normal melting point (Tm)
    "BOILING_POINT": 7,  # normal boiling point (Tb)
    "CRITICAL_TEMP":  8,  # critical temperature (Tc)
    "CRITICAL_PRESSURE":  9,  # critical pressure (Pc)
    "CRITICAL_VOLUME": 10,  # critical volume (Vc)
    "GIBBS":  11,  # standard Gibbs free energy, 298K
    "ENTHALPY_FORMATION": 12,  # standard Enthalpy of formation, 298K
    "ENTHALPY_VAP": 13,  # enthalpy of vaporization, 298K
    "ENTHALPY_FUS": 14,  # enthalpy of fusion, 298K
    "ENTHALPY_VAP_BP": 15,  # enthalpy of vaporization at Tb
    "AUTOIG_TEMP": 16,  # autoignition temp
    "PKA": 17,  # pKa
    "ENTHALPY_SOL": 18,  # heat of solution at P, hsol
    "LC50": 19,  # LC50
    "LOGP": 20,  # logP
    "LOGWS": 21,  # logWs
    "LD50": 22,  # ld50
    "OSHA_TWA": 23,  # osha-twa
    "HSP": 24,  # hild solubility param
    "BCF": 25,
    "VAPOR_PRESSURE": 26,  # vapor pressure
    "VISCOSITY": 27  # v iscosity
}

mu = 0.5
sigma = 0.5


def constraints_gen(constraints_raw):
    """ Generates a formatted constraints matrix.

    Args:
        `constraints_raw`: a list of unprocessed constraints, each of which is formatted as a tuple (type, min, max).

    Returns:
        an nx3 formatted constraints matrix, with the columns ordered as:
            property number | min | max"""
    constraints = np.zeros((len(constraints_raw), 3))
    for i in range(0, len(constraints_raw)):
        constraints[i] = [constraint_idx_map[(constraints_raw[i])[0]],
                          constraints_raw[i][1], constraints_raw[i][2]]
    return constraints[constraints[:, 0].argsort()]


def spec_bounds(min, max, l):
    """
    Args:
        min: vector of length l denoting the minimum possible value of an element.
        max: vector of length l denoting the maximum possible value of an element.

    Returns: a tuple `(min, max)` where min and max are each vectors of length l
    """
    return [min for iter in range(l)], [max for iter in range(l)]


def pretty_format_molecule_set(self, properties, full_valence_valid,  order, full_vec=np.zeros((220,)), constr=None, ms=None,):
    """Converts the found solutions to human readable format. Adds molecules only if they meet all
    constraints (property, group, valency)

    Args:
        `self`: set to the MolecularSearchEnv obj if called during the RL run; None otherwise.
        `properties`: the properties of the correspoding molecule, polled from the property models.
        `full_valence_valid`: a boolean that indicates whether a zero-valency integral structure
                            has been found.
        `building_blocks`: is the set of building blocks to consider printing.
        `full_vec`: a full representation of the state vector, if called outside the RL run.
        `constraints`: CAMD constraints, if called outside the RL run.
        `molecule set`: list of found molecules, if called outside the RL run.

    Returns: a tuple representing the set of functional groups in a discovered solution, formatted
             as `(NUM GROUP_NAME, NUM2 GROUP_NAME2, ..)`

    Example: (1,1,0,0,0,0) --> (1 CH3, 1 CH2)"""
    if (order == 1):
        found_molecules = full_vec
        molecule_set = ms
        constraints = constr
        if (full_vec.any()):
            building_blocks = found_molecules.nonzero()
        if (not (full_vec.any())):
            constraints = self.constraints()
            building_blocks = self._building_blocks
            found_molecules = np.zeros((220,))
            found_molecules[building_blocks] = self._state['observation']
            molecule_set = self._molecule_set
        if group_constraints_check(found_molecules, constraints) and full_valence_valid:
            names = first_order_group_names[building_blocks]
            idx = found_molecules[building_blocks].nonzero()
            found_molecules = np.char.mod(
                "%d ", found_molecules[building_blocks])
            if show_prop:
                molecule_set.add((
                    tuple(np.char.add(found_molecules[idx], names[idx])), tuple(properties[:, 0].flatten()), tuple(properties[:, 1].flatten())))
            else:
                molecule_set.add((
                    tuple(np.char.add(found_molecules[idx], names[idx]))))


def print_state_visitation(hist):
    for key in hist.keys():
        print(str(np.frombuffer(key, dtype=np.int32)) + " " + str(hist[key]))


# @deprecated
def target_second_order_graph_gen(groups):
    """Generates a random, valid, connected graph for a given found solution of 1st order molecules.

    Arg:
        `groups`: A set of first order groups.

    Returns: A connected graph of the 1st order groups.
    """
    # TODO - half implemented
    graphs = []
    # not sure why indexing is off by 1
    graph_info = load_graph_info()
    G = nx.Graph()
    for i in range(0, len(groups)):
        G.add_node(i+1, group=dataset[groups[i], 1])

    return G


def generate_sog_cands(first_order_groups):
    # TODO
    """Generates all second order group candidates derived from the provided set `first_order_groups`
    through enumeration. Generated candidates will be used in subsequent subgraph matching step.

     Arg:
        `first_order_groups': A set of first order groups

     Returns: List of all possible second order groups
    """
    # Match all subsets of first order groups existing in the graph info for each second order group in the dataset


def enumerate_first_order_sets(n, group_constraints):
    """Enumerates all possible combinations of the first-order groups given
    the group constraints. Does not take into account any property constraints.

     Arg:
        `n`: number of candidate building blocks
        `group_constraints`: min of repeats,

     Returns: List of all possible first order group sets, given the group constraints
    """
    min, max = group_constraints[0][0]
    max = group_constraints[0][1]

    repeat_min = group_constraints[1][0]
    repeat_max = group_constraints[1][1]

    sets = np.zeros((1, n))
    # restrict total count of groups to [min, max]
    for k in range(min, max + 1):
        dividers = itertools.combinations(range(1, k+n), n-1)
        tmp = [(0,) + x + (k+n,) for x in dividers]
        sequences = np.diff(tmp) - 1
        sets = np.concatenate([sets, sequences[::1]])
        sets = sets[1:]
        # print(sets)

    # filter out all sets that violate repeat group constraints

    return sets[np.where(np.all(np.logical_and(
        sets <= repeat_max, sets >= repeat_min), axis=1))]


def filter_enumerated_sets(sets, constraints, building_blocks):
    """Extracts the enumerated sets that satisfy the priority constraints.

     Arg:
        `sets`: enumerated sets of first-order groups.
        `constraints': all constraints for the CAMD problem. nx2 matrix
        `building blocks': Set of building block first-order groups

     Returns: List of all possible first order group sets satisfying property constraints
    """

    property_model_input = np.zeros((np.shape(sets)[0], 220))
    property_model_input[:, building_blocks] = sets

    results = []

    def valid_func(x):
        property_valid, property_vals, _, full_valence_valid = is_valid(
            x, 1, constraints)
        return property_vals, (property_valid and full_valence_valid)

    ms = set()

    for i in tqdm(range(np.shape(property_model_input)[0])):
        property_vals, isv = valid_func(property_model_input[i])
        if (isv):
            print(property_model_input[i][building_blocks])
            heapq.heappush(results, (rewardFunction(RewardFunction.L1_EQUAL_WEIGHT, constraints,
                           property_model_input[i], property_vals, isv), property_vals, property_model_input[i]))
            pretty_format_molecule_set(
                None, property_vals, is_valid, 1, property_model_input[i], constraints, ms)

    print(sorted(results[:10], reverse=True))
    return ms


def brute_force_solution(constraints, building_blocks):
    """
    Generates the molecules found by a generate-and-test approach. Brute force enumeration of all
    possible group combinations, and subsequent testing to see if each candidate satisfies the
    property and group constraints.

    Arg:
        `constraints`: unformatted CAMD constraints.
        `building_blocks`: Set of building block first-order groups

    Returns: set of all molecules satisfying CAMD constraints
    """
    c = constraints_gen(constraints)
    all_comb = enumerate_first_order_sets(
        len(building_blocks), c[:, 1:3].astype(int))
    solutions = filter_enumerated_sets(all_comb, c, building_blocks)

    return solutions


def print_v(arg):
    """
    Helper function that prints only if the verbose flag is set.
    Arg: any value
    """
    if (verbose == 1):
        print(arg)


def visualize_second_order_graph(G):
    """Helper function that plots a collection of first order functional groups defining a single
    second order functional group"""
    labels = nx.get_node_attributes(G, 'group')
    nx.draw(G, labels=labels)
    plt.show(G)

# def valid_tails(G, node_idx):
#     """
#     Takes a node index in a network X graph G as input, and outputs the list of other possible nodes
#     that it may be legally attached to.

#     Arg:
#         `G`: a networkx graph G of the molecular fragment at a given step in the RL trajectory.
#              consists of exactly one isolated node (the node added due to the RL action)
#               and one connected subgraph (which represents the molecular fragment at the previous RL step)
#         `node_idx`: index/id of the disconnected node

#     Returns:
#         `node_list`: list of feasible attachment points (nodes).
#     """

#     n = G.nodes()[node_idx]
#     c = n['group_class']

# def subunit_check():
#     """
#     Checks if a set of first order groups can be combined to create another first order group
#     """

# def can_bond(a, b):
# # Group 17 is special in that is both aromatic and cyclic and possesses a cyclic bond (vb = 1)
# # Distinction is made between aromatic and cyclic groups -- aromatic groups do not count as cyclic groups
# # Cyclic groups - bond type a and b are cyclic bonds, type c is non-cyclic (can connect to aromatic or acyclic)
# # Aromatic groups - bond type a is aromatic bond, type b and c is non-aromatic (can connect to cyclic and acyclic)
# # Missing constraints: group17 constraints, group 3 and 4 constraints,
# # Heuristics:
# #   - One aromatic group and a non-aromatic group are combined.
# #    - One urea/amide group and a standard group or urea/amide subgroup
# #     are combined.
# #   - Two urea/amide subgroups are combined.
# #   - Two or more standard groups are combined o r one urea/amide
# #      subgroup and a standard group are combined.


#     if (a['vtotal'] > 0 and b['vtotal'] > 0):
#         if (a['group_class'] == 1): #standard
#             # If a is cyclic, can only have a cyclic bond connected to another cyclic group (or group 17),
#             # or a noncyclic bond connected to a non-cyclic group
#             if (a['cyclic']):
#                 # a and b both cyclic, so must each have available cyclic bonds
#                 if (b['cyclic'] and (a['va'] > 0 or a['vb'] > 0) and (b['va'] > 0 or b['vb'] > 0)):
#                     return True
#                 # a cyclic, b noncyclic, so a must have noncyclic bonds. a cyclic cannot be linked to
#                 # an aromatic unless it is to group 17. #TODO
#                 elif (not b['cyclic'] and b['group_class'] != 2 and (a['vb'] > 0 or a['vc'] > 0)):
#                     return True
#             # if a is not cyclic, cannot to any groups with no noncyclic bonds left
#             else:
#                 if (b['group_class'] == 1):
#                     if (b['cyclic']):
#                         return can_bond(b, a)
#                     else:
#                         return True
#                 if (b['group_class'] == 2 and ((b['vb'] > 0) or (b['vc'] > 0))):
#                     return True
#                 if (b['group_class'] == 3 or b['group_class'] == 4):
#                     return True
#         if (a['group_class'] == 2):
#             # a is aromatic, b is nonaromatic and standard
#             if (b['group_class'] == 1):
#                 return can_bond(b,a)
#             # a is aromatic and b is aromatic, must both have aromatic bonds available
#             if (b['group_class'] == 2 and (a['va'] > 0 and b['va'] > 0)):
#                 return True
#             # all urea/amide groups/subgroups are acyclic and nonaromatic
#             if (b['group_class'] == 3 or b['group_class'] == 4):
#                 return True
#         elif (a['group_class']== 3): #UA
#             if (b['group_class'] == 1 or b['group_class'] == 2):
#                 return can_bond(b, a)
#             if (b['group_class'] == 3):
#             if (b['group_class'] == 4):
#         elif (a['group_class'] == 4): #UAS
#             if (b['group_class'] == 4):

#             else:
#                 return can_bond(b, a)

#     return False

def fgroup(df):
    """Prints group assignments in human readable format given a list of molecules in a dataframe"""
    df = df.iloc[:, 1:425]
    bt = df.apply(lambda x: x > 0)
    print(bt.apply(lambda x: list(df.columns[x.values] + 1), axis=1))
