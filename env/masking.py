import numpy as np
from structs.dataset import DataSet
from dataclasses import dataclass, InitVar, field

dataset = DataSet.instance()


@dataclass
class MaskParams:
    env: InitVar

    def __post_init__(self, env):
        self.p = env.p
        self.bb_idx = env.case.bb
        self.invalids = env.invalids
        self.num_bb = env.num_bb
        self.bond_types = env.bond_types
        self.max_per_group = env.max_per_group
        self.state = env.state
        self.action = env.action
        self.action_dims = env.action_dims
        self.n = np.prod(self.action_dims)


def mask(params):
    """ Generates the entire action mask for the current time step.
    """
    if (np.sum(params.state.group_count) == 0):
        return np.logical_and(adj_mask(params), invalids_mask(params))
    mask = adj_mask(params)
    mask = np.logical_and(mask, valency_head_mask(params))
    mask = np.logical_and(mask, valency_tail_mask(params))
    mask = np.logical_and(mask, invalids_mask(params))
    mask = np.logical_and(mask, max_per_group_mask(params))
    mask = np.logical_and(mask, compatibility_chemical(params))
    return mask


def adj_mask(params):
    """ Masks out actions that connect to nonexistent nodes.
    """
    adj_mask = np.zeros(params.action_dims)
    inst = 1 if np.sum(params.state.group_count) == 0 else 0
    # nodes that do not exist in the graph
    for i in range(params.num_bb):
        if params.state.group_count[i] != 0:
            adj_mask[:,
                    :,
                    i,
                    :,
                    :int(params.state.group_count[i] + inst),
                    ] = np.ones(int(params.state.group_count[i] + inst), dtype=bool)
        else:
            adj_mask[:,
                    :,
                    i,
                    :,
                    0,
                    ] = np.ones(1, dtype=bool)            

    return adj_mask


def valency_head_mask(params):
    """ Masks out actions that attempt to connect bond types with 0 valency in the head group
    """
    # print(params.state.valency.va)
    mask = np.stack([params.state.valency.va_full(params.bb_idx),
                    params.state.valency.vb_full(params.bb_idx),
                    params.state.valency.vc_full(params.bb_idx)],
                    axis=1)
    mask = np.expand_dims(mask, axis=(2,3, 4))
    print(np.where(mask))
    return mask.astype(bool)


def valency_tail_mask(params):
    """ Masks out actions that connect to tail bond types with no more valency
    """

    mask = np.stack([params.state.valency.va,
                    params.state.valency.vb,
                    params.state.valency.vc],
                    axis=1).astype(bool)
    return np.broadcast_to(mask, params.action_dims)

def max_per_group_mask(params):
    """Mask out group additions for which the max number has already been reached.
    """
    mask = np.ones(params.action_dims, dtype=bool)
    mask[params.state.group_count >= params.max_per_group, :, :, :, :] = False
    return mask


def invalids_mask(params):
    """ Masks out actions that actions that the agent has historically experienced
    termination from, given the current state.
    
    """
    # 
    if (params.invalids.get(params.state.key()) is not None):
        return params.invalids[params.state.key()].reshape(params.action_dims)
    else:
        return True


def aromatics_mask(params):
    """ Masks out all aromatic groups if p = 1.
    
    """
    if (params.p == 1):
        aromatics = dataset.aromatics[0]
        idxs = ~np.isin(params.bb_idx - 1, aromatics)
        idxs = np.repeat(idxs, params.max_per_group)
        idxs = np.broadcast_to(idxs, (params.action_dims))
    return idxs


def compatibility_chemical(params):
    """ Generates the mask for forbidden 
    bondtype to bondtype bonds based on actual
    chemical rules.
    
    """
    A = forbidden_chem_A(params)
    S = forbidden_chem_S(params)
    UA = forbidden_chem_UA(params)
    UAS = forbidden_chem_UAS(params)
    to_A = forbidden_chem_to_A(params)
    to_C = forbidden_chem_to_C(params)


    return A & S & UA & UAS & to_A & to_C


def compatibility_heur(params):
    """ Generates the mask for forbidden bonds based 
    on MG heuristic uniqueness rules
    """
    A = forbidden_heur_A_nonA(params)
    S = forbidden_heur_UA_SUAS(params)
    UA = forbidden_heur_UAS_UAS(params)
    UAS = forbidden_heur_S_SUAS(params)

    return A & S & UA & UAS


def forbidden_chem_A(params):
    """
    """
    aromatics = dataset.aromatics[0]
    mask = np.ones((params.action_dims), dtype=bool)
    aromatic_bb = np.isin(params.bb_idx, aromatics)
    nonaromatic_bb = ~np.isin(params.bb_idx, aromatics)

    # (a-b), (a-c) not allowed for A-A
    mask[aromatic_bb][:, 0, aromatic_bb, 1, :] = False
    mask[aromatic_bb][:, 0, aromatic_bb, 2, :] = False

    # a to ANY not allowed for A-nonA
    mask[aromatic_bb][:, 0, nonaromatic_bb, :, :] = False
    mask[aromatic_bb][:, 0, nonaromatic_bb, :, :] = False
    mask[nonaromatic_bb][:, :, aromatic_bb, 0, :] = False
    mask[nonaromatic_bb][:, :, aromatic_bb, 0, :] = False

    # (b-a), (c-a) not allowed A-A
    mask[aromatic_bb][:, 1, aromatic_bb, 0, :] = False
    mask[aromatic_bb][:, 2, aromatic_bb, 0, :] = False

    return mask


def forbidden_chem_C(params):
    """
    """
    cyclics = dataset.cyclics[0]
    cyclic_bb = np.isin(params.bb_idx, cyclics)

    noncyclics = dataset.noncyclics[0]
    noncyclic_bb = np.isin(params.bb_idx, noncyclics)

    mask = np.zeros((params.action_dims), dtype=bool)

    # (a-a, a-b, tail cyclic)
    mask[cyclic_bb, 0, cyclic_bb, 0, :] = True
    mask[cyclic_bb, 0, cyclic_bb, 1, :] = True

    # (b-a, b-b, tail cyclic)
    mask[cyclic_bb, 1, cyclic_bb, 0, :] = True
    mask[cyclic_bb, 1, cyclic_bb, 1, :] = True

    # (c-a, c-b, tail noncyclic)
    mask[cyclic_bb, 2, noncyclic_bb, :, :] = True
    return mask


def forbidden_chem_S(params):
    """
    """
    # cyclics = dataset.cyclics[0]
    # cyclic_bb = np.isin(params.bb, cyclics)

    # noncyclics = dataset.noncyclics[0]
    # noncyclic_bb = np.isin(params.bb, noncyclics)

    # mask = np.zeros((params.action_dims))
    return np.ones(params.action_dims, dtype=bool)


def forbidden_chem_UA(params):
    """
    """
    return np.ones(params.action_dims, dtype=bool)


def forbidden_chem_UAS(params):
    """
    """
    return np.ones(params.action_dims, dtype=bool)


def forbidden_chem_to_A(params):
    """
    """
    aromatics = dataset.aromatics[0]
    aromatic_bb = np.isin(params.bb_idx, aromatics)

    nonaromatics = dataset.nonaromatics[0]
    nonaromatic_bb = np.isin(params.bb_idx, nonaromatics)

    mask = np.ones((params.action_dims), dtype=bool)
    mask[nonaromatic_bb][:, 0, aromatic_bb, 0, :] = False
    return mask


def forbidden_chem_to_C(params):
    cyclics = dataset.cyclics[0]
    cyclic_bb = np.isin(params.bb_idx, cyclics)

    noncyclics = dataset.noncyclics[0]
    noncyclic_bb = np.isin(params.bb_idx, noncyclics)

    mask = np.ones((params.action_dims), dtype=bool)
    mask[noncyclic_bb][:, 0, cyclic_bb, 0, :] = False
    mask[noncyclic_bb][:, 0, cyclic_bb, 1, :] = False
    return mask


def forbidden_heur_A_nonA(params):
    group, bond_type, tail = params.action
    tail = tail // params.max_per_group
    classes = (dataset.get_class(tail),
               dataset.get_class(group))
    # if classes == set('A', 'UA') or \
    #    classes == set('A', 'S') or \
    #    classes == set('A', 'UAS') or \


def forbidden_heur_UA_SUAS(params):
    pass


def forbidden_heur_UAS_UAS(params):
    pass


def forbidden_heur_S_SUAS(params):
    pass


def gen_A_nonA(params):
    aromatics = group_filter(params.bb_idx, "A")
    # print(aromatics)
    nonaromatics = group_filter(params.bb_idx, "NA")
    # print(nonaromatics)
    pairs = [set([a, na]) for a in aromatics for na in nonaromatics]
    return pairs


def gen_UA_SUAS(params):
    urea_amides = group_filter(params.bb_idx, "UA")
    urea_amide_subgroups = group_filter(params.bb_idx, "UAS")
    standards = group_filter(params.bb_idx, "S")
    pairs_ua_s = [set([ua, s]) for ua in urea_amides for s in standards]
    pairs_ua_uas = [set([ua, uas])
                    for ua in urea_amides for uas in urea_amide_subgroups]
    return pairs_ua_s, pairs_ua_uas


def gen_UAS_UAS(params):
    urea_amide_subgroups = group_filter(params.bb_idx, "UAS")
    pairs = [set([ua, ua2])
             for ua in urea_amide_subgroups for ua2 in urea_amide_subgroups]
    return pairs


def gen_forbidden_pair_S_SUAS(params):
    # TODO:
    # aromatics = filter(params.bb_idx, "S")
    # nonaromatics = filter(params.bb_idx, "UAS")
    # pairs = [(a, na) for na in aromatics for a in aromatics]
    pass


def group_filter(bb_idx, cls):
    if cls == "A":
        return bb_idx[np.isin(bb_idx, dataset.aromatics[0])]
    if cls == "NA":
        return bb_idx[np.isin(bb_idx, dataset.nonaromatics[0])]
    if cls == "UA":
        return bb_idx[np.isin(bb_idx, dataset.urea_amides[0])]
    if cls == "UAS":
        return bb_idx[np.isin(bb_idx, dataset.urea_amide_subgroups[0])]
    if cls == "C":
        return bb_idx[np.isin(bb_idx, dataset.cyclics[0])]
    if cls == "NC":
        return bb_idx[np.isin(bb_idx, dataset.noncyclics[0])]
    if cls == "S":
        return bb_idx[np.isin(bb_idx, dataset.standards[0])]
