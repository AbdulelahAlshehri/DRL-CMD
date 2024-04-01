import heapq
import os
import pprint
import random
from ast import Sub
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rdkit
import torch
import pickle
from networkx.algorithms import isomorphism
from rdkit import Chem
from torch.utils.data import TensorDataset

from structs.dataset import DataSet
from structs.smarts import GroupSMARTS
from util.dataload import open_dataset
from structs.dataset import vec_to_plaintext


class GraphDataLoader:
    """Loader of data needed to generate full molecule graphs from SMILES.

    Attributes:
        data: dataset as a pytorch Dataset
        data_pd: dataset in Pandas
        data_np: dataset with labels in NumPy
        n: number of examples to generate
    """

    def __init__(self, path: str, split: float = 0, num_examples=500):
        dataset = pd.read_csv(path, encoding="ISO-8859-1")
        self.n = num_examples
        self.data_pd = dataset

    @property
    def data(self) -> pd.DataFrame:
        return

    @property
    def data_np(self) -> Tuple[pd.DataFrame, np.ndarray]:
        name = self.data_pd.iloc[:self.n, 0]
        smiles = self.data_pd.iloc[:self.n, 1]
        labels = self.data_pd.iloc[:self.n, 2:222].values
        smarts = [self.get_minimum_smarts_queries(label) for label in labels]
        return smiles, smarts, labels

    def get_minimum_smarts_queries(self, count_vector):
        smarts_idxs = np.argwhere(count_vector)
        return [DataSet.instance().smarts[i] for i in list(smarts_idxs.flatten())]


class GraphQueryDataLoader:
    """Loader for adjacency and parent information of second order query groups.
    """

    def __init__(self, path: str = None):
        if path is None:
            self.data = self.generate()
        else:
            self.data = self.load_graphs(path)

    def generate(self, start: int = 221, stop: int = 340):
        df = self.load_adj()
        graph_info = load_graph_info(df)
        G_list = []
        for i in range(start, stop):
            graphs = second_order_graph_gen(graph_info, i)
            for G in graphs:
                G_list.append(G)
            # print(G_list[0].nodes)
        plot(G_list, 'second_order_groups.png')
        return "Successfully generated"

    def load_graphs(self, path: str):
        with open(path, 'r') as f:
            return np.load(f)

    def load_adj(self):
        data_dir = os.path.realpath(os.path.join(
            os.path.dirname(__file__),  'data', 'valence.xlsx'))
        return pd.read_excel(data_dir)


class QueryGraph:
    """Representation of query graph used during GNN training"""

    def __init__(self, glist: List[nx.Graph] == None):
        self.list = []

    def add(self, glist):
        self.list.append(glist)


df = pd.read_excel("data/valence.xlsx")
NUM_SAMPLES = 50000

smarts = []

@ dataclass
class SubstructMatch:
    """Representation of a matched substructure during fragmentation.

    Attributes:
        smarts: the corresponding SMART
        num_atoms: number of atoms in the substructure
        atom_idxs: indices of the substructure's atoms in the orinal rdkit mol graph
        group_num: MG functional group number of the substructure.
    """
    raw_smarts: str
    smarts: str
    num_atoms: int
    atom_idxs: list = field(default_factory=list)
    group_num: int = 0


@ dataclass
class Fragmentation:
    """Representation of a Marrero-Gani fragmentation of a molecule.
    """

    matches: List[SubstructMatch]
    groups_idx: List[int] = field(init=False)
    vec: np.ndarray = field(init=False)
    all_matches: List[SubstructMatch] = None

    # TODO: refactor
    def __post_init__(self):
        cnts = {}
        for group in self.matches:
            cnts[group.group_num] = cnts.get(group.group_num, 0) + 1
        aug_cnts = []
        for k, v in cnts.items():
            aug_cnts.append((k, v))
        cnts = sorted(aug_cnts)
        counts = [c[1] for c in cnts]
        idxs = [(id[0] - 1) for id in cnts]
        self.vec = np.zeros((220))
        self.vec[np.array(idxs)] = counts
        self.groups_idx = [match.group_num for match in self.matches]

    def has(self, idx: int) -> bool:
        return self.vec[idx-1] != 0

    def counts(self):
        return vec_to_plaintext(self.vec)

    @ property
    def num_groups(self):
        return len(self.matches)

    def __repr__(self):
        msg = ""
        msg += f"{'Counts:': <17}\n"
        for g in self.counts():
            msg += f"{'': <8}{g} \n"
        # msg += f"{'All matches:': <17}{self.matches}"
        return msg


def find_matches(smarts: List[GroupSMARTS],
                 mol: rdkit.Chem.rdchem.Mol) -> List[SubstructMatch]:
    """Finds all matches of each SMART in a given list of SMARTS.

    Args:
        smarts: the SMARTS substructures to be matched.
        mol: the target molecule

    Returns:
        List[SubstructMatch]: list of matched substructures with metadata.
    """
    matches = []
    for groupSMART in smarts:
        safe_match(groupSMART, mol, matches)
    return matches


def safe_match(groupSMART: GroupSMARTS, mol, matches: List[SubstructMatch]) -> None:
    for q in groupSMART.smart_str[0]:
        if q != '[placeholder]':
            item, num_atoms = rdkit_match(q, mol)
            for idx_set in item:
                append_match(groupSMART.raw_smarts, q, num_atoms,
                             idx_set, groupSMART.group_num, matches)


def append_match(raw: str, q: str, num_atoms: int, idxs: List[int], group_num: int,
                 matches: List[SubstructMatch]) -> None:
    matches.append(SubstructMatch(raw, q, num_atoms, idxs, group_num))


def rdkit_match(q: str, mol: rdkit.Chem.rdchem.Mol):
    item = mol.GetSubstructMatches(Chem.MolFromSmarts(q))
    num_atoms = 0 if len(item) == 0 else len(item[0])
    return item, num_atoms


def find_fragmentation(mol: rdkit.Chem.rdchem.Mol, smart_cands: List[GroupSMARTS]) -> Fragmentation:
    """
    Args:
        mol: rdkit molecule
        smart_cands: subset of SMARTS used for substructure search

    Returns: A list of matched substructures forming a valid fragmentation
        as per the MG-method.
    """
    heap = []
    for i, smart in enumerate(smart_cands):
        heapq.heappush(heap,  (smart.weight, smart.group_num, smart))

    smarts_sorted = [s[-1] for s in sorted(heap, reverse=True)]
    # pprint.pprint(smarts_sorted)
    results = []
    num_atoms_in_mol = mol.GetNumAtoms()
    # print(num_atoms_in_mol)
    # pprint.pprint(smarts_sorted)
    matches = find_matches(smarts_sorted, mol)
    # pprint.pprint(matches)
    # pprint.pprint("Matches: {}".format([m.raw_smarts for m in matches]))
    for i in range(len(matches) + 1):
        results += combinations(matches, i)
    # pprint.pprint(results)
    seen = set()
    covers = []
    size = 999999
    min_cover = None

    for i, comb in enumerate(results):
        # pprint.pprint(comb)
        # print(total_match_atoms(comb))
        seen = set()
        terminate = None
        for match in comb:
            # print('Loop', i, seen)
            if has_overlaps(match, seen):
                terminate = True
        # print(total_match_atoms(comb))
        if total_match_atoms(comb) == num_atoms_in_mol and not terminate:
            covers.append(comb)
            # print(len(comb))
            if len(comb) <= size:
                min_cover = comb
                size = len(min_cover)
    # for cover in covers:
    #     pprint.pprint([match.group_num for match in cover])

    if min_cover is not None:
        return Fragmentation(min_cover)
    else:
        return None


def has_overlaps(match: SubstructMatch, seen: Set[SubstructMatch]) -> bool:
    for atom_idx in match.atom_idxs:
        # print(atom_idx, seen)
        if atom_idx not in seen:
            seen.add(atom_idx)
        else:
            return True

    return False


def total_match_atoms(matches: List[SubstructMatch]) -> int:
    return sum([match.num_atoms for match in matches])

    # def smile_to_func_graph(smile, substructs):
    #     mol = get_rdmol(smile)
    #     indices = find_fragmentation(mol, substructs, substructs)
    #     # print(indices)
    #     fg_dict = {}
    #     # Map from atom idx to bucket (functional groups)
    #     for i in range(len(indices)):
    #         for el in indices[i][0]:
    #             fg_dict[el] = i

    #     # Get edges between the connected components
    #     adj_list = []
    #     for group in indices:
    #         out = set()
    #     for el in group[0]:
    #         bonds = mol.GetAtomWithIdx(el).GetBonds()
    #         for b in bonds:
    #             dst = fg_dict[b.GetOtherAtomIdx(el)]
    #             src = fg_dict[el]
    #         # Avoid adding duplicates
    #         if (dst != src) and ((len(adj_list) <= dst) or (not (src in adj_list[dst]))):
    #             out.add(dst)
    #     adj_list.append(out)

    #     G = nx.Graph()
    #     # print(adj_list)

    #     # Node features are the group number, group class (aromatic, urea/amide, or standard), va, vb, vc,
    #     # and total valency
    #     for i in range(len(indices)):
    #     group_num = indices[i][1]
    #         feature_vec = torch.tensor(
    #             [group_num, group_class_map[util.group_class[group_num]], util.valency[group_num]]).float()
    #         print(feature_vec)
    #         G.add_node(i, node_feature=feature_vec)

    #       for i in range(len(indices)):
    #         for j in adj_list[i]:
    #           G.add_edge(i, j)

    #       results.append(G)
    #       # nf = nx.get_node_attributes(G, 'node_feature')
    #       # nx.draw(G, labels=nf)
    #       # plt.show(G)


def get_rdmol(smile: str) -> rdkit.Chem.rdchem.Mol:
    """Returns an RDKit Mol object given a smile string.

    Args:
        smile: _description_

    Returns:
        rdkit.Chem.rdchem.Mol: _description_
    """
    smile = smile.strip()
    mol = Chem.MolFromSmiles(smile)
    Chem.SanitizeMol(mol)

    return mol


def G_add_edge(G: nx.Graph, bond: rdkit.Chem.rdchem.Bond) -> None:
    G.add_edge(bond.GetBeginAtomIdx(),
               bond.GetEndAtomIdx(),
               bond_type=bond.GetBondType())


def G_set_group_prop(atom: rdkit.Chem.rdchem.Atom) -> str:
    group = None
    try:
        group = atom.GetProp('group')
    except:
        group = 'None'
    return group


def G_add_node(G: nx.Graph, atom: rdkit.Chem.rdchem.Atom, group: str) -> None:
    G.add_node(atom.GetIdx(),
               atomic_num=atom.GetAtomicNum(),
               group=group,
               formal_charge=atom.GetFormalCharge(),
               chiral_tag=atom.GetChiralTag(),
               hybridization=atom.GetHybridization(),
               is_aromatic=atom.GetIsAromatic())


def mol_to_nx(mol: rdkit.Chem.rdchem.Mol) -> nx.Graph:
    G = nx.Graph()

    for atom in mol.GetAtoms():
        group = G_set_group_prop(atom)
        G_add_node(G, atom, group)
    for bond in mol.GetBonds():
        G_add_edge(G, bond)
    return G


def frag_to_nx_graph(fragmentation: Fragmentation,
                     mol: rdkit.Chem.rdchem.Mol) -> nx.Graph:
    fg_dict = {}
    dataset = DataSet.instance()

    # Map from atom idx to functional group
    for substruct_idx, substruct in enumerate(fragmentation.matches):
        for atom_idx in substruct.atom_idxs:
            fg_dict[atom_idx] = substruct_idx

    # print(fg_dict)

    # Get edges between the connected components
    adj_list = []
    for group in fragmentation.matches:
        edges_out = set()
        for atom_idx in group.atom_idxs:
            bonds = mol.GetAtomWithIdx(atom_idx).GetBonds()
            for b in bonds:
                dst = fg_dict[b.GetOtherAtomIdx(atom_idx)]
                src = fg_dict[atom_idx]
                # Avoid adding duplicates
                if (dst != src) and ((len(adj_list) <= dst) or (not (src in adj_list[dst]))):
                    edges_out.add(dst)
        adj_list.append(edges_out)

    G = nx.Graph()
    # print(adj_list)

    total_valence, va, vb, vc = dataset.valences
    # Node features are the group number, total valency, va, vb, vc,
    # and total valency
    for i, group in enumerate(fragmentation.matches):
        idx = group.group_num - 1
        G.add_node(i, group_num=group.group_num,
                   total_valence=total_valence[idx],
                   va=va[idx],
                   vb=vb[idx],
                   vc=vc[idx])

    for i in range(fragmentation.num_groups):
        for j in adj_list[i]:
            G.add_edge(i, j)

    nf = nx.get_node_attributes(G, 'group_num')
    nx.draw(G, labels=nf)
    # plt.show()

    return G


def frag_to_pyg_graph():
    pass


def load_graph_info(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Imports the combinations of first order groups and edge information that makes the corresponding
    sets of second order groups.

    Returns: tuple of pandas dataframes `(nodes, edges)` with the relevant information.
    """

    mask = df["1st order set"] != 'N/A'
    df.where(mask, inplace=True)
    second_order_nodes = df.iloc[220:350]["1st order set"].str.extractall(
        r'(?:\()(([A-Za-z0-9]*,)+[A-Za-z0-9]*)+(?:\))')
    second_order_edges = df.iloc[220:350]["single bond edges"].str.extractall(
        r'(?:\()([A-Za-z0-9]*(?:,)[A-Za-z0-9]*)+(?:\))')

    second_order_nodes = second_order_nodes[0].str.split(',')
    second_order_edges = second_order_edges[0].str.split(',')

    return second_order_nodes, second_order_edges


def second_order_graph_gen(graph_info, second_order_group):
    """Generates all possible fully connected graphs representing a given second order functional group,
    with connections between first order functional groups.

    Arg:
        second_order_group: The index of the second order group in the dataset.

    Returns: a generated list of fully connected graphs

    """
    graphs = []
    # not sure why indexing is off by 1
    second_order_group -= 1
    try:
        nodesets = graph_info[0][second_order_group]
        edges = graph_info[1][second_order_group]
    except KeyError as e:
        return [nx.Graph(group=9999)]
    for ns in nodesets:
        G = nx.Graph(group=second_order_group)
        for i in range(0, len(ns)):
            # Dataset nodes are indexed by 1, hence i+1
            G.add_node(
                i+1, group=int(ns[i]))
        for j in edges:
            G.add_edge(int(j[0]), int(j[1]))
        graphs.append(G)
    # TODO

    return graphs


def plot(G_list: List[nx.Graph], filename: str) -> None:
    """Helper function for plotting generated 2nd order query graphs.

    Args:
        G_list: list of graphs corresponding to the second order queries
        str: filename of output image
    """
    fig, axes = plt.subplots(len(G_list) // 5 + 1, 5,
                             figsize=(20, 2*(len(G_list) // 5 + 1)))
    ax = axes.flat
    plt.tight_layout()
    for a in ax:
        a.margins(0.50)
    for i, graph in zip(range(0, len(G_list)), G_list):
        labels = nx.get_node_attributes(graph, 'group')
        ax[i].set_title(graph.graph['group'])
        nx.draw_networkx(
            graph, ax=ax[i], labels=labels)

    plt.savefig('fig/' + filename)


def is_isomorphic(G1, G2, appr='vf2'):
    if (appr == 'vf2'):
        return isomorphism.GraphMatcher(G1, G2)
    elif (appr == 'nn'):
        return None


def generate_gnn_training_data():
    graphDataLoader_obj = GraphDataLoader(
        "test/csv/restricted_groups_test_small.csv", num_examples=5)
    smiles, smarts, labels = graphDataLoader_obj.data_np
    results = []
    for i, smile in enumerate(smiles):
        mol = get_rdmol(smile)
        frag = find_fragmentation(mol, smarts[i])
        if frag is not None:
            results.append(frag_to_nx_graph(frag, mol))

    pickle.dump(results, open('gnn/train/train_data.pickle', 'wb'))

# VISUALIZATION
