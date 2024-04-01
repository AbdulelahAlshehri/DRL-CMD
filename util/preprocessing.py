import random
from torch_geometric import utils as pyg_utils
import torch
import torch.nn as nn
from torch_scatter import scatter_add
import networkx as nx
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
from typing import Tuple, List


def device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device


AUGMENT_METHOD = "concat"
FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = [], []


class DiskDataSource:
    """ Uses a set of graphs saved in a dataset file to train the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.

    See the load_dataset function for supported datasets.
    """

    def __init__(self, dataset_name, node_anchored=True, min_size=5,
                 max_size=15):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
                         use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=15, min_size=5, seed=None,
                  filter_negs=False, sample_method="tree-pair") -> Tuple:
        batch_size = a
        train_set, test_set = self.dataset
        graphs = train_set if train else test_set
        if seed is not None:
            random.seed(seed)

        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size // 2):
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph, a = sample_neigh(graphs, size)
                b = a[:random.randint(min_size, len(a) - 1)]
            elif sample_method == "subgraph-tree":
                graph = None
                while graph is None or len(graph) < min_size + 1:
                    graph = random.choice(graphs)
                a = graph.nodes
                _, b = sample_neigh([graph], random.randint(min_size,
                                                            len(graph) - 1))
            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)
            neigh_a, neigh_b = graph.subgraph(
                a), graph.subgraph(b)
            pos_a.append(neigh_a)
            pos_b.append(neigh_b)

        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph_a, a = sample_neigh(graphs, size)
                graph_b, b = sample_neigh(graphs, random.randint(min_size,
                                                                 size - 1))
            elif sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < min_size + 1:
                    graph_a = random.choice(graphs)
                a = graph_a.nodes
                graph_b, b = sample_neigh(graphs, random.randint(min_size,
                                                                 len(graph_a) - 1))
            if self.node_anchored:
                neg_a_anchors.append(list(graph_a.nodes)[0])
                neg_b_anchors.append(list(graph_b.nodes)[0])
            neigh_a, neigh_b = graph_a.subgraph(
                a), graph_b.subgraph(b)
            if filter_negs:
                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    neigh_a, neigh_b)
                if matcher.subgraph_is_isomorphic():  # a <= b (b is subgraph of a)
                    continue
            neg_a.append(neigh_a)
            neg_b.append(neigh_b)

        pos_a = batch_nx_graphs(pos_a, anchors=pos_a_anchors if
                                self.node_anchored else None)
        pos_b = batch_nx_graphs(pos_b, anchors=pos_b_anchors if
                                self.node_anchored else None)
        neg_a = batch_nx_graphs(neg_a, anchors=neg_a_anchors if
                                self.node_anchored else None)
        neg_b = batch_nx_graphs(neg_b, anchors=neg_b_anchors if
                                self.node_anchored else None)
        return pos_a, pos_b, neg_a, neg_b


def batch_nx_graphs(graphs: List[nx.Graph], anchors: List[int] = None) -> Batch:

    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                if (g.nodes[v].get('node_feature').size()[0] != 6):
                    g.nodes[v]["node_feature"] = torch.cat(
                        (torch.tensor([float(v == anchor)]),
                         g.nodes[v]["node_feature"]), 0)

    # print(graphs)
    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = batch.to(device())
    return batch


def compute_identity(edge_index, n, k):
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float,
                             device=edge_index.device)
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
        edge_index, edge_weight, 1, n)
    adj_sparse = torch.sparse.FloatTensor(edge_index, edge_weight,
                                          torch.Size([n, n]))
    adj = adj_sparse.to_dense()

    deg = torch.diag(torch.sum(adj, -1))
    deg_inv_sqrt = deg.pow(-0.5)
    adj = deg_inv_sqrt @ adj @ deg_inv_sqrt

    diag_all = [torch.diag(adj)]
    adj_power = adj
    for i in range(1, k):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all


class FeatureAugment(nn.Module):
    def __init__(self):
        super(FeatureAugment, self).__init__()

    #     def degree_fun(graph, feature_dim):
    #         graph.node_degree = self._one_hot_tensor(
    #             [d for _, d in graph.G.degree()],
    #             one_hot_dim=feature_dim)
    #         return graph

    #     def centrality_fun(graph, feature_dim):
    #         nodes = list(graph.G.nodes)
    #         centrality = nx.betweenness_centrality(graph.G)
    #         graph.betweenness_centrality = torch.tensor(
    #             [centrality[x] for x in
    #              nodes]).unsqueeze(1)
    #         return graph

    #     def path_len_fun(graph, feature_dim):
    #         nodes = list(graph.G.nodes)
    #         graph.path_len = self._one_hot_tensor(
    #             [np.mean(list(nx.shortest_path_length(graph.G,
    #                                                   source=x).values())) for x in nodes],
    #             one_hot_dim=feature_dim)
    #         return graph

    #     def pagerank_fun(graph, feature_dim):
    #         nodes = list(graph.G.nodes)
    #         pagerank = nx.pagerank(graph.G)
    #         graph.pagerank = torch.tensor([pagerank[x] for x in
    #                                        nodes]).unsqueeze(1)
    #         return graph

    #     def identity_fun(graph, feature_dim):
    #         graph.identity = compute_identity(
    #             graph.edge_index, graph.num_nodes, feature_dim)
    #         return graph

    #     def clustering_coefficient_fun(graph, feature_dim):
    #         node_cc = list(nx.clustering(graph.G).values())
    #         if feature_dim == 1:
    #             graph.node_clustering_coefficient = torch.tensor(
    #                 node_cc, dtype=torch.float).unsqueeze(1)
    #         else:
    #             graph.node_clustering_coefficient = FeatureAugment._bin_features(
    #                 node_cc, feature_dim=feature_dim)

    #     def node_features_base_fun(graph, feature_dim):
    #         for v in graph.G.nodes:
    #             if "node_feature" not in graph.G.nodes[v]:
    #                 graph.G.nodes[v]["node_feature"] = torch.ones(feature_dim)
    #         return graph

    #     def valency_fun(graph, feature_dim):
    #         for v in graph.G.nodes:
    #             valency = torch.Tensor(
    #                 [graph.total_valence, graph.va, graph.vb, graph.vc])
    #             graph.G.nodes[v]["node_feature"] = torch.cat(
    #                 (graph.G.nodes[v]))
    #         return graph

    #     def group_fun(graph, feature_dim):
    #         for v in graph.G.nodes:
    #             if "node_feature" not in graph.G.nodes[v]:
    #                 graph.G.nodes[v]["node_feature"] = torch.ones(feature_dim)
    #         return graph

    #     self.node_features_base_fun = node_features_base_fun

    #     self.node_feature_funs = {"node_degree": degree_fun,
    #                               "betweenness_centrality": centrality_fun,
    #                               "path_len": path_len_fun,
    #                               "pagerank": pagerank_fun,
    #                               'node_clustering_coefficient': clustering_coefficient_fun,
    #                               #    "valency": valency_fun,
    #                               #   'group_fun': group_fun,
    #                               "identity": identity_fun}

    # @staticmethod
    # def _bin_features(list_scalars, feature_dim=2):
    #     arr = np.array(list_scalars)
    #     min_val, max_val = np.min(arr), np.max(arr)
    #     bins = np.linspace(min_val, max_val, num=feature_dim)
    #     feat = np.digitize(arr, bins) - 1
    #     assert np.min(feat) == 0
    #     assert np.max(feat) == feature_dim - 1
    #     return FeatureAugment._one_hot_tensor(feat, one_hot_dim=feature_dim)

    # @staticmethod
    # def _one_hot_tensor(list_scalars, one_hot_dim=1):
    #     if not isinstance(list_scalars, list) and not list_scalars.ndim == 1:
    #         raise ValueError("input to _one_hot_tensor must be 1-D list")
    #     vals = torch.LongTensor(list_scalars).view(-1, 1)
    #     vals = vals - min(vals)
    #     vals = torch.min(vals, torch.tensor(one_hot_dim - 1))
    #     vals = torch.max(vals, torch.tensor(0))
    #     one_hot = torch.zeros(len(list_scalars), one_hot_dim)
    #     one_hot.scatter_(1, vals, 1.0)
    #     return one_hot

    def augment(self, dataset):
        for key, dim in zip(FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS):
            dataset = dataset.apply_transform(self.node_feature_funs[key],
                                              feature_dim=dim)
        return dataset


class Preprocess(nn.Module):
    def __init__(self, dim_in):
        super(Preprocess, self).__init__()
        self.dim_in = dim_in
        if AUGMENT_METHOD == 'add':
            self.module_dict = {
                key: nn.Linear(aug_dim, dim_in)
                for key, aug_dim in zip(FEATURE_AUGMENT,
                                        FEATURE_AUGMENT_DIMS)
            }

    @ property
    def dim_out(self):
        if AUGMENT_METHOD == 'concat':
            return self.dim_in + sum(
                [aug_dim for aug_dim in FEATURE_AUGMENT_DIMS])
        elif AUGMENT_METHOD == 'add':
            return self.dim_in
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                AUGMENT_METHOD))

    def forward(self, batch):
        if AUGMENT_METHOD == 'concat':
            feature_list = [batch.node_feature]
            for key in FEATURE_AUGMENT:
                feature_list.append(batch[key])
            batch.node_feature = torch.cat(feature_list, dim=-1)
        elif AUGMENT_METHOD == 'add':
            for key in FEATURE_AUGMENT:
                batch.node_feature = batch.node_feature + self.module_dict[key](
                    batch[key])
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                AUGMENT_METHOD))
        return batch


def sample_neigh(graphs: List[nx.Graph], size: int) -> Tuple[nx.Graph, List[int]]:
    ps = np.array([len(g) for g in graphs], dtype=np.float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        # graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            # new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh


cached_masks = None


def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    # v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    # v = [np.sum(v) for mask in cached_masks]
    return v


def load_dataset(dataset):
    """ Load real-world datasets, available in PyTorch Geometric.
    Used as a helper for DiskDataSource.
    """
    train_len = int(0.8 * len(dataset))
    train, test = [], []
    dataset = list(dataset)
    random.shuffle(dataset)
    for i, graph in enumerate(dataset):
        if i < train_len:
            train.append(graph)
        else:
            test.append(graph)
    return train, test
