
from dataclasses import dataclass, field
import torch
import networkx as nx


@ dataclass
class Features:
    """Represents the node/edge/graph features to be used during GNN
    training and prediction.

    Attributes:
        G: The graph for which the features are computed.
        laplacian: Normalized graph laplacian.
        num_nodes: The total number of nodes in the graph.
        degree: List of node degrees in the graph.
        centrality: List of betweenness centrality for each node.
        pagerank: The pagerank of the graph.
        clustering_coefficient: The clustering coefficient of the graph.
    """
    G: nx.Graph
    # laplacian: np.ndarray = field(init=False)
    use_graph_stats: bool = False
    num_nodes: torch.Tensor = field(init=False)
    degree: torch.Tensor = field(init=False)
    centrality: torch.Tensor = field(init=False)
    # path: torch.Tensor = field(init=False)
    pagerank: torch.Tensor = field(init=False)
    clustering_coefficient: torch.Tensor = field(init=False)
    valency: torch.Tensor = field(init=False)
    group_num: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.use_graph_stats:
            self.update_graph_stats()
        self.init_node_features()
        self.init_edge_features()

    def update_graph_stats(self):
        self.laplacian = nx.normalized_laplacian_matrix(
            self.G, weight='valency')
        self.num_nodes = self.G.number_of_nodes()
        self.degree = (torch.Tensor(
            [d for _, d in self.G.degree()])/self.G.number_of_nodes())
        self.centrality = torch.Tensor(
            list(nx.betweenness_centrality(self.G).values()))
        self.pagerank = torch.Tensor(list(nx.pagerank_numpy(self.G).values()))
        self.clustering_coefficient = torch.Tensor(
            list(nx.clustering(self.G)))

    def init_node_features(self):
        valency = list(self.G.nodes(data="valency"))

    def init_edge_features(self):
        pass

    def to_feature_matrix(self):
        pass

    def to_tensor(self, node):
        return torch.Tensor()

    def __repr__(self):
        return f"num_nodes: {self.num_nodes}\n" + \
            f"graph degree: {self.degree}\n" + \
            f"centrality coefficient: {self.centrality}\n" + \
            f"clustering coefficient: {self.clustering_coefficient}\n" + \
            f"pagerank: {self.pagerank}"
