"""Defines all graph embedding models"""
from util.preprocessing import DiskDataSource
from tqdm import tqdm

import datetime
import pickle
import random
from collections import defaultdict
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch_geometric import utils as pyg_utils
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent) + "gnn/")


search_space = {
    "lr": tune.grid_search([1e-4]),
    "batch_size": tune.grid_search([64]),
    "hidden_dim": tune.grid_search([64, 128]),
    # "dropout": tune.grid_search([0, 0.1, 0.3]),
    "n_layers": tune.grid_search([8, 12]),
    # "skip_layers": tune.grid_search([True, False]),
    # "num_train_samples": tune.grid_search([150, 300, 450, 600, 750]),
    "eval_interval": tune.grid_search([200]),
    "num_batches": tune.grid_search([1000]),
}

# MWE
# search_space = {
#     "lr": tune.grid_search([1e-4]),
#     "batch_size": tune.grid_search([64]),
#     "hidden_dim": tune.grid_search([64, 128]),
#     # "dropout": tune.grid_search([0, 0.1, 0.3]),
#     "n_layers": tune.grid_search([8, 12]),
#     # "skip_layers": tune.grid_search([True, False]),
#     # "num_train_samples": tune.grid_search([150, 300, 450, 600, 750]),
#     "eval_interval": tune.grid_search([10]),
#     "num_batches": tune.grid_search([100]),
# }


class OrderEmbedder(nn.Module):
    """Implementation of the Order Embedding model for subgraph matching,
    as in NeuroMatch (2020)
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(
            input_dim, hidden_dim, hidden_dim, n_layers)
        self.margin = 0.1
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred: Tuple[List]) -> List:
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.
        pred: list (emb_as, emb_bs) of embeddings of graph pairs
        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
                                                 device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, pred, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0);
        for negative examples, the e term is trained to be at least greater than self.margin.
        pred: lists of embeddings outputted by forward
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
                                                 device=device()), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0,
                                                device=device()), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss


# def valency_fun(graph, feature_dim):
#     for v in graph.G.nodes:
#         valency = torch.cat(
#             [graph.total_valence, graph.va, graph.vb, graph.vc], -1)
#         graph.G.nodes[v]["node_feature"] = torch.cat(
#             (graph.G.nodes[v], valency), 0)
#     return graph


# def group_fun(graph, feature_dim):
#     for v in graph.G.nodes:
#         graph.G.nodes[v]["node_feature"] = torch.cat(
#             (graph.G.nodes[v]), v.group_num)
#     return graph


# def add_features(batch):
#     batch.apply_transform(valency_fun, feature_dim=4)
#     batch.apply_transform(group_fun, feature_dim=1)


class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(SkipLastGNN, self).__init__()
        self.dropout = 0.0
        self.n_layers = n_layers
        feature_dim = 10  # group_num, total_valence, va, vb, vc

        input_dim = 1
        preprocessed_dim = input_dim + feature_dim

        self.pre_mp = nn.Sequential(nn.Linear(preprocessed_dim, hidden_dim))

        conv_model = SAGEConv
        self.convs = nn.ModuleList()

        self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                                                      self.n_layers))

        for l in range(n_layers):
            hidden_input_dim = hidden_dim * (l + 1)
            self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (n_layers + 1)
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        self.skip = "learnable"
        self.conv_type = 'SAGE'

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                                                :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                x = self.convs[i](curr_emb, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        # print(emb)

        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
                                    out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GINConv(pyg_nn.MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
                                                              edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
                                                          edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


def device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device


def train(model, config):
    """Train the order embedding model.
    model: OrderEmbedding model.
    logger: logger for logging progress

    emb_pos_a is a list of whole graph embeddings for the positive examples,
    emb_pos_b is a list of subgraph embeddings for the positive examples
    similar rationale for emb_neg_a, emb_neg_b
    """
    opt = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)
    clf_opt = optim.Adam(model.clf_model.parameters(), lr=config["lr"])

    f = pickle.load(open(Path(__file__).parent.parent /
                    "gnn" / "train" / "train_data.pickle", 'rb'))
    data_source = DiskDataSource(f)
    loaders = data_source.gen_data_loaders(config["batch_size"] *
                                           config["eval_interval"], config["batch_size"], train=True)
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        # train
        model.train()
        model.zero_grad()

        # generate positive examples, with positive subgraphs, and negatt
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                                                           batch_neg_target, batch_neg_query, True)
        emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
        emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)
        # print(emb_pos_a.shape, emb_neg_a.shape, emb_neg_b.shape)
        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)

        # label vector of batch size
        labels = torch.tensor([1]*pos_a.num_graphs + [0]
                              * neg_a.num_graphs).to(device())

        pred = model(emb_as, emb_bs)
        loss = model.criterion(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            pred = model.predict(pred)
        model.clf_model.zero_grad()
        pred = model.clf_model(pred.unsqueeze(1))
        criterion = nn.NLLLoss()
        clf_loss = criterion(pred, labels)
        clf_loss.backward()
        clf_opt.step()
        pred = pred.argmax(dim=-1)
        acc = torch.mean((pred == labels).type(torch.float))
    return loss.item(), acc


def validation(config, model, test_pts, epoch, batch_n, verbose=True):
    # test on new motifs
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    for pos_a, pos_b, neg_a, neg_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(device())
            pos_b = pos_b.to(device())
        neg_a = neg_a.to(device())
        neg_b = neg_b.to(device())
        labels = torch.tensor([1]*(pos_a.num_graphs if pos_a else 0) +
                              [0]*neg_a.num_graphs).to(device())
        with torch.no_grad():
            emb_neg_a, emb_neg_b = (model.emb_model(neg_a),
                                    model.emb_model(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (model.emb_model(pos_a),
                                        model.emb_model(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b
            pred = model(emb_as, emb_bs)
            raw_pred = model.predict(pred)

            pred = model.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
            raw_pred *= -1
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
            torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
              torch.sum(labels).item() if torch.sum(labels) > 0 else
              float("NaN"))

    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    auroc = roc_auc_score(labels, raw_pred)
    avg_prec = average_precision_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    return acc, prec, recall, auroc, \
        avg_prec, tp, tn, fp, fn
    # torch.save(model.state_dict(), 'gnn/model/')

    if verbose:
        conf_mat_examples = defaultdict(list)
        idx = 0
        for pos_a, pos_b, neg_a, neg_b in test_pts:
            if pos_a:
                pos_a = pos_a.to(device())
                pos_b = pos_b.to(device())
            neg_a = neg_a.to(device())
            neg_b = neg_b.to(device())
            for list_a, list_b in [(pos_a, pos_b), (neg_a, neg_b)]:
                if not list_a:
                    continue
                for a, b in zip(list_a.G, list_b.G):
                    correct = pred[idx] == labels[idx]
                    conf_mat_examples[correct, pred[idx]].append((a, b))
                    idx += 1


def train_loop(config):
    batch_n = 0
    eval_interval = config["eval_interval"]
    input_dim = 1
    model = OrderEmbedder(
        input_dim, config["hidden_dim"], config["n_layers"]).to(device())

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device())
    test_pts = []
    f = pickle.load(open(Path(__file__).parent.parent /
                    "gnn" / "train" / "train_data.pickle", 'rb'))
    data_source = DiskDataSource(f)
    loaders = data_source.gen_data_loaders(
        config["eval_interval"] * config["batch_size"], config["batch_size"], train=True)
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                                                           batch_neg_target, batch_neg_query, False)
        if pos_a:
            pos_a = pos_a.to(torch.device(device()))
            pos_b = pos_b.to(torch.device(device()))
        neg_a = neg_a.to(torch.device(device()))
        neg_b = neg_b.to(torch.device(device()))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))
    for epoch in range(config["num_batches"] // config["eval_interval"]):
        train_losses = []
        train_accs = []
        for i in range(config["eval_interval"]):
            params = train(model, config)
            train_loss, train_acc = params
            # print(batch_n)
            # print("Batch {}. Loss: {:.4f}. Training acc: {:.4f}".format(
            #     batch_n, train_loss, train_acc))
            batch_n += 1
            train_losses.append(train_loss)
            train_accs.append(train_accs)
        epoch_train_loss = (torch.sum(train_losses) /
                            train_losses.size[0]).item()
        epoch_train_acc = (torch.sum(train_accs)/train_accs.size[0]).item()
        val_acc, prec, recall, auroc, \
            avg_prec, tp, tn, fp, fn = validation(
                config, model, test_pts, epoch, batch_n)
        tune.report(epoch=epoch, train_loss=epoch_train_loss, train_acc=epoch_train_acc,
                    val_acc=val_acc, prec=prec, recall=recall, auroc=auroc, avg_prec=avg_prec, tp=tp, tn=tn, fp=fp, fn=fn)


def train_gnn():
    analysis = tune.run(train_loop, name='train_gnn_hp',
                        resources_per_trial={"cpu": 6, "gpu": 0.25},
                        config=search_space,
                        checkpoint_freq=2,
                        log_to_file=True,
                        scheduler=ASHAScheduler(
                            metric="val_acc", mode="max"),
                        checkpoint_at_end=True)
    print(analysis.get_best_config(metric='val_acc', mode='max'))


if __name__ == '__main__':
    train_gnn()
