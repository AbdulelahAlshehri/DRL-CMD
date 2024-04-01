import pytest
from gnn.gnn import OrderEmbedder, train
from graph import GraphDataLoader, GraphQueryDataLoader, find_fragmentation, frag_to_nx_graph, get_rdmol
from conftest import _slim, _slim_reason
from unittest.mock import Mock
import pickle
import numpy as np
import torch
from deepsnap.graph import Graph as DSGraph
from util.preprocessing import DiskDataSource


@pytest.fixture
def graphDataLoader_obj():
    return GraphDataLoader("test/csv/restricted_groups_test.csv", num_examples=800)


def test_GraphQueryDataLoader():
    pass

# @pytest.mark.skipif(_slim, reason=_slim_reason)
# @pytest.mark.skip(reason="omit for speed")
# def test_GraphQueryDataLoader_generate(graphQueryDataLoader_obj):
#     print(graphQueryDataLoader_obj.generate())


# def test_GraphSampler_sample():
#     pass

# @pytest.mark.parametrize
def test_GraphDataLoader(graphDataLoader_obj):
    print(graphDataLoader_obj.data_np)


def test_generate_gnn_training_graphs(graphDataLoader_obj):
    smiles, smarts, labels = graphDataLoader_obj.data_np
    results = []
    for i, smile in enumerate(smiles):
        mol = get_rdmol(smile)
        frag = find_fragmentation(mol, smarts[i])
        # print(frag)
        print(i)
        if frag is not None and np.array_equal(frag.vec, labels[i]):
            results.append(frag_to_nx_graph(frag, mol))

    pickle.dump(results, open('gnn/train/train_data_nx.pickle', 'wb'))


def test_batching_gnn():
    f = pickle.load(open('gnn/train/train_data.pickle', 'rb'))
    data_source = DiskDataSource(f)
    loaders = data_source.gen_data_loaders(4096, 64,
                                           train=False, use_distributed_sampling=False)
    # print(loaders)
    test_pts = []
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                                                           batch_neg_target, batch_neg_query, False)

        if pos_a:
            pos_a = pos_a.to(torch.device("cpu"))
            pos_b = pos_b.to(torch.device("cpu"))
        neg_a = neg_a.to(torch.device("cpu"))
        neg_b = neg_b.to(torch.device("cpu"))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))
    # print(test_pts)


def test_emb_model():
    f = pickle.load(open('gnn/train/train_data.pickle', 'rb'))
    data_source = DiskDataSource(f)
    loaders = data_source.gen_data_loaders(4096, 64,
                                           train=False, use_distributed_sampling=False)
    # print(loaders)
    test_pts = []
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                                                           batch_neg_target, batch_neg_query, False)

        if pos_a:
            pos_a = pos_a.to(torch.device("cpu"))
            pos_b = pos_b.to(torch.device("cpu"))
        neg_a = neg_a.to(torch.device("cpu"))
        neg_b = neg_b.to(torch.device("cpu"))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))
    emb = OrderEmbedder(1, 64)
    train(emb)
