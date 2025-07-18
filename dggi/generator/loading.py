import os
from collections.abc import MutableSequence

import torch
import numpy as np
import networkx as nx
import graph_tool as gt

from dggi.utils import ProgressBar
from dggi.generator.graph_utils import from_gt_to_nx
from dggi.generator.preprocessing import bfs_seq, encode_adj


def split_data(graphs, rng, with_val=False, inplace=False, progress_bar_qt=None):
    num_graphs = len(graphs)
    graph_index = np.arange(num_graphs)
    test_size = int(0.15 * num_graphs)
    if with_val:
        val_size = int(0.15 * num_graphs)
    else:
        val_size = 0
    rng.shuffle(graph_index)
    train_graphs = Graphs(
        list=[graphs._list[i] for i in graph_index[(test_size + val_size) :]],
        from_path=False,
        inplace=inplace,
        ext="gt",
        progress_bar_qt=progress_bar_qt,
    )
    test_graphs = Graphs(
        list=[graphs._list[i] for i in graph_index[0:test_size]],
        from_path=False,
        inplace=inplace,
        ext="gt",
        progress_bar_qt=progress_bar_qt,
    )
    val_graphs = Graphs(
        list=[graphs._list[i] for i in graph_index[test_size : (val_size + test_size)]],
        from_path=False,
        inplace=inplace,
        ext="gt",
        progress_bar_qt=progress_bar_qt,
    )
    return train_graphs, val_graphs, test_graphs


# max_prev_node: Max previous node that looks back (if none, automatically defined)
def create(
    source_path,
    data_size,
    min_num_nodes,
    max_num_nodes,
    check_size,
    progress_bar_qt=None,
):
    graphs = []
    graphs = Graphs(
        source_path,
        data_size,
        min_num_nodes,
        max_num_nodes,
        check_size=check_size,
        progress_bar_qt=progress_bar_qt,
    )
    # max_prev_node = 246  # Use None for compute estimation
    return graphs


class Graphs(MutableSequence):
    def __init__(
        self,
        source_path=None,
        data_size=None,
        min_num_node=None,
        max_num_node=None,
        list=None,
        from_path=True,
        check_size=True,
        inplace=False,
        ext="gt",
        progress_bar_qt=None,
    ):
        super(Graphs, self).__init__()
        self._list = []
        self._inplace = inplace
        if from_path:
            if max_num_node is None and min_num_node is not None:
                max_num_node = 1000
            elif max_num_node is not None and min_num_node is None:
                min_num_node = 0
            graph_names = os.listdir(source_path)
            bar = ProgressBar(
                total=len(graph_names),
                desc="Loading data graphs",
                progress_bar_qt=progress_bar_qt,
            )
            for graph_name in graph_names:
                if graph_name.endswith(f".{ext}"):
                    graph_path = os.path.join(source_path, graph_name)
                    if min_num_node is None and max_num_node is None:
                        self._list.append(graph_path)
                    elif check_size:
                        G = gt.load_graph(graph_path)
                        num_nodes = G.num_vertices()
                        if num_nodes >= min_num_node and num_nodes <= max_num_node:
                            self._list.append(graph_path)
                    else:
                        self._list.append(graph_path)
                bar.update()
                if progress_bar_qt is not None and progress_bar_qt.stop_running:
                    break
            bar.close()
            self._list = self._list[:data_size]
            print(f"{data_size} graphs are being considered...")
        else:
            self._list = list if list is not None else []
        if self._inplace:
            self._graphs = []
            bar = ProgressBar(
                len(self._list),
                desc="Adding data graphs to memory",
                progress_bar_qt=progress_bar_qt,
            )
            for graph_path in self._list:
                G = from_gt_to_nx(gt.load_graph(graph_path))
                self._graphs.append(G)
                bar.update()
                if progress_bar_qt is not None and progress_bar_qt.stop_running:
                    break
            bar.close()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, ii):
        if self._inplace:
            G = self._graphs[ii]
        else:
            G = from_gt_to_nx(gt.load_graph(self._list[ii]))
        return G

    def __str__(self):
        return str(self._list)

    def __delitem__(self, ii):
        del self._list[ii]

    def __setitem__(self, ii, val):
        self._list[ii] = val

    def insert(self, ii, val):
        self._list.insert(ii, val)

    def append(self, val):
        self.insert(len(self._list), val)

    def change_root(self, root_path):
        list = []
        for graph_path in self._list:
            graph_name = os.path.basename(graph_path)
            list.append(os.path.join(root_path, graph_name))
        self._list = list


class GraphSequenceSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node, max_prev_node):
        self.G_list = G_list
        self.max_num_node = max_num_node
        self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.G_list)

    def __getitem__(self, idx):
        adj = np.asarray(nx.to_numpy_matrix(self.G_list[idx]))
        x_batch = np.zeros(
            (self.max_num_node, self.max_prev_node)
        )  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        y_batch = np.zeros(
            (self.max_num_node, self.max_prev_node)
        )  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj.shape[0]
        x_idx = np.random.permutation(adj.shape[0])
        adj = adj[np.ix_(x_idx, x_idx)]
        adj = np.asmatrix(adj)
        G = nx.from_numpy_matrix(adj)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj = adj[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj.copy(), max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0 : adj_encoded.shape[0], :] = adj_encoded
        x_batch[1 : adj_encoded.shape[0] + 1, :] = adj_encoded
        return {"x": x_batch, "y": y_batch, "len": len_batch}
