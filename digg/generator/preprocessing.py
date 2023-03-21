import os

import numpy as np
import networkx as nx
import graph_tool as gt

from digg.utils import ProgressBar
from digg.generator.graph_utils import from_gt_to_nx


def bfs_seq(G, start_id):
    """
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    """
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output


def encode_adj(adj, max_prev_node=10, is_full=False):
    """
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    """
    if is_full:
        max_prev_node = adj.shape[0] - 1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0 : n - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order
    return adj_output


def encode_adj_flexible(adj):
    """
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    """
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0 : n - 1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end - len(adj_slice) + np.amin(non_zero)
    return adj_output


def decode_adj(adj_output):
    """
    recover to adj from adj_output
    note: here adj_output have shape (n-1)*m
    """
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i, ::-1][
            output_start:output_end
        ]  # reverse order
    adj_full = np.zeros((adj_output.shape[0] + 1, adj_output.shape[0] + 1))
    n = adj_full.shape[0]
    adj_full[1:n, 0 : n - 1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T
    return adj_full


def get_data_params(G_list, iter=20000, topk=10, progress_bar_qt=None):
    adj_all = []
    len_all = []
    for G in G_list:
        adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
        len_all.append(G.number_of_nodes())
    max_prev_node = []
    bar = ProgressBar(
        total=iter, desc="Estimating max prev", progress_bar_qt=progress_bar_qt
    )
    for _ in range(iter):
        if progress_bar_qt is not None and progress_bar_qt.stop_running:
            break
        adj_idx = np.random.randint(len(adj_all))
        adj_copy = adj_all[adj_idx].copy()
        # print('Graph size', adj_copy.shape[0])
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        # encode adj
        adj_encoded = encode_adj_flexible(adj_copy.copy())
        max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
        max_prev_node.append(max_encoded_len)
        bar.set_postfix(max_prev=max(max_prev_node))
        bar.update()
    bar.close()
    max_prev_node = sorted(max_prev_node)[-1 * topk :]
    return max_prev_node, max(len_all)


def get_graph_params(
    source_path, data_size, min_num_node, max_num_node, progress_bar_qt=None
):
    """
    get graph dataset information
    :param source_path: path to data graph in gt format
    :param min_num_node: minimum number of nodes to be considered
    :param max_num_node: maximum number of nodes to be considered
    :return:
    """
    G_list = []
    bar = ProgressBar(
        total=data_size, desc="Loading graph", progress_bar_qt=progress_bar_qt
    )
    min_num_node = 0 if min_num_node is None else min_num_node
    max_num_node = 1000 if max_num_node is None else max_num_node
    graph_names = [p for p in os.listdir(source_path) if p.endswith(".xz.gt")]
    for name in graph_names[0:data_size]:
        if progress_bar_qt is not None and progress_bar_qt.stop_running:
            break
        graph_path = os.path.join(source_path, name)
        g = from_gt_to_nx(gt.load_graph(graph_path))
        num_nodes = g.number_of_nodes()
        if num_nodes >= min_num_node and num_nodes <= max_num_node:
            G_list.append(g)
        bar.update()
    bar.close()
    return get_data_params(G_list, progress_bar_qt)
