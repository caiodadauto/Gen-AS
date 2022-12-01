import numpy as np
import networkx as nx
import graph_tool as gt


def from_nx_to_gt(nx_graph):
    gt_graph = gt.Graph(directed=False)
    gt_graph.add_vertex(nx_graph.number_of_nodes())
    gt_graph.add_edge_list(list(nx_graph.edges()))
    return gt_graph


def from_gt_to_nx(gt_graph):
    """
    convert graphs from garph_tool to networkx structure
    :param gt_graph: graph in graph_tool format
    :return:
    """
    # TODO: Deal with properties
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(gt_graph.get_edges())
    return nx_graph


def get_graph(adj, min_num_node, max_num_nodes=np.infty):
    """
    get a graph from zero-padded adj
    :param adj:
    :return:
    """
    # remove all zeros rows and columns
    subgraphs = []
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    connected_components = nx.connected_components(G)
    for node_set in connected_components:
        if len(node_set) >= min_num_node:
            subgraphs.append(G.subgraph(node_set).copy())
    return subgraphs
