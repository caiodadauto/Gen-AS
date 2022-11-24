import os
from collections.abc import MutableSequence

import graph_tool as gt
from tqdm import tqdm

from digg.generator.graph_utils import from_gt_to_nx
from digg.generator.preprocessing import get_data_params


def get_caida_params(caida_source_path, data_size, min_num_node, max_num_node):
    """
    get caida dataset information
    :param caida_source_path: path to caida graph in gt format
    :param min_num_node: minimum number of nodes to be considered
    :param max_num_node: maximum number of nodes to be considered
    :return:
    """
    G_list = []
    bar = tqdm(total=data_size, desc="Loading caida")
    min_num_node = 0 if min_num_node is None else min_num_node
    max_num_node = 1000 if max_num_node is None else max_num_node
    graph_names = [p for p in os.listdir(caida_source_path) if p.endswith(".xz.gt")]
    for name in graph_names[0:data_size]:
        graph_path = os.path.join(caida_source_path, name)
        g = from_gt_to_nx(gt.load_graph(graph_path))
        num_nodes = g.number_of_nodes()
        if num_nodes >= min_num_node and num_nodes <= max_num_node:
            G_list.append(g)
        bar.update()
    bar.close()
    return get_data_params(G_list)


class GraphsCAIDA(MutableSequence):
    def __init__(
        self,
        caida_source_path=None,
        data_size=None,
        min_num_node=None,
        max_num_node=None,
        list=None,
        from_path=True,
        check_size=True,
        in_memory=False,
    ):
        super(GraphsCAIDA, self).__init__()
        self._list = []
        self._in_memory = in_memory
        if from_path:
            if max_num_node is None and min_num_node is not None:
                max_num_node = 1000
            elif max_num_node is not None and min_num_node is None:
                min_num_node = 0
            for graph_name in tqdm(
                os.listdir(caida_source_path), desc="Loading caida data"
            ):
                if graph_name.endswith(".xz.gt"):
                    graph_path = os.path.join(caida_source_path, graph_name)
                    if min_num_node is None and max_num_node is None:
                        self._list.append(graph_path)
                    elif check_size:
                        G = gt.load_graph(graph_path)
                        num_nodes = G.num_vertices()
                        if num_nodes >= min_num_node and num_nodes <= max_num_node:
                            self._list.append(graph_path)
                    else:
                        self._list.append(graph_path)
            self._list = self._list[:data_size]
        else:
            self._list = list if list is not None else []
        if self._in_memory:
            self._graphs = []
            for graph_path in tqdm(self._list, desc="Adding caida data to memory"):
                G = from_gt_to_nx(gt.load_graph(graph_path))
                self._graphs.append(G)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, ii):
        if self._in_memory:
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

# max_prev_node: Max previous node that looks back (if none, automatically defined)
def create(
    dataset,
    caida_source_path,
    data_size,
    min_num_nodes,
    max_num_nodes,
    check_size,
):
    graphs = []
    if dataset == "caida":
        graphs = GraphsCAIDA(
            caida_source_path,
            data_size,
            min_num_nodes,
            max_num_nodes,
            check_size=check_size,
        )
        max_prev_node = 246  # Use None for compute estimation
    return graphs, max_prev_node, max_num_nodes
