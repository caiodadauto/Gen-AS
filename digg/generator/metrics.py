import concurrent.futures
from functools import partial

import pyemd
import numpy as np
from scipy.linalg import toeplitz
from networkx.algorithms.cluster import clustering
from graph_tool.centrality import betweenness, pagerank
from graph_tool.correlations import assortativity


from digg.generator.graph_utils import from_nx_to_gt


def _degree(graph):
    degrees = list(dict(graph.degree()).values())
    hist, _ = np.histogram(degrees, bins=np.arange(max(degrees) + 2), density=False)
    hist_sum = np.sum(hist)
    hist = np.array([h / hist_sum for h in hist])
    return hist


def _clustering(graph, bins):
    clustering_coef = list(clustering(graph).values())
    hist, _ = np.histogram(clustering_coef, bins=bins, range=(0.0, 1.0), density=False)
    hist_sum = np.sum(hist)
    hist = np.array([h / hist_sum for h in hist])
    return hist


def _betweenness(graph, bins):
    graph_gt = from_nx_to_gt(graph)
    node_bt = betweenness(graph_gt)[0].get_array()
    hist, _ = np.histogram(node_bt, bins=bins, range=(0.0, 1.0), density=False)
    hist_sum = np.sum(hist)
    hist = np.array([h / hist_sum for h in hist])
    return hist


def _pagerank(graph, bins):
    graph_gt = from_nx_to_gt(graph)
    pg = pagerank(graph_gt).get_array()
    hist, _ = np.histogram(pg, bins=bins, range=(0.0, 1.0), density=False)
    hist_sum = np.sum(hist)
    hist = np.array([h / hist_sum for h in hist])
    return hist


def _assortativity(graph):
    graph_gt = from_nx_to_gt(graph)
    ast, _ = assortativity(graph_gt, deg="out")
    if np.isnan(ast):
        ast == 0
    return [ast]


def correct_assortativity(samples, bins):
    asts = [ast[0] for ast in samples]
    hist, _ = np.histogram(asts, bins=bins, range=(-1.0, 1.0), density=False)
    hist_sum = np.sum(hist)
    hist = np.array([h / hist_sum for h in hist])
    return [hist]


def get_mmd(ref_graphs, pred_graphs, metric, is_parallel=True, bins=100):
    sample_ref = []
    sample_pred = []
    if metric == "degree":
        metric_fn = _degree
        kernel_fn = gaussian_emd
    elif metric == "clustering":
        metric_fn = partial(_clustering, bins=bins)
        kernel_fn = partial(gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    elif metric == "betweenness":
        metric_fn = partial(_betweenness, bins=bins)
        kernel_fn = partial(gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    elif metric == "pagerank":
        metric_fn = partial(_pagerank, bins=bins)
        kernel_fn = partial(gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    elif metric == "assortativity":
        metric_fn = partial(_assortativity)
        kernel_fn = partial(gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    else:
        raise ValueError(f"Unknown metric name {metric}.")
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(metric_fn, ref_graphs):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(metric_fn, pred_graphs):
                sample_pred.append(deg_hist)
    else:
        for graph in ref_graphs:
            sample_ref.append(metric_fn(graph))
        for graph in pred_graphs:
            sample_pred.append(metric_fn(graph))
    if metric == "assortativity":
        sample_ref = correct_assortativity(sample_ref, bins)
        sample_pred = correct_assortativity(sample_pred, bins)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel_fn)
    return mmd_dist


def compute_mmd(samples1, samples2, kernel):
    return (
        disc(samples1, samples1, kernel)
        + disc(samples2, samples2, kernel)
        - 2 * disc(samples1, samples2, kernel)
    )


def in_kernel(s1, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(s1, s2)
    return d


def disc(samples1, samples2, kernel, is_parallel=True):
    d = 0
    if is_parallel:
        partial_fn = partial(in_kernel, samples2=samples2, kernel=kernel)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(partial_fn, samples1):
                d += dist
    else:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2)
    d /= len(samples1) * len(samples2)
    return d


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))
