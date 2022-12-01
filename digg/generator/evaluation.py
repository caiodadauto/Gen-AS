from os.path import join, isdir, basename

import torch
import numpy as np
import pandas as pd
import mlflow as mlf
import networkx as nx
import graph_tool as gt
from tqdm import tqdm
from os import listdir
from torch.autograd import Variable

from digg.generator.metrics import get_mmd
from digg.generator.preprocessing import decode_adj
from digg.generator.eval_utils import sample_sigmoid, save_obj
from digg.generator.graph_utils import from_gt_to_nx, get_graph
from digg.generator.mlf_utils import (
    mlf_get_run,
    mlf_fix_artifact_path,
    mlf_get_data_paths,
    mlf_get_synthetic_graph,
)


def eval(exp_name, metrics, seed, baseline, n_samples, save_dir):
    mmd_data = {}
    mmd_data["from"] = []
    mmd_data["metric"] = []
    mmd_data["value"] = []
    mlf_fix_artifact_path()
    rng = np.random.default_rng(seed)
    mlf_run = mlf_get_run(exp_name=exp_name, load_runs=True)
    with mlf.start_run(run_id=mlf_run.info.run_id):
        _, _, test_paths = mlf_get_data_paths()
        print("Getting CAIDA graphs for test...")
        test_caida_graphs = [
            from_gt_to_nx(gt.load_graph(p)) for p in test_paths if p.endswith(".xz.gt")
        ]
        get_metrics(
            test_caida_graphs,
            metrics,
            ["best_mmd_degree"],  # , "best_mmd_clustering"],
            mlf_get_synthetic_graph,
            n_samples,
            mmd_data,
            save_dir,
            rng,
            gen_name="ditg",
        )
    graph_set_paths = [
        join("data", baseline, p)
        for p in listdir(join("data", baseline))
        if isdir(join("data", baseline, p))
    ]
    graph_fn = lambda s: [
        nx.read_gpickle(join(s, "graphs", p))
        for p in listdir(join(s, "graphs"))[:500]
        if p.endswith(".gpickle")
    ]
    get_metrics(
        test_caida_graphs,
        metrics,
        graph_set_paths,
        graph_fn,
        n_samples,
        mmd_data,
        save_dir,
        rng,
    )
    mmd_data = pd.DataFrame(mmd_data)
    mmd_data.to_csv("all_values_mmd.csv", index=False)
    # plot_graphs(mmd_data, save_dir)


def bootstrap_eval(true_graphs, pred_graphs, rng, metrics, n_samples=2000):
    mmd_ci = {}
    mmd_means = {}
    mmd_values = {}
    true_graph_idx = np.arange(len(true_graphs))
    num_pred_graphs = len(pred_graphs)
    for metric in metrics:
        mmd_values[metric] = []
    bar = tqdm(total=n_samples, desc="Bootstraping")
    for _ in range(n_samples):
        sampled_caida_idx = rng.choice(
            true_graph_idx, replace=True, size=num_pred_graphs
        )
        sampled_true_graphs = [true_graphs[i] for i in sampled_caida_idx]
        for metric in metrics:
            bar.set_postfix(metric=metric)
            mmd = get_mmd(sampled_true_graphs, pred_graphs, metric)
            mmd_values[metric].append(mmd)
        bar.update()
    for metric in metrics:
        mmd_means[metric] = np.mean(mmd_values[metric])
        mmd_ci[metric] = np.percentile(mmd_values[metric], [2.5, 97.5])
    return mmd_means, mmd_ci, mmd_values


def get_metrics(
    true_graphs,
    metrics,
    keys,
    graph_fn,
    n_samples,
    data_output,
    save_dir,
    rng,
    gen_name=None,
):
    gen_names = (
        [basename(key) for key in keys] if gen_name is None else [gen_name] * len(keys)
    )
    for key, gen_name in zip(keys, gen_names):
        graphs = graph_fn(key)
        mmd_means, mmd_ci, mmd_values = bootstrap_eval(
            true_graphs, graphs, rng, metrics, n_samples=n_samples
        )
        for name, obj in [
            (f"{basename(key)}_ci.pkl", mmd_ci),
            (f"{basename(key)}_means.pkl", mmd_means),
            (f"{basename(key)}_values.pkl", mmd_values),
        ]:
            save_obj(name, join(save_dir, key), obj)
        for metric in metrics:
            data_output["value"].extend(mmd_values[metric])
            data_output["metric"].extend([metric] * n_samples)
            data_output["from"].extend([basename(key)] * n_samples)


def synthesize_graph_sample(
    rnn,
    output,
    min_num_node,
    max_num_node,
    max_prev_node,
    num_layer,
    device,
    test_batch_size,
    test_total_size,
):
    G_pred = []
    bar = tqdm(total=test_total_size, desc="Creating synthetic samples")
    while len(G_pred) < test_total_size:
        G_pred_step = synthesize_graph(
            rnn,
            output,
            min_num_node,
            max_num_node,
            max_prev_node,
            num_layer,
            device,
            test_batch_size=test_batch_size,
        )
        G_pred.extend(G_pred_step)
        delta = test_total_size - len(G_pred)
        if delta < 0:
            step = len(G_pred_step) + delta
        else:
            step = len(G_pred_step)
        bar.update(step)
    bar.close()
    return G_pred[:test_total_size]


def synthesize_graph(
    rnn,
    output,
    min_num_node,
    max_num_node,
    max_prev_node,
    num_layer,
    device,
    test_batch_size=16,
):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    max_num_node = int(max_num_node)
    y_pred_long = Variable(
        torch.zeros(test_batch_size, max_num_node, max_prev_node)
    ).to(
        device
    )  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1, max_prev_node)).to(device)
    for i in range(max_num_node):
        h = rnn(x_step)
        hidden_null = Variable(torch.zeros(num_layer - 1, h.size(0), h.size(2))).to(
            device
        )
        output.hidden = torch.cat(
            (h.permute(1, 0, 2), hidden_null), dim=0
        )  # num_layer, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size, 1, max_prev_node)).to(device)
        output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).to(device)
        for j in range(min(max_prev_node, i + 1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(
                output_y_pred_step, device=device, sample=True, sample_time=1
            )
            x_step[:, :, j : j + 1] = output_x_step
            output.hidden = Variable(output.hidden.data).to(device)
        y_pred_long[:, i : i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(device)
    y_pred_long_data = y_pred_long.data.long()

    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        subgraphs = get_graph(
            adj_pred, min_num_node
        )  # get a graph from zero-padded adj
        G_pred_list.extend(subgraphs)
    return G_pred_list
