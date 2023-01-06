from os import makedirs
from os.path import join

import torch
import numpy as np
import pandas as pd
import mlflow as mlf
import networkx as nx
import graph_tool as gt
from tqdm import tqdm
from torch.autograd import Variable

from digg.generator.metrics import get_mmd
from digg.generator.preprocessing import decode_adj
from digg.generator.eval_utils import sample_sigmoid, plot_metrics
from digg.generator.graph_utils import from_gt_to_nx, get_graph
from digg.generator.mlf_utils import (
    mlf_get_model,
    mlf_get_all_run_models,
    mlf_get_data_paths,
    mlf_set_env,
    mlf_save_text,
    mlf_save_pickle,
)


def generate(cfg, num_graphs, min_num_node, max_num_node, run_dir):
    _, mlf_run, save_dir = mlf_set_env(
        cfg.generation.seed,
        run_dir,
        cfg.mlflow.exp_name,
        root_dir=cfg.generation.save_dir,
        load_runs=True,
        fix_path=True,
    )
    with mlf.start_run(run_id=mlf_run.info.run_id):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rnn, output, _ = mlf_get_model(mlf_run, device)
        _ = generate_from_env(
            rnn,
            output,
            device,
            mlf_run,
            num_graphs,
            min_num_node,
            max_num_node,
            cfg.generation.test_batch_size,
            cfg.generation.test_total_size,
            save_dir,
        )


def generate_from_env(
    rnn,
    output,
    device,
    mlf_run,
    num_graphs,
    min_num_node,
    max_num_node,
    test_batch_size,
    test_total_size,
    save_dir=None,
):
    min_num_node = (
        int(mlf_run.data.params["data.min_num_node"])
        if min_num_node is None
        else min_num_node
    )
    max_num_node = (
        int(mlf_run.data.params["data.max_num_node"])
        if max_num_node is None
        else max_num_node
    )
    max_prev_node = int(mlf_run.data.params["data.max_prev_node"])
    num_layer = int(mlf_run.data.params["model.num_layer"])
    pred_graphs = synthesize_graph_sample(
        rnn,
        output,
        min_num_node,
        max_num_node,
        max_prev_node,
        num_layer,
        device,
        test_batch_size,
        test_total_size if num_graphs is None else num_graphs,
    )
    if save_dir is None:
        # mlf_save_pickle(f"synthetic_graphs", f"model_dir", pred_graphs)
        pass
    else:
        save_graph_dir = join(save_dir, "synthetic_graphs")
        makedirs(save_graph_dir, exist_ok=True)
        for i, g in enumerate(pred_graphs):
            nx.write_gpickle(g, join(save_graph_dir, f"{i}.gpickle"))
    return pred_graphs


def evaluate(cfg, num_graphs, run_dir):
    mmd_data = {}
    mmd_data["run_id"] = []
    mmd_data["for"] = []
    mmd_data["metric"] = []
    mmd_data["value"] = []
    rng, mlf_run, _ = mlf_set_env(
        cfg.evaluation.seed,
        run_dir,
        cfg.mlflow.exp_name,
        load_runs=True,
        fix_path=True,
    )
    with mlf.start_run(run_id=mlf_run.info.run_id):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = mlf_get_all_run_models(mlf_run, device)
        for model_for, (rnn, output) in models.items():
            pred_graphs = generate_from_env(
                rnn,
                output,
                device,
                mlf_run,
                num_graphs,
                None,
                None,
                cfg.evaluation.test_batch_size,
                cfg.evaluation.test_total_size,
            )
            _, _, true_test_paths = mlf_get_data_paths()
            true_test_graphs = [
                from_gt_to_nx(gt.load_graph(p))
                for p in tqdm(true_test_paths, desc="Getting graphs for test")
                if p.endswith(".xz.gt")
            ]
            mmd_values, _, _ = get_metrics(
                true_test_graphs,
                pred_graphs,
                cfg.evaluation.metrics,
                cfg.evaluation.n_bootstrap_samples,
                rng,
            )
            for metric in cfg.evaluation.metrics:
                mmd_data["run_id"].extend(
                    [f"{mlf_run.info.run_id}"] * cfg.evaluation.n_bootstrap_samples
                )
                mmd_data["for"].extend([model_for] * cfg.evaluation.n_bootstrap_samples)
                mmd_data["metric"].extend([metric] * cfg.evaluation.n_bootstrap_samples)
                mmd_data["value"].extend(mmd_values[metric])
        mmd_data = pd.DataFrame(mmd_data)
        mlf_save_text(
            "mmd_data.csv", "evaluation", mmd_data.to_csv(index=False)
        )
        plot_metrics(mmd_data, "evaluation")


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
        sampled_data_idx = rng.choice(
            true_graph_idx, replace=True, size=num_pred_graphs
        )
        sampled_true_graphs = [true_graphs[i] for i in sampled_data_idx]
        for metric in metrics:
            bar.set_postfix(metric=metric)
            mmd = get_mmd(sampled_true_graphs, pred_graphs, metric)
            mmd_values[metric].append(mmd)
        bar.update()
    for metric in metrics:
        mmd_means[metric] = np.mean(mmd_values[metric])
        mmd_ci[metric] = np.percentile(mmd_values[metric], [2.5, 97.5])
    return mmd_means, mmd_ci, mmd_values


def get_metrics(true_graphs, pred_graphs, metrics, n_samples, rng):
    mmd_means, mmd_ci, mmd_values = bootstrap_eval(
        true_graphs, pred_graphs, rng, metrics, n_samples=n_samples
    )
    return mmd_values, mmd_means, mmd_ci


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
            adj_pred, min_num_node, max_num_node
        )  # get a graph from zero-padded adj
        G_pred_list.extend(subgraphs)
    return G_pred_list
