import random
from datetime import datetime
from os import makedirs
from os.path import join

import torch
import numpy as np
import pandas as pd
import mlflow as mlf
import networkx as nx
import graph_tool as gt
from torch.autograd import Variable
from graph_tool.clustering import local_clustering
from graph_tool.correlations import assortativity
from graph_tool.centrality import betweenness, pagerank

from dggi.utils import ProgressBar
from dggi.generator.metrics import get_mmd
from dggi.generator.preprocessing import decode_adj
from dggi.generator.eval_utils import sample_sigmoid, plot_metrics
from dggi.generator.graph_utils import from_gt_to_nx, get_graph, from_nx_to_gt
from dggi.generator.mlf_utils import (
    mlf_get_model,
    mlf_get_all_run_models,
    mlf_get_data_paths,
    mlf_set_env,
    mlf_save_text,
    mlf_save_pickle,
)


def generate(
    cfg,
    num_graphs,
    min_num_node,
    max_num_node,
    run_dir,
    progress_bar_qt=None,
    model_metric=None,
    mlf_run=None,
):
    min_num_node = cfg.generation.min_num_node if min_num_node is None else min_num_node
    max_num_node = cfg.generation.max_num_node if max_num_node is None else max_num_node
    if mlf_run is None:
        _, mlf_run, save_dir = mlf_set_env(
            cfg.generation.seed,
            run_dir,
            cfg.mlflow.exp_name,
            root_dir=cfg.generation.save_dir,
            load_runs=True,
            fix_path=True,
        )
    else:
        save_dir = join(
            cfg.generation.save_dir, datetime.strftime(datetime.now(), "%y%m%d%H%M%S")
        )
    if mlf_run is None:
        print("There is no model to be used")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rnn, output, _ = mlf_get_model(mlf_run, device, model_metric=model_metric)
    generate_from_env(
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
        progress_bar_qt=progress_bar_qt,
        return_graphs=False,
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
    progress_bar_qt=None,
    return_graphs=True,
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
        progress_bar_qt=progress_bar_qt,
        return_graphs=return_graphs,
        save_dir=save_dir,
    )
    return pred_graphs


def evaluate(cfg, num_graphs, run_dir, mlf_run=None, progress_bar_qt=None):
    raw_metrics_models = {}
    mmd_data = {}
    mmd_data["run_id"] = []
    mmd_data["Model"] = []
    mmd_data["Metric"] = []
    mmd_data["MMD"] = []
    if mlf_run is None:
        rng, mlf_run, _ = mlf_set_env(
            cfg.evaluation.seed,
            run_dir,
            cfg.mlflow.exp_name,
            load_runs=True,
            fix_path=True,
        )
    else:
        seed = cfg.evaluation.seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
    with mlf.start_run(run_id=mlf_run.info.run_id):
        print(cfg.evaluation.metrics)
        true_test_graphs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = mlf_get_all_run_models(mlf_run, device)
        _, _, true_test_paths = mlf_get_data_paths()
        bar = ProgressBar(
            len(true_test_paths),
            desc="Getting graphs for test",
            progress_bar_qt=progress_bar_qt,
        )
        for p in true_test_paths:
            if progress_bar_qt is not None and progress_bar_qt.stop_running:
                return None
            if p.endswith(".gt"):
                graph = from_gt_to_nx(gt.load_graph(p))
            else:
                graph = nx.read_gpickle(p)
            true_test_graphs.append(graph)
            bar.update()
        bar.close()
        raw_metrics_models["Ground truth"] = raw_metrics_eval(
            true_test_graphs, cfg.evaluation.metrics, progress_bar_qt=progress_bar_qt
        )
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
                progress_bar_qt=progress_bar_qt,
            )
            _, _, mmd_values = bootstrap_eval(
                true_test_graphs,
                pred_graphs,
                rng,
                cfg.evaluation.metrics,
                n_samples=cfg.evaluation.n_bootstrap_samples,
                progress_bar_qt=progress_bar_qt,
            )
            raw_metrics_models[model_for] = raw_metrics_eval(
                pred_graphs, cfg.evaluation.metrics, progress_bar_qt=progress_bar_qt
            )
            for metric in cfg.evaluation.metrics:
                mmd_data["run_id"].extend(
                    [f"{mlf_run.info.run_id}"] * cfg.evaluation.n_bootstrap_samples
                )
                mmd_data["Model"].extend(
                    [model_for] * cfg.evaluation.n_bootstrap_samples
                )
                mmd_data["Metric"].extend([metric] * cfg.evaluation.n_bootstrap_samples)
                mmd_data["MMD"].extend(mmd_values[metric])
        mmd_data = pd.DataFrame(mmd_data)
        mlf_save_text("mmd_data.csv", "evaluation", mmd_data.to_csv(index=False))
        mlf_save_pickle("raw_metrics_models", "evaluation", raw_metrics_models)
        plot_metrics(mmd_data, raw_metrics_models, cfg.evaluation.metrics, "evaluation")


def bootstrap_eval(
    true_graphs, pred_graphs, rng, metrics, n_samples=2000, progress_bar_qt=None
):
    mmd_ci = {}
    mmd_means = {}
    mmd_values = {}
    true_graph_idx = np.arange(len(true_graphs))
    num_pred_graphs = len(pred_graphs)
    for metric in metrics:
        mmd_values[metric] = []
    bar = ProgressBar(
        total=n_samples,
        desc="Bootstrap sampling evaluation",
        progress_bar_qt=progress_bar_qt,
    )
    for _ in range(n_samples):
        sampled_data_idx = rng.choice(
            true_graph_idx, replace=True, size=num_pred_graphs
        )
        sampled_true_graphs = [true_graphs[i] for i in sampled_data_idx]
        for metric in metrics:
            bar.set_postfix(metric=metric)
            mmd = get_mmd(sampled_true_graphs, pred_graphs, metric)
            mmd_values[metric].append(mmd)
            if progress_bar_qt is not None and progress_bar_qt.stop_running:
                break
        if progress_bar_qt is not None and progress_bar_qt.stop_running:
            return {}, {}, {}
        bar.update()
    for metric in metrics:
        mmd_means[metric] = np.mean(mmd_values[metric])
        mmd_ci[metric] = np.percentile(mmd_values[metric], [2.5, 97.5])
    return mmd_means, mmd_ci, mmd_values


def raw_metrics_eval(graphs, metrics, progress_bar_qt=None):
    raw_metrics = {}
    for m in metrics:
        raw_metrics[m] = []
    bar = ProgressBar(
        total=len(graphs),
        desc="Getting raw graph metrics",
        progress_bar_qt=progress_bar_qt,
    )
    for g in graphs:
        g = from_nx_to_gt(g)
        if "degree" in metrics:
            raw_metrics["degree"].append(g.get_total_degrees(range(g.num_vertices())))
        if "clustering" in metrics:
            raw_metrics["clustering"].append(local_clustering(g).a)
        if "betweenness" in metrics:
            raw_metrics["betweenness"].append(betweenness(g)[0].a)
        if "assortativity" in metrics:
            ast = assortativity(g, deg="out")[0]
            if np.isnan(ast):
                ast = 0
            raw_metrics["assortativity"].append(ast)
        if "pagerank" in metrics:
            raw_metrics["pagerank"].append(pagerank(g).a)
        if progress_bar_qt is not None and progress_bar_qt.stop_running:
            break
        bar.update()
    if progress_bar_qt is not None and progress_bar_qt.stop_running:
        return {}
    for k, v in raw_metrics.items():
        if k == "assortativity":
            raw_metrics[k] = np.array(v)
        else:
            raw_metrics[k] = np.concatenate(v)
    return raw_metrics


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
    progress_bar_qt=None,
    return_graphs=True,
    save_dir=None,
):
    n_graphs = 0
    bar = ProgressBar(
        total=test_total_size,
        desc="Creating synthetic samples",
        progress_bar_qt=progress_bar_qt,
    )
    bar.set_suffix(f"0 / {test_total_size} synthetic graphs")
    if save_dir is not None:
        save_graph_dir = join(save_dir, "synthetic_graphs")
        makedirs(save_graph_dir)
    if return_graphs:
        G_pred = []
    else:
        G_pred = None
    while n_graphs < test_total_size:
        if progress_bar_qt is not None and progress_bar_qt.stop_running:
            break
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
        if save_dir is not None:
            for i, g in enumerate(G_pred_step):
                nx.write_gpickle(g, join(save_graph_dir, f"{n_graphs + i}.gpickle"))
        n_graphs += len(G_pred_step)
        delta = test_total_size - n_graphs
        if return_graphs:
            G_pred.extend(G_pred_step)
        if delta < 0:
            step = len(G_pred_step) + delta
        else:
            step = len(G_pred_step)
        bar.update(step)
        bar.set_suffix(
            f"{n_graphs if n_graphs < test_total_size else test_total_size} / {test_total_size} synthetic graphs"
        )
    bar.close()
    if return_graphs:
        return G_pred[:test_total_size]
    return G_pred


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
