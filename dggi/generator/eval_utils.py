import warnings
from os import makedirs
from os.path import join
from functools import partial

import numpy as np
import pandas as pd

import torch
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable

from dggi.generator.mlf_utils import mlf_save_figure


def sample_sigmoid(y, sample, device, thresh=0.5, sample_time=2):
    """
    do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    """
    # do sigmoid first
    y = torch.sigmoid(y)
    # do sampling
    if sample:
        if sample_time > 1:
            y_result = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for _ in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).to(device)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).to(
            device
        )
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def save_obj(name, save_dir, obj):
    makedirs(save_dir, exist_ok=True)
    with open(join(save_dir, name), "wb") as f:
        joblib.dump(obj, f)


def bar_plot(data, log, name, artifact_path, **kwargs):
    sns.plotting_context(context="paper")
    sns.set_palette("Set1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = plt.figure(figsize=(15, 10), dpi=250)
    ax = fig.subplots(1, 1, sharey=False)
    sns.barplot(
        x="Metric", y="MMD", hue="Model", data=data, ax=ax, edgecolor="0.15", **kwargs
    )
    ax.yaxis.grid(True)
    fig.tight_layout()
    if log:
        ax.set_yscale("log")
    mlf_save_figure(f"{name}{'_log' if log else ''}", artifact_path, "png")
    plt.close()


def _lines_plot(
    true_metrics, models_metrics, metric, artifact_path, func_t=None, log_x=True
):
    def get_freqs(data, metric, model, func_t):
        metrics = func_t(data[[metric]])
        metric_freq = metrics.groupby(by=[metric]).size() / len(metrics)
        metric_freq = metric_freq.reset_index()
        metric_freq.rename(columns={0: "Frequency"}, inplace=True)
        metric_freq["Model"] = model
        return metric_freq.loc[metric_freq["Frequency"] > 0]

    freqs = []
    sns.plotting_context(context="paper")
    sns.set_palette("Set1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = plt.figure(figsize=(7, 4), dpi=200)
    ax = fig.subplots(1, 1, sharey=False)
    func_t = func_t if func_t is not None else lambda x: x
    for model, model_metrics in models_metrics.items():
        freqs.append(get_freqs(model_metrics, metric, model, func_t))
    freqs.append(get_freqs(true_metrics, metric, "Ground truth", func_t))
    freqs = pd.concat(freqs, axis=0, ignore_index=True)
    ax = sns.lineplot(
        data=freqs,
        x=metric,
        y="Frequency",
        hue="Model",
        hue_order=["Ground truth"] + sorted(list(models_metrics.keys())),
        markers=True,
        style="Model",
        style_order=["Ground truth"] + sorted(list(models_metrics.keys())),
        lw=2,
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    mlf_save_figure(f"lines_plot_{metric}", artifact_path, "png")
    plt.close()


def bin_float_metrics(data, bins):
    metric = data.columns[0]
    df = pd.DataFrame(
        {
            metric: pd.cut(
                data[metric].values,
                bins=bins[metric],
                labels=bins[metric][:-1],
                include_lowest=True,
            )
        }
    )
    return df


def lines_plot(raw_metrics_models, metric_names, artifact_path):
    bins = {}
    models_ast = {}
    models_metrics = {}
    gt_metrics = raw_metrics_models.pop("Ground truth")
    gt_ast = gt_metrics.pop("assortativity")
    gt_metrics = pd.DataFrame(gt_metrics)
    gt_ast = pd.DataFrame(gt_ast, columns=["assortativity"])
    for k, v in raw_metrics_models.items():
        models_ast[k] = pd.DataFrame(v.pop("assortativity"), columns=["assortativity"])
        models_metrics[k] = pd.DataFrame(v)
    for m in metric_names:
        if m == "assortativity":
            bins[m] = np.histogram_bin_edges(gt_ast[m], bins="doane")
        else:
            bins[m] = np.histogram_bin_edges(gt_metrics[m], bins="doane")
    for m in metric_names:
        if m == "assortativity":
            _lines_plot(
                gt_ast,
                models_ast,
                m,
                artifact_path,
                func_t=partial(bin_float_metrics, bins=bins),
                log_x=False,
            )
        else:
            _lines_plot(
                gt_metrics,
                models_metrics,
                m,
                artifact_path,
                func_t=partial(bin_float_metrics, bins=bins),
            )


def plot_metrics(mmd_data, raw_metrics_models, metric_names, artifact_path):
    bar_plot(mmd_data, True, "bar_plot", artifact_path, capsize=0.2)
    lines_plot(raw_metrics_models, metric_names, artifact_path)
