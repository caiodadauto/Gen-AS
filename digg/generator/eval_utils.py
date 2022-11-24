from os import makedirs
from os.path import join

import torch
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable


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


def plot_graphs(data, save_dir):
    if isinstance(data, str):
        data = pd.read_csv(join(save_dir, data))
    save_dir = join(save_dir, "imgs")
    makedirs(save_dir, exist_ok=True)
    plot(data, False, save_dir, sns.boxplot, "box_plot")
    plot(data, True, save_dir, sns.boxplot, "box_plot")
    plot(data, False, save_dir, sns.barplot, "bar_plot", capsize=0.2)
    plot(data, True, save_dir, sns.barplot, "bar_plot", capsize=0.2)


def plot(data, log, save_dir, plot_fn, name, **kwargs):
    fig = plt.figure(figsize=(15, 8), dpi=200)
    ax = fig.subplots(1, 1, sharey=False)
    plot_fn(x="from", y="value", hue="metric", data=data, ax=ax, **kwargs)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    if log:
        ax.set_yscale("log")
    plt.savefig(join(save_dir, f"{name}{'_log' if log else ''}.png"))
    plt.savefig(join(save_dir, f"{name}{'_log' if log else ''}.pdf"))
    plt.close()
