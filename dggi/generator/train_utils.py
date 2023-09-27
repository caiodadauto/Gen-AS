import warnings

import torch
import mlflow as mlf
import torch.nn.functional as F

from dggi.generator.evaluation import synthesize_graph_sample
from dggi.generator.mlf_utils import (
    mlf_save_text,
    mlf_save_pickle,
)


def binary_cross_entropy_weight(
    y_pred, y, has_weight=False, weight_length=1, weight_max=10
):
    """
    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    """
    if has_weight:
        weight = torch.ones(y.size(0), y.size(1), y.size(2))
        weight_linear = torch.arange(1, weight_length + 1) / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(
            y.size(0), 1, y.size(2)
        )
        weight[:, -1 * weight_length :, :] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.cpu())
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def save_model(rnn, output, model_dir, raw_signatures):
    rnn_signature = mlf.models.infer_signature(
        raw_signatures["rnn"][0], raw_signatures["rnn"][1]
    )
    output_signature = mlf.models.infer_signature(
        raw_signatures["output"][0], raw_signatures["output"][1]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlf.pytorch.log_model(
            rnn,
            f"{model_dir}/rnn",
            signature=rnn_signature,
            pip_requirements=["torch"],
        )
        mlf.pytorch.log_model(
            output,
            f"{model_dir}/output",
            signature=output_signature,
            pip_requirements=["torch"],
        )


def save_best(rnn, output, pred_graphs, best_value, metric_name, epoch, raw_signatures):
    mlf_save_pickle(f"synthetic_graphs", f"best_{metric_name}", pred_graphs)
    save_model(rnn, output, f"best_{metric_name}", raw_signatures)
    mlf_save_text("epoch.csv", f"best_{metric_name}", f"{epoch}\n")
    mlf_save_text("best_value.csv", f"best_{metric_name}", f"{best_value}\n")


def checkpoint(
    rnn,
    output,
    epoch,
    idx,
    min_num_node,
    max_num_node,
    max_prev_node,
    num_layer,
    device,
    test_batch_size,
    test_total_size,
    raw_signatures,
    progress_bar_qt=None,
):
    pred_graphs = synthesize_graph_sample(
        rnn,
        output,
        min_num_node,
        max_num_node,
        max_prev_node,
        num_layer,
        device,
        test_batch_size,
        test_total_size,
        progress_bar_qt=progress_bar_qt,
    )
    # mlf_save_pickle(f"synthetic_graphs", f"checkpoint-{idx}", pred_graphs)
    save_model(rnn, output, f"checkpoint-{idx}", raw_signatures)
    mlf_save_text("epoch.csv", f"checkpoint-{idx}", f"{epoch}\n")
    print("checkpoint return")
    return pred_graphs
