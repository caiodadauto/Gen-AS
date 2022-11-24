import os
import re
import joblib
import tempfile

import yaml
import torch
import mlflow as mlf
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig
from prettytable import PrettyTable


def mlf_log_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlf.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlf.log_param(f"{parent_name}.{i}", v)


def mlf_save_pickle(name, artifact_path, obj):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, f"{name}.pkl")
        with open(tmp_path, "wb") as f:
            joblib.dump(obj, f)
        mlf.log_artifact(tmp_path, artifact_path=artifact_path)


def mlf_save_text(name, artifact_path, obj):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, f"{name}")
        with open(tmp_path, "w") as f:
            f.write(obj)
        mlf.log_artifact(tmp_path, artifact_path=artifact_path)


def mlf_save_figure(name, artifact_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, f"{name}.png")
        plt.savefig(tmp_path)
        mlf.log_artifact(tmp_path, artifact_path)
    plt.close()


def mlf_load_pickle(name):
    path = mlf.get_artifact_uri(f"{name}.pkl").split(":")[-1]
    return joblib.load(path)


def mlf_load_text(path):
    path = mlf.get_artifact_uri(f"{path}").split(":")[-1]
    with open(path, "r") as f:
        lines = [l.rstrip() for l in f if l != ""]
    return lines


def mlf_get_data_paths():
    paths = {}
    stages = ["train", "validation", "test"]
    for stage in stages:
        paths[stage] = mlf_load_text(
            os.path.join("caida_graphs", f"{stage}_graphs.csv")
        )
    return paths["train"], paths["validation"], paths["test"]


def mlf_get_all_checkpoints():
    path = mlf.get_artifact_uri().split(":")[-1]
    checkpoint_names = [p for p in os.listdir(path) if re.match(r"(checkpoint_).+", p)]
    return checkpoint_names


def mlf_get_epoch(stage):
    return int(mlf_load_text(os.path.join(stage, "epoch.csv"))[0])


def mlf_get_best_value(metric):
    return float(mlf_load_text(os.path.join(f"best_mmd_{metric}", "best_value.csv"))[0])


def mlf_get_synthetic_graph(stage):
    return mlf_load_pickle(os.path.join(stage, "synthetic_graphs"))


def mlf_get_model(run_id, device):
    model_uri = f"runs:/{run_id}/best_mmd_degree"
    rnn = mlf.pytorch.load_model(
        os.path.join(model_uri, f"rnn"), map_location=torch.device(device)
    )
    output = mlf.pytorch.load_model(
        os.path.join(model_uri, f"output"), map_location=torch.device(device)
    )
    rnn.device = device
    output.device = device
    return rnn, output


def mlf_get_run(
    exp_name,
    exp_tags=None,
    run_tags=None,
    run_id=None,
    load_runs=False,
):
    mlf.set_tracking_uri(f"file://{get_original_cwd()}/mlruns")
    client = mlf.tracking.MlflowClient()
    experiment = mlf.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = client.create_experiment(name=exp_name)
        experiment = mlf.get_experiment(experiment_id)
        if exp_tags is not None:
            for name, tag in exp_tags.items():
                mlf.set_experiment_tag(experiment_id, name, tag)
    else:
        experiment_id = experiment.experiment_id
    if run_id is None and not load_runs:
        run_tags = {} if run_tags is None else run_tags
        run_tags = mlf.tracking.context.registry.resolve_tags(run_tags)
        run = client.create_run(experiment_id=experiment_id, tags=run_tags)
    else:
        if load_runs:
            runs_df = mlf.search_runs(experiment_ids=[experiment_id])
            t = PrettyTable(
                [
                    "",
                    "Status",
                    "Graph Type",
                    "Final Loss",
                    "Final Degree MMD",
                    "Final Clustering MMD",
                    "Duration (h)",
                    "Start Time",
                ]
            )
            for i, row in runs_df.iterrows():
                try:
                    duration = (row["end_time"] - row["start_time"]).total_seconds() / 3600
                except TypeError:
                    duration = 0.0
                t.add_row(
                    [
                        i,
                        row["status"],
                        row["params.data.graph_type"],
                        "{:.3f}".format(row["metrics.loss"]),
                        "{:.3f}".format(row["metrics.mmd_degree"]),
                        "{:.3f}".format(row["metrics.mmd_clustering"]),
                        "{:.3f}".format(duration),
                        row["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    ]
                )
            print(t)
            idx = int(input("Choose the number of the desired run: "))
            run_id = runs_df.loc[idx, "run_id"]
        run = mlf.get_run(run_id=run_id)
    return run


def mlf_fix_artifact_path():
    def update_artifact_path(meta, key):
        if key in meta:
            artifact_path = meta[key]
            new_artifact_path = os.path.join(
                "file://" + cwd, artifact_path[artifact_path.find("mlruns") :]
            )
            meta[key] = new_artifact_path

    cwd = os.path.abspath(os.getcwd())
    for root, _, files in os.walk(os.path.join(cwd, "mlruns")):
        for f_name in files:
            if f_name == "meta.yaml":
                path = os.path.join(cwd, "mlruns", root, f_name)
                with open(path, "r") as f:
                    meta = yaml.safe_load(f)
                update_artifact_path(meta, "artifact_location")
                update_artifact_path(meta, "artifact_uri")
                with open(path, "w") as f:
                    meta = yaml.dump(meta, f)
