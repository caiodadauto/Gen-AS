import re
import joblib
import tempfile
import random
import warnings
from datetime import datetime
from os.path import join, abspath
from os import getcwd, walk, listdir

import yaml
import torch
import numpy as np
import pandas as pd
import mlflow as mlf
import matplotlib.pyplot as plt
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
        tmp_path = join(temp_dir, f"{name}.pkl")
        with open(tmp_path, "wb") as f:
            joblib.dump(obj, f)
        mlf.log_artifact(tmp_path, artifact_path=artifact_path)


def mlf_save_text(name, artifact_path, obj):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = join(temp_dir, f"{name}")
        with open(tmp_path, "w") as f:
            f.write(obj)
        mlf.log_artifact(tmp_path, artifact_path=artifact_path)


def mlf_save_figure(name, artifact_path, ext):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = join(temp_dir, f"{name}.{ext}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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


def mlf_get_evaluation_imgs(mlf_run):
    with mlf.start_run(run_id=mlf_run.info.run_id):
        path = mlf.get_artifact_uri("evaluation").split(":")[-1]
        try:
            imgs_paths = [
                join(path, p)
                for p in listdir(path)
                if (p.startswith("lines_plot") and p.endswith(".png"))
                or p == "bar_plot_log.png"
            ]
        except FileNotFoundError:
            imgs_paths = []
    return imgs_paths


def mlf_get_data_paths():
    paths = {}
    stages = ["train", "validation", "test"]
    for stage in stages:
        paths[stage] = mlf_load_text(join("graphs", f"{stage}_graphs.csv"))
    return paths["train"], paths["validation"], paths["test"]


def mlf_get_all_checkpoints():
    path = mlf.get_artifact_uri().split(":")[-1]
    checkpoint_names = [p for p in listdir(path) if re.match(r"(checkpoint_).+", p)]
    return checkpoint_names


def mlf_get_epoch(stage):
    return int(mlf_load_text(join(stage, "epoch.csv"))[0])


def mlf_get_best_value(metric):
    return float(mlf_load_text(join(f"best_mmd_{metric}", "best_value.csv"))[0])


def mlf_get_synthetic_graph(stage):
    return mlf_load_pickle(join(stage, "synthetic_graphs"))


def mlf_get_model(run, device, model_metric=None):
    if model_metric is None:
        metrics = [k for k in run.data.metrics.keys() if k.startswith("mmd")]
        print("Found saved models:")
        for i, m in enumerate(metrics):
            print(f"\t{i}. Model for the best {m}")
        idx = int(input("Choose the number of the desired model: "))
        model_metric = f"{metrics[idx]}"
    model_uri = f"runs:/{run.info.run_id}/best_{model_metric}"
    rnn = mlf.pytorch.load_model(
        join(model_uri, "rnn"), map_location=torch.device(device)
    )
    output = mlf.pytorch.load_model(
        join(model_uri, "output"), map_location=torch.device(device)
    )
    rnn.device = device
    output.device = device
    return rnn, output, f"best_{model_metric}"


def mlf_get_all_run_models(run, device):
    models = {}
    metrics = [k for k in run.data.metrics.keys() if k.startswith("mmd")]
    for m in metrics:
        model_uri = f"runs:/{run.info.run_id}/best_{m}"
        rnn = mlf.pytorch.load_model(
            join(model_uri, "rnn"), map_location=torch.device(device)
        )
        output = mlf.pytorch.load_model(
            join(model_uri, "output"), map_location=torch.device(device)
        )
        rnn.device = device
        output.device = device
        models[f"Model for Best {m}"] = (rnn, output)
    return models


def mlf_get_run_list(experiment_id):
    def get_duration(row):
        if row["start_time"] is None or row["end_time"] is None:
            return None
        return (row["end_time"] - row["start_time"]).total_seconds() / 3600

    data = {}
    runs_df = mlf.search_runs(experiment_ids=[experiment_id])
    metric_columns = []
    for c in runs_df.columns:
        if c.startswith("params.training.metrics"):
            metric_columns.append(c)
    metrics = (
        runs_df[metric_columns]
        .melt()
        .loc[lambda d: ~d["value"].isnull()]
        .drop_duplicates(subset=["value"])
        .set_index("variable")["value"]
    )
    data["Run ID"] = runs_df["run_id"].tolist()
    data["Status"] = runs_df["status"].tolist()
    data["Start Time"] = (
        runs_df["start_time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")).tolist()
    )
    data["Duration (h)"] = (
        runs_df[["start_time", "end_time"]].apply(get_duration, axis=1).tolist()
    )
    data["Final Loss"] = runs_df["metrics.loss"].tolist()
    for m in metrics:
        try:
            data[f"Final {m} MMD"] = runs_df[f"metrics.mmd_{m}"].tolist()
        except KeyError:
            pass
    runs_df = pd.DataFrame(data)
    if runs_df.shape[1] < 6:
        return pd.DataFrame(
            columns=["Run ID", "Status", "Start time", "Duration (h)", "Final loss"]
        )
    else:
        return (
            runs_df.loc[~runs_df.iloc[:, 5:].isna().all(axis=1)]
            .sort_values("Start Time", ascending=False)
            .reset_index(drop=True)
        )


def mlf_get_run(
    exp_name,
    run_dir,
    exp_tags=None,
    run_tags=None,
    run_id=None,
    load_runs=False,
):
    mlf.set_tracking_uri(f"file://{run_dir}/mlruns")
    client = mlf.tracking.MlflowClient()
    experiment = mlf.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = client.create_experiment(name=exp_name)
        experiment = mlf.get_experiment(experiment_id)
        if exp_tags is not None:
            mlf.set_experiment_tags(exp_tags)
    else:
        experiment_id = experiment.experiment_id
    if run_id is None and not load_runs:
        run_tags = {} if run_tags is None else run_tags
        run_tags = mlf.tracking.context.registry.resolve_tags(run_tags)
        run = client.create_run(experiment_id=experiment_id, tags=run_tags)
    else:
        if load_runs:
            runs_df = mlf_get_run_list(experiment_id)
            if runs_df.empty:
                return None
            t = PrettyTable([""] + runs_df.columns.tolist())
            for i, row in runs_df.iterrows():
                table_row = [
                    i,
                    row["Status"],
                    row["Start Time"],
                    "{:.3f}".format(row["Duration (h)"]),
                    "{:.3f}".format(row["Final Loss"]),
                ]
                for i in range(4, runs_df.shape[1]):
                    table_row.append("{:.3f}".format(row.iloc[i]))
                t.add_row(table_row)
            print(t)
            idx = int(input("Choose the number of the desired run: "))
            run_id = runs_df.loc[idx, "Run ID"]
        run = mlf.get_run(run_id=run_id)
    return run


def mlf_fix_artifact_path():
    def update_artifact_path(meta, key):
        if key in meta:
            artifact_path = meta[key]
            new_artifact_path = join(
                "file://" + cwd, artifact_path[artifact_path.find("mlruns") :]
            )
            meta[key] = new_artifact_path

    cwd = abspath(getcwd())
    for root, _, files in walk(join(cwd, "mlruns")):
        for f_name in files:
            if f_name == "meta.yaml":
                path = join(cwd, "mlruns", root, f_name)
                with open(path, "r") as f:
                    meta = yaml.safe_load(f)
                update_artifact_path(meta, "artifact_location")
                update_artifact_path(meta, "artifact_uri")
                with open(path, "w") as f:
                    meta = yaml.dump(meta, f)


def mlf_set_env(
    seed,
    run_dir,
    exp_name,
    root_dir=None,
    fix_path=False,
    load_runs=False,
    return_runs_list=False,
    **kwargs,
):
    save_dir = (
        None
        if root_dir is None
        else join(root_dir, datetime.strftime(datetime.now(), "%y%m%d%H%M%S"))
    )
    _ = mlf_fix_artifact_path() if fix_path else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if return_runs_list:
        mlf.set_tracking_uri(f"file://{run_dir}/mlruns")
        experiment = mlf.get_experiment_by_name(exp_name)
        if experiment is None:
            return pd.DataFrame(
                columns=["Run ID", "Status", "Start time", "Duration (h)", "Final loss"]
            )
        else:
            experiment_id = experiment.experiment_id
        return mlf_get_run_list(experiment_id)
    mlf_run = mlf_get_run(
        run_dir=run_dir, exp_name=exp_name, load_runs=load_runs, **kwargs
    )
    return rng, mlf_run, save_dir
