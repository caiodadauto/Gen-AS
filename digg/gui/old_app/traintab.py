from multiprocessing import cpu_count

from omegaconf import OmegaConf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QCheckBox

from digg.generator.training import train
from digg.gui.test import CollapsibleDialog
from digg.gui.utils import (
    LineEditLabel,
    get_grid_layout,
    get_label,
    LineEditLabel,
    SpinBoxLabel,
    DoubleSpinBoxLabel,
    ProgressBarButton,
)


class MLFlowSection(QWidget):
    def __init__(self, mlflow_kwargs, mlf_dir):
        super(MLFlowSection, self).__init__()
        window_layout = get_grid_layout(1, 4, [1], [1] * 4)
        self.w_exp_name = LineEditLabel("Experiment Name", mlflow_kwargs.exp_name)
        self.w_run_tag = LineEditLabel(
            "Exp Tags", mlflow_kwargs.run_tags, "tag1, tag2, ..."
        )
        self.w_exp_tag = LineEditLabel(
            "Run Tags", mlflow_kwargs.exp_tags, "tag1, tag2, ..."
        )
        self.w_mlf_dir = LineEditLabel("Root Directory", mlf_dir)
        window_layout.addWidget(self.w_exp_name, 0, 0)
        window_layout.addWidget(self.w_run_tag, 0, 1)
        window_layout.addWidget(self.w_exp_tag, 0, 2)
        window_layout.addWidget(self.w_mlf_dir, 0, 3)
        self.setLayout(window_layout)

    def get_cfg(self):
        return dict(
            exp_name=self.w_exp_name.value(),
            run_tags=None
            if self.w_run_tag.value() == ""
            else self.w_run_tag.value().split(","),
            exp_tags=None
            if self.w_exp_tag.value() == ""
            else self.w_exp_tag.value().split(","),
            mlf_dir=self.w_mlf_dir.value(),
        )


class ModelSection(QWidget):
    def __init__(self, model_kwargs):
        super(ModelSection, self).__init__()
        window_layout = get_grid_layout(2, 4, [1] * 2, [1] * 4)
        self.w_in_gru_hidden_size = SpinBoxLabel(
            "Input GRU Hidden Size", model_kwargs.hidden_size_rnn, 1, 1e4
        )
        self.w_out_gru_hidden_size = SpinBoxLabel(
            "Output GRU Hidden Size", model_kwargs.hidden_size_rnn_output, 1, 1e4
        )
        self.w_in_gru_embedding_size = SpinBoxLabel(
            "Input GRU Embedding Size", model_kwargs.embedding_size_rnn, 1, 1e4
        )
        self.w_out_gru_embedding_size = SpinBoxLabel(
            "Output GRU Embedding Size", model_kwargs.embedding_size_rnn_output, 1, 1e4
        )
        self.w_out_mlp_embedding_size = SpinBoxLabel(
            "Output MLP Embedding Size", model_kwargs.embedding_size_output, 1, 1e4
        )
        self.w_num_layers = SpinBoxLabel(
            "Num of Layers", model_kwargs.num_layer, 1, 1e4
        )
        window_layout.addWidget(self.w_in_gru_hidden_size, 0, 0)
        window_layout.addWidget(self.w_out_gru_hidden_size, 0, 1)
        window_layout.addWidget(self.w_in_gru_embedding_size, 0, 2)
        window_layout.addWidget(self.w_out_gru_embedding_size, 0, 3)
        window_layout.addWidget(self.w_out_mlp_embedding_size, 1, 0)
        window_layout.addWidget(self.w_num_layers, 1, 1)
        self.setLayout(window_layout)

    def get_cfg(self):
        return dict(
            hidden_size_rnn=self.w_in_gru_hidden_size.value(),
            hidden_size_rnn_output=self.w_out_gru_hidden_size.value(),
            embedding_size_rnn=self.w_in_gru_embedding_size.value(),
            embedding_size_rnn_output=self.w_out_gru_embedding_size.value(),
            embedding_size_output=self.w_out_mlp_embedding_size.value(),
            num_layer=self.w_num_layers.value(),
        )


class TrainingSection(QWidget):
    def __init__(self, train_kwargs):
        super(TrainingSection, self).__init__()
        window_layout = get_grid_layout(3, 4, [1] * 3, [1] * 4)
        self.w_seed = SpinBoxLabel("Seed", train_kwargs.seed, 1, 1e9)
        self.w_num_epochs = SpinBoxLabel(
            "Num of Epochs", train_kwargs.num_epochs, 1, 1e4
        )
        self.w_eval_start = SpinBoxLabel(
            "Elapsed Epochs to Start Eval", train_kwargs.epochs_test_start, 1, 1e4
        )
        self.w_eval_epochs = SpinBoxLabel(
            "Elapsed Epochs for Eval", train_kwargs.epochs_test, 1, 1e4
        )
        self.w_eval_batch_size = SpinBoxLabel(
            "Eval Batch Size", train_kwargs.test_batch_size, 1, 1e4
        )
        self.w_eval_size = SpinBoxLabel(
            "Num of Graphs for Eval", train_kwargs.test_total_size, 1, 1e4
        )
        self.w_num_bootstrap_samples = SpinBoxLabel(
            "Num of Bootstrap Samples for Eval",
            train_kwargs.n_bootstrap_samples,
            1,
            1e4,
        )
        self.w_lr = DoubleSpinBoxLabel("Inital LR", train_kwargs.lr, 1e-6, 50, 6, 5e-6)
        self.w_lr_rate = DoubleSpinBoxLabel(
            "LR Ratio Decayment", train_kwargs.lr_rate, 1e-6, 50, 6, 5e-6
        )
        self.w_lr_milestones = LineEditLabel(
            "LR Decayment Milestones",
            train_kwargs.milestones,
        )
        self.w_checkpoints = SpinBoxLabel(
            "Num of Checkpoints", train_kwargs.n_checkpoints, 0, 50
        )
        self.w_eval_metrics = LineEditLabel("Metrics for Eval", train_kwargs.metrics)
        window_layout.addWidget(self.w_seed, 0, 0)
        window_layout.addWidget(self.w_num_epochs, 0, 1)
        window_layout.addWidget(self.w_eval_epochs, 0, 2)
        window_layout.addWidget(self.w_eval_start, 0, 3)
        window_layout.addWidget(self.w_eval_batch_size, 1, 0)
        window_layout.addWidget(self.w_eval_size, 1, 1)
        window_layout.addWidget(self.w_num_bootstrap_samples, 1, 2)
        window_layout.addWidget(self.w_lr, 1, 3)
        window_layout.addWidget(self.w_lr_rate, 2, 0)
        window_layout.addWidget(self.w_checkpoints, 2, 1)
        window_layout.addWidget(self.w_lr_milestones, 2, 2)
        window_layout.addWidget(self.w_eval_metrics, 2, 3)
        self.setLayout(window_layout)

    def get_cfg(self):
        return dict(
            seed=self.w_seed.value(),
            num_epochs=self.w_num_epochs.value(),
            epochs_test_start=self.w_eval_start.value(),
            epochs_test=self.w_eval_epochs.value(),
            test_batch_size=self.w_eval_batch_size.value(),
            test_total_size=self.w_eval_size.value(),
            n_bootstrap_samples=self.w_num_bootstrap_samples.value(),
            lr=self.w_lr.value(),
            lr_rate=self.w_lr_rate.value(),
            milestones=self.w_lr_milestones.value().split(","),  # TODO: Check input
            n_checkpoints=self.w_checkpoints.value(),
            metrics=self.w_eval_metrics.value().split(","),
        )


class DataSection(QWidget):
    def __init__(self, data_kwargs):
        super(DataSection, self).__init__()
        window_layout = get_grid_layout(2, 4, [1] * 2, [1] * 4)
        self.w_num_workers = SpinBoxLabel(
            "Num of Workers", data_kwargs.num_workers, 1, cpu_count()
        )
        self.w_data_size = SpinBoxLabel(
            "Num of Graph Samples", data_kwargs.data_size, 1, 1e9
        )
        self.w_batch_size = SpinBoxLabel(
            "Data Loader Batch Size", data_kwargs.batch_size, 1, 1e9
        )
        self.w_source_path = LineEditLabel("Source Path", data_kwargs.source_path)
        self.w_min_node = SpinBoxLabel(
            "Min Num of Nodes", data_kwargs.min_num_node, 2, 1e9
        )
        self.w_max_node = SpinBoxLabel(
            "Max Num of Nodes", data_kwargs.max_num_node, 2, 1e9
        )
        self.w_max_prev_node = SpinBoxLabel(
            "Max Prev Nodes", data_kwargs.max_prev_node, 2, 1e9
        )
        self.w_check_size = QCheckBox("Check the num of nodes")
        self.w_check_size.setChecked(data_kwargs.check_size)
        window_layout.addWidget(self.w_num_workers, 0, 0)
        window_layout.addWidget(self.w_data_size, 0, 1)
        window_layout.addWidget(self.w_batch_size, 0, 2)
        window_layout.addWidget(self.w_source_path, 0, 3)
        window_layout.addWidget(self.w_min_node, 1, 0)
        window_layout.addWidget(self.w_max_node, 1, 1)
        window_layout.addWidget(self.w_max_prev_node, 1, 2)
        window_layout.addWidget(self.w_check_size, 1, 3)
        self.setLayout(window_layout)

    def get_cfg(self):
        return dict(
            num_workers=self.w_num_workers.value(),
            data_size=self.w_data_size.value(),
            batch_size=self.w_batch_size.value(),
            min_num_node=self.w_min_node.value(),
            max_num_node=self.w_max_node.value(),
            max_prev_node=self.w_max_prev_node.value(),
            check_size=self.w_check_size.isChecked(),
            inplace=False,
            source_path=self.w_source_path.value(),
        )


class TrainTab(QWidget):
    def __init__(self, cfg, mlf_dir):
        super(TrainTab, self).__init__()
        num_cols = 1
        num_rows = 9
        row_ratios = [1] * num_rows
        col_ratios = [1] * num_cols
        self.window_layout = get_grid_layout(num_rows, num_cols, row_ratios, col_ratios)
        self.progress_sec = ProgressBarButton(self.run_training)
        self.mlflow_sec = MLFlowSection(cfg.mlflow, mlf_dir)
        self.data_sec = DataSection(cfg.data)
        self.model_sec = ModelSection(cfg.model)
        self.training_sec = TrainingSection(cfg.training)
        self.window_layout.addWidget(
            get_label("MLFlow Parameters", 12, True),
            0,
            0,
            Qt.AlignLeft | Qt.AlignBottom,
        )
        self.window_layout.addWidget(
            get_label("Data Parameters", 12, True), 2, 0, Qt.AlignLeft | Qt.AlignBottom
        )
        self.window_layout.addWidget(
            get_label("Model Parameters", 12, True), 4, 0, Qt.AlignLeft | Qt.AlignBottom
        )
        self.window_layout.addWidget(
            get_label("Training Parameters", 12, True),
            6,
            0,
            Qt.AlignLeft | Qt.AlignBottom,
        )
        self.window_layout.addWidget(self.mlflow_sec, 1, 0)
        self.window_layout.addWidget(self.data_sec, 3, 0)
        self.window_layout.addWidget(self.model_sec, 5, 0)
        self.window_layout.addWidget(self.training_sec, 7, 0)
        self.window_layout.addWidget(self.progress_sec, 8, 0)
        self.setLayout(self.window_layout)

    def run_training(self, progress_bar):
        cfg = {}
        cfg["mlflow"] = self.mlflow_sec.get_cfg()
        cfg["data"] = self.data_sec.get_cfg()
        cfg["training"] = self.training_sec.get_cfg()
        cfg["model"] = self.model_sec.get_cfg()
        mlf_dir = cfg["mlflow"].pop("mlf_dir")
        cfg = OmegaConf.create(cfg)
        train(cfg, mlf_dir, progress_bar)
