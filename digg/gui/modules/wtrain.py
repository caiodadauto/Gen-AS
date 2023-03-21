from os import getcwd, listdir
from os.path import expanduser, join, abspath

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from PySide6.QtWidgets import QWidget, QFileDialog
from PySide6.QtCore import QThread, QObject, Signal, Slot

from digg.generator.training import train


class TrainingConfig(QWidget):
    def __init__(self, ui):
        super(TrainingConfig, self).__init__()
        cwd = getcwd()
        config_name = "config.yaml"
        if config_name in [p for p in listdir(cwd) if p.endswith("yaml")]:
            config_dir = cwd
        else:
            config_dir = join(expanduser("~"), ".config", "digg_dggm")
        initialize_config_dir(
            version_base=None, config_dir=config_dir, job_name="train"
        )
        self.default_mlf_dir = cwd
        self.default_cfg = compose(config_name=config_name)

        self.spin_seed_tr = ui.spin_seed_tr
        self.spin_num_ep_tr = ui.spin_num_ep_tr
        self.spin_ep_to_ev_tr = ui.spin_ep_to_ev_tr
        self.spin_n_ckt_tr = ui.spin_n_ckt_tr
        self.spin_batch_ev_tr = ui.spin_batch_ev_tr
        self.spin_n_graphs_ev_tr = ui.spin_n_graphs_ev_tr
        self.spin_bs_samples_tr = ui.spin_bs_samples_tr
        self.spin_init_lr_tr = ui.spin_init_lr_tr
        self.spin_lr_decay_tr = ui.spin_lr_decay_tr
        self.line_lr_mile_tr = ui.line_lr_mile_tr
        self.check_a_ev_tr = ui.check_a_ev_tr
        self.check_b_ev_tr = ui.check_b_ev_tr
        self.check_c_ev_tr = ui.check_c_ev_tr
        self.check_d_ev_tr = ui.check_d_ev_tr
        self.check_pr_ev_tr = ui.check_pr_ev_tr
        self.spin_seed_tr.setRange(1, 1e9)
        self.spin_num_ep_tr.setRange(1, 1e4)
        self.spin_ep_to_ev_tr.setRange(1, 1e4)
        self.spin_n_ckt_tr.setRange(0, 100)
        self.spin_batch_ev_tr.setRange(1, 1e4)
        self.spin_n_graphs_ev_tr.setRange(1, 1e4)
        self.spin_bs_samples_tr.setRange(1, 1e4)
        self.spin_init_lr_tr.setRange(1e-6, 50)
        self.spin_init_lr_tr.setDecimals(6)
        self.spin_init_lr_tr.setSingleStep(5e-6)
        self.spin_lr_decay_tr.setRange(1e-6, 50)
        self.spin_lr_decay_tr.setDecimals(6)
        self.spin_lr_decay_tr.setSingleStep(5e-6)

        self.spin_in_hd_tr = ui.spin_in_hd_tr
        self.spin_out_hd_tr = ui.spin_out_hd_tr
        self.spin_in_emb_tr = ui.spin_in_emb_tr
        self.spin_out_emb_tr = ui.spin_out_emb_tr
        self.spin_out_mlp_emb_tr = ui.spin_out_mlp_emb_tr
        self.spin_layers_tr = ui.spin_layers_tr
        self.spin_in_hd_tr.setRange(1, 1e4)
        self.spin_out_hd_tr.setRange(1, 1e4)
        self.spin_in_emb_tr.setRange(1, 1e4)
        self.spin_out_emb_tr.setRange(1, 1e4)
        self.spin_out_mlp_emb_tr.setRange(1, 1e4)
        self.spin_layers_tr.setRange(1, 1e4)

        self.spin_n_workers_tr = ui.spin_n_workers_tr
        self.spin_n_graphs_tr = ui.spin_n_graphs_tr
        self.spin_min_nodes_tr = ui.spin_min_nodes_tr
        self.spin_max_nodes_tr = ui.spin_max_nodes_tr
        self.spin_nodes_prev_tr = ui.spin_nodes_prev_tr
        self.spin_batch_loader_tr = ui.spin_batch_loader_tr
        self.check_num_nodes_tr = ui.check_num_nodes_tr
        self.btn_data_path_tr = ui.btn_data_path_tr
        self.line_data_path_tr = ui.line_data_path_tr
        self.btn_data_path_tr.pressed.connect(self.search_training_data)
        self.spin_n_workers_tr.setRange(1, 100)
        self.spin_batch_loader_tr.setRange(1, 1e4)
        self.spin_n_graphs_tr.setRange(1, 1e7)
        self.spin_min_nodes_tr.setRange(1, 1e4)
        self.spin_max_nodes_tr.setRange(1, 1e4)
        self.spin_nodes_prev_tr.setRange(1, 1e4)

        self.line_exp_name_tr = ui.line_exp_name_tr
        self.btn_root_path_tr = ui.btn_root_path_tr
        self.line_root_path_tr = ui.line_root_path_tr
        self.btn_root_path_tr.pressed.connect(self.search_mlf_root)
        self.set_defaults()

    def search_training_data(self):
        dir_name = QFileDialog().getExistingDirectory(self)
        self.line_data_path_tr.setText(dir_name)

    def search_mlf_root(self):
        dir_name = QFileDialog().getExistingDirectory(self)
        self.line_root_path_tr.setText(dir_name)

    def set_defaults(self):
        self.spin_seed_tr.setValue(self.default_cfg["training"]["seed"])
        self.spin_num_ep_tr.setValue(self.default_cfg["training"]["num_epochs"])
        self.spin_ep_to_ev_tr.setValue(self.default_cfg["training"]["epochs_test"])
        self.spin_n_ckt_tr.setValue(self.default_cfg["training"]["n_checkpoints"])
        self.spin_batch_ev_tr.setValue(self.default_cfg["training"]["test_batch_size"])
        self.spin_n_graphs_ev_tr.setValue(
            self.default_cfg["training"]["test_total_size"]
        )
        self.spin_bs_samples_tr.setValue(
            self.default_cfg["training"]["n_bootstrap_samples"]
        )
        self.spin_init_lr_tr.setValue(self.default_cfg["training"]["lr"])
        self.spin_lr_decay_tr.setValue(self.default_cfg["training"]["lr_rate"])
        self.line_lr_mile_tr.setText(
            ",".join(map(str, self.default_cfg["training"]["milestones"]))
        )
        for metric_name in self.default_cfg["training"]["metrics"]:
            if metric_name == "degree":
                self.check_d_ev_tr.setChecked(True)
            elif metric_name == "assortativity":
                self.check_a_ev_tr.setChecked(True)
            elif metric_name == "betweenness":
                self.check_b_ev_tr.setChecked(True)
            elif metric_name == "clustering":
                self.check_c_ev_tr.setChecked(True)
            elif metric_name == "pagerank":
                self.check_pr_ev_tr.setChecked(True)
        self.spin_in_hd_tr.setValue(self.default_cfg["model"]["hidden_size_rnn"])
        self.spin_out_hd_tr.setValue(
            self.default_cfg["model"]["hidden_size_rnn_output"]
        )
        self.spin_in_emb_tr.setValue(self.default_cfg["model"]["embedding_size_rnn"])
        self.spin_out_emb_tr.setValue(
            self.default_cfg["model"]["embedding_size_rnn_output"]
        )
        self.spin_out_mlp_emb_tr.setValue(
            self.default_cfg["model"]["embedding_size_output"]
        )
        self.spin_layers_tr.setValue(self.default_cfg["model"]["num_layer"])
        self.spin_n_workers_tr.setValue(self.default_cfg["data"]["num_workers"])
        self.spin_n_graphs_tr.setValue(self.default_cfg["data"]["data_size"])
        self.spin_min_nodes_tr.setValue(self.default_cfg["data"]["min_num_node"])
        self.spin_max_nodes_tr.setValue(self.default_cfg["data"]["max_num_node"])
        self.spin_batch_loader_tr.setValue(self.default_cfg["data"]["batch_size"])
        self.spin_nodes_prev_tr.setValue(self.default_cfg["data"]["max_prev_node"])
        self.check_num_nodes_tr.setChecked(self.default_cfg["data"]["check_size"])
        self.line_data_path_tr.setText(abspath(self.default_cfg["data"]["source_path"]))
        self.line_exp_name_tr.setText(self.default_cfg["mlflow"]["exp_name"])
        self.line_root_path_tr.setText(self.default_mlf_dir)

    def get_training_cfg(self):
        metrics = []
        if self.check_d_ev_tr.isChecked():
            metrics.append("degree")
        if self.check_a_ev_tr.isChecked():
            metrics.append("assortativity")
        if self.check_b_ev_tr.isChecked():
            metrics.append("betweenness")
        if self.check_c_ev_tr.isChecked():
            metrics.append("clustering")
        if self.check_pr_ev_tr.isChecked():
            metrics.append("pagerank")
        return dict(
            seed=self.spin_seed_tr.value(),
            num_epochs=self.spin_num_ep_tr.value(),
            epochs_test_start=1,
            epochs_test=self.spin_ep_to_ev_tr.value(),
            test_batch_size=self.spin_batch_ev_tr.value(),
            test_total_size=self.spin_n_graphs_tr.value(),
            n_bootstrap_samples=self.spin_bs_samples_tr.value(),
            lr=self.spin_init_lr_tr.value(),
            lr_rate=self.spin_lr_decay_tr.value(),
            milestones=list(
                map(int, self.line_lr_mile_tr.text().split(","))
            ),  # TODO: Check input
            n_checkpoints=self.spin_n_ckt_tr.value(),
            metrics=metrics,
        )

    def get_model_cfg(self):
        return dict(
            hidden_size_rnn=self.spin_in_hd_tr.value(),
            hidden_size_rnn_output=self.spin_out_hd_tr.value(),
            embedding_size_rnn=self.spin_in_emb_tr.value(),
            embedding_size_rnn_output=self.spin_out_emb_tr.value(),
            embedding_size_output=self.spin_out_mlp_emb_tr.value(),
            num_layer=self.spin_layers_tr.value(),
        )

    def get_data_cfg(self):
        return dict(
            num_workers=self.spin_n_workers_tr.value(),
            data_size=self.spin_n_graphs_tr.value(),
            batch_size=self.spin_batch_loader_tr.value(),
            min_num_node=self.spin_min_nodes_tr.value(),
            max_num_node=self.spin_max_nodes_tr.value(),
            max_prev_node=self.spin_nodes_prev_tr.value(),
            check_size=self.check_num_nodes_tr.isChecked(),
            inplace=False,
            source_path=self.line_data_path_tr.text(),
        )

    def get_mlflow_cfg(self):
        return dict(
            exp_name=self.line_exp_name_tr.text(), mlf_dir=self.line_root_path_tr.text()
        )


class TrackRunning:
    def __init__(self, progress):
        super(TrackRunning, self).__init__()
        self.stop_running = False
        self.progress = progress

    def emit(self, value):
        self.progress.emit(value)


class AppTrainingWindow(QObject):
    def __init__(self, ui, training_config):
        super(AppTrainingWindow, self).__init__()
        self.btn_style_before_run = "QPushButton { font-size: 14px; color: #282a36; background-color: #69FF94; } QPushButton:hover { background-color: #50fa7b; } QPushButton:pressed { background-color: #69FF94; }"
        self.btn_style_after_run = "QPushButton { color: #282a36; background-color: #FF6E6E; } QPushButton:hover {	background-color: #ff5555; } QPushButton:pressed { background-color: #FF6E6E; }"
        self.btn_run_tr = ui.btn_run_tr
        self.training_config = training_config
        self.progress_bar = ui.progress_tr
        self.progress_bar.setValue(0)
        self.label_progress_bar = ui.label_progress_tr
        self.label_sufix = ui.label_sufix_tr
        self.btn_run_tr.setStyleSheet(self.btn_style_before_run)
        self.crete_worker()
        print("INIT")

    def crete_worker(self):
        self.worker = WorkerTraining(
            self.btn_run_tr,
            self.btn_style_before_run,
            self.btn_style_after_run,
            self.stop_running,
            self.training_config,
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()
        # self.worker.finished.connect(self.worker_thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker.progress.connect(self.update_gui)
        # self.parent().aboutToQuit.connect(self.wo)
        self.btn_run_tr.clicked.connect(self.worker.run_training)

    def update_gui(self, sig):
        if sig.startswith("set"):
            self.progress_bar.setValue(int(sig.split("|")[-1]))
        elif sig.startswith("max"):
            self.progress_bar.setMaximum(int(sig.split("|")[-1]))
        elif sig.startswith("desc"):
            self.label_progress_bar.setText(sig.split("|")[-1])
        elif sig.startswith("sufix"):
            self.label_sufix.setText(sig.split("|")[-1])

    def stop_running(self):
        if self.btn_run_tr.text() == "Stop":
            self.worker.track_running.stop_running = True
            self.btn_run_tr.setText("Run")
            self.btn_run_tr.setStyleSheet(self.btn_style_before_run)
            # print("FINISHED", self.worker_thread.isFinished())
            # print("RUNNING", self.worker_thread.isRunning())
            self.btn_run_tr.clicked.disconnect()
            self.btn_run_tr.clicked.connect(self.worker.run_training)


class WorkerTraining(QObject):
    # finished = Signal()
    progress = Signal(str)

    def __init__(
        self,
        btn_run_tr,
        btn_style_before_run,
        btn_style_after_run,
        stop_running_fn,
        training_config,
    ):
        super(self.__class__, self).__init__()
        self.btn_run_tr = btn_run_tr
        self.btn_style_before_run = btn_style_before_run
        self.btn_style_after_run = btn_style_after_run
        self.stop_running_fn = stop_running_fn
        self.training_config = training_config
        self.track_running = TrackRunning(self.progress)

    @Slot()
    def run_training(self):
        if self.btn_run_tr.text() == "Run":
            self.btn_run_tr.setStyleSheet(self.btn_style_after_run)
            self.btn_run_tr.setText("Stop")
            self.btn_run_tr.clicked.disconnect()
            self.btn_run_tr.clicked.connect(self.stop_running_fn)
            cfg = {}
            cfg["mlflow"] = self.training_config.get_mlflow_cfg()
            cfg["data"] = self.training_config.get_data_cfg()
            cfg["training"] = self.training_config.get_training_cfg()
            cfg["model"] = self.training_config.get_model_cfg()
            mlf_dir = cfg["mlflow"].pop("mlf_dir")
            cfg = OmegaConf.create(cfg)
            train(
                cfg=cfg,
                run_dir=mlf_dir,
                progress_bar_qt=self.track_running,
            )
            # self.finished.emit()
