import subprocess
from os import getcwd
from os.path import abspath

from omegaconf import OmegaConf
from hydra import compose
from PySide6.QtWidgets import QWidget, QFileDialog
from PySide6.QtCore import QThread, QObject, Signal, Slot, Qt

from dggi.generator.training import train


class TrainingConfig(QWidget):
    def __init__(self, ui, global_config):
        super(TrainingConfig, self).__init__()
        self.global_config = global_config
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

        # self.line_exp_name_tr = ui.line_exp_name_tr
        # self.btn_root_path_tr = ui.btn_root_path_tr
        # self.line_root_path_tr = ui.line_root_path_tr
        # self.btn_root_path_tr.pressed.connect(self.search_mlf_root)
        self.set_defaults()
        self.update_config()

    def search_training_data(self):
        dir_name = QFileDialog().getExistingDirectory(self)
        self.line_data_path_tr.setText(dir_name)

    # def search_mlf_root(self):
    #     dir_name = QFileDialog().getExistingDirectory(self)
    #     self.line_root_path_tr.setText(dir_name)

    def set_defaults(self):
        self.spin_seed_tr.setValue(self.global_config.default_cfg["training"]["seed"])
        self.spin_num_ep_tr.setValue(
            self.global_config.default_cfg["training"]["num_epochs"]
        )
        self.spin_ep_to_ev_tr.setValue(
            self.global_config.default_cfg["training"]["epochs_test"]
        )
        self.spin_n_ckt_tr.setValue(
            self.global_config.default_cfg["training"]["n_checkpoints"]
        )
        self.spin_batch_ev_tr.setValue(
            self.global_config.default_cfg["training"]["test_batch_size"]
        )
        self.spin_n_graphs_ev_tr.setValue(
            self.global_config.default_cfg["training"]["test_total_size"]
        )
        self.spin_bs_samples_tr.setValue(
            self.global_config.default_cfg["training"]["n_bootstrap_samples"]
        )
        self.spin_init_lr_tr.setValue(self.global_config.default_cfg["training"]["lr"])
        self.spin_lr_decay_tr.setValue(
            self.global_config.default_cfg["training"]["lr_rate"]
        )
        self.line_lr_mile_tr.setText(
            ",".join(map(str, self.global_config.default_cfg["training"]["milestones"]))
        )
        self.check_d_ev_tr.setChecked(False)
        self.check_a_ev_tr.setChecked(False)
        self.check_b_ev_tr.setChecked(False)
        self.check_c_ev_tr.setChecked(False)
        self.check_pr_ev_tr.setChecked(False)
        for metric_name in self.global_config.default_cfg["training"]["metrics"]:
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
        self.spin_in_hd_tr.setValue(
            self.global_config.default_cfg["model"]["hidden_size_rnn"]
        )
        self.spin_out_hd_tr.setValue(
            self.global_config.default_cfg["model"]["hidden_size_rnn_output"]
        )
        self.spin_in_emb_tr.setValue(
            self.global_config.default_cfg["model"]["embedding_size_rnn"]
        )
        self.spin_out_emb_tr.setValue(
            self.global_config.default_cfg["model"]["embedding_size_rnn_output"]
        )
        self.spin_out_mlp_emb_tr.setValue(
            self.global_config.default_cfg["model"]["embedding_size_output"]
        )
        self.spin_layers_tr.setValue(
            self.global_config.default_cfg["model"]["num_layer"]
        )
        self.spin_n_workers_tr.setValue(
            self.global_config.default_cfg["data"]["num_workers"]
        )
        self.spin_n_graphs_tr.setValue(
            self.global_config.default_cfg["data"]["data_size"]
        )
        self.spin_min_nodes_tr.setValue(
            self.global_config.default_cfg["data"]["min_num_node"]
        )
        self.spin_max_nodes_tr.setValue(
            self.global_config.default_cfg["data"]["max_num_node"]
        )
        self.spin_batch_loader_tr.setValue(
            self.global_config.default_cfg["data"]["batch_size"]
        )
        self.spin_nodes_prev_tr.setValue(
            self.global_config.default_cfg["data"]["max_prev_node"]
        )
        self.check_num_nodes_tr.setChecked(
            self.global_config.default_cfg["data"]["check_size"]
        )
        self.line_data_path_tr.setText(
            abspath(self.global_config.default_cfg["data"]["source_path"])
        )
        # self.line_exp_name_tr.setText(self.global_config.default_cfg["mlflow"]["exp_name"])
        # self.line_root_path_tr.setText(self.global_config.default_mlf_dir)

    def update_config(self):
        cfg = {}
        cfg["mlflow"] = self.get_mlflow_cfg()
        cfg["data"] = self.get_data_cfg()
        cfg["training"] = self.get_training_cfg()
        cfg["model"] = self.get_model_cfg()
        self.mlf_dir = cfg["mlflow"].pop("mlf_dir")
        self.cfg = OmegaConf.create(cfg)

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
            test_total_size=self.spin_n_graphs_ev_tr.value(),
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
            exp_name=self.global_config.default_cfg["mlflow"][
                "exp_name"
            ],  # line_exp_name_tr.text(),
            mlf_dir=self.global_config.default_mlf_dir,  # line_root_path_tr.text()
        )


class TrackRunning:
    def __init__(self, progress_sig):
        super(TrackRunning, self).__init__()
        self.stop_running = False
        self.running = False
        self.progress_sig = progress_sig

    def emit(self, value):
        self.progress_sig.emit(value)


class AppTrainingWindow(QObject):
    def __init__(self, ui, training_config, parent):
        super(AppTrainingWindow, self).__init__(parent=parent)
        self.btn_style_before_run = "QPushButton { font-size: 14px; color: #282a36; background-color: #69FF94; } QPushButton:hover { background-color: #50fa7b; } QPushButton:pressed { background-color: #69FF94; }"
        self.btn_style_after_run = "QPushButton { font-size: 14px; color: #282a36; background-color: #FF6E6E; } QPushButton:hover {	background-color: #ff5555; } QPushButton:pressed { background-color: #FF6E6E; }"
        self.btn_style_stopping = "QPushButton { font-size: 14px; color: #282a36; background-color: #FFFFA5; } QPushButton:hover {	background-color: #F1FA8C; } QPushButton:pressed { background-color: #FFFFA5; }"
        self.btn_run_tr = ui.btn_run_tr
        self.training_config = training_config
        self.progress_bar = ui.progress_tr
        self.progress_bar.setValue(0)
        self.label_progress_bar = ui.label_progress_tr
        self.label_suffix = ui.label_suffix_tr
        self.label_suffix.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.btn_run_tr.setStyleSheet(self.btn_style_before_run)
        self.create_runner_thread()
        self.create_stopper_thread()
        self.create_mlflow_ui_thread()
        self.parent().aboutToQuit.connect(self.quit_threads)
        self.parent().aboutToQuit.connect(self.quit_mlflow_ui)
        self.btn_run_tr.clicked.connect(self.runner.run_training)
        self.mlflow_ui.run()

    def create_runner_thread(self):
        self.runner = TrainRunner(self.training_config)
        self.runner_thread = QThread()
        self.runner.moveToThread(self.runner_thread)
        self.runner_thread.start()
        self.runner_thread.finished.connect(self.runner_thread.deleteLater)
        self.runner.progress_sig.connect(self.update_progress)
        self.runner.btn_sig.connect(self.update_btn)

    def create_stopper_thread(self):
        self.stopper = TrainStopper()
        self.stopper_thread = QThread()
        self.stopper.moveToThread(self.stopper_thread)
        self.stopper.add_runner(self.runner_thread, self.runner.track_running)
        self.stopper_thread.start()
        self.stopper_thread.finished.connect(self.stopper_thread.deleteLater)
        self.stopper.btn_sig.connect(self.update_btn)

    def create_mlflow_ui_thread(self):
        self.mlflow_ui = MlFlowUI()
        self.mlflow_ui_thread = QThread()
        self.mlflow_ui.moveToThread(self.mlflow_ui_thread)
        self.mlflow_ui_thread.start()
        self.mlflow_ui_thread.finished.connect(self.mlflow_ui_thread.deleteLater)

    def quit_threads(self):
        if self.runner_thread.isRunning():
            self.runner_thread.quit()
        if self.stopper_thread.isRunning():
            self.stopper_thread.quit()

    def quit_mlflow_ui(self):
        if self.mlflow_ui_thread.isRunning():
            subprocess.call(["pkill", "-f", "gunicorn"])
            self.mlflow_ui_thread.quit()

    def update_progress(self, sig):
        if sig.startswith("set"):
            self.progress_bar.setValue(int(sig.split("|")[-1]))
        elif sig.startswith("max"):
            self.progress_bar.setMaximum(int(sig.split("|")[-1]))
        elif sig.startswith("reset"):
            self.progress_bar.reset()
        elif sig.startswith("desc"):
            self.label_progress_bar.setText(
                f'<html><head/><body style="text-align: left;"><p><span style="font-size:11pt;">{sig.split("|")[-1]}</span></p></body></html>'
            )
        elif sig.startswith("suffix"):
            self.label_suffix.setText(
                f'<html><head/><body style="text-align: right;"><p><span style="font-size:11pt;">{sig.split("|")[-1]}</span></p></body></html>'
            )

    def update_btn(self, sig):
        if sig == "run":
            self.btn_run_tr.clicked.disconnect()
            self.btn_run_tr.clicked.connect(self.stopper.stop_training)
            self.btn_run_tr.setStyleSheet(self.btn_style_after_run)
            self.btn_run_tr.setText("Stop")
        elif sig == "stopping":
            self.btn_run_tr.setStyleSheet(self.btn_style_stopping)
            self.btn_run_tr.setText("Stopping")
        elif sig == "stop":
            print("STOP")
            self.create_runner_thread()
            self.stopper.add_runner(self.runner_thread, self.runner.track_running)
            # self.btn_run_tr.clicked.disconnect()
            self.btn_run_tr.clicked.connect(self.runner.run_training)
            self.btn_run_tr.setStyleSheet(self.btn_style_before_run)
            self.btn_run_tr.setText("Run")
        elif sig == "endrun":
            self.btn_run_tr.clicked.disconnect()
            self.btn_run_tr.clicked.connect(self.runner.run_training)
            self.btn_run_tr.setStyleSheet(self.btn_style_before_run)
            self.btn_run_tr.setText("Run")


class TrainRunner(QObject):
    progress_sig = Signal(str)
    btn_sig = Signal(str)

    def __init__(self, training_config):
        super(self.__class__, self).__init__()
        self.training_config = training_config
        self.track_running = TrackRunning(self.progress_sig)

    @Slot()
    def run_training(self):
        if not self.track_running.running:
            self.progress_sig.emit("desc|")
            self.progress_sig.emit("suffix|")
            self.track_running.stop_running = False
            self.track_running.running = True
            self.btn_sig.emit("run")
            train(
                cfg=self.training_config.cfg,
                run_dir=self.training_config.mlf_dir,
                progress_bar_qt=self.track_running,
            )
            self.track_running.running = False
            self.btn_sig.emit("endrun")


class TrainStopper(QObject):
    btn_sig = Signal(str)

    def __init__(self):
        super(self.__class__, self).__init__()

    def add_runner(self, runner_thread, track_running):
        self.runner_thread = runner_thread
        self.track_running = track_running

    @Slot()
    def stop_training(self):
        if not self.track_running.stop_running and self.track_running.running:
            self.track_running.stop_running = True
            self.runner_thread.quit()
            self.btn_sig.emit("stopping")
            self.runner_thread.wait()
            self.runner_thread = None
            self.track_running = None
            self.btn_sig.emit("stop")


class MlFlowUI(QObject):
    def __init__(self):
        super(self.__class__, self).__init__()

    def run(self):
        self.cmd = subprocess.Popen(["mlflow", "ui"])
