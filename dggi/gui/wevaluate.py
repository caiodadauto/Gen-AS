import re
from os.path import join
from os import listdir
from posixpath import basename

import numpy as np
import pandas as pd
import mlflow as mlf
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PySide6.QtWidgets import QWidget, QTreeWidgetItem, QMainWindow
from PySide6.QtCore import QThread, QObject, Signal, Slot, Qt, QSize
from PySide6.QtGui import QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PIL import Image

from dggi.generator.evaluation import evaluate
from dggi.generator.mlf_utils import (
    mlf_set_env,
    mlf_get_run_list,
    mlf_get_evaluation_imgs,
)
from dggi.gui.ui_vis_eval import Ui_VisEvalWindow


class EvaluationConfig(QWidget):
    def __init__(self, ui, global_config):
        super(EvaluationConfig, self).__init__()
        self.global_config = global_config
        self.spin_seed_ev = ui.spin_seed_ev
        self.spin_num_graphs_ev = ui.spin_num_graphs_ev
        self.spin_batch_size_ev = ui.spin_batch_size_ev
        self.spin_bs_samples_ev = ui.spin_bs_samples_ev
        self.check_a_ev_ev = ui.check_a_ev_ev
        self.check_b_ev_ev = ui.check_b_ev_ev
        self.check_c_ev_ev = ui.check_c_ev_ev
        self.check_d_ev_ev = ui.check_d_ev_ev
        self.check_pr_ev_ev = ui.check_pr_ev_ev
        self.spin_seed_ev.setRange(1, 1e9)
        self.spin_num_graphs_ev.setRange(1, 1e6)
        self.spin_batch_size_ev.setRange(1, 1e4)
        self.spin_bs_samples_ev.setRange(1, 1e4)
        self.set_defaults()
        self.update_config()

    def set_defaults(self):
        self.spin_seed_ev.setValue(self.global_config.default_cfg["evaluation"]["seed"])
        self.spin_num_graphs_ev.setValue(
            self.global_config.default_cfg["evaluation"]["test_total_size"]
        )
        self.spin_batch_size_ev.setValue(
            self.global_config.default_cfg["evaluation"]["test_batch_size"]
        )
        self.spin_bs_samples_ev.setValue(
            self.global_config.default_cfg["evaluation"]["n_bootstrap_samples"]
        )

        self.check_d_ev_ev.setChecked(False)
        self.check_a_ev_ev.setChecked(False)
        self.check_b_ev_ev.setChecked(False)
        self.check_c_ev_ev.setChecked(False)
        self.check_pr_ev_ev.setChecked(False)
        for metric_name in self.global_config.default_cfg["evaluation"]["metrics"]:
            if metric_name == "degree":
                self.check_d_ev_ev.setChecked(True)
            elif metric_name == "assortativity":
                self.check_a_ev_ev.setChecked(True)
            elif metric_name == "betweenness":
                self.check_b_ev_ev.setChecked(True)
            elif metric_name == "clustering":
                self.check_c_ev_ev.setChecked(True)
            elif metric_name == "pagerank":
                self.check_pr_ev_ev.setChecked(True)

    def update_config(self):
        cfg = {}
        cfg["mlflow"] = self.get_mlflow_cfg()
        cfg["evaluation"] = self.get_evaluation_cfg()
        self.mlf_dir = cfg["mlflow"].pop("mlf_dir")
        self.cfg = OmegaConf.create(cfg)

    def get_evaluation_cfg(self):
        metrics = []
        if self.check_d_ev_ev.isChecked():
            metrics.append("degree")
        if self.check_a_ev_ev.isChecked():
            metrics.append("assortativity")
        if self.check_b_ev_ev.isChecked():
            metrics.append("betweenness")
        if self.check_c_ev_ev.isChecked():
            metrics.append("clustering")
        if self.check_pr_ev_ev.isChecked():
            metrics.append("pagerank")
        return dict(
            seed=self.spin_seed_ev.value(),
            test_total_size=self.spin_num_graphs_ev.value(),
            test_batch_size=self.spin_batch_size_ev.value(),
            n_bootstrap_samples=self.spin_bs_samples_ev.value(),
            metrics=metrics,
        )

    def get_mlflow_cfg(self):
        return dict(
            exp_name=self.global_config.default_cfg["mlflow"]["exp_name"],
            mlf_dir=self.global_config.default_mlf_dir,
        )


class TrackRunning:
    def __init__(self, progress_sig):
        super(TrackRunning, self).__init__()
        self.stop_running = False
        self.running = False
        self.progress_sig = progress_sig

    def emit(self, value):
        self.progress_sig.emit(value)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=4, height=4, dpi=150):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def __del__(self):
        plt.close(self.fig)


class AppEvaluationWindow(QObject):
    def __init__(self, ui, evaluation_config, parent):
        super(AppEvaluationWindow, self).__init__(parent=parent)
        self.btn_style_before_run = "QPushButton { font-size: 14px; color: #282a36; background-color: #69FF94; } QPushButton:hover { background-color: #50fa7b; } QPushButton:pressed { background-color: #69FF94; }"
        self.btn_style_after_run = "QPushButton { font-size: 14px; color: #282a36; background-color: #FF6E6E; } QPushButton:hover {	background-color: #ff5555; } QPushButton:pressed { background-color: #FF6E6E; }"
        self.btn_style_stopping = "QPushButton { font-size: 14px; color: #282a36; background-color: #FFFFA5; } QPushButton:hover {	background-color: #F1FA8C; } QPushButton:pressed { background-color: #FFFFA5; }"
        self.btn_style_neutral = "QPushButton { border: 2px solid rgb(52, 59, 72); border-radius: 5px;	background-color: rgb(52, 59, 72); } QPushButton:hover { background-color: rgb(57, 65, 80); border: 2px solid rgb(61, 70, 86); } QPushButton:pressed {	background-color: rgb(35, 40, 49); border: 2px solid rgb(43, 50, 61); }"
        self.re_datetime = re.compile(
            "(\d\d)(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(\d\d)([0-5]\d|60)([0-5]\d|60)"
        )
        self.evaluation_img_paths = None
        self.icon_ban = QIcon()
        self.icon_diag = QIcon()
        self.icon_ban.addFile(
            ":/icons/images/icons/cil-ban.png", QSize(), QIcon.Normal, QIcon.Off
        )
        self.icon_diag.addFile(
            ":/icons/images/icons/icon_project_diagram.png",
            QSize(),
            QIcon.Normal,
            QIcon.Off,
        )
        self.tree_ev = ui.tree_ev
        self.btn_run_ev = ui.btn_run_ev
        self.btn_vis_ev = ui.btn_vis_ev
        self.label_tree_ev = ui.label_tree_ev
        self.evaluation_config = evaluation_config
        self.progress_bar = ui.progress_ev
        self.label_progress_bar = ui.label_progress_ev
        self.label_suffix = ui.label_suffix_ev
        self.label_progress_bar.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.progress_bar.setValue(0)
        self.mlf_run = None
        self.runs_df = mlf_set_env(
            1234,
            self.evaluation_config.global_config.default_mlf_dir,
            self.evaluation_config.global_config.default_cfg["mlflow"]["exp_name"],
            root_dir=".",
            fix_path=True,
            return_runs_list=True,
        )
        self.label_tree_ev.setStyleSheet("QLabel { font-size: 16px; }")
        self.btn_run_ev.setIcon(self.icon_ban)
        self.btn_vis_ev.setIcon(self.icon_ban)
        self.btn_run_ev.setText("  No run selected")
        self.btn_vis_ev.setText("  No visualization")
        self.tree_ev.itemDoubleClicked.connect(self.get_run)
        self.create_runner_thread()
        self.create_stopper_thread()
        self.parent().aboutToQuit.connect(self.quit_threads)
        self.btn_run_ev.clicked.connect(self.runner.run_evaluation)
        self.btn_vis_ev.clicked.connect(self.open_vis_window)
        self.load_runs()

    def get_graph_paths(self, path):
        try:
            graph_paths = sorted(
                [join(path, p) for p in listdir(path) if p.endswith(".gpickle")],
                key=lambda s: int(basename(s).split(".")[0]),
            )
        except FileNotFoundError:
            return []
        return graph_paths

    # def save_graph(self):
    #     if len(self.graph_paths) > 0:
    #         graph_path = self.graph_paths[self.graph_idx]
    #         file_name = QFileDialog().getSaveFileName(
    #             self.evaluation_config,
    #             "Save Graph",
    #             f"{basename(graph_path).split('.')[0]}.png",
    #         )[0]
    #         fig = plt.Figure(figsize=(5, 4), dpi=150)
    #         ax = fig.add_subplot(111)
    #         graph = nx.read_gpickle(graph_path)
    #         pos = nx.nx_agraph.graphviz_layout(graph, prog="sfdp")
    #         nx.draw_networkx_nodes(
    #             graph,
    #             pos=pos,
    #             node_size=80,
    #             alpha=0.9,
    #             edgecolors="k",
    #             node_color="#8be9fd",
    #             ax=ax,
    #         )
    #         nx.draw_networkx_edges(graph, pos=pos, node_size=100, ax=ax)
    #         ax.spines["top"].set_visible(False)
    #         ax.spines["right"].set_visible(False)
    #         ax.spines["bottom"].set_visible(False)
    #         ax.spines["left"].set_visible(False)
    #         plt.show()
    #         plt.savefig(file_name)

    def update_graph_paths(self, path):
        self.graph_idx = 0
        self.graph_paths = self.get_graph_paths(path)
        self.draw_graph(increment=False)

    def open_vis_window(self):
        if self.evaluation_img_paths is not None and len(self.evaluation_img_paths) > 0:
            self.sc = None
            self.vis_window = QMainWindow()
            self.vis_ui = Ui_VisEvalWindow()
            self.vis_ui.setupUi(self.vis_window)
            combo_names = []
            for p in self.evaluation_img_paths:
                bname = basename(p)
                if bname.startswith("bar"):
                    combo_names.append("MMD Bar Plot")
                elif bname.startswith("line"):
                    metric_name = bname.split(".")[0].split("_")[-1].title()
                    combo_names.append(f"{metric_name} Metric Line Plot")
            self.vis_ui.comboBox.insertItems(0, combo_names)
            self.vis_ui.comboBox.currentIndexChanged.connect(self.draw_graph)
            self.vis_window.show()
            self.draw_graph(self.vis_ui.comboBox.currentIndex())

    def draw_graph(self, index, checked=False, increment=True):
        if self.sc is not None:
            self.vis_ui.layout_graph_vis.takeAt(0).widget().deleteLater()
            self.sc = None
            del self.sc
        self.sc = MplCanvas(self, width=5, height=4, dpi=150)
        self.vis_ui.layout_graph_vis.addWidget(self.sc)
        img = np.asarray(Image.open(self.evaluation_img_paths[index]))
        self.sc.ax.imshow(img)
        self.sc.ax.xaxis.set_visible(False)
        self.sc.ax.yaxis.set_visible(False)
        self.sc.ax.spines["top"].set_visible(False)
        self.sc.ax.spines["right"].set_visible(False)
        self.sc.ax.spines["bottom"].set_visible(False)
        self.sc.ax.spines["left"].set_visible(False)
        self.sc.fig.tight_layout()

    def check_for_visualization(self):
        self.evaluation_img_paths = sorted(mlf_get_evaluation_imgs(self.mlf_run))
        if len(self.evaluation_img_paths) > 0:
            self.btn_vis_ev.setIcon(self.icon_diag)
            self.btn_vis_ev.setText("  Visualization")
        else:
            self.btn_vis_ev.setIcon(self.icon_ban)
            self.btn_vis_ev.setText("  No visualization")

    def get_run(self, *args):
        item = args[0]
        run_name = item.text(0)
        if not self.runner.track_running.running:
            self.btn_run_ev.setIcon(QIcon())
            self.btn_run_ev.setText(f"Evaluate best Models in {run_name.rstrip()}")
            self.btn_run_ev.setStyleSheet(self.btn_style_before_run)
            run_id = self.runs_df.iloc[int(run_name.split(" ")[-1]) - 1]["Run ID"]
            self.mlf_run = mlf.get_run(run_id=run_id)
            self.runner.set_run(self.mlf_run)
            self.run_name = run_name
            self.check_for_visualization()

    def load_runs(self, update_runs=False):
        self.tree_ev.clear()
        if update_runs:
            experiment = mlf.get_experiment_by_name(
                self.evaluation_config.global_config.default_cfg["mlflow"]["exp_name"]
            )
            if experiment is not None:
                experiment_id = experiment.experiment_id
                self.runs_df = mlf_get_run_list(experiment_id)
            else:
                self.runs_df = pd.DataFrame()
        if not self.runs_df.empty:
            number_cols = self.runs_df.columns[3:]
            _runs_df = self.runs_df.copy()
            _runs_df.loc[:, number_cols] = self.runs_df.loc[:, number_cols].applymap(
                lambda x: "{0:.3f}".format(x)
            )
            columns = [""] + [
                "",
                "Start Time",
                "Duration (h)",
                "Final Loss",
            ]
            for metric_name in _runs_df.columns[5:]:
                columns.append("Best " + metric_name.split(" ")[1].title() + " MMD")
            self.tree_ev.setHeaderLabels(columns)
            for i, row in _runs_df.iterrows():
                row_list = row.iloc[2:].values.tolist()
                row_list = [f"RUN {i + 1}"] + [""] + row_list
                tree_item = QTreeWidgetItem(row_list)
                self.tree_ev.addTopLevelItem(tree_item)
            for i in range(len(columns)):
                self.tree_ev.resizeColumnToContents(i)
            self.tree_ev.setSortingEnabled(True)
        else:
            pass

    def create_runner_thread(self):
        self.runner = EvaluationRunner(self.evaluation_config)
        self.runner_thread = QThread()
        self.runner.moveToThread(self.runner_thread)
        self.runner_thread.start()
        self.runner_thread.finished.connect(self.runner_thread.deleteLater)
        self.runner.progress_sig.connect(self.update_progress)
        self.runner.btn_sig.connect(self.update_btn)
        self.runner.set_run(self.mlf_run)

    def create_stopper_thread(self):
        self.stopper = EvaluationStopper()
        self.stopper_thread = QThread()
        self.stopper.moveToThread(self.stopper_thread)
        self.stopper.add_runner(self.runner_thread, self.runner.track_running)
        self.stopper_thread.start()
        self.stopper_thread.finished.connect(self.stopper_thread.deleteLater)
        self.stopper.btn_sig.connect(self.update_btn)

    def quit_threads(self):
        if self.runner_thread.isRunning():
            self.runner_thread.quit()
        if self.stopper_thread.isRunning():
            self.stopper_thread.quit()

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
            self.btn_run_ev.clicked.disconnect()
            self.btn_run_ev.clicked.connect(self.stopper.stop_evaluation)
            self.btn_run_ev.setStyleSheet(self.btn_style_after_run)
            self.btn_run_ev.setText("Stop")
        elif sig == "stopping":
            self.btn_run_ev.setStyleSheet(self.btn_style_stopping)
            self.btn_run_ev.setText("Stopping")
        elif sig == "stop":
            self.create_runner_thread()
            self.stopper.add_runner(self.runner_thread, self.runner.track_running)
            self.btn_run_ev.clicked.disconnect()
            self.btn_run_ev.clicked.connect(self.runner.run_evaluation)
            if self.run_name is None:
                self.btn_run_ev.setStyleSheet("")
                self.btn_run_ev.setIcon(self.icon_ban)
                self.btn_run_ev.setText("  No run selected")
            else:
                self.btn_run_ev.setStyleSheet(self.btn_style_before_run)
                self.btn_run_ev.setIcon(QIcon())
                self.btn_run_ev.setText(
                    f"Evaluate best Modes in {self.run_name.rstrip()}"
                )
        elif sig == "endrun":
            self.btn_run_ev.clicked.disconnect()
            self.btn_run_ev.clicked.connect(self.runner.run_evaluation)
            self.check_for_visualization()
            if self.run_name is None:
                self.btn_run_ev.setStyleSheet("")
                self.btn_run_ev.setIcon(self.icon_ban)
                self.btn_run_ev.setText("  No run selected")
            else:
                self.btn_run_ev.setStyleSheet(self.btn_style_before_run)
                self.btn_run_ev.setIcon(QIcon())
                self.btn_run_ev.setText(
                    f"Evaluate best Modes in {self.run_name.rstrip()}"
                )


class EvaluationRunner(QObject):
    progress_sig = Signal(str)
    btn_sig = Signal(str)

    def __init__(self, evaluation_config):
        super(self.__class__, self).__init__()
        self.evaluation_config = evaluation_config
        self.track_running = TrackRunning(self.progress_sig)

    def set_run(self, mlf_run):
        self.mlf_run = mlf_run

    @Slot()
    def run_evaluation(self):
        if not self.track_running.running and self.mlf_run is not None:
            self.progress_sig.emit("suffix|")
            self.track_running.stop_running = False
            self.track_running.running = True
            self.btn_sig.emit("run")
            evaluate(
                self.evaluation_config.cfg,
                None,
                self.evaluation_config.mlf_dir,
                mlf_run=self.mlf_run,
                progress_bar_qt=self.track_running,
            )
            self.track_running.running = False
            self.btn_sig.emit("endrun")


class EvaluationStopper(QObject):
    btn_sig = Signal(str)

    def __init__(self):
        super(self.__class__, self).__init__()

    def add_runner(self, runner_thread, track_running):
        self.runner_thread = runner_thread
        self.track_running = track_running

    @Slot()
    def stop_evaluation(self):
        if not self.track_running.stop_running and self.track_running.running:
            self.track_running.stop_running = True
            self.runner_thread.quit()
            self.btn_sig.emit("stopping")
            self.runner_thread.wait()
            self.runner_thread = None
            self.track_running = None
            self.btn_sig.emit("stop")
