import re
from datetime import datetime
from os.path import join, isdir
from os import listdir
from posixpath import basename

import pandas as pd
import mlflow as mlf
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PySide6.QtWidgets import QWidget, QFileDialog, QTreeWidgetItem, QMainWindow
from PySide6.QtCore import QThread, QObject, Signal, Slot, Qt, QSize
from PySide6.QtGui import QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from dggi.generator.evaluation import generate
from dggi.generator.mlf_utils import mlf_set_env, mlf_get_run_list
from dggi.gui.ui_vis import Ui_VisWindow


class GenerationConfig(QWidget):
    def __init__(self, ui, global_config):
        super(GenerationConfig, self).__init__()
        self.global_config = global_config
        self.spin_seed_gr = ui.spin_seed_gr
        self.spin_min_num_nodes_gr = ui.spin_min_num_nodes_gr
        self.spin_max_num_nodes_gr = ui.spin_max_num_nodes_gr
        self.spin_num_graphs_gr = ui.spin_num_graphs_gr
        self.spin_batch_size_gr = ui.spin_batch_size_gr
        self.btn_save_loc_gr = ui.btn_save_loc_gr
        self.line_save_loc_gr = ui.line_save_loc_gr
        self.btn_save_loc_gr.pressed.connect(self.search_gr_location)
        self.spin_seed_gr.setRange(1, 1e9)
        self.spin_min_num_nodes_gr.setRange(1, 1e6)
        self.spin_max_num_nodes_gr.setRange(1, 1e6)
        self.spin_num_graphs_gr.setRange(1, 1e6)
        self.spin_batch_size_gr.setRange(1, 1e4)
        self.set_defaults()
        self.update_config()

    def search_gr_location(self):
        dir_name = QFileDialog().getExistingDirectory(self)
        self.line_save_loc_gr.setText(dir_name)

    def set_defaults(self):
        self.spin_seed_gr.setValue(self.global_config.default_cfg["generation"]["seed"])
        self.spin_min_num_nodes_gr.setValue(
            self.global_config.default_cfg["data"]["min_num_node"]
        )
        self.spin_max_num_nodes_gr.setValue(
            self.global_config.default_cfg["data"]["max_num_node"]
        )
        self.spin_num_graphs_gr.setValue(
            self.global_config.default_cfg["generation"]["test_total_size"]
        )
        self.spin_batch_size_gr.setValue(
            self.global_config.default_cfg["generation"]["test_batch_size"]
        )
        self.line_save_loc_gr.setText(
            self.global_config.default_cfg["generation"]["save_dir"]
        )

    def update_config(self):
        cfg = {}
        cfg["mlflow"] = self.get_mlflow_cfg()
        cfg["generation"] = self.get_generation_cfg()
        self.mlf_dir = cfg["mlflow"].pop("mlf_dir")
        self.min_num_nodes = cfg["generation"]["min_num_nodes"]
        self.max_num_nodes = cfg["generation"]["max_num_nodes"]
        self.cfg = OmegaConf.create(cfg)

    def get_generation_cfg(self):
        return dict(
            seed=self.spin_seed_gr.value(),
            test_total_size=self.spin_num_graphs_gr.value(),
            test_batch_size=self.spin_batch_size_gr.value(),
            save_dir=self.line_save_loc_gr.text(),
            min_num_nodes=self.spin_min_num_nodes_gr.value(),
            max_num_nodes=self.spin_max_num_nodes_gr.value(),
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


class AppGenerationWindow(QObject):
    def __init__(self, ui, generation_config, parent):
        super(AppGenerationWindow, self).__init__(parent=parent)
        self.btn_style_before_run = "QPushButton { font-size: 14px; color: #282a36; background-color: #69FF94; } QPushButton:hover { background-color: #50fa7b; } QPushButton:pressed { background-color: #69FF94; }"
        self.btn_style_after_run = "QPushButton { font-size: 14px; color: #282a36; background-color: #FF6E6E; } QPushButton:hover {	background-color: #ff5555; } QPushButton:pressed { background-color: #FF6E6E; }"
        self.btn_style_stopping = "QPushButton { font-size: 14px; color: #282a36; background-color: #FFFFA5; } QPushButton:hover {	background-color: #F1FA8C; } QPushButton:pressed { background-color: #FFFFA5; }"
        self.re_datetime = re.compile(
            "(\d\d)(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(\d\d)([0-5]\d|60)([0-5]\d|60)"
        )
        self.graph_paths = None
        self.icon_ban = QIcon()
        self.icon_ban.addFile(u":/icons/images/icons/cil-ban.png", QSize(), QIcon.Normal, QIcon.Off)
        self.tree_gr = ui.tree_gr
        self.btn_run_gr = ui.btn_run_gr
        self.btn_vis_gr = ui.btn_vis_gr
        self.label_tree_gr = ui.label_tree_gr
        self.generation_config = generation_config
        self.progress_bar = ui.progress_gr
        self.label_progress_bar = ui.label_progress_gr
        self.label_suffix = ui.label_suffix_gr
        self.label_progress_bar.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.progress_bar.setValue(0)
        self.mlf_run = None
        self.model_metric = None
        self.runs_df = mlf_set_env(
            1234,
            self.generation_config.global_config.default_mlf_dir,
            self.generation_config.global_config.default_cfg["mlflow"]["exp_name"],
            root_dir=".",
            fix_path=True,
            return_runs_list=True,
        )
        self.label_tree_gr.setStyleSheet("QLabel { font-size: 16px; }")
        self.btn_run_gr.setIcon(self.icon_ban)
        self.btn_run_gr.setText("  No model selected")
        self.tree_gr.itemDoubleClicked.connect(self.get_model)
        self.create_runner_thread()
        self.create_stopper_thread()
        self.parent().aboutToQuit.connect(self.quit_threads)
        self.btn_run_gr.clicked.connect(self.runner.run_generation)
        self.btn_vis_gr.clicked.connect(self.open_vis_window)
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

    def get_graphs_for_last_run(self):
        graph_source_path = self.generation_config.cfg.generation.save_dir
        try:
            last_run_path = sorted(
                [
                    join(graph_source_path, p)
                    for p in listdir(graph_source_path)
                    if isdir(join(graph_source_path, p)) and self.re_datetime.match(p)
                ],
                key=lambda s: datetime.strptime(basename(s), "%y%m%d%H%M%S"),
                reverse=True,
            )[0]
        except IndexError:
            last_run_path = ""
        graph_source = join(last_run_path, "synthetic_graphs")
        return (
            self.get_graph_paths(graph_source),
            graph_source,
        )

    # def save_graph(self):
    #     if len(self.graph_paths) > 0:
    #         graph_path = self.graph_paths[self.graph_idx]
    #         file_name = QFileDialog().getSaveFileName(
    #             self.generation_config,
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

    def search_graph_source(self):
        dir_name = QFileDialog().getExistingDirectory(self.generation_config)
        self.vis_ui.line_find_source_vis.setText(dir_name)

    def update_graph_paths(self, path):
        self.graph_idx = 0
        self.graph_paths = self.get_graph_paths(path)
        self.draw_graph(increment=False)

    def open_vis_window(self):
        self.sc = None
        self.vis_window = QMainWindow()
        self.vis_ui = Ui_VisWindow()
        self.vis_ui.setupUi(self.vis_window)
        self.sc = MplCanvas(self, width=5, height=4, dpi=150)
        self.vis_ui.layout_graph_vis.addWidget(self.sc)
        if self.graph_paths is None:
            self.graph_idx = 0
            self.graph_paths, self.last_run_path = self.get_graphs_for_last_run()
        if len(self.graph_paths) > 0:
            self.draw_graph(increment=False)
        else:
            self.sc.ax.xaxis.set_visible(False)
            self.sc.ax.yaxis.set_visible(False)
            self.sc.ax.spines["top"].set_visible(False)
            self.sc.ax.spines["right"].set_visible(False)
            self.sc.ax.spines["bottom"].set_visible(False)
            self.sc.ax.spines["left"].set_visible(False)
        self.vis_ui.btn_change_graph_vis.clicked.connect(self.draw_graph)
        self.vis_ui.line_find_source_vis.setText(self.last_run_path)
        self.vis_ui.line_find_source_vis.textChanged.connect(self.update_graph_paths)
        self.vis_ui.btn_find_source_vis.clicked.connect(self.search_graph_source)
        self.vis_window.show()

    def draw_graph(self, checked=False, increment=True):
        if self.sc is not None:
            self.vis_ui.layout_graph_vis.takeAt(0).widget().deleteLater()
            self.sc = None
            del self.sc
        self.sc = MplCanvas(self, width=5, height=4, dpi=150)
        self.vis_ui.layout_graph_vis.addWidget(self.sc)
        if len(self.graph_paths) > 0:
            if increment:
                self.graph_idx = (
                    self.graph_idx + 1
                    if self.graph_idx < len(self.graph_paths) - 1
                    else 0
                )
            graph_path = self.graph_paths[self.graph_idx]
            graph = nx.read_gpickle(graph_path)
            pos = nx.nx_agraph.graphviz_layout(graph, prog="sfdp")
            nx.draw_networkx_nodes(
                graph,
                pos=pos,
                node_size=80,
                alpha=0.9,
                edgecolors="k",
                node_color="#8be9fd",
                ax=self.sc.ax,
            )
            nx.draw_networkx_edges(graph, pos=pos, node_size=100, ax=self.sc.ax)
            self.sc.ax.set_title(basename(graph_path))
        else:
            self.sc.ax.xaxis.set_visible(False)
            self.sc.ax.yaxis.set_visible(False)
        self.sc.ax.spines["top"].set_visible(False)
        self.sc.ax.spines["right"].set_visible(False)
        self.sc.ax.spines["bottom"].set_visible(False)
        self.sc.ax.spines["left"].set_visible(False)

    def get_model(self, *args):
        item = args[0]
        model_name = item.text(0)
        if model_name.startswith("MODEL") and not self.runner.track_running.running:
            run_name = item.parent().text(0)
            metric = item.text(1).split(" ")[1]
            self.btn_run_gr.setIcon(QIcon())
            self.btn_run_gr.setText(
                f"Generate Graphs using {model_name} from {run_name.rstrip()}"
            )
            self.btn_run_gr.setStyleSheet(self.btn_style_before_run)
            run_id = self.runs_df.iloc[int(run_name.split(" ")[-1]) - 1]["Run ID"]
            self.mlf_run = mlf.get_run(run_id=run_id)
            self.model_metric = f"mmd_{metric.lower()}"
            self.generation_config.spin_min_num_nodes_gr.setValue(
                int(self.mlf_run.data.params["data.min_num_node"])
            )
            self.generation_config.spin_max_num_nodes_gr.setValue(
                int(self.mlf_run.data.params["data.max_num_node"])
            )
            self.runner.set_run_model(self.model_metric, self.mlf_run)
            self.run_name = run_name
            self.model_name = model_name

    def load_runs(self, update_runs=False):
        self.tree_gr.clear()
        if update_runs:
            experiment = mlf.get_experiment_by_name(
                self.generation_config.global_config.default_cfg["mlflow"]["exp_name"]
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
            self.tree_gr.setHeaderLabels(columns)
            for i, row in _runs_df.iterrows():
                model_idx = 0
                row_list = row.iloc[2:].values.tolist()
                row_list = [f"RUN {i + 1}"] + [""] + row_list
                tree_item = QTreeWidgetItem(row_list)
                self.tree_gr.addTopLevelItem(tree_item)
                for metric_name in row.iloc[5:].index:
                    if row[metric_name].strip() != "nan":
                        row_model_list = [
                            f"MODEL {model_idx + 1}",
                            "for " + metric_name.split(" ")[1] + " MMD",
                        ] + [""] * (len(columns) - 2)
                        child_item = QTreeWidgetItem(row_model_list)
                        tree_item.addChild(child_item)
                        model_idx += 1
            self.tree_gr.expandAll()
            for i in range(len(columns)):
                self.tree_gr.resizeColumnToContents(i)
            self.tree_gr.collapseAll()
            self.tree_gr.setSortingEnabled(True)
            # self.tree_gr.sortItems(2, Qt.DescendingOrder)
        else:
            pass

    def create_runner_thread(self):
        self.runner = GenerationRunner(self.generation_config)
        self.runner_thread = QThread()
        self.runner.moveToThread(self.runner_thread)
        self.runner_thread.start()
        self.runner_thread.finished.connect(self.runner_thread.deleteLater)
        self.runner.progress_sig.connect(self.update_progress)
        self.runner.btn_sig.connect(self.update_btn)
        self.runner.set_run_model(self.model_metric, self.mlf_run)

    def create_stopper_thread(self):
        self.stopper = GenerationStopper()
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
            self.btn_run_gr.clicked.disconnect()
            self.btn_run_gr.clicked.connect(self.stopper.stop_generation)
            self.btn_run_gr.setStyleSheet(self.btn_style_after_run)
            self.btn_run_gr.setText("Stop")
        elif sig == "stopping":
            self.btn_run_gr.setStyleSheet(self.btn_style_stopping)
            self.btn_run_gr.setText("Stopping")
        elif sig == "stop":
            self.create_runner_thread()
            self.stopper.add_runner(self.runner_thread, self.runner.track_running)
            self.btn_run_gr.clicked.disconnect()
            self.btn_run_gr.clicked.connect(self.runner.run_generation)
            if self.model_name is None or self.run_name is None:
                self.btn_run_gr.setStyleSheet("")
                self.btn_run_gr.setIcon(self.icon_ban)
                self.btn_run_gr.setText("  No model selected")
            else:
                self.btn_run_gr.setIcon(QIcon())
                self.btn_run_gr.setStyleSheet(self.btn_style_before_run)
                self.btn_run_gr.setText(
                    f"Generate Graphs using {self.model_name} from {self.run_name.rstrip()}"
                )
        elif sig == "endrun":
            self.btn_run_gr.clicked.disconnect()
            self.btn_run_gr.clicked.connect(self.runner.run_generation)
            if self.model_name is None or self.run_name is None:
                self.btn_run_gr.setStyleSheet("")
                self.btn_run_gr.setIcon(self.icon_ban)
                self.btn_run_gr.setText("  No model selected")
            else:
                self.btn_run_gr.setIcon(QIcon())
                self.btn_run_gr.setStyleSheet(self.btn_style_before_run)
                self.btn_run_gr.setText(
                    f"Generate Graphs using {self.model_name} from {self.run_name.rstrip()}"
                )


class GenerationRunner(QObject):
    progress_sig = Signal(str)
    btn_sig = Signal(str)

    def __init__(self, generation_config):
        super(self.__class__, self).__init__()
        self.generation_config = generation_config
        self.track_running = TrackRunning(self.progress_sig)

    def set_run_model(self, model_metric, mlf_run):
        self.model_metric = model_metric
        self.mlf_run = mlf_run

    @Slot()
    def run_generation(self):
        if (
            not self.track_running.running
            and self.model_metric is not None
            and self.mlf_run is not None
        ):
            self.progress_sig.emit("suffix|")
            self.track_running.stop_running = False
            self.track_running.running = True
            self.btn_sig.emit("run")

            generate(
                self.generation_config.cfg,
                None,
                self.generation_config.min_num_nodes,
                self.generation_config.max_num_nodes,
                self.generation_config.mlf_dir,
                progress_bar_qt=self.track_running,
                model_metric=self.model_metric,
                mlf_run=self.mlf_run,
            )
            self.track_running.running = False
            self.btn_sig.emit("endrun")


class GenerationStopper(QObject):
    btn_sig = Signal(str)

    def __init__(self):
        super(self.__class__, self).__init__()

    def add_runner(self, runner_thread, track_running):
        self.runner_thread = runner_thread
        self.track_running = track_running

    @Slot()
    def stop_generation(self):
        if not self.track_running.stop_running and self.track_running.running:
            self.track_running.stop_running = True
            self.runner_thread.quit()
            self.btn_sig.emit("stopping")
            self.runner_thread.wait()
            self.runner_thread = None
            self.track_running = None
            self.btn_sig.emit("stop")
