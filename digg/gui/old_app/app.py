import sys
from os import getcwd, listdir
from os.path import join, expanduser

from hydra import compose, initialize_config_dir
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QTabWidget,
    QMainWindow,
    QWidget,
)
from PySide6.QtQuickControls2 import QQuickStyle

from digg.gui.traintab import TrainTab


class Color(QWidget):
    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        cwd = getcwd()
        config_name = "config.yaml"
        if config_name in [p for p in listdir(cwd) if p.endswith("yaml")]:
            config_dir = cwd
        else:
            config_dir = join(expanduser("~"), ".config", "digg_dggm")
        self.mlf_dir = cwd
        initialize_config_dir(
            version_base=None, config_dir=config_dir, job_name="train"
        )
        self.cfg = compose(config_name=config_name)

        self.setWindowTitle("DIGG")
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        tabs.setMovable(False)
        tabs.addTab(
            TrainTab(self.cfg, self.mlf_dir),
            "Training",
        )
        self.setCentralWidget(tabs)


app = QApplication(sys.argv)
QQuickStyle.setStyle("Material")

window = MainWindow()
window.show()

app.exec()
