# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////
import sys
import os

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from dggi.gui.ui_functions import MainWindow

os.environ["QT_FONT_DPI"] = "96"  # FIX Problem for High DPI and Scale above 100%

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow(app)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
