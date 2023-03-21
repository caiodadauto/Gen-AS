import time
from os.path import expanduser, join, abspath

from PySide6.QtWidgets import QWidget, QCheckBox, QFileDialog
from PySide6.QtCore import QThread, QObject, Signal, Slot


class Example(QObject):
    # signalStatus = Signal(int)

    def __init__(self, ui):
        super(self.__class__, self).__init__()
        self.btn_run_tr = ui.btn_run_tr
        self.progress_bar = ui.progress_tr
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.createWorkerThread()
        # self._connectSignals()

    # def _connectSignals(self):
    #     self.signalStatus.connect(self.updateStatus)

    def createWorkerThread(self):
        # Setup the worker object and the worker_thread.
        self.worker = WorkerObject()
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        # Connect any worker signals
        self.worker.signalStatus.connect(self.updateStatus)
        self.btn_run_tr.clicked.connect(self.worker.startWork)

    def updateStatus(self, n):
        self.progress_bar.setValue(n)


class WorkerObject(QObject):
    signalStatus = Signal(int)

    def __init__(self):
        super(self.__class__, self).__init__()

    @Slot()
    def startWork(self):
        for i in range(100):
            print(i)
            time.sleep(1)
            self.signalStatus.emit(i + 1)
