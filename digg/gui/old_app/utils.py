from functools import partial

from omegaconf.listconfig import ListConfig
from PySide6.QtGui import QValidator
from PySide6.QtCore import QRegularExpression
from PySide6.QtWidgets import (
    QProgressBar,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QWidget,
    # QFrame,
)


def get_grid_layout(num_rows, num_cols, row_ratios, col_ratios):
    layout = QGridLayout()
    for i, r in zip(range(num_rows), row_ratios):
        layout.setRowStretch(i, r)
    for j, r in zip(range(num_cols), col_ratios):
        layout.setColumnStretch(j, r)
    return layout


def get_label(text, font_size, bold):
    widget = QLabel(text)
    font = widget.font()
    font.setPointSize(font_size)
    font.setBold(bold)
    widget.setFont(font)
    return widget


class ProgressBarButton(QWidget):
    def __init__(self, pressed_func):
        super(ProgressBarButton, self).__init__()
        layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.button = QPushButton("Run")
        self.button.pressed.connect(
            partial(pressed_func, progress_bar=self.progress_bar)
        )
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button)
        self.setLayout(layout)


class SpinBoxLabel(QWidget):
    def __init__(self, label_text, default_value, min_value, max_value):
        super(SpinBoxLabel, self).__init__()
        layout = QVBoxLayout()
        self.label = QLabel(label_text)
        self.spinbox = QSpinBox()
        self.spinbox.setRange(min_value, max_value)
        self.spinbox.setValue(
            default_value if default_value <= max_value else max_value
        )
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)

    def value(self):
        return self.spinbox.value()


class DoubleSpinBoxLabel(QWidget):
    def __init__(
        self, label_text, default_value, min_value, max_value, decimals, step_size
    ):
        super(DoubleSpinBoxLabel, self).__init__()
        layout = QVBoxLayout()
        self.label = QLabel(label_text)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_value, max_value)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setSingleStep(step_size)
        self.spinbox.setValue(
            default_value if default_value <= max_value else max_value
        )
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)

    def value(self):
        return self.spinbox.value()

class LineEditLabel(QWidget):
    def __init__(self, label_text, default_value, placeholder=None, validator_regex=None):
        super(LineEditLabel, self).__init__()
        layout = QVBoxLayout()
        self.label = QLabel(label_text)
        self.line_edit = QLineEdit()
        if default_value is None:
            self.line_edit.setPlaceholderText(placeholder)
        elif isinstance(default_value, list) or isinstance(default_value, ListConfig):
            self.line_edit.setText(",".join(map(str, default_value)))
        else:
            self.line_edit.setText(default_value)
        if validator_regex is not None:
            QValidator(QRegularExpression(validator_regex), self.line_edit)
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        self.setLayout(layout)

    def value(self):
        return self.line_edit.text()
