# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainWDfNaF.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QCheckBox, QComboBox,
    QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpacerItem, QSpinBox, QStackedWidget,
    QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget)
from dggi.gui.resources_rc import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1203, 736)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(940, 0))
        self.styleSheet = QWidget(MainWindow)
        self.styleSheet.setObjectName(u"styleSheet")
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        self.styleSheet.setFont(font)
        self.styleSheet.setStyleSheet(u"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"\n"
"SET APP STYLESHEET - FULL STYLES HERE\n"
"DARK THEME - DRACULA COLOR BASED\n"
"\n"
"///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"\n"
"QWidget{\n"
"	color: rgb(221, 221, 221);\n"
"	font: 10pt \"Segoe UI\";\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Tooltip */\n"
"QToolTip {\n"
"	color: #ffffff;\n"
"	background-color: rgba(33, 37, 43, 180);\n"
"	border: 1px solid rgb(44, 49, 58);\n"
"	background-image: none;\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 2px solid rgb(255, 121, 198);\n"
"	text-align: left;\n"
"	padding-left: 8px;\n"
"	margin: 0px;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Bg App */\n"
"#bgApp {	\n"
"	background"
                        "-color: rgb(40, 44, 52);\n"
"	border: 1px solid rgb(44, 49, 58);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Left Menu */\n"
"#leftMenuBg {	\n"
"	background-color: rgb(33, 37, 43);\n"
"}\n"
"#topLogo {\n"
"	background-color: rgb(33, 37, 43);\n"
"	background-image: url(:/images/images/images/PyDracula.png);\n"
"	background-position: centered;\n"
"	background-repeat: no-repeat;\n"
"}\n"
"#titleLeftApp { font: 63 12pt \"Segoe UI Semibold\"; }\n"
"#titleLeftDescription { font: 8pt \"Segoe UI\"; color: rgb(189, 147, 249); }\n"
"\n"
"/* MENUS */\n"
"#topMenu .QPushButton {	\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color: transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"}\n"
"#topMenu .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#topMenu .QPushButton:pressed {	\n"
"	background-color: rgb(18"
                        "9, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"#bottomMenu .QPushButton {	\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"}\n"
"#bottomMenu .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#bottomMenu .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"#leftMenuFrame{\n"
"	border-top: 3px solid rgb(44, 49, 58);\n"
"}\n"
"\n"
"/* Toggle Button */\n"
"#toggleButton {\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	background-color: rgb(37, 41, 48);\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"	color: rgb(113, 126, 149);\n"
"}\n"
"#toggleButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#toggleButton:pressed {\n"
"	background-color: rgb("
                        "189, 147, 249);\n"
"}\n"
"\n"
"/* Title Menu */\n"
"#titleRightInfo { padding-left: 10px; }\n"
"\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Extra Tab */\n"
"#extraLeftBox {	\n"
"	background-color: rgb(44, 49, 58);\n"
"}\n"
"#extraTopBg{	\n"
"	background-color: rgb(189, 147, 249)\n"
"}\n"
"\n"
"/* Icon */\n"
"#extraIcon {\n"
"	background-position: center;\n"
"	background-repeat: no-repeat;\n"
"	background-image: url(:/icons/images/icons/icon_settings.png);\n"
"}\n"
"\n"
"/* Label */\n"
"#extraLabel { color: rgb(255, 255, 255); }\n"
"\n"
"/* Btn Close */\n"
"#extraCloseColumnBtn { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px; }\n"
"#extraCloseColumnBtn:hover { background-color: rgb(196, 161, 249); border-style: solid; border-radius: 4px; }\n"
"#extraCloseColumnBtn:pressed { background-color: rgb(180, 141, 238); border-style: solid; border-radius: 4px; }\n"
"\n"
"/* Extra Content */\n"
"#extraContent{\n"
"	border"
                        "-top: 3px solid rgb(40, 44, 52);\n"
"}\n"
"\n"
"/* Extra Top Menus */\n"
"#extraTopMenu .QPushButton {\n"
"background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"}\n"
"#extraTopMenu .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#extraTopMenu .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Content App */\n"
"#contentTopBg{	\n"
"	background-color: rgb(33, 37, 43);\n"
"}\n"
"#contentBottom{\n"
"	border-top: 3px solid rgb(44, 49, 58);\n"
"}\n"
"\n"
"/* Top Buttons */\n"
"#rightButtons .QPushButton { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px; }\n"
"#rightButtons .QPushButton:hover { background-color: rgb(44, 49, 57); border-sty"
                        "le: solid; border-radius: 4px; }\n"
"#rightButtons .QPushButton:pressed { background-color: rgb(23, 26, 30); border-style: solid; border-radius: 4px; }\n"
"\n"
"/* Theme Settings */\n"
"#extraRightBox { background-color: rgb(44, 49, 58); }\n"
"#themeSettingsTopDetail { background-color: rgb(189, 147, 249); }\n"
"\n"
"/* Bottom Bar */\n"
"#bottomBar { background-color: rgb(44, 49, 58); }\n"
"#bottomBar QLabel { font-size: 11px; color: rgb(113, 126, 149); padding-left: 10px; padding-right: 10px; padding-bottom: 2px; }\n"
"\n"
"/* CONTENT SETTINGS */\n"
"/* MENUS */\n"
"#contentSettings QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"#contentSettings QPushButton:hover {\n"
"	background-color: rgb(57, 65, 80);\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"#contentSettings QPushButton:pressed {	\n"
"	background-color: rgb(35, 40, 49);\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}\n"
"/* #contentSettings .QPushButton {	\n"
"	backgr"
                        "ound-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color:transparent;\n"
"	padding-left: 44px;\n"
"}\n"
"#contentSettings .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#contentSettings .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"} */\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QTableWidget\n"
"QTableWidget {	\n"
"	background-color: transparent;\n"
"	selection-background-color: blue;\n"
"	padding: 10px;\n"
"	border-radius: 5px;\n"
"	gridline-color: rgb(44, 49, 58);\n"
"	border-bottom: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item{\n"
"	border-color: rgb(44, 49, 60);\n"
"	padding-left: 5px;\n"
"	padding-right: 5px;\n"
"	gridline-color: rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item:selected{\n"
"	background-color: rgb(189, 147, 249);\n"
"}\n"
"QHeaderView::secti"
                        "on{\n"
"	background-color: rgb(33, 37, 43);\n"
"	max-width: 30px;\n"
"	border: 1px solid rgb(44, 49, 58);\n"
"	border-style: none;\n"
"    border-bottom: 1px solid rgb(44, 49, 60);\n"
"    border-right: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::horizontalHeader {	\n"
"	background-color: rgb(33, 37, 43);\n"
"}\n"
"QHeaderView::section:horizontal\n"
"{\n"
"    border: 1px solid rgb(33, 37, 43);\n"
"	background-color: rgb(33, 37, 43);\n"
"	padding: 3px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"}\n"
"QHeaderView::section:vertical\n"
"{\n"
"    border: 1px solid rgb(44, 49, 60);\n"
"	selected-background-color: rgb(189, 147, 249);\n"
"}\n"
"QHeaderView::section:vertical:selected\n"
"{\n"
"	background-color: rgb(189, 147, 249);\n"
"}\n"
" */\n"
"QTableWidget {	\n"
"	selection-background-color: rgb(189, 147, 249);\n"
"	gridline-color: rgb(44, 49, 58);\n"
"	border-bottom: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QHeaderView::section{\n"
"	background-color: rgb(33, 37, 43);\n"
"	m"
                        "ax-width: 30px;\n"
"	border: 1px solid rgb(44, 49, 58);\n"
"	border-style: none;\n"
"    border-bottom: 1px solid rgb(44, 49, 60);\n"
"    border-right: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::horizontalHeader {	\n"
"	background-color: rgb(33, 37, 43);\n"
"}\n"
"QHeaderView::section:horizontal\n"
"{\n"
"    border: 1px solid rgb(33, 37, 43);\n"
"	background-color: rgb(33, 37, 43);\n"
"	padding: 3px;\n"
"}\n"
"QHeaderView::section:vertical\n"
"{\n"
"    border: 1px solid rgb(44, 49, 60);\n"
"	selected-background-color: rgb(189, 147, 249);\n"
"}\n"
"QHeaderView::section:vertical:selected\n"
"{\n"
"	background-color: rgb(189, 147, 249);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"LineEdit */\n"
"QLineEdit {\n"
"	background-color: rgb(33, 37, 43);\n"
"	border-radius: 5px;\n"
"	border: 2px solid rgb(33, 37, 43);\n"
"	padding-left: 10px;\n"
"	selection-color: rgb(255, 255, 255);\n"
"	selection-background-color: rgb(255, 121, 198);"
                        "\n"
"}\n"
"QLineEdit:hover {\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QLineEdit:focus {\n"
"	border: 2px solid rgb(91, 101, 124);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"PlainTextEdit */\n"
"QPlainTextEdit {\n"
"	background-color: rgb(27, 29, 35);\n"
"	border-radius: 5px;\n"
"	padding: 10px;\n"
"	selection-color: rgb(255, 255, 255);\n"
"	selection-background-color: rgb(255, 121, 198);\n"
"}\n"
"QPlainTextEdit  QScrollBar:vertical {\n"
"    width: 8px;\n"
" }\n"
"QPlainTextEdit  QScrollBar:horizontal {\n"
"    height: 8px;\n"
" }\n"
"QPlainTextEdit:hover {\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QPlainTextEdit:focus {\n"
"	border: 2px solid rgb(91, 101, 124);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ScrollBars */\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    height: 8px;\n"
"    margin: 0"
                        "px 21px 0 21px;\n"
"	border-radius: 0px;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"    background: rgb(189, 147, 249);\n"
"    min-width: 25px;\n"
"	border-radius: 4px\n"
"}\n"
"QScrollBar::add-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"	border-top-right-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::sub-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"	border-top-left-radius: 4px;\n"
"    border-bottom-left-radius: 4px;\n"
"    subcontrol-position: left;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
" QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"   "
                        " width: 8px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }\n"
" QScrollBar::handle:vertical {	\n"
"	background: rgb(189, 147, 249);\n"
"    min-height: 25px;\n"
"	border-radius: 4px\n"
" }\n"
" QScrollBar::add-line:vertical {\n"
"     border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"	border-bottom-left-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"     subcontrol-position: bottom;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::sub-line:vertical {\n"
"	border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"	border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"     subcontrol-position: top;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
" QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
"/* ///////////////////////////////////////////////////"
                        "//////////////////////////////////////////////\n"
"CheckBox */\n"
"QCheckBox::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"	width: 15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QCheckBox::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    background: 3px solid rgb(52, 59, 72);\n"
"	border: 3px solid rgb(52, 59, 72);	\n"
"	background-image: url(:/icons/images/icons/cil-check-alt.png);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"RadioButton */\n"
"QRadioButton::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"	width: 15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QRadioButton::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background: 3px solid rgb(94, 106, 130);\n"
"	border: 3px solid rgb("
                        "52, 59, 72);	\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ComboBox */\n"
"QComboBox{\n"
"	background-color: rgb(27, 29, 35);\n"
"	border-radius: 5px;\n"
"	border: 2px solid rgb(33, 37, 43);\n"
"	padding: 5px;\n"
"	padding-left: 10px;\n"
"}\n"
"QComboBox:hover{\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QComboBox::drop-down {\n"
"	subcontrol-origin: padding;\n"
"	subcontrol-position: top right;\n"
"	width: 25px; \n"
"	border-left-width: 3px;\n"
"	border-left-color: rgba(39, 44, 54, 150);\n"
"	border-left-style: solid;\n"
"	border-top-right-radius: 3px;\n"
"	border-bottom-right-radius: 3px;	\n"
"	background-image: url(:/icons/images/icons/cil-arrow-bottom.png);\n"
"	background-position: center;\n"
"	background-repeat: no-reperat;\n"
" }\n"
"QComboBox QAbstractItemView {\n"
"	color: rgb(255, 121, 198);	\n"
"	background-color: rgb(33, 37, 43);\n"
"	padding: 10px;\n"
"	selection-background-color: rgb(39, 44, 54);\n"
"}\n"
"\n"
"/* //"
                        "///////////////////////////////////////////////////////////////////////////////////////////////\n"
"Sliders */\n"
"QSlider::groove:horizontal {\n"
"    border-radius: 5px;\n"
"    height: 10px;\n"
"	margin: 0px;\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:horizontal:hover {\n"
"	background-color: rgb(55, 62, 76);\n"
"}\n"
"QSlider::handle:horizontal {\n"
"    background-color: rgb(189, 147, 249);\n"
"    border: none;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	border-radius: 5px;\n"
"}\n"
"QSlider::handle:horizontal:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:horizontal:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"    border-radius: 5px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:vertical:hover {\n"
"	background-color: rgb(55, 62, 76);\n"
"}\n"
"QSlider::handle:vertical {\n"
"    background-color: rgb(189, 1"
                        "47, 249);\n"
"	border: none;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	border-radius: 5px;\n"
"}\n"
"QSlider::handle:vertical:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:vertical:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"CommandLinkButton */\n"
"QCommandLinkButton {	\n"
"	color: rgb(255, 121, 198);\n"
"	border-radius: 5px;\n"
"	padding: 5px;\n"
"	color: rgb(255, 170, 255);\n"
"}\n"
"QCommandLinkButton:hover {	\n"
"	color: rgb(255, 170, 255);\n"
"	background-color: rgb(44, 49, 60);\n"
"}\n"
"QCommandLinkButton:pressed {	\n"
"	color: rgb(189, 147, 249);\n"
"	background-color: rgb(52, 58, 71);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Button */\n"
"#pagesContainer QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
""
                        "	background-color: rgb(52, 59, 72);\n"
"}\n"
"#pagesContainer QPushButton:hover {\n"
"	background-color: rgb(57, 65, 80);\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"#pagesContainer QPushButton:pressed {	\n"
"	background-color: rgb(35, 40, 49);\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}\n"
"\n"
"")
        self.appMargins = QVBoxLayout(self.styleSheet)
        self.appMargins.setSpacing(0)
        self.appMargins.setObjectName(u"appMargins")
        self.appMargins.setContentsMargins(10, 10, 10, 10)
        self.bgApp = QFrame(self.styleSheet)
        self.bgApp.setObjectName(u"bgApp")
        self.bgApp.setStyleSheet(u"")
        self.bgApp.setFrameShape(QFrame.NoFrame)
        self.bgApp.setFrameShadow(QFrame.Raised)
        self.appLayout = QHBoxLayout(self.bgApp)
        self.appLayout.setSpacing(0)
        self.appLayout.setObjectName(u"appLayout")
        self.appLayout.setContentsMargins(0, 0, 0, 0)
        self.leftMenuBg = QFrame(self.bgApp)
        self.leftMenuBg.setObjectName(u"leftMenuBg")
        self.leftMenuBg.setMinimumSize(QSize(60, 0))
        self.leftMenuBg.setMaximumSize(QSize(60, 16777215))
        self.leftMenuBg.setFrameShape(QFrame.NoFrame)
        self.leftMenuBg.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.leftMenuBg)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.topLogoInfo = QFrame(self.leftMenuBg)
        self.topLogoInfo.setObjectName(u"topLogoInfo")
        self.topLogoInfo.setMinimumSize(QSize(0, 70))
        self.topLogoInfo.setMaximumSize(QSize(16777215, 70))
        self.topLogoInfo.setFrameShape(QFrame.NoFrame)
        self.topLogoInfo.setFrameShadow(QFrame.Raised)
        self.titleLeftApp = QLabel(self.topLogoInfo)
        self.titleLeftApp.setObjectName(u"titleLeftApp")
        self.titleLeftApp.setGeometry(QRect(70, 8, 160, 20))
        font1 = QFont()
        font1.setFamilies([u"Segoe UI Semibold"])
        font1.setPointSize(12)
        font1.setBold(False)
        font1.setItalic(False)
        self.titleLeftApp.setFont(font1)
        self.titleLeftApp.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.titleLeftDescription = QLabel(self.topLogoInfo)
        self.titleLeftDescription.setObjectName(u"titleLeftDescription")
        self.titleLeftDescription.setGeometry(QRect(70, 27, 160, 31))
        self.titleLeftDescription.setMaximumSize(QSize(16777215, 100))
        font2 = QFont()
        font2.setFamilies([u"Segoe UI"])
        font2.setPointSize(8)
        font2.setBold(False)
        font2.setItalic(False)
        self.titleLeftDescription.setFont(font2)
        self.titleLeftDescription.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.frame_62 = QFrame(self.topLogoInfo)
        self.frame_62.setObjectName(u"frame_62")
        self.frame_62.setGeometry(QRect(10, 10, 50, 50))
        self.frame_62.setAutoFillBackground(False)
        self.frame_62.setStyleSheet(u"\n"
"image: url(:/images/images/images/logo.png);")
        self.frame_62.setFrameShape(QFrame.NoFrame)
        self.frame_62.setFrameShadow(QFrame.Raised)

        self.verticalLayout_3.addWidget(self.topLogoInfo)

        self.leftMenuFrame = QFrame(self.leftMenuBg)
        self.leftMenuFrame.setObjectName(u"leftMenuFrame")
        self.leftMenuFrame.setFrameShape(QFrame.NoFrame)
        self.leftMenuFrame.setFrameShadow(QFrame.Raised)
        self.verticalMenuLayout = QVBoxLayout(self.leftMenuFrame)
        self.verticalMenuLayout.setSpacing(0)
        self.verticalMenuLayout.setObjectName(u"verticalMenuLayout")
        self.verticalMenuLayout.setContentsMargins(0, 0, 0, 0)
        self.toggleBox = QFrame(self.leftMenuFrame)
        self.toggleBox.setObjectName(u"toggleBox")
        self.toggleBox.setMaximumSize(QSize(16777215, 45))
        self.toggleBox.setFrameShape(QFrame.NoFrame)
        self.toggleBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.toggleBox)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.toggleButton = QPushButton(self.toggleBox)
        self.toggleButton.setObjectName(u"toggleButton")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.toggleButton.sizePolicy().hasHeightForWidth())
        self.toggleButton.setSizePolicy(sizePolicy1)
        self.toggleButton.setMinimumSize(QSize(0, 45))
        self.toggleButton.setFont(font)
        self.toggleButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggleButton.setLayoutDirection(Qt.LeftToRight)
        self.toggleButton.setStyleSheet(u"background-image: url(:/icons/images/icons/icon_menu.png);")

        self.verticalLayout_4.addWidget(self.toggleButton)


        self.verticalMenuLayout.addWidget(self.toggleBox)

        self.topMenu = QFrame(self.leftMenuFrame)
        self.topMenu.setObjectName(u"topMenu")
        self.topMenu.setFrameShape(QFrame.NoFrame)
        self.topMenu.setFrameShadow(QFrame.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.topMenu)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.btn_train = QPushButton(self.topMenu)
        self.btn_train.setObjectName(u"btn_train")
        sizePolicy1.setHeightForWidth(self.btn_train.sizePolicy().hasHeightForWidth())
        self.btn_train.setSizePolicy(sizePolicy1)
        self.btn_train.setMinimumSize(QSize(0, 45))
        self.btn_train.setFont(font)
        self.btn_train.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_train.setLayoutDirection(Qt.LeftToRight)
        self.btn_train.setStyleSheet(u"background-image: url(:/icons/images/icons/icon_layer_solid.png);")

        self.verticalLayout_8.addWidget(self.btn_train)

        self.btn_generate = QPushButton(self.topMenu)
        self.btn_generate.setObjectName(u"btn_generate")
        sizePolicy1.setHeightForWidth(self.btn_generate.sizePolicy().hasHeightForWidth())
        self.btn_generate.setSizePolicy(sizePolicy1)
        self.btn_generate.setMinimumSize(QSize(0, 45))
        self.btn_generate.setFont(font)
        self.btn_generate.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_generate.setLayoutDirection(Qt.LeftToRight)
        self.btn_generate.setStyleSheet(u"background-image: url(:/icons/images/icons/icon_project_diagram.png);")

        self.verticalLayout_8.addWidget(self.btn_generate)

        self.btn_evaluate = QPushButton(self.topMenu)
        self.btn_evaluate.setObjectName(u"btn_evaluate")
        sizePolicy1.setHeightForWidth(self.btn_evaluate.sizePolicy().hasHeightForWidth())
        self.btn_evaluate.setSizePolicy(sizePolicy1)
        self.btn_evaluate.setMinimumSize(QSize(0, 45))
        self.btn_evaluate.setFont(font)
        self.btn_evaluate.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_evaluate.setLayoutDirection(Qt.LeftToRight)
        self.btn_evaluate.setStyleSheet(u"background-image: url(:/icons/images/icons/icon_chart_bar.png);")

        self.verticalLayout_8.addWidget(self.btn_evaluate)


        self.verticalMenuLayout.addWidget(self.topMenu, 0, Qt.AlignTop)

        self.bottomMenu = QFrame(self.leftMenuFrame)
        self.bottomMenu.setObjectName(u"bottomMenu")
        self.bottomMenu.setFrameShape(QFrame.NoFrame)
        self.bottomMenu.setFrameShadow(QFrame.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.bottomMenu)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.toggleLeftBox = QPushButton(self.bottomMenu)
        self.toggleLeftBox.setObjectName(u"toggleLeftBox")
        sizePolicy1.setHeightForWidth(self.toggleLeftBox.sizePolicy().hasHeightForWidth())
        self.toggleLeftBox.setSizePolicy(sizePolicy1)
        self.toggleLeftBox.setMinimumSize(QSize(0, 45))
        self.toggleLeftBox.setFont(font)
        self.toggleLeftBox.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggleLeftBox.setLayoutDirection(Qt.LeftToRight)
        self.toggleLeftBox.setStyleSheet(u"background-image: url(:/icons/images/icons/icon_info.png);")

        self.verticalLayout_9.addWidget(self.toggleLeftBox)


        self.verticalMenuLayout.addWidget(self.bottomMenu, 0, Qt.AlignBottom)


        self.verticalLayout_3.addWidget(self.leftMenuFrame)


        self.appLayout.addWidget(self.leftMenuBg)

        self.extraLeftBox = QFrame(self.bgApp)
        self.extraLeftBox.setObjectName(u"extraLeftBox")
        self.extraLeftBox.setMinimumSize(QSize(0, 0))
        self.extraLeftBox.setMaximumSize(QSize(0, 16777215))
        self.extraLeftBox.setFrameShape(QFrame.NoFrame)
        self.extraLeftBox.setFrameShadow(QFrame.Raised)
        self.extraColumLayout = QVBoxLayout(self.extraLeftBox)
        self.extraColumLayout.setSpacing(0)
        self.extraColumLayout.setObjectName(u"extraColumLayout")
        self.extraColumLayout.setContentsMargins(0, 0, 0, 0)
        self.extraTopBg = QFrame(self.extraLeftBox)
        self.extraTopBg.setObjectName(u"extraTopBg")
        self.extraTopBg.setMinimumSize(QSize(0, 10))
        self.extraTopBg.setMaximumSize(QSize(16777215, 50))
        self.extraTopBg.setFrameShape(QFrame.NoFrame)
        self.extraTopBg.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.extraTopBg)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)

        self.extraColumLayout.addWidget(self.extraTopBg)

        self.extraContent = QFrame(self.extraLeftBox)
        self.extraContent.setObjectName(u"extraContent")
        self.extraContent.setFrameShape(QFrame.NoFrame)
        self.extraContent.setFrameShadow(QFrame.Raised)
        self.verticalLayout_11 = QVBoxLayout(self.extraContent)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.extraCenter = QFrame(self.extraContent)
        self.extraCenter.setObjectName(u"extraCenter")
        sizePolicy.setHeightForWidth(self.extraCenter.sizePolicy().hasHeightForWidth())
        self.extraCenter.setSizePolicy(sizePolicy)
        self.extraCenter.setMinimumSize(QSize(0, 265))
        self.extraCenter.setMaximumSize(QSize(16777215, 265))
        self.extraCenter.setFrameShape(QFrame.NoFrame)
        self.extraCenter.setFrameShadow(QFrame.Raised)
        self.verticalLayout_10 = QVBoxLayout(self.extraCenter)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(9, -1, 9, 0)
        self.textEdit = QTextEdit(self.extraCenter)
        self.textEdit.setObjectName(u"textEdit")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy2)
        self.textEdit.setMinimumSize(QSize(222, 265))
        self.textEdit.setMaximumSize(QSize(222, 265))
        self.textEdit.setStyleSheet(u"background: transparent;")
        self.textEdit.setFrameShape(QFrame.NoFrame)
        self.textEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.textEdit.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.textEdit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.textEdit.setReadOnly(True)

        self.verticalLayout_10.addWidget(self.textEdit)


        self.verticalLayout_11.addWidget(self.extraCenter)

        self.frame_61 = QFrame(self.extraContent)
        self.frame_61.setObjectName(u"frame_61")
        self.frame_61.setMaximumSize(QSize(16777215, 180))
        self.frame_61.setFrameShape(QFrame.NoFrame)
        self.frame_61.setFrameShadow(QFrame.Raised)
        self.verticalLayout_12 = QVBoxLayout(self.frame_61)
        self.verticalLayout_12.setSpacing(0)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.plainTextEdit_2 = QPlainTextEdit(self.frame_61)
        self.plainTextEdit_2.setObjectName(u"plainTextEdit_2")
        self.plainTextEdit_2.setMinimumSize(QSize(222, 0))
        self.plainTextEdit_2.setMaximumSize(QSize(16777215, 240))
        self.plainTextEdit_2.setStyleSheet(u"background: transparent;")
        self.plainTextEdit_2.setFrameShape(QFrame.NoFrame)
        self.plainTextEdit_2.setLineWrapMode(QPlainTextEdit.NoWrap)

        self.verticalLayout_12.addWidget(self.plainTextEdit_2)


        self.verticalLayout_11.addWidget(self.frame_61)

        self.verticalSpacer_3 = QSpacerItem(20, 194, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_3)


        self.extraColumLayout.addWidget(self.extraContent)


        self.appLayout.addWidget(self.extraLeftBox)

        self.contentBox = QFrame(self.bgApp)
        self.contentBox.setObjectName(u"contentBox")
        self.contentBox.setFrameShape(QFrame.NoFrame)
        self.contentBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.contentBox)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.contentTopBg = QFrame(self.contentBox)
        self.contentTopBg.setObjectName(u"contentTopBg")
        self.contentTopBg.setMinimumSize(QSize(0, 70))
        self.contentTopBg.setMaximumSize(QSize(16777215, 70))
        self.contentTopBg.setFrameShape(QFrame.NoFrame)
        self.contentTopBg.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.contentTopBg)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 10, 0)
        self.leftBox = QFrame(self.contentTopBg)
        self.leftBox.setObjectName(u"leftBox")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.leftBox.sizePolicy().hasHeightForWidth())
        self.leftBox.setSizePolicy(sizePolicy3)
        self.leftBox.setFrameShape(QFrame.NoFrame)
        self.leftBox.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.leftBox)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.titleRightInfo = QLabel(self.leftBox)
        self.titleRightInfo.setObjectName(u"titleRightInfo")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.titleRightInfo.sizePolicy().hasHeightForWidth())
        self.titleRightInfo.setSizePolicy(sizePolicy4)
        self.titleRightInfo.setMaximumSize(QSize(16777215, 45))
        self.titleRightInfo.setFont(font)
        self.titleRightInfo.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.titleRightInfo)


        self.horizontalLayout.addWidget(self.leftBox)

        self.rightButtons = QFrame(self.contentTopBg)
        self.rightButtons.setObjectName(u"rightButtons")
        self.rightButtons.setMinimumSize(QSize(0, 28))
        self.rightButtons.setFrameShape(QFrame.NoFrame)
        self.rightButtons.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.rightButtons)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.settingsTopBtn = QPushButton(self.rightButtons)
        self.settingsTopBtn.setObjectName(u"settingsTopBtn")
        self.settingsTopBtn.setMinimumSize(QSize(28, 28))
        self.settingsTopBtn.setMaximumSize(QSize(28, 28))
        self.settingsTopBtn.setCursor(QCursor(Qt.PointingHandCursor))
        icon = QIcon()
        icon.addFile(u":/icons/images/icons/icon_settings.png", QSize(), QIcon.Normal, QIcon.Off)
        self.settingsTopBtn.setIcon(icon)
        self.settingsTopBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_2.addWidget(self.settingsTopBtn)

        self.minimizeAppBtn = QPushButton(self.rightButtons)
        self.minimizeAppBtn.setObjectName(u"minimizeAppBtn")
        self.minimizeAppBtn.setMinimumSize(QSize(28, 28))
        self.minimizeAppBtn.setMaximumSize(QSize(28, 28))
        self.minimizeAppBtn.setCursor(QCursor(Qt.PointingHandCursor))
        icon1 = QIcon()
        icon1.addFile(u":/icons/images/icons/icon_minimize.png", QSize(), QIcon.Normal, QIcon.Off)
        self.minimizeAppBtn.setIcon(icon1)
        self.minimizeAppBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_2.addWidget(self.minimizeAppBtn)

        self.maximizeRestoreAppBtn = QPushButton(self.rightButtons)
        self.maximizeRestoreAppBtn.setObjectName(u"maximizeRestoreAppBtn")
        self.maximizeRestoreAppBtn.setMinimumSize(QSize(28, 28))
        self.maximizeRestoreAppBtn.setMaximumSize(QSize(28, 28))
        font3 = QFont()
        font3.setFamilies([u"Segoe UI"])
        font3.setPointSize(10)
        font3.setBold(False)
        font3.setItalic(False)
        font3.setStyleStrategy(QFont.PreferDefault)
        self.maximizeRestoreAppBtn.setFont(font3)
        self.maximizeRestoreAppBtn.setCursor(QCursor(Qt.PointingHandCursor))
        icon2 = QIcon()
        icon2.addFile(u":/icons/images/icons/icon_maximize.png", QSize(), QIcon.Normal, QIcon.Off)
        self.maximizeRestoreAppBtn.setIcon(icon2)
        self.maximizeRestoreAppBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_2.addWidget(self.maximizeRestoreAppBtn)

        self.closeAppBtn = QPushButton(self.rightButtons)
        self.closeAppBtn.setObjectName(u"closeAppBtn")
        self.closeAppBtn.setMinimumSize(QSize(28, 28))
        self.closeAppBtn.setMaximumSize(QSize(28, 28))
        self.closeAppBtn.setCursor(QCursor(Qt.PointingHandCursor))
        icon3 = QIcon()
        icon3.addFile(u":/icons/images/icons/icon_close.png", QSize(), QIcon.Normal, QIcon.Off)
        self.closeAppBtn.setIcon(icon3)
        self.closeAppBtn.setIconSize(QSize(20, 20))

        self.horizontalLayout_2.addWidget(self.closeAppBtn)


        self.horizontalLayout.addWidget(self.rightButtons, 0, Qt.AlignRight)


        self.verticalLayout_2.addWidget(self.contentTopBg)

        self.contentBottom = QFrame(self.contentBox)
        self.contentBottom.setObjectName(u"contentBottom")
        self.contentBottom.setFrameShape(QFrame.NoFrame)
        self.contentBottom.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.contentBottom)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.content = QFrame(self.contentBottom)
        self.content.setObjectName(u"content")
        self.content.setFrameShape(QFrame.NoFrame)
        self.content.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.content)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.pagesContainer = QFrame(self.content)
        self.pagesContainer.setObjectName(u"pagesContainer")
        self.pagesContainer.setStyleSheet(u"")
        self.pagesContainer.setFrameShape(QFrame.NoFrame)
        self.pagesContainer.setFrameShadow(QFrame.Raised)
        self.verticalLayout_15 = QVBoxLayout(self.pagesContainer)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(10, 10, 10, 10)
        self.stackedWidget = QStackedWidget(self.pagesContainer)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setStyleSheet(u"background: transparent;")
        self.train = QWidget()
        self.train.setObjectName(u"train")
        self.train.setStyleSheet(u"")
        self.verticalLayout_21 = QVBoxLayout(self.train)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.frame_title_tr = QFrame(self.train)
        self.frame_title_tr.setObjectName(u"frame_title_tr")
        self.frame_title_tr.setFrameShape(QFrame.StyledPanel)
        self.frame_title_tr.setFrameShadow(QFrame.Raised)
        self.verticalLayout_22 = QVBoxLayout(self.frame_title_tr)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.frame_2 = QFrame(self.frame_title_tr)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_23 = QVBoxLayout(self.frame_2)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.title_tr = QLabel(self.frame_2)
        self.title_tr.setObjectName(u"title_tr")
        self.title_tr.setFont(font)

        self.verticalLayout_23.addWidget(self.title_tr)


        self.verticalLayout_22.addWidget(self.frame_2)


        self.verticalLayout_21.addWidget(self.frame_title_tr)

        self.frame_progress_tr = QFrame(self.train)
        self.frame_progress_tr.setObjectName(u"frame_progress_tr")
        self.frame_progress_tr.setMaximumSize(QSize(16777215, 80))
        self.frame_progress_tr.setFrameShape(QFrame.StyledPanel)
        self.frame_progress_tr.setFrameShadow(QFrame.Raised)
        self.verticalLayout_24 = QVBoxLayout(self.frame_progress_tr)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.progress_tr = QProgressBar(self.frame_progress_tr)
        self.progress_tr.setObjectName(u"progress_tr")
        self.progress_tr.setStyleSheet(u"QProgressBar{\n"
"	font-size: 12px;\n"
"	background-color: rgb(81, 82, 124);\n"
"	color: rgb(241, 241, 241);\n"
"	border-style: none;\n"
"	border-radius: 10px;\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: rgb(189, 147, 249);\n"
"	border-radius: 10px;\n"
"}")
        self.progress_tr.setValue(24)

        self.verticalLayout_24.addWidget(self.progress_tr)

        self.frame_16 = QFrame(self.frame_progress_tr)
        self.frame_16.setObjectName(u"frame_16")
        self.frame_16.setMinimumSize(QSize(0, 0))
        self.frame_16.setFrameShape(QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_16)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.label_progress_tr = QLabel(self.frame_16)
        self.label_progress_tr.setObjectName(u"label_progress_tr")

        self.horizontalLayout_13.addWidget(self.label_progress_tr)

        self.label_suffix_tr = QLabel(self.frame_16)
        self.label_suffix_tr.setObjectName(u"label_suffix_tr")

        self.horizontalLayout_13.addWidget(self.label_suffix_tr)


        self.verticalLayout_24.addWidget(self.frame_16)


        self.verticalLayout_21.addWidget(self.frame_progress_tr)

        self.frame_run_tr = QFrame(self.train)
        self.frame_run_tr.setObjectName(u"frame_run_tr")
        self.frame_run_tr.setFrameShape(QFrame.StyledPanel)
        self.frame_run_tr.setFrameShadow(QFrame.Raised)
        self.verticalLayout_25 = QVBoxLayout(self.frame_run_tr)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.btn_run_tr = QPushButton(self.frame_run_tr)
        self.btn_run_tr.setObjectName(u"btn_run_tr")
        self.btn_run_tr.setMinimumSize(QSize(0, 45))
        self.btn_run_tr.setFont(font)

        self.verticalLayout_25.addWidget(self.btn_run_tr)


        self.verticalLayout_21.addWidget(self.frame_run_tr)

        self.frame_mlf = QFrame(self.train)
        self.frame_mlf.setObjectName(u"frame_mlf")
        self.frame_mlf.setMaximumSize(QSize(16777215, 30))
        self.frame_mlf.setFrameShape(QFrame.StyledPanel)
        self.frame_mlf.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_mlf)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.link_mlf = QLabel(self.frame_mlf)
        self.link_mlf.setObjectName(u"link_mlf")
        self.link_mlf.setOpenExternalLinks(True)

        self.horizontalLayout_5.addWidget(self.link_mlf)


        self.verticalLayout_21.addWidget(self.frame_mlf)

        self.stackedWidget.addWidget(self.train)
        self.generate = QWidget()
        self.generate.setObjectName(u"generate")
        self.generate.setStyleSheet(u"b")
        self.verticalLayout_18 = QVBoxLayout(self.generate)
        self.verticalLayout_18.setSpacing(0)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.title_gr = QLabel(self.generate)
        self.title_gr.setObjectName(u"title_gr")
        self.title_gr.setMinimumSize(QSize(0, 150))
        self.title_gr.setFont(font)

        self.verticalLayout_18.addWidget(self.title_gr)

        self.frame_22 = QFrame(self.generate)
        self.frame_22.setObjectName(u"frame_22")
        self.frame_22.setFrameShape(QFrame.StyledPanel)
        self.frame_22.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame_22)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.progress_gr = QProgressBar(self.frame_22)
        self.progress_gr.setObjectName(u"progress_gr")
        self.progress_gr.setStyleSheet(u"QProgressBar{\n"
"	font-size: 12px;\n"
"	background-color: rgb(81, 82, 124);\n"
"	color: rgb(241, 241, 241);\n"
"	border-style: none;\n"
"	border-radius: 10px;\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: rgb(189, 147, 249);\n"
"	border-radius: 10px;\n"
"}")
        self.progress_gr.setValue(24)

        self.verticalLayout.addWidget(self.progress_gr)

        self.frame_59 = QFrame(self.frame_22)
        self.frame_59.setObjectName(u"frame_59")
        self.frame_59.setMinimumSize(QSize(0, 0))
        self.frame_59.setFrameShape(QFrame.StyledPanel)
        self.frame_59.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_17 = QHBoxLayout(self.frame_59)
        self.horizontalLayout_17.setSpacing(0)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.label_progress_gr = QLabel(self.frame_59)
        self.label_progress_gr.setObjectName(u"label_progress_gr")

        self.horizontalLayout_17.addWidget(self.label_progress_gr)

        self.label_suffix_gr = QLabel(self.frame_59)
        self.label_suffix_gr.setObjectName(u"label_suffix_gr")

        self.horizontalLayout_17.addWidget(self.label_suffix_gr)


        self.verticalLayout.addWidget(self.frame_59)


        self.verticalLayout_18.addWidget(self.frame_22)

        self.frame_18 = QFrame(self.generate)
        self.frame_18.setObjectName(u"frame_18")
        self.frame_18.setMinimumSize(QSize(0, 0))
        self.frame_18.setFrameShape(QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_18)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(50, 0, 50, 0)
        self.frame_25 = QFrame(self.frame_18)
        self.frame_25.setObjectName(u"frame_25")
        self.frame_25.setMaximumSize(QSize(16777215, 16777215))
        self.frame_25.setFrameShape(QFrame.StyledPanel)
        self.frame_25.setFrameShadow(QFrame.Raised)
        self.verticalLayout_17 = QVBoxLayout(self.frame_25)
        self.verticalLayout_17.setSpacing(0)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.frame_21 = QFrame(self.frame_25)
        self.frame_21.setObjectName(u"frame_21")
        self.frame_21.setFrameShape(QFrame.StyledPanel)
        self.frame_21.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_11 = QHBoxLayout(self.frame_21)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_tree_gr = QLabel(self.frame_21)
        self.label_tree_gr.setObjectName(u"label_tree_gr")

        self.horizontalLayout_11.addWidget(self.label_tree_gr)


        self.verticalLayout_17.addWidget(self.frame_21)

        self.tree_gr = QTreeWidget(self.frame_25)
        self.tree_gr.headerItem().setText(0, "")
        self.tree_gr.setObjectName(u"tree_gr")
        self.tree_gr.setStyleSheet(u"\n"
"\n"
"QTreeView::item:selected {\n"
"		color: #282a36;\n"
"        background: #8BE9FD;\n"
"}\n"
"")
        self.tree_gr.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tree_gr.setSortingEnabled(False)

        self.verticalLayout_17.addWidget(self.tree_gr)


        self.horizontalLayout_9.addWidget(self.frame_25)


        self.verticalLayout_18.addWidget(self.frame_18)

        self.frame_23 = QFrame(self.generate)
        self.frame_23.setObjectName(u"frame_23")
        self.frame_23.setFrameShape(QFrame.StyledPanel)
        self.frame_23.setFrameShadow(QFrame.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.frame_23)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(-1, 0, -1, 80)
        self.btn_run_gr = QPushButton(self.frame_23)
        self.btn_run_gr.setObjectName(u"btn_run_gr")
        self.btn_run_gr.setMinimumSize(QSize(0, 45))
        self.btn_run_gr.setFont(font)

        self.verticalLayout_16.addWidget(self.btn_run_gr)

        self.btn_vis_gr = QPushButton(self.frame_23)
        self.btn_vis_gr.setObjectName(u"btn_vis_gr")
        self.btn_vis_gr.setMinimumSize(QSize(0, 30))
        self.btn_vis_gr.setFont(font)
        self.btn_vis_gr.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(57, 65, 80);\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"QPushButton:pressed {	\n"
"	background-color: rgb(35, 40, 49);\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}")
        icon4 = QIcon()
        icon4.addFile(u":/icons/images/icons/icon_project_diagram.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_vis_gr.setIcon(icon4)

        self.verticalLayout_16.addWidget(self.btn_vis_gr)


        self.verticalLayout_18.addWidget(self.frame_23)

        self.stackedWidget.addWidget(self.generate)
        self.evaluate = QWidget()
        self.evaluate.setObjectName(u"evaluate")
        self.verticalLayout_48 = QVBoxLayout(self.evaluate)
        self.verticalLayout_48.setSpacing(0)
        self.verticalLayout_48.setObjectName(u"verticalLayout_48")
        self.title_ev = QLabel(self.evaluate)
        self.title_ev.setObjectName(u"title_ev")
        self.title_ev.setMinimumSize(QSize(0, 150))
        self.title_ev.setFont(font)

        self.verticalLayout_48.addWidget(self.title_ev)

        self.frame_49 = QFrame(self.evaluate)
        self.frame_49.setObjectName(u"frame_49")
        self.frame_49.setFrameShape(QFrame.StyledPanel)
        self.frame_49.setFrameShadow(QFrame.Raised)
        self.verticalLayout_42 = QVBoxLayout(self.frame_49)
        self.verticalLayout_42.setObjectName(u"verticalLayout_42")
        self.progress_ev = QProgressBar(self.frame_49)
        self.progress_ev.setObjectName(u"progress_ev")
        self.progress_ev.setStyleSheet(u"QProgressBar{\n"
"	font-size: 12px;\n"
"	background-color: rgb(81, 82, 124);\n"
"	color: rgb(241, 241, 241);\n"
"	border-style: none;\n"
"	border-radius: 10px;\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: rgb(189, 147, 249);\n"
"	border-radius: 10px;\n"
"}")
        self.progress_ev.setValue(24)

        self.verticalLayout_42.addWidget(self.progress_ev)

        self.frame_60 = QFrame(self.frame_49)
        self.frame_60.setObjectName(u"frame_60")
        self.frame_60.setMinimumSize(QSize(0, 0))
        self.frame_60.setFrameShape(QFrame.StyledPanel)
        self.frame_60.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_18 = QHBoxLayout(self.frame_60)
        self.horizontalLayout_18.setSpacing(0)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.label_progress_ev = QLabel(self.frame_60)
        self.label_progress_ev.setObjectName(u"label_progress_ev")

        self.horizontalLayout_18.addWidget(self.label_progress_ev)

        self.label_suffix_ev = QLabel(self.frame_60)
        self.label_suffix_ev.setObjectName(u"label_suffix_ev")

        self.horizontalLayout_18.addWidget(self.label_suffix_ev)


        self.verticalLayout_42.addWidget(self.frame_60)


        self.verticalLayout_48.addWidget(self.frame_49)

        self.frame_50 = QFrame(self.evaluate)
        self.frame_50.setObjectName(u"frame_50")
        self.frame_50.setMinimumSize(QSize(0, 0))
        self.frame_50.setFrameShape(QFrame.StyledPanel)
        self.frame_50.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_12 = QHBoxLayout(self.frame_50)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(50, 0, 50, 0)
        self.frame_51 = QFrame(self.frame_50)
        self.frame_51.setObjectName(u"frame_51")
        self.frame_51.setMaximumSize(QSize(16777215, 16777215))
        self.frame_51.setFrameShape(QFrame.StyledPanel)
        self.frame_51.setFrameShadow(QFrame.Raised)
        self.verticalLayout_43 = QVBoxLayout(self.frame_51)
        self.verticalLayout_43.setSpacing(0)
        self.verticalLayout_43.setObjectName(u"verticalLayout_43")
        self.verticalLayout_43.setContentsMargins(0, 0, 0, 0)
        self.frame_52 = QFrame(self.frame_51)
        self.frame_52.setObjectName(u"frame_52")
        self.frame_52.setFrameShape(QFrame.StyledPanel)
        self.frame_52.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_52)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_tree_ev = QLabel(self.frame_52)
        self.label_tree_ev.setObjectName(u"label_tree_ev")

        self.horizontalLayout_14.addWidget(self.label_tree_ev)


        self.verticalLayout_43.addWidget(self.frame_52)

        self.tree_ev = QTreeWidget(self.frame_51)
        self.tree_ev.headerItem().setText(0, "")
        self.tree_ev.setObjectName(u"tree_ev")
        self.tree_ev.setStyleSheet(u"\n"
"\n"
"QTreeView::item:selected {\n"
"		color: #282a36;\n"
"        background: #8BE9FD;\n"
"}\n"
"")
        self.tree_ev.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tree_ev.setSortingEnabled(False)

        self.verticalLayout_43.addWidget(self.tree_ev)


        self.horizontalLayout_12.addWidget(self.frame_51)


        self.verticalLayout_48.addWidget(self.frame_50)

        self.frame_32 = QFrame(self.evaluate)
        self.frame_32.setObjectName(u"frame_32")
        self.frame_32.setFrameShape(QFrame.StyledPanel)
        self.frame_32.setFrameShadow(QFrame.Raised)
        self.verticalLayout_26 = QVBoxLayout(self.frame_32)
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.verticalLayout_26.setContentsMargins(-1, 0, -1, 80)
        self.btn_run_ev = QPushButton(self.frame_32)
        self.btn_run_ev.setObjectName(u"btn_run_ev")
        self.btn_run_ev.setMinimumSize(QSize(0, 45))
        self.btn_run_ev.setFont(font)

        self.verticalLayout_26.addWidget(self.btn_run_ev)

        self.btn_vis_ev = QPushButton(self.frame_32)
        self.btn_vis_ev.setObjectName(u"btn_vis_ev")
        self.btn_vis_ev.setMinimumSize(QSize(0, 30))
        self.btn_vis_ev.setFont(font)
        self.btn_vis_ev.setStyleSheet(u"")

        self.verticalLayout_26.addWidget(self.btn_vis_ev)


        self.verticalLayout_48.addWidget(self.frame_32)

        self.stackedWidget.addWidget(self.evaluate)

        self.verticalLayout_15.addWidget(self.stackedWidget)


        self.horizontalLayout_4.addWidget(self.pagesContainer)

        self.extraRightBox = QFrame(self.content)
        self.extraRightBox.setObjectName(u"extraRightBox")
        self.extraRightBox.setMinimumSize(QSize(0, 0))
        self.extraRightBox.setMaximumSize(QSize(0, 16777215))
        self.extraRightBox.setFrameShape(QFrame.NoFrame)
        self.extraRightBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.extraRightBox)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.contentSettings = QFrame(self.extraRightBox)
        self.contentSettings.setObjectName(u"contentSettings")
        self.contentSettings.setFrameShape(QFrame.NoFrame)
        self.contentSettings.setFrameShadow(QFrame.Raised)
        self.verticalLayout_13 = QVBoxLayout(self.contentSettings)
        self.verticalLayout_13.setSpacing(0)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self.contentSettings)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(0, 40))
        self.frame.setMaximumSize(QSize(16777215, 16777215))
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_40 = QVBoxLayout(self.frame)
        self.verticalLayout_40.setObjectName(u"verticalLayout_40")
        self.frame_26 = QFrame(self.frame)
        self.frame_26.setObjectName(u"frame_26")
        self.frame_26.setFrameShape(QFrame.NoFrame)
        self.frame_26.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_26)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(-1, 0, -1, 0)
        self.frame_31 = QFrame(self.frame_26)
        self.frame_31.setObjectName(u"frame_31")
        self.frame_31.setFrameShape(QFrame.NoFrame)
        self.frame_31.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_31)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(-1, 6, -1, 6)
        self.btn_save_conf = QPushButton(self.frame_31)
        self.btn_save_conf.setObjectName(u"btn_save_conf")
        self.btn_save_conf.setMinimumSize(QSize(95, 25))
        self.btn_save_conf.setStyleSheet(u"")
        icon5 = QIcon()
        icon5.addFile(u":/icons/images/icons/cil-save.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_save_conf.setIcon(icon5)
        self.btn_save_conf.setIconSize(QSize(14, 14))

        self.horizontalLayout_6.addWidget(self.btn_save_conf)

        self.btn_load_conf = QPushButton(self.frame_31)
        self.btn_load_conf.setObjectName(u"btn_load_conf")
        self.btn_load_conf.setMinimumSize(QSize(95, 25))
        self.btn_load_conf.setStyleSheet(u"")
        icon6 = QIcon()
        icon6.addFile(u":/icons/images/icons/cil-cloud-upload.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_load_conf.setIcon(icon6)
        self.btn_load_conf.setIconSize(QSize(14, 14))

        self.horizontalLayout_6.addWidget(self.btn_load_conf)

        self.btn_restore_conf = QPushButton(self.frame_31)
        self.btn_restore_conf.setObjectName(u"btn_restore_conf")
        self.btn_restore_conf.setMinimumSize(QSize(95, 25))
        self.btn_restore_conf.setStyleSheet(u"")
        icon7 = QIcon()
        icon7.addFile(u":/icons/images/icons/cil-reload.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_restore_conf.setIcon(icon7)
        self.btn_restore_conf.setIconSize(QSize(14, 14))

        self.horizontalLayout_6.addWidget(self.btn_restore_conf)

        self.btn_apply_conf = QPushButton(self.frame_31)
        self.btn_apply_conf.setObjectName(u"btn_apply_conf")
        self.btn_apply_conf.setMinimumSize(QSize(95, 32))
        self.btn_apply_conf.setStyleSheet(u"QPushButton { font-size: 14px; color: #282a36; background-color: #69FF94; } QPushButton:hover { background-color: #50fa7b; } QPushButton:pressed { background-color: #69FF94; }")

        self.horizontalLayout_6.addWidget(self.btn_apply_conf)


        self.horizontalLayout_7.addWidget(self.frame_31)


        self.verticalLayout_40.addWidget(self.frame_26)

        self.conf_selection = QComboBox(self.frame)
        self.conf_selection.addItem("")
        self.conf_selection.addItem("")
        self.conf_selection.addItem("")
        self.conf_selection.setObjectName(u"conf_selection")
        self.conf_selection.setFont(font)
        self.conf_selection.setAutoFillBackground(False)
        self.conf_selection.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.conf_selection.setIconSize(QSize(16, 16))
        self.conf_selection.setFrame(True)

        self.verticalLayout_40.addWidget(self.conf_selection)


        self.verticalLayout_13.addWidget(self.frame)

        self.frame_3 = QFrame(self.contentSettings)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.NoFrame)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_20 = QVBoxLayout(self.frame_3)
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.stackedWidget_2 = QStackedWidget(self.frame_3)
        self.stackedWidget_2.setObjectName(u"stackedWidget_2")
        self.stackedWidget_2.setStyleSheet(u"background: transparent;")
        self.conf_tr = QWidget()
        self.conf_tr.setObjectName(u"conf_tr")
        self.horizontalLayout_10 = QHBoxLayout(self.conf_tr)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_2 = QScrollArea(self.conf_tr)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setFrameShape(QFrame.NoFrame)
        self.scrollArea_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 414, 1043))
        self.verticalLayout_28 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_28.setSpacing(11)
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.verticalLayout_28.setContentsMargins(5, 10, 0, 9)
        self.conf_tr_row_1 = QFrame(self.scrollAreaWidgetContents_2)
        self.conf_tr_row_1.setObjectName(u"conf_tr_row_1")
        self.conf_tr_row_1.setFrameShape(QFrame.StyledPanel)
        self.conf_tr_row_1.setFrameShadow(QFrame.Raised)
        self.verticalLayout_14 = QVBoxLayout(self.conf_tr_row_1)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.label_row_1_tr = QLabel(self.conf_tr_row_1)
        self.label_row_1_tr.setObjectName(u"label_row_1_tr")
        self.label_row_1_tr.setMaximumSize(QSize(16777215, 25))

        self.verticalLayout_14.addWidget(self.label_row_1_tr)

        self.frame_5 = QFrame(self.conf_tr_row_1)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.gridLayout_3 = QGridLayout(self.frame_5)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.frame_10 = QFrame(self.frame_5)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setFrameShape(QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QFrame.Raised)
        self.verticalLayout_32 = QVBoxLayout(self.frame_10)
        self.verticalLayout_32.setSpacing(3)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.label_n_ckt_tr = QLabel(self.frame_10)
        self.label_n_ckt_tr.setObjectName(u"label_n_ckt_tr")
        self.label_n_ckt_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_32.addWidget(self.label_n_ckt_tr)

        self.spin_n_ckt_tr = QSpinBox(self.frame_10)
        self.spin_n_ckt_tr.setObjectName(u"spin_n_ckt_tr")

        self.verticalLayout_32.addWidget(self.spin_n_ckt_tr)


        self.gridLayout_3.addWidget(self.frame_10, 2, 0, 1, 1)

        self.frame_7 = QFrame(self.frame_5)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.verticalLayout_27 = QVBoxLayout(self.frame_7)
        self.verticalLayout_27.setSpacing(3)
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.label_seed_tr = QLabel(self.frame_7)
        self.label_seed_tr.setObjectName(u"label_seed_tr")
        self.label_seed_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_27.addWidget(self.label_seed_tr)

        self.spin_seed_tr = QSpinBox(self.frame_7)
        self.spin_seed_tr.setObjectName(u"spin_seed_tr")

        self.verticalLayout_27.addWidget(self.spin_seed_tr)


        self.gridLayout_3.addWidget(self.frame_7, 1, 0, 1, 1)

        self.frame_17 = QFrame(self.frame_5)
        self.frame_17.setObjectName(u"frame_17")
        self.frame_17.setFrameShape(QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QFrame.Raised)
        self.verticalLayout_37 = QVBoxLayout(self.frame_17)
        self.verticalLayout_37.setSpacing(3)
        self.verticalLayout_37.setObjectName(u"verticalLayout_37")
        self.label_lr_mile_tr = QLabel(self.frame_17)
        self.label_lr_mile_tr.setObjectName(u"label_lr_mile_tr")
        self.label_lr_mile_tr.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_37.addWidget(self.label_lr_mile_tr)

        self.line_lr_mile_tr = QLineEdit(self.frame_17)
        self.line_lr_mile_tr.setObjectName(u"line_lr_mile_tr")
        self.line_lr_mile_tr.setMinimumSize(QSize(0, 25))
        self.line_lr_mile_tr.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.verticalLayout_37.addWidget(self.line_lr_mile_tr)


        self.gridLayout_3.addWidget(self.frame_17, 4, 0, 1, 3)

        self.frame_12 = QFrame(self.frame_5)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setFrameShape(QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Raised)
        self.verticalLayout_36 = QVBoxLayout(self.frame_12)
        self.verticalLayout_36.setSpacing(3)
        self.verticalLayout_36.setObjectName(u"verticalLayout_36")
        self.label_n_graphs_ev_tr = QLabel(self.frame_12)
        self.label_n_graphs_ev_tr.setObjectName(u"label_n_graphs_ev_tr")
        self.label_n_graphs_ev_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_36.addWidget(self.label_n_graphs_ev_tr)

        self.spin_n_graphs_ev_tr = QSpinBox(self.frame_12)
        self.spin_n_graphs_ev_tr.setObjectName(u"spin_n_graphs_ev_tr")

        self.verticalLayout_36.addWidget(self.spin_n_graphs_ev_tr)


        self.gridLayout_3.addWidget(self.frame_12, 2, 2, 1, 1)

        self.frame_8 = QFrame(self.frame_5)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.verticalLayout_30 = QVBoxLayout(self.frame_8)
        self.verticalLayout_30.setSpacing(3)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.label_num_ep_tr = QLabel(self.frame_8)
        self.label_num_ep_tr.setObjectName(u"label_num_ep_tr")
        self.label_num_ep_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_30.addWidget(self.label_num_ep_tr)

        self.spin_num_ep_tr = QSpinBox(self.frame_8)
        self.spin_num_ep_tr.setObjectName(u"spin_num_ep_tr")

        self.verticalLayout_30.addWidget(self.spin_num_ep_tr)


        self.gridLayout_3.addWidget(self.frame_8, 1, 1, 1, 1)

        self.frame_15 = QFrame(self.frame_5)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setFrameShape(QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QFrame.Raised)
        self.verticalLayout_33 = QVBoxLayout(self.frame_15)
        self.verticalLayout_33.setSpacing(3)
        self.verticalLayout_33.setObjectName(u"verticalLayout_33")
        self.label_lr_decay_tr = QLabel(self.frame_15)
        self.label_lr_decay_tr.setObjectName(u"label_lr_decay_tr")
        self.label_lr_decay_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_33.addWidget(self.label_lr_decay_tr)

        self.spin_lr_decay_tr = QDoubleSpinBox(self.frame_15)
        self.spin_lr_decay_tr.setObjectName(u"spin_lr_decay_tr")

        self.verticalLayout_33.addWidget(self.spin_lr_decay_tr)


        self.gridLayout_3.addWidget(self.frame_15, 3, 2, 1, 1)

        self.frame_11 = QFrame(self.frame_5)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setFrameShape(QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Raised)
        self.verticalLayout_29 = QVBoxLayout(self.frame_11)
        self.verticalLayout_29.setSpacing(3)
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.label_batch_ev_tr = QLabel(self.frame_11)
        self.label_batch_ev_tr.setObjectName(u"label_batch_ev_tr")
        self.label_batch_ev_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_29.addWidget(self.label_batch_ev_tr)

        self.spin_batch_ev_tr = QSpinBox(self.frame_11)
        self.spin_batch_ev_tr.setObjectName(u"spin_batch_ev_tr")

        self.verticalLayout_29.addWidget(self.spin_batch_ev_tr)


        self.gridLayout_3.addWidget(self.frame_11, 2, 1, 1, 1)

        self.frame_9 = QFrame(self.frame_5)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.verticalLayout_31 = QVBoxLayout(self.frame_9)
        self.verticalLayout_31.setSpacing(3)
        self.verticalLayout_31.setObjectName(u"verticalLayout_31")
        self.label_ep_to_ev_tr = QLabel(self.frame_9)
        self.label_ep_to_ev_tr.setObjectName(u"label_ep_to_ev_tr")
        self.label_ep_to_ev_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_31.addWidget(self.label_ep_to_ev_tr)

        self.spin_ep_to_ev_tr = QSpinBox(self.frame_9)
        self.spin_ep_to_ev_tr.setObjectName(u"spin_ep_to_ev_tr")

        self.verticalLayout_31.addWidget(self.spin_ep_to_ev_tr)


        self.gridLayout_3.addWidget(self.frame_9, 1, 2, 1, 1)

        self.frame_14 = QFrame(self.frame_5)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setFrameShape(QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QFrame.Raised)
        self.verticalLayout_34 = QVBoxLayout(self.frame_14)
        self.verticalLayout_34.setSpacing(3)
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.label_init_lr_tr = QLabel(self.frame_14)
        self.label_init_lr_tr.setObjectName(u"label_init_lr_tr")
        self.label_init_lr_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_34.addWidget(self.label_init_lr_tr)

        self.spin_init_lr_tr = QDoubleSpinBox(self.frame_14)
        self.spin_init_lr_tr.setObjectName(u"spin_init_lr_tr")

        self.verticalLayout_34.addWidget(self.spin_init_lr_tr)


        self.gridLayout_3.addWidget(self.frame_14, 3, 1, 1, 1)

        self.frame_13 = QFrame(self.frame_5)
        self.frame_13.setObjectName(u"frame_13")
        self.frame_13.setFrameShape(QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QFrame.Raised)
        self.verticalLayout_35 = QVBoxLayout(self.frame_13)
        self.verticalLayout_35.setSpacing(3)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.label_bs_samples_tr = QLabel(self.frame_13)
        self.label_bs_samples_tr.setObjectName(u"label_bs_samples_tr")
        self.label_bs_samples_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_35.addWidget(self.label_bs_samples_tr)

        self.spin_bs_samples_tr = QSpinBox(self.frame_13)
        self.spin_bs_samples_tr.setObjectName(u"spin_bs_samples_tr")

        self.verticalLayout_35.addWidget(self.spin_bs_samples_tr)


        self.gridLayout_3.addWidget(self.frame_13, 3, 0, 1, 1)


        self.verticalLayout_14.addWidget(self.frame_5)

        self.frame_6 = QFrame(self.conf_tr_row_1)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.verticalLayout_38 = QVBoxLayout(self.frame_6)
        self.verticalLayout_38.setObjectName(u"verticalLayout_38")
        self.frame_19 = QFrame(self.frame_6)
        self.frame_19.setObjectName(u"frame_19")
        self.frame_19.setFrameShape(QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QFrame.Raised)
        self.gridLayout_5 = QGridLayout(self.frame_19)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.check_d_ev_tr = QCheckBox(self.frame_19)
        self.check_d_ev_tr.setObjectName(u"check_d_ev_tr")
        self.check_d_ev_tr.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.check_d_ev_tr, 1, 0, 1, 1)

        self.check_b_ev_tr = QCheckBox(self.frame_19)
        self.check_b_ev_tr.setObjectName(u"check_b_ev_tr")
        self.check_b_ev_tr.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.check_b_ev_tr, 1, 2, 1, 2)

        self.check_a_ev_tr = QCheckBox(self.frame_19)
        self.check_a_ev_tr.setObjectName(u"check_a_ev_tr")
        self.check_a_ev_tr.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.check_a_ev_tr, 2, 0, 1, 1)

        self.label_metrics_ev_tr = QLabel(self.frame_19)
        self.label_metrics_ev_tr.setObjectName(u"label_metrics_ev_tr")
        self.label_metrics_ev_tr.setMaximumSize(QSize(16777215, 20))

        self.gridLayout_5.addWidget(self.label_metrics_ev_tr, 0, 0, 1, 5)

        self.check_c_ev_tr = QCheckBox(self.frame_19)
        self.check_c_ev_tr.setObjectName(u"check_c_ev_tr")
        self.check_c_ev_tr.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.check_c_ev_tr, 1, 1, 1, 1)

        self.check_pr_ev_tr = QCheckBox(self.frame_19)
        self.check_pr_ev_tr.setObjectName(u"check_pr_ev_tr")
        self.check_pr_ev_tr.setStyleSheet(u"")

        self.gridLayout_5.addWidget(self.check_pr_ev_tr, 2, 1, 1, 1)


        self.verticalLayout_38.addWidget(self.frame_19)


        self.verticalLayout_14.addWidget(self.frame_6)


        self.verticalLayout_28.addWidget(self.conf_tr_row_1)

        self.conf_tr_row_2 = QFrame(self.scrollAreaWidgetContents_2)
        self.conf_tr_row_2.setObjectName(u"conf_tr_row_2")
        self.conf_tr_row_2.setMinimumSize(QSize(0, 0))
        self.conf_tr_row_2.setFrameShape(QFrame.StyledPanel)
        self.conf_tr_row_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_61 = QVBoxLayout(self.conf_tr_row_2)
        self.verticalLayout_61.setSpacing(0)
        self.verticalLayout_61.setObjectName(u"verticalLayout_61")
        self.verticalLayout_61.setContentsMargins(0, 0, 0, 0)
        self.label_row_2_tr = QLabel(self.conf_tr_row_2)
        self.label_row_2_tr.setObjectName(u"label_row_2_tr")
        self.label_row_2_tr.setMaximumSize(QSize(16777215, 25))

        self.verticalLayout_61.addWidget(self.label_row_2_tr)

        self.frame_20 = QFrame(self.conf_tr_row_2)
        self.frame_20.setObjectName(u"frame_20")
        self.frame_20.setFrameShape(QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QFrame.Raised)
        self.gridLayout_7 = QGridLayout(self.frame_20)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.frame_42 = QFrame(self.frame_20)
        self.frame_42.setObjectName(u"frame_42")
        self.frame_42.setFrameShape(QFrame.StyledPanel)
        self.frame_42.setFrameShadow(QFrame.Raised)
        self.verticalLayout_58 = QVBoxLayout(self.frame_42)
        self.verticalLayout_58.setSpacing(3)
        self.verticalLayout_58.setObjectName(u"verticalLayout_58")
        self.label_in_emb_tr = QLabel(self.frame_42)
        self.label_in_emb_tr.setObjectName(u"label_in_emb_tr")
        self.label_in_emb_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_58.addWidget(self.label_in_emb_tr)

        self.spin_in_emb_tr = QSpinBox(self.frame_42)
        self.spin_in_emb_tr.setObjectName(u"spin_in_emb_tr")

        self.verticalLayout_58.addWidget(self.spin_in_emb_tr)


        self.gridLayout_7.addWidget(self.frame_42, 2, 2, 1, 1)

        self.frame_34 = QFrame(self.frame_20)
        self.frame_34.setObjectName(u"frame_34")
        self.frame_34.setFrameShape(QFrame.StyledPanel)
        self.frame_34.setFrameShadow(QFrame.Raised)
        self.verticalLayout_51 = QVBoxLayout(self.frame_34)
        self.verticalLayout_51.setSpacing(3)
        self.verticalLayout_51.setObjectName(u"verticalLayout_51")
        self.label_in_hd_tr = QLabel(self.frame_34)
        self.label_in_hd_tr.setObjectName(u"label_in_hd_tr")
        self.label_in_hd_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_51.addWidget(self.label_in_hd_tr)

        self.spin_in_hd_tr = QSpinBox(self.frame_34)
        self.spin_in_hd_tr.setObjectName(u"spin_in_hd_tr")

        self.verticalLayout_51.addWidget(self.spin_in_hd_tr)


        self.gridLayout_7.addWidget(self.frame_34, 2, 0, 1, 1)

        self.frame_41 = QFrame(self.frame_20)
        self.frame_41.setObjectName(u"frame_41")
        self.frame_41.setFrameShape(QFrame.StyledPanel)
        self.frame_41.setFrameShadow(QFrame.Raised)
        self.verticalLayout_57 = QVBoxLayout(self.frame_41)
        self.verticalLayout_57.setSpacing(3)
        self.verticalLayout_57.setObjectName(u"verticalLayout_57")
        self.label_out_mlp_emb_tr = QLabel(self.frame_41)
        self.label_out_mlp_emb_tr.setObjectName(u"label_out_mlp_emb_tr")
        self.label_out_mlp_emb_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_57.addWidget(self.label_out_mlp_emb_tr)

        self.spin_out_mlp_emb_tr = QSpinBox(self.frame_41)
        self.spin_out_mlp_emb_tr.setObjectName(u"spin_out_mlp_emb_tr")

        self.verticalLayout_57.addWidget(self.spin_out_mlp_emb_tr)


        self.gridLayout_7.addWidget(self.frame_41, 3, 1, 1, 1)

        self.frame_36 = QFrame(self.frame_20)
        self.frame_36.setObjectName(u"frame_36")
        self.frame_36.setFrameShape(QFrame.StyledPanel)
        self.frame_36.setFrameShadow(QFrame.Raised)
        self.verticalLayout_53 = QVBoxLayout(self.frame_36)
        self.verticalLayout_53.setSpacing(3)
        self.verticalLayout_53.setObjectName(u"verticalLayout_53")
        self.label_layers_tr = QLabel(self.frame_36)
        self.label_layers_tr.setObjectName(u"label_layers_tr")
        self.label_layers_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_53.addWidget(self.label_layers_tr)

        self.spin_layers_tr = QSpinBox(self.frame_36)
        self.spin_layers_tr.setObjectName(u"spin_layers_tr")

        self.verticalLayout_53.addWidget(self.spin_layers_tr)


        self.gridLayout_7.addWidget(self.frame_36, 3, 2, 1, 1)

        self.frame_37 = QFrame(self.frame_20)
        self.frame_37.setObjectName(u"frame_37")
        self.frame_37.setFrameShape(QFrame.StyledPanel)
        self.frame_37.setFrameShadow(QFrame.Raised)
        self.verticalLayout_54 = QVBoxLayout(self.frame_37)
        self.verticalLayout_54.setSpacing(3)
        self.verticalLayout_54.setObjectName(u"verticalLayout_54")
        self.label_out_hd_tr = QLabel(self.frame_37)
        self.label_out_hd_tr.setObjectName(u"label_out_hd_tr")
        self.label_out_hd_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_54.addWidget(self.label_out_hd_tr)

        self.spin_out_hd_tr = QSpinBox(self.frame_37)
        self.spin_out_hd_tr.setObjectName(u"spin_out_hd_tr")

        self.verticalLayout_54.addWidget(self.spin_out_hd_tr)


        self.gridLayout_7.addWidget(self.frame_37, 2, 1, 1, 1)

        self.frame_33 = QFrame(self.frame_20)
        self.frame_33.setObjectName(u"frame_33")
        self.frame_33.setFrameShape(QFrame.StyledPanel)
        self.frame_33.setFrameShadow(QFrame.Raised)
        self.verticalLayout_50 = QVBoxLayout(self.frame_33)
        self.verticalLayout_50.setSpacing(3)
        self.verticalLayout_50.setObjectName(u"verticalLayout_50")
        self.label_out_emb_tr = QLabel(self.frame_33)
        self.label_out_emb_tr.setObjectName(u"label_out_emb_tr")
        self.label_out_emb_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_50.addWidget(self.label_out_emb_tr)

        self.spin_out_emb_tr = QSpinBox(self.frame_33)
        self.spin_out_emb_tr.setObjectName(u"spin_out_emb_tr")

        self.verticalLayout_50.addWidget(self.spin_out_emb_tr)


        self.gridLayout_7.addWidget(self.frame_33, 3, 0, 1, 1)


        self.verticalLayout_61.addWidget(self.frame_20)


        self.verticalLayout_28.addWidget(self.conf_tr_row_2)

        self.conf_tr_row_3 = QFrame(self.scrollAreaWidgetContents_2)
        self.conf_tr_row_3.setObjectName(u"conf_tr_row_3")
        self.conf_tr_row_3.setFrameShape(QFrame.StyledPanel)
        self.conf_tr_row_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_63 = QVBoxLayout(self.conf_tr_row_3)
        self.verticalLayout_63.setSpacing(0)
        self.verticalLayout_63.setObjectName(u"verticalLayout_63")
        self.verticalLayout_63.setContentsMargins(0, 0, 0, 0)
        self.label_row_3_tr = QLabel(self.conf_tr_row_3)
        self.label_row_3_tr.setObjectName(u"label_row_3_tr")
        self.label_row_3_tr.setMaximumSize(QSize(16777215, 25))

        self.verticalLayout_63.addWidget(self.label_row_3_tr)

        self.frame_38 = QFrame(self.conf_tr_row_3)
        self.frame_38.setObjectName(u"frame_38")
        self.frame_38.setFrameShape(QFrame.StyledPanel)
        self.frame_38.setFrameShadow(QFrame.Raised)
        self.gridLayout_8 = QGridLayout(self.frame_38)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.frame_43 = QFrame(self.frame_38)
        self.frame_43.setObjectName(u"frame_43")
        self.frame_43.setFrameShape(QFrame.StyledPanel)
        self.frame_43.setFrameShadow(QFrame.Raised)
        self.verticalLayout_59 = QVBoxLayout(self.frame_43)
        self.verticalLayout_59.setSpacing(3)
        self.verticalLayout_59.setObjectName(u"verticalLayout_59")
        self.label_batch_loader_tr = QLabel(self.frame_43)
        self.label_batch_loader_tr.setObjectName(u"label_batch_loader_tr")
        self.label_batch_loader_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_59.addWidget(self.label_batch_loader_tr)

        self.spin_batch_loader_tr = QSpinBox(self.frame_43)
        self.spin_batch_loader_tr.setObjectName(u"spin_batch_loader_tr")

        self.verticalLayout_59.addWidget(self.spin_batch_loader_tr)


        self.gridLayout_8.addWidget(self.frame_43, 1, 2, 1, 1)

        self.frame_39 = QFrame(self.frame_38)
        self.frame_39.setObjectName(u"frame_39")
        self.frame_39.setFrameShape(QFrame.StyledPanel)
        self.frame_39.setFrameShadow(QFrame.Raised)
        self.verticalLayout_52 = QVBoxLayout(self.frame_39)
        self.verticalLayout_52.setSpacing(3)
        self.verticalLayout_52.setObjectName(u"verticalLayout_52")
        self.label_n_workers_tr = QLabel(self.frame_39)
        self.label_n_workers_tr.setObjectName(u"label_n_workers_tr")
        self.label_n_workers_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_52.addWidget(self.label_n_workers_tr)

        self.spin_n_workers_tr = QSpinBox(self.frame_39)
        self.spin_n_workers_tr.setObjectName(u"spin_n_workers_tr")

        self.verticalLayout_52.addWidget(self.spin_n_workers_tr)


        self.gridLayout_8.addWidget(self.frame_39, 1, 0, 1, 1)

        self.frame_40 = QFrame(self.frame_38)
        self.frame_40.setObjectName(u"frame_40")
        self.frame_40.setFrameShape(QFrame.StyledPanel)
        self.frame_40.setFrameShadow(QFrame.Raised)
        self.verticalLayout_55 = QVBoxLayout(self.frame_40)
        self.verticalLayout_55.setSpacing(3)
        self.verticalLayout_55.setObjectName(u"verticalLayout_55")
        self.label_nodes_prev_tr = QLabel(self.frame_40)
        self.label_nodes_prev_tr.setObjectName(u"label_nodes_prev_tr")
        self.label_nodes_prev_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_55.addWidget(self.label_nodes_prev_tr)

        self.spin_nodes_prev_tr = QSpinBox(self.frame_40)
        self.spin_nodes_prev_tr.setObjectName(u"spin_nodes_prev_tr")

        self.verticalLayout_55.addWidget(self.spin_nodes_prev_tr)


        self.gridLayout_8.addWidget(self.frame_40, 2, 2, 1, 1)

        self.frame_44 = QFrame(self.frame_38)
        self.frame_44.setObjectName(u"frame_44")
        self.frame_44.setFrameShape(QFrame.StyledPanel)
        self.frame_44.setFrameShadow(QFrame.Raised)
        self.verticalLayout_60 = QVBoxLayout(self.frame_44)
        self.verticalLayout_60.setSpacing(3)
        self.verticalLayout_60.setObjectName(u"verticalLayout_60")
        self.label_max_nodes_tr = QLabel(self.frame_44)
        self.label_max_nodes_tr.setObjectName(u"label_max_nodes_tr")
        self.label_max_nodes_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_60.addWidget(self.label_max_nodes_tr)

        self.spin_max_nodes_tr = QSpinBox(self.frame_44)
        self.spin_max_nodes_tr.setObjectName(u"spin_max_nodes_tr")

        self.verticalLayout_60.addWidget(self.spin_max_nodes_tr)


        self.gridLayout_8.addWidget(self.frame_44, 2, 1, 1, 1)

        self.frame_48 = QFrame(self.frame_38)
        self.frame_48.setObjectName(u"frame_48")
        self.frame_48.setMinimumSize(QSize(0, 0))
        self.frame_48.setFrameShape(QFrame.StyledPanel)
        self.frame_48.setFrameShadow(QFrame.Raised)
        self.gridLayout_9 = QGridLayout(self.frame_48)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.gridLayout_9.setVerticalSpacing(3)
        self.btn_data_path_tr = QPushButton(self.frame_48)
        self.btn_data_path_tr.setObjectName(u"btn_data_path_tr")
        self.btn_data_path_tr.setMinimumSize(QSize(150, 25))
        self.btn_data_path_tr.setFont(font)
        self.btn_data_path_tr.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_data_path_tr.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon8 = QIcon()
        icon8.addFile(u":/icons/images/icons/cil-folder-open.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_data_path_tr.setIcon(icon8)
        self.btn_data_path_tr.setIconSize(QSize(14, 14))

        self.gridLayout_9.addWidget(self.btn_data_path_tr, 1, 1, 1, 1)

        self.line_data_path_tr = QLineEdit(self.frame_48)
        self.line_data_path_tr.setObjectName(u"line_data_path_tr")
        self.line_data_path_tr.setMinimumSize(QSize(0, 25))
        self.line_data_path_tr.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.gridLayout_9.addWidget(self.line_data_path_tr, 1, 0, 1, 1)

        self.label_data_path_tr = QLabel(self.frame_48)
        self.label_data_path_tr.setObjectName(u"label_data_path_tr")

        self.gridLayout_9.addWidget(self.label_data_path_tr, 0, 0, 1, 2)


        self.gridLayout_8.addWidget(self.frame_48, 3, 0, 1, 3)

        self.frame_47 = QFrame(self.frame_38)
        self.frame_47.setObjectName(u"frame_47")
        self.frame_47.setFrameShape(QFrame.StyledPanel)
        self.frame_47.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_15 = QHBoxLayout(self.frame_47)
        self.horizontalLayout_15.setSpacing(3)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.check_num_nodes_tr = QCheckBox(self.frame_47)
        self.check_num_nodes_tr.setObjectName(u"check_num_nodes_tr")
        self.check_num_nodes_tr.setStyleSheet(u"")

        self.horizontalLayout_15.addWidget(self.check_num_nodes_tr)


        self.gridLayout_8.addWidget(self.frame_47, 4, 0, 1, 3)

        self.frame_45 = QFrame(self.frame_38)
        self.frame_45.setObjectName(u"frame_45")
        self.frame_45.setFrameShape(QFrame.StyledPanel)
        self.frame_45.setFrameShadow(QFrame.Raised)
        self.verticalLayout_56 = QVBoxLayout(self.frame_45)
        self.verticalLayout_56.setSpacing(3)
        self.verticalLayout_56.setObjectName(u"verticalLayout_56")
        self.label_n_graphs_tr = QLabel(self.frame_45)
        self.label_n_graphs_tr.setObjectName(u"label_n_graphs_tr")
        self.label_n_graphs_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_56.addWidget(self.label_n_graphs_tr)

        self.spin_n_graphs_tr = QSpinBox(self.frame_45)
        self.spin_n_graphs_tr.setObjectName(u"spin_n_graphs_tr")

        self.verticalLayout_56.addWidget(self.spin_n_graphs_tr)


        self.gridLayout_8.addWidget(self.frame_45, 1, 1, 1, 1)

        self.frame_46 = QFrame(self.frame_38)
        self.frame_46.setObjectName(u"frame_46")
        self.frame_46.setFrameShape(QFrame.StyledPanel)
        self.frame_46.setFrameShadow(QFrame.Raised)
        self.verticalLayout_62 = QVBoxLayout(self.frame_46)
        self.verticalLayout_62.setSpacing(3)
        self.verticalLayout_62.setObjectName(u"verticalLayout_62")
        self.label_min_nodes_tr = QLabel(self.frame_46)
        self.label_min_nodes_tr.setObjectName(u"label_min_nodes_tr")
        self.label_min_nodes_tr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_62.addWidget(self.label_min_nodes_tr)

        self.spin_min_nodes_tr = QSpinBox(self.frame_46)
        self.spin_min_nodes_tr.setObjectName(u"spin_min_nodes_tr")

        self.verticalLayout_62.addWidget(self.spin_min_nodes_tr)


        self.gridLayout_8.addWidget(self.frame_46, 2, 0, 1, 1)


        self.verticalLayout_63.addWidget(self.frame_38)


        self.verticalLayout_28.addWidget(self.conf_tr_row_3)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.horizontalLayout_10.addWidget(self.scrollArea_2)

        self.stackedWidget_2.addWidget(self.conf_tr)
        self.conf_gr = QWidget()
        self.conf_gr.setObjectName(u"conf_gr")
        self.horizontalLayout_8 = QHBoxLayout(self.conf_gr)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_3 = QScrollArea(self.conf_gr)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setFrameShape(QFrame.NoFrame)
        self.scrollArea_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 306, 326))
        self.verticalLayout_39 = QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_39.setSpacing(11)
        self.verticalLayout_39.setObjectName(u"verticalLayout_39")
        self.verticalLayout_39.setContentsMargins(5, 10, 0, 9)
        self.conf_gr_row_1 = QFrame(self.scrollAreaWidgetContents_3)
        self.conf_gr_row_1.setObjectName(u"conf_gr_row_1")
        self.conf_gr_row_1.setFrameShape(QFrame.StyledPanel)
        self.conf_gr_row_1.setFrameShadow(QFrame.Raised)
        self.verticalLayout_19 = QVBoxLayout(self.conf_gr_row_1)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.label_row_1_gr = QLabel(self.conf_gr_row_1)
        self.label_row_1_gr.setObjectName(u"label_row_1_gr")
        self.label_row_1_gr.setMaximumSize(QSize(16777215, 25))

        self.verticalLayout_19.addWidget(self.label_row_1_gr)

        self.frame_24 = QFrame(self.conf_gr_row_1)
        self.frame_24.setObjectName(u"frame_24")
        self.frame_24.setFrameShape(QFrame.StyledPanel)
        self.frame_24.setFrameShadow(QFrame.Raised)
        self.gridLayout = QGridLayout(self.frame_24)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame_27 = QFrame(self.frame_24)
        self.frame_27.setObjectName(u"frame_27")
        self.frame_27.setFrameShape(QFrame.StyledPanel)
        self.frame_27.setFrameShadow(QFrame.Raised)
        self.verticalLayout_41 = QVBoxLayout(self.frame_27)
        self.verticalLayout_41.setSpacing(3)
        self.verticalLayout_41.setObjectName(u"verticalLayout_41")
        self.label_seed_gr = QLabel(self.frame_27)
        self.label_seed_gr.setObjectName(u"label_seed_gr")
        self.label_seed_gr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_41.addWidget(self.label_seed_gr)

        self.spin_seed_gr = QSpinBox(self.frame_27)
        self.spin_seed_gr.setObjectName(u"spin_seed_gr")

        self.verticalLayout_41.addWidget(self.spin_seed_gr)


        self.gridLayout.addWidget(self.frame_27, 0, 0, 1, 1)

        self.frame_30 = QFrame(self.frame_24)
        self.frame_30.setObjectName(u"frame_30")
        self.frame_30.setFrameShape(QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QFrame.Raised)
        self.verticalLayout_44 = QVBoxLayout(self.frame_30)
        self.verticalLayout_44.setSpacing(3)
        self.verticalLayout_44.setObjectName(u"verticalLayout_44")
        self.label_num_graphs_gr = QLabel(self.frame_30)
        self.label_num_graphs_gr.setObjectName(u"label_num_graphs_gr")
        self.label_num_graphs_gr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_44.addWidget(self.label_num_graphs_gr)

        self.spin_num_graphs_gr = QSpinBox(self.frame_30)
        self.spin_num_graphs_gr.setObjectName(u"spin_num_graphs_gr")

        self.verticalLayout_44.addWidget(self.spin_num_graphs_gr)


        self.gridLayout.addWidget(self.frame_30, 0, 1, 1, 1)

        self.frame_35 = QFrame(self.frame_24)
        self.frame_35.setObjectName(u"frame_35")
        self.frame_35.setFrameShape(QFrame.StyledPanel)
        self.frame_35.setFrameShadow(QFrame.Raised)
        self.verticalLayout_47 = QVBoxLayout(self.frame_35)
        self.verticalLayout_47.setSpacing(3)
        self.verticalLayout_47.setObjectName(u"verticalLayout_47")
        self.label_batch_size_gr = QLabel(self.frame_35)
        self.label_batch_size_gr.setObjectName(u"label_batch_size_gr")
        self.label_batch_size_gr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_47.addWidget(self.label_batch_size_gr)

        self.spin_batch_size_gr = QSpinBox(self.frame_35)
        self.spin_batch_size_gr.setObjectName(u"spin_batch_size_gr")

        self.verticalLayout_47.addWidget(self.spin_batch_size_gr)


        self.gridLayout.addWidget(self.frame_35, 0, 2, 1, 1)

        self.frame_28 = QFrame(self.frame_24)
        self.frame_28.setObjectName(u"frame_28")
        self.frame_28.setFrameShape(QFrame.StyledPanel)
        self.frame_28.setFrameShadow(QFrame.Raised)
        self.verticalLayout_45 = QVBoxLayout(self.frame_28)
        self.verticalLayout_45.setSpacing(3)
        self.verticalLayout_45.setObjectName(u"verticalLayout_45")
        self.label_min_num_nodes_gr = QLabel(self.frame_28)
        self.label_min_num_nodes_gr.setObjectName(u"label_min_num_nodes_gr")
        self.label_min_num_nodes_gr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_45.addWidget(self.label_min_num_nodes_gr)

        self.spin_min_num_nodes_gr = QSpinBox(self.frame_28)
        self.spin_min_num_nodes_gr.setObjectName(u"spin_min_num_nodes_gr")

        self.verticalLayout_45.addWidget(self.spin_min_num_nodes_gr)


        self.gridLayout.addWidget(self.frame_28, 1, 0, 1, 1)

        self.frame_29 = QFrame(self.frame_24)
        self.frame_29.setObjectName(u"frame_29")
        self.frame_29.setFrameShape(QFrame.StyledPanel)
        self.frame_29.setFrameShadow(QFrame.Raised)
        self.verticalLayout_46 = QVBoxLayout(self.frame_29)
        self.verticalLayout_46.setSpacing(3)
        self.verticalLayout_46.setObjectName(u"verticalLayout_46")
        self.label_max_num_nodes_gr = QLabel(self.frame_29)
        self.label_max_num_nodes_gr.setObjectName(u"label_max_num_nodes_gr")
        self.label_max_num_nodes_gr.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_46.addWidget(self.label_max_num_nodes_gr)

        self.spin_max_num_nodes_gr = QSpinBox(self.frame_29)
        self.spin_max_num_nodes_gr.setObjectName(u"spin_max_num_nodes_gr")

        self.verticalLayout_46.addWidget(self.spin_max_num_nodes_gr)


        self.gridLayout.addWidget(self.frame_29, 1, 1, 1, 1)

        self.frame_4 = QFrame(self.frame_24)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)

        self.gridLayout.addWidget(self.frame_4, 1, 2, 1, 1)


        self.verticalLayout_19.addWidget(self.frame_24)

        self.frame_69 = QFrame(self.conf_gr_row_1)
        self.frame_69.setObjectName(u"frame_69")
        self.frame_69.setMinimumSize(QSize(0, 0))
        self.frame_69.setFrameShape(QFrame.StyledPanel)
        self.frame_69.setFrameShadow(QFrame.Raised)
        self.gridLayout_13 = QGridLayout(self.frame_69)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.gridLayout_13.setVerticalSpacing(3)
        self.line_save_loc_gr = QLineEdit(self.frame_69)
        self.line_save_loc_gr.setObjectName(u"line_save_loc_gr")
        self.line_save_loc_gr.setMinimumSize(QSize(0, 25))
        self.line_save_loc_gr.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.gridLayout_13.addWidget(self.line_save_loc_gr, 1, 0, 1, 1)

        self.label_save_loc_gr = QLabel(self.frame_69)
        self.label_save_loc_gr.setObjectName(u"label_save_loc_gr")

        self.gridLayout_13.addWidget(self.label_save_loc_gr, 0, 0, 1, 2)

        self.btn_save_loc_gr = QPushButton(self.frame_69)
        self.btn_save_loc_gr.setObjectName(u"btn_save_loc_gr")
        self.btn_save_loc_gr.setMinimumSize(QSize(150, 25))
        self.btn_save_loc_gr.setFont(font)
        self.btn_save_loc_gr.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_save_loc_gr.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.btn_save_loc_gr.setIcon(icon8)
        self.btn_save_loc_gr.setIconSize(QSize(14, 14))

        self.gridLayout_13.addWidget(self.btn_save_loc_gr, 1, 1, 1, 1)


        self.verticalLayout_19.addWidget(self.frame_69)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_19.addItem(self.verticalSpacer)


        self.verticalLayout_39.addWidget(self.conf_gr_row_1)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.horizontalLayout_8.addWidget(self.scrollArea_3)

        self.stackedWidget_2.addWidget(self.conf_gr)
        self.conf_ev = QWidget()
        self.conf_ev.setObjectName(u"conf_ev")
        self.horizontalLayout_16 = QHBoxLayout(self.conf_ev)
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_4 = QScrollArea(self.conf_ev)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        self.scrollArea_4.setFrameShape(QFrame.NoFrame)
        self.scrollArea_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 391, 523))
        self.verticalLayout_49 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_49.setSpacing(11)
        self.verticalLayout_49.setObjectName(u"verticalLayout_49")
        self.verticalLayout_49.setContentsMargins(5, 10, 0, 9)
        self.conf_ev_row = QFrame(self.scrollAreaWidgetContents_4)
        self.conf_ev_row.setObjectName(u"conf_ev_row")
        self.conf_ev_row.setFrameShape(QFrame.StyledPanel)
        self.conf_ev_row.setFrameShadow(QFrame.Raised)
        self.verticalLayout_64 = QVBoxLayout(self.conf_ev_row)
        self.verticalLayout_64.setObjectName(u"verticalLayout_64")
        self.label_row_1_ev = QLabel(self.conf_ev_row)
        self.label_row_1_ev.setObjectName(u"label_row_1_ev")
        self.label_row_1_ev.setMaximumSize(QSize(16777215, 25))

        self.verticalLayout_64.addWidget(self.label_row_1_ev)

        self.frame_53 = QFrame(self.conf_ev_row)
        self.frame_53.setObjectName(u"frame_53")
        self.frame_53.setFrameShape(QFrame.StyledPanel)
        self.frame_53.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame_53)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.frame_54 = QFrame(self.frame_53)
        self.frame_54.setObjectName(u"frame_54")
        self.frame_54.setFrameShape(QFrame.StyledPanel)
        self.frame_54.setFrameShadow(QFrame.Raised)
        self.verticalLayout_65 = QVBoxLayout(self.frame_54)
        self.verticalLayout_65.setSpacing(3)
        self.verticalLayout_65.setObjectName(u"verticalLayout_65")
        self.label_seed_ev = QLabel(self.frame_54)
        self.label_seed_ev.setObjectName(u"label_seed_ev")
        self.label_seed_ev.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_65.addWidget(self.label_seed_ev)

        self.spin_seed_ev = QSpinBox(self.frame_54)
        self.spin_seed_ev.setObjectName(u"spin_seed_ev")

        self.verticalLayout_65.addWidget(self.spin_seed_ev)


        self.gridLayout_2.addWidget(self.frame_54, 0, 0, 1, 1)

        self.frame_55 = QFrame(self.frame_53)
        self.frame_55.setObjectName(u"frame_55")
        self.frame_55.setFrameShape(QFrame.StyledPanel)
        self.frame_55.setFrameShadow(QFrame.Raised)
        self.verticalLayout_66 = QVBoxLayout(self.frame_55)
        self.verticalLayout_66.setSpacing(3)
        self.verticalLayout_66.setObjectName(u"verticalLayout_66")
        self.label_num_graphs_ev = QLabel(self.frame_55)
        self.label_num_graphs_ev.setObjectName(u"label_num_graphs_ev")
        self.label_num_graphs_ev.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_66.addWidget(self.label_num_graphs_ev)

        self.spin_num_graphs_ev = QSpinBox(self.frame_55)
        self.spin_num_graphs_ev.setObjectName(u"spin_num_graphs_ev")

        self.verticalLayout_66.addWidget(self.spin_num_graphs_ev)


        self.gridLayout_2.addWidget(self.frame_55, 0, 1, 1, 1)

        self.frame_58 = QFrame(self.frame_53)
        self.frame_58.setObjectName(u"frame_58")
        self.frame_58.setFrameShape(QFrame.StyledPanel)
        self.frame_58.setFrameShadow(QFrame.Raised)
        self.verticalLayout_69 = QVBoxLayout(self.frame_58)
        self.verticalLayout_69.setSpacing(3)
        self.verticalLayout_69.setObjectName(u"verticalLayout_69")
        self.label_batch_size_ev = QLabel(self.frame_58)
        self.label_batch_size_ev.setObjectName(u"label_batch_size_ev")
        self.label_batch_size_ev.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_69.addWidget(self.label_batch_size_ev)

        self.spin_batch_size_ev = QSpinBox(self.frame_58)
        self.spin_batch_size_ev.setObjectName(u"spin_batch_size_ev")

        self.verticalLayout_69.addWidget(self.spin_batch_size_ev)


        self.gridLayout_2.addWidget(self.frame_58, 0, 2, 1, 1)

        self.frame_56 = QFrame(self.frame_53)
        self.frame_56.setObjectName(u"frame_56")
        self.frame_56.setFrameShape(QFrame.StyledPanel)
        self.frame_56.setFrameShadow(QFrame.Raised)
        self.verticalLayout_67 = QVBoxLayout(self.frame_56)
        self.verticalLayout_67.setSpacing(3)
        self.verticalLayout_67.setObjectName(u"verticalLayout_67")
        self.label_bs_samples_ev = QLabel(self.frame_56)
        self.label_bs_samples_ev.setObjectName(u"label_bs_samples_ev")
        self.label_bs_samples_ev.setMaximumSize(QSize(16777215, 32))

        self.verticalLayout_67.addWidget(self.label_bs_samples_ev)

        self.spin_bs_samples_ev = QSpinBox(self.frame_56)
        self.spin_bs_samples_ev.setObjectName(u"spin_bs_samples_ev")

        self.verticalLayout_67.addWidget(self.spin_bs_samples_ev)


        self.gridLayout_2.addWidget(self.frame_56, 0, 3, 1, 1)


        self.verticalLayout_64.addWidget(self.frame_53)

        self.frame_57 = QFrame(self.conf_ev_row)
        self.frame_57.setObjectName(u"frame_57")
        self.frame_57.setFrameShape(QFrame.StyledPanel)
        self.frame_57.setFrameShadow(QFrame.Raised)
        self.gridLayout_6 = QGridLayout(self.frame_57)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.check_d_ev_ev = QCheckBox(self.frame_57)
        self.check_d_ev_ev.setObjectName(u"check_d_ev_ev")
        self.check_d_ev_ev.setStyleSheet(u"")

        self.gridLayout_6.addWidget(self.check_d_ev_ev, 1, 0, 1, 1)

        self.check_b_ev_ev = QCheckBox(self.frame_57)
        self.check_b_ev_ev.setObjectName(u"check_b_ev_ev")
        self.check_b_ev_ev.setStyleSheet(u"")

        self.gridLayout_6.addWidget(self.check_b_ev_ev, 1, 2, 1, 2)

        self.check_a_ev_ev = QCheckBox(self.frame_57)
        self.check_a_ev_ev.setObjectName(u"check_a_ev_ev")
        self.check_a_ev_ev.setStyleSheet(u"")

        self.gridLayout_6.addWidget(self.check_a_ev_ev, 2, 0, 1, 1)

        self.label_metrics_ev_ev = QLabel(self.frame_57)
        self.label_metrics_ev_ev.setObjectName(u"label_metrics_ev_ev")
        self.label_metrics_ev_ev.setMaximumSize(QSize(16777215, 20))

        self.gridLayout_6.addWidget(self.label_metrics_ev_ev, 0, 0, 1, 5)

        self.check_c_ev_ev = QCheckBox(self.frame_57)
        self.check_c_ev_ev.setObjectName(u"check_c_ev_ev")
        self.check_c_ev_ev.setStyleSheet(u"")

        self.gridLayout_6.addWidget(self.check_c_ev_ev, 1, 1, 1, 1)

        self.check_pr_ev_ev = QCheckBox(self.frame_57)
        self.check_pr_ev_ev.setObjectName(u"check_pr_ev_ev")
        self.check_pr_ev_ev.setStyleSheet(u"")

        self.gridLayout_6.addWidget(self.check_pr_ev_ev, 2, 1, 1, 1)


        self.verticalLayout_64.addWidget(self.frame_57)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_64.addItem(self.verticalSpacer_2)


        self.verticalLayout_49.addWidget(self.conf_ev_row)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)

        self.horizontalLayout_16.addWidget(self.scrollArea_4)

        self.stackedWidget_2.addWidget(self.conf_ev)

        self.verticalLayout_20.addWidget(self.stackedWidget_2)


        self.verticalLayout_13.addWidget(self.frame_3)


        self.verticalLayout_7.addWidget(self.contentSettings)


        self.horizontalLayout_4.addWidget(self.extraRightBox)


        self.verticalLayout_6.addWidget(self.content)


        self.verticalLayout_2.addWidget(self.contentBottom)


        self.appLayout.addWidget(self.contentBox)


        self.appMargins.addWidget(self.bgApp)

        MainWindow.setCentralWidget(self.styleSheet)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(2)
        self.stackedWidget_2.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.titleLeftApp.setText(QCoreApplication.translate("MainWindow", u"DGGI", None))
        self.titleLeftDescription.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Deep-generative graphs<br/>for the Internet</p></body></html>", None))
        self.toggleButton.setText(QCoreApplication.translate("MainWindow", u"Hide", None))
        self.btn_train.setText(QCoreApplication.translate("MainWindow", u"Train", None))
        self.btn_generate.setText(QCoreApplication.translate("MainWindow", u"Generate", None))
        self.btn_evaluate.setText(QCoreApplication.translate("MainWindow", u"Evaluate", None))
        self.toggleLeftBox.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.textEdit.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600; color:#ff79c6;\">Deep-Generative Graphs for the Internet (DGGI)</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">A graph generator based on deep learning originally developed to syntesize graphs that reproduce the propotities of "
                        "intra-AS networks, for more details <a href=\"https://doi.org/10.48550/arXiv.2308.05254\"><span style=\" text-decoration: underline; color:#bd93f9;\">see</span></a>.</p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:700; color:#ff79c6;\">Interface</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Interface created by <span style=\" font-style:italic;\">Wanderson M. Pimenta</span> and modified by <span style=\" font-style:italic;\">Caio Vinicius Dadauto</span>.</p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:700; color:#ff79c6;\">Cite</span></p></body></html>", None))
        self.plainTextEdit_2.setPlainText(QCoreApplication.translate("MainWindow", u"@ARTICLE{10589568,\n"
"author={Vinicius Dadauto, Caio and Fonseca, Nelson L. S. da and Torres, Ricardo da S.},\n"
"journal={IEEE Transactions on Network and Service Management}, \n"
"title={Data-Driven Intra-Autonomous Systems Graph Generator}, \n"
"year={2024},\n"
"volume={21},\n"
"number={5},\n"
"pages={5491-5504},\n"
"doi={10.1109/TNSM.2024.3425508}\n"
"}", None))
        self.titleRightInfo.setText("")
#if QT_CONFIG(tooltip)
        self.settingsTopBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Settings", None))
#endif // QT_CONFIG(tooltip)
        self.settingsTopBtn.setText("")
#if QT_CONFIG(tooltip)
        self.minimizeAppBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Minimize", None))
#endif // QT_CONFIG(tooltip)
        self.minimizeAppBtn.setText("")
#if QT_CONFIG(tooltip)
        self.maximizeRestoreAppBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Maximize", None))
#endif // QT_CONFIG(tooltip)
        self.maximizeRestoreAppBtn.setText("")
#if QT_CONFIG(tooltip)
        self.closeAppBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Close", None))
#endif // QT_CONFIG(tooltip)
        self.closeAppBtn.setText("")
        self.title_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-size:40pt; font-weight:700;\">Model</span><span style=\" font-size:40pt;\"> Training</span></p></body></html>", None))
        self.label_progress_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><br/></p></body></html>", None))
        self.label_suffix_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"right\"><br/></p></body></html>", None))
        self.btn_run_tr.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.link_mlf.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>For more taining datails access <a href=\"http://127.0.0.1:5000\"><span style=\" text-decoration: underline; color:#bd93f9;\">MlFlow UI</span></a></p></body></html>", None))
        self.title_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-size:40pt; font-weight:700;\">Graphs</span><span style=\" font-size:40pt;\"> Generation</span></p></body></html>", None))
        self.label_progress_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><br/></p></body></html>", None))
        self.label_suffix_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"right\"><br/></p></body></html>", None))
        self.label_tree_gr.setText(QCoreApplication.translate("MainWindow", u"Select the model to be used for graph generation:", None))
        self.btn_run_gr.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.btn_vis_gr.setText(QCoreApplication.translate("MainWindow", u"  Visualization", None))
        self.title_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-size:40pt; font-weight:700;\">Generator </span><span style=\" font-size:40pt;\">Evaluation</span></p></body></html>", None))
        self.label_progress_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><br/></p></body></html>", None))
        self.label_suffix_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"right\"><br/></p></body></html>", None))
        self.label_tree_ev.setText(QCoreApplication.translate("MainWindow", u"Select the model to be used for graph generation:", None))
        self.btn_run_ev.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.btn_vis_ev.setText(QCoreApplication.translate("MainWindow", u"  Visualization", None))
        self.btn_save_conf.setText(QCoreApplication.translate("MainWindow", u" Save    ", None))
        self.btn_load_conf.setText(QCoreApplication.translate("MainWindow", u" Load    ", None))
        self.btn_restore_conf.setText(QCoreApplication.translate("MainWindow", u" Restore", None))
        self.btn_apply_conf.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.conf_selection.setItemText(0, QCoreApplication.translate("MainWindow", u"Training Settings", None))
        self.conf_selection.setItemText(1, QCoreApplication.translate("MainWindow", u"Generation Settings", None))
        self.conf_selection.setItemText(2, QCoreApplication.translate("MainWindow", u"Evaluation Settings", None))

        self.label_row_1_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:700; color:#bd93f9;\">Procedure </span><span style=\" font-size:14pt;\">Parameters</span></p></body></html>", None))
        self.label_n_ckt_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Num of<br/>Checkpoints</p></body></html>", None))
        self.label_seed_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Seed</p></body></html>", None))
        self.label_lr_mile_tr.setText(QCoreApplication.translate("MainWindow", u"LR Decayment Milestones", None))
        self.line_lr_mile_tr.setText("")
        self.line_lr_mile_tr.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.label_n_graphs_ev_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Num of Graphs<br/>to Eval</p></body></html>", None))
        self.label_num_ep_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Num Epochs</p></body></html>", None))
        self.label_lr_decay_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">LR<br/>Decayment</p></body></html>", None))
        self.label_batch_ev_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Batch Size<br/>to Eval</p></body></html>", None))
        self.label_ep_to_ev_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Num Epochs<br/>to Eval</p></body></html>", None))
        self.label_init_lr_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Start LR</p></body></html>", None))
        self.label_bs_samples_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Bootstrap<br/>Sample Size</p></body></html>", None))
        self.check_d_ev_tr.setText(QCoreApplication.translate("MainWindow", u"Degree", None))
        self.check_b_ev_tr.setText(QCoreApplication.translate("MainWindow", u"Betweenness", None))
        self.check_a_ev_tr.setText(QCoreApplication.translate("MainWindow", u"Assortativity", None))
        self.label_metrics_ev_tr.setText(QCoreApplication.translate("MainWindow", u"Metrics to Eval", None))
        self.check_c_ev_tr.setText(QCoreApplication.translate("MainWindow", u"Clustering", None))
        self.check_pr_ev_tr.setText(QCoreApplication.translate("MainWindow", u"Pagerank", None))
        self.label_row_2_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:700; color:#bd93f9;\">Model </span><span style=\" font-size:14pt;\">Parameters</span></p></body></html>", None))
        self.label_in_emb_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Input GRU<br/>Embedding Size</p></body></html>", None))
        self.label_in_hd_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Input GRU<br/>Hidden Size</p></body></html>", None))
        self.label_out_mlp_emb_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Output MLP<br/>Embedding Size</p></body></html>", None))
        self.label_layers_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Num of<br/>Layers</p></body></html>", None))
        self.label_out_hd_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Output GRU<br/>Hidden SIze</p></body></html>", None))
        self.label_out_emb_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Output GRU<br/>Embedding Size</p></body></html>", None))
        self.label_row_3_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:700; color:#bd93f9;\">Data </span><span style=\" font-size:14pt;\">Parameters</span></p></body></html>", None))
        self.label_batch_loader_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Data Loader<br/>Batch Size</p></body></html>", None))
        self.label_n_workers_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Num of<br/>Workers</p></body></html>", None))
        self.label_nodes_prev_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Max Num of Nodes<br/>Preview</p></body></html>", None))
        self.label_max_nodes_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Max Num<br/>of Nodes</p></body></html>", None))
        self.btn_data_path_tr.setText(QCoreApplication.translate("MainWindow", u" Search", None))
        self.line_data_path_tr.setText("")
        self.line_data_path_tr.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.label_data_path_tr.setText(QCoreApplication.translate("MainWindow", u"Data Path", None))
        self.check_num_nodes_tr.setText(QCoreApplication.translate("MainWindow", u"Check num of nodes before load the graph", None))
        self.label_n_graphs_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Num of<br/>Graphs</p></body></html>", None))
        self.label_min_nodes_tr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Min Num<br/>of Nodes</p></body></html>", None))
        self.label_row_1_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:700; color:#bd93f9;\">Procedure </span><span style=\" font-size:14pt;\">Parameters</span></p></body></html>", None))
        self.label_seed_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Seed</p></body></html>", None))
        self.label_num_graphs_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Number<br/>Graphs</p></body></html>", None))
        self.label_batch_size_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Batch Size</p></body></html>", None))
        self.label_min_num_nodes_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Min Num<br/>Nodes</p></body></html>", None))
        self.label_max_num_nodes_gr.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Max Num<br/>Nodes</p></body></html>", None))
        self.line_save_loc_gr.setText("")
        self.line_save_loc_gr.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.label_save_loc_gr.setText(QCoreApplication.translate("MainWindow", u"Path to Save Graphs", None))
        self.btn_save_loc_gr.setText(QCoreApplication.translate("MainWindow", u" Search", None))
        self.label_row_1_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:700; color:#bd93f9;\">Procedure </span><span style=\" font-size:14pt;\">Parameters</span></p></body></html>", None))
        self.label_seed_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Seed</p></body></html>", None))
        self.label_num_graphs_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Number<br/>Graphs</p></body></html>", None))
        self.label_batch_size_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Batch Size</p></body></html>", None))
        self.label_bs_samples_ev.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\">Bootstrap<br/>Sample Size</p></body></html>", None))
        self.check_d_ev_ev.setText(QCoreApplication.translate("MainWindow", u"Degree", None))
        self.check_b_ev_ev.setText(QCoreApplication.translate("MainWindow", u"Betweenness", None))
        self.check_a_ev_ev.setText(QCoreApplication.translate("MainWindow", u"Assortativity", None))
        self.label_metrics_ev_ev.setText(QCoreApplication.translate("MainWindow", u"Metrics to Eval", None))
        self.check_c_ev_ev.setText(QCoreApplication.translate("MainWindow", u"Clustering", None))
        self.check_pr_ev_ev.setText(QCoreApplication.translate("MainWindow", u"Pagerank", None))
    # retranslateUi

