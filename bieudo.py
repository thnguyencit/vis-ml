# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bieudo1.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!
import os
import pathlib

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_bieudo(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
    def setupUi(self, bieudo):
        bieudo.setObjectName("bieudo")
        bieudo.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(bieudo)
        self.centralwidget.resize(1200,800)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.texemanh = QtWidgets.QTextEdit(self.centralwidget)
        self.texemanh.setObjectName("texemanh")
        self.gridLayout.addWidget(self.texemanh, 0, 0, 1000, 1000)
        # bieudo.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(bieudo)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1140, 22))
        # self.menubar.setObjectName("menubar")
        # bieudo.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(bieudo)
        # self.statusbar.setObjectName("statusbar")
        # bieudo.setStatusBar(self.statusbar)

        self.retranslateUi(bieudo)
        QtCore.QMetaObject.connectSlotsByName(bieudo)

    def retranslateUi(self, bieudo):
        _translate = QtCore.QCoreApplication.translate
        bieudo.setWindowTitle(_translate("bieudo", "Biểu đồ"))

        k = pathlib.Path(__file__).parent
        print('path chart ' + str(k))
        #path = str(k)+r'\anhbieudo'
        # nthai
        path = os.path.join(k,'anhbieudo')
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    files.append(os.path.join(r, file))
                    print(files)
        html=""
        files.sort()
        for f in files:
            html+=f'<img src="{f}">'
        self.texemanh.setHtml(html)