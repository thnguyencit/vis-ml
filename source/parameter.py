from PyQt5 import QtWidgets, uic
import pathlib
import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox
# from  import mainWindows
app = QtWidgets.QApplication([])
dlP = uic.loadUi("./interface/Parameter.ui")
dig = uic.loadUi("./interface/mainwindow.ui")
Estimators, Visual, Momentum, KnnN, Cmapvmin, Cmapvmax, Quantile, NumBin, Lossfunc, Margin = - \
    1, 300, 0, 5, 0, 1, 1000, 10, "binary_crossentropy", "n"
Padding, model_cnn, Delo, Optimizer, SVM, MethodlPLE, ModePreImg, Coffe = "n", "", "n", "adam", "binary_crossentropy", "linear", "standard", "caffe"


def MessageBox(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(s)
    msg.exec_()


def checkIntegerandFloat(number):
    print(number)
    try:
        val = int(number)
        # print("Input is an integer number. Number = ", val)
    except ValueError:
        # print("No.. input is not a number. It's a string")
        err = number + " is not number"
        MessageBox(err)


def Apply():
    if (dlP.lineRuntime.text() != ""):
        checkIntegerandFloat(dlP.lineRuntime.text())
    # if (dlP.lr_visual.text() != ""):
    #     checkIntegerandFloat(dlP.lr_visual.text())
    if (dlP.lineIterVisual.text() != ""):
        checkIntegerandFloat(dlP.lineIterVisual.text())
    if (dlP.fig_size.text() != ""):
        checkIntegerandFloat(dlP.fig_size.text())
    if (dlP.lineKnnN.text() != ""):
        checkIntegerandFloat(dlP.lineKnnN.text())
    if (dlP.lineCmapvmin.text() != ""):
        checkIntegerandFloat(dlP.lineCmapvmin.text())
    if (dlP.lineCmapvmax.text() != ""):
        checkIntegerandFloat(dlP.lineCmapvmax.text())
    if (dlP.lineQuantile.text() != ""):
        checkIntegerandFloat(dlP.lineQuantile.text())
    if (dlP.lineNumBin.text() != ""):
        checkIntegerandFloat(dlP.lineNumBin.text())
    if (dlP.lineLossfunc.text() != ""):
        checkIntegerandFloat(dlP.lineLossfunc.text())
    if (dlP.newdim.text() != ""):
        checkIntegerandFloat(dlP.newdim.text())
    # print(dig.mainWindows.lineK.text())
    # command = f"python D:/GSOM-Application/deepmg_v34/__main__.py -i  -t  -y  -z 255 " \
    #     f"--preprocess_img  --colormap  -k  -e 100 --search_already n --channel 3 "\
    #     f"--save_w y "
    # runtime = dlP.lr_visual.text()
    # # Cmd = ""
    # if (runtime != ""):
    #     command = command + "--runtime " + runtime
    # print(command)
    # print(Cmd)
    dlP.close()


def reloadParameter():
    dlP.lr_visual.setText("")
    dlP.lineIterVisual.setText("")
    dlP.fig_size.setText("")
    dlP.lineKnnN.setText("")
    dlP.lineCmapvmin.setText("")
    dlP.lineCmapvmax.setText("")
    dlP.lineQuantile.setText("")
    dlP.lineNumBin.setText("")
    dlP.lineLossfunc.setText("")
    dlP.newdim.setText("")
    dlP.lineRuntime.setText("")
    dlP.cbPadding.setCurrentText("")
    dlP.cbModel.setCurrentText("")
    dlP.cbDelo.setCurrentText("")
    dlP.cbOptimizer.setCurrentText("")
    dlP.cbLossfunc.setCurrentText("")
    dlP.cbSVM.setCurrentText("")
    dlP.cbMethodLLE.setCurrentText("")
    dlP.cbModeReduceDim.setCurrentText("")
    dlP.cbCoffe.setText("")

    # print("hello world")


def Show():
    dlP.pBtnApply.clicked.connect(Apply)
    # dlP.pBtnApply.clicked.connect(reloadParameter)
    dlP.exec_()
