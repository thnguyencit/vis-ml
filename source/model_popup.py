from PyQt5 import QtWidgets, uic
import pathlib
import sys
import os
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox
app = QtWidgets.QApplication([])
dig = uic.loadUi("./interface/mainwindow.ui")
# dl = uic.loadUi("./interface/model.ui")


def MessageBox(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(s)
    msg.exec_()


def modelcheckrun():  # numlayer,numfilter,save_d,model_cnn,comboxSearch = checkEmpty()
    data = dig.lineFile1.text()
    print(dig)
    print(data)
    b = data.rstrip('_x.csv')
    o = pathlib.Path.cwd().parent / 'data'

    x = str(o)

    d = b[(len(x)+18):]
    k_fold = dig.lineK.text()
    comboText1 = dig.cbEMB.currentText()
    print(comboText1)
    comboText2 = dig.cbBin.currentText()
    comboText3 = dig.cbPreIMG.currentText()
    comboText4 = dig.cbColormap.currentText()
    linkfileh5 = dl.lineLinkfileh5.text()
    command = f"python -m deepmg -i {d} -t {comboText1} -y {comboText2} -z 255 " \
              f"--preprocess_img {comboText3}  --run_time 2 --colormap {comboText4} -k {k_fold} -e 100 --search_already=y   -a predict --pretrained_w_path {linkfileh5}  --model pretrained -n 2 -f 7  --channel 3 --save_w y"
    print(command)


def files():
    options = QFileDialog.Options()

    options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(dig, "QFileDialog.getOpenFileNames()", "",
                                            "All Files (*);;Python Files (*.h5)", options=options)
    if files:
        dl.lineLinkfileh5.setText(files[0])
        print(files[0])


def ShowModel():
    dl.pBtRunModel.clicked.connect(modelcheckrun)
    dl.pBtnLinkFileh5.clicked.connect(files)
    dl.exec_()
