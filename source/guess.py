from PyQt5 import QtWidgets, uic
def Show():
    dig = uic.loadUi("./interface/guess.ui")
    dig.exec_()