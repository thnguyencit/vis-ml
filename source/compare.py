from PyQt5 import QtWidgets, uic
import pathlib
import sys
import os
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
app = QtWidgets.QApplication([])
# dig = uic.loadUi("./interface/mainwindow.ui")
dl = uic.loadUi("./interface/compare.ui")


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=10):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = fig.add_axes([0,0,1,1])
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va) 

def FileSumOk1():
    options = QFileDialog.Options()

    options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(dl, "QFileDialog.getOpenFileNames()", "",
                                            "All Files (*);;Python Files (*.txt)", options=options)
    if files:
        dl.lineChooseFile1.setText(files[0])
        print(files[0])
        x = list()
        files = dl.lineChooseFile1.text()
        i = 0
        j = 0
        with open(files) as f:
            for line in f:
                # print(line)
                # print("hello")
                # print()
                arr = line.split()
                #           0       1       2       3       4       5       6   7       8   9       10      11  12          13      14  15  16      17      18
                if(arr == ['time', 'tr_ac', 'va_ac', 'sd', 'va_au', 'sd', 'tn', 'fn', 'fp', 'tp', 'preci', 'sd', 'recall', 'sd', 'f1', 'sd', 'mcc', 'sd', 'epst']):
                    j = i
                x.append(line)
                i = i+1
        # x = np.asarray(x)
        arr = x[j].split()
        arr1 = x[j+1].split()
        
        
        # create horizontal list i.e x-axis
        x = [0.1, 0.2, 0.3, 0.4]


        # show values from sum file to compare using index from arr
        # nthai: load data from file sum, 
        # data
        print('arr1')
        print(arr1)
        y1 = [float(arr1[1]), float(arr1[2]), float(arr1[4]), float(arr1[16])]
        print(y1)
        # label of arr        
        label = [arr[1], arr[2], arr[4], arr[16]]
        # nthai
        print("processing in compare.py")
        print(label)
        
        # create pyqt5graph bar graph item
        # with width = 0.6
        # with bar colors = green
        # bargraph = pg.BarGraphItem(x = x, height = y1, width = 0.05, brush ='g')
        sc = MplCanvas(None, width=1, height=1, dpi=50)
        sc.axes.bar(label, y1, width=0.6)
        add_value_labels(sc.axes)
        content_widget = QtWidgets.QWidget()
        dl.showchart1.setWidget(content_widget)
        layout = QtWidgets.QGridLayout(content_widget)
        layout.addWidget(sc)
        content_widget.setLayout(layout)
        dl.tbguess1.setItem(0, 0, QTableWidgetItem(arr[1]))
        dl.tbguess1.setItem(0, 1, QTableWidgetItem(arr[2]))
        dl.tbguess1.setItem(0, 2, QTableWidgetItem(arr[4]))
        dl.tbguess1.setItem(0, 3, QTableWidgetItem(arr[16]))
        # dl.tbguess1.setItem(0, 4, QTableWidgetItem(arr[4]))
        dl.tbguess1.setItem(1, 0, QTableWidgetItem(arr1[1]))
        dl.tbguess1.setItem(1, 1, QTableWidgetItem(arr1[2]))
        dl.tbguess1.setItem(1, 2, QTableWidgetItem(arr1[4]))
        dl.tbguess1.setItem(1, 3, QTableWidgetItem(arr1[16]))
        # dl.tbguess1.setItem(1, 4, QTableWidgetItem(arr1[4]))


def FileSumOk2():
    options = QFileDialog.Options()

    options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(dl, "QFileDialog.getOpenFileNames()", "",
                                            "All Files (*);;Python Files (*.txt)", options=options)
    if files:
        dl.lineChooseFile2.setText(files[0])
        print(files[0])
        x = list()
        files = dl.lineChooseFile2.text()
        i = 0
        j = 0
        with open(files) as f:
            for line in f:
                # print(line)
                # print("hello")
                # print()
                arr = line.split()
                if(arr == ['time', 'tr_ac', 'va_ac', 'sd', 'va_au', 'sd', 'tn', 'fn', 'fp', 'tp', 'preci', 'sd', 'recall', 'sd', 'f1', 'sd', 'mcc', 'sd', 'epst']):
                    j = i
                x.append(line)
                i = i+1
        # x = np.asarray(x)
        arr = x[j].split()
        arr1 = x[j+1].split()
        print(arr[1])


        # y1 = [float(arr1[0]), float(arr1[1]), float(arr1[2]), float(arr1[4])]
        # print(y1)      
        # label = [arr[0], arr[1], arr[2], arr[4]]
        # print(label)

        y1 = [float(arr1[1]), float(arr1[2]), float(arr1[4]), float(arr1[16])]
        print(y1)
        # label of arr        
        label = [arr[1], arr[2], arr[4], arr[16]]
        # nthai
        print("processing in compare.py")
        print(label)

        # create pyqt5graph bar graph item
        # with width = 0.6
        # with bar colors = green
        # bargraph = pg.BarGraphItem(x = x, height = y1, width = 0.05, brush ='g')
        sc = MplCanvas(None, width=1, height=1, dpi=50)
        sc.axes.bar(label, y1, width=0.6)
        add_value_labels(sc.axes)
        content_widget = QtWidgets.QWidget()
        dl.showchart2.setWidget(content_widget)
        layout = QtWidgets.QGridLayout(content_widget)
        layout.addWidget(sc)
        content_widget.setLayout(layout)
        

        dl.tbguess2.setItem(0, 0, QTableWidgetItem(arr[1]))
        dl.tbguess2.setItem(0, 1, QTableWidgetItem(arr[2]))
        dl.tbguess2.setItem(0, 2, QTableWidgetItem(arr[4]))
        dl.tbguess2.setItem(0, 3, QTableWidgetItem(arr[16]))
        # dl.tbguess1.setItem(0, 4, QTableWidgetItem(arr[4]))
        dl.tbguess2.setItem(1, 0, QTableWidgetItem(arr1[1]))
        dl.tbguess2.setItem(1, 1, QTableWidgetItem(arr1[2]))
        dl.tbguess2.setItem(1, 2, QTableWidgetItem(arr1[4]))
        dl.tbguess2.setItem(1, 3, QTableWidgetItem(arr1[16]))
        # dl.tbguess1.setItem(1, 4, QTableWidgetItem(arr1[4]))


        # dl.tbguess2.setItem(0, 0, QTableWidgetItem(arr[0]))
        # dl.tbguess2.setItem(0, 1, QTableWidgetItem(arr[1]))
        # dl.tbguess2.setItem(0, 2, QTableWidgetItem(arr[2]))
        # dl.tbguess2.setItem(0, 3, QTableWidgetItem(arr[4]))
        # # dl.tbguess1.setItem(0, 4, QTableWidgetItem(arr[4]))
        # dl.tbguess2.setItem(1, 0, QTableWidgetItem(arr1[0]))
        # dl.tbguess2.setItem(1, 1, QTableWidgetItem(arr1[1]))
        # dl.tbguess2.setItem(1, 2, QTableWidgetItem(arr1[2]))
        # dl.tbguess2.setItem(1, 3, QTableWidgetItem(arr1[4]))
        # # dl.tbguess1.setItem(1, 4, QTableWidgetItem(arr1[4]))


# nthai: draw couple-chart to compare
def drawchart12():
    data1 = dl.lineChooseFile1.text()
    data2 = dl.lineChooseFile2.text()
    i1 = 0
    j1 = 0
    i2 = 0
    j2 = 0
    x1 = list()
    x2 = list()
    with open(data1) as f:
            for line in f:
                # print(line)
                # print("hello")
                # print()
                arr = line.split()
                if(arr == ['time', 'tr_ac', 'va_ac', 'sd', 'va_au', 'sd', 'tn', 'fn', 'fp', 'tp', 'preci', 'sd', 'recall', 'sd', 'f1', 'sd', 'mcc', 'sd', 'epst']):
                    j1 = i1
                x1.append(line)
                i1 = i1+1
    arr = x1[j1].split()
    arr1 = x1[j1+1].split()
    
    
    #y1 = [float(arr1[0]), float(arr1[1]), float(arr1[2]), float(arr1[4])]
    #print(y1)
    y1 = [float(arr1[1]), float(arr1[2]), float(arr1[4]), float(arr1[16])]
    print(y1)
    # create horizontal list i.e x-axis
    #label = [arr[0], arr[1], arr[2], arr[4]]
    label = [arr[1], arr[2], arr[4], arr[16]]
    #label = [arr[0], arr[1], arr[2], arr[4]]


    with open(data2) as f:
            for line in f:
                # print(line)
                # print("hello")
                # print()
                arr = line.split()
                if(arr == ['time', 'tr_ac', 'va_ac', 'sd', 'va_au', 'sd', 'tn', 'fn', 'fp', 'tp', 'preci', 'sd', 'recall', 'sd', 'f1', 'sd', 'mcc', 'sd', 'epst']):
                    j2 = i2
                x2.append(line)
                i2 = i2+1
    arr2 = x2[j2+1].split()
    
    y2 = [float(arr2[1]), float(arr2[2]), float(arr2[4]), float(arr2[16])]
    #y2 = [float(arr2[0]), float(arr2[1]), float(arr2[2]), float(arr2[4])]
    print(y2)

    sc = MplCanvas(None, width=1, height=1, dpi=50)
    width=0.35
    ind = np.arange(len(y1))
    sc.axes.bar(ind - width/2, y1, width=0.35,label="Method 1")
    sc.axes.bar(ind + width/2, y2, width=0.35,label="Method 2")
    sc.axes.set_xticks(ind)
    sc.axes.set_xticklabels(label)
    sc.axes.legend()
    add_value_labels(sc.axes)
    content_widget = QtWidgets.QWidget()
    dl.showchart1_2.setWidget(content_widget)
    layout = QtWidgets.QGridLayout(content_widget)
    layout.addWidget(sc)
    content_widget.setLayout(layout)
def Show():
    dl.pBtChooseFile1.clicked.connect(FileSumOk1)
    dl.pBtChooseFile2.clicked.connect(FileSumOk2)
    dl.showchart.clicked.connect(drawchart12)
    dl.exec_()
