import sys
import os
import pathlib
import pymysql
import pandas as pd
from source import parameter
from source import detail
from source import compare
from PyQt5.QtWidgets import QFileDialog
from source import guess

x = list()
o = pathlib.Path.cwd() / "results"
with open("D:\GSOM-Application\results\wt2dphy_fillspix_r0p1.0spbm0.0a1.0Reds_nb10_aun_0.0_1.0\a1_k2_vgg16caffe_estopc5_fc_o1adam_lr-1.0de0.0e100_20201114_153419c255.0di-1ch3dfc0.0file_sum_ok.txt") as f:
    for line in f:
        # print(line)
        # print("hello")
        # print()
        x.append(line)

arr = x[22].split()
arr1 = x[23].split()
for i in range(0, 5):
    print(arr[i])
for j in range(0, 5):
    print(arr1[j])
