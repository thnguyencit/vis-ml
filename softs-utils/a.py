# from PyQt5 import QtWidgets, uic
# import pathlib
# import sys
# import os
# import pandas as pd
# import numpy as np
# from PyQt5.QtWidgets import QMessageBox


# def MessageBox(s):
#     msg = QMessageBox()
#     msg.setIcon(QMessageBox.Warning)
#     msg.setText(s)
#     msg.exec_()


def checkIntegerandFloat(number):
    try:
        val = int(number)
        print("Input is an integer number. Number = ", val)
    except ValueError:
        try:
            val = float(number)
            print("Input is a float  number. Number = ", val)
        except ValueError:
            print("No.. input is not a number. It's a string")
            # MessageBox(number+" is not number")


def Apply():
    # a = dlP.lineEstimators.text
    # print(str(a))
    a = "1"
    checkIntegerandFloat(str(a))


a = "python3 -m deepmg -i wt2dphy -t zfills -y spb -z 255 --preprocess_img vgg16 --run_time=2  -k 3 -e 100 --colormap rainbow --search_already=n  --channel 3 --save_w y --padding y  --model model_vgglike --del0 y  -o sgd -l mae --svm_kernel linear --method_lle lts --mode_pre_img caffe -z 5 --rf_max_depth 10 --iter_visual 350 --momentum 5 --knn_n_neighbors 10 --cmap_vmin 2  --cmap_vmax 4 --n_quantile 1000 --num_bin 20"
Apply()
