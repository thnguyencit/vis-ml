from PyQt5 import QtWidgets, uic
def Show(thu, mau, sl, std, ma, mi, avg):
    dig = uic.loadUi("./interface/detail.ui")
    dig.tetinh.setText(str(thu))
    dig.temau.setText(str(mau))
    dig.lEsl.setText(str(sl))
    dig.tegt.setText("Giá trị std:" + "\n\n" + str(std) + "\n\n\n\n\n\n" + \
                         f"Giá trị max của csv:" + "\n\n" + str(ma) + "\n\n\n\n\n\n" + \
                         f"Giá trị min của csv:" + "\n\n" + str( mi) + "\n\n\n\n\n" + \
                         f"Giá trị trung bình csv:" + "\n\n" + str(avg))

    dig.exec_()