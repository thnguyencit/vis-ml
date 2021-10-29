from PyQt5 import QtWidgets, uic, QtGui, QtCore
import sys
import os
import glob
#import os.path
from time import gmtime, strftime
from keras.applications.vgg16 import preprocess_input as pre_vgg16
import pathlib
import pymysql
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bieudo import Ui_bieudo
from source import parameter
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from source import detail
from source import compare
from PyQt5.QtWidgets import QFileDialog
from source import guess
from PyQt5.QtWidgets import QMessageBox
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, InputLayer, Conv2D, MaxPooling2D
from source import model_popup
from keras.models import load_model
from keras import Sequential
from keras.models import model_from_json
from keras.preprocessing import image
app = QtWidgets.QApplication([])
dig = uic.loadUi("./interface/mainwindow.ui")
# dlP = uic.loadUi("./interface/model.ui")

def MessageBox(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(s)
    msg.exec_()

# nthai: load model file
def filesModel():
    options = QFileDialog.Options()

    options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(dig,"Select File h5 for pretraining", "",
                                                     "All Files (*);;Python Files (*.h5)", options=options)
    print('path file model chosen')
    print(files)
    if files:
        return files[0]
        
def history():
    conn = pymysql.connect(host="localhost", db="deepmg", user="root", password="")
    cur = conn.cursor()
    cur.execute("select value_applied from para_applied")
    for i in range(cur.rowcount):
        result = cur.fetchall()
        for row in result:
            # print(row)
            dig.list_hisstory.addItems(row)
    cur.close()
    conn.close()

# nthai
import ntpath
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def checkEmpty():
        if dig.lineFile1.text() == "":
            dig.lineFile1.setFocus()
            dig.lineFile1.setPlaceholderText("Data with csv extension should be provided")
            return
        else:
            data = dig.lineFile1.text()
        if dig.lineFile2.text() == "":
            dig.lineFile2.setFocus()
            dig.lineFile2.setPlaceholderText("Label with csv extension should be provided")
            return
        else:
            label = dig.lineFile2.text()
        
        # b = data.rstrip('_x.csv')
        # o = pathlib.Path.cwd().parent / 'data'
        # print(o)
        # x=str(o)
        # print(x)
        # d=b[(len(x)+18):]
        # print('test path test pathtest pathtest path ')
        # print(d)

        # nthai
        file_name=path_leaf(data)
        d = file_name.rstrip('_x.csv') # cut "_x_csv"   
        print(d)

        # k = pathlib.Path.cwd().parent
        # os.chdir(k)
        if dig.lineK.text() == "":
            dig.lineK.setFocus()
            dig.lineK.setPlaceholderText("Empty Value Not Allowed")
            return
        else:
            k_fold = dig.lineK.text()
        # print(k)
        if dig.cbEMB.currentText() == "":
            dig.cbEMB.setFocus()
            MessageBox("EMB empty value not allowed")
            return
        else:
            comboText1 = dig.cbEMB.currentText()
        if dig.cbBin.currentText() == "":
            dig.cbBin.setFocus()
            MessageBox("Bin empty value not allowed")
            return
        else:
            comboText2 = dig.cbBin.currentText()
        if dig.cbPreIMG.currentText() == "":
            dig.cbPreIMG.setFocus()
            MessageBox("Preprocess Image empty value not allowed")
            return
        else:
            comboText3 = dig.cbPreIMG.currentText()
        if dig.cbColormap.currentText() == "":
            dig.cbColormap.setFocus()
            MessageBox("Colormap empty value not allowed")
            return
        else:
            comboText4 = dig.cbColormap.currentText()
        run(data,label,d,k_fold,comboText1,comboText2,comboText3,comboText4)

        pathfile_main_deepmg= os.path.join(pathlib.Path.cwd(),'deepmg_v37','__main__.py')

        if k_fold in ['']:
            k_fold = 2

        if (comboText1 in ['se']):
            command = f"python {pathfile_main_deepmg} -i {d} -t {comboText1} -y {comboText2} -z 255 " \
                    f"--preprocess_img {comboText3} --colormap {comboText4} -k {k_fold} --search_already n --channel 3 "\
                    f"--save_w y "\
                    f"--eigen_solver arpack "
        else:
            command = f"python {pathfile_main_deepmg} -i {d} -t {comboText1} -y {comboText2} -z 255 " \
                    f"--preprocess_img {comboText3} --colormap {comboText4} -k {k_fold}  --search_already n --channel 3 "\
                    f"--save_w y "
      
        xuly1_2(data,label,command)

def files():
    options = QFileDialog.Options()

    options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(dig,"QFileDialog.getOpenFileNames()", "",
                                                     "All Files (*);;Python Files (*.csv)", options=options)
    if files:
        dig.lineFile1.setText(files[0])
        print(files[0])

def files1():
    options = QFileDialog.Options()

    options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(dig,"QFileDialog.getOpenFileNames()", "",
                                                     "All Files (*);;Python Files (*.csv)", options=options)
    if files:
        dig.lineFile2.setText(files[0])
        print(files[0])
def link_image():
    path = QFileDialog.getExistingDirectory(dig,"Select path image")
    dig.link_Image.setText(path)

def xuly1_2(data,label,command):
        xuly1 = pd.read_csv(data, encoding='utf-8', header='infer', sep=',')
        mau = xuly1.axes  # hoắcj columns
        sl = xuly1.shape
        thu = xuly1['Unnamed: 0'].values
        std = xuly1.std
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        ma = xuly1.max
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        mi = xuly1.min
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        avg = xuly1.mean
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        xuly2 = pd.read_csv(label, encoding='utf-8', header='infer', sep=',')
        nhan1 = xuly2[xuly2['x'] == 1]
        dem1 = xuly2['x'].value_counts()
        nhan2 = xuly2[xuly2['x'] == 0]
        dem2 = xuly2['x'].value_counts()
        dem = f'{dem1} {dem2}'
        conn = pymysql.connect(host="localhost", db="deepmg", user="root", password="")
        cursor = conn.cursor()
        conn.begin()
        #query1 = ('insert into dataset(X,Y) values (%s,%s)')
        #cursor.execute(query1, (data, label))
        query2 = ('insert into para_applied(value_applied) values (%s)')
        cursor.execute(query2, command)
        query3 = ('insert into class_contribution(SAMPLES_BELONG_TO_CLASS ) values (%s)')
        cursor.execute(query3, dem)

        conn.commit()
        cursor.close()
        conn.close()

# nthai: this place starts the experiment
def run(data,label,d,k_fold,comboText1,comboText2,comboText3,comboText4,deep_v='deepmg_v37'):
        #command = f"python C:\LV\GSOM-Application\deepmg_v37\__main__.py -i {d} -t {comboText1} -y {comboText2} -z 255 " \
        #          f"--preprocess_img {comboText3} --colormap {comboText4} -k {k_fold}  --search_already n --channel 3 "\
        #          f"--save_w y "


        pathfile_main_deepmg= os.path.join(pathlib.Path.cwd(),deep_v,'__main__.py')
        if k_fold in ['']:
            k_fold = 2
        

        
        # nthai: add options for the command 
        if (comboText1 in ['se']):
            command = f"python {pathfile_main_deepmg} -i {d} -t {comboText1} -y {comboText2} -z 255 " \
                  f"--preprocess_img {comboText3} --colormap {comboText4} -k {k_fold}  --search_already n --channel 3 "\
                  f"--save_w y "\
                  f"--eigen_solver arpack "
        else:
            command = f"python {pathfile_main_deepmg} -i {d} -t {comboText1} -y {comboText2} -z 255 " \
                  f"--preprocess_img {comboText3} --colormap {comboText4} -k {k_fold}  --search_already n --channel 3 "\
                  f"--save_w y "
        
        #nthai: additonal parameters
        if (parameter.dlP.lineRuntime.text() != ""):
            command = command + "--run_time "+ parameter.dlP.lineRuntime.text()
        if (parameter.dlP.cbPadding.currentText() != ""):
            command = command + " --padding "+ parameter.dlP.cbPadding.currentText()
        if (parameter.dlP.cbModel.currentText() != ""):
            command = command + " --model "+ parameter.dlP.cbModel.currentText()
        if (parameter.dlP.cbDelo.currentText() != ""):
            command = command + " --del0 "+ parameter.dlP.cbDelo.currentText()
        if (parameter.dlP.cbOptimizer.currentText() != ""):
            command = command + " --optimizer "+ parameter.dlP.cbOptimizer.currentText()
        if (parameter.dlP.cbLossfunc.currentText() != ""):
            command = command + " --loss_func "+ parameter.dlP.cbLossfunc.currentText()
        if (parameter.dlP.cbSVM.currentText() != ""):
            command = command + " --svm_kernel "+ parameter.dlP.cbSVM.currentText()
        if (parameter.dlP.cbMethodLLE.currentText() != ""):
            command = command + " --method_lle "+ parameter.dlP.cbMethodLLE.currentText()
        if (parameter.dlP.cbModeReduceDim.currentText() != ""):
            command = command + " --algo_redu "+ parameter.dlP.cbModeReduceDim.currentText()
        if (parameter.dlP.cbCoffe.text() != ""):
            command = command + " --coeff "+ parameter.dlP.cbCoffe.text()
        if (parameter.dlP.lr_visual.text() != ""):
            command = command + " --lr_visual "+ parameter.dlP.lr_visual.text()
        if (parameter.dlP.lineIterVisual.text() != ""):
            command = command + " --iter_visual "+ parameter.dlP.lineIterVisual.text()
        if (parameter.dlP.fig_size.text() != ""):
            command = command + " --momentum "+ parameter.dlP.fig_size.text()
        if (parameter.dlP.lineKnnN.text() != ""):
            command = command + " --knn_n_neighbors "+ parameter.dlP.lineKnnN.text()
        if (parameter.dlP.lineCmapvmin.text() != ""):
            command = command + " --cmap_vmin "+ parameter.dlP.lineCmapvmin.text()
        if (parameter.dlP.lineCmapvmax.text() != ""):
            command = command + " --cmap_vmax "+ parameter.dlP.lineCmapvmax.text()
        if (parameter.dlP.lineQuantile.text() != ""):
            command = command + " --n_quantile "+ parameter.dlP.lineQuantile.text()
        if (parameter.dlP.lineNumBin.text() != ""):
            command = command + " --num_bin "+ parameter.dlP.lineNumBin.text()
        if (parameter.dlP.lineLossfunc.text() != ""):
            command = command + " --loss_func "+ parameter.dlP.lineLossfunc.text()
        if (parameter.dlP.newdim.text() != ""):
            command = command + " --new_dim "+ parameter.dlP.newdim.text()
        
        print(command)
        xuly = os.popen(command)
        z = xuly.read()
        print(z)

        if(parameter.dlP.lineNumBin.text()=="" and comboText2 == "pr"):
            textcmd = "2"
        elif(parameter.dlP.lineNumBin.text()=="" and comboText2 != "pr"):
            textcmd = "10"
        else:
            textcmd = parameter.dlP.lineNumBin.text()

        
        # nthai: if we use dim reduction, see in function naming_folder in deepmg
        if not (parameter.dlP.cbModeReduceDim.currentText() in ['','none']):  #if use reducing dimension
            if parameter.dlP.cbModeReduceDim.currentText() == 'rd_pro':
                #if options.rd_pr_seed != "None":
                #    d = d + parameter.dlP.cbModeReduceDim.currentText() + str(parameter.dlP.newdim.text()) + '_'+ str(10)   + '_s' + str(options.rd_pr_seed)
                #else:
                d = d+ parameter.dlP.cbModeReduceDim.currentText() + str(parameter.dlP.newdim.text()) + '_'+ str(10)  
            else:
                d = d + parameter.dlP.cbModeReduceDim.currentText() + str(parameter.dlP.newdim.text()) + '_'+ str(10)   


        if(comboText1 == "zfill" or comboText1=="fill"):
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0/"
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0")

        # elif(comboText1 in ["tsne",'se','pca','isomap','mds']):
        #     #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
        #     #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
        #     path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"k1")

        elif(comboText1 in ['minisom',"tsne",'se','pca','isomap','mds'] and  str(parameter.dlP.lr_visual.text())==''):
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"k1")


        elif(comboText1 in ['minisom',"tsne",'se','pca','isomap','mds'] and str(parameter.dlP.lr_visual.text())!=''):
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_p5l"+ str(parameter.dlP.lr_visual.text()) + "i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"k1")
        
        else:
            # nb = parameter.dlP.lineNumBin.text()
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"k1")

        if(z==None):
            return
        else:
            dig.link_Image.setText(path_link_folder_img)
            url = dig.link_Image.text()
            show_img_default(url)
            FileSumOk(data,label,d,k_fold,comboText1,comboText2,comboText3,comboText4)
        history()

def modelcheckrun():# numlayer,numfilter,save_d,model_cnn,comboxSearch = checkEmpty()
        data = dig.lineFile1.text()
        label = dig.lineFile2.text()
        
        
        # print(dig)
        # print(data)
        # b = data.rstrip('_x.csv')
        # o = pathlib.Path.cwd().parent / 'data'

        # x=str(o)
        
        # d=b[(len(x)+18):]

        file_name=path_leaf(data)
        d = file_name.rstrip('_x.csv') # cut "_x_csv"   
        print(d)

        k_fold = dig.lineK.text()
        comboText1 = dig.cbEMB.currentText()
        print(comboText1)
        comboText2 = dig.cbBin.currentText()
        comboText3 = dig.cbPreIMG.currentText()
        comboText4 = dig.cbColormap.currentText()
        linkfileh5 = filesModel()

        linkfileh5 = linkfileh5.rstrip('.h5')

        pathfile_main_deepmg= os.path.join(pathlib.Path.cwd(),'deepmg_v37','__main__.py')

        if k_fold in ['']:
            k_fold = 2
        # in prediction without k
        if (comboText1 in ['se']):
            command =   f"python {pathfile_main_deepmg} -i {d} -t {comboText1} -y {comboText2} -z 255 " \
                        f"--preprocess_img {comboText3}  --colormap {comboText4} -k {k_fold}  --search_already=n -a predict --pretrained_w_path {linkfileh5} --model pretrained"\
                        f"--eigen_solver arpack "
        else:
            command =   f"python {pathfile_main_deepmg} -i {d} -t {comboText1} -y {comboText2} -z 255 " \
                        f"--preprocess_img {comboText3}  --colormap {comboText4} -k {k_fold}  --search_already=n -a predict --pretrained_w_path {linkfileh5} --model pretrained"
                    
        print('command for predictin')
        print(command)
        
        # nthai: run the prediction from trained model
        xuly = os.popen(command)
        
        #dig.pBtRunModel.clicked.connect(xuly)
        z = xuly.read()
        print('zzzzzzzzzzz')
        print(z)
       
        if(comboText2 == "pr"):
            textcmd = "2"
        else:
            textcmd = "10"
        
        # nthai: review image in whole for test in model
        if(comboText1 == "zfill" or comboText1=="fill"):
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0/"
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0","whole")

        # elif(comboText1 in ["tsne",'se','pca','isomap','mds']):
        #     #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
        #     #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
        #     path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"whole")

        elif(comboText1 in ['minisom',"tsne",'se','pca','isomap','mds'] and  str(parameter.dlP.lr_visual.text())==''):
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"whole")


        elif(comboText1 in ['minisom',"tsne",'se','pca','isomap','mds'] and  str(parameter.dlP.lr_visual.text())!=''):
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"_p5l"+ str(parameter.dlP.lr_visual.text()) + "i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"whole")

        else:
            # nb = parameter.dlP.lineNumBin.text()
            #path_link_folder_img= str(pathlib.Path.cwd())+"/images/"+d+"_"+comboText1+"pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/s1/nfold"+k_fold+"/k1/"
            path_link_folder_img= os.path.join(pathlib.Path.cwd(),'images',d+"_"+comboText1+"pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0",'s1',"nfold"+str(k_fold),"whole")


  
        if(z==None):
            return
        else:
            print('path after prediction')
            print(path_link_folder_img)
            dig.link_Image.setText(path_link_folder_img)
            url = dig.link_Image.text()
            print('url')
            print(url)
            show_img_default(url)
            FileSumOk(data,label,d,k_fold,comboText1,comboText2,comboText3,comboText4,predict='yes')


def show_img_default(highlight_dir):
    scrollArea =  dig.scrollArea
    content_widget =  QtWidgets.QWidget()
    scrollArea.setWidget(content_widget)
    lay =  QtWidgets.QGridLayout(content_widget)
    lay.setColumnStretch(5, 5)
    for file in os.listdir(highlight_dir):
        image_profile = QtGui.QImage(os.path.join(highlight_dir, file))
        image_profile = image_profile.scaled(100,100)
        pixmap = QtGui.QPixmap(image_profile)
        print(os.path.join(highlight_dir, file))
        if not pixmap.isNull():
            label = QtWidgets.QLabel()
            labeltext = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            labeltext.setText(file)
            labeltext.setAlignment(Qt.AlignBottom)
            lay.addWidget(label)
            lay.addWidget(labeltext)

def show_img():
    highlight_dir = dig.link_Image.text()
    scrollArea =  dig.scrollArea
    content_widget =  QtWidgets.QWidget()
    scrollArea.setWidget(content_widget)
    lay =  QtWidgets.QGridLayout(content_widget)
    lay.setColumnStretch(5, 5)
    for file in os.listdir(highlight_dir):
        image_profile = QtGui.QImage(os.path.join(highlight_dir, file))
        image_profile = image_profile.scaled(100,100)
        pixmap = QtGui.QPixmap(image_profile)
        print(os.path.join(highlight_dir, file))
        if not pixmap.isNull():
            label = QtWidgets.QLabel()
            labeltext = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            labeltext.setText(file)
            labeltext.setAlignment(Qt.AlignBottom)
            lay.addWidget(label)
            lay.addWidget(labeltext)
            
def showchart():
        label = dig.lineFile2.text()
        data = dig.lineFile1.text()
        xuly1 = pd.read_csv(data, encoding='utf-8', header='infer', sep=',')
        k = pathlib.Path(__file__).parent
        # dliu theo mẫu
        df = xuly1.drop('Unnamed: 0', axis=1)
        plt.figure(figsize=(10, 5))
        plt.title('Phân phối dữ liệu theo thuộc tính')
        plt.plot(df)
        # plt.savefig('C:/Users/DELL/Downloads/pyqt5/pyqt/anhbieudo/a.png')
        # plt.savefig(k+'/anhbieudo/a.png')
        print("duong dan", str(k)+r'/anhbieudo/a.png')
        plt.savefig(str(k)+r'/anhbieudo/a.png')
        # plt.savefig(k+'/anhbieudo/a.png')
        # f=k+'/anhbieudo/a.png'
        # print("f ne",f)
        # Thuoc tinh
        dao = xuly1.T
        # print(a)
        # dao.to_csv('C:/Users/DELL/Downloads/pyqt5/pyqt/ghifile/thuoctinh.csv', index=True, header=None)
        dao.to_csv(str(k)+r'/ghifile/thuoctinh.csv', index=True, header=None)
        # docdao = pd.read_csv('C:/Users/DELL/Downloads/pyqt5/pyqt/ghifile/thuoctinh.csv', encoding='utf-8',
        #                      header='infer', sep=',')
        docdao = pd.read_csv(str(k)+r'/ghifile/thuoctinh.csv', encoding='utf-8',header='infer', sep=',')


        xoa2 = docdao.drop('Unnamed: 0', axis=1)
        tt = plt.figure(figsize=(10, 5))
        plt.title('Phân phối dữ liệu theo các mẫu')
        plt.plot(xoa2)
        # tt.savefig('C:/Users/DELL/Downloads/pyqt5/pyqt/anhbieudo/e.png')
        tt.savefig(str(k)+r'/anhbieudo/e.png')

        # Biểu đồ bệnh nhân bị bệnh
        xuly2 = pd.read_csv(label, encoding='utf-8', header='infer', sep=',')
        nhan = ['có bệnh', 'không bệnh']
        plt.title('Phân phối các mẫu theo phân lớp')
        xuly2['x'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=nhan, subplots=True)
        # plt.savefig('C:/Users/DELL/Downloads/pyqt5/pyqt/anhbieudo/i.png')
        plt.savefig(str(k)+r'/anhbieudo/i.png')
        bieudo = QtWidgets.QDialog()
        ui = Ui_bieudo()
        ui.setupUi(bieudo)
        bieudo.exec_()

def showdetail():
    data = dig.lineFile1.text()
    xuly1 = pd.read_csv(data, encoding='utf-8', header='infer', sep=',')
    mau = xuly1.axes  # hoắcj columns
    sln = xuly1.shape
    sl = (sln[0],sln[1]-1)
    thu = xuly1['Unnamed: 0'].values
    std = xuly1.std
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    ma = xuly1.max
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    mi = xuly1.min
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    avg = xuly1.mean
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    detail.Show(thu, mau, sl, std, ma, mi, avg)

def reload():
    # print("hello")
    dig.close()
    os.system("python mainWindows.py")
    # print("Restarting...")
    exit()

# nthai: read from fileSum to plot on the charts
def FileSumOk(data,label,d,k_fold,comboText1,comboText2,comboText3,comboText4,predict='no'):#i

    
    i = 0
    j= 0
    x = list()
    if(parameter.dlP.lineNumBin.text()=="" and comboText2 == "pr"):
        textcmd = "2"
    elif(parameter.dlP.lineNumBin.text()=="" and comboText2 != "pr"):
        textcmd = "10"
    else:
        textcmd = parameter.dlP.lineNumBin.text()
    if(comboText1 in ["zfill", "fill"]):
        # folder_path= str(pathlib.Path.cwd())+"/results/"+d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0/"
        folder_path= os.path.join(pathlib.Path.cwd(),'results',d+"_"+comboText1+"_pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"bi"+textcmd+"_0.0_1.0/")
    # elif(comboText1 in ['tsne','se','pca','isomap','mds']):
    #     #folder_path= str(pathlib.Path.cwd())+"/results/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/"
    #     folder_path= os.path.join(pathlib.Path.cwd(),'results',d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/")
    
    elif(comboText1 in ['minisom','tsne','se','pca','isomap','mds'] and str(parameter.dlP.lr_visual.text())!=''):
        #folder_path= str(pathlib.Path.cwd())+"/results/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/"
        folder_path= os.path.join(pathlib.Path.cwd(),'results',d+"_"+comboText1+"_p5l" + str(parameter.dlP.lr_visual.text()) + "i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/")
    
    elif(comboText1 in ['minisom','tsne','se','pca','isomap','mds'] and str(parameter.dlP.lr_visual.text())==''):
        #folder_path= str(pathlib.Path.cwd())+"/results/"+d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/"
        folder_path= os.path.join(pathlib.Path.cwd(),'results',d+"_"+comboText1+"_p5l100.0i300pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/")
        
    else:
        #folder_path= str(pathlib.Path.cwd())+"/results/"+d+"_"+comboText1+"pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/"
        folder_path= os.path.join(pathlib.Path.cwd(),'results',d+"_"+comboText1+"pix_r0p1.0"+comboText2+"m0.0a1.0"+comboText4+"_nb"+textcmd+"_aun_0.0_1.0/")

    print('###folder_pathfolder_pathfolder_path####')
    print(folder_path)
    # find sum_ok.txt files
    files_in_folder = glob.glob(folder_path + '*_ok.txt')   
    # filesname = ''
    print('folder_files###folder_files###folder_files###folder_files###')
    print(files_in_folder) 
    # determine the one with the latest time
    files = max(files_in_folder, key=os.path.getctime)
    print (files)

    # nthai, merge path
    #files=os.path.join(folder_path,filesname)
    # print('files')
    # print(files)
    if predict == 'no':
        with open(files) as f:
            for line in f:
                arr = line.split()
                #           0       1       2       3       4       5       6   7       8       9   10      11  12          13      14  15      16  17      18  
                if(arr == ['time', 'tr_ac', 'va_ac', 'sd', 'va_au', 'sd', 'tn', 'fn', 'fp', 'tp', 'preci', 'sd', 'recall', 'sd', 'f1', 'sd', 'mcc', 'sd', 'epst']):
                    j = i
                x.append(line)
                i= i+1
        arr = x[j].split()
        arr1 = x[j+1].split()
        dig.tbguess.setItem(0,0, QTableWidgetItem(arr[0])) 
        dig.tbguess.setItem(0,1, QTableWidgetItem(arr[1])) 
        dig.tbguess.setItem(0,2, QTableWidgetItem(arr[2])) 
        dig.tbguess.setItem(0,3, QTableWidgetItem(arr[4])) 
        dig.tbguess.setItem(0,4, QTableWidgetItem(arr[14])) 
        dig.tbguess.setItem(0,5, QTableWidgetItem(arr[16])) 
        dig.tbguess.setItem(0,6, QTableWidgetItem(arr[18])) 

        dig.tbguess.setItem(1,0, QTableWidgetItem(arr1[0])) 
        dig.tbguess.setItem(1,1, QTableWidgetItem(arr1[1])) 
        dig.tbguess.setItem(1,2, QTableWidgetItem(arr1[2])) 
        dig.tbguess.setItem(1,3, QTableWidgetItem(arr1[4]))
        dig.tbguess.setItem(1,4, QTableWidgetItem(arr1[14])) 
        dig.tbguess.setItem(1,5, QTableWidgetItem(arr1[16]))
        dig.tbguess.setItem(1,6, QTableWidgetItem(arr1[18])) 
    
    else: # with prediction from model
        # show accuracy
        with open(files) as f:
            for line in f:
                arr = line.split()
                if(arr == ['acc',	'auc',	'mcc'	,'tn',	'fp',	'fn',	'tp']):
                    j = i
                x.append(line)
                i= i+1
        arr = x[j].split()
        arr1 = x[j+1].split()
        print('arr1')
        print(arr1)
        dig.tbguess.setItem(0,0, QTableWidgetItem(arr[0])) 
        dig.tbguess.setItem(0,1, QTableWidgetItem(arr[1])) 
        dig.tbguess.setItem(0,2, QTableWidgetItem(arr[2])) 
        dig.tbguess.setItem(0,3, QTableWidgetItem(arr[3])) 
        dig.tbguess.setItem(0,4, QTableWidgetItem(arr[4])) 
        dig.tbguess.setItem(0,5, QTableWidgetItem(arr[5])) 
        dig.tbguess.setItem(0,6, QTableWidgetItem(arr[6])) 

        dig.tbguess.setItem(1,0, QTableWidgetItem(arr1[0])) 
        dig.tbguess.setItem(1,1, QTableWidgetItem(arr1[1])) 
        dig.tbguess.setItem(1,2, QTableWidgetItem(arr1[2])) 
        dig.tbguess.setItem(1,3, QTableWidgetItem(arr1[3]))
        dig.tbguess.setItem(1,4, QTableWidgetItem(arr1[4]))
        dig.tbguess.setItem(1,5, QTableWidgetItem(arr1[5]))
        dig.tbguess.setItem(1,6, QTableWidgetItem(arr1[6]))  

        # find sum_ok.txt files
        files_in_folder = glob.glob(folder_path + '*predicted.txt')   
        # filesname = ''
        print('folder_files')
        print(files_in_folder) 
        # determine the one with the latest time
        files = max(files_in_folder, key=os.path.getctime)
        print (files)

        # show each sample for diagnosis
        i = 0
        with open(files) as f:
            #dig.tbguess_2 = QTableWidget(2,3,dig)
            #dig.tbguess_2.setHorizontalHeaderLabels(["Name","I","City",''])
            for line in f:
                arr = line.split()
                #dig.tbguess_2.setItem(i,0, QTableWidgetItem(i)) 
                dig.tbguess_2.setItem(i,0, QTableWidgetItem(arr[0])) 
                dig.tbguess_2.setItem(i,1, QTableWidgetItem(arr[1])) 
                dig.tbguess_2.setItem(i,2, QTableWidgetItem(arr[2])) 
                dig.tbguess_2.setItem(i,3, QTableWidgetItem(arr[3])) 
                i = i + 1


def resultParameter():
    parameter.reloadParameter()
    parameter.Show()

def model_cnn(input_reshape=(32,32,3),num_classes=2,optimizers_func='Adam',numfilter=20, 
        filtersize=3,numlayercnn_per_maxpool=1,nummaxpool=1,maxpoolsize=2,
        dropout_cnn=0, dropout_fc=0,lr_rate=0.0005,lr_decay=0, loss_func='binary_crossentropy', padded='n',type_emb='n' ) :
    """ architecture CNNs with specific filters, pooling...
    Args:
        input_reshape (array): dimension of input
        num_classes (int): the number of output of the network
        optimizers_func (string): optimizers function
        lr_rate (float): learning rate, if use -1 then use default values of the optimizer
        lr_decay (float): learning rate decay
        loss_func (string): loss function

        numfilter (int): number of filters (kernels) for each cnn layer
        filtersize (int): filter size
        numlayercnn_per_maxpool (int): the number of convolutional layers before each max pooling
        nummaxpool (int): the number of max pooling layer
        maxpoolsize (int): pooling size
        dropout_cnn (float): dropout at each cnn layer
        dropout_fc (float): dropout at the FC (fully connected) layers
        padded: padding 'same' to input to keep the same size after 1st conv layer
        
    Returns:
        model
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_reshape))  
    if not (type_emb in ['none']):
        try:
            model.add(InputLayer(input_shape=input_reshape))  # commented by tbtoan
        except:
            print("there is something wrong with InputLayer")



    for j_pool in range(1,nummaxpool+1):
        print(("j_pool"+str(j_pool)))
        for i_layer in range(1,numlayercnn_per_maxpool+1):
            if i_layer==1:
                if padded=='y':
                    #use padding
                    model.add(Conv2D(numfilter, (filtersize, filtersize), padding='same'))
                else:
                    #do not use padding
                    model.add(Conv2D(numfilter, (filtersize, filtersize)))
            else:                
                model.add(Conv2D(numfilter, (filtersize, filtersize)))
            model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(maxpoolsize,maxpoolsize)))
        if dropout_cnn > 0:
            model.add(Dropout(dropout_cnn))
        #model.add(Conv2D(1, (1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))
    
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))   
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
       

    if lr_rate > -1:           
        if optimizers_func=="adam":
            optimizers_func = kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            optimizers_func = kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
        elif optimizers_func=="rmsprop":
            optimizers_func = kr.optimizers.rmsprop(lr=lr_rate, decay=lr_decay) 
  

    model.compile(loss=loss_func,
            optimizer=optimizers_func,
            metrics=['accuracy'])
        
    return model

# def read_fileimg():
    #file_mcnn = "C:/GSOM-Application/results/wt2dphy_fillspix_r0p1.0spbm0.0a1.0rainbow_nb10_aun_0.0_1.0/models/a1_k3_vgg16caffe_estopc5_fc_o1adam_lr-1.0de0.0e100_20210103_144651c255.0di-1ch3dfc0.0model_s1k1.json"
    #file_wcnn = "C:/GSOM-Application/results/wt2dphy_fillspix_r0p1.0spbm0.0a1.0rainbow_nb10_aun_0.0_1.0/models/a1_k3_vgg16caffe_estopc5_fc_o1adam_lr-1.0de0.0e100_20210103_144651c255.0di-1ch3dfc0.0model_s1k1.h5"

    ##read structures and weights of model
    json_file = open(file_mcnn, 'r')
    loaded_model_jsoncnn = json_file.read()
    json_file.close()
    model = model_cnn()
    # model = model_from_json(loaded_model_jsoncnn)
    model.summary()
    # model.load_weights(file_wcnn)
    print(model)
    #path_read = "C:/GSOM-Application/images/wt2dphy_fillspix_r0p1.0spbm0.0a1.0rainbow_nb10_aun_0.0_1.0/s1/nfold3/k1/"
    temp_proceed=[]
    images_folder = os.listdir(path_read)
    mode_image = 'color'
    ##read proceed images with vgg16
    for n,images in enumerate(images_folder):
        image_path = os.path.join(path_read, images)   
        
        if mode_image != 'gray':
            #print 'color'
            img = image.load_img(image_path)
            x = image.img_to_array(img) 
            #temp_ori.append(x) 
            x = pre_vgg16(x)        
        else:
            #print 'gray' + str(i)
            x = image.load_img(image_path,grayscale=True)
            x = image.img_to_array(x) 
            #print x.shape
        temp_proceed.append(x) 

    temp_proceed=np.stack(temp_proceed)

    temp_proceed /= 255 #this is very important so that the network works
    print ('shape of whole dataset = '+ str(temp_proceed.shape))
    # model.fit()
    #print temp.shape
    res_proceed= model.predict(temp_proceed)
    acc_score_global = accuracy_score(labels, res_proceed.round())
    print ('global performance on whole dataset (in accuracy) = ' + str(acc_score_global))   

def main():
    # read_fileimg()
    dig.pBtDetail.clicked.connect(showdetail)
    dig.pBtParameter.clicked.connect(resultParameter)
    dig.pBtCompare.clicked.connect(compare.Show)

    dig.pBtModel.clicked.connect(modelcheckrun)
    dig.pBtChooseFile1.clicked.connect(files)
    dig.pBtChooseFile2.clicked.connect(files1)
    dig.pBtRun.clicked.connect(checkEmpty)
    dig.pBtnImage.clicked.connect(link_image)
    dig.pBtnShow_Img.clicked.connect(show_img)
    dig.pBtnChart.clicked.connect(showchart)
    dig.pBtReload.clicked.connect(reload)
    history()

    dig.show()

if __name__ == '__main__':
    main()
app.exec()