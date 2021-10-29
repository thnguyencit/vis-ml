#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
#this file aims to visualize the filters and outputs of the network using Grad-CAM and LIME
#DATE: 27/07/2019 (v.2) --> convert to python3, changed path
#DATE: 27/03/2018 (v.1)

#SECTION 1: Load models trained (*.json) and (*.h5)
'''

'''### important! set PATH of machine used '''
#path_machine = "/Users/hainguyen/deepMG_tf/"
path_machine = "/Users/hainguyen/Documents/nthai/PhD/workspace/p3_deepmg/"

from time import gmtime, strftime
import time
start_time = time.time() 

from time import gmtime, strftime
#now, supporting 2 modes of color and gray, please set mode here:
mode_image = 'color'

#libraries
from keras.models import model_from_json
import pandas as pd

len_square = 24
#dataset images
if mode_image=='gray':
    #gray
    data_folder = 'cirphy_fill_pix_r0p1abm0a1graybi10_1e-07_0.0065536/'
else:
    #color
    #data_folder = 'cirphy_fill_pix_r0p1abm0a1colorbi10_1e-07_0.0065536/'
    data_folder = 'cirphy_fill_pix_r0p1abm0a1colorjet_rbi10_1e-07_0.0065536/'
    
#folder contains jupyter
path_jupyter = path_machine + '/W_ONLY_dev/jupyter/'
    
#folder contains data
image_path1 = path_jupyter+ '/models/' + data_folder

#folder containing image results
path_saved_img = path_jupyter + '/generated_img/'

##files of models and weights
if mode_image=='gray':
    name_m = 'nonecaffe_estopc5_cnn_o1adam_lr-1de0e500_20180416_093507c255.0di-1ch1l1p1f64dcnn0dfc0padweightmodel_s9k1'
else:
    #name_m = 'vgg16caffe_estopc5_cnn_o1adam_lr-1de0e500_20180320_102149c255.0di-1ch3l1p1f32dcnn0dfc0padmodel_s1k1'
    name_m = 'vgg16caffe_estopc5_cnn_o1adam_lr-1de0e500_20180510_131029c255.0di-1ch3l1p1f64dcnn0dfc0padweightmodel_s1k1'
file_mcnn = image_path1 + name_m+ str(".json")
file_wcnn = image_path1 + name_m+ str(".h5")

##read structures and weights of model
json_file = open(file_mcnn, 'r')
loaded_model_jsoncnn = json_file.read()
json_file.close()
model = model_from_json(loaded_model_jsoncnn)
model.load_weights(file_wcnn)
print (model)


# In[7]:


'''
#SECTION 2: make predictions
**Notes: 
    please check whether images used for prediction exist or not in folder ./images/
    if those images do not exist, run this command to generate
        python dev_met2img.py -i cirphy -t fill -y ab --colormap jet_r --fig_size 0
    after having 232 images of cirrhosis, we stop the command and run this script.
'''
##load images and make prediction based a trained network
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input as pre_resnet50
from keras.applications.vgg16 import preprocess_input as pre_vgg16

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


##read labels
path_read = path_machine + '/images/' + data_folder 
labels = pd.read_csv(os.path.join(path_read, 'y.csv'),header=None)
labels = labels.iloc[:,1] 
  
##read original images
temp_ori=[]   
for i in range(0,len(labels)):
    image_path = os.path.join(path_read, 'fill_'+str(i)+'.png')  
    img = image.load_img(image_path)
    x = image.img_to_array(img) 
    temp_ori.append(x)    
temp_ori=np.stack(temp_ori)

#if mode_image <> 'gray':
#    temp_ori /= 255
#    res1= model.predict(temp_ori)
#    acc_score_global = accuracy_score(labels, res1.round())
#    print acc_score_global #global accuracy 
#    print res1[1:5] #print 5 first samples


temp_proceed=[]
##read proceed images with vgg16
for i in range(0,len(labels)):
    image_path = os.path.join(path_read, 'fill_'+str(i)+'.png')   
    
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
#print temp.shape
res_proceed= model.predict(temp_proceed)
acc_score_global = accuracy_score(labels, res_proceed.round())
print ('global performance on whole dataset (in accuracy) = ' + str(acc_score_global))
#print res_proceed[3]


# In[3]:


##show information on model to know the name of layers
print model.summary()


# In[4]:


'''
visualize outputs (PAIRED: Original - OUTPUT) of the network using visualize_cam
'''
#visualize the results as well original
#Gradient-weighted Class Activation Mapping
get_ipython().run_line_magic('matplotlib', 'inline')
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from vis.visualization import visualize_cam
from keras import activations
import math
from time import gmtime, strftime
import time
time_text = str(strftime("%Y%m%d_%H%M%S", gmtime()))
layer_idx = utils.find_layer_idx(model, 'dense_1')
off_set = 76
num_cols=6
num_rows=6
size_f = 22
plt.rcParams['figure.figsize'] = (size_f, size_f * num_rows/num_cols)

for modifier in ['guided']: #we can see guided is better
    plt.figure()
    f, ax = plt.subplots(num_rows, num_cols)
    plt.suptitle(modifier)
    for i in range(0,num_cols*num_rows):    
        # 20 is the imagenet index corresponding to `ouzel`
        #print str(int(math.ceil(i/num_cols)))+'_'+str(int(i%num_cols))
        if i==0:
            index_img = off_set
        if i%2==0:
            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(temp_ori[index_img]*255,extent=[(len_square-1)*(-1),0,(len_square-1)*(-1),0])
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_l'+str(labels[index_img]))
            
        else:
            #temp_ori: original images, temp_vgg16: vgg16
            grads = visualize_cam(model, layer_idx, filter_indices=0, 
                                       seed_input=temp_proceed[index_img], backprop_modifier=modifier)
            # Lets overlay the heatmap onto original image.    

            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(grads,extent=[(len_square-1)*(-1),0,(len_square-1)*(-1),0])
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_p'+ str(res_proceed[index_img].round(2)))
            index_img = index_img + 1
    plt.show();
    f.savefig(path_saved_img+'visualize_cam_all.'+time_text+'.png')


# In[5]:


'''
visualize outputs (PAIRED: Original - OUTPUT) of the network using visualize_saliency
'''
#visualize the results as well original
#Gradient-weighted Class Activation Mapping
get_ipython().run_line_magic('matplotlib', 'inline')
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from vis.visualization import visualize_cam
from keras import activations
import math
time_text = str(strftime("%Y%m%d_%H%M%S", gmtime()))
layer_idx = utils.find_layer_idx(model, 'dense_1') #for model with 3 channels input
#layer_idx = utils.find_layer_idx(model, 'dense_81')#1 channels
#plt.rcParams['figure.figsize'] = (12, 12)
#off_set = 82
#num_cols=6
#num_rows=4
off_set = 76
num_cols=6
num_rows=6
size_f = 22 #set the size of view to show images
plt.rcParams['figure.figsize'] = (size_f, size_f * num_rows/num_cols)

for modifier in ['guided']: #we can see guided is better
    plt.figure()
    f, ax = plt.subplots(num_rows, num_cols)
    
    plt.suptitle(modifier)
    for i in range(0,num_cols*num_rows):    
        # 20 is the imagenet index corresponding to `ouzel`
        #print str(int(math.ceil(i/num_cols)))+'_'+str(int(i%num_cols))
        if i==0:
            index_img = off_set
        if i%2==0:
            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(temp_ori[index_img]*255,interpolation='none', extent=[(len_square-1)*(-1),0,(len_square-1)*(-1),0])
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_l'+str(labels[index_img]))
            
        else:
            grads = visualize_saliency(model, layer_idx, filter_indices=0, 
                                       seed_input=temp_proceed[index_img], backprop_modifier=modifier)
            # Lets overlay the heatmap onto original image.    
            #use interpolation='none', extent=[-23,0,-23,0] to change values of axis
            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(grads,interpolation='none', extent=[(len_square-1)*(-1),0,(len_square-1)*(-1),0])
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_p'+ str(res_proceed[index_img].round(2)))
            index_img = index_img + 1
        #ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_ylim([-23,0])
        #ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_xlim([-23,0])
        #ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_xlabel([-23,0])
        #ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_ylabel([-23,0]) 
        #ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_xticklabels(edges)
    plt.show();
    f.savefig(path_saved_img+'visualize_saliency_all.'+time_text+'.png')


# In[ ]:





# In[6]:


'''
visualize outputs (PAIRED: Original - OUTPUT) of the network using LIME
'''
#visualize the results as well original
#Gradient-weighted Class Activation Mapping
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from vis.visualization import visualize_cam
from keras import activations
import math

import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from skimage.segmentation import mark_boundaries
from lime import lime_image

layer_idx = utils.find_layer_idx(model, 'dense_1')
#plt.rcParams['figure.figsize'] = (12, 12)
#off_set = 82
#num_cols=6
#num_rows=4
off_set = 76
num_cols=6
num_rows=6
plt.rcParams['figure.figsize'] = (15, 18 * num_rows/num_cols)

for modifier in ['guided']: #we can see guided is better
    plt.figure()
    f, ax = plt.subplots(num_rows, num_cols)
    plt.suptitle(modifier)
    for i in range(0,num_cols*num_rows):    
        # 20 is the imagenet index corresponding to `ouzel`
        #print str(int(math.ceil(i/num_cols)))+'_'+str(int(i%num_cols))
        if i==0:
            index_img = off_set
        if i%2==0:
            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(temp_ori[index_img]*255)
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_l'+str(labels[index_img]))
            
        else:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(temp_proceed[index_img], model.predict, top_labels=1)
            temp, mask = explanation.get_image_and_mask(0,positive_only=True, num_features=10, hide_rest=True)
            grads = mark_boundaries(temp/2 +0.5 , mask)
            #grads = mark_boundaries(temp , mask)
            #print grads
            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(grads,cmap='jet')
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_p'+ str(res_proceed[index_img].round(2)))
            index_img = index_img + 1
    f.savefig(path_saved_img+'visualize_lime_all.'+time_text+'.png')


# In[7]:


'''
visualize outputs (PAIRED: Original - OUTPUT) of the network using LIME
use pros and cons
'''
#visualize the results as well original
#Gradient-weighted Class Activation Mapping
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from vis.visualization import visualize_cam
from keras import activations
import math

import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from skimage.segmentation import mark_boundaries
from lime import lime_image

layer_idx = utils.find_layer_idx(model, 'dense_1')
#plt.rcParams['figure.figsize'] = (12, 12)
#off_set = 82
#num_cols=6
#num_rows=4
off_set = 76
num_cols=6
num_rows=6
size_f = 22
plt.rcParams['figure.figsize'] = (size_f, size_f * num_rows/num_cols)

for modifier in ['guided']: #we can see guided is better
    plt.figure()
    f, ax = plt.subplots(num_rows, num_cols)
    plt.suptitle(modifier)
    for i in range(0,num_cols*num_rows):    
        # 20 is the imagenet index corresponding to `ouzel`
        #print str(int(math.ceil(i/num_cols)))+'_'+str(int(i%num_cols))
        if i==0:
            index_img = off_set
        if i%2==0:
            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(temp_ori[index_img]*255,extent=[(len_square-1)*(-1),0,(len_square-1)*(-1),0])
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_l'+str(labels[index_img]))
            
        else:
            explainer = lime_image.LimeImageExplainer()
            
            #unprocessed images
            #explanation = explainer.explain_instance(temp_ori[index_img], model.predict, top_labels=1)
            #print temp_proceed
            explanation = explainer.explain_instance(temp_ori[index_img]/255, model.predict, top_labels=1)
            temp, mask = explanation.get_image_and_mask(0,positive_only=False, num_features=10, hide_rest=False)
            #img = mark_boundaries(temp/2 +0.5 , mask)
            #plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            #grads = mark_boundaries(temp , mask)
            grads = mark_boundaries(temp, mask)
            #print grads
            #print grads
            #sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(grads*10,cmap=plt.cm.nipy_spectral,extent=[(len_square-1)*(-1),0,(len_square-1)*(-1),0])
            
            sc=ax[int(math.ceil(i/num_cols))][int(i%num_cols)].imshow(grads,extent=[(len_square-1)*(-1),0,(len_square-1)*(-1),0])
            #f.colorbar(sc, ax=ax[int(math.ceil(i/num_cols))][int(i%num_cols)],orientation='horizontal')
            ax[int(math.ceil(i/num_cols))][int(i%num_cols)].set_title(str(index_img)+'_p'+ str(res_proceed[index_img].round(2)))
            index_img = index_img + 1
f.savefig(path_saved_img+'visualize_lime_all.'+time_text+'.png')


# In[8]:


print 'execution time all cells = ' + str(time.time() - start_time)


# In[ ]:





# In[ ]:




