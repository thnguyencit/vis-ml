"""
======================================================================================
utils_deepmg functions
======================================================================================
Author: Thanh Hai Nguyen
date: 20/12/2017 (updated to 31/10/2018, stable version)'
'this module includes:
'1. find_files: find files based on a given pattern, aiming to avoid repeating the experiments
'2. load_img_util: load images
'3. textcolor_display: show messages in the screen with color 
"""

#from scipy.misc import imread
import numpy as np
import os, fnmatch
import math
from sys import exit

def find_files(pattern, path):
    """ find files in path based on pattern
    Args:
        pattern (string): pattern of file
        path (string): path to look for the files
    Return 
        list of names found
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def load_img_util (num_sample,path_write,dim_img,preprocess_img,channel,mode_pre_img, pattern_img='train',type_emb='none',data='train',suf_after_i=''):
    """ load and reading images save to data array 
    Args:
        data_dir (array): folder contains images
        pattern_img (string): the pattern names of images (eg. 'img_1.png, img_2.png,...' so pattern='img') 
        num_sample (int) : the number of images read
        dim_img: dimension of images
        preprocess_img: preprocessing images, support: vgg16, vgg19, resnet50, incep
        path_write: path to save images

    Returns:
        array
    """
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input as pre_resnet50
    from keras.applications.vgg16 import preprocess_input as pre_vgg16
    from keras.applications.vgg19 import preprocess_input as pre_vgg19
    from keras.applications.inception_v3 import  preprocess_input as pre_incep
    
    # this var is value which is returned for the function
    temp = []
    for i in range(0,num_sample): #load samples for learning
        #image_path = os.path.join(path_write, str(pattern_img) +str("_") + str(i) +".png")               
        
        # added by tbtoan ##begin ## begin
        if type_emb == 'none': # added by tbtoan
            image_path = data[i]
        else:
            if suf_after_i != '':
                image_path = os.path.join(path_write, str(pattern_img) +str("_") + str(i) + str(suf_after_i[i]) +".png")               
            else:
                image_path = os.path.join(path_write, str(pattern_img) +str("_") + str(i) + ".png")               
        # added by tbtoan ##end ## end

        if dim_img==-1: #if use real img
            if channel == 4:
                #img = imread(image_path)
                #img = img.astype('float32')
                print ('waiting for fixing issues from from scipy.misc import imread')
                exit()
            elif channel == 1:
                img = image.load_img(image_path,grayscale=True)
            else:   
                img = image.load_img(image_path)
        else: #if select dimension
           
            if channel == 1:
                img = image.load_img(image_path,grayscale=True, target_size=(dim_img, dim_img))
            
            else:
                img = image.load_img(image_path,target_size=(dim_img, dim_img))

        x = image.img_to_array(img)         
        # x = preprocess_input(x)
        if preprocess_img=='vgg16':
            x = pre_vgg16(x, mode= mode_pre_img)
        elif preprocess_img=='resnet50':
            x = pre_resnet50(x, mode= mode_pre_img)
        elif preprocess_img=='vgg19':
            x = pre_vgg19(x, mode= mode_pre_img)
        elif preprocess_img=='incep':
            x = pre_incep(x)
    
        temp.append(x)      

    return np.stack(temp)             

def textcolor_display(text,type_mes='er'):
    '''
    show text in format with color
    Agr:
        text: text to format with color
        type_mes : type of messgage could be 'er' or 'inf'
    return 
        text formatted with color
    '''
    import platform
    end = '\x1b[0m'
    if type_mes in ['er','error']:
        
        if platform.system() == 'Windows':
            begin = ''
            end = ''  
        else:
            begin = '\x1b[1;33;41m' 
        return begin + text + end
    
    if type_mes in ['inf','information']:
        if platform.system() == 'Windows':
            begin = ''
            end = '' 
        else:
            begin = '\x1b[0;36;44m'        
        return begin + text + end



# added by tbtoan  ##begin ######begin ######begin ######begin ####
def read_from_directoty(directoty):
    '''
    # Arguments
        directory: string, path to the target directory.
            It should contain one subdirectory per class.
            Any PNG or JPG images
            inside each of the subdirectories directory tree
            will be included.

    #  Directory tree structures:
        directoty
            --class1
                --img1.png
                --img2.png
            --class2
                --img1.png
                --img2.png
    # Return
        A tuple of `(x, y)` where `x` is list contains all path-to-images in subdirectories
        `y` is a numpy array of corresponding labels
    '''
    
    import multiprocessing.pool

    def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):

        """Lists paths of files in `subdir` with extensions in `white_list_formats`.
        # Arguments
            directory: absolute path to a directory containing the files to list.
                The directory name is used as class label
                and must be a key of `class_indices`.
            white_list_formats: set of strings containing allowed extensions for
                the files to be counted.
            split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
                account a certain fraction of files in each directory.
                E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
                of images in each directory.
            class_indices: dictionary mapping a class name to its index.
            follow_links: boolean, follow symbolic links to subdirectories.

        # Returns
            classes: a list of class indices
            filenames: the path of valid files in `directory`, relative from
                `directory`'s parent (e.g., if `directory` is "dataset/class1",
                the filenames will be
                `["class1/file1.jpg", "class1/file2.jpg", ...]`).
        """

        dirname = os.path.basename(directory)
        if split:
            all_files = list(_iter_valid_files(directory, white_list_formats,
                                            follow_links))
            num_files = len(all_files)
            start, stop = int(split[0] * num_files), int(split[1] * num_files)
            valid_files = all_files[start: stop]
        else:
            valid_files = _iter_valid_files(
                directory, white_list_formats, follow_links)
        classes = []
        filenames = []
        for root, fname in valid_files:
            classes.append(class_indices[dirname])
            absolute_path = os.path.join(root, fname)
            relative_path = os.path.join(
                dirname, os.path.relpath(absolute_path, directory))
            filenames.append(relative_path)

        return classes, filenames

    def _iter_valid_files(directory, white_list_formats, follow_links):

        """Iterates on files with extension in `white_list_formats` contained in `directory`.

        # Arguments
            directory: Absolute path to the directory
                containing files to be counted
            white_list_formats: Set of strings containing allowed extensions for
                the files to be counted.
            follow_links: Boolean, follow symbolic links to subdirectories.

        # Yields
            Tuple of (root, filename) with extension in `white_list_formats`.
        """

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links),
                        key=lambda x: x[0])

        for root, _, files in _recursive_list(directory):
            for fname in sorted(files):
                if fname.lower().endswith(white_list_formats):
                    yield root, fname

    
    i = 0
    samples = 0
    classes = []
    filenames = []
    classes_list = []
    results = []
    pool = multiprocessing.pool.ThreadPool()

    for subdir in sorted(os.listdir(directoty)):
        if os.path.isdir(os.path.join(directoty, subdir)):
            classes.append(subdir)

    num_classes = len(classes)
    class_indices = dict(zip(classes, range(len(classes))))
    white_list_formats = ('png', 'jpg', 'jpeg')
    follow_links = False


    for dirpath in (os.path.join(directoty, subdir) for subdir in classes):
        results.append(
                    pool.apply_async(_list_valid_filenames_in_directory,
                                    (dirpath, white_list_formats, None,
                                    class_indices, follow_links)))

    for res in results:
        classes, filename = res.get()
        classes_list.append(classes)
        filenames += filename

    samples = len(filenames)
    classes = np.zeros((samples,), dtype='int32')

    for _class in classes_list:
        classes[i:i + len(_class)] = _class
        i += len(_class)

    print('Found %d images belonging to %d classes.' %
            (samples, num_classes))

    pool.close()
    pool.join()

    filepaths = [os.path.join(directoty, fname) for fname in filenames]

    return filepaths, classes
# added by tbtoan  ##end ######end ######end ######end ####
