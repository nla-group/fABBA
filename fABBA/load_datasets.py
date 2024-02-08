# Copyright (c) 2021, 
# Authors: Stefan Güttel, Xinye Chen

# All rights reserved.


# load demo image samples

import os
import requests
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
import sys



def loadData(name="Beef"):
    """Load the example data. 
    
    Parameters
    ----------
    name : str
        The dataset name, current support 'AtrialFibrillation', 'BasicMotions', 'Beef', 
              'CharacterTrajectories', 'LSST', 'Epilepsy', 
              'NATOPS', 'UWaveGestureLibrary', 'JapaneseVowels'. 
        For more datasets, we refer the users to https://www.timeseriesclassification.com/ 
        or https://archive.ics.uci.edu/datasets. 
        
        
    Returns
    -------
    train, test : numpy.ndarray
        Return data for train and test, respectively.
        
    """
    current_dir, current_filename = os.path.split(__file__)
    
    if name == "Beef":
        train = np.load(os.path.join(current_dir, "jabba/data/beef_train.npy"))
        test = np.load(os.path.join(current_dir, "jabba/data/beef_test.npy"))
            
    elif name in ['AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'LSST',
             'Epilepsy', 'NATOPS', 'UWaveGestureLibrary', 'JapaneseVowels', 
            ]:
        train = preprocess(arff.loadarff(os.path.join(current_dir, "jabba/data/"+name+'_TRAIN.arff')))
        test = preprocess(arff.loadarff(os.path.join(current_dir, "jabba/data/"+name+'_TEST.arff')))
                
    return train, test


        
def load_synthetic_sample(length=1000, freq=20):
    try:
        assert(type(length)==int and type(freq)==int)
    except:
        print("Please ensure both parameters are integer.")
    # generate synthetic sine time series
    sample = np.zeros(length)
    j = 0
    for i in np.arange(0, length, 1)*(1/freq):
        sample[j] = np.sin(i)
        j = j + 1
    return sample


def load_images():
    """Load the example images (a total of 24 images for test). 
    
    Parameters
    ----------
    No parameter input.
        
    Returns
    -------
    images : list
        Return list storing image data.
    
    
    """
    samples_list = [ 'n02086646_2069.jpg',
                     'n02088094_3593.jpg',
                     'n02089078_2021.jpg',
                     'n02090379_2083.jpg',
                     'n02091134_14363.jpg',
                     'n02091134_17788.jpg',
                     'n02093428_17280.jpg',
                     'n02093428_1746.jpg',
                     'n02093428_1767.jpg',
                     'n02093428_19443.jpg',
                     'n02093859_2579.jpg',
                     'n02096585_2947.jpg',
                     'n02099601_5857.jpg',
                     'n02101556_4241.jpg',
                     'n02101556_8093.jpg',
                     'n02101556_8168.jpg',
                     'n02107312_5862.jpg',
                     'n02107683_5115.jpg',
                     'n02109525_6019.jpg',
                     'n02110063_1034.jpg',
                     'n02110185_3406.jpg',
                     'n02112706_637.jpg',
                     'n02113023_1825.jpg',
                     'n02115913_4117.jpg'
                   ]
    
    store_dir = "samples/img"
    
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
        sys.stdout.write("Downloading: [ %s" % ("" * len(samples_list)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (len(samples_list)+1)) 

        for img in samples_list:
            get_img(img, store_dir)
            sys.stdout.write("=")
            sys.stdout.flush()
        sys.stdout.write("]\n") 
        
    elif len(os.listdir(store_dir)) == 0:
        sys.stdout.write("Progress: [ %s" % ("" * len(samples_list)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (len(samples_list)+1)) 

        for img in samples_list:
            get_img(img, store_dir)
            sys.stdout.write("=")
            sys.stdout.flush()
        sys.stdout.write("]\n") 
        
    images = list()
    figs = os.listdir(store_dir)
    for filename in figs:
        img = plt.imread(os.path.join(store_dir,filename)) 
        if img is not None:
            images.append(img)
    return images




def get_img(file, store_dir):
    url_parent = "https://raw.githubusercontent.com/nla-group/fABBA/master/samples/img/"
    img_data = requests.get(url_parent + file).content
    with open(store_dir + "/" + file, 'wb') as handler:
        handler.write(img_data)
        


def preprocess(data):
    time_series = list()
    for ii in data[0]:
        database = list()
        for i in ii[0]:
            database.append(list(i))
        time_series.append(database)
    return np.nan_to_num(np.array(time_series))


