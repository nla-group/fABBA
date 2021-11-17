#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Copyright (c) 2021, Stefan Güttel, Xinye Chen
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# load demo image samples

import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_img(file, store_dir):
    url_parent = "https://raw.githubusercontent.com/nla-group/fABBA/master/fABBA/samples/img/"
    img_data = requests.get(url_parent + file).content
    with open(store_dir + "/" + file, 'wb') as handler:
        handler.write(img_data)
        
        
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
