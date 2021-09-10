import os
import matplotlib.pyplot as plt
import numpy as np


def load_images():
    images = list()
    folder = os.path.dirname(os.path.realpath(__file__))+'/samples/img'
    figs = os.listdir(folder)
    for filename in figs:
        img = plt.imread(os.path.join(folder,filename)) 
        if img is not None:
            images.append(img)
    return images