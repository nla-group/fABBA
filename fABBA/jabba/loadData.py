import os
import numpy as np

def loadData(name="Beef"):
    "The funcationality is to load example dataset for demonstration."
    current_dir, current_filename = os.path.split(__file__)
    
    if name == "Beef":
        train = np.load(os.path.join(current_dir, "data/beef_train.npy"))
        test = np.load(os.path.join(current_dir, "data/beef_test.npy"))
            
    return train, test
