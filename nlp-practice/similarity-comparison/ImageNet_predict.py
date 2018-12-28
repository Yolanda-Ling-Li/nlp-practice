from PIL import Image
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt

size = 224,224
def create_predict_data(images_dir):
    data = np.zeros((2,224,224,3),dtype = np.uint8)
    pair = []
    i = 0
    pathDir = os.listdir(images_dir)

    for allDir in pathDir:
        dirs = os.path.join('%s/%s' % (images_dir,allDir))
        image = Image.open(dirs)
        image = image.resize(size)
        data[i] = image
        i+=1
    data = data/225

    pair += [[data[0],data[1]]]
    pair = np.array(pair)

    return pair
