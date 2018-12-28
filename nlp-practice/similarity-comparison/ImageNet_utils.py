from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.applications.resnet50 import preprocess_input
train_path = './all_data/train'
test_path = './all_data/test'
size = 224,224

#预处理数据
def create_data(path,count,count_by_kinds):
    data = np.zeros((count,224,224,3),dtype = np.uint8)
    kind = np.zeros(count,dtype = np.uint8)
    i = 0
    pathDir = os.listdir(path)

    for allDir in pathDir:
        dirs = os.path.join('%s/%s' % (path,allDir))
        image = Image.open(dirs)
        image = image.resize(size)
        if random.randint(0,5)==0:
            image = pepper(image,0.1)
        data[i] = image
        kind[i] = i//count_by_kinds
        i+=1
    data = data/225
    return data,kind
#生成训练数据
def create_pairs(data,kind,count_by_kinds):
    pairs = []
    labels = []
    c = count_by_kinds
    for i in range(data.shape[0]):
        i1 = i
        i2 = random.randint(kind[i]*c,kind[i]*c+c-1)
        pairs += [[data[i1],data[i2]]]
        inc =(random.randint(1,5)+kind[i])%5 
        i2 = random.randint(inc*c,inc*c+c-1)

        pairs += [[data[i1],data[i2]]]
        labels +=[0,1]

    return np.array(pairs),np.array(labels)
#预测准确率
def compute_accuracy(y_true, y_pred):
    pred = (np.ravel(y_pred)+0.5)//1
    j=0
    for i in range(len(y_pred)):
        if y_true[i]==pred[i]:
            j+=1
    return j/len(y_pred)

#添加噪声
def pepper(image,SNR):
    row = 224
    col = 224

    image = np.array(image)
    pepper_size = int(row*col*SNR)

    for i in range(pepper_size):
        noise_row = random.randint(0,row-1)
        noise_col = random.randint(0,col-1)

        if random.randint(0,1)==0:
            image[noise_row,noise_col,0]=0
        else:
            image[noise_row,noise_col,0]=255
    
    image = Image.fromarray(np.uint8(image))
    return image


train_data,train_kind = create_data(train_path,400,80)
train_pairs,train_labels = create_pairs(train_data,train_kind,80) 

input_shape = train_data.shape[1:]