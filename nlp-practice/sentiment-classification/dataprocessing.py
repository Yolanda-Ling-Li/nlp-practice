import re
import os
import urllib.request
import tarfile
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb


def rm_tags(text):
    #去标签
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype,data_dir):

    path = data_dir + "/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('Read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)
    all_labels = np.array(all_labels)
    all_texts = []

    for fi in file_list:
        the_text = []
        with open(fi, encoding='utf8') as file_input:
            the_line = rm_tags(" ".join(file_input.readlines())) #去标签
            the_string = re.sub(r'[^\w\s]','',the_line)   #去标点
            the_text += the_string.split(' ')
        all_texts += [the_text]

    return all_labels, all_texts


def w2v_load_data(data_dir):
    #生成训练集验证集并返回

    print("Loading data...")
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filepath = data_dir + "/aclImdb_v1.tar.gz"
    if not os.path.isfile(filepath):    #未下载则下载
        result = urllib.request.urlretrieve(url, filepath)
        print('downloaded:', result)

    if not os.path.exists(data_dir+"/aclImdb"): # 未解压则解压
        tfile = tarfile.open(data_dir+"/aclImdb_v1.tar.gz", 'r:gz')

    y_train, train_text = read_files("train",data_dir)   # 读文件
    y_test, test_text = read_files("test",data_dir)

    return train_text, y_train, test_text, y_test



def embeding_load_data(max_len,vocabulary_size=2000):
    #筛去过多词语，截断过长句子，生成训练集验证集并返回

    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary_size)

    print("Pad sequences (samples x time)...")  # 列表需要有一个定长，过长截断过短补零
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print("x trian:", x_train.shape)
    print("x test:", x_test.shape)

    return x_train, y_train, x_test, y_test
