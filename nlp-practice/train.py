import sys
import getopt
import numpy as np
import multiprocessing
from keras.callbacks import ModelCheckpoint,TensorBoard
from sklearn.model_selection import train_test_split
from model import *
from dataprocessing import *
from predict import *


class Config():

    model = 1 #embeding--1 w2v--2
    vocabulary_size = 2000  #词典大小
    max_len = 512   #句长
    vocabulary_dim = 128    #词向量维数
    batch_size = 48
    keep_prob = 0.5 #防过拟合
    num_epoch = 30  #30
    window_size = 8  # 窗口大小
    n_iterations = 5 # 迭代次数，默认为5 #定义词向量模型
    cpu_count = multiprocessing.cpu_count()
    data_dir = os.path.abspath(os.path.join(os.path.curdir,"data"))
    log_dir_file = os.path.abspath(os.path.join(os.path.curdir,"data/logs"))

    def __init__(self, value=1):
        self.model = value


def train_embedding(config):

    x_train, y_train, x_test, y_test = embeding_load_data(max_len=config.max_len,vocabulary_size=config.vocabulary_size)

    print("Setting up Arrays for Keras Embedding Layer...")
    model = Embedding_CNN_LSTM_Model(config=config)

    print('Compiling the Model Embedding+CNN+Reshape+LSTM...')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Train the Model Embedding+CNN+Reshape+LSTM...')
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
    cbs = [ModelCheckpoint(os.path.join(config.data_dir,  'Model1.hdf5'),   #存储模型参数的路径
                           monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False), #只存储效果最好loss最小时的模型参数
           TensorBoard(log_dir=config.log_dir_file)]   #存储loss，acc曲线文件的路径，可以用命令行+6006打开
    model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.num_epoch, validation_data=(x_test, y_test), callbacks=cbs)

    print("Evaluate...")
    score, acc = model.evaluate(x_test, y_test, batch_size=config.batch_size)
    print("Embedding_CNN_LSTM_Model_model:the test data score is %f" % (score))
    print("Embedding_CNN_LSTM_Model_model:the test data accuracy is %f" % (acc))
    x_predict, y_predict, _, _ = train_test_split(x_test, y_test, test_size=0.008, random_state=0)

    return  x_predict, y_predict


    '''
    print('Giving the internal output...')
    from keras.models import Model
    dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Flatten').output)
    dense1_output = dense1_layer_model.predict(x_test)
    print(dense1_output.shape)
    print(str(dense1_output))
    '''


def train_w2v(config):
    train_text, y_train, test_text, y_test = w2v_load_data(data_dir=config.data_dir)
    combine = train_text+test_text

    print('Training a Word2vec model...')
    n_symbols, embedding_weights, combine = Word2Vec_Model(combine=combine,config=config)

    x_train = combine[0:len(combine)//2]
    x_test = combine[len(combine)//2:]

    print('Setting up Arrays for Keras Embedding Layer...')
    model = W2v_CNN_LSTM_Model(n_symbols=n_symbols, embedding_weights=embedding_weights,config=config)

    print('Compiling the Model Word2vec+CNN+Reshape+LSTM...')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Train the Model Word2vec+CNN+Reshape+LSTM...')
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
    cbs = [ModelCheckpoint(os.path.join(config.data_dir, 'Model2.hdf5'),  # 存储模型参数的路径
                           monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False), # 只存储效果最好loss最小时的模型参数
           TensorBoard(log_dir=config.log_dir_file)]  # 存储loss，acc曲线文件的路径，可以用命令行+6006打开
    model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.num_epoch, validation_data=(x_test, y_test),callbacks=cbs)

    print("Evaluate...")
    score, acc = model.evaluate(x_test, y_test, batch_size=config.batch_size)
    print("Embedding_CNN_LSTM_Model_model:the test data score is %f" % (score))
    print("Embedding_CNN_LSTM_Model_model:the test data accuracy is %f" % (acc))
    x_predict, y_predict, _, _ = train_test_split(x_test, y_test, test_size=0.008, random_state=0)

    return x_predict, y_predict


def main():
    opts, args = getopt.getopt(sys.argv[1:], '-m', ['model='])  #1-embedding 2-w2v
    num=int(args[0])

    print("Initialize...")
    config = Config(num)


    if config.model == 1:
        x_predict, y_predict = train_embedding(config)
        predict_save(config.data_dir,x_predict, y_predict)
    elif config.model == 2:
        x_predict, y_predict = train_w2v(config)
        predict_save(config.data_dir,x_predict, y_predict)
    elif config.model == 3 or config.model == 4 :
       predict(config)
    else:
        print("Nonstandard input:"+config.model )



#当.py文件被直接运行时将被运行，当.py文件以模块形式被导入时不被运行。
if __name__ == "__main__":
    main()



