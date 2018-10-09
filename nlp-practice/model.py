import os
import sys
import importlib
importlib.reload(sys)
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import Embedding, LSTM, Dropout, Reshape
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary




def embedding_cnn_lstm_model(config):

    # embedding layer
    model = Sequential()
    model.add(Embedding(input_dim=config.vocabulary_size, output_dim=config.vocabulary_dim, mask_zero=True, input_length=config.max_len,name="Embedding"))
    model.add(Reshape((-1, config.max_len, config.vocabulary_dim), name="Reshape_1"))
    model.add(Conv2D(filters=256, kernel_size=(1, 3), strides=1, padding='valid', activation='relu', name="Conv2D"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=1, padding="valid", name="MaxPooling2D"))
    model.add(Reshape((1, -1), name="Reshape_2"))
    model.add(LSTM(units=config.vocabulary_dim, activation='sigmoid', recurrent_activation='hard_sigmoid', name="LSTM"))
    model.add(Dropout(config.keep_prob, name="Dropout"))
    model.add(Dense(1, activation='sigmoid', name="Dense"))

    return model



#创建词语字典，并返回每个词语的索引，词向量，以及每个文本所对应的词语索引
def create_dictionaries(vocab_dim,max_len,model=None, combine=None):

    if (combine is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True) #把该查询文档（词集）更改为（词袋模型）即：字典格式，key是单词，value是该单词在该文档中出现次数。
        w2indx = {v: k+1 for k, v in gensim_dict.items()} #所有的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()} #所有词语的词向量

        def parse_dataset(thecombine):    #Words become index
            data=[]
            for sentence in thecombine:
                new_txt = []
                for word in sentence:
                    if word in w2indx:
                        new_txt.append(w2indx[word])
                    else:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined=parse_dataset(combine) #每个句子所含词语对应的词语索引
        combined = sequence.pad_sequences(combined, maxlen=max_len)  # maxlen设置最大的序列长度，长于该长度的序列将会截短，短于该长度的序列将会填充
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


#创建词语字典，并返回词典大小，权重矩阵以及每个词语索引矩阵
def word2vec_model(config,combine):

    model = Word2Vec(size=config.vocabulary_dim,min_count=15,window=config.window_size,workers=config.cpu_count, iter=config.n_iterations,sorted_vocab=True)
    model.build_vocab(combine) #建立词典，必须步骤，不然会报错
    model.train(combine,total_examples=model.corpus_count,epochs=model.iter) #训练词向量模型
    model.save(os.path.join(config.data_dir,  "Word2vec_model.pkl")) #保存词向量模型
    index_dict, word_vectors, combined = create_dictionaries(vocab_dim=config.vocabulary_dim,max_len=config.max_len,model=model,combine=combine)
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, config.vocabulary_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    return n_symbols, embedding_weights, combined


def w2v_cnn_lstm_model(n_symbols,embedding_weights,config):

    # embedding layer
    model = Sequential()
    model.add(Embedding(output_dim=config.vocabulary_dim, input_dim=n_symbols,
                        weights=[embedding_weights],input_length=config.max_len,name="Word2vec"))  # Adding Input Length Embedding层只能作为模型的第一层
    model.add(Reshape((-1, config.max_len, config.vocabulary_dim), name="Reshape_1"))
    model.add(Conv2D(filters=256, kernel_size=(1, 3), strides=1, padding='valid', activation='relu', name="Conv2D"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=1, padding="valid", name="MaxPooling2D"))
    model.add(Reshape((1, -1), name="Reshape_2"))
    model.add(LSTM(units=config.vocabulary_dim, activation='sigmoid', recurrent_activation='hard_sigmoid', name="LSTM"))
    model.add(Dropout(config.keep_prob, name="Dropout"))
    model.add(Dense(1, activation='sigmoid', name="Dense"))

    return model





















