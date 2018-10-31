from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from random import choice
import numpy as np
import random
import shutil
import MeCab
import math
import json
import gc
import os
import re


def data_initialization():
    """
       Copy useful data into folder.(./data/input)
       Returns:
        data_ready(int): label if data ready(1 if ready else 0)
    """
    thedata_dir = os.path.abspath(os.path.join(os.path.curdir, "thedata"))
    input_dir = os.path.abspath(os.path.join(os.path.curdir, "data//input//txt"))
    if not os.path.exists(thedata_dir):
        print("###!!Wrong: thedata folder is not detected!###")
        return 0
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    one_path = os.path.join(thedata_dir, "txt//gaigo_txt")
    for f in os.listdir(one_path):
        shutil.copyfile(one_path + "//" + f, input_dir + "//" + f)

    two_path = os.path.join(thedata_dir, "txt//internet_txt")
    for f in os.listdir(two_path):
        shutil.copyfile(two_path + "//" + f, input_dir + "//" + f)

    return 1


def data_input(input_dir):
    """
    Load document from files and make sentence segmentation
    Args:
        input_dir (path): relative path to the data folder
    Returns:
        all_texts(list): matrix generated after document sentence segmentation
    """
    file_list = []
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    for f in os.listdir(input_dir):
        file_list += [input_dir + '//' + f]

    all_texts = []
    for fi in file_list:
        the_text = []
        with open(fi, encoding='utf8') as file_input:
            the_document = ("".join(file_input.readlines())).replace("\n", "")
            the_document = the_document.replace("\n\r", "")
            the_text += the_document.split('。')
        all_texts += [the_text[:-1]]
    return all_texts


def data_cleaning(the_document):
    """
    Remove punctuation marks and special characters, and then do word segmentation
    Args:
        the_document (list): document after sentence segmentation
    Returns:
        the_document_3(list): pure document after word segmentation
    """
    mecab = MeCab.Tagger("-Owakati")
    the_document_1 = re.sub(u'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FA5\uF900-\uFA2D]', "", "".join(the_document))
    the_document_2 = mecab.parse(the_document_1).strip()
    the_document_3 = the_document_2.split(' ')
    return the_document_3


def data_plagiarism(the_document):
    """
    Simulate article plagiarism, remove some sentences and switch some sentences
    Args:
        the_document (list): original document after sentence segmentation
    Returns:
        the_document_3(list): pure document after word segmentation
    """
    some = random.sample(the_document, math.ceil(0.002 * len(the_document)))
    for x in some:
        the_document.remove(x)

    the = random.sample(range(0, len(the_document)), math.ceil(0.005 * len(the_document)))
    plag = list(the)
    plag_document = list(the_document)
    random.shuffle(the)
    for x, y in zip(plag, the):
        plag_document[x] = the_document[y]

    document_plagiarism = data_cleaning(plag_document)
    return document_plagiarism


def data_load(data_dir):
    """
    Load data from file system, remove special characters, do word segmentation and produce control group
    Args:
        data_dir (path): relative path to the data folder
    Returns:
        documents1(list): list of word list of original document
        documents2(list): list of word list of control group document
        is_similar(list): list containing labels if respective documents in document1 and document2 are same or not
                         (1 if same else 0)
    """
    documents1 = []
    documents2 = []
    is_similar = []
    input_dir = os.path.abspath(os.path.join(data_dir, "input//txt"))
    all_texts = data_input(input_dir)

    for document in all_texts:
        document1 = data_cleaning(document)
        similar_value = np.random.randint(0, 2)
        if similar_value == 0:
            other_texts = all_texts.copy()
            other_texts.remove(document)
            document2 = data_cleaning(choice(other_texts))
        elif similar_value == 1:
            document2 = data_plagiarism(document)
        else:
            print("###!!Warning: wrong similar_value.###")
            break
        documents1 += [document1]
        documents2 += [document2]
        is_similar += str(similar_value)
#        if len(document1)> 400:
#            num +=1
#    print("%d-%d"%(len(is_similar),num))
    return documents1, documents2, is_similar


def data_predict(data_dir):
    """
    Load data from file system, remove special characters, do word segmentation and produce control group
    Args:
        data_dir (path): relative path to the data folder
    Returns:
        documents1(list): list of word list of original document
        documents2(list): list of word list of control group document
    """
#   input_dir = os.path.abspath(os.path.join(data_dir, "predict"))
    all_texts = data_input(data_dir)

    for document in all_texts:
        data_cleaning(document)

    documents1 = list(all_texts[::2])
    documents2 = list(all_texts[1::2])

    return documents1, documents2


def parse_dataset(w2indx, the_documents):  
    """
    Map words of documents into index
    Args:
        w2indx(dict): dict containing words and their respective index
        the_documents(list): list of word list of document
    Returns:
        document_index(array):array of input features for train set or test set, which is built by word index 
    """
    data = []
    for document in the_documents:
        new_txt = []
        for word in document:
            if word in w2indx:
                new_txt.append(w2indx[word])
            else:
                new_txt.append(0)
        data.append(new_txt)
    document_index = np.array(data)
    return document_index


def create_dictionaries(model=None, combine=None):
    """
    Create dictionary through word2vector, and translate the word of documents into its index
    Args:
        model(model): trained word2vec model
        combine(list): list of documents after word segmentation
    Returns:
        w2indx(dict): dict containing words and their respective index
        w2vec(dict): dict containing words and their respective vectors
        combined(array):array of input features for train set or test set, which is built by word index
    """
    if (combine is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}
        combined = parse_dataset(w2indx=w2indx, the_documents=combine)
        return w2indx, w2vec, combined
    else:
        print('###!!Wrong:No data provided!###')


def train_word2vec(combine, vocab_dim, min_count, window_size, n_iterations, data_dir):
    """
    Train word2vector over training documents
    Args:
        combine (list): list of documents after word segmentation
        others: config of word2vec model
    Returns:
        n_symbols(int): size of vocabulary
        embedding_weights(dict): dict with word_index and vector mapping
        combined(list):list of input features for train set or test set, which is built by word index
    """
    model_dir = os.path.abspath(os.path.join(data_dir, "model//txt"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = Word2Vec(size=vocab_dim, min_count=min_count, window=window_size, iter=n_iterations, sorted_vocab=True)
    model.build_vocab(combine)  # To build a dictionary. You must take this steps, or you will get an error.
    model.train(combine, total_examples=model.corpus_count, epochs=model.iter)
    model.save(os.path.join(model_dir,  "Word2vec_model.pkl"))
    index_dict, word_vectors, combined = create_dictionaries(model=model, combine=combine)
    n_symbols = len(index_dict) + 1
    # The number of index of all words. The index of word which frequents are less than MIN_FREQUENT is 0, and so add 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # The vector of the word which index is 0 is whole 0.
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    f = open(os.path.join(model_dir, "w2index.txt"), 'w')   #save index_dict
    w2index = json.dumps(index_dict)
    f.write(str(w2index))
    f.close()
    return n_symbols, embedding_weights, combined


def create_train_dev_set(documents_pair, is_similar, max_len, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        documents_pair (list): list of tuple of sentences pairs
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_len (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data

    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features

        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    documents1 = [x[0] for x in documents_pair]
    documents2 = [x[1] for x in documents_pair]

    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(documents1, documents2)]

    train_padded_data_1 = pad_sequences(documents1, maxlen=max_len)
    train_padded_data_2 = pad_sequences(documents1, maxlen=max_len)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))  # reshuffing
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1  # Release memory
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]  #Divide dataset
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val


def create_test_data(index_dict, test_documents_pair, max_document_length):
    """
    Create training and validation dataset
    Args:
        index_dict(dict): dict containing words and their respective index
        test_documents_pair (list): list of tuple of documents pairs
        max_document_length (int): max document length of documents to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from test_documents1
        test_data_2 (list): list of input features for training set from test_documents2
    """
    test_documents1 = [data_cleaning(x[0]) for x in test_documents_pair]
    test_documents2 = [data_cleaning(x[1]) for x in test_documents_pair]
    test_documents_1 = parse_dataset(index_dict, test_documents1)
    test_documents_2 = parse_dataset(index_dict, test_documents2)


    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_documents_1, test_documents_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_documents_1, maxlen=max_document_length)
    test_data_2 = pad_sequences(test_documents_2, maxlen=max_document_length)

    return test_data_1, test_data_2, leaks_test
