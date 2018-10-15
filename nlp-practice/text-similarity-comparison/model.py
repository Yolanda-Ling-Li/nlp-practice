# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
import os

from input_handler import create_train_dev_set


class SiameseBiLSTM:
    def __init__(self, embedding_dim, max_document_length, number_lstm, number_dense, rate_drop_lstm,
                 rate_drop_dense, hidden_activation, validation_split_ratio, num_epoch, batch_size):
        self.embedding_dim = embedding_dim
        self.max_document_length = max_document_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def train_model(self, documents_pair, is_similar, embedding_meta_data, data_dir):
        """
        Train Siamese network to find similarity between sentences in `documents_pair`
            Steps Involved:
                1. Pass the each from documents_pairs  to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            documents_pair (list): list of tuple of sentence pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing vocabulary size and word embedding matrix
            data_dir (str): working directory for where to save data

        Returns:
            return (best_model_path): path of best model
            test_data(list): list of input features for test from ）
            test_labels(array): array containing similarity score for test data
        """
        vocab_size, embedding_matrix = embedding_meta_data['vocab_size'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_labels, leaks_train, \
        test_data_x1, test_data_x2, test_labels, leaks_test = create_train_dev_set(
            documents_pair, is_similar, self.max_document_length, self.validation_split_ratio)

        if train_data_x1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        # Creating word embedding layer
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_document_length, trainable=False)

        # Creating LSTM Encoder
        lstm_layer = Bidirectional(
            LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))    #双向预测

        # Creating LSTM Encoder layer for First Sentence
        document_1_input = Input(shape=(self.max_document_length,), dtype='int32')
        embedded_documents_1 = embedding_layer(document_1_input)
        x1 = lstm_layer(embedded_documents_1)

        # Creating LSTM Encoder layer for Second Sentence
        document_2_input = Input(shape=(self.max_document_length,), dtype='int32')
        embedded_documents_2 = embedding_layer(document_2_input)
        x2 = lstm_layer(embedded_documents_2)

        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))
        leaks_dense = Dense(int(self.number_dense_units/2), activation=self.activation_function)(leaks_input)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2, leaks_dense])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[document_1_input, document_2_input, leaks_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])


        model_dir = os.path.abspath(os.path.join(data_dir, "model"))
        logs_dir = os.path.abspath(os.path.join(data_dir, "logs"))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint(
            os.path.join(model_dir, "Model.hdf5"),  monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False)
        tensorboard = TensorBoard(log_dir=logs_dir)  # 存储loss，acc曲线文件的路径，可以用命令行+6006打开

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels, epochs=self.num_epoch,
                  batch_size=self.batch_size, validation_split=self.validation_split_ratio, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        test_data = [test_data_x1, test_data_x2, leaks_test]

        return os.path.join(model_dir, 'Model.hdf5'), test_data, test_labels


    def update_model(self, data_dir, new_documents_pair, is_similar):
        """
        Update trained siamese model for given new sentences pairs 
            Steps Involved:
                1. Pass the each from sentences from new_documents_pair to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            data_dir (str): dara path of siamese model
            new_documents_pair (list): list of tuple of new sentences pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0

        Returns:
            return (model_path):  path of best model
        """
        model_path = os.path.abspath(os.path.join(data_dir, "model//Model.hdf5"))
        logs_path = os.path.abspath(os.path.join(data_dir, "logs"))
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(new_documents_pair,
                                                                               is_similar, self.max_document_length,
                                                                               self.validation_split_ratio)

        model = load_model(model_path)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint(
            os.path.join(model_path), monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False)
        tensorboard = TensorBoard(log_dir=logs_path)  # 存储loss，acc曲线文件的路径，可以用命令行+6006打开

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=3, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return model_path
