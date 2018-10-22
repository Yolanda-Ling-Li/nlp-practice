from model import SiameseBiLSTM
from input_handler import data_initialization, data_load, train_word2vec
from config import siamese_config
from keras.models import load_model


class ConfiGuration(object):
    """Dump stuff here"""


config = ConfiGuration()
config.max_len = siamese_config['MAX_DOCUMENT_LENGTH']
config.number_lstm_units = siamese_config['NUMBER_LSTM']
config.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
config.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
config.activation_function = siamese_config['ACTIVATION_FUNCTION']
config.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
config.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

config.min_count = siamese_config['MIN_FREQUENT']
config.vocab_dim = siamese_config['VOCABULARY_DIM']
config.batch_size = siamese_config['BATCH_SIZE']
config.num_epoch = siamese_config['NUM_EPOCH']
config.window_size = siamese_config['WINDOW_SIZE']
config.n_iterations = siamese_config['N_ITERATIONS']
config.data_dir = siamese_config['DATA_DIR']



def data_preperation():
    """
    Data preperation and word embedding
    Returns:
        documents_pair (list): list of tuple of sentence pairs
        is_similar (list): target value 1 if same sentences pair are similar otherwise 0
        embedding_meta_data (dict): dict containing vocabulary size and word embedding matrix
    """
    # Data Preperation
    documents1, documents2, is_similar = data_load(config.data_dir)

    # Word Embedding
    print("Creating word embedding meta data for word embeddin...")
    vocab_size, embedding_matrix, combine = train_word2vec(
        documents1 + documents2, config.vocab_dim,
        config.min_count, config.window_size,
        config.n_iterations, config.data_dir)
    del documents1
    del documents2

    embedding_meta_data = {
        'vocab_size': vocab_size,
        'embedding_matrix': embedding_matrix
    }

    print("Creating document pairs...")
    documents1 = combine[0:len(combine)//2]
    documents2 = combine[len(combine)//2:]
    documents_pair = [(x1, x2) for x1, x2 in zip(documents1, documents2)]
    del documents1
    del documents2

    return documents_pair, is_similar, embedding_meta_data


def train(documents_pair, is_similar, embedding_meta_data):
    """
    Train Siamese network
    Args:
        documents_pair (list): list of tuple of sentence pairs
        is_similar (list): target value 1 if same sentences pair are similar otherwise 0
        embedding_meta_data (dict): dict containing vocabulary size and word embedding matrix

    Returns:
        return (best_model_path): path of best model
        test_data(list): list of input features for test from ）
        test_labels(array): array containing similarity score for test data
    """
    print("Train the model SiameseBiLSTM...")
    siamese = SiameseBiLSTM(
        config.vocab_dim, config.max_len, config.number_lstm_units, config.number_dense_units, config.rate_drop_lstm,
        config.rate_drop_dense, config.activation_function, config.validation_split_ratio, config.num_epoch,
        config.batch_size)

    best_model_path, test_data, test_labels = siamese.train_model(
        documents_pair, is_similar, embedding_meta_data, data_dir=config.data_dir)

    return best_model_path, test_data, test_labels


def evaluate(best_model_path, test_data, test_labels):
    """
    Evaluate Siamese model
    Args:
        best_model_path(atr): path of best model
        test_data(list): list of input features for test from ）
        test_labels(array): array containing similarity score for test data
    """
    print("Evaluate the model SiameseBiLSTM...")
    model = load_model(best_model_path)
    score, acc = model.evaluate(test_data, test_labels, batch_size=config.batch_size)
    print("Embedding_cnn_lstm_model_model:the test data score is %f" % score)
    print("Embedding_cnn_lstm_model_model:the test data accuracy is %f" % acc)


def main():
    data_ready = data_initialization()
    if data_ready == 0:
        return
    else:
        documents_pair, is_similar, embedding_meta_data = data_preperation()
        best_model_path, test_data, test_labels = train(documents_pair, is_similar, embedding_meta_data)
        evaluate(best_model_path, test_data, test_labels)


# When the.Py file is run directly, it will be run, and the.Py file will not be run when it is imported in module form.
if __name__ == "__main__":
    main()
