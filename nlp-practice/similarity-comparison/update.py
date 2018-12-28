from model import SiameseBiLSTM
from input_handler import data_cleaning, data_predict, create_dictionaries
from config import siamese_config
from gensim.models import Word2Vec
import os


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



def update_load(data_dir):
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
    same_dir = os.path.abspath(os.path.join(data_dir, "update//txt//same"))
    dif_dir = os.path.abspath(os.path.join(data_dir, "update//txt//different"))

    same_documents1, same_documents2 = data_predict(same_dir)
    dif_documents1, dif_documents2 = data_predict(dif_dir)
    for x, y in zip(same_documents1, same_documents2):
        documents1 += [data_cleaning(x)]
        documents2 += [data_cleaning(y)]
        is_similar += str(1)
    for x, y in zip(dif_documents1, dif_documents2):
        documents1 += [data_cleaning(x)]
        documents2 += [data_cleaning(y)]
        is_similar += str(0)

    return documents1, documents2, is_similar


def update_preperation():
    """
    Data preperation and word embedding
    Returns:
        documents_pair (list): list of tuple of sentence pairs
        is_similar (list): target value 1 if same sentences pair are similar otherwise 0
        embedding_meta_data (dict): dict containing vocabulary size and word embedding matrix
    """
    # Data Preperation
    documents1, documents2, is_similar = update_load(config.data_dir)

    # Load Word Embedding
    model = Word2Vec.load(os.path.join(config.data_dir, "model//txt//Word2vec_model.pkl"))
    index_dict, word_vectors, combine = create_dictionaries(model = model, combine = documents1 + documents2)

    del documents1
    del documents2

    print("Creating document pairs...")
    documents1 = combine[0:len(combine)//2]
    documents2 = combine[len(combine)//2:]
    documents_pair = [(x1, x2) for x1, x2 in zip(documents1, documents2)]
    del documents1
    del documents2

    return documents_pair, is_similar


def update(documents_pair, is_similar):
    """
    Train Siamese network
    Args:
        documents_pair (list): list of tuple of sentence pairs
        is_similar (list): target value 1 if same sentences pair are similar otherwise 0

    Returns:
        return (best_model_path): path of best model
        test_data(list): list of input features for test from ）
        test_labels(array): array containing similarity score for test data
    """
    print("Update the model SiameseBiLSTM...")
    siamese = SiameseBiLSTM(
        config.vocab_dim, config.max_len, config.number_lstm_units, config.number_dense_units, config.rate_drop_lstm,
        config.rate_drop_dense, config.activation_function, config.validation_split_ratio, config.num_epoch,
        config.batch_size)

    siamese.update_model(documents_pair, is_similar, data_dir=config.data_dir)


def main():
    documents_pair, is_similar = update_preperation()
    update(documents_pair, is_similar)


# When the.Py file is run directly, it will be run, and the.Py file will not be run when it is imported in module form.
if __name__ == "__main__":
    main()
