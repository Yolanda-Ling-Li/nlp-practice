## Text Similarity Using Siamese Deep Neural Network

Siamese neural network is a class of **neural network architectures that contain two or more** **identical** **subnetworks**. *identical* here means they have the same configuration with the same parameters 
and weights. Parameter updating is mirrored across both subnetworks.

It is a keras based implementation of deep siamese Bidirectional LSTM network to capture phrase/sentence similarity using word embeddings.

Below is the architecture description for the same.

![rch_imag](images/arch_image.png)



## Usage

#### Config

```python
from model import SiameseBiLSTM
from input_handler import data_load, train_word2vec, create_test_data
from config import siamese_config
from keras.models import load_model
from operator import itemgetter

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
```



#### Training

```python
############ Data Preperation ##########


print("Loading data...")
documents1, documents2, is_similar = data_load(config.data_dir, 'initialize')


######## Word Embedding ############


print("Creating word embedding meta data for word embeddin...")
vocab_size, embedding_matrix, combine, index_dict = train_word2vec(
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


######## Training ########


print("Train the model SiameseBiLSTM...")
siamese = SiameseBiLSTM(
	config.vocab_dim, config.max_len, config.number_lstm_units, config.number_dense_units, config.rate_drop_lstm,
	config.rate_drop_dense, config.activation_function, config.validation_split_ratio, config.num_epoch, config.batch_size)

best_model_path, test_data, test_labels = siamese.train_model(
	documents_pair, is_similar, embedding_meta_data, data_dir=config.data_dir)
```

#### Evaluating

```python
from keras.models import load_model

print("Evaluate the model SiameseBiLSTM...")
model = load_model(best_model_path)
score, acc = model.evaluate(test_data, test_labels, batch_size=config.batch_size)
print("Embedding_cnn_lstm_model_model:the test data score is %f" % score)
print("Embedding_cnn_lstm_model_model:the test data accuracy is %f" % acc)
```

#### Predicting

```python
print("Predict documents through model SiameseBiLSTM...")
predict_documents1, predict_documents2, predict_is_similar = data_load(config.data_dir, 'predict')
predict_document_pairs = [(x1, x2) for x1, x2 in zip(predict_documents1, predict_documents2)]
predict_data_x1, predict_data_x2, leaks_predict = create_test_data(index_dict, predict_document_pairs, config.max_len)

preds = list(model.predict([predict_data_x1, predict_data_x2, leaks_predict], verbose=1).ravel())
results = [(c, d, "".join(a), "".join(b)) for (a, b), c, d in zip(predict_document_pairs, preds, predict_is_similar)]
results.sort(key=itemgetter(0), reverse=True)

for result in results:
    print(str(result))
```

### References:

1. [Siamese Recurrent Architectures for Learning Sentence Similarity (2016)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)
2. Inspired from Tensorflow Implementation of  https://github.com/dhwajraj/deep-siamese-text-similarity
3. [Lstm Siamese Text Similarity from Github](https://github.com/amansrivastava17/lstm-siamese-text-similarity)
4. [Mecab Japanese word segmentation system](http://taku910.github.io/mecab/)
