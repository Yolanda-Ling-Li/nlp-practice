from input_handler import create_test_data, data_predict
from config import siamese_config
from keras.models import load_model
from operator import itemgetter
import json
import os


class ConfiGuration(object):
    """Dump stuff here"""


config = ConfiGuration()
config.max_len = siamese_config['MAX_DOCUMENT_LENGTH']
config.data_dir = siamese_config['DATA_DIR']

model = load_model(os.path.join(config.data_dir, "model//Model.hdf5"))
f = open(os.path.join(config.data_dir, "model//w2index.txt"), 'r')
w2index = f.read()
index_dict = json.loads(w2index)
f.close()

print("Predict documents through model SiameseBiLSTM...")
predict_documents1, predict_documents2 = data_predict(config.data_dir)
predict_document_pairs = [(x1, x2) for x1, x2 in zip(predict_documents1, predict_documents2)]
predict_data_x1, predict_data_x2, leaks_predict = create_test_data(index_dict, predict_document_pairs, config.max_len)

preds = list(model.predict([predict_data_x1, predict_data_x2, leaks_predict], verbose=1).ravel())
results = [(c, "".join(a), "".join(b)) for (a, b), c in zip(predict_document_pairs, preds)]
results.sort(key=itemgetter(0), reverse=True)

for result in results:
    print(str(result))
