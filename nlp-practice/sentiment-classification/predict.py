import os
import numpy as np
from keras.models import load_model


def predict_save(data_dir, x_predict, y_predict):

    x_predict = np.toarray(x_predict)
    np.save(os.path.join(data_dir, 'x.npy'), x_predict)
    y_predict = np.toarray(y_predict)
    np.save(os.path.join(data_dir, 'y.npy'), y_predict)
    return


def predict_load(data_dir):

    x_predict = np.load(os.path.join(data_dir, 'x.npy'))
    y_predict = np.load(os.path.join(data_dir, 'y.npy'))
    return x_predict,y_predict



def predict(config):

    thestr =  'Model'+ str(config.model-2) +'.hdf5'
    model = load_model(os.path.join(config.data_dir,thestr))
    x_predict, y_predict = predict_load(config.data_dir)
    result = model.predict_classes(x_predict)
    print(result)


