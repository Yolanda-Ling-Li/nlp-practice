import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras import backend as K

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    return (shapes[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def create_resnet50(input_shape):
    inputs = Input(shape=input_shape)
    resnet50 = keras.applications.resnet50.ResNet50(include_top=False,pooling='avg',weights='imagenet')(inputs)
    dense = Dense(128,activation='relu')(resnet50)

    return Model(inputs,dense)
def create_siamese(input_shape):
    resnet50 = create_resnet50(input_shape)
    
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    processed_1 = resnet50(input_1)
    processed_2 = resnet50(input_2)
    
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_1,processed_2])
    model = Model([input_1,input_2],distance)

    return model