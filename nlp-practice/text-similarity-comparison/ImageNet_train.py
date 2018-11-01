import ImageNet_utils as inu
import ImageNet_model as inm
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os

rms = RMSprop(lr=0.01,decay=0.5)
adam = Adam(lr=0.005,decay=0.5)
nadam = Nadam()
model = inm.create_siamese(inu.input_shape)

model.compile(loss=inm.contrastive_loss,optimizer=adam,metrics=['accuracy'])
#将效果最好的epoch保存为模型
modelspath="./models/model_best_weights.hdf5"
logspath="./logs"
tensorboard = TensorBoard(log_dir=logspath)
checkpoint = ModelCheckpoint(modelspath, monitor='val_loss',verbose=1, 
                            save_best_only=True,save_weights_only=True)

model.fit([inu.train_pairs[:,0],inu.train_pairs[:,1]],inu.train_labels,
            batch_size=16,
            epochs=50,
            validation_split=0.2,
            callbacks=[checkpoint,tensorboard])

model.save('./models/last_model.hdf5')