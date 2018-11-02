from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D, UpSampling2D, Reshape
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import Model
import numpy as np

import mycoco
mycoco.setmode('train')

## Model
inputlayer = Input(shape=(200,200,3))

encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(inputlayer)
encoder = MaxPooling2D(pool_size=(2, 2))(encoder)
encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder)
encoder = MaxPooling2D(pool_size=(2, 2))(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
encoder = MaxPooling2D(pool_size=(2, 2))(encoder)

flatten = Flatten(name='encoded_flat')(encoder)
flatten = Reshape((25, 25, 8))(flatten)

decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(flatten)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder)

autoencoder = Model(inputlayer, decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
autoencoder.summary()

## Use same network but stop at flattened encoded layer to extract
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_flat').output)

## 

def only_img_iter(img_iter):
    for i in img_iter:
        yield (i[0], i[0])
        
# Pretty hacky to train on all the images
all_ids = mycoco.query([['']])
all_img = mycoco.iter_images(all_ids, [None])
        
all_only_img = only_img_iter(all_img)

csv_logger = CSVLogger('./autoencoder.csv', append=True, separator=',')
filepath="/scratch/gussteen/autoencoder/autoencoder.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
autoencoder.fit_generator(all_only_img, steps_per_epoch=10000, epochs=50, callbacks=[checkpoint, csv_logger])
autoencoder.save('./autoencoder.model.hdf5')