import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import Activation, Input
from keras.utils.image_utils import ResizeMethod
from keras_preprocessing.image import load_img, img_to_array
from tensorflow import keras

from keras.models import Model, load_model
from keras.layers.merging import concatenate
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from skimage.io import imshow

import matplotlib.pyplot as plt

input_dir = 'Dataset/images'
target_dir = 'Dataset/masks'

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png")
    ]
)

IMG_WIDTH_HEIGHT = 512
IMG_CHANNELS = 3
classes = 5

X = np.zeros((len(input_img_paths), IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, 3), dtype=np.float32)
Y = np.zeros((len(input_img_paths), IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, 1), dtype=np.uint8)

for i in range(len(input_img_paths)):
    img_path = input_img_paths[i]
    img = load_img(img_path)
    img = img_to_array(img)
    img = tf.image.resize_with_pad(img, IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, method=ResizeMethod.BILINEAR,
                                   antialias=False).numpy()
    X[i] = img.astype('float32') / 255.0

    mask_path = target_img_paths[i]
    mask = load_img(mask_path, color_mode='grayscale')
    mask = img_to_array(mask)
    mask = tf.image.resize_with_pad(mask, IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, method=ResizeMethod.BILINEAR,
                                    antialias=False).numpy()
    Y[i] = mask

train_test_split = int(len(input_img_paths) * 0.8)
X_train = X
Y_train = Y

X_test = X
Y_test = Y

def convolutional_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters,
                  kernel_size=3,
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    conv = Conv2D(n_filters,
                  kernel_size=3,
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)


    else:
        next_layer = conv

    # conv = BatchNormalization()(conv)
    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(
        n_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='same')(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                  kernel_size=3,
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(merge)
    conv = Conv2D(n_filters,
                  kernel_size=3,
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(conv)

    return conv


def unet_model(input_size=(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, IMG_CHANNELS), n_filters=32, n_classes=5):
    inputs = Input(input_size)

    # contracting path
    cblock1 = convolutional_block(inputs, n_filters)
    cblock2 = convolutional_block(cblock1[0], 2 * n_filters)
    cblock3 = convolutional_block(cblock2[0], 4 * n_filters)
    cblock4 = convolutional_block(cblock3[0], 8 * n_filters, dropout_prob=0.2)
    cblock5 = convolutional_block(cblock4[0], 16 * n_filters, dropout_prob=0.2, max_pooling=None)

    # expanding path
    ublock6 = upsampling_block(cblock5[0], cblock4[1], 8 * n_filters)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_classes,
                   1,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    # conv10 = Conv2D(n_classes, kernel_size=1, padding='same', activation = 'softmax')(conv9)
    conv10 = Activation('softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


unet = unet_model((IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, IMG_CHANNELS), n_classes=5)

EPOCHS = 20

unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

earlystopper = EarlyStopping(patience=5, verbose=1)
model_history = unet.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=EPOCHS, callbacks=[earlystopper])

predictions = unet.predict(X_test)
i = 1
imshow(X_test[i])
plt.show()
imshow(np.squeeze(Y_test[i]))
plt.show()
pred = predictions[i]
pred = pred.argmax(axis=2)
colour_mappings = {
    'wall': (255, 255, 255),  # white
    'insufficient': (255, 0, 0),  # red
    'sufficient': (0, 255, 0),  # green
    'window': (0, 0, 255),  # blue
    'bg': (0, 0, 0)
}
pred_img = np.ones((512, 512, 3))
pred_img[pred == 0] = colour_mappings['wall']
pred_img[pred == 1] = colour_mappings['insufficient']
pred_img[pred == 2] = colour_mappings['sufficient']
pred_img[pred == 3] = colour_mappings['window']
pred_img[pred == 4] = colour_mappings['bg']
imshow(pred_img)
plt.show()
