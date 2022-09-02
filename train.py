import os

import numpy as np
import tensorflow as tf
from keras.layers import Activation, Input
from keras.utils.image_utils import ResizeMethod
from keras_preprocessing.image import load_img, img_to_array

from keras.layers.merging import concatenate
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

# Load images from directories
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

# Pad images to 512x512, transform ground truth to grayscale and convert them into an array
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
    mask = tf.image.resize_with_pad(mask, IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, method=ResizeMethod.NEAREST_NEIGHBOR,
                                    antialias=False).numpy()
    # set correct class labels
    mask[mask == 110] = 1
    mask[mask == 114] = 2
    mask[mask == 119] = 3
    mask[mask == 169] = 4
    Y[i] = mask

# Split into train and test set
train_test_split = int(len(input_img_paths) * 0.9)
X_train = X[:train_test_split]
Y_train = Y[:train_test_split]
X_test = X[train_test_split:]
Y_test = Y[train_test_split:]

# Build the U-Net
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

    conv10 = Activation('softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


unet = unet_model((IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, IMG_CHANNELS), n_classes=5)
unet.summary()

# Perform training
EPOCHS = 100

unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_history = unet.fit(X_train, Y_train, validation_split=0.2, batch_size=8, epochs=EPOCHS)

unet.save("model/unet")
