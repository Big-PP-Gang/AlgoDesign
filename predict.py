import os

import numpy as np
import tensorflow as tf
from keras.utils.image_utils import ResizeMethod
from keras_preprocessing.image import load_img, img_to_array

from skimage.io import imshow

import matplotlib.pyplot as plt

# Load and transform the images, same as in train script
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
    mask = tf.image.resize_with_pad(mask, IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, method=ResizeMethod.NEAREST_NEIGHBOR,
                                    antialias=False).numpy()
    mask[mask == 110] = 1
    mask[mask == 114] = 2
    mask[mask == 119] = 3
    mask[mask == 169] = 4
    Y[i] = mask

train_test_split = int(len(input_img_paths) * 0.9)
X_train = X[:train_test_split]
Y_train = Y[:train_test_split]
X_test = X[train_test_split:]
Y_test = Y[train_test_split:]

# Load saved model and print accuracy
unet = tf.keras.models.load_model('model/unet')
loss, acc = unet.evaluate(X_test, Y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# Show the result of one input, ground truth and the model prediction
predictions = unet.predict(X_test)
i = 1
imshow(X_test[i])
plt.show()
imshow(np.squeeze(Y_test[i]))
plt.show()
pred = predictions[i]
pred = pred.argmax(axis=2)

# Transform gray-scaled images back to RGB
colour_mappings = {
    'wall': (255, 0, 255),  # purple
    'insufficient': (255, 0, 0),  # red
    'sufficient': (0, 255, 0),  # green
    'window': (0, 0, 255),  # blue
    'bg': (0, 0, 0)  # black
}
pred_img = np.ones((512, 512, 3))
pred_img[pred == 0] = colour_mappings['bg']
pred_img[pred == 1] = colour_mappings['window']
pred_img[pred == 2] = colour_mappings['insufficient']
pred_img[pred == 3] = colour_mappings['wall']
pred_img[pred == 4] = colour_mappings['sufficient']

imshow(pred_img)
plt.show()
