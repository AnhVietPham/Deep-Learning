from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, concatenate, Conv2DTranspose, \
    BatchNormalization, Dropout, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from skimage.io import imshow
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanIoU

# https://towardsdatascience.com/semantic-segmentation-of-aerial-imagery-captured-by-a-drone-using-different-u-net-approaches-91e32c92803c

# Dataset link: https://www.kaggle.com/awsaf49/semantic-drone-dataset



train_path = '/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/DLTensorflow/semantic-segmentation/dataset/semantic_drone_dataset/training_set/images/*.jpg'


def importing_data(path):
    sample = []
    for filename in glob.glob(path):
        img = Image.open(filename, 'r')
        img = img.resize((256, 256))
        img = np.array(img)
        sample.append(img)
    return sample


mask_path = "/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/DLTensorflow/semantic-segmentation/dataset/semantic_drone_dataset/training_set/gt/semantic/label_images/*.png"


def importing_data_mask(path):
    sample = []
    for filename in glob.glob(path):
        img = Image.open(filename, 'r')
        img = img.resize((256, 256))
        img = np.array(img)
        sample.append(img)
    return sample


def image_labels(label):
    image_labels = np.zeros(label.shape, dtype=np.uint8)
    for i in range(24):
        image_labels[np.all(label == labels[i, :], axis=-1)] = i
    image_labels = image_labels[:, :, 0]
    return image_labels


def multiclass_unet_architecture(n_classes=2, height=256, width=256, channels=3):
    inputs = Input((height, width, channels))

    conv_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv_1 = Dropout(0.1)(conv_1)
    conv_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_1)
    pool_1 = MaxPooling2D((2, 2))(conv_1)

    conv_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_1)
    conv_2 = Dropout(0.1)(conv_2)
    conv_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_2)
    pool_2 = MaxPooling2D((2, 2))(conv_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_2)
    conv_3 = Dropout(0.1)(conv_3)
    conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_3)
    pool_3 = MaxPooling2D((2, 2))(conv_3)

    conv_4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_3)
    conv_4 = Dropout(0.1)(conv_4)
    conv_4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_4)
    pool_4 = MaxPooling2D((2, 2))(conv_4)

    conv_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_4)
    conv_5 = Dropout(0.2)(conv_5)
    conv_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_5)
    u6 = concatenate([u6, conv_4])
    conv_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    conv_6 = Dropout(0.2)(conv_6)
    conv_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_6)
    u7 = concatenate([u7, conv_3])
    conv_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    conv_7 = Dropout(0.1)(conv_7)
    conv_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_7)
    u8 = concatenate([u8, conv_2])
    conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    conv_8 = Dropout(0.2)(conv_8)
    conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_8)
    u9 = concatenate([u9, conv_1], axis=3)
    conv_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    conv_9 = Dropout(0.1)(conv_9)
    conv_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(conv_9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model


def jacard(y_true, y_pred):
    y_true_c = K.flatten(y_true)
    y_pred_c = K.flatten(y_pred)
    intersection = K.sum(y_true_c * y_pred_c)
    return (intersection * 1.0) / (K.sum(y_true_c) + K.sum(y_pred_c) - intersection + 1.0)


def jacard_loss(y_true, y_pred):
    return -jacard(y_true, y_pred)


metrics = ['accuracy', jacard]

if __name__ == '__main__':
    data_train = importing_data(train_path)
    data_train = np.asarray(data_train)

    data_mask = importing_data_mask(mask_path)
    data_mask = np.asarray(data_mask)

    # Normalization
    scaler = MinMaxScaler()
    nsamples, nx, ny, nz = data_train.shape
    d2_data_train = data_train.reshape((nsamples, nx * ny * nz))
    train_images = scaler.fit_transform(d2_data_train)
    train_images = train_images.reshape(400, 256, 256, 3)

    # Labels of the masks
    labels = pd.read_csv(
        "/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/DLTensorflow/semantic-segmentation/dataset/semantic_drone_dataset/training_set/gt/semantic/class_dict.csv")
    labels = labels.drop(['name'], axis=1)
    labels = np.array(labels)

    labels_final = []
    for i in range(data_mask.shape[0]):
        label = image_labels(data_mask[i])
        labels_final.append(label)
    labels_final = np.array(labels_final)

    # train_Test
    n_classes = len(np.unique(labels_final))
    labels_cat = to_categorical(labels_final, num_classes=n_classes)
    x_train, x_test, y_train, y_test = train_test_split(train_images, labels_cat, test_size=0.2, random_state=42)

    # U-net
    img_height = x_train.shape[1]
    img_width = x_train.shape[2]
    img_channels = x_train.shape[3]

    model = multiclass_unet_architecture(n_classes=n_classes, height=img_height, width=img_width, channels=img_channels)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=2,
                        validation_data=(x_test, y_test),
                        shuffle=False)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
