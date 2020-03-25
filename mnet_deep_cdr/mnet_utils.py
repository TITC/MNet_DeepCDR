# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing import image
from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np


def pro_process(temp_img, input_size):
    img = np.asarray(temp_img).astype('float32')
    img = np.array(Image.fromarray(img, mode='RGB').resize((input_size, input_size)))
    return img


def train_loader(data_list, data_path, mask_path, input_size):
    while 1:
        for lineIdx, temp_txt in enumerate(data_list):
            train_img = np.asarray(image.load_img(os.path.join(data_path, temp_txt),
                                                  target_size=(input_size, input_size, 3))
                                   ).astype('float32')
            img_mask = np.asarray(
                image.load_img(os.path.join(mask_path, temp_txt),
                               target_size=(input_size, input_size, 3))
            ) / 255.0

            train_img = np.reshape(train_img, (1,) + train_img.shape)
            img_mask = np.reshape(img_mask, (1,) + img_mask.shape)
            yield ([train_img], [img_mask, img_mask, img_mask, img_mask, img_mask])

def train_loader_2(data_list, data_path, mask_path, input_size):
    while 1:
        for lineIdx, temp_txt in enumerate(data_list):
            train_img = np.asarray(image.load_img(os.path.join(data_path, temp_txt),
                                                  target_size=(input_size, input_size, 3))
                                   ).astype('float32')
            img_mask = np.asarray(
                image.load_img(os.path.join(mask_path, temp_txt),
                               target_size=(input_size, input_size, 2))
            ) / 255.0

            train_img = np.reshape(train_img, (1,) + train_img.shape)
            img_mask = np.reshape(img_mask, (1,) + img_mask.shape)
            yield ([train_img], [img_mask, img_mask, img_mask, img_mask, img_mask])

def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    e
