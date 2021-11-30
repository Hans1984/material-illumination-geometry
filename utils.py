import os
import numpy as np
import scipy.misc
import scipy.ndimage
from easydict import EasyDict as edict
from scipy.io import loadmat
import tensorlayer as tl
from tensorlayer.layers import *
from sklearn.metrics import mean_squared_error
import random
import cv2
import csv
import pandas as pd
import math
import tensorflow as tf
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift

##load cvs and read images
def load_data_cvs_exclude(arg_dict, is_train=False):
    input_path = arg_dict.input_path
    gt_path = arg_dict.gt_path

    data = pd.read_csv(gt_path, usecols=['BRDF', 'glossiness', 'refsharp', 'contgloss', 'metallicness', 'lightness', 'anisotropy'])
    data = np.array(data)

    d_validation = []
    d_train = []
    for i in range(len(data[:,0])):
       if '[bunny_vt]'in data[:,0][i] and '1_1_cambridge_2k' in data[:,0][i]:
             d_validation.append(data[i,0:7])
       if not '[bunny_vt]' in data[:,0][i] and not '1_1_cambridge_2k' in data[:,0][i]:
             d_train.append(data[i,0:7])

    d_train = np.array(d_train)
    d_validation = np.array(d_validation)

    file_paths = []
    if is_train:
        file_names =  d_train[:,0]
        gt_values = (d_train[:,1:7] - 1.0)/6.0
        input_files = list(os.path.join(input_path, file_name) for file_name in file_names)
    else:
        file_names =  d_validation[:,0]
        gt_values = (d_validation[:,1:7] - 1.0)/6.0
        input_files = list(os.path.join(input_path, file_name) for file_name in file_names)

    return input_files, gt_values

def read_imgs_test_cv(img_paths):
    img = (cv2.imread(img_paths)/255.).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img[np.newaxis,:,:,:]

    return img


#data augmentation
def filp_horizontal(img):
    img = img[:, ::-1, :]
    img = img_resize(img)
    return img

def flip_vertical(img):
    img = img[::-1, :, :]
    img = img_resize(img)
    return img

def img_padding(img):
    img = np.pad(img, ((128, 128), (128, 128), (0, 0)), 'edge')
    return img

def img_resize(img, size=(512, 512)):
    img = cv2.resize(img, size)#imresize(img, size = size)
    return img

def random_rotation(img, angle_range=(0, 360)):
    h, w, _ = img.shape
    angle = np.random.randint(*angle_range)
    img = rotate(img, angle)
    img = img_resize(img)
    return img
    
def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image

def scale_augmentation(img, scale_range, crop_size):
    scale_size = np.random.randint(*scale_range)
    img = cv2.resize(img, (scale_size, scale_size))#imresize(img, (scale_size, scale_size))
    img = random_crop(img, crop_size)
    img = img_resize(img)
    return img

def scale_augmentation_distored(img, scale_range_h, scale_range_w, crop_size):
    scale_size_h = np.random.randint(*scale_range_h)
    scale_size_w = np.random.randint(*scale_range_w)
    img = cv2.resize(img, (scale_size_h, scale_size_w))#imresize(img, (scale_size_h, scale_size_w))
    img = random_crop(img, crop_size)
    img = img_resize(img)
    return img

def shift_horizontal(img, shift_range):
    shift_size = np.random.randint(*shift_range)
    img = shift(img, shift = [0, shift_size, 0], cval = 0)
    img = img_resize(img)
    return img


def shift_vertical(img, shift_range):
    shift_size = np.random.randint(*shift_range)
    img = shift(img, shift = [shift_size, 0, 0], cval = 0)
    img = img_resize(img)
    return img

def gasuss_noise(img, mean=0, var=0.005):
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    out = img_resize(out)
    return out

def poisson_noise(img):
    noise = np.random.poisson(img) 
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    out = img_resize(out)
    return out

def data_augmentation(img, ep):
    if ep == 0:
        img = poisson_noise(img)
    if ep == 1:
        img = filp_horizontal(img)
    if ep == 2:
        img = flip_vertical(img)
    if ep == 3:
        img = random_rotation(img)
    if ep == 4:
        img = scale_augmentation(img, (900, 1480), 800)
    if ep == 5:
        img = scale_augmentation_distored(img, (920, 1280), (1000, 1280), 800)
    if ep == 6:
        img = shift_horizontal(img, (64, 128))
    if ep == 7:
        img = shift_vertical(img, (64, 128))
    if ep == 8:
        img = gasuss_noise(img)
    if ep == 9:
        img = img_resize(img)
    if ep == 10:
        img = img_resize(img)
    return img


def read_imgs_augmentation(img_paths):
    """ Input an image path and name, return an image array """
    ep = np.random.randint(0, 10)
    imgs = []

    for idx in range(len(img_paths)):
        img = (cv2.imread(img_paths[idx])/255.).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = data_augmentation(img, ep)
        imgs.append(img)

    return imgs
    
##loss function for training
#RMLSE Loss Function
def RMSLE(y_predict, gt):
    log_predict = tf.log(y_predict + 1.0)
    log_gt = tf.log(gt + 1.0)
    rmlse_loss = tf.reduce_mean(tf.square(tf.subtract(log_predict, log_gt)))
    return rmlse_loss
