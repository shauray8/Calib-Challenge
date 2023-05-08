from __future__ import division
import os, glob
import shutil
import matplotlib.pyplot as plt
import cv2

import torch
import scipy.ndimage as ndimage
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
from imageio import imread
import pickle

import random
import numpy as np
import numbers
import types
import argparse

from model import *

with torch.no_grad():
    imgs_dir = "E:\data\comma10k\comma10k\imgs"
    masks_dir = "E:\data\comma10k\comma10k\masks"
    pretrained_model = "E:\data\comma10k\pretrained/pretraining_on_sample.pkl"

    ids = [file for file in os.listdir(imgs_dir)]

    model = Unet_square(4,4)
    with open(pretrained_model, 'rb') as pickle_file:
        network_data = pickle.load(pickle_file)

    model.load_state_dict(network_data["state_dict"])
    main_epoch = network_data['epoch']
    print(main_epoch)

    H, W = 1928, 1208
    img_nd = np.asarray(Image.open("E:\data\comma10k\comma10k\imgs\\"+ids[i]).resize((H//8,W//8)))
    fig, ax = plt.subplots()
    ax.imshow(img_nd)
    while img_nd.shape[2] < 4:
        img_nd = np.concatenate((img_nd, np.zeros((W//8,H//8,1))), axis=2)

    img_trans = img_nd
    img_trans = img_trans.reshape(1, img_nd.shape[0],img_nd.shape[1], img_nd.shape[2]).transpose((0,3,1,2))
    print(img_trans.shape)
    test = torch.tensor(img_trans, dtype=torch.float32)
    img = model(test)
    another = np.array(img[0])
    img = img[0].permute(0,2,3,1)
    print(img[0,:,:,:3].shape)
    img = img[0,:,:,0:]
    print(img_nd.shape)
    ax.imshow(img, alpha=.4)
    plt.show()
