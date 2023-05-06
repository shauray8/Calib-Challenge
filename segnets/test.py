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

imgs_dir = "E:\data\comma10k\comma10k\imgs"
pretrained_model = "./pretrained/checkpoint.pkl"

ids = [file for file in os.listdir(imgs_dir)]

model = Unet_square(4,4)
with open(pretrained_model, 'rb') as pickle_file:
    network_data = pickle.load(pickle_file)

model.load_state_dict(network_data["state_dict"])
main_epoch = network_data['epoch']

print(ids[0])

test = np.asarray(Image.open(ids[0]).resize((H//8,W//8)))
img = model(test)
print(img.shape)
