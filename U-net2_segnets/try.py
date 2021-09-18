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

imgs_dir = "E:\data\comma10k\comma10k\imgs"

ids = [file for file in os.listdir(imgs_dir)]



