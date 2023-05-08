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


## ---------------- For FlowNetCorr ---------------- ##

class comma10k_dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.scale = scale

        assert 0 < scale <= 1, 'scale must be between 0 and 1'

#        self.ids = [os.path.splitext(file)[0].split("_mask")[0] for file in os.listdir(masks_dir)
#                if not file.startswith('.')]

        self.ids = [file for file in os.listdir(masks_dir)]

        print(f"=> Creating dataset with {len(self.ids)} example")
        

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_image, scale=1):
        try:
            w, h, d = pil_image.shape
        except:
            w, h = pil_image.shape
            d = 1
        newW, newH = int(scale * w), int(scale * h)
        img_nd = pil_image.reshape((newW, newH, d))

        while img_nd.shape[2] < 4:
                img_nd = np.concatenate((img_nd, np.zeros((w,h,1))), axis=2)

        img_trans = img_nd.transpose((2,0,1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):

        idx = self.ids[i]
        self.img_file = glob.glob(self.imgs_dir+ "/" + idx)
        self.mask_file = glob.glob(self.masks_dir+ "/" + idx)
        H, W = 1928, 1208

        assert len(self.img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {self.img_file} : {self.mask_file} : {glob.glob(self.imgs_dir+ "/" + idx)}'
        img = Image.open(self.img_file[0]).resize((H//8,W//8))
        mask = np.asarray(Image.open(self.mask_file[0]).resize((H//8,W//8)))
        if self.transform is not None:
            img = self.transform(img)

        img = np.asarray(img)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        #print(f"img {img.shape} == mask {mask.shape}")
        
        return {
            'image': torch.from_numpy((np.array(img))).type(torch.FloatTensor),
            'mask': torch.from_numpy((np.array(mask))).type(torch.FloatTensor)
        }

## ---------------- Make Dataset --> [[img1, img2], [yaw, pitch]] ---------------- ##
def loader(path_imgs, yaw, pitch):
    #return [np.array((Image.open(img)), dtype=np.float64) for img in path_imgs], target
    return [(Image.open(img)) for img in path_imgs], yaw, pitch

## ---------------- return all the usefull stuff ---------------- ##

class ListDataset(Dataset):
    def __init__(self, root, path_list, yaw_classes, pitch_classes, transform=None):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.yaw_classes = yaw_classes
        self.pitch_classes = pitch_classes

    def __getitem__(self, index):
        
        inputs, yaw, pitch = self.path_list[index]
        yaw, pitch = onehot_vector(yaw, self.yaw_classes), onehot_vector(pitch, self.pitch_classes)
        inputs, yaw, pitch = loader(inputs, yaw, pitch)

        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])

        return inputs, np.float32(yaw), np.float32(pitch)

    def __len__(self):
        return len(self.path_list)

## ---------------- grab all the data from respective files ---------------- ##

def DATA_LOADER(root, split):        
    img_data = []
    mode = 5
    yaw_array = []
    pitch_array = []
    for i in range(mode):
        input_img = (glob.glob(os.path.join(root, f"{str(i)}") + '/*.jpg'))
        target_num = (glob.glob(os.path.join("../labeled") + f'/{i}.txt'))
        drive_img = []

        w = open(target_num[0],'r')
        drive_img.append(w.read())
        drive_img = drive_img[0].split("\n")
    
        for i in range(len(input_img)-1):
            yaw, pitch = drive_img[i].split(" ")
            yaw = 0 if yaw == "nan" else yaw
            pitch = 0 if pitch == "nan" else pitch
            img_data.append([[ input_img[i], input_img[i+1] ], float(yaw), float(pitch) ])
            yaw_array.append(float(yaw))
            pitch_array.append(float(pitch))
            
## ---------------- initializing class ararys with value 0. ---------------- ##

    yaw_array = np.sort(yaw_array)
    pitch_array = np.sort(pitch_array)
    yaw_classes, pitch_classes = [0.], [0.]

## ---------------- breaking into classes of 100 each ---------------- ##

    for yaw in range(len(yaw_array)):
        if yaw % 100 == 0 and yaw_array[yaw] > 0:
            yaw_classes.append(yaw_array[yaw])
    yaw_classes.append(1.)

    for pitch in range(len(pitch_array)):
        if pitch % 100 == 0 and pitch_array[pitch] > 0:
            pitch_classes.append(pitch_array[pitch])
    pitch_classes.append(1.)

## ---------------- train, validation split ---------------- ##

    train, test = [], []
    if split is not None:
        for sample in range( int(split*len(img_data)) ):
            train.append(img_data[sample])

        for sample in range( int(split*len(img_data)), len(img_data) ):
            test.append(img_data[sample])

    return train, test, yaw_classes, pitch_classes


def Transformed_data(root, transform=None, split=None):
    train, test, yaw_classes, pitch_classes = DATA_LOADER(root, split)
    #print("YAW",yaw_classes, "PITCH",pitch_classes)
    train_dataset = ListDataset(root, train, yaw_classes, pitch_classes, transform)
    test_dataset = ListDataset(root, test, yaw_classes, pitch_classes, transform)

    return train_dataset, test_dataset

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pkl'):
    with open(os.path.join(save_path, filename), 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))

if __name__ == "__main__":

## ---------------- Displaying the dataset with respective yaw and pitch ---------------- ##

    parser = argparse.ArgumentParser(description='Wanna Watch some random dude drive',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default=2, type=int, metavar='N',
                    help='number of video')
    args = parser.parse_args()
    show(args.data)
 
## ---------------- Breaking frames into images for the dataset ---------------- ##

    break_into_images()
