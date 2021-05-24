from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import torch
import shutil
from PIL import Image
from imageio import imread

import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
from torch.utils.data import DataLoader, Dataset
import argparse

def openit(path, line):
    # opening and reading the yaw pitch files
    with open(path, "r") as files:
        data = files.read()
        data = data.split("\n")
        for f in data:
            line.append(f)

line = []
## ---------------- Visualizing Stuff Here ---------------- ##
def show(data):
    i = 0
    stuff = data
    cap = cv2.VideoCapture(f'../labeled/{stuff}.hevc')

    # path to the dataset and stuff
    path = f"../labeled/{stuff}.txt"
    openit(path, line)

    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 
                    f"GT - {line[i]}", 
                    (50, 50), 
                     font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
        
        # prints images with yaw and pitch on it 
        cv2.imshow('frame',frame)

        ## prints out all the frames and corosponding yaw and pitch
        #print(len(frame))
        #print(line[i])

        i += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



## ---------------- For FlowNetCorr ---------------- ##

class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th

        return inputs, target


## ---------------- Make Dataset --> [[img1, img2], [yaw, pitch]] ---------------- ##
def loader(path_imgs, yaw, pitch):
    #return [np.array((Image.open(img)), dtype=np.float64) for img in path_imgs], target
    return [(Image.open(img)) for img in path_imgs], yaw, pitch

class ListDataset(Dataset):
    def __init__(self, root, path_list, transform=None):

        self.root = root
        self.path_list = path_list
        self.transform = transform

    def __getitem__(self, index):
        
        inputs, yaw, pitch = self.path_list[index]
        inputs, yaw, pitch = loader(inputs, yaw, pitch)

        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])

        return inputs, yaw, pitch

    def __len__(self):
        return len(self.path_list)

def DATA_LOADER(root, split):        
    img_data = []
    mode = 5
    for i in range(mode):
        input_img = (glob.glob(os.path.join(root, f"{str(i)}") + '/*.jpg'))
        target_num = (glob.glob(os.path.join("../labeled") + f'/{i}.txt'))
        drive_img = []

        w = open(target_num[0],'r')
        drive_img.append(w.read())
        drive_img = drive_img[0].split("\n")
        for i in range(len(input_img)-1):
            yaw, pitch = drive_img[i].split(" ")
            img_data.append([[ input_img[i], input_img[i+1] ], float(yaw), float(pitch) ])

    train, test = [], []
    for sample in range( int(split*len(img_data)) ):
        train.append(img_data[sample])
    for sample in range( int(split*len(img_data)), len(img_data) ):
        test.append(img_data[sample])

    return train, test

def Transformed_data(root, transform=None, split=None):
    train, test = DATA_LOADER(root, split)
    train_dataset = ListDataset(root, train, transform )
    test_dataset = ListDataset(root, test, transform )

    return train_dataset, test_dataset


## ---------------- Buy new RAM! I have to break stuff into images and save them ---------------- ##

def frame_by_frame(input):
    frames = []
    for j in input:
        cap= cv2.VideoCapture(j)
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frames.append(frame)
            i+=1

        cap.release()
        cv2.destroyAllWindows()
        frames.append(torch.zeros(frames[0].shape))
        print("inner",len(frames))
        final_images = [torch.cat((torch.tensor(frames[i]), torch.tensor(frames[i+1])), 1) for i in range(len(frames))]
    print(len(final_image))
    return final_image

## ---------------- This thing is no good but for now i dont think i have a better idea ---------------- ##

def break_into_images():
    input_vid = sorted(glob.glob(os.path.join('../labeled') + '/*.HEVC'))
    folder = 0
    for i in input_vid:
        folder += 1
        vidcap = cv2.VideoCapture(i)
        success,image = vidcap.read()
        count = 0
        while success:
          cv2.imwrite(f"../data/{folder}/frame%d.jpg" % count, image)     # save frame as JPEG file      
          success,image = vidcap.read()
          print('Read a new frame: ', success)
          count += 1

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))

## ---------------- For Global Motion Aggregation ---------------- ##

class InputPadder:
    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1)* 8 - self.ht) % 8
        pad_wd = (((self.ht // 8) + 1)* 8 - self.wd) % 8

        self._pad = [pad_wd // 2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, model='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2: ]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arrange(wd), np.arrange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wanna Watch some random dude drive',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default=1, type=int, metavar='N',
                    help='number of video')
    args = parser.parse_args()
    #show(args.data)
    #frame_by_frame('../labeled/2.hevc')
    #DATA_LOADER("../labeled", "Transform", 22).frame_by_frame
    #break_into_images()
