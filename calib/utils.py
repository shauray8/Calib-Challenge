import matplotlib.pyplot as plt
import cv2
import os
import torch

from __future__ import division
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage

def openit(path, line):
    # opening and reading the yaw pitch files
    with open(path, "r") as files:
        data = files.read()
        data = data.split("\n")
        for f in data:
            line.append(f)

line = []
## ---------------- Visualizing Stuff Here ---------------- ##
def show():
    i = 0
    stuff = 4
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
        #print(frame)
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


## ---------------- For Global Motion Aggregation ---------------- ##

class InputPadder:
    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1)* 8 - self.ht) % 8
        pad_wd = (((self.ht // 8) + 1)* 8 - self.wd) % 8

        self._pad = [pad_wd // 2. pad_wd - pad_wd//2, 0, pad_ht]

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
    show()
