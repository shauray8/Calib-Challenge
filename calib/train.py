import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import torchvision 

from FlownetCorr import *
import FlownetCorr

def callable():
    kwargs = sorted(name for name in FlownetCorr.__dict__
        if name.islower() and not name.startswith("__")
        and callable(FlownetCorr.__dict__[name]))
    return kwargs




