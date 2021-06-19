## ------------- test for the pretrained weights goes here ------------- ##
import torch
import matplotlib.pyplot as plt
import numpy 
import pickle

from FlownetCorr import *
from utils import *
import FlownetCorr

def load_in_the_weights():
    pretrained = pickle.load("weights.pkl")
    

if __name__ == "__main__":
    print("Visualizing stuff with pretrained weights !")
