## ------------- test for the pretrained weights goes here ------------- ##
import torch
import matplotlib.pyplot as plt
import numpy 
import pickle

from FlownetCorr import *
from utils import *
import FlownetCorr

def test_on_data():
    pretrained = pickle.loads("./pretrained/checkpoint.pkl") 
    model = flownetc()
    model.load_state_dict(pretrained["state_dict"])

    test_transform = transforms.Compose([
            transforms.Resize((100)))

    test_set = Transformed_data(data, transform=None, split=None)
    test_loader = DataLoader(
            test_set, batch_size = batch_size, num_workers=args.workers,
            pin_memory=True, shuffle=False)



if __name__ == "__main__":
    test_on_data()
    print("Visualizing stuff with pretrained weights !")
