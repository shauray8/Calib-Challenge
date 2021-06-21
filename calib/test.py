## ------------- test for the pretrained weights goes here ------------- ##
import torch
import matplotlib.pyplot as plt
import numpy 
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from FlownetCorr import *
from utils import *
import FlownetCorr

def test_on_data():
    data = "../../data/calib_image_data"
    batch_size = 16
    folder = 0
    with open("../../data/checkpoint.pkl", 'rb') as pickle_file:
        pretrained = pickle.load(pickle_file)
    model = flownetc()
    model.load_state_dict(pretrained["state_dict"])

    test_transform = transforms.Compose([
            transforms.Resize((100))])

    test_set = Transformed_data(data, transform=test_transform, split=None)
    test_loader = DataLoader(
            test_set, batch_size = batch_size, pin_memory=True, shuffle=False)
    

    validation(test_loader, model)        

def validation(val_loader, model):
    model.eval()
    
    for i, (input, yaw, pitch) in enumerate(val_loader):
        yaw = yaw.to(device)
        pitch = pitch.to(device)
        input = torch.cat(input,1).to(device)

        pred_yaw, pred_pitch = model(input)

        with open('./predicted/{folder}.txt', 'w') as f:
            f.write(yaw, " ", pitch)

    folder += 1

if __name__ == "__main__":
    test_on_data()
    print("Visualizing stuff with pretrained weights !")
