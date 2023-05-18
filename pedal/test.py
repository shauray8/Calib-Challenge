## ------------- test for the pretrained weights goes here ------------- ##
import torch
import matplotlib.pyplot as plt
import numpy 
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt

from FlownetCorr import *
from utils import *
import FlownetCorr

def test_on_data():
    data = "../../data/calib_image_data"
    batch_size = 16
    folder = 0
    with open("../../data/calib_pretrainde/checkpoint217.pkl", 'rb') as pickle_file:
        pretrained = pickle.load(pickle_file)
    model = flownetc()
    model.load_state_dict(pretrained["state_dict"])

    test_transform = transforms.Compose([
            transforms.Resize((100))])


    image = cv2.imread(f"{data}/1/frame0000.jpg", 1)

    shaoe = image.shape
    print(shaoe)
    bigger = cv2.resize(image, (100,134))
    plt.imshow(bigger)
    plt.show()

    test_set = Transformed_data(data, transform=test_transform, split=None)
    test_loader = DataLoader(
            test_set, batch_size = batch_size, pin_memory=True, shuffle=False)

    validation(test_loader, model)        

def validation(val_loader, model):
    model.eval()
    
    for i, (input, yaw, pitch) in enumerate(val_loader):
        input = torch.cat(input,1).to(device)

        pred_yaw, pred_pitch = model(input)
        print(pred_yaw)

        with open('./predicted/{folder}.txt', 'w') as f:
            f.write(pred_yaw, " ", pred_pitch)

    folder += 1

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


if __name__ == "__main__":
    test_on_data()
    print("Visualizing stuff with pretrained weights !")
