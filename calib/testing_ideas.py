import matplotlib.pyplot as plt
import cv2
import os

# capturing videos from the file
stuff = 2
cap = cv2.VideoCapture(f'../labeled/{stuff}.hevc')

# path to the dataset and stuff
path = f"../labeled/{stuff}.txt"
line = []

# opening and reading the yaw pitch files
with open(path, "r") as files:
    data = files.read()
    data = data.split("\n")
    for f in data:
        line.append(f)

# shows the data set 
def show():
    i = 0
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 
                    line[i], 
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

show()
