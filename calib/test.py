import matplotlib.pyplot as plt
import cv2
import os

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./labeled/1.hevc')

path = "./labeled/1.txt"
line = []
with open(path, "r") as files:
    data = files.read()
    data = data.split("\n")
    for f in data:
        line.append(f)

i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 
                line[i], 
                (50, 50), 
                 font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

