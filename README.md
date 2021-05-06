# Commaai's Calib-Challange
## Goal
the goal is to predict the direction of travel (in camera frame) from provided dashcam video. (yaw and pitch, fortunately, no roll)
</br></br>
Commaai's repo provides 10 videos. Every video is 1min long and 20 fps.</br>
5 videos are labelled with a 2D array describing the direction of travel at every frame of the video with a pitch and yaw angle in radians.</br>
5 videos are unlabeled. It is your task to generate the labels for them.</br>
The example labels are generated using a Neural Network, and the labels were confirmed with a SLAM algorithm.</br>
You can estimate the focal length to be 910 pixels.</br>
</br>
![](./Docs/yaw-pitch-roll.png)

## Evaluation
They will evaluate our mean squared error against our ground truth labels. Errors for frames where the car speed is less than 4m/s will be ignored. Those are also labeled as NaN in the example labels.
</br></br>
commaai's repo includes an eval script that will give an error score (lower is better). You can use it to test your solutions against the labeled examples. They will use this script to evaluate your solution.

## Navigation
[Labelled dataset](./labeled)</br>
[Unlabeled test dataset](./unlabeled)</br>
[Eval script](eval.py)</br>
[My Model and training script](./calib)</br>
[Setup]("setup.py") (adding soon dont worry) </br>

## example of how opensource is changing the world !! 
[comma ai](https://github.com/commaai)
