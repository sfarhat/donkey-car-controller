# donkey-car-controller
Research project in UC Berkeley Swarm Lab to create an autonomous RC car

# Files

pid.ipynb - In this file, I experimented will several methods to get the controller working, some of which made it into the final approach

pid.py - This was simply the iPython notebook condensed into the parts necessary to run the script

trajectory.py - Given a path to trajectory plots computed in pid.py, it will display them all in sequence

vo-python.py - Implementation of monocular visual odometry in python.

# Requirements

python3, numpy, and cv2 (OpenCV's python library)

# Method

The process of going from image to trajectory involves several steps:

1. Load the image
2. Apply a binary mask denoting "good" and "bad" areas
  a. For now, I have used Canny Edge Detection to implement an algorithm that will provide a pretty accurate mask
  b. In the future, we hope to transfer this process to a pre-trained CNN known as SqueezeNet
3. From the mask, compute the desired trajectory using an algorithm that takes the weighted sum of "good" pixels in each row and offsets the trajectory vector appropriately
4. Using a PID controller where the feedback is the trajectory from the previous frame and the target is the trajectory for the current frame, compute the actual trajectory.
