import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
import time
import cv2
import os

IMAGE_DIM = 128
P = 1.2
I = 1
D = 0.00001
canny_low = 150
canny_high = 250


def get_trajectory(img, weight_factor):
    
    # Iterative algorithm. Begins at bottom of image and computes middle of green
    # area for each succesive row. The weight of that decision becomes less and 
    # less as the row index "decreases" -> gets farther away from car, scaled the weight_factor passed in. 
    # A lower weight_factor (< 1 leads to problems) leads to more sensitive changes. Stops when first all 
    # "red" row encountered or reached top of image and returns (x, y) coordinates of trajectory vector
    
    weight, scaling_factor = 1, weight_factor
    left, right = 0, IMAGE_DIM - 1
    center, trajectory_x, trajectory_y = 0, 0, 0
    
    for i in np.arange(IMAGE_DIM - 1, -1, -1):
        for j in np.arange(IMAGE_DIM):
            
            if img[i][j] == 1 and left is 0:
                left = j
            elif img[i][j] == 0 and right is IMAGE_DIM - 1 and left is not 0:
                right = j - 1
                
        if left is 0 and img[i][0] == 0: # reached all "red" row
            trajectory_y = i + 1 # this seems weird but it's because we are iterating through rows in reverse
            break
            
        mid = (left + right) / 2
        if i == IMAGE_DIM - 1:
            center = mid
            trajectory_x = center
            
        trajectory_x = trajectory_x + ((mid - trajectory_x) * weight)
        left, right = 0, IMAGE_DIM - 1
        weight /= scaling_factor
        
    return [(trajectory_x, trajectory_y), center]


def get_arrow_info(trajectory, center):
    x = center
    y = IMAGE_DIM - 1
    dx = trajectory[0] - x
    dy = -1 * (y - trajectory[1])
    
    return [x, y, dx, dy]

class PIDController():
    
    def __init__(self, P = 0, I = 0, D = 0):
        
        # Constants to be tuned
        self.Kp = P
        self.Ki = I
        self.Kd = D
        
        # For integral term
        self.cumulative_error = 0
        
        # Initializing the time to time.time() leads to non-determinism
        self.prev_time = time.time()
        self.curr_time = 0
        
        # Error values
        self.prev_error = 0
        self.curr_error = 0
        
        
    def update(self, target, feedback, debug = False):
        
        self.curr_error = target - feedback
        de = self.curr_error - self.prev_error
        
        self.curr_time = time.time()
        dt = self.curr_time - self.prev_time
        
        p = self.Kp * self.curr_error
        
        self.cumulative_error += self.curr_error * dt
        i = self.Ki * self.cumulative_error
        
        d = self.Kd * (de / dt)
        
        output = p + i + d
        
        self.prev_error = self.curr_error
        self.prev_time = self.curr_time
        
        if debug:
            print('p:', p)
            print('i:', i)
            print('d:', d)
            print('de:', de)
            print('dt:', dt)

        return output     


def edges_to_mask(edges):
    mask = np.ones((IMAGE_DIM, IMAGE_DIM))
    bad_cols = []
    buffer_x = np.arange(20, 100) ### NOT WORKING FOR SOME REASON
    buffer_y = np.arange(110, IMAGE_DIM) # Buffer so that front bumper is considered ok to drive towards
    
    for i in np.arange(IMAGE_DIM - 1, -1, -1):
        for j in np.arange(IMAGE_DIM):
            
            if j in bad_cols:
                mask[i][j] = 0
            
            if edges[i][j] and i not in buffer_y:
                mask[i][j] = 0
                # If an edge is detected, nowhere "ahead" in that path can be driven towards, 
                # so mask all further rows in that column
                bad_cols.append(j)
            
                
    return mask

edges_list = []
mask_list = []
img_list = []

for x in range(35, 1024):

    filename = "../donkey_car/tub/" + str(x) + "_cam-image_array_.jpg"
    img = cv2.imread(filename, 0)
    img_list += [img]

    edges = cv2.Canny(img, 100, 200)
    edges_list += [edges]
        
    mask = edges_to_mask(edges)
    mask_list += [mask]

    print(x)

print("FINISHED COMPUTING MASKS")


pid = PIDController(P, I, D)

mask = mask_list[0]
img = img_list[0]
feedback_trajectory, feedback_center = get_trajectory(mask, 1.05)
feedback_scaled = feedback_trajectory[0] / IMAGE_DIM
x_f, y_f, dx_f, dy_f = get_arrow_info(feedback_trajectory, feedback_center)

for i in np.arange(1, len(mask_list)):
    
    mask = mask_list[i]
    img = img_list[i]
    target_trajectory, target_center = get_trajectory(mask, 1.05)
    
    target_scaled = target_trajectory[0] / IMAGE_DIM
    output = pid.update(target_scaled, feedback_scaled, debug = False)
    actual = output * IMAGE_DIM
    pid_trajectory = (actual + feedback_trajectory[0], feedback_trajectory[1])
    
    x_t, y_t, dx_t, dy_t = get_arrow_info(target_trajectory, target_center)
    x_pid, y_pid, dx_pid, dy_pid = get_arrow_info(pid_trajectory, target_center)
    
    fig = plt.figure()
    plt.text(10, 150, "Feedback: " + str(feedback_trajectory))
    plt.text(10, 160, "Target: " + str(target_trajectory))
    plt.text(10, 170, "PID: " + str(pid_trajectory))
    plt.arrow(x_f, y_f, dx_f, dy_f, width = 1, color = "yellow")
    plt.arrow(x_t, y_t, dx_t, dy_t, width = 1, color = "lightblue")
    plt.arrow(x_pid, y_pid, dx_pid, dy_pid, width = 1, color = "lightgreen")
    plt.imshow(mask, cmap = "RdYlGn",  interpolation='none')
    plt.imshow(img, alpha = 0.5)
    fig.savefig("./images/pid_frame_" + str(i) + ".png")
    
    feedback_trajectory, feedback_center = target_trajectory, target_center
    feedback_scaled = target_scaled

    print(i)

print("FINISHED COMPUTING TRAJECTORIES")

os.system("ffmpeg -f image2 -framerate 1 -i pid_frame_%d.png pid.gif")
os.system("mv pid.gif ../")
