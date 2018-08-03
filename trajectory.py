import cv2
import numpy as np

while 1:
	for x in np.arange(989):
		filename = "./images/frame_" + str(x) + ".png"
		img = cv2.imread(filename)
		cv2.imshow("Car POV", img)
		k = cv2.waitKey(1) #### ALWAYS NEED WAITKEY WHEN DOING IMSHOW

