import cv2
import numpy as np

kMinNumFeature = 1500
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
cur_t = None
cur_R = None
x, y, z = 0, 0, 0
trueX, trueY, trueZ = 0, 0, 0

path = "/Users/seanfarhat/Downloads/dataset 2/poses/00.txt"
with open(path) as f:
	annotations = f.readlines()

focal = 718.8560
pp = (607.1928, 185.2157)

def feature_tracking(img1, img2, points_1):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, points_1, None)

	st = st.reshape(st.shape[0])
	kp1 = points_1[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2

def feature_detection(img):
	kp = fast.detect(img)
	a = np.array([x.pt for x in kp], dtype=np.float32)
	return a

def getAbsoluteScale(frame_id):  #specialized for KITTI odometry dataset
	ss = annotations[frame_id-1].strip().split()
	x_prev = float(ss[3])
	y_prev = float(ss[7])
	z_prev = float(ss[11])
	ss = annotations[frame_id].strip().split()
	x = float(ss[3])
	y = float(ss[7])
	z = float(ss[11])
	trueX, trueY, trueZ = x, y, z
	return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev)), trueX, trueY, trueZ

# img1 = cv2.imread("/Users/seanfarhat/Desktop/researchsu18/donkey_car/mono-vo/tub/35_cam-image_array_.jpg", 0)
# img2 = cv2.imread("/Users/seanfarhat/Desktop/researchsu18/donkey_car/mono-vo/tub/36_cam-image_array_.jpg", 0)
img1 = cv2.imread('/Users/seanfarhat/Downloads/dataset/sequences/00/image_0/'+str(0).zfill(6)+'.png', 0)
img2 = cv2.imread('/Users/seanfarhat/Downloads/dataset/sequences/00/image_0/'+str(1).zfill(6)+'.png', 0)


points_1 = feature_detection(img1)
points_1, points_2 = feature_tracking(img1, img2, points_1)

E, mask = cv2.findEssentialMat(points_2, points_1, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, cur_R, cur_t, mask = cv2.recoverPose(E, points_2, points_1, focal=focal, pp = pp)

img1 = img2
points_1 = points_2

x1, y1, z1 = cur_t[0], cur_t[1], cur_t[2]

cv2.namedWindow( "Road facing camera", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "Trajectory", cv2.WINDOW_AUTOSIZE )

traj = np.zeros((600,600,3), dtype=np.uint8)

# for img_id in range(37, 1024):
for img_id in range(2, 4541):

	# filename = "/Users/seanfarhat/Desktop/researchsu18/donkey_car/mono-vo/tub/" + str(img_id) + "_cam-image_array_.jpg"
	filename = "/Users/seanfarhat/Downloads/dataset/sequences/00/image_0/"+str(img_id).zfill(6)+".png"
	img2 = cv2.imread(filename, 0) 

	# Feature detection already done by points_1 = points_2

	points_1, points_2 = feature_tracking(img1, img2, points_1);

	E, mask = cv2.findEssentialMat(points_2, points_1, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
	_, R, t, mask = cv2.recoverPose(E, points_2, points_1, focal=focal, pp = pp, mask=mask)

	# x2, y2, z2 = t[0], t[1], t[2]

	# absolute_scale = np.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) + (z2 - z1)*(z2 - z1))
	absolute_scale, trueX, trueY, trueZ = getAbsoluteScale(img_id)
	if(absolute_scale > 0.1):
		cur_t = cur_t + absolute_scale * cur_R.dot(t) 
		cur_R = R.dot(cur_R)
	if(points_1.shape[0] < kMinNumFeature):
		points_2 = feature_detection(img2)

	img1 = img2;
	points_1 = points_2
	# x1, y1, z1 = x2, y2, z2
	x, y, z = cur_t[0], cur_t[1], cur_t[2]


	### Handles drawing trajectory on map

	draw_x, draw_y = int(x)+290, int(z)+90
	true_x, true_y = int(trueX)+290, int(trueZ)+90

	cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
	cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x,y,z)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img1)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)




