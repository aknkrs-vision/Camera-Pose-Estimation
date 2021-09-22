import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


MIN_MATCH_COUNT = 8
FLANN_INDEX_KDTREE = 1

# Camera Calibration
def calibrate_camera():
    global camera_matrix
    focal_length = 100
    cx, cy = 960, 540
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # no distortion

    vr2d = np.load("./points/vr2d.npy")  
    vr3d = np.load("./points/vr3d.npy") 

    vr2d = np.squeeze(vr2d)  
    vr3d = np.squeeze(vr3d) 

    img1 = cv2.imread("./images/img1.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)  # camera matrix

    ret, camera_matrix, dist_coeffs, rotation_vector, translation_vector = cv2.calibrateCamera(
        [vr3d], [vr2d], img1.shape[::-1], K, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    print(rotation_vector,translation_vector)

# Feature Matching
def get_best_matches(ref_des, q_des, ratio=0.8):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ref_des, q_des, k=2) 

    best_matches = []

    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            best_matches.append(m)
    return best_matches

 
def get_rotation(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def get_translation(ref_img, query_img):
    global rotation_estimation
    sift = cv2.SIFT_create()

    ref_kp, ref_des = sift.detectAndCompute(ref_img, None)
    q_kp, q_des = sift.detectAndCompute(query_img, None)
    best_matches = get_best_matches(ref_des, q_des, ratio=0.95)

    if len(best_matches) > MIN_MATCH_COUNT:
        
        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in best_matches])
        dst_pts = np.float32([q_kp[m.trainIdx].pt for m in best_matches])

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix, method=cv2.RANSAC,  prob=0.999)

        points, rotation_estimation, translation_estimation, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix)

        return  translation_estimation

def print_results():
    ref_img = cv2.imread('./images/img1.png')
    query_img = cv2.imread('./images/img2.png')

    print("Reference image: img1 ")
    print("Query image: img2 ")
    print(get_translation(ref_img, query_img))
    print(get_rotation(rotation_estimation))

    ref_img = cv2.imread('./images/img1.png')
    query_img = cv2.imread('./images/img3.png')

    print("Reference image: img1 ")
    print("Query image: img3 ")
    print(get_translation(ref_img, query_img))
    print(get_rotation(rotation_estimation))

def plot_results():
    ref_img = cv2.imread('./images/img1.png')
    query_img = cv2.imread('./images/img2.png')
    coordinates = get_translation(ref_img, query_img)
    x, y, z = coordinates[0], coordinates[1], coordinates[2]

    ref_img = cv2.imread('./images/img1.png')
    query_img = cv2.imread('./images/img3.png')
    coordinates1 = get_translation(ref_img, query_img)
    x1, y1, z1 = coordinates1[0], coordinates1[1], coordinates1[2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-8, 2)
    ax.set_ylim(-8, 2)
    ax.set_zlim(-8, 2)
    ax.scatter(x,y,z)
    ax.scatter(x1,y1,z1)
    ax.scatter(0,0,0)
    plt.show()

 
