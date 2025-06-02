import cv2
import sys
import numpy as np

# TODO: the Gazebo simulation has 4x4 aruco, make sure the irl lab has the same 
# TODO: the Gzebo simulation has a 5x7 calibration checkerboard, make sure the irl lab also does
# TODO: check if our assumption of aruco marker x always being in the forward direction for us

"""
Calculates the camera K matrix and distortion coefficients so that aruco frames can be determined in other functions

args:
    - frames: frames to use for the calibration (recommended 5-10 total)

returns:
    - [K, dist]:
        - K:    3x3 camera matrix
        - dist: distortion coefficient matrix
"""
def calculate_camera_matrix(frames):
    # we only consider the inner corners (ie, those that touch two black squares)
    SIZE = (5, 7) # 6 rows by 8 columns of inner corners
    DIM = 0.03 # each box is known to be 2cm by 2cm (0.02m)

    # we need world points (3d, known)
    corners = np.zeros((SIZE[0] * SIZE[1], 1, 3), np.float32) # pixel_points will be compared to this later

    # array has been manually entered for a 5x7 checkerboard for now
    count = 0
    for x in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for y in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
            corners[count][0] = [x, y, 0]
            count += 1

    print(f"---> Size of calibration board: {SIZE}")
    print(f"---> Size of individual squares on board: {DIM}m x {DIM}m")
    print(f"---> Total number of corners: {len(corners)}")

    pixel_points = []   # where the corner pixels are in our FOV
    world_points = []   # where the corners are in the workspace
    for frame in frames:
        r, c = cv2.findChessboardCorners(frame, (SIZE[1], SIZE[0]))
        if r: # only append to the list if valid corner has been found
            pixel_points.append(c)
            world_points.append(corners)
        else:
            print("*** FATAL: No corners found for camera calibration")
            print("(tip: ensure that the checkerboard is reasonably visible to the camera and that there are no aruco markers in view)")
            sys.exit(2)

    # perform the calibration
    _, K, dist, R, t = cv2.calibrateCamera(world_points, pixel_points, (frames[0].shape[0], frames[0].shape[1]), None, None)
    
    # reverse transform for sanity check
    #K_new = cv2.getOptimalNewCameraMatrix(K, dist, (frames[0].shape[0], frames[0].shape[1]), 1, (frames[0].shape[0], frames[0].shape[1]))

    print(f"---> Camera matrix: {K}")
    print(f"---> Distortion coefficients: {dist}")

    return [K, dist]

"""
Detects the aruco markers visible in the camera FOV

args:
    - frame:     single frame to check
    - cam_param: camera K and distortion

returns:
    - frame:     frame with coordinate frames at aruco centers drawn ontop
"""
def detect_aruco(frame, cam_param):
    # get the frame in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # define the aruco markers and parameters using standard functions
    #aruco_markers = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_markers = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_parameters = cv2.aruco.DetectorParameters()

    # create the detector and use it to find markers ids and corners
    detector = cv2.aruco.ArucoDetector(aruco_markers, aruco_parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    # extract the passed camera parameters
    K = cam_param[0]
    dist = cam_param[1]

    # get the pose information of the markers
    aruco_list_t = []
    aruco_list_r = []
    if ids is not None:
        for i in range(0, len(ids)):

            # get and draw the rotation and translation vectors for the marker
            r, t, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, K, dist)
            cv2.drawFrameAxes(frame, K, dist, r, t, 0.1)
            
            # add the transformations to a list we can filter later for our aruco of interest in navigation
            aruco_list_t.append(t)
            aruco_list_r.append(r)

        # draw marker on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        #print(aruco_list_t)
        closest_aruco = (aruco_list_t[0], aruco_list_r[0])
        for i in range(0, len(aruco_list_t)):
            if abs(aruco_list_t[i][0][0][2]) < abs(closest_aruco[0][0][0][2]):
                closest_aruco = [aruco_list_t[i], aruco_list_r[i]]
        return frame, closest_aruco
    
    return frame, None