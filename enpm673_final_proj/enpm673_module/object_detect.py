import cv2
import numpy as np

# TODO: Add connection to movement/stopping
# TODO: Prep code that saves reference frame for when object isn't moving (optical flow not detected) but is still there - assigned to EB

def detect_obstacle(prev_frame,frame):
    # prep colors
    blue = [255,0,0]
    red = [0,0,255]
    # Resize image for consistency
    last_frame = cv2.resize(prev_frame, (640,480))
    last_frame_g = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    curr_frame = cv2.resize(frame, (640,480))
    curr_frame_g = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    #=====Lucas-Kanade Optical Flow Method=====
    # OpenCV uses Shi-Tomasi corner detection for features to track
    shitomasi_params = dict(maxCorners = 100, qualityLevel=0.3, minDistance=5, blockSize=7)
    # Lucas-Kanade parameters
    lk_params = dict(winSize=(15,15), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Need to find important features in the first frame
    features_last = cv2.goodFeaturesToTrack(last_frame_g, mask=None, **shitomasi_params)
    
    # Calculate optical flow
    if features_last is not None:
        features_current, flow_status, error = cv2.calcOpticalFlowPyrLK(last_frame_g, curr_frame_g, features_last, None, **lk_params)
    else:
        features_current = None
    
    # Select points that were successfully tracked between 2 frames (and optical flow was detected)
    moving_points = [] # for drawing box around obstacle
    last_moving_points = []
    # Section runs if at least one points has been tracked
    if features_current is not None:
        # Only getting points that were detected to have optical flow
        tracked_last = features_last[flow_status==1]
        tracked_current = features_current[flow_status==1]
        # Getting mean angle and magnitude of each vector to separate obstacle vectors from environment vectors caused by turtlebot moving
        all_vectors = tracked_current - tracked_last
        magnitudes = np.linalg.norm(all_vectors,axis=1) # compute magnitude for each pair and put it into new array
        # First filtering out points that aren't actually moving
        static_points_indices = np.where(magnitudes > 1)[0]
        moving_vectors = all_vectors[static_points_indices]
        moving_current = tracked_current[static_points_indices]
        moving_last = tracked_last[static_points_indices]
        # Continuing onto angle stuff
        dx = moving_vectors[:,0]
        dy = moving_vectors[:,1]
        angles = np.arctan2(dy,dx) # make array of angle of each point pair
        mean_angle = np.mean(angles)
        # Going through each point that was detected to have optical flow
        angle_differences = np.abs(angles - mean_angle)
        normalized_angle_differences = np.mod(angle_differences,2*np.pi) # normalizes angle to this range
        # Get indices of points/vectors that are different enough from mean
        obstacle_vector_indices = np.where(normalized_angle_differences > 1.6)[0] # this is 90 degrees in rads
        # Get only the current points that are suspected to be the moving objects
        obstacle_points = moving_current[obstacle_vector_indices]
        last_points = moving_last[obstacle_vector_indices]
        
        # Draw tracks between the 2 frames (prob don't need to keep in code for actual version but want to see for now)
        # basically copied from opencv just for code testing purposes - not to go into final submission version
        for current, last in zip(obstacle_points, last_points):
            a, b = current.ravel()
            c, d = last.ravel()
            cv2.line(curr_frame, (int(a), int(b)), (int(c), int(d)), red, 2)
            cv2.circle(curr_frame, (int(a), int(b)), 5, blue, -1)
        
        # If robot and environment is not moving, this will not run, nothing will get drawn
        if len(obstacle_points) != 0:
            # print(moving_points)
            # # Using all points to get the mean center coordinates for the object
            # pts = np.array(moving_points,dtype=np.int32)
            # print(pts)
            center_x = np.mean(obstacle_points[:,0])
            center_y = np.mean(obstacle_points[:,1])
            rect_corner_x = int(center_x - 50)
            rect_corner_y = int(center_y - 90)
            w = 100
            h = 180
            # Since it's definitely going to be the oat box then can use those set values and the center of the collected obstacle movement points
            cv2.rectangle(curr_frame, (rect_corner_x,rect_corner_y),(rect_corner_x+w,rect_corner_y+h),red,2)
            # make it stop for one second
            # add boolean so that can run while loop after obstacle is detected
            # draws smallest rectangle possible to contain all of the moving points
            # Can be further improved with filtering out outliers, will do this
            # rect_corner_x,rect_corner_y,w,h = cv2.boundingRect(pts)
            return curr_frame, True
    # Returning drawed on frame to show
    return curr_frame, False

# psuedocode for keeping turtlebot stopped while obstacle is still in FOV
# when optical flow is detected, change status to moving
# while status is moving, save frame to remember it, this will be resaved while status of flow is still moving
# when optical flow isn't detected, change status to stopped
# if status is stopped, 
