#!/usr/bin/env python3

from enpm673_module.lane_following import calculate_camera_matrix, detect_aruco
from enpm673_module.horizon_overlay import horizon_overlay
from enpm673_module.stop_sign_detect import detect_stop_sign
from enpm673_module.object_detect import detect_obstacle

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.qos import ReliabilityPolicy

from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from tf_transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation

from cv_bridge import CvBridge
import cv2

import numpy as np
import math

import sys

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            #CompressedImage,
            #'/tb4_2/oakd/rgb/image_raw/compressed',
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )

        # create a publisher for the aruco marker so the nav node can read it
        self.aruco_pub = self.create_publisher(
            Twist,
            "/closest_aruco",
            ReliabilityPolicy.RELIABLE
        )

        # create a publisher for stopping
        self.stop_pub = self.create_publisher(
            Bool,
            "/stop",
            ReliabilityPolicy.RELIABLE
        )

        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        self.frame_count = 0
        self.frame_list_for_calibration = []
        self.cam_param = []
        self.FLAG_ONCE = False
        self.FLAG_HORIZON = 0
        self.recalibrate_camera = False # this should only be True if you are camera calibrating and doing nothing else
        # For optical flow
        self.prev_frame = None
        self.HORIZON_LINE = 0.0

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #np_arr = np.frombuffer(msg.data, np.uint8)
        #frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        #frame = cv2.resize(frame, (int(frame.shape[0]/2), int(frame.shape[1]/2)))

        frame_of = frame.copy()
        # TODO: cascade the frames such that all the detection information is displayed on one camera stream

        # Call the horizon detection function
        if not self.FLAG_HORIZON:
            processed_frame, self.HORIZON_LINE = horizon_overlay(frame)
            self.FLAG_HORIZON = True
        # else:
        #     cv2.line(frame, (0, self.HORIZON_LINE), (frame.shape[1], self.HORIZON_LINE), (0, 255, 0), 2)  # Green line

        # if we need to calibrate our camera (ie, we don't know the K matrix)
        if self.recalibrate_camera:
            if not self.FLAG_ONCE:
                self.frame_list_for_calibration.append(frame)
                self.frame_count += 1
                if self.frame_count > 10:
                    self.FLAG_ONCE = True
                    self.cam_param = calculate_camera_matrix(self.frame_list_for_calibration)            
            #else:
                # call the lane detection function
                #lane_centers, _ = detect_aruco(frame, self.cam_param)
                # cv2.imshow("Camera Frame LANES", lanes)
            sys.exit(2)
        else: # if we already know the coefficients, no need to recalculate

            K = np.array([[4.18404789e+03, 0.00000000e+00, 3.38875214e+02],
                          [0.00000000e+00, 4.20664741e+03, 6.75762736e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

            dist = np.array([[ 9.12787690e-01, -2.73742883e+01,  2.47619406e-02, -4.51819478e-02, 4.26597072e+02]])

            # before Gazebo was changed:
            #K = np.array([[3.85232023e+03, 0.00000000e+00, 1.41557688e+02],
            #              [0.00000000e+00, 4.08970185e+03, 2.76145748e+02],
            #              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            
            #dist = np.array([[-3.01932588e+00, 2.77595150e+02, 1.36824583e-02, 2.86416942e-02, 4.80133005e+00]])

            # call the lane detection function
            lane_centers, aruco = detect_aruco(frame, [K, dist])
            #cv2.imshow("Lane (aruco) Frame", lane_centers)
            #cv2.waitKey(1)  # Required to refresh the OpenCV window

            if aruco:
                #print(aruco)
                aruco_msg = Twist()
                aruco_msg.linear.x = float(aruco[0][0][0][0])
                aruco_msg.linear.y = float(aruco[0][0][0][1])
                aruco_msg.linear.z = float(aruco[0][0][0][2])
                aruco_msg.angular.x = float(aruco[1][0][0][0])
                aruco_msg.angular.y = float(aruco[1][0][0][1])
                aruco_msg.angular.z = float(aruco[1][0][0][2])
                self.aruco_pub.publish(aruco_msg)
        
        stop, mask, stop_sign_frame = detect_stop_sign(frame)
        self.get_logger().info("stop sign stop")
        
        # Optical Flow Section
        # Skip first one frame where there is no previous frame
        self.get_logger().info("obstacle detect start")
        if self.prev_frame is None:
            self.prev_frame = frame_of
            obstacle = False
        else:
            obstacle_frame, obstacle = detect_obstacle(self.prev_frame,frame_of)
            self.prev_frame = frame_of
            cv2.imshow("Obstacle Frame", obstacle_frame)
        
        stop_msg = Bool()
        if obstacle or stop: # if there is an obstacle in the way or a stop sign
            stop_msg.data = True
        else:
            stop_msg.data = False
        self.stop_pub.publish(stop_msg)

        # Display the processed frames and stop sign frame
        # cv2.imshow("Horizon Frame", processed_frame)
        cv2.imshow("Stop Sign Frame", stop_sign_frame)
        cv2.waitKey(1)  # Required to refresh the OpenCV window

# node for turtlebot movement
class NavNode(Node):
    def __init__(self):
        super().__init__('nav_node')

        # NOTE: be sure to change the topics to align with lab robot!!

        # subscriber to the odom
        self.odom_sub = self.create_subscription(
            Odometry,
            #"tb4_2/odom",
            "/odom",
            self.odom_cb,
            10,
        )

        # subscriber to the custom aruco topic
        self.aurco_sub = self.create_subscription(
            Twist,
            "/closest_aruco",
            self.aruco_cb,
            10,
        )

        # subscriber to the custom stop topic
        self.stop_sub = self.create_subscription(
            Bool,
            "/stop",
            self.stop_cb,
            10
        )

        # publisher to the velocity
        self.vel_pub = self.create_publisher(
            Twist,
            #"tb4_2/cmd_vel",
            "/cmd_vel",
            ReliabilityPolicy.RELIABLE
        )

        '''
        # create timer for the publisher
        self.timer_vel_pub = self.create_timer(
            2, # Hz, professor said this is about the lag we can expect
            self.timer_vel_pub_cb
        )
        '''
        self.aruco_lin = []
        self.aruco_ang = []

    def aruco_cb(self, msg : Twist):
        # need to convert into Euler angles
        ang_rot = Rotation.from_rotvec([msg.angular.x, msg.angular.y, msg.angular.z]).as_euler('xyz', degrees=False)
        self.aruco_lin = [msg.linear.x, msg.linear.y, msg.linear.z]
        self.aruco_ang = [ang_rot[0], ang_rot[1], ang_rot[2]]

    def odom_cb(self, msg : Odometry):
        # actually, we don't even need to check the odom, since the aruco frame represents the difference wrt camera frame
        x_now = msg.pose.pose.position.x   # x position
        y_now = msg.pose.pose.position.y   # y position
        q = msg.pose.pose.orientation      # quarternion
        _, _, o_now = euler_from_quaternion([q.x, q.y, q.z, q.w])  # z orientation

        # since the odom has updated, we ought to update the nav
        #self.calculate_nav()
    
    def stop_cb(self, msg : Bool):
        if msg.data: # stop sign or obstacle
            vel_msg = Twist()
            vel_msg.angular.z = 0.0
            vel_msg.linear.x = 0.0
            self.vel_pub.publish(vel_msg)
        else:
            self.calculate_nav()


    def calculate_nav(self):
        # distance error
        # angle to goal (which is used for heading error)

        vel_msg = Twist()
        vel_msg.angular.z = 0.0
        vel_msg.linear.x = 0.0

        if not self.aruco_lin: # if we haven't seen an aruco yet
            vel_msg.angular.z = 0.2 # keep rotating until we find one
            vel_msg.linear.x = 0.0
            self.vel_pub.publish(vel_msg)
            print("No aruco")
            return
        else:
            dist_err = math.sqrt(self.aruco_lin[0]**2 + self.aruco_lin[1]**2) # + 0.25 to purposefully saturate it early at long distances
            heading_err = self.aruco_ang[1] + math.pi/2 - math.pi/12
            #print(self.aruco_ang)
            print(f"HEADING ERROR: {heading_err} ({heading_err * 360 / (2*math.pi)})")
            print(f"DISTANCE ERROR: {dist_err}\n")

            # PID
            Kp_v = 0.6  # P for velocity
            Kp_o = 0.5  # P for orientation

            if abs(heading_err) > math.pi/15:   # threshold before we start correcting
                #vel_msg.linear.x = 0.0   # stop so that we can turn independantly for now
                vel_msg.angular.z = Kp_o * heading_err
                vel_msg.linear.x = Kp_v * dist_err
            else:
                vel_msg.linear.x = Kp_v * dist_err
                #vel_msg.angular.z = Kp_o * heading_err
                vel_msg.angular.z = 0.0
            
            # saturation caps
            if vel_msg.linear.x > 0.15:
                vel_msg.linear.x = 0.15
            if vel_msg.angular.x < -0.4:
                vel_msg.angular.x = -0.4
            if vel_msg.angular.z > 0.4:
                vel_msg.angular.z = 0.4

            # ignore distance PID for now
            vel_msg.linear.x = 0.25

        self.vel_pub.publish(vel_msg)


def main(args=None):
    rclpy.init(args=args)

    cam_node = CameraSubscriber()
    cam_node.get_logger().info("Camera node created")

    nav_node = NavNode()
    nav_node.get_logger().info("Navigation node created")

    # so we can have multiple nodes running simultanously (since camera callback is process-intensive)
    exec_thread = SingleThreadedExecutor()
    exec_thread.add_node(cam_node)
    exec_thread.add_node(nav_node)
    
    try:
        exec_thread.spin()
    except KeyboardInterrupt:
        pass

    # Cleanup
    cam_node.destroy_node()
    nav_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()







        # # Create a directory to save frames
        # self.save_dir = "saved_frames"
        # os.makedirs(self.save_dir, exist_ok=True)


        # # Save the frame with a timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # e.g., 20250429_123456789
        # filename = os.path.join(self.save_dir, f"frame_{timestamp}.png")
        # cv2.imwrite(filename, frame)
        # self.get_logger().info(f"Saved frame: {filename}")