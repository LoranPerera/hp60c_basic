import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import mediapipe as mp
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

class FingertipPose(Node):
    def __init__(self):
        super().__init__('fingertip_pose')

        self.bridge = CvBridge()
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Camera intrinsics (populated from camera_info)
        self.fx = self.fy = self.cx = self.cy = None

        # Latest depth image
        self.depth_image = None

        # Subscriptions
        self.create_subscription(
            CameraInfo,
            '/ascamera_hp60c/camera_publisher/depth0/camera_info',
            self.camera_info_cb, 1)

        self.create_subscription(
            Image,
            '/hp60c/depth',
            self.depth_cb, 10)

        self.create_subscription(
            Image,
            '/hp60c/rgb',
            self.rgb_cb, 10)

        # Publishers
        self.pub_pose = self.create_publisher(PoseStamped, '/hp60c/fingertip_pose', 10)
        self.pub_vis  = self.create_publisher(Image, '/hp60c/fingertip_pose_vis', 10)

        self.get_logger().info('Fingertip Pose node started')

    def camera_info_cb(self, msg):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg):
        # Depth is in millimetres (uint16) — convert to metres
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth_at_point(self, px, py):
        """Get depth value at pixel, averaging a 5x5 patch for stability"""
        h, w = self.depth_image.shape
        px = np.clip(int(px), 0, w - 1)
        py = np.clip(int(py), 0, h - 1)

        patch = self.depth_image[
            max(0, py-2):py+3,
            max(0, px-2):px+3
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0  # mm → metres

    def pixel_to_3d(self, px, py, depth_m):
        """Convert pixel coordinates + depth to 3D point"""
        X = (px - self.cx) * depth_m / self.fx
        Y = (py - self.cy) * depth_m / self.fy
        Z = depth_m
        return np.array([X, Y, Z])

    def estimate_finger_orientation(self, hand_landmarks, rgb_shape):
        """
        Estimate finger orientation from hand landmarks.
        Uses the vector from wrist (landmark 0) through fingertip (landmark 8)
        and constructs an orthonormal frame.
        
        Returns: 3x3 rotation matrix (camera frame)
        """
        h, w = rgb_shape[:2]

        # Get wrist position (landmark 0)
        wrist = hand_landmarks.landmark[0]
        wrist_px = wrist.x * w
        wrist_py = wrist.y * h
        wrist_depth = self.get_depth_at_point(wrist_px, wrist_py)

        # Get fingertip position (landmark 8)
        tip = hand_landmarks.landmark[8]
        tip_px = tip.x * w
        tip_py = tip.y * h
        tip_depth = self.get_depth_at_point(tip_px, tip_py)

        if wrist_depth is None or tip_depth is None:
            # Fallback: use Z-axis pointing forward
            return np.eye(3)

        # Get 3D points
        wrist_3d = self.pixel_to_3d(wrist_px, wrist_py, wrist_depth)
        tip_3d = self.pixel_to_3d(tip_px, tip_py, tip_depth)

        # Finger direction (Z-axis of finger frame)
        finger_vec = tip_3d - wrist_3d
        finger_length = np.linalg.norm(finger_vec)
        if finger_length < 1e-6:
            return np.eye(3)
        z_axis = finger_vec / finger_length

        # Get middle finger joint (landmark 6) to create a plane
        middle = hand_landmarks.landmark[6]
        middle_px = middle.x * w
        middle_py = middle.y * h
        middle_depth = self.get_depth_at_point(middle_px, middle_py)

        if middle_depth is not None:
            middle_3d = self.pixel_to_3d(middle_px, middle_py, middle_depth)
            # Vector from wrist to middle joint
            helper_vec = middle_3d - wrist_3d
        else:
            # Fallback: use right-hand perpendicular in image plane
            helper_vec = np.array([-1, 0, 0])  # arbitrary vector

        # Orthogonalize: X-axis perpendicular to finger direction
        x_axis = np.cross(helper_vec, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            # helper_vec is parallel to z_axis; use fallback
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = x_axis / x_norm

        # Y-axis = Z × X (right-hand rule)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Build rotation matrix: columns are x, y, z axes
        R = np.column_stack([x_axis, y_axis, z_axis])
        return R

    def rgb_cb(self, msg):
        if self.fx is None or self.depth_image is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h, w = rgb.shape[:2]

        results = self.hands.process(rgb)

        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                vis, hand, mp.solutions.hands.HAND_CONNECTIONS)

            # Landmark 8 = index fingertip
            tip = hand.landmark[8]
            px = int(tip.x * w)
            py = int(tip.y * h)

            # Get depth
            Z = self.get_depth_at_point(px, py)
            if Z is None:
                self.get_logger().warn('No valid depth at fingertip')
                self.pub_vis.publish(
                    self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
                return

            # Back-project to 3D
            X = (px - self.cx) * Z / self.fx
            Y = (py - self.cy) * Z / self.fy

            # Estimate orientation
            R = self.estimate_finger_orientation(hand, rgb.shape)

            # Convert rotation matrix to quaternion
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # [x, y, z, w]

            # Publish pose
            pose = PoseStamped()
            pose.header = msg.header
            pose.header.frame_id = 'camera_link'
            
            # Position
            pose.pose.position.x = X
            pose.pose.position.y = Y
            pose.pose.position.z = Z
            
            # Orientation (quaternion)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            
            self.pub_pose.publish(pose)

            self.get_logger().info(
                f'Fingertip Pose → Pos:({X:.3f}, {Y:.3f}, {Z:.3f})m  '
                f'Quat:({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})')

            # Draw overlay
            cv2.circle(vis, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(vis,
                f'Pos: ({X:.2f}, {Y:.2f}, {Z:.2f})',
                (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            cv2.putText(vis,
                f'Quat: ({quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f})',
                (px + 10, py + 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

        self.pub_vis.publish(
            self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = FingertipPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
