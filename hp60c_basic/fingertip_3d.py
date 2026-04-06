import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import mediapipe as mp
import numpy as np
import cv2

class Fingertip3D(Node):
    def __init__(self):
        super().__init__('fingertip_3d')

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
        self.pub_point = self.create_publisher(PointStamped, '/hp60c/fingertip_3d', 10)
        self.pub_vis   = self.create_publisher(Image, '/hp60c/fingertip_vis', 10)

        self.get_logger().info('Fingertip 3D node started')

    def camera_info_cb(self, msg):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg):
        # Depth is in millimetres (uint16) — convert to metres
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

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

            # Clamp to image bounds
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)

            # Get depth — average a 5x5 patch for stability
            patch = self.depth_image[
                max(0, py-2):py+3,
                max(0, px-2):px+3
            ]
            valid = patch[patch > 0]
            if valid.size == 0:
                self.get_logger().warn('No valid depth at fingertip')
                return

            Z = float(np.median(valid)) / 1000.0  # mm → metres

            # Back-project to 3D
            X = (px - self.cx) * Z / self.fx
            Y = (py - self.cy) * Z / self.fy

            # Publish 3D point
            pt = PointStamped()
            pt.header = msg.header
            pt.header.frame_id = 'camera_link'
            pt.point.x = X
            pt.point.y = Y
            pt.point.z = Z
            self.pub_point.publish(pt)

            self.get_logger().info(
                f'Fingertip → X:{X:.3f}m  Y:{Y:.3f}m  Z:{Z:.3f}m')

            # Draw overlay
            cv2.circle(vis, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(vis,
                f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m',
                (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

        self.pub_vis.publish(
            self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = Fingertip3D()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
