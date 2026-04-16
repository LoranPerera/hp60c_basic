import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

class FingertipYOLO(Node):
    def __init__(self):
        super().__init__('fingertip_yolo')

        self.bridge = CvBridge()
        
        # Load YOLOv8 pose model (detects human pose keypoints including hands)
        # Use 'yolov8n-pose' for faster inference, 'yolov8m-pose' for better accuracy
        try:
            self.model = YOLO('yolov8n-pose.pt')
            self.get_logger().info('YOLOv8 Pose model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            self.model = None

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
        self.pub_pose = self.create_publisher(PoseStamped, '/hp60c/fingertip_yolo_pose', 10)
        self.pub_vis  = self.create_publisher(Image, '/hp60c/fingertip_yolo_vis', 10)

        self.get_logger().info('Fingertip YOLO node started')

    def camera_info_cb(self, msg):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg):
        # Depth is in millimetres (uint16)
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth_at_point(self, px, py):
        """Get depth value at pixel, averaging a 5x5 patch for stability"""
        if self.depth_image is None:
            return None
        
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

    def estimate_hand_orientation(self, keypoints, rgb_shape):
        """
        Estimate hand orientation from YOLO keypoints.
        YOLO pose model keypoints (for hand):
        - Wrist (0), Thumb (1-4), Index (5-8), Middle (9-12), 
          Ring (13-16), Pinky (17-20)
        
        Uses vector from wrist to index fingertip as the primary direction.
        Returns: 3x3 rotation matrix
        """
        h, w = rgb_shape[:2]
        
        # Extract visible keypoints (YOLO returns [x, y, confidence])
        kpt = keypoints.reshape(-1, 3)  # N keypoints × 3 (x, y, conf)
        
        # Keypoint indices (COCO hand format from YOLO)
        # 0: wrist, 5-8: index finger
        wrist_idx = 0
        index_tip_idx = 8  # Index fingertip
        index_pip_idx = 6  # Index PIP joint
        
        if kpt[wrist_idx, 2] < 0.5 or kpt[index_tip_idx, 2] < 0.5:
            return np.eye(3)  # Fallback if keypoints not visible
        
        # Get wrist
        wrist_px, wrist_py = kpt[wrist_idx, :2]
        wrist_depth = self.get_depth_at_point(wrist_px, wrist_py)
        
        # Get index fingertip
        tip_px, tip_py = kpt[index_tip_idx, :2]
        tip_depth = self.get_depth_at_point(tip_px, tip_py)
        
        # Get index PIP for reference
        pip_px, pip_py = kpt[index_pip_idx, :2]
        pip_depth = self.get_depth_at_point(pip_px, pip_py)
        
        if wrist_depth is None or tip_depth is None:
            return np.eye(3)
        
        # Convert to 3D
        wrist_3d = self.pixel_to_3d(wrist_px, wrist_py, wrist_depth)
        tip_3d = self.pixel_to_3d(tip_px, tip_py, tip_depth)
        
        # Z-axis: finger direction (wrist to tip)
        z_axis = tip_3d - wrist_3d
        z_length = np.linalg.norm(z_axis)
        if z_length < 1e-6:
            return np.eye(3)
        z_axis = z_axis / z_length
        
        # Helper vector: use PIP joint if available, else use arbitrary
        if pip_depth is not None:
            pip_3d = self.pixel_to_3d(pip_px, pip_py, pip_depth)
            helper_vec = pip_3d - wrist_3d
        else:
            helper_vec = np.array([-1, 0, 0])
        
        # X-axis: perpendicular to finger direction
        x_axis = np.cross(helper_vec, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = x_axis / x_norm
        
        # Y-axis: complete orthonormal frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Build rotation matrix
        R = np.column_stack([x_axis, y_axis, z_axis])
        return R

    def rgb_cb(self, msg):
        if self.fx is None or self.depth_image is None or self.model is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h, w = rgb.shape[:2]

        # Run YOLO inference
        results = self.model(rgb, verbose=False)
        
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints
            
            # Process first person detected
            if keypoints.xy is not None and len(keypoints.xy) > 0:
                kpt = keypoints.xy[0]  # First person's keypoints (N×2)
                conf = keypoints.conf[0] if keypoints.conf is not None else np.ones(len(kpt))
                
                # Index fingertip is keypoint 8 in COCO hand format
                if len(kpt) > 8 and conf[8] > 0.5:
                    tip_px, tip_py = kpt[8]
                    wrist_px, wrist_py = kpt[0]
                    
                    # Get depth at fingertip
                    Z = self.get_depth_at_point(tip_px, tip_py)
                    if Z is None:
                        self.get_logger().warn('No valid depth at fingertip')
                        self.pub_vis.publish(
                            self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
                        return
                    
                    # Check that keypoints exist
                    keypoints_all = np.concatenate([kpt, conf.reshape(-1, 1)], axis=1)
                    
                    # Back-project to 3D
                    X = float((tip_px - self.cx) * Z / self.fx)
                    Y = float((tip_py - self.cy) * Z / self.fy)
                    Z = float(Z)

                    # Estimate orientation
                    R = self.estimate_hand_orientation(keypoints_all.flatten(), rgb.shape)

                    # Convert rotation matrix to quaternion
                    rot = Rotation.from_matrix(R)
                    quat = [float(v) for v in rot.as_quat()]  # [x, y, z, w]
                    # Publish pose
                    pose = PoseStamped()
                    pose.header = msg.header
                    pose.header.frame_id = 'camera_link'
                    
                    # Position
                    pose.pose.position.x = X
                    pose.pose.position.y = Y
                    pose.pose.position.z = Z
                    
                    # Orientation
                    pose.pose.orientation.x = quat[0]
                    pose.pose.orientation.y = quat[1]
                    pose.pose.orientation.z = quat[2]
                    pose.pose.orientation.w = quat[3]
                    
                    self.pub_pose.publish(pose)
                    
                    self.get_logger().info(
                        f'YOLO Fingertip → Pos:({X:.3f}, {Y:.3f}, {Z:.3f})m  '
                        f'Quat:({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})')
                    
                    # Draw keypoints and skeleton
                    for i, (kpt_x, kpt_y) in enumerate(kpt):
                        if conf[i] > 0.5:
                            cv2.circle(vis, (int(kpt_x), int(kpt_y)), 4, (0, 255, 0), -1)
                    
                    # Draw fingertip highlight
                    cv2.circle(vis, (int(tip_px), int(tip_py)), 8, (0, 0, 255), -1)
                    cv2.putText(vis,
                        f'Pos: ({X:.2f}, {Y:.2f}, {Z:.2f})',
                        (int(tip_px) + 10, int(tip_py) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
                    cv2.putText(vis,
                        f'Quat: ({quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f})',
                        (int(tip_px) + 10, int(tip_py) + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        
        self.pub_vis.publish(
            self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = FingertipYOLO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
