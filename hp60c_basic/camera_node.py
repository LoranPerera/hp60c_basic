import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

class HP60CBasic(Node):
    def __init__(self):
        super().__init__('hp60c_basic')

        # Subscribers
        self.sub_depth = self.create_subscription(
            Image,
            '/ascamera_hp60c/camera_publisher/depth0/image_raw',
            self.depth_cb, 10)

        self.sub_rgb = self.create_subscription(
            Image,
            '/ascamera_hp60c/camera_publisher/rgb0/image',
            self.rgb_cb, 10)

        self.sub_points = self.create_subscription(
            PointCloud2,
            '/ascamera_hp60c/camera_publisher/depth0/points',
            self.points_cb, 10)

        # Publishers with clean names
        self.pub_depth = self.create_publisher(Image, '/hp60c/depth', 10)
        self.pub_rgb   = self.create_publisher(Image, '/hp60c/rgb', 10)
        self.pub_points = self.create_publisher(PointCloud2, '/hp60c/points', 10)

        self.get_logger().info('HP60C Basic node started')

    def depth_cb(self, msg):
        self.pub_depth.publish(msg)

    def rgb_cb(self, msg):
        self.pub_rgb.publish(msg)

    def points_cb(self, msg):
        self.pub_points.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HP60CBasic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
