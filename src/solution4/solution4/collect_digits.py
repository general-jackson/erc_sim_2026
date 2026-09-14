import os

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class DigitDataCollectorNode(Node):

    def __init__(self):
        super().__init__('digit_data_collector')

        # Parameters
        self.declare_parameter(
                'image_topic', '/head_front_camera/head_front_camera/color/image_raw'
        )
        self.declare_parameter(
                'depth_topic', '/head_front_camera/head_front_camera/depth/image_rect_raw'
        )
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter(
                'rotate_speed', 0.5
        )  # Speed in rad/s (Left turn is positive)
        self.declare_parameter('save_dir', '/opt/erc_ws/digit_files')
        self.declare_parameter(
                'save_interval', 0.2
        )  # Time interval between saved images (seconds)

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.rotate_speed = abs(self.get_parameter('rotate_speed').value)
        self.save_dir = self.get_parameter('save_dir').value
        self.save_interval = self.get_parameter('save_interval').value
        self.max_images = 50
        self.depth_change_threshold = 0.5
        self.depth_patch_size = 20

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(
                f'Saving captured frames to directory: {self.save_dir}'
        )

        # Initialize tools & tracking state
        self.bridge = CvBridge()
        self.image_count = 0
        self.last_save_time = 0.0
        self.previous_center_depth = None
        self.capture_started = False

        # ROS 2 Interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.image_sub = message_filters.Subscriber(
                self, Image, image_topic, qos_profile=10
        )
        self.depth_sub = message_filters.Subscriber(
                self, Image, depth_topic, qos_profile=10
        )
        self.image_sync = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.image_sync.registerCallback(self.image_callback)

        # Continuous movement timer (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_rotation)

        self.get_logger().info('Digit collector initialized. Rotating left...')

    def publish_rotation(self):
        if self.image_count >= self.max_images:
            return
        twist = Twist()
        twist.angular.z = self.rotate_speed  # Positive value forces left rotation
        self.cmd_vel_pub.publish(twist)

    def _center_depth(self, depth_msg: Image):
        depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
        )
        if depth_image.ndim != 2:
            raise ValueError('Depth image must be a single-channel image')

        height, width = depth_image.shape
        half_size = self.depth_patch_size // 2
        center_y, center_x = height // 2, width // 2
        patch = depth_image[
                max(0, center_y - half_size):min(height, center_y + half_size),
                max(0, center_x - half_size):min(width, center_x + half_size),
        ].astype('float32')
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None

        center_depth = float(valid.mean())
        if depth_msg.encoding == '16UC1':
            center_depth /= 1000.0
        return center_depth

    def image_callback(self, image_msg: Image, depth_msg: Image):
        try:
            center_depth = self._center_depth(depth_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to read depth image: {str(e)}')
            return

        if center_depth is None:
            return

        if self.previous_center_depth is None:
            self.previous_center_depth = center_depth
            return

        depth_change = abs(center_depth - self.previous_center_depth)
        self.previous_center_depth = center_depth
        if not self.capture_started:
            if depth_change < self.depth_change_threshold:
                return
            self.capture_started = True
            self.get_logger().info(
                    f'Depth change detected at stream center: {depth_change:.2f} m. '
                    f'Starting capture of up to {self.max_images} images.'
            )

        if self.image_count >= self.max_images:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        # Throttle image saves to avoid filling disk too quickly
        if current_time - self.last_save_time >= self.save_interval:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                filename = os.path.join(
                        self.save_dir, f'digit_frame_{self.image_count:04d}.png'
                )
                cv2.imwrite(filename, cv_image)
                self.image_count += 1
                self.last_save_time = current_time
                self.get_logger().info(f'Saved image #{self.image_count}: {filename}')
                if self.image_count == self.max_images:
                    self.cmd_vel_pub.publish(Twist())
                    self.timer.cancel()
                    self.get_logger().info(
                            f'Maximum image count ({self.max_images}) reached; stopped rotation.'
                    )
            except Exception as e:
                self.get_logger().error(f'Failed to convert/save image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = DigitDataCollectorNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ensure robot stops turning on node exit
        stop_twist = Twist()
        node.cmd_vel_pub.publish(stop_twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
