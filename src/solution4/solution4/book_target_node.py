from collections import deque
import math
import os

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    ReliabilityPolicy,
    DurabilityPolicy
)

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from sensor_msgs.msg import Image, LaserScan, Imu
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Time
from ament_index_python.packages import get_package_share_directory


class BookTargetNode(Node):

    def __init__(self):
        super().__init__(
            'book_target_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        # 1. Launch Parameters
        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter('book_colour', 'red')

        raw_shelf = self.get_parameter('shelf_column_number').value
        raw_colour = self.get_parameter('book_colour').value

        if raw_shelf is None or raw_colour is None:
            self.get_logger().error("[INIT] Missing required launch parameters!")
            return

        self.shelf_column_number = str(raw_shelf)
        self.book_colour = str(raw_colour).strip().lower()

        # 2. ONNX Model Initialization
        try:
            package_share = get_package_share_directory('solution4')
            model_path = os.path.join(package_share, 'models', 'gazebo_digit_model.onnx')
            self.net = cv2.dnn.readNetFromONNX(model_path) if os.path.exists(model_path) else None
        except Exception as e:
            self.net = None
            self.get_logger().error(f"[INIT] Failed loading ONNX model: {e}")

        # State Variables & Tracking
        self.arms_ready = False
        self.processing_complete = False
        self.table_position = 'unknown'
        self.rotation_direction = 0
        self.is_rotating = False
        self.rotation_command_sent = False
        self.current_yaw = None
        self.rotation_start_yaw = None
        self.table_detection_started = None

        # Depth trigger state tracking
        self.depth_change_count = 0
        self.previous_patch_depth = None
        self.digit_recog_active = False
        self.target_column_box = None

        self.column_image_saved = False
        self.image1_path = None
        self.image2_path = None

        self.ROTATE_SPEED = 0.4
        self.SHELF_APPROACH_STANDOFF_M = 0.6
        self.shelf_approach_pose_sent = False

        # OpenCV Resources
        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.color_ranges = {
            'red': [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
            'blue': [((100, 100, 100), (140, 255, 255))],
            'green': [((40, 50, 50), (80, 255, 255))],
            'yellow': [((20, 100, 100), (40, 255, 255))],
        }

        # Publishers
        latched_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.point_pub = self.create_publisher(PointStamped, '/detected_book_pose', 10)
        self.arm_left_pub = self.create_publisher(JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.arm_right_pub = self.create_publisher(JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.shelf_approach_pub = self.create_publisher(PoseStamped, '/erc/shelf_approach_pose', latched_qos)

        # Synchronized Camera Subscribers
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.colour_sub = message_filters.Subscriber(
            self, Image, '/head_front_camera/head_front_camera/color/image_raw', qos_profile=camera_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/head_front_camera/head_front_camera/depth/image_rect_raw', qos_profile=camera_qos
        )
        self.camera_sync = message_filters.ApproximateTimeSynchronizer(
            [self.colour_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.camera_sync.registerCallback(self.rgbd_callback)

        # LiDAR & IMU Subscriptions
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.front_scan_history = deque(maxlen=5)
        self.rear_scan_history = deque(maxlen=5)
        self.create_subscription(LaserScan, '/scan_front_raw', self._lidar_callback, lidar_qos)
        self.create_subscription(LaserScan, '/scan_rear_raw', self._rear_lidar_callback, lidar_qos)
        self.create_subscription(Imu, '/base_imu', self._imu_callback, lidar_qos)

        self.arms_ready = False

        # Step 1 Execution: Move arms simultaneously
        self.arm_timer = self.create_timer(0.5, self._move_arms_simultaneously)

    # =========================================================================
    # STEP 1: MOVE ARMS SIMULTANEOUSLY
    # =========================================================================
    def _move_arms_simultaneously(self):
        self.arm_timer.cancel()
        stamp = Time(sec=0, nanosec=0)

        # Left Arm Trajectory
        msg_left = JointTrajectory()
        msg_left.header.stamp = stamp
        msg_left.joint_names = [f'arm_left_{i}_joint' for i in range(1, 8)]
        pt_l = JointTrajectoryPoint(positions=[0.0, 2.0, 0.0, -2.3, 0.0, 2.0, 0.0])
        pt_l.time_from_start.sec = 2
        msg_left.points = [pt_l]

        # Right Arm Trajectory
        msg_right = JointTrajectory()
        msg_right.header.stamp = stamp
        msg_right.joint_names = [f'arm_right_{i}_joint' for i in range(1, 8)]
        pt_r = JointTrajectoryPoint(positions=[0.0, 2.0, 0.0, -2.3, 0.0, 2.0, 0.0])
        pt_r.time_from_start.sec = 2
        msg_right.points = [pt_r]

        # Publish simultaneously
        self.arm_left_pub.publish(msg_left)
        self.arm_right_pub.publish(msg_right)
        self.arms_ready = True
        self.get_logger().info("[STEP 1] Both arms command sent simultaneously.")

        self.vision_timer = self.create_timer(2.0, self._start_rgbd_callback)

    # =========================================================================
    # MAIN SEQUENTIAL PIPELINE CALLBACK
    # =========================================================================
    def _start_rgbd_callback(self):
        self.vision_timer.cancel()

        # Register the 2-argument message filter callback HERE
        self.arms_ready = True
        
    def rgbd_callback(self, colour_msg, depth_msg):
        if not self.arms_ready:
            return

        self.vision_timer.cancel()
        self.arms_ready = True
        
        
        if self.processing_complete or not self.arms_ready:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(colour_msg, desired_encoding='bgr8')
        except Exception:
            return

        # ---------------------------------------------------------------------
        # STEP 2 & 3: CHECK RED BOX VIA RGB -> LIDAR FALLBACK -> DETERMINE MOTION
        # ---------------------------------------------------------------------
        if self.table_position == 'unknown':
             # Allow sensors to stabilize

            if self._detect_red_box_rgb(frame):
                self.table_position = 'front'
                self.rotation_direction = -1
            else:
                # LiDAR Fallback
                self.table_position = self._determine_table_via_lidar()
                if self.table_position == 'left':
                    self.rotation_direction = -1
                elif self.table_position == 'right':
                    self.rotation_direction = 1
                elif self.table_position == 'front':
                    self.rotation_direction = -1

                    return

            self.table_detection_started = None
            self.get_logger().info(f"[STEP 2/3] Table located at '{self.table_position}'.")
            self.is_rotating = True

        # Perform Base Rotation
        if self.is_rotating:
            twist = Twist()
            twist.angular.z = float(self.rotation_direction) * self.ROTATE_SPEED
            self.cmd_vel_pub.publish(twist)
            self.rotation_command_sent = True
            if self.rotation_start_yaw is None and self.current_yaw is not None:
                self.rotation_start_yaw = self.current_yaw

        # Do not process depth changes until a base-rotation command has
        # actually been published.
        if not self.rotation_command_sent:
            return

        # ---------------------------------------------------------------------
        # STEP 4: MONITOR DEPTH JUMPS 
        # ---------------------------------------------------------------------
        depth_jump = self._check_depth_patch_change(depth_msg)

        if depth_jump and self.depth_change_count < 2:
            self.depth_change_count += 1
            self.get_logger().info(
                f"[STEP 4] Depth jump #{self.depth_change_count} detected "
                "(>= 1.3m)."
            )

            if self.depth_change_count == 1:
                # First change: Enable Digit Recognition
                self.digit_recog_active = True
            elif self.depth_change_count == 2:
                if not self._base_rotated_90_degrees():
                    self.depth_change_count = 1
                    self.get_logger().info(
                        '[STEP 4] Second depth jump ignored until the base has '
                        'rotated 90 degrees.'
                    )
                    return

                self.depth_change_count = 2
                # STEP 9: Second change after 90 degrees: passed table/shelf.
                self._stop_robot()
                self.processing_complete = True
                self.get_logger().warn(
                    '[STEP 9] Second depth change detected after 90 degrees '
                    'of base rotation. Motion stopped.'
                )
                return

        # ---------------------------------------------------------------------
        # STEP 5 & 6: DIGIT RECOGNITION -> BOOK RECOGNITION
        # ---------------------------------------------------------------------
        if self.digit_recog_active and self.target_column_box is None:
            col_box = self._scan_for_column_digit(frame)
            if col_box is not None:
                self.target_column_box = col_box
                self._save_column_image(frame, col_box)
                self.get_logger().info(f"[STEP 5] Digit '{self.shelf_column_number}' recognized.")

        # Once digit is recognized, proceed to book recognition
        if self.target_column_box is not None:
            book_center = self._scan_for_target_book(frame, self.target_column_box)
            if book_center is not None:
                # -------------------------------------------------------------
                # STEP 7: PUBLISH DATA & COMPLETE
                # -------------------------------------------------------------
                self._stop_robot()

                # Publish Book Pixel Target
                pt = PointStamped()
                pt.header = colour_msg.header
                pt.point.x, pt.point.y, pt.point.z = float(book_center[0]), float(book_center[1]), 0.0
                self.point_pub.publish(pt)

                # Publish Shelf Approach Pose for Nav2 / Node 2
                self._publish_shelf_approach_pose()

                # Terminal Log Image Paths
                self._save_final_annotated_image(frame, self.target_column_box, book_center)
                self.processing_complete = True
                self.get_logger().info("[STEP 7] Target book identified and pose data published. Task Complete.")

    # =========================================================================
    # HELPER PROCESSING METHODS
    # =========================================================================
    def _detect_red_box_rgb(self, frame):
        if frame is None or frame.size == 0 or len(frame.shape) < 3:
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) | \
               cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 800:
            x, _, width, _ = cv2.boundingRect(largest)
            center_x = x + width / 2.0
            image_center_x = frame.shape[1] / 2.0
            if abs(center_x - image_center_x) <= frame.shape[1] * 0.6:
                self.get_logger().info(
                    f'[STEP 2] Red box detected (area={area:.0f}, '
                    f'center_x={center_x:.0f}).'
                )
                return True
        return False

    def _determine_table_via_lidar(self):
        # Allow sensors to stabilize
        if not self.front_scan_history and not self.rear_scan_history:
            self.get_logger().warn(
            '[STEP 3] LiDAR scan history is empty. Waiting for data...'
              )
            return 'none'

        front_left = self._average_sector_min(130.0, 5.0, self.front_scan_history)
        front_right = self._average_sector_min(-5.0, -130.0, self.front_scan_history)
       
        rear_left = self._average_sector_min(5.0, 130.0, self.rear_scan_history)
        rear_right = self._average_sector_min(-130.0, -5.0, self.rear_scan_history)
     

        left_dist = min(front_left, rear_left)
        right_dist = min(front_right, rear_right)
        threshold = 3.5

        candidates = {
            'left': left_dist,
            'right': right_dist,
            
        }
        valid_candidates = {
            position: distance
            for position, distance in candidates.items()
            if math.isfinite(distance) and distance < threshold
        }
        self.get_logger().info(
            f'[STEP 3] LiDAR distances: left={left_dist:.2f}m, '
            f'right={right_dist:.2f}m, '
          
            f'valid={list(valid_candidates)}',
            throttle_duration_sec=2.0,
        )
        if valid_candidates:
            return min(valid_candidates, key=valid_candidates.get)
        return 'none'

    def _check_depth_patch_change(self, depth_msg):
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        h, w = depth_img.shape[:2]
        patch = depth_img[max(0, h // 2 - 10): min(h, h // 2 + 10), max(0, w - 20): w].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return False

        curr_depth = float(np.median(valid))
        if depth_msg.encoding == '16UC1':
            curr_depth /= 1000.0

        if self.previous_patch_depth is None:
            self.previous_patch_depth = curr_depth
            return False

        diff = abs(curr_depth - self.previous_patch_depth)
        self.previous_patch_depth = curr_depth
        return diff >= 1.3

    def _deskew_roi(self, gray_roi, pts):
        """Aligns rotated text contours straight before feeding to the CNN."""
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        h, w = gray_roi.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            gray_roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def _scan_for_column_digit(self, frame):
        if self.net is None or frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        mser = cv2.MSER_create(min_area=60, max_area=14000)
        try:
            regions, _ = mser.detectRegions(enhanced)
        except cv2.error:
            return None
        
        for pts in regions:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
            if 10 < h < 150 and 0.2 < (w / float(h)) < 1.4:
                roi_gray = enhanced[y:y + h, x:x + w]
                deskew_roi = self._deskew_roi(roi_gray, pts)
                digit = self._predict_digit(deskew_roi)
                if digit == self.shelf_column_number:
                    return (x, y, w, h)
        return None

    def _predict_digit(self, roi):
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = thresh.shape[:2]
        scale = 20.0 / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[(28 - nh) // 2: (28 - nh) // 2 + nh, (28 - nw) // 2: (28 - nw) // 2 + nw] = cv2.resize(thresh, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (28, 28))
        self.net.setInput(blob)
        probs = self.net.forward()[0]
        pred = int(np.argmax(probs))
        return str(pred) if 1 <= pred <= 5 else None

    def _scan_for_target_book(self, frame, col_box):
        x, y, w, h = col_box
        fh, fw = frame.shape[:2]
        roi = frame[min(y + h, fh): min(y + h + 180, fh), max(x - 30, 0): min(x + w + 30, fw)]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in self.color_ranges.get(self.book_colour, []):
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= 100.0]
        if valid:
            bx, by, bw, bh = cv2.boundingRect(max(valid, key=cv2.contourArea))
            return (max(x - 30, 0) + bx + bw // 2, min(y + h, fh) + by + bh // 2)
        return None

    def _publish_shelf_approach_pose(self):
        fwd_dist = self._average_sector_min(-10.0, 10.0)
        approach = max(0.0, fwd_dist - self.SHELF_APPROACH_STANDOFF_M) if math.isfinite(fwd_dist) else 0.0

        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(approach)
        pose.pose.orientation.w = 1.0
        self.shelf_approach_pub.publish(pose)

    def _stop_robot(self):
        self.is_rotating = False
        self.cmd_vel_pub.publish(Twist())

    def _lidar_callback(self, msg):
        self.front_scan_history.append(msg)

    def _rear_lidar_callback(self, msg):
        self.rear_scan_history.append(msg)

    def _imu_callback(self, msg):
        siny_cosp = 2.0 * (msg.orientation.w * msg.orientation.z
                           + msg.orientation.x * msg.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (msg.orientation.y * msg.orientation.y
                                 + msg.orientation.z * msg.orientation.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        if (
            self.rotation_command_sent
            and self.is_rotating
            and self.rotation_start_yaw is None
        ):
            self.rotation_start_yaw = self.current_yaw

    def _base_rotated_90_degrees(self):
        if self.current_yaw is None or self.rotation_start_yaw is None:
            return False

        yaw_change = self.current_yaw - self.rotation_start_yaw
        while yaw_change > math.pi:
            yaw_change -= 2.0 * math.pi
        while yaw_change < -math.pi:
            yaw_change += 2.0 * math.pi
        return abs(math.degrees(yaw_change)) >= 90.0

    def _average_sector_min(self, min_deg, max_deg, scan_history=None):
        if scan_history is None:
            scan_history = self.front_scan_history
        if not scan_history:
            return float('inf')
        ranges = []
        for scan in scan_history:
            if scan.angle_increment <= 0.0:
                continue
            ranges_array = np.asarray(scan.ranges, dtype=np.float32)
            angles = scan.angle_min + np.arange(len(ranges_array)) * scan.angle_increment
            mask = (
                (angles >= np.radians(min_deg))
                & (angles <= np.radians(max_deg))
                & np.isfinite(ranges_array)
                & (ranges_array >= max(scan.range_min, 0.3))
                & (ranges_array <= scan.range_max)
            )
            r = ranges_array[mask]
            if len(r) > 0:
                ranges.append(np.min(r))
        return float(np.mean(ranges)) if ranges else float('inf')

    def _save_column_image(self, frame, col_box):
        x, y, w, h = col_box
        img = frame.copy()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        filename = f"detected_column_{self.get_clock().now().nanoseconds}.png"
        cv2.imwrite(filename, img)
        self.image1_path = os.path.abspath(filename)
        self.get_logger().info(f"[STEP 5] shelf column image---> '{filename}' saved.")
        

    def _save_final_annotated_image(self, frame, col_box, book_center):
        img = frame.copy()
        x, y, w, h = col_box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, book_center, 5, (0, 0, 255), -1)
        filename = f"detected_book_{self.get_clock().now().nanoseconds}.png"
        cv2.imwrite(filename, img)
        self.image2_path = os.path.abspath(filename)
        self.get_logger().info(f"[LOG] Image 1: {self.image1_path} | Image 2: {self.image2_path}")


def main(args=None):
    rclpy.init(args=args)
    node = BookTargetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()