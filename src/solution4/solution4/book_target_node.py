from collections import deque
import datetime
import math
import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    ReliabilityPolicy,
)
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
from std_msgs.msg import Int32
import tf2_geometry_msgs  # noqa: F401  - registers PointStamped with the TF buffer
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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
            self.get_logger().error('[INIT] Missing required launch parameters!')
            return

        self.shelf_column_number = str(raw_shelf)
        self.book_colour = str(raw_colour).strip().lower()

        # Row numbering base. The Phase 1 PDF asks for "the shelf row (1-4)",
        # while the simulator names those same four rows 2..5 internally
        # (ACTIVE_ROWS = [1,2,3,4] rendered as row_{i+1}). We publish 1 for the
        # topmost active row; set this parameter to 2 if the organisers confirm
        # the other convention.
        self.declare_parameter('row_index_base', 1)
        self.row_index_base = int(self.get_parameter('row_index_base').value)

        # Height of each active row's books above the floor, topmost first.
        # From simulation.launch.py: books spawn at SHELF_Z (1.1) plus the
        # row's offset, and ACTIVE_ROWS covers offsets 0.495, 0.165, -0.165
        # and -0.495. The rows are 0.33 m apart and a book is 0.16 m tall, so
        # a book's height names its row outright - which is what makes this
        # work when only some of the column is visible.
        self.declare_parameter('row_heights', [1.595, 1.265, 0.935, 0.605])
        self.declare_parameter('row_height_tolerance', 0.16)
        # Frame to measure that height in. base_footprint sits on the floor,
        # so a point's z in it is height above the floor.
        self.declare_parameter('row_reference_frame', 'base_footprint')
        self.row_heights = [float(v) for v in self.get_parameter('row_heights').value]
        self.row_height_tolerance = float(self.get_parameter('row_height_tolerance').value)
        self.row_reference_frame = str(self.get_parameter('row_reference_frame').value)

        # Digit-detection gating. The classifier was trained to tell digits
        # apart, not to decide whether a region is a digit at all, so it labels
        # whatever it is handed with high confidence. What keeps it honest is
        # never showing it anything but the inside of a confirmed marker plate;
        # the floor and the agreement across frames are a backstop, not the
        # discriminator.
        self.declare_parameter('digit_confidence', 0.90)
        self.declare_parameter('digit_confirm_frames', 3)
        # Fraction of the frame height, measured from the top, that the digit
        # search looks at. The markers sit at z = 2.26 m so this is a cheap way
        # to skip most of the frame; it is no longer what rejects false
        # positives, and can be widened to 1.0 at the cost of some work.
        self.declare_parameter('digit_search_band', 0.5)

        # Marker plate geometry. Each plate is a 0.3 m square whose texture is
        # a tight crop of the digit stretched to fill it, so the glyph covers
        # roughly 0.78 of the plate's width and 0.83 of its height and sits
        # centred, with only a thin margin around it. Confirming a plate means
        # segmenting its flat face, checking the result is a quadrilateral of
        # about the right shape, and checking the glyph fills it in that
        # proportion - tests the grippers, the shelf uprights and the lettering
        # on the hall banner all fail on shape rather than on brightness.
        #
        # The tolerance is tight because it has to be: the plate face and the
        # shelf top it rests on are only about eleven grey levels apart, so a
        # looser band grows straight through the join and swallows the shelf.
        # It is safe to be that tight because the value is measured from the
        # plate itself each time, never assumed.
        self.declare_parameter('glyph_max_level', 90)       # a printed glyph is near-black
        self.declare_parameter('glyph_max_fill', 0.80)      # a solid block is not a digit
        self.declare_parameter('plate_tolerance', 8.0)
        self.declare_parameter('plate_min_contrast', 40.0)
        self.declare_parameter('plate_min_rect_fill', 0.80)  # a plate fills its own bounding rect
        self.declare_parameter('plate_aspect_min', 0.55)
        self.declare_parameter('plate_aspect_max', 1.90)
        self.declare_parameter('plate_glyph_span_min', 0.55)
        self.declare_parameter('plate_glyph_span_max', 0.97)
        self.declare_parameter('plate_glyph_offset', 0.15)   # glyph must sit near the centre
        self.declare_parameter('plate_frame_margin', 2)

        self.digit_confidence = float(self.get_parameter('digit_confidence').value)
        self.digit_confirm_frames = int(self.get_parameter('digit_confirm_frames').value)
        self.digit_search_band = float(self.get_parameter('digit_search_band').value)
        self.glyph_max_level = int(self.get_parameter('glyph_max_level').value)
        self.glyph_max_fill = float(self.get_parameter('glyph_max_fill').value)
        self.plate_tolerance = float(self.get_parameter('plate_tolerance').value)
        self.plate_min_contrast = float(self.get_parameter('plate_min_contrast').value)
        self.plate_min_rect_fill = float(self.get_parameter('plate_min_rect_fill').value)
        self.plate_aspect = (
            float(self.get_parameter('plate_aspect_min').value),
            float(self.get_parameter('plate_aspect_max').value),
        )
        self.plate_glyph_span = (
            float(self.get_parameter('plate_glyph_span_min').value),
            float(self.get_parameter('plate_glyph_span_max').value),
        )
        self.plate_glyph_offset = float(self.get_parameter('plate_glyph_offset').value)
        self.plate_frame_margin = int(self.get_parameter('plate_frame_margin').value)
        self._digit_streak = 0
        self._last_digit_confidence = 0.0

        # 2. ONNX Model Initialization
        self.net: Any = None
        try:
            package_share = get_package_share_directory('solution4')
            model_path = os.path.join(package_share, 'models', 'gazebo_digit_model.onnx')
            self.net = cv2.dnn.readNetFromONNX(model_path) if os.path.exists(model_path) else None
        except Exception as e:
            self.net = None
            self.get_logger().error(f'[INIT] Failed loading ONNX model: {e}')

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

        # Scoring-topic latch state (publish once per trial)
        self.column_id_published = False
        self.row_id_published = False

        self.ROTATE_SPEED = 0.4
        # Stand-off is measured to the marker plate, which is flush with the
        # shelf edge, but the books sit 0.145 m further in (simulation.launch
        # puts the plate at SHELF_X - 0.245 and the books at SHELF_X - 0.1).
        # A 0.6 m stand-off therefore left the book 0.75 m away before any
        # measurement error, and the arm reaches about 0.8 m at full stretch:
        # a trial parked with the book at 1.76 m and no IK solution existed.
        # This is deliberately shorter than the LiDAR stop distance in node 2,
        # so the LiDAR is what actually halts the approach.
        # Stand-off is measured to the marker plate, which sits flush with the
        # shelf edge, while the books are 0.145 m further in. In practice the
        # shelf itself stops the base at about 0.7 m whatever this says, so it
        # mainly governs how firmly the robot commits to closing the gap.
        self.SHELF_APPROACH_STANDOFF_M = 0.35
        # Where the identified column stands, in the base frame. Set once
        # the column is recognised and used to aim the approach pose so the
        # base arrives in front of that column rather than wherever it
        # happened to be facing.
        self.column_point_base = None
        self.camera_k = None
        self.shelf_approach_pose_sent = False

        # Annotated images must land in erc_images/ inside the team repository
        # (Phase 1 spec, "Saving Images").
        #
        # Only src/ is bind-mounted into the container, so anything written to
        # the container's working directory is lost when the container is
        # recreated. Resolving from __file__ instead puts the images on the
        # mounted tree, where they persist and can be committed. Requires
        # --symlink-install, which the README mandates anyway; falls back to
        # the working directory if that ever stops holding.
        self.images_dir = os.environ.get('ERC_IMAGES_DIR') or self._default_images_dir()
        try:
            os.makedirs(self.images_dir, exist_ok=True)
        except OSError as e:
            self.get_logger().error(f'[INIT] Cannot create {self.images_dir}: {e}')

        # OpenCV Resources
        self.bridge = CvBridge()
        self.color_ranges = {
            'red': [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
            'blue': [((100, 100, 100), (140, 255, 255))],
            'green': [((40, 50, 50), (80, 255, 255))],
            'yellow': [((20, 100, 100), (40, 255, 255))],
        }

        # Publishers
        latched_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.point_pub = self.create_publisher(PointStamped, '/detected_book_pose', 10)
        self.arm_left_pub = self.create_publisher(
            JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.arm_right_pub = self.create_publisher(
            JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.shelf_approach_pub = self.create_publisher(
            PoseStamped, '/erc/shelf_approach_pose', latched_qos)

        # Competition scoring topics (Phase 1 rubric, Table 1: +1 point each).
        # Latched so a monitor that subscribes after we publish still receives
        # the value.
        self.column_id_pub = self.create_publisher(
            Int32, '/erc/shelf_column_identification', latched_qos
        )
        self.row_id_pub = self.create_publisher(
            Int32, '/erc/shelf_row_identification', latched_qos
        )

        # Synchronized Camera Subscribers
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.colour_sub = message_filters.Subscriber(
            self, Image, '/head_front_camera/head_front_camera/color/image_raw',
            qos_profile=camera_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/head_front_camera/head_front_camera/depth/image_rect_raw',
            qos_profile=camera_qos
        )
        self.camera_sync = message_filters.ApproximateTimeSynchronizer(
            [self.colour_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.camera_sync.registerCallback(self.rgbd_callback)

        # Intrinsics, for turning the marker plate's pixel into a bearing.
        self.create_subscription(
            CameraInfo,
            '/head_front_camera/head_front_camera/depth/camera_info',
            self._on_camera_info,
            camera_qos,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # LiDAR & IMU Subscriptions
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.front_scan_history: deque = deque(maxlen=5)
        self.rear_scan_history: deque = deque(maxlen=5)
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
        self.get_logger().info('[STEP 1] Both arms command sent simultaneously.')

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
                # LiDAR Fallback. The scan history is empty for the first few
                # frames after start-up; latching the 'none' it returns then
                # would leave rotation_direction at 0, so the base never turns
                # and the trial deadlocks. Stay 'unknown' and retry instead.
                detected = self._determine_table_via_lidar()
                if detected not in ('left', 'right', 'front'):
                    self.get_logger().info(
                        '[STEP 3] Table position not resolved yet - retrying '
                        'on the next frame.',
                        throttle_duration_sec=2.0,
                    )
                    return

                self.table_position = detected
                if self.table_position == 'left':
                    self.rotation_direction = -1
                elif self.table_position == 'right':
                    self.rotation_direction = 1
                elif self.table_position == 'front':
                    self.rotation_direction = -1

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
        # Once the target column is in hand the search is over; the "rotated
        # past the shelf" terminator below must not fire and end the trial
        # while we are standing still reading the books.
        depth_jump = (
            self._check_depth_patch_change(depth_msg)
            if self.target_column_box is None
            else False
        )

        if depth_jump and self.depth_change_count < 2:
            self.depth_change_count += 1
            self.get_logger().info(
                f'[STEP 4] Depth jump #{self.depth_change_count} detected '
                '(>= 1.3m).'
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
            if col_box is None:
                self._digit_streak = 0
            else:
                self._digit_streak += 1

            if col_box is not None and self._digit_streak < self.digit_confirm_frames:
                self.get_logger().info(
                    f"[STEP 5] Candidate digit '{self.shelf_column_number}' "
                    f'(confidence {self._last_digit_confidence:.2f}), '
                    f'{self._digit_streak}/{self.digit_confirm_frames} frames - '
                    'keep looking.',
                    throttle_duration_sec=1.0,
                )
                col_box = None

            if col_box is not None:
                self.target_column_box = col_box
                # Stop the base the moment the column is identified. The box is
                # captured once from this frame; if the base keeps rotating,
                # every later frame has the column somewhere else and the book
                # search below ends up examining a stale region of a moving
                # image - which is why the run used to spin past the shelf
                # without ever finding the book.
                self._stop_robot()
                self.column_point_base = self._column_point_in_base(depth_msg, col_box)
                self._publish_column_identification()
                self._save_column_image(frame, col_box)
                self.get_logger().info(f"[STEP 5] Digit '{self.shelf_column_number}' recognized.")

        # Once digit is recognized, proceed to book recognition
        if self.target_column_box is not None:
            books = self._scan_column_books(frame, self.target_column_box)
            target_box = books.get(self.book_colour)
            if target_box is not None:
                bx, by, bw, bh = target_box
                book_center = (bx + bw // 2, by + bh // 2)

                # -------------------------------------------------------------
                # STEP 7: PUBLISH DATA & COMPLETE
                # -------------------------------------------------------------
                self._stop_robot()

                # Row comes from the book's height where the depth allows
                # it, because that holds even when the rest of the column is
                # hidden. Ranking is the fallback, and says so in the log.
                row = self._row_from_height(depth_msg, target_box)
                if row is None:
                    row = self._row_for_colour(books, self.book_colour)
                if row is not None:
                    self._publish_row_identification(row)

                # Publish Book Pixel Target
                pt = PointStamped()
                pt.header = colour_msg.header
                pt.point.x, pt.point.y, pt.point.z = (
                    float(book_center[0]), float(book_center[1]), 0.0)
                self.point_pub.publish(pt)

                # Publish Shelf Approach Pose for Nav2 / Node 2
                self._publish_shelf_approach_pose()

                # Terminal Log Image Paths
                self._save_final_annotated_image(frame, self.target_column_box, target_box, row)
                self.processing_complete = True
                self.get_logger().info(
                    '[STEP 7] Target book identified and pose data published. Task Complete.')

    @staticmethod
    def _default_images_dir():
        """Locate erc_images/ on the bind-mounted source tree.

        This file lives at <src>/solution4/solution4/book_target_node.py, so
        three levels up is the mounted repository content.
        """
        try:
            # colcon --symlink-install executes this module from build/, so
            # __file__ must be resolved through the symlink to reach the real
            # source path before walking up to the workspace's src/.
            path = os.path.dirname(os.path.realpath(__file__))
            while True:
                path, tail = os.path.split(path)
                if not tail:
                    break
                if tail == 'src':
                    return os.path.join(path, 'src', 'erc_images')
        except (OSError, NameError):
            pass
        return os.path.join(os.getcwd(), 'erc_images')

    # =========================================================================
    # COMPETITION SCORING PUBLISHERS
    # =========================================================================
    def _publish_column_identification(self):
        """Publish the identified shelf column (+1 point, rubric Table 1)."""
        if self.column_id_published:
            return
        try:
            value = int(self.shelf_column_number)
        except (TypeError, ValueError):
            self.get_logger().error(
                f"[SCORE] Cannot publish column id: '{self.shelf_column_number}' is not an int."
            )
            return
        self.column_id_pub.publish(Int32(data=value))
        self.column_id_published = True
        self.get_logger().info(
            f'[SCORE] Published {value} to /erc/shelf_column_identification'
        )

    def _publish_row_identification(self, row):
        """Publish the identified shelf row (+1 point, rubric Table 1)."""
        if self.row_id_published:
            return
        self.row_id_pub.publish(Int32(data=int(row)))
        self.row_id_published = True
        self.get_logger().info(
            f'[SCORE] Published {row} to /erc/shelf_row_identification'
        )

    # =========================================================================
    # HELPER PROCESSING METHODS
    # =========================================================================
    def _detect_red_box_rgb(self, frame):
        if frame is None or frame.size == 0 or len(frame.shape) < 3:
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
                | cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])))
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
            return min(valid_candidates, key=lambda side: valid_candidates[side])
        return 'none'

    def _check_depth_patch_change(self, depth_msg):
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        h, w = depth_img.shape[:2]
        patch = depth_img[max(0, h // 2 - 10): min(h, h // 2 + 10),
                          max(0, w - 20): w].astype(np.float32)
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

    # Side of the square a confirmed plate is rectified onto before reading.
    PLATE_RECTIFIED_PX = 64

    def _glyph_blobs(self, gray):
        """Near-black connected regions that could be a printed glyph."""
        dark = (gray < self.glyph_max_level).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area < 40 or w < 4 or h < 6:
                continue
            if not 0.15 < w / float(h) < 1.5:
                continue
            # A glyph leaves gaps inside its bounding box; a shelf upright or a
            # shadow fills its own box almost completely.
            if area / float(w * h) > self.glyph_max_fill:
                continue
            yield (x, y, w, h), (labels == i)

    def _plate_grey(self, gray, glyph):
        """Return the plate's own grey, read from a ring just clear of the glyph.

        Measured every time rather than assumed, because how bright a plate
        renders depends on where the robot is standing. Sampled a couple of
        pixels out so the glyph's anti-aliased rim does not drag it down.
        """
        kernel = np.ones((3, 3), np.uint8)
        outer = cv2.dilate(glyph.astype(np.uint8), kernel, iterations=3).astype(bool)
        inner = cv2.dilate(glyph.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = gray[outer & ~inner]
        return float(np.median(ring)) if ring.size >= 12 else None

    @staticmethod
    def _order_quad(pts):
        """Corners as top-left, top-right, bottom-right, bottom-left."""
        total, diff = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
        return np.array([pts[np.argmin(total)], pts[np.argmin(diff)],
                         pts[np.argmax(total)], pts[np.argmax(diff)]], dtype=np.float32)

    def _find_marker_plates(self, gray):
        """Locate the 0.3 m marker plates, before anything is classified.

        Measured on the raw greyscale, never a CLAHE-equalised copy: a plate
        reads as a flat grey distinctly darker than the wall behind it, and
        equalisation destroys exactly that separation.

        Returns a list of dicts with the plate's bounding box, its corners and
        the rectified square that the classifier is later shown.
        """
        height, width = gray.shape[:2]
        margin = self.plate_frame_margin
        side = self.PLATE_RECTIFIED_PX
        found = []

        for (x, y, w, h), glyph in self._glyph_blobs(gray):
            # The plate always extends past the glyph, so a glyph touching the
            # frame edge sits on a plate that is certainly cut off. Such a
            # marker is useless even when read correctly - the shelf column
            # beneath it is off-screen, leaving the book search nothing to rank.
            if (x <= margin or y <= margin
                    or x + w >= width - margin or y + h >= height - margin):
                continue
            plate_grey = self._plate_grey(gray, glyph)
            if plate_grey is None:
                continue

            # Grow the plate's face out from the glyph, inside a window a
            # couple of glyph-widths across so a same-coloured surface further
            # off cannot be joined onto it.
            pad = int(max(w, h) * 0.8) + 6
            x0, x1 = max(0, x - pad), min(width, x + w + pad)
            y0, y1 = max(0, y - pad), min(height, y + h + pad)
            window = gray[y0:y1, x0:x1]
            face = (np.abs(window.astype(np.int16) - plate_grey) <= self.plate_tolerance)
            face |= glyph[y0:y1, x0:x1]
            face = cv2.morphologyEx(
                face.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )

            count, labels, stats, _ = cv2.connectedComponentsWithStats(face, 8)
            seed = labels[y - y0 + h // 2, x - x0 + w // 2]
            if seed == 0:
                ids = labels[glyph[y0:y1, x0:x1]]
                ids = ids[ids > 0]
                if ids.size == 0:
                    continue
                seed = int(np.bincount(ids).argmax())
            contours, _ = cv2.findContours(
                (labels == seed).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if rw < 8 or rh < 8:
                continue
            aspect = rw / rh
            # A plate is solid, so it fills its own minimum-area rectangle. A
            # region that merely happens to be the right colour does not.
            rect_fill = cv2.contourArea(contour) / (rw * rh)
            if not self.plate_aspect[0] <= aspect <= self.plate_aspect[1]:
                continue
            if rect_fill < self.plate_min_rect_fill:
                continue

            # Rectify the plate onto a square. It is one in the world, so this
            # undoes the tilt and foreshortening of a plate seen from an angle
            # and presents near and far markers at the same scale. It also
            # makes the checks below rotation-invariant.
            quad = self._order_quad(cv2.boxPoints(rect) + np.array([x0, y0], np.float32))
            target = np.array([[0, 0], [side - 1, 0],
                               [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
            flat = cv2.warpPerspective(
                gray, cv2.getPerspectiveTransform(quad, target), (side, side)
            )

            ink = (flat < self.glyph_max_level).astype(np.uint8)
            gcount, glabels, gstats, _ = cv2.connectedComponentsWithStats(ink, 8)
            if gcount < 2:
                continue
            biggest = 1 + int(np.argmax(gstats[1:, cv2.CC_STAT_AREA]))
            gx, gy, gw, gh = gstats[biggest, :4]
            span_x, span_y = gw / float(side), gh / float(side)
            off_x = abs((gx + gw / 2.0) / side - 0.5)
            off_y = abs((gy + gh / 2.0) / side - 0.5)

            body = (glabels == biggest)
            around = flat[~cv2.dilate(
                body.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2
            ).astype(bool)]
            if around.size < 30:
                continue
            contrast = float(np.median(around)) - float(np.median(flat[body]))

            if contrast < self.plate_min_contrast:
                continue
            if not self.plate_glyph_span[0] <= span_x <= self.plate_glyph_span[1]:
                continue
            if not self.plate_glyph_span[0] <= span_y <= self.plate_glyph_span[1]:
                continue
            if off_x > self.plate_glyph_offset or off_y > self.plate_glyph_offset:
                continue

            found.append({
                'bbox': cv2.boundingRect(quad.astype(np.int32)),
                'quad': quad,
                'flat': flat,
            })
        return found

    def _read_plate(self, flat):
        """Classify the glyph on a rectified plate, or None if unconvincing."""
        _, ink = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        if count < 2:
            return None
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = stats[biggest, :4]
        glyph = (labels[y:y + h, x:x + w] == biggest).astype(np.uint8) * 255
        return self._predict_digit(glyph)

    def _scan_for_column_digit(self, frame):
        """Return the bounding box of the plate carrying the column number.

        Plates are confirmed geometrically first and the classifier only ever
        sees the inside of one, so a region that is not a marker is rejected on
        its shape rather than on the network's opinion of it.
        """
        if self.net is None or frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_height = max(1, int(gray.shape[0] * self.digit_search_band))
        band = gray[0:search_height, :]

        for plate in self._find_marker_plates(band):
            if self._read_plate(plate['flat']) == self.shelf_column_number:
                # The band starts at row 0, so band and frame coordinates agree.
                return plate['bbox']
        return None

    def _predict_digit(self, glyph):
        """Classify an isolated glyph, rejecting anything unconvincing.

        The network assigns every region it is shown some class, so the
        confidence floor stays as a backstop. It is no longer what keeps
        non-digits out - _find_marker_plates is, by never handing this
        anything that is not the inside of a marker plate.
        """
        h, w = glyph.shape[:2]
        scale = 20.0 / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[(28 - nh) // 2: (28 - nh) // 2 + nh,
               (28 - nw) // 2: (28 - nw) // 2 + nw] = cv2.resize(glyph, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (28, 28))
        self.net.setInput(blob)

        logits = self.net.forward()[0].astype(np.float64)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)

        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        if not 1 <= pred <= 5:
            return None
        if confidence < self.digit_confidence:
            return None
        self._last_digit_confidence = confidence
        return str(pred)

    def _scan_column_books(self, frame, col_box):
        """Locate every coloured book sitting under the target column marker.

        Each column holds exactly four books - one of each colour, one per
        active row - so ranking what we find top-to-bottom by pixel y gives the
        row index directly, with no camera intrinsics or TF lookup needed.

        Returns {colour: (x, y, w, h)} in full-frame coordinates.
        """
        x, y, w, h = col_box
        fh, fw = frame.shape[:2]
        top = min(y + h, fh)
        # The four active rows span roughly a metre of shelf; 420 px covers
        # them at the distances this runs at, where the old 180 px could clip
        # the lower rows out of the search.
        bottom = min(y + h + 420, fh)
        left = max(x - 30, 0)
        right = min(x + w + 30, fw)

        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            return {}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        found = {}
        for colour, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low, high in ranges:
                mask |= cv2.inRange(hsv, np.array(low), np.array(high))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if cv2.contourArea(c) >= 100.0]
            if not valid:
                continue
            bx, by, bw, bh = cv2.boundingRect(max(valid, key=cv2.contourArea))
            found[colour] = (left + bx, top + by, bw, bh)
        return found

    def _row_for_colour(self, books, colour):
        """Fallback row estimate: rank the detected books top to bottom.

        Only correct when all four books in the column are visible, since a
        missing one shifts every book below it up a rank. Used when the
        book's height could not be measured - see _row_from_height.
        """
        if colour not in books:
            return None

        ordered = sorted(books.items(), key=lambda item: item[1][1])
        if len(ordered) < 4:
            self.get_logger().warn(
                f'[SCORE] Only {len(ordered)} of 4 books visible in the column - '
                'the row index is an estimate.'
            )
        for index, (name, _) in enumerate(ordered):
            if name == colour:
                return index + self.row_index_base
        return None

    def _publish_shelf_approach_pose(self):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.orientation.w = 1.0

        if self.column_point_base is not None:
            # Stand off in front of the column itself. The base is holonomic,
            # so the sideways component costs nothing and is what actually
            # earns "navigate to the column" - driving straight ahead only
            # reaches whichever column the robot happened to stop facing.
            forward = self.column_point_base.point.x - self.SHELF_APPROACH_STANDOFF_M
            pose.pose.position.x = float(max(0.0, forward))
            pose.pose.position.y = float(self.column_point_base.point.y)
        else:
            # Fallback: no depth or no TF for the plate, so approach on LiDAR
            # alone and accept whatever column that lands in front of.
            fwd_dist = self._average_sector_min(-10.0, 10.0)
            approach = (max(0.0, fwd_dist - self.SHELF_APPROACH_STANDOFF_M)
                        if math.isfinite(fwd_dist) else 0.0)
            pose.pose.position.x = float(approach)
            self.get_logger().warn(
                '[STEP 7] Column position unknown - approaching straight ahead.'
            )

        self.get_logger().info(
            f'[STEP 7] Approach pose: x={pose.pose.position.x:.2f}m, '
            f'y={pose.pose.position.y:+.2f}m in base_link.'
        )
        self.shelf_approach_pub.publish(pose)

    def _on_camera_info(self, msg):
        self.camera_k = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy

    def _deproject(self, depth_msg, box, target_frame):
        """Turn the centre of a pixel box into a 3-D point in `target_frame`.

        Depth and colour are published at the same resolution and with the same
        intrinsics here, so a pixel in the colour image indexes the depth image
        directly. Returns None if the depth there is missing or TF is not ready.
        """
        if self.camera_k is None:
            self.get_logger().warn('[DEPTH] No camera_info yet.', throttle_duration_sec=5.0)
            return None

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warn(f'[DEPTH] Image unreadable: {exc}')
            return None

        x, y, w, h = box
        u, v = x + w // 2, y + h // 2
        dh, dw = depth.shape[:2]
        if not (0 <= u < dw and 0 <= v < dh):
            return None

        # Median of a small patch, so one dropped pixel at an edge does not
        # decide the answer.
        half = 3
        patch = depth[max(0, v - half):min(dh, v + half + 1),
                      max(0, u - half):min(dw, u + half + 1)].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
        if valid.size < 4:
            self.get_logger().warn('[DEPTH] No usable depth at that pixel.')
            return None
        z = float(np.median(valid))

        fx, fy, cx, cy = self.camera_k
        point = PointStamped()
        point.header.frame_id = depth_msg.header.frame_id
        # Zero stamp asks for the latest transform. The base is already stopped
        # by the time this runs, so latest and exact agree.
        point.header.stamp = RclpyTime().to_msg()
        point.point.x = (u - cx) * z / fx
        point.point.y = (v - cy) * z / fy
        point.point.z = z

        try:
            return self.tf_buffer.transform(
                point, target_frame, timeout=RclpyDuration(seconds=2.0)
            )
        except Exception as exc:
            self.get_logger().warn(
                f'[DEPTH] {point.header.frame_id} -> {target_frame} failed: {exc}'
            )
            return None

    def _column_point_in_base(self, depth_msg, col_box):
        """Where the identified column stands, as a point in the base frame.

        The marker plate sits directly above its column, so deprojecting the
        plate's centre gives the column's position without needing to see the
        shelf itself.
        """
        in_base = self._deproject(depth_msg, col_box, 'base_link')
        if in_base is None:
            return None
        self.get_logger().info(
            f'[STEP 7] Column {self.shelf_column_number} is at '
            f'x={in_base.point.x:.2f}m, y={in_base.point.y:+.2f}m in the base frame.'
        )
        return in_base

    def _row_from_height(self, depth_msg, book_box):
        """Which row a book is on, from how far off the floor it sits.

        The four active rows are fixed heights 0.33 m apart, so a book's height
        names its row on its own. Ranking the column's books top to bottom
        needs all four of them in view, and one arm across the frame is enough
        to break that - which is what the "row index is an estimate" warning
        has been reporting.
        """
        point = self._deproject(depth_msg, book_box, self.row_reference_frame)
        if point is None:
            return None

        height = point.point.z
        nearest = min(range(len(self.row_heights)),
                      key=lambda i: abs(self.row_heights[i] - height))
        error = abs(self.row_heights[nearest] - height)
        if error > self.row_height_tolerance:
            self.get_logger().warn(
                f'[SCORE] Book sits {height:.2f} m up, {error:.2f} m off the nearest '
                'row - too far to call from height.'
            )
            return None

        row = nearest + self.row_index_base
        self.get_logger().info(
            f'[SCORE] Book is {height:.2f} m off the floor, {error:.2f} m from the '
            f'row at {self.row_heights[nearest]:.2f} m -> row {row}.'
        )
        return row

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

    # =========================================================================
    # ANNOTATED IMAGE OUTPUT
    # =========================================================================
    def _stamp_image(self, img):
        """Burn a timestamp into the frame.

        The spec requires every saved image to carry one so the committee can
        verify it was captured live during the trial rather than prepared
        beforehand. We write both wall-clock and simulation time.
        """
        sim_seconds = self.get_clock().now().nanoseconds / 1e9
        wall = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        label = f'{wall}   sim_t={sim_seconds:.3f}s'

        height, width = img.shape[:2]
        cv2.rectangle(img, (0, height - 26), (width, height), (0, 0, 0), -1)
        cv2.putText(
            img, label, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        return img

    def _write_image(self, img, prefix):
        filename = os.path.join(
            self.images_dir, f'{prefix}_{self.get_clock().now().nanoseconds}.png'
        )
        if not cv2.imwrite(filename, img):
            self.get_logger().error(f'[IMAGE] Failed to write {filename}')
            return None
        return os.path.abspath(filename)

    def _save_column_image(self, frame, col_box):
        """Bounding box around the target shelf column (+2 points)."""
        x, y, w, h = col_box
        img = frame.copy()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, f'column {self.shelf_column_number}', (x, max(14, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
        self._stamp_image(img)

        self.image1_path = self._write_image(img, 'detected_column')
        if self.image1_path:
            self.column_image_saved = True
            self.get_logger().info(f'[STEP 5] Column image saved: {self.image1_path}')

    def _save_final_annotated_image(self, frame, col_box, book_box, row=None):
        """Bounding box around the target book (+2 points)."""
        img = frame.copy()

        x, y, w, h = col_box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        bx, by, bw, bh = book_box
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        caption = f'{self.book_colour} book'
        if row is not None:
            caption += f' (row {row})'
        cv2.putText(
            img, caption, (bx, max(14, by - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )
        self._stamp_image(img)

        self.image2_path = self._write_image(img, 'detected_book')
        self.get_logger().info(
            f'[LOG] Image 1: {self.image1_path} | Image 2: {self.image2_path}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = BookTargetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
