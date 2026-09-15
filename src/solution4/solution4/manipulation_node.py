"""Grasp the target book at the shelf and deliver it to the collection bin.

Node 3 of the pipeline. It waits for node 2 to report REACHED_SHELF on
/erc/nav_status and for node 1's row on /erc/shelf_row_identification, then:

  grasp    clear both arms out of the camera's view, square the base to the
           shelf on the front LiDAR, find the book of the requested colour on
           that row, slide the base until the book is in front of the left arm,
           and reach, close, lift and retract;
  deliver  back off the shelf, tuck the book in against the body, return to
           the start zone on odometry, line up on the red collection bin by
           vision, lower the book over its rim and let go.

Every number below was measured in the ERC simulation against Gazebo ground
truth, and the comments say what went wrong without it. Three findings shape
the whole node:

  * The left gripper's fingers extend along +X of gripper_left_grasping_link.
    Inverse kinematics is solved for that frame directly (arm_kinematics.py).
  * A mecanum wheel grips only along its roller, and the wheel joints are weak.
    With the book held out in front, a turn at 0.35 rad/s slid the base up to
    0.5 m without the wheels turning, so odometry never saw it. With the book
    tucked in and turns held to 0.15 rad/s, no turn slid more than 3 mm.
  * Odometry drifts, the bin does not. Odometry only brings the robot roughly
    home; the final approach is closed on the bin itself.

Progress is published on /erc/manipulation_status (latched).
"""

import math
from typing import Any, Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from solution4.arm_kinematics import ArmKinematics, GRASP_LINK, ROOT_LINK
from solution4.camera_intrinsics import intrinsics_from_urdf
from std_msgs.msg import Int32, String
import tf2_geometry_msgs  # noqa: F401  - registers PointStamped with the TF buffer
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# --- the shelf and the books ------------------------------------------------
# Heights of the four rows above the floor, from simulation.launch.py.
ROW_HEIGHTS = {1: 1.595, 2: 1.265, 3: 0.935, 4: 0.605}
BASE_LINK_HEIGHT = 0.076
COLOUR_RANGES = {
    'red': [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
    'blue': [((100, 100, 100), (140, 255, 255))],
    'green': [((40, 50, 50), (80, 255, 255))],
    'yellow': [((20, 100, 100), (40, 255, 255))],
}
# Parked at the shelf the camera sees a narrow band, so no single head angle
# frames every row. Sweep the tilt and keep the best candidate at the row's
# height. head_2_joint's real range is [-1.0472, 0.34907]; a command outside it
# wedges the head controller for the rest of the run.
HEAD_2_LOWER, HEAD_2_UPPER = -1.0472, 0.34907
HEAD_1_LIMIT = 1.2
HEAD_TILTS = (0.0, -0.30, -0.60, -0.90, -1.04, 0.20)
BOOK_MIN_LONG_SIDE = 0.08       # m; drops specks
BOOK_FACE_PERCENTILE = 15       # depth percentile over the book's pixels = its near face
ARM_LINKS = ('gripper_right_grasping_link', 'gripper_right_base_link',
             'arm_right_tool_link', 'gripper_left_grasping_link',
             'gripper_left_base_link')
SELF_RADIUS = 0.22              # m; a blob this close to a gripper is the robot

# --- the arm --------------------------------------------------------------
# The torso carries the camera as well as the shoulder: low frames a low row,
# high frames a high one. The grasp itself is always made from 0.32, where the
# reach was verified.
TORSO_FOR_ROW = {1: 0.32, 2: 0.32, 3: 0.18, 4: 0.05}
TORSO_FOR_GRASP = 0.32
# Both arms swung down and out of the camera's view. Photographed at the shelf:
# the robot fills 10 % of the frame here against 21 % in the start-up pose. The
# arms are mirrored, so one set of joint values cannot serve both.
ARM_CLEAR_LEFT = [0.0, -2.40, 0.0, -2.30, 0.0, 0.0, 0.0]
ARM_CLEAR_RIGHT = [-2.50, 1.10, 0.0, -2.40, 0.0, 0.0, 0.0]
# Parked at the shelf, both arms swing through straight ahead on the way to the
# clear pose and stall against the shelf (left shoulder stopped at +0.73 against
# -2.40), where they then hide the lower rows. So back away first.
ARM_CLEAR_BACKOFF = 0.35
ARM_CLEAR_TOLERANCE = 0.15      # rad, on the two joints that swing furthest
# Pad gap 56 mm at 0.065 against a 2 cm spine. The organisers' clamp on the
# public gripper topic rejects anything outside [0.00, 0.069].
GRIPPER_OPEN = 0.065
# Commanding the gripper fully closed drives the finger linkage straight through
# the book: the fingertips are mimic joints driven kinematically, with no force
# limit (the organisers' issue #2). Their workaround is to close under position
# control and keep publishing a position just inside the spine. The pad gap is
# 5 mm at 0.0 and 37.5 mm at 0.04, so a 2 cm spine is touched at about 0.0185.
# Validated against ground truth: held at 0.016, a 2 cm book was lifted 3.9 cm
# and pulled 16.7 cm off the shelf, and still hung there 1.5 minutes later.
# Tighter holds were not better: finger effort reads 0.000 throughout the close,
# so there is no contact signal to stop on, and the position alone decides.
GRIPPER_HOLD = 0.016
GRIPPER_CLOSE_STEP = 0.005
GRIPPER_HOLD_PERIOD = 0.5       # s between re-published hold commands
PREGRASP_BACKOFF = 0.16         # m behind the book
PREGRASP_MIN_X = 0.45           # nearer than this the elbow cannot fold
GRASP_TARGET_X = 0.62           # book here: pre-grasp and grasp both reachable
GRASP_TARGET_TOLERANCE = 0.04
ARM_LATERAL_SWEET_SPOT = 0.00
LATERAL_TOLERANCE = 0.08
LATERAL_ATTEMPTS = 3
# 2 cm inside the spine face: the linkage behind the pads starts 2.2 cm back
# and rams the spine any deeper.
GRASP_DEPTH = 0.02
# Lift before pulling out: dragged flat, the book fights the shelf's friction.
LIFT_HEIGHT = 0.04
REACH_ATTEMPTS = 3
REACH_TOLERANCE = 0.04          # m at the grasp point
JOINT_TRACK_TOLERANCE = 0.05    # rad

# --- base motion ------------------------------------------------------------
# Strafing arcs on a freshly started sim, so it is done in short bursts with
# the heading restored between them.
STRAFE_BURST = 0.12
STRAFE_BURST_SECONDS = 4.0
STRAFE_YAW_TOLERANCE_DEG = 2.0
# 0.15 rad/s: see the module docstring.
TURN_SPEED = 0.15
# Odometry reports 89.1 deg for a true 84.8 deg turn. Angles measured from the
# scene are true angles.
ODOM_YAW_PER_TRUE_YAW = 1.051
TURN_SHIFT_WARN = 0.05

# --- delivery ---------------------------------------------------------------
# The robot spawns at the start-zone centre, so the odometry origin IS the
# start zone. The bin is 1 m behind it: odometry +y, i.e. yaw +90 deg.
HOME_ODOM = (0.0, 0.0)
FACE_BIN_ODOM_YAW = math.pi / 2
HOME_TOLERANCE = 0.06
BACK_OFF_SHELF = 0.40
TORSO_FOR_DELIVERY = 0.32
# Tucked close to the body for every base move (see the module docstring).
CARRY = (0.30, 0.15, 0.95)
# Line-up on the bin: base centre this far short of its near rim.
TARGET_NEAR_X = 0.545
ALIGN_YAW_TOL_DEG = 3.0
ALIGN_YAW_GAIN = 0.8
ALIGN_AXIS_MAX_DEG = 60.0
ALIGN_BEARING_MAX_DEG = 20.0
ALIGN_LATERAL_TOL = 0.03
ALIGN_RANGE_TOL = 0.03
ALIGN_PASSES = 6
# Steep tilts first: at -0.35 the near rim can fall out of the frame, and the
# near edge then measured 0.11 m too far (probe_bin.py against ground truth).
BIN_TILTS = (-0.55, -0.75, -0.35)
BIN_PANS = (0.0, 0.5, -0.5, 1.0, -1.0)
RED = COLOUR_RANGES['red']
BIN_LENGTH = 0.56
BIN_RIM_Z = 0.95 - BASE_LINK_HEIGHT
TABLE_TOP_Z = 0.74 - BASE_LINK_HEIGHT
# What the held book and gripper occupy, relative to the grasp point.
BOOK_BELOW_GRASP = 0.14
BOOK_AHEAD_OF_GRASP = 0.14
GRIPPER_BEHIND_GRASP = 0.157
ABOVE_RIM_CLEARANCE = 0.19
RELEASE_RIM_CLEARANCE = 0.04


def wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quat_matrix(q):
    """Return the rotation matrix of a geometry_msgs quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class ManipulationNode(Node):
    """Grasp the book at the shelf, then deliver it to the bin."""

    def __init__(self):
        super().__init__('node3_manipulation')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('book_colour', 'red')
        self.declare_parameter('row_index_base', 1)
        colour = str(self.get_parameter('book_colour').value).lower()
        if colour not in COLOUR_RANGES:
            self.get_logger().error(f"Unknown book colour '{colour}'; using red.")
            colour = 'red'
        self.book_colour = colour
        self.row_index_base = int(self.get_parameter('row_index_base').value)

        latched = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.bridge = CvBridge()
        # Filled in by callbacks, or once the trigger arrives.
        self.urdf: Optional[str] = None
        self.joints: Any = None
        self.intrinsics: Any = None
        self.colour: Any = None
        self.depth: Any = None
        self.scan: Any = None
        self.nav_status: Optional[str] = None
        self.row: Optional[int] = None
        self.last_size = (0.0, 0.0)
        self.kin: Any = None
        self.hold_position: Optional[float] = None

        self.create_subscription(String, '/robot_description', self._on_urdf, latched)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        for topic in ('/head_front_camera/head_front_camera/depth/camera_info',
                      '/head_front_camera/head_front_camera/color/camera_info'):
            self.create_subscription(CameraInfo, topic, self._on_camera_info, sensor)
        colour_sub = message_filters.Subscriber(
            self, Image, '/head_front_camera/head_front_camera/color/image_raw',
            qos_profile=sensor)
        depth_sub = message_filters.Subscriber(
            self, Image, '/head_front_camera/head_front_camera/depth/image_rect_raw',
            qos_profile=sensor)
        message_filters.ApproximateTimeSynchronizer(
            [colour_sub, depth_sub], 10, 0.1).registerCallback(self._on_rgbd)
        self.create_subscription(LaserScan, '/scan_front_raw', self._on_scan, sensor)
        self.create_subscription(String, '/erc/nav_status', self._on_nav_status, latched)
        self.create_subscription(Int32, '/erc/shelf_row_identification', self._on_row, latched)

        self.arm = self.create_publisher(
            JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.arm_right = self.create_publisher(
            JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.torso = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10)
        self.head = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)
        # The public topic: the organisers' clamp node checks the range and
        # forwards it to the controller.
        self.grip = self.create_publisher(
            JointTrajectory, '/gripper_left_controller/joint_trajectory', 10)
        self.cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/erc/manipulation_status', latched)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Re-publishes the hold position while a book is held; every wait spins it.
        self.create_timer(GRIPPER_HOLD_PERIOD, self._hold_gripper)

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------
    def _on_urdf(self, msg):
        self.urdf = msg.data

    def _on_joints(self, msg):
        self.joints = msg

    def _on_camera_info(self, msg):
        self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_rgbd(self, colour, depth):
        self.colour, self.depth = colour, depth

    def _on_scan(self, msg):
        self.scan = msg

    def _on_nav_status(self, msg):
        self.nav_status = msg.data

    def _on_row(self, msg):
        self.row = int(msg.data) - self.row_index_base + 1

    def log(self, text):
        self.get_logger().info(text)

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.log(f"[NODE 3 STATUS] Published /erc/manipulation_status: '{status}'")

    def wait(self, seconds):
        """Spin for `seconds` of simulated time."""
        start = self.get_clock().now()
        duration = Duration(seconds=float(seconds))
        while rclpy.ok() and self.get_clock().now() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

    def elapsed(self, start):
        return (self.get_clock().now() - start).nanoseconds * 1e-9

    def pos(self, name):
        if self.joints is None:
            return float('nan')
        return dict(zip(self.joints.name, self.joints.position)).get(name, float('nan'))

    def send(self, pub, names, values, seconds):
        msg = JointTrajectory()
        msg.joint_names = list(names)
        point = JointTrajectoryPoint(positions=[float(v) for v in values])
        point.time_from_start.sec = int(seconds)
        msg.points = [point]
        pub.publish(msg)

    def gripper(self, opening, seconds=2):
        self.send(self.grip, ['gripper_left_finger_joint'], [opening], seconds)

    def _hold_gripper(self):
        if self.hold_position is not None:
            self.gripper(self.hold_position, seconds=1)

    def close_on_book(self):
        """Close onto the spine and keep commanding a position just inside it.

        The gripper steps closed under position control to GRIPPER_HOLD, and
        that position is then re-published on a timer until release(), so the
        finger drive never pushes through the book.
        """
        opening = GRIPPER_OPEN
        while opening > GRIPPER_HOLD + 1e-6:
            opening = max(GRIPPER_HOLD, opening - GRIPPER_CLOSE_STEP)
            self.gripper(opening, seconds=1)
            self.wait(1.5)
        self.hold_position = GRIPPER_HOLD
        self.wait(3.0)
        self.log(f'[NODE 3 GRASP] Holding the gripper at {GRIPPER_HOLD:.3f}.')

    def release(self):
        """Stop holding and open the gripper."""
        self.hold_position = None
        self.gripper(GRIPPER_OPEN)
        self.wait(4.0)

    def grasp_point(self):
        t = self.tf_buffer.lookup_transform(ROOT_LINK, GRASP_LINK, Time()).transform
        return t.translation.x, t.translation.y, t.translation.z

    def base_pose(self):
        tf = self.tf_buffer.lookup_transform('odom', ROOT_LINK, Time()).transform
        q = tf.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tf.translation.x, tf.translation.y, yaw

    def fresh_image(self, timeout=15.0):
        """Block until a newer camera frame arrives.

        The synchroniser sometimes stops delivering for a while, and a
        measurement taken after moving the base would then come from the old
        picture.
        """
        if self.colour is None:
            return False
        was = (self.colour.header.stamp.sec, self.colour.header.stamp.nanosec)
        start = self.get_clock().now()
        while rclpy.ok() and self.elapsed(start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            now = (self.colour.header.stamp.sec, self.colour.header.stamp.nanosec)
            if now != was:
                return True
        self.get_logger().warn('No new camera frame; measuring from a stale image.')
        return False

    # ------------------------------------------------------------------
    # Head, torso and arm
    # ------------------------------------------------------------------
    def look(self, tilt, pan=0.0, seconds=2, retries=2):
        """Aim the head within its limits and check it got there."""
        tilt = max(HEAD_2_LOWER, min(HEAD_2_UPPER, tilt))
        pan = max(-HEAD_1_LIMIT, min(HEAD_1_LIMIT, pan))
        for _ in range(retries + 1):
            self.send(self.head, ['head_1_joint', 'head_2_joint'], [pan, tilt], seconds)
            self.wait(seconds + 2)
            if abs(self.pos('head_2_joint') - tilt) <= 0.05:
                return True
        self.get_logger().warn(
            f"Head did not reach tilt {tilt:+.2f} (at {self.pos('head_2_joint'):+.2f}).")
        return False

    def ramp_torso(self, target):
        """Move the torso in steps, then re-solve kinematics at its new height.

        The torso controller silently declines a single distant setpoint, so
        the move is sent as points spaced inside its 0.035 m/s limit.
        """
        points = self.kin.torso_ramp(self.pos('torso_lift_joint'), target)
        if points:
            msg = JointTrajectory()
            msg.joint_names = ['torso_lift_joint']
            for position, t in points:
                point = JointTrajectoryPoint(positions=[float(position)])
                point.time_from_start.sec = int(t)
                msg.points.append(point)
            self.torso.publish(msg)
            self.wait(points[-1][1] + 4.0)
        self.kin = ArmKinematics(self.urdf, seed_attempts=150,
                                 torso_position=self.pos('torso_lift_joint'))
        self.log(f"Torso at {self.pos('torso_lift_joint'):.3f} (wanted {target:.2f}).")

    def arms_clear(self):
        return (abs(self.pos('arm_left_2_joint') - ARM_CLEAR_LEFT[1]) <= ARM_CLEAR_TOLERANCE
                and abs(self.pos('arm_right_1_joint') - ARM_CLEAR_RIGHT[0])
                <= ARM_CLEAR_TOLERANCE)

    def clear_arms(self, seconds=10, backoff=ARM_CLEAR_BACKOFF):
        """Swing both arms down and out of the camera's view, away from the shelf.

        The base backs off first so the arms do not sweep into the shelf, and
        returns the same distance afterwards. Returns whether the arms arrived.
        """
        if self.arms_clear():
            return True
        if backoff:
            self.drive_forward(-backoff, speed=0.1)
        for _ in range(2):
            for side, pub, q in (('left', self.arm, ARM_CLEAR_LEFT),
                                 ('right', self.arm_right, ARM_CLEAR_RIGHT)):
                self.send(pub, [f'arm_{side}_{i}_joint' for i in range(1, 8)], q, seconds)
            # The right arm can take well over the trajectory time to arrive.
            start = self.get_clock().now()
            while self.elapsed(start) < seconds + 20 and not self.arms_clear():
                self.wait(1.0)
            if self.arms_clear():
                break
        cleared = self.arms_clear()
        if not cleared:
            self.get_logger().warn(
                f"Arms not clear: left shoulder {self.pos('arm_left_2_joint'):+.2f}, "
                f"right shoulder {self.pos('arm_right_1_joint'):+.2f}.")
        if backoff:
            self.drive_forward(backoff, speed=0.1)
        return cleared

    def reach(self, x, y, z, seconds=8, settle=12):
        """Put the grasp point at (x, y, z) in base_link, verifying the result.

        Seeded with where the arm is, so a retry after a joint stalls differs
        from the failed attempt, and differs towards something reachable.
        """
        pitch, arm = None, []
        for attempt in range(REACH_ATTEMPTS):
            current = [self.pos(name) for name in self.kin.joint_names]
            q, pitch = self.kin.solve(x, y, z, current=current)
            if q is None:
                self.log(f'No IK for ({x:.3f}, {y:+.3f}, {z:.3f}), attempt {attempt + 1}.')
                continue
            _torso, arm = self.kin.split_solution(q)
            self.send(self.arm, [n for n, _v in arm], [v for _n, v in arm], seconds)
            self.wait(settle)
            stuck = [n for n, v in arm if abs(self.pos(n) - v) > JOINT_TRACK_TOLERANCE]
            if not stuck:
                break
            self.log(f"Attempt {attempt + 1}: {', '.join(stuck)} did not track.")
        if not arm:
            return False
        try:
            gx, gy, gz = self.grasp_point()
        except Exception as exc:
            self.get_logger().warn(f'Grasp point TF failed: {exc}')
            return False
        err = math.sqrt((gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2)
        self.log(f'Arm -> ({gx:.3f}, {gy:+.3f}, {gz:.3f}), error {err:.3f} m, '
                 f'pitch {pitch if pitch is not None else float("nan"):+.0f} deg.')
        return err < REACH_TOLERANCE

    # ------------------------------------------------------------------
    # Base motion
    # ------------------------------------------------------------------
    def rotate_to(self, yaw_target, tol_deg=1.0, speed=TURN_SPEED, timeout=90.0):
        """Turn to an odometry yaw with angular.z alone.

        Returns how far the turn moved the base, which it should not.
        """
        x0, y0, _yaw0 = self.base_pose()
        start = self.get_clock().now()
        cmd = Twist()
        while rclpy.ok() and self.elapsed(start) < timeout:
            _x, _y, yaw = self.base_pose()
            err = wrap(yaw_target - yaw)
            if abs(math.degrees(err)) <= tol_deg:
                break
            cmd.angular.z = max(-speed, min(speed, 2.0 * err))
            self.cmd_vel.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.cmd_vel.publish(Twist())
        self.wait(0.5)
        x1, y1, _yaw1 = self.base_pose()
        shift = math.hypot(x1 - x0, y1 - y0)
        if shift > TURN_SHIFT_WARN:
            self.get_logger().warn(f'That turn also moved the base {shift:.3f} m.')
        return shift

    def rotate_by(self, true_angle):
        """Turn by an angle measured from the scene."""
        _x, _y, yaw = self.base_pose()
        return self.rotate_to(wrap(yaw + true_angle * ODOM_YAW_PER_TRUE_YAW))

    def drive_forward(self, dx, speed=0.06, tolerance=0.02, timeout=60.0):
        """Drive straight ahead by dx metres; return the distance achieved."""
        x0, y0, yaw0 = self.base_pose()
        start = self.get_clock().now()
        cmd = Twist()
        while rclpy.ok() and self.elapsed(start) < timeout:
            x, y, _yaw = self.base_pose()
            moved = (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)
            if abs(dx - moved) <= tolerance:
                break
            cmd.linear.x = math.copysign(min(speed, max(0.03, abs(dx - moved))), dx - moved)
            self.cmd_vel.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.cmd_vel.publish(Twist())
        self.wait(1.0)
        x, y, _yaw = self.base_pose()
        return (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)

    def strafe(self, dy, speed=0.08, tolerance=0.02, timeout=120.0):
        """Slide sideways by dy metres in short bursts, restoring the heading between them.

        On a freshly started sim a linear.y command arcs (3.3 deg/s measured),
        and correcting it with angular.z at the same time freezes the base, so
        the two are interleaved instead.
        """
        x0, y0, yaw0 = self.base_pose()

        def lateral():
            x, y, _yaw = self.base_pose()
            return -(x - x0) * math.sin(yaw0) + (y - y0) * math.cos(yaw0)

        start = self.get_clock().now()
        while rclpy.ok() and self.elapsed(start) < timeout:
            remaining = dy - lateral()
            if abs(remaining) <= tolerance:
                break
            burst_start = lateral()
            burst_t0 = self.get_clock().now()
            cmd = Twist()
            while rclpy.ok():
                moved = lateral()
                remaining = dy - moved
                if (abs(moved - burst_start) >= STRAFE_BURST or abs(remaining) <= tolerance
                        or self.elapsed(burst_t0) > STRAFE_BURST_SECONDS):
                    break
                cmd.linear.y = math.copysign(min(speed, max(0.03, abs(remaining))), remaining)
                self.cmd_vel.publish(cmd)
                rclpy.spin_once(self, timeout_sec=0.05)
            self.cmd_vel.publish(Twist())
            self.wait(0.5)
            self.hold_heading(yaw0)

        # Each heading correction nudges the base back ~14 mm; drive it out again.
        x, y, _yaw = self.base_pose()
        lost = (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)
        if abs(lost) > 0.03:
            self.drive_forward(-lost)
        achieved = lateral()
        self.log(f'Strafed {achieved:+.3f} m of {dy:+.3f} m.')
        return achieved

    def hold_heading(self, yaw0):
        """Turn back to square after a lateral burst, measured on the shelf when possible.

        The wheels can slide the base round without odometry seeing it: holding
        the odometry heading once left the base 81 deg off the shelf. So the
        shelf face seen by the front LiDAR is the reference, and odometry is
        used only when the shelf cannot be measured.
        """
        err = self.shelf_yaw_error()
        if err is not None:
            if abs(err) > STRAFE_YAW_TOLERANCE_DEG:
                self.rotate_by(math.radians(err))
            return
        _x, _y, yaw = self.base_pose()
        if abs(math.degrees(wrap(yaw0 - yaw))) > STRAFE_YAW_TOLERANCE_DEG:
            self.rotate_to(yaw0, tol_deg=1.5)

    def sidestep(self, dy):
        """Move sideways by dy (left +) as turn, drive, turn back, closed on odometry."""
        x0, y0, yaw0 = self.base_pose()
        gx, gy = x0 - dy * math.sin(yaw0), y0 + dy * math.cos(yaw0)
        for _ in range(3):
            x, y, _yaw = self.base_pose()
            if math.hypot(gx - x, gy - y) <= 0.02:
                break
            self.rotate_to(math.atan2(gy - y, gx - x))
            x, y, yaw = self.base_pose()
            ahead = (gx - x) * math.cos(yaw) + (gy - y) * math.sin(yaw)
            self.drive_forward(ahead, speed=0.1)
        self.rotate_to(yaw0)

    def go_to_odom(self, gx, gy, tol=0.05):
        """Turn towards an odometry point and drive to it, re-checking up to three times."""
        for _ in range(3):
            x, y, _yaw = self.base_pose()
            dist = math.hypot(gx - x, gy - y)
            if dist <= tol:
                break
            self.rotate_to(math.atan2(gy - y, gx - x))
            self.drive_forward(dist, speed=0.2, tolerance=0.015)

    # ------------------------------------------------------------------
    # Perception at the shelf
    # ------------------------------------------------------------------
    def find_blobs(self):
        """Bounding boxes of every blob of the book's colour, largest first."""
        frame = self.bridge.imgmsg_to_cv2(self.colour, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in COLOUR_RANGES[self.book_colour]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 200]
        contours.sort(key=cv2.contourArea, reverse=True)
        return [cv2.boundingRect(c) for c in contours]

    def deproject(self, box):
        """Return the book's near face under a box as a PointStamped in base_link, or None.

        Depth is a low percentile over the box's pixels of the book's colour,
        not the median at its centre. Seen even slightly from the side, the box
        also covers the book's side face, which reads up to 16 cm deeper; a
        grasp aimed there pushed the book 7 cm into the shelf.
        """
        depth = self.bridge.imgmsg_to_cv2(self.depth, 'passthrough').astype(np.float32)
        frame = self.bridge.imgmsg_to_cv2(self.colour, 'bgr8')
        x, y, w, h = box
        u, v = x + w // 2, y + h // 2
        dh, dw = depth.shape[:2]
        x0, x1, y0, y1 = max(0, x), min(dw, x + w), max(0, y), min(dh, y + h)
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in COLOUR_RANGES[self.book_colour]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        region = depth[y0:y1, x0:x1]
        good = region[(mask > 0) & np.isfinite(region) & (region > 0.05)]
        if good.size < 4:
            return None
        z = float(np.percentile(good, BOOK_FACE_PERCENTILE))
        fx, fy, cx, cy = self.intrinsics
        point = PointStamped()
        point.header.frame_id = self.depth.header.frame_id
        point.header.stamp = Time().to_msg()
        point.point.x = (u - cx) * z / fx
        point.point.y = (v - cy) * z / fy
        point.point.z = z
        self.last_size = (w * z / fx, h * z / fy)
        for _ in range(12):
            try:
                return self.tf_buffer.transform(point, ROOT_LINK,
                                                timeout=Duration(seconds=1.0))
            except Exception:
                self.wait(1.0)
        return None

    def _on_own_arm(self, point):
        """Return True if the candidate sits on one of the robot's own gripper links."""
        for link in ARM_LINKS:
            try:
                t = self.tf_buffer.lookup_transform(ROOT_LINK, link, Time()).transform
            except Exception:
                continue
            d = math.sqrt((point.point.x - t.translation.x) ** 2
                          + (point.point.y - t.translation.y) ** 2
                          + (point.point.z - t.translation.z) ** 2)
            if d < SELF_RADIUS:
                return True
        return False

    def find_row(self, row):
        """Sweep the head and return (box, point) of the best book on `row`, or None."""
        want = ROW_HEIGHTS[row] - BASE_LINK_HEIGHT
        best = None
        for tilt in HEAD_TILTS:
            self.look(tilt)
            self.fresh_image()
            for box in self.find_blobs():
                point = self.deproject(box)
                if point is None or max(self.last_size) < BOOK_MIN_LONG_SIDE:
                    continue
                if self._on_own_arm(point):
                    continue
                dz = abs(point.point.z - want)
                if dz <= 0.15 and abs(point.point.y) <= 0.40:
                    if best is None or dz < best[2]:
                        best = (box, point, dz)
        if best is None:
            return None
        return best[0], best[1]

    def shelf_yaw_error(self):
        """Degrees to turn so the base faces the shelf square, or None.

        A RANSAC line through the front laser's returns ahead of the robot: the
        shelf face is the dominant straight line there. Validated against
        ground truth: -22.9 deg measured, -22.8 deg true.
        """
        self.wait(1.0)
        scan = self.scan
        if scan is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(ROOT_LINK, scan.header.frame_id,
                                                 Time()).transform
        except Exception:
            return None
        r = np.asarray(scan.ranges, dtype=np.float64)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > 0.15) & (r < 3.0)
        local = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum())])
        offset = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        xy = ((quat_matrix(tf.rotation) @ local).T + offset)[:, :2]
        xy = xy[(xy[:, 0] > 0.25) & (np.abs(xy[:, 1]) < 1.8)]
        if len(xy) < 30:
            return None
        rng = np.random.default_rng(0)
        best = None
        for _ in range(500):
            i, j = rng.choice(len(xy), 2, replace=False)
            d = xy[j] - xy[i]
            if np.linalg.norm(d) < 0.2:
                continue
            normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            inliers = np.abs((xy - xy[i]) @ normal) < 0.015
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.sum() < 30:
            return None
        points = xy[best]
        _, _, vt = np.linalg.svd(points - points.mean(axis=0))
        # Square means the line runs along base y (90 deg); fold the line's
        # direction into [0, 180) so either orientation of the fit reads the same.
        err = math.degrees(math.atan2(vt[0][1], vt[0][0])) % 180.0 - 90.0
        return err if abs(err) < 45.0 else None

    def square_up(self, tol_deg=1.0, passes=3):
        """Rotate the base until it faces the shelf square."""
        for _ in range(passes):
            err = self.shelf_yaw_error()
            if err is None:
                # An arm still moving past the laser blocks it; look again.
                self.wait(3.0)
                err = self.shelf_yaw_error()
            if err is None:
                self.get_logger().warn('Could not measure the shelf; not squaring up.')
                return
            if abs(err) <= tol_deg:
                return
            _x, _y, yaw = self.base_pose()
            self.rotate_to(yaw + math.radians(err), tol_deg=0.5)

    # ------------------------------------------------------------------
    # Perception at the bin
    # ------------------------------------------------------------------
    def detect_bin(self):
        """Measure the bin in base_link from red pixels.

        Returns near rim, centre line, rim height, long axis and bearing, or
        None. Only points at bin height are kept, and points where a held book
        hangs are dropped (a red book reaches down into that band). The long
        axis is the major principal direction of the rim, trusted only when the
        rim is clearly elongated and roughly points at the robot.
        """
        if self.colour is None or self.depth is None or self.intrinsics is None:
            return None
        colour_msg, depth_msg = self.colour, self.depth
        frame = self.bridge.imgmsg_to_cv2(colour_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in RED:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        vs, us = np.nonzero(mask)
        zc = depth[vs, us] if len(us) else np.array([])
        ok = np.isfinite(zc) & (zc > 0.1) & (zc < 3.0)
        us, vs, zc = us[ok], vs[ok], zc[ok]
        if len(us) < 200:
            return None
        fx, fy, cx, cy = self.intrinsics
        cam = np.stack([(us - cx) * zc / fx, (vs - cy) * zc / fy, zc], axis=1)
        try:
            tf = self.tf_buffer.lookup_transform(ROOT_LINK, depth_msg.header.frame_id, Time(),
                                                 timeout=Duration(seconds=2.0)).transform
            gx, gy, gz = self.grasp_point()
        except Exception:
            return None
        offset = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        pts = cam @ quat_matrix(tf.rotation).T + offset
        band = ((pts[:, 2] > TABLE_TOP_Z - 0.03) & (pts[:, 2] < BIN_RIM_Z + 0.06)
                & (pts[:, 0] > 0.25))
        held = ((pts[:, 0] > gx - 0.05) & (pts[:, 0] < gx + BOOK_AHEAD_OF_GRASP + 0.04)
                & (np.abs(pts[:, 1] - gy) < 0.08)
                & (pts[:, 2] > gz - BOOK_BELOW_GRASP - 0.04))
        pts = pts[band & ~held]
        if len(pts) < 150:
            return None
        rim_z = float(np.percentile(pts[:, 2], 97))
        top = pts[pts[:, 2] > rim_z - 0.05][:, :2]
        axis, ratio, reliable = 0.0, 0.0, False
        if len(top) > 50:
            centred = top - top.mean(axis=0)
            w, vecs = np.linalg.eigh(centred.T @ centred / len(centred))
            ratio = math.sqrt(w[1] / max(w[0], 1e-9))
            major = vecs[:, 1] if vecs[0, 1] >= 0 else -vecs[:, 1]
            axis = math.atan2(major[1], major[0])
            reliable = ratio > 1.4 and abs(math.degrees(axis)) < ALIGN_AXIS_MAX_DEG
        ax = axis if reliable else 0.0
        u = np.array([math.cos(ax), math.sin(ax)])
        v = np.array([-u[1], u[0]])
        along, across = pts[:, :2] @ u, pts[:, :2] @ v
        width = float(np.percentile(across, 95) - np.percentile(across, 5))
        if width < 0.15:
            return None
        # Centre line from the rim's two outer edges. The median of every red
        # point leans towards whichever inner wall the camera sees more of, and
        # read 0.14 m off the true centre line.
        if len(top) > 50:
            rim_across = top @ v
            centre = 0.5 * float(np.percentile(rim_across, 3) + np.percentile(rim_across, 97))
        else:
            centre = float(np.median(across))
        result = {
            'axis': axis, 'reliable': reliable, 'rim_z': rim_z,
            'near': float(np.percentile(along, 5)),
            'centre': centre,
            'bearing': math.atan2(float(np.median(pts[:, 1])), float(np.median(pts[:, 0]))),
        }
        trust = 'trusted' if reliable else 'not trusted'
        self.log(f'Bin: axis {math.degrees(axis):+.1f} deg ({trust}), '
                 f"near {result['near']:.3f} m, centre {result['centre']:+.3f} m, "
                 f"bearing {math.degrees(result['bearing']):+.1f} deg, rim z {rim_z:.3f} m.")
        return result

    def find_bin(self):
        """Look ahead at a few tilts, then pan the head to either side."""
        for pan in BIN_PANS:
            for tilt in BIN_TILTS:
                self.look(tilt, pan=pan)
                self.fresh_image()
                found = self.detect_bin()
                if found:
                    return found
        self.look(BIN_TILTS[0])
        return None

    def line_up_on_bin(self):
        """Face the bin, square to its axis, get on its centre line and stop at range."""
        found = None
        for _ in range(ALIGN_PASSES):
            found = self.find_bin()
            if found is None:
                return None
            if not found['reliable'] and \
                    abs(math.degrees(found['bearing'])) > ALIGN_BEARING_MAX_DEG:
                self.rotate_by(found['bearing'])
            elif found['reliable'] and abs(math.degrees(found['axis'])) > ALIGN_YAW_TOL_DEG:
                self.rotate_by(ALIGN_YAW_GAIN * found['axis'])
            elif abs(found['centre']) > ALIGN_LATERAL_TOL:
                self.sidestep(found['centre'])
            elif abs(found['near'] - TARGET_NEAR_X) > ALIGN_RANGE_TOL:
                self.drive_forward(found['near'] - TARGET_NEAR_X, speed=0.1, tolerance=0.015)
            else:
                return found
        self.get_logger().warn('Not fully lined up on the bin; using the last measurement.')
        return self.find_bin() or found

    # ------------------------------------------------------------------
    # The task
    # ------------------------------------------------------------------
    def wait_for_trigger(self):
        """Wait for node 2 to reach the shelf and node 1 to publish the row."""
        self.log('[NODE 3] Waiting for REACHED_SHELF on /erc/nav_status.')
        while rclpy.ok() and self.nav_status not in ('REACHED_SHELF', 'NAV_FAILED'):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.nav_status != 'REACHED_SHELF':
            return False
        start = self.get_clock().now()
        while rclpy.ok() and self.elapsed(start) < 60.0 and not (
                self.row in ROW_HEIGHTS and self.urdf and self.joints
                and self.colour is not None and self.depth is not None):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.urdf is None or self.joints is None or self.colour is None:
            self.get_logger().error('[NODE 3] Robot description, joints or camera missing.')
            return False
        if self.intrinsics is None:
            self.intrinsics = intrinsics_from_urdf(self.urdf)
        # Let the TF listener assemble the tree before the first lookup.
        self.wait(4.0)
        self.kin = ArmKinematics(self.urdf, seed_attempts=150,
                                 torso_position=self.pos('torso_lift_joint'))
        return True

    def grasp(self, row):
        """Find the book on `row`, line the base up with it and pick it up."""
        self.log(f'[NODE 3 GRASP] Picking up the {self.book_colour} book on row {row}.')
        self.gripper(GRIPPER_OPEN)
        self.ramp_torso(TORSO_FOR_ROW[row])
        self.clear_arms()
        # Square to the shelf before looking: parked 23 deg off, the approach
        # swept the book over.
        self.square_up()

        found = self.find_row(row)
        if found is None:
            self.get_logger().error(f'[NODE 3 GRASP] No {self.book_colour} book on row {row}.')
            return False
        _box, point = found
        bx, by, bz = point.point.x, point.point.y, point.point.z
        self.log(f'[NODE 3 GRASP] Book at ({bx:.3f}, {by:+.3f}, {bz:.3f}) in base_link.')

        # Close the lateral loop on the measured book, not on odometry.
        for _ in range(LATERAL_ATTEMPTS):
            if abs(by - ARM_LATERAL_SWEET_SPOT) <= LATERAL_TOLERANCE:
                break
            self.strafe(by - ARM_LATERAL_SWEET_SPOT)
            self.fresh_image()
            found = self.find_row(row)
            if found is None:
                self.get_logger().error('[NODE 3 GRASP] Lost the book after sliding.')
                return False
            _box, point = found
            bx, by, bz = point.point.x, point.point.y, point.point.z

        # Sliding can turn the base; square up again and re-measure before reaching.
        self.square_up()
        self.fresh_image()
        again = self.find_row(row)
        if again is not None:
            bx, by, bz = again[1].point.x, again[1].point.y, again[1].point.z

        # Put the book where the whole pre-grasp -> grasp stroke is reachable,
        # and measure again while the torso is still low enough to see it.
        if abs(bx - GRASP_TARGET_X) > GRASP_TARGET_TOLERANCE:
            bx -= self.drive_forward(bx - GRASP_TARGET_X)
            self.fresh_image()
            again = self.find_row(row)
            if again is not None:
                bx, by, bz = again[1].point.x, again[1].point.y, again[1].point.z
        return self.pick(bx, by, bz)

    def pick(self, bx, by, bz):
        """Reach the book at (bx, by, bz) in base_link, close on it, lift and retract."""
        self.log(f'[NODE 3 GRASP] Grasping at ({bx:.3f}, {by:+.3f}, {bz:.3f}).')
        # base_link coordinates do not move with the torso, so they survive this.
        if abs(self.pos('torso_lift_joint') - TORSO_FOR_GRASP) > 0.01:
            self.ramp_torso(TORSO_FOR_GRASP)
        self.gripper(GRIPPER_OPEN)
        self.wait(3.0)

        pre_x = max(bx - PREGRASP_BACKOFF, PREGRASP_MIN_X)
        if not self.reach(pre_x, by, bz):
            self.get_logger().error('[NODE 3 GRASP] Could not reach the pre-grasp pose.')
            return False
        self.reach(bx + GRASP_DEPTH, by, bz, seconds=5, settle=9)
        self.close_on_book()
        self.reach(bx + GRASP_DEPTH, by, bz + LIFT_HEIGHT, seconds=4, settle=8)
        self.reach(pre_x, by, bz + LIFT_HEIGHT, seconds=6, settle=10)
        self.log('[NODE 3 GRASP] Closed, lifted and retracted.')
        return True

    def deliver(self):
        """Carry the book back to the start zone and drop it into the bin."""
        self.log('[NODE 3 DELIVER] Taking the book to the collection bin.')
        if self.pos('torso_lift_joint') < TORSO_FOR_DELIVERY - 0.02:
            self.ramp_torso(TORSO_FOR_DELIVERY)

        # Back off first: raising or tucking the arm at the shelf scrapes the
        # book on the board above.
        self.drive_forward(-BACK_OFF_SHELF, speed=0.1)
        if not self.reach(*CARRY):
            self.get_logger().warn('[NODE 3 DELIVER] Carry pose not reached; continuing.')

        for _ in range(3):
            self.go_to_odom(*HOME_ODOM)
            self.rotate_to(FACE_BIN_ODOM_YAW)
            x, y, _yaw = self.base_pose()
            if math.hypot(x - HOME_ODOM[0], y - HOME_ODOM[1]) <= HOME_TOLERANCE:
                break
        found = self.line_up_on_bin()
        if found is None:
            self.get_logger().error('[NODE 3 DELIVER] Bin not found; not releasing blind.')
            return False

        near, bin_y, rim_z = found['near'], found['centre'], found['rim_z']
        # Gripper wholly inside the bin's footprint, book clear of the far wall.
        rel_x = min(near + GRIPPER_BEHIND_GRASP + 0.04,
                    near + BIN_LENGTH - 0.04 - BOOK_AHEAD_OF_GRASP)
        above_z = rim_z + ABOVE_RIM_CLEARANCE
        release_z = rim_z + RELEASE_RIM_CLEARANCE
        self.reach(rel_x, bin_y, above_z)
        self.reach(rel_x, bin_y, 0.5 * (above_z + release_z), seconds=4, settle=8)
        self.reach(rel_x, bin_y, release_z, seconds=4, settle=8)
        self.release()
        self.log('[NODE 3 DELIVER] Released over the bin.')
        # Lift clear and tuck in before anything else moves: moving the base
        # with the arm over the bin once dragged the bin 0.34 m.
        self.reach(rel_x, bin_y, above_z)
        self.reach(*CARRY)
        return True

    def run(self):
        if not self.wait_for_trigger():
            self.publish_status('MANIPULATION_ABORTED')
            return
        row = self.row
        if row not in ROW_HEIGHTS:
            self.get_logger().error('[NODE 3] No shelf row received; cannot grasp.')
            self.publish_status('GRASP_FAILED')
            return
        self.publish_status('GRASPING')
        if not self.grasp(row):
            self.publish_status('GRASP_FAILED')
            return
        self.publish_status('DELIVERING')
        if self.deliver():
            self.log('=' * 50)
            self.log('[NODE 3 SUCCESS] Book delivered to the collection bin.')
            self.log('=' * 50)
            self.publish_status('DELIVERED')
        else:
            self.publish_status('DELIVERY_FAILED')


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    try:
        node.run()
        # Stay up so the latched status stays available to the graders' tools.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
