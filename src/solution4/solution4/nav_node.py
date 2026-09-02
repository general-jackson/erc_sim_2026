"""Drive the base to the shelf once node 1 has identified the target column.

This used to hand the goal to Nav2 in the 'map' frame. Nothing in the ERC
simulation provides that: there is no map server, no AMCL, no costmaps and no
/navigate_to_pose action server, so every run ended with

    "map" passed to lookupTransform argument target_frame does not exist

and the robot never moved. The Nav2 packages are installed in the image, but
standing them up would mean supplying a map of the hall and localising in it,
which the organisers' world does not ask for.

What the simulation does provide is wheel odometry - /odom, and an
odom -> base_link transform on /tf - and an omni_drive_controller base, which
can translate sideways as well as forwards. The goal is a few metres away and
the run lasts seconds, so odometry drift over that distance is small compared
with the 0.6 m standoff. This node therefore closes the loop on odometry
directly and treats the front LiDAR as the authority on when to stop.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  - registers PoseStamped with the TF buffer


class NavigationNode(Node):

    def __init__(self):
        super().__init__(
            'navigation_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )

        # Everything else in this stack runs on simulation time. Without this
        # the TF buffer stamps its queries with the wall clock and no lookup
        # ever matches.
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        # The frame the goal is held in while the base drives. Odometry, not a
        # map - see the module docstring.
        self.declare_parameter('goal_frame', 'odom')
        self.declare_parameter('max_linear_speed', 0.25)     # m/s
        self.declare_parameter('max_yaw_speed', 0.30)        # rad/s
        self.declare_parameter('goal_tolerance', 0.10)       # m
        self.declare_parameter('obstacle_stop_distance', 0.45)   # m, front LiDAR
        self.declare_parameter('nav_timeout', 45.0)          # s
        self.declare_parameter('control_period', 0.1)        # s

        self.goal_frame = str(self.get_parameter('goal_frame').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_yaw_speed = float(self.get_parameter('max_yaw_speed').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.obstacle_stop_distance = float(self.get_parameter('obstacle_stop_distance').value)
        self.nav_timeout = float(self.get_parameter('nav_timeout').value)
        self.control_period = float(self.get_parameter('control_period').value)

        self._goal = None
        self._started_at = None
        self._timer = None
        self._front_scan = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.status_pub = self.create_publisher(String, '/erc/nav_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Node 1 latches this pose, so a queue is enough to catch it even if
        # this node came up first.
        self.create_subscription(
            PoseStamped, '/erc/shelf_approach_pose', self._on_approach_pose, 10
        )

        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, '/scan_front_raw', self._on_front_scan, lidar_qos)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _on_front_scan(self, msg):
        self._front_scan = msg

    def _front_clearance(self, half_angle_deg=10.0):
        """Nearest return straight ahead, or inf if the LiDAR has not spoken."""
        scan = self._front_scan
        if scan is None or not scan.ranges:
            return float('inf')
        half = math.radians(half_angle_deg)
        best = float('inf')
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r <= scan.range_min:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if -half <= angle <= half:
                best = min(best, r)
        return best

    def _on_approach_pose(self, pose: PoseStamped):
        self.get_logger().info(
            '[NODE 2 TRIGGER] Received /erc/shelf_approach_pose: '
            f'x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}, '
            f'frame={pose.header.frame_id}'
        )
        if self._goal is not None:
            self.get_logger().warn('[NODE 2 BUSY] Navigation already active; ignoring repeat trigger.')
            self._publish_status('NAV_BUSY')
            return

        goal = self._to_goal_frame(pose)
        if goal is None:
            self.get_logger().error('[NODE 2 ERROR] Could not anchor the goal; aborting navigation.')
            self._publish_status('NAV_FAILED')
            return

        self._goal = goal
        self._started_at = self.get_clock().now()
        self.get_logger().info(
            f"[NODE 2 NAV] Goal anchored in '{self.goal_frame}' at "
            f'({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}). Driving.'
        )
        self._timer = self.create_timer(self.control_period, self._tick)

    def _to_goal_frame(self, pose: PoseStamped):
        """Anchor the base-relative goal to odometry so it survives the drive."""
        probe = PoseStamped()
        probe.header.frame_id = pose.header.frame_id
        # Zero stamp means "the latest transform available". The goal is
        # anchored the moment it arrives and the base is stationary then, so
        # the latest transform is the right one and this avoids an
        # extrapolation error if the pose stamp leads the TF buffer.
        probe.header.stamp = Time().to_msg()
        probe.pose = pose.pose
        try:
            return self.tf_buffer.transform(
                probe, self.goal_frame, timeout=Duration(seconds=3.0)
            )
        except Exception as exc:
            self.get_logger().error(f'[NODE 2 TF] {pose.header.frame_id} -> {self.goal_frame} failed: {exc}')
            return None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _tick(self):
        if self._goal is None:
            return

        elapsed = (self.get_clock().now() - self._started_at).nanoseconds * 1e-9
        if elapsed > self.nav_timeout:
            self.get_logger().error(f'[NODE 2 FAILURE] Gave up after {elapsed:.0f}s.')
            self._finish('NAV_FAILED')
            return

        # The shelf is the thing we are driving at, so a close LiDAR return
        # means the standoff is reached, not that something is in the way.
        clearance = self._front_clearance()
        forward_blocked = clearance <= self.obstacle_stop_distance

        try:
            local = self.tf_buffer.transform(
                self._stamped_goal(), 'base_link', timeout=Duration(seconds=0.5)
            )
        except Exception as exc:
            self.get_logger().warn(
                f'[NODE 2 TF] goal -> base_link unavailable: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        dx = local.pose.position.x
        dy = local.pose.position.y
        remaining = math.hypot(dx, dy)

        # The two axes are controlled separately, because they finish for
        # different reasons. Forward motion ends when the LiDAR says the shelf
        # is a standoff away; sideways motion has to keep going until the base
        # is actually in front of the target column. Driving the pair as one
        # vector meant the standoff cut the sideways correction short and the
        # robot stopped a fifth of a metre off the column.
        forward_done = forward_blocked or dx <= self.goal_tolerance
        lateral_done = abs(dy) <= self.goal_tolerance
        if forward_done and lateral_done:
            self.get_logger().info(
                f'[NODE 2 NAV] In position: {dx:.2f} m ahead, {dy:+.2f} m across, '
                f'shelf at {clearance:.2f} m.'
            )
            self._finish('REACHED_SHELF')
            return

        cmd = Twist()
        cmd.linear.x = 0.0 if forward_done else self._axis_speed(dx)
        cmd.linear.y = 0.0 if lateral_done else self._axis_speed(dy)  # base is holonomic
        # Hold the heading the goal was anchored with; the column was
        # identified from it and the book search depends on keeping it.
        cmd.angular.z = max(-self.max_yaw_speed,
                            min(self.max_yaw_speed, 1.0 * self._heading_error(local)))
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(
            f'[NODE 2 PROGRESS] {remaining:.2f} m remaining '
            f'({dx:.2f} ahead, {dy:+.2f} across), clearance {clearance:.2f} m',
            throttle_duration_sec=1.0,
        )

    def _axis_speed(self, error):
        """Proportional, capped, and eased down close in - but never a crawl."""
        speed = min(self.max_linear_speed, max(0.05, 0.6 * abs(error)))
        return math.copysign(speed, error)

    def _stamped_goal(self):
        probe = PoseStamped()
        probe.header.frame_id = self.goal_frame
        probe.header.stamp = Time().to_msg()
        probe.pose = self._goal.pose
        return probe

    @staticmethod
    def _heading_error(local_goal):
        """Yaw of the goal orientation expressed in the base frame."""
        q = local_goal.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _finish(self, status):
        self.cmd_vel_pub.publish(Twist())
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._goal = None
        if status == 'REACHED_SHELF':
            self.get_logger().info('=' * 50)
            self.get_logger().info('[NODE 2 SUCCESS] Navigation complete: REACHED_SHELF')
            self.get_logger().info('=' * 50)
        self._publish_status(status)

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"[NODE 2 STATUS] Published /erc/nav_status: '{status}'")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
