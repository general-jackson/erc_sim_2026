import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap

import tf2_ros
import tf2_geometry_msgs  # Enables PoseStamped transforms for TF2 Buffer


class NavigationNode(Node):

    def __init__(self):
        super().__init__("navigation_node")

        cb_group = ReentrantCallbackGroup()

        # State tracking
        self._nav_in_progress = False
        self._retry_used = False
        self._current_pose = None

        # TF Buffer for frame transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers & Subscribers
        self.status_pub = self.create_publisher(String, "/erc/nav_status", 10)

        # Queue depth 10 allows compatibility with both transient local & CLI publishers
        self.subscription = self.create_subscription(
            PoseStamped, 
            "/erc/shelf_approach_pose", 
            self._on_approach_pose, 
            10,
            callback_group=cb_group,
        )

        # Nav2 Action Client
        self.nav_client = ActionClient(
            self, NavigateToPose, "/navigate_to_pose",
            callback_group=cb_group,
        )

        # Costmap Clearing Services
        self.clear_global_client = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap",
            callback_group=cb_group,
        )
        self.clear_local_client = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap",
            callback_group=cb_group,
        )

    def _transform_to_map_frame(self, base_pose: PoseStamped) -> PoseStamped:
        """Transforms base_link relative pose into global map frame for Nav2."""
        self.get_logger().info(
            f"[NODE 2 TF] Transform lookup: '{base_pose.header.frame_id}' -> 'map'..."
        )
        try:
            map_pose = self.tf_buffer.transform(
                base_pose,
                "map",
                timeout=Duration(seconds=3.0)
            )
            self.get_logger().info(
                f"[NODE 2 TF] Transform succeeded: base ({base_pose.pose.position.x:.2f}, {base_pose.pose.position.y:.2f}) "
                f"--> map ({map_pose.pose.position.x:.2f}, {map_pose.pose.position.y:.2f})"
            )
            return map_pose
        except Exception as e:
            self.get_logger().error(f"[NODE 2 TF] Transform lookup failed: {e}")
            return None

    def _on_approach_pose(self, pose: PoseStamped):
        self.get_logger().info(
            f"[NODE 2 TRIGGER] Received /erc/shelf_approach_pose! "
            f"Position: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}, frame={pose.header.frame_id}"
        )

        if self._nav_in_progress:
            self.get_logger().warn("[NODE 2 BUSY] Navigation already active; ignoring repeat trigger.")
            self._publish_status("NAV_BUSY")
            return

        self._retry_used = False
        map_pose = self._transform_to_map_frame(pose)
        if map_pose is not None:
            self._send_nav_goal(map_pose)
        else:
            self.get_logger().error("[NODE 2 ERROR] Pose transformation failed; aborting navigation.")
            self._publish_status("NAV_FAILED")

    def _send_nav_goal(self, pose: PoseStamped):
        self.get_logger().info("[NODE 2 NAV] Connecting to /navigate_to_pose action server...")
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("[NODE 2 NAV] Action server /navigate_to_pose unavailable after 5s.")
            self._nav_in_progress = False
            self._publish_status("NAV_FAILED")
            return

        self._nav_in_progress = True
        self._current_pose = pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f"[NODE 2 NAV] Dispatching goal: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}"
        )

        send_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self._on_nav_feedback
        )
        send_future.add_done_callback(self._on_goal_response)

    def _on_nav_feedback(self, feedback_msg):
        remaining = feedback_msg.feedback.distance_remaining
        self.get_logger().info(f"[NODE 2 PROGRESS] Distance remaining: {remaining:.2f} m")

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"[NODE 2 NAV] Goal request failed: {exc}")
            self._handle_failure()
            return

        if not goal_handle.accepted:
            self.get_logger().error("[NODE 2 NAV] Nav2 controller rejected the target goal.")
            self._handle_failure()
            return

        self.get_logger().info("[NODE 2 NAV] Goal accepted. Executing path...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().error(f"[NODE 2 NAV] Result callback failed: {exc}")
            self._handle_failure()
            return

        self.get_logger().info(f"[NODE 2 NAV] Execution completed with status: {status}")
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._on_nav_success()
        else:
            self.get_logger().warn(f"[NODE 2 NAV] Navigation goal failed (status={status}).")
            self._handle_failure()

    def _on_nav_success(self):
        self._nav_in_progress = False
        self.get_logger().info("==================================================")
        self.get_logger().info("[NODE 2 SUCCESS] Navigation complete: REACHED_SHELF")
        self.get_logger().info("==================================================")
        self._publish_status("REACHED_SHELF")

    def _handle_failure(self):
        if not self._retry_used:
            self.get_logger().warn("[NODE 2 RECOVERY] Navigation failed. Initiating costmap recovery...")
            self._retry_used = True
            self._clear_costmaps(retry_pose=self._current_pose)
        else:
            self.get_logger().error("[NODE 2 FAILURE] Recovery failed. Reporting NAV_FAILED.")
            self._nav_in_progress = False
            self._publish_status("NAV_FAILED")

    def _clear_costmaps(self, retry_pose: PoseStamped):
        if retry_pose is None:
            self.get_logger().error("[NODE 2 RECOVERY] No previous pose available; cannot retry.")
            self._nav_in_progress = False
            self._publish_status("NAV_FAILED")
            return

        if not self.clear_global_client.wait_for_service(timeout_sec=3.0) or \
           not self.clear_local_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("[NODE 2 RECOVERY] Costmap services unavailable; re-sending goal directly.")
            self._nav_in_progress = False
            self._send_nav_goal(retry_pose)
            return

        req = ClearEntireCostmap.Request()
        gfut = self.clear_global_client.call_async(req)
        lfut = self.clear_local_client.call_async(req)

        def _after_clear(_):
            self.get_logger().info("[NODE 2 RECOVERY] Costmaps cleared. Re-issuing goal...")
            self._nav_in_progress = False
            self._send_nav_goal(retry_pose)

        gfut.add_done_callback(_after_clear)

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"[NODE 2 STATUS] Published /erc/nav_status: '{status}'")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()