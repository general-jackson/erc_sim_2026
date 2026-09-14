# solution4

ERC 2026 Phase 1 entry for the Library Assistant Robot task on TIAGo Pro: read
the target shelf column from its digit marker, find the book of the requested
colour, grasp it, and deliver it to the collection bin.

## Running

```bash
ros2 launch solution4 solution.launch.py shelf_column_number:=<1-5> book_colour:=<red|green|blue|yellow>
```

Build with `colcon build --packages-select solution4`. Runtime dependencies are
listed in `package.xml`; the only one beyond the base image's ROS packages is
`python3-pykdl`, used for the arm's inverse kinematics.

## Nodes

The three nodes hand over through latched topics, so each stage starts only
when the previous one has finished, and a node that starts late still receives
its trigger.

| node | file | does | publishes |
|---|---|---|---|
| `node1_search` | `book_target_node.py` | confirms the marker plate, reads the digit, finds the book by colour, derives the row from the book's measured height, writes annotated images to `src/erc_images/` | `/erc/shelf_column_identification`, `/erc/shelf_row_identification`, `/erc/shelf_approach_pose` |
| `node2_navigation` | `nav_node.py` | anchors the approach pose in odometry and drives to the column, stopping on the front LiDAR | `/erc/nav_status` (`REACHED_SHELF`) |
| `node3_manipulation` | `manipulation_node.py` | squares up to the shelf, finds and grasps the book, carries it back, lines up on the bin by vision and releases it | `/erc/manipulation_status` |

`/erc/manipulation_status` reports `GRASPING`, `DELIVERING`, `DELIVERED`, or
`GRASP_FAILED` / `DELIVERY_FAILED` / `MANIPULATION_ABORTED`.

Supporting modules:

- `arm_kinematics.py` - a KDL chain for the left arm built from
  `/robot_description` with the standard library, and joint-limited IK.
- `camera_intrinsics.py` - pinhole intrinsics from the robot description,
  since `camera_info` is published only around start-up.

## Parameters

| node | parameter | default | meaning |
|---|---|---|---|
| `node1_search` | `shelf_column_number` | `1` | target column (launch argument) |
| `node1_search`, `node3_manipulation` | `book_colour` | `red` | target colour (launch argument) |
| `node1_search`, `node3_manipulation` | `row_index_base` | `1` | number given to the top row |

## Design notes that matter when changing the code

Each of these was measured against Gazebo ground truth; the comments at the
relevant constants in `manipulation_node.py` give the numbers.

- **Gripper.** Commanding it fully closed drives the finger linkage through the
  book (organisers' issue #2). The node steps it closed and keeps re-publishing
  a hold position of 0.016 just inside the 2 cm spine.
- **Base motion.** Turning with the book held out in front slides the mecanum
  base without odometry seeing it. The book is carried tucked in and turns are
  held to 0.15 rad/s.
- **Arms at the shelf.** Clearing the arms out of the camera's view is done
  after backing away from the shelf, or they stall against it.
- **Bin line-up** is closed on the camera, not on odometry.
- **Gripper topic.** Only the public `/gripper_left_controller` topic is used,
  so the organisers' range clamp stays in the loop.
