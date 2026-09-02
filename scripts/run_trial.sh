#!/usr/bin/env bash
# Run one ERC Phase 1 trial end to end and report what scored.
#
#   ./scripts/run_trial.sh [column] [colour] [seed]
#   ./scripts/run_trial.sh 2 red 42        # defaults
#
# Set HEADLESS=false to watch it happen in the Gazebo window instead of only
# reading the result:
#
#   HEADLESS=false ./scripts/run_trial.sh 2 red 42
#
# That needs an X server the container can reach - ./docker/up.sh arranges it
# on this machine - and it is slower, because there is no GPU here and the
# window is rendered in software.
#
# Run from the host (not inside the container). The container must already be
# up: ./docker/up.sh
#
# Proven on 2026-09-01: publishes both scoring topics, writes both annotated
# images, reaches STEP 7. See KNOWN ISSUES at the bottom before trusting the
# row value.
set -uo pipefail

COLUMN="${1:-2}"
COLOUR="${2:-red}"
SEED="${3:-42}"
CONTAINER="erc_sim"
HEADLESS="${HEADLESS:-true}"
SIM_WARMUP=30       # world populates on staggered timers: robot 3s, books 5s, controllers 8s
# Long enough to cover the whole pipeline with margin: the solution
# reaches STEP 7 about 15s after launch and the drive to the shelf takes
# another 40s. Raise it if a step downstream of navigation is added.
TRIAL_SECONDS=90

say() { printf '\n\033[1;34m[trial]\033[0m %s\n' "$*"; }

docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true || {
  echo "Container '$CONTAINER' is not running. Start it with ./docker/up.sh" >&2
  exit 1
}

dexec()  { docker exec "$CONTAINER" /entrypoint.sh bash -c "$1"; }
dexecd() { docker exec -d "$CONTAINER" /entrypoint.sh bash -c "$1"; }

say "Clean slate"
docker restart "$CONTAINER" >/dev/null
sleep 8
dexec "rm -f /opt/erc_ws/src/erc_images/*.png /tmp/sol.log /tmp/sim.log"

say "Building solution4"
dexec "cd /opt/erc_ws && colcon build --symlink-install --packages-select solution4 2>&1 | tail -2"

say "Starting simulator (headless=$HEADLESS, ERC_SEED=$SEED)"
dexecd "cd /opt/erc_ws && source install/setup.bash && export ERC_SEED=$SEED && \
        ros2 launch erc_bringup simulation.launch.py headless:=$HEADLESS > /tmp/sim.log 2>&1"
sleep "$SIM_WARMUP"

say "Running solution: column=$COLUMN colour=$COLOUR"
dexecd "cd /opt/erc_ws && source install/setup.bash && \
        ros2 launch solution4 solution.launch.py \
        shelf_column_number:=$COLUMN book_colour:=$COLOUR > /tmp/sol.log 2>&1"
sleep "$TRIAL_SECONDS"

say "Results"
dexec "source /opt/erc_ws/install/setup.bash
# The ros2 CLI daemon caches the ROS graph and does not survive the container
# restart at the top of this script. A stale daemon reports no publishers, so
# 'topic echo' below would print nothing on a perfectly good run. Stop it and
# bypass it entirely - slower discovery, but the results are trustworthy.
ros2 daemon stop >/dev/null 2>&1 || true
echo '--- /erc/shelf_column_identification ---'
timeout 20 ros2 topic echo --once --no-daemon --spin-time 5 /erc/shelf_column_identification std_msgs/msg/Int32 2>&1 | head -2
echo '--- /erc/shelf_row_identification ---'
timeout 20 ros2 topic echo --once --no-daemon --spin-time 5 /erc/shelf_row_identification std_msgs/msg/Int32 2>&1 | head -2
echo '--- erc_images/ ---'
ls -1 /opt/erc_ws/src/erc_images/ 2>/dev/null
echo '--- pipeline log ---'
grep node1_search /tmp/sol.log | grep -E 'SCORE|STEP 5|STEP 7|STEP 9|Candidate' | tail -12"

say "Images are on the host at src/erc_images/ (written as root; sudo chown if git complains)"

# ---------------------------------------------------------------------------
# KNOWN ISSUES - read before trusting a run
#
# 1. Row value is unreliable. A passing run logged
#      "Only 1 of 4 books visible in the column - the row index is an estimate"
#    and then published row 1. The row is derived by ranking the four coloured
#    books top-to-bottom, so seeing only one makes the rank meaningless. Treat
#    any run carrying that warning as a column-only score.
#
# 2. Digit detection confirms the marker plate before classifying anything.
#    Each plate is a 0.3m square whose texture is a tight crop of the digit
#    stretched to fill it, so the glyph covers ~0.78 of the plate's width and
#    ~0.83 of its height. The detector segments the plate's flat face, checks
#    the result is a quadrilateral of about the right shape, rectifies it onto
#    a square, and checks the glyph fills that square in the proportion above.
#    Only then is anything classified. This replaced a stack of brightness and
#    position filters that each rejected one false positive and revealed the
#    next. digit_search_band still trims the search to the top of the frame,
#    but it is an optimisation now, not what rejects the grippers.
#
# 3. The arms occlude the camera. STEP 1 swings both into view; only one arm
#    is allowed for the task anyway, so tucking the unused one would free a
#    large part of the frame.
#
# 4. The base drives to the target column on odometry, not Nav2. The ERC
#    simulation ships no Nav2 stack at all - no /navigate_to_pose, no map
#    server, no AMCL, no costmaps - so goals in the 'map' frame failed on
#    every run and the robot never moved. Node 1 deprojects the confirmed
#    marker plate to a point in base_link and node 2 closes the loop on
#    odometry and the front LiDAR, using the holonomic base to strafe into
#    line with the column. Arrives ~0.09m off centre with ~0.43m clearance.
#
# 5. Nothing downstream of navigation is implemented: no grasp, no drive to
#    the bin, no placement. That is 10 of the 19 points.
#
# 6. The simulator sometimes starts up incomplete, and the run has to be
#    thrown away rather than debugged. Seen twice, differently: once the
#    camera never delivered a callback and the log stopped after STEP 1, once
#    every LiDAR beam returned inf so STEP 3 never resolved the table. Both
#    are upstream of any solution logic. Check the front scan is returning
#    (818 beams, all finite when healthy) before blaming a change.
#
# 7. Software rendering only (no NVIDIA), so the camera runs ~10Hz against the
#    30Hz spec. Timings here will not match the graders' machine.
# ---------------------------------------------------------------------------
