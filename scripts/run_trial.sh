#!/usr/bin/env bash
# Run one ERC Phase 1 trial end to end and report what scored.
#
#   ./scripts/run_trial.sh [column] [colour] [seed]
#   ./scripts/run_trial.sh 2 red 42        # defaults
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
SIM_WARMUP=30       # world populates on staggered timers: robot 3s, books 5s, controllers 8s
TRIAL_SECONDS=150

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

say "Starting simulator (headless, ERC_SEED=$SEED)"
dexecd "cd /opt/erc_ws && source install/setup.bash && export ERC_SEED=$SEED && \
        ros2 launch erc_bringup simulation.launch.py headless:=true > /tmp/sim.log 2>&1"
sleep "$SIM_WARMUP"

say "Running solution: column=$COLUMN colour=$COLOUR"
dexecd "cd /opt/erc_ws && source install/setup.bash && \
        ros2 launch solution4 solution.launch.py \
        shelf_column_number:=$COLUMN book_colour:=$COLOUR > /tmp/sol.log 2>&1"
sleep "$TRIAL_SECONDS"

say "Results"
dexec "source /opt/erc_ws/install/setup.bash
echo '--- /erc/shelf_column_identification ---'
timeout 6 ros2 topic echo --once /erc/shelf_column_identification std_msgs/msg/Int32 2>&1 | head -2
echo '--- /erc/shelf_row_identification ---'
timeout 6 ros2 topic echo --once /erc/shelf_row_identification std_msgs/msg/Int32 2>&1 | head -2
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
# 2. Digit detection depends on the search band. The classifier labels the
#    robot's own gripper as a digit with >0.9 confidence. What rejects it is
#    digit_search_band (default 0.5 = top half of frame), because the markers
#    sit at z=2.26m and the grippers do not. If detection starts failing,
#    that parameter is the first thing to look at.
#
# 3. The arms occlude the camera. STEP 1 swings both into view; only one arm
#    is allowed for the task anyway, so tucking the unused one would free a
#    large part of the frame.
#
# 4. Nothing downstream of detection is implemented: no grasp, no navigation
#    to the bin, no placement. That is 10 of the 19 points.
#
# 5. Software rendering only (no NVIDIA), so the camera runs ~10Hz against the
#    30Hz spec. Timings here will not match the graders' machine.
# ---------------------------------------------------------------------------
