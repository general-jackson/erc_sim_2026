"""Kinematics for the left arm, built from the URDF with only the standard library.

kdl_parser_py is not in the ERC image and the graders run that image as it
ships, so installing a URDF parser is not an option. The conversion KDL needs is
small enough to do here: a joint's origin translation, its axis rotated into the
parent frame, and the full origin transform as the segment's frame. Verified
against TF - forward kinematics on this chain matched the transform TF reported
for the same frame to 0.00000 m.

The chain runs base_link -> gripper_left_grasping_link and includes
torso_lift_joint, so raising the torso is part of the solution rather than a
separate decision. Note the torso is driven by its own controller, so callers
must split the answer - see split_solution().
"""
import math
import random
import xml.etree.ElementTree as ET

import PyKDL as kdl

ROOT_LINK = 'base_link'
GRASP_LINK = 'gripper_left_grasping_link'
TORSO_JOINT = 'torso_lift_joint'
# From the URDF. The torso is slow and the controller enforces it.
TORSO_VELOCITY_LIMIT = 0.035  # m/s

# The gripper reaches along -X of its grasping frame and its fingers close along
# +Y (measured from the fingertip links, not assumed). This rotation therefore
# The fingers extend along the grasping link's local +X (measured from TF
# 2026-09-11: +0.157 m from gripper base to grasp point). Yaw 0 keeps that
# axis on base +X, into the shelf, with the fingers opening along base Y
# across the book's 3 cm spine. A yaw of pi was used until 2026-09-11 and
# pointed the fingers back at the robot - every grasp closed on air.
LEVEL_APPROACH = kdl.Rotation.RPY(0.0, 0.0, 0.0)

# How far the wrist may tilt away from straight-on, tried in order so a level
# grasp still wins whenever one exists.
#
# The range matters more than it looks. Sampling the workspace by forward
# kinematics, the furthest the gripper reaches at a book's height is 0.73 m
# while it stays within 37 degrees of level - and the books sit at 0.76 m,
# with the base unable to close the gap. Allowing 53 degrees lifts that to
# 0.77-0.81 m and brings rows 2 to 4 into range; 66 degrees reaches 0.85 m.
# Stopping at -45 was why the arm kept reporting no solution at the shelf.
#
# Row 1 is not rescued by any of this: it tops out at 0.69 m even at 66
# degrees, so the top row cannot be grasped from where the base can stand.
# Level first, then small tilts either way (negative = fingertips down).
# Upward tilts added 2026-09-11: with the fingers pointing into the shelf,
# row 1 only solves tilted up.
APPROACH_PITCHES_DEG = (0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0,
                        -60.0, -70.0)


def _xyz(node, attr, default=(0.0, 0.0, 0.0)):
    if node is None or node.get(attr) is None:
        return default
    return tuple(float(v) for v in node.get(attr).split())


def _link(joint, tag):
    """Return the link a URDF joint names as its `tag` ('parent' or 'child')."""
    element = joint.find(tag)
    link = element.get('link') if element is not None else None
    if link is None:
        raise ValueError('joint %s has no %s link' % (joint.get('name'), tag))
    return link


def build_chain(urdf_text, root, tip):
    """Build the serial KDL chain from `root` to `tip`, with each movable joint's limits."""
    robot = ET.fromstring(urdf_text)
    by_child = {_link(j, 'child'): j for j in robot.findall('joint')}

    path, link = [], tip
    while link != root:
        joint = by_child.get(link)
        if joint is None:
            raise ValueError('no path from %s up to %s (stuck at %s)' % (tip, root, link))
        path.append(joint)
        link = _link(joint, 'parent')
    path.reverse()

    chain = kdl.Chain()
    limits = []
    for joint in path:
        origin = joint.find('origin')
        frame = kdl.Frame(kdl.Rotation.RPY(*_xyz(origin, 'rpy')),
                          kdl.Vector(*_xyz(origin, 'xyz')))
        jtype, name = joint.get('type'), joint.get('name')
        if jtype in ('revolute', 'continuous', 'prismatic'):
            axis = kdl.Vector(*_xyz(joint.find('axis'), 'xyz', (1.0, 0.0, 0.0)))
            kind = kdl.Joint.TransAxis if jtype == 'prismatic' else kdl.Joint.RotAxis
            kjoint = kdl.Joint(name, frame.p, frame.M * axis, kind)
            lim = joint.find('limit')
            if jtype == 'continuous' or lim is None:
                limits.append((name, -math.pi, math.pi))
            else:
                limits.append((name, float(lim.get('lower', -math.pi)),
                               float(lim.get('upper', math.pi))))
        else:
            kjoint = kdl.Joint(name, kdl.Joint.Fixed)
        chain.addSegment(kdl.Segment(_link(joint, 'child'), kjoint, frame))
    return chain, limits


class ArmKinematics:
    """Position IK for the left arm that respects the joint limits.

    The Levenberg-Marquardt solver converges beautifully here but ignores
    limits, and its favourite trick is to slide the prismatic torso to 0.69 m
    when it only travels 0.35. The Newton-Raphson solver that does respect
    limits succeeds from only a minority of starting guesses, so this tries
    many, which is cheap.
    """

    def __init__(self, urdf_text, seed_attempts=150, position_tolerance=0.012,
                 torso_position=None):
        """`torso_position` pins the torso instead of letting IK choose it.

        The torso runs on its own controller, so a solution that moves it is
        only half executable by the arm publisher. Callers raise the torso
        first with torso_ramp() and then solve with it pinned at its measured
        height, leaving the arm as the only thing IK is free to choose.
        """
        self.chain, self.limits = build_chain(urdf_text, ROOT_LINK, GRASP_LINK)
        self.joint_names = [n for n, _lo, _hi in self.limits]
        self.n = self.chain.getNrOfJoints()
        self.seed_attempts = seed_attempts
        self.position_tolerance = position_tolerance

        self._lo = kdl.JntArray(self.n)
        self._hi = kdl.JntArray(self.n)
        for i, (nm, lo, hi) in enumerate(self.limits):
            if nm == TORSO_JOINT and torso_position is not None:
                lo = hi = float(torso_position)
                lo -= 1e-4
                hi += 1e-4
            self._lo[i], self._hi[i] = lo, hi

        # These are held as attributes deliberately. PyKDL's NR_JL solver keeps
        # raw pointers to the solvers it is given, so passing either as a
        # temporary leaves it pointing at freed memory and the process dies
        # with a segfault the first time IK runs.
        self._fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self._vel = kdl.ChainIkSolverVel_pinv(self.chain)
        self._ik = kdl.ChainIkSolverPos_NR_JL(
            self.chain, self._lo, self._hi, self._fk, self._vel, 300, 1e-4,
        )
        self._rng = random.Random(0)

    def _within_limits(self, q):
        return all(self._lo[i] - 1e-6 <= q[i] <= self._hi[i] + 1e-6 for i in range(self.n))

    def forward(self, q):
        frame = kdl.Frame()
        self._fk.JntToCart(q, frame)
        return frame

    def solve(self, x, y, z, pitches=APPROACH_PITCHES_DEG, current=None,
              keep=6):
        """Joint angles putting the gripper at (x, y, z) in base_link.

        Returns (joint_positions, pitch_used) or (None, None). Pitches are
        tried in order, so a level grasp wins whenever one exists.

        A 7-jointed arm reaches the same point many ways, and the random seeds
        this needs will happily return one on the far side of the workspace.
        Taking the first answer meant the arm swinging through a huge
        reconfiguration between two nearby targets and not arriving inside the
        time allowed. So several solutions are collected for the first pitch
        that works and the one closest to `current` is returned - shorter
        motion, and less chance of sweeping books off the shelf on the way.
        """
        for pitch in pitches:
            # -pitch: with yaw 0, positive rotation about Y tips local +X down, and
            # APPROACH_PITCHES_DEG uses negative to mean fingertips down.
            rotation = kdl.Rotation.RPY(0.0, -math.radians(pitch), 0.0)
            target = kdl.Frame(rotation, kdl.Vector(x, y, z))
            found = []
            for attempt in range(self.seed_attempts):
                seed = kdl.JntArray(self.n)
                for i in range(self.n):
                    if attempt == 0 and current is not None:
                        seed[i] = float(current[i])
                    elif attempt == 0:
                        seed[i] = 0.0
                    else:
                        seed[i] = self._rng.uniform(self._lo[i], self._hi[i])
                out = kdl.JntArray(self.n)
                if self._ik.CartToJnt(seed, target, out) < 0:
                    continue
                if not self._within_limits(out):
                    continue
                reached = self.forward(out)
                error = math.sqrt(sum((reached.p[i] - target.p[i]) ** 2 for i in range(3)))
                if error <= self.position_tolerance:
                    found.append([out[i] for i in range(self.n)])
                    if len(found) >= keep:
                        break
            if found:
                if current is not None:
                    found.sort(key=lambda q: max(abs(q[i] - current[i])
                                                 for i in range(self.n)))
                return found[0], pitch
        return None, None

    def torso_ramp(self, start, target, step_seconds=2.0, fraction=0.5):
        """Return stepped positions and times that raise or lower the torso.

        The torso will not accept a single distant setpoint: it travels at
        0.035 m/s and the controller declines a trajectory implying more, so a
        lone "go to 0.32" command leaves it sitting still while the controller
        reports the command going out. Points spaced inside the limit track it
        exactly. Returns [(position, time_from_start), ...].
        """
        speed = TORSO_VELOCITY_LIMIT * fraction
        points, position, elapsed = [], float(start), 0.0
        rising = target > position
        while (position < target) if rising else (position > target):
            elapsed += step_seconds
            move = speed * step_seconds
            position = (min(target, position + move) if rising
                        else max(target, position - move))
            points.append((position, elapsed))
        return points

    def split_solution(self, q):
        """Separate the torso from the arm; they have different controllers."""
        torso, arm = None, []
        for name, value in zip(self.joint_names, q):
            if name == TORSO_JOINT:
                torso = value
            else:
                arm.append((name, value))
        return torso, arm
