"""
Robot-arm objective: reach a target with a 6-joint arm without colliding.

A pure-Python planar inverse-kinematics problem. A 6-link arm is anchored at a
base; the HumpDay objective takes a 6-D point in [0,1]^6, decodes it into the six
joint angles (each clamped to ±90°), runs forward kinematics, and scores how
close the tip gets to a target — minus heavy penalties for any link that
intersects one of three circular obstacles. It returns the negative score.

The landscape is a **constrained inverse-kinematics** problem with multiple
disjoint solution branches (elbow-up / elbow-down / wrap-around): naive poses
crash an arm through an obstacle, and the feasible corridors to the target are
narrow. Most optimisers find a collision-free pose that lands the tip on the
target; the ones that don't get stuck against the penalty cliffs.

Mirrors the browser demo docs/applications/robot-arm.html.
"""

from __future__ import annotations

import math

N_JOINTS = 6
N_DIM = N_JOINTS
LINK_LEN = (120, 105, 90, 75, 60, 45)
JOINT_ANGLE_MAX = math.pi / 2  # each joint clamped to ±90°
LINK_HW = 9  # link half-thickness (collision)
BASE = (400.0, 478.0)
TARGET = (700.0, 180.0)
OBSTACLES = ((450.0, 360.0, 55.0), (600.0, 320.0, 50.0), (540.0, 190.0, 38.0))


def decode(u):
    return [(v - 0.5) * 2 * JOINT_ANGLE_MAX for v in u]


def _forward_kinematics(angles):
    pos = [BASE]
    acc = -math.pi / 2  # base joint points up
    for i in range(N_JOINTS):
        acc += angles[i]
        px, py = pos[-1]
        pos.append((px + LINK_LEN[i] * math.cos(acc), py + LINK_LEN[i] * math.sin(acc)))
    return pos


def _seg_circle_dist(p1, p2, cx, cy):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    len2 = dx * dx + dy * dy
    if len2 < 1e-9:
        return math.hypot(p1[0] - cx, p1[1] - cy)
    t = ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / len2
    t = max(0.0, min(1.0, t))
    return math.hypot(p1[0] + t * dx - cx, p1[1] + t * dy - cy)


def evaluate_pose(u):
    """Return (score, tip_error, collisions) for a pose in [0,1]^6."""
    positions = _forward_kinematics(decode(u))
    tip = positions[-1]
    tip_err = math.hypot(tip[0] - TARGET[0], tip[1] - TARGET[1])
    collisions = 0
    deepest = 0.0
    for i in range(N_JOINTS):
        p1, p2 = positions[i], positions[i + 1]
        for ox, oy, r in OBSTACLES:
            d = _seg_circle_dist(p1, p2, ox, oy)
            if d < r + LINK_HW:
                collisions += 1
                deepest = max(deepest, r + LINK_HW - d)
    reach = max(0.0, 100 - tip_err * 0.25)
    score = reach - (collisions * 25 + deepest * 0.4)
    return score, tip_err, collisions


def objective(u):
    """HumpDay objective: negative reach-minus-collision score (minimise)."""
    return -evaluate_pose(u)[0]
