"""
Tests for mykinematics.get_wrist_xz.

Convention: knife_tilt_deg is the angle of L4 (knife tip link) from horizontal.
  0°  = horizontal (tip pointing forward)
  negative = tip down, positive = tip up
L3 (wrist_knife) is always 45° behind L4: L3 angle = tilt - 45°.

Run with:  py -3.9 -m pytest test_mykinematics.py -v
"""

import numpy as np
import pytest
from mykinematics import get_wrist_xz, robot_links

L3 = robot_links["wrist_knife"]       # 97.0 mm  (wrist → knife base)
L4 = robot_links["knife_knifetip"]    # 80.0 mm  (knife base → tip)


def make_pos(x, z):
    """Build a 3-element world-coord vector [x, 0, z] for convenience."""
    return np.array([x, 0.0, z])


# ── Basic tilt=0 case (horizontal knife) ─────────────────────────────────────

def test_tilt_zero_x():
    """
    tilt = 0°: L4 horizontal, L3 at -45°.
      wrist_x = 10 - L4*cos(0°) - L3*cos(-45°)
              = 10 - L4 - L3*cos(45°)
    """
    wx, wz = get_wrist_xz(make_pos(10, 0), knife_tilt_deg=0.0)
    alpha = np.deg2rad(0.0)
    expected_x = 10.0 - L4 * np.cos(alpha) - L3 * np.cos(alpha - np.deg2rad(45.0))
    assert abs(wx - expected_x) < 1e-9


def test_tilt_zero_z():
    """
    tilt = 0°: L4 horizontal, L3 at -45°.
      wrist_z = 0 - L4*sin(0°) - L3*sin(-45°)
              = L3*sin(45°)
    """
    wx, wz = get_wrist_xz(make_pos(10, 0), knife_tilt_deg=0.0)
    alpha = np.deg2rad(0.0)
    expected_z = 0.0 - L4 * np.sin(alpha) - L3 * np.sin(alpha - np.deg2rad(45.0))
    assert abs(wz - expected_z) < 1e-9


# ── Consistency with FK: round-trip ──────────────────────────────────────────

def test_round_trip_via_fk():
    """
    get_wrist_xz should invert the wrist→tip portion of FK.

    For any tip position produced by FK at a known tilt, get_wrist_xz
    should recover the wrist position that FK also reported.

    knife_tilt_deg = sum_trigo_deg + 45°  (L4 direction from horizontal)
    """
    from mykinematics import forward_kinematics

    # Joint angles that give a well-defined configuration
    joint_angles = [0.0, -30.0, 10.0, 45.0]   # degrees (servo frame)
    tip = forward_kinematics(joint_angles)      # [x, y, z] in mm

    # Determine the tilt: L4 direction = sum_trigo + 45°
    from mykinematics import joint_angles_to_trigo
    trigo = joint_angles_to_trigo(joint_angles)
    alpha_rad = np.sum(np.deg2rad(trigo))
    knife_tilt_deg = np.degrees(alpha_rad) + 45.0

    # forward_kinematics adds Z_FK_BIAS_MM to the tip z; subtract it before inverting
    from mykinematics import Z_FK_BIAS_MM
    tip_corrected = tip.copy()
    tip_corrected[2] -= Z_FK_BIAS_MM
    wx, wz = get_wrist_xz(tip_corrected, knife_tilt_deg=knife_tilt_deg)

    # FK wrist position
    import mykinematics as mk
    theta1, theta2, theta3 = np.deg2rad(trigo)
    from mykinematics import upper_arm_coordinates, elbow_wrist_angle, upper_arm_elbow_angle
    x1 = mk.robot_links["upperarm_forearm"] * np.cos(theta1 - np.deg2rad(upper_arm_elbow_angle)) + upper_arm_coordinates[0]
    z1 = mk.robot_links["upperarm_forearm"] * np.sin(theta1 - np.deg2rad(upper_arm_elbow_angle)) + upper_arm_coordinates[2]
    x2 = mk.robot_links["forearm_wrist"] * np.cos(theta1 + theta2 + np.deg2rad(elbow_wrist_angle)) + x1
    z2 = mk.robot_links["forearm_wrist"] * np.sin(theta1 + theta2 + np.deg2rad(elbow_wrist_angle)) + z1

    assert abs(wx - x2) < 1e-6, f"wrist_x mismatch: got {wx:.4f}, expected {x2:.4f}"
    assert abs(wz - z2) < 1e-6, f"wrist_z mismatch: got {wz:.4f}, expected {z2:.4f}"


# ── Tip at origin ─────────────────────────────────────────────────────────────

def test_tip_at_origin_tilt_zero():
    """Tip at (0, 0), tilt=0: L4 horizontal, L3 at -45°."""
    wx, wz = get_wrist_xz(make_pos(0, 0), knife_tilt_deg=0.0)
    alpha = np.deg2rad(0.0)
    expected_x = -L4 * np.cos(alpha) - L3 * np.cos(alpha - np.deg2rad(45.0))
    expected_z = -L4 * np.sin(alpha) - L3 * np.sin(alpha - np.deg2rad(45.0))
    assert abs(wx - expected_x) < 1e-9
    assert abs(wz - expected_z) < 1e-9


# ── Linearity in tip position ─────────────────────────────────────────────────

def test_linear_offset():
    """
    Shifting the tip by (dx, dz) at fixed tilt should shift the wrist
    by exactly the same (dx, dz).
    """
    dx, dz = 50.0, -30.0
    wx0, wz0 = get_wrist_xz(make_pos(0, 0),  knife_tilt_deg=15.0)
    wx1, wz1 = get_wrist_xz(make_pos(dx, dz), knife_tilt_deg=15.0)
    assert abs((wx1 - wx0) - dx) < 1e-9
    assert abs((wz1 - wz0) - dz) < 1e-9
