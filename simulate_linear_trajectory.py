"""
simulate_linear_trajectory.py
==============================
Simulate _jacobian_move_to_xz in PyBullet GUI.

The robot follows a linear wrist trajectory using the same
jacobian_control_step function used for real execution.

"Observe" = read joint angles back from PyBullet after each step
"Send"    = resetJointState in PyBullet

Run with:  py -3.9 simulate_linear_trajectory.py
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

from ninja_kinematics import URDF_PATH
from mykinematics import (
    jacobian_control_step,
    calculate_linear_trajectory,
    get_tilt_from_joints,
    forward_kinematics,
    forward_kinematics_wrist,
)

# ── Trajectory parameters (cm, matching _jacobian_move_to_xz convention) ─────
TARGET_X_CM     = 32.0   # target knife-tip X in cm
TARGET_Z_CM     =  1.0   # target knife-tip Z in cm
TARGET_TILT_DEG =  -20.0  # target tilt (0 = horizontal, negative = tip down)

SPEED_CM_S       = 3.0   # cm/s
STEPS_PER_SECOND = 30     # waypoints per second

# Starting joint angles [shoulder_lift, elbow_flex, wrist_flex] degrees (robot frame)
INITIAL_JOINTS = [0.0, 0.0, 45.0]

# PyBullet joint indices
LIFT_IDX, ELBOW_IDX, WRIST_IDX = 1, 2, 3

STEP_DELAY_S = 0.001   # seconds between steps for visualization


# ── PyBullet helpers ──────────────────────────────────────────────────────────

def _observe(robot_id):
    """Read [shoulder_lift, elbow_flex, wrist_flex] from PyBullet in degrees."""
    return np.array([
        np.degrees(p.getJointState(robot_id, LIFT_IDX)[0]),
        np.degrees(p.getJointState(robot_id, ELBOW_IDX)[0]),
        np.degrees(p.getJointState(robot_id, WRIST_IDX)[0]),
    ])


def _send(robot_id, joint_angles_3, shoulder_pan_deg=0.0):
    """Apply [shoulder_lift, elbow_flex, wrist_flex] + pan to PyBullet."""
    for idx, deg in zip(
        [0, LIFT_IDX, ELBOW_IDX, WRIST_IDX],
        [shoulder_pan_deg, joint_angles_3[0], joint_angles_3[1], joint_angles_3[2]],
    ):
        p.resetJointState(robot_id, idx, np.deg2rad(deg))


# ── Main simulation ───────────────────────────────────────────────────────────

def main():
    # ── PyBullet GUI setup ────────────────────────────────────────────────────
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(URDF_PATH, useFixedBase=True)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.6, cameraYaw=60, cameraPitch=-20,
        cameraTargetPosition=[0.25, 0, 0.1],
    )

    # ── Set initial pose ──────────────────────────────────────────────────────
    _send(robot_id, INITIAL_JOINTS)
    time.sleep(0.5)

    # ── Build trajectory from current FK state (mirrors _jacobian_move_to_xz) ─
    current_joints = _observe(robot_id)
    starting_pos   = forward_kinematics(current_joints)          # mm
    starting_tilt  = get_tilt_from_joints(current_joints)        # degrees
    target_pos     = np.array([TARGET_X_CM * 10.0, 0.0, TARGET_Z_CM * 10.0])  # mm

    print(f"FK start: X={starting_pos[0]/10:.2f} cm  Z={starting_pos[2]/10:.2f} cm  tilt={starting_tilt:.1f}°")
    print(f"Target:   X={TARGET_X_CM:.2f} cm  Z={TARGET_Z_CM:.2f} cm  tilt={TARGET_TILT_DEG:.1f}°")

    try:
        trajectory = calculate_linear_trajectory(
            target_pos, TARGET_TILT_DEG, starting_tilt, starting_pos,
            speed=SPEED_CM_S * 10.0,          # cm/s → mm/s
            steps_per_second=STEPS_PER_SECOND,
        )
    except ValueError as e:
        print(f"Cannot compute trajectory: {e}")
        p.disconnect()
        return

    print(f"Trajectory: {len(trajectory)} waypoints  "
          f"({SPEED_CM_S} cm/s, {STEPS_PER_SECOND} steps/s)\n")

    # ── Simulate closed-loop execution (mirrors _execute_linear_trajectory) ────
    for i, (x_wrist_mm, z_wrist_mm, tilt_deg) in enumerate(trajectory):
        # 1. Observe (simulates encoder readback)
        current_angles = _observe(robot_id)

        # 2. Jacobian step
        new_angles = jacobian_control_step(current_angles, x_wrist_mm, z_wrist_mm, tilt_deg)
        if new_angles is None:
            print(f"[STEP {i:3d}] Jacobian step failed — stopping.")
            break

        # 3. Send
        _send(robot_id, new_angles)

        # 4. Print progress
        wx, wz    = forward_kinematics_wrist(new_angles)
        fk_tip    = forward_kinematics(new_angles)
        wrist_err = np.sqrt((wx - x_wrist_mm)**2 + (wz - z_wrist_mm)**2)
        print(f"[STEP {i:3d}] "
              f"tgt_wrist=({x_wrist_mm:6.1f}, {z_wrist_mm:6.1f}) mm  "
              f"fk_wrist=({wx:6.1f}, {wz:6.1f}) mm  err={wrist_err:.2f} mm  "
              f"tip=({fk_tip[0]/10:.2f}, {fk_tip[2]/10:.2f}) cm  tilt={tilt_deg:+.1f}°")

        time.sleep(STEP_DELAY_S)

    print("\nSimulation complete. Press Enter to exit.")
    input()
    p.disconnect()


if __name__ == "__main__":
    main()
