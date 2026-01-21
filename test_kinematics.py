import numpy as np
from ninja_kinematics import SimpleIK

# Initialize PyBullet kinematics solver (loads URDF)
ik_solver = SimpleIK()

print("=== Test 1: FK then IK back to same position ===")
joint_angles = np.array([0.0, 10.0, 10.0, 45.0])
position = ik_solver.forward_kinematics(joint_angles)
print(f"Original joints (deg): {joint_angles}")
print(f"End-effector XYZ (m): {position}")

# IK back to same position - should return similar joint angles

new_joints = ik_solver.inverse_kinematics(position, current_joints=joint_angles)
print(f"IK solution (deg): {new_joints}")
print(f"Joint difference: {np.abs(new_joints - joint_angles)}")

# Verify
verify_pos = ik_solver.forward_kinematics(new_joints)
print(f"Verification XYZ (m): {verify_pos}")
print(f"Position error (mm): {np.linalg.norm(verify_pos - position) * 1000:.3f}")

print("\n=== Test 2: Move up 5cm in Z ===")
target_pos = position.copy()
target_pos[2] += 0.05  # +5cm in Z
print(f"Original joints (deg): {joint_angles}")
print(f"Target XYZ (m): {target_pos}")

# Try IK
new_joints_2 = ik_solver.inverse_kinematics(target_pos, current_joints=joint_angles)
print(f"IK solution (deg): {new_joints_2}")

# Verify - but let's check intermediate steps
print("\n[Debug] Setting joints to IK solution and reading back:")
import pybullet as p
for i, angle in enumerate(np.deg2rad(new_joints_2)):
    p.resetJointState(ik_solver.robot_id, i, angle)
    actual = p.getJointState(ik_solver.robot_id, i)[0]
    print(f"  Joint {i}: Set={angle:.10f} rad, Read={actual:.10f} rad, Diff={abs(angle-actual):.2e}")

verify_pos_2 = ik_solver.forward_kinematics(new_joints_2)
print(f"\nVerification XYZ (m): {verify_pos_2}")
print(f"Target XYZ (m):       {target_pos}")
print(f"Difference (mm):      {(verify_pos_2 - target_pos) * 1000}")
print(f"Position error (mm): {np.linalg.norm(verify_pos_2 - target_pos) * 1000:.6f}")
