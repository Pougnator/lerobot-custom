import pybullet as p
import pybullet_data
import time
import numpy as np
import os

# Start PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load the robot URDF
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "so101_ninja1.urdf")
robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

# Get joint info
num_joints = p.getNumJoints(robot_id)
print(f"Robot loaded with {num_joints} joints")

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: {joint_info[1].decode('utf-8')} (Type: {joint_info[2]})")

# Change color of entire upper_arm_link (link 2) to green - includes motor and arm
for shape_id in range(-1, 10):  # -1 means all shapes
    try:
        p.changeVisualShape(robot_id, 2, shape_id, rgbaColor=[0, 1, 0, 1])
    except:
        pass

# Add sliders for joint control
joint_ids = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
        joint_ids.append(i)
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        param_id = p.addUserDebugParameter(
            joint_info[1].decode('utf-8'),
            lower_limit,
            upper_limit,
            0
        )

# Simulation loop
print("\nUse the sliders to control the robot joints!")
print("Close the PyBullet window to exit.")

try:
    while True:
        # Read slider values and set joint positions
        for idx, joint_id in enumerate(joint_ids):
            slider_value = p.readUserDebugParameter(idx)
            p.setJointMotorControl2(
                robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=slider_value
            )
        
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    p.disconnect()