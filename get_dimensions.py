from ninja_kinematics import SimpleIK
import numpy as np
import pybullet as p
import pybullet_data
import time
# --- 1. SETUP SIMULATION ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Fix: Add search path for plane.urdf
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

def display_all_joint_positions(ik_solver, joint_angles_deg):
    print("\n=== SYSTEM HIERARCHY & DIMENSIONS ===")
    
    # 1. Check Base State
    base_pos, base_orn = p.getBasePositionAndOrientation(ik_solver.robot_id)
    base_rpy = np.rad2deg(p.getEulerFromQuaternion(base_orn))
    print(f"Base Frame (Link -1):")
    print(f"  Pos (World): {np.array(base_pos)*100} cm")
    print(f"  Rot (Euler): {base_rpy} deg")
    
    # Set angles
    for i, angle in enumerate(np.deg2rad(joint_angles_deg)):
        p.resetJointState(ik_solver.robot_id, i, angle)

    print("\n=== JOINT CHAIN ===")
    print(f"{'Idx':<4} {'Name':<20} {'Parent':<6} {'XYZ World (cm)':<20} {'Offset from Parent (cm)':<25}")
    print("-" * 80)

    # Dictionary to store positions for distance calc
    frame_positions = {-1: np.array(base_pos)}

    for i in range(p.getNumJoints(ik_solver.robot_id)):
        info = p.getJointInfo(ik_solver.robot_id, i)
        joint_name = info[1].decode('utf-8')
        parent_idx = info[16]
        
        # Static URDF Offset
        urdf_offset = np.array(info[14]) * 100
        
        # World Position of the Joint Frame
        link_state = p.getLinkState(ik_solver.robot_id, i, computeForwardKinematics=1)
        world_pos_m = np.array(link_state[4]) 
        world_pos_cm = world_pos_m * 100
        
        # Store for reference
        frame_positions[i] = world_pos_m

        # Calculate actual distance from parent in World Space
        parent_pos_m = frame_positions[parent_idx]
        real_dist = np.linalg.norm(world_pos_m - parent_pos_m) * 100

        print(f"{i:<4} {joint_name:<20} {parent_idx:<6} "
              f"[{world_pos_cm[0]:>5.1f}, {world_pos_cm[1]:>5.1f}, {world_pos_cm[2]:>5.1f}]   "
              f"URDF: {urdf_offset} (Dist: {real_dist:.1f})")

    return frame_positions

# Run output
ik_solver = SimpleIK()
# Test with Zeros
display_all_joint_positions(ik_solver, [0.0, 0.0, 0.0, 0.0])

# Visual Target (Red Sphere)
target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 1])
target_body = p.createMultiBody(baseVisualShapeIndex=target_visual, basePosition=[0.2, 0, 0.2])  # Fixed: Use hardcoded position

# Setup sliders for manual control
# p.configureDebugVisualizer(p.COV_ENABLE_EXPLORER, 0) # Disable explorer
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0.1])

joint_params = []
print("\nControls:")
for i in range(p.getNumJoints(ik_solver.robot_id)):
    info = p.getJointInfo(ik_solver.robot_id, i)
    joint_name = info[1].decode('utf-8')
    joint_type = info[2]
    
    if joint_type == p.JOINT_REVOLUTE:
        lower = info[8]
        upper = info[9]
        if lower >= upper: # Fix bad limits
            lower = -np.pi
            upper = np.pi
            
        param_id = p.addUserDebugParameter(joint_name, np.rad2deg(lower), np.rad2deg(upper), 0)
        joint_params.append((i, param_id))
        print(f"  Added slider for {joint_name}")

# Helper to draw coordinate frames
def draw_frame(pos, orn, length=0.1, duration=0.1):
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    p.addUserDebugLine(pos, pos + rot_mat[:, 0] * length, [1, 0, 0], lineWidth=2, lifeTime=duration)
    p.addUserDebugLine(pos, pos + rot_mat[:, 1] * length, [0, 1, 0], lineWidth=2, lifeTime=duration)
    p.addUserDebugLine(pos, pos + rot_mat[:, 2] * length, [0, 0, 1], lineWidth=2, lifeTime=duration)

print("\nStarting simulation loop...")
print("Move the sliders to change joint angles. Press Ctrl+C to exit.\n")

frame_counter = 0
try:
    while True:
        # A. Read Sliders and Update Robot
        current_angles_deg = []
        for joint_idx, param_id in joint_params:
            angle_deg = p.readUserDebugParameter(param_id)
            angle_rad = np.deg2rad(angle_deg)
            p.resetJointState(ik_solver.robot_id, joint_idx, angle_rad)
            current_angles_deg.append(angle_deg)

        # B. Visualize frames and print positions every 30 frames (~1 second)
        for i in range(p.getNumJoints(ik_solver.robot_id)):
            link_state = p.getLinkState(ik_solver.robot_id, i, computeForwardKinematics=1)
            joint_frame_pos = link_state[4]  # worldLinkFramePosition = joint axis position
            link_com_pos = link_state[0]     # linkWorldPosition = center of mass
            orn = link_state[5]
            
            # Draw coordinate frame at JOINT axis
            draw_frame(joint_frame_pos, orn, length=0.05)
        
        # Print positions to console periodically
        if frame_counter % 30 == 0:
            print("\n=== LINK STATE ANALYSIS ===")
            print(f"{'Link':<20} {'COM Pos [0]':<30} {'Link Frame Pos [4]':<30} {'Difference'}")
            print("-" * 90)
            
            for i in range(p.getNumJoints(ik_solver.robot_id)):
                link_state = p.getLinkState(ik_solver.robot_id, i, computeForwardKinematics=1)
                com_pos = np.array(link_state[0]) * 100  # linkWorldPosition (COM)
                frame_pos = np.array(link_state[4]) * 100  # worldLinkFramePosition
                diff = frame_pos - com_pos
                
                joint_info = p.getJointInfo(ik_solver.robot_id, i)
                link_name = joint_info[12].decode('utf-8')
                
                print(f"{link_name:<20} [{com_pos[0]:>6.2f} {com_pos[1]:>6.2f} {com_pos[2]:>6.2f}]   "
                      f"[{frame_pos[0]:>6.2f} {frame_pos[1]:>6.2f} {frame_pos[2]:>6.2f}]   "
                      f"[{diff[0]:>5.2f} {diff[1]:>5.2f} {diff[2]:>5.2f}]")
            
            print("\n=== DEBUGGING PARENT FRAMES ===")
            
            for i in range(p.getNumJoints(ik_solver.robot_id)):
                joint_info = p.getJointInfo(ik_solver.robot_id, i)
                j_name = joint_info[1].decode('utf-8')
                parent_idx = joint_info[16]
                
                if parent_idx == -1:
                    print(f"Joint {i} ({j_name}): parent = base_link")
                else:
                    parent_joint_info = p.getJointInfo(ik_solver.robot_id, parent_idx)
                    parent_joint_name = parent_joint_info[1].decode('utf-8')
                    # The parent link name is the child link of the parent joint
                    parent_link_name = parent_joint_info[12].decode('utf-8')
                    print(f"Joint {i} ({j_name}): parent joint idx = {parent_idx} ({parent_joint_name}), parent link = {parent_link_name}")
            
            print("\n=== JOINT POSITIONS & URDF VERIFICATION ===")
            print("Testing if: Child Joint Pos = Parent Link Frame + Parent Link Rotation @ URDF offset")
            print(f"{'Joint Name':<20} {'Parent Link':<20} {'Joint Pos (World)':<30} {'Parent Link Frame':<30} {'Match?'}")
            print("-" * 130)
            
            # Store all positions
            base_pos, base_orn = p.getBasePositionAndOrientation(ik_solver.robot_id)
            base_rot_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
            
            for i in range(p.getNumJoints(ik_solver.robot_id)):
                joint_info = p.getJointInfo(ik_solver.robot_id, i)
                j_name = joint_info[1].decode('utf-8')
                parent_idx = joint_info[16]
                urdf_offset_xyz = np.array(joint_info[14])
                urdf_offset_orn = joint_info[15]
                
                # Get this joint's world position
                link_state = p.getLinkState(ik_solver.robot_id, i, computeForwardKinematics=1)
                joint_pos_world = np.array(link_state[4])
                
                # Get parent link frame
                if parent_idx == -1:
                    parent_name = "base_link"
                    parent_frame_pos = base_pos
                    parent_frame_rot = base_rot_mat
                else:
                    parent_joint_info = p.getJointInfo(ik_solver.robot_id, parent_idx)
                    parent_link_name = parent_joint_info[12].decode('utf-8')
                    parent_name = parent_link_name
                    
                    # Get parent link's frame (NOT the parent joint position, but the link's frame)
                    parent_link_state = p.getLinkState(ik_solver.robot_id, parent_idx, computeForwardKinematics=1)
                    parent_frame_pos = np.array(parent_link_state[4])
                    parent_frame_orn = parent_link_state[5]
                    parent_frame_rot = np.array(p.getMatrixFromQuaternion(parent_frame_orn)).reshape(3, 3)
                
                # Calculate: parent_frame + parent_rot @ urdf_offset
                calculated_joint_pos = parent_frame_pos + parent_frame_rot @ urdf_offset_xyz
                
                # Check error
                error = np.linalg.norm(joint_pos_world - calculated_joint_pos) * 100
                match_str = "✓ MATCH" if error < 0.01 else f"✗ err:{error:.2f}cm"
                
                # Convert to cm for display
                joint_pos_cm = joint_pos_world * 100
                parent_frame_cm = parent_frame_pos * 100
                calc_cm = calculated_joint_pos * 100
                urdf_cm = urdf_offset_xyz * 100
                
                print(f"{j_name:<20} {parent_name:<20} [{joint_pos_cm[0]:>6.2f} {joint_pos_cm[1]:>6.2f} {joint_pos_cm[2]:>6.2f}]   "
                      f"[{parent_frame_cm[0]:>6.2f} {parent_frame_cm[1]:>6.2f} {parent_frame_cm[2]:>6.2f}]   {match_str}")
                
                if error >= 0.01:
                    print(f"  └─ URDF offset: [{urdf_cm[0]:>6.2f} {urdf_cm[1]:>6.2f} {urdf_cm[2]:>6.2f}], Calculated: [{calc_cm[0]:>6.2f} {calc_cm[1]:>6.2f} {calc_cm[2]:>6.2f}]")

        frame_counter += 1
        p.stepSimulation()
        time.sleep(1./30.)

except KeyboardInterrupt:
    p.disconnect()