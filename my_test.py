from custom_so100 import CustomSO100
from lerobot.robots.so_follower import SOFollowerRobotConfig
import time
import threading
from pynput import keyboard
from ninja_kinematics import SimpleIK
import numpy as np
from mykinematics import forward_kinematics
from mykinematics import inverse_kinematics as my_inverse_kinematics



DEBUG = False

# Global flag for emergency stop
emergency_stop_requested = False


def build_action(angles):
    """Build action dictiionnary from numpy array."""
    
    action = {
        "shoulder_pan.pos": angles[0],
        "shoulder_lift.pos": angles[1],
        "elbow_flex.pos": angles[2],
        "wrist_flex.pos": angles[3],
    }
    return action



def on_press(key):
    global emergency_stop_requested
    try:
        if key == keyboard.Key.space:
            print("\n[SPACEBAR DETECTED] Emergency stop requested!")
            emergency_stop_requested = True
    except AttributeError:
        pass

# Start keyboard listener in background thread
listener = keyboard.Listener(on_press=on_press)
listener.daemon = True  # Thread will exit when main program exits
listener.start()
print("[INFO] Emergency stop listener started. Press SPACEBAR at any time to stop.")

print("Setting FORCE_CALIBRATION = False")
CustomSO100.FORCE_CALIBRATION = False
config = SOFollowerRobotConfig(port="COM11", use_degrees=True, id="ninja_so101")
robot = CustomSO100(config=config)
# Set emergency stop callback
robot.external_stop_check = lambda: emergency_stop_requested
print("Calling robot.connect()...")
robot.connect()


print("Connection complete")

robot.set_acceleration(20)
robot.set_max_velocity(50)

# Print calibration info
if DEBUG:
    print("\n=== CALIBRATION INFO ===")
    if robot.bus.calibration:
        print(f"Number of calibrated motors: {len(robot.bus.calibration)}")
        for motor_name, calib in robot.bus.calibration.items():
            print(f"\n{motor_name}:")
            print(f"  ID: {calib.id}")
            print(f"  Range: [{calib.range_min}, {calib.range_max}]")
            print(f"  Homing Offset: {calib.homing_offset}")
    else:
        print("NO CALIBRATION FOUND!")

# Read raw and normalized positions from all motors
if DEBUG:
    print("\n=== MOTOR POSITIONS ===")
    obs = robot.get_observation()
    for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
        if f"{motor_name}.pos" in obs:
            print(f"{motor_name}: {obs[f'{motor_name}.pos']:.2f}°")
        else:
            print(f"{motor_name}: NOT FOUND")

try:
    print("\n=== STARTING MOTION TEST ===")
    
   
    robot.bus.enable_torque()
    print("Torque enabled")

    # Move first motor 0 degrees
    action = build_action([0.0, 0.0, 0.0, 0.0])
 
    robot.act_and_observe(action)
    
    if emergency_stop_requested:
        raise KeyboardInterrupt("Emergency stop detected")
    
    action = build_action([0.0, 0.0, 0.0, 45.0])
    input("Press ENTER to continue to IK test...")
    obs = robot.act_and_observe(action)
    print(f"Observed position: shoulder_pan = {obs['shoulder_pan.pos']:.2f}°, shoulder_lift = {obs['shoulder_lift.pos']:.2f}°,elbow_flex = {obs['elbow_flex.pos']:.2f}°, Wrist flex = {obs['wrist_flex.pos']:.2f}°")
    
    
    myxyz= forward_kinematics(np.array([obs["shoulder_lift.pos"], obs["elbow_flex.pos"], obs["wrist_flex.pos"]]))
    print(f"Knife tip position according to mykinematics: X={myxyz[0]:.3f} mm, Y={myxyz[1]:.3f} mm, Z={myxyz[2]:.3f} mm")

  
    # robot.set_max_velocity(20)
    # robot.set_acceleration(10)

    # target_xyz = np.array([0.3, 0.0, 0.02])  # Set target to 20cm depth and 10cm height
   
    # current_joint_angles  = np.array([
    #     obs["shoulder_pan.pos"],
    #     obs["shoulder_lift.pos"],
    #     obs["elbow_flex.pos"],
    #     obs["wrist_flex.pos"],
    # ])
    # print(f"Current joints: {current_joint_angles}")
    # # target_joints = ik_solver.inverse_kinematics(target_xyz, current_joints= current_joint_angles, calibration=robot.bus.calibration )
    # # for joint in target_joints:
    # #     print(f"  Target joint angle: {joint:.2f}°")
    # my_target_joints = np.zeros(4)
    # my_target_joints[0] = 0  # shoulder_pan
    # ik_result = my_inverse_kinematics(target_xyz * 1000)  # Returns [shoulder_lift, elbow_flex, wrist_flex]
    # my_target_joints[1] = ik_result[0]  # shoulder_lift
    # my_target_joints[2] = ik_result[1]  # elbow_flex
    # my_target_joints[3] = ik_result[2]  # wrist_flex

    
    
    

    # print(f"my_target_joints: Shoulder pan = {my_target_joints[0]:.2f}°, Shoulder lift = {my_target_joints[1]:.2f}°, Elbow={my_target_joints[2]:.2f}°, Wrist={my_target_joints[3]:.2f}°")
    # input("Press ENTER to move to my IK target...")
    # action = build_action(my_target_joints)
   
    # obs = robot.act_and_observe(action) 
    # obs = robot.act_and_observe(action) 
    # print(f"Observed position: Shoulder pan = {obs['shoulder_pan.pos']:.2f}°, Shoulder lift = {obs['shoulder_lift.pos']:.2f}°, Elbow={obs['elbow_flex.pos']:.2f}°, Wrist={obs['wrist_flex.pos']:.2f}°")
    
    # observed_angles = np.array([obs["shoulder_pan.pos"], obs["shoulder_lift.pos"], obs["elbow_flex.pos"], obs["wrist_flex.pos"]])


    # xyz_position_mykin = forward_kinematics(np.array([observed_angles[1], observed_angles[2], observed_angles[3]]))
    # print(f"Knife tip position according to mykinematics: X={xyz_position_mykin[0]:.3f} mm, Y={xyz_position_mykin[1]:.3f} mm, Z={xyz_position_mykin[2]:.3f} mm")

        
except KeyboardInterrupt:
    print("\n[INTERRUPT] Emergency stop triggered!")
    robot.emergency_stop("User interrupt")
finally:
    listener.stop()  # Stop keyboard listener
    time.sleep(1)
    robot.disconnect()
    print("Disconnected")