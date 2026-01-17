from custom_so100 import CustomSO100
from lerobot.robots.so_follower import SOFollowerRobotConfig
import time
import threading
from pynput import keyboard
DEBUG = True

# Global flag for emergency stop
emergency_stop_requested = False

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

    # Move first motor 10 degrees
    action = {
        "shoulder_pan.pos": -5.0,
        "shoulder_lift.pos": -5.0,
        "elbow_flex.pos": -5.0,
        "wrist_flex.pos": -5.0,
    }
 
    robot.act_and_observe(action)
    
    if emergency_stop_requested:
        raise KeyboardInterrupt("Emergency stop detected")
    
    action = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 45.0,
    }
    robot.act_and_observe(action)

    action = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 10.0,
        "elbow_flex.pos": 10.0,
        "wrist_flex.pos": 0.0,
    }
    robot.act_and_observe(action)

    action = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 40.0,
        "elbow_flex.pos": 30.0,
        "wrist_flex.pos": -10.0,
    }
    robot.act_and_observe(action)
    robot.set_max_velocity(20)
    robot.set_acceleration(10)

    # action = {
    #     "shoulder_pan.pos": 0.0,
    #     "shoulder_lift.pos": 40.0,
    #     "elbow_flex.pos": 30.0,
    #     "wrist_flex.pos": 45.0,
    # }
    # robot.act_and_observe(action) 
        
except KeyboardInterrupt:
    print("\n[INTERRUPT] Emergency stop triggered!")
    robot.emergency_stop("User interrupt")
finally:
    listener.stop()  # Stop keyboard listener
    time.sleep(1)
    robot.disconnect()
    print("Disconnected")