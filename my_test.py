
from custom_so100 import CustomSO100
from lerobot.robots.so_follower import SOFollowerRobotConfig
import time

print("Setting FORCE_CALIBRATION = True")
CustomSO100.FORCE_CALIBRATION = False
config = SOFollowerRobotConfig(port="COM9", use_degrees=True)
robot = CustomSO100(config=config)

print("Calling robot.connect()...")
robot.connect()  # This calls calibrate() automatically if needed
print("Connection complete")

# Print calibration info
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
print("\n=== MOTOR POSITIONS ===")
obs = robot.get_observation()
for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
    if f"{motor_name}.pos" in obs:
        print(f"{motor_name}: {obs[f'{motor_name}.pos']:.2f}°")
    else:
        print(f"{motor_name}: NOT FOUND")

print("\n=== STARTING MOTION TEST ===")
# Enable torque
robot.bus.enable_torque()
print("Torque enabled")

# Move first motor 10 degrees
target_pos = 0.0
action = {"elbow_flex.pos": target_pos}
print(f"Sending action: {action}")
result = robot.send_action(action)
print(f"Action sent: {result}")
# Wait for motor to move
time.sleep(1)
action = {"shoulder_pan.pos": 0}
print(f"Sending action: {action}")
result = robot.send_action(action)
print(f"Action sent: {result}")

# Wait for motor to move
time.sleep(1)
# Check new position
obs = robot.get_observation()
new_pos = obs["shoulder_pan.pos"]
action = {
    "shoulder_pan.pos": 5.0,
    "shoulder_lift.pos": 5.0,
    "elbow_flex.pos": 5.0,
    "wrist_flex.pos": 5.0,
}
robot.send_action(action)
time.sleep(1)
# Check new position
obs = robot.get_observation()
print(f"Motor positions: \n" + "\n".join([f"{item}: {obs[item]:.2f}°" for item in action.keys()]))


action = {"wrist_flex.pos": 45}
print(f"Sending action: {action}")

result = robot.send_action(action)
time.sleep(1)
print(f"Action sent: {result}")
obs = robot.get_observation()
print(f"Motor positions: \n" + "\n".join([f"{item}: {obs[item]:.2f}°" for item in action.keys()]))

robot.disconnect()
