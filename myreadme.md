# Ninjabot Project

This repository contains the robot arm control system for a custom SO101 robotic arm. The wrist has been replaced with a custom part for specialized manipulation tasks.

**Main files:** [my_test.py](my_test.py), [mykinematics.py](mykinematics.py), and [custom_so100.py](custom_so100.py)

---

## 🏗️ High Level Architecture

The robot control functions are stored in this repository. Related project components:
- **ninjabot-mechanics repository**: Custom 3D files and SpaceClaim projects
- **3D-vision repository**: Camera operation and detection algorithms
- **To Do**: Unify all components into a single cohesive system


## 🤖 my_test.py

This is a comprehensive **testing script** for the custom SO100/SO101 robot that demonstrates motor control, emergency stop functionality, and inverse kinematics testing.

### Features

#### 1. Emergency Stop System
- Uses `pynput` keyboard listener to monitor for spacebar press
- Sets global `emergency_stop_requested` flag
- Runs in background daemon thread
- Not finished, need to be made more reliable

#### 2. **Robot Initialization**
- DisablRobot InitializationFORCE_CALIBRATION = False`)
- Connects to robot on COM11 port
- Configures for SO101 model using degrees

#### 3. Motor Configuration
- Reads and displays current motor settings (model number, torque, current limits)
- Sets acceleration and max velocity

#### 4. Motion Testing Sequence

The script executes the following test sequence:

1. **Small movement test**: Moves all motors -5° then +5° to verify basic control
2. **Home position**: Moves to `[0°, 0°, 0°, 45°]` configuration
3. **IK testing**: Series of target positions using `robot.go_to_xz()`:
   - (35 cm, 5 cm)
   - (37 cm, 7 cm)
   - (35 cm, 3 cm)
   - (35 cm, 2 cm)
   - (32 cm, 2 cm)

Each target waits for user confirmation before proceeding.

---

## 🔧 custom_so100.py

This file contains the **CustomSO100 class**, which extends the LeRobot `SOFollower` base class to support the modified SO101 arm with only 4 motors (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex). The gripper and wrist_roll motors have been removed/replaced.

### Key Features

#### Motor Configuration
- Supports 4 Feetech STS3215 servo motors
- Custom calibration offsets for mechanical inaccuracies:
  - `elbow_angle_calib_offset = 8°`
  - `wrist_angle_calib_offset = 5°`
  - `shoulder_lift_calib_offset = 0.5°`
  - `shoulder_pan_calib_offset = -1.3°`

Not really accurate, but these adjustements were made to keep the arm aligned properly when given zero angles as command. This should be refined in the future and probably integrated to the calibration function

#### Safety Systems

**Emergency Stop**
- Monitors torque limits during all movements (default threshold: 1350 units)
- External stop check callback for spacebar integration
- Automatic torque disable on emergency conditions
- `reset_emergency_stop()` method to re-enable after stop

**Real-time Monitoring**
- `get_motor_torques()`: Read current load on all motors
- `get_motors_velocities()`: Monitor motor speeds in RPM
- `check_torque_limits()`: Automatic safety threshold checking
- Velocity-based movement completion detection

#### Motion Control Methods

**`build_action(angles)`** (static)
Converts numpy array `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex]` to action dictionary format.

**`act_and_observe(action)`**
- Sends action with calibration offset corrections
- Monitors torque and velocity during movement
- Waits for motion completion (velocities ≈ 0)
- Returns corrected observation dict
- Includes emergency stop checks

**`go_to_xz(x, z, previous_observation, estimate_xyz)`**
Most important function
High-level cartesian motion control:
- **Input**: Target X,Z coordinates in centimeters (Y assumed 0)
- Calls `inverse_kinematics()` from [mykinematics.py](mykinematics.py)
- Executes movement via `act_and_observe()`
- **Returns**: `(observation, target_joints, estimated_xyz)`
- Optional forward kinematics verification

#### Configuration & Calibration

**`set_acceleration(acceleration)`**
Sets acceleration for all motors (0-254, default 50)

**`set_max_velocity(max_velocity)`**
Sets maximum velocity limit (0-254, default 100)

**`calibrate()`**
- Supports forced recalibration via `FORCE_CALIBRATION` class variable
- Records homing offsets and range of motion for all motors
- Saves calibration to persistent storage
- Loads existing calibration if available and not forcing

#### Debug Features
- Extensive debug logging when `DEBUG = True`
- `print_motor_velocities()`: Display current velocities
- `print_motor_torques()`: Display current torque/load values
---

## 📋 To Do

- [ ] Build an inverse kinematics function where the constraint for horizontal positioning is relaxed
- [ ] Add the joint ranges from URDF files and use them as constraints for IK
- [ ] Refactor [custom_so100.py](custom_so100.py) to make it suitable for import in the broader LeRobot project
- [ ] Improve emergency stop reliability in [my_test.py](my_test.py)
- [ ] Unify robot control, 3D mechanics, and vision components into single system
- [ ] Improve calibration
- [ ] Implement a more accurate approach to target


from lerobot.robots.so_follower import SOFollowerRobotConfig

# Initialize
config = SOFollowerRobotConfig(port="COM11", use_degrees=True)
robot = CustomSO100(config=config)
robot.connect()

# Set motion parameters
robot.set_acceleration(20)
robot.set_max_velocity(50)

# Move to cartesian target (35cm X, 5cm Z)
obs, joints, xyz = robot.go_to_xz(35, 5, obs, estimate_xyz=True)


---

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Optimization algorithms for IK
- `pynput`: Keyboard input monitoring
- `custom_so100`: Custom robot control class
- `lerobot.robots.so_follower`: Robot configuration

---

## 📐 mykinematics.py

This file implements **forward and inverse kinematics** for a 4-DOF robotic arm operating in a 2D plane (Y=0).

### Robot Configuration

The robot consists of the following link segments (in millimeters):
- **shoulder_upperarm**: 79.21 mm
- **upperarm_forearm**: 119 mm  
- **forearm_wrist**: 134.29 mm
- **wrist_knife**: 97.0 mm
- **knife_knifetip**: 80.0 mm

The coordinate system origin is at the **shoulder base** located at `[61, 0, 47]` mm in world coordinates, with the upper arm starting at `[93, 0, 120]` mm.

### Key Functions

#### `forward_kinematics(joint_angles)`
Computes the end-effector (knife tip) position in world coordinates from given joint angles.

**Input:** Joint angles in degrees `[shoulder_lift, elbow_flex, wrist_flex]`  
**Output:** 3D position `[x, y, z]` in millimeters

The function calculates positions for all key points:
- Elbow position
- Wrist position  
- Knife base position
- **Knife tip position** (the end-effector)

#### `inverse_kinematics(xyz_target, current_joints=None, calibration=None)`
Computes the joint angles needed to reach a target end-effector position.

**Input:** 
- `xyz_target`: Target position in mm `[x, y, z]`
- `current_joints`: Optional current joint configuration for better initial guess
- `calibration`: Optional calibration data

**Output:** Joint angles in degrees `[shoulder_lift, elbow_flex, wrist_flex]` or `None` if no solution found

**Constraint:** The knife is kept parallel to the horizontal plane (θ₁ + θ₂ + θ₃ + 45° = 0°)

**Algorithm:** Uses scipy's `minimize` function with L-BFGS-B optimization method and bounded constraints:
- θ₁ (shoulder): 0° to 90°
- θ₂ (elbow): -180° to 0°
- θ₃ (wrist): Calculated from constraint

Returns `None` if the solution error exceeds 1 mm.


#### Helper Functions

- `joint_angles_to_trigo(joint_angles)`: Converts from robot frame to trigonometric world frame
- `trigo_to_joint_angles(trigo_angles)`: Converts from world frame back to robot frame



---

