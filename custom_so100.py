
import json
import os
import time
from pathlib import Path
from typing import Callable
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.cameras.utils import make_cameras_from_configs
from mykinematics import forward_kinematics, inverse_kinematics
import numpy as np

DEBUG = False
MAX_TORQUE_THRESHOLD = 1350.0  # Torque/load threshold to trigger emergency stop

class CustomSOLeader(SOLeader):
    """Custom SO leader with only 4 motors (missing wrist_roll and gripper)."""

    def __init__(self, config: SOLeaderTeleopConfig):
        # Skip SOLeader.__init__ bus setup — call Teleoperator.__init__ directly
        super(SOLeader, self).__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        from lerobot.motors.feetech import FeetechMotorsBus
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan":  Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex":    Motor(3, "sts3215", norm_mode_body),
                "wrist_flex":    Motor(4, "sts3215", norm_mode_body),
                # wrist_roll (5) and gripper (6) not present on custom build
            },
            calibration=self.calibration,
        )

    def get_motor_torques(self) -> dict:
        """Return signed load for all leader motors (same encoding as follower).

        Magnitude is bits 0–9 (0–1023); bit 10 is the direction sign.
        """
        torques = {}
        for motor_name in self.bus.motors.keys():
            raw = self.bus.read("Present_Load", motor_name)
            magnitude = raw & 0x3FF
            torques[motor_name] = -magnitude if (raw & 0x400) else magnitude
        return torques


# ── Joint angle calibration offsets ──────────────────────────────────────────
# Definition:  offset[j] = raw_encoder_reading[j] − true_physical_angle[j]
#   send path     raw  = true + offset   (_to_raw)
#   observe path  true = raw  − offset   (_to_true)
#
# Fitted by vibe_cam_project/calibrate_angles.py, which parks the arm at the home
# pose (pan 0°, shoulder_lift 0°, elbow_flex 0°, wrist_flex 45°) releasing one
# joint at a time.
#
# Replaced four module-level constants on 2026-07-30. They were applied by eight
# duplicated if-blocks across four methods, so any new offset — or this loader —
# would have had to be wired into all eight without missing one.
_ANGLE_CALIB_REL = "vibe_cam_project/sensors/angle_calibration.json"


def _find_beside(rel):
    """Locate `rel` inside a repo checked out beside this one, else None.

    lerobot-custom and 3D-vision sit side by side and the pair has already moved
    once (C:\\Repos_Razor -> C:\\Repos), so no machine root is hardcoded here:
    walk up from this file and look one level down into each sibling directory.
    """
    for parent in Path(__file__).resolve().parents:
        hits = sorted(parent.glob("*/" + rel))
        if hits:
            return str(hits[0])
    return None


ANGLE_CALIB_PATH = os.environ.get("NINJA_ANGLE_CALIB") or _find_beside(_ANGLE_CALIB_REL)

# The four joints a calibration file must cover. These were ALSO the hardcoded
# fallback offsets until 2026-10-08 (-4.0 / 0.5 / 8.0 / 5.0); they are no longer
# applied automatically. The real calibration had drifted 5.9° away on wrist_flex
# and the fallback hid that behind one easily-missed print line.
_KNOWN_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

_BOOTSTRAP_OFFSETS = {
    "shoulder_pan":  -4.0,
    "shoulder_lift":  0.5,
    "elbow_flex":     8.0,
    "wrist_flex":     5.0,
}

CALIB_OFFSETS = {}


def load_angle_calibration(path=ANGLE_CALIB_PATH, allow_defaults=False) -> dict:
    """Load joint offsets from JSON into CALIB_OFFSETS. Returns the active offsets.

    Mutates CALIB_OFFSETS in place rather than rebinding it, so callers that did
    `from custom_so100 import CALIB_OFFSETS` see reloads.

    RAISES on a missing, unreadable or incomplete file. There is deliberately no
    automatic fallback: offset[j] feeds both the send path (_to_raw) and the
    observe path (_to_true), so a wrong offset silently moves the whole FK chain,
    the knife tilt and every force estimate derived from it — indistinguishable
    from a real measurement. Same reasoning as
    ArduinoSensors.load_normal_force_calibration.

    `allow_defaults=True` seeds _BOOTSTRAP_OFFSETS instead of raising, and says so
    loudly. It exists for calibrate_angles.py, which has to start somewhere when
    it is generating the file for the first time; nothing on a motion path should
    pass it. The NINJA_ALLOW_DEFAULT_CALIB env var does the same for the
    import-time load below.
    """
    def _fail(msg):
        if not allow_defaults:
            raise RuntimeError(msg)
        CALIB_OFFSETS.clear()
        CALIB_OFFSETS.update(_BOOTSTRAP_OFFSETS)
        print("[CALIB][BOOTSTRAP] " + msg)
        print("[CALIB][BOOTSTRAP] Seeding built-in offsets:  "
              + "  ".join(f"{j}={CALIB_OFFSETS[j]:+.2f}°" for j in _KNOWN_JOINTS))
        return CALIB_OFFSETS

    if not path:
        return _fail(
            f"Angle calibration not found: no {_ANGLE_CALIB_REL} in any repo beside "
            f"{Path(__file__).resolve().parent}. Check out 3D-vision next to "
            "lerobot-custom, or set NINJA_ANGLE_CALIB to the file."
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return _fail(f"Angle calibration missing: {path}. "
                     "Run vibe_cam_project/tools/calibrate_angles.py, or point "
                     "NINJA_ANGLE_CALIB at the file.")
    except (OSError, ValueError) as e:
        return _fail(f"Angle calibration unreadable: {path} ({e}).")

    loaded  = data.get("offsets", data)
    unknown = sorted(set(loaded) - set(_KNOWN_JOINTS))
    if unknown:
        print(f"[CALIB][WARN] Ignoring unknown joints in {path}: {unknown}")
    absent = sorted(set(_KNOWN_JOINTS) - set(loaded))
    if absent:
        # Previously these kept their built-in default — the silent-partial case.
        return _fail(f"Angle calibration {path} has no entry for {absent}; "
                     "re-run calibrate_angles.py covering all four joints.")

    CALIB_OFFSETS.clear()
    for j in _KNOWN_JOINTS:
        CALIB_OFFSETS[j] = float(loaded[j])
    print(f"[CALIB] Angle offsets from {path}:  "
          + "  ".join(f"{j}={CALIB_OFFSETS[j]:+.2f}°" for j in _KNOWN_JOINTS))
    return CALIB_OFFSETS


load_angle_calibration(
    allow_defaults=bool(os.environ.get("NINJA_ALLOW_DEFAULT_CALIB")))


def _to_raw(action: dict) -> dict:
    """True angles → raw motor commands. Returns a NEW dict; the caller's is untouched.

    Copying matters: act_and_observe() used to mutate the caller's action dict, so
    re-sending the same dict applied the offset twice.
    """
    out = dict(action)
    for j, off in CALIB_OFFSETS.items():
        key = f"{j}.pos"
        if key in out:
            out[key] += off
    return out


def _to_true(obs: dict) -> dict:
    """Raw motor readings → true angles. Mutates and returns obs (it is ours already)."""
    for j, off in CALIB_OFFSETS.items():
        key = f"{j}.pos"
        if key in obs:
            obs[key] -= off
    return obs


class CustomSO100(SOFollower):
    """Custom SO100 with only 4 motors (missing wrist_roll and gripper)."""
    
    # Set this to True to force recalibration
    FORCE_CALIBRATION = False

    @staticmethod
    def build_action(angles):
        """Build action dictiionnary from numpy array.
        Args:
            angles: Numpy array of (true) joint angles in degrees [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex]
        Returns:
            action: Dictionary mapping motor names to target positions in degrees.
             
             action = {
            "shoulder_pan.pos": angles[0],
            "shoulder_lift.pos": angles[1],
            "elbow_flex.pos": angles[2],
            "wrist_flex.pos": angles[3],
        }
        """
        
        action = {
            "shoulder_pan.pos": angles[0],
            "shoulder_lift.pos": angles[1],
            "elbow_flex.pos": angles[2],
            "wrist_flex.pos": angles[3],
        }
        return action
    
    
    def __init__(self, config: SOFollowerRobotConfig):
        # Call parent __init__ but skip motor bus setup
        super(SOFollower, self).__init__(config)  # Call Robot.__init__ directly
        self.config = config
        
         # Initialize emergency stop
        # Annotated Callable so assigning a real callback type-checks — without it
        # Pylance infers `() -> Literal[False]` from the default lambda and rejects
        # every genuine e-stop hookup (e.g. MotionController.setup's lambda).
        self.external_stop_check: Callable[[], bool] = lambda: False  # Default: no stop
        self._emergency_stop_triggered = False

        # Clear calibration if forcing recalibration
        if self.FORCE_CALIBRATION:
            self.calibration = {}
            print("[DEBUG] Cleared calibration due to FORCE_CALIBRATION=True")
        
        # Set up normalization mode
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # Define only the 4 motors you have
        from lerobot.motors.feetech import FeetechMotorsBus
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                # "wrist_roll" and "gripper" removed
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
    

    def set_acceleration(self, acceleration: int = 50) -> None:
        """Set acceleration for all motors.
        
        Args:
            acceleration: Acceleration value (0-254)
                - Range: 0-254
                - LeRobot default: 254 (maximum)
                - Current default: 50 (conservative)
        """
        
        for motor in self.bus.motors.keys():
            self.bus.write("Acceleration", motor, acceleration)
        print(f"[DEBUG] Set acceleration to {acceleration} for all motors")

    def set_max_velocity(self, max_velocity: int = 100) -> None:
        """Set maximum velocity for all motors.

        Args:
            max_velocity: Maximum velocity value
                - Range: 0-254
                - Current default: 100
                - Factory default: 254 (maximum)
        """
        for motor in self.bus.motors.keys():
            self.bus.write("Maximum_Velocity_Limit", motor, max_velocity)
        print(f"[DEBUG] Set maximum velocity to {max_velocity} for all motors")

    def velocities_are_zero(self, velocities) -> bool:
        """Checks if all motor velocities are effectively zero.

        Args:
            velocities: Dictionary mapping motor names to their current velocities.

        Returns:
            True if all velocities are effectively zero.
        """
        for motor_name,   velocity in velocities.items():
            if abs(velocity) > 1e-2:  # Consider a small threshold for zero
                return False
        return True
        
    def get_motors_velocities(self) -> dict[str, float]:
        """Check current motor velocities.
        
        
        Returns:
            Dictionary mapping motor names to their current load velocities in revolutions per minute (RRM).
        """
        velocities = {}
        for motor_name in self.bus.motors.keys():
            velocity = abs(self.bus.read("Present_Velocity", motor_name))
            velocities[motor_name] = velocity * 0.0732  # Convert to RRM (Rev/min)
        return velocities
        
    def get_motor_torques(self) -> dict[str, float]:
        """Get current torque/load for all motors.
        
        Returns:
            Dictionary mapping motor names to their current load values.
            Load range: -2047 to +2047 (negative = CCW, positive = CW)
        """
        torques = {}
        for motor_name in self.bus.motors.keys():
            raw = self.bus.read("Present_Load", motor_name)
            # Present_Load uses sign-magnitude encoding: bit 10 is the direction
            # bit, bits 0-9 are the magnitude. Positive = CW, negative = CCW.
            magnitude = raw & 0x3FF
            load = -magnitude if (raw & 0x400) else magnitude
            torques[motor_name] = load
        return torques

    def set_wrist_torque_limit(self, limit: int) -> None:
        """Set Torque_Limit on wrist_flex only. Range 0–1000 (1000 = full torque)."""
        self.bus.write("Torque_Limit", "wrist_flex", limit)

    def reset_emergency_stop(self):
        """Reset emergency stop flag and re-enable torque."""
        print("[RESET] Clearing emergency stop and re-enabling torque...")
        self._emergency_stop_triggered = False
        self.bus.enable_torque()
        print("[RESET] Emergency stop cleared, torque re-enabled")

    def send_action_calibrated(self, action: dict) -> None:
        """Apply calibration offsets and fire the action without waiting for completion.

        Unlike act_and_observe, this method does NOT wait for motors to settle or
        add any sleep delay.  Used by high-frequency trajectory loops (e.g. cutting
        motion) where the caller manages timing and torque checks directly.
        """
        self.send_action(_to_raw(action))   # _to_raw copies; caller's dict is safe

    def get_observation_calibrated(self) -> dict:
        """Get observation with calibration offsets removed (IK-convention angles).

        Counterpart of send_action_calibrated — returns angles in the same
        convention as go_to_xz / act_and_observe.
        """
        return _to_true(self.get_observation())

    def emergency_stop(self, reason: str = "Torque limit exceeded"):
        """Execute emergency stop - disable all motor torque."""
        print(f"\n{'='*60}")
        print(f"[EMERGENCY STOP] {reason}")
        print(f"{'='*60}")
        
        self._emergency_stop_triggered = True
        self.bus.disable_torque()
        print("[EMERGENCY STOP] All motors disabled")
        
        # # Print final torque readings
        # self.print_motor_torques()

    def check_torque_limits(self, torques: dict[str, float], threshold: float = 1900.0) -> None:
        """Check if any motor torque exceeds the specified threshold.

        Args:
            torques: Dictionary mapping motor names to their current load values.
            threshold: Torque/load threshold to trigger emergency stop."""
        for motor_name, load in torques.items():
            if abs(load) >= threshold:
                print(f"[WARNING] Torque limit exceeded on {motor_name}: {load} (threshold: {threshold})")
                self.emergency_stop(f"Torque limit exceeded on {motor_name}")
              
    def print_motor_velocities(self, velocities) -> None:
        """Print current motor velocities."""
        
        print("\n=== MOTOR VELOCITIES ===")
        for motor_name, velocity in velocities.items():
            direction = "CW" if velocity > 0 else "CCW" if velocity < 0 else "NONE"
            print(f"{motor_name}: {velocity:6.1f} ({direction})")  

    def print_motor_torques(self, torques) -> None:
        """Print current torque/load for all motors."""
        
        print("\n=== MOTOR TORQUES ===")
        for motor_name, load in torques.items():
            direction = "CW" if load > 0 else "CCW" if load < 0 else "NONE"
            print(f"{motor_name}: {load:6.1f} ({direction})")

 
    def go_to_xz(self, x, z, previous_observation=None, estimate_xyz = False):
        ''' Move robot to specified X,Z position using IK. Y is assumed to be 0 (2D plane).
        x,y are in cm


        previous_observation: Previous observation dict to get current joint angles.
        Returns:
            obs: New observation after movement
            target_joints: Target joint angles used for movement
            estimated_xyz: Forward kinematics estimate of knife tipafter movement based on observed joint angles
        '''
        target_xyz = np.array([x, 0.0, z])  # Y=0 for 2D plane
        target_xyz = target_xyz * 10 # convert to mm
        current_joints = previous_observation['shoulder_pan.pos'], previous_observation['shoulder_lift.pos'], previous_observation['elbow_flex.pos'], previous_observation['wrist_flex.pos']
        ik_result = inverse_kinematics(target_xyz, current_joints=current_joints)
        if DEBUG:
            print(f"[DEBUG] IK result for target X={x:.3f} m, Z={z:.3f} m: {ik_result}")
            print(f"[DEBUG] Current joints: {current_joints}")
        # Build 4-DOF action (add shoulder_pan = 0)
        if ik_result is None:
            print("[ERROR] IK solution not found for the target position.")
            return None, None, None
        target_joints = np.array([0.0, ik_result[0], ik_result[1], ik_result[2]])
        action = self.build_action(target_joints)
        obs = self.act_and_observe(action)
        estimated_xyz = None
        if estimate_xyz:
            estimated_xyz = forward_kinematics(np.array([obs["shoulder_lift.pos"], obs["elbow_flex.pos"], obs["wrist_flex.pos"]]))
        return obs, target_joints, estimated_xyz

    def act_and_observe(self, action: dict) -> dict:
        """Send action and get observation in one step."""

        action = _to_raw(action)   # true angles → raw motor commands
        # Check if emergency stop was triggered
        if self.external_stop_check and self.external_stop_check() == True:
            print("[WARNING] Emergency stop active - action ignored")
            return self.get_observation()
        
        if DEBUG == True:
            print(f"[DEBUG] sending action: {action} to robot")
        self.send_action(action)
        # Monitor torque during movement
        start_time = time.time()
        # while time.time() - start_time < 0.1 or not self.velocities_are_zero(velocities):  # Monitor for 3 seconds
        while time.time() - start_time < 0.1: 
            torques = self.get_motor_torques()
            velocities = self.get_motors_velocities()
            self.check_torque_limits(torques, threshold=MAX_TORQUE_THRESHOLD)
            if self.external_stop_check and self.external_stop_check() == True:
                self.emergency_stop("Spacebar pressed")
                break
            if DEBUG:
                self.print_motor_velocities(velocities)
                self.print_motor_torques(torques)
            time.sleep(0.1)  # Sample every 100ms
         
        #This was for debug purposes
        # time.sleep(3)
        obs = self.get_observation()
        if DEBUG == True:
            print(f"[DEBUG][CustomSO100] received observation:")
            for key, value in obs.items():
                print(f"-{key} ---> {value:.2f} degrees")
            print("-"*50)

        return _to_true(obs)   # raw motor readings → true angles
    
    def act_and_observe_precise(self, action: dict) -> dict:
        """Send action and get observation in one step."""

        action = _to_raw(action)   # true angles → raw motor commands
        # Check if emergency stop was triggered
        if self.external_stop_check and self.external_stop_check() == True:
            print("[WARNING] Emergency stop active - action ignored")
            return self.get_observation()
        
        if DEBUG == True:
            print(f"[DEBUG] sending action: {action} to robot")
        self.send_action(action)
        # Wait until the servos report stopped, but never longer than SETTLE_TIMEOUT_S:
        # total wait = min(settle time, timeout). If the timeout fires the returned
        # obs is a snapshot of an arm that is STILL MOVING.
        SETTLE_TIMEOUT_S = 3.0
        start_time = time.time()
        settled = False
        while time.time() - start_time < SETTLE_TIMEOUT_S:
            torques = self.get_motor_torques()
            velocities = self.get_motors_velocities()
            self.check_torque_limits(torques, threshold=MAX_TORQUE_THRESHOLD)
            if self.external_stop_check and self.external_stop_check() == True:
                self.emergency_stop("Spacebar pressed")
                break
            if DEBUG:
                self.print_motor_velocities(velocities)
                self.print_motor_torques(torques)
            # Only trust "stopped" after the servo has had time to start moving.
            if time.time() - start_time >= 0.1 and self.velocities_are_zero(velocities):
                settled = True
                break
            time.sleep(0.1)  # Sample every 100ms

        waited_s = time.time() - start_time
        print(f"[PRECISE] waited {waited_s:.2f}s  "
              f"({'settled' if settled else 'TIMEOUT — arm may still be moving'})")
        obs = self.get_observation()
        if DEBUG == True:
            print(f"[DEBUG][CustomSO100] received observation:")
            for key, value in obs.items():
                print(f"-{key} ---> {value:.2f} degrees")
            print("-"*50)

        return _to_true(obs)   # raw motor readings → true angles
        
    # def configure(self) -> None:
    #     """Custom configuration for the 4-motor SO100."""
    #     print(f"[DEBUG] Configuring CustomSO100 with 4 motors")
    #     self.bus.disable_torque()
    #     for motor in self.bus.motors.keys():
    #         #Acceleration: default 50, range 0-254
    #         #Maximum_Velocity_Limit: default unknown, range 0-255


    #         # print("Do nothing special for motor configuration")
    #         #There are different operating modes, and step might be interesting
    #         # self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)  # Position mode
    #         self.bus.write("Acceleration", motor, 50)  # Set acceleration
    #         self.bus.write("Maximum_Velocity_Limit", motor, 100)  # Limit max speed
    #         # self.bus.write("Goal_Velocity", motor, 75)  # Set target velocity
    #     print(f"[DEBUG] Configuration complete")
    def calibrate(self) -> None:
        """Override calibration to support forced recalibration."""
        print(f"[DEBUG] calibrate() called. FORCE_CALIBRATION={self.FORCE_CALIBRATION}")
        
        # Check if we should skip calibration
        if not self.FORCE_CALIBRATION:
            print("[DEBUG] Not forcing calibration, checking for existing...")
            # Load existing calibration only for the 4 motors we have
            try:
                existing_calib = self._load_calibration()
                if existing_calib:
                    # Filter calibration to only include our 4 motors
                    filtered_calib = {motor: calib for motor, calib in existing_calib.items() 
                                    if motor in self.bus.motors}
                    if filtered_calib:
                        print(f"[DEBUG] Using existing calibration for motors: {list(filtered_calib.keys())}")
                        self.bus.write_calibration(filtered_calib)
                        return
            except FileNotFoundError:
                print("[DEBUG] No existing calibration file found, proceeding to calibrate.")
        
        # Force new calibration
        print("[DEBUG] Running new calibration...")
        from lerobot.motors.feetech import OperatingMode
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()
        
        # Record all motors including wrist_flex
        print(
            f"Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(None)  # Record all motors
        
        from lerobot.motors import MotorCalibration
        self.calibration = {}
        for motor_name, motor_obj in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor_obj.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )
        
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved")

