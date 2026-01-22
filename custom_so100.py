
import time
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.cameras.utils import make_cameras_from_configs
from mykinematics import forward_kinematics, inverse_kinematics
import numpy as np

DEBUG = False 
MAX_TORQUE_THRESHOLD = 1350.0  # Torque/load threshold to trigger emergency stop

elbow_angle_calib_offset = 8
wrist_angle_calib_offset = 5.0
shoulder_lift_calib_offset = 0.5
shoulder_pan_calib_offset = -1.3

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
        self.external_stop_check = lambda: False  # Default: no stop
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
            load = self.bus.read("Present_Load", motor_name)
            torques[motor_name] = load
        return torques
    
    def reset_emergency_stop(self):
        """Reset emergency stop flag and re-enable torque."""
        print("[RESET] Clearing emergency stop and re-enabling torque...")
        self._emergency_stop_triggered = False
        self.bus.enable_torque()
        print("[RESET] Emergency stop cleared, torque re-enabled")

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

        # Correct the angles because two of them are off
        if "elbow_flex.pos" in action:
            action["elbow_flex.pos"] += elbow_angle_calib_offset
        if "wrist_flex.pos" in action:
            action["wrist_flex.pos"] += wrist_angle_calib_offset
        if "shoulder_lift.pos" in action:
            action["shoulder_lift.pos"] += shoulder_lift_calib_offset  
        if "shoulder_pan.pos" in action:
            action["shoulder_pan.pos"] += shoulder_pan_calib_offset
        # Check if emergency stop was triggered
        if self.external_stop_check and self.external_stop_check() == True:
            print("[WARNING] Emergency stop active - action ignored")
            return self.get_observation()
        
        if DEBUG == True:
            print(f"[DEBUG] sending action: {action} to robot")
        self.send_action(action)
        # Monitor torque during movement
        start_time = time.time()
        while time.time() - start_time < 0.1 or not self.velocities_are_zero(velocities):  # Monitor for 3 seconds
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
         
        
        time.sleep(3)
        obs = self.get_observation()
        if DEBUG == True:
            print(f"[DEBUG][CustomSO100] received observation:")
            for key, value in obs.items():
                print(f"-{key} ---> {value:.2f} degrees")
            print("-"*50)

        #correct the observed angles because elbow and wrist are off by 4 degrees
        if "elbow_flex.pos" in obs:
            obs["elbow_flex.pos"] -= elbow_angle_calib_offset
        if "wrist_flex.pos" in obs:
            obs["wrist_flex.pos"] -= wrist_angle_calib_offset
        if "shoulder_lift.pos" in obs:
            obs["shoulder_lift.pos"] -= shoulder_lift_calib_offset
        if "shoulder_pan.pos" in obs:
            obs["shoulder_pan.pos"] -= shoulder_pan_calib_offset
        return obs
    
    def act_and_observe_precise(self, action: dict) -> dict:
        """Send action and get observation in one step."""

        # Correct the angles because two of them are off
        if "elbow_flex.pos" in action:
            action["elbow_flex.pos"] += elbow_angle_calib_offset
        if "wrist_flex.pos" in action:
            action["wrist_flex.pos"] += wrist_angle_calib_offset
        if "shoulder_lift.pos" in action:
            action["shoulder_lift.pos"] += shoulder_lift_calib_offset  
        if "shoulder_pan.pos" in action:
            action["shoulder_pan.pos"] += shoulder_pan_calib_offset
        # Check if emergency stop was triggered
        if self.external_stop_check and self.external_stop_check() == True:
            print("[WARNING] Emergency stop active - action ignored")
            return self.get_observation()
        
        if DEBUG == True:
            print(f"[DEBUG] sending action: {action} to robot")
        self.send_action(action)
        # Monitor torque during movement
        start_time = time.time()
        while time.time() - start_time < 0.1 or not self.velocities_are_zero(velocities):  # Monitor for 3 seconds
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
         
        
        time.sleep(3)
        obs = self.get_observation()
        if DEBUG == True:
            print(f"[DEBUG][CustomSO100] received observation:")
            for key, value in obs.items():
                print(f"-{key} ---> {value:.2f} degrees")
            print("-"*50)

        #correct the observed angles because elbow and wrist are off by 4 degrees
        if "elbow_flex.pos" in obs:
            obs["elbow_flex.pos"] -= elbow_angle_calib_offset
        if "wrist_flex.pos" in obs:
            obs["wrist_flex.pos"] -= wrist_angle_calib_offset
        if "shoulder_lift.pos" in obs:
            obs["shoulder_lift.pos"] -= shoulder_lift_calib_offset
        if "shoulder_pan.pos" in obs:
            obs["shoulder_pan.pos"] -= shoulder_pan_calib_offset
        return obs
        
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

