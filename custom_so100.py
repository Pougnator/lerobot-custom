from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.cameras.utils import make_cameras_from_configs


class CustomSO100(SOFollower):
    """Custom SO100 with only 4 motors (missing wrist_roll and gripper)."""
    
    # Set this to True to force recalibration
    FORCE_CALIBRATION = False
    
    def __init__(self, config: SOFollowerRobotConfig):
        # Call parent __init__ but skip motor bus setup
        super(SOFollower, self).__init__(config)  # Call Robot.__init__ directly
        self.config = config
        
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
    

    def configure(self) -> None:
        """Custom configuration for the 4-motor SO100."""
        print(f"[DEBUG] Configuring CustomSO100 with 4 motors")
        self.bus.disable_torque()
        for motor in self.bus.motors.values():
            print("Do nothing special for motor configuration")
            #There are different operating modes, and step might be interesting
            # self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)  # Position mode
            # self.bus.write("Acceleration", motor, 50)  # Set acceleration
            # self.bus.write("Maximum_Velocity_Limit", motor, 100)  # Limit max speed
            # self.bus.write("Goal_Velocity", motor, 75)  # Set target velocity
        print(f"[DEBUG] Configuration complete")
    def calibrate(self) -> None:
        """Override calibration to support forced recalibration."""
        print(f"[DEBUG] calibrate() called. FORCE_CALIBRATION={self.FORCE_CALIBRATION}")
        
        # Check if we should skip calibration
        if not self.FORCE_CALIBRATION:
            print("[DEBUG] Not forcing calibration, checking for existing...")
            # Load existing calibration only for the 4 motors we have
            existing_calib = self._load_calibration()
            if existing_calib:
                # Filter calibration to only include our 4 motors
                filtered_calib = {motor: calib for motor, calib in existing_calib.items() 
                                if motor in self.bus.motors}
                if filtered_calib:
                    print(f"[DEBUG] Using existing calibration for motors: {list(filtered_calib.keys())}")
                    self.bus.write_calibration(filtered_calib)
                    return
        
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
        for motor, msts3215 in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved")

