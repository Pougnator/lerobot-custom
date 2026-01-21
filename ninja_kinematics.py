import pybullet as p
import numpy as np

# Path to SO100 URDF
URDF_PATH = r"C:\Repos_Razor\ninjabot-mechanics\Simulation\SO101\so101_ninja1.urdf"

class SimpleIK:
    def __init__(self, urdf_path=URDF_PATH, end_effector_link=4):
        self.physics_client = p.connect(p.DIRECT)  # No GUI
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        self.end_effector_link = end_effector_link
        
    def forward_kinematics(self, joint_angles):
        """Joint angles (degrees) -> end-effector position (meters)"""
        for i, angle in enumerate(np.deg2rad(joint_angles)):
            p.resetJointState(self.robot_id, i, angle)
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link)
        return np.array(ee_state[0])  # XYZ position in meters
    
    def inverse_kinematics(self, target_pos, current_joints=None, calibration=None):
        """Target XYZ (meters) -> joint angles (degrees)
        
        Args:
            target_pos: Target XYZ position in meters
            current_joints: Current joint angles in degrees (optional)
            calibration: Robot calibration dict with MotorCalibration objects (REQUIRED)
        """
        if calibration is None:
            raise ValueError("Calibration is required!")
        
        # Set current joint positions as initial guess
        if current_joints is not None:
            current_joints_rad = np.deg2rad(current_joints)
            for i, angle in enumerate(current_joints_rad[:4]):  # Only 4 motors
                p.resetJointState(self.robot_id, i, angle)
         # Convert motor units to degrees: (motor_pos / 4095) * 360 - 180
        def motor_to_deg(motor_val):
            return (motor_val / 4095.0) * 360.0 - 180.0
        # Get motor names for looking up calibration
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
        
        # Get joint limits from calibration
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        rest_poses = []
        
        for i in range(4):  # 4 motors
            motor_name = motor_names[i]
            
            if motor_name not in calibration:
                raise ValueError(f"Motor '{motor_name}' not found in calibration!")
            
            motor_calib = calibration[motor_name]
            
            # Convert motor range (motor units) to degrees
           
            ll_deg = motor_to_deg(motor_calib.range_min)  
            ul_deg = motor_to_deg(motor_calib.range_max)  

            print(f"[DEBUG] {motor_name}: range_min={motor_calib.range_min}, range_max={motor_calib.range_max}")
            print(f"        Converted to degrees: ll={ll_deg:.2f}°, ul={ul_deg:.2f}°")
            ll = np.deg2rad(ll_deg)
            ul = np.deg2rad(ul_deg)
            
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)
            rest_poses.append(current_joints_rad[i] if current_joints is not None else 0)

            if motor_name == "elbow_flex":
                # Adjust elbow flex rest pose to avoid singularity
                ll = np.deg2rad(ll_deg) 
                ul = 0 
                joint_ranges[-1] = ul - ll
                rest_poses[-1] = np.deg2rad(-10.0)  # Slightly bent elbow
            if motor_name == "wrist_flex":
                # Adjust wrist flex limits
                ll = np.deg2rad(-40.0)
                ul = np.deg2rad(40.0)
                joint_ranges[-1] = ul - ll
                rest_poses[-1] = np.deg2rad(0.0)  # Slightly bent elbow

        
        # Calculate IK
        joint_angles = p.calculateInverseKinematics(
            self.robot_id, 
            self.end_effector_link,
            target_pos,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=500,
            residualThreshold=1e-8
        )
        return np.rad2deg(joint_angles[:4])  # Return only 4 motors in degrees