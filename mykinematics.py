import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize

DEBUG = True
robot_links = {
    "shoulder_upperarm": 79.21,
    "upperarm_forearm": 119,
    "forearm_wrist": 134.29,
    "wrist_knife": 97.0,
    "knife_knifetip": 80.0,
}

elbow_wrist_angle = 1.77 # degrees
upper_arm_elbow_angle = 13.81  # degrees
shoulder_coordinates = np.array([61, 0.0, 47.0])  # Shoulder base at origin
upper_arm_coordinates = shoulder_coordinates + np.array([32, 0.0, 73.0])

def forward_kinematics(joint_angles):
    """Compute end-effector position from joint angles (degrees) in world coordinates. The origin of the world coordinate system is at the shoulder base."""
    trigo = joint_angles_to_trigo(joint_angles)
    theta1, theta2, theta3= np.deg2rad(trigo)
    
    # Position calculations
    # We assume that the robot stays in 2d plane with y=0
    # x1, y1, z1: position of elbow
    # x2, y2, z2: position of wrist
    # x3, y3, z3: position of knife base
    # x4, y4, z4: position of knife tip

    x1 = robot_links["upperarm_forearm"] * np.cos(theta1-np.deg2rad(upper_arm_elbow_angle)) + upper_arm_coordinates[0]
    y1 = 0
    z1 = robot_links["upperarm_forearm"] * np.sin(theta1-np.deg2rad(upper_arm_elbow_angle)) + upper_arm_coordinates[2]
    
    x2 = robot_links["forearm_wrist"] * np.cos(theta1 + theta2+ np.deg2rad(elbow_wrist_angle)) + x1
    y2 = 0
    z2 = robot_links["forearm_wrist"] * np.sin(theta1 + theta2 + np.deg2rad(elbow_wrist_angle)) + z1
    
    x3 = robot_links["wrist_knife"] * np.cos(theta1 + theta2 + theta3) + x2
    y3 = 0
    z3 = robot_links["wrist_knife"] * np.sin(theta1 + theta2 + theta3) + z2

    x4 = robot_links["knife_knifetip"] * np.cos(theta1 + theta2 + theta3 + np.deg2rad(45)) + x3
    y4 = 0
    z4 = robot_links["knife_knifetip"] * np.sin(theta1 + theta2 + theta3 + np.deg2rad(45)) + z3
   
    
    positions = {
        "shoulder": shoulder_coordinates,
        "upper_arm": upper_arm_coordinates,
        "elbow": np.array([x1, y1, z1]),
        "wrist": np.array([x2, y2, z2]),
        "knife": np.array([x3, y3, z3]),
        "knife_tip": np.array([x4, y4, z4]),
    }
    if DEBUG:
        for part, pos in positions.items():
            print(f"[DEBUG][mykinematics] {part} position: X={pos[0]:.3f} mm, Y={pos[1]:.3f} mm, Z={pos[2]:.3f} mm")
    return np.array([x4, y4, z4])  # Return word coordinates in mm



def joint_angles_to_trigo(joint_angles):
    """Convert joint angles from robot frame to world frame."""
    trigo = [0, 0, 0]  # Initialize list with 4 elements
    trigo[0]= 90 - joint_angles[0]  # Shoulder
    trigo[1]= -(joint_angles[1])-90   # Elbow
    trigo[2]= -(joint_angles[2])     # Wrist
    return np.array(trigo)


def trigo_to_joint_angles(trigo_angles):
    """Convert joint angles from world frame to robot frame."""
    joint_angles = [0, 0, 0]  # Initialize list with 4 elements
    joint_angles[0]= 90 - trigo_angles[0]  # Shoulder
    joint_angles[1]= -(trigo_angles[1]+90)   # Elbow
    joint_angles[2]= -(trigo_angles[2])   # Wrist
    return np.array(joint_angles)


def inverse_kinematics(xyz_target):

    """Compute joint angles (degrees) from target end-effector position (mm) in 2 dimensions.
    We constrain the knife to be parallel to the horisontal plane 
    
    """
    # Equations
     # We assume that the robot stays in 2d plane with y=0
    theta1, theta2, theta3 = 0, 0, 0  # Initialize angles

    x4_target = xyz_target[0]
    z4_target = xyz_target[2]

    # Link lengths
    L1 = robot_links["upperarm_forearm"]
    L2 = robot_links["forearm_wrist"]
    L3 = robot_links["wrist_knife"]
    L4 = robot_links["knife_knifetip"]

     # Base position
    x0 = upper_arm_coordinates[0]
    z0 = upper_arm_coordinates[2]

   # Constraint: θ1 + θ2 + θ3 + 45° = 0° (knife parallel to horizontal)
    # This means: θ1 + θ2 + θ3 = -45° = -π/4 rad
    
    def objective(vars):
        """Objective function to minimize (distance to target).
        vars = [theta1, theta2] (in radians)
        theta3 is determined by constraint: theta3 = -π/4 - theta1 - theta2
        """
        theta1, theta2 = vars
        theta3 = -np.pi/4 - theta1 - theta2  # Constraint
        
        # Forward kinematics
        x1 = L1 * np.cos(theta1-np.deg2rad(upper_arm_elbow_angle)) + x0
        z1 = L1 * np.sin(theta1-np.deg2rad(upper_arm_elbow_angle)) + z0
        
        x2 = L2 * np.cos(theta1 + theta2+ np.deg2rad(elbow_wrist_angle)) + x1
        z2 = L2 * np.sin(theta1 + theta2 + np.deg2rad(elbow_wrist_angle)) + z1
        
        x3 = L3 * np.cos(theta1 + theta2 + theta3) + x2
        z3 = L3 * np.sin(theta1 + theta2 + theta3) + z2
        
        x4 = L4 * np.cos(theta1 + theta2 + theta3 + np.pi/4) + x3
        z4 = L4 * np.sin(theta1 + theta2 + theta3 + np.pi/4) + z3
        
        # Return squared error
        return (x4 - x4_target)**2 + (z4 - z4_target)**2
    
     # Initial guess (straight down configuration)
    theta1_init = np.pi/4  # Start at 45° (middle of 0 to 90°)
    theta2_init = -np.pi/2  # Start at -90° (middle of -180° to 0°)
    initial_guess = [theta1_init, theta2_init]
    
     # Bounds: theta2 must be < -π/2
    bounds = [
        (0, np.pi/2),           # theta1: no bounds
        (-np.pi, 0)        # theta2: upper bound at -π/2
    ]
    
    # Solve with bounds
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    if not result.success:
        print(f"[WARNING] IK optimization failed: {result.message}")

    theta1_rad, theta2_rad = result.x
    theta3_rad = -np.pi/4 - theta1_rad - theta2_rad  # From constraint
    # Convert to degrees
    theta1_deg = np.rad2deg(theta1_rad)
    theta2_deg = np.rad2deg(theta2_rad)
    theta3_deg = np.rad2deg(theta3_rad)
    
    # Convert from trigo frame to robot frame
    trigo_angles = np.array([theta1_deg, theta2_deg, theta3_deg])
    joint_angles = trigo_to_joint_angles(trigo_angles)

    if DEBUG:
        print(f"[DEBUG] IK Solution: θ1={theta1_deg:.2f}°, θ2={theta2_deg:.2f}°, θ3={theta3_deg:.2f}°")
        print(f"  RMS error: {np.sqrt(result.fun):.3f} mm")
        
    return joint_angles


   

xyz = forward_kinematics([0,0, 45])
print(f"End-effector position from FK: X={xyz[0]:.3f} mm, Y={xyz[1]:.3f} mm, Z={xyz[2]:.3f} mm")
joint_angles = inverse_kinematics(xyz)
print(f"Joint angles from IK: Shoulder={joint_angles[0]:.2f}°, Elbow={joint_angles[1]:.2f}°, Wrist={joint_angles[2]:.2f}°")

