#!/usr/bin/env python3
"""
URDF Joint Analyzer
Parses a URDF file and extracts joint positions relative to parent links,
calculating 3D distances between connected links.
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse
from pathlib import Path

# URDF_PATH = Path(__file__).parent / "so101_ninja1.urdf"
URDF_PATH = r"C:\Repos_Razor\ninjabot-mechanics\Simulation\SO101\so101_ninja1.urdf"
def parse_xyz(xyz_string):
    """Parse xyz string into numpy array."""
    return np.array([float(x) for x in xyz_string.split()])


def calculate_3d_distance(xyz):
    """Calculate 3D Euclidean distance from origin."""
    return np.linalg.norm(xyz)


def rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (in radians) to rotation matrix.
    Uses ZYX Euler angle convention (standard in robotics).
    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # ZYX order: R = Rz * Ry * Rx
    return R_z @ R_y @ R_x


def create_transform_matrix(xyz, rpy):
    """
    Create 4x4 homogeneous transformation matrix from xyz and rpy.
    
    Args:
        xyz: position [x, y, z] in meters
        rpy: orientation [roll, pitch, yaw] in radians
    
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = xyz
    return T


def calculate_world_positions(joints):
    """
    Calculate the world frame position of each joint by following the kinematic chain.
    
    Args:
        joints: List of joint dictionaries
        
    Returns:
        Updated joints list with 'world_position' added
    """
    # Build a dictionary for quick lookup
    joint_dict = {j['child']: j for j in joints}
    
    # Start from base_link (assumed at origin)
    for joint in joints:
        # Get the chain from base to this joint
        chain = []
        current = joint
        
        while current is not None:
            chain.insert(0, current)
            # Find parent joint
            parent_link = current['parent']
            current = None
            for j in joints:
                if j['child'] == parent_link:
                    current = j
                    break
        
        # Compute cumulative transformation
        T_world = np.eye(4)
        
        for link in chain:
            # Get xyz in meters and rpy in radians
            xyz_m = link['xyz'] / 100.0  # Convert back to meters
            rpy_rad = np.radians(link['rpy'])
            
            T_joint = create_transform_matrix(xyz_m, rpy_rad)
            T_world = T_world @ T_joint
        
        # Extract world position and convert to cm
        world_pos = T_world[:3, 3] * 100.0
        joint['world_position'] = world_pos
        joint['world_distance'] = np.linalg.norm(world_pos)
    
    return joints


def parse_urdf_joints(urdf_file):
    """
    Parse URDF file and extract joint information.
    
    Args:
        urdf_file: Path to URDF file
        
    Returns:
        List of dictionaries containing joint information
    """
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    joints = []
    
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        parent = joint.find('parent')
        child = joint.find('child')
        origin = joint.find('origin')
        
        if parent is not None and child is not None:
            parent_link = parent.get('link')
            child_link = child.get('link')
            
            # Get xyz position
            xyz = np.array([0.0, 0.0, 0.0])
            rpy = np.array([0.0, 0.0, 0.0])
            
            if origin is not None:
                xyz_str = origin.get('xyz', '0 0 0')
                rpy_str = origin.get('rpy', '0 0 0')
                xyz = parse_xyz(xyz_str)
                xyz = xyz * 100 # Convert to cm
                rpy = parse_xyz(rpy_str)
                rpy = np.degrees(rpy)
            
            # Calculate 3D distance
            distance_3d = calculate_3d_distance(xyz)
            
            joints.append({
                'name': joint_name,
                'type': joint_type,
                'parent': parent_link,
                'child': child_link,
                'xyz': xyz,
                'rpy': rpy,
                'distance_3d': distance_3d
            })
    
    return joints


def print_joint_info(joints):
    """Print formatted joint information."""
    print("\n" + "="*80)
    print("URDF Joint Analysis - Relative Positions")
    print("="*80 + "\n")
    
    for joint in joints:
        print(f"Joint: {joint['name']}")
        print(f"  Type: {joint['type']}")
        print(f"  Parent: {joint['parent']} → Child: {joint['child']}")
        print(f"  Position (xyz) [cm] - Relative to Parent:")
        print(f"    X: {joint['xyz'][0]:>10.2f}")
        print(f"    Y: {joint['xyz'][1]:>10.2f}")
        print(f"    Z: {joint['xyz'][2]:>10.2f}")
        print(f"  Orientation (rpy) [deg]:")
        print(f"    Roll:  {joint['rpy'][0]:>10.2f}")
        print(f"    Pitch: {joint['rpy'][1]:>10.2f}")
        print(f"    Yaw:   {joint['rpy'][2]:>10.2f}")
        print(f"  3D Distance: {joint['distance_3d']:.2f} cm ")
        
        if 'world_position' in joint:
            print(f"  World Position [cm]:")
            print(f"    X: {joint['world_position'][0]:>10.2f}")
            print(f"    Y: {joint['world_position'][1]:>10.2f}")
            print(f"    Z: {joint['world_position'][2]:>10.2f}")
            print(f"  World Distance: {joint['world_distance']:.2f} cm")
        print()


def print_summary_table(joints):
    """Print a summary table of all joints."""
    print("\n" + "="*135)
    print("Summary Table - Relative and World Positions")
    print("="*135 + "\n")
    
    # Header
    print(f" {'Parent → Child':<30} "
          f"{'Rel X':<8} {'Rel Y':<8} {'Rel Z':<8} {'Rel D':<8} "
          f"{'World X':<8} {'World Y':<8} {'World Z':<8} {'World D':<8}")
    print("-" * 135)
    
    # Data rows
    for joint in joints:
        parent_child = f"{joint['parent']} → {joint['child']}"
        if len(parent_child) > 30:
            parent_child = parent_child[:27] + "..."
        
        world_x = joint.get('world_position', [0, 0, 0])[0]
        world_y = joint.get('world_position', [0, 0, 0])[1]
        world_z = joint.get('world_position', [0, 0, 0])[2]
        world_d = joint.get('world_distance', 0.0)
        
        print(f"{parent_child:<30} "
              f"{joint['xyz'][0]:>7.2f} {joint['xyz'][1]:>7.2f} {joint['xyz'][2]:>7.2f} {joint['distance_3d']:>7.2f} "
              f"{world_x:>7.2f} {world_y:>7.2f} {world_z:>7.2f} {world_d:>7.2f}")
    
    print("\nNote: All positions in cm. Relative positions are w.r.t. parent link frame.\n")


def main():

   
    
    # Parse URDF
    joints = parse_urdf_joints(URDF_PATH)
    
    if not joints:
        print("No joints found in URDF file!")
        return 1
    
    # Calculate world positions
    joints = calculate_world_positions(joints)
   
    print_joint_info(joints)
    print_summary_table(joints)
    
    return 0


if __name__ == '__main__':
    exit(main())
