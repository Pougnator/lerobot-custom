import math

import numpy as np
from scipy.optimize import minimize

DEBUG =True
robot_links = {
    "shoulder_upperarm": 79.21,
    "upperarm_forearm": 119,
    "forearm_wrist": 134.29,
    "wrist_knife": 97.0,
    "knife_knifetip": 80.0,
}

elbow_wrist_angle = 1.77        # degrees
upper_arm_elbow_angle = 13.81   # degrees
shoulder_coordinates = np.array([61, 0.0, 47.0])
upper_arm_coordinates = shoulder_coordinates + np.array([32, 0.0, 73.0])

# ─── Z-axis calibration correction ──────────────────────────────────────────
# Physical calibration on 2026-03-02 revealed that the FK model systematically
# UNDERESTIMATES the knife-tip Z coordinate by ~2.7 mm.  Root cause: the link
# lengths / joint offsets above describe a chain that is slightly shorter in
# the vertical direction than the real hardware (accumulated modelling error
# across shoulder, upper-arm, forearm, wrist, and knife links).
#
# Z_FK_BIAS_MM is added to z4 in BOTH:
#   • forward_kinematics()  — FK readback now matches physical measurements.
#   • _fk_chain() used inside inverse_kinematics() — the IK optimiser also uses
#     the corrected model, so it drives the chain to the physically correct height.
#
# RELATIONSHIP WITH config.py → BOARD_ORIGIN_IN_ROBOT[2]:
#   BOARD_ORIGIN_IN_ROBOT[2] has been reset to 0.0 — the Z error is handled here.
Z_FK_BIAS_MM = 2.7   # mm — physical calibration correction, 2026-03-02

# ─── Joint angle bounds (trigo frame, radians) ────────────────────────────────
# Used by both the IK optimiser (as scipy bounds) and the Jacobian controller
# (as explicit guard checks).  Single source of truth — edit here only.
TRIGO_A1_MIN = -10.0
TRIGO_A1_MAX = np.deg2rad(130)
TRIGO_A2_MIN = -np.pi
TRIGO_A2_MAX = 0.0
TRIGO_A3_MIN = -3 * np.pi / 4
TRIGO_A3_MAX = np.pi / 4

# ─── Module-level link constants ─────────────────────────────────────────────
# Pre-extracted so functions don't repeat the same dict lookups and deg2rad calls.
_L1 = robot_links["upperarm_forearm"]    # 119.00 mm
_L2 = robot_links["forearm_wrist"]       # 134.29 mm
_L3 = robot_links["wrist_knife"]         #  97.00 mm
_L4 = robot_links["knife_knifetip"]      #  80.00 mm
_x0 = upper_arm_coordinates[0]
_z0 = upper_arm_coordinates[2]
_d1 = np.deg2rad(upper_arm_elbow_angle)  # 13.81°
_d2 = np.deg2rad(elbow_wrist_angle)      #  1.77°


# ─── Private helpers ─────────────────────────────────────────────────────────

def _fk_chain(t1, t2, t3):
    """Full FK from trigo-frame angles (radians).

    Returns (x_wrist, z_wrist, x_tip, z_tip) in mm.
    Z_FK_BIAS_MM is applied to z_tip so both FK and IK use the same corrected model.
    """
    x1 = _L1 * np.cos(t1 - _d1) + _x0
    z1 = _L1 * np.sin(t1 - _d1) + _z0
    x2 = _L2 * np.cos(t1 + t2 + _d2) + x1
    z2 = _L2 * np.sin(t1 + t2 + _d2) + z1
    x3 = _L3 * np.cos(t1 + t2 + t3) + x2
    z3 = _L3 * np.sin(t1 + t2 + t3) + z2
    x4 = _L4 * np.cos(t1 + t2 + t3 + np.pi / 4) + x3
    z4 = _L4 * np.sin(t1 + t2 + t3 + np.pi / 4) + z3 + Z_FK_BIAS_MM
    return x2, z2, x4, z4


def _wrist_pos_and_jacobian(a1, a2):
    """Wrist position and Jacobian trig terms for (a1, a2) in trigo-frame radians.

    Returns (x_wrist, z_wrist, s1, c1, s12, c12, D) where:
      D = det(Jacobian) = L1·L2·sin(a2 + d1 + d2)
    """
    s1  = np.sin(a1 - _d1);  c1  = np.cos(a1 - _d1)
    s12 = np.sin(a1 + a2 + _d2);  c12 = np.cos(a1 + a2 + _d2)
    x_wrist = _L1 * c1 + _x0 + _L2 * c12
    z_wrist = _L1 * s1 + _z0 + _L2 * s12
    D = _L1 * _L2 * np.sin(a2 + _d1 + _d2)
    return x_wrist, z_wrist, s1, c1, s12, c12, D


def _check_trigo_bounds(a1, a2, a3, tag=""):
    """Return an error string if any trigo-frame angle is out of bounds, else None."""
    for val, lo, hi, name in [
        (a1, TRIGO_A1_MIN, TRIGO_A1_MAX, "a1"),
        (a2, TRIGO_A2_MIN, TRIGO_A2_MAX, "a2"),
        (a3, TRIGO_A3_MIN, TRIGO_A3_MAX, "a3"),
    ]:
        if not (lo <= val <= hi):
            return f"{tag}{name}={np.degrees(val):.1f}° out of [{np.degrees(lo):.0f},{np.degrees(hi):.0f}]°"
    return None


# ─── Public functions ─────────────────────────────────────────────────────────

def forward_kinematics(joint_angles):
    """Compute knife-tip position (mm) from joint angles (degrees, robot frame)."""
    trigo = joint_angles_to_trigo(joint_angles)
    t1, t2, t3 = np.deg2rad(trigo)
    _, _, x4, z4 = _fk_chain(t1, t2, t3)
    if DEBUG:
        x2, z2, _, _ = _fk_chain(t1, t2, t3)
        print(f"[DEBUG][FK] wrist=({x2:.1f},{z2:.1f})  tip=({x4:.1f},{z4:.1f}) mm")
    return np.array([x4, 0.0, z4])


def forward_kinematics_wrist(joint_angles):
    """Compute wrist position (x, z) in mm from joint angles (degrees, robot frame)."""
    trigo = joint_angles_to_trigo(joint_angles)
    t1, t2, t3 = np.deg2rad(trigo)
    x2, z2, _, _ = _fk_chain(t1, t2, t3)
    return x2, z2


def get_wrist_xz(target_knife_pos, knife_tilt_deg=0.0):
    """Invert the wrist→tip chain to find the wrist position.

    Given a target knife-tip position and tilt, returns the wrist (x, z) in mm.

    knife_tilt_deg: angle of L4 (knife_knifetip link) from horizontal.
      0° = horizontal, negative = tip down, positive = tip up.
    L3 (wrist_knife) is always 45° behind L4: L3 angle = tilt - 45°.
    """
    alpha = np.deg2rad(knife_tilt_deg)
    knife_x = target_knife_pos[0]
    knife_z = target_knife_pos[2]
    wrist_x = knife_x - _L4 * np.cos(alpha) - _L3 * np.cos(alpha - np.deg2rad(45))
    wrist_z = knife_z - _L4 * np.sin(alpha) - _L3 * np.sin(alpha - np.deg2rad(45))
    return wrist_x, wrist_z


def get_tilt_from_joints(joint_angles):
    """Return knife tilt in degrees from robot-frame joint angles.

    Tilt = sum of trigo-frame angles + 45°
    (0° = horizontal, negative = tip down, positive = tip up)
    """
    trigo = joint_angles_to_trigo(np.array(joint_angles))
    return float(np.sum(trigo) + 45.0)


def joint_angles_to_trigo(joint_angles):
    """Convert joint angles from robot frame to trigo frame (degrees)."""
    return np.array([
        90 - joint_angles[0],       # Shoulder
        -joint_angles[1] - 90,      # Elbow
        -joint_angles[2],           # Wrist
    ])


def trigo_to_joint_angles(trigo_angles):
    """Convert joint angles from trigo frame to robot frame (degrees)."""
    return np.array([
        90 - trigo_angles[0],       # Shoulder
        -(trigo_angles[1] + 90),    # Elbow
        -trigo_angles[2],           # Wrist
    ])


def _check_wrist_waypoint(x_wrist_mm, z_wrist_mm, tilt_deg):
    """Return a string describing why (x_wrist, z_wrist, tilt) is unreachable, or None if valid.

    Uses closed-form 2R IK for shoulder+elbow, then derives a3 from the tilt constraint.
    """
    px = x_wrist_mm - _x0
    pz = z_wrist_mm - _z0
    r2 = px ** 2 + pz ** 2

    cos_sum = (r2 - _L1 ** 2 - _L2 ** 2) / (2 * _L1 * _L2)
    if abs(cos_sum) > 1.0:
        return f"wrist unreachable (r={np.sqrt(r2):.1f} mm, cos_sum={cos_sum:.3f})"

    sum_angle = -np.arccos(cos_sum)   # elbow-down solution
    a2 = sum_angle - _d1 - _d2
    beta = np.arctan2(pz, px)
    gamma = np.arctan2(_L2 * np.sin(sum_angle), _L1 + _L2 * np.cos(sum_angle))
    a1 = beta - gamma + _d1
    a3 = np.deg2rad(tilt_deg - 45.0) - a1 - a2

    return _check_trigo_bounds(a1, a2, a3)


def calculate_linear_trajectory(target_pos, target_tilt, starting_tilt, starting_pos,
                                 speed=3.0, steps_per_second=None):
    """Calculate a linear wrist trajectory from starting_pos to target_pos.

    The wrist moves linearly in time; tilt interpolates linearly between
    starting_tilt and target_tilt.

    Args:
        target_pos    : target knife-tip position in mm [x, y, z]
        target_tilt   : target knife tilt in degrees
        starting_tilt : starting knife tilt in degrees
        starting_pos  : starting knife-tip position in mm [x, y, z]
        speed         : travel speed in mm/s
        steps_per_second : waypoints per second

    Returns:
        list of (x_wrist_mm, z_wrist_mm, tilt_deg) tuples

    Raises:
        ValueError if start and target are too close, or any waypoint is out of reach.
    """
    x_wrist_target, z_wrist_target = get_wrist_xz(target_pos, knife_tilt_deg=target_tilt)
    x_wrist_start,  z_wrist_start  = get_wrist_xz(starting_pos, knife_tilt_deg=starting_tilt)

    distance   = np.sqrt((x_wrist_target - x_wrist_start) ** 2
                         + (z_wrist_target - z_wrist_start) ** 2)
    print(f"Distnce: {distance:.1f} mm  speed: {speed:.1f} mm/s  steps/s: {steps_per_second}")
    print(f'time steps: {int(distance / speed * steps_per_second)}')

    time_steps = int((distance / speed) * steps_per_second)
    if time_steps == 0:
        raise ValueError(
            "Start and target are too close to compute a trajectory at the given "
            "speed and steps_per_second."
        )

    trajectory = []
    for step in range(time_steps + 1):
        k=5.0  # tilt interpolation speed factor (higher = faster approach to target tilt)
        t = step / time_steps
        xw = x_wrist_start + t * (x_wrist_target - x_wrist_start)
        zw = z_wrist_start  + t * (z_wrist_target  - z_wrist_start)
        tilt_t = target_tilt + (target_tilt - starting_tilt) * math.exp(-k*t)

        # tilt_t = starting_tilt + t * (target_tilt - starting_tilt)
        # tilt_t = starting_tilt + (1 - math.cos(t * math.pi/2)) * (target_tilt - starting_tilt)
        trajectory.append((xw, zw, tilt_t))

    invalid = []
    for i, (xw, zw, tilt_t) in enumerate(trajectory):
        reason = _check_wrist_waypoint(xw, zw, tilt_t)
        if reason:
            invalid.append((i, reason))

    if invalid:
        msgs = "; ".join(f"step {i}: {r}" for i, r in invalid[:3])
        if len(invalid) > 3:
            msgs += f" … and {len(invalid) - 3} more"
        raise ValueError(f"{len(invalid)}/{len(trajectory)} waypoints out of reach: {msgs}")

    return trajectory


def calculate_cutting_trajectory(x_tip_cm, z_start_cm, tilt_deg, t_offset,
                                  amplitude_cm, frequency_hz, descent_rate_cm_s,
                                  z_floor_cm=None, steps_per_second=10, batch_size=30,
                                  tilt_rate_deg_per_cm=2.0):
    """Generate a batch of wrist waypoints for the oscillating+descending cutting motion.

    The knife-tip follows:
        x(t) = x_tip_cm + amplitude_cm * sin(2π * frequency_hz * t)
        z(t) = z_start_cm - descent_rate_cm_s * t

    Tilt rotates as the knife descends:
        tilt(t) = tilt_deg + tilt_rate_deg_per_cm * depth_cm(t)
    where depth_cm(t) = z_start_cm - z(t) = descent_rate_cm_s * t.
    Positive tilt_rate rotates the tip upward (toward 0°) as depth increases,
    e.g. tilt_deg=-20°, rate=2.0 → -14° after 3 cm descent.

    Each tip position is converted to wrist coordinates via get_wrist_xz at the
    per-step tilt so the result can be fed directly into _execute_trajectory.

    Parameters
    ----------
    x_tip_cm, z_start_cm : nominal tip position in cm (robot world frame)
    tilt_deg             : knife tilt at z_start_cm in degrees (0 = horizontal, negative = tip down)
    t_offset             : time in seconds at the start of this batch (for trajectory continuity)
    amplitude_cm         : lateral oscillation half-amplitude in cm
    frequency_hz         : oscillation frequency in Hz
    descent_rate_cm_s    : Z descent rate in cm/s
    z_floor_cm           : hard commanded-Z lower bound; generation stops if z(t) ≤ this
    steps_per_second      : waypoints per second (dt = 1 / steps_per_second)
    batch_size           : maximum number of waypoints to generate
    tilt_rate_deg_per_cm : tilt change per cm of descent (positive = tip rotates toward horizontal)

    Returns
    -------
    List of (x_wrist_mm, z_wrist_mm, tilt_deg) tuples (may be shorter than batch_size).
    Returns an empty list if the first waypoint already violates z_floor_cm.
    """
    dt = 1.0 / steps_per_second
    trajectory = []
    for i in range(batch_size):
        t = t_offset + i * dt
        x_t = x_tip_cm + amplitude_cm * np.sin(2 * np.pi * frequency_hz * t)
        z_t = z_start_cm - descent_rate_cm_s * t
        if z_floor_cm is not None and z_t <= z_floor_cm:
            break
        depth_cm  = z_start_cm - z_t                          # cm descended from start
        step_tilt = tilt_deg + tilt_rate_deg_per_cm * depth_cm
        tip_pos   = np.array([x_t * 10.0, 0.0, z_t * 10.0])  # cm → mm
        x_wrist, z_wrist = get_wrist_xz(tip_pos, knife_tilt_deg=step_tilt)
        trajectory.append((x_wrist, z_wrist, step_tilt))
    return trajectory


def wrist_rotation_matrix(joint_angles):
    """Return the 3×3 rotation matrix of the wrist frame in robot world coordinates.

    Simple Ry(alpha) rotation — at alpha=0, wrist axes align with robot axes.
    """
    trigo = joint_angles_to_trigo(joint_angles)
    alpha = np.sum(np.deg2rad(trigo))
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ ca,  0., -sa],
        [ 0.,  1.,  0.],
        [ sa,  0.,  ca],
    ])


def wrist_rotation_matrix_urdf(joint_angles):
    """Return the 3×3 rotation matrix matching the URDF wrist_link frame convention."""
    trigo = joint_angles_to_trigo(joint_angles)
    alpha = np.sum(np.deg2rad(trigo))
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ 0., -ca, -sa],
        [ 1.,  0.,  0.],
        [ 0., -sa,  ca],
    ])


def inverse_kinematics(xyz_target, current_joints=None, calibration=None):
    """Compute joint angles (degrees) from target end-effector position (mm).

    The knife is kept as close to horizontal as possible via a large penalty
    term (soft constraint) rather than a hard equality.
    """
    x4_target, z4_target = xyz_target[0], xyz_target[2]
    KNIFE_PENALTY_WEIGHT = 5000.0

    def objective(vars):
        t1, t2, t3 = vars
        _, _, x4, z4 = _fk_chain(t1, t2, t3)
        pos_err    = (x4 - x4_target) ** 2 + (z4 - z4_target) ** 2
        knife_tilt = t1 + t2 + t3 + np.pi / 4
        return pos_err + KNIFE_PENALTY_WEIGHT * knife_tilt ** 2

    t1i, t2i = np.pi / 4, -np.pi / 2
    t3i = -np.pi / 4 - t1i - t2i
    if current_joints is not None:
        trigo_c = joint_angles_to_trigo(np.array(current_joints[1:4]))
        t1i, t2i, t3i = np.deg2rad(trigo_c)

    bounds = [(TRIGO_A1_MIN, TRIGO_A1_MAX),
              (TRIGO_A2_MIN, TRIGO_A2_MAX),
              (TRIGO_A3_MIN, TRIGO_A3_MAX)]

    result = minimize(objective, [t1i, t2i, t3i], bounds=bounds, method='L-BFGS-B')
    if not result.success:
        print(f"[WARNING] IK optimization failed: {result.message}")

    t1, t2, t3 = result.x
    _, _, x4, z4 = _fk_chain(t1, t2, t3)
    pos_rms = np.sqrt((x4 - x4_target) ** 2 + (z4 - z4_target) ** 2)
    if pos_rms > 3.0:
        print(f"[WARNING] IK solution has high error: {pos_rms:.3f} mm")
        return None

    knife_tilt_deg = np.degrees(t1 + t2 + t3 + np.pi / 4)
    if abs(knife_tilt_deg) > 1.0:
        print(f"[INFO] Knife tilt: {knife_tilt_deg:+.2f}° from horizontal")

    t1d, t2d, t3d = np.rad2deg([t1, t2, t3])
    if DEBUG:
        print(f"[DEBUG] IK: θ1={t1d:.2f}°  θ2={t2d:.2f}°  θ3={t3d:.2f}°"
              f"  pos_err={pos_rms:.3f} mm  tilt={knife_tilt_deg:+.2f}°")
    return trigo_to_joint_angles(np.array([t1d, t2d, t3d]))


def jacobian_control_step(current_joint_angles, target_wrist_x_mm, target_wrist_z_mm,
                          target_tilt_deg, obs_tilt=None):
    """Compute a single Jacobian control step toward the target wrist position and tilt.

    Controls three independent DOF:
      - (a1, a2) → wrist position via 2×2 analytical Jacobian inverse
      - a3       → absorbs the remaining tilt error (da3 = dtilt − da1 − da2)

    Parameters
    ----------
    current_joint_angles : array-like [shoulder_lift, elbow_flex, wrist_flex] degrees (robot frame)
    target_wrist_x_mm    : target wrist X in mm (robot world frame)
    target_wrist_z_mm    : target wrist Z in mm (robot world frame)
    target_tilt_deg      : target knife tilt in degrees (0 = horizontal, positive = tip up)
    obs_tilt             : observed current tilt in degrees (from IMU if available, else None)

    Returns
    -------
    np.ndarray [shoulder_lift, elbow_flex, wrist_flex] in robot-frame degrees,
    or None if the configuration is singular or a joint bound would be exceeded.
    """
    trigo = joint_angles_to_trigo(np.array(current_joint_angles))
    a1, a2, a3 = np.deg2rad(trigo)

    x_wrist, z_wrist, s1, c1, s12, c12, D = _wrist_pos_and_jacobian(a1, a2)

    dx = target_wrist_x_mm - x_wrist
    dz = target_wrist_z_mm - z_wrist

    current_tilt_deg = obs_tilt if obs_tilt is not None else (np.degrees(a1 + a2 + a3) + 45.0)
    if obs_tilt is None and DEBUG:
        print(f"[DEBUG][JAC-CTRL] Observed tilt unavailble! Using FK tilt")
    dtilt = np.deg2rad(target_tilt_deg - current_tilt_deg)

    if abs(D) < 500.0:
        print(f"[JAC-CTRL] Near-singular configuration (D={D:.1f} mm²) — skipping step.")
        return None

    da1 = (_L2 * c12 * dx + _L2 * s12 * dz) / D
    da2 = -((_L1 * c1 + _L2 * c12) * dx + (_L1 * s1 + _L2 * s12) * dz) / D
    da3 = dtilt - da1 - da2

    a1_new, a2_new, a3_new = a1 + da1, a2 + da2, a3 + da3

    if DEBUG:
        print(f"[DEBUG][JAC-CTRL] da1={np.degrees(da1):.2f}°, da2={np.degrees(da2):.2f}°, da3={np.degrees(da3):.2f}°"
              f"  tilt error={np.degrees(dtilt):.2f}°")

    err = _check_trigo_bounds(a1_new, a2_new, a3_new, tag="[JAC-CTRL] ")
    if err:
        print(err)
        return None

    return trigo_to_joint_angles(np.rad2deg([a1_new, a2_new, a3_new]))


def jacobian_cutting_step(current_joint_angles, x_des_mm, z_des_mm, knife_tilt_deg=0.0):
    """Compute the next joint-angle command for one closed-loop cutting step.

    Holds the knife tilt constant at knife_tilt_deg via the constraint:
        a1 + a2 + a3 + π/4 = β    (β = knife_tilt_deg in radians)

    Uses the 2×2 analytical Jacobian inverse — no optimizer called.

    Parameters
    ----------
    current_joint_angles : array-like [shoulder_lift, elbow_flex, wrist_flex] degrees (robot frame)
    x_des_mm, z_des_mm   : desired knife-tip position in mm (robot world frame)
    knife_tilt_deg       : desired knife tilt in degrees (0 = horizontal, negative = tip down)

    Returns
    -------
    np.ndarray [shoulder_lift, elbow_flex, wrist_flex] in robot-frame degrees,
    or None if the configuration is singular or out of joint bounds.
    """
    beta = np.deg2rad(knife_tilt_deg)
    dx_fixed = _L3 * np.cos(beta - np.pi / 4) + _L4 * np.cos(beta)
    dz_fixed = _L3 * np.sin(beta - np.pi / 4) + _L4 * np.sin(beta)

    trigo = joint_angles_to_trigo(np.array(current_joint_angles))
    a1, a2, _ = np.deg2rad(trigo)

    x_wrist, z_wrist, s1, c1, s12, c12, D = _wrist_pos_and_jacobian(a1, a2)
    x_cur = x_wrist + dx_fixed
    z_cur = z_wrist + dz_fixed + Z_FK_BIAS_MM

    dx = x_des_mm - x_cur
    dz = z_des_mm - z_cur

    if abs(D) < 500.0:
        print(f"[CUT-JAC] Near-singular configuration (D={D:.1f} mm²) — skipping step.")
        return None

    da1 = (_L2 * c12 * dx + _L2 * s12 * dz) / D
    da2 = -((_L1 * c1 + _L2 * c12) * dx + (_L1 * s1 + _L2 * s12) * dz) / D

    a1_new = a1 + da1
    a2_new = a2 + da2
    # a3 enforced directly from tilt constraint (damped to avoid wrist whip)
    a3_cur = np.deg2rad(trigo[2])
    da3 = beta - np.pi / 4 - a1_new - a2_new - a3_cur
    a3_new = a3_cur + da3 / 4

    err = _check_trigo_bounds(a1_new, a2_new, a3_new, tag="[CUT-JAC] ")
    if err:
        print(err)
        return None

    return trigo_to_joint_angles(np.rad2deg([a1_new, a2_new, a3_new]))


xyz = forward_kinematics([0, 0, 45.77])
xyz = [0.36, 0.0, 0.03]  # Target position in mm
xyz = np.array(xyz) * 1000  # Convert to mm
print(f"End-effector position from FK: X={xyz[0]:.3f} mm, Y={xyz[1]:.3f} mm, Z={xyz[2]:.3f} mm")
joint_angles = inverse_kinematics(xyz)
if joint_angles is not None:
    print(f"Joint angles from IK: Shoulder={joint_angles[0]:.2f}°, Elbow={joint_angles[1]:.2f}°, Wrist={joint_angles[2]:.2f}°")
