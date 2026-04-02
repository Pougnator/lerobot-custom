import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize

DEBUG = False
robot_links = {
    "shoulder_upperarm": 79.21,
    "upperarm_forearm": 119,
    "forearm_wrist": 134.29,
    "wrist_knife": 97.0,
    "knife_knifetip": 80.0,
}

elbow_wrist_angle = 1.77 # degrees
# elbow_wrist_angle = 0 # degrees
upper_arm_elbow_angle = 13.81  # degrees
shoulder_coordinates = np.array([61, 0.0, 47.0])  # Shoulder base at origin
upper_arm_coordinates = shoulder_coordinates + np.array([32, 0.0, 73.0])

# ─── Z-axis calibration correction ──────────────────────────────────────────
# Physical calibration on 2026-03-02 revealed that the FK model systematically
# UNDERESTIMATES the knife-tip Z coordinate by ~2.7 mm.  Root cause: the link
# lengths / joint offsets above describe a chain that is slightly shorter in
# the vertical direction than the real hardware (accumulated modelling error
# across shoulder, upper-arm, forearm, wrist, and knife links).
#
# Rather than re-fitting all link lengths (which requires a full geometric
# calibration rig), we apply a single additive correction Z_FK_BIAS_MM to the
# knife-tip Z in both FK and IK.
#
# Z_FK_BIAS_MM is added to z4 in BOTH:
#   • forward_kinematics()  — FK readback now matches physical measurements.
#   • _fk_tip() inside inverse_kinematics() — the IK optimiser also uses the
#     corrected model, so it drives the chain to the physically correct height.
#     Without this, IK would overshoot the requested Z target by ~2.7 mm.
#
# RELATIONSHIP WITH config.py → BOARD_ORIGIN_IN_ROBOT[2]:
#   Before this fix, BOARD_ORIGIN_IN_ROBOT[2] was set to −0.0027 m during
#   calibration to compensate for this same FK error from the board-frame side.
#   Now that mykinematics corrects the error directly, BOARD_ORIGIN_IN_ROBOT[2]
#   has been reset to 0.0.  Leaving it at −0.0027 would DOUBLE-COUNT the
#   correction and introduce a +2.7 mm error in the opposite direction.
Z_FK_BIAS_MM = 2.7   # mm — physical calibration correction, 2026-03-02

# ─── Joint angle bounds (trigo frame, radians) ────────────────────────────────
# Used by both the IK optimiser (as scipy bounds) and the Jacobian controller
# (as explicit guard checks).  Single source of truth — edit here only.
TRIGO_A1_MIN = 0.0                   # shoulder: 0°   → lift =  90° (arm vertical)
TRIGO_A1_MAX = np.deg2rad(130)       # shoulder: 130° → lift ≈ −40° (leaning forward)
TRIGO_A2_MIN = -np.pi                # elbow: −180°
TRIGO_A2_MAX = 0.0                   # elbow:    0°   (fully folded)
TRIGO_A3_MIN = -3 * np.pi / 4       # wrist: −135°
TRIGO_A3_MAX = np.pi / 4            # wrist:  +45°

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
    # Apply the Z calibration bias so FK matches physical measurements.
    # See Z_FK_BIAS_MM comment block above for full explanation.
    z4 += Z_FK_BIAS_MM

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



def wrist_rotation_matrix(joint_angles):
    """Return the 3×3 rotation matrix of the wrist frame in robot world coordinates.

    Simple Ry(alpha) rotation — at alpha=0, wrist axes align with robot axes:
      X_wrist = X_robot,  Y_wrist = Y_robot,  Z_wrist = Z_robot

    At angle alpha:
      X_robot = X_wrist * cos(alpha) - Z_wrist * sin(alpha)
      Z_robot = X_wrist * sin(alpha) + Z_wrist * cos(alpha)

    Use this when tag positions and knife offsets are expressed in this
    simple physical wrist frame (not URDF frame).
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
    """Return the 3×3 rotation matrix matching the URDF wrist_link frame convention.

    Columns are the URDF wrist-frame axes expressed in the FK world frame:
      col 0  wrist X = [0, 1, 0]             (robot Y, perpendicular to arm plane)
      col 1  wrist Y = [-cos α, 0, -sin α]   (toward shoulder; −arm-forward direction)
      col 2  wrist Z = [-sin α, 0,  cos α]   (perpendicular to arm, outward)

    where α = theta1 + theta2 + theta3 in the trigo frame (radians).

    Convention matches the URDF wrist_link frame (joint rpy "0 0 −π/2"):
    use when offsets are given in URDF coordinates.
    """
    trigo = joint_angles_to_trigo(joint_angles)
    alpha = np.sum(np.deg2rad(trigo))
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ 0., -ca, -sa],
        [ 1.,  0.,  0.],
        [ 0., -sa,  ca],
    ])


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


def inverse_kinematics(xyz_target, current_joints=None, calibration=None):
    """Compute joint angles (degrees) from target end-effector position (mm).

    The knife is kept as close to horizontal as possible via a large penalty
    term (soft constraint) rather than a hard equality.  This allows the
    optimizer to tilt the blade by a few degrees near the edges of the
    workspace where the hard constraint would leave no feasible solution.

    The shoulder bound is extended to 110° (trigo frame) — equivalent to
    shoulder_lift ≈ -20° — to cover positions that require the upper arm to
    lean slightly past vertical.
    """
    x4_target = xyz_target[0]
    z4_target = xyz_target[2]

    L1 = robot_links["upperarm_forearm"]
    L2 = robot_links["forearm_wrist"]
    L3 = robot_links["wrist_knife"]
    L4 = robot_links["knife_knifetip"]
    x0 = upper_arm_coordinates[0]
    z0 = upper_arm_coordinates[2]

    # Weight on knife-angle penalty.  At 5000, a 1° tilt adds ~1.5 to the
    # objective — comparable to a 1.2 mm position error — so the solver stays
    # near-horizontal everywhere inside the workspace.
    KNIFE_PENALTY_WEIGHT = 5000.0

    def _fk_tip(t1, t2, t3):
        """Return knife-tip (x4, z4) from trigo-frame angles (radians)."""
        x1 = L1 * np.cos(t1 - np.deg2rad(upper_arm_elbow_angle)) + x0
        z1 = L1 * np.sin(t1 - np.deg2rad(upper_arm_elbow_angle)) + z0
        x2 = L2 * np.cos(t1 + t2 + np.deg2rad(elbow_wrist_angle)) + x1
        z2 = L2 * np.sin(t1 + t2 + np.deg2rad(elbow_wrist_angle)) + z1
        x3 = L3 * np.cos(t1 + t2 + t3) + x2
        z3 = L3 * np.sin(t1 + t2 + t3) + z2
        x4 = L4 * np.cos(t1 + t2 + t3 + np.pi / 4) + x3
        z4 = L4 * np.sin(t1 + t2 + t3 + np.pi / 4) + z3
        # MUST match forward_kinematics: the optimiser must target the same
        # corrected Z that FK would report, otherwise IK overshoots by
        # Z_FK_BIAS_MM (~2.7 mm).  See Z_FK_BIAS_MM comment block above.
        z4 += Z_FK_BIAS_MM
        return x4, z4

    def objective(vars):
        t1, t2, t3 = vars
        x4, z4 = _fk_tip(t1, t2, t3)
        pos_err    = (x4 - x4_target) ** 2 + (z4 - z4_target) ** 2
        # Knife horizontal ↔ t1+t2+t3+π/4 = 0
        knife_tilt = t1 + t2 + t3 + np.pi / 4
        return pos_err + KNIFE_PENALTY_WEIGHT * knife_tilt ** 2

    # Initial guess: horizontal knife from current joint state (or default)
    theta1_init = np.pi / 4
    theta2_init = -np.pi / 2
    theta3_init = -np.pi / 4 - theta1_init - theta2_init  # horizontal
    if current_joints is not None:
        trigo_current = joint_angles_to_trigo(np.array(current_joints[1:4]))
        theta1_init = np.deg2rad(trigo_current[0])
        theta2_init = np.deg2rad(trigo_current[1])
        theta3_init = np.deg2rad(trigo_current[2])

    bounds = [
        (TRIGO_A1_MIN, TRIGO_A1_MAX),
        (TRIGO_A2_MIN, TRIGO_A2_MAX),
        (TRIGO_A3_MIN, TRIGO_A3_MAX),
    ]

    result = minimize(objective, [theta1_init, theta2_init, theta3_init],
                      bounds=bounds, method='L-BFGS-B')
    if not result.success:
        print(f"[WARNING] IK optimization failed: {result.message}")

    t1, t2, t3 = result.x

    # Evaluate position error separately (objective also contains penalty)
    x4, z4 = _fk_tip(t1, t2, t3)
    pos_rms = np.sqrt((x4 - x4_target) ** 2 + (z4 - z4_target) ** 2)
    if pos_rms > 3.0:
        print(f"[WARNING] IK solution has high error: {pos_rms:.3f} mm")
        return None

    knife_tilt_deg = np.degrees(t1 + t2 + t3 + np.pi / 4)
    if abs(knife_tilt_deg) > 1.0:
        print(f"[INFO] Knife tilt: {knife_tilt_deg:+.2f}° from horizontal")

    theta1_deg, theta2_deg, theta3_deg = np.rad2deg([t1, t2, t3])
    joint_angles = trigo_to_joint_angles(np.array([theta1_deg, theta2_deg, theta3_deg]))

    if DEBUG:
        print(f"[DEBUG] IK: θ1={theta1_deg:.2f}°  θ2={theta2_deg:.2f}°  θ3={theta3_deg:.2f}°"
              f"  pos_err={pos_rms:.3f} mm  tilt={knife_tilt_deg:+.2f}°")

    return joint_angles


def inverse_kinematics_prefer_up(xyz_target, current_joints=None, calibration=None):
    """IK with an asymmetric knife-tilt penalty: zero tilt is optimal, but
    negative tilt (tip pointing down) is penalised more heavily than positive
    tilt (tip pointing up).

    Identical to inverse_kinematics() except the objective uses:
        W_pos * tilt²  +  (W_neg - W_pos) * max(0, -tilt)²
    so that a 1° downward tilt costs W_neg/W_pos times more than 1° upward.
    """
    x4_target = xyz_target[0]
    z4_target = xyz_target[2]

    L1 = robot_links["upperarm_forearm"]
    L2 = robot_links["forearm_wrist"]
    L3 = robot_links["wrist_knife"]
    L4 = robot_links["knife_knifetip"]
    x0 = upper_arm_coordinates[0]
    z0 = upper_arm_coordinates[2]

    KNIFE_PENALTY_POS = 5000.0    # penalty weight for negative (tip pointing downward) tilt — baseline
    KNIFE_PENALTY_NEG = 20000.0   # penalty weight f  or positive (tip pointing upward) tilt — 4× stronger

    def _fk_tip(t1, t2, t3):
        x1 = L1 * np.cos(t1 - np.deg2rad(upper_arm_elbow_angle)) + x0
        z1 = L1 * np.sin(t1 - np.deg2rad(upper_arm_elbow_angle)) + z0
        x2 = L2 * np.cos(t1 + t2 + np.deg2rad(elbow_wrist_angle)) + x1
        z2 = L2 * np.sin(t1 + t2 + np.deg2rad(elbow_wrist_angle)) + z1
        x3 = L3 * np.cos(t1 + t2 + t3) + x2
        z3 = L3 * np.sin(t1 + t2 + t3) + z2
        x4 = L4 * np.cos(t1 + t2 + t3 + np.pi / 4) + x3
        z4 = L4 * np.sin(t1 + t2 + t3 + np.pi / 4) + z3
        z4 += Z_FK_BIAS_MM
        return x4, z4

    def objective(vars):
        t1, t2, t3 = vars
        x4, z4 = _fk_tip(t1, t2, t3)
        pos_err    = (x4 - x4_target) ** 2 + (z4 - z4_target) ** 2
        knife_tilt = t1 + t2 + t3 + np.pi / 4
        tilt_penalty = (KNIFE_PENALTY_POS * knife_tilt ** 2
                        + (KNIFE_PENALTY_NEG - KNIFE_PENALTY_POS) * np.maximum(0.0, knife_tilt) ** 2)
        return pos_err + tilt_penalty

    theta1_init = np.pi / 4
    theta2_init = -np.pi / 2
    theta3_init = -np.pi / 4 - theta1_init - theta2_init
    if current_joints is not None:
        trigo_current = joint_angles_to_trigo(np.array(current_joints[1:4]))
        theta1_init = np.deg2rad(trigo_current[0])
        theta2_init = np.deg2rad(trigo_current[1])
        theta3_init = np.deg2rad(trigo_current[2])

    bounds = [
        (TRIGO_A1_MIN, TRIGO_A1_MAX),
        (TRIGO_A2_MIN, TRIGO_A2_MAX),
        (TRIGO_A3_MIN, TRIGO_A3_MAX),
    ]

    result = minimize(objective, [theta1_init, theta2_init, theta3_init],
                      bounds=bounds, method='L-BFGS-B')
    if not result.success:
        print(f"[WARNING] IK (prefer_up) optimization failed: {result.message}")

    t1, t2, t3 = result.x

    x4, z4 = _fk_tip(t1, t2, t3)
    pos_rms = np.sqrt((x4 - x4_target) ** 2 + (z4 - z4_target) ** 2)
    if pos_rms > 3.0:
        print(f"[WARNING] IK (prefer_up) solution has high error: {pos_rms:.3f} mm")
        return None

    knife_tilt_deg = np.degrees(t1 + t2 + t3 + np.pi / 4)
    if abs(knife_tilt_deg) > 1.0:
        print(f"[INFO] Knife tilt (prefer_up): {knife_tilt_deg:+.2f}° from horizontal")

    theta1_deg, theta2_deg, theta3_deg = np.rad2deg([t1, t2, t3])
    joint_angles = trigo_to_joint_angles(np.array([theta1_deg, theta2_deg, theta3_deg]))

    if DEBUG:
        print(f"[DEBUG] IK (prefer_up): θ1={theta1_deg:.2f}°  θ2={theta2_deg:.2f}°  θ3={theta3_deg:.2f}°"
              f"  pos_err={pos_rms:.3f} mm  tilt={knife_tilt_deg:+.2f}°")

    return joint_angles




def compute_cutting_trajectory_jacobian(
    initial_joint_angles,
    x_center_mm, z_start_mm,
    amplitude_mm=15.0, frequency_hz=0.5, descent_rate_mm_s=10.0,
    dt_s=0.20, z_stop_mm=5.0, z_floor_mm=-50.0,
):
    """Precompute a cutting trajectory as a list of joint-angle waypoints.

    Uses the 2-DOF Jacobian method under the knife-horizontal constraint
    (t1 + t2 + t3 + π/4 = 0).  Because t3 is fully determined by t1 and t2
    through this constraint, the Jacobian reduces to a 2×2 system that is
    inverted analytically at each step — no optimizer is called.

    Trajectory (t = elapsed seconds):
        x(t) = x_center_mm + amplitude_mm * sin(2π * frequency_hz * t)
        z(t) = z_start_mm  - descent_rate_mm_s * t

    Parameters
    ----------
    initial_joint_angles : array-like [shoulder_lift, elbow_flex, wrist_flex]
        Current joint angles in robot-frame degrees — used as the seed for
        the first Jacobian step.
    x_center_mm, z_start_mm : float
        Trajectory starting position in mm (robot world frame).
    amplitude_mm : float
        Lateral (X) oscillation half-amplitude in mm.
    frequency_hz : float
        Lateral oscillation frequency in Hz.
    descent_rate_mm_s : float
        Rate of Z descent in mm/s.
    dt_s : float
        Time between waypoints in seconds.
    z_stop_mm : float
        Stop when the commanded Z reaches this value (primary criterion).
    z_floor_mm : float
        Hard safety floor: stop immediately if commanded Z falls below this.

    Returns
    -------
    list of dict, each with keys:
        'joint_angles' : [shoulder_lift, elbow_flex, wrist_flex] degrees (robot frame)
        'x_mm'         : commanded X (mm)
        'z_mm'         : commanded Z (mm)
        't_s'          : elapsed time (s)
        'pos_err_mm'   : residual tip-position error after Jacobian solve (mm)
    Returns a partial list (possibly empty) if a singularity or bound violation
    is encountered mid-trajectory; the caller should check the length.
    """
    L1 = robot_links["upperarm_forearm"]    # 119.00 mm  (elbow link)
    L2 = robot_links["forearm_wrist"]       # 134.29 mm  (forearm link)
    L3 = robot_links["wrist_knife"]         #  97.00 mm  (wrist link)
    L4 = robot_links["knife_knifetip"]      #  80.00 mm  (knife link, fixed +45°)
    x0 = upper_arm_coordinates[0]
    z0 = upper_arm_coordinates[2]
    d1 = np.deg2rad(upper_arm_elbow_angle)  # 13.81° — elbow joint offset
    d2 = np.deg2rad(elbow_wrist_angle)      #  1.77° — elbow-wrist joint offset

    # ── Horizontal constraint: t1 + t2 + t3 + π/4 = 0  →  t3 = -π/4 - t1 - t2
    # Under this constraint the wrist→tip chain collapses to a fixed offset:
    #   L3 points at angle -π/4  →  (cos=-π/4, sin=-π/4)
    #   L4 points at angle  0    →  (cos=0 = 1, sin=0 = 0)   (horizontal)
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    dx_fixed = L3 * sqrt2_inv + L4     # fixed X offset wrist → tip
    dz_fixed = -L3 * sqrt2_inv         # fixed Z offset wrist → tip

    def _tip_constrained(a1, a2):
        """Knife-tip (x, z) from trigo angles (rad) with horizontal constraint."""
        x_wrist = L1 * np.cos(a1 - d1) + x0 + L2 * np.cos(a1 + a2 + d2)
        z_wrist = L1 * np.sin(a1 - d1) + z0 + L2 * np.sin(a1 + a2 + d2)
        return x_wrist + dx_fixed, z_wrist + dz_fixed + Z_FK_BIAS_MM

    def _jacobian(a1, a2):
        """2×2 Jacobian of (x_tip, z_tip) wrt (a1, a2) under horizontal constraint.

        J = [[ ∂x/∂a1,  ∂x/∂a2 ],
             [ ∂z/∂a1,  ∂z/∂a2 ]]

        The wrist→tip fixed offset contributes zero to the Jacobian, so only
        the L1 and L2 links appear here.
        """
        s1   = np.sin(a1 - d1)
        c1   = np.cos(a1 - d1)
        s12  = np.sin(a1 + a2 + d2)
        c12  = np.cos(a1 + a2 + d2)
        J11  = -L1 * s1  - L2 * s12   # ∂x/∂a1
        J12  = -L2 * s12               # ∂x/∂a2
        J21  =  L1 * c1  + L2 * c12   # ∂z/∂a1
        J22  =  L2 * c12               # ∂z/∂a2
        return np.array([[J11, J12], [J21, J22]])

    # ── Initialise from current joint angles ──────────────────────────────────
    trigo0 = joint_angles_to_trigo(np.array(initial_joint_angles))
    a1 = np.deg2rad(trigo0[0])
    a2 = np.deg2rad(trigo0[1])
    # a3 is derived from constraint — no need to track separately

    # ── Angle bounds ──────────────────────────────────────────────────────────
    waypoints = []
    t_elapsed = 0.0

    while True:
        x_des = x_center_mm + amplitude_mm * np.sin(2 * np.pi * frequency_hz * t_elapsed)
        z_des = z_start_mm - descent_rate_mm_s * t_elapsed

        if z_des <= z_floor_mm:
            print(f"[CUT-JAC] Safety floor z={z_floor_mm:.1f} mm reached at t={t_elapsed:.2f}s — stopping.")
            break
        if z_des <= z_stop_mm:
            print(f"[CUT-JAC] Stop z={z_stop_mm:.1f} mm reached at t={t_elapsed:.2f}s — stopping.")
            break

        # ── Jacobian Newton iterations to reach (x_des, z_des) ───────────────
        a1_new, a2_new = a1, a2
        converged = False
        for _ in range(10):
            x_cur, z_cur = _tip_constrained(a1_new, a2_new)
            err = np.array([x_des - x_cur, z_des - z_cur])
            if np.linalg.norm(err) < 0.01:   # 0.01 mm — converged
                converged = True
                break
            J = _jacobian(a1_new, a2_new)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-4:
                print(f"[CUT-JAC] Singular Jacobian (det={det:.2e}) at t={t_elapsed:.2f}s — stopping.")
                return waypoints
            delta = np.linalg.solve(J, err)
            a1_new = a1_new + delta[0]
            a2_new = a2_new + delta[1]

        if not converged:
            # Accept the solution anyway if the residual is small enough
            x_cur, z_cur = _tip_constrained(a1_new, a2_new)
            if np.linalg.norm([x_des - x_cur, z_des - z_cur]) > 3.0:
                print(f"[CUT-JAC] Did not converge at t={t_elapsed:.2f}s — stopping.")
                return waypoints

        # ── Enforce tilt constraint for a3 ────────────────────────────────────
        a3_new = -np.pi / 4 - a1_new - a2_new

        # ── Bounds check ──────────────────────────────────────────────────────
        if not (TRIGO_A1_MIN <= a1_new <= TRIGO_A1_MAX):
            print(f"[CUT-JAC] a1={np.degrees(a1_new):.1f}° out of bounds at t={t_elapsed:.2f}s — stopping.")
            break
        if not (TRIGO_A2_MIN <= a2_new <= TRIGO_A2_MAX):
            print(f"[CUT-JAC] a2={np.degrees(a2_new):.1f}° out of bounds at t={t_elapsed:.2f}s — stopping.")
            break
        if not (TRIGO_A3_MIN <= a3_new <= TRIGO_A3_MAX):
            print(f"[CUT-JAC] a3={np.degrees(a3_new):.1f}° out of bounds at t={t_elapsed:.2f}s — stopping.")
            break

        a1, a2 = a1_new, a2_new

        # ── Convert back to robot-frame joint angles ───────────────────────────
        a1_deg, a2_deg, a3_deg = np.rad2deg([a1, a2, a3_new])
        joint_angles = trigo_to_joint_angles(np.array([a1_deg, a2_deg, a3_deg]))

        x_achieved, z_achieved = _tip_constrained(a1, a2)
        pos_err = np.sqrt((x_achieved - x_des)**2 + (z_achieved - z_des)**2)

        waypoints.append({
            'joint_angles': joint_angles,   # [shoulder_lift, elbow_flex, wrist_flex] deg
            'x_mm': x_des,
            'z_mm': z_des,
            't_s':  t_elapsed,
            'pos_err_mm': pos_err,
        })

        t_elapsed += dt_s

    return waypoints


def jacobian_cutting_step(current_joint_angles, x_des_mm, z_des_mm,
                          knife_tilt_deg=0.0):
    """Compute the next joint-angle command for one closed-loop cutting step.

    Reads the *actual* current joint angles (from robot observation), computes
    the position error to the desired tip location, then applies one analytical
    Jacobian-inverse step to produce the commanded joint angles.

    The knife tilt is held constant at knife_tilt_deg via the constraint:
        a1 + a2 + a3 + π/4 = β    (β = knife_tilt_deg in radians)
    β = 0   → horizontal knife
    β < 0   → tip pointing down  (e.g. −10° for a slight downward tilt)
    β > 0   → tip pointing up

    The wrist→tip fixed offset under this constraint:
        knife-base direction = β − π/4
        knife-tip  direction = β
        dx_fixed = L3·cos(β−π/4) + L4·cos(β)
        dz_fixed = L3·sin(β−π/4) + L4·sin(β)

    The Jacobian matrix is independent of β (the fixed offset cancels in
    differentiation), so only the constraint enforcement and the fixed offset
    change with knife_tilt_deg.

    Convention note — mykinematics uses:
        x = L * cos(θ),   z = L * sin(θ)
    where θ is measured from the X-axis (horizontal forward) in the trigo
    frame.  This differs from the "sin/cos" notation common in textbooks where
    θ is measured from vertical.

    Jacobian of the constrained 2-DOF system wrt (a1, a2):
        J = [[ -L1·sin(a1-d1) - L2·sin(a1+a2+d2),  -L2·sin(a1+a2+d2) ],
             [  L1·cos(a1-d1) + L2·cos(a1+a2+d2),   L2·cos(a1+a2+d2) ]]
        D = det(J) = L1·L2·sin(a2 + d1 + d2)

    Analytical inverse (one shot, no iteration):
        da1 = (1/D) · [ L2·cos(a1+a2+d2)·dx  +  L2·sin(a1+a2+d2)·dz ]
        da2 = −(1/D) · [ (L1·cos(a1-d1)+L2·cos(a1+a2+d2))·dx
                        +(L1·sin(a1-d1)+L2·sin(a1+a2+d2))·dz ]
        da3 = −da1 − da2

    Parameters
    ----------
    current_joint_angles : array-like [shoulder_lift, elbow_flex, wrist_flex]
        Actual observed joint angles in robot-frame degrees at this timestep.
    x_des_mm, z_des_mm : float
        Desired knife-tip position in robot world frame (mm).
    knife_tilt_deg : float
        Desired knife tilt in degrees. 0 = horizontal, negative = tip down.

    Returns
    -------
    np.ndarray [shoulder_lift, elbow_flex, wrist_flex] in robot-frame degrees,
    or None if the configuration is singular or out of joint bounds.
    """
    L1 = robot_links["upperarm_forearm"]    # 119.00 mm
    L2 = robot_links["forearm_wrist"]       # 134.29 mm
    L3 = robot_links["wrist_knife"]         #  97.00 mm
    L4 = robot_links["knife_knifetip"]      #  80.00 mm
    x0 = upper_arm_coordinates[0]
    z0 = upper_arm_coordinates[2]
    d1 = np.deg2rad(upper_arm_elbow_angle)  # 13.81° — upper-arm/elbow joint offset
    d2 = np.deg2rad(elbow_wrist_angle)      #  1.77° — elbow/wrist joint offset

    # Under the tilt constraint (a1+a2+a3+π/4 = β) the wrist→tip chain is a
    # fixed vector determined by β = knife_tilt_deg.
    beta = np.deg2rad(knife_tilt_deg)
    dx_fixed = L3 * np.cos(beta - np.pi / 4) + L4 * np.cos(beta)
    dz_fixed = L3 * np.sin(beta - np.pi / 4) + L4 * np.sin(beta)

    # ── Convert observed robot-frame angles → trigo-frame radians ─────────────
    trigo = joint_angles_to_trigo(np.array(current_joint_angles))
    a1 = np.deg2rad(trigo[0])
    a2 = np.deg2rad(trigo[1])
    a3 = np.deg2rad(trigo[2])
    # a3 is not needed; it is recomputed from the constraint after the update.

    # ── Current tip position (constrained FK, same formula as forward_kinematics)
    s1  = np.sin(a1 - d1)
    c1  = np.cos(a1 - d1)
    s12 = np.sin(a1 + a2 + d2)
    c12 = np.cos(a1 + a2 + d2)

    x_wrist = L1 * c1 + x0 + L2 * c12
    z_wrist = L1 * s1 + z0 + L2 * s12
    x_cur   = x_wrist + dx_fixed
    z_cur   = z_wrist + dz_fixed + Z_FK_BIAS_MM

    dx = x_des_mm - x_cur
    dz = z_des_mm - z_cur

    # ── Determinant ───────────────────────────────────────────────────────────
    # D = L1·L2·sin(a2 + d1 + d2)
    # Singular when elbow is nearly fully extended (arm and forearm aligned).
    D = L1 * L2 * np.sin(a2 + d1 + d2)
    if abs(D) < 500.0:   # ~2° from singularity; units: mm²
        print(f"[CUT-JAC] Near-singular configuration (D={D:.1f} mm²) — skipping step.")
        return None

    # ── Analytical Jacobian inverse (one shot, no iteration) ──────────────────
    da1 = (L2 * c12 * dx + L2 * s12 * dz) / D
    da2 = -((L1 * c1 + L2 * c12) * dx + (L1 * s1 + L2 * s12) * dz) / D
    # da3 = -(da1 + da2)
     # enforce constraint for a3 update

    # ── Apply delta and enforce tilt constraint ────────────────────────────────
    a1_new = a1 + da1
    a2_new = a2 + da2


    da3 = beta - np.pi / 4 - a1_new - a2_new - a3  # compute da3 from constraint  
    a3_new = a3 + da3 / 4
    # a3_new = beta - np.pi / 4 - a1_new - a2_new   # constraint: a1+a2+a3 = β−π/4

    # ── Joint bounds ──────────────────────────────────────────────────────────
    if not (TRIGO_A1_MIN <= a1_new <= TRIGO_A1_MAX):
        print(f"[CUT-JAC] a1={np.degrees(a1_new):.1f}° out of bounds.")
        return None
    if not (TRIGO_A2_MIN <= a2_new <= TRIGO_A2_MAX):
        print(f"[CUT-JAC] a2={np.degrees(a2_new):.1f}° out of bounds.")
        return None
    if not (TRIGO_A3_MIN <= a3_new <= TRIGO_A3_MAX):
        print(f"[CUT-JAC] a3={np.degrees(a3_new):.1f}° out of bounds.")
        return None

    a1_deg, a2_deg, a3_deg = np.rad2deg([a1_new, a2_new, a3_new])
    return trigo_to_joint_angles(np.array([a1_deg, a2_deg, a3_deg]))


xyz = forward_kinematics([0, 0, 45.77])
xyz = [0.36, 0.0, 0.03]  # Target position in mm
xyz = np.array(xyz) * 1000  # Convert to mm
print(f"End-effector position from FK: X={xyz[0]:.3f} mm, Y={xyz[1]:.3f} mm, Z={xyz[2]:.3f} mm")
joint_angles = inverse_kinematics(xyz)
if joint_angles is not None:
    print(f"Joint angles from IK: Shoulder={joint_angles[0]:.2f}°, Elbow={joint_angles[1]:.2f}°, Wrist={joint_angles[2]:.2f}°")

