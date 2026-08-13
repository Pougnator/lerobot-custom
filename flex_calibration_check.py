#!/usr/bin/env python3
"""flex_calibration_check.py — does the torsional-spring flex relation hold?

Tests, on recorded cutting parquets, the per-joint flex model from the sim plan:

        slip_j = (1/k_j) * (load_j - tau_gravity_j)          (the relation to check)
    <=> load_j = k_j * slip_j + tau_gravity_j                (linear, what we fit)

where
    slip_j         = q_commanded - q_actual   (encoder deflection, obs idx 10..12)
    load_j         = Present_Load proxy        (motor torque,        obs idx 5..7)
    tau_gravity_j  = Y_j(q) @ p                 (mykinematics.gravity_regressor)

The 4 lumped gravity params `p` are NOT assumed known: we fit them jointly with
the 3 joint stiffnesses `k` (and optional per-joint backlash offsets `b`) in a
single linear least-squares over all samples and all 3 joints. If the relation
holds, the fit explains `slip` with high R^2 and structureless residuals.

For comparison it also fits the two naive per-joint baselines:
    (A) slip ~ load              (no gravity term)
    (B) slip ~ (load - grav)     (gravity subtracted, the literal requested form)

Usage: edit the settings block at the bottom (under `if __name__ == "__main__":`)
— PATHS (parquet files/dirs), STATIC_MODE (cork-press Kp_eff fit), etc. — then run:
    python flex_calibration_check.py

Outputs: prints a per-joint report and saves diagnostic PNGs next to this script.
"""
from __future__ import annotations

import sys
import glob
import os

import numpy as np

# --------------------------------------------------------------------------- #
# Config — adjust to your machine                                             #
# --------------------------------------------------------------------------- #

# Where the recorded parquets live (Razer). Override via PATHS in __main__.
PARQUET_DIR = r"C:\Repos_Razor\3D-vision\cut_datasets"

# Directory containing mykinematics.py (for gravity_regressor). This file ships
# next to it, so the default just adds its own folder.
MYKINEMATICS_DIR = os.path.dirname(os.path.abspath(__file__))

# Fitted gravity model. Two formats are accepted:
#   * 3-param  {"p1","p2","p3"}        — recompute_offline_rewards.py model
#                                        (shoulder+elbow only, wrist gravity = 0)
#   * 4-param  {"gravity_params":[..]} — sensor_hub.py / gravity_regressor model
# Set to None to fit the gravity params from the data instead.
GRAVITY_MODEL_PATH = r"C:\Repos_Razor\3D-vision\outputs\hilserl\gravity_model.json"

# Present_Load over-reads torque by this much per deg/s of joint speed (back-EMF).
# STS3215 @12V: stall 1000 PL-units, no-load 270 deg/s  ->  1000/270.
BACK_EMF_PER_DEG_S = 1000.0 / 270.0

# cutting_obs.py 13D layout (source of truth). Used when the parquet stores the
# observation as a single array column instead of named scalar columns.
OBS_IDX = {
    "shoulder_lift": 0, "elbow_flex": 1, "wrist_flex": 2,
    "imu_tilt": 3, "fsr": 4,
    "load": (5, 6, 7),          # current_motor_1/2/3  == Present_Load proxy
    "slip": (10, 11, 12),       # slip_j1/j2/j3
}

# Candidate names for an array-valued observation column (LeRobot style).
STATE_COL_CANDIDATES = ["observation.state", "obs", "state", "observation"]

# Candidate names for explicit per-joint scalar columns (custom recorder style).
SCALAR_CANDIDATES = {
    "q":    [["shoulder_lift", "elbow_flex", "wrist_flex"]],
    "load": [["current_motor_1", "current_motor_2", "current_motor_3"],
             ["load_j1", "load_j2", "load_j3"]],
    "slip": [["slip_j1", "slip_j2", "slip_j3"],
             ["slip_1", "slip_2", "slip_3"]],
    "fsr":  [["fsr_value"], ["fsr"]],
}

JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]

# --------------------------------------------------------------------------- #


def _import_gravity_regressor():
    if MYKINEMATICS_DIR not in sys.path:
        sys.path.insert(0, MYKINEMATICS_DIR)
    try:
        from mykinematics import gravity_regressor
    except Exception as e:  # pragma: no cover
        sys.exit(f"ERROR: could not import gravity_regressor from "
                 f"{MYKINEMATICS_DIR!r}: {e}")
    return gravity_regressor


def _collect_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)
        elif os.path.isfile(p):
            files.append(p)
        else:
            files += glob.glob(p, recursive=True)
    return sorted(set(files))


def _first_present(df_cols, candidate_groups):
    """Return the first candidate name-group fully present in df_cols, else None."""
    cols = set(df_cols)
    for group in candidate_groups:
        if all(c in cols for c in group):
            return group
    return None


def _stack_array_col(series):
    """Turn a parquet array/list column into an (N, D) float array."""
    return np.array([np.asarray(v, dtype=float).ravel() for v in series.values])


def _extract(df):
    """Return q, load, slip, fsr, ts for one episode DataFrame (auto-schema)."""
    q_cols    = _first_present(df.columns, SCALAR_CANDIDATES["q"])
    load_cols = _first_present(df.columns, SCALAR_CANDIDATES["load"])
    slip_cols = _first_present(df.columns, SCALAR_CANDIDATES["slip"])

    if q_cols and load_cols and slip_cols:
        q    = df[list(q_cols)].to_numpy(float)
        load = df[list(load_cols)].to_numpy(float)
        slip = df[list(slip_cols)].to_numpy(float)
        fsr_cols = _first_present(df.columns, SCALAR_CANDIDATES["fsr"])
        fsr = df[fsr_cols[0]].to_numpy(float) if fsr_cols else np.full(len(df), np.nan)
    else:
        state_col = next((c for c in STATE_COL_CANDIDATES if c in df.columns), None)
        if state_col is None:
            sys.exit("ERROR: could not find joint/load/slip columns nor an "
                     f"array observation column.\nColumns: {list(df.columns)}\n"
                     "Edit SCALAR_CANDIDATES / STATE_COL_CANDIDATES / OBS_IDX.")
        state = _stack_array_col(df[state_col])
        if state.shape[1] < 13:
            sys.exit(f"ERROR: observation column {state_col!r} width {state.shape[1]} (<13).")
        q    = state[:, [OBS_IDX["shoulder_lift"], OBS_IDX["elbow_flex"], OBS_IDX["wrist_flex"]]]
        load = state[:, list(OBS_IDX["load"])]
        slip = state[:, list(OBS_IDX["slip"])]
        fsr  = state[:, OBS_IDX["fsr"]]

    ts = df["timestamp"].to_numpy(float) if "timestamp" in df.columns else None
    return q, load, slip, fsr, ts


def load_episodes(paths):
    """Return q (N,3), load (N,3), slip (N,3), fsr (N,), vel (N,3).

    `vel` = actual-joint velocity (deg/s), differenced PER EPISODE so episode
    boundaries don't create spurious spikes. Used to separate elastic flex from
    servo tracking lag.
    """
    import pandas as pd

    files = _collect_files(paths)
    if not files:
        sys.exit(f"ERROR: no parquet files found under {paths}")

    print(f"Found {len(files)} parquet file(s).")
    q_all, load_all, slip_all, fsr_all, vel_all = [], [], [], [], []

    printed_schema = False
    for f in files:
        df = pd.read_parquet(f)
        if not printed_schema:
            print(f"  Columns in {os.path.basename(f)}: {list(df.columns)}")
            printed_schema = True
        q, load, slip, fsr, ts = _extract(df)

        # per-episode joint velocity
        if ts is not None and len(ts) > 2:
            dt = np.gradient(ts); dt[dt <= 0] = np.median(dt[dt > 0]) if (dt > 0).any() else 0.1
            vel = np.gradient(q, axis=0) / dt[:, None]
        else:
            vel = np.full_like(q, np.nan)

        q_all.append(q); load_all.append(load); slip_all.append(slip)
        fsr_all.append(fsr); vel_all.append(vel)

    q    = np.vstack(q_all);  load = np.vstack(load_all)
    slip = np.vstack(slip_all); fsr = np.concatenate(fsr_all)
    vel  = np.vstack(vel_all)

    ok = np.isfinite(q).all(1) & np.isfinite(load).all(1) & np.isfinite(slip).all(1)
    print(f"Loaded {len(q)} rows; {ok.sum()} usable after NaN filter.")
    return q[ok], load[ok], slip[ok], fsr[ok], vel[ok]


def build_gravity(q, gravity_regressor):
    """Return Y of shape (N, 3, 4): per-sample gravity regressor."""
    return np.stack([gravity_regressor(qi) for qi in q], axis=0)


def load_gravity_model(path):
    """Return (grav_fn, label) or (None, None) if no usable model file.

    grav_fn(q (N,3)) -> tau_gravity (N,3) in Present_Load units.
    """
    if not path or not os.path.isfile(path):
        return None, None
    import json
    d = json.loads(open(path).read())

    if all(k in d for k in ("p1", "p2", "p3")):           # 3-param reduced model
        p1, p2, p3 = float(d["p1"]), float(d["p2"]), float(d["p3"])

        def grav_fn(q):
            a1 = np.radians(90.0 - q[:, 0])
            a2 = np.radians(-q[:, 1] - 90.0)
            a3 = np.radians(-q[:, 2])
            c12 = np.cos(a1 + a2); c123 = np.cos(a1 + a2 + a3)
            tau1 = p1 * np.cos(a1) + p2 * c12 + p3 * c123
            tau2 = p2 * c12 + p3 * c123
            return np.column_stack([tau1, tau2, np.zeros_like(tau1)])  # wrist=0

        return grav_fn, f"3-param (p1={p1:.1f}, p2={p2:.1f}, p3={p3:.1f}), wrist gravity=0"

    if "gravity_params" in d:                              # 4-param regressor model
        gr = _import_gravity_regressor()
        p = np.asarray(d["gravity_params"], float)

        def grav_fn(q):
            return np.stack([gr(qi) @ p for qi in q], axis=0)

        return grav_fn, f"4-param gravity_regressor params={np.round(p,1)}"

    return None, None


def joint_fit(q, load, slip, Y, backlash=True):
    """Joint least-squares: load_j = k_j * slip_j + Y_j @ p [+ b_j].

    Unknowns: [k1,k2,k3, p1..p4 (shared), (b1,b2,b3 if backlash)].
    Returns dict with k, p, b, and predicted load/slip + R^2 per joint.
    """
    N = len(q)
    n_b = 3 if backlash else 0
    n_unknown = 3 + 4 + n_b

    A = np.zeros((N * 3, n_unknown))
    y = np.zeros(N * 3)
    for j in range(3):
        rows = slice(j * N, (j + 1) * N)
        A[rows, j] = slip[:, j]          # k_j coefficient
        A[rows, 3:7] = Y[:, j, :]        # shared gravity params p
        if backlash:
            A[rows, 7 + j] = 1.0         # b_j offset
        y[rows] = load[:, j]

    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    k = theta[0:3]
    p = theta[3:7]
    b = theta[7:10] if backlash else np.zeros(3)

    # gravity torque and predictions
    tau_grav = np.einsum("njk,k->nj", Y, p)              # (N,3)
    load_pred = k * slip + tau_grav + b                  # predict load
    # invert to predict slip (the quantity the relation is about)
    slip_pred = (load - tau_grav - b) / k

    return dict(k=k, p=p, b=b, tau_grav=tau_grav,
                load_pred=load_pred, slip_pred=slip_pred)


def r2(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def ols(x, y):
    """Return slope, intercept, R^2 of simple y = slope*x + intercept."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    return slope, intercept, r2(y, slope * x + intercept)


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def _mlr(X, y):
    """Multiple linear regression y = X @ c (+ intercept col assumed in X). Returns c, R2."""
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2v = 1.0 - np.sum((y - pred) ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return c, r2v


def lag_vs_flex_diagnostic(slip, load, vel):
    """Separate servo tracking lag (~velocity) from elastic flex (~torque).

    For each joint fits   slip = a*omega + (1/k)*load + b   and compares against
    the velocity-only model. Elastic flex is real only if the load term carries
    extra explanatory power (positive 1/k, meaningful dR2). Returns True iff a
    usable flex signal is found on at least one joint.
    """
    print("\n" + "=" * 78)
    print("DIAGNOSTIC: is `slip` elastic flex (~load) or servo tracking lag (~velocity)?")
    print("=" * 78)
    have_vel = np.isfinite(vel).all()
    if not have_vel:
        print("  No timestamp/velocity available -> cannot separate. Inconclusive.")
        print("=" * 78)
        return True

    print("Per joint: simple correlations, then partial load effect after removing velocity.")
    print(f"{'joint':<14}{'r(slip,vel)':>12}{'r(slip,load)':>13}"
          f"{'1/k|vel':>10}{'k=1/.':>9}{'R2_vel':>8}{'R2_+load':>9}{'dR2_load':>9}")
    flex_ok = False
    for j, name in enumerate(JOINTS):
        o, l, y = vel[:, j], load[:, j], slip[:, j]
        r_vel  = _pearson(y, o)
        r_load = _pearson(y, l)
        # velocity-only vs velocity+load
        _, r2_v  = _mlr(np.column_stack([o, np.ones_like(o)]), y)
        c2, r2_vl = _mlr(np.column_stack([o, l, np.ones_like(o)]), y)
        kinv = c2[1]
        k = (1.0 / kinv) if abs(kinv) > 1e-9 else float("inf")
        dr2 = r2_vl - r2_v
        # a usable flex signal: positive stiffness AND a non-trivial dR2
        if kinv > 0 and dr2 > 0.05:
            flex_ok = True
        print(f"{name:<14}{r_vel:>12.3f}{r_load:>13.3f}{kinv:>10.5f}{k:>9.1f}"
              f"{r2_v:>8.3f}{r2_vl:>9.3f}{dr2:>9.4f}")

    print("\nVerdict:")
    if flex_ok:
        print("  A load term adds real explanatory power with POSITIVE stiffness ->")
        print("  an elastic-flex signal is present; the calibration below is meaningful.")
    else:
        print("  `slip` is dominated by SERVO TRACKING LAG (slip ~ joint velocity).")
        print("  After removing velocity, the load term is negligible / negative-sign,")
        print("  i.e. NO usable elastic-flex signal. The torsional-spring stiffness")
        print("  CANNOT be calibrated from these motion parquets. Collect a STATIC")
        print("  loading test instead:")
        print("    1. command a fixed pose; let joint velocity settle to ~0,")
        print("    2. apply known static tip loads (hang weights / press a scale),")
        print("    3. regress the settled slip vs load -> k_j.")
        print("  (Root causes here: arm is always moving, and FSR shows no board")
        print("   contact in any frame -> zero statically-loaded samples.)")
    print("=" * 78)
    return flex_ok


def fit_static_stiffness(slip, load, vel, vel_thresh=8.0):
    """Static-press calibration: Kp_eff[j] = slope of load vs slip on SETTLED rows.

    At quasi-static equilibrium the servo holds  slip = load / Kp_eff, so a per-
    joint regression load ~ Kp_eff*slip + b gives the effective joint stiffness
    directly (b = backlash/deadband offset). Requires settled samples spanning a
    range of load, so press the knife with VARYING force through the episode.
    """
    print("\n" + "=" * 78)
    print("STATIC-PRESS STIFFNESS   load_j = Kp_eff_j * slip_j + b_j   (settled rows)")
    print("=" * 78)

    have_vel = np.isfinite(vel).all()
    if have_vel:
        settled = np.abs(vel).max(1) < vel_thresh
    else:
        settled = np.ones(len(slip), bool)
        print("  (no velocity available -> using all rows; ensure they are settled)")
    n = int(settled.sum())
    print(f"Settled rows (|vel|<{vel_thresh} deg/s): {n}/{len(slip)}")
    if n < 15:
        print("  Too few settled rows to fit. Record a slower, settled press.")
        print("=" * 78)
        return None

    print(f"{'joint':<14}{'Kp_eff(load/deg)':>18}{'backlash b':>12}"
          f"{'R2':>8}{'slip range':>16}{'load range':>16}")
    kp = {}
    for j, name in enumerate(JOINTS):
        s, l = slip[settled, j], load[settled, j]
        slope, intercept, r2v = ols(s, l)          # load = slope*slip + b
        kp[name] = slope
        s_rng = f"[{s.min():.2f},{s.max():.2f}]"
        l_rng = f"[{l.min():.0f},{l.max():.0f}]"
        warn = "" if (s.max() - s.min()) > 0.5 and r2v > 0.5 else "  <-- weak/no spread"
        print(f"{name:<14}{slope:>18.2f}{intercept:>12.2f}{r2v:>8.3f}"
              f"{s_rng:>16}{l_rng:>16}{warn}")

    print("\n  Kp_eff = load-units per degree of droop. Use p+ = load/Kp_eff to")
    print("  cancel the servo droop. Low R2 or tiny slip range => vary the press")
    print("  force more (press progressively harder) and keep the arm settled.")
    print("=" * 78)
    return kp


def report(q, load, slip, fit, tau_grav_model=None):
    grav_for_baseline = tau_grav_model if tau_grav_model is not None else fit["tau_grav"]
    print("\n" + "=" * 70)
    print("JOINT FLEX FIT   load_j = k_j * slip_j + Y_j @ p  [+ b_j]")
    print("=" * 70)
    print(f"Shared gravity params p = {np.array2string(fit['p'], precision=4)}")
    print(f"{'joint':<14}{'k (load/deg)':>14}{'compliance 1/k':>16}"
          f"{'backlash b':>12}{'R2(slip)':>10}{'R2(load)':>10}")
    for j, name in enumerate(JOINTS):
        r2_slip = r2(slip[:, j], fit["slip_pred"][:, j])
        r2_load = r2(load[:, j], fit["load_pred"][:, j])
        k = fit["k"][j]
        comp = 1.0 / k if k != 0 else float("nan")
        print(f"{name:<14}{k:>14.4f}{comp:>16.5f}{fit['b'][j]:>12.3f}"
              f"{r2_slip:>10.3f}{r2_load:>10.3f}")

    src = "loaded model" if tau_grav_model is not None else "fitted p"
    print(f"\nBaselines per joint (sanity / model-form check; gravity from {src}):")
    print(f"{'joint':<14}{'(A) slip~load R2':>18}{'(B) slip~(load-grav) R2':>26}")
    for j, name in enumerate(JOINTS):
        _, _, r2A = ols(load[:, j], slip[:, j])
        _, _, r2B = ols(load[:, j] - grav_for_baseline[:, j], slip[:, j])
        print(f"{name:<14}{r2A:>18.3f}{r2B:>26.3f}")

    print("\nInterpretation:")
    print("  * High R2(slip) (> ~0.7) and small backlash  => linear torsional")
    print("    spring holds; use 1/k as the DR center for that joint.")
    print("  * (B) >> (A)  => gravity term matters (relation needs it).")
    print("  * Low R2 everywhere or fan-shaped residuals => wrong model form")
    print("    (suspect backlash/hysteresis); inspect the residual plots.")
    print("=" * 70)


def make_plots(load, slip, fit, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping plots — matplotlib unavailable: {e})")
        return

    # effective torque the spring should see, per the fitted model
    eff_torque = load - fit["tau_grav"] - fit["b"]      # = load - grav - backlash

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, name in enumerate(JOINTS):
        # row 0: slip vs (load - grav), with fitted line slip = (1/k)*eff
        ax = axes[0, j]
        ax.scatter(eff_torque[:, j], slip[:, j], s=4, alpha=0.3)
        xs = np.linspace(eff_torque[:, j].min(), eff_torque[:, j].max(), 100)
        ax.plot(xs, xs / fit["k"][j], "r-", lw=2, label=f"1/k = {1/fit['k'][j]:.4f}")
        ax.set_title(f"{name}: slip vs (load - grav)")
        ax.set_xlabel("load - gravity torque"); ax.set_ylabel("slip (deg)")
        ax.legend(); ax.grid(alpha=0.3)

        # row 1: residuals of slip prediction vs fitted slip
        ax = axes[1, j]
        resid = slip[:, j] - fit["slip_pred"][:, j]
        ax.scatter(fit["slip_pred"][:, j], resid, s=4, alpha=0.3)
        ax.axhline(0, color="r", lw=1)
        ax.set_title(f"{name}: slip residuals")
        ax.set_xlabel("predicted slip (deg)"); ax.set_ylabel("residual (deg)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(outdir, "flex_calibration_check.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved diagnostic plot -> {out}")


def main(paths, static=False, no_backlash=False, contact_only=False,
         fsr_thresh=900.0, vel_thresh=8.0):
    gravity_regressor = _import_gravity_regressor()
    q, load, slip, fsr, vel = load_episodes(paths)

    if contact_only:
        m = np.isfinite(fsr) & (fsr < fsr_thresh)
        print(f"contact-only: keeping {m.sum()}/{len(q)} rows (fsr < {fsr_thresh}).")
        if m.sum() < 20:
            sys.exit(f"ERROR: only {m.sum()} contact rows (fsr<{fsr_thresh}); "
                     "this dataset has no statically-loaded contact data.")
        q, load, slip, fsr, vel = q[m], load[m], slip[m], fsr[m], vel[m]

    if len(q) < 20:
        sys.exit("ERROR: too few samples to fit.")

    # Real fitted gravity model, if available.
    grav_fn, grav_label = load_gravity_model(GRAVITY_MODEL_PATH)
    if grav_fn is not None:
        print(f"Gravity model: {grav_label}")
        tau_grav_model = grav_fn(q)
    else:
        print("Gravity model: none loaded -> gravity params will be fit from data.")
        tau_grav_model = None

    # First decide whether `slip` even carries an elastic-flex signal.
    flex_ok = lag_vs_flex_diagnostic(slip, load, vel)

    if static:
        # Cork-press calibration: this is the intended path for Kp_eff.
        fit_static_stiffness(slip, load, vel, vel_thresh=vel_thresh)
        return

    Y = build_gravity(q, gravity_regressor)
    fit = joint_fit(q, load, slip, Y, backlash=not no_backlash)
    report(q, load, slip, fit, tau_grav_model=tau_grav_model)
    if not flex_ok:
        print("\n*** WARNING: diagnostic says slip is lag-dominated -- the fit "
              "above is NOT a valid flex calibration. See verdict. ***")
    make_plots(load, slip, fit, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    # ── settings — edit here ────────────────────────────────────────────────
    PATHS        = [PARQUET_DIR]  # parquet files or directories (recursed)
    STATIC_MODE  = False          # True -> cork-press Kp_eff fit (settled load-vs-slip)
    NO_BACKLASH  = False          # True -> fit without per-joint backlash offset b_j
    CONTACT_ONLY = False          # True -> keep only rows in board contact
    FSR_THRESH   = 900.0          # FSR contact threshold (ADC)
    VEL_THRESH   = 8.0            # max joint speed (deg/s) counted as settled (static mode)
    # ────────────────────────────────────────────────────────────────────────
    main(PATHS, static=STATIC_MODE, no_backlash=NO_BACKLASH,
         contact_only=CONTACT_ONLY, fsr_thresh=FSR_THRESH, vel_thresh=VEL_THRESH)
