"""
Step 6: Real Field Gravity Inversion (Dizaj Karst Cavity, Naghadeh)
===================================================================
Applies the Harris Hawks Optimization (HHO) algorithm to real field gravity
profile data (Profiles_2_Naghade.xlsx).

Methodological Highlights:
  - Automatic detection of data file inside 'RealData/', 'data/', or root
  - First-Order Trend Surface Correction (Variable Projection / Golub-Pereyra)
  - Simultaneous estimation of buried karst cavity parameters [x0, z, R, drho]
  - Publication-ready Figure 7 generation (3-Panel Conference Standard)
  - Outputs saved directly to 'conf_figures/' folder with comprehensive terminal logging.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try importing project modules
try:
    from src.hho_optimizer import HarrisHawksOptimizer
    from src.metrics import compute_rmse
except ModuleNotFoundError:
    print("[ERROR] Please run this script from the root directory of 'gravity-hho-inversion'!")
    sys.exit(1)

# -------------------------------------------------------------------------
# Logging Helper Function
# -------------------------------------------------------------------------
def log(msg, tag="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag:<5}] {msg}", flush=True)

# -------------------------------------------------------------------------
# 1. Physics Forward Model for Microgravity Cavity (Buried Sphere)
# -------------------------------------------------------------------------
G_SI = 6.67430e-11  # m^3 / (kg * s^2)

def sphere_gravity_real(x, x0, z, R, drho):
    """
    Computes vertical gravity anomaly of a spherical cavity in mGal.
    x0, z, R in meters; drho in kg/m^3.
    """
    mass_contrast = (4.0 / 3.0) * np.pi * (R ** 3) * drho  # kg
    gz_si = G_SI * mass_contrast * z / ((x - x0) ** 2 + z ** 2) ** 1.5
    return gz_si * 1e5  # convert to mGal


def run_real_inversion():
    total_start = time.perf_counter()
    log("Starting Step 6: Real Field Gravity Inversion Routine...")

    # Base & Output Directory Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "conf_figures")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output directory verified: '{out_dir}'")

    # Dynamic Data File Locator (Checks RealData, data, and root)
    potential_paths = [
        os.path.join(base_dir, "RealData", "Profiles_2_Naghade.xlsx"),
        os.path.join(base_dir, "RealData", "Profiles_2_Naghadeh.xlsx"),
        os.path.join(base_dir, "data", "Profiles_2_Naghade.xlsx"),
        os.path.join(base_dir, "Profiles_2_Naghade.xlsx"),
        os.path.join(os.getcwd(), "RealData", "Profiles_2_Naghade.xlsx"),
        os.path.join(os.getcwd(), "data", "Profiles_2_Naghade.xlsx"),
        os.path.join(os.getcwd(), "Profiles_2_Naghade.xlsx"),
    ]
    data_path = next((p for p in potential_paths if os.path.isfile(p)), None)

    # Recursive fallback if explicit paths fail
    if not data_path:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if "naghade" in f.lower() and f.endswith(".xlsx"):
                    data_path = os.path.join(root, f)
                    break
            if data_path:
                break

    if not data_path:
        log("Could not locate 'Profiles_2_Naghade.xlsx' in RealData/, data/, or root directory!", "ERROR")
        raise FileNotFoundError("Profiles_2_Naghade.xlsx missing.")

    log(f"Loaded field dataset from: '{data_path}'")

    # Ingest Profile Observations
    df = pd.read_excel(data_path, header=None)
    x = df.iloc[:, 0].values.astype(float)
    g_obs = df.iloc[:, 1].values.astype(float)
    log(f"Data ingested: {len(x)} observation stations along profile [0.0 -> {x.max():.2f} m]")
    log(f"Anomaly amplitude span: Min = {g_obs.min():.4f} mGal, Max = {g_obs.max():.4f} mGal")

    # ---------------------------------------------------------------------
    # 2. Variable Projection (Golub-Pereyra) for Background Trend
    # ---------------------------------------------------------------------
    A_matrix = np.column_stack([np.ones_like(x), x])
    log("Constructed Vandermonde matrix for 1st-order regional trend correction.")

    eval_counter = [0]
    def cost_fn(m):
        eval_counter[0] += 1
        x0, z, R, drho = m
        g_model = sphere_gravity_real(x, x0, z, R, drho)
        # Solve least-squares for linear regional background [c0, c1]
        coef, _, _, _ = np.linalg.lstsq(A_matrix, g_obs - g_model, rcond=None)
        g_total = g_model + A_matrix @ coef
        return compute_rmse(g_obs, g_total)

    # Search Bounds: [x0, z, R, drho]
       # به جای -3000 تا -200:
    lb = np.array([5.0,  1.0,  0.5, -2400.0])
    ub = np.array([20.0, 12.0, 10.0,  -800.0])

    log(f"Search bounds configured:")
    log(f"  x0 (Center)     : [{lb[0]:.1f}, {ub[0]:.1f}] m")
    log(f"  z0 (Depth)      : [{lb[1]:.1f}, {ub[1]:.1f}] m")
    log(f"  R  (Radius)     : [{lb[2]:.1f}, {ub[2]:.1f}] m")
    log(f"  drho (Contrast) : [{lb[3]:.1f}, {ub[3]:.1f}] kg/m^3")

    # ---------------------------------------------------------------------
    # 3. Optimization via Harris Hawks Optimization (HHO)
    # ---------------------------------------------------------------------
    n_agents = 30
    max_iter = 500
    log(f"Initializing Harris Hawks Optimizer (Agents={n_agents}, MaxIter={max_iter}, Seed=42)...")
    
    hho = HarrisHawksOptimizer(
        cost_func=cost_fn,
        lb=lb,
        ub=ub,
        dim=4,
        search_agents_no=n_agents,
        max_iter=max_iter,
        random_seed=42
    )

    log("Commencing inversion optimization iterations...")
    opt_start = time.perf_counter()
    best_m, best_rmse, conv_curve = hho.optimize()
    opt_duration = time.perf_counter() - opt_start

    log(f"Optimization completed in {opt_duration:.2f} seconds ({eval_counter[0]} evaluations).", "DONE")
    log(f"Best RMSE Achieved: {best_rmse:.5f} mGal")

    # Model parameters reconstruction
    x0_opt, z_opt, R_opt, drho_opt = best_m
    z_top_opt = z_opt - R_opt
    g_sphere_opt = sphere_gravity_real(x, x0_opt, z_opt, R_opt, drho_opt)
    trend_coef, _, _, _ = np.linalg.lstsq(A_matrix, g_obs - g_sphere_opt, rcond=None)
    g_trend = A_matrix @ trend_coef
    g_pred = g_sphere_opt + g_trend
    residuals = g_obs - g_pred
    r2_score = 1.0 - (np.sum(residuals ** 2) / np.sum((g_obs - np.mean(g_obs)) ** 2))

    print("\n" + "=" * 72)
    print("           OPTIMAL PARAMETERS INVERTED BY HHO (TABLE 4)")
    print("=" * 72)
    print(f"  {'Parameter':<28} | {'Value':<12} | {'Unit'}")
    print("  " + "-" * 68)
    print(f"  {'Cavity Center (x0)':<28} | {x0_opt:<12.2f} | m")
    print(f"  {'Cavity Depth (z0)':<28} | {z_opt:<12.2f} | m")
    print(f"  {'Cavity Radius (R)':<28} | {R_opt:<12.2f} | m")
    print(f"  {'Density Contrast (drho)':<28} | {drho_opt:<12.2f} | kg/m^3")
    print(f"  {'Cavity Roof Depth (z_top)':<28} | {z_top_opt:<12.2f} | m")
    print(f"  {'Inversion RMSE':<28} | {best_rmse:<12.4f} | mGal")
    print(f"  {'Coefficient of Det. (R^2)':<28} | {r2_score:<12.4f} | -")
    print(f"  {'Regional Trend c0':<28} | {trend_coef[0]:<12.6f} | mGal")
    print(f"  {'Regional Trend c1':<28} | {trend_coef[1]:<12.6e} | mGal/m")
    print("=" * 72 + "\n")

    # ---------------------------------------------------------------------
    # 4. Save Table 4 Summary Report
    # ---------------------------------------------------------------------
    report_file = os.path.join(out_dir, "real_data_hho_results.txt")
    log(f"Writing numerical inversion summary to '{report_file}'...")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("HHO INVERSION RESULTS — DIZAJ REAL GRAVITY PROFILE 2\n")
        f.write("=" * 60 + "\n")
        f.write(f"x0 (m)           : {x0_opt:.4f}\n")
        f.write(f"z0 (m)           : {z_opt:.4f}\n")
        f.write(f"R (m)            : {R_opt:.4f}\n")
        f.write(f"drho (kg/m3)     : {drho_opt:.2f}\n")
        f.write(f"z_top (m)        : {z_top_opt:.4f}\n")
        f.write(f"RMSE (mGal)      : {best_rmse:.4f}\n")
        f.write(f"R^2              : {r2_score:.4f}\n")
        f.write(f"Runtime (s)      : {opt_duration:.2f}\n")
        f.write(f"Regional Trend   : c0 = {trend_coef[0]:.6f}, c1 = {trend_coef[1]:.6e}\n")
    log("Report saved successfully.")

    # ---------------------------------------------------------------------
    # 5. Generate Figure 7 (3-Panel Publication Format)
    # ---------------------------------------------------------------------
    log("Generating 3-Panel publication Figure 7...")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8.5, 10.5), dpi=300,
        gridspec_kw={"height_ratios": [2.2, 1.2, 2.0]}
    )

    # Panel (a): Gravity Anomaly Fit & Trend
    ax1.scatter(x, g_obs, color="black", s=18, facecolors="none", edgecolors="black", linewidth=1.0, label="Observed Field Data (CG-5)")
    ax1.plot(x, g_pred, color="crimson", linewidth=2.0, label="HHO Forward Response + Regional Trend")
    ax1.plot(x, g_trend, color="dimgray", linestyle="--", linewidth=1.2, label=f"Regional Trend ($c_0 + c_1 x$)")
    ax1.set_ylabel("Gravity Anomaly (mGal)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Field Gravity Anomaly and Inversion Model Fit", fontsize=11, fontweight="bold", pad=8)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel (b): Inversion Residuals
    ax2.plot(x, residuals, color="black", marker="o", markersize=3, linestyle="-", linewidth=0.8, alpha=0.85)
    ax2.axhline(0, color="crimson", linestyle="--", linewidth=1.0)
    std_res = np.std(residuals)
    ax2.axhline(std_res, color="gray", linestyle=":", linewidth=0.8, label=f"$\pm 1\sigma$ ({std_res:.4f} mGal)")
    ax2.axhline(-std_res, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Residual (mGal)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Profile Distance (m)", fontsize=10, fontweight="bold")
    ax2.set_title(f"(b) Inversion Residuals (Final RMSE = {best_rmse:.4f} mGal)", fontsize=11, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel (c): Subsurface Geological Section
    ax3.axhspan(-1.5, 0, color="#dff0fc", alpha=0.6)  # Surface/Atmosphere
    ax3.axhline(0, color="black", linewidth=1.8)       # Ground line
    cavity_patch = plt.Circle(
        (x0_opt, z_opt), R_opt,
        facecolor="#f3d3b3", edgecolor="#8b3a0f",
        linewidth=1.8, hatch="//", label="Karst Cavity Target"
    )
    ax3.add_patch(cavity_patch)
    ax3.plot([x0_opt, x0_opt], [0, z_opt], color="black", linestyle="--", linewidth=0.9)
    ax3.plot(x0_opt, z_opt, marker="x", color="black", markersize=7, markeredgewidth=1.5)

    text_info = (
        f"$x_0 = {x0_opt:.2f}$ m\n"
        f"$z_0 = {z_opt:.2f}$ m\n"
        f"$R = {R_opt:.2f}$ m\n"
        f"$z_{{top}} = {z_top_opt:.2f}$ m\n"
        f"$\\Delta\\rho = {drho_opt:.1f}$ kg/m$^3$"
    )
    ax3.text(
        1.2, 7.5, text_info, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    ax3.set_xlim(0, x.max())
    ax3.set_ylim(9.0, -1.0)  # Inverted depth axis
    ax3.set_xlabel("Profile Distance (m)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Depth (m)", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Inverted Subsurface Cavity Geometry", fontsize=11, fontweight="bold", pad=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Saving Figures to conf_figures/
    fig7_png = os.path.join(out_dir, "fig7_real_data_hho.png")
    fig7_pdf = os.path.join(out_dir, "fig7_real_data_hho.pdf")

    log(f"Writing raster figure to: '{fig7_png}'...")
    plt.savefig(fig7_png, dpi=300, bbox_inches="tight")
    
    log(f"Writing vector figure to: '{fig7_pdf}'...")
    plt.savefig(fig7_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    # Verification
    if os.path.exists(fig7_png) and os.path.getsize(fig7_png) > 0:
        log(f"Figure PNG confirmed ({os.path.getsize(fig7_png) / 1024:.1f} KB)", "SUCCESS")
    else:
        log("Figure PNG write failed!", "ERROR")

    if os.path.exists(fig7_pdf) and os.path.getsize(fig7_pdf) > 0:
        log(f"Figure PDF confirmed ({os.path.getsize(fig7_pdf) / 1024:.1f} KB)", "SUCCESS")
    else:
        log("Figure PDF write failed!", "ERROR")

    log(f"Entire routine finished in {time.perf_counter() - total_start:.2f} seconds.", "SUCCESS")


if __name__ == "__main__":
    run_real_inversion()
