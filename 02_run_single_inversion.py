"""
Step 2: Single Run Inversion & Performance Visualization
Loads data from Excel, executes HHO for clean, 5%, and 10% noisy anomalies,
prints the summary table, and plots Figures 2 (Fit & Residuals) and Figure 3 (Convergence).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forward_model import sphere_gravity_forward
from src.hho_optimizer import HarrisHawksOptimizer
from src.metrics import compute_mae, compute_relative_error, compute_rmse


def run_single_inversion():
    excel_path = os.path.join("data", "synthetic_profiles.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"[-] Could not find '{excel_path}'. Please run 01_generate_synthetic_data.py first!"
        )

    # 1. Read Frozen Profiles
    df_profiles = pd.read_excel(excel_path, sheet_name="Profiles")
    x = df_profiles["x (m)"].values
    g_clean = df_profiles["g_clean (mGal)"].values
    g_noisy_5 = df_profiles["g_noisy_5pct (mGal)"].values
    g_noisy_10 = df_profiles["g_noisy_10pct (mGal)"].values

    # True Parameters: [z, x0, A]
    true_params = {"z": 500.0, "x0": 0.0, "A": 13978.62}

    # Search Bounds
    lb = np.array([300.0, -300.0, 6000.0])
    ub = np.array([700.0, 300.0, 22000.0])

    cases = [
        ("Noise-free (0%)", g_clean, "black", "-"),
        ("5% Gaussian Noise", g_noisy_5, "crimson", "--"),
        ("10% Gaussian Noise", g_noisy_10, "dodgerblue", "-."),
    ]

    results = []
    curves = {}
    predicted_anomalies = {}

    print("\n" + "=" * 75)
    print(
        f"{'Dataset':<20} | {'z (m)':<9} | {'x0 (m)':<8} | {'A (mGal*m^2)':<13} | {'RMSE':<8}"
    )
    print("=" * 75)

    for name, g_obs, _, _ in cases:
        # Define Objective (RMSE)
        def cost_fn(m):
            g_cal = sphere_gravity_forward(x, z=m[0], x0=m[1], A=m[2])
            return compute_rmse(g_obs, g_cal)

        # Execute HHO Optimizer
        hho = HarrisHawksOptimizer(
            cost_func=cost_fn,
            lb=lb,
            ub=ub,
            dim=3,
            search_agents_no=30,
            max_iter=100,
            random_seed=42,
        )
        best_m, best_rmse, conv_curve = hho.optimize()

        # Compute calculated anomaly
        g_pred = sphere_gravity_forward(
            x, z=best_m[0], x0=best_m[1], A=best_m[2]
        )

        curves[name] = conv_curve
        predicted_anomalies[name] = g_pred

        # Parameter Relative Errors
        err_z = compute_relative_error(true_params["z"], best_m[0])
        err_x0 = abs(best_m[1] - true_params["x0"])  # Absolute error for zero
        err_A = compute_relative_error(true_params["A"], best_m[2])
        mae = compute_mae(g_obs, g_pred)

        results.append(
            {
                "Case": name,
                "z_est": best_m[0],
                "z_err%": err_z,
                "x0_est": best_m[1],
                "x0_abs": err_x0,
                "A_est": best_m[2],
                "A_err%": err_A,
                "RMSE": best_rmse,
                "MAE": mae,
            }
        )

        print(
            f"{name:<20} | {best_m[0]:<9.2f} | {best_m[1]:<8.2f} | {best_m[2]:<13.2f} | {best_rmse:<8.5f}"
        )

    print("=" * 75)
    print(
        f"{'True Model':<20} | {true_params['z']:<9.2f} | {true_params['x0']:<8.2f} | {true_params['A']:<13.2f} | {'-':<8}"
    )
    print("=" * 75 + "\n")

    # -------------------------------------------------------------
    # Plot Figure 2: Inversion Fits & Residuals (2-Row Publication Style)
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7.5), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    colors = {"Noise-free (0%)": "black", "5% Gaussian Noise": "crimson", "10% Gaussian Noise": "dodgerblue"}

    # Observed points
    ax1.scatter(x, g_clean, color="black", s=25, alpha=0.5, label="Obs (0%)")
    ax1.scatter(x, g_noisy_5, color="crimson", marker="o", s=25, alpha=0.5, label="Obs (5%)")
    ax1.scatter(x, g_noisy_10, color="dodgerblue", marker="s", s=25, alpha=0.5, label="Obs (10%)")

    # Inverted curves
    for name, _, _, ls in cases:
        ax1.plot(
            x,
            predicted_anomalies[name],
            color=colors[name],
            linestyle=ls,
            linewidth=2.0,
            label=f"HHO Fit ({name})",
        )
        residual = df_profiles[
            "g_clean (mGal)"
            if "0%" in name
            else ("g_noisy_5pct (mGal)" if "5%" in name else "g_noisy_10pct (mGal)")
        ].values - predicted_anomalies[name]
        ax2.plot(
            x,
            residual,
            color=colors[name],
            linestyle=ls,
            linewidth=1.5,
            label=f"Res: {name}",
        )

    ax1.set_ylabel("Gravity Anomaly (mGal)", fontsize=11, fontweight="bold")
    ax1.set_title("HHO Inversion Fit for Clean and Noisy Datasets", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Profile Distance (m)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Residual (mGal)", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax2.legend(loc="upper right", frameon=True, fontsize=8, ncol=3)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join("figures", "fig2_inversion_fit.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[*] Figure 2 saved to: {fig2_path}")

    # -------------------------------------------------------------
    # Plot Figure 3: Convergence Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for name, _, col, ls in cases:
        ax.semilogy(
            np.arange(1, 101),
            curves[name],
            color=col,
            linestyle=ls,
            linewidth=2.0,
            label=f"{name} (Final RMSE={results[[r['Case'] for r in results].index(name)]['RMSE']:.4e})",
        )

    ax.set_title("HHO Optimization Convergence History", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Iteration Number", fontsize=11, fontweight="bold")
    ax.set_ylabel("Objective Function (RMSE in mGal, Log Scale)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, shadow=True, fontsize=9.5)

    plt.tight_layout()
    fig3_path = os.path.join("figures", "fig3_convergence.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[*] Figure 3 saved to: {fig3_path}")


if __name__ == "__main__":
    run_single_inversion()
