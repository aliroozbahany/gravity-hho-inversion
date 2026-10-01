"""
Step 3: Monte Carlo Stability Analysis (30 Independent Runs)
Executes 30 independent runs of HHO for Clean, 5%, and 10% noise levels.
Saves raw runs and full statistical summary to 'data/monte_carlo_results.xlsx'.
Generates Figure 4 (Publication-ready Boxplots for Parameter Stability).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forward_model import sphere_gravity_forward
from src.hho_optimizer import HarrisHawksOptimizer
from src.metrics import compute_relative_error, compute_rmse


def run_monte_carlo_stability():
    excel_profiles = os.path.join("data", "synthetic_profiles.xlsx")
    if not os.path.exists(excel_profiles):
        raise FileNotFoundError(
            f"[-] Could not find '{excel_profiles}'. Please run 01_generate_synthetic_data.py first!"
        )

    # 1. Load Frozen Profiles
    df_profiles = pd.read_excel(excel_profiles, sheet_name="Profiles")
    x = df_profiles["x (m)"].values
    datasets = {
        "Noise-free (0%)": df_profiles["g_clean (mGal)"].values,
        "5% Gaussian Noise": df_profiles["g_noisy_5pct (mGal)"].values,
        "10% Gaussian Noise": df_profiles["g_noisy_10pct (mGal)"].values,
    }

    # True Parameters: [z, x0, A]
    true_params = {"z": 500.0, "x0": 0.0, "A": 13978.62}

    # Search Bounds
    lb = np.array([300.0, -300.0, 6000.0])
    ub = np.array([700.0, 300.0, 22000.0])

    num_runs = 30
    raw_results = []

    print("\n" + "=" * 70)
    print(f"[*] Starting Monte Carlo Stability Test ({num_runs} Independent Runs per Noise Level)...")
    print("=" * 70)

    for noise_label, g_obs in datasets.items():
        print(f"\n>>> Processing: {noise_label}")

        def cost_fn(m):
            g_cal = sphere_gravity_forward(x, z=m[0], x0=m[1], A=m[2])
            return compute_rmse(g_obs, g_cal)

        for run_id in range(1, num_runs + 1):
            # Seed changes per run to ensure independent exploration
            seed = 1000 + run_id
            hho = HarrisHawksOptimizer(
                cost_func=cost_fn,
                lb=lb,
                ub=ub,
                dim=3,
                search_agents_no=30,
                max_iter=100,
                random_seed=seed,
            )
            best_m, best_rmse, _ = hho.optimize()

            err_z = compute_relative_error(true_params["z"], best_m[0])
            err_x0 = abs(best_m[1] - true_params["x0"])
            err_A = compute_relative_error(true_params["A"], best_m[2])

            raw_results.append(
                {
                    "Noise Level": noise_label,
                    "Run": run_id,
                    "z (m)": best_m[0],
                    "x0 (m)": best_m[1],
                    "A (mGal*m^2)": best_m[2],
                    "RMSE (mGal)": best_rmse,
                    "z_RE%": err_z,
                    "x0_AbsErr (m)": err_x0,
                    "A_RE%": err_A,
                }
            )

            if run_id % 10 == 0 or run_id == num_runs:
                print(f"    Completed run {run_id:02d}/{num_runs} | Latest RMSE: {best_rmse:.5e}")

    df_raw = pd.DataFrame(raw_results)

    # 2. Compute Summary Statistics
    summary_list = []
    for noise_label in datasets.keys():
        sub_df = df_raw[df_raw["Noise Level"] == noise_label]
        for param, true_val, unit in [("z", true_params["z"], "m"), ("x0", true_params["x0"], "m"), ("A", true_params["A"], "mGal*m^2")]:
            vals = sub_df[f"{param} ({unit})"].values
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            if param == "x0":
                mre = np.mean(sub_df["x0_AbsErr (m)"].values)
            else:
                mre = np.mean(sub_df[f"{param}_RE%"].values)

            summary_list.append(
                {
                    "Noise Level": noise_label,
                    "Parameter": param,
                    "True Value": true_val,
                    "Mean (mu)": round(mean_val, 2),
                    "Std Dev (sigma)": round(std_val, 2),
                    "Min": round(min_val, 2),
                    "Max": round(max_val, 2),
                    "Mean Error": round(mre, 3),
                    "Unit": unit if param != "x0" else "m (abs)",
                }
            )

    df_summary = pd.DataFrame(summary_list)

    # 3. Save to Excel
    out_excel = os.path.join("data", "monte_carlo_results.xlsx")
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Statistical_Summary", index=False)
        df_raw.to_excel(writer, sheet_name="Raw_Runs", index=False)

    print("\n" + "=" * 70)
    print(f"[*] Monte Carlo results successfully saved to: {out_excel}")
    print("=" * 70)
    print("\n--- Summary Table ---")
    print(df_summary.to_string(index=False))

    # 4. Generate Figure 4: Boxplots for Parameter Stability
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), dpi=300)

    params_info = [
        ("z (m)", true_params["z"], "Depth z (m)", axes[0]),
        ("x0 (m)", true_params["x0"], "Horizontal Location x0 (m)", axes[1]),
        ("A (mGal*m^2)", true_params["A"], "Amplitude Factor A (mGal*m²)", axes[2]),
    ]

    labels = ["Noise-free", "5% Noise", "10% Noise"]

    for col_name, true_v, title, ax in params_info:
        data_to_plot = [
            df_raw[df_raw["Noise Level"] == "Noise-free (0%)"][col_name].values,
            df_raw[df_raw["Noise Level"] == "5% Gaussian Noise"][col_name].values,
            df_raw[df_raw["Noise Level"] == "10% Gaussian Noise"][col_name].values,
        ]

        box = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="darkred", linewidth=1.8),
            boxprops=dict(facecolor="lightblue", alpha=0.7, edgecolor="navy"),
            whiskerprops=dict(color="navy", linewidth=1.2),
            capprops=dict(color="navy", linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=5, alpha=0.6),
        )

        ax.axhline(true_v, color="darkgreen", linestyle="--", linewidth=1.8, label=f"True ({true_v})")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right" if col_name != "x0 (m)" else "upper right", fontsize=9.5)

    fig.suptitle("HHO Stability Analysis Across 30 Independent Monte Carlo Runs", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig4_path = os.path.join("figures", "fig4_stability_boxplot.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    print(f"\n[*] Figure 4 successfully saved to: {fig4_path}")


if __name__ == "__main__":
    run_monte_carlo_stability()
