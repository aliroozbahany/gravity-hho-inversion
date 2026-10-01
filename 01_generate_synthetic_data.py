"""
Step 1: Synthetic Data Generation and Freezing
Produces clean and noisy synthetic gravity anomaly profiles for a buried sphere,
exports them into Excel (data/synthetic_profiles.xlsx), and generates Figure 1.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forward_model import calculate_amplitude_factor, sphere_gravity_forward


def generate_and_freeze_synthetic_data():
    # 1. Ensure required directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 2. Define Physical & Geometric Parameters
    R_true = 100.0  # Radius (m)
    rho_contrast = 500.0  # Density contrast (kg/m^3)
    z_true = 500.0  # Depth (m)
    x0_true = 0.0  # Horizontal location (m)

    # Calculate theoretical amplitude factor A
    A_true = calculate_amplitude_factor(R_true, rho_contrast)  # ~13978.62 mGal*m^2

    # 3. Profile Observation Grid
    x_min, x_max, dx = -1200.0, 1200.0, 50.0
    x_profile = np.arange(x_min, x_max + dx, dx)  # 49 stations

    # 4. Generate Clean Anomaly
    g_clean = sphere_gravity_forward(x_profile, z_true, x0_true, A_true)
    g_max = np.max(np.abs(g_clean))

    # 5. Add Gaussian Noise (Reproducible Seed)
    np.random.seed(42)  # Fixed seed for reproducibility
    noise_5pct = 0.05 * g_max * np.random.normal(0, 1, size=len(x_profile))
    noise_10pct = 0.10 * g_max * np.random.normal(0, 1, size=len(x_profile))

    g_noisy_5 = g_clean + noise_5pct
    g_noisy_10 = g_clean + noise_10pct

    # 6. Save Data to Excel with Two Worksheets
    excel_path = os.path.join("data", "synthetic_profiles.xlsx")

    df_metadata = pd.DataFrame(
        {
            "Parameter": [
                "True Depth (z)",
                "True Center (x0)",
                "True Amplitude (A)",
                "True Radius (R)",
                "Density Contrast (delta_rho)",
                "Max Clean Anomaly (g_max)",
                "Profile Range (x_min, x_max, dx)",
                "Total Stations (N)",
                "Random Seed",
            ],
            "Value": [
                z_true,
                x0_true,
                round(A_true, 4),
                R_true,
                rho_contrast,
                round(g_max, 5),
                f"[{x_min}, {x_max}, {dx}]",
                len(x_profile),
                42,
            ],
            "Unit": [
                "m",
                "m",
                "mGal*m^2",
                "m",
                "kg/m^3",
                "mGal",
                "m",
                "-",
                "-",
            ],
        }
    )

    df_profiles = pd.DataFrame(
        {
            "x (m)": x_profile,
            "g_clean (mGal)": g_clean,
            "g_noisy_5pct (mGal)": g_noisy_5,
            "g_noisy_10pct (mGal)": g_noisy_10,
            "noise_5pct_residual (mGal)": noise_5pct,
            "noise_10pct_residual (mGal)": noise_10pct,
        }
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_metadata.to_excel(writer, sheet_name="Metadata", index=False)
        df_profiles.to_excel(writer, sheet_name="Profiles", index=False)

    print(f"[*] Synthetic data successfully saved to: {excel_path}")

    # 7. Generate Figure 1 (Publication Quality 300 DPI)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax.plot(
        x_profile,
        g_clean,
        "k-",
        linewidth=2.2,
        label="Noise-free (Synthetic)",
        zorder=4,
    )
    ax.scatter(
        x_profile,
        g_noisy_5,
        color="crimson",
        marker="o",
        s=30,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
        label="Noisy (5% Gaussian noise)",
        zorder=3,
    )
    ax.scatter(
        x_profile,
        g_noisy_10,
        color="dodgerblue",
        marker="s",
        s=30,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
        label="Noisy (10% Gaussian noise)",
        zorder=2,
    )

    ax.set_title("Synthetic Gravity Anomaly Over Buried Sphere", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Profile Distance (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gravity Anomaly (mGal)", fontsize=11, fontweight="bold")
    ax.set_xlim(x_min - 50, x_max + 50)
    ax.set_ylim(-0.005, g_max * 1.15)
    ax.legend(loc="upper right", frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join("figures", "fig1_synthetic_profiles.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[*] Figure 1 generated and saved to: {fig_path}")


if __name__ == "__main__":
    generate_and_freeze_synthetic_data()
