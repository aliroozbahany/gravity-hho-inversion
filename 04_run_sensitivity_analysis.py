"""
Step 4: Sensitivity & Equivalence Analysis
1. OAT (One-at-a-Time) Sensitivity curves around the true model.
2. 2D Objective Function Landscape / Contour Map in (z, A) space showing convergence path.
Generates Figures 6 and 7 (300 DPI publication standards).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forward_model import sphere_gravity_forward
from src.hho_optimizer import HarrisHawksOptimizer
from src.metrics import compute_rmse


def run_sensitivity_analysis():
    excel_path = os.path.join("data", "synthetic_profiles.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"[-] Could not find '{excel_path}'. Please run 01_generate_synthetic_data.py first!"
        )

    df_profiles = pd.read_excel(excel_path, sheet_name="Profiles")
    x = df_profiles["x (m)"].values
    g_clean = df_profiles["g_clean (mGal)"].values

    # True Parameters
    z_true = 500.0
    x0_true = 0.0
    A_true = 13978.62

    print("\n" + "=" * 70)
    print("[*] Running Step 4: Sensitivity & Parameter Equivalence Analysis...")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. One-at-a-Time (OAT) Sensitivity Analysis
    # -------------------------------------------------------------
    perturbations = np.linspace(-0.30, 0.30, 101)  # -30% to +30%
    rmse_z = []
    rmse_x0 = []
    rmse_A = []

    for p in perturbations:
        # Varying z
        z_val = z_true * (1 + p)
        g_sim_z = sphere_gravity_forward(x, z=z_val, x0=x0_true, A=A_true)
        rmse_z.append(compute_rmse(g_clean, g_sim_z))

        # Varying x0 (since x0_true=0, perturb by percentage of profile extent e.g. 500m)
        x0_val = x0_true + (p * 200.0)
        g_sim_x0 = sphere_gravity_forward(x, z=z_true, x0=x0_val, A=A_true)
        rmse_x0.append(compute_rmse(g_clean, g_sim_x0))

        # Varying A
        A_val = A_true * (1 + p)
        g_sim_A = sphere_gravity_forward(x, z=z_true, x0=x0_true, A=A_val)
        rmse_A.append(compute_rmse(g_clean, g_sim_A))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    ax.plot(perturbations * 100, rmse_z, "r-", linewidth=2.2, label=r"Depth $z$ ($\pm 30\%$)")
    ax.plot(perturbations * 100, rmse_A, "b--", linewidth=2.2, label=r"Amplitude Factor $A$ ($\pm 30\%$)")
    ax.plot(perturbations * 100, rmse_x0, "g-.", linewidth=2.2, label=r"Horizontal Location $x_0$ ($\pm 60\text{ m}$)")

    ax.set_title("OAT Sensitivity Analysis of Misfit Function (RMSE)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Perturbation from True Model (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSE (mGal)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", frameon=True, fontsize=10)

    plt.tight_layout()
    fig6_path = os.path.join("figures", "fig6_sensitivity_oat.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"[*] Figure 6 (OAT Sensitivity) saved to: {fig6_path}")

    # -------------------------------------------------------------
    # 2. 2D Landscape / Contour Analysis in (z, A) Space
    # -------------------------------------------------------------
    z_grid = np.linspace(350.0, 650.0, 80)
    A_grid = np.linspace(8000.0, 20000.0, 80)
    Z_mesh, A_mesh = np.meshgrid(z_grid, A_grid)
    cost_surface = np.zeros_like(Z_mesh)

    for i in range(Z_mesh.shape[0]):
        for j in range(Z_mesh.shape[1]):
            g_sim = sphere_gravity_forward(x, z=Z_mesh[i, j], x0=x0_true, A=A_mesh[i, j])
            cost_surface[i, j] = compute_rmse(g_clean, g_sim)

    # Run a quick tracked HHO run for trajectory recording
    lb = np.array([300.0, -300.0, 6000.0])
    ub = np.array([700.0, 300.0, 22000.0])

    trajectory = []

    class TrackingHHO(HarrisHawksOptimizer):
        def optimize(self):
            np.random.seed(42)
            rabbit_location = np.zeros(self.dim)
            rabbit_energy = float("inf")
            x_agents = self.lb + np.random.rand(self.search_agents_no, self.dim) * (self.ub - self.lb)
            
            for t in range(self.max_iter):
                for i in range(self.search_agents_no):
                    x_agents[i, :] = np.clip(x_agents[i, :], self.lb, self.ub)
                    fit = self.cost_func(x_agents[i, :])
                    if fit < rabbit_energy:
                        rabbit_energy = fit
                        rabbit_location = x_agents[i, :].copy()
                trajectory.append(rabbit_location.copy())

                e0 = 2 * np.random.rand() - 1
                e = 2 * e0 * (1 - (t / self.max_iter))
                for i in range(self.search_agents_no):
                    if np.abs(e) >= 1:
                        rand_idx = np.random.randint(0, self.search_agents_no)
                        x_agents[i, :] = x_agents[rand_idx, :] - np.random.rand() * np.abs(
                            x_agents[rand_idx, :] - 2 * np.random.rand() * x_agents[i, :]
                        )
                    else:
                        delta_x = rabbit_location - x_agents[i, :]
                        x_agents[i, :] = rabbit_location - e * np.abs(delta_x)

            return rabbit_location, rabbit_energy, np.array(trajectory)

    def cost_fn_track(m):
        return compute_rmse(g_clean, sphere_gravity_forward(x, z=m[0], x0=m[1], A=m[2]))

    tracker = TrackingHHO(cost_fn_track, lb, ub, dim=3, search_agents_no=25, max_iter=30)
    _, _, traj = tracker.optimize()

    # Plot Contour Map
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    cp = ax.contourf(Z_mesh, A_mesh, cost_surface, levels=30, cmap="viridis_r")
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label("Misfit RMSE (mGal)", fontsize=11, fontweight="bold")

    # Contours lines
    ax.contour(Z_mesh, A_mesh, cost_surface, levels=15, colors="white", alpha=0.3, linewidths=0.7)

    # Plot True Point
    ax.plot(z_true, A_true, "r*", markersize=14, markeredgecolor="black", label=r"True Solution ($z=500\text{ m}, A=13978.6$)")

    # Plot Optimizer Trajectory
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 2], "w.-", linewidth=1.5, markersize=5, alpha=0.85, label="HHO Best Agent Path")
    ax.plot(traj[0, 0], traj[0, 2], "yo", markersize=8, markeredgecolor="black", label="Start Location")

    ax.set_title("Objective Function Topology & HHO Trajectory in $(z, A)$ Space", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Depth $z$ (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Amplitude Factor $A$ ($\text{mGal}\cdot\text{m}^2$)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    fig7_path = os.path.join("figures", "fig7_sensitivity_map.png")
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"[*] Figure 7 (Sensitivity Topology Map) saved to: {fig7_path}")


if __name__ == "__main__":
    run_sensitivity_analysis()
