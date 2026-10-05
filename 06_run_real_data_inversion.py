import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math  # این کتابخانه اضافه شد

# ==============================================================================
# Configuration & B&W Publication Styling
# ==============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{level:<5}] {msg}")

def safe_read_excel(file_path):
    temp_path = file_path + ".tmp_read"
    try:
        shutil.copyfile(file_path, temp_path)
        df = pd.read_excel(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return df
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return pd.read_excel(file_path)

# ==============================================================================
# Forward Modeling & Variable Projection
# ==============================================================================
def sphere_gravity(x, x0, z0, R, delta_rho):
    G = 6.67430e-11
    SI_to_mGal = 1e5
    V = (4.0 / 3.0) * np.pi * (R**3)
    mass = V * delta_rho
    r2 = (x - x0)**2 + z0**2
    gz_SI = G * mass * z0 / (r2**1.5)
    return gz_SI * SI_to_mGal

def solve_varproj(x, g_obs, x0, z0, R, delta_rho):
    g_sph = sphere_gravity(x, x0, z0, R, delta_rho)
    A = np.column_stack([np.ones_like(x), x])
    rhs = g_obs - g_sph
    coef, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    c0, c1 = coef[0], coef[1]
    g_pred = g_sph + A @ coef
    rmse = np.sqrt(np.mean((g_obs - g_pred)**2))
    return rmse, g_pred, c0, c1

# ==============================================================================
# Harris Hawks Optimization (HHO)
# ==============================================================================
def levy_flight(dim):
    beta = 1.5
    # اصلاح شده: استفاده از math.gamma به جای np.math.gamma
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
            (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step

def hho_inversion_real(x, g_obs, n_hawks=30, max_iter=500, seed=42):
    np.random.seed(seed)
    lb = np.array([x.min(), 0.5, 0.5, -3000.0])
    ub = np.array([x.max(), 10.0, 5.0, -1000.0])
    dim = len(lb)
    X = lb + np.random.rand(n_hawks, dim) * (ub - lb)
    fitness = np.zeros(n_hawks)
    
    for i in range(n_hawks):
        fitness[i], _, _, _ = solve_varproj(x, g_obs, X[i,0], X[i,1], X[i,2], X[i,3])
        
    best_idx = np.argmin(fitness)
    Rabbit_Location = X[best_idx].copy()
    Rabbit_Energy = fitness[best_idx]
    
    for t in range(max_iter):
        E0 = 2.0 * np.random.rand() - 1.0
        E = 2.0 * (1.0 - (t / max_iter)) * E0
        
        for i in range(n_hawks):
            if np.abs(E) >= 1.0:
                q = np.random.rand()
                rand_hawk = X[np.random.randint(0, n_hawks)]
                if q >= 0.5:
                    X[i] = rand_hawk - np.random.rand() * np.abs(rand_hawk - 2.0 * np.random.rand() * X[i])
                else:
                    X[i] = (Rabbit_Location - X.mean(axis=0)) - np.random.rand() * (lb + np.random.rand() * (ub - lb))
            else:
                r = np.random.rand()
                J = 2.0 * (1.0 - np.random.rand())
                if r >= 0.5 and np.abs(E) >= 0.5:
                    X[i] = (Rabbit_Location - X[i]) - E * np.abs(J * Rabbit_Location - X[i])
                elif r >= 0.5 and np.abs(E) < 0.5:
                    X[i] = Rabbit_Location - E * np.abs(Rabbit_Location - X[i])
                elif r < 0.5 and np.abs(E) >= 0.5:
                    LF = levy_flight(dim)
                    Y = Rabbit_Location - E * np.abs(J * Rabbit_Location - X[i])
                    Y = np.clip(Y, lb, ub)
                    fY, _, _, _ = solve_varproj(x, g_obs, Y[0], Y[1], Y[2], Y[3])
                    Z = Y + np.random.randn(dim) * LF
                    Z = np.clip(Z, lb, ub)
                    fZ, _, _, _ = solve_varproj(x, g_obs, Z[0], Z[1], Z[2], Z[3])
                    if fY < fitness[i]: X[i] = Y
                    elif fZ < fitness[i]: X[i] = Z
                    continue
                else:
                    LF = levy_flight(dim)
                    Y = Rabbit_Location - E * np.abs(J * Rabbit_Location - X.mean(axis=0))
                    Y = np.clip(Y, lb, ub)
                    fY, _, _, _ = solve_varproj(x, g_obs, Y[0], Y[1], Y[2], Y[3])
                    Z = Y + np.random.randn(dim) * LF
                    Z = np.clip(Z, lb, ub)
                    fZ, _, _, _ = solve_varproj(x, g_obs, Z[0], Z[1], Z[2], Z[3])
                    if fY < fitness[i]: X[i] = Y
                    elif fZ < fitness[i]: X[i] = Z
                    continue
            X[i] = np.clip(X[i], lb, ub)
            fitness[i], _, _, _ = solve_varproj(x, g_obs, X[i,0], X[i,1], X[i,2], X[i,3])
            if fitness[i] < Rabbit_Energy:
                Rabbit_Energy = fitness[i]
                Rabbit_Location = X[i].copy()
    return Rabbit_Location, Rabbit_Energy

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    output_dir = os.path.join(script_dir, "conf_figures")
    os.makedirs(output_dir, exist_ok=True)
    
    data_file = os.path.join(script_dir, "RealData", "Profiles_2_Naghade.xlsx")
    if not os.path.exists(data_file): data_file = os.path.join(script_dir, "Profiles_2_Naghade.xlsx")
    
    df = safe_read_excel(data_file)
    x_obs, g_obs = df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)
    
    best_params, best_rmse = hho_inversion_real(x_obs, g_obs)
    x0_opt, z0_opt, R_opt, drho_opt = best_params
    _, g_pred, c0_opt, c1_opt = solve_varproj(x_obs, g_obs, x0_opt, z0_opt, R_opt, drho_opt)
    residuals = g_obs - g_pred
    z_top = z0_opt - R_opt

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.5, 9.5), gridspec_kw={'height_ratios': [1, 0.7, 1]})
    ax1.plot(x_obs, g_obs, 'ko', mfc='none', ms=3.5, label='Observed')
    ax1.plot(x_obs, g_pred, 'k-', lw=1.8, label='HHO Model')
    ax1.plot(x_obs, c0_opt + c1_opt * x_obs, 'k--', lw=1.0, label='Regional Trend')
    ax1.set_title('(a) Field Gravity Anomaly', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    
    ax2.plot(x_obs, residuals, 'k.-', lw=0.7, ms=3)
    ax2.axhline(0, color='black', lw=0.8); ax2.axhline(best_rmse, color='gray', linestyle=':')
    ax2.axhline(-best_rmse, color='gray', linestyle=':')
    ax2.set_title(f'(b) Residuals (RMSE = {best_rmse:.4f} mGal)', fontweight='bold')
    
    circle = plt.Circle((x0_opt, z0_opt), R_opt, edgecolor='black', facecolor='white', hatch='///', lw=1.5)
    ax3.add_patch(circle)
    ax3.plot(x0_opt, z0_opt, 'kx', ms=8); ax3.axhline(0, color='black', lw=1.5)
    ax3.set_ylim(z0_opt + R_opt + 1.5, -0.5); ax3.set_xlim(x_obs.min(), x_obs.max())
    ax3.set_title('(c) Inverted Subsurface Geometry', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_real_data_hho_bw.png"), dpi=300)
    print("Successfully generated B&W plots.")
    plt.show()
