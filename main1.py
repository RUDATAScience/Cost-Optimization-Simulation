import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import zipfile
import os
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("インフラ劣化逆問題：全フェーズ(1〜4) 試行回数比較・統合シミュレーションを開始します...\n")
filenames = []

# =====================================================================
# パラメータ共通設定
# =====================================================================
mu_c = 0.2          # 連続劣化
sigma_c = 0.1       # 環境ノイズ
Threshold_C = 1.0   # 崩壊閾値
lambda_j = 0.5      # ジャンプ発生率
mu_j = 0.3          # ジャンプ量

N_small = 1000
N_large = 10000000  # Phase 2のみメモリ保護のため1,000,000を使用

# =====================================================================
# Phase 1 & 2: 連続モデルの限界 と ジャンプ崩壊のFPT極値現象
# =====================================================================
print("Phase 1 & 2 を実行中: FPT(初到達時間)のN規模比較シミュレーション...")
def simulate_fpt(N_samples, include_jump=True):
    np.random.seed(42)
    X = np.zeros(N_samples)
    time_elapsed = np.zeros(N_samples)
    active = np.ones(N_samples, dtype=bool)
    FPTs = np.full(N_samples, np.nan)

    dt = 0.05
    t_step = 0
    # 最大2000ステップ (時間100) まで追跡
    while np.any(active) and t_step < 2000:
        n_active = np.sum(active)
        Z = np.random.normal(0, np.sqrt(dt), n_active)

        if include_jump:
            jumps = np.random.poisson(lambda_j * dt, n_active) * np.random.normal(mu_j, 0.1, n_active)
            jumps = np.maximum(0, jumps)
        else:
            jumps = 0

        X[active] += mu_c * dt + sigma_c * Z + jumps
        time_elapsed[active] += dt

        crossed = X[active] >= Threshold_C
        if np.any(crossed):
            newly_crossed = np.where(active)[0][crossed]
            FPTs[newly_crossed] = time_elapsed[newly_crossed]
            active[newly_crossed] = False
        t_step += 1
    return FPTs[~np.isnan(FPTs)]

# 大規模シミュレーションの実行 (N=1,000 vs N=1,000,000)
fpt_jump_small = simulate_fpt(N_small, include_jump=True)
fpt_jump_large = simulate_fpt(1000000, include_jump=True) # RAM保護のため1M

# FPTの統計データをCSV化
df_fpt_stats = pd.DataFrame({
    'Metric': ['Count', 'Mean FPT', '95th Percentile (Tail)', 'Max FPT (Extreme)'],
    'N_1000': [len(fpt_jump_small), np.mean(fpt_jump_small), np.percentile(fpt_jump_small, 95), np.max(fpt_jump_small)],
    'N_1000000': [len(fpt_jump_large), np.mean(fpt_jump_large), np.percentile(fpt_jump_large, 95), np.max(fpt_jump_large)]
})
df_fpt_stats.to_csv('Phase2_FPT_Stats.csv', index=False)
filenames.append('Phase2_FPT_Stats.csv')

plt.figure(figsize=(12, 6))
plt.hist(fpt_jump_large, bins=100, density=True, alpha=0.5, color='purple', label='N = 1,000,000')
plt.hist(fpt_jump_small, bins=30, density=True, alpha=0.8, color='orange', edgecolor='black', label='N = 1,000')
plt.title(r'Phase 2: FPT Distribution (Extreme Value Tail Revealed by Large $N$)')
plt.xlabel('Time to Tipping Point')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Phase2_FPT_Comparison.png')
filenames.append('Phase2_FPT_Comparison.png')
plt.close()

# =====================================================================
# Phase 3: コスト最適化とサンプリング検証 (N=1,000 vs N=10,000,000)
# =====================================================================
print("Phase 3 を実行中: 大規模MCによるコスト関数の最適化...")
dt_eval = np.logspace(np.log10(0.01), np.log10(5.0), 100)
C_insp = 10.0
C_fail = 20000.0

def calc_cost(N_samples):
    np.random.seed(42)
    costs = []
    for dt in dt_eval:
        jumps = np.random.poisson(lambda_j * dt, N_samples)
        J_tot = np.where(jumps > 0, np.random.normal(mu_j * jumps, 0.1 * np.sqrt(jumps)), 0)
        delta_X = mu_c * dt + sigma_c * np.sqrt(dt) * np.random.normal(0, 1, N_samples) + J_tot
        P_fail = np.sum(delta_X >= Threshold_C) / N_samples
        Total_Cost = (C_insp / dt) + C_fail * (P_fail / dt)
        costs.append(Total_Cost)
    return np.array(costs)

cost_small = calc_cost(N_small)
cost_large = calc_cost(N_large)

df_cost = pd.DataFrame({'dt_years': dt_eval, 'Cost_N1000': cost_small, 'Cost_N10M': cost_large})
df_cost.to_csv('Phase3_Cost_Optimization.csv', index=False)
filenames.append('Phase3_Cost_Optimization.csv')

plt.figure(figsize=(10, 6))
plt.plot(dt_eval, cost_small, color='lightblue', lw=2, label=r'N = 1,000 (False Safety / Noisy)')
plt.plot(dt_eval, cost_large, color='navy', lw=3, label=r'N = 10,000,000 (True Sweet Spot)')
opt_dt = dt_eval[np.argmin(cost_large)]
plt.scatter(opt_dt, np.min(cost_large), color='gold', s=150, edgecolors='black', zorder=5, label=f'Optimal dt: {opt_dt:.2f}y')
plt.title(r'Phase 3: Cost Optimization Verification')
plt.xlabel(r'Inspection Interval $\Delta t$ (Years)')
plt.ylabel('Total Annualized Cost')
plt.xscale('log')
plt.ylim(0, np.min(cost_large)*4)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Phase3_CostOptimization.png')
filenames.append('Phase3_CostOptimization.png')
plt.close()

# =====================================================================
# Phase 4: PINNsストレステスト (経験的エイリアシング誤差)
# =====================================================================
print("Phase 4 を実行中: PINNs逆推定限界の動的保全マップ (N=1K vs N=10M)...")
lambda_B0 = 0.05
alpha = 2.5
ERROR_THRESHOLD = 0.03

def empirical_pinn_error(N_samples, lam, dt_arr):
    errors = []
    np.random.seed(42)
    for dt in dt_arr:
        jumps = np.random.poisson(lam * dt, N_samples)
        total_jumps = np.sum(jumps)
        if total_jumps == 0:
            err = 0.0 # ジャンプ未発生(偽の安全)
        else:
            aliased = np.sum(np.maximum(0, jumps - 1))
            err = aliased / total_jumps
        errors.append(err)
    return np.array(errors)

X_range = np.linspace(0.0, 1.2, 50)
dt_crit_dynamic_small = []
dt_crit_dynamic_large = []

for x in X_range:
    lam_x = lambda_B0 * np.exp(alpha * x)

    # N=1000 の限界
    err_small = empirical_pinn_error(N_small, lam_x, dt_eval)
    v_small = np.where(err_small <= ERROR_THRESHOLD)[0]
    dt_crit_dynamic_small.append(dt_eval[v_small[-1]] if len(v_small) > 0 else dt_eval[0])

    # N=10,000,000 の限界
    err_large = empirical_pinn_error(N_large, lam_x, dt_eval)
    v_large = np.where(err_large <= ERROR_THRESHOLD)[0]
    dt_crit_dynamic_large.append(dt_eval[v_large[-1]] if len(v_large) > 0 else dt_eval[0])

df_dynamic = pd.DataFrame({
    'State_X': X_range,
    'Critical_dt_N1000': dt_crit_dynamic_small,
    'Critical_dt_N10M': dt_crit_dynamic_large
})
df_dynamic.to_csv('Phase4_Dynamic_Inspection.csv', index=False)
filenames.append('Phase4_Dynamic_Inspection.csv')

plt.figure(figsize=(10, 6))
plt.plot(X_range, np.array(dt_crit_dynamic_small) * 12, color='gray', linestyle='--', lw=2, label='N = 1,000 (Underestimates Risk)')
plt.plot(X_range, np.array(dt_crit_dynamic_large) * 12, color='darkgreen', lw=3, label='N = 10,000,000 (True Safe Boundary)')
plt.fill_between(X_range, np.array(dt_crit_dynamic_large) * 12, 60, color='red', alpha=0.1, label=r'Breakdown Zone (Error > 3%)')
plt.fill_between(X_range, 0, np.array(dt_crit_dynamic_large) * 12, color='lightgreen', alpha=0.3, label=r'Safe Zone (Error $\leq$ 3%)')

plt.title(r'Phase 4: Dynamic Inspection Map (Empirical Proof)')
plt.xlabel(r'Degradation State $X_t$ (Proximity to Tipping Point)')
plt.ylabel(r'Maximum Allowable Interval $\Delta t_{critical}$ (Months)')
plt.axvline(1.0, color='black', linestyle='--', label='Threshold C')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.4)
plt.savefig('Phase4_DynamicInspection.png')
filenames.append('Phase4_DynamicInspection.png')
plt.close()

# =====================================================================
# 5. ZIP圧縮とダウンロード
# =====================================================================
zip_filename = 'Infrastructure_Full_Simulations_with_CSV.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for f in filenames:
        zipf.write(f)

print(f"\nすべてのシミュレーションとCSVの生成が完了しました。{zip_filename} をダウンロードします...")

if IN_COLAB:
    files.download(zip_filename)
