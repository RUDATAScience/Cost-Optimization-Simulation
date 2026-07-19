import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("モデルA(定常)とモデルB(状態依存)の統合検証シミュレーションを開始します...")

# ==========================================
# 1. パラメータ設定
# ==========================================
# 共通物理パラメータ
mu = 0.05          # 連続劣化(ドリフト)
sigma = 0.05       # 環境ノイズ
mu_J = 0.1         # ジャンプ強度の平均
sigma_J = 0.02     # ジャンプ強度のばらつき

# モデルA: 定常ポアソン過程のパラメータ
lambda_A = 0.2     # 一定のジャンプ発生率 (回/年)

# モデルB: 状態依存ポアソン過程のパラメータ
lambda_B0 = 0.05   # 初期状態でのジャンプ発生率
alpha = 2.5        # 脆弱性の感度パラメータ (状態悪化によるリスク増大率)

# 許容相対誤差の閾値 (Safety-Critical: 3%)
ERROR_THRESHOLD = 0.03

# 点検間隔 dt の配列 (0.01年(約3日) ～ 5.0年)
dt_array = np.logspace(np.log10(0.01), np.log10(5.0), 200)

# ==========================================
# 2. 理論的エイリアシング誤差（PINNの逆推定誤差Proxy）の計算
# ==========================================
def calculate_pinn_error(lambda_rate, dt):
    """
    ジャンプ発生率λと点検間隔dtに基づく、PINN推定値の理論的相対誤差。
    (ポアソン過程において、dt間に2回以上のジャンプが起きる確率に比例して誤差が増大)
    """
    # 誤差モデル: k * (lambda * dt)^1.5  (エイリアシングによる非線形な情報の欠落)
    base_error = 0.5 * (lambda_rate * dt) ** 1.5
    return base_error

# モデルAの誤差 (状態に関わらず一定)
error_A = calculate_pinn_error(lambda_A, dt_array)

# モデルBの誤差 (状態 X に依存して変化)
X_early = 0.2  # 供用初期
X_mid = 0.6    # 供用中期
X_late = 1.0   # 供用末期 (閾値C寸前)

lambda_B_early = lambda_B0 * np.exp(alpha * X_early)
lambda_B_mid = lambda_B0 * np.exp(alpha * X_mid)
lambda_B_late = lambda_B0 * np.exp(alpha * X_late)

error_B_early = calculate_pinn_error(lambda_B_early, dt_array)
error_B_mid = calculate_pinn_error(lambda_B_mid, dt_array)
error_B_late = calculate_pinn_error(lambda_B_late, dt_array)

# データフレーム化
df_error = pd.DataFrame({
    'dt_years': dt_array,
    'Error_Model_A': error_A,
    'Error_Model_B_Early': error_B_early,
    'Error_Model_B_Mid': error_B_mid,
    'Error_Model_B_Late': error_B_late
})

# ==========================================
# 3. 最適点検間隔 (dt_critical) の算出
# ==========================================
def find_critical_dt(error_array, dt_array, threshold=ERROR_THRESHOLD):
    # 誤差が3%を超える直前のdtを特定
    valid_idx = np.where(error_array <= threshold)[0]
    if len(valid_idx) > 0:
        return dt_array[valid_idx[-1]]
    return dt_array[0] # 常に超えている場合は最小値

dt_crit_A = find_critical_dt(error_A, dt_array)
dt_crit_B_early = find_critical_dt(error_B_early, dt_array)
dt_crit_B_mid = find_critical_dt(error_B_mid, dt_array)
dt_crit_B_late = find_critical_dt(error_B_late, dt_array)

# 動的保全マップ用データ (状態 X に対する限界 dt)
X_range = np.linspace(0.0, 1.2, 100)
dt_crit_dynamic = []
for x in X_range:
    lam_x = lambda_B0 * np.exp(alpha * x)
    err_x = calculate_pinn_error(lam_x, dt_array)
    dt_crit_dynamic.append(find_critical_dt(err_x, dt_array))

df_dynamic = pd.DataFrame({'State_X': X_range, 'Critical_dt': dt_crit_dynamic})

# ==========================================
# 4. 可視化とグラフ保存
# ==========================================
print("データを生成・可視化しています...")
filenames = []

# (グラフ1) モデルAとモデルBの相対誤差と3%閾値の交点
plt.figure(figsize=(12, 7))
plt.plot(dt_array, error_A * 100, label='Model A (Stationary)', color='black', lw=3)
plt.plot(dt_array, error_B_early * 100, label='Model B (Early State: X=0.2)', color='royalblue', linestyle='-.', lw=2)
plt.plot(dt_array, error_B_mid * 100, label='Model B (Mid State: X=0.6)', color='orange', linestyle='--', lw=2)
plt.plot(dt_array, error_B_late * 100, label='Model B (Late State: X=1.0)', color='red', linestyle='-', lw=2)

# 3% 閾値ライン (raw文字列を使用)
plt.axhline(ERROR_THRESHOLD * 100, color='red', linestyle=':', lw=2, label=r'Strict Tolerance Limit ($\pm 3\%$)')

# ブレイクダウン・ポイントのマーキング
plt.scatter([dt_crit_A, dt_crit_B_early, dt_crit_B_mid, dt_crit_B_late],
            [3.0]*4, color='gold', s=100, edgecolors='black', zorder=5)

plt.annotate(f'{dt_crit_A*12:.1f} m', (dt_crit_A, 3.2), ha='center')
plt.annotate(f'{dt_crit_B_early*12:.1f} m', (dt_crit_B_early, 2.5), ha='center', color='royalblue')
plt.annotate(f'{dt_crit_B_mid*12:.1f} m', (dt_crit_B_mid, 3.2), ha='center', color='darkorange')
plt.annotate(f'{dt_crit_B_late*12:.1f} m', (dt_crit_B_late, 2.5), ha='center', color='darkred')

plt.title('PINN Estimation Relative Error vs Inspection Interval ($\Delta t$)')
# raw文字列を使用
plt.xlabel(r'Inspection Interval $\Delta t$ (Years)')
plt.ylabel(r'Relative Error of Estimated $\lambda$ (%)')
plt.xscale('log')
plt.ylim(0, 10)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('1_PINN_Error_Limits.png')
filenames.append('1_PINN_Error_Limits.png')
plt.close()

# (グラフ2) CBM(状態基準保全)から予測型動的保全への進化マップ
plt.figure(figsize=(10, 6))
plt.plot(df_dynamic['State_X'], df_dynamic['Critical_dt'] * 12, color='darkgreen', lw=3)
# raw文字列と \leq を使用
plt.fill_between(df_dynamic['State_X'], df_dynamic['Critical_dt'] * 12, 60, color='red', alpha=0.1, label='Breakdown Zone (Error > 3%)')
plt.fill_between(df_dynamic['State_X'], 0, df_dynamic['Critical_dt'] * 12, color='lightgreen', alpha=0.3, label=r'Safe Zone (Error $\leq$ 3%)')

plt.title('Dynamic Inspection Scheduling (Model B: State-Dependent)')
# raw文字列を使用
plt.xlabel(r'Degradation State $X_t$ (Proximity to Collapse)')
plt.ylabel(r'Maximum Allowable Interval $\Delta t_{critical}$ (Months)')
plt.axvline(1.0, color='black', linestyle='--', label='Threshold C (Tipping Point)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.4)
plt.savefig('2_Dynamic_Inspection_Map.png')
filenames.append('2_Dynamic_Inspection_Map.png')
plt.close()

# CSVの保存
df_error.to_csv('PINN_error_vs_dt.csv', index=False)
df_dynamic.to_csv('Dynamic_dt_critical_map.csv', index=False)
filenames.extend(['PINN_error_vs_dt.csv', 'Dynamic_dt_critical_map.csv'])

# ==========================================
# 5. ZIP圧縮とダウンロード
# ==========================================
zip_filename = 'Dynamic_Inspection_Strategy_Results.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for f in filenames:
        zipf.write(f)

print(f"{zip_filename} を作成しました。ダウンロードを開始します...")

if IN_COLAB:
    files.download(zip_filename)
