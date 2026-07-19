import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import gc
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("「状態Cトリガー」動的保全ルールの有効性証明シミュレーションを開始します...\n")

# ==========================================
# 1. シミュレーション設定
# ==========================================
N_paths = 100000       # パス数 (メモリ考慮で10万パス)
T_years = 5.0          # シミュレーション期間 (5年)
dt_sim = 1.0 / 120.0   # ベースのシミュレーション刻み (1ヶ月=10ステップ, 1年=120ステップ)
steps = int(T_years / dt_sim)

# 物理パラメータ (状態依存ポアソン過程)
mu_c = 0.2
sigma_c = 0.1
lam_0 = 0.05
alpha = 2.5
Threshold_C = 0.8      # 状態Cへのトリガー閾値

# 吸収境界 (物理的崩壊の上限値)
# 現実にはX=1.0付近で崩壊すると定義しているため、計算上の発散を防ぐためのキャップ
MAX_STATE_X = 1.5

# ==========================================
# 2. 真の劣化パス生成 (ベクトル化)
# ==========================================
print("真の劣化パス(Ground Truth)を生成中...")
np.random.seed(42)

X = np.zeros((N_paths, steps + 1))
jump_counts = np.zeros((N_paths, steps), dtype=int)
active_paths = np.ones(N_paths, dtype=bool) # 崩壊していないパスのフラグ

for i in range(steps):
    # すでに崩壊境界(MAX_STATE_X)を超えたパスは進行させない
    active_idx = np.where(active_paths)[0]
    if len(active_idx) == 0:
        break

    # 現在の有効な状態を取得（計算の安定化のため上限を設ける）
    current_X = np.clip(X[active_idx, i], 0, MAX_STATE_X)

    # 状態依存のジャンプ発生率
    lam_curr = lam_0 * np.exp(alpha * current_X)

    # ジャンプの発生
    j_curr = np.random.poisson(lam_curr * dt_sim)
    jump_counts[active_idx, i] = j_curr

    # ジャンプサイズ (平均0.3)
    j_size = j_curr * np.maximum(0, np.random.normal(0.3, 0.05, len(active_idx)))

    # 状態の更新
    delta_X = mu_c * dt_sim + sigma_c * np.sqrt(dt_sim) * np.random.randn(len(active_idx)) + j_size
    X[active_idx, i+1] = X[active_idx, i] + delta_X

    # 吸収境界を超えたパスを非アクティブにする（完全に崩壊したとみなす）
    active_paths[active_idx] = X[active_idx, i+1] < MAX_STATE_X

# 崩壊して非アクティブになったパスの以降のXを固定
for p in range(N_paths):
    last_valid_idx = np.where(X[p] >= MAX_STATE_X)[0]
    if len(last_valid_idx) > 0:
        idx = last_valid_idx[0]
        X[p, idx:] = X[p, idx]

# ==========================================
# 3. 3つの点検戦略の評価
# ==========================================
print("3つの点検戦略を評価中...")

def evaluate_strategy(strategy_type):
    total_inspections = np.zeros(N_paths)
    aliased_jumps = np.zeros(N_paths)
    total_actual_jumps = np.sum(jump_counts, axis=1)

    for p in range(N_paths):
        inspection_points = [0]
        curr_step = 0

        while curr_step < steps:
            # 戦略に基づく次の点検までのステップ数 (1ヶ月=10, 1年=120)
            if strategy_type == 'Fixed_1Year':
                step_size = 120
            elif strategy_type == 'Fixed_1Month':
                step_size = 10
            elif strategy_type == 'Dynamic_Trigger':
                # 現在の状態がトリガー閾値を超えていれば1ヶ月、未満なら1年
                if X[p, curr_step] >= Threshold_C:
                    step_size = 10
                else:
                    step_size = 120

            next_step = min(curr_step + step_size, steps)
            inspection_points.append(next_step)
            curr_step = next_step
            total_inspections[p] += 1

            # すでに吸収境界に達している場合は点検を終了（またはこれ以上ジャンプしない）
            if X[p, curr_step] >= MAX_STATE_X:
                break

        # エイリアシング誤差の計算 (区間内に2回以上ジャンプがあれば見逃し発生)
        for i in range(len(inspection_points)-1):
            start_idx = inspection_points[i]
            end_idx = inspection_points[i+1]
            jumps_in_interval = np.sum(jump_counts[p, start_idx:end_idx])
            if jumps_in_interval > 1:
                aliased_jumps[p] += (jumps_in_interval - 1)

    return total_inspections, aliased_jumps, total_actual_jumps

inspections_1y, aliased_1y, actual_jumps = evaluate_strategy('Fixed_1Year')
inspections_1m, aliased_1m, _ = evaluate_strategy('Fixed_1Month')
inspections_dyn, aliased_dyn, _ = evaluate_strategy('Dynamic_Trigger')

# エラー率(%)の計算
valid = actual_jumps > 0
error_1y = np.sum(aliased_1y[valid]) / np.sum(actual_jumps[valid]) * 100
error_1m = np.sum(aliased_1m[valid]) / np.sum(actual_jumps[valid]) * 100
error_dyn = np.sum(aliased_dyn[valid]) / np.sum(actual_jumps[valid]) * 100

avg_insp_1y = np.mean(inspections_1y)
avg_insp_1m = np.mean(inspections_1m)
avg_insp_dyn = np.mean(inspections_dyn)

df_results = pd.DataFrame({
    'Strategy': ['Static (1 Year)', 'Static (1 Month)', 'Dynamic (State C Trigger)'],
    'Avg_Inspections_per_5Years': [avg_insp_1y, avg_insp_1m, avg_insp_dyn],
    'Jump_Miss_Rate_Error_Pct': [error_1y, error_1m, error_dyn]
})

# ==========================================
# 4. グラフの生成と保存
# ==========================================
filenames = []
print("グラフを生成しています...")

fig, ax1 = plt.subplots(figsize=(10, 6))

x_pos = np.arange(3)
width = 0.35

# 点検回数の棒グラフ (左軸)
bars1 = ax1.bar(x_pos - width/2, df_results['Avg_Inspections_per_5Years'], width, color='royalblue', label='Avg Inspections (Cost)')
ax1.set_ylabel('Average Number of Inspections (Cost)', color='royalblue', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='royalblue')
ax1.set_ylim(0, max(df_results['Avg_Inspections_per_5Years']) * 1.2)

# エラー率の棒グラフ (右軸)
ax2 = ax1.twinx()
bars2 = ax2.bar(x_pos + width/2, df_results['Jump_Miss_Rate_Error_Pct'], width, color='tomato', label='PINN Error Rate (Risk)')
ax2.set_ylabel('Jump Miss Rate / AI Error (%)', color='tomato', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='tomato')
ax2.set_ylim(0, max(df_results['Jump_Miss_Rate_Error_Pct']) * 1.2)

# 安全基準ライン (raw文字列を使用してエラー回避)
ax2.axhline(3.0, color='black', linestyle='--', lw=2, label=r'Strict Safety Limit ($\pm 3\%$)')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_results['Strategy'], fontweight='bold')
plt.title('Cost vs Risk: Validation of Dynamic "State C Trigger" Strategy')

# 凡例の結合
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Dynamic_Strategy_Validation.png')
filenames.append('Dynamic_Strategy_Validation.png')
plt.close()

# CSVの保存
df_results.to_csv('Dynamic_Strategy_Results.csv', index=False)
filenames.append('Dynamic_Strategy_Results.csv')

# ==========================================
# 5. ZIP圧縮とダウンロード
# ==========================================
zip_filename = 'Dynamic_Strategy_Validation.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for f in filenames:
        zipf.write(f)

print(f"\n全処理が完了しました。{zip_filename} をダウンロードします...")
print("\n--- シミュレーション結果サマリー ---")
print(df_results.to_string(index=False))

if IN_COLAB:
    files.download(zip_filename)
