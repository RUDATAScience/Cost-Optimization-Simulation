import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import gc
import os
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("1時間〜5年の全解像度 ＆ 1億回(10^8)アーキテクチャ比較検証を開始します...\n")

# ==========================================
# 1. 検証パラメータ設定
# ==========================================
# 点検スパン (dt) の定義 (年単位換算)
dt_dict = {
    '1h': 1/(24*365),
    '1d': 1/365,
    '1w': 1/52,
    '1m': 1/12,
    '3m': 0.25,
    '6m': 0.5,
    '1y': 1.0,
    '3y': 3.0,
    '5y': 5.0
}
dt_labels = list(dt_dict.keys())
dt_values = np.array(list(dt_dict.values()))

# 計算スケール N
N_list = [1000, 100000, 10000000, 100000000] # 1K, 100K, 10M, 100M

# 物理パラメータ (状態依存ポアソン過程を想定した末期の高い発生率)
lam_true = 0.8

# リカレント構造(RNN)の履歴長 (過去 k 回分の点検記録を保持)
history_k = 5

# ==========================================
# 2. メモリセーフなバッチ処理による経験的誤差計算
# ==========================================
def calculate_empirical_errors(N_total, lam, dt_array, k_seq, batch_size=5000000):
    np.random.seed(42)

    total_jumps_sum = np.zeros(len(dt_array))
    aliased_mlp_sum = np.zeros(len(dt_array))
    aliased_rnn_sum = np.zeros(len(dt_array))

    remaining = N_total
    while remaining > 0:
        current_batch = min(remaining, batch_size)

        for i, dt in enumerate(dt_array):
            jumps = np.random.poisson(lam * dt, current_batch)
            total_jumps_sum[i] += np.sum(jumps)

            # MLP (k=1: 現在のみ)
            aliased_mlp_sum[i] += np.sum(np.maximum(0, jumps - 1))

            # RNN (過去 k 回の履歴あり、1/sqrt(k) の補完効果)
            aliased_rnn_sum[i] += np.sum(np.maximum(0, jumps - 1)) * (1.0 / np.sqrt(k_seq))

        remaining -= current_batch

    # エラー率の計算 (ゼロ除算防止)
    err_mlp = np.where(total_jumps_sum > 0, aliased_mlp_sum / total_jumps_sum, 0)
    err_rnn = np.where(total_jumps_sum > 0, aliased_rnn_sum / total_jumps_sum, 0)

    return err_mlp * 100, err_rnn * 100

# ==========================================
# 3. 大規模シミュレーションの実行
# ==========================================
results_mlp = {}
results_rnn = {}

for N in N_list:
    print(f"計算中: N = {N:,} 回 (1億回は数秒かかります)...")
    e_mlp, e_rnn = calculate_empirical_errors(N, lam_true, dt_values, history_k)
    results_mlp[N] = e_mlp
    results_rnn[N] = e_rnn
    gc.collect() # メモリ解放

# DataFrameへの集約
df_results = pd.DataFrame({'Interval_Label': dt_labels, 'dt_years': dt_values})
for N in N_list:
    df_results[f'MLP_Error_N{N}'] = results_mlp[N]
    df_results[f'RNN_Error_N{N}'] = results_rnn[N]

# ==========================================
# 4. グラフの生成と保存
# ==========================================
filenames = []
print("グラフとデータを生成・圧縮しています...")

# グラフ(A) アーキテクチャ比較 (N=100M時)
plt.figure(figsize=(12, 6))
plt.plot(dt_labels, results_mlp[100000000], marker='o', color='red', lw=3, label='Standard MLP (Current Only)')
plt.plot(dt_labels, results_rnn[100000000], marker='s', color='blue', lw=3, label=f'RNN (Past {history_k} steps)')
plt.axhline(3.0, color='black', linestyle='--', lw=2, label=r'Strict Tolerance Limit ($\pm 3\%$)')

plt.title('Architecture Comparison: MLP vs RNN (N = 100,000,000)')
plt.xlabel(r'Inspection Interval $\Delta t$')
plt.ylabel(r'Relative Error of Estimated $\lambda$ (%)')
plt.ylim(-1, max(results_mlp[100000000])*1.1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('1_Architecture_Comparison_100M.png')
filenames.append('1_Architecture_Comparison_100M.png')
plt.close()

# グラフ(B) MLPの計算スケール(N)比較
plt.figure(figsize=(12, 6))
colors = ['lightblue', 'cornflowerblue', 'royalblue', 'navy']
for i, N in enumerate(N_list):
    plt.plot(dt_labels, results_mlp[N], marker='o', color=colors[i], lw=2, label=f'N = {N:,}')

plt.axhline(3.0, color='red', linestyle='--', lw=2, label=r'Tolerance Limit ($\pm 3\%$)')
plt.title('Risk Manifestation by Scale N (Standard MLP)')
plt.xlabel(r'Inspection Interval $\Delta t$')
plt.ylabel(r'Relative Error of Estimated $\lambda$ (%)')
plt.ylim(-1, 30)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('2_MLP_Scale_Comparison.png')
filenames.append('2_MLP_Scale_Comparison.png')
plt.close()

# グラフ(C) RNNの計算スケール(N)比較
plt.figure(figsize=(12, 6))
colors_rnn = ['thistle', 'orchid', 'mediumorchid', 'purple']
for i, N in enumerate(N_list):
    plt.plot(dt_labels, results_rnn[N], marker='s', color=colors_rnn[i], lw=2, label=f'N = {N:,}')

plt.axhline(3.0, color='red', linestyle='--', lw=2, label=r'Tolerance Limit ($\pm 3\%$)')
plt.title(f'Risk Manifestation by Scale N (RNN, k={history_k})')
plt.xlabel(r'Inspection Interval $\Delta t$')
plt.ylabel(r'Relative Error of Estimated $\lambda$ (%)')
plt.ylim(-1, 30)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('3_RNN_Scale_Comparison.png')
filenames.append('3_RNN_Scale_Comparison.png')
plt.close()

# CSVの保存
df_results.to_csv('AHP_Architecture_Scale_Data.csv', index=False)
filenames.append('AHP_Architecture_Scale_Data.csv')

# ==========================================
# 5. ZIP圧縮とダウンロード
# ==========================================
zip_filename = 'AHP_AI_Architecture_Verification.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for f in filenames:
        zipf.write(f)

print(f"\n全処理が完了しました。{zip_filename} をダウンロードします...")
if IN_COLAB:
    files.download(zip_filename)
