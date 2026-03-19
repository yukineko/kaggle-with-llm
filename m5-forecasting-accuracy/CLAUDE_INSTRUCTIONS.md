# Claude Code Instructions for M5 Forecasting Accuracy (v7)

## 現在の状況 (2026-03-19)

### 達成済み
- Step 1-4 (SNAP切断・二極化・特徴量削減): **完了** → RMSE 2.1263
- Step C (FOODS 残差学習): **完了** → RMSE 2.1106, Feature Importance 均等化
- Kaggle 提出: Public 0.730, **Private 0.981** (汎化に課題)

### 最大の課題
**Public/Private ギャップ (+0.25)**: Val 期間への過適合。
tree が深すぎて Val 期間の固有パターンを暗記している。

---

## Step D: ハイパーパラメータ正則化 (Private スコア改善)

### 背景
Private 0.98 の主因は「tree の複雑さ」。特徴量の追加/復活ではなく、
モデルの容量を制限するのが汎化への最短ルート。

### 指示
1. **FOODS モデル** の LightGBM パラメータを変更:
   - `num_leaves`: 127 → 63
   - `min_child_samples`: 20 → 50
   - `feature_fraction`: 0.8 → 0.7
2. **NON_FOODS モデル** も同様:
   - `num_leaves`: 127 → 63
   - `min_child_samples`: 20 → 50
3. 再学習後、Val RMSE と Kaggle Public/Private スコアを比較。

---

## Step E: relative_trend_28_56 の導入

### 背景
roll_mean_56 は残差学習後も #8 で有用だが、roll_mean_28 との情報重複が激しい。
比率化により「トレンドの方向」を1値で表現できる。

### 指示
1. preprocess.py / pipeline_cpu.ipynb の Phase 1 に追加:
   `relative_trend_28_56 = roll_mean_28 / (roll_mean_56 + 1e-8)`
2. roll_mean_56 を維持したまま追加し、importance を比較。
3. relative_trend が roll_mean_56 を上回れば、roll_mean_56 を削除候補に。

---

## Step F: NON_FOODS 残差学習 (実験的)

### 背景
NON_FOODS で ewma_28 が 16倍独裁のまま。FOODS と同じ残差学習を試す。
ただし間欠需要 (ゼロ売上が多い) で残差学習が機能するかは不確実。

### 指示
1. NON_FOODS に `residual_target: True` を設定。
2. objective を `tweedie` → `regression` に変更 (残差は負になるため)。
3. RMSE が悪化した場合は即座に revert し、通常学習に戻す。

---

## 記録と報告
各ステップの実行後、RMSE・Feature Importance Top 10・Kaggle スコアを
[`PROCESS.md`](m5-forecasting-accuracy/PROCESS.md) に追記。
