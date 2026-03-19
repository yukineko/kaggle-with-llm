# Gemini への報告: Claude Code 実行結果

報告日: 2026-03-19 (v6 + Step 13 EDA)

---

## 最新状況サマリ

### RMSE 推移
| Version | Val RMSE | 変更 | Kaggle Public | Kaggle Private |
|---|---|---|---|---|
| v1 (ベースライン) | 2.1357 | 3モデル, roll_mean_56 支配 | - | - |
| v3 (Step A修正) | 2.1324 | parquet再生成, value_gap Top 10 | - | - |
| v4 (Decision) | 2.1327 | SNAP切断 + snap_x_high/low | - | - |
| v5 (削減) | 2.1263 | SNAP 13→2, 低寄与14列削除 | - | - |
| **v6 (残差学習)** | **2.1106** | FOODS: target=sales-roll_mean_28 | **0.730** | **0.981** |

### 最大の課題
**Public/Private ギャップ (+0.25)**: Val 期間への過適合。WRMSSE 最適化未実施。

---

## v6 Feature Importance (残差学習後)

### FOODS モデル (target = sales - roll_mean_28, objective = regression)
| rank | feature | importance |
|---|---|---|
| 1 | month | 5.68e+07 |
| 2 | roll_mean_28 | 4.41e+07 |
| 3 | zeros_last_28 | 3.94e+07 |
| 4 | sell_price | 3.91e+07 |
| 5 | lag_28 | 3.36e+07 |
| 6 | discount_ratio | 3.14e+07 |
| 7 | **value_gap** | **3.14e+07** |
| 8 | roll_mean_56 | 3.04e+07 |
| 9 | roll_median_7 | 2.92e+07 |
| 10 | wday | 2.64e+07 |

**ewma_28 が Top 10 から消失。** importance が均等化 (#1/#10 比 = 2.2倍、v5 では 66倍)。

### NON_FOODS モデル (target = sales, objective = tweedie)
| rank | feature | importance |
|---|---|---|
| 1 | ewma_28 | 1.42e+08 |
| 2 | roll_mean_28 | 8.73e+06 |
| 3 | roll_mean_56_weighted | 4.74e+06 |
| 4 | roll_mean_56 | 3.47e+06 |
| 5 | roll_std_56 | 3.07e+06 |
| 6 | value_gap | 2.02e+06 |
| 7 | month | 1.98e+06 |
| 8 | discount_ratio | 1.81e+06 |
| 9 | sell_price | 1.67e+06 |
| 10 | days_since_last_sale | 1.64e+06 |

NON_FOODS は ewma_28 が依然として独裁 (16倍)。

---

## Step 13: 家計セグメンテーション EDA 結果

### 店舗セグメント分類 (snap_lift × payday_lift)
| Store | snap_lift | payday_lift | Segment |
|---|---|---|---|
| CA_1 | 1.098 | 1.031 | Type-B (Balanced) |
| CA_2 | 1.024 | 1.040 | Type-B |
| CA_3 | 1.108 | 1.027 | Type-S (SNAP) |
| CA_4 | 1.047 | 1.026 | Type-B |
| TX_1 | 1.127 | 1.125 | Type-B |
| TX_2 | 1.104 | 1.094 | Type-P (Payroll) |
| TX_3 | 1.118 | 1.113 | Type-B |
| WI_1 | 1.050 | 1.048 | Type-B |
| WI_2 | 1.329 | 1.132 | Type-B |
| WI_3 | 1.252 | 1.085 | Type-B |

**問題: 中央値ベースの分割では 8/10 店が Type-B に集中。** WI_2 (snap_lift=1.33) と CA_2 (snap_lift=1.02) が同じ Type-B になってしまう。セグメント定義の再検討が必要。

### Payday Decay (給料日からの減衰)
全セグメントで、給料日→給料日間に **-5% ～ -12% の売上減衰** が確認された:

| Segment | FOODS decay | HOBBIES decay | HOUSEHOLD decay |
|---|---|---|---|
| Type-S (SNAP) | -8.0% | -8.5% | -5.6% |
| Type-P (Payroll) | **-12.1%** | -5.7% | -5.7% |
| Type-B (Balanced) | -8.8% | -7.3% | **-10.9%** |

**Type-P (Payroll) の FOODS が -12.1%** で最も大きな減衰。給料日依存店舗の食品需要は給料日に集中している証拠。

### 図一覧
| 図 | 内容 | パス |
|---|---|---|
| 35 | セグメント散布図 | `figures/35_segment_scatter.png` |
| 36 | 月内売上カーブ | `figures/36_monthly_curve.png` |
| 37 | Payday Decay | `figures/37_payday_decay.png` |
| 38 | カテゴリ構成比 | `figures/38_category_composition.png` |
| 40 | 月末枯渇 (平均単価) | `figures/40_depletion_effect.png` |

テキスト出力: `figures/step13_results.txt`

---

## 現在のモデル構成
- **FOODS**: 残差学習 (target=sales-roll_mean_28), objective=regression, 64特徴量
- **NON_FOODS**: 通常学習 (target=sales), objective=tweedie, 59特徴量
- **Step D** (実装済み): num_leaves=63, min_child_samples=50, FOODS ff=0.7

## 次のアクション候補
1. セグメント定義の改善 (WI_2/WI_3 は明らかに SNAP 型だが Type-B に分類されている)
2. `days_since_payday` 特徴量の導入 (decay が全セグメントで確認されたため)
3. NON_FOODS 残差学習の実験
4. WRMSSE カスタム objective / sample_weight の導入

## 参考ファイル
- `FEATURES.md`: 全特徴量一覧 (v5: 79列)
- `PROCESS.md`: 作業履歴
- `CLAUDE_INSTRUCTIONS.md`: 実装指示 (v7: Step D-F)
