# Gemini への報告: Claude Code 実行結果

報告日: 2026-03-19 (v7 + Payday Deep Dive)

---

## 最新状況サマリ

### RMSE・Kaggle スコア推移
| Version | Val RMSE | Kaggle Public | Kaggle Private | 備考 |
|---|---|---|---|---|
| v1 (ベースライン) | 2.1357 | - | - | 3モデル, roll_mean_56 支配 |
| v5 (削減) | 2.1263 | - | - | SNAP 13→2, 低寄与14列削除 |
| v6 (残差学習) | **2.1106** | 0.730 | 0.981 | FOODS残差学習, ewma_28独裁崩壊 |
| **v7 (正則化)** | 2.1256 | 0.736 | **0.842** | num_leaves=63, min_child=50 |

### 最大の成果
- **Private が 0.981 → 0.842 に大幅改善** (ハイパラ正則化のみで -0.139)
- Public/Private ギャップが 0.251 → 0.106 に半減

### 残る課題
- NON_FOODS が ewma_28 に 70倍依存したまま
- Private 0.842 はまだ改善余地あり (Top 100 は ~0.55)

---

## v7 Feature Importance

### FOODS モデル (残差学習 + 正則化)
| rank | feature | importance |
|---|---|---|
| 1 | month | 4.51e+07 |
| 2 | roll_mean_28 | 4.25e+07 |
| 3 | zeros_last_28 | 3.46e+07 |
| 4 | sell_price | 3.29e+07 |
| 5 | ewma_28 | 3.20e+07 |
| 6 | lag_28 | 2.68e+07 |
| 7 | **value_gap** | 2.46e+07 |
| 8 | discount_ratio | 2.42e+07 |
| 9 | price_rolling_mean_56 | 2.34e+07 |
| 10 | roll_median_7 | 2.26e+07 |

**特徴: importance が均等化** (#1/#10 = 2.0倍。v5 では 66倍だった)

### NON_FOODS モデル (通常学習, tweedie)
| rank | feature | importance |
|---|---|---|
| 1 | ewma_28 | 1.38e+08 |
| 2 | roll_mean_28 | 1.98e+07 |
| 3 | roll_std_28 | 2.43e+06 |
| 4 | month | 2.19e+06 |
| 5 | value_gap | 2.01e+06 |

**ewma_28 が依然 70倍独裁。** 次の Phase 2 で NON_FOODS にも残差学習を適用予定。

---

## Payday Deep Dive 分析結果 (新規)

### 分析1: SNAP 減衰 — FOODS vs NON_FOODS の決定的差異

SNAP 日を基準にした売上減衰を、Type-S (SNAP依存: WI_2, WI_3, CA_3) で計測:

| 経過日数 | FOODS | HOBBIES | HOUSEHOLD |
|---|---|---|---|
| day 0 (SNAP日) | 基準 | 基準 | 基準 |
| day 1 | -23.5% | -23.9% | -21.6% |
| day 5 | **-20.8%** | **-2.6%** | -5.3% |
| day 10 | **-27.3%** | **-5.7%** | -8.4% |
| day 14 | **-26.9%** | **-3.3%** | -3.9% |
| day 18 | +0.9% | **+47.4%** | +43.0% |

**結論:**
- **FOODS は SNAP 後ずっと -20〜27% で沈み続ける** → `days_since_snap` が極めて有効
- **HOBBIES/HOUSEHOLD は day 2 で即回復** → SNAP サイクルは無関係
- **HOBBIES/HOUSEHOLD は day 18-20 で +47〜55%** → 次の SNAP 前の先行購買
- 図: `figures/41_typeS_snap_vs_payday_decay.png`, `figures/42_snap_decay_foods_vs_nonfood.png`

### 分析2: Payday (給料日) 減衰 — dept 別・store_type 別 (平日のみ)

週末効果を排除し、純粋な Salary 影響を dept 単位で計測:

| dept | 内容 | Payroll店 | SNAP店 | Independent店 | Salary感度 |
|---|---|---|---|---|---|
| **HOBBIES_1** | おもちゃ・ゲーム | **-11.2%** | **-15.0%** | -8.3% | **高い** |
| HOBBIES_2 | スポーツ用品 | -3.1% | +2.0% | +2.4% | **ゼロ** |
| **HOUSEHOLD_1** | 洗剤・紙製品 | **-10.8%** | **-17.5%** | -9.5% | **高い** |
| HOUSEHOLD_2 | 調理器具・家具 | -8.0% | -9.6% | -5.7% | 中程度 |

**結論:**
- **HOBBIES_1 (おもちゃ) と HOUSEHOLD_1 (消耗品) が強く Salary に反応**
- **HOBBIES_2 (スポーツ) は Salary に無反応** — 計画購買品
- **SNAP 店が最大の減衰** (-15〜17.5%) — 低所得ほど給料日サイクルに支配される
- 図: `figures/46_dept_payday_decay_weekday.png`

### 分析3: FOODS の価格帯別 Salary 感度

| 価格帯 | 全体減衰 | Payroll店 | SNAP店 | Independent店 |
|---|---|---|---|---|
| $0-1 | -11.9% | -18.0% | -10.8% | -6.2% |
| $1-3 | -9.9% | -14.7% | -8.1% | -6.2% |
| $3-5 | -9.4% | -15.3% | -6.7% | -6.8% |
| $5-10 | -11.3% | -17.8% | -8.0% | -6.9% |
| **$10+** | **-18.1%** | **-26.7%** | -10.9% | **-13.5%** |

**結論:**
- **$10+ の食品が最大の Salary 感度** — 全 store_type で最も減衰が激しい
- **Payroll 店 × $10+: -26.7%** — 最強のシグナル。「給料日にだけ高級食材を買う」層
- **Independent 店でも $10+: -13.5%** — 高所得でも高級食材は給料日を意識
- 図: `figures/47_foods_price_payday_decay.png`

### 分析4: 「給料日後の最初の週末」の効果

| カテゴリ | Payday Weekend Premium | Cohen's d | 判定 |
|---|---|---|---|
| HOBBIES | +2.5% | 0.014 | **negligible** |
| HOUSEHOLD | +3.9% | 0.022 | **negligible** |

**p 値は有意 (p<0.001) だが効果量はゼロに等しい。** 週末効果 (+24〜48%) に比べて +2〜4% は誤差。
→ `payday_weekend_window` フラグは不要。

### 分析5: HOUSEHOLD 必需品 Budget/Premium 比率 (所得 proxy)

| Store | Budget% | Premium% | B/P比 | snap_lift |
|---|---|---|---|---|
| WI_2 (最低所得) | 55.6% | 17.9% | 3.11 | 1.329 |
| CA_4 (最高所得) | 44.3% | 24.5% | 1.80 | 1.047 |

**B/P 比 × SNAP lift: r = 0.70** — SNAP だけでは捉えきれない所得の別側面を補完。
→ `store_bp_ratio` として Phase 1.5 に追加済み。

---

## 次の実装プラン (Phase 2)

### 実装済み (Colab 実行待ち)
| 特徴量 | 箇所 | 内容 |
|---|---|---|
| `relative_trend_28_56` | Phase 1 | roll_mean_28 / roll_mean_56 (トレンド方向) |
| `days_since_payday` | Phase 1 | 1日/15日からの最短距離 (0-14) |
| `store_bp_ratio` | Phase 1.5 | HOUSEHOLD B/P 売上比率 (所得 proxy) |
| NON_FOODS 残差学習 | GPU | target=sales-roll_mean_28, objective=regression |

### 検討中 (Gemini の意見を求める)
上記 EDA の結果を踏まえ、以下の **「給料日感度スコア」** の事前計算を検討中:

| 特徴量 | 定義 | 粒度 | 根拠 |
|---|---|---|---|
| `item_payday_sensitivity` | item ごとの payday日平均 / 非payday日平均 | ~3,049値 | HOBBIES_2(+2%) vs HOUSEHOLD_1(-17.5%) を item 単位で弁別 |
| `store_payday_decay` | store ごとの dsp=0 平均 / dsp=13 平均 | 10値 | Payroll店(-27%) vs Independent店(-13%) の差 |
| `dept_payday_sensitivity` | dept ごとの payday lift | 7値 | HOBBIES_1(-11%) vs HOBBIES_2(-3%) |

**設計思想:** `item_elasticity` (価格弾力性) が `value_gap` の効果を増幅したのと同じパターン。
tree が `days_since_payday` で split する際に、「この item/store は給料日に敏感か」を事前に知っていれば、split の gain が跳ね上がる。
手動の交差特徴量 (A×B) は LightGBM では冗長だが、**item 固有の感度スコア** は tree の探索を効率化する。

### 質問
1. `item_payday_sensitivity` の設計方針は妥当か？
2. 上記3つの感度スコアに加えて、他に有効な切り口はあるか？
3. NON_FOODS 残差学習のリスク (間欠需要でのノイズ) について見解は？

---

## 参考ファイル
- `FEATURES.md`: 全特徴量一覧 (v6: 82列予定)
- `PROCESS.md`: 作業履歴
- `CLAUDE_INSTRUCTIONS.md`: 実装指示 (v7: Step D-F)
- 図一覧: `figures/35-47_*.png`
