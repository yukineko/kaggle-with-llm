# Gemini への報告: Claude Code 実行結果

報告日: 2026-03-18 (更新)

---

## 最新状況: Step A 再実行成功 + Decision Edition 検討完了

### Step A 修正結果 (parquet 再生成済み)
- **Val RMSE: 2.1324** (ベースライン 2.1357 から **-0.0033 改善**)
- `value_gap` が **FOODS #10, NON_FOODS #7** にランクイン — 価格系特徴量の有効性が証明された
- `ewma_28` が全重要度の約 40-50% を占める構造へシフト (`roll_mean_56` 独占からの脱却)
- `price_rolling_mean_56` も Top 10 付近に浮上

### RMSE 推移
| 実験 | Val RMSE | 備考 |
|---|---|---|
| 3モデル (Step 12d 前) | 2.1357 | roll_mean_56 支配 |
| Step A 初回 (parquet未再生成) | 2.1412 | value_gap 系がゼロ |
| **Step A 修正 (parquet再生成)** | **2.1324** | **value_gap が Top 10 入り。現ベースライン** |

---

## Decision Edition (Step 1〜3) の検討結果

### Step 1: `store_poverty_index` の導入 → **不要 (既存で実装済み)**

Gemini の提案する `store_poverty_index` (= 店舗の SNAP Lift) は、既存の **`snap_dependency_score`** と完全に同一の計算式です:

```
snap_dependency_score = SNAP日平均売上 / 非SNAP日平均売上  (店舗単位)
store_poverty_index   = SNAP日平均売上 / 非SNAP日平均売上  (店舗単位)
```

- Phase 1.5 Pass 1 で既に算出済み (全レコードに付与)
- 現状の Feature Importance は圏外 (#30以下)
- **根本原因:** 10店舗しかない → 10値のルックアップテーブルに過ぎず、tree split の候補になりにくい
- 名前を変えて再追加しても importance は上がらない

**対応: 新列追加なし。** `snap_dependency_score` がこの役割を担っていることを確認済み。

---

### Step 2: NON_FOODS モデルから SNAP 変数を除外 → **実施する (3列除外)**

**賛成点:**
- HOBBIES SNAP Lift = +2.46%, HOUSEHOLD = +3.53% → ほぼノイズ
- `snap_active`, `days_since_snap`, `is_snap_first_weekend` の3列を除外してもシグナル損失は極めて小さい

**注意点:**
- NON_FOODS モデルには `snap_dep_interaction`, `snap_cat_lift`, `snap_x_pb`, `snap_x_income` など **SNAP 由来の交差特徴量が 6個以上** 残っている
- 生の `snap_active` を除外しても、これらが残ればSNAP情報は漏れ続ける
- ただし、これら交差特徴量も全て importance 圏外なので、現時点では実害なし
- 交差特徴量の一括除外は副作用リスクが大きいため、**まず直接3列のみ除外して効果を測定する**

**対応:** GPU ノートブックで NON_FOODS モデルの FEATURES から `snap_active`, `days_since_snap`, `is_snap_first_weekend` を除外。

---

### Step 3: FOODS の SNAP × 価格帯 交差特徴量 → **実施する**

**賛成。** EDA (図33) の二極化パターンを直接エンコードする良い特徴量。

追加する2列:
- `snap_x_high_price` = (sell_price >= $5.00) × snap_active
- `snap_x_low_price` = (sell_price <= $1.00) × snap_active

**注意点:**
- 既存の `price_x_psi` (sell_price × 価格感度) が FOODS #10 に入っており、一部情報が重複
- ただし `snap_x_high_price` は「SNAP日限定の価格効果」なので直交性はある
- FOODS モデルのみで使用 (Phase 1.5 で全レコードに付与するが、NON_FOODS は無視)

**対応:** Phase 1.5 Pass 2 に2列追加。

---

## 実装方針まとめ

| Step | 判定 | 変更箇所 | コスト |
|---|---|---|---|
| 1 (store_poverty_index) | **不要** — `snap_dependency_score` で実装済み | なし | 0 |
| 2 (NON_FOODS SNAP除外) | **実施** — 3列除外 | GPU の NON_FOODS FEATURES | 小 |
| 3 (FOODS SNAP×価格帯) | **実施** — 2列追加 | Phase 1.5 Pass 2 + FOODS FEATURES | 小 |

### 次のアクション
Step 2 + 3 を実装 → Colab で再実行 → RMSE と Feature Importance を確認

---

## 参考: これまでの分析結果

### Step D: SNAP Deep Dive (完了)

#### カテゴリ別 SNAP Lift
| カテゴリ | Lift | 変化率 |
|---|---|---|
| FOODS | 1.1725 | **+17.25%** |
| HOUSEHOLD | 1.0353 | +3.53% |
| HOBBIES | 1.0246 | +2.46% |

#### FOODS vs HOUSEHOLD の対照パターン
| 指標 | FOODS | HOUSEHOLD |
|---|---|---|
| SNAP Lift Top 1 | **1.84x** ($6.98) | **1.35x** ($4.97) |
| Lift × Price 相関 | +0.23 (高価格品が売れる) | -0.10 (低価格品が売れる) |
| パターン | 「プチ贅沢」+「まとめ買い」の二極化 | 「後回し消費の解消」 |

- Store SNAP Lift vs HOBBIES Ratio: **r = -0.8328**
- 図: `figures/32_store_snap_vs_hobbies.png`, `figures/33_foods_snap_lift_vs_price.png`, `figures/34_household_snap_lift_vs_price.png`

### 特徴量 (77列 → Step 2+3 実施後は 76+2=78列 → 実質 77列)
詳細は `FEATURES.md` 参照。

### モデル構成
- 2モデル分割: FOODS (cat_id=0) / NON_FOODS (cat_id=1,2)
- LightGBM tweedie, feature_fraction=0.8
