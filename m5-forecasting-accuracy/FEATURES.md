# M5 Feature Registry

特徴量の設計変更を version 管理するドキュメント。
各 version はパイプライン実行ごとの parquet スキーマに対応する。

---

## Version Log

| Version | Date | 変更概要 | RMSE | 備考 |
|---|---|---|---|---|
| v1 | 2026-03-15 | 初版: 48列 (lag/rolling/price/SNAP/store profile) | 2.1353 | 3モデル (FOODS/HOBBIES/HOUSEHOLD) |
| v2 | 2026-03-16 | +6列 (Step 12d 価格弾力性系) + store profile 既存29列 | 2.1357 | 3モデル, 新特徴量は importance 圏外 |
| v3 | 2026-03-18 | Step A修正版: `price_rolling_mean_56` 再生成 | 2.1324 | `value_gap` が Top 10 入り、大幅改善 |
| v4 | 2026-03-18 | Decision Edition: +2列, FOODS -3列, NON_FOODS -6列 | ? | SNAP切断 + 低寄与削除 |

---

## v4 — 79 Features (Current)

### v3 → v4 変更点
- **追加 (+2):** `snap_x_high_price`, `snap_x_low_price` (Phase 1.5)
- **FOODS モデル除外 (-3):** `deal_intensity`, `above_price_wall`, `days_since_spike` (Step 4: 低寄与)
- **NON_FOODS モデル除外 (-6):**
  - SNAP切断 (Step 2): `snap_active`, `days_since_snap`, `is_snap_first_weekend`
  - 低寄与削除 (Step 4): `luxury_pressure_x_payday`, `impulse_buy_index`, `event_consumption_type`
- parquet スキーマは変更なし — GPU 学習時に `drop_features` で除外

### モデル別の実効特徴量数
| モデル | parquet 列数 | 除外 | **実効** |
|---|---|---|---|
| FOODS | 79 | 3 | **76** |
| NON_FOODS | 79 | 6 | **73** |

---

## v3 — 77 Features

### 概要
- Phase 1 起源: 45列 (CSV → parquet ストリーミング処理)
- Phase 1.5 起源: 32列 (parquet 上で追加、train期間の集計ベース)
- Step 12d 新規: 6列 (価格弾力性・需要構造分析)
- モデル構成: FOODS (feature_fraction=0.7) / NON_FOODS (HOBBIES+HOUSEHOLD, ff=0.8)

### 全特徴量一覧

| # | Feature | Category | Phase | 意味 |
|---|---|---|---|---|
| 1 | wday | Calendar | 1 | 曜日 (1=土〜7=金) |
| 2 | month | Calendar | 1 | 月 (1〜12) |
| 3 | year | Calendar | 1 | 年 |
| 4 | event_name_1 | Event | 1 | イベント名1 (カテゴリエンコード済) |
| 5 | event_name_2 | Event | 1 | イベント名2 |
| 6 | event_type_1 | Event | 1 | イベント種別1 (Sporting/Cultural/National/Religious) |
| 7 | event_type_2 | Event | 1 | イベント種別2 |
| 8 | event_nearby | Event | 1 | イベント前後3日以内フラグ |
| 9 | lag_28 | Lag | 1 | 28日前の売上 |
| 10 | lag_35 | Lag | 1 | 35日前の売上 |
| 11 | lag_42 | Lag | 1 | 42日前の売上 |
| 12 | lag_56 | Lag | 1 | 56日前の売上 |
| 13 | roll_mean_7 | Rolling | 1 | 直近7日の売上平均 (28日シフト) |
| 14 | roll_mean_28 | Rolling | 1 | 直近28日の売上平均 (28日シフト) |
| 15 | roll_mean_56 | Rolling | 1 | 直近56日の売上平均 (28日シフト) |
| 16 | roll_std_7 | Rolling | 1 | 直近7日の売上標準偏差 |
| 17 | roll_std_28 | Rolling | 1 | 直近28日の売上標準偏差 |
| 18 | roll_std_56 | Rolling | 1 | 直近56日の売上標準偏差 |
| 19 | roll_median_7 | Rolling | 1 | 直近7日の売上中央値 (外れ値に頑健) |
| 20 | ewma_7 | Rolling | 1 | 指数平滑移動平均 (span=7) |
| 21 | ewma_28 | Rolling | 1 | 指数平滑移動平均 (span=28) |
| 22 | sell_price | Price | 1 | 販売価格 |
| 23 | discount_ratio | Price | 1 | 割引率 (1 - price/max_price) |
| 24 | price_change_rel | Price | 1 | 前日比価格変化率 |
| 25 | not_on_shelf | Price | 1 | 棚なしフラグ (sell_price欠損=未取扱) |
| 26 | price_rolling_mean_56 | Price | 1 | 売価の56日移動平均 (value_gapの材料) **NEW** |
| 27 | value_gap | Price Elasticity | 1.5 | (売価 - 売価MA56) / 売価MA56 **NEW** |
| 28 | value_gap_x_elasticity | Price Elasticity | 1.5 | value_gap × 弾力性 × Volatile判定 **NEW** |
| 29 | deal_intensity | Price Elasticity | 1.5 | max(0,-VG) × 弾力性 × SNAP × Volatile **NEW** |
| 30 | above_price_wall | Price Elasticity | 1.5 | 価格の壁フラグ (FOODS$5/HOB$8/HH$10超) **NEW** |
| 31 | price_rank_in_dept | Price Elasticity | 1.5 | 部門内の価格相対位置 (0=最安〜1=最高) **NEW** |
| 32 | snap_active | SNAP | 1 | SNAP支給日フラグ (州別統合) |
| 33 | snap_wday | SNAP | 1 | SNAP日×曜日 |
| 34 | days_since_snap | SNAP | 1 | SNAP支給日からの経過日数 |
| 35 | is_snap_first_weekend | SNAP | 1 | SNAP期間内の最初の週末 |
| 36 | event_consumption_type | Event | 1 | イベント消費クラスター (内食/外食/購買型) |
| 37 | impulse_buy_index | Event | 1 | せっかく買い指数 (イベント誘発衝動購買度) |
| 38 | day | Calendar | 1 | 日 (1〜31) |
| 39 | is_weekend | Calendar | 1 | 週末フラグ |
| 40 | is_month_end | Calendar | 1 | 月末フラグ |
| 41 | is_month_start | Calendar | 1 | 月初フラグ |
| 42 | is_christmas_nearby | Calendar | 1 | クリスマス前後 (12/23〜27) |
| 43 | payday_flag | Calendar | 1 | 給料日フラグ (1日/15日) |
| 44 | payday_weekend | Calendar | 1 | 給料日ウィンドウ×週末 |
| 45 | snap_first_10d | SNAP | 1 | SNAP日かつ月初10日以内 |
| 46 | zeros_last_28 | Demand Pattern | 1 | 直近28日のゼロ売上日数 |
| 47 | days_since_last_sale | Demand Pattern | 1 | 最後に売れた日からの経過日数 |
| 48 | days_since_spike | Demand Pattern | 1 | 直近スパイクからの経過日数 (補充サイクル) |
| 49 | store_poverty_index | Store Profile | 1.5 | 旧 snap_dependency_score。店舗のSNAP依存度から算出した住民の購買力指標 (r=-0.83 with HOBBIES ratio) |
| 50 | payroll_dependency_score | Store Profile | 1.5 | 店舗の給料日依存度 |
| 51 | weekend_intensity | Store Profile | 1.5 | 店舗の週末売上集中度 |
| 52 | luxury_affinity_score | Store Profile | 1.5 | 店舗のHOBBIES売上比率 (贅沢品親和性)。Poverty Index と強烈な負の相関 |
| 53 | price_sensitivity_index | Store Profile | 1.5 | 店舗の価格感度 (割引時/通常時 売上比) |
| 54 | pb_ratio | Store Profile | 1.5 | 店舗のPB品売上比率 (低価格帯シェア) |
| 55 | store_income_type | Store Profile | 1.5 | 店舗所得タイプ (0=SNAP型/1=給料日型/2=均衡型) |
| 56 | income_event_sensitivity | Store Profile | 1.5 | 店舗が最も反応するイベント種別 |
| 57 | cat_income_elasticity | Store×Cat | 1.5 | 店舗×カテゴリのイベント日売上リフト率 |
| 58 | cat_snap_sensitivity | Store×Cat | 1.5 | 店舗×カテゴリのSNAP感度 |
| 59 | cat_payday_sensitivity | Store×Cat | 1.5 | 店舗×カテゴリの給料日感度 |
| 60 | stockpiling_index | Store×Cat | 1.5 | 店舗のHOUSEHOLD買い溜め傾向 (28日自己相関) |
| 61 | weekday_density_ratio | Store×Dept | 1.5 | 店舗×部門の平日/週末売上比 |
| 62 | store_dept_wday_avg | Store×Dept | 1.5 | 店舗×部門×曜日の平均売上 (イベント/SNAP除外) |
| 63 | store_dept_premium_share | Store×Dept | 1.5 | 店舗×部門のプレミアム品売上比率 |
| 64 | snap_dep_interaction | Interaction | 1.5 | snap_active × SNAP依存度 |
| 65 | weekend_interaction | Interaction | 1.5 | is_weekend × 週末集中度 |
| 66 | snap_x_income | Interaction | 1.5 | snap_active × 所得タイプ |
| 67 | price_x_psi | Interaction | 1.5 | sell_price × 価格感度 |
| 68 | snap_x_pb | Interaction | 1.5 | snap_active × PB比率 |
| 69 | snap_cat_lift | Interaction | 1.5 | snap_active × カテゴリ別SNAP感度 |
| 70 | payday_cat_lift | Interaction | 1.5 | payday_flag × カテゴリ別給料日感度 |
| 71 | luxury_pressure | Interaction | 1.5 | sell_price × 給料日依存度 (所得制約の強さ) |
| 72 | luxury_pressure_x_payday | Interaction | 1.5 | luxury_pressure × 給料日フラグ |
| 73 | is_CA4 | Store Flag | 1 | CA_4店舗フラグ (特異な売れ方をする店) |
| 74 | CA4_x_evt_type | Store Flag | 1 | CA_4 × イベント消費タイプ |
| 75 | te_store_dept_lag28 | Target Encoding | 1.5 | 店舗×部門のターゲットエンコーディング (28日ラグ) |
| 76 | roll_mean_56_weighted | Rolling | 1.5 | roll_mean_56 / cat_income_elasticity |
| 77 | spike_hint | Demand Pattern | 1.5 | 当日の期待スパイク量 (SNAP/給料日/週末別) |
| 78 | snap_x_high_price | SNAP×Price | 1.5 | (sell_price >= $5) × snap_active **v4 NEW** (FOODS二極化: プチ贅沢) |
| 79 | snap_x_low_price | SNAP×Price | 1.5 | (sell_price <= $1) × snap_active **v4 NEW** (FOODS二極化: まとめ買い) |

### Feature Importance ランキング (v3 修正版)

#### FOODS Top 10
| Rank | Feature | Importance | 備考 |
|---|---|---|---|
| 1 | `ewma_28` | 1.51e+08 | 支配的 (~40%) |
| 2 | `roll_mean_56` | 4.24e+07 | |
| 3 | `roll_mean_56_weighted` | 3.94e+07 | |
| 4 | `roll_mean_28` | 1.40e+07 | |
| 5 | `roll_std_56` | 1.25e+07 | |
| 6 | `ewma_7` | 4.91e+06 | |
| 7 | `discount_ratio` | 4.69e+06 | |
| 8 | `price_rolling_mean_56` | 4.54e+06 | **NEW (Step 12d)** |
| 9 | `month` | 4.13e+06 | |
| 10 | `value_gap` | 3.47e+06 | **NEW (Step 12d)** |

#### NON_FOODS Top 10
| Rank | Feature | Importance | 備考 |
|---|---|---|---|
| 1 | `ewma_28` | 1.22e+08 | |
| 2 | `roll_mean_28` | 2.30e+07 | |
| 3 | `roll_mean_56_weighted` | 7.83e+06 | |
| 4 | `roll_mean_56` | 5.47e+06 | |
| 5 | `roll_std_56` | 3.13e+06 | |
| 6 | `month` | 2.03e+06 | |
| 7 | `value_gap` | 2.01e+06 | **NEW (Step 12d)** |
| 8 | `discount_ratio` | 1.79e+06 | |
| 9 | `days_since_last_sale` | 1.41e+06 | |
| 10 | `wday` | 1.35e+06 | |

#### Step 12d 新特徴量の順位 (FOODS)
| Feature | Rank | Importance | 備考 |
|---|---|---|---|
| `price_rolling_mean_56` | 8 | 4.54e+06 | 有効 |
| `value_gap` | 10 | 3.47e+06 | 有効 |
| `price_rank_in_dept` | 15 | 1.94e+06 | 有効 |
| `value_gap_x_elasticity` | 18 | 1.48e+06 | |
| `deal_intensity` | 49 | 2.37e+04 | 弱い |
| `above_price_wall` | 58 | 4.27e+03 | 弱い |

---

## 変更履歴の詳細

### v1 → v2 (2026-03-16)
**追加:** store_dept_wday_avg, store_dept_premium_share, weekday_density_ratio,
luxury_pressure, luxury_pressure_x_payday, event_consumption_type, impulse_buy_index,
days_since_snap, is_snap_first_weekend, is_CA4, CA4_x_evt_type 他
**結果:** RMSE 2.1353 → 2.1357 (微悪化)。新特徴量は importance 圏外。
roll_mean_56 が importance の50%を独占する構造は不変。

### v2 → v3 (2026-03-17)
**追加:** price_rolling_mean_56, value_gap, value_gap_x_elasticity,
deal_intensity, above_price_wall, price_rank_in_dept
**モデル変更:** 3モデル → 2モデル (FOODS ff=0.7 / NON_FOODS ff=0.8)
**結果:** RMSE 2.1357 → 2.1412 (悪化)。ただし Phase 1 parquet 未再生成のため
value_gap 系3特徴量がゼロ。price_rank_in_dept (#12) のみ有効に動作。
**要再実行:** 既存 parquet 削除 → Phase 1 再実行で price_rolling_mean_56 を含むスキーマを生成。

---

## 今後の検討 (Future Ideas)
- **snap_x_high_price**: FOODS かつ高価格帯 ($5-$11) × snap_active。SNAP支給日の「プチ贅沢」行動。
- **snap_x_low_price**: FOODS かつ超低価格帯 ($0.20) × snap_active。SNAP支給日の「まとめ買い」行動。
- **is_snap (Non-Foods からの除外)**: HOBBIES/HOUSEHOLD のモデルから SNAP 関連変数を削除し、因果の希釈を防ぐ。
- **is_snap_payday**: SNAP支給日かつ給料日の重なり（月による）に対する特異的な反応。
