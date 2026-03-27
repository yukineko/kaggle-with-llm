# M5 Forecasting Accuracy - Analysis & Instruction Process Log

## プロジェクト概要

| 項目 | 内容 |
|---|---|
| コンペ | [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) |
| 目標 | Walmart 30,490 SKU × 10店舗の日次売上を28日先まで予測 (WRMSSE) |
| データ | calendar.csv, sales_train_evaluation.csv, sell_prices.csv, sample_submission.csv |
| モデル | LightGBM (予定) |
| 現在のフェーズ | EDA完了 → 前処理実装済み → **Colab学習待ち** |

---

## ファイル構成

```
m5-forecasting-accuracy/
├── eda.ipynb              # EDA (Step 1〜12d, 55 cells)
├── preprocess.py          # ストリーミング前処理スクリプト (ローカル実行用)
├── preprocess.ipynb       # preprocess.py のノートブック版
├── features.py            # 特徴量生成モジュール (旧版、pipeline.ipynb に統合済み)
├── pipeline.ipynb         # 統合パイプライン (Phase 1→1.5→2→Training→Eval→Submit)
├── pipeline_cpu.ipynb     # CPU用: Phase 1→1.5→2 (前処理のみ)
├── pipeline_gpu.ipynb     # GPU用: Training→Eval→Submit
├── run_eda_step7.py       # EDA Step 7 一括実行スクリプト
├── figures/               # EDA 出力図 (28枚)
├── PROCESS.md             # 本ファイル
├── FEATURES.md            # 特徴量レジストリ (version管理)
└── *.csv                  # 元データ
```

---

## 2026-03-12: 「店舗性格診断」の深化と特徴量拡充戦略

### 1. 分析サマリ：住民の所得構造と購買行動の因果関係
- **コア仮説:** Walmartの売上は「個人の好み」よりも「現金の流動性（残高）」に支配されている。
- **特定された属性:**
    - **SNAP依存層:** 月初（1-10日）の支給日に売上が集中。
    - **Payroll依存層:** 15日・末日の給与日に反応。
    - **生活防衛層:** 価格変更（安売り）に対する反応（Price Sensitivity）が極めて高い。
- **現状評価:** RMSE 2.1353。roll_mean_56（長期平均）への依存を減らし、より解像度の高い「所得・価格」に関連する特徴量へのシフトが必要。

### 2. Claude Code への指示記録

#### 指示 No.1: 価格感度とPB依存度の実装
- **対象ファイル:** `preprocess.py`, `pipeline.ipynb`
- **指示内容:**
    1. **Price_Sensitivity_Index (PSI):** `d < 1886` 期間で `discount_ratio > 0.1` 時の売上リフト値を店舗別に算出。
    2. **PB_Ratio:** カテゴリ内下位20%価格帯（Great Value等）の売上構成比を算出。
    3. **交差特徴量:**
        - `price_x_psi_interaction`: (discount_ratio > 0.1) フラグ × PSI
        - `snap_x_pb`: snap_active × PB_Ratio
        - `price_x_psi`: sell_price × PSI
- **期待される結果:** WI_2（低所得型）などの店舗において、セール時の予測精度が向上し、PSIが Feature Importance の上位にランクインすること。

---

#### 指示 No.2: Zero-Sales 排除 + Luxury Pressure 導入 (実装済み)
- **対象ファイル:** `preprocess.py`, `pipeline.ipynb`
- **実装日:** 2026-03-12
- **指示内容:**

##### ① `not_on_shelf` フラグ（Zero-Sales の正体判別）
- **Phase 1:** `sell_price` が NaN = その週は物理的に未取扱い → `not_on_shelf` (int8) フラグを付与
- **Phase 2:** `not_on_shelf == 1` の行を train データから除外（val/eval はそのまま保持）
- **理由:** 「棚にない商品の売上 0」は需要ゼロではなく供給ゼロ。これを学習に含めるとモデルが「需要は0」と誤学習する。
- **不採用とした代替案:**
    - `is_active`（100日以上未販売フラグ）→ 閾値が恣意的。sell_price NaN のほうが物理的根拠が明確
    - SKU密度（store×category の品揃え数）→ 優先度3として保留

##### ② `luxury_pressure` 連続特徴量
- **Phase 1.5 Pass 2 で算出:**
    - `luxury_pressure = sell_price × payroll_dependency_score`
    - `luxury_pressure_x_payday = luxury_pressure × payday_flag`
- **設計意図:** 高価格 × 給料日依存度が高い店舗 = 所得制約が強く、給料日に高額品需要が集中する。この「購入圧力」を連続値で表現。
- **不採用とした代替案:**
    - 価格デシルによる離散的な閾値 → パス追加のコスト、閾値の不安定性、LightGBMが自動で分割点を見つけるため不要
    - Phase 1b（別パス）での実装 → Phase 1.5 Pass 2 に統合することで追加パス不要

##### ③ 後方互換性
- Phase 1.5 Pass 2 の先頭で `new_cols` チェック（既に列が存在すればスキップ）
- `not_on_shelf` がスキーマにない旧 parquet にはフォールバック計算を実行

---

#### 指示 No.3: 所得分布の「指紋（Signature）」解析（検討段階・未実装）
- **目的:** 同一ジャンル内の低価格帯 vs 高価格帯の売れ行きを対照させ、各店舗の住民所得分布を可視化・特徴量化する。
- **分析ステップ:**
    - **Step A:** カテゴリ内で商品を価格帯に分類（tercile推奨）
    - **Step B:** 各価格帯の SNAP期間リフト・給与日リフトを店舗ごとに集計
    - **Step C:** 「ゆとり境界（Affluence Threshold）」＝ 低価格帯は常に売れ、高価格帯は収入直後のみ売れる境界ラインを特定
- **Claude の見解:**

| 論点 | 判断 |
|---|---|
| 現行3群クラスタとの関係 | 置き換えではなく補完。現行は「いつ買うか」、本手法は「何を・いつ買うか」で直交的 |
| カテゴリ選択 | **FOODS 単独で十分**（購買頻度・価格連続性・即時性の3点で優位） |
| 価格帯分割 | **tercile（3分割）** で十分。デシルは10店舗×10ビン＝100セルでサンプル不足 |
| 特徴量の形 | `affluence_score = lift_high / lift_low`（1変数に圧縮） |
| リーケージ対策 | Phase 1.5 と同じ train 期間集約方式で安全（val/eval の sales は不使用） |
| リスク | 10店舗しかないため marginal gain が小さい可能性あり |
| 実装優先度 | **luxury_pressure の RMSE 効果を先に確認してから判断** |

- **補足（FOODS が HOUSEHOLD より強いシグナルである理由）:**
    1. 購買頻度が日次〜週次（HOUSEHOLD は月次〜隔月でリフト推定がノイジー）
    2. 価格帯が $1〜$15 で滑らかなグラデーション（HOUSEHOLD は二極化しやすい）
    3. 食品は即時需要で所得制約の影響が最も直接的に現れる

---

### 3. 分析・評価の基本指針 (M5 Project Standard)
- **分析対象の優先順位:** 「分析の依頼」があった際は、常に `pipeline.ipynb` の最新の実行出力を最優先で読み取り、評価する。
- **評価ポイント:**
    - 全体およびカテゴリ別の RMSE の変化。
    - Feature Importance における新規導入特徴量の順位と寄与度。
    - 特筆すべき店舗（WI_2 vs CA_1等）の予測挙動。
- **継続性:** 本プロジェクトが継続する限り、この「ノートブック出力に基づく定量的評価」を私の標準的な分析動作とする。

### 4. 特徴量の優先度ロードマップ

| 優先度 | 特徴量 | ステータス | 備考 |
|---|---|---|---|
| 1 | `not_on_shelf` + train除外 | **実装済み** | sell_price NaN ベース |
| 2 | `luxury_pressure` / `luxury_pressure_x_payday` | **実装済み** | 連続値、Phase 1.5 Pass 2 |
| 3 | SKU密度（store×category 品揃え数） | 保留 | Zero-Sales 関連の補助指標 |
| 4 | `affluence_score`（所得指紋） | 検討中 | luxury_pressure の効果確認後に判断 |
| 5 | 離散的 luxury threshold | 見送り | 連続値で代替済み |

---

## 2026-03-13: EDA Step 6-7 深化 + 構造ベースライン特徴量の実装

### 5. EDA 分析結果 (Step 6-7)

#### Step 6: イベント消費クラスター & 店舗特異性

- **イベント消費クラスター分類:**
  - A=Outing/Premium-Up (Easter, SuperBowl, LaborDay, ColumbusDay, VeteransDay)
  - B=Home-Party/Bulk-Up (IndependenceDay, Halloween, MemorialDay)
  - C=Closed (Christmas, Thanksgiving)
  - D=Others/No-Event
- **せっかく買い指数 (Impulse Buy Index):** Easter +36.7%, LaborDay +12.8% が突出。Christmas/Thanksgiving は店舗閉鎖のため 0。
- **CA_4 特異性:** イベント反応が他店舗と異なるパターン → `is_CA4` フラグ + `CA4_x_evt_type` 交差特徴量で対応。
- **SNAP 心理的ラグ:** 支給日当日よりも「最初の週末」に消費が集中 → `is_snap_first_weekend`, `days_since_snap` を導入。

#### Step 7a: イベント普遍性分析
- **ユニバーサルイベント:** LaborDay (+24.9%, std=6.5), Christmas (-99.9%, std=0.1)
- **リージョナルイベント:** Easter (std=18.5), SuperBowl (std=8.8) → 店舗間分散が大きい
- 分析スクリプト: `run_eda_step7.py` / 図: `figures/26_event_universality.png`

#### Step 7b: ラマダン深掘り分析
- CA_2 が最も感度が高い (score=5.16, FOODS_2 P75+ lift +10.4%)
- ただしラマダンはM5期間中の出現頻度が限定的
- 図: `figures/27_ramadan_deep_lift.png`

#### Step 7c: 価格プロファイリング
- **item_premium_flag:** dept内 Z-score > 2.0 の高価格アイテムを判定
- **store_dept_premium_share:** 高価格品の数量シェア → pb_ratio との相関 r=-0.176（独立 → 両方保持）
- **発見:** TX_1, TX_2 = "taste signal" stores（低 luxury_index だが高 premium_share）
- 図: `figures/28_price_profiling.png`

### 6. 指示 No.4: EDA知見の特徴量化 (実装済み)

#### ① Phase 1 追加特徴量 (process_chunk 内)
| 特徴量 | 型 | 説明 |
|---|---|---|
| `event_consumption_type` | int8 | イベント消費クラスター (0-3) |
| `impulse_buy_index` | float32 | せっかく買い指数 |
| `days_since_snap` | int16 | SNAP支給日からの経過日数 (州別→統合) |
| `is_snap_first_weekend` | int8 | SNAP期間内の最初の土日フラグ |
| `is_CA4` | int8 | CA_4 店舗フラグ |
| `CA4_x_evt_type` | int8 | CA_4 × イベントクラスター交差 (0-4) |

- **カレンダー前処理にも追加:** `days_since_snap_{CA,TX,WI}`, `is_snap_first_we_{CA,TX,WI}` を calendar DataFrame に事前計算

#### ② Phase 1.5 追加特徴量 (構造ベースライン)
| 特徴量 | 型 | 説明 |
|---|---|---|
| `store_dept_wday_avg` | float32 | store×dept×wday の「日常」平均売上 (イベント日・SNAP日を除外) |
| `store_dept_premium_share` | float32 | store×dept 内の高価格品 (Z>2.0) 数量シェア |
| `weekday_density_ratio` | float32 | store×dept の平日/週末 売上密度比 |

- **Pass 0b 追加:** `item_premium_flag` (dept内 Z-score > 2.0) を全 row_group スキャンで算出
- **Pass 1 追加:** `sdw_agg`, `sdps_agg`, `wdr_agg` の集計ループ (train期間のみ)
- **Pass 2 追加:** lookup からの列書き込み

#### ③ 実装方針
- `preprocess.py` と `pipeline.ipynb` の両方にインラインで同一ロジックを実装
- pipeline.ipynb は Colab 実行用（`import preprocess` ではなくインライン）
- 既存 parquet を削除して Phase 1 から再実行が必要

### 7. 検討中・未実装の分析と特徴量

#### 指示 No.3 (継続検討): 所得指紋 (Affluence Score)
- ステータス: **保留** — luxury_pressure + premium_share の効果確認後に判断
- 内容: カテゴリ内 tercile 分割 → 価格帯別 SNAP/給与日リフト → `affluence_score = lift_high / lift_low`

#### roll_mean_56 の要否検討
- **ユーザーの問題提起:** 56日周期の根拠は何か？入金イベントベースなら28日で十分では？
- **結論:** roll_mean_56 は「2ヶ月の季節トレンド」を捕捉する目的で有効だが、Feature Importance で確認後に判断
- **検証方法:** roll_mean_56 あり/なしで RMSE 比較（Colab 実行後に実施）

---

## 2026-03-13: Pipeline 分割 (CPU/GPU)

### 8. Notebook 分割

Colab の無料枠制約（GPU セッションのタイムアウト）に対応するため、`pipeline.ipynb` を3つに分割:

| Notebook | 用途 | 実行環境 |
|---|---|---|
| `pipeline.ipynb` | 統合版 (全ステップ一括) | Colab Pro / ローカル |
| `pipeline_cpu.ipynb` | 前処理のみ (Phase 1→1.5→2: CSV→parquet→split) | Colab CPU |
| `pipeline_gpu.ipynb` | 学習・評価・提出 (split files→LightGBM→submit) | Colab GPU |

- **分割の利点:** CPU前処理を先に完了 → Google Drive に保存 → GPU セッションを別途起動して学習。GPU時間を節約。
- **データ受け渡し:** `train_X.dat`, `train_y.dat`, `train_cat.dat`, `val.parquet`, `eval.parquet` を Google Drive 経由で共有。

---

## 2026-03-16: Colab 初回学習結果 + EDA Step 12 (価格弾力性・需要構造分析)

### 9a. LightGBM 初回学習結果

#### 全体スコア
| 指標 | 値 |
|---|---|
| **Val RMSE (全体)** | **2.1357** |
| FOODS RMSE | 2.5835 (n=446,348) |
| HOBBIES RMSE | 1.4500 (n=359,324) |
| HOUSEHOLD RMSE | 1.8207 (n=48,048) |

#### Feature Importance (全カテゴリ合算 Top 10)
| Rank | Feature | Importance |
|---|---|---|
| 1 | `roll_mean_56` | 1.86e+08 (**~50% 独占**) |
| 2 | `roll_mean_28` | 1.30e+08 |
| 3 | `roll_mean_56_weighted` | 5.50e+07 |
| 4 | `roll_std_56` | 3.00e+07 |
| 5 | `roll_mean_7` | 1.35e+07 |
| 6 | `sell_price` | 1.34e+07 |
| 7 | `discount_ratio` | 1.04e+07 |
| 8 | `days_since_last_sale` | 8.54e+06 |
| 9 | `month` | 8.15e+06 |
| 10 | `price_x_psi` | 4.24e+06 |

#### 診断
- **roll_mean_56 が全 importance の約50%を独占** → ナイーブ予測 (過去平均≒将来) に近い状態
- 新規特徴量で Top 10 入りしたのは `price_x_psi` (FOODS #10) のみ
- `luxury_pressure`, `event_consumption_type`, `impulse_buy_index`, `days_since_snap`, `is_snap_first_weekend` は全て圏外
- **問題:** roll_mean 系が他の特徴量の学習機会を奪っている

---

### 9b. EDA Step 12: 価格弾力性・需要構造分析

roll_mean_56 依存を打破するため、価格因子を3つの観点から深掘り分析。

#### Step 12a: アイテム別 Price Elasticity
- **結果:** 9,929 item×store ペアで弾力性を算出
  - mean=-3.324, **median=-0.283** (弱い負の弾力性), std=67.7 (極端にばらつく)
- **強い弾力性 (値下げで爆売れ):** FOODS_3 に集中 ($0.82-$7.24 の低価格帯)
- **価格不感応品:** HOUSEHOLD_1, FOODS_2 に多い (生活必需品・定番品)
- 図: `figures/29_price_elasticity_analysis.png`

#### Step 12b: 「価格の壁」と家計収支の限界分析
- **明確な壁:** $2-3 (NZ Rate -20%), $5-8 (-18%), **$20-50 (-38~48%)**
- **Q5「死に筋」定量化 (Low Income):**
  - HOUSEHOLD_2: Q5 NZ Rate = 0.1214, Q1 比 **48.1% 抑制** (最悪)
  - FOODS_3: Q5 NZ Rate = 0.3726, Q1 比 31.8% 抑制 (最良)
- **所得格差は意外と小さい** (High vs Low の Q5 差: 0.5-5pp)
  - → `above_price_wall` は所得クラスタ問わず全店で有効
- 図: `figures/30_price_wall_analysis.png`

#### Step 12c: アンカリング（価格の慣れ）の検証
- **重要な発見 — Stable品は逆パターン:**
  - Stable (CV<2%): 値下げ時 **-12.2%**, 値上げ時 **+33.0%**
  - 原因: Stable品=生活必需品 (牛乳等) で、値上げ=インフレ期=消費増の confounding
- **Volatile品 (CV>5%) は期待通り:**
  - 値下げリフト **+62.0%**, 値上げ時 +47.8%
- **結論:** `value_gap` は **Volatile品限定** で有効。Stable品には逆効果のリスクあり。
  - → `deal_intensity`, `value_gap_x_elasticity` は `price_cv > 0.05` の条件付きで適用すべき
- 図: `figures/31_price_anchoring.png`

#### Step 12d: 新・複合特徴量の提案 (6種)

| # | 特徴量 | 計算式 | 設計意図 |
|---|---|---|---|
| 1 | `value_gap` | `(price - price_MA8w) / price_MA8w` | アンカリング効果 (Volatile品で有効) |
| 2 | `value_gap_x_elasticity` | `value_gap × item_elasticity` | 弾力性の高い商品のセール効果を強調 |
| 3 | `price_rank_in_dept` | `(price - dept_min) / (dept_max - dept_min)` | 部門内の市場ポジション |
| 4 | `above_price_wall` | `1 if price > wall_threshold else 0` | カテゴリ×所得別の不連続点フラグ |
| 5 | `deal_intensity` | `max(0, -VG) × elasticity × snap` | セール×弾力性×SNAP の三重交差 |
| 6 | `price_memory_ratio` | `price / item_max_price` | 過去最高値に対する位置 |

#### roll_mean_56 脱却の段階的戦略
- **Step A:** 上記6特徴量を追加 + roll_mean_56 維持 → RMSE 変化を確認
- **Step B:** roll_mean_56 を削除 → 価格系の importance 浮上を確認
- **Step C:** 残差学習: `target = sales - roll_mean_56` として「平均からのズレ」を予測

---

## 2026-03-16: EDA Step 8-11 追加分析

### 9. EDA 分析結果 (Step 8-11)

#### Step 8: Item-level Periodicity Analysis
- `statsmodels.tsa.stattools.acf` を使用して商品レベルの自己相関を分析
- 週次（7日）周期の普遍性を確認

#### Step 9: Steady vs Burst 判別 (HOBBIES / HOUSEHOLD)
- **Step 9a:** 全商品を「安定需要型 (Steady)」と「バースト型 (Burst)」に分類
- **Step 9b:** 代表アイテムの時系列比較で Steady vs Burst の挙動差を可視化
- HOBBIES/HOUSEHOLD カテゴリで特にバースト型が多い

#### Step 10: 店舗クラスタリング深掘り
- **Step 10a:** 店舗クラスタリング + HOBBIES Non-zero Rate 検証
- **Step 10b:** 部門内ランキングの所得による逆転現象 (Spearman相関)
- **Step 10c:** 価格 Quintile 別の販売シェア分析 — 所得クラスタごとの購買パターンの違い

#### Step 11: 所得クラスタ × 価格構造の深層分析
- **Step 11a:** 所得クラスタ別「価格の壁」— 高価格帯での購入が落ちる閾値を特定
- **Step 11b:** HOBBIES ゼロ率解剖 — 間欠需要の原因構造を分析
- **Step 11c:** 店舗×部門別の売上変動ヒートマップ (Val期間 d_1886〜d_1913)

### 10. EDA 出力図一覧 (figures/)

| # | ファイル名 | 内容 |
|---|---|---|
| 01 | `01_total_daily_sales.png` | 全期間の合計売上時系列 |
| 02 | `02_weekday_month_sales.png` | 曜日別・月別の平均売上 |
| 03 | `03_event_sales.png` | イベント日の売上変動 |
| 04 | `04_hierarchy_sales.png` | 階層別売上 |
| 05 | `05_state_category_timeseries.png` | 州別カテゴリ別時系列 |
| 07 | `07_price_distribution.png` | 価格分布とカテゴリ別価格帯 |
| 10 | `10_zero_sales_distribution.png` | ゼロ売上分布 |
| 11 | `11_first_sale_date.png` | 初回販売日分布 |
| 12 | `12_zero_streak_distribution.png` | ゼロ連続日数分布 |
| 14 | `14_store_profiling.png` | 店舗プロファイリング |
| 15 | `15_snap_lift_vs_luxury.png` | SNAPリフト vs 高価格品比率 |
| 16 | `16_4quadrant_snap_weekend.png` | 4象限: SNAP×週末 |
| 17 | `17_income_sensitivity.png` | 所得感度 |
| 18 | `18_price_elasticity_correction.png` | 価格弾力性 |
| 19 | `19_payday_lag.png` | 給料日ラグ |
| 20 | `20_weekend_lift_decomposition.png` | 週末リフト分解 |
| 21 | `21_assortment.png` | 品揃え分析 |
| 22 | `22_weekday_density.png` | 平日/週末密度 |
| 23 | `23_visitor_attribution.png` | 来客帰属 |
| 24 | `24_variance_decomposition.png` | 分散分解 |
| 25 | `25_impulse_buy.png` | せっかく買い指数 |
| 26 | `26_event_universality.png` | イベント普遍性 vs 地域特異性 |
| 27 | `27_ramadan_deep_lift.png` | ラマダン深層リフト |
| 28 | `28_price_profiling.png` | 価格プロファイリング |
| 29 | `29_price_elasticity_analysis.png` | アイテム別価格弾力性 (6パネル) |
| 30 | `30_price_wall_analysis.png` | 価格の壁・家計制約分析 (6パネル) |
| 31 | `31_price_anchoring.png` | アンカリング・Value Gap 分析 (6パネル) |
| 32 | `32_store_snap_vs_hobbies.png` | SNAP リフト vs HOBBIES 売上比率 |
| 33 | `33_foods_snap_lift_vs_price.png` | FOODS SNAP リフト vs 価格帯分析 |
| 34 | `34_household_snap_lift_vs_price.png` | HOUSEHOLD SNAP リフト vs 価格帯分析 |

---

## 前処理パイプライン構造 (preprocess.py)

### Phase 1: CSV → df_features.parquet (ストリーミング)
- sales_train_evaluation.csv を chunksize 行ずつ読み込み
- melt (wide→long) → calendar マージ → 価格マージ → 特徴量計算
- カテゴリエンコーディング (グローバル辞書方式)
- 店舗所得クラスタリング (SNAP/給料日依存スコア → KMeans 3群)
- parquet 逐次書き出し

### Phase 1.5: parquet 上で追加特徴量
- **Pass 0:** item_max_price 事前計算 (prices チャンク読み)
- **Pass 0b:** item_premium_flag (dept内 Z-score > 2.0)
- **Pass 0c:** item_elasticity, item_price_cv, dept_price_range (Step 12d 新特徴量用)
- **Pass 1:** 集約テーブル構築 (store×dept×wday平均, premium_share, weekday密度比)
- **Pass 2:** row_group 単位で読み込み → lookup から列追加 → 上書き

### Phase 2: Train/Val/Eval 分割
- train: d < 1886 (not_on_shelf 除外)
- val: 1886 ≤ d < 1914
- eval: 1914 ≤ d ≤ 1941
- 出力: `train_X.dat`, `train_y.dat`, `train_cat.dat`, `val.parquet`, `eval.parquet`

---

## 特徴量の優先度ロードマップ (最新)

| # | 特徴量 | ステータス | 備考 |
|---|---|---|---|
| 1 | `not_on_shelf` + train除外 | **実装済み** | sell_price NaN ベース |
| 2 | `luxury_pressure` / `luxury_pressure_x_payday` | **実装済み** | importance 圏外 → 効果なし |
| 3 | `event_consumption_type` / `impulse_buy_index` | **実装済み** | importance 圏外 → 効果なし |
| 4 | `days_since_snap` / `is_snap_first_weekend` | **実装済み** | importance 圏外 → 効果なし |
| 5 | `is_CA4` / `CA4_x_evt_type` | **実装済み** | importance 圏外 → 効果なし |
| 6 | `store_dept_wday_avg` | **実装済み** | importance 圏外 → 効果なし |
| 7 | `store_dept_premium_share` | **実装済み** | importance 圏外 → 効果なし |
| 8 | `weekday_density_ratio` | **実装済み** | importance 圏外 → 効果なし |
| 9 | `price_x_psi` | **実装済み** | **FOODS #10, ALL #10** (唯一 Top 10 入り) |
| 10 | `price_rolling_mean_56` | **Step A 実装済み** | Phase 1 で算出 (value_gap の材料) |
| 11 | `value_gap` | **Step A 実装済み** | (price - price_MA56) / price_MA56 |
| 12 | `value_gap_x_elasticity` | **Step A 実装済み** | VG × elas × (CV>0.05) |
| 13 | `deal_intensity` | **Step A 実装済み** | max(0,-VG) × elas × snap × (CV>0.05) |
| 14 | `above_price_wall` | **Step A 実装済み** | FOODS $5, HOBBIES $8, HOUSEHOLD $10 |
| 15 | `price_rank_in_dept` | **Step A 実装済み** | (price - dept_min) / (dept_max - dept_min) |
| 16 | roll_mean_56 脱却 | **Step A: 維持** → Step B で検証 | Step A→B→C で段階実施 |
| 17 | SKU密度 | 保留 | 優先度低下 |
| 18 | `affluence_score` | 保留 | luxury_pressure 効果なしで優先度低下 |

---

## 2026-03-16: Step A 実装 (新特徴量 + 2モデル分割)

### 11. Step A: 実装内容

#### Phase 1 変更
- `price_rolling_mean_56` を `process_chunk()` に追加 (sell_price の56日ローリング平均)

#### Phase 1.5 変更 (Pass 0c 新設 + Pass 2 拡張)
- **Pass 0c:** train期間のみでアイテム弾力性・価格CV・部門価格レンジを算出
  - `item_elasticity`: cov(price, sales) / var(price) × mean_p / (mean_s + 1)
  - `item_price_cv`: sell_price の変動係数 (CV)
  - `dept_price_range`: 部門ごとの [min, max] 価格
  - `PRICE_WALL`: {FOODS: $5, HOBBIES: $8, HOUSEHOLD: $10} (EDA Step 12b)
- **Pass 2 追加:** 5特徴量を row_group 単位で算出
  - `value_gap`: (sell_price - price_rolling_mean_56) / price_rolling_mean_56
  - `value_gap_x_elasticity`: value_gap × item_elasticity × (price_cv > 0.05)
  - `deal_intensity`: max(0, -value_gap) × |item_elasticity| × snap_active × (price_cv > 0.05)
  - `above_price_wall`: sell_price > カテゴリ別壁閾値
  - `price_rank_in_dept`: (sell_price - dept_min) / (dept_max - dept_min)

#### GPU 変更 (2モデル分割)
- **FOODS モデル** (cat_id=0): `feature_fraction=0.8`
- **NON_FOODS モデル** (cat_id=1,2): HOBBIES + HOUSEHOLD 統合 (48K の HOUSEHOLD サンプル保護)
  - `num_boost_round=2500`, `early_stopping=80`

#### Volatile 条件付き適用
- `item_price_cv > 0.05` の商品のみ `value_gap_x_elasticity` と `deal_intensity` が非ゼロ
- Stable品 (CV<2%) のアンカリング逆効果を回避 (EDA Step 12c の知見)

---

## 2026-03-18: Step A 初回結果 + 修正

### Step A 初回結果 (parquet 未再生成)
- **RMSE: 2.1412** (前回 2.1357 → +0.0055 悪化)
- FOODS: 2.5914, HOBBIES: 1.4523, HOUSEHOLD: 1.8172 (改善)
- **value_gap 系3特徴量が importance=0.00** — `price_rolling_mean_56` が parquet に存在しなかった
- `price_rank_in_dept` は FOODS #12, NON_FOODS #13 で有効
- `feature_fraction=0.7` は悪化要因 → 0.8 に戻す

### Step A 修正結果 (2026-03-18)
- **Val RMSE (全体): 2.1324** (前回 2.1412 → 0.0088 改善、ベースライン 2.1357 からも改善)
  - FOODS: 2.5783
  - HOBBIES: 1.4513
  - HOUSEHOLD: 1.8120
- **重要度変化**:
  - `value_gap` が FOODS #10, NON_FOODS #7 にランクイン。価格系特徴量の有効性が証明された。
  - `ewma_28` が全重要度の約 40-50% を占める構造へシフト（`roll_mean_56` 独占からの脱却）。
  - `price_rolling_mean_56` も Top 10 付近に浮上。

---

## 2026-03-18: SNAP Deep Dive 分析結果

### 12. SNAP 依存度と需要構造の相関 (Deep Dive)

#### Store SNAP Lift vs HOBBIES Ratio
- **相関係数:** r = -0.8328 (強い負の相関)
- **洞察:** SNAP依存度（支給日の売上リフト）が高い店舗ほど、HOBBIESカテゴリの売上比率が低い。
- **背景:** 低所得商圏においては、可処分所得が食品などの必需品に集中し、嗜好品への支出が抑制されている構造が鮮明に現れている。
- 図: `figures/32_store_snap_vs_hobbies.png`

#### Item SNAP Lift vs Price (FOODS)
- **相関係数:** r = +0.2296 (弱い正の相関)
- **発見:** 意外にも、高価格帯の食品の方が SNAP 支給日のリフトが大きい傾向。
- **購買パターン:**
    - **「プチ贅沢」行動:** FOODS_2 (飲料等) の $5-$11 帯が SNAP リフト Top 20 に多数ランクイン。支給日に「普段買えない少し良い食品」を購入する行動パターンが示唆される。
    - **「大量まとめ買い」行動:** 一方で $0.20 帯の超低価格品 (FOODS_3) も Top 20 入りしており、支給日に生活必需品を安価に大量確保する動きも併存。
- 図: `figures/33_foods_snap_lift_vs_price.png`

#### Item SNAP Lift vs Price (HOUSEHOLD) — 分析完了
- **背景:** SNAP支給による家計内の現金余裕が HOUSEHOLD カテゴリに波及する仮説。
- **相関係数:** r = -0.1031 (弱い負の相関)
- **結論:** FOODS（1.8倍のリフト）に比べ、HOUSEHOLD は 1.0 付近（無反応）に密集。SNAP の直接的な影響は極めて限定的（ノイズに近い）。
- 図: `figures/34_household_snap_lift_vs_price.png`

### 13. 戦略的転換：因果関係の「外科的整理」 (Gemini's Insight)

#### 現状の課題
- `roll_mean_56` が Feature Importance の 50% を独占し、モデルが平均値に逃げている。
- **原因:** HOUSEHOLD/HOBBIES など、SNAP と事実上無関係なカテゴリにまで SNAP フラグを入れているため、情報の希釈（Dilution）が起き、モデルが因果を特定できなくなっている。

#### 新戦略：因果の物理的切断
1. **Poverty Index (住民購買力) の数値化**:
   - `store_id` という単なる「名前」ではなく、図32の強い負の相関 (r=-0.83) を活用した `store_poverty_index` (SNAP依存度) を導入し、地域の「背景」を直接モデルに教える。
2. **HOUSEHOLD/HOBBIES からの SNAP 変数除外**:
   - これらカテゴリの学習から `is_snap` 関連を物理的に切断し、代わりに「給与日」や「週末」にモデルの注意力を集中させる。
3. **FOODS の二極化モデル化**:
   - 図33の「安値まとめ買い」と「中高価格プチ贅沢」という SNAP 支給日の二極化行動を、価格帯別の交差特徴量で捉える。

---

## 2026-03-19: Step 13 (Decision Edition) — 因果関係の純鋭化と特徴量削減

### 14. 分析サマリ：慣性モデル（EWMA）からの脱却
- **現状評価:** RMSE 2.1327。`value_gap` 等の価格系特徴量は Top 10 入りを果たしたが、依然として `ewma_28` が支配的。
- **根本課題:** モデルが「なぜ今日売れるのか」という因果を理解するための「状況証拠（コンテキスト）」が不足しており、0/1 フラグによる情報の希釈（Dilution）が起きている。
- **新戦略:** 
    - **FOODS**: 「プチ贅沢（Treat Yourself）」行動への焦点化。
    - **NON_FOODS**: SNAPというノイズの完全排除と、給料日サイクルへの集中。

### 15. 戦略：特徴量の少数精鋭化 (Surgical Pruning)

#### ① FOODS モデル (78 → 64 features)
- **残す SNAP 特徴量:** `snap_active`, `snap_x_high_price` (+$5) の2列のみ。
- **削除 (14列):** 
    - 冗長な交差: `snap_wday`, `is_snap_first_weekend`, `snap_first_10d`
    - 低寄与スコア: `snap_dependency_score`, `snap_dep_interaction`, `snap_x_income`, `snap_x_pb`, `snap_cat_lift`, `cat_snap_sensitivity`
    - 効果不明: `snap_x_low_price` (-$1)
    - 低寄与 (Step 4): `deal_intensity`, `above_price_wall`, `days_since_spike`

#### ② NON_FOODS モデル (78 → 59 features)
- **SNAP 排除:** すべての SNAP 関連変数（13列）を完全に削除。
- **低寄与削除 (Step 4):** `luxury_pressure_x_payday`, `impulse_buy_index`, `event_consumption_type`, `deal_intensity`, `above_price_wall`, `days_since_spike`

---

## 2026-03-19: v5 結果 — SNAP削減 + 特徴量 pruning

### 16. RMSE: 2.1263 (過去最高)

| 指標 | ベースライン (v1) | v4 (Decision Ed.) | **v5 (pruning)** | 累計改善 |
|---|---|---|---|---|
| 全体 | 2.1357 | 2.1327 | **2.1263** | **-0.0094** |
| FOODS | 2.5835 | 2.5784 | **2.5692** | -0.0143 |
| HOBBIES | 1.4500 | 1.4519 | **1.4502** | ±0 |
| HOUSEHOLD | 1.8207 | 1.8130 | **2.8117** | -0.0090 |

### 17. Feature Importance (v5)

**FOODS Top 10:**
| Rank | Feature | Importance |
|---|---|---|
| 1 | ewma_28 | 1.94e+08 |
| 2 | roll_mean_56_weighted | 3.08e+07 |
| 3 | roll_mean_56 | 2.40e+07 |
| 4 | roll_std_56 | 1.13e+07 |
| 5 | **price_rolling_mean_56** | 5.57e+06 ★ |
| 6 | discount_ratio | 5.54e+06 |
| 7 | days_since_last_sale | 5.52e+06 |
| 8 | month | 4.99e+06 |
| 9 | **value_gap** | 4.08e+06 ★ |
| 10 | ewma_7 | 2.94e+06 |

**NON_FOODS Top 10:**
| Rank | Feature | Importance |
|---|---|---|
| 1 | ewma_28 | 1.42e+08 |
| 2 | roll_mean_28 | 8.69e+06 |
| 3 | roll_mean_56_weighted | 5.13e+06 |
| 4 | roll_mean_56 | 3.46e+06 |
| 5 | roll_std_56 | 3.06e+06 |
| 6 | **value_gap** | 2.02e+06 ★ |
| 7 | month | 1.87e+06 |
| 8 | discount_ratio | 1.81e+06 |
| 9 | days_since_last_sale | 1.62e+06 |
| 10 | wday | 1.41e+06 |

### 18. 分析

- **特徴量削減が改善に直結:** 78→65/60 に減らして RMSE -0.0064。情報の希釈が解消された
- **価格系が浮上:** `price_rolling_mean_56` が FOODS #5、`value_gap` が FOODS #9 / NON_FOODS #6
- **SNAP 全削除は正解:** NON_FOODS で SNAP 13列を全削除しても RMSE 微改善
- **ewma_28 の支配度はさらに上昇:** FOODS 6.3倍、NON_FOODS 16.3倍 (削除された特徴量の importance を吸収)
- **deal_intensity, above_price_wall は NOT FOUND:** 削除が正しく反映されている

---

## 2026-03-19: v6 結果 — Step C 残差学習 (FOODS のみ)

### 19. RMSE: 2.1106 (過去最高、ベースラインから -0.0251)

| 指標 | v5 (pruning) | **v6 (残差学習)** | 改善 | ベースラインからの累計 |
|---|---|---|---|---|
| 全体 | 2.1263 | **2.1106** | **-0.0157** | **-0.0251** |
| FOODS | 2.5692 | **2.5451** | **-0.0241** | -0.0384 |
| HOBBIES | 1.4502 | 1.4488 | -0.0014 | -0.0012 |
| HOUSEHOLD | 1.8117 | 1.8115 | -0.0002 | -0.0092 |

### 20. Step C 実装内容
- **FOODS**: target = sales - roll_mean_28 (残差学習), objective = `regression` (MSE)
- **NON_FOODS**: target = sales (通常学習), objective = `tweedie` (変更なし)
- 推論時: FOODS の予測値 = model.predict() + roll_mean_28 → clip(0)

### 21. Feature Importance (v6) — ewma_28 独裁の崩壊

**FOODS Top 10 (残差学習):**
| Rank | Feature | Importance | v5比 |
|---|---|---|---|
| 1 | **month** | 5.68e+07 | v5 #8 → **#1** |
| 2 | roll_mean_28 | 4.41e+07 | — |
| 3 | zeros_last_28 | 3.94e+07 | 圏外 → **#3** |
| 4 | **sell_price** | 3.91e+07 | 圏外 → **#4** |
| 5 | lag_28 | 3.36e+07 | — |
| 6 | discount_ratio | 3.14e+07 | v5 #6 |
| 7 | **value_gap** | **3.14e+07** | v5 #9 → **#7** (imp 7.7倍) |
| 8 | roll_mean_56 | 3.04e+07 | v5 #3 |
| 9 | roll_median_7 | 2.92e+07 | 圏外 → **#9** |
| 10 | **wday** | 2.64e+07 | 圏外 → **#10** |

**NON_FOODS Top 10 (変更なし):**
| Rank | Feature | Importance |
|---|---|---|
| 1 | ewma_28 | 1.42e+08 |
| 2 | roll_mean_28 | 8.73e+06 |
| 3 | roll_mean_56_weighted | 4.74e+06 |
| 4 | roll_mean_56 | 3.47e+06 |
| 5 | roll_std_56 | 3.31e+06 |
| 6 | **value_gap** | 2.02e+06 ★ |
| 7 | month | 1.98e+06 |
| 8 | discount_ratio | 1.81e+06 |
| 9 | sell_price | 1.67e+06 |
| 10 | days_since_last_sale | 1.64e+06 |

### 22. 分析

- **ewma_28 が FOODS Top 10 から消失**: 残差学習により「売上レベル」が roll_mean_28 で分離され、ewma_28 の独裁が終了
- **importance の均等化**: #1 (5.7e7) と #10 (2.6e7) の比率が **2.2倍** (v5 では 66倍)
- **因果系特徴量の躍進**: sell_price (#4), value_gap (#7), wday (#10) が初めて Top 10 入り
- **「なぜ今日売れるか」の学習が開始**: 季節性 (month), 価格 (sell_price, value_gap), 曜日 (wday), 需要パターン (zeros_last_28) がバランスよく配置
- **FOODS の改善が支配的**: -0.0241 (全体改善 -0.0157 の大部分)
- **NON_FOODS は安定**: 通常学習を維持し、ewma_28 が引き続き有効

---

## 2026-03-19: Kaggle 初回提出結果

### 23. Kaggle Score (v6, Late Submission)

| | Score (WRMSSE) |
|---|---|
| Public | **0.72977** |
| Private | 0.98067 |

- **M5 の評価指標は WRMSSE** (Weighted Root Mean Squared Scaled Error)。ローカルの RMSE (2.1106) とは別指標。
- Public 0.73 は M5 全体の中央値付近 (1位: ~0.52, Top 100: ~0.55)。
- **Public → Private で +0.25 の大幅悪化** が最大の課題。

### 24. Public/Private ギャップの原因分析

1. **Val 期間への過剰適合**: roll_mean_28 等の統計量が Val 直前のデータに最適化されており、28日先の Eval 期間では精度が落ちる
2. **残差学習のベースライン劣化**: target = sales - roll_mean_28 で学習するが、Eval 期間の roll_mean_28 は Val 期間のデータから計算される → Val と Eval でベースラインの質が異なる
3. **WRMSSE の重み構造**: 売上が大きいアイテム/部門ほど重みが大きい。RMSE では見えない偏りが存在
4. **Eval 期間は COVID-19 直前 (2016年5月頃)**: 消費パターンの構造変化がある可能性

### 25. 改善の方向性

| 課題 | 対策 | 優先度 |
|---|---|---|
| WRMSSE 非最適化 | WRMSSE カスタム目的関数 or 重み付き RMSE の導入 | 高 |
| Public/Private ギャップ | 複数 28日ウィンドウでの time-series CV | 高 |
| Eval の roll_mean_28 劣化 | recursive prediction (28日を逐次予測) | 中 |
| NON_FOODS の改善余地 | 残差学習の適用検討 (間欠需要のリスクあり) | 中 |
| さらなる特徴量削減 | FOODS #20 以下を刈り込み | 低 |

---

## 2026-03-19: v7 結果 — Step D ハイパラ正則化 + days_since_snap 復活

### 26. Kaggle Score: Private 大幅改善

| | v6 | **v7 (正則化)** | 改善 |
|---|---|---|---|
| Val RMSE | 2.1106 | 2.1256 | +0.0150 悪化 |
| Public (WRMSSE) | 0.730 | 0.736 | +0.006 微悪化 |
| **Private (WRMSSE)** | **0.981** | **0.842** | **-0.139 大幅改善** |
| **Gap (Private - Public)** | 0.251 | **0.106** | **ギャップ半減以下** |

### 27. Step D 実装内容
- `num_leaves`: 127 → 63 (両モデル)
- `min_child_samples`: 20 → 50 (両モデル)
- `feature_fraction`: 0.8 → 0.7 (FOODS のみ)
- `days_since_snap`: FOODS モデルに復活 (v5 で削除していたが、Type-S 店舗で -27.7% の減衰が確認されたため)

### 28. Feature Importance (v7)

**FOODS Top 10 (残差学習 + 正則化):**
| Rank | Feature | Importance | v6比 |
|---|---|---|---|
| 1 | month | 4.51e+07 | #1 維持 |
| 2 | roll_mean_28 | 4.25e+07 | #2 維持 |
| 3 | zeros_last_28 | 3.46e+07 | #3 維持 |
| 4 | sell_price | 3.29e+07 | #4 維持 |
| 5 | **ewma_28** | 3.20e+07 | 圏外 → **#5 復帰** |
| 6 | lag_28 | 2.68e+07 | #5 → #6 |
| 7 | value_gap | 2.46e+07 | #7 維持 |
| 8 | discount_ratio | 2.42e+07 | #6 → #8 |
| 9 | price_rolling_mean_56 | 2.34e+07 | #12 → **#9** |
| 10 | roll_median_7 | 2.26e+07 | #9 → #10 |

**NON_FOODS Top 10:**
| Rank | Feature | Importance |
|---|---|---|
| 1 | ewma_28 | 1.38e+08 |
| 2 | roll_mean_28 | 1.98e+07 |
| 3 | roll_std_28 | 2.43e+06 |
| 4 | month | 2.19e+06 |
| 5 | value_gap | 2.01e+06 |
| 6 | discount_ratio | 1.98e+06 |
| 7 | sell_price | 1.87e+06 |
| 8 | days_since_last_sale | 1.78e+06 |
| 9 | wday | 1.39e+06 |
| 10 | ewma_7 | 1.32e+06 |

### 29. 分析

- **ハイパラ正則化は汎化に極めて有効**: Val RMSE は +0.015 悪化したが、Private は -0.139 の大幅改善。過適合抑制が正しく機能
- **ewma_28 が FOODS #5 に復帰**: 正則化で tree が浅くなり、ewma_28 を「手っ取り早い shortcut」として再利用。ただし独裁ではない (5.7e7→3.2e7, 均等化は維持)
- **NON_FOODS は改善余地が大きい**: ewma_28 が依然 70倍独裁。残差学習の適用が次の最大施策

### 30. Step 13: 家計セグメンテーション EDA 結果

#### SNAP 減衰分析 (Type-S vs Type-B-low)
SNAP 日を基準にした売上減衰を Type-S (SNAP依存: WI_2, WI_3, CA_3) と Type-B-low (自立型: CA_2, CA_4) で比較:

| カテゴリ | Type-S 最大減衰 | Type-B-low 最大減衰 | 差分 |
|---|---|---|---|
| **FOODS** | **-27.7%** (day 15) | -12.4% (day 15) | **2.2倍** |
| HOBBIES | -23.9% (day 1のみ) → 以降 -3〜6% | -7.6% | SNAP 無関係 |
| HOUSEHOLD | -21.6% (day 1のみ) → 以降 -4〜8% | -10.9% | SNAP ほぼ無関係 |

**核心的発見:**
- **FOODS はSNAP後ずっと -20〜27% で沈み続ける** → `days_since_snap` が極めて有効
- **HOBBIES/HOUSEHOLD は day 2 で即回復** → SNAP サイクルは無関係
- **HOBBIES/HOUSEHOLD は day 18-20 で +47〜55% に跳ねる** → 次の SNAP 前の先行購買
- `days_since_snap` の FOODS 復活は正しかった (v7 で実施済み)

#### HOUSEHOLD 所得 proxy (Budget/Premium 比率)
| Store | Budget% | Premium% | B/P比 | snap_lift |
|---|---|---|---|---|
| WI_2 | 55.6% | 17.9% | 3.11 | 1.329 (最低所得) |
| CA_4 | 44.3% | 24.5% | 1.80 | 1.047 (最高所得) |

- Budget/Premium 比率 × SNAP lift: **r = 0.70**
- SNAP lift だけでは捉えきれない所得の別側面を B/P 比が補完

---

## 2026-03-19: Payday Deep Dive EDA

### 31. SNAP 減衰の FOODS vs NON_FOODS 差異 (Type-S 店舗)
- **FOODS**: SNAP 後 -20〜27% で沈み続ける。day 15 で最大 -27.7%
- **HOBBIES/HOUSEHOLD**: day 2 で即回復 (-3〜8%)。day 18-20 で +47〜55% (次の SNAP 前の先行購買)
- **結論**: `days_since_snap` は FOODS にのみ有効。NON_FOODS には不要

### 32. Payday 減衰: dept 別 (平日のみ、週末効果排除)
| dept | Payroll店 | SNAP店 | Independent店 |
|---|---|---|---|
| HOBBIES_1 (おもちゃ) | -11.2% | **-15.0%** | -8.3% |
| HOBBIES_2 (スポーツ) | -3.1% | +2.0% | +2.4% |
| HOUSEHOLD_1 (洗剤等) | -10.8% | **-17.5%** | -9.5% |
| HOUSEHOLD_2 (調理器具) | -8.0% | -9.6% | -5.7% |

- **HOBBIES_1 と HOUSEHOLD_1 が Salary に強く反応。HOBBIES_2 は無反応**
- **SNAP 店が最大の減衰** — 低所得ほど給料日サイクルに支配される

### 33. FOODS 価格帯別 Salary 感度
| 価格帯 | 全体減衰 | Payroll店 | Independent店 |
|---|---|---|---|
| $0-1 | -11.9% | -18.0% | -6.2% |
| $10+ | **-18.1%** | **-26.7%** | **-13.5%** |

- **$10+ が最大**: 高級食材は全 store_type で最も Salary に敏感
- Independent 店でも -13.5% — 高所得でも高級食材は給料日を意識

### 34. 給料日後の週末プレミアム → negligible
- Cohen's d = 0.014 (HOBBIES), 0.022 (HOUSEHOLD) → **効果量ゼロ**
- `payday_weekend_window` フラグは不要

---

## 2026-03-19: Phase 2 実装 (v8, Colab 実行待ち)

### 35. Phase 1 追加特徴量
| 特徴量 | 定義 | 根拠 |
|---|---|---|
| `relative_trend_28_56` | roll_mean_28 / (roll_mean_56 + 1e-8) | トレンド方向の1値表現 |
| `days_since_payday` | 1日/15日からの最短距離 (0-14) | 全セグメントで -5〜18% 減衰 |

### 36. Phase 1.5 追加特徴量
| 特徴量 | 定義 | 粒度 | 根拠 |
|---|---|---|---|
| `store_bp_ratio` | HOUSEHOLD 必需品 Budget/Premium 売上比率 | 10値 | 所得 proxy (r=0.70) |
| `item_payday_sensitivity` | item 別 payday日平均 / 非payday日平均 | ~3,049値 | HOBBIES_2(+2%) vs HOUSEHOLD_1(-17.5%) を弁別 |
| `store_payday_decay` | store 別 near-payday avg / far-payday avg | 10値 | Payroll店(-27%) vs Independent店(-13%) |
| `dept_payday_sensitivity` | dept 別 payday lift | 7値 | HOBBIES_1(-11%) vs HOBBIES_2(-3%) |

### 37. GPU 変更
- **NON_FOODS**: `residual_target: True`, objective `tweedie` → `regression`
- 両モデルとも残差学習 (target = sales - roll_mean_28)

### 38. 実行手順
1. parquet 削除セル実行 (pipeline_cpu.ipynb cell 3)
2. Phase 1 → 1.5 → 2 (parquet 再生成)
3. GPU 学習 → 評価 → Kaggle 提出

---

## 2026-03-24: v8 結果 — Phase 2 (全モデル残差学習 + 家計セグメント特徴量)

### 39. Kaggle Score: 過去最高 — Public/Private 両方で大幅改善

| | v6 | v7 (正則化) | **v8 (Phase 2)** | v7→v8 改善 |
|---|---|---|---|---|
| Val RMSE | 2.1106 | 2.1256 | 2.1165 | -0.0091 |
| Public (WRMSSE) | 0.730 | 0.736 | **0.723** | **-0.013** |
| **Private (WRMSSE)** | 0.981 | 0.842 | **0.755** | **-0.087** |
| **Gap** | 0.251 | 0.106 | **0.032** | ギャップほぼ解消 |

### 40. Val RMSE 内訳

| 指標 | v7 (正則化) | **v8** | 差分 |
|---|---|---|---|
| FOODS | 2.5677 | 2.5486 | -0.0191 |
| HOBBIES | 1.4508 | 1.4615 | +0.0107 悪化 |
| HOUSEHOLD | 1.8145 | **1.8108** | -0.0037 (過去最高) |

- HOBBIES の Val 悪化 (+0.013) にもかかわらず Private は大幅改善 → **Val RMSE だけで判断してはいけない**
- HOUSEHOLD が全バージョンで過去最高 → payday sensitivity scores + store_bp_ratio が効果を発揮

### 41. Feature Importance (v8)

**FOODS Top 10 (残差学習 + 正則化 + Phase 2 特徴量):**
| Rank | Feature | Importance | v7比 |
|---|---|---|---|
| 1 | month | 4.30e+07 | #1 維持 |
| 2 | **relative_trend_28_56** | **3.96e+07** | **新規 #2** ★ |
| 3 | roll_mean_28 | 3.70e+07 | #2 → #3 |
| 4 | lag_28 | 3.23e+07 | #6 → #4 |
| 5 | zeros_last_28 | 2.92e+07 | #3 → #5 |
| 6 | ewma_28 | 2.80e+07 | #5 → #6 |
| 7 | value_gap | 2.50e+07 | #7 維持 |
| 8 | discount_ratio | 2.39e+07 | #8 維持 |
| 9 | price_rolling_mean_56 | 2.24e+07 | #9 維持 |
| 10 | roll_median_7 | 2.16e+07 | #10 維持 |

**NON_FOODS Top 10 (残差学習に移行):**
| Rank | Feature | Importance | v7比 |
|---|---|---|---|
| 1 | roll_mean_28 | 4.71e+06 | **ewma_28 (#1, 1.38e+08) が消滅** |
| 2 | **relative_trend_28_56** | **3.89e+06** | **新規 #2** ★ |
| 3 | zeros_last_28 | 3.09e+06 | 圏外 → #3 |
| 4 | roll_mean_7 | 2.29e+06 | 圏外 → #4 |
| 5 | month | 2.21e+06 | #4 → #5 |
| 6 | lag_28 | 1.88e+06 | 圏外 → #6 |
| 7 | **wday** | 1.86e+06 | #9 → **#7** |
| 8 | lag_56 | 1.77e+06 | 圏外 → #8 |
| 9 | roll_median_7 | 1.60e+06 | 圏外 → #9 |
| 10 | lag_42 | 1.54e+06 | 圏外 → #10 |

### 42. 分析

#### 最大の成果: NON_FOODS の ewma_28 独裁が崩壊
- v7: ewma_28 が 1.38e+08 で #1 (2位との比 70倍)
- v8: ewma_28 が **Top 10 外**。#1/#10 比 = **3.1倍** (70倍 → 3.1倍)
- NON_FOODS 残差学習が成功し、FOODS と同様に importance が均等化

#### `relative_trend_28_56` が両モデルで #2
- roll_mean_28 / roll_mean_56 という単純な比率が「トレンド方向」として極めて有効
- FOODS: 3.96e+07 (month に次ぐ2位)
- NON_FOODS: 3.89e+06 (roll_mean_28 に次ぐ2位)

#### Private ギャップが 0.032 に縮小
- v6: 0.251 → v7: 0.106 → **v8: 0.032**
- 過学習がほぼ解消。正則化 + 残差学習 + 情報密度の高い特徴量の相乗効果

#### 全施策の累計効果
| 施策 | Public 改善 | Private 改善 |
|---|---|---|
| v6→v7: ハイパラ正則化 | +0.006 | **-0.139** |
| v7→v8: NON_FOODS 残差学習 + Phase 2 特徴量 | **-0.013** | **-0.087** |
| **v6→v8 累計** | **-0.007** | **-0.226** |

---

## 2026-03-24: v9b 失敗と v8+α (v9c) への再試行

### 43. v9b (40/44 drops) の失敗分析
- **結果:** Public 0.767 / Private 0.830 (v8 より大幅悪化)
- **原因:** 特徴量の削減しすぎ。特に **Store profile (8列)**, **Interaction (6列)**, **Store×Cat/Dept (4列)** を一括削除したことで、店舗の性格（所得分布や需要構造）をモデルが判別できなくなった。
- **教訓:** 個別の importance が低くても、それらが集合的に「文脈（誰が・いつ・何を）」を提供している場合、一括削除は危険。

### 44. v9c 戦略: v8 (15/20 drops) + 安全な7列のみ追加削除
- **FOODS:** v8 (15) + 7 safe = **22 drops**
- **NON_FOODS:** v8 (20) + 7 safe = **27 drops**
- **追加削除した7列:**
    - `year`, `event_name_2`, `event_type_2`, `is_month_start`, `payday_weekend` (Calendar系)
    - `deal_intensity`, `above_price_wall` (低寄与確定)
- **復活させた列:** Store profile, Interaction, Store×Cat/Dept 関連の全列。
- **意図:** 「誰が・いつ・何を」の文脈を維持しつつ、明らかに不要なノイズ列のみを排除する。

### 45. v9c/v9d 結果: v8 より悪化
| | v8 | v9c (22/27 drops) | v9d (21/26, roll_mean_56復帰) |
|---|---|---|---|
| Public | **0.723** | 0.768 | 0.769 |
| Private | **0.755** | 0.833 | 0.839 |

**教訓: feature_fraction=0.7 が暗黙の正則化を担っている。** 低 importance の特徴量も tree のバリエーション生成に寄与しており、手動削除はアンサンブルの多様性を壊す。v8 の 15/20 drops が最適水準。**これ以上の削減は行わない。**

---

## 2026-03-25: v10 結果 — item_snap_sensitivity + item_month_index

### 46. v10 実装内容

#### Phase 1.5 新特徴量
| 特徴量 | Pass | 定義 | 粒度 |
|---|---|---|---|
| `item_snap_sensitivity` | 0f | FOODS item 別 SNAP 日平均/非SNAP 日平均 | ~1,400値 |
| `item_month_index` | 0g | item × month の季節性指数 (月別avg/年間avg) | ~36,000値 |

#### EDA 知見 (新規分析)
- **FOODS_2 は 98% が SNAP 感応** (冷凍食品・肉 = プチ贅沢 + 備蓄)
- **FOODS_1 は 40% が SNAP 中立** (飲料 = 日常ルーチン)
- **SNAP と Salary の相関: r = 0.41** — 異なる情報を持つ
- **SNAP-Pay+ (Salary のみ反応): 78 items (5.5%)** — FOODS_1 の 18% が該当
- **季節性が最も強い: HOBBIES_2** (CV=0.171, 10月+12月にスパイク)
- **季節性が最もフラット: HOBBIES_1** (CV=0.032)

#### GPU 変更
- drop_features: **v8 に復帰** (FOODS 14, NON_FOODS 19)
- NON_FOODS: **50% サンプリング** (CPU メモリ制約対策)
- NON_FOODS: rounds 2500 → 1500

#### インフラ改善
- Phase 1.5 Pass 統合: 8回 → 5回スキャン
- FOODS/NON_FOODS 学習を別セルに分離 (OOM 対策)
- models dict + pkl バックアップ方式

### 47. Kaggle Score: **過去最高を更新**

| | v8 (前ベスト) | **v10** | 改善 |
|---|---|---|---|
| Val RMSE | 2.1165 | 2.1213 | +0.005 |
| Public (WRMSSE) | 0.723 | **0.724** | ±0 (同等) |
| **Private (WRMSSE)** | 0.755 | **0.750** | **-0.005 改善** |
| **Gap** | 0.032 | **0.026** | さらに縮小 |

### 48. Feature Importance (v10)

**FOODS Top 10:**
| Rank | Feature | Importance | v8比 |
|---|---|---|---|
| 1 | month | 3.80e+07 | #1 維持 |
| 2 | roll_mean_28 | 3.75e+07 | #3 → #2 |
| 3 | relative_trend_28_56 | 3.54e+07 | #2 → #3 |
| 4 | roll_mean_56 | 3.03e+07 | — |
| 5 | lag_28 | 2.65e+07 | #4 維持 |
| 6 | zeros_last_28 | 2.50e+07 | #5 → #6 |
| 7 | sell_price | 2.50e+07 | — |
| 8 | value_gap | 2.31e+07 | #7 → #8 |
| 9 | roll_median_7 | 2.01e+07 | #10 → #9 |
| 10 | discount_ratio | 1.91e+07 | #8 → #10 |

**NON_FOODS Top 10:**
| Rank | Feature | Importance | v8比 |
|---|---|---|---|
| 1 | roll_mean_28 | 2.34e+06 | #1 維持 |
| 2 | relative_trend_28_56 | 2.19e+06 | #2 維持 |
| 3 | zeros_last_28 | 1.63e+06 | #3 維持 |
| 4 | roll_mean_7 | 1.45e+06 | #4 維持 |
| 5 | lag_56 | 1.24e+06 | #8 → #5 |
| 6 | wday | 1.19e+06 | #7 → #6 |
| 7 | month | 1.08e+06 | #5 → #7 |
| 8 | lag_28 | 0.97e+06 | #6 → #8 |
| 9 | roll_median_7 | 0.84e+06 | — |
| 10 | lag_35 | 0.73e+06 | — |

### 49. 分析

- **Private 0.750 — 過去最高。** Val RMSE は v8 より +0.005 高いが、Private は -0.005 改善。汎化性能が向上
- **Gap 0.026 — 過学習がほぼ解消。** v6 の 0.251 → v10 の 0.026 (10分の1以下)
- **item_snap_sensitivity / item_month_index は Top 10 外** だが、parquet にタプルカラム名バグがあった。修正済みなので次回再生成後に正しく機能する見込み
- **NON_FOODS 50% サンプリング + rounds 1500 の制約下** でこの結果。GPU 枠復活後にフルデータで改善余地あり
- **feature pruning は v8 レベルが最適。** これ以上の削減は多様性を壊す

### 50. 累計スコア推移

| Version | Val RMSE | Public | Private | Gap | 主要変更 |
|---|---|---|---|---|---|
| v1 (baseline) | 2.1357 | — | — | — | 3モデル, 48列 |
| v5 (pruning) | 2.1263 | — | — | — | SNAP 13→2 (FOODS)/0 (NF) |
| v6 (残差FOODS) | 2.1106 | 0.730 | 0.981 | 0.251 | FOODS 残差学習 |
| v7 (正則化) | 2.1256 | 0.736 | 0.842 | 0.106 | num_leaves 63, min_child 50 |
| v8 (Phase 2) | 2.1165 | 0.723 | 0.755 | 0.032 | NON_FOODS 残差学習 + 家計特徴量 |
| v9b (大量削減) | 2.1247 | 0.767 | 0.830 | 0.063 | 40/44 drops → 失敗 |
| **v10 (snap+season)** | 2.1213 | **0.724** | **0.750** | **0.026** | item_snap_sensitivity + item_month_index |

---

## 2026-03-25: EDA — FOODS 季節性 × 所得の交差分析

### 51. month が FOODS #1 の本当の理由

**残差学習 (target = sales - roll_mean_28) の後で month が #1 なのは「季節の食べ物の違い」ではなく、「roll_mean_28 の遅延を補正するカレンダー情報」として機能しているため。**

roll_mean_28 は季節の変わり目 (11月→12月の年末商戦) に追いつけない。month = 12 で「今は年末」と直接伝える役割。

### 52. FOODS dept 別: 所得で季節パターンが逆転する

**FOODS_1 (飲料):**
| 所得タイプ | 夏(6-8月) | 冬(12-2月) | 夏/冬比 | 解釈 |
|---|---|---|---|---|
| SNAP (低所得) | 0.98 | **1.06** | **0.92** (冬型) | ホットドリンク消費 |
| Payroll | 0.96 | **1.09** | **0.88** (冬型) | 同上 |
| **Independent (高所得)** | **1.05** | 1.01 | **1.04** (夏型) | 清涼飲料水のケース買い |

**同じ「飲料」でも低所得は冬ピーク、高所得は夏ピーク。** 全店平均の `item_month_index` では相殺される。

**FOODS_2 (肉/冷凍):**
| 所得タイプ | 夏/冬比 | 特徴 |
|---|---|---|
| SNAP | 0.91 (冬型) | 年末の肉需要 |
| Payroll | 1.01 (フラット) | 通年 |
| Independent | 0.88 (冬型) | 年末の高級肉 |

**FOODS_3 (食品全般): 全所得で夏偏重** (Payroll 店が最大 1.17)。BBQ/ピクニック食品。

### 53. 11月→12月の急変 = 年末商戦の所得別差異

| Dept | SNAP 12月急増 | Payroll 12月急増 | Independent 12月急増 |
|---|---|---|---|
| FOODS_1 (飲料) | +19.3% | +21.5% | **+27.7%** |
| FOODS_2 (肉) | -3.5% | -3.2% | -4.7% |
| FOODS_3 (食品) | -3.2% | -1.1% | -2.0% |

**FOODS_1 のクリスマス急増は高所得 (Independent) が最大 (+27.7%)。** 低所得でも +19% — クリスマスは全所得層で飲料が増える。FOODS_2/3 は 11月 (感謝祭) がピークで 12月はむしろ微減。

### 54. 価格帯の違い

| Dept | 平均価格 | 中央値 | Q75 |
|---|---|---|---|
| FOODS_2 (肉) | **$4.15** | $2.97 | **$5.34** |
| FOODS_1 (飲料) | $3.33 | $2.51 | $4.75 |
| FOODS_3 (食品) | $2.87 | $2.52 | $3.51 |

FOODS_2 が最も高価格。低所得は安い肉 (ひき肉)、高所得はステーキ肉 — **肉 × 所得の相関が最も大きい** dept。

### 55. 特徴量設計への示唆

**現在の `item_month_index` (全店平均) では SNAP 店の冬型と Independent 店の夏型が相殺される。** 所得別の `item_month_index` が将来の改善候補だが、lookup サイズが 3倍 (36K → 108K) になるため、まず現在の `item_month_index` がタプルバグ修正後に正しく機能するかを確認してから判断する。

### 56. 季節性が顕著な dept / item

**季節型 (高 CV):**
| 対象 | CV | ピーク月 | トリガー |
|---|---|---|---|
| **HOBBIES_2** (スポーツ用品) | **0.171** (dept 最大) | 10月 +38%, 12月 +12% | ハロウィン + クリスマス |
| HOUSEHOLD_2 (調理器具) | 0.056 | 夏 (6-8月) | BBQ/ガーデニング |
| FOODS_2_255 | **1.315** (item 最大) | 11月に **5.1倍** | 感謝祭ターキー |
| FOODS_3_069 | 1.251 | 12月に **4.9倍** | クリスマス食品 |
| HOUSEHOLD_2_340 | 1.199 | 6月に 3.9倍, 11月に 0.07倍 | 夏アウトドア用品 |
| HOUSEHOLD_1_049/297 | 1.0+ | 7月ピーク | 夏の清掃用品 |

**通年型 (低 CV):**
| 対象 | CV | 特徴 |
|---|---|---|
| **HOBBIES_1** | **0.032** (dept 最小) | おもちゃ/ゲームは通年売れる |
| FOODS_3_458 | 0.038 | daily=8.67 の高売上定番品 (パン/卵系) |
| HOBBIES_1_254 | 0.028 | daily=4.57 の完全通年品 |

### 57. ピーク月の分布 — 「イベントカレンダー」

| カテゴリ | 最多ピーク月 | トリガー |
|---|---|---|
| FOODS | **1-2月** | 年始 + スーパーボウル (2月) |
| HOBBIES | **10月 + 12月** | ハロウィン + クリスマス |
| HOUSEHOLD | **8月** | 新学期準備 (Back to School) |

**これらは `item_month_index` で捉えられるはず。** ただし FOODS_1 の所得による夏/冬逆転 (Section 52) は全店平均 index では相殺される。

---

## v13: 低寄与特徴量の整理 (2026-03-27)

### 変更内容
- `above_price_wall` と `deal_intensity` を FOODS / NON_FOODS 両モデルの `drop_features` に追加
- 対象: `pipeline_gpu.ipynb` Cell 6 (gpu_006) + Cell 7 (gpu_006b) — 本体と復元ブロック両方を同期

### 根拠 (Step 12d 新特徴量の重要度分析)

| 特徴量 | FOODS (順位) | NON_FOODS (順位) | 判定 |
|--------|-------------|-----------------|------|
| value_gap | 11位 | 5位 | 大当り — 残す |
| price_rolling_mean_56 | 16位 | 15位 | 優秀 — 残す |
| value_gap_x_elasticity | 22位 | 20位 | 良好 — 残す |
| price_rank_in_dept | 28位 | 17位 | 良好 — 残す |
| deal_intensity | 43位 | 52位 | 微妙 — **drop** |
| above_price_wall | 62位 | 54位 | 不要 — **drop** |

### 期待効果
- FOODS: feature_fraction=0.7 で低寄与特徴量がサンプリングされる確率を排除
- NON_FOODS: 20→20 drop (2特徴量追加で次元削減)
- 学習の高速化・汎化改善

### v12 ベースライン RMSE
- FOODS: 2.5696 / HOBBIES: 1.4477 / HOUSEHOLD: 1.8090 / Overall: 2.1257

### v13 RMSE → (Colab学習後に記入)

### 構造分析メモ
- **FOODS**: バランス型。item_month_index, roll_mean_28, lag_28, zeros_last_28 が上位 — 間欠需要・在庫切れの信号が強い
- **NON_FOODS**: ewma_28 一極集中型 (2位の6倍の重要度)。価格・カレンダー要因が効きにくい

---

## Next Steps

1. **間欠需要系特徴量** — zero_to_nonzero_transition, sale_burst_after_zero, item_zero_volatility (FOODS RMSE 改善狙い)
2. **価格系深掘り** — price_zscore_in_dept (price_rank_in_dept の連続値版), cross_price_effect (カニバリ)
3. **NON_FOODS EWMA依存解消** — ewma_28_residual の特徴量化
4. **Multi-seed ensemble** — チューニング完了後に実施 (seed=42,123,456 の3モデル平均)
5. **所得別 item_month_index の検討** — FOODS_1 で「低所得=冬型、高所得=夏型」が確認された
