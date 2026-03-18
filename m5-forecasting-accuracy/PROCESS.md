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

## Next Steps

1. **Claude Code による Step 1〜3 の実行 (Decision Edition)** — [`CLAUDE_INSTRUCTIONS.md`](CLAUDE_INSTRUCTIONS.md) に基づき、店舗購買力の数値化、Non-Foods の SNAP 切断、FOODS の二極化モデル化を順次実施。
2. **因果関係の学習検証** — 新戦略により、`ewma_28` への過度な依存が低下し、因果系特徴量（Poverty Index, 週末, 給料日）の importance が上昇するかを確認。
3. **Kaggle 提出** — 全体 RMSE が **2.1324** を安定して下回った段階で、提出用ファイルを生成。
