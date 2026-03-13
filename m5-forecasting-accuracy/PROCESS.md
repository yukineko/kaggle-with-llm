# M5 Forecasting Accuracy - Analysis & Instruction Process Log

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

#### Step 7b: ラマダン深掘り分析
- CA_2 が最も感度が高い (score=5.16, FOODS_2 P75+ lift +10.4%)
- ただしラマダンはM5期間中の出現頻度が限定的

#### Step 7c: 価格プロファイリング
- **item_premium_flag:** dept内 Z-score > 2.0 の高価格アイテムを判定
- **store_dept_premium_share:** 高価格品の数量シェア → pb_ratio との相関 r=-0.176（独立 → 両方保持）
- **発見:** TX_1, TX_2 = "taste signal" stores（低 luxury_index だが高 premium_share）

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

### 8. 特徴量の優先度ロードマップ (更新)

| 優先度 | 特徴量 | ステータス | 備考 |
|---|---|---|---|
| 1 | `not_on_shelf` + train除外 | **実装済み** | sell_price NaN ベース |
| 2 | `luxury_pressure` / `luxury_pressure_x_payday` | **実装済み** | 連続値、Phase 1.5 Pass 2 |
| 3 | `event_consumption_type` / `impulse_buy_index` | **実装済み** | EDA Step 6 |
| 4 | `days_since_snap` / `is_snap_first_weekend` | **実装済み** | SNAP心理ラグ |
| 5 | `is_CA4` / `CA4_x_evt_type` | **実装済み** | CA_4 特異性 |
| 6 | `store_dept_wday_avg` | **実装済み** | 構造ベースライン (日常平均) |
| 7 | `store_dept_premium_share` | **実装済み** | 高価格品シェア |
| 8 | `weekday_density_ratio` | **実装済み** | 平日/週末密度比 |
| 9 | SKU密度（store×category 品揃え数） | 保留 | Zero-Sales 関連の補助指標 |
| 10 | `affluence_score`（所得指紋） | 検討中 | premium_share 効果確認後 |
| 11 | roll_mean_56 要否 | 検証待ち | Feature Importance で判断 |

---
*Next Step: Colab で pipeline.ipynb 実行 (parquet 削除 → Phase 1 → 1.5 → 2 → Training) → RMSE + Feature Importance 確認 → roll_mean_56 判断 → affluence_score 実装可否判断*
