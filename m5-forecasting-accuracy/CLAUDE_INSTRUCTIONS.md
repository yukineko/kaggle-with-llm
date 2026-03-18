# Claude Code Instructions for M5 Forecasting Accuracy (Decision Edition)

最新の SNAP Deep Dive 分析（図32, 33, 34）の結果を考慮した、モデル精度の飛躍を目指す「外科的」特徴量エンジニアリングの指示です。

## 全体戦略
「SNAPフラグ」という強力だが限定的なシグナルを、真に相関のあるカテゴリ（FOODS）にのみ濃縮し、無関係なカテゴリ（HOBBIES/HOUSEHOLD）からは物理的に切断することで、モデルが `roll_mean_56` や `ewma_28` といった平均値に逃げるのを防ぎ、因果関係を学習させる。

---

## Step 1: 店舗の「背景（住民購買力）」の数値化 (図32に基づく)

### 背景
SNAP依存度とHOBBIES売上比率には強烈な負の相関 (r = -0.83) がある。これは店舗ごとの `store_id` という記号を、物理的な「地域の経済力」という連続値に置き換える最強の指標である。

### 指示
1.  Phase 1.5 において、店舗ごとの SNAP リフト（SNAP日平均売上 / 非SNAP日平均売上）を計算し、`store_poverty_index` として全レコードに付与する。
2.  これにより、モデルが「この店舗は低所得商圏にあるため、支給日に売上が跳ね、嗜好品が売れにくい」という背景を理解できるようにする。

---

## Step 2: HOBBIES / HOUSEHOLD カテゴリの「SNAP切断」 (図34に基づく)

### 背景
図34の分析により、HOUSEHOLD カテゴリにおいて SNAP はほとんどノイズであることが判明した。無関係な変数を与えることは情報の希釈（Dilution）を招き、モデルが平均値（EWMA）に逃げる原因となる。

### 指示
1.  **NON_FOODS モデル (HOBBIES + HOUSEHOLD)** の学習・推論において、以下の変数を入力から除外する：
    *   `snap_active`, `days_since_snap`, `is_snap_first_weekend`
2.  これにより、モデルを真の因果である「給料日（Salary）」、「週末」、「価格の壁」というシグナルに集中させる。

---

## Step 3: FOODS カテゴリの「二極化行動」のモデル化 (図33に基づく)

### 背景
図33により、SNAP支給日には「超安値品（生活防衛のまとめ買い）」と「中高価格帯（プチ贅沢）」の両方が跳ねる二極化行動が確認された。

### 指示
1.  **FOODS モデル** において、以下の交差特徴量を導入する：
    *   `snap_x_high_price`: (sell_price が $5.00 以上) × `snap_active`
    *   `snap_x_low_price`: (sell_price が $1.00 以下) × `snap_active`
2.  これにより、「支給日だから食品なら何でも売れる」という曖昧な学習を、「安いものはまとめ買い、高いものは贅沢買い」という具体的な行動パターンの学習へと進化させる。

---

## Step 4: 不要な特徴量の削減 (Ablation Study / Strategic Removal)

### 背景
現在 77 個の特徴量があるが、重要度下位（40位以下）の多くがモデルの「情報の希釈」を招き、移動平均（EWMA）への依存を助長している。特に今回導入した `value_gap` 系の下位項目や、論理的な裏付けが弱い変数を整理し、モデルの因果学習を「純鋭化」させる。

### 指示
1.  **FOODS モデル** において、以下の寄与の低い変数を削除し、情報の密度を高める：
    *   `deal_intensity`, `above_price_wall`, `days_since_spike`
2.  **NON_FOODS モデル** において、前述の「SNAP 排除」に加え、以下の寄与の低い変数を削除する：
    *   `luxury_pressure_x_payday`, `impulse_buy_index`, `event_consumption_type`
3.  削除後の再学習で RMSE が維持、または改善されるかを確認する。もし悪化する場合は、重要な因果が失われた可能性があるため、原因を分析して [`PROCESS.md`](m5-forecasting-accuracy/PROCESS.md) に記録する。

---

## 期待される結果と報告
1.  **重要度の変化**: 支配的な `ewma_28` の重要度が低下し、上記の因果系特徴量（Step 1-3）が上位に浮上すること。
2.  **RMSE の改善**: 現在のベースライン **2.1324** を下回ること。
3.  **モデルの軽量化**: 特徴量数を 77 から 60 前後まで絞り込み、学習速度と解釈性を向上させる。
4.  **報告**: 各ステップ実行後の結果を [`m5-forecasting-accuracy/PROCESS.md`](m5-forecasting-accuracy/PROCESS.md) に追記してください。
