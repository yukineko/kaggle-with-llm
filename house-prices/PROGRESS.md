# House Prices 進捗記録

## セッション1 (2026-03-03)

### Step 1: EDA (探索的データ分析)
- **データ概要**: Train 1460行×81列 (数値37 + カテゴリカル43 + 目的変数SalePrice)、Test 1459行×80列
- **SalePrice分布**: 右裾が長い (Skew=1.88, Kurt=6.54)。log1p変換後はほぼ正規分布 (Skew=0.12, Kurt=0.81)
- **SalePriceとの相関Top5**: OverallQual(0.79), GrLivArea(0.71), GarageCars(0.64), GarageArea(0.62), TotalBsmtSF(0.61)
- **カテゴリカル重要度Top5**: PoolQC, ExterQual, BsmtQual, KitchenQual, Condition2 (グループ間SalePrice中央値のstdで評価)
- **欠損値**: 19列に欠損。PoolQC(99.5%), MiscFeature(96.3%), Alley(93.8%)等はNA="その設備がない"の意味
- **多重共線性**: 相関>0.7のペア6組 (GarageCars-GarageArea=0.88, YearBuilt-GarageYrBlt=0.83等)
- **外れ値**: GrLivArea>4000かつ低価格の2件を発見 → 除去対象

### Step 2: 前処理パイプライン構築 (HousePricesPreprocessor)
Titanicで学んだfold内前処理の教訓を適用。sklearn Pipeline内にTransformerを配置し、統計量はtrain_foldのみから計算。

- **欠損値処理**:
  - NA="なし"の15カテゴリカル列 (Alley, BsmtQual, FireplaceQu, GarageType, PoolQC等) → "None"で埋め
  - Garage/Basement数値系 → ガレージ/地下室なしは0
  - LotFrontage → Neighborhood別中央値で補完 (近隣の敷地前面は似た値になる)
  - MasVnrArea → 0で補完
  - 残りの数値 → 中央値、カテゴリカル → 最頻値
- **エンコーディング**:
  - 順序カテゴリ20列 (Ex/Gd/TA/Fa/Po → 5/4/3/2/1等) を手動マッピング
  - 残りのカテゴリカルはLabel Encoding (fitでマッピングを学習、未知カテゴリは-1)
- **特徴量エンジニアリング**:
  - TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF (総面積)
  - TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
  - TotalBath = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath
  - HouseAge = YrSold - YearBuilt、RemodAge = YrSold - YearRemodAdd
  - HasRemod, HasGarage, HasBsmt, Has2ndFlr, HasPool (バイナリフラグ)
- **歪度補正**: 連続量19特徴量 + TotalSF, TotalPorchSF にlog1p変換
- **目的変数**: SalePrice → log1p変換 (RMSLE評価に最適化。MSE最小化=RMSLE最小化)
- **外れ値除去**: GrLivArea>4000 & SalePrice<300000 の2件を除去

### Step 3: ベースラインモデル比較 (5-fold KFold CV)
6モデルをPipeline(前処理→モデル)で統一的に評価。評価指標はRMSLE (= log1p空間でのRMSE)。

| モデル | RMSLE | ±std | 備考 |
|--------|-------|------|------|
| **GBR** | **0.11460** | 0.00599 | n_estimators=3000, lr=0.05, max_depth=4, loss=huber |
| XGBoost | 0.11907 | 0.00571 | n_estimators=2000, lr=0.05, max_depth=4 |
| Lasso | 0.11983 | 0.00899 | alpha=0.0005 |
| ElasticNet | 0.11995 | 0.00880 | alpha=0.0005, l1_ratio=0.5 |
| Ridge | 0.12102 | 0.00855 | alpha=10.0 |
| LightGBM | 0.12573 | 0.00663 | n_estimators=2000, lr=0.05, num_leaves=31 |

→ GBR単体で提出: **Public 0.12933** (CV対比で+0.015の乖離)

### Step 4: OOFスタッキング
CV-Public乖離を縮めるため、6モデルのOOF予測をメタモデル(Ridge)で結合。

- **方法**: 5-fold CVで各モデルのOut-of-Fold予測を取得 → 6モデル×1458行のOOF行列を作成 → Ridge(alpha=1.0)で結合
- **テスト予測**: 各foldのテスト予測を5fold分平均してからメタモデルに入力
- **結果**: Stack CV = 0.11134 (GBR単体0.11460から-0.003改善)
- **メタウェイト**: GBR=0.43, XGBoost=0.22, Lasso=0.18, ElasticNet=0.18, Ridge=0.04, LightGBM=-0.03
  - GBRが最大の貢献。LightGBMは負のウェイト(他モデルと冗長)

→ 提出: **Public 0.12271** (0.12933から-0.007改善)

### Step 5: ブレンディング (比較実験)
スタッキングとの比較として、手動の加重平均も試行。
- Ridge=0.15, Lasso=0.15, GBR=0.30, XGBoost=0.20, LightGBM=0.20
→ Public 0.12323。スタッキングに劣る (メタモデルが最適ウェイトを学習する方が良い)

### Step 6: マルチシードスタッキング
KFoldのseedを変えて3回スタッキングし、テスト予測を平均。seed間のランダム性を吸収。

- seed=42: Stack CV=0.11134
- seed=123: Stack CV=0.11142
- seed=456: Stack CV=0.11047
- 3 seedのテスト予測を単純平均

→ 提出: **Public 0.12246** (0.12271から微改善)

### Step 7: LOO Target Encoding (Neighborhood)
Neighborhoodは25カテゴリで、中央値$88k(MeadowV)〜$315k(NridgHt)と3.6倍の価格差がある (ANOVA F=79.5, p≈0)。
Label Encoding (アルファベット順0-24) ではこの価格情報が失われるため、Target Encodingで各カテゴリを「期待価格」に変換。

- **手法**: LOO (Leave-One-Out) Target Encoding
  - 訓練データ: 自分自身のyを除外して平均を計算 → 目的変数リーク防止
  - テスト/バリデーションデータ: 訓練データのカテゴリ平均を使用
  - Smoothing (smooth=10): 小カテゴリ (Blueste n=2等) はglobal meanに引き寄せ → 過学習防止
  - 計算式 (LOO): `(Σy_c - y_i + smooth × global_mean) / (n_c - 1 + smooth)`
- **実装**: `fit_transform()`をオーバーライドしてLOOを適用。`transform()`では通常のTE
- **バグ修正**: pandas Index不整合 → `.values`で位置ベース代入に修正

- **結果 (各モデル単体CV)**:
  | モデル | Before TE | After TE | 差分 |
  |--------|-----------|----------|------|
  | Ridge | 0.12102 | 0.11826 | **-0.00276** |
  | Lasso | 0.11983 | 0.11700 | **-0.00283** |
  | ElasticNet | 0.11995 | 0.11699 | **-0.00296** |
  | GBR | 0.11460 | 0.11424 | -0.00036 |
  | XGBoost | 0.11907 | 0.11819 | -0.00088 |
  | LightGBM | 0.12573 | 0.12405 | -0.00168 |

→ **線形モデルが最も恩恵** (約-0.003)。tree系は微改善 (自動でNeighborhoodの分割を学習できるため)
→ Stack CV: 0.11134→0.10993 (-0.00141)
→ 提出: **Public 0.12063** (0.12246から-0.00183改善)

### Step 8: ドメイン知識特徴量 (広さ軸の再定義)
GrLivArea単体では説明しきれない「高価格帯の誤差」を解消するため、「広さ」を多角的に捉える3つの特徴量を追加。

- **Is_SFL (Is_Senior_Friendly_Large)**: 2階なし (`2ndFlrSF==0`) かつ1階面積が中央値以上 → 平屋の広い家フラグ
  - バイナリ (0/1)。1stFlrSF中央値はfitで学習
- **Living_Space_Ratio**: `GrLivArea / LotArea` → 敷地に対する居住面積の密度
  - 低い=敷地が広い (郊外型)、高い=密集型
- **Luxury_Space_Index**: `TotalSF × OverallQual` → 面積と品質の掛け合わせ
  - 「広くて高品質」を一発で捉える。log1p変換適用

- **結果 (各モデル単体CV)**:
  | モデル | Before | After | 差分 |
  |--------|--------|-------|------|
  | Ridge | 0.11826 | 0.11818 | -0.00008 |
  | Lasso | 0.11700 | 0.11661 | -0.00039 |
  | ElasticNet | 0.11699 | 0.11670 | -0.00029 |
  | **GBR** | **0.11424** | **0.11268** | **-0.00156** |
  | XGBoost | 0.11819 | 0.11714 | -0.00105 |
  | LightGBM | 0.12405 | 0.12341 | -0.00064 |

→ **GBRが最大の改善 (-0.00156)**。TotalSF×OverallQualの明示的な交互作用はtreeモデルでも有効
→ Stack CV: 0.10993→0.10971 (-0.00022)
→ 提出: **Public 0.12030** (0.12063から-0.00033改善、**新ベスト**)
→ CV-Public gap: 0.01070→0.01059 (縮小 = 汎化性も改善)

- **SHAP分析**:
  - Luxury_Space_Index: 全体#2 (SHAP=0.03689)、高価格帯(>$300k)では#2 (SHAP=0.08546, 2.3x増幅)
  - Living_Space_Ratio: 全体#18 (SHAP=0.00574)
  - Is_SFL: 全体#30 (SHAP=0.00190)。平屋フラグとしてのニッチな寄与
  - 高価格帯でLSIの寄与が大幅増幅 → 「広くて高品質」が高価格帯の予測精度向上に直結

### Step 9: CatBoost 追加 (7モデルスタッキング)
カテゴリカル変数43列を文字列のままCatBoostに渡すことで、Label Encodingの情報損失を回避。

- **実装**: `CatBoostPreprocessor` を `HousePricesPreprocessor` から継承
  - Label Encodingをスキップ → 残存する21個の名義カテゴリカルを文字列のまま保持
  - `cat_features` 引数で CatBoost にカテゴリカル列を明示
  - 順序カテゴリ (ExterQual等20列) は従来通り数値に変換 (順序関係の保持)
  - LOO Target Encoding, ドメイン特徴量, 歪度補正は既存パイプラインと同一
- **ハイパーパラメータ**: iterations=3000, lr=0.05, depth=6, RMSE, early_stopping=100

- **CatBoost単体CV**: **0.11321 ± 0.00668** (GBR 0.11268 に次ぐ第2位)
- **7モデル比較**:
  | モデル | RMSLE | ±std |
  |--------|-------|------|
  | GBR | 0.11268 | 0.00695 |
  | **CatBoost** | **0.11321** | **0.00668** |
  | ElasticNet | 0.11655 | 0.00785 |
  | Lasso | 0.11660 | 0.00803 |
  | Ridge | 0.11808 | 0.00793 |
  | XGBoost | 0.11883 | 0.00543 |
  | LightGBM | 0.12347 | 0.00715 |

- **スタッキング (7モデル → Ridge)**:
  - seed=42: Stack CV=0.10909 (6-model: 0.10971, **-0.00062**)
  - seed=123: Stack CV=0.11029 (6-model: 0.11042, -0.00013)
  - seed=456: Stack CV=0.10931 (6-model: 0.10971, -0.00040)
- **メタウェイト (seed=42)**: GBR=0.375, **CatBoost=0.266**, ElasticNet=0.181, Lasso=0.132, XGBoost=0.051, Ridge=0.042, LightGBM=-0.029
  - CatBoostが第2位のウェイト → GBR/XGBoostと異なる予測パターンでアンサンブルに大きく貢献

→ 提出: **Public 0.11994** (0.12030から-0.00036改善、**初の0.12突破!**)
→ CV-Public gap: 0.01059→0.01085 (seed=42基準で微増、ただし絶対スコアは両方とも改善)

- **Feature Importance分析**:
  - CatBoostでもLuxury_Space_Indexが圧倒的#1 (28.6%、GBRの14.8%の約2倍)
  - カテゴリカル変数の合計寄与は7.77% (21列) — 上位はMSZoning(#14, 1.7%), SaleCondition(#17, 1.4%)
  - 順序カテゴリ (KitchenQual#10, FireplaceQu#11等) は数値変換後も高い寄与
  - CatBoostはOverallCond(#5), RemodAge(#9)をGBRより重視 → 異なる特徴量選好がアンサンブル多様性に貢献
  - ドメイン特徴量: LSI=#1(28.6), LSR=#32(0.68), Is_SFL=#89(0.005) — GBRと同様の傾向

### Step 10: QOL_Score (Quality of Life スコア)
OOF残差分析 (figure 14) から、「低QOLの家が系統的に過大評価される」パターンを発見。
9つのバイナリアメニティ指標を合計したQOL_Score (0-9) を追加。

- **9つのQOLコンポーネント**:
  CentralAir, HasGarage, HasBsmt, HasFireplace, Functional=Typ,
  GoodKitchen(≥Gd), GoodExterior(≥Gd), GoodBasement(≥Gd), ModernBath(FullBath≥2)

- **Ablation Study (Stack CV, seed=42基準)**:
  | 特徴量の組み合わせ | Stack CV | vs baseline |
  |---------------------|----------|-------------|
  | **QOL_Score only** | **0.10868** | **-0.00041** |
  | QOL_Score + Qual_QOL_Gap | 0.10925 | +0.00016 |
  | Is_Distressed only | 0.10993 | +0.00084 |
  | Score+Gap+Distressed | 0.10912 | +0.00003 |
  | All 4 (Score+Gap+Distressed+ModDeficit) | 0.10971 | +0.00062 |

→ **QOL_Score単体のみが改善**。他の特徴量 (Qual_QOL_Gap, Is_Distressed, Modernization_Deficit) は冗長・ノイズとなり悪化。

- **各モデル単体CV (QOL_Score only)**:
  | モデル | Before | After | 差分 |
  |--------|--------|-------|------|
  | Ridge | 0.11808 | 0.11821 | +0.00013 |
  | Lasso | 0.11660 | 0.11658 | -0.00002 |
  | ElasticNet | 0.11655 | 0.11656 | +0.00001 |
  | GBR | 0.11268 | 0.11260 | -0.00008 |
  | **XGBoost** | **0.11883** | **0.11716** | **-0.00167** |
  | **LightGBM** | **0.12347** | **0.12329** | **-0.00018** |
  | CatBoost | 0.11321 | 0.11345 | +0.00024 |

→ XGBoostが最大の改善 (-0.00167)。QOL_Scoreは各アメニティの「総合評価」を明示化し、tree系の分割効率を向上。

- **マルチシード Stack CV**:
  - seed=42: 0.10909→**0.10868** (-0.00041)
  - seed=123: 0.11029→**0.10901** (-0.00128)
  - seed=456: 0.10931→**0.10876** (-0.00055)

→ 提出: **Public 0.11992** (0.11994から-0.00002微改善、新ベスト)
→ CV-Public gap: 0.01124 (seed=42基準、0.01085から微増)

### 失敗した改善試行
| 手法 | 結果 | 理由 |
|------|------|------|
| 交互作用特徴量 (OverallQual×GrLivArea等) | CV悪化 0.11134→0.11237 | tree系モデルは自動で交互作用を捉える。冗長な特徴量がノイズに |
| RobustScaler (線形モデルに適用) | 改善なし | log1p変換済み特徴量にはスケーリングの追加効果が薄い |
| KernelRidge (polynomial kernel) | CV 0.12793 (劣化) | 特徴量数が多い状態では線形モデルに劣る |
| IsPartial×OverallQual交互作用 | Public 0.12095 (+0.00032悪化) | 傾き差1.54xは有意だが、Partial=125件(8.6%)で効果薄。tree系が微悪化し相殺 |
| NbhdCluster (K-Means k=5) | CV改善0.10993→0.10956だがPublic 0.12089 (+0.00026悪化) | 離散境界がテストデータと微妙にずれる。CV-Public gap拡大 (0.01070→0.01133) = 軽度の過学習 |
| Garage PCA統合 (3変数→1) | CV微改善0.10993→0.10981だがPublic 0.12091 (+0.00028悪化) | PC1は79.7%のみ説明。GarageFinishの順序情報が失われ、情報損失がPublicで表面化。Gap拡大 (0.01070→0.01110) |
| Optunaハイパーパラメータ最適化 | CV大幅改善(0.10868→0.10791)だがPublic 0.12048 (+0.00056悪化) | n=1460の小データでCV過適合。個別CV改善がPublicに反映されず |
| Value_Standard_Score + VSS×GrLivArea | 線形/CatBoost改善だがGBR大幅悪化。Public 0.12033 (+0.00041悪化) | GBRが自力で交互作用を捉えるため明示的特徴量がノイズに。GBR高重みのスタックで全体悪化 |

---

## 提出履歴

| # | モデル | CV | Public | 改善幅 |
|---|--------|----|--------|--------|
| 1 | GBR単体 | 0.11478 | 0.12933 | (ベースライン) |
| 2 | OOFスタッキング (6モデル→Ridge) | 0.11134 | 0.12271 | -0.00662 |
| 3 | ブレンディング (手動加重平均) | - | 0.12323 | -0.00610 |
| 4 | マルチシードスタッキング (3 seeds) | 0.11047-0.11142 | 0.12246 | -0.00687 |
| 5 | **マルチシード + LOO TE (Neighborhood)** | **0.10993-0.11042** | **0.12063** | **-0.00870** |
| 6 | +IsPartial×OverallQual交互作用 | 0.11011 | 0.12095 | +0.00032 (悪化) |
| 7 | +NbhdCluster (K-Means k=5) | 0.10956-0.11021 | 0.12089 | +0.00026 (悪化) |
| 8 | マルチシード + ドメイン特徴量 (Is_SFL, LSR, LSI) | 0.10971 | 0.12030 | -0.00903 |
| 9 | **7モデル マルチシード + CatBoost追加** | **0.10909-0.11029** | **0.11994** | **-0.00939** |
| 10 | **+ QOL_Score** | **0.10868-0.10901** | **0.11992** | **-0.00941** |
| 11 | Optuna最適化 6モデル(CB除外) | 0.10763-0.10829 | 0.12048 | +0.00056 (悪化) |
| 12 | 6モデル(CB除外) 元パラメータ | 0.10898-0.10910 | 0.12012 | +0.00020 (悪化) |
| 13 | +Value_Standard_Score + VSS×GrLivArea | 0.10873-0.10923 | 0.12033 | +0.00041 (悪化) |

---

### Step 12: Value_Standard_Score (2026-03-04)

**目的**: 生活基盤の充実度スコア + GrLivArea交互作用

**Value_Standard_Score 定義 (range: -2 ~ +4):**
- 加点(+1): HeatingQC≥Gd, FullBath≥2, GarageCars≥2, CentralAir=Y
- 減点(-1): Condition1が騒音源(Artery/Feedr/RRNn/RRNe/RRAn/RRAe), Functional≠Typ
- 交互作用: VSS_x_GrLivArea = Value_Standard_Score × GrLivArea

**Ablation結果 (seed=42, Stack CV):**
| Variant | Stack CV | vs #10 |
|---------|----------|--------|
| Baseline (#10) | 0.10868 | - |
| VSS only | 0.10900 | +0.00032 |
| VSS + interaction | 0.10923 | +0.00055 |
| Interaction only | 0.10977 | +0.00109 |

**個別モデル影響 (VSS+interaction, seed=42):**
| モデル | #10 | +VSS | 変化 |
|--------|------|------|------|
| Ridge | 0.11808 | 0.11745 | **-0.00063** |
| Lasso | 0.11652 | 0.11613 | **-0.00039** |
| ElasticNet | 0.11645 | 0.11599 | **-0.00046** |
| GBR | 0.11260 | 0.11451 | +0.00191 |
| CatBoost | 0.11345 | 0.11219 | **-0.00126** |

**考察:**
- 線形モデルとCatBoostは改善（交互作用を明示的に渡す効果）
- GBRが大幅悪化（+0.00191）→ GBRは自力で交互作用を捉えており、明示的特徴量がノイズに
- GBRのスタック重みが高いため（~0.35）、全体が沈む
- Public LB: 0.12033（#10の0.11992より+0.00041悪化）

→ 交互作用特徴量はtree系モデルにはノイズとなりやすい（Step 2の知見と一致）

---

### Step 11: Optunaハイパーパラメータ最適化 (2026-03-04)

**目的**: GBR, XGBoost, LightGBMのハイパーパラメータをOptuna (各40試行) で最適化

**Optuna結果 (個別モデルCV):**
| モデル | Before | After | 改善幅 |
|--------|--------|-------|--------|
| GBR | 0.11260 | 0.10868 | -0.00392 |
| XGBoost | 0.11716 | 0.11431 | -0.00285 |
| LightGBM | 0.12329 | 0.11716 | -0.00613 |
| CatBoost | 0.11345 | (未完了 - 実行時間過大で中止) | - |

**CatBoost除外理由**: 1試行あたり~16分 (5-fold CV × 2000-5000 iterations) → 40試行で~10時間以上

**スタッキング結果 (CatBoost除外):**
- Optunaパラメータ: Stack CV 0.10791 → Public **0.12048** (悪化)
- 元パラメータ: Stack CV 0.10905 → Public **0.12012** (悪化)

**考察:**
1. Optuna最適化はCV過適合: 個別CVは大幅改善だがPublic LBは悪化
2. CatBoost除外でPublic LBが~0.002悪化: アンサンブルの多様性に貢献していた
3. GBRの重みが~0.49に上昇 (CatBoost除外でGBR依存が過剰に)

→ CatBoostは実行速度の問題があるが、アンサンブルへの貢献は大きい
→ Optunaの個別CVベストは汎化性能を保証しない (特に小データ n=1460)

---

## 現在の状態

### ベストモデル
- **マルチシード OOF スタッキング + CatBoost + LOO TE + ドメイン特徴量 + QOL_Score** (提出#10)
  - Base: Ridge, Lasso, ElasticNet, GBR, XGBoost, LightGBM, CatBoost (各5-fold CV)
  - Meta: Ridge (alpha=1.0)
  - Neighborhood LOO TE (smooth=10)
  - ドメイン特徴量: Is_SFL, Living_Space_Ratio, Luxury_Space_Index
  - QOL_Score: 9バイナリアメニティ指標の合計 (0-9)
- **CV: 0.10868** (seed=42), **Public: 0.11992**
- **CV-Public gap: 0.01124**

### 現在のmain.py状態
- CatBoost除外済み (6モデル構成)、パラメータは元の値に戻し済み
- CatBoostPreprocessor クラスはコードに残存 (将来復活用)

### 主要ファイル
- `house-prices/main.py` — メインパイプライン (前処理 + LOO TE + ドメイン特徴量 + QOL + 6モデル比較 + スタッキング)
- `house-prices/optuna_tune.py` — Optunaチューニングスクリプト (GBR, XGB, LGBM, CatBoost)
- `house-prices/eda.ipynb` — EDAノートブック
- `house-prices/data/` — train.csv, test.csv, data_description.txt
- `house-prices/figures/` — 01〜14の連番図

### 今後の改善案
1. 他カテゴリへのTE拡張 (MSSubClass, Exterior1st/2nd等)
2. Repeated KFoldによる有意差検定
3. CV-Public gap縮小のための正則化強化
4. 特徴量選択 (Boruta / Permutation Importance)
5. CatBoost復活の検討 (速度改善策: iterations削減+early_stopping厳格化)
