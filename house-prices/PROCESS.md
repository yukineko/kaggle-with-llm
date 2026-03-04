# House Prices: 作業引き継ぎドキュメント

## コンペ概要
- **Kaggle**: House Prices - Advanced Regression Techniques
- **目標**: 住宅価格 (SalePrice) の予測、評価指標は RMSLE
- **データ**: Train 1460行×81列、Test 1459行×80列

---

## 現在のベストスコア (提出#10)

| 指標 | 値 |
|------|----|
| **Public LB** | **0.11992** |
| CV (seed=42) | 0.10868 |
| CV-Public gap | 0.01124 |

### ベストモデル構成
- **7モデル マルチシード OOF スタッキング**
  - Base models: Ridge, Lasso, ElasticNet, GBR, XGBoost, LightGBM, CatBoost
  - Meta model: Ridge (alpha=1.0)
  - Seeds: 42, 123, 456 の3seed平均
- **前処理パイプライン** (`HousePricesPreprocessor`):
  - 欠損値処理 (NA="なし"の15カテゴリカル列、LotFrontageはNeighborhood中央値補完等)
  - 順序カテゴリ20列の手動マッピング
  - Neighborhood LOO Target Encoding (smooth=10)
  - ドメイン特徴量: Is_SFL, Living_Space_Ratio, Luxury_Space_Index
  - QOL_Score (9バイナリアメニティ指標合計)
  - 歪度補正 (log1p)、外れ値除去 (GrLivArea>4000 & 低価格の2件)
- **CatBoost**: `CatBoostPreprocessor` (Label Encodingスキップ、カテゴリカル列を文字列のまま保持)

---

## 現在のコード状態 (WIP・未コミット)

### main.py の変更点 (前回コミットからの差分)
1. **CatBoost復活**: iterations=3000→1000、l2_leaf_reg=3.0 追加
2. **CatBoostPipeline クラス追加**: cat_features 自動渡しのラッパー (コード簡略化)
3. **evaluate_models / stacking_predict リファクタリング**: CatBoostの手動CV処理を `make_pipeline(use_catboost=True)` で統一
4. **新特徴量追加 (実験中)**:
   - `_qol_features()`: QOL_Score → **検証済み、改善あり** (Step 10)
   - `_value_standard_features()`: Value_Standard_Score + VSS_x_GrLivArea → **Public LB悪化** (Step 12)
   - `_pure_comfort_features()`: Pure_Comfort_Score (Z-score標準化) → **未検証**
   - `_family_stability_features()`: Family_Stability_Index + Family_Premium → **未呼び出し** (transform()から呼ばれていない)
   - `Livable_Area`: 1stFlrSF + 2ndFlrSF + TotalBsmtSF × BsmtFinType1 → **未検証**

### 重要: ベスト再現に必要なこと
現在のmain.pyはベスト提出 (#10) の状態ではない。ベストを再現するには:
1. `_value_standard_features()` の呼び出しを除去 (or コメントアウト)
2. `_pure_comfort_features()` の呼び出しを除去
3. `Livable_Area` 特徴量の追加を除去
4. CatBoostのパラメータを元に戻す: `iterations=3000, depth=6, loss_function='RMSE'` (l2_leaf_reg除去)

あるいは、これらの実験的特徴量を一旦無効化してから再実行する方が安全。

---

## 改善の履歴 (時系列)

| Step | 内容 | CV | Public | 結果 |
|------|------|-----|--------|------|
| 3 | 6モデル比較 (GBR単体ベスト) | 0.11460 | 0.12933 | baseline |
| 4 | OOFスタッキング (6→Ridge) | 0.11134 | 0.12271 | 改善 |
| 6 | マルチシード (3 seeds) | 0.11047-0.11142 | 0.12246 | 改善 |
| 7 | LOO Target Encoding (Neighborhood) | 0.10993-0.11042 | 0.12063 | 大幅改善 |
| 8 | ドメイン特徴量 (Is_SFL, LSR, LSI) | 0.10971 | 0.12030 | 改善 |
| 9 | CatBoost追加 (7モデル) | 0.10909-0.11029 | 0.11994 | 改善 |
| **10** | **QOL_Score** | **0.10868-0.10901** | **0.11992** | **ベスト** |
| 11 | Optuna最適化 | 0.10791 | 0.12048 | 悪化 (過学習) |
| 12 | Value_Standard_Score | 0.10873-0.10923 | 0.12033 | 悪化 |

---

## 失敗した手法まとめ

| 手法 | 失敗理由 |
|------|----------|
| 交互作用特徴量 (手動) | tree系が自動で捉えるため冗長 |
| RobustScaler | log1p変換済みには効果薄 |
| KernelRidge | 高次元で線形モデルに劣る |
| IsPartial×OverallQual | Partial=125件(8.6%)で効果薄 |
| NbhdCluster (K-Means) | テストと微妙にずれる、CV-Public gap拡大 |
| Garage PCA | 情報損失がPublicで表面化 |
| Optuna最適化 | n=1460でCV過適合 |
| Value_Standard_Score | GBRが大幅悪化、スタック全体が沈む |

### 学んだ教訓
1. **tree系モデルに明示的交互作用を渡すとノイズになりやすい** — tree系は自動で交互作用を学習する
2. **小データ (n=1460) ではCV過適合に注意** — 個別CV改善がPublicに反映されない
3. **CatBoostはアンサンブル多様性に大きく貢献** — 除外するとPublic ~0.002悪化
4. **CV-Public gap が拡大する特徴量は過学習の兆候** — gap<0.011を目安に

---

## 今後の改善候補 (優先度順)

### 高優先度
1. **他カテゴリへのTE拡張** — MSSubClass, Exterior1st/2nd 等の高カーディナリティ名義変数
   - Neighborhoodで-0.00183改善した実績あり
   - LOO + smoothing で過学習防止を維持
2. **CatBoostパラメータ調整** — iterations=3000に戻し、early_stopping活用
   - 現状iterations=1000は保守的すぎる可能性
3. **特徴量選択** — Boruta or Permutation Importance で冗長特徴量を除去
   - 特徴量数が増えてきたので効果的かも

### 中優先度
4. **Repeated KFold** — 5-fold × 3回で安定度を検証 (有意差検定)
5. **正則化強化** — CV-Public gap (0.01124) を縮める
   - メタモデルの alpha 調整、ベースモデルの正則化パラメータ見直し
6. **ElasticNet/Lasso のalpha grid search** — 現状固定値

### 低優先度 (リスク高)
7. **NN (MLP) 追加** — アンサンブル多様性のため。ただし小データでは過学習リスク
8. **ターゲットエンコーディングの平滑化パラメータチューニング** — smooth=10は経験則

---

## 環境・実行方法

### Python環境
```
Python 3.13: c:/Users/hiroyuki_nakayama/AppData/Local/Programs/Python/Python313/python.exe
パッケージ: numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, kaggle
```

### 実行
```bash
cd house-prices
python main.py
```
- 全モデル比較 + スタッキング + マルチシード提出ファイル生成
- 実行時間: ~10-15分 (CatBoost含む)

### 提出 (curl)
```bash
# Bearer認証でKaggle APIに直接提出 (kaggle CLIはKGATトークン非対応)
# 詳細は MEMORY.md の「Kaggle API認証」セクション参照
```

### ファイル構成
```
house-prices/
├── main.py                    # メインパイプライン (前処理+モデル+スタッキング)
├── optuna_tune.py             # Optunaチューニング (参考用、結果は過学習)
├── eda.ipynb                  # EDAノートブック
├── PROGRESS.md                # 詳細な実験記録
├── PROCESS.md                 # ← このファイル (引き継ぎ用)
├── data/                      # train.csv, test.csv, data_description.txt
├── figures/                   # 分析図 (01-16)
└── submission_*.csv           # 各種提出ファイル
```

---

## 次のセッションでやるべきこと

1. **まずmain.pyをベスト状態に戻す**:
   - `_value_standard_features()` と `_pure_comfort_features()` の呼び出しを `transform()` から除去
   - `Livable_Area` 特徴量を除去
   - CatBoostパラメータを `iterations=3000, loss_function='RMSE'` に戻す
   - 実行して Public 0.11992 が再現できることを確認

2. **新しい改善を試す** (上記「今後の改善候補」参照):
   - MSSubClass等へのTE拡張が最有望
   - 1つずつ追加してCV + Public LBで検証
   - CV-Public gap が拡大しないことを確認
