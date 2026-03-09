# House Prices: 作業引き継ぎドキュメント

## コンペ概要
- **Kaggle**: House Prices - Advanced Regression Techniques
- **目標**: 住宅価格 (SalePrice) の予測、評価指標は RMSLE
- **データ**: Train 1460行×81列、Test 1459行×80列

---

## 現在のベストスコア

| 指標 | 値 |
|------|----|
| **Public LB** | **0.11935** |
| 構成 | Feature cleanup + conservative XGB/LGBM (depth=3, lr=0.01) |

### 提出履歴 (主要)

| # | 日時 | Public | 内容 |
|---|------|--------|------|
| 50738317 | 3/6 | **0.11935** | Feature cleanup + conservative models |
| 50738791 | 3/6 | 0.12048 | HighValue interaction (悪化→revert) |
| 50734300 | 3/6 | 0.11993 | Simplified pipeline |
| - | 3/9 | (pending) | Geo+Dist_to_Center + Residual Top-40 multiseed |
| - | 3/6 | (pending) | SHAP Top-25 feature selection + multiseed |
| - | 3/5 | 0.11992 | QOL_Score (旧ベスト) |

---

## 現在のコード状態

### メインファイル: `pipeline.ipynb`
- **Colab/Local 両対応ノートブック** (main.py はレガシー、参考用)
- Colab 優先実行 (ローカル 3050 LTI は遅い)

### アーキテクチャ: 2-Stage Residual Learning
1. **Stage 1**: Ridge baseline (TotalSF, OverallQual, HouseAge の3変数) → OOF ベースライン予測
2. **Stage 2**: 7モデル OOF スタッキングで「残差（実際の値 − ベースライン予測）」を学習
3. **最終予測**: baseline_pred + residual_pred
4. **マルチシード平均** (seeds: 42, 123, 456)

### Pipeline クラス
1. **`HousePricesPreprocessor`**: `selected_features` パラメータで変数選抜対応
2. **`CatBoostPipeline`**: cat_features 自動渡し (カテゴリカル=object のまま)
3. **`SklearnPipeline`**: object 列を LabelEncode して数値化
4. **`make_pipeline(model, use_catboost, selected_features)`**: 統一インターフェース

### transform() パイプライン順序
```
_fill_missing → _ordinal_encode → _apply_te(廃止) → _geo_features →
_feature_engineering → _qol_features → _drop_redundant → _drop_cols →
_fix_skewness → [SHAP Top-N 変数選抜]
```

### ノートブック実行順序
```
Setup → Imports → 前処理定義 → Pipeline定義 → データ読み込み →
KNN Geo Price (OOF) → SHAP Top-40 選抜 → Stage 1 Baseline →
モデル定義 & CV評価 → 単体ベスト予測 → OOF スタッキング →
SHAP 値可視化 → マルチシード スタッキング → ファイル出力 → 自動提出
```

### 7モデル マルチシード OOF スタッキング
- Base: Ridge, Lasso, ElasticNet, GBR, XGBoost, LightGBM, CatBoost
- Meta: Ridge (alpha=1.0)
- Seeds: 42, 123, 456 の3seed平均

### モデルパラメータ (現在)
```python
'Ridge': Ridge(alpha=5.0)
'Lasso': Lasso(alpha=0.0003, max_iter=10000)
'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=10000)
'GBR': GBR(n_estimators=3000, lr=0.05, max_depth=4, min_samples_split=15,
           min_samples_leaf=10, max_features='sqrt', loss='huber', subsample=0.75)
'XGBoost': XGB(n_estimators=3000, lr=0.01, max_depth=3, subsample=0.8, colsample=0.8)
'LightGBM': LGBM(n_estimators=4000, lr=0.01, max_depth=3, num_leaves=7, min_child=50)
'CatBoost': CB(iterations=3000/4000(GPU), lr=0.03, depth=6, l2_leaf_reg=10, od_wait=100)
```

---

## 特徴量設計

### 地理的特徴量 (Neighborhood_te の代替)
- **Neighborhood_te は廃止** — Target Encoding は leak リスクがあり、地理的特徴量に移行
- **Latitude / Longitude**: Neighborhood → 代表座標 (AmesHousing R package + GIS data)
- **Dist_to_Center**: Ames 市中心 (42.05, -93.65) からのユークリッド距離
- **Dist_to_HighPrice_Center**: 高価格エリア中心 (NridgHt, NoRidge, StoneBr の重心) からの距離
- **KNN_Geo_Price**: 地理的に近い K=10 物件の LogPrice 平均 (OOF leak-free)
- SHAP Top-40 選抜で自動的に有効な地理特徴量が選択される

### 合成特徴量
- **数値**: TotalSF, TotalBath, HouseAge, TotalPorchSF, RemodAge, Living_Space_Ratio, Luxury_Space_Index, QOL_Score
- **バイナリ**: IsNew, HasRemod, HasGarage, HasBsmt, Has2ndFlr, HasPool, Is_SFL

### REDUNDANT_COLS (削除)
TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea, FullBath, HalfBath, BsmtFullBath, BsmtHalfBath, YearBuilt, GarageArea

### SHAP Top-40 変数選抜
- LightGBM feature importance で動的に上位40変数を選択
- `_te` サフィックスの変数は明示的に除外
- 予測クリッピング: CLIP_MIN=35000, CLIP_MAX=600000

### Colab ファイル出力 (絶対パス固定)
```
/content/submissions/          # submission CSV
/content/figures/              # SHAP 等の図
/content/final_submission.csv  # マルチシード最終提出
```
- 最終セルで Google Drive に自動同期: `MyDrive/kaggle-output/house-prices/`

### Kaggle 自動提出
- requests で Bearer 認証 (KGAT トークン)
- StartSubmissionUpload → PUT GCS → CreateSubmission (blobToken 単一文字列) → ポーリング
- 失敗時は blobFileTokens 配列形式にフォールバック

---

## 改善の履歴

| Step | 内容 | Public | 結果 |
|------|------|--------|------|
| 3 | 6モデル比較 (GBR単体) | 0.12933 | baseline |
| 4 | OOFスタッキング | 0.12271 | 改善 |
| 6 | マルチシード | 0.12246 | 改善 |
| 7 | LOO Target Encoding | 0.12063 | 大幅改善 |
| 8 | ドメイン特徴量 | 0.12030 | 改善 |
| 9 | CatBoost追加 (7モデル) | 0.11994 | 改善 |
| 10 | QOL_Score | 0.11992 | ベスト(旧) |
| 11 | Optuna最適化 | 0.12048 | 悪化 (過学習) |
| 19 | pipeline.ipynb化 + simplified | 0.11993 | 中立 |
| **20** | **Feature cleanup + conservative models** | **0.11935** | **現ベスト** |
| 21 | HighValue_Area interaction | 0.12048 | 悪化→revert |
| 22 | SHAP Top-25 選抜 | (pending) | CV=0.11455 |
| 23 | SHAP Top-40 + Neighborhood_te廃止 + Geo features | (pending) | Top-25→40に拡張 |
| **24** | **2-Stage Residual Learning + Dist_to_Center** | **(pending)** | Geo 5特徴量, 提出API修正 |

---

## 失敗した手法まとめ

| 手法 | 失敗理由 |
|------|----------|
| 交互作用特徴量 (手動) | tree系が自動で捉えるため冗余 |
| Optuna最適化 | n=1460でCV過適合 |
| 24精鋭変数のみ | n=1460では情報損失 > ノイズ削減 |
| HighValue_Area x OverallQual | 3エリアのフラグはノイズ化 (0.12048) |
| CatBoost depth=4 | 弱すぎ、depth=6が最適 |

### 学んだ教訓
1. **tree系に明示的交互作用を渡すとノイズになりやすい**
2. **小データ (n=1460) ではCV過適合に注意** — 個別CV改善がPublicに反映されない
3. **CatBoostはアンサンブル多様性に大きく貢献** — 除外/弱体化でPublic ~0.002悪化
4. **冗余変数の削除は有効** — GrLivArea, GarageArea 等の削除で 0.11992→0.11935
5. **保守的ハイパラ (低depth, 低lr) が小データでは効く** — XGB/LGBM depth=3, lr=0.01
6. **新特徴量は慎重に** — HighValue interaction で +0.001 悪化
7. **CV悪化 = 必ずしも悪いわけではない** — 過学習解消ならPublicで改善の可能性
8. **Neighborhood_te → 地理的特徴量への移行**: TE は leak リスクがあり、Lat/Lon/Distance/KNN で代替可能

---

## ファイル構成
```
house-prices/
├── pipeline.ipynb             # メインパイプライン (Colab/Local両対応)
├── main.py                    # レガシー (参考用)
├── PROGRESS.md                # 詳細実験記録
├── PROCESS.md                 # このファイル (引き継ぎ用)
├── data/                      # train.csv, test.csv
├── submissions/               # 提出CSV
└── figures/                   # 分析図
```

---

## 次のセッションでやるべきこと

1. **Colab で再実行して Public スコア確認** — Dist_to_Center 追加 + Residual Learning の効果検証
2. **SHAP 値で Neighborhood 依存度を確認** — Geo 特徴量が適切に分散しているか
3. **提出API の動作確認** — blobToken 形式で成功するか
4. **メタモデルのチューニング**: Ridge alpha=1.0 以外 (Lasso meta等)
5. **Top-40 vs Top-30 vs Top-50 の比較** — 最適な変数数の探索
