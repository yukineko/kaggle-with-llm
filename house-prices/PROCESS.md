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

| 日時 | Public | 内容 |
|------|--------|------|
| 3/6 | **0.11935** | **Feature cleanup + conservative models (現ベスト)** |
| 3/9 | **0.11946** | pipeline.ipynb 微調整 (2位) |
| 3/9 | 0.11993 | pipeline.ipynb |
| 3/9 | 0.12277 | pipeline.ipynb |
| 3/6 | 0.12048 | HighValue interaction (悪化→revert) |
| 3/6 | 0.12340 | SHAP Top-25 (削りすぎ) |
| 3/5 | 0.11992-0.11993 | CleanHybrid + NbhdBin 等 (Phase 2 各種) |
| 3/10 | 0.17501 | tabnet_pipeline v4 stacked (大幅悪化) |
| | | **全34回提出、詳細は PROGRESS.md 参照** |

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

| Phase | Step | 内容 | Public | 結果 |
|-------|------|------|--------|------|
| 1 | 3 | 6モデル比較 (GBR単体) | 0.12933 | baseline |
| 1 | 4 | OOFスタッキング | 0.12271 | 改善 |
| 1 | 6 | マルチシード | 0.12246 | 改善 |
| 1 | 7 | LOO Target Encoding | 0.12063 | 大幅改善 |
| 1 | 8 | ドメイン特徴量 | 0.12030 | 改善 |
| 1 | 9 | CatBoost追加 (7モデル) | 0.11994 | 改善 |
| 1 | 10 | QOL_Score | 0.11992 | main.pyベスト |
| 1 | 11 | Optuna最適化 | 0.12048 | 悪化 (過学習) |
| 1 | 12-16 | VSS, Starter_Eff, Comfort等 | 0.12033-0.12077 | 全て悪化 |
| 2 | 17 | Elite 24 features | 0.12505 | 悪化 (情報不足) |
| 2 | 18-22 | Hybrid/NbhdBin/ThermalEff等 | 0.11992-0.12417 | 中立〜悪化 |
| **2** | **17(cleanup)** | **Feature cleanup + conservative models** | **0.11935** | **全期間ベスト** |
| 2 | - | HighValue interaction | 0.12048 | 悪化→revert |
| 2 | - | SHAP Top-25 選抜 | 0.12340 | 悪化 (削りすぎ) |
| 3 | - | pipeline.ipynb 微調整 (3/9) | 0.11946 | 2位 |
| 3 | - | tabnet_pipeline v4 stacked | 0.17501 | 大幅悪化 |

---

## 失敗した手法まとめ

### Phase 1: main.py 時代の失敗 (3/3〜3/4)

| 手法 | Public | 悪化幅 | 失敗理由 |
|------|--------|--------|----------|
| IsPartial×OverallQual交互作用 | 0.12095 | +0.00032 | Partial=125件(8.6%)で効果薄。tree系が微悪化し相殺 |
| NbhdCluster (K-Means k=5) | 0.12089 | +0.00026 | 離散境界がtestデータとずれる。CV-Public gap拡大=軽度過学習 |
| Garage PCA統合 (3変数→1) | 0.12091 | +0.00028 | PC1は79.7%のみ説明。GarageFinishの順序情報が消失 |
| 交互作用特徴量 (OverallQual×GrLivArea等) | - | CV悪化 | tree系が自動で捉えるため冗余。スタックでノイズ化 |
| RobustScaler (線形モデルに適用) | - | 改善なし | log1p変換済み特徴量にはスケーリングの追加効果が薄い |
| KernelRidge (polynomial kernel) | - | CV 0.12793 | 特徴量数が多い状態では線形モデルに劣る |
| Optuna最適化 (GBR/XGB/LGBM) | 0.12048 | +0.00056 | 個別CVは大幅改善だがPublicで悪化。n=1460でCV過適合 |
| CatBoost除外 (6モデルスタック) | 0.12012 | +0.00020 | アンサンブル多様性の喪失。GBR依存が過剰に |
| Value_Standard_Score + VSS×GrLivArea | 0.12033 | +0.00041 | 線形/CatBoost改善だがGBR +0.00191大幅悪化。GBR高重みで全体沈む |
| Starter_Efficiency | 0.12066 | +0.00074 | QOL_Scoreと冗長。CatBoost除外の影響も |
| Pure_Comfort_Score (Z-score, 4 comp) | 0.12077 | +0.00085 | 同上。複合スコアの追加効果なし |

### Phase 2: pipeline.ipynb 特徴量選択の失敗 (3/5〜3/6)

| 手法 | Public | 悪化幅 | 失敗理由 |
|------|--------|--------|----------|
| Elite 24 features + lightweight CatBoost | 0.12505 | +0.00513 | 情報不足。24変数では住宅価格の多面性を捉えきれない |
| Hybrid SetA(24)/SetB(30) | 0.12417 | +0.00425 | 線形/tree別に変数を分けてもスタック時に相互補完できない |
| 96feat + NbhdBin(5q) + VSS/Comfort | 0.12060 | +0.00068 | 特徴量が多すぎてノイズ化。NbhdBin(量子化)は情報損失 |
| HighValue interaction + feature cleanup | 0.12048 | +0.00056 | HighValueエリアフラグはtree系にノイズ |
| SHAP Top-25 feature selection | 0.12340 | +0.00348 | 特徴量削りすぎ。Top-40からTop-25への15変数削除で情報喪失 |

### Phase 3: tabnet_pipeline.ipynb の失敗 (3/5, 3/10)

| 手法 | Public | 悪化幅 | 失敗理由 |
|------|--------|--------|----------|
| tabnet_pipeline 初期 (submission_final) | 0.13319 | +0.01327 | TabNet単体/初期構成が不安定 |
| tabnet_pipeline 最適化 | 0.13299 | +0.01307 | 微改善だがまだ大幅に劣る |
| tabnet_pipeline 簡素化 | 0.12858 | +0.00866 | 改善傾向だが不十分 |
| **Iowa v4 Stacked (TabNet+LGB+CB)** | **0.17501** | **+0.05566** | **Generational Adaptive Ensembleが壊滅的過学習。世代別重み付けがtestに合わない。TabNetは小データ(n=1460)で不安定** |

### 「中立」だった手法 (悪化はしないが改善もしない)

| 手法 | Public | 備考 |
|------|--------|------|
| CleanHybrid + ThermalEff + NbhdBin | 0.11992 | #10タイ。Phase 2の構成変更は中立 |
| Location bias LINEAR_ONLY | 0.11993 | ほぼ中立 |
| Simplified pipeline (no LINEAR_ONLY) | 0.11993 | コード簡素化。スコアは不変 |

### 学んだ教訓

**特徴量に関して:**
1. **tree系に明示的交互作用を渡すとノイズになりやすい** — GBRが自力で捉えるため冗余。特にGBRのスタック重みが高い(~0.35)ので全体を引きずり下ろす
2. **冗余変数の削除は有効** — GrLivArea(TotalSFと相関0.83), GarageArea(GarageCarsと相関0.88)の除去で0.11992→0.11935
3. **特徴量選択は40前後が最適** — 25は情報喪失、96はノイズ過多。SHAP Top-40がバランス点
4. **新特徴量は慎重に** — 34回中、特徴量追加で改善したのはLOO TE, ドメイン特徴量(LSI等), QOL_Scoreのみ
5. **Neighborhood_te → 地理的特徴量への移行**: TE は leak リスクがあり、Lat/Lon/Distance/KNN で代替可能
6. **PCA統合は情報損失** — Garage PCA (3→1) で順序情報が消失。元変数を残す方が良い

**モデル・CV に関して:**
7. **小データ (n=1460) ではCV過適合に注意** — Optuna個別CVは大幅改善だがPublicは悪化。CV-Public gapを常に監視
8. **CatBoostはアンサンブル多様性に大きく貢献** — 除外/弱体化でPublic ~0.002悪化。OOFの予測パターンがGBR/XGBと異なる
9. **保守的ハイパラ (低depth, 低lr) が小データでは効く** — XGB/LGBM depth=3, lr=0.01で過学習抑制
10. **TabNetは小データ (n=1460) で不安定** — 単体もアンサンブルも信頼性が低い。Iowa v4の0.17501は教訓
11. **CV悪化 = 必ずしも悪いわけではない** — 過学習解消ならPublicで改善の可能性

**プロセスに関して:**
12. **成功した改善は少ない** — 34回提出中、ベストを更新したのは#1→#2→#4→#5→#8→#9→#10→#28の8回のみ。残り26回は悪化または中立
13. **Phase 1 (main.py) の限界** — Step 10 (QOL_Score, 0.11992) 以降は6回連続悪化。パイプライン刷新が必要だった
14. **大きな改善は「構造変更」から** — スタッキング導入(-0.00662), CatBoost追加(-0.00939), 冗余削除(-0.00057)。小手先の特徴量追加ではない

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

### 優先度高
1. **#28 (0.11935) の構成を正確に再現・固定** — 現ベストの再現性を確保
2. **メタモデルのチューニング**: Ridge alpha=1.0 以外 (Lasso meta, HuberRegressor等)
3. **Top-40 vs Top-30 vs Top-50 の比較** — 最適な変数数の探索

### 優先度中
4. **Geo特徴量の効果検証** — Dist_to_Center, KNN_Geo_Price が実際にPublicで効いているか
5. **2-Stage Residual Learning の効果検証** — Residual Learning あり/なしの比較
6. **CatBoost のハイパラ微調整** — depth=6 vs 5, iterations, l2_leaf_reg

### やらないこと
- TabNet系 (小データで不安定、0.17501の教訓)
- 24特徴量以下への極端な絞り込み
- Optuna (CV過適合のリスク)
- 手動交互作用特徴量 (tree系にノイズ)
