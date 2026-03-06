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

※ 提出#13 (Location bias) も Public 0.11993 でほぼ同等

### ベストモデル構成
- **7モデル マルチシード OOF スタッキング**
  - Base models: Ridge, Lasso, ElasticNet, GBR, XGBoost, LightGBM, CatBoost
  - Meta model: Ridge (alpha=1.0)
  - Seeds: 42, 123, 456 の3seed平均
- **ハイブリッド特徴量アーキテクチャ**:
  - LINEAR モデル (Ridge, Lasso, ElasticNet): 全変数 + 合成特徴量
  - TREE モデル (GBR, XGBoost, LightGBM, CatBoost): 全変数 − 合成特徴量 (LINEAR_ONLY除外)
  - `MODEL_FEATURE_SET` dict + `_select_elite()` メソッドで制御
- **前処理パイプライン** (`HousePricesPreprocessor`):
  - 欠損値処理 (NA="なし"の15カテゴリカル列、LotFrontageはNeighborhood中央値補完等)
  - 順序カテゴリ20列の手動マッピング
  - Neighborhood LOO Target Encoding (smooth=10)
  - Neighborhood 5分位ビニング (`NbhdBin`)
  - ドメイン特徴量: Is_SFL, Living_Space_Ratio, Luxury_Space_Index
  - QOL_Score (9バイナリアメニティ指標合計)
  - Value_Standard_Score + VSS_x_GrLivArea (LINEAR_ONLY)
  - Pure_Comfort_Score (LINEAR_ONLY)
  - Livable_Area (LINEAR_ONLY)
  - Thermal_Efficiency (LINEAR_ONLY)
  - Location bias特徴量 (LINEAR_ONLY): IDOTRR_Distress, IDOTRR_QualCap, Elite_Area_Premium, Is_Distress_Sale, Distress_x_Qual, Snow_Maint_Burden, Burden_x_Area
  - 歪度補正 (log1p)、外れ値除去 (GrLivArea>4000 & 低価格の2件)
- **CatBoost**: `CatBoostPreprocessor` (Label Encodingスキップ、カテゴリカル列を文字列のまま保持)

### モデルパラメータ (現在)
```python
'Ridge': Ridge(alpha=5.0)
'Lasso': Lasso(alpha=0.0003, max_iter=10000)
'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=10000)
'GBR': GradientBoostingRegressor(n_estimators=3000, lr=0.05, max_depth=4, ...)
'XGBoost': XGBRegressor(n_estimators=2000, lr=0.05, max_depth=4, ...)
'LightGBM': LGBMRegressor(n_estimators=2000, lr=0.05, max_depth=5, ...)
'CatBoost': CatBoostRegressor(iterations=3000, lr=0.03, depth=6, l2_leaf_reg=10, od_wait=100)
```

---

## 現在のコード状態

### main.py アーキテクチャ
1. **`LINEAR_ONLY_FEATURES`** リスト: tree系モデルから除外する合成特徴量一覧
2. **`HousePricesPreprocessor`**: `feature_set` パラメータで LINEAR/TREE を切り替え
   - `_select_elite()`: TREE の場合 LINEAR_ONLY_FEATURES を除外
3. **`CatBoostPreprocessor`**: Label Encoding スキップ版
4. **`CatBoostPipeline`**: cat_features 自動渡しラッパー
5. **`MODEL_FEATURE_SET`** dict: モデル名 → 'LINEAR' or 'TREE'
6. **transform() パイプライン順序**:
   ```
   _fill_missing → _ordinal_encode → _apply_te → _nbhd_binning →
   _qol_features → _value_standard_features → _pure_comfort_features →
   _thermal_efficiency → _location_bias_features →
   _label_encode → _feature_engineering → _drop_cols → _fix_skewness → _select_elite
   ```

### main.py は現在のベスト (#10) の状態ではない
ベスト (#10, Public 0.11992) 以降に追加された実験的特徴量が含まれている:
- Location bias 特徴量 (IDOTRR, Elite, Distress, Snow) → Public 0.11993 (中立)
- Thermal_Efficiency → Public 0.11992 (中立、gap改善)
- VSS, Pure_Comfort, Livable_Area → これらは以前から含まれているが LINEAR_ONLY なので tree に影響なし

**現在の状態のまま実行すれば Public ~0.11993 が得られる。**

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
| 13 | 24精鋭変数のみ | — | 0.12505 | 大幅悪化 |
| 14 | ハイブリッド SetA/SetB 分割 | — | 0.12417 | 悪化 |
| 15 | 96変数復帰 + NbhdBin + CatBoost(d4) | — | 0.12060 | 悪化 |
| 16 | クリーンハイブリッド (LINEAR/TREE分割 + alpha調整) | — | 0.12019 | 悪化 |
| 17 | CatBoost復活 (iter3000,d6) + Thermal_Efficiency | 0.10868 | 0.11992 | ベスト同点 |
| 18 | Location bias (IDOTRR+Elite+Distress+Snow) | 0.10914 | 0.11993 | 中立 |

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
| 24精鋭変数のみ | n=1460では情報損失 > ノイズ削減 |
| ハイブリッド SetA/SetB | 変数の厳選が裏目 |
| CatBoost depth=4, iter=1000 | 弱すぎ (weight 0.076)、depth=6/iter=3000が最適 |
| Location bias (IDOTRR等) | 情報量不足で中立 (悪化はしない) |

### 学んだ教訓
1. **tree系モデルに明示的交互作用を渡すとノイズになりやすい** — tree系は自動で交互作用を学習する
2. **小データ (n=1460) ではCV過適合に注意** — 個別CV改善がPublicに反映されない
3. **CatBoostはアンサンブル多様性に大きく貢献** — 除外/弱体化するとPublic ~0.002悪化
4. **CV-Public gap が拡大する特徴量は過学習の兆候** — gap<0.011を目安に
5. **特徴量削減は小データでは逆効果** — 96変数→24変数でPublic 0.005悪化
6. **LINEAR_ONLY アーキテクチャは安全** — 線形モデルにのみ合成特徴量を渡せば tree に悪影響なし
7. **CatBoostの設定は重要** — iter=3000, depth=6, l2=10, od_wait=100 が最適

---

## 今後の改善候補 (優先度順)

### 高優先度
1. **他カテゴリへのTE拡張** — MSSubClass, Exterior1st/2nd 等の高カーディナリティ名義変数
   - Neighborhoodで-0.00183改善した実績あり
   - LOO + smoothing で過学習防止を維持
2. **特徴量選択** — Boruta or Permutation Importance で冗長特徴量を除去
   - LINEAR_ONLY特徴量が増えてきたので、効果のないものを刈り取る
3. **メタモデルのチューニング** — Ridge alpha=1.0 以外を試す (Lasso meta, ElasticNet meta)

### 中優先度
4. **Repeated KFold** — 5-fold × 3回で安定度を検証 (有意差検定)
5. **正則化強化** — CV-Public gap を縮める
6. **LightGBM改善** — 現在weight=-0.015〜-0.129 (負の重み)、チューニングか除外検討

### 低優先度 (リスク高)
7. **NN (MLP) 追加** — アンサンブル多様性のため。ただし小データでは過学習リスク
8. **ターゲットエンコーディングの平滑化パラメータチューニング** — smooth=10は経験則

---

## OOF残差分析の知見

### 系統的に外すパターン
- **IDOTRR (鉄道隣接地域)**: 平均|残差| 0.152 (全体平均の2倍以上)
- **SaleCondition=Abnorml/Family**: 過大評価する傾向 (市場外取引の割引を捉えきれない)
- **StoneBr/Crawfor**: 過小評価する傾向 (プレミアムを捉えきれない)

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
"c:/Users/hiroyuki_nakayama/AppData/Local/Programs/Python/Python313/python.exe" main.py
```
- 全モデル比較 + スタッキング + マルチシード提出ファイル生成
- 実行時間: ~10-15分 (CatBoost含む)

### 提出 (curl)
```bash
# Bearer認証でKaggle APIに直接提出 (kaggle CLIはKGATトークン非対応)
# 3ステップ: StartSubmissionUpload → PUT GCS → CreateSubmission
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

1. **現在のmain.pyはそのまま実行可能** (Public ~0.11993):
   - Location bias 特徴量は中立 (改善も悪化もない)
   - 除去してもしなくても Public ~0.11992-0.11993

2. **新しい改善を試す** (上記「今後の改善候補」参照):
   - MSSubClass等へのTE拡張が最有望
   - 1つずつ追加してCV + Public LBで検証
   - CV-Public gap が拡大しないことを確認

3. **LightGBMの扱いを検討**:
   - 現在メタモデルで負の重み (-0.015〜-0.129)
   - チューニングするか、6モデル構成に戻すか
