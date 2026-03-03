# Kaggle Titanic 進捗記録

## セッション1 (2026-03-02)

### 初期構築
1. **EDA**: 基本統計量、欠損値、生存率(Sex/Pclass別)、年齢分布、運賃分布、相関行列
2. **ベースラインモデル構築**: LightGBM → XGBoost → SVM
3. **特徴量追加**: Title, Deck, AgePclass, TicketFreq, GroupSurvival
4. **GroupKFold導入** (Union-Find: 姓+チケット統合, 609グループ)
5. **GridSearchCV** で全5モデル (LGB, XGB, LR, SVM, KNN) チューニング

### 提出結果
| # | モデル | Public |
|---|--------|--------|
| 1 | LightGBM + GroupSurvival | 0.76555 |
| 2 | XGBoost (GroupSurvivalなし) | 0.76315 |
| 3 | LightGBM 7特徴量 | 0.75358 |
| 4 | XGBoost 15特徴量 + GroupKFold | 0.76794 |
| 5 | LightGBM 18特徴量 + GroupKFold | 0.76076 |
| 6 | SVM 12特徴量, GroupKFold | **0.77751** |

---

## セッション2 (2026-03-03 前半)

### リーク修正
- train-only stats での前処理修正 → Public 0.76794 (一旦悪化)
- SVM C=0.5 gamma=0.05 で再チューニング → Public **0.77751** 復帰

### 失敗した手法
- **Seed Averaging** (23モデル平均): Public 0.75837 — 弱モデル(LR,KNN)がSVMの良い予測を希釈
- **Post Processing** (チケット/姓グループ補正, 79件反転): Public 0.72727 — ルールベース補正が大量に誤り

### fold内前処理リークの発見と修正
- **問題**: Age/Fare補完等がCV前に全データで実行 → val_foldの情報がtrain_foldの統計量に混入
- **影響**: CV 0.034 過大評価 (fold外 0.8283 vs fold内 0.7946)
- **修正**: TitanicPreprocessor クラス (sklearn BaseEstimator/TransformerMixin) を作成し Pipeline 内に配置
- **結果**: SVM CV 0.8160 (Public 0.77751), XGBoost CV 0.8238 (Public 0.76794)

### 提出結果
| # | モデル | CV | Public |
|---|--------|----|--------|
| 7 | SVM リーク修正 | - | 0.76794 |
| 8 | SVM C=0.5 gamma=0.05 リーク修正済 | 0.8283* | **0.77751** |
| 9 | Seed Averaging 23モデル | - | 0.75837 |
| 10 | SVM + Post Processing | - | 0.72727 |
| 11 | XGBoost fold内前処理 | 0.8238 | 0.76794 |
| 12 | SVM C=0.1 fold内前処理 | 0.8160 | **0.77751** |

*#8のCV 0.8283はfold外前処理時代の値（リーク含み）

---

## セッション2 (2026-03-03 後半) — 上級CV手法

### Step 1: CVを確率分布として扱う
- **Repeated GroupKFold** (10回×5fold): CV 0.8130 ± 0.0045, **有意差閾値 ±0.009**
- **Paired t-test**: SVM C=0.1 > C=0.5 (p=0.0000)、SVM ≈ XGBoost ≈ LightGBM (有意差なし)
- **OOF予測分析**: 129人の不確実層 (P=0.3-0.7)
  - 3等女性: 34人, 53% accuracy
  - 1等男性: 45人, 53% accuracy

### Step 2: OOF Target Encoding
- **Surname/Ticket TE**: GroupKFoldと構造的非互換 (val_foldでoverlap=0%) → 使用不可
- **GroupKFold互換カテゴリ**: Embarked×Pclass, Deck, Title×Pclass, FamilySize (overlap=100%)
  - SVM: +0.009 (p=0.0000 有意改善)
  - LightGBM: +0.004 (p=0.079 ボーダーライン)
  - Public: SVM 0.77751 (変わらず), XGBoost 0.74880 (悪化)

### 提出結果
| # | モデル | CV | Public |
|---|--------|----|--------|
| 13 | XGBoost + TE (16feat) | 0.8272 | 0.74880 |
| 14 | SVM + TE (16feat) | 0.8227 | **0.77751** |

---

## セッション3 (2026-03-03 続き) — EDA知見の特徴量化

### 家族生存戦略の階層別分析 (eda.ipynb Task A,B,C)

**仮説: 3等の大家族は共倒れ、1等は子供を優先救出** → 支持された

- **Task A** (FamilySize×Pclass生存率曲線):
  - 3等 FamilySize 5以上で生存率 **0%** に急落 (42人)
  - 1等は FamilySize 5でも 50-100%
- **Task B** (All-or-Nothing分析):
  - 3等家族の **58%** が全員死亡 (1等は15%)
  - 3等大家族(FS>=5) 10家族中 **8家族が全員死亡** (Goodwin, Sage, Rice, Panula等)
- **Task C** (子供の生存率):
  - 3等大家族の子供: **18.8%** (n=32) vs 小家族: 75.0% (n=20)
  - 1等/2等の子供: 83-100%

### 新特徴量の検証

| 特徴量 | SVM差分 | p値 | 採用 |
|--------|---------|-----|------|
| **Deadly_Large_Family** (Pclass=3 & FS>=5) | **+0.0100** | 0.0000 | 採用 |
| Ticket_Group_Survival (同一チケット他者の生存率) | - | - | 廃止 (GroupKFold非互換) |
| Female_Upper_Count (上等女性の人数, LOO) | -0.0006 | 0.2796 | 不採用 |
| Master_Count (男児の人数, LOO) | (上に含む) | (上に含む) | 不採用 |

**Ticket_Group_Survival廃止の理由**: GroupKFoldで同一チケット=同一グループ=同一fold → val_foldのチケットはtrain_foldに存在しない → val_foldで常に-1 (定数) → CV評価不能。train_fold内では自分のSurvivedを含むため目的変数リーク。

### 最終提出
| # | モデル | CV | Public |
|---|--------|----|--------|
| 15 | SVM C=1.0 gamma=0.01 + TE + DLF (17feat) | **0.8351** | **0.77751** |

---

## 現在の状態

### ベストモデル
- **SVM** (C=1.0, gamma=0.01, rbf)
- **17特徴量**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, FamilySize, TicketFreq, Deck, AgePclass, Embarked_Pclass_te, Deck_te, Title_Pclass_te, FamilySize_te, Deadly_Large_Family
- **CV: 0.8351**, Train-CV Gap: 0.0022, **Public: 0.77751**

### 主要ファイル
- `titanic/main.py` — メインパイプライン (TitanicPreprocessor + Pipeline + GridSearchCV)
- `titanic/eda.ipynb` — EDAノートブック (家族生存戦略分析含む)
- `titanic/figures/` — EDAグラフ画像
- `titanic/experiment_*.py` — 各実験スクリプト

### 学んだ重要な教訓
1. **fold内前処理**: 統計量はtrain_foldのみから計算する (Pipeline内にTransformerを配置)
2. **GroupKFold制約**: 同一グループ=同一fold → グループ固有の特徴量はCV不可能
3. **有意差検定**: Repeated GKF + paired t-test で ±0.009 が閾値 (それ以下はノイズ)
4. **アンサンブルの落とし穴**: 弱モデルを混ぜると最強モデルの予測を希釈
5. **CVとPublicの乖離**: CVは改善してもPublicは0.77751で停滞（テストデータ418人の分散が大きい）
