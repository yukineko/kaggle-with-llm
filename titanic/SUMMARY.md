# Kaggle Titanic プロジェクト サマリ

## やったこと

### Phase 1: 基盤構築
- **EDA**: 欠損値、生存率(Sex/Pclass/Age別)、運賃分布、相関行列を可視化
- **特徴量エンジニアリング**: Title抽出、Deck、AgePclass、TicketFreq、FamilySize等 12個の基本特徴量を作成
- **5モデル比較**: LightGBM, XGBoost, LogisticRegression, SVM, KNN をGridSearchCVでチューニング
- **GroupKFold導入**: Union-Findで姓+チケットを統合し、家族を同一foldに配置 (609グループ)

### Phase 2: リーク修正
- **fold外前処理リークの発見**: Age/Fare補完がCV前に全データで実行されていた → CVが0.034過大評価
- **Pipeline化**: TitanicPreprocessor (sklearn BaseEstimator/TransformerMixin) を作成し、Pipeline内に配置することでfold内前処理を実現
- **GroupKFold×特徴量の構造的非互換**: Surname/Ticket系の特徴量はGroupKFoldでval_foldとのoverlap=0% → CV評価不能

### Phase 3: 上級CV手法
- **CVを確率分布として扱う**: Repeated GroupKFold (10回×5fold) で分散を測定。有意差閾値 ±0.009 を確立
- **Paired t-test**: 同一fold分割での比較で、これまでの「改善」の多くがノイズだったと判明
- **OOF Target Encoding**: GroupKFold互換カテゴリ (Embarked×Pclass, Deck, Title×Pclass, FamilySize) で有意改善

### Phase 4: EDA駆動の特徴量
- **家族生存戦略分析**: 3等大家族(FS>=5)の生存率0%、全員死亡率80%、子供も18.8%しか生存しないことを発見
- **Deadly_Large_Family**: 上記知見をバイナリ特徴量化。SVM CV +0.01 (p=0.0000) の有意改善
- **Ticket_Group_Survival**: 目的変数を使うためGroupKFoldと構造的非互換 → 廃止
- **Female_Upper_Count / Master_Count**: 説明変数のみだがDLFと冗長 → 追加効果なし

---

## 学び

### 1. リーク管理が最も重要
前処理をfold外で行うと、知らないうちにCVが0.034も過大評価される。**Pipeline内にTransformerを置く**ことで、sklearn/GridSearchCVが自動的にfold内前処理を保証する。「CVが高い」は信頼できない — 正しいCV設計が先。

### 2. GroupKFoldは強力だが制約もある
家族を同一foldにまとめることでリーク防止できるが、同時に**グループ固有の情報(Surname, Ticket)をval_foldに持ち込めない**という制約が生まれる。Target Encodingや家族生存情報はGroupKFoldと構造的に矛盾する。この制約を理解せずに特徴量を作ると、train_foldではリーク、val_foldでは定数という最悪のパターンになる。

### 3. 統計的検定なしの「改善」は信用できない
Repeated GroupKFold + paired t-test で初めて、+0.002程度の差はノイズだと判明。**有意差閾値(±0.009)を知らずにチューニングすると、ランダム変動を改善と誤認してPublicで裏切られる**。

### 4. EDA → 仮説 → 特徴量 の流れが有効
漫然と特徴量を追加するのではなく、EDAで「3等大家族は全員死亡する」という具体的な仮説を立て、それをDeadly_Large_Familyとして特徴量化したことで有意な改善が得られた。**データの構造を理解してから特徴量を作る**。

### 5. モデル選択よりも特徴量とCV設計
SVM, XGBoost, LightGBMの間に統計的有意差はなかった(paired t-test)。モデルの違いよりも、**正しいCV設計(リーク排除)と意味のある特徴量**(DLF)の方がはるかに大きなインパクトを持つ。

### 6. アンサンブルの落とし穴
23モデルのSeed Averagingは、弱モデル(LR, KNN)がSVMの良い予測を希釈して悪化した。**全モデルが同程度に強くないと、アンサンブルは害になる**。

---

## 結果

| ベストモデル | SVM (C=1.0, gamma=0.01, rbf) |
|---|---|
| 特徴量数 | 17 (12基本 + 4TE + DLF) |
| CV Accuracy | **0.8351** |
| Public Score | **0.77751** |
| Train-CV Gap | 0.0022 |
| 提出回数 | 15回 |

---

## 今後やること

### 短期: スコア改善に直結しうる施策

1. **OOF不確実層の深掘り**
   - 3等女性(34人, 53%acc) と 1等男性(45人, 53%acc) の誤分類パターンを個別分析
   - この層に効く特徴量やルールを探す

2. **特徴量の再選択**
   - 17特徴量の中で効いていないものを除外 (Permutation Importance等)
   - ノイズ特徴量の削減でSVMの汎化性能が上がる可能性

3. **ハイパーパラメータの広域探索**
   - SVM: C, gamma の連続空間をBayesian Optimization (Optuna) で探索
   - 現在のGridSearchは離散グリッドなので最適値を逃している可能性

### 中期: 新しいアプローチ

4. **Stacking**
   - OOF予測を1段目、メタモデルを2段目とするStacking
   - Seed AveragingではなくOOFベースならモデルの多様性が活きる

5. **テストデータの説明変数情報の活用**
   - train+testで共通のTicketグループがある場合、testの説明変数(Sex, Pclass等)からグループの属性を推定できる → これはリークではない

6. **Pseudo Labeling**
   - 確信度の高いtest予測をtrainに追加して再学習
   - CVで効果測定が難しいので慎重に

### 長期: スキル拡張

7. **別コンペへの応用**
   - Titanicで学んだCV設計・リーク管理・特徴量検証のフレームワークを、テーブルデータコンペ(Housing Prices等)に適用
   - より大きなデータセットでRepeated KFoldやTEの効果を体験する

8. **Neural Network / Deep Learning**
   - TabNet, Transformer系のテーブルデータモデル
   - Titanicのような小データでは不利だが、手法として知っておく
