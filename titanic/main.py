import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb

train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")

# =====================================================
# 前処理クラス (sklearn Transformer)
#   fit(): train_fold のみから統計量を学習
#   transform(): 学習した統計量で任意のデータを変換
#   Pipeline内に置くことで、CV各foldで正しくリークなし前処理される
# =====================================================
TITLE_MAP = {
    'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
    'Mme': 'Mrs', 'Ms': 'Mrs', 'Mlle': 'Miss',
    'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
    'Dr': 'Officer', 'Rev': 'Officer',
    'Don': 'Royalty', 'Sir': 'Royalty', 'Dona': 'Royalty',
    'Lady': 'Royalty', 'Countess': 'Royalty', 'Jonkheer': 'Royalty',
}

FEATURE_COLS_BASE = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                     'Title', 'FamilySize', 'TicketFreq', 'Deck', 'AgePclass']

# GroupKFoldと矛盾しないTarget Encoding対象
TE_TARGETS = ['Embarked_Pclass', 'Deck', 'Title_Pclass', 'FamilySize']

# EDA知見: Pclass=3 & FamilySize>=5 は生存率7.4% (54人)
FEATURE_COLS = FEATURE_COLS_BASE + [t + '_te' for t in TE_TARGETS] + ['Deadly_Large_Family']

TE_ALPHA = 5  # スムージングの強さ


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """前処理 + Target Encoding (GroupKFold互換カテゴリのみ)

    fit(): train_foldのみから統計量とTarget Encoding用の生存率を学習
    transform(): 学習した統計量で任意のデータを変換
    """
    def fit(self, X, y=None):
        df = X.copy()
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')[0].map(TITLE_MAP).fillna('Other')
        df['Deck'] = df['Cabin'].str[0].fillna('U')

        # 統計量をtrain_foldから学習
        self.age_medians_ = df.groupby(['Pclass', 'Sex', 'Title'])['Age'].median()
        self.age_title_medians_ = df.groupby('Title')['Age'].median()
        self.age_overall_median_ = df['Age'].median()
        self.embarked_mode_ = df['Embarked'].mode()[0]
        self.fare_medians_ = df.groupby('Pclass')['Fare'].median()
        self.ticket_counts_ = df['Ticket'].value_counts()

        # LabelEncoder: train_foldのカテゴリで学習
        self.label_encoders_ = {}
        for col in ['Sex', 'Embarked', 'Title', 'Deck']:
            le = LabelEncoder()
            le.fit(df[col].dropna())
            self.label_encoders_[col] = le

        # Target Encoding (yが渡された場合のみ)
        if y is not None:
            y_arr = np.array(y)
            self.global_mean_ = y_arr.mean()
            df['_y'] = y_arr
            df['Embarked'] = df['Embarked'].fillna(self.embarked_mode_)

            self.te_maps_ = {}

            # Embarked × Pclass
            df['Embarked_Pclass'] = df['Embarked'] + '_' + df['Pclass'].astype(str)
            self.te_maps_['Embarked_Pclass'] = self._compute_te(df, 'Embarked_Pclass')

            # Deck
            self.te_maps_['Deck'] = self._compute_te(df, 'Deck')

            # Title × Pclass
            df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
            self.te_maps_['Title_Pclass'] = self._compute_te(df, 'Title_Pclass')

            # FamilySize
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            self.te_maps_['FamilySize'] = self._compute_te(df, 'FamilySize')

        return self

    def _compute_te(self, df, col):
        agg = df.groupby(col)['_y'].agg(['mean', 'count'])
        smoothed = (agg['count'] * agg['mean'] + TE_ALPHA * self.global_mean_) / (agg['count'] + TE_ALPHA)
        return smoothed.to_dict()

    def transform(self, X, y=None):
        df = X.copy()

        # Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')[0].map(TITLE_MAP).fillna('Other')

        # Age補完 (fitで学習した中央値を使用)
        for (p, s, t), med in self.age_medians_.items():
            mask = (df['Pclass'] == p) & (df['Sex'] == s) & (df['Title'] == t) & (df['Age'].isna())
            df.loc[mask, 'Age'] = med
        for t, med in self.age_title_medians_.items():
            mask = (df['Title'] == t) & (df['Age'].isna())
            df.loc[mask, 'Age'] = med
        df['Age'] = df['Age'].fillna(self.age_overall_median_)

        # Embarked補完
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode_)

        # Fare補完
        for p, med in self.fare_medians_.items():
            mask = (df['Pclass'] == p) & (df['Fare'].isna())
            df.loc[mask, 'Fare'] = med

        # 特徴量作成
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['TicketFreq'] = df['Ticket'].map(self.ticket_counts_).fillna(0).astype(int)
        df['Deck'] = df['Cabin'].str[0].fillna('U')
        df['AgePclass'] = df['Age'] * df['Pclass']

        # LabelEncode (fitで学習したエンコーダを使用、未知カテゴリは先頭カテゴリに)
        for col in ['Sex', 'Embarked', 'Title', 'Deck']:
            le = self.label_encoders_[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x, k=known, c=le.classes_[0]: x if x in k else c)
            df[col] = le.transform(df[col])

        # Target Encoding の適用
        if hasattr(self, 'te_maps_'):
            gm = self.global_mean_

            emb_raw = X['Embarked'].fillna(self.embarked_mode_)
            df['Embarked_Pclass_te'] = (emb_raw + '_' + X['Pclass'].astype(str)).map(
                self.te_maps_['Embarked_Pclass']).fillna(gm)

            deck_raw = X['Cabin'].str[0].fillna('U')
            df['Deck_te'] = deck_raw.map(self.te_maps_['Deck']).fillna(gm)

            title_raw = X['Name'].str.extract(r' ([A-Za-z]+)\.')[0].map(TITLE_MAP).fillna('Other')
            df['Title_Pclass_te'] = (title_raw + '_' + X['Pclass'].astype(str)).map(
                self.te_maps_['Title_Pclass']).fillna(gm)

            fs = X['SibSp'] + X['Parch'] + 1
            df['FamilySize_te'] = fs.map(self.te_maps_['FamilySize']).fillna(gm)

        # Deadly_Large_Family: Pclass=3 & FamilySize>=5
        df['Deadly_Large_Family'] = ((df['FamilySize'] >= 5) & (X['Pclass'] == 3)).astype(int)

        return df[FEATURE_COLS]


# ======================================================
# データ準備
# ======================================================
X_raw = train_raw.drop(columns=['Survived'])
y = train_raw['Survived'].astype(int)
X_test_raw = test_raw.copy()

# 前処理後の特徴量を確認
prep = TitanicPreprocessor()
prep.fit(X_raw, y)
X_check = prep.transform(X_raw)
print(f"特徴量 ({len(FEATURE_COLS)}個): {FEATURE_COLS}")
print(f"X shape: {X_check.shape}, y shape: {y.shape}")

# ======================================================
# GroupKFold用のグループID (Union-Find: 姓+チケット統合)
# ======================================================
n_train = len(train_raw)
surnames = train_raw['Name'].str.split(',').str[0].values
tickets = train_raw['Ticket'].values

parent = list(range(n_train))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

surname_first = {}
for i, s in enumerate(surnames):
    if s in surname_first:
        union(i, surname_first[s])
    else:
        surname_first[s] = i

ticket_first = {}
for i, t in enumerate(tickets):
    if t in ticket_first:
        union(i, ticket_first[t])
    else:
        ticket_first[t] = i

groups = np.array([find(i) for i in range(n_train)])
n_groups = len(set(groups))

gkf = GroupKFold(n_splits=5)
print(f"GroupKFold: {n_groups}グループを5分割 (同じ姓 or 同じチケット → 同じfold)")

# ======================================================
# 各モデルのパイプラインとグリッドサーチ
#   TitanicPreprocessor がパイプライン内にあるため
#   CV各foldでtrain_foldのみから統計量を計算する (リークなし)
# ======================================================

# --- LightGBM ---
print("\n=== LightGBM グリッドサーチ ===")
lgb_pipe = Pipeline([
    ('prep', TitanicPreprocessor()),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
])
lgb_param_grid = {
    'lgb__max_depth': [3, 5, 7],
    'lgb__num_leaves': [8, 16, 31],
    'lgb__min_child_samples': [10, 20, 30, 50],
    'lgb__reg_alpha': [0, 0.1, 1.0],
    'lgb__reg_lambda': [0, 0.1, 1.0],
}
print(f"探索中... ({np.prod([len(v) for v in lgb_param_grid.values()])}通り)")

lgb_grid = GridSearchCV(lgb_pipe, lgb_param_grid, cv=gkf, scoring='accuracy', n_jobs=1)
lgb_grid.fit(X_raw, y, groups=groups)
print(f"ベストパラメータ: {lgb_grid.best_params_}")
print(f"CV Accuracy    : {lgb_grid.best_score_:.4f}")

# --- XGBoost ---
print("\n=== XGBoost グリッドサーチ ===")
xgb_pipe = Pipeline([
    ('prep', TitanicPreprocessor()),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
])
xgb_param_grid = {
    'xgb__max_depth': [3, 5, 7],
    'xgb__min_child_weight': [1, 3, 5, 10],
    'xgb__reg_alpha': [0, 0.1, 1.0],
    'xgb__reg_lambda': [0, 0.1, 1.0],
}
print(f"探索中... ({np.prod([len(v) for v in xgb_param_grid.values()])}通り)")

xgb_grid = GridSearchCV(xgb_pipe, xgb_param_grid, cv=gkf, scoring='accuracy', n_jobs=1)
xgb_grid.fit(X_raw, y, groups=groups)
print(f"ベストパラメータ: {xgb_grid.best_params_}")
print(f"CV Accuracy    : {xgb_grid.best_score_:.4f}")

# --- LogisticRegression ---
print("\n=== LogisticRegression グリッドサーチ ===")
lr_pipe = Pipeline([
    ('prep', TitanicPreprocessor()),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
])
lr_param_grid = {'lr__C': [0.01, 0.1, 1.0, 10.0]}
print(f"探索中... ({np.prod([len(v) for v in lr_param_grid.values()])}通り)")

lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=gkf, scoring='accuracy', n_jobs=1)
lr_grid.fit(X_raw, y, groups=groups)
print(f"ベストパラメータ: {lr_grid.best_params_}")
print(f"CV Accuracy    : {lr_grid.best_score_:.4f}")

# --- SVM ---
print("\n=== SVM グリッドサーチ (詳細) ===")
svm_pipe = Pipeline([
    ('prep', TitanicPreprocessor()),
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42)),
])
svm_param_grid = {
    'svm__C': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
    'svm__gamma': ['scale', 'auto', 0.01, 0.05, 0.1],
    'svm__kernel': ['rbf', 'poly', 'linear'],
}
print(f"探索中... ({np.prod([len(v) for v in svm_param_grid.values()])}通り)")

svm_grid = GridSearchCV(svm_pipe, svm_param_grid, cv=gkf, scoring='accuracy', n_jobs=1)
svm_grid.fit(X_raw, y, groups=groups)
print(f"ベストパラメータ: {svm_grid.best_params_}")
print(f"CV Accuracy    : {svm_grid.best_score_:.4f}")

# --- KNN ---
print("\n=== KNN グリッドサーチ ===")
knn_pipe = Pipeline([
    ('prep', TitanicPreprocessor()),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier()),
])
knn_param_grid = {'knn__n_neighbors': [3, 5, 7, 9, 11]}
print(f"探索中... ({np.prod([len(v) for v in knn_param_grid.values()])}通り)")

knn_grid = GridSearchCV(knn_pipe, knn_param_grid, cv=gkf, scoring='accuracy', n_jobs=1)
knn_grid.fit(X_raw, y, groups=groups)
print(f"ベストパラメータ: {knn_grid.best_params_}")
print(f"CV Accuracy    : {knn_grid.best_score_:.4f}")

# ------------------------------------------
# アンサンブル (soft voting)
#   各estimatorは完全なパイプライン → CV内でも前処理が正しく行われる
# ------------------------------------------
print("\n=== アンサンブル (5モデル Voting) ===")
ensemble = VotingClassifier(
    estimators=[
        ('lgb', lgb_grid.best_estimator_),
        ('xgb', xgb_grid.best_estimator_),
        ('lr', lr_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_),
        ('knn', knn_grid.best_estimator_),
    ],
    voting='soft',
)
ensemble_scores = cross_val_score(ensemble, X_raw, y, cv=gkf, scoring='accuracy', groups=groups)
print(f"CV Accuracy    : {ensemble_scores.mean():.4f} (±{ensemble_scores.std():.4f})")

# ------------------------------------------
# 結果比較
# ------------------------------------------
print("\n=== 比較 ===")
print(f"{'モデル':25s}  {'CV Accuracy':>12s}")
print("-" * 41)
results = {
    'LightGBM': lgb_grid.best_score_,
    'XGBoost': xgb_grid.best_score_,
    'LogisticRegression': lr_grid.best_score_,
    'SVM': svm_grid.best_score_,
    'KNN': knn_grid.best_score_,
    'Ensemble (5モデル)': ensemble_scores.mean(),
}
for name, score in results.items():
    marker = ' <<<' if score == max(results.values()) else ''
    print(f"{name:25s}: {score:.4f}{marker}")

# ------------------------------------------
# ベストモデルで予測
# ------------------------------------------
best_name = max(results, key=results.get)
print(f"\n>>> ベスト: {best_name}")

model_map = {
    'LightGBM': lgb_grid.best_estimator_,
    'XGBoost': xgb_grid.best_estimator_,
    'LogisticRegression': lr_grid.best_estimator_,
    'SVM': svm_grid.best_estimator_,
    'KNN': knn_grid.best_estimator_,
    'Ensemble (5モデル)': ensemble,
}
final_model = model_map[best_name]

final_model.fit(X_raw, y)
train_acc = (final_model.predict(X_raw) == y).mean()
best_cv = results[best_name]
print(f"Train Accuracy: {train_acc:.4f}")
print(f"CV Accuracy   : {best_cv:.4f}")
print(f"Gap           : {train_acc - best_cv:.4f}")

predictions = final_model.predict(X_test_raw).astype(int)
submission = pd.DataFrame({
    'PassengerId': test_raw['PassengerId'].astype(int),
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
print(f"\n=== 提出ファイル作成完了 ===")
print(f"submission.csv ({len(submission)}行)")
print(submission.head())
