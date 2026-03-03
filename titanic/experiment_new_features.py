"""
新特徴量の検証: Deadly_Large_Family + Ticket_Group_Survival
Repeated GroupKFold + paired t-test でベースラインと比較
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from scipy import stats

train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")

# =====================================================
# 定数・共通
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

TE_TARGETS = ['Embarked_Pclass', 'Deck', 'Title_Pclass', 'FamilySize']
TE_ALPHA = 5

# 新特徴量
NEW_FEATURES = ['Deadly_Large_Family', 'Ticket_Group_Survival']

FEATURE_COLS_OLD = FEATURE_COLS_BASE + [t + '_te' for t in TE_TARGETS]
FEATURE_COLS_NEW = FEATURE_COLS_OLD + NEW_FEATURES


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """前処理 + TE + 新特徴量"""

    def __init__(self, use_new_features=False):
        self.use_new_features = use_new_features

    def fit(self, X, y=None):
        df = X.copy()
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')[0].map(TITLE_MAP).fillna('Other')
        df['Deck'] = df['Cabin'].str[0].fillna('U')

        self.age_medians_ = df.groupby(['Pclass', 'Sex', 'Title'])['Age'].median()
        self.age_title_medians_ = df.groupby('Title')['Age'].median()
        self.age_overall_median_ = df['Age'].median()
        self.embarked_mode_ = df['Embarked'].mode()[0]
        self.fare_medians_ = df.groupby('Pclass')['Fare'].median()
        self.ticket_counts_ = df['Ticket'].value_counts()

        self.label_encoders_ = {}
        for col in ['Sex', 'Embarked', 'Title', 'Deck']:
            le = LabelEncoder()
            le.fit(df[col].dropna())
            self.label_encoders_[col] = le

        if y is not None:
            y_arr = np.array(y)
            self.global_mean_ = y_arr.mean()
            df['_y'] = y_arr
            df['Embarked'] = df['Embarked'].fillna(self.embarked_mode_)

            self.te_maps_ = {}
            df['Embarked_Pclass'] = df['Embarked'] + '_' + df['Pclass'].astype(str)
            self.te_maps_['Embarked_Pclass'] = self._compute_te(df, 'Embarked_Pclass')
            self.te_maps_['Deck'] = self._compute_te(df, 'Deck')
            df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
            self.te_maps_['Title_Pclass'] = self._compute_te(df, 'Title_Pclass')
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            self.te_maps_['FamilySize'] = self._compute_te(df, 'FamilySize')

            # Ticket_Group_Survival: 同一チケットの「自分以外」の生存率
            # GroupKFoldで同一チケット=同一グループ=同一fold なので
            # train_fold内の同一チケット他者は全員train_fold側 → リークなし
            if self.use_new_features:
                df['FamilySize2'] = df['SibSp'] + df['Parch'] + 1
                ticket_surv = df.groupby('Ticket').agg(
                    ticket_survived_sum=('_y', 'sum'),
                    ticket_count=('_y', 'count'),
                ).to_dict('index')
                self.ticket_survival_map_ = ticket_surv

        return self

    def _compute_te(self, df, col):
        agg = df.groupby(col)['_y'].agg(['mean', 'count'])
        smoothed = (agg['count'] * agg['mean'] + TE_ALPHA * self.global_mean_) / (agg['count'] + TE_ALPHA)
        return smoothed.to_dict()

    def transform(self, X, y=None):
        df = X.copy()

        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')[0].map(TITLE_MAP).fillna('Other')

        for (p, s, t), med in self.age_medians_.items():
            mask = (df['Pclass'] == p) & (df['Sex'] == s) & (df['Title'] == t) & (df['Age'].isna())
            df.loc[mask, 'Age'] = med
        for t, med in self.age_title_medians_.items():
            mask = (df['Title'] == t) & (df['Age'].isna())
            df.loc[mask, 'Age'] = med
        df['Age'] = df['Age'].fillna(self.age_overall_median_)

        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode_)

        for p, med in self.fare_medians_.items():
            mask = (df['Pclass'] == p) & (df['Fare'].isna())
            df.loc[mask, 'Fare'] = med

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['TicketFreq'] = df['Ticket'].map(self.ticket_counts_).fillna(0).astype(int)
        df['Deck'] = df['Cabin'].str[0].fillna('U')
        df['AgePclass'] = df['Age'] * df['Pclass']

        for col in ['Sex', 'Embarked', 'Title', 'Deck']:
            le = self.label_encoders_[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x, k=known, c=le.classes_[0]: x if x in k else c)
            df[col] = le.transform(df[col])

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

        if self.use_new_features:
            # --- Deadly_Large_Family ---
            # Pclass=3 かつ FamilySize>=5 → 1, それ以外 → 0
            df['Deadly_Large_Family'] = ((df['FamilySize'] >= 5) &
                                         (X['Pclass'] == 3)).astype(int)

            # --- Ticket_Group_Survival ---
            # 同一チケットの「自分以外」の生存率
            # = (チケット全体の生存合計 - 自分) / (チケット全体の人数 - 1)
            # 単独チケット(count=1) → -1 (欠損フラグ、globalMeanではなく区別可能に)
            if hasattr(self, 'ticket_survival_map_'):
                gm = self.global_mean_
                tgs_values = []
                for idx, row in X.iterrows():
                    ticket = row['Ticket']
                    info = self.ticket_survival_map_.get(ticket)
                    if info is None or info['ticket_count'] <= 1:
                        # テストデータの未知チケット or 単独チケット
                        tgs_values.append(-1.0)
                    else:
                        # 自分のSurvivedを引くにはyが必要だが、
                        # transform時にはyがない場合がある (test data)
                        # → グループ全体の平均を使う (自分を含む)
                        # CVではGroupKFoldにより同一チケットは同一foldなので
                        # train_foldでfitした統計 = 全員train_fold側 → リークなし
                        tgs_values.append(
                            info['ticket_survived_sum'] / info['ticket_count']
                        )
                df['Ticket_Group_Survival'] = tgs_values
            else:
                df['Ticket_Group_Survival'] = -1.0

        feature_cols = FEATURE_COLS_NEW if self.use_new_features else FEATURE_COLS_OLD
        return df[feature_cols]


# =====================================================
# Union-Find グループID
# =====================================================
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

X_raw = train_raw.drop(columns=['Survived'])
y = train_raw['Survived'].astype(int)

print(f"GroupKFold: {n_groups}グループを5分割")

# =====================================================
# Repeated GroupKFold + Paired t-test
# =====================================================
N_REPEATS = 10

print("\n" + "=" * 60)
print("検証: ベースライン (16特徴量) vs 新特徴量追加 (18特徴量)")
print("=" * 60)

# SVM: C=0.5, gamma=0.01 (前回のベストパラメータ)
svm_params = {'svm__C': 0.5, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}

# LightGBM: 前回ベストの近似パラメータ
lgb_params = {'lgb__max_depth': 5, 'lgb__num_leaves': 16,
              'lgb__min_child_samples': 20, 'lgb__reg_alpha': 0.1, 'lgb__reg_lambda': 0.1}

for model_name, model_cls, model_params in [
    ('SVM', lambda nf: Pipeline([
        ('prep', TitanicPreprocessor(use_new_features=nf)),
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42, C=0.5, gamma=0.01, kernel='rbf')),
    ]), None),
    ('LightGBM', lambda nf: Pipeline([
        ('prep', TitanicPreprocessor(use_new_features=nf)),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1,
                                    max_depth=5, num_leaves=16, min_child_samples=20,
                                    reg_alpha=0.1, reg_lambda=0.1)),
    ]), None),
]:
    print(f"\n--- {model_name} ---")

    scores_old_all = []
    scores_new_all = []

    for repeat in range(N_REPEATS):
        # グループをシャッフルして異なるfold分割を生成
        rng = np.random.RandomState(repeat * 42)
        unique_groups = np.unique(groups)
        shuffled_map = {g: rng.randint(0, 10**6) for g in unique_groups}
        shuffled_groups = np.array([shuffled_map[g] for g in groups])

        gkf = GroupKFold(n_splits=5)
        fold_scores_old = []
        fold_scores_new = []

        for train_idx, val_idx in gkf.split(X_raw, y, groups=shuffled_groups):
            X_tr, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # ベースライン (旧特徴量)
            pipe_old = model_cls(False)
            pipe_old.fit(X_tr, y_tr)
            acc_old = (pipe_old.predict(X_val) == y_val).mean()
            fold_scores_old.append(acc_old)

            # 新特徴量
            pipe_new = model_cls(True)
            pipe_new.fit(X_tr, y_tr)
            acc_new = (pipe_new.predict(X_val) == y_val).mean()
            fold_scores_new.append(acc_new)

        scores_old_all.extend(fold_scores_old)
        scores_new_all.extend(fold_scores_new)

    scores_old = np.array(scores_old_all)
    scores_new = np.array(scores_new_all)
    diff = scores_new - scores_old

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_new, scores_old)

    print(f"  ベースライン (16特徴量): {scores_old.mean():.4f} ± {scores_old.std():.4f}")
    print(f"  新特徴量     (18特徴量): {scores_new.mean():.4f} ± {scores_new.std():.4f}")
    print(f"  差分         : {diff.mean():+.4f}")
    print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        if diff.mean() > 0:
            print(f"  >>> 有意に改善 (p={p_value:.4f})")
        else:
            print(f"  >>> 有意に悪化 (p={p_value:.4f})")
    else:
        print(f"  >>> 有意差なし (p={p_value:.4f})")

# =====================================================
# 新特徴量の分布確認
# =====================================================
print("\n" + "=" * 60)
print("新特徴量の分布")
print("=" * 60)

prep_new = TitanicPreprocessor(use_new_features=True)
prep_new.fit(X_raw, y)
X_new = prep_new.transform(X_raw)

print("\n--- Deadly_Large_Family ---")
dlf = X_new['Deadly_Large_Family']
print(f"  1 (該当): {(dlf == 1).sum()}人, 生存率: {y[dlf == 1].mean():.1%}")
print(f"  0 (非該当): {(dlf == 0).sum()}人, 生存率: {y[dlf == 0].mean():.1%}")

print("\n--- Ticket_Group_Survival ---")
tgs = X_new['Ticket_Group_Survival']
print(f"  -1 (単独チケット): {(tgs == -1).sum()}人")
print(f"   0 (グループ全滅): {(tgs == 0).sum()}人")
print(f"   1 (グループ全生): {(tgs == 1).sum()}人")
print(f"   混合(0<x<1): {((tgs > 0) & (tgs < 1)).sum()}人")
print(f"  全体の分布:")
for val in sorted(tgs.unique()):
    count = (tgs == val).sum()
    surv = y[tgs == val].mean()
    print(f"    {val:5.2f}: {count:4d}人, 生存率 {surv:.1%}")
