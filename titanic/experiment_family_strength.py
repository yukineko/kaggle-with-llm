"""
検証: 家族の社会的強さ特徴量
  - Deadly_Large_Family (前回有効と確認済み)
  - Female_Upper_Count: 同一チケットグループの (Sex==female & Pclass<3) の人数 (LOO: 自分を除く)
  - Master_Count: 同一チケットグループの (Title==Master) の人数 (LOO: 自分を除く)

リーク安全性:
  - 集計対象は Sex, Pclass, Title = 全て説明変数(X)のみ → 目的変数(y)を使わない
  - train+test全体から計算しても理論上リークしない
  - ただし正確を期して、fit時はtrain_foldのみから集計し、
    transform時にtrain_foldで見たチケット情報を適用する

  注意: GroupKFoldで同一チケット=同一fold なので、
  val_foldのチケットはtrain_foldに存在しない
  → val_foldでは「自分のチケットのfit時統計」が存在しない
  → 対策: fit時ではなく、transform時にX自体から直接計算する
           (Xの説明変数のみなのでリークしない)
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
X_raw = train_raw.drop(columns=['Survived'])
y = train_raw['Survived'].astype(int)

# Union-Find
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
FEATURE_COLS_OLD = FEATURE_COLS_BASE + [t + '_te' for t in TE_TARGETS]

# 新特徴量の定義
FAMILY_STRENGTH_FEATURES = ['Deadly_Large_Family', 'Female_Upper_Count', 'Master_Count']
FEATURE_COLS_NEW = FEATURE_COLS_OLD + FAMILY_STRENGTH_FEATURES


class Prep(BaseEstimator, TransformerMixin):
    def __init__(self, add_family_strength=False):
        self.add_family_strength = add_family_strength

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
            self.te_maps_['Embarked_Pclass'] = self._te(df, 'Embarked_Pclass')
            self.te_maps_['Deck'] = self._te(df, 'Deck')
            df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
            self.te_maps_['Title_Pclass'] = self._te(df, 'Title_Pclass')
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            self.te_maps_['FamilySize'] = self._te(df, 'FamilySize')
        return self

    def _te(self, df, col):
        agg = df.groupby(col)['_y'].agg(['mean', 'count'])
        return ((agg['count'] * agg['mean'] + TE_ALPHA * self.global_mean_) / (agg['count'] + TE_ALPHA)).to_dict()

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
            df['Deck_te'] = X['Cabin'].str[0].fillna('U').map(self.te_maps_['Deck']).fillna(gm)
            title_raw = X['Name'].str.extract(r' ([A-Za-z]+)\.')[0].map(TITLE_MAP).fillna('Other')
            df['Title_Pclass_te'] = (title_raw + '_' + X['Pclass'].astype(str)).map(
                self.te_maps_['Title_Pclass']).fillna(gm)
            df['FamilySize_te'] = (X['SibSp'] + X['Parch'] + 1).map(
                self.te_maps_['FamilySize']).fillna(gm)

        if self.add_family_strength:
            # --- Deadly_Large_Family ---
            df['Deadly_Large_Family'] = ((df['FamilySize'] >= 5) & (X['Pclass'] == 3)).astype(int)

            # --- Female_Upper_Count & Master_Count (LOO) ---
            # Xの説明変数のみから計算 → yを使わない → リークなし
            # transform時にX自体から直接集計する
            # (fit時のtrain_foldに限定しなくても安全)
            title_raw = X['Name'].str.extract(r' ([A-Za-z]+)\.')[0].map(TITLE_MAP).fillna('Other')
            is_female_upper = ((X['Sex'] == 'female') & (X['Pclass'] < 3)).astype(int)
            is_master = (title_raw == 'Master').astype(int)

            # チケットグループ内の合計 (自分を含む)
            ticket_female_upper = X['Ticket'].map(
                pd.DataFrame({'Ticket': X['Ticket'], 'val': is_female_upper})
                .groupby('Ticket')['val'].sum()
            )
            ticket_master = X['Ticket'].map(
                pd.DataFrame({'Ticket': X['Ticket'], 'val': is_master})
                .groupby('Ticket')['val'].sum()
            )

            # LOO: 自分を引く
            df['Female_Upper_Count'] = (ticket_female_upper - is_female_upper).fillna(0).astype(int)
            df['Master_Count'] = (ticket_master - is_master).fillna(0).astype(int)

        cols = FEATURE_COLS_NEW if self.add_family_strength else FEATURE_COLS_OLD
        return df[cols]


# =====================================================
# まず特徴量の分布を確認
# =====================================================
print("=" * 60)
print("新特徴量の分布")
print("=" * 60)

prep = Prep(add_family_strength=True)
prep.fit(X_raw, y)
X_new = prep.transform(X_raw)

for feat in FAMILY_STRENGTH_FEATURES:
    print(f"\n--- {feat} ---")
    for val in sorted(X_new[feat].unique()):
        mask = X_new[feat] == val
        n = mask.sum()
        surv = y[mask].mean()
        print(f"  {val:3.0f}: {n:4d}人, 生存率 {surv:.1%}")

# =====================================================
# Repeated GroupKFold + Paired t-test
# =====================================================
print("\n" + "=" * 60)
print("検証: ベースライン (16特徴量) vs +家族強さ (19特徴量)")
print("  追加: Deadly_Large_Family, Female_Upper_Count, Master_Count")
print("=" * 60)

N_REPEATS = 10

for model_name, make_pipe in [
    ('SVM (C=0.5, gamma=0.01)', lambda fs: Pipeline([
        ('prep', Prep(add_family_strength=fs)),
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42, C=0.5, gamma=0.01, kernel='rbf')),
    ])),
    ('LightGBM', lambda fs: Pipeline([
        ('prep', Prep(add_family_strength=fs)),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1,
            max_depth=5, num_leaves=16, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1)),
    ])),
]:
    print(f"\n--- {model_name} ---")
    old_all, new_all = [], []
    for rep in range(N_REPEATS):
        rng = np.random.RandomState(rep * 42)
        ugroups = np.unique(groups)
        smap = {g: rng.randint(0, 10**6) for g in ugroups}
        sgroups = np.array([smap[g] for g in groups])
        gkf = GroupKFold(n_splits=5)
        for tr_idx, va_idx in gkf.split(X_raw, y, groups=sgroups):
            Xtr, Xva = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
            p0 = make_pipe(False)
            p0.fit(Xtr, ytr)
            old_all.append((p0.predict(Xva) == yva).mean())
            p1 = make_pipe(True)
            p1.fit(Xtr, ytr)
            new_all.append((p1.predict(Xva) == yva).mean())

    old_arr, new_arr = np.array(old_all), np.array(new_all)
    diff = new_arr - old_arr
    t_stat, p_val = stats.ttest_rel(new_arr, old_arr)
    print(f"  ベースライン(16特徴量): {old_arr.mean():.4f} ± {old_arr.std():.4f}")
    print(f"  +家族強さ   (19特徴量): {new_arr.mean():.4f} ± {new_arr.std():.4f}")
    print(f"  差分: {diff.mean():+.4f}, t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        direction = "改善" if diff.mean() > 0 else "悪化"
        print(f"  >>> 有意に{direction}")
    else:
        print(f"  >>> 有意差なし")

# =====================================================
# DLFのみ vs DLF+家族強さ の差分も確認
# =====================================================
print("\n" + "=" * 60)
print("追加検証: DLFのみ(17) vs DLF+Female_Upper+Master(19)")
print("  Female_Upper_Count, Master_Count の純粋な効果")
print("=" * 60)

FEATURE_COLS_DLF = FEATURE_COLS_OLD + ['Deadly_Large_Family']

class PrepDLF(Prep):
    """DLFのみ追加版"""
    def transform(self, X, y=None):
        df = super().transform(X, y)
        if self.add_family_strength:
            # 親クラスでFEATURE_COLS_NEWが返されるので、DLFのみに絞る
            return df[FEATURE_COLS_DLF]
        return df

# DLFのみ vs DLF+FUC+MC
for model_name, make_pipe_dlf, make_pipe_full in [
    ('SVM',
     lambda: Pipeline([
         ('prep', Prep(add_family_strength=False)),  # ← ここをハックする
         ('scaler', StandardScaler()),
         ('svm', SVC(probability=True, random_state=42, C=0.5, gamma=0.01, kernel='rbf')),
     ]),
     lambda: Pipeline([
         ('prep', Prep(add_family_strength=True)),
         ('scaler', StandardScaler()),
         ('svm', SVC(probability=True, random_state=42, C=0.5, gamma=0.01, kernel='rbf')),
     ]),
    ),
]:
    # DLFのみ用のPrep
    class PrepDLFOnly(BaseEstimator, TransformerMixin):
        def __init__(self):
            self._inner = Prep(add_family_strength=True)
        def fit(self, X, y=None):
            self._inner.fit(X, y)
            return self
        def transform(self, X, y=None):
            df = self._inner.transform(X, y)
            return df[FEATURE_COLS_DLF]

    print(f"\n--- {model_name} ---")
    dlf_all, full_all = [], []
    for rep in range(N_REPEATS):
        rng = np.random.RandomState(rep * 42)
        ugroups = np.unique(groups)
        smap = {g: rng.randint(0, 10**6) for g in ugroups}
        sgroups = np.array([smap[g] for g in groups])
        gkf = GroupKFold(n_splits=5)
        for tr_idx, va_idx in gkf.split(X_raw, y, groups=sgroups):
            Xtr, Xva = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

            # DLFのみ
            p_dlf = Pipeline([
                ('prep', PrepDLFOnly()),
                ('scaler', StandardScaler()),
                ('svm', SVC(probability=True, random_state=42, C=0.5, gamma=0.01, kernel='rbf')),
            ])
            p_dlf.fit(Xtr, ytr)
            dlf_all.append((p_dlf.predict(Xva) == yva).mean())

            # DLF + FUC + MC
            p_full = Pipeline([
                ('prep', Prep(add_family_strength=True)),
                ('scaler', StandardScaler()),
                ('svm', SVC(probability=True, random_state=42, C=0.5, gamma=0.01, kernel='rbf')),
            ])
            p_full.fit(Xtr, ytr)
            full_all.append((p_full.predict(Xva) == yva).mean())

    dlf_arr, full_arr = np.array(dlf_all), np.array(full_all)
    diff = full_arr - dlf_arr
    t_stat, p_val = stats.ttest_rel(full_arr, dlf_arr)
    print(f"  DLFのみ     (17特徴量): {dlf_arr.mean():.4f} ± {dlf_arr.std():.4f}")
    print(f"  +FUC+MC     (19特徴量): {full_arr.mean():.4f} ± {full_arr.std():.4f}")
    print(f"  差分: {diff.mean():+.4f}, t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        direction = "改善" if diff.mean() > 0 else "悪化"
        print(f"  >>> Female_Upper_Count + Master_Count は有意に{direction}")
    else:
        print(f"  >>> Female_Upper_Count + Master_Count の追加効果は有意差なし")
