"""
検証: Ticket_Group_Survivalの構造問題 + Deadly_Large_Familyのみの効果
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

# === 検証: val_foldでのTicket overlap ===
print("=== Ticket_Group_Survival のGroupKFold構造問題 ===")
print()

gkf = GroupKFold(n_splits=5)
for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups=groups)):
    train_tickets = set(train_raw.iloc[train_idx]['Ticket'].values)
    val_tickets = set(train_raw.iloc[val_idx]['Ticket'].values)
    overlap = train_tickets & val_tickets
    print(f"Fold {fold_i}: train tickets {len(train_tickets)}, "
          f"val tickets {len(val_tickets)}, overlap {len(overlap)}")

print()
print(">>> overlap = 0: val_foldのチケットはtrain_foldに一切存在しない")
print(">>> GroupKFoldで同一チケット=同一グループ=同一foldのため")
print(">>> val_foldではTicket_Group_Survival = -1 (常に定数)")
print()

# === Deadly_Large_Familyのみの検証 ===
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
FEATURE_COLS_DLF = FEATURE_COLS_OLD + ['Deadly_Large_Family']


class Prep(BaseEstimator, TransformerMixin):
    def __init__(self, add_dlf=False):
        self.add_dlf = add_dlf

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
        if self.add_dlf:
            df['Deadly_Large_Family'] = ((df['FamilySize'] >= 5) & (X['Pclass'] == 3)).astype(int)
        cols = FEATURE_COLS_DLF if self.add_dlf else FEATURE_COLS_OLD
        return df[cols]


print("=" * 60)
print("検証: Deadly_Large_Family のみ追加 (17特徴量) vs ベースライン (16特徴量)")
print("=" * 60)

N_REPEATS = 10

for model_name, make_pipe in [
    ('SVM', lambda dlf: Pipeline([
        ('prep', Prep(add_dlf=dlf)),
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42, C=0.5, gamma=0.01, kernel='rbf')),
    ])),
    ('LightGBM', lambda dlf: Pipeline([
        ('prep', Prep(add_dlf=dlf)),
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
    print(f"  +DLFのみ    (17特徴量): {new_arr.mean():.4f} ± {new_arr.std():.4f}")
    print(f"  差分: {diff.mean():+.4f}, t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        direction = "改善" if diff.mean() > 0 else "悪化"
        print(f"  >>> 有意に{direction}")
    else:
        print(f"  >>> 有意差なし")
