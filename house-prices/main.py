"""
House Prices: Advanced Regression Techniques
Main pipeline: preprocessing → CV → model training → submission
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


# ============================================================
# 前処理
# ============================================================

# Neighborhood価格帯ビン (LOO TEの値をビンに分け、線形モデルにカテゴリ序列を与える)
NBHD_BIN_N = 5  # 5分位

# 線形専用の合成特徴量 (tree系には除外する)
# 合成特徴量に集約済みのため削除する元変数
REDUNDANT_COLS = [
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',           # → TotalSF
    'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',  # → TotalBath
    'YearBuilt',                                       # → HouseAge
]

LINEAR_ONLY_FEATURES = [
    'Value_Standard_Score', 'VSS_x_GrLivArea',
    'Pure_Comfort_Score', 'Livable_Area',
    'Thermal_Efficiency',
    'IDOTRR_Distress', 'IDOTRR_QualCap',
    'Elite_Area_Premium',
    'Is_Distress_Sale', 'Distress_x_Qual',
    'Snow_Maint_Burden', 'Burden_x_Area',
]


class HousePricesPreprocessor(BaseEstimator, TransformerMixin):
    """fold内前処理: 統計量はtrain_foldのみから計算"""

    # NA = "その設備がない" のカテゴリカル特徴量
    NA_MEANS_NONE = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
        'MasVnrType',
    ]

    # 順序カテゴリのマッピング
    ORDINAL_MAP = {
        'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
        'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
        'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
        'CentralAir': {'N': 0, 'Y': 1},
        'Street': {'Grvl': 0, 'Pave': 1},
        'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
        'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
    }

    # ドロップするカラム (ほぼ定数 or 情報量なし)
    DROP_COLS = ['Id', 'Utilities', 'Street']

    # 歪度補正対象 (log1p変換する連続量特徴量)
    SKEW_FEATURES = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
    ]

    # Target Encoding対象カラム
    TE_COLS = ['Neighborhood']
    TE_SMOOTH = 10  # smoothing parameter (小カテゴリの過学習防止)

    def __init__(self, feature_set=None):
        self.feature_set = feature_set  # None=全変数, 'A'=精鋭24, 'B'=木モデル30
        self.lot_frontage_medians_ = None
        self.num_medians_ = None
        self.cat_modes_ = None
        self.label_encoders_ = {}
        self.feature_names_ = None
        self.te_stats_ = {}
        self.te_global_mean_ = None
        self._train_y_ = None
        self.first_floor_median_ = None
        self.comfort_mean_ = None
        self.comfort_std_ = None
        self.nbhd_bin_edges_ = None

    def fit(self, X, y=None):
        df = X.copy()

        # LotFrontage: Neighborhoodごとの中央値で補完
        self.lot_frontage_medians_ = df.groupby('Neighborhood')['LotFrontage'].median()
        self.lot_frontage_global_ = df['LotFrontage'].median()

        # 数値の欠損は中央値で補完
        num_cols = df.select_dtypes(include=[np.number]).columns
        self.num_medians_ = df[num_cols].median()

        # カテゴリカルの欠損は最頻値で補完
        cat_cols = df.select_dtypes(include=['object']).columns
        self.cat_modes_ = df[cat_cols].mode().iloc[0]

        # Label Encoding用のマッピング
        temp = self._fill_missing(df)
        temp = self._ordinal_encode(temp)
        remaining_cats = temp.select_dtypes(include=['object']).columns
        for col in remaining_cats:
            if col in self.DROP_COLS:
                continue
            vals = sorted(temp[col].unique())
            self.label_encoders_[col] = {v: i for i, v in enumerate(vals)}

        # Target Encoding統計量 (カテゴリ別のy合計・件数)
        if y is not None:
            self._fit_te(temp, y)

        # Is_Senior_Friendly_Large用の閾値 (train foldから学習)
        if '1stFlrSF' in df.columns:
            self.first_floor_median_ = df['1stFlrSF'].median()

        # Pure_Comfort_Score用: Z-score標準化パラメータ (train foldから学習)
        # temp は既に _fill_missing + _ordinal_encode 済み
        raw_comfort = (
            (temp['HeatingQC'] == 5).astype(int) +
            (temp['KitchenQual'] >= 4).astype(int) +
            (temp['GarageFinish'] >= 2).astype(int) +
            (temp['FireplaceQu'] >= 4).astype(int)
        )
        self.comfort_mean_ = float(raw_comfort.mean())
        self.comfort_std_ = float(raw_comfort.std())
        if self.comfort_std_ < 1e-8:
            self.comfort_std_ = 1.0

        # Neighborhood Binning: TE値を5分位ビンに分割 (線形モデル補助)
        if self.te_stats_ and 'Neighborhood' in self.te_stats_:
            te_vals = []
            for cat, stats in self.te_stats_['Neighborhood'].items():
                te_vals.append(stats['sum'] / max(stats['count'], 1))
            te_arr = np.array(te_vals)
            self.nbhd_bin_edges_ = np.percentile(
                te_arr, np.linspace(0, 100, NBHD_BIN_N + 1)[1:-1])

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """LOO Target Encoding のためにfit+transformをオーバーライド"""
        self.fit(X, y)
        self._train_y_ = y  # transform内でLOO計算に使用
        result = self.transform(X)
        self._train_y_ = None
        return result

    def transform(self, X):
        df = X.copy()
        df = self._fill_missing(df)
        df = self._ordinal_encode(df)
        df = self._apply_te(df)
        df = self._nbhd_binning(df)
        df = self._qol_features(df)
        df = self._value_standard_features(df)
        df = self._pure_comfort_features(df)
        df = self._thermal_efficiency(df)
        df = self._location_bias_features(df)
        df = self._label_encode(df)
        df = self._feature_engineering(df)
        df = self._drop_redundant(df)
        df = self._drop_cols(df)
        df = self._fix_skewness(df)
        df = self._select_elite(df)

        self.feature_names_ = df.columns.tolist()
        return df.values.astype(np.float64)

    def _select_elite(self, df):
        """feature_setに応じて変数を調整"""
        if self.feature_set == 'TREE':
            # 木モデル: 線形専用合成変数を除外
            drop = [f for f in LINEAR_ONLY_FEATURES if f in df.columns]
            if drop:
                df = df.drop(columns=drop)
        # 'LINEAR' or None: 全変数をそのまま使用
        return df

    def _fill_missing(self, df):
        # NA = "なし" を "None" に置換
        for col in self.NA_MEANS_NONE:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        # Garage数値系: ガレージなしは0
        for col in ['GarageYrBlt', 'GarageCars', 'GarageArea']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Basement数値系: 地下室なしは0
        for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                     'BsmtFullBath', 'BsmtHalfBath']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # LotFrontage: Neighborhoodごとの中央値
        if 'LotFrontage' in df.columns:
            for idx in df[df['LotFrontage'].isnull()].index:
                neighborhood = df.loc[idx, 'Neighborhood']
                if neighborhood in self.lot_frontage_medians_:
                    df.loc[idx, 'LotFrontage'] = self.lot_frontage_medians_[neighborhood]
                else:
                    df.loc[idx, 'LotFrontage'] = self.lot_frontage_global_

        # MasVnrArea
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

        # 残りの数値は中央値
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any() and col in self.num_medians_:
                df[col] = df[col].fillna(self.num_medians_[col])

        # 残りのカテゴリカルは最頻値
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().any() and col in self.cat_modes_:
                df[col] = df[col].fillna(self.cat_modes_[col])

        return df

    def _ordinal_encode(self, df):
        for col, mapping in self.ORDINAL_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                # マッピングに無い値は0で補完
                df[col] = df[col].fillna(0).astype(int)
        return df

    def _label_encode(self, df):
        for col, mapping in self.label_encoders_.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                df[col] = df[col].fillna(-1).astype(int)
        return df

    def _feature_engineering(self, df):
        # 住める広さ: 地上階 + 地下室仕上がり加重面積
        df['Livable_Area'] = (df['1stFlrSF'] + df['2ndFlrSF']
                              + df['TotalBsmtSF'] * df['BsmtFinType1'])
        # 総面積
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        # 総ポーチ面積
        df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                               df['3SsnPorch'] + df['ScreenPorch'])
        # 総バスルーム数
        df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                           df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
        # 築年数
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        # リモデルからの年数
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        # リモデル有無
        df['HasRemod'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        # ガレージ有無
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        # 地下室有無
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        # 2階有無
        df['Has2ndFlr'] = (df['2ndFlrSF'] > 0).astype(int)
        # プール有無
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        # ドメイン特徴量: 広さ軸
        # 平屋かつ広い (Senior Friendly Large)
        if self.first_floor_median_ is not None:
            df['Is_SFL'] = ((df['2ndFlrSF'] == 0) &
                            (df['1stFlrSF'] >= self.first_floor_median_)).astype(int)
        # 建蔽率的指標 (建物面積 / 土地面積)
        df['Living_Space_Ratio'] = df['GrLivArea'] / df['LotArea'].clip(lower=1)
        # 広さ×品質の相乗効果
        df['Luxury_Space_Index'] = df['TotalSF'] * df['OverallQual']

        return df

    def _drop_redundant(self, df):
        """合成特徴量の元変数を削除 (TotalSF, TotalBath, HouseAge に集約済み)"""
        drop = [c for c in REDUNDANT_COLS if c in df.columns]
        return df.drop(columns=drop)

    def _drop_cols(self, df):
        drop = [c for c in self.DROP_COLS if c in df.columns]
        df = df.drop(columns=drop)
        return df

    def _fix_skewness(self, df):
        for feat in self.SKEW_FEATURES:
            if feat in df.columns:
                df[feat] = np.log1p(np.maximum(df[feat], 0))
        for feat in ['TotalSF', 'TotalPorchSF', 'Luxury_Space_Index', 'Livable_Area']:
            if feat in df.columns:
                df[feat] = np.log1p(np.maximum(df[feat], 0))
        return df

    def _fit_te(self, df, y):
        """Target Encoding統計量を訓練データから計算"""
        self.te_global_mean_ = float(y.mean())
        self.te_stats_ = {}
        for col in self.TE_COLS:
            if col in df.columns:
                y_arr = y.values if hasattr(y, 'values') else np.array(y)
                tmp = pd.DataFrame({'cat': df[col].values, 'y': y_arr})
                agg = tmp.groupby('cat')['y'].agg(['sum', 'count'])
                self.te_stats_[col] = {
                    idx: {'sum': row['sum'], 'count': row['count']}
                    for idx, row in agg.iterrows()
                }

    def _target_encode_train(self, df, y):
        """LOO Target Encoding (訓練データ用: 自分自身のyを除外して平均計算)"""
        smooth = self.TE_SMOOTH
        global_mean = self.te_global_mean_
        for col in self.TE_COLS:
            if col not in df.columns or col not in self.te_stats_:
                continue
            y_arr = y.values if hasattr(y, 'values') else np.array(y)
            tmp = pd.DataFrame({'cat': df[col].values, 'y': y_arr})
            cat_sum = tmp.groupby('cat')['y'].transform('sum')
            cat_count = tmp.groupby('cat')['y'].transform('count')
            # LOO: 自分自身のyを除外
            loo_sum = cat_sum - tmp['y']
            loo_count = cat_count - 1
            # Smoothing: 小カテゴリはglobal meanに引き寄せる
            # .values で位置ベース代入 (fold分割後のdf.indexと不整合を防止)
            df[f'{col}_te'] = ((loo_sum + smooth * global_mean)
                               / (loo_count + smooth)).values
        return df

    def _target_encode_test(self, df):
        """通常のTarget Encoding (テスト/バリデーションデータ用)"""
        smooth = self.TE_SMOOTH
        global_mean = self.te_global_mean_
        for col in self.TE_COLS:
            if col not in df.columns or col not in self.te_stats_:
                continue
            encoded = np.full(len(df), global_mean)
            for cat, stats in self.te_stats_[col].items():
                mask = (df[col] == cat)
                n = stats['count']
                s = stats['sum']
                encoded[mask] = (s + smooth * global_mean) / (n + smooth)
            df[f'{col}_te'] = encoded
        return df

    def _apply_te(self, df):
        """Target Encodingを適用 (LOO for train, regular for test)"""
        if self.te_global_mean_ is None:
            return df
        if self._train_y_ is not None:
            return self._target_encode_train(df, self._train_y_)
        return self._target_encode_test(df)

    def _nbhd_binning(self, df):
        """Neighborhood TEを5分位ビンに変換 (線形モデル補助: 連続値→序列)"""
        if self.nbhd_bin_edges_ is not None and 'Neighborhood_te' in df.columns:
            df['Nbhd_Bin'] = np.digitize(df['Neighborhood_te'], self.nbhd_bin_edges_)
        return df

    def _qol_features(self, df):
        """QOL Score: 9つのバイナリアメニティ指標の合計 (0-9)"""
        df['QOL_Score'] = (
            (df['CentralAir'] == 1).astype(int) +
            (df['GarageArea'] > 0).astype(int) +
            (df['TotalBsmtSF'] > 0).astype(int) +
            (df['Fireplaces'] > 0).astype(int) +
            (df['Functional'] == 8).astype(int) +
            (df['KitchenQual'] >= 4).astype(int) +
            (df['ExterQual'] >= 4).astype(int) +
            (df['BsmtQual'] >= 4).astype(int) +
            (df['FullBath'] >= 2).astype(int)
        )
        return df

    def _family_stability_features(self, df):
        """Family Stability Index: 家族定着の土台 (0-3)
        FullBath>=2 (家族利用), BedroomAbvGr>=3 (子供部屋確保),
        KitchenQual>=TA (即入居可キッチン)
        交互作用: 全条件充足(FSI==3)フラグ × GrLivArea (二値×連続で独立性確保)
        """
        fsi = (
            (df['FullBath'] >= 2).astype(int) +
            (df['BedroomAbvGr'] >= 3).astype(int) +
            (df['KitchenQual'] >= 3).astype(int)
        )
        df['Family_Stability_Index'] = fsi
        df['Family_Premium'] = (fsi == 3).astype(int) * df['GrLivArea']
        return df

    def _value_standard_features(self, df):
        """Value Standard Score: 生活基盤の充実度 + GrLivArea交互作用"""
        NOISE_CONDITIONS = {'Artery', 'Feedr', 'RRNn', 'RRNe', 'RRAn', 'RRAe'}
        score = (
            (df['HeatingQC'] >= 4).astype(int) +
            (df['FullBath'] >= 2).astype(int) +
            (df['GarageCars'] >= 2).astype(int) +
            (df['CentralAir'] == 1).astype(int) -
            df['Condition1'].isin(NOISE_CONDITIONS).astype(int) -
            (df['Functional'] != 8).astype(int)
        )
        df['Value_Standard_Score'] = score
        df['VSS_x_GrLivArea'] = score * df['GrLivArea']
        return df

    def _pure_comfort_features(self, df):
        """Pure Comfort Score: 面積非依存の設備品質指標 (Z-score標準化)
        4成分: HeatingQC==Ex, KitchenQual>=Gd, GarageFinish∈{Fin,RFn}, FireplaceQu>=Gd
        raw: 0-4 → Z-score (train fold mean/stdで標準化)
        """
        raw = (
            (df['HeatingQC'] == 5).astype(int) +
            (df['KitchenQual'] >= 4).astype(int) +
            (df['GarageFinish'] >= 2).astype(int) +
            (df['FireplaceQu'] >= 4).astype(int)
        )
        mean = self.comfort_mean_ if self.comfort_mean_ is not None else 0.0
        std = self.comfort_std_ if self.comfort_std_ is not None else 1.0
        df['Pure_Comfort_Score'] = (raw - mean) / std
        return df

    def _thermal_efficiency(self, df):
        """Thermal Efficiency: 暖房・外装品質 / 面積 (線形モデル専用)
        (HeatingQC + ExterQual) / log(GrLivArea) — 広さあたりの断熱性能"""
        heat = df['HeatingQC'].clip(1, 5)
        exter = df['ExterQual'].clip(1, 5)
        area_log = np.log(df['GrLivArea'].clip(lower=1))
        df['Thermal_Efficiency'] = (heat + exter) / area_log.clip(lower=0.1)
        return df

    def _location_bias_features(self, df):
        """地域特性バイアス: IDOTRR減衰 + 高級エリアプレミアム + 非市場取引フラグ"""
        # --- 1. IDOTRR Distress (線路沿い減衰) ---
        is_idotrr = (df['Neighborhood'] == 'IDOTRR').astype(int) if df['Neighborhood'].dtype == object \
            else (df['Neighborhood_te'] < df['Neighborhood_te'].quantile(0.1)).astype(int)
        # フラグ + Qualとの交互作用 (高Qualでも天井がある)
        df['IDOTRR_Distress'] = is_idotrr
        df['IDOTRR_QualCap'] = is_idotrr * df['OverallQual']

        # --- 2. Elite Area Premium (高級エリア×面積) ---
        elite_areas = {'StoneBr', 'NridgHt', 'NoRidge', 'Crawfor'}
        is_elite = df['Neighborhood'].isin(elite_areas).astype(int) if df['Neighborhood'].dtype == object \
            else (df['Neighborhood_te'] > df['Neighborhood_te'].quantile(0.85)).astype(int)
        df['Elite_Area_Premium'] = is_elite * df['GrLivArea']

        # --- 3. Distress Sale (非市場取引フラグ) ---
        distress_conds = {'Abnorml', 'Family', 'Alloca'}
        is_distress = df['SaleCondition'].isin(distress_conds).astype(int) if df['SaleCondition'].dtype == object \
            else 0
        df['Is_Distress_Sale'] = is_distress
        df['Distress_x_Qual'] = is_distress * df['OverallQual']

        # --- 4. Snow Maintenance Burden (雪害リスク・シニア忌避) ---
        # 平屋根 or 2階建て古家 → 維持困難フラグ
        is_flat = (df['RoofStyle'] == 'Flat').astype(int) if df['RoofStyle'].dtype == object \
            else pd.Series(0, index=df.index)
        is_old_2story = pd.Series(0, index=df.index)
        if 'HouseStyle' in df.columns and df['HouseStyle'].dtype == object:
            is_old_2story = ((df['HouseStyle'] == '2Story') &
                             (df['YearBuilt'] < 1970)).astype(int)
        burden = (is_flat | is_old_2story).astype(int)
        df['Snow_Maint_Burden'] = burden
        # 広さのプラス効果を抑制する交互作用 (burden=1のとき面積が負に作用)
        df['Burden_x_Area'] = burden * df['GrLivArea']

        return df


class CatBoostPreprocessor(HousePricesPreprocessor):
    """CatBoost用前処理: カテゴリカル変数を文字列のまま保持し cat_features で渡す"""

    def __init__(self, feature_set=None):
        super().__init__(feature_set=feature_set)
        self.cat_feature_indices_ = None
        self.cat_feature_names_ = None

    def transform(self, X):
        df = X.copy()
        df = self._fill_missing(df)
        df = self._ordinal_encode(df)
        df = self._apply_te(df)
        df = self._nbhd_binning(df)
        df = self._qol_features(df)
        df = self._value_standard_features(df)
        df = self._pure_comfort_features(df)
        df = self._thermal_efficiency(df)
        df = self._location_bias_features(df)
        # Label Encoding をスキップ — カテゴリカルは文字列のまま保持
        df = self._feature_engineering(df)
        df = self._drop_redundant(df)
        df = self._drop_cols(df)
        df = self._fix_skewness(df)
        df = self._select_elite(df)

        # 残存する文字列列 = CatBoost用 cat_features
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.cat_feature_names_ = cat_cols
        self.cat_feature_indices_ = [df.columns.get_loc(c) for c in cat_cols]
        self.feature_names_ = df.columns.tolist()

        # カテゴリカルの残存NaN対策
        for col in cat_cols:
            df[col] = df[col].fillna('Missing')

        return df


class CatBoostPipeline:
    """CatBoost用パイプライン: cat_features を自動的に渡す"""

    def __init__(self, model, feature_set=None):
        self.preprocess = CatBoostPreprocessor(feature_set=feature_set)
        self.model = model

    def fit(self, X, y):
        X_t = self.preprocess.fit_transform(X, y)
        cat_idx = self.preprocess.cat_feature_indices_
        self.model.fit(X_t, y, cat_features=cat_idx)
        return self

    def predict(self, X):
        X_t = self.preprocess.transform(X)
        return self.model.predict(X_t)


# ============================================================
# データ読み込み
# ============================================================

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # 外れ値除去 (GrLivArea > 4000 & 低価格)
    outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
    train = train.drop(outliers).reset_index(drop=True)
    print(f'外れ値除去: {len(outliers)}件 → Train: {len(train)}行')

    y = np.log1p(train['SalePrice'])
    X = train.drop(columns=['SalePrice'])
    X_test = test.copy()
    test_ids = test['Id']

    return X, y, X_test, test_ids


# ============================================================
# CV
# ============================================================

def get_models(seed=42):
    """全モデル定義"""
    return {
        'Ridge': Ridge(alpha=5.0),
        'Lasso': Lasso(alpha=0.0003, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=10000),
        'GBR': GradientBoostingRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=4,
            min_samples_split=15, min_samples_leaf=10, max_features='sqrt',
            loss='huber', subsample=0.75, random_state=seed),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=seed, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.01, max_depth=4,
            num_leaves=15, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, verbose=-1),
        'CatBoost': CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=6,
            l2_leaf_reg=10, od_type='Iter', od_wait=100,
            random_seed=seed, verbose=0),
    }

# モデル → 特徴量セットの割り当て
# LINEAR: 全変数 + 線形専用合成変数 (VSS, Comfort, Livable)
# TREE: 全変数 - 線形専用合成変数
MODEL_FEATURE_SET = {
    'Ridge': 'LINEAR', 'Lasso': 'LINEAR', 'ElasticNet': 'LINEAR',
    'GBR': 'TREE', 'XGBoost': 'TREE', 'LightGBM': 'TREE',
    'CatBoost': 'TREE',
}


def make_pipeline(model, use_catboost=False, feature_set=None):
    """前処理 + モデルのパイプライン"""
    if use_catboost:
        return CatBoostPipeline(model, feature_set=feature_set)
    return Pipeline([
        ('preprocess', HousePricesPreprocessor(feature_set=feature_set)),
        ('model', model),
    ])


def evaluate_models(X, y, n_splits=5, seed=42):
    """KFold CVで複数モデルを評価 (RMSLE = RMSE on log1p target)"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = get_models(seed)

    results = {}
    for name, model in models.items():
        is_cb = name == 'CatBoost'
        fset = MODEL_FEATURE_SET.get(name)
        if is_cb:
            rmse_list = []
            for train_idx, val_idx in kf.split(X):
                pipe_cb = make_pipeline(model, use_catboost=True, feature_set=fset)
                pipe_cb.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = pipe_cb.predict(X.iloc[val_idx])
                rmse_list.append(np.sqrt(mean_squared_error(y.iloc[val_idx], pred)))
            rmse_scores = np.array(rmse_list)
        else:
            pipe = make_pipeline(model, feature_set=fset)
            scores = cross_val_score(pipe, X, y, cv=kf,
                                     scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
        results[name] = {
            'mean': rmse_scores.mean(),
            'std': rmse_scores.std(),
            'scores': rmse_scores,
        }
        print(f'{name:12s}: RMSLE = {rmse_scores.mean():.5f} ± {rmse_scores.std():.5f} [Set {fset}]')

    return results


# ============================================================
# 予測 & 提出
# ============================================================

def train_and_predict(X, y, X_test, test_ids, model_name='GBR', seed=42):
    """フル学習 → テスト予測 → submission.csv"""
    models = get_models(seed)
    model = models[model_name]
    fset = MODEL_FEATURE_SET.get(model_name)
    is_cb = model_name == 'CatBoost'
    pipe = make_pipeline(model, use_catboost=is_cb, feature_set=fset)

    pipe.fit(X, y)
    preds_log = pipe.predict(X_test)
    preds = np.expm1(preds_log)
    preds = np.maximum(preds, 0)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    filename = f'submission_{model_name.lower()}.csv'
    submission.to_csv(filename, index=False)
    print(f'\n{filename} を作成しました ({len(submission)}行)')
    print(f'SalePrice: mean={preds.mean():.0f}, median={np.median(preds):.0f}')

    return submission


# ============================================================
# スタッキング & ブレンディング
# ============================================================

def stacking_predict(X, y, X_test, test_ids, seed=42):
    """OOFスタッキング: 複数モデル → Ridgeメタモデル"""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    base_models = get_models(seed)

    n_train = len(X)
    n_test = len(X_test)
    n_models = len(base_models)

    oof_preds = np.zeros((n_train, n_models))
    test_preds = np.zeros((n_test, n_models))

    for i, (name, model) in enumerate(base_models.items()):
        fset = MODEL_FEATURE_SET.get(name)
        print(f'  {name} [Set {fset}]...', end='', flush=True)
        test_preds_fold = np.zeros((n_test, 5))
        is_cb = name == 'CatBoost'

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            pipe = make_pipeline(model, use_catboost=is_cb, feature_set=fset)
            pipe.fit(X_train_fold, y_train_fold)
            oof_preds[val_idx, i] = pipe.predict(X_val_fold)
            test_preds_fold[:, fold] = pipe.predict(X_test)

        test_preds[:, i] = test_preds_fold.mean(axis=1)
        rmse = np.sqrt(mean_squared_error(y, oof_preds[:, i]))
        print(f' OOF RMSLE={rmse:.5f}')

    # メタモデル
    meta = Ridge(alpha=1.0)
    meta.fit(oof_preds, y)
    meta_pred = meta.predict(test_preds)

    # OOF CVスコア
    meta_cv = np.sqrt(-cross_val_score(
        Ridge(alpha=1.0), oof_preds, y,
        cv=KFold(5, shuffle=True, random_state=seed),
        scoring='neg_mean_squared_error'))
    print(f'\nStack CV RMSLE = {meta_cv.mean():.5f} ± {meta_cv.std():.5f}')
    print(f'Meta weights: {dict(zip(base_models.keys(), meta.coef_.round(3)))}')

    preds = np.expm1(meta_pred)
    preds = np.maximum(preds, 0)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    submission.to_csv('submission_stack.csv', index=False)
    print(f'\nsubmission_stack.csv を作成しました ({len(submission)}行)')
    print(f'SalePrice: mean={preds.mean():.0f}, median={np.median(preds):.0f}')

    return submission, meta_cv.mean()


# ============================================================
# メイン
# ============================================================

if __name__ == '__main__':
    X, y, X_test, test_ids = load_data()

    print('=' * 60)
    print('モデル比較 (5-fold CV, RMSLE)')
    print('=' * 60)
    results = evaluate_models(X, y)

    # ベストモデルで予測
    best_model = min(results, key=lambda k: results[k]['mean'])
    print(f'\nベストモデル: {best_model} (RMSLE={results[best_model]["mean"]:.5f})')

    # 単体ベストの提出
    train_and_predict(X, y, X_test, test_ids, model_name=best_model)

    # スタッキング
    print('\n' + '=' * 60)
    print('スタッキング')
    print('=' * 60)
    stacking_predict(X, y, X_test, test_ids)

    # マルチシードスタッキング (seed averaging)
    print('\n' + '=' * 60)
    print('マルチシード スタッキング (seeds: 42, 123, 456)')
    print('=' * 60)
    multi_preds = []
    for seed in [42, 123, 456]:
        print(f'\n--- seed={seed} ---')
        sub, cv = stacking_predict(X, y, X_test, test_ids, seed=seed)
        multi_preds.append(sub['SalePrice'].values)

    avg_preds = np.mean(multi_preds, axis=0)
    submission_multi = pd.DataFrame({'Id': test_ids, 'SalePrice': avg_preds})
    submission_multi.to_csv('submission_multiseed.csv', index=False)
    print(f'\nsubmission_multiseed.csv を作成しました')
    print(f'SalePrice: mean={avg_preds.mean():.0f}, median={np.median(avg_preds):.0f}')
