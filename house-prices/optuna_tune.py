"""
Optuna hyperparameter optimization for GBR, XGBoost, LightGBM, CatBoost.
Feature set is fixed (current best: LOO TE + domain features + QOL_Score).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from main import load_data, HousePricesPreprocessor, CatBoostPreprocessor, make_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y, X_test, test_ids = load_data()
KF = KFold(n_splits=5, shuffle=True, random_state=42)


def cv_score_pipeline(model):
    """Standard pipeline CV (RMSLE)"""
    pipe = make_pipeline(model)
    scores = cross_val_score(pipe, X, y, cv=KF, scoring='neg_mean_squared_error')
    return np.sqrt(-scores).mean()


def cv_score_catboost(params, seed=42):
    """CatBoost with CatBoostPreprocessor (cat_features)"""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        prep = CatBoostPreprocessor()
        X_tr = prep.fit_transform(X.iloc[train_idx], y.iloc[train_idx])
        X_va = prep.transform(X.iloc[val_idx])
        cat_idx = prep.cat_feature_indices_

        cb = CatBoostRegressor(**params, loss_function='RMSE', verbose=0)
        cb.fit(X_tr, y.iloc[train_idx],
               eval_set=(X_va, y.iloc[val_idx]),
               cat_features=cat_idx,
               early_stopping_rounds=100)
        preds = cb.predict(X_va)
        rmse_scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
    return np.mean(rmse_scores)


# ============================================================
# GBR
# ============================================================
def objective_gbr(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 5000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'loss': 'huber',
        'random_state': 42,
    }
    model = GradientBoostingRegressor(**params)
    return cv_score_pipeline(model)


# ============================================================
# XGBoost
# ============================================================
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1500, 4000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'random_state': 42,
        'verbosity': 0,
    }
    model = xgb.XGBRegressor(**params)
    return cv_score_pipeline(model)


# ============================================================
# LightGBM
# ============================================================
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1500, 4000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'random_state': 42,
        'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    return cv_score_pipeline(model)


# ============================================================
# CatBoost
# ============================================================
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 5000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-2, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
    }
    return cv_score_catboost(params)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    N_TRIALS = 40

    results = {}

    # --- GBR ---
    print('=' * 60)
    print(f'GBR Optimization ({N_TRIALS} trials)')
    print('=' * 60)
    study_gbr = optuna.create_study(direction='minimize')
    study_gbr.optimize(objective_gbr, n_trials=N_TRIALS, show_progress_bar=True)
    print(f'  Best RMSLE: {study_gbr.best_value:.5f} (current: 0.11260)')
    print(f'  Best params: {study_gbr.best_params}')
    results['GBR'] = {'score': study_gbr.best_value, 'params': study_gbr.best_params}

    # --- XGBoost ---
    print('\n' + '=' * 60)
    print(f'XGBoost Optimization ({N_TRIALS} trials)')
    print('=' * 60)
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)
    print(f'  Best RMSLE: {study_xgb.best_value:.5f} (current: 0.11716)')
    print(f'  Best params: {study_xgb.best_params}')
    results['XGBoost'] = {'score': study_xgb.best_value, 'params': study_xgb.best_params}

    # --- LightGBM ---
    print('\n' + '=' * 60)
    print(f'LightGBM Optimization ({N_TRIALS} trials)')
    print('=' * 60)
    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=True)
    print(f'  Best RMSLE: {study_lgbm.best_value:.5f} (current: 0.12329)')
    print(f'  Best params: {study_lgbm.best_params}')
    results['LightGBM'] = {'score': study_lgbm.best_value, 'params': study_lgbm.best_params}

    # --- CatBoost ---
    print('\n' + '=' * 60)
    print(f'CatBoost Optimization ({N_TRIALS} trials)')
    print('=' * 60)
    study_cb = optuna.create_study(direction='minimize')
    study_cb.optimize(objective_catboost, n_trials=N_TRIALS, show_progress_bar=True)
    print(f'  Best RMSLE: {study_cb.best_value:.5f} (current: 0.11345)')
    print(f'  Best params: {study_cb.best_params}')
    results['CatBoost'] = {'score': study_cb.best_value, 'params': study_cb.best_params}

    # --- Summary ---
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    before = {'GBR': 0.11260, 'XGBoost': 0.11716, 'LightGBM': 0.12329, 'CatBoost': 0.11345}
    for name in results:
        b = before[name]
        a = results[name]['score']
        print(f'{name:12s}: {b:.5f} -> {a:.5f}  ({a-b:+.5f})')
