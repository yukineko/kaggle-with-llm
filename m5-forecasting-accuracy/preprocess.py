"""
M5 Forecasting Accuracy — 前処理スクリプト (省メモリ・ストリーミング版)

全ファイルをチャンク / ストリーミングで処理し、メモリに全量ロードしない。
- calendar : 小さいので全量ロード (1969行)
- prices   : チャンク読みで item_max_price を計算、sales チャンクごとに再スキャン
- sales    : chunksize 行ずつ読み込み → melt → 特徴量 → parquet 逐次書き出し

Phase 1: CSV → df_features.parquet
Phase 2: df_features.parquet → train_X.dat, train_y.dat, val.parquet, eval.parquet

Usage:
    python preprocess.py [--keep-from-day 1100] [--chunk-size 1000]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# 設定
# ============================================================
DATA_DIR = Path(__file__).parent
OUT_PATH = DATA_DIR / 'df_features.parquet'
PRICES_PATH = DATA_DIR / 'sell_prices.csv'
SALES_PATH  = DATA_DIR / 'sales_train_evaluation.csv'
CAL_PATH    = DATA_DIR / 'calendar.csv'

TRAIN_X_PATH   = DATA_DIR / 'train_X.dat'
TRAIN_Y_PATH   = DATA_DIR / 'train_y.dat'
TRAIN_CAT_PATH = DATA_DIR / 'train_cat.dat'
VAL_PATH       = DATA_DIR / 'val.parquet'
EVAL_PATH      = DATA_DIR / 'eval.parquet'

PRED_HORIZON = 28
N_DAYS       = 1941
VAL_START_DAY  = 1886
EVAL_START_DAY = 1914

META_COLS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
CAL_COLS  = ['d', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
             'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
             'snap_CA', 'snap_TX', 'snap_WI']
CAT_COLS  = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
             'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

PRICES_CHUNK = 500_000

# 給料日定義
PAYDAY_DAYS = {14, 15, 16, 28, 29, 30, 31}  # 15日・末日周辺
PAYDAY_EXACT = {15, 28, 29, 30, 31}          # 給料日当日
# cat_id エンコード (sorted): FOODS=0, HOBBIES=1, HOUSEHOLD=2
FOODS_CAT_ID = 0
HOBBIES_CAT_ID = 1
HOUSEHOLD_CAT_ID = 2

# ============================================================
# EDA Step 6: イベント消費クラスター & せっかく買い指数
# ============================================================
# A=Outing/Premium-Up, B=Home-Party/Bulk-Up, C=Closed, D=Others/No-Event
EVENT_CONSUMPTION_CLUSTER: dict[str, int] = {
    'Easter': 0, 'SuperBowl': 0, 'LaborDay': 0,
    'ColumbusDay': 0, 'VeteransDay': 0,
    'IndependenceDay': 1, 'Halloween': 1, 'MemorialDay': 1,
    'Christmas': 2, 'Thanksgiving': 2,
}
EVT_CLUSTER_DEFAULT = 3  # D: Others / No-Event

IMPULSE_BUY_INDEX: dict[str, float] = {
    'Easter': 36.7, 'LaborDay': 12.8, 'SuperBowl': 5.2,
    'ColumbusDay': 3.5, 'VeteransDay': 2.1,
    'MemorialDay': -4.3, 'IndependenceDay': -8.5, 'Halloween': -11.2,
    'Christmas': 0.0, 'Thanksgiving': 0.0,
}


# ============================================================
# グローバルカテゴリマッピング (チャンク間で一貫したエンコード)
# ============================================================
def build_cat_mappings(sales_path: Path, cal_path: Path) -> dict[str, dict]:
    """全 CSV をスキャンしてカテゴリ列 → int のグローバルマッピングを構築。"""
    # Sales CSV からメタ列のユニーク値を取得
    uniques: dict[str, set] = {}
    for chunk in pd.read_csv(sales_path, usecols=['item_id', 'dept_id', 'cat_id',
                                                   'store_id', 'state_id'],
                             chunksize=5000):
        for col in chunk.columns:
            if col not in uniques:
                uniques[col] = set()
            uniques[col].update(chunk[col].unique())
    # Calendar CSV からカレンダー列のユニーク値を取得
    cal = pd.read_csv(cal_path)
    for col in ['weekday', 'event_name_1', 'event_type_1',
                'event_name_2', 'event_type_2']:
        uniques[col] = set(cal[col].dropna().unique())
    del cal
    # ソートして固定マッピング作成 (NaN は -1)
    mappings = {}
    for col, vals in uniques.items():
        sorted_vals = sorted(vals)
        mappings[col] = {v: i for i, v in enumerate(sorted_vals)}
    return mappings


# ============================================================
# ストリーミングユーティリティ
# ============================================================
def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes('integer').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes('float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def stream_item_max_price(prices_path: Path) -> dict:
    """prices CSV をチャンク読みして {item_id: max_price} を構築。"""
    result: dict[str, float] = {}
    for chunk in pd.read_csv(prices_path, chunksize=PRICES_CHUNK,
                             usecols=['item_id', 'sell_price']):
        for item_id, price in zip(chunk['item_id'], chunk['sell_price']):
            if item_id not in result or price > result[item_id]:
                result[item_id] = price
    return result


def stream_prices_for_items(
    prices_path: Path, item_ids: set, store_ids: set,
) -> pd.DataFrame:
    """prices CSV から特定の item/store の行だけ抽出 (チャンク読み)。"""
    parts = []
    for chunk in pd.read_csv(prices_path, chunksize=PRICES_CHUNK):
        mask = chunk['item_id'].isin(item_ids) & chunk['store_id'].isin(store_ids)
        filtered = chunk[mask]
        if len(filtered) > 0:
            parts.append(filtered)
    if parts:
        return reduce_mem(pd.concat(parts, ignore_index=True))
    return pd.DataFrame(columns=['store_id', 'item_id', 'wm_yr_wk', 'sell_price'])


# ============================================================
# チャンク処理
# ============================================================
def process_chunk(
    chunk: pd.DataFrame,
    d_cols: list[str],
    calendar: pd.DataFrame,
    prices_path: Path,
    item_max_price: dict,
    keep_from_day: int,
    cat_mappings: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """1 チャンク (複数アイテムの wide 行) → 特徴量付き long DataFrame。"""

    # --- first_sale_day ---
    mat      = chunk[d_cols].values
    has_sale = (mat > 0)
    first_idx       = has_sale.argmax(axis=1)
    never_sold_mask = has_sale.sum(axis=1) == 0
    fsd = pd.DataFrame({
        'id': chunk['id'].values,
        'first_sale_day': np.where(never_sold_mask, N_DAYS + 1, first_idx + 1),
    })
    del mat, has_sale, first_idx, never_sold_mask

    # --- wide → long ---
    df = chunk.melt(id_vars=META_COLS, value_vars=d_cols,
                    var_name='d', value_name='sales')
    df['d_num'] = df['d'].str[2:].astype('int16')
    df['sales'] = df['sales'].astype('float32')

    # --- フィルタ ---
    df = df[df['d_num'] >= keep_from_day].reset_index(drop=True)
    df = df.merge(fsd, on='id', how='left')
    df = df[df['d_num'] >= df['first_sale_day']].drop('first_sale_day', axis=1)
    df = df.reset_index(drop=True)
    del fsd

    if len(df) == 0:
        return df

    # --- カレンダー結合 ---
    df = df.merge(calendar, on='d', how='left')

    # --- 価格結合 (prices をチャンクスキャンして必要行だけ取得) ---
    prices = stream_prices_for_items(
        prices_path,
        set(df['item_id'].unique()),
        set(df['store_id'].unique()),
    )
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    del prices

    # === 棚なしフラグ: sell_price が欠落 = その週は物理的に未取扱い ===
    df['not_on_shelf'] = df['sell_price'].isna().astype('int8')

    # --- ソート ---
    df = df.sort_values(['id', 'd_num']).reset_index(drop=True)

    # === 価格特徴量 (カテゴリエンコード前) ===
    df['price_change_rel'] = (
        df.groupby('id')['sell_price'].pct_change().astype('float32')
    )
    mp = df['item_id'].map(item_max_price).values.astype('float32')
    df['discount_ratio'] = (
        (mp - df['sell_price'].values) / np.where(mp == 0, np.nan, mp)
    ).astype('float32')
    del mp

    # === SNAP 交差特徴量 (カテゴリエンコード前: state_id が文字列) ===
    _snap_map = {'CA': 'snap_CA', 'TX': 'snap_TX', 'WI': 'snap_WI'}
    df['snap_active'] = np.int8(0)
    for _state, _col in _snap_map.items():
        _m = df['state_id'] == _state
        df.loc[_m, 'snap_active'] = df.loc[_m, _col].values.astype('int8')
    df['snap_wday'] = (df['snap_active'] * df['wday']).astype('int8')

    # === [EDA Step 6] イベント消費クラスター + せっかく買い指数 ===
    df['event_consumption_type'] = (
        df['event_name_1'].map(EVENT_CONSUMPTION_CLUSTER)
        .fillna(EVT_CLUSTER_DEFAULT).astype('int8')
    )
    df['impulse_buy_index'] = (
        df['event_name_1'].map(IMPULSE_BUY_INDEX)
        .fillna(0.0).astype('float32')
    )

    # === [EDA Step 6] SNAP 支給日からの経過日数 (州別 → 統合) ===
    df['days_since_snap'] = np.int16(999)
    for _state, _dss_col in [('CA', 'days_since_snap_CA'),
                              ('TX', 'days_since_snap_TX'),
                              ('WI', 'days_since_snap_WI')]:
        _m = df['state_id'] == _state
        if _m.any():
            df.loc[_m, 'days_since_snap'] = df.loc[_m, _dss_col].values
    df.drop(columns=['days_since_snap_CA', 'days_since_snap_TX',
                      'days_since_snap_WI'], inplace=True)

    # === is_snap_first_weekend: SNAP支給後の最初の週末 (州別 → 統合) ===
    df['is_snap_first_weekend'] = np.int8(0)
    for _state, _sfwe_col in [('CA', 'is_snap_first_we_CA'),
                                ('TX', 'is_snap_first_we_TX'),
                                ('WI', 'is_snap_first_we_WI')]:
        _m = df['state_id'] == _state
        if _m.any():
            df.loc[_m, 'is_snap_first_weekend'] = df.loc[_m, _sfwe_col].values
    df.drop(columns=['is_snap_first_we_CA', 'is_snap_first_we_TX',
                      'is_snap_first_we_WI'], inplace=True)

    # === [EDA Step 6] CA_4 特異性フラグ ===
    df['is_CA4'] = (df['store_id'] == 'CA_4').astype('int8')
    # CA_4 × event_consumption_type (スパース対策: クラスタレベルのみ)
    # 0=非CA4, 1=CA4_Outing, 2=CA4_HomePty, 3=CA4_Closed, 4=CA4_Others
    df['CA4_x_evt_type'] = np.where(
        df['is_CA4'] == 1,
        df['event_consumption_type'] + 1,
        0
    ).astype('int8')

    # === カテゴリ列エンコード (グローバルマッピング使用) ===
    if cat_mappings is not None:
        for col in CAT_COLS:
            df[col] = df[col].map(cat_mappings[col]).fillna(-1).astype('int16')
    else:
        for col in CAT_COLS:
            df[col] = df[col].astype('category').cat.codes.astype('int16')

    # === カレンダー特徴量 ===
    df['is_weekend']          = (df['wday'] >= 6).astype('int8')
    df['day']                 = df['date'].dt.day.astype('int8')
    df['is_month_end']        = df['date'].dt.is_month_end.astype('int8')
    df['is_month_start']      = df['date'].dt.is_month_start.astype('int8')
    df['is_christmas_nearby'] = (
        (df['date'].dt.month == 12) & (df['date'].dt.day.between(23, 27))
    ).astype('int8')

    # === 給料日・SNAP月初フラグ ===
    df['payday_flag']    = df['day'].isin(PAYDAY_EXACT).astype('int8')
    _in_window           = df['day'].isin(PAYDAY_DAYS)
    df['payday_weekend'] = (_in_window & (df['is_weekend'] == 1)).astype('int8')
    df['snap_first_10d'] = ((df['snap_active'] == 1) & (df['day'] <= 10)).astype('int8')
    del _in_window

    # === ラグ特徴量 ===
    for lag in [28, 35, 42, 56]:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag).astype('float32')

    # === ローリング統計 ===
    df['_s28'] = df.groupby('id')['sales'].shift(PRED_HORIZON).astype('float32')
    for win in [7, 28, 56]:
        grp = df.groupby('id')['_s28']
        df[f'roll_mean_{win}'] = (
            grp.rolling(win, min_periods=1).mean()
               .reset_index(level=0, drop=True).astype('float32')
        )
        df[f'roll_std_{win}'] = (
            grp.rolling(win, min_periods=1).std()
               .reset_index(level=0, drop=True).astype('float32')
        )
    # roll_median_7: 外れ値に頑健な中央値
    df['roll_median_7'] = (
        df.groupby('id')['_s28']
          .rolling(7, min_periods=1).median()
          .reset_index(level=0, drop=True).astype('float32')
    )
    # EWMA: 指数平滑移動平均 (直近トレンド捕捉)
    df['ewma_7'] = (
        df.groupby('id')['_s28']
          .ewm(span=7, adjust=False).mean()
          .reset_index(level=0, drop=True).astype('float32')
    )
    df['ewma_28'] = (
        df.groupby('id')['_s28']
          .ewm(span=28, adjust=False).mean()
          .reset_index(level=0, drop=True).astype('float32')
    )

    # === 補充サイクル (スパイクからの経過日数) ===
    _rm7 = df['roll_mean_7'].values
    _s28_vals = df['_s28'].values
    _spike_mask = (_s28_vals > 2 * _rm7) & (_rm7 > 0)
    df['_spike_day'] = df['d_num'].where(pd.Series(_spike_mask, index=df.index)).astype('float32')
    df['_spike_day'] = df.groupby('id')['_spike_day'].ffill()
    df['days_since_spike'] = (
        df.groupby('id')['d_num'].shift(PRED_HORIZON).astype('float32')
        - df.groupby('id')['_spike_day'].shift(PRED_HORIZON)
    ).fillna(0).clip(0, 365).astype('float32')
    df.drop(columns=['_spike_day'], inplace=True)
    del _rm7, _s28_vals, _spike_mask

    # === 間欠需要 ===
    _zero = (df['_s28'] == 0).astype('float32')
    df['zeros_last_28'] = (
        _zero.groupby(df['id'])
             .rolling(28, min_periods=1).sum()
             .reset_index(level=0, drop=True).astype('float32')
    )
    _mask = df['sales'] > 0
    df['_last_sale_day'] = df['d_num'].where(_mask).astype('float32')
    df['_last_sale_day'] = df.groupby('id')['_last_sale_day'].ffill()
    df['days_since_last_sale'] = (
        df.groupby('id')['d_num'].shift(PRED_HORIZON)
        - df.groupby('id')['_last_sale_day'].shift(PRED_HORIZON)
    ).astype('float32').fillna(0)
    df.drop(columns=['_s28', '_last_sale_day'], inplace=True)

    return reduce_mem(df)


# ============================================================
# Phase 1: CSV → df_features.parquet
# ============================================================
def phase1_features(keep_from_day: int, chunk_size: int) -> None:
    """sales CSV をチャンク処理して df_features.parquet を生成。"""
    if OUT_PATH.exists():
        pf = pq.ParquetFile(OUT_PATH)
        print(f'[Phase 1 SKIP] {OUT_PATH} が既に存在 ({pf.metadata.num_rows:,} rows)')
        del pf
        return

    t0 = time.time()

    print('\n[Phase 1] CSV → df_features.parquet')
    calendar = reduce_mem(pd.read_csv(CAL_PATH, parse_dates=['date'])[CAL_COLS])
    # イベント前後3日フラグ (±3日以内にイベントがあれば 1)
    calendar = calendar.sort_values('d').reset_index(drop=True)
    _has_ev = calendar['event_name_1'].notna().astype(float)
    calendar['event_nearby'] = (
        _has_ev.rolling(7, center=True, min_periods=1).max().astype('int8')
    )
    del _has_ev

    # === SNAP 支給日からの経過日数 (州別) ===
    _d_num_cal = calendar['d'].str[2:].astype('int16')
    for _state in ['CA', 'TX', 'WI']:
        _snap_col = f'snap_{_state}'
        _last_snap_day = _d_num_cal.where(calendar[_snap_col] == 1)
        _last_snap_day = _last_snap_day.ffill()
        calendar[f'days_since_snap_{_state}'] = (
            (_d_num_cal - _last_snap_day).fillna(999).clip(0, 999).astype('int16')
        )

    # === is_snap_first_weekend: SNAP期間内の最初の土日フラグ (州別) ===
    # wday: M5では 1=Sat,2=Sun,...,7=Fri → 土日 = wday in {1,2}
    _is_sat_or_sun = calendar['wday'].isin({1, 2}).astype('int8')
    for _state in ['CA', 'TX', 'WI']:
        _snap_col = f'snap_{_state}'
        _dss = calendar[f'days_since_snap_{_state}']
        # SNAP支給中(days_since_snap <= 9) かつ 土日 かつ 初回の週末
        # 初回 = days_since_snap が最小の土日 → days_since_snap <= 6 で最初の土日をカバー
        calendar[f'is_snap_first_we_{_state}'] = (
            (calendar[_snap_col] == 1) &  # SNAP active期間
            (_is_sat_or_sun == 1) &        # 土日
            (_dss <= 6)                     # 支給開始から6日以内 (最初の週末)
        ).astype('int8')
    del _d_num_cal, _last_snap_day, _is_sat_or_sun

    print(f'  calendar: {calendar.shape}')

    print('  item_max_price 計算中...')
    item_max_price = stream_item_max_price(PRICES_PATH)
    print(f'  {len(item_max_price)} items')

    print('  カテゴリマッピング構築中...')
    cat_mappings = build_cat_mappings(SALES_PATH, CAL_PATH)
    for col, m in cat_mappings.items():
        print(f'    {col}: {len(m)} categories')

    header = pd.read_csv(SALES_PATH, nrows=0)
    d_cols = [c for c in header.columns if c.startswith('d_')]
    print(f'  D_COLS: {d_cols[0]} ~ {d_cols[-1]} ({len(d_cols)} days)')

    reader = pd.read_csv(SALES_PATH, chunksize=chunk_size)
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for i, chunk in enumerate(reader):
        df_chunk = process_chunk(
            chunk, d_cols, calendar, PRICES_PATH, item_max_price, keep_from_day,
            cat_mappings=cat_mappings,
        )
        if len(df_chunk) == 0:
            continue

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PATH, table.schema)
        writer.write_table(table)
        total_rows += len(df_chunk)

        elapsed = time.time() - t0
        print(f'  chunk {i+1:>3}  |  items ~{(i+1)*chunk_size:>6,}  |  '
              f'rows {total_rows:>12,}  |  {elapsed:.0f}s')
        del df_chunk, table

    if writer is not None:
        writer.close()

    print(f'  完了: {total_rows:,} rows  ({OUT_PATH.stat().st_size/1e6:.0f} MB)  '
          f'{(time.time()-t0)/60:.1f} min')


# ============================================================
# Phase 1.5: Target Encoding + Store Profiling (4指標 + 交差特徴量)
# ============================================================
def phase1_5_target_encoding() -> None:
    """df_features.parquet に店舗プロファイリング特徴量を追加 (train 期間のみで算出):
    - te_store_dept_lag28: (store_id, dept_id, d_num-28) の平均売上
    - snap_dependency_score: SNAP支給期間 / 非支給期間の売上比
    - payroll_dependency_score: 給料日周辺 / 月平均の売上比
    - weekend_intensity: 土日 / 平日の売上比
    - luxury_affinity_score: HOBBIES カテゴリの売上比率
    - store_income_type: 上記からクラスタリング (0=Type-S, 1=Type-P, 2=Type-B)
    - snap_dep_interaction: snap_active × snap_dependency_score
    - weekend_interaction: is_weekend × weekend_intensity
    - snap_x_income: snap_active × 3 + store_income_type (6カテゴリ交差)
    """
    pf = pq.ParquetFile(OUT_PATH)
    existing_cols = [f.name for f in pf.schema_arrow]
    # 新指標が揃っていればスキップ
    new_cols = ['snap_dependency_score', 'payroll_dependency_score',
                'weekend_intensity', 'luxury_affinity_score',
                'snap_dep_interaction', 'weekend_interaction',
                'price_sensitivity_index', 'pb_ratio',
                'price_x_psi', 'snap_x_pb',
                'cat_income_elasticity', 'stockpiling_index',
                'roll_mean_56_weighted',
                'income_event_sensitivity', 'spike_hint',
                'cat_snap_sensitivity', 'cat_payday_sensitivity',
                'snap_cat_lift', 'payday_cat_lift',
                'luxury_pressure', 'luxury_pressure_x_payday',
                'weekday_density_ratio',
                'store_dept_wday_avg', 'store_dept_premium_share']
    if all(c in existing_cols for c in new_cols):
        print(f'[Phase 1.5 SKIP] 店舗プロファイリング特徴量が既に存在')
        del pf
        return

    t0 = time.time()
    print('\n[Phase 1.5] Store Profiling + Target Encoding (train期間のみ)')
    n_rg = pf.metadata.num_row_groups

    # === Pass 0: PB_Ratio 用に店舗ごとの価格 P20 閾値を算出 ===
    print('  [0] 店舗別 P20 価格閾値を算出中...')
    store_prices: dict[int, list] = {}  # store_id → [sell_price, ...]
    for i in range(n_rg):
        rg = pf.read_row_group(i, columns=['store_id', 'd_num', 'sell_price']).to_pandas()
        rg_tr = rg[rg['d_num'].values < VAL_START_DAY]
        for sid, price in zip(rg_tr['store_id'].values, rg_tr['sell_price'].values):
            p = float(price)
            if p > 0 and not np.isnan(p):
                sid_int = int(sid)
                if sid_int not in store_prices:
                    store_prices[sid_int] = []
                store_prices[sid_int].append(p)
        del rg, rg_tr
    p20_threshold: dict[int, float] = {}
    for sid, prices_list in store_prices.items():
        p20_threshold[sid] = float(np.percentile(prices_list, 20))
    del store_prices
    print(f'    P20 thresholds: {dict(sorted(p20_threshold.items()))}')

    # === Pass 0b: item_premium_flag (dept内 Z-score > 2.0) ===
    print('  [0b] プレミアム品判定 (Z>2.0)...')
    item_price_acc: dict[int, list] = {}  # item_id(encoded) → [sum, count]
    item_to_dept: dict[int, int] = {}     # item_id(encoded) → dept_id(encoded)
    for i in range(n_rg):
        rg = pf.read_row_group(i, columns=['item_id', 'dept_id', 'sell_price']).to_pandas()
        for item, dept, price in zip(rg['item_id'].values, rg['dept_id'].values,
                                      rg['sell_price'].values):
            p = float(price)
            if p > 0 and not np.isnan(p):
                ik = int(item)
                if ik not in item_price_acc:
                    item_price_acc[ik] = [0.0, 0]
                    item_to_dept[ik] = int(dept)
                item_price_acc[ik][0] += p
                item_price_acc[ik][1] += 1
        del rg
    # dept別の平均・標準偏差
    from collections import defaultdict
    dept_item_prices: dict[int, list] = defaultdict(list)
    item_avg_price: dict[int, float] = {}
    for ik, (s, c) in item_price_acc.items():
        avg = s / c
        item_avg_price[ik] = avg
        dept_item_prices[item_to_dept[ik]].append(avg)
    del item_price_acc
    dept_stats: dict[int, tuple] = {}
    for dk, prices in dept_item_prices.items():
        arr = np.array(prices)
        dept_stats[dk] = (float(arr.mean()), float(arr.std()))
    del dept_item_prices
    item_premium_flag: dict[int, int] = {}  # item_id(encoded) → 1 if Z>2.0
    n_premium = 0
    for ik, avg in item_avg_price.items():
        dk = item_to_dept[ik]
        mean, std = dept_stats[dk]
        z = (avg - mean) / std if std > 0 else 0
        if z > 2.0:
            item_premium_flag[ik] = 1
            n_premium += 1
    del item_avg_price, item_to_dept
    print(f'    Premium items (Z>2.0): {n_premium}')

    # === Pass 1: train期間 (d_num < VAL_START_DAY) のみで集計 ===
    print('  [1/3] 集計中 (d_num < %d のみ)...' % VAL_START_DAY)
    te_agg: dict[tuple, list] = {}          # (store_id, dept_id, d_num) → [sum, count]
    snap_agg: dict[int, list] = {}          # store_id → [snap_sum, snap_cnt, nosnap_sum, nosnap_cnt]
    payroll_agg: dict[int, list] = {}       # store_id → [payday_sum, payday_cnt, total_sum, total_cnt]
    weekend_agg: dict[int, list] = {}       # store_id → [weekend_sum, weekend_cnt, weekday_sum, weekday_cnt]
    luxury_agg: dict[int, list] = {}        # store_id → [hobbies_sum, total_sum]
    psi_agg: dict[int, list] = {}           # store_id → [disc_sum, disc_cnt, nodisc_sum, nodisc_cnt]
    pb_agg: dict[int, list] = {}            # store_id → [pb_sales, total_sales]
    # category_income_elasticity: (store_id, cat_id) → [event_sum, event_cnt, normal_sum, normal_cnt]
    cie_agg: dict[tuple, list] = {}
    # stockpiling_index: store_id → {d_num: household_sales} (HOUSEHOLD のみ)
    stockpile_daily: dict[int, dict] = {}
    # カテゴリ別SNAP/給料日感度: (store_id, cat_id) → [event_sum, event_cnt, nonevent_sum, nonevent_cnt]
    cat_snap_agg: dict[tuple, list] = {}
    cat_payday_agg: dict[tuple, list] = {}
    # 乖離率集計: (store_id, cat_id, segment) → [residual_sum, count]
    # segment: 0=SNAP, 1=payday, 2=weekend, 3=other
    resid_agg: dict[tuple, list] = {}
    # weekday_density_ratio: (store_id, dept_id) → [weekday_sum, wd_cnt, weekend_sum, we_cnt]
    wdr_agg: dict[tuple, list] = {}
    # store_dept_wday_avg: (store_id, dept_id, wday) → [sum, count]
    # イベント日・SNAP日を除外した「真の日常」平均
    sdw_agg: dict[tuple, list] = {}
    # store_dept_premium_share: (store_id, dept_id) → [premium_sales, total_sales]
    sdps_agg: dict[tuple, list] = {}

    has_snap_active = 'snap_active' in existing_cols
    has_is_weekend = 'is_weekend' in existing_cols
    base_cols = ['store_id', 'dept_id', 'd_num', 'sales', 'day', 'cat_id',
                 'discount_ratio', 'sell_price', 'roll_mean_28', 'wday',
                 'event_name_1', 'item_id']
    if has_snap_active:
        base_cols.append('snap_active')
    else:
        base_cols.extend(['state_id', 'snap_CA', 'snap_TX', 'snap_WI'])
        print('    (snap_active 列なし → snap_CA/TX/WI から再計算)')
    if has_is_weekend:
        base_cols.append('is_weekend')
    else:
        base_cols.append('wday')
        print('    (is_weekend 列なし → wday から再計算)')
    read_cols = list(dict.fromkeys(base_cols))  # 重複除去

    for i in range(n_rg):
        rg = pf.read_row_group(i, columns=read_cols).to_pandas()
        # snap_active フォールバック
        if not has_snap_active:
            rg['snap_active'] = np.int8(0)
            rg.loc[rg['state_id'] == 0, 'snap_active'] = rg.loc[rg['state_id'] == 0, 'snap_CA'].values.astype('int8')
            rg.loc[rg['state_id'] == 1, 'snap_active'] = rg.loc[rg['state_id'] == 1, 'snap_TX'].values.astype('int8')
            rg.loc[rg['state_id'] == 2, 'snap_active'] = rg.loc[rg['state_id'] == 2, 'snap_WI'].values.astype('int8')
        # is_weekend フォールバック
        if not has_is_weekend:
            rg['is_weekend'] = (rg['wday'] >= 6).astype('int8')
        # train期間のみ
        mask = rg['d_num'].values < VAL_START_DAY
        rg_tr = rg[mask]
        # --- TE 集計 ---
        for sid, did, dnum, sales in zip(
            rg_tr['store_id'].values, rg_tr['dept_id'].values,
            rg_tr['d_num'].values, rg_tr['sales'].values,
        ):
            key = (int(sid), int(did), int(dnum))
            if key in te_agg:
                te_agg[key][0] += float(sales)
                te_agg[key][1] += 1
            else:
                te_agg[key] = [float(sales), 1]
        # --- weekday_density_ratio 集計 (store × dept) ---
        for sid, did, s, is_we in zip(
            rg_tr['store_id'].values, rg_tr['dept_id'].values,
            rg_tr['sales'].values, rg_tr['is_weekend'].values,
        ):
            wdr_key = (int(sid), int(did))
            if wdr_key not in wdr_agg:
                wdr_agg[wdr_key] = [0.0, 0, 0.0, 0]
            if int(is_we) == 0:  # weekday
                wdr_agg[wdr_key][0] += float(s)
                wdr_agg[wdr_key][1] += 1
            else:  # weekend
                wdr_agg[wdr_key][2] += float(s)
                wdr_agg[wdr_key][3] += 1
        # --- store_dept_wday_avg 集計 (イベント日・SNAP日を除外した日常平均) ---
        for sid, did, s, wday_v, snap_v, ev_v in zip(
            rg_tr['store_id'].values, rg_tr['dept_id'].values,
            rg_tr['sales'].values, rg_tr['wday'].values,
            rg_tr['snap_active'].values, rg_tr['event_name_1'].values,
        ):
            # イベント日とSNAP日を除外 → 純粋な「日常」
            if int(snap_v) == 1:
                continue
            # event_name_1 は encode 済み: -1=NoEvent, >=0=有イベント
            if int(ev_v) >= 0:
                continue
            sdw_key = (int(sid), int(did), int(wday_v))
            if sdw_key not in sdw_agg:
                sdw_agg[sdw_key] = [0.0, 0]
            sdw_agg[sdw_key][0] += float(s)
            sdw_agg[sdw_key][1] += 1
        # --- store_dept_premium_share 集計 (Z>2.0品の数量シェア) ---
        if 'item_id' in rg_tr.columns and 'sell_price' in rg_tr.columns:
            for sid, did, s, item_v in zip(
                rg_tr['store_id'].values, rg_tr['dept_id'].values,
                rg_tr['sales'].values, rg_tr['item_id'].values,
            ):
                sdps_key = (int(sid), int(did))
                if sdps_key not in sdps_agg:
                    sdps_agg[sdps_key] = [0.0, 0.0]
                sf = float(s)
                sdps_agg[sdps_key][1] += sf  # total
                item_key = int(item_v)
                if item_key in item_premium_flag:
                    sdps_agg[sdps_key][0] += sf  # premium
        # --- 6指標の集計 ---
        for sid, snap, sales, day, is_we, cat, dr, sp, dnum, rm28 in zip(
            rg_tr['store_id'].values, rg_tr['snap_active'].values,
            rg_tr['sales'].values, rg_tr['day'].values,
            rg_tr['is_weekend'].values, rg_tr['cat_id'].values,
            rg_tr['discount_ratio'].values, rg_tr['sell_price'].values,
            rg_tr['d_num'].values, rg_tr['roll_mean_28'].values,
        ):
            sid_int = int(sid)
            s = float(sales)
            # SNAP dependency
            if sid_int not in snap_agg:
                snap_agg[sid_int] = [0.0, 0, 0.0, 0]
            if int(snap) == 1:
                snap_agg[sid_int][0] += s
                snap_agg[sid_int][1] += 1
            else:
                snap_agg[sid_int][2] += s
                snap_agg[sid_int][3] += 1
            # Payroll dependency
            if sid_int not in payroll_agg:
                payroll_agg[sid_int] = [0.0, 0, 0.0, 0]
            payroll_agg[sid_int][2] += s
            payroll_agg[sid_int][3] += 1
            if int(day) in PAYDAY_DAYS:
                payroll_agg[sid_int][0] += s
                payroll_agg[sid_int][1] += 1
            # Weekend intensity
            if sid_int not in weekend_agg:
                weekend_agg[sid_int] = [0.0, 0, 0.0, 0]
            if int(is_we) == 1:
                weekend_agg[sid_int][0] += s
                weekend_agg[sid_int][1] += 1
            else:
                weekend_agg[sid_int][2] += s
                weekend_agg[sid_int][3] += 1
            # Luxury affinity (HOBBIES ratio)
            if sid_int not in luxury_agg:
                luxury_agg[sid_int] = [0.0, 0.0]
            luxury_agg[sid_int][1] += s
            if int(cat) == HOBBIES_CAT_ID:
                luxury_agg[sid_int][0] += s
            # Price Sensitivity Index (discount_ratio > 0.1)
            dr_f = float(dr)
            if not np.isnan(dr_f):
                if sid_int not in psi_agg:
                    psi_agg[sid_int] = [0.0, 0, 0.0, 0]
                if dr_f > 0.1:
                    psi_agg[sid_int][0] += s
                    psi_agg[sid_int][1] += 1
                else:
                    psi_agg[sid_int][2] += s
                    psi_agg[sid_int][3] += 1
            # PB Ratio (sell_price <= P20 threshold)
            sp_f = float(sp)
            if not np.isnan(sp_f) and sp_f > 0:
                if sid_int not in pb_agg:
                    pb_agg[sid_int] = [0.0, 0.0]
                pb_agg[sid_int][1] += s
                if sp_f <= p20_threshold.get(sid_int, 0.0):
                    pb_agg[sid_int][0] += s
            # category_income_elasticity: 給料日/SNAP日 vs 通常日
            cat_int = int(cat)
            cie_key = (sid_int, cat_int)
            if cie_key not in cie_agg:
                cie_agg[cie_key] = [0.0, 0, 0.0, 0]
            is_event_day = (int(snap) == 1) or (int(day) in PAYDAY_DAYS)
            if is_event_day:
                cie_agg[cie_key][0] += s
                cie_agg[cie_key][1] += 1
            else:
                cie_agg[cie_key][2] += s
                cie_agg[cie_key][3] += 1
            # cat_snap: カテゴリ別SNAP感度
            sc_key = (sid_int, cat_int)
            if sc_key not in cat_snap_agg:
                cat_snap_agg[sc_key] = [0.0, 0, 0.0, 0]
            if int(snap) == 1:
                cat_snap_agg[sc_key][0] += s
                cat_snap_agg[sc_key][1] += 1
            else:
                cat_snap_agg[sc_key][2] += s
                cat_snap_agg[sc_key][3] += 1
            # cat_payday: カテゴリ別給料日感度
            if sc_key not in cat_payday_agg:
                cat_payday_agg[sc_key] = [0.0, 0, 0.0, 0]
            if int(day) in PAYDAY_DAYS:
                cat_payday_agg[sc_key][0] += s
                cat_payday_agg[sc_key][1] += 1
            else:
                cat_payday_agg[sc_key][2] += s
                cat_payday_agg[sc_key][3] += 1
            # stockpiling_index: HOUSEHOLD 日次売上を店舗ごとに蓄積
            if cat_int == HOUSEHOLD_CAT_ID:
                dnum_int = int(dnum)
                if sid_int not in stockpile_daily:
                    stockpile_daily[sid_int] = {}
                if dnum_int not in stockpile_daily[sid_int]:
                    stockpile_daily[sid_int][dnum_int] = 0.0
                stockpile_daily[sid_int][dnum_int] += s
            # 乖離率 (residual from roll_mean_28)
            rm28_f = float(rm28)
            if rm28_f > 0 and not np.isnan(rm28_f):
                resid = (s - rm28_f) / rm28_f
                # セグメント判定: SNAP > payday > weekend > other
                if int(snap) == 1:
                    seg = 0
                elif int(day) in PAYDAY_DAYS:
                    seg = 1
                elif int(is_we) == 1:
                    seg = 2
                else:
                    seg = 3
                rkey = (sid_int, cat_int, seg)
                if rkey not in resid_agg:
                    resid_agg[rkey] = [0.0, 0]
                resid_agg[rkey][0] += resid
                resid_agg[rkey][1] += 1
        del rg, rg_tr

    te_lookup = {k: v[0] / v[1] for k, v in te_agg.items()}
    del te_agg
    print(f'    TE lookup entries: {len(te_lookup):,}')

    # --- 6指標の算出 ---
    snap_dep: dict[int, float] = {}
    for sid, (s_sum, s_cnt, ns_sum, ns_cnt) in snap_agg.items():
        snap_avg = s_sum / s_cnt if s_cnt > 0 else 0.0
        nosnap_avg = ns_sum / ns_cnt if ns_cnt > 0 else 0.0
        snap_dep[sid] = (snap_avg / nosnap_avg) if nosnap_avg > 0 else 1.0
    del snap_agg

    payroll_dep: dict[int, float] = {}
    for sid, (pd_sum, pd_cnt, t_sum, t_cnt) in payroll_agg.items():
        pd_avg = pd_sum / pd_cnt if pd_cnt > 0 else 0.0
        t_avg = t_sum / t_cnt if t_cnt > 0 else 0.0
        payroll_dep[sid] = (pd_avg / t_avg) if t_avg > 0 else 1.0
    del payroll_agg

    we_intensity: dict[int, float] = {}
    for sid, (we_sum, we_cnt, wd_sum, wd_cnt) in weekend_agg.items():
        we_avg = we_sum / we_cnt if we_cnt > 0 else 0.0
        wd_avg = wd_sum / wd_cnt if wd_cnt > 0 else 0.0
        we_intensity[sid] = (we_avg / wd_avg) if wd_avg > 0 else 1.0
    del weekend_agg

    lux_affinity: dict[int, float] = {}
    for sid, (hob_sum, tot_sum) in luxury_agg.items():
        lux_affinity[sid] = (hob_sum / tot_sum) if tot_sum > 0 else 0.0
    del luxury_agg

    # Price Sensitivity Index = discounted_avg / normal_avg
    price_sens: dict[int, float] = {}
    for sid, (d_sum, d_cnt, n_sum, n_cnt) in psi_agg.items():
        d_avg = d_sum / d_cnt if d_cnt > 0 else 0.0
        n_avg = n_sum / n_cnt if n_cnt > 0 else 0.0
        price_sens[sid] = (d_avg / n_avg) if n_avg > 0 else 1.0
    del psi_agg

    # PB Ratio = low-price sales / total sales
    pb_ratio: dict[int, float] = {}
    for sid, (pb_sum, tot_sum) in pb_agg.items():
        pb_ratio[sid] = (pb_sum / tot_sum) if tot_sum > 0 else 0.0
    del pb_agg

    # category_income_elasticity = event_avg / normal_avg (store × cat)
    cie_lookup: dict[tuple, float] = {}
    for key, (e_sum, e_cnt, n_sum, n_cnt) in cie_agg.items():
        e_avg = e_sum / e_cnt if e_cnt > 0 else 0.0
        n_avg = n_sum / n_cnt if n_cnt > 0 else 0.0
        cie_lookup[key] = (e_avg / n_avg) if n_avg > 0 else 1.0
    del cie_agg
    print(f'    CIE entries: {len(cie_lookup)} (store×cat)')

    # cat_snap_sensitivity = snap_avg / nosnap_avg (store × cat)
    cat_snap_sens: dict[tuple, float] = {}
    for key, (s_sum, s_cnt, ns_sum, ns_cnt) in cat_snap_agg.items():
        s_avg = s_sum / s_cnt if s_cnt > 0 else 0.0
        ns_avg = ns_sum / ns_cnt if ns_cnt > 0 else 0.0
        cat_snap_sens[key] = (s_avg / ns_avg) if ns_avg > 0 else 1.0
    del cat_snap_agg
    print(f'    Cat SNAP sens entries: {len(cat_snap_sens)}')

    # cat_payday_sensitivity = payday_avg / nopayday_avg (store × cat)
    cat_payday_sens: dict[tuple, float] = {}
    for key, (p_sum, p_cnt, np_sum, np_cnt) in cat_payday_agg.items():
        p_avg = p_sum / p_cnt if p_cnt > 0 else 0.0
        np_avg = np_sum / np_cnt if np_cnt > 0 else 0.0
        cat_payday_sens[key] = (p_avg / np_avg) if np_avg > 0 else 1.0
    del cat_payday_agg
    print(f'    Cat Payday sens entries: {len(cat_payday_sens)}')

    # stockpiling_index = lag-28 autocorrelation of HOUSEHOLD daily sales per store
    stockpile_idx: dict[int, float] = {}
    for sid, daily in stockpile_daily.items():
        if len(daily) < 56:  # 最低56日必要
            stockpile_idx[sid] = 0.0
            continue
        d_sorted = sorted(daily.keys())
        vals = np.array([daily[d] for d in d_sorted], dtype='float64')
        if vals.std() < 1e-8:
            stockpile_idx[sid] = 0.0
            continue
        n = len(vals)
        if n <= 28:
            stockpile_idx[sid] = 0.0
            continue
        mean = vals.mean()
        var = vals.var()
        autocov = np.mean((vals[:n-28] - mean) * (vals[28:] - mean))
        stockpile_idx[sid] = float(autocov / var) if var > 0 else 0.0
    del stockpile_daily
    print(f'    Stockpiling index: {dict(sorted(stockpile_idx.items()))}')

    # weekday_density_ratio = weekday_avg / weekend_avg (store × dept)
    wdr_lookup: dict[tuple, float] = {}
    for key, (wd_sum, wd_cnt, we_sum, we_cnt) in wdr_agg.items():
        wd_avg = wd_sum / wd_cnt if wd_cnt > 0 else 0.0
        we_avg = we_sum / we_cnt if we_cnt > 0 else 0.0
        wdr_lookup[key] = (wd_avg / we_avg) if we_avg > 0 else 1.0
    del wdr_agg
    print(f'    Weekday density ratio entries: {len(wdr_lookup)} (store×dept)')

    # store_dept_wday_avg = イベント・SNAP除外の日常平均 (store × dept × wday)
    sdw_lookup: dict[tuple, float] = {}
    for key, (s_sum, s_cnt) in sdw_agg.items():
        sdw_lookup[key] = s_sum / s_cnt if s_cnt > 0 else 0.0
    del sdw_agg
    print(f'    Store-dept-wday avg entries: {len(sdw_lookup)} (store×dept×wday)')

    # store_dept_premium_share = プレミアム品の数量シェア (store × dept)
    sdps_lookup: dict[tuple, float] = {}
    for key, (prem, total) in sdps_agg.items():
        sdps_lookup[key] = (prem / total) if total > 0 else 0.0
    del sdps_agg
    print(f'    Store-dept premium share entries: {len(sdps_lookup)} (store×dept)')

    # 乖離率の平均 per (store_id, cat_id, segment)
    resid_mean: dict[tuple, float] = {}
    for rkey, (rsum, rcnt) in resid_agg.items():
        resid_mean[rkey] = rsum / rcnt if rcnt > 0 else 0.0
    del resid_agg

    # income_event_sensitivity: 店舗ごとに最大乖離セグメントで分類
    # 0=SNAP依存, 1=給料日依存, 2=週末依存
    SEG_NAMES = {0: 'SNAP', 1: 'Payday', 2: 'Weekend'}
    income_event_sens: dict[int, int] = {}
    # spike_expected: (store_id, cat_id, segment) → 期待乖離率
    spike_expected: dict[tuple, float] = {}
    all_store_ids_r = set()
    for (sid, cid, seg), val in resid_mean.items():
        all_store_ids_r.add(sid)
        spike_expected[(sid, cid, seg)] = val
    for sid in all_store_ids_r:
        # 全カテゴリの平均乖離率を seg=0,1,2 で集約
        seg_totals = [0.0, 0.0, 0.0]
        seg_counts = [0, 0, 0]
        for (s, c, seg), val in resid_mean.items():
            if s == sid and seg < 3:
                seg_totals[seg] += val
                seg_counts[seg] += 1
        seg_avgs = [seg_totals[i] / seg_counts[i] if seg_counts[i] > 0 else 0.0
                    for i in range(3)]
        best_seg = int(np.argmax(seg_avgs))
        income_event_sens[sid] = best_seg
    print(f'\n    Income Event Sensitivity:')
    for sid in sorted(income_event_sens.keys()):
        seg = income_event_sens[sid]
        # 各セグメントの代表的乖離率 (FOODS cat_id=0)
        snap_r = resid_mean.get((sid, 0, 0), 0.0)
        pay_r = resid_mean.get((sid, 0, 1), 0.0)
        we_r = resid_mean.get((sid, 0, 2), 0.0)
        print(f'      store {sid}: {SEG_NAMES[seg]:>8} '
              f'(SNAP:{snap_r:+.3f} Pay:{pay_r:+.3f} WE:{we_r:+.3f})')

    # --- クラスタリング: Type-S / Type-P / Type-B ---
    store_ids = sorted(snap_dep.keys())
    snap_vals = np.array([snap_dep[s] for s in store_ids])
    pay_vals  = np.array([payroll_dep[s] for s in store_ids])
    snap_z = (snap_vals - snap_vals.mean()) / (snap_vals.std() + 1e-8)
    pay_z  = (pay_vals - pay_vals.mean()) / (pay_vals.std() + 1e-8)

    INCOME_LABELS = {0: 'Type-S', 1: 'Type-P', 2: 'Type-B'}
    store_income: dict[int, int] = {}
    print(f'\n    Store Profiling Results:')
    print(f'    {"store":>6} {"SNAP_dep":>9} {"Payroll":>8} {"Weekend":>8} '
          f'{"Luxury":>7} {"PSI":>6} {"PB%":>6} {"type":>8}')
    for idx, sid in enumerate(store_ids):
        sz, pz = snap_z[idx], pay_z[idx]
        if sz > 0 and sz >= pz:
            t = 0  # Type-S
        elif pz > 0 and pz > sz:
            t = 1  # Type-P
        else:
            t = 2  # Type-B
        store_income[sid] = t
        print(f'    {sid:>6} {snap_dep[sid]:>9.4f} {payroll_dep[sid]:>8.4f} '
              f'{we_intensity[sid]:>8.4f} {lux_affinity[sid]:>7.4f} '
              f'{price_sens[sid]:>6.3f} {pb_ratio[sid]:>6.3f} '
              f'{INCOME_LABELS[t]:>8}')

    # === Pass 2: 列を追加して新 parquet に書き出し ===
    print('\n  [2/3] 書き出し中...')
    tmp_path = OUT_PATH.with_suffix('.tmp.parquet')
    writer: pq.ParquetWriter | None = None
    for i in range(n_rg):
        rg = pf.read_row_group(i).to_pandas()
        # te_store_dept_lag28
        keys = list(zip(
            rg['store_id'].astype(int).values,
            rg['dept_id'].astype(int).values,
            (rg['d_num'].astype(int) - 28).values,
        ))
        rg['te_store_dept_lag28'] = pd.Series(keys).map(te_lookup).astype('float32').values
        del keys
        # 6 指標 (store-level, 全行にマッピング)
        sid_int = rg['store_id'].astype(int)
        rg['snap_dependency_score']    = sid_int.map(snap_dep).astype('float32').values
        rg['payroll_dependency_score'] = sid_int.map(payroll_dep).astype('float32').values
        rg['weekend_intensity']        = sid_int.map(we_intensity).astype('float32').values
        rg['luxury_affinity_score']    = sid_int.map(lux_affinity).astype('float32').values
        rg['price_sensitivity_index']  = sid_int.map(price_sens).astype('float32').values
        rg['pb_ratio']                 = sid_int.map(pb_ratio).astype('float32').values
        # store_income_type (categorical cluster)
        rg['store_income_type'] = sid_int.map(store_income).astype('int8').values
        # snap_active フォールバック
        if 'snap_active' not in rg.columns:
            rg['snap_active'] = np.int8(0)
            rg.loc[rg['state_id'] == 0, 'snap_active'] = rg.loc[rg['state_id'] == 0, 'snap_CA'].values.astype('int8')
            rg.loc[rg['state_id'] == 1, 'snap_active'] = rg.loc[rg['state_id'] == 1, 'snap_TX'].values.astype('int8')
            rg.loc[rg['state_id'] == 2, 'snap_active'] = rg.loc[rg['state_id'] == 2, 'snap_WI'].values.astype('int8')
        if 'is_weekend' not in rg.columns:
            rg['is_weekend'] = (rg['wday'] >= 6).astype('int8')
        # 交差特徴量
        rg['snap_dep_interaction'] = (
            rg['snap_active'].values.astype('float32') * rg['snap_dependency_score'].values
        ).astype('float32')
        rg['weekend_interaction'] = (
            rg['is_weekend'].values.astype('float32') * rg['weekend_intensity'].values
        ).astype('float32')
        rg['snap_x_income'] = (
            rg['snap_active'].values.astype('int8') * 3
            + rg['store_income_type'].values.astype('int8')
        ).astype('int8')
        # 新交差特徴量: sell_price × PSI, snap_active × PB_Ratio
        rg['price_x_psi'] = (
            rg['sell_price'].values.astype('float32') * rg['price_sensitivity_index'].values
        ).astype('float32')
        rg['snap_x_pb'] = (
            rg['snap_active'].values.astype('float32') * rg['pb_ratio'].values
        ).astype('float32')
        # cat_income_elasticity (store × cat)
        cie_keys = list(zip(
            rg['store_id'].astype(int).values,
            rg['cat_id'].astype(int).values,
        ))
        rg['cat_income_elasticity'] = (
            pd.Series(cie_keys).map(cie_lookup).fillna(1.0).astype('float32').values
        )
        del cie_keys
        # stockpiling_index (store-level)
        rg['stockpiling_index'] = sid_int.map(stockpile_idx).fillna(0.0).astype('float32').values
        # roll_mean_56_weighted: 弾力性が高い店×カテゴリほど平均を抑制
        if 'roll_mean_56' in rg.columns:
            cie_vals = rg['cat_income_elasticity'].values
            rg['roll_mean_56_weighted'] = (
                rg['roll_mean_56'].values / np.where(cie_vals == 0, 1.0, cie_vals)
            ).astype('float32')
        else:
            rg['roll_mean_56_weighted'] = np.float32(0)
        # income_event_sensitivity (store-level cluster)
        rg['income_event_sensitivity'] = sid_int.map(income_event_sens).fillna(0).astype('int8').values
        # spike_hint: 期待乖離率 × 当日イベントフラグ
        snap_vals = rg['snap_active'].values.astype('int8')
        is_we_vals = rg['is_weekend'].values.astype('int8') if 'is_weekend' in rg.columns else np.int8(0)
        day_vals = rg['day'].values if 'day' in rg.columns else np.int8(0)
        store_vals = rg['store_id'].astype(int).values
        cat_vals = rg['cat_id'].astype(int).values
        spike = np.zeros(len(rg), dtype='float32')
        for j in range(len(rg)):
            sid_j, cid_j = int(store_vals[j]), int(cat_vals[j])
            if int(snap_vals[j]) == 1:
                spike[j] = spike_expected.get((sid_j, cid_j, 0), 0.0)
            elif int(day_vals[j]) in PAYDAY_DAYS:
                spike[j] = spike_expected.get((sid_j, cid_j, 1), 0.0)
            elif int(is_we_vals[j]) == 1:
                spike[j] = spike_expected.get((sid_j, cid_j, 2), 0.0)
        rg['spike_hint'] = spike.astype('float32')
        # payday_flag フォールバック (旧 Phase 1 の parquet にない場合)
        if 'payday_flag' not in rg.columns:
            rg['payday_flag'] = rg['day'].isin(PAYDAY_EXACT).astype('int8')
        if 'payday_weekend' not in rg.columns:
            rg['payday_weekend'] = (
                rg['day'].isin(PAYDAY_DAYS) & (rg['is_weekend'] == 1)
            ).astype('int8')
        if 'snap_first_10d' not in rg.columns:
            rg['snap_first_10d'] = (
                (rg['snap_active'] == 1) & (rg['day'] <= 10)
            ).astype('int8')
        # cat_snap_sensitivity / cat_payday_sensitivity (store × cat)
        sc_keys = list(zip(store_vals, cat_vals))
        rg['cat_snap_sensitivity'] = (
            pd.Series(sc_keys).map(cat_snap_sens).fillna(1.0).astype('float32').values
        )
        rg['cat_payday_sensitivity'] = (
            pd.Series(sc_keys).map(cat_payday_sens).fillna(1.0).astype('float32').values
        )
        del sc_keys
        # snap_cat_lift: SNAP日 × カテゴリ別SNAP感度 (FOODS で高い)
        rg['snap_cat_lift'] = (
            rg['snap_active'].values.astype('float32') * rg['cat_snap_sensitivity'].values
        ).astype('float32')
        # payday_cat_lift: 給料日 × カテゴリ別給料日感度 (HOBBIES で高い)
        rg['payday_cat_lift'] = (
            rg['payday_flag'].values.astype('float32') * rg['cat_payday_sensitivity'].values
        ).astype('float32')
        # not_on_shelf フォールバック (旧 Phase 1 の parquet にない場合)
        if 'not_on_shelf' not in rg.columns:
            rg['not_on_shelf'] = rg['sell_price'].isna().astype('int8')
        # luxury_pressure: sell_price × payroll_dependency_score
        # 高価格 × 給料日依存度高 = 所得制約が強い店舗にとっての購入圧力
        _sp = rg['sell_price'].fillna(0).values.astype('float32')
        rg['luxury_pressure'] = (
            _sp * rg['payroll_dependency_score'].values
        ).astype('float32')
        # luxury_pressure × payday_flag: 給料日に高価格品の需要が集中する度合い
        rg['luxury_pressure_x_payday'] = (
            rg['luxury_pressure'].values * rg['payday_flag'].values.astype('float32')
        ).astype('float32')
        del _sp
        # weekday_density_ratio (store × dept, EDA Step 6b)
        wdr_keys = list(zip(
            rg['store_id'].astype(int).values,
            rg['dept_id'].astype(int).values,
        ))
        rg['weekday_density_ratio'] = (
            pd.Series(wdr_keys).map(wdr_lookup).fillna(1.0).astype('float32').values
        )
        del wdr_keys
        # store_dept_wday_avg (store × dept × wday → 日常平均)
        sdw_keys = list(zip(
            rg['store_id'].astype(int).values,
            rg['dept_id'].astype(int).values,
            rg['wday'].astype(int).values,
        ))
        rg['store_dept_wday_avg'] = (
            pd.Series(sdw_keys).map(sdw_lookup).fillna(0.0).astype('float32').values
        )
        del sdw_keys
        # store_dept_premium_share (store × dept → プレミアム品シェア)
        sdps_keys = list(zip(
            rg['store_id'].astype(int).values,
            rg['dept_id'].astype(int).values,
        ))
        rg['store_dept_premium_share'] = (
            pd.Series(sdps_keys).map(sdps_lookup).fillna(0.0).astype('float32').values
        )
        del sdps_keys
        # 旧列を削除 (存在する場合)
        for old_col in ['snap_uplift_store']:
            if old_col in rg.columns:
                rg.drop(columns=[old_col], inplace=True)
        table = pa.Table.from_pandas(rg, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(tmp_path), table.schema)
        writer.write_table(table)
        del rg, table
        print(f'    rg {i+1}/{n_rg}')
    del pf
    if writer:
        writer.close()

    # 置き換え
    tmp_path.replace(OUT_PATH)
    elapsed = time.time() - t0
    print(f'  完了 ({elapsed:.0f}s, {OUT_PATH.stat().st_size/1e6:.0f} MB)')


# ============================================================
# Phase 2: df_features.parquet → train/val/eval 分割
# ============================================================
def phase2_split() -> None:
    """parquet を row_group ごとに読み、train(memmap)/val(parquet)/eval(parquet) に分割。"""
    split_files = [TRAIN_X_PATH, TRAIN_Y_PATH, TRAIN_CAT_PATH, VAL_PATH, EVAL_PATH]

    # 古い split ファイルとの整合性チェック
    if VAL_PATH.exists():
        _val_cols = set(f.name for f in pq.ParquetFile(str(VAL_PATH)).schema_arrow)
        _expected = set(_get_features())
        if not _expected.issubset(_val_cols):
            _missing_feats = _expected - _val_cols
            print(f'  [STALE] split ファイルに不足列: {_missing_feats}')
            print('  → 古い split ファイルを削除して再生成します')
            for p in split_files:
                if p.exists():
                    p.unlink()
                    print(f'    deleted: {p.name}')

    if all(p.exists() for p in split_files):
        n_train = os.path.getsize(TRAIN_X_PATH) // (len(_get_features()) * 4)
        print(f'[Phase 2 SKIP] 分割済み (train: {n_train:,} rows)')
        return

    t0 = time.time()
    print('\n[Phase 2] parquet → train/val/eval 分割')

    pf = pq.ParquetFile(OUT_PATH)
    n_rg = pf.metadata.num_row_groups
    features = _get_features()
    target = 'sales'

    # Pass 1: train 行数カウント
    print('  [1/3] train 行数カウント...')
    has_not_on_shelf = 'not_on_shelf' in [f.name for f in pf.schema_arrow]
    count_cols = ['d_num', 'lag_56']
    if has_not_on_shelf:
        count_cols.append('not_on_shelf')
    n_train = 0
    for i in range(n_rg):
        rg = pf.read_row_group(i, columns=count_cols).to_pandas()
        _mask = (rg['d_num'] < VAL_START_DAY) & rg['lag_56'].notna()
        if has_not_on_shelf:
            _mask = _mask & (rg['not_on_shelf'] == 0)
        n_train += int(_mask.sum())
        del rg
    print(f'    train rows: {n_train:,}')

    # memmap 作成
    X_mm = np.memmap(str(TRAIN_X_PATH), dtype='float32', mode='w+',
                     shape=(n_train, len(features)))
    y_mm = np.memmap(str(TRAIN_Y_PATH), dtype='float32', mode='w+',
                     shape=(n_train,))
    cat_mm = np.memmap(str(TRAIN_CAT_PATH), dtype='int8', mode='w+',
                       shape=(n_train,))

    # Pass 2: 分割書き出し
    print('  [2/3] 分割中...')
    offset = 0
    val_writer: pq.ParquetWriter | None = None
    eval_writer: pq.ParquetWriter | None = None
    keep_cols_ve = ['id', 'cat_id', 'd_num', target] + features

    for i in range(n_rg):
        rg = pf.read_row_group(i).to_pandas()

        # Train → memmap (fillna 前にマスク計算: lag_56 NaN + not_on_shelf 行を除外)
        mask_t = (rg['d_num'] < VAL_START_DAY) & rg['lag_56'].notna()
        if has_not_on_shelf:
            mask_t = mask_t & (rg['not_on_shelf'] == 0)
        rg[features] = rg[features].fillna(0)
        tr = rg[mask_t]
        if len(tr) > 0:
            n = len(tr)
            X_mm[offset:offset+n] = tr[features].values.astype('float32')
            y_mm[offset:offset+n] = tr[target].values.astype('float32')
            cat_mm[offset:offset+n] = tr['cat_id'].values.astype('int8')
            offset += n

        # Val → parquet
        mask_v = (rg['d_num'] >= VAL_START_DAY) & (rg['d_num'] < EVAL_START_DAY)
        v = rg[mask_v]
        if len(v) > 0:
            tbl = pa.Table.from_pandas(v[keep_cols_ve], preserve_index=False)
            if val_writer is None:
                val_writer = pq.ParquetWriter(str(VAL_PATH), tbl.schema)
            val_writer.write_table(tbl)

        # Eval → parquet
        mask_e = rg['d_num'] >= EVAL_START_DAY
        e = rg[mask_e]
        if len(e) > 0:
            tbl = pa.Table.from_pandas(e[keep_cols_ve], preserve_index=False)
            if eval_writer is None:
                eval_writer = pq.ParquetWriter(str(EVAL_PATH), tbl.schema)
            eval_writer.write_table(tbl)

        del rg, tr, v, e
        print(f'    rg {i+1}/{n_rg}  train: {offset:,}')

    X_mm.flush(); y_mm.flush(); cat_mm.flush()
    if val_writer:  val_writer.close()
    if eval_writer: eval_writer.close()
    del X_mm, y_mm, cat_mm, pf

    n_val  = pq.ParquetFile(str(VAL_PATH)).metadata.num_rows
    n_eval = pq.ParquetFile(str(EVAL_PATH)).metadata.num_rows
    elapsed = time.time() - t0
    print(f'  [3/3] 完了 ({elapsed:.0f}s)')
    print(f'    train : {n_train:,} rows  ({os.path.getsize(TRAIN_X_PATH)/1e6:.0f} MB memmap)')
    print(f'    val   : {n_val:,} rows  ({os.path.getsize(VAL_PATH)/1e6:.0f} MB)')
    print(f'    eval  : {n_eval:,} rows  ({os.path.getsize(EVAL_PATH)/1e6:.0f} MB)')


def _get_features() -> list[str]:
    """parquet schema から特徴量列を取得。"""
    pf = pq.ParquetFile(OUT_PATH)
    all_cols = [f.name for f in pf.schema_arrow]
    del pf
    drop = set(META_COLS + ['d', 'date', 'sales', 'wm_yr_wk', 'd_num',
                            'snap_CA', 'snap_TX', 'snap_WI'])
    return [c for c in all_cols if c not in drop]


# ============================================================
# メイン
# ============================================================
def main(keep_from_day: int = 1100, chunk_size: int = 1000) -> None:
    for p in [CAL_PATH, SALES_PATH, PRICES_PATH]:
        if not p.exists():
            raise FileNotFoundError(f'{p} が見つかりません')

    print('=== M5 前処理 (ストリーミング版) ===')
    print(f'  keep_from_day : {keep_from_day}')
    print(f'  chunk_size    : {chunk_size}')
    print(f'  DATA_DIR      : {DATA_DIR}')

    phase1_features(keep_from_day, chunk_size)
    phase1_5_target_encoding()
    phase2_split()

    print('\n=== 出力ファイル ===')
    for p in [OUT_PATH, TRAIN_X_PATH, TRAIN_Y_PATH, TRAIN_CAT_PATH, VAL_PATH, EVAL_PATH]:
        if p.exists():
            print(f'  {p.name:<25} {os.path.getsize(p)/1e6:>8.0f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M5 前処理 (ストリーミング版)')
    parser.add_argument('--keep-from-day', type=int, default=1100)
    parser.add_argument('--chunk-size', type=int, default=1000)
    args = parser.parse_args()
    main(keep_from_day=args.keep_from_day, chunk_size=args.chunk_size)
