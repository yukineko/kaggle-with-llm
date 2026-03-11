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
# Phase 1.5: Target Encoding + snap_uplift_store + store_income_type
# ============================================================
PAYDAY_DAYS = {14, 15, 16, 28, 29, 30, 31}  # 15日・末日周辺

def phase1_5_target_encoding() -> None:
    """df_features.parquet に以下の列を追加 (train 期間のみで算出):
    - te_store_dept_lag28: (store_id, dept_id, d_num-28) の平均売上
    - snap_uplift_store: store ごとの SNAP uplift 比率
    - store_income_type: 店舗クラスタ (0=Type-S, 1=Type-P, 2=Type-B)
    - snap_x_income: snap_active × store_income_type 交差特徴量
    """
    pf = pq.ParquetFile(OUT_PATH)
    existing_cols = [f.name for f in pf.schema_arrow]
    need_te = 'te_store_dept_lag28' not in existing_cols
    need_income = 'store_income_type' not in existing_cols
    if not need_te and not need_income:
        print(f'[Phase 1.5 SKIP] te_store_dept_lag28, store_income_type が既に存在')
        del pf
        return

    t0 = time.time()
    print('\n[Phase 1.5] Target Encoding + store_income_type (train期間のみ)')
    n_rg = pf.metadata.num_row_groups

    # Pass 1: train期間 (d_num < VAL_START_DAY) のみで集計
    print('  [1/2] 集計中 (d_num < %d のみ)...' % VAL_START_DAY)
    te_agg: dict[tuple, list] = {}          # (store_id, dept_id, d_num) → [sum, count]
    snap_agg: dict[int, list] = {}          # store_id → [snap_sum, snap_cnt, nosnap_sum, nosnap_cnt]
    payroll_agg: dict[int, list] = {}       # store_id → [payday_sum, payday_cnt, total_sum, total_cnt]
    read_cols = ['store_id', 'dept_id', 'd_num', 'sales', 'snap_active', 'day']
    for i in range(n_rg):
        rg = pf.read_row_group(i, columns=read_cols).to_pandas()
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
        # --- SNAP / Payroll 集計 ---
        for sid, snap, sales, day in zip(
            rg_tr['store_id'].values, rg_tr['snap_active'].values,
            rg_tr['sales'].values, rg_tr['day'].values,
        ):
            sid_int = int(sid)
            s = float(sales)
            # SNAP
            if sid_int not in snap_agg:
                snap_agg[sid_int] = [0.0, 0, 0.0, 0]
            if int(snap) == 1:
                snap_agg[sid_int][0] += s
                snap_agg[sid_int][1] += 1
            else:
                snap_agg[sid_int][2] += s
                snap_agg[sid_int][3] += 1
            # Payroll
            if sid_int not in payroll_agg:
                payroll_agg[sid_int] = [0.0, 0, 0.0, 0]
            payroll_agg[sid_int][2] += s
            payroll_agg[sid_int][3] += 1
            if int(day) in PAYDAY_DAYS:
                payroll_agg[sid_int][0] += s
                payroll_agg[sid_int][1] += 1
        del rg, rg_tr

    te_lookup = {k: v[0] / v[1] for k, v in te_agg.items()}
    del te_agg
    print(f'    TE lookup entries: {len(te_lookup):,}')

    # SNAP_Impact = snap_avg / nosnap_avg
    snap_impact: dict[int, float] = {}
    for sid, (s_sum, s_cnt, ns_sum, ns_cnt) in snap_agg.items():
        snap_avg = s_sum / s_cnt if s_cnt > 0 else 0.0
        nosnap_avg = ns_sum / ns_cnt if ns_cnt > 0 else 0.0
        snap_impact[sid] = (snap_avg / nosnap_avg) if nosnap_avg > 0 else 1.0
    del snap_agg

    # Payroll_Impact = payday_avg / overall_avg
    payroll_impact: dict[int, float] = {}
    for sid, (pd_sum, pd_cnt, t_sum, t_cnt) in payroll_agg.items():
        pd_avg = pd_sum / pd_cnt if pd_cnt > 0 else 0.0
        t_avg = t_sum / t_cnt if t_cnt > 0 else 0.0
        payroll_impact[sid] = (pd_avg / t_avg) if t_avg > 0 else 1.0
    del payroll_agg

    # --- クラスタリング: Type-S / Type-P / Type-B ---
    store_ids = sorted(snap_impact.keys())
    snap_vals = np.array([snap_impact[s] for s in store_ids])
    pay_vals = np.array([payroll_impact[s] for s in store_ids])
    # z-score 正規化
    snap_z = (snap_vals - snap_vals.mean()) / (snap_vals.std() + 1e-8)
    pay_z = (pay_vals - pay_vals.mean()) / (pay_vals.std() + 1e-8)

    # 0=Type-S (SNAP型), 1=Type-P (Payroll型), 2=Type-B (Balanced)
    INCOME_LABELS = {0: 'Type-S', 1: 'Type-P', 2: 'Type-B'}
    store_income: dict[int, int] = {}
    print(f'\n    Store Income Clustering:')
    print(f'    {"store":>6} {"SNAP_Impact":>12} {"Payroll_Impact":>15} {"snap_z":>7} {"pay_z":>7} {"type":>8}')
    for idx, sid in enumerate(store_ids):
        sz, pz = snap_z[idx], pay_z[idx]
        if sz > 0 and sz >= pz:
            t = 0  # Type-S
        elif pz > 0 and pz > sz:
            t = 1  # Type-P
        else:
            t = 2  # Type-B
        store_income[sid] = t
        print(f'    {sid:>6} {snap_vals[idx]:>12.4f} {pay_vals[idx]:>15.4f} '
              f'{sz:>7.2f} {pz:>7.2f} {INCOME_LABELS[t]:>8}')

    # Pass 2: 列を追加して新 parquet に書き出し
    print('\n  [2/2] 書き出し中...')
    tmp_path = OUT_PATH.with_suffix('.tmp.parquet')
    writer: pq.ParquetWriter | None = None
    for i in range(n_rg):
        rg = pf.read_row_group(i).to_pandas()
        # te_store_dept_lag28: lag-28 の日の (store_id, dept_id) 平均売上
        keys = list(zip(
            rg['store_id'].astype(int).values,
            rg['dept_id'].astype(int).values,
            (rg['d_num'].astype(int) - 28).values,
        ))
        rg['te_store_dept_lag28'] = pd.Series(keys).map(te_lookup).astype('float32').values
        del keys
        # snap_uplift_store: store ごとの SNAP uplift 比率
        rg['snap_uplift_store'] = (
            rg['store_id'].astype(int).map(snap_impact).astype('float32').values
        )
        # store_income_type: 店舗クラスタ (0=S, 1=P, 2=B)
        rg['store_income_type'] = (
            rg['store_id'].astype(int).map(store_income).astype('int8').values
        )
        # snap_x_income: snap_active × store_income_type 交差
        rg['snap_x_income'] = (
            rg['snap_active'].values.astype('int8') * 3
            + rg['store_income_type'].values.astype('int8')
        ).astype('int8')
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
    n_train = 0
    for i in range(n_rg):
        rg = pf.read_row_group(i, columns=['d_num', 'lag_56']).to_pandas()
        n_train += int(((rg['d_num'] < VAL_START_DAY) & rg['lag_56'].notna()).sum())
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

        # Train → memmap (fillna 前にマスク計算: lag_56 NaN 行を除外)
        mask_t = (rg['d_num'] < VAL_START_DAY) & rg['lag_56'].notna()
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
