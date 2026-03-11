"""
M5 Forecasting Accuracy — Feature Engineering for LightGBM
==========================================================
EDA知見に基づく特徴量生成:
- Graph 11: First Sale 以前のデータ削除
- Graph 12: 0売上関連のラグ・ストリーク変数
- Graph 03: イベントフラグ (Christmas=0, SuperBowl/Easter=+)

メモリ効率最優先:
- wide→long変換を避け、numpy配列で処理
- dtype最適化 (int8/int16/float32)
- 不要列の即時削除
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path

DATA_DIR = Path(__file__).parent
TRAIN_DAYS = 1913       # d_1 ~ d_1913 (validation target: d_1914~d_1941)
EVAL_DAYS = 1941        # d_1 ~ d_1941 (evaluation target: d_1942~d_1969)
PRED_HORIZON = 28


# =====================================================================
# 1. データ読み込み (dtype最適化)
# =====================================================================
def load_data(use_evaluation=True):
    """メモリ効率を重視したデータ読み込み"""
    print("Loading calendar...")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])

    print("Loading sales...")
    fname = "sales_train_evaluation.csv" if use_evaluation else "sales_train_validation.csv"
    sales = pd.read_csv(DATA_DIR / fname)

    print("Loading sell_prices...")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

    # --- dtype最適化 ---
    # calendar
    for col in ["snap_CA", "snap_TX", "snap_WI", "wday", "month", "year"]:
        calendar[col] = calendar[col].astype(np.int8)
    calendar["wm_yr_wk"] = calendar["wm_yr_wk"].astype(np.int16)

    # prices
    prices["sell_price"] = prices["sell_price"].astype(np.float32)
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype(np.int16)

    # sales: d_* 列を int16 に
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    sales[d_cols] = sales[d_cols].astype(np.int16)

    # カテゴリカル列を category 型に
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for col in cat_cols:
        sales[col] = sales[col].astype("category")
    for col in ["store_id", "item_id"]:
        prices[col] = prices[col].astype("category")

    print(f"  calendar: {calendar.shape}, {calendar.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"  sales: {sales.shape}, {sales.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"  prices: {prices.shape}, {prices.memory_usage(deep=True).sum()/1e6:.1f} MB")

    return calendar, sales, prices


# =====================================================================
# 2. First Sale 検出 & Pre-launch 行削除 (Graph 11)
# =====================================================================
def compute_first_sale_day(sales, d_cols):
    """各商品の初回販売日インデックスを返す (0-indexed)"""
    mat = sales[d_cols].values  # (n_items, n_days)
    # argmax は最初の True を返す
    first_idx = np.argmax(mat > 0, axis=1)
    # 全部0の商品は argmax=0 になるので補正
    never_sold = mat.sum(axis=1) == 0
    first_idx[never_sold] = mat.shape[1]  # 全期間外
    return first_idx


# =====================================================================
# 3. Wide → Long 変換 (メモリ効率版)
# =====================================================================
def melt_sales(sales, calendar, d_cols, first_sale_idx, last_n_days=None):
    """
    Wide → Long 変換。First Sale 以前の行を除外。
    last_n_days: 直近N日分のみ変換 (メモリ節約)
    """
    n_items = len(sales)
    n_days = len(d_cols)

    if last_n_days is not None:
        start_day = max(0, n_days - last_n_days)
        d_cols_use = d_cols[start_day:]
    else:
        start_day = 0
        d_cols_use = d_cols

    # メタ情報
    meta = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()

    print(f"Melting {len(d_cols_use)} days × {n_items} items...")

    # チャンク処理で melt (一度に全部はメモリ不足の可能性)
    chunk_size = 200  # 200日ずつ
    frames = []

    for i in range(0, len(d_cols_use), chunk_size):
        cols = d_cols_use[i:i + chunk_size]
        chunk = sales[["id"] + list(cols)].melt(id_vars=["id"], var_name="d", value_name="sales")
        chunk["sales"] = chunk["sales"].astype(np.int16)
        frames.append(chunk)

    long = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()

    # メタ情報をマージ
    long = long.merge(meta, on="id", how="left")

    # d → day_num (d_1 → 1)
    long["day_num"] = long["d"].str[2:].astype(np.int16)

    # First Sale 以前を削除 (Graph 11)
    id_to_idx = dict(zip(sales["id"], first_sale_idx))
    long["first_sale_day"] = long["id"].map(id_to_idx).astype(np.int16)
    # day_num は 1-indexed, first_sale_idx は 0-indexed → +1
    n_before = len(long)
    long = long[long["day_num"] >= (long["first_sale_day"] + 1)].copy()
    print(f"  Pre-launch rows removed: {n_before - len(long):,} ({(n_before - len(long))/n_before*100:.1f}%)")

    # calendar マージ
    d_to_info = calendar[["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                           "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                           "snap_CA", "snap_TX", "snap_WI"]].copy()
    long = long.merge(d_to_info, on="d", how="left")

    # --- Graph 11: days_on_shelf (商品の成熟度) ---
    # first_sale_day からの経過日数 → 新商品 vs 定番商品の区別
    long["days_on_shelf"] = (long["day_num"] - (long["first_sale_day"] + 1)).astype(np.int16)

    # 不要列削除
    long.drop(columns=["first_sale_day"], inplace=True)

    print(f"  Final long shape: {long.shape}, {long.memory_usage(deep=True).sum()/1e6:.1f} MB")
    return long


# =====================================================================
# 4. カレンダー特徴量 + イベントフラグ (Graph 03)
# =====================================================================
def add_calendar_features(df):
    """イベント・カレンダー特徴量"""

    # --- イベントフラグ (Graph 03 の知見) ---
    # 売上激減イベント
    df["is_christmas"] = (df["event_name_1"] == "Christmas").astype(np.int8)
    df["is_thanksgiving"] = (df["event_name_1"] == "Thanksgiving").astype(np.int8)
    df["is_new_year"] = (df["event_name_1"] == "NewYear").astype(np.int8)

    # 売上増加イベント
    df["is_superbowl"] = (df["event_name_1"] == "SuperBowl").astype(np.int8)
    df["is_easter"] = (df["event_name_1"] == "Easter").astype(np.int8)
    df["is_labor_day"] = (df["event_name_1"] == "LaborDay").astype(np.int8)
    df["is_independence_day"] = (df["event_name_1"] == "IndependenceDay").astype(np.int8)
    df["is_memorial_day"] = (df["event_name_1"] == "MemorialDay").astype(np.int8)

    # イベントタイプ (one-hot)
    for etype in ["Cultural", "National", "Religious", "Sporting"]:
        col = f"event_type_{etype}"
        df[col] = ((df["event_type_1"] == etype) | (df["event_type_2"] == etype)).astype(np.int8)

    # イベント有無
    df["has_event"] = df["event_name_1"].notna().astype(np.int8)

    # --- SNAP (州別フードスタンプ) ---
    # 州に対応するSNAPフラグ
    df["snap"] = np.int8(0)
    for state in ["CA", "TX", "WI"]:
        mask = df["state_id"] == state
        df.loc[mask, "snap"] = df.loc[mask, f"snap_{state}"].astype(np.int8)

    # --- 曜日・週末 ---
    df["is_weekend"] = (df["wday"].isin([1, 2])).astype(np.int8)  # 1=Sat, 2=Sun

    # --- 月初/月末/給料日周辺 ---
    dom = df["date"].dt.day.astype(np.int8)
    df["is_month_start"] = (dom <= 3).astype(np.int8)
    df["is_month_end"] = (dom >= 28).astype(np.int8)

    # --- 年内の週番号 ---
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8)

    # --- 不要列削除 ---
    drop_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2",
                 "snap_CA", "snap_TX", "snap_WI", "weekday"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


# =====================================================================
# 5. ラグ特徴量 + ゼロストリーク (Graph 12)
# =====================================================================
def add_lag_features(df):
    """
    ラグ・移動統計量・ゼロストリーク特徴量。
    商品×店舗 (id) でグループ化して計算。
    day_numでソート済みを前提。
    """
    df.sort_values(["id", "day_num"], inplace=True)

    # --- 売上ラグ ---
    # 予測対象が28日先なので、lag >= 28 が安全 (リーク防止)
    lag_days = [28, 29, 30, 35, 42, 56, 91, 182, 364]
    for lag in lag_days:
        col = f"lag_{lag}"
        df[col] = df.groupby("id")["sales"].shift(lag).astype("float32")

    # --- 移動統計量 (lag=28 起点) ---
    windows = [7, 14, 28, 56, 91]
    grouped = df.groupby("id")["sales"]

    for w in windows:
        shifted = grouped.shift(28)
        roll = shifted.rolling(w, min_periods=1)

        df[f"rmean_{w}"] = roll.mean().astype(np.float32)
        df[f"rstd_{w}"] = roll.std().astype(np.float32)

        # max/min は間欠需要の把握に有効
        if w >= 28:
            df[f"rmax_{w}"] = roll.max().astype(np.float32)
            df[f"rmin_{w}"] = roll.min().astype(np.float32)

    del shifted, roll; gc.collect()

    # --- ゼロストリーク特徴量 (Graph 12) ---
    # 「今、何日連続で0が続いているか」を lag=28 時点で計算
    # CDFの50th=2日, 90th=7日, 99th=38日 → 重要な予測シグナル
    _compute_zero_streak_features(df)

    # --- 曜日別平均 (lag=28起点, 過去8週分) ---
    shifted_sales = df.groupby("id")["sales"].shift(28)
    df["_shifted_28"] = shifted_sales.astype(np.float32)
    df["dow_mean_8w"] = (
        df.groupby(["id", "wday"])["_shifted_28"]
        .transform(lambda x: x.rolling(8, min_periods=1).mean())
        .astype(np.float32)
    )
    df.drop(columns=["_shifted_28"], inplace=True)

    return df


def _compute_zero_streak_features(df):
    """
    各行について lag=28 時点での:
    - zero_streak: 連続0日数
    - days_since_last_sale: 最後に売れてからの日数
    - zero_ratio_28: 過去28日の0比率
    - zero_ratio_91: 過去91日の0比率
    """
    df.sort_values(["id", "day_num"], inplace=True)
    is_zero = (df["sales"] == 0).astype(np.int8)

    grouped = df.groupby("id")

    # --- zero_ratio: 過去N日間の0比率 ---
    shifted = grouped["sales"].shift(28)
    for w in [28, 91]:
        zr = (shifted == 0).rolling(w, min_periods=1).mean()
        df[f"zero_ratio_{w}"] = zr.astype(np.float32)
    del shifted; gc.collect()

    # --- zero_streak: lag=28時点での連続0日数 ---
    # ベクトル化: cumsum trick
    # 非ゼロでリセットされるカウンター
    sales_shifted = grouped["sales"].shift(28)  # lag=28
    is_nz = (sales_shifted > 0)

    # cumsum of non-zero → グループ境界
    nz_cumsum = is_nz.groupby(df["id"]).cumsum()
    # 各グループ内で0の連続カウント
    streak = nz_cumsum.groupby([df["id"], nz_cumsum]).cumcount()
    # 非ゼロの行はstreak=0ではなく、直前のゼロカウントが欲しいが
    # ここでは「直前の連続0日数」として使う
    df["zero_streak_28"] = np.where(sales_shifted == 0, streak + 1, 0).astype(np.int16)

    # --- days_since_last_sale ---
    # lag=28時点から遡って最後に売れた日からの日数
    last_sale_day = is_nz.groupby(df["id"]).cumsum()
    # 簡易版: zero_streak_28 で代替 (0でなければ0)
    df["days_since_sale_28"] = df["zero_streak_28"].copy()

    del sales_shifted, is_nz, nz_cumsum, streak, last_sale_day
    gc.collect()


# =====================================================================
# 6. 価格特徴量
# =====================================================================
def add_price_features(df, prices, calendar):
    """価格変動・相対価格特徴量 (Graph 07: 右歪み分布を考慮)"""
    # prices: store_id, item_id, wm_yr_wk, sell_price
    prices = prices.copy()

    # df と prices を wm_yr_wk + store_id + item_id でマージ
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["sell_price"] = df["sell_price"].astype(np.float32)

    # --- Graph 07: 価格が右歪み ($1-5に集中) → log変換 ---
    df["log_price"] = np.log1p(df["sell_price"]).astype(np.float32)

    # --- 価格特徴量 ---
    grouped = df.groupby("id")["sell_price"]

    # 価格の移動平均からの乖離 (セール検出)
    df["price_momentum"] = (df["sell_price"] / grouped.transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    ) - 1.0).astype(np.float32)

    # 価格変化フラグ
    df["price_change"] = grouped.diff().fillna(0).astype(np.float32)
    df["price_change_pct"] = grouped.pct_change().fillna(0).astype(np.float32)

    # 商品の最大/最小価格に対する相対位置
    price_max = grouped.transform("max")
    price_min = grouped.transform("min")
    price_range = price_max - price_min
    df["price_norm"] = np.where(
        price_range > 0,
        (df["sell_price"] - price_min) / price_range,
        0.5
    ).astype(np.float32)

    del price_max, price_min, price_range; gc.collect()

    # カテゴリ内相対価格 (Graph 07: FOODS=$2.68, HOBBIES=$3.97, HOUSEHOLD=$4.94)
    df["price_cat_ratio"] = (
        df["sell_price"] / df.groupby("cat_id")["sell_price"].transform("mean")
    ).astype(np.float32)

    # 店舗内相対価格
    df["price_store_ratio"] = (
        df["sell_price"] / df.groupby("store_id")["sell_price"].transform("mean")
    ).astype(np.float32)

    return df


# =====================================================================
# 7. 静的商品特徴量 (Graph 10, 11)
# =====================================================================
def add_item_static_features(df):
    """
    商品レベルの静的特徴量。
    - Graph 10: 商品ごとのゼロ売上比率 (mean=68%, 80-95%にピーク)
      → 間欠需要の度合いを表す重要な特徴量
    - Graph 11: 商品の存在期間
    """
    # --- Graph 10: 商品別ゼロ率 (lag=28で計算してリーク回避) ---
    shifted = df.groupby("id")["sales"].shift(28)
    df["item_zero_rate"] = (
        (shifted == 0)
        .groupby(df["id"])
        .transform("mean")
        .astype(np.float32)
    )
    del shifted; gc.collect()

    # --- Graph 10: カテゴリ別の相対ゼロ率 ---
    # HOUSEHOLD > HOBBIES > FOODS の順でゼロ率が高い
    cat_zero_mean = df.groupby("cat_id")["item_zero_rate"].transform("mean")
    df["item_zero_rate_vs_cat"] = (df["item_zero_rate"] - cat_zero_mean).astype(np.float32)
    del cat_zero_mean; gc.collect()

    # --- Graph 11: 商品の成熟度カテゴリ ---
    # days_on_shelf が短い商品は行動が異なる可能性
    df["is_new_item"] = (df["days_on_shelf"] < 90).astype(np.int8)

    return df


# =====================================================================
# 8. エンコーディング
# =====================================================================
def encode_categoricals(df):
    """カテゴリカル変数を LightGBM 用にエンコード"""
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype(np.int16)
    return df


# =====================================================================
# 9. メインパイプライン
# =====================================================================
def build_features(last_n_days=365 * 3, use_evaluation=True):
    """
    全特徴量を構築して返す。

    Parameters
    ----------
    last_n_days : int
        直近何日分を使うか (メモリ節約)。
        LightGBMでは直近2-3年で十分。
    use_evaluation : bool
        True: sales_train_evaluation (d_1~d_1941)
        False: sales_train_validation (d_1~d_1913)
    """
    calendar, sales, prices = load_data(use_evaluation)
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    # First Sale 検出
    print("\nComputing first sale days...")
    first_sale_idx = compute_first_sale_day(sales, d_cols)
    print(f"  d_1 から存在: {(first_sale_idx == 0).sum()}")
    print(f"  途中参入: {((first_sale_idx > 0) & (first_sale_idx < len(d_cols))).sum()}")

    # Wide → Long (pre-launch除外)
    print("\nMelting to long format...")
    df = melt_sales(sales, calendar, d_cols, first_sale_idx, last_n_days=last_n_days)
    del sales; gc.collect()

    # カレンダー + イベント特徴量
    print("\nAdding calendar & event features...")
    df = add_calendar_features(df)

    # ラグ + ゼロストリーク特徴量
    print("\nAdding lag & zero-streak features...")
    df = add_lag_features(df)
    gc.collect()

    # 価格特徴量
    print("\nAdding price features...")
    df = add_price_features(df, prices, calendar)
    del prices; gc.collect()

    # 静的商品特徴量 (Graph 10, 11)
    print("\nAdding item static features...")
    df = add_item_static_features(df)
    gc.collect()

    # カテゴリカルエンコーディング
    print("\nEncoding categoricals...")
    df = encode_categoricals(df)

    # --- 最終クリーンアップ ---
    drop_cols = ["d", "date", "id"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print(f"\nFinal dataset: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"Features: {[c for c in df.columns if c != 'sales']}")

    return df


# =====================================================================
# 特徴量リスト (LightGBM用)
# =====================================================================
FEATURE_COLS = [
    # カテゴリカル
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    # カレンダー
    "wday", "month", "year", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end",
    # イベント (Graph 03)
    "has_event",
    "is_christmas", "is_thanksgiving", "is_new_year",
    "is_superbowl", "is_easter", "is_labor_day",
    "is_independence_day", "is_memorial_day",
    "event_type_Cultural", "event_type_National",
    "event_type_Religious", "event_type_Sporting",
    "snap",
    # ラグ
    "lag_28", "lag_29", "lag_30", "lag_35", "lag_42",
    "lag_56", "lag_91", "lag_182", "lag_364",
    # 移動統計量
    "rmean_7", "rmean_14", "rmean_28", "rmean_56", "rmean_91",
    "rstd_7", "rstd_14", "rstd_28", "rstd_56", "rstd_91",
    "rmax_28", "rmin_28", "rmax_56", "rmin_56", "rmax_91", "rmin_91",
    # ゼロストリーク (Graph 12)
    "zero_ratio_28", "zero_ratio_91",
    "zero_streak_28", "days_since_sale_28",
    # 曜日別
    "dow_mean_8w",
    # 価格 (Graph 07)
    "sell_price", "log_price",
    "price_momentum", "price_change", "price_change_pct",
    "price_norm", "price_cat_ratio", "price_store_ratio",
    # 静的商品特徴量 (Graph 10, 11)
    "item_zero_rate", "item_zero_rate_vs_cat",
    "days_on_shelf", "is_new_item",
]

CATEGORICAL_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


if __name__ == "__main__":
    df = build_features(last_n_days=365 * 3)
    print(f"\n=== Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    print(f"\nNull counts (top 10):")
    nulls = df.isnull().sum().sort_values(ascending=False)
    print(nulls[nulls > 0].head(10))
    print(f"\nSample:")
    print(df.head())
