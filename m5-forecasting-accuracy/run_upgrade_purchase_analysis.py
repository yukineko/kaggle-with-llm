"""
値引き時のアップグレード購買分析 (省メモリ・高速版)
====================================================
全処理をベクトル化。Python ループなし。
チャンクごとに weekly 集計して捨てる。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent
FIGURES_DIR = DATA_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

PRICES_PATH = DATA_DIR / 'sell_prices.csv'
SALES_PATH = DATA_DIR / 'sales_train_evaluation.csv'
CAL_PATH = DATA_DIR / 'calendar.csv'

KEEP_FROM_DAY = 1600
CHUNK_SIZE = 1000

print("=" * 60)
print("値引き × アップグレード購買 分析")
print("=" * 60)

# --- 1. カレンダー ---
cal = pd.read_csv(CAL_PATH, usecols=['d', 'wm_yr_wk'])
cal['d_num'] = cal['d'].str[2:].astype(int)
d_cols_all = [f'd_{i}' for i in range(KEEP_FROM_DAY, 1942)]
# d_col → wm_yr_wk mapping
dcol_wk = cal[cal['d_num'] >= KEEP_FROM_DAY].set_index('d')['wm_yr_wk'].to_dict()
# unique weeks in range
weeks_in_range = sorted(set(dcol_wk.values()))
# group d_cols by week
wk_to_dcols = {}
for d, wk in dcol_wk.items():
    wk_to_dcols.setdefault(wk, []).append(d)
print(f"  期間: d_{KEEP_FROM_DAY}~ ({len(dcol_wk)} days, {len(weeks_in_range)} weeks)")

# --- 2. 価格 → tercile + 値引き週 ---
print("\n[1/3] 価格処理...")

cat_map = {'FOODS': 0, 'HOBBIES': 1, 'HOUSEHOLD': 2}
meta = pd.read_csv(SALES_PATH, usecols=['item_id', 'dept_id', 'cat_id'], nrows=30490)
meta = meta.drop_duplicates('item_id')
meta['cat_num'] = meta['cat_id'].map(cat_map).astype('int8')

# チャンク読み → item 平均価格 & 週次価格
price_agg = []
for chunk in pd.read_csv(PRICES_PATH, chunksize=500_000):
    price_agg.append(
        chunk.groupby(['store_id', 'item_id', 'wm_yr_wk'])['sell_price']
        .first().reset_index()
    )
prices = pd.concat(price_agg, ignore_index=True)
del price_agg

# 値引き判定 (>5% drop)
prices = prices.sort_values(['store_id', 'item_id', 'wm_yr_wk'])
prices['prev_price'] = prices.groupby(['store_id', 'item_id'])['sell_price'].shift(1)
prices['is_discount'] = (
    ((prices['sell_price'] - prices['prev_price']) / prices['prev_price']) < -0.05
).fillna(False).astype('int8')
prices.drop(columns='prev_price', inplace=True)

# item 平均価格 → tercile
prices = prices.merge(meta[['item_id', 'dept_id', 'cat_num']], on='item_id', how='left')
item_avg = prices.groupby(['store_id', 'dept_id', 'item_id'])['sell_price'].mean().reset_index()
item_avg.rename(columns={'sell_price': 'avg_price'}, inplace=True)
item_avg['tercile'] = (
    item_avg.groupby(['store_id', 'dept_id'])['avg_price']
    .transform(lambda x: pd.qcut(x, 3, labels=[0, 1, 2], duplicates='drop'))
).astype('int8')

# discount lookup: (store, item, wk) → is_discount
disc_df = prices[prices['is_discount'] == 1][['store_id', 'item_id', 'wm_yr_wk']].copy()
disc_df['_disc'] = 1
print(f"  Discount events: {len(disc_df):,}")

# tercile lookup
terc_df = item_avg[['store_id', 'item_id', 'tercile']].copy()

# 高価格値引き dept-weeks
high_disc = disc_df.merge(terc_df, on=['store_id', 'item_id'], how='inner')
high_disc = high_disc[high_disc['tercile'] == 2]
high_disc = high_disc.merge(meta[['item_id', 'dept_id']], on='item_id', how='left')
high_disc_dw = high_disc[['store_id', 'dept_id', 'wm_yr_wk']].drop_duplicates()
high_disc_dw['_high_disc'] = 1
print(f"  High-tier discount dept-weeks: {len(high_disc_dw):,}")

# 不要な大きいDFを削除
del prices, item_avg, high_disc

# --- 3. 売上チャンク処理 (週単位ベクトル化) ---
print("\n[2/3] 売上集計...")

# 結果を DataFrame で蓄積 (小さい集計結果のみ)
lift_parts = []
cannib_parts = []
dept_high_parts = []

read_cols = ['item_id', 'store_id'] + [d for d in d_cols_all if d in dcol_wk]
reader = pd.read_csv(SALES_PATH, usecols=read_cols, chunksize=CHUNK_SIZE)

for ci, chunk in enumerate(reader):
    # meta merge
    chunk = chunk.merge(meta[['item_id', 'dept_id', 'cat_num']], on='item_id', how='left')
    chunk = chunk.merge(terc_df, on=['store_id', 'item_id'], how='left')
    chunk = chunk.dropna(subset=['tercile'])
    chunk['tercile'] = chunk['tercile'].astype('int8')
    if len(chunk) == 0:
        continue

    # 週単位で集計
    for wk in weeks_in_range:
        dcols_wk = wk_to_dcols.get(wk, [])
        dcols_present = [d for d in dcols_wk if d in chunk.columns]
        if not dcols_present:
            continue

        # 週の平均日販
        chunk['_wk_sales'] = chunk[dcols_present].mean(axis=1).astype('float32')

        # discount flag merge
        wk_disc = disc_df[disc_df['wm_yr_wk'] == wk][['store_id', 'item_id', '_disc']]
        merged = chunk.merge(wk_disc, on=['store_id', 'item_id'], how='left')
        merged['_disc'] = merged['_disc'].fillna(0).astype('int8')

        # === lift 集計: cat × tercile × is_disc ===
        g = merged.groupby(['cat_num', 'tercile', '_disc'])['_wk_sales'].agg(['sum', 'count'])
        g = g.reset_index()
        g['wk'] = wk
        lift_parts.append(g)

        # === dept high tier ===
        high = merged[merged['tercile'] == 2]
        if len(high) > 0:
            gh = high.groupby(['dept_id', '_disc'])['_wk_sales'].agg(['sum', 'count']).reset_index()
            dept_high_parts.append(gh)

        # === cannib: low tier + high_disc_week flag ===
        low = merged[merged['tercile'] == 0].copy()
        if len(low) > 0:
            hdw = high_disc_dw[high_disc_dw['wm_yr_wk'] == wk][['store_id', 'dept_id', '_high_disc']]
            low = low.merge(hdw, on=['store_id', 'dept_id'], how='left')
            low['_high_disc'] = low['_high_disc'].fillna(0).astype('int8')
            gc = low.groupby(['cat_num', '_high_disc'])['_wk_sales'].agg(['sum', 'count']).reset_index()
            cannib_parts.append(gc)

    # cleanup
    chunk.drop(columns=[c for c in chunk.columns if c.startswith('_')], inplace=True, errors='ignore')

    if (ci + 1) % 10 == 0:
        print(f"  {(ci+1)*CHUNK_SIZE:,} items...")

print(f"  完了")

# --- 4. 集約 & 出力 ---
print("\n[3/3] 結果...")

cat_names = {0: 'FOODS', 1: 'HOBBIES', 2: 'HOUSEHOLD'}
terc_names = {0: 'Low', 1: 'Mid', 2: 'High'}

# lift
lift_df = pd.concat(lift_parts, ignore_index=True)
lift_total = lift_df.groupby(['cat_num', 'tercile', '_disc'])[['sum', 'count']].sum().reset_index()
lift_total['avg'] = lift_total['sum'] / lift_total['count'].clip(lower=1)

print("\n=== 分析1: 値引き時の売上リフト (price tier × カテゴリ) ===")
for cat in [0, 1, 2]:
    print(f"\n  {cat_names[cat]}:")
    for t in [0, 1, 2]:
        n_row = lift_total[(lift_total['cat_num']==cat) & (lift_total['tercile']==t) & (lift_total['_disc']==0)]
        d_row = lift_total[(lift_total['cat_num']==cat) & (lift_total['tercile']==t) & (lift_total['_disc']==1)]
        avg_n = float(n_row['avg'].iloc[0]) if len(n_row) else 0
        avg_d = float(d_row['avg'].iloc[0]) if len(d_row) else 0
        cnt_n = int(n_row['count'].iloc[0]) if len(n_row) else 0
        cnt_d = int(d_row['count'].iloc[0]) if len(d_row) else 0
        lift = (avg_d / max(avg_n, 0.001) - 1) * 100
        print(f"    {terc_names[t]:>4s}: normal={avg_n:.3f} (n={cnt_n:,}), "
              f"discount={avg_d:.3f} (n={cnt_d:,}), lift={lift:+.1f}%")

# cannibalization
cannib_df = pd.concat(cannib_parts, ignore_index=True)
cannib_total = cannib_df.groupby(['cat_num', '_high_disc'])[['sum', 'count']].sum().reset_index()
cannib_total['avg'] = cannib_total['sum'] / cannib_total['count'].clip(lower=1)

print("\n=== 分析2: カニバリゼーション (高価格値引き → 低価格の売上変化) ===")
for cat in [0, 1, 2]:
    no = cannib_total[(cannib_total['cat_num']==cat) & (cannib_total['_high_disc']==0)]
    hi = cannib_total[(cannib_total['cat_num']==cat) & (cannib_total['_high_disc']==1)]
    avg_no = float(no['avg'].iloc[0]) if len(no) else 0
    avg_hi = float(hi['avg'].iloc[0]) if len(hi) else 0
    chg = (avg_hi / max(avg_no, 0.001) - 1) * 100
    print(f"  {cat_names[cat]:>10s}: normal={avg_no:.3f}, high_disc_week={avg_hi:.3f}, change={chg:+.1f}%")

# dept high tier
dept_df = pd.concat(dept_high_parts, ignore_index=True)
dept_total = dept_df.groupby(['dept_id', '_disc'])[['sum', 'count']].sum().reset_index()
dept_total['avg'] = dept_total['sum'] / dept_total['count'].clip(lower=1)

print("\n=== 分析3: dept別 High tier 値引きリフト ===")
dp = dept_total.pivot(index='dept_id', columns='_disc', values='avg').fillna(0)
dp.columns = ['normal', 'discount']
dp['lift_pct'] = (dp['discount'] / dp['normal'].clip(0.001) - 1) * 100
print(dp.sort_values('lift_pct', ascending=False).to_string())

# --- 可視化 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, cat in enumerate([0, 1, 2]):
    ax = axes[idx]
    normals, discounts = [], []
    for t in [0, 1, 2]:
        n_row = lift_total[(lift_total['cat_num']==cat) & (lift_total['tercile']==t) & (lift_total['_disc']==0)]
        d_row = lift_total[(lift_total['cat_num']==cat) & (lift_total['tercile']==t) & (lift_total['_disc']==1)]
        normals.append(float(n_row['avg'].iloc[0]) if len(n_row) else 0)
        discounts.append(float(d_row['avg'].iloc[0]) if len(d_row) else 0)
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, normals, w, label='Normal', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, discounts, w, label='Discount', color='#FF5722', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(['Low', 'Mid', 'High'])
    ax.set_xlabel('Price Tier'); ax.set_ylabel('Mean Daily Sales')
    ax.set_title(cat_names[cat]); ax.legend()
    for i in range(3):
        if normals[i] > 0:
            lift = (discounts[i] / normals[i] - 1) * 100
            ax.annotate(f'{lift:+.0f}%', xy=(i + w/2, discounts[i]),
                       ha='center', va='bottom', fontsize=10, fontweight='bold', color='#FF5722')
plt.suptitle('値引き時の売上リフト: Price Tier別\n(High tierのリフトが大 = アップグレード購買)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'upgrade_purchase_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIGURES_DIR / 'upgrade_purchase_analysis.png'}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, cat in enumerate([0, 1, 2]):
    ax = axes[idx]
    no = cannib_total[(cannib_total['cat_num']==cat) & (cannib_total['_high_disc']==0)]
    hi = cannib_total[(cannib_total['cat_num']==cat) & (cannib_total['_high_disc']==1)]
    vals = [float(no['avg'].iloc[0]) if len(no) else 0, float(hi['avg'].iloc[0]) if len(hi) else 0]
    chg = (vals[1] / max(vals[0], 0.001) - 1) * 100
    ax.bar(['Normal\nWeek', 'High-tier\nDiscount Week'], vals, color=['#2196F3', '#FF5722'], alpha=0.8)
    ax.set_title(f'{cat_names[cat]}\nLow-tier change: {chg:+.1f}%')
    ax.set_ylabel('Mean Daily Sales (Low tier)')
plt.suptitle('カニバリゼーション: 高価格帯の値引き週に低価格帯はどうなる？',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cannibalization_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURES_DIR / 'cannibalization_analysis.png'}")

print("\n完了")
