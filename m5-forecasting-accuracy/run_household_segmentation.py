"""
Step 13: 家計セグメンテーション EDA
3つの家計タイプ (SNAP依存 / Payroll依存 / 自立型) の購買行動を可視化する。

出力:
  figures/35_segment_scatter.png      — snap_lift × payday_lift 散布図
  figures/36_monthly_curve.png        — 月内売上カーブ (day 1-31) × セグメント × カテゴリ
  figures/37_payday_decay.png         — payday_flag 経過日数 × セグメント別売上推移
  figures/38_category_composition.png — セグメント別カテゴリ構成比
  figures/39_price_sensitivity.png    — 割引感度 × セグメント
  figures/40_depletion_effect.png     — 月末枯渇: 平均単価の月内推移
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    sns = None
import gc
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 11
DATA_DIR = 'm5-forecasting-accuracy'
FIG_DIR = f'{DATA_DIR}/figures/'

# ============================================================
# データ読み込み
# ============================================================
print("Loading data...")
calendar = pd.read_csv(f'{DATA_DIR}/calendar.csv', parse_dates=['date'])
prices = pd.read_csv(f'{DATA_DIR}/sell_prices.csv')
sales_header = pd.read_csv(f'{DATA_DIR}/sales_train_evaluation.csv', nrows=0)
d_cols = [c for c in sales_header.columns if c.startswith('d_')]

# calendar から day, payday_flag, snap を取得
cal_info = {}
PAYDAY_DAYS = {1, 2, 14, 15, 16}
PAYDAY_EXACT = {1, 15}
for _, r in calendar.iterrows():
    d = r['d']
    day_of_month = r['date'].day
    cal_info[d] = {
        'day': day_of_month,
        'month': r['date'].month,
        'wday': r['wday'],
        'payday': 1 if day_of_month in PAYDAY_EXACT else 0,
        'snap_CA': r['snap_CA'], 'snap_TX': r['snap_TX'], 'snap_WI': r['snap_WI'],
    }

# アイテム平均価格
item_avg_price = prices.groupby('item_id')['sell_price'].mean()

# ============================================================
# Phase 1: 店舗別 snap_lift / payday_lift を算出
# ============================================================
print("Computing store-level snap_lift / payday_lift...")
store_agg = {}  # store_id -> {snap_sum, snap_cnt, nosnap_sum, nosnap_cnt,
                #              pay_sum, pay_cnt, nopay_sum, nopay_cnt, total_sum}

reader = pd.read_csv(f'{DATA_DIR}/sales_train_evaluation.csv', chunksize=3000)
for chunk in reader:
    for _, row in chunk.iterrows():
        store = row['store_id']
        state = row['state_id']
        if store not in store_agg:
            store_agg[store] = {
                'snap_sum': 0, 'snap_cnt': 0, 'nosnap_sum': 0, 'nosnap_cnt': 0,
                'pay_sum': 0, 'pay_cnt': 0, 'nopay_sum': 0, 'nopay_cnt': 0,
            }
        sa = store_agg[store]
        snap_col = f'snap_{state}'
        for d in d_cols:
            s = float(row[d])
            ci = cal_info[d]
            # SNAP
            if ci[snap_col]:
                sa['snap_sum'] += s; sa['snap_cnt'] += 1
            else:
                sa['nosnap_sum'] += s; sa['nosnap_cnt'] += 1
            # Payday
            if ci['payday']:
                sa['pay_sum'] += s; sa['pay_cnt'] += 1
            else:
                sa['nopay_sum'] += s; sa['nopay_cnt'] += 1
    del chunk; gc.collect()

# Lift 算出
store_profiles = {}
for store, sa in store_agg.items():
    snap_avg = sa['snap_sum'] / sa['snap_cnt'] if sa['snap_cnt'] > 0 else 0
    nosnap_avg = sa['nosnap_sum'] / sa['nosnap_cnt'] if sa['nosnap_cnt'] > 0 else 0
    pay_avg = sa['pay_sum'] / sa['pay_cnt'] if sa['pay_cnt'] > 0 else 0
    nopay_avg = sa['nopay_sum'] / sa['nopay_cnt'] if sa['nopay_cnt'] > 0 else 0
    store_profiles[store] = {
        'snap_lift': snap_avg / nosnap_avg if nosnap_avg > 0 else 1.0,
        'payday_lift': pay_avg / nopay_avg if nopay_avg > 0 else 1.0,
    }
del store_agg; gc.collect()

# セグメント分類
# Type-S: snap_lift が中央値以上 かつ payday_lift が中央値未満
# Type-P: payday_lift が中央値以上 かつ snap_lift が中央値未満
# Type-B: それ以外 (均衡型)
snap_lifts = [v['snap_lift'] for v in store_profiles.values()]
pay_lifts = [v['payday_lift'] for v in store_profiles.values()]
snap_med = np.median(snap_lifts)
pay_med = np.median(pay_lifts)

store_segment = {}
for store, prof in store_profiles.items():
    sl, pl = prof['snap_lift'], prof['payday_lift']
    if sl >= snap_med and pl < pay_med:
        store_segment[store] = 'Type-S (SNAP)'
    elif pl >= pay_med and sl < snap_med:
        store_segment[store] = 'Type-P (Payroll)'
    else:
        store_segment[store] = 'Type-B (Balanced)'

print("\n=== Store Segments ===")
for store in sorted(store_profiles.keys()):
    p = store_profiles[store]
    print(f"  {store:6s}: snap_lift={p['snap_lift']:.4f}, payday_lift={p['payday_lift']:.4f} → {store_segment[store]}")

# ============================================================
# 図35: セグメント散布図
# ============================================================
print("\nPlotting segment scatter...")
fig, ax = plt.subplots(figsize=(10, 8))
colors = {'Type-S (SNAP)': '#e74c3c', 'Type-P (Payroll)': '#3498db', 'Type-B (Balanced)': '#2ecc71'}
for store, prof in store_profiles.items():
    seg = store_segment[store]
    ax.scatter(prof['snap_lift'], prof['payday_lift'], c=colors[seg], s=120, edgecolors='k', zorder=3)
    ax.annotate(store, (prof['snap_lift'], prof['payday_lift']),
                fontsize=9, ha='center', va='bottom', xytext=(0, 8), textcoords='offset points')
ax.axvline(snap_med, color='gray', ls='--', alpha=0.5, label=f'SNAP median={snap_med:.3f}')
ax.axhline(pay_med, color='gray', ls=':', alpha=0.5, label=f'Payday median={pay_med:.3f}')
for seg, color in colors.items():
    ax.scatter([], [], c=color, s=80, edgecolors='k', label=seg)
ax.set_xlabel('SNAP Lift (SNAP日 / 非SNAP日)')
ax.set_ylabel('Payday Lift (給料日 / 非給料日)')
ax.set_title('Step 13a: Store Segmentation — SNAP vs Payday Dependency')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR + '35_segment_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}35_segment_scatter.png")

# ============================================================
# Phase 2: day 別・セグメント別の売上集計 (チャンク読み)
# ============================================================
print("\nComputing daily sales by segment...")
# seg × cat × day → [sum, count]
seg_cat_day = {}
# seg × day → [price_sum, price_cnt]  (平均単価追跡)
seg_day_price = {}
# seg × discount_bin → [sum, count]
seg_disc = {}

reader = pd.read_csv(f'{DATA_DIR}/sales_train_evaluation.csv', chunksize=2000)
for chunk in reader:
    items = chunk['item_id'].unique()
    chunk_prices = prices[prices['item_id'].isin(items)]
    # item_id → avg price
    ip = chunk_prices.groupby('item_id')['sell_price'].mean().to_dict()
    # item_id → max price (for discount_ratio)
    ip_max = chunk_prices.groupby('item_id')['sell_price'].max().to_dict()

    for _, row in chunk.iterrows():
        store = row['store_id']
        cat = row['cat_id']
        item = row['item_id']
        seg = store_segment[store]
        avg_p = ip.get(item, 0)
        max_p = ip_max.get(item, 0)

        for d in d_cols:
            s = float(row[d])
            ci = cal_info[d]
            day = ci['day']

            # seg × cat × day
            key_scd = (seg, cat, day)
            if key_scd not in seg_cat_day:
                seg_cat_day[key_scd] = [0.0, 0]
            seg_cat_day[key_scd][0] += s
            seg_cat_day[key_scd][1] += 1

            # seg × day × price (売上加重)
            if s > 0 and avg_p > 0:
                key_sdp = (seg, day)
                if key_sdp not in seg_day_price:
                    seg_day_price[key_sdp] = [0.0, 0.0]
                seg_day_price[key_sdp][0] += avg_p * s  # 売上金額
                seg_day_price[key_sdp][1] += s           # 販売数

    del chunk, chunk_prices; gc.collect()

print(f"  seg_cat_day entries: {len(seg_cat_day)}")

# ============================================================
# 図36: 月内売上カーブ (day 1-31) × セグメント × カテゴリ
# ============================================================
print("\nPlotting monthly curves...")
cats = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
segs = ['Type-S (SNAP)', 'Type-P (Payroll)', 'Type-B (Balanced)']

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for idx, cat in enumerate(cats):
    ax = axes[idx]
    for seg, color in colors.items():
        days = range(1, 32)
        avgs = []
        for day in days:
            key = (seg, cat, day)
            if key in seg_cat_day and seg_cat_day[key][1] > 0:
                avgs.append(seg_cat_day[key][0] / seg_cat_day[key][1])
            else:
                avgs.append(0)
        ax.plot(days, avgs, color=color, lw=2, marker='o', ms=4, label=seg)
    # Mark payday (1, 15)
    for pd_day in [1, 15]:
        ax.axvline(pd_day, color='orange', ls='--', alpha=0.4)
    ax.set_xlabel('Day of Month')
    ax.set_ylabel('Average Sales')
    ax.set_title(f'{cat}: Monthly Sales Curve by Segment')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Step 13b: Monthly Sales Curve — Household Segments', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR + '36_monthly_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}36_monthly_curve.png")

# ============================================================
# 図37: payday_flag 経過日数 × セグメント別売上推移
# ============================================================
print("\nPlotting payday decay...")
# days_since_payday: 1日/15日を0として、次の給料日までの日数 (0-15)
def days_since_payday(day_of_month):
    """1日と15日を給料日として、最寄りの給料日からの経過日数を返す"""
    d1 = (day_of_month - 1) % 31   # 1日からの距離
    d15 = (day_of_month - 15) % 31 if day_of_month >= 15 else (day_of_month + 16)
    # 1日からの経過: day - 1 (1→0, 2→1, ..., 14→13)
    # 15日からの経過: day - 15 (15→0, 16→1, ..., 28→13, 29→14)
    if day_of_month <= 14:
        return day_of_month - 1
    elif day_of_month <= 28:
        return day_of_month - 15
    else:
        return day_of_month - 15  # 29-31: 月末は次の給料日(1日)に近い

# seg × days_since_payday → [sum, count] (全カテゴリ合算)
seg_dsp = {}
for (seg, cat, day), (s_sum, s_cnt) in seg_cat_day.items():
    dsp = days_since_payday(day)
    key = (seg, dsp)
    if key not in seg_dsp:
        seg_dsp[key] = [0.0, 0]
    seg_dsp[key][0] += s_sum
    seg_dsp[key][1] += s_cnt

# カテゴリ別にも
seg_cat_dsp = {}
for (seg, cat, day), (s_sum, s_cnt) in seg_cat_day.items():
    dsp = days_since_payday(day)
    key = (seg, cat, dsp)
    if key not in seg_cat_dsp:
        seg_cat_dsp[key] = [0.0, 0]
    seg_cat_dsp[key][0] += s_sum
    seg_cat_dsp[key][1] += s_cnt

fig, axes = plt.subplots(1, 4, figsize=(28, 6))

# All categories
ax = axes[0]
for seg, color in colors.items():
    dsps = range(0, 16)
    vals = []
    for dsp in dsps:
        key = (seg, dsp)
        if key in seg_dsp and seg_dsp[key][1] > 0:
            vals.append(seg_dsp[key][0] / seg_dsp[key][1])
        else:
            vals.append(0)
    # Normalize to day 0 = 1.0
    base = vals[0] if vals[0] > 0 else 1
    norm_vals = [v / base for v in vals]
    ax.plot(dsps, norm_vals, color=color, lw=2, marker='o', ms=5, label=seg)
ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
ax.set_xlabel('Days Since Payday (0 = payday)')
ax.set_ylabel('Relative Sales (payday = 1.0)')
ax.set_title('ALL Categories')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

for idx, cat in enumerate(cats):
    ax = axes[idx + 1]
    for seg, color in colors.items():
        dsps = range(0, 16)
        vals = []
        for dsp in dsps:
            key = (seg, cat, dsp)
            if key in seg_cat_dsp and seg_cat_dsp[key][1] > 0:
                vals.append(seg_cat_dsp[key][0] / seg_cat_dsp[key][1])
            else:
                vals.append(0)
        base = vals[0] if vals[0] > 0 else 1
        norm_vals = [v / base for v in vals]
        ax.plot(dsps, norm_vals, color=color, lw=2, marker='o', ms=5, label=seg)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('Days Since Payday')
    ax.set_ylabel('Relative Sales')
    ax.set_title(f'{cat}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Step 13b: Payday Decay — Sales Relative to Payday', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR + '37_payday_decay.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}37_payday_decay.png")

# ============================================================
# 図38: セグメント別カテゴリ構成比
# ============================================================
print("\nPlotting category composition...")
fig, ax = plt.subplots(figsize=(10, 6))
cat_colors = {'FOODS': '#2ecc71', 'HOBBIES': '#3498db', 'HOUSEHOLD': '#e74c3c'}
seg_cat_totals = {}
for (seg, cat, day), (s_sum, s_cnt) in seg_cat_day.items():
    key = (seg, cat)
    seg_cat_totals[key] = seg_cat_totals.get(key, 0) + s_sum

bar_data = {}
for seg in segs:
    total = sum(seg_cat_totals.get((seg, cat), 0) for cat in cats)
    bar_data[seg] = {cat: seg_cat_totals.get((seg, cat), 0) / total * 100 if total > 0 else 0 for cat in cats}

x = range(len(segs))
bottom = np.zeros(len(segs))
for cat in cats:
    vals = [bar_data[seg][cat] for seg in segs]
    ax.barh(x, vals, left=bottom, color=cat_colors[cat], edgecolor='k', alpha=0.8, label=cat)
    for i, v in enumerate(vals):
        if v > 5:
            ax.text(bottom[i] + v/2, i, f'{v:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
    bottom += vals

ax.set_yticks(x)
ax.set_yticklabels(segs)
ax.set_xlabel('Sales Composition (%)')
ax.set_title('Step 13c: Category Composition by Household Segment')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR + '38_category_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}38_category_composition.png")

# ============================================================
# 図39: 月末枯渇効果 — 平均購入単価の月内推移
# ============================================================
print("\nPlotting depletion effect...")
fig, ax = plt.subplots(figsize=(14, 6))
for seg, color in colors.items():
    days = range(1, 32)
    avg_prices = []
    for day in days:
        key = (seg, day)
        if key in seg_day_price and seg_day_price[key][1] > 0:
            avg_prices.append(seg_day_price[key][0] / seg_day_price[key][1])
        else:
            avg_prices.append(np.nan)
    ax.plot(days, avg_prices, color=color, lw=2, marker='o', ms=4, label=seg)

for pd_day in [1, 15]:
    ax.axvline(pd_day, color='orange', ls='--', alpha=0.4, label='Payday' if pd_day == 1 else '')
ax.set_xlabel('Day of Month')
ax.set_ylabel('Average Unit Price ($)')
ax.set_title('Step 13c: Average Purchase Price by Day — Depletion Effect')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR + '40_depletion_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}40_depletion_effect.png")

# ============================================================
# サマリ出力
# ============================================================
print("\n" + "=" * 60)
print("Step 13: Household Segmentation Summary")
print("=" * 60)

for seg in segs:
    stores = [s for s, sg in store_segment.items() if sg == seg]
    print(f"\n{seg}: {sorted(stores)}")
    for cat in cats:
        day0 = seg_cat_dsp.get((seg, cat, 0), [0, 1])
        day7 = seg_cat_dsp.get((seg, cat, 7), [0, 1])
        day13 = seg_cat_dsp.get((seg, cat, 13), [0, 1])
        avg0 = day0[0] / day0[1] if day0[1] > 0 else 0
        avg7 = day7[0] / day7[1] if day7[1] > 0 else 0
        avg13 = day13[0] / day13[1] if day13[1] > 0 else 0
        decay = (avg13 / avg0 - 1) * 100 if avg0 > 0 else 0
        print(f"  {cat:12s}: payday={avg0:.3f} → mid={avg7:.3f} → pre-payday={avg13:.3f} (decay={decay:+.1f}%)")

print("\n=== Done ===")
