"""
EDA Step 7 一括実行スクリプト (チャンク読み版)
7a: イベント普遍性 vs 地域特異性
7b: ラマダン深層リフト分析
7c: 価格帯プロファイリング & 冗長性検証
"""
import os, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)
SALES_PATH = DATA_DIR / 'sales_train_evaluation.csv'
PRICES_PATH = DATA_DIR / 'sell_prices.csv'
CAL_PATH = DATA_DIR / 'calendar.csv'
CHUNK_SIZE = 500

t0 = time.time()

# ============================================================
# Phase 0: Calendar + Item prices
# ============================================================
print("=== Phase 0: Calendar & Price Data ===")
calendar = pd.read_csv(CAL_PATH, parse_dates=['date'])
header = pd.read_csv(SALES_PATH, nrows=0)
d_cols = [c for c in header.columns if c.startswith('d_')]
meta_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
print(f"  calendar: {len(calendar)} rows, d_cols: {d_cols[0]}~{d_cols[-1]} ({len(d_cols)} days)")

# Item average prices (chunked)
item_price_acc = {}
for chunk in pd.read_csv(PRICES_PATH, chunksize=500_000):
    for item, price in zip(chunk['item_id'], chunk['sell_price']):
        if item not in item_price_acc:
            item_price_acc[item] = [0.0, 0]
        item_price_acc[item][0] += price
        item_price_acc[item][1] += 1
item_avg = {k: v[0]/v[1] for k, v in item_price_acc.items() if v[1] > 0}
del item_price_acc
print(f"  Items with prices: {len(item_avg)}")

# Dept quartiles
dept_prices_map = defaultdict(list)
for item, price in item_avg.items():
    dept_prices_map['_'.join(item.split('_')[:2])].append(price)
dept_q = {d: np.percentile(p, [25, 50, 75]) for d, p in dept_prices_map.items()}
dept_stats = {}
for d, prices in dept_prices_map.items():
    arr = np.array(prices)
    dept_stats[d] = {'mean': arr.mean(), 'std': arr.std()}
del dept_prices_map

# Item → tier (0-3) and z-premium
def get_tier(item):
    dept = '_'.join(item.split('_')[:2])
    q = dept_q.get(dept, [0, 0, 999])
    p = item_avg.get(item, 0)
    if p >= q[2]: return 3
    if p >= q[1]: return 2
    if p >= q[0]: return 1
    return 0

item_tier = {item: get_tier(item) for item in item_avg}

# Z-score premium (z > 2.0)
item_zprem = {}
for item, price in item_avg.items():
    dept = '_'.join(item.split('_')[:2])
    st = dept_stats[dept]
    z = (price - st['mean']) / st['std'] if st['std'] > 0 else 0
    item_zprem[item] = int(z > 2.0)

# PB items (P20 or below)
dept_p20 = {d: np.percentile([item_avg[i] for i in item_avg
             if '_'.join(i.split('_')[:2]) == d], 20) for d in dept_q}
item_pb = {item: int(item_avg[item] <= dept_p20.get('_'.join(item.split('_')[:2]), 0))
           for item in item_avg}

print(f"  Premium (Z>2): {sum(item_zprem.values())} items")
print(f"  PB (P20): {sum(item_pb.values())} items")
for d in sorted(dept_stats.keys()):
    st = dept_stats[d]
    n_prem = sum(1 for i, v in item_zprem.items() if v == 1 and '_'.join(i.split('_')[:2]) == d)
    print(f"    {d}: mean=${st['mean']:.2f}, std=${st['std']:.2f}, P75=${dept_q[d][2]:.2f}, premium={n_prem}")

# ============================================================
# Phase 1: Accumulate from sales chunks
# ============================================================
print(f"\n=== Phase 1: Accumulate from sales chunks (chunksize={CHUNK_SIZE}) ===")

# Event setup
d_to_idx = {d: i for i, d in enumerate(d_cols)}
ev_d_set = set(calendar[calendar['event_name_1'].notna()]['d'])
non_ev_idx = np.array([d_to_idx[d] for d in d_cols if d not in ev_d_set])

stores_set = set()
n_days = len(d_cols)

# Accumulators
store_all = {}        # store → FOODS daily totals (n_days,)
store_p75 = {}        # store → FOODS P75+ daily totals
grp_daily = {}        # (store, dept, tier) → daily totals
prem_share_num = {}   # (store, dept) → premium sales total
prem_share_den = {}   # (store, dept) → total sales
pb_share_num = {}     # store → PB sales total
pb_share_den = {}     # store → total sales
anchor_candidates = defaultdict(lambda: np.zeros(n_days))  # item → daily sums (FOODS_3 only)

for i, chunk in enumerate(pd.read_csv(SALES_PATH, chunksize=CHUNK_SIZE)):
    for _, row in chunk.iterrows():
        s = row['store_id']
        dept = row['dept_id']
        cat = row['cat_id']
        item = row['item_id']
        stores_set.add(s)

        vals = row[d_cols].values.astype(float)
        t = item_tier.get(item, 0)
        zp = item_zprem.get(item, 0)
        pb = item_pb.get(item, 0)

        # grp_daily
        key = (s, dept, t)
        if key not in grp_daily:
            grp_daily[key] = np.zeros(n_days)
        grp_daily[key] += vals

        # FOODS totals
        if cat == 'FOODS':
            if s not in store_all:
                store_all[s] = np.zeros(n_days)
                store_p75[s] = np.zeros(n_days)
            store_all[s] += vals
            if t == 3:
                store_p75[s] += vals

        # Premium share (store × dept)
        sd = (s, dept)
        if sd not in prem_share_den:
            prem_share_num[sd] = 0.0
            prem_share_den[sd] = 0.0
        total = vals.sum()
        prem_share_den[sd] += total
        if zp == 1:
            prem_share_num[sd] += total

        # PB share (store-level)
        if s not in pb_share_den:
            pb_share_num[s] = 0.0
            pb_share_den[s] = 0.0
        pb_share_den[s] += total
        if pb == 1:
            pb_share_num[s] += total

        # Anchor candidates (FOODS_3, for luxury index)
        if dept == 'FOODS_3':
            anchor_candidates[item] += vals

    if (i + 1) % 10 == 0:
        print(f"  chunk {i+1} processed ({(i+1)*CHUNK_SIZE} items)")

stores = sorted(stores_set)
print(f"  Total chunks: {i+1}, stores: {stores}")

# Anchor items → luxury index
item_totals = {item: vals.sum() for item, vals in anchor_candidates.items()}
item_cv_vals = {}
for item, vals in anchor_candidates.items():
    m = vals.mean()
    std = vals.std()
    item_cv_vals[item] = std / (m + 1e-8)
low_cv_thresh = np.percentile(list(item_cv_vals.values()), 30)
low_cv_items = [item for item, cv in item_cv_vals.items() if cv <= low_cv_thresh]
anchors = sorted(low_cv_items, key=lambda x: item_totals[x], reverse=True)[:5]
print(f"  Anchor items: {anchors}")
del anchor_candidates

# Market score & luxury index
market_score = {}
for s in stores:
    ms = 0.0
    for a in anchors:
        key_search = [(s, 'FOODS_3', t) for t in range(4)]
        # anchor items are in FOODS_3, we need their sales per store
        # Since we accumulated by (store, dept, tier), we can't isolate single items
        # Use a simpler proxy: total FOODS_3 tier-0 and tier-1 sales
    # Fallback: use total FOODS sales as proxy for market size
    market_score[s] = store_all[s].sum() if s in store_all else 1.0

luxury_index = {}
for s in stores:
    high = store_p75[s].sum() if s in store_p75 else 0
    luxury_index[s] = high / market_score[s] if market_score[s] > 0 else 0

# Premium share
premium_share = {sd: (prem_share_num[sd] / prem_share_den[sd] * 100)
                 if prem_share_den[sd] > 0 else 0.0 for sd in prem_share_den}
# PB ratio
pb_ratio = {s: (pb_share_num[s] / pb_share_den[s] * 100)
            if pb_share_den[s] > 0 else 0.0 for s in stores}

elapsed = time.time() - t0
print(f"\n  Phase 1 done ({elapsed:.0f}s)")

# ============================================================
# Phase 2: Analysis 7a — Event Universality
# ============================================================
print("\n=== Step 7a: Event Universality vs Regional Specificity ===")

target_events = [
    'Christmas', 'Easter', 'Thanksgiving', 'SuperBowl', 'IndependenceDay',
    'Halloween', 'LaborDay', 'MemorialDay',
    'Ramadan starts', 'Eid al-Fitr', 'EidAlAdha',
    'OrthodoxEaster', 'OrthodoxChristmas', 'Pesach End', 'Chanukah End', 'LentStart',
]

rows_7a = []
for ev in target_events:
    ev_idx = np.array([d_to_idx[d] for d in
                       calendar[calendar['event_name_1'] == ev]['d'] if d in d_to_idx])
    if len(ev_idx) == 0:
        continue
    for s in stores:
        if s not in store_all:
            continue
        a, p = store_all[s], store_p75[s]
        base = a[non_ev_idx].mean()
        lift = (a[ev_idx].mean() / base - 1) * 100 if base > 0 else 0
        p75_norm = p[non_ev_idx].sum() / (a[non_ev_idx].sum() + 1e-8)
        p75_ev = p[ev_idx].sum() / (a[ev_idx].sum() + 1e-8)
        rows_7a.append({'event': ev, 'store': s, 'lift': lift,
                        'p75_shift': (p75_ev - p75_norm) * 100})

df_lift = pd.DataFrame(rows_7a)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(22, 16))

pv = df_lift.pivot(index='event', columns='store', values='lift')
sns.heatmap(pv, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=axes[0,0], linewidths=0.5)
axes[0,0].set_title('FOODS Total Lift % by Event × Store', fontsize=12)

pv2 = df_lift.pivot(index='event', columns='store', values='p75_shift')
sns.heatmap(pv2, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            ax=axes[0,1], linewidths=0.5)
axes[0,1].set_title('P75+ Share Shift (pp) on Event Days — FOODS', fontsize=12)

es = df_lift.groupby('event').agg(
    mean_lift=('lift','mean'), std_lift=('lift','std')).reset_index()
axes[1,0].scatter(es['mean_lift'], es['std_lift'], s=100, c='steelblue', edgecolors='k')
for _, r in es.iterrows():
    axes[1,0].annotate(r['event'], (r['mean_lift'], r['std_lift']),
                       fontsize=8, xytext=(5,5), textcoords='offset points')
axes[1,0].axhline(es['std_lift'].median(), color='r', ls='--', alpha=.5, label='median std')
axes[1,0].axvline(0, color='gray', ls='--', alpha=.5)
axes[1,0].set_xlabel('Mean Lift %'); axes[1,0].set_ylabel('Std Lift %')
axes[1,0].set_title('Universal (low std) vs Regional (high std)', fontsize=12)
axes[1,0].legend()

indf = df_lift[df_lift['lift'].abs() < 2].groupby('store').size().reindex(stores, fill_value=0)
axes[1,1].barh(indf.index, indf.values, color='salmon', edgecolor='k')
axes[1,1].set_xlabel('# events with |lift| < 2%')
axes[1,1].set_title('"Indifferent" Stores', fontsize=12)

plt.suptitle('Step 7a: Event Universality vs Regional Specificity', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / '26_event_universality.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 26_event_universality.png")

# Print results
med = es['std_lift'].median()
print("\n  Universal Events (std < median):")
for _, r in es[es['std_lift'] < med].sort_values('mean_lift', ascending=False).iterrows():
    print(f"    {r['event']:>20}: lift={r['mean_lift']:+.1f}%, std={r['std_lift']:.1f}")
print("\n  Regional Events (std >= median):")
for _, r in es[es['std_lift'] >= med].sort_values('std_lift', ascending=False).iterrows():
    print(f"    {r['event']:>20}: lift={r['mean_lift']:+.1f}%, std={r['std_lift']:.1f}")
print("\n  空振りイベント per store (|lift| < 2%):")
for s in stores:
    blanks = df_lift[(df_lift['store']==s) & (df_lift['lift'].abs() < 2)]['event'].tolist()
    if blanks:
        print(f"    {s}: {', '.join(blanks)}")

# ============================================================
# Phase 3: Analysis 7b — Ramadan Deep Lift
# ============================================================
print("\n=== Step 7b: Ramadan Deep Lift Analysis ===")

ram_starts = calendar[calendar['event_name_1'] == 'Ramadan starts'][['d','date','year']].copy()
eid_dates  = calendar[calendar['event_name_1'] == 'Eid al-Fitr'][['d','date','year']].copy()
ram_starts['d_num'] = ram_starts['d'].str[2:].astype(int)
eid_dates['d_num']  = eid_dates['d'].str[2:].astype(int)

ram_periods = []
for _, rs in ram_starts.iterrows():
    yr = rs['year']
    eid = eid_dates[eid_dates['year'] == yr]
    end_d = (eid.iloc[0]['d_num'] + 3) if len(eid) > 0 else (rs['d_num'] + 32)
    end_d = min(end_d, n_days)
    base_start = max(1, rs['d_num'] - 30)
    ram_periods.append({
        'year': yr, 'ram_start': rs['d_num'], 'ram_end': end_d,
        'base_start': base_start, 'base_end': rs['d_num'] - 1,
    })
    print(f"  {yr}: Ramadan d_{rs['d_num']}~d_{end_d}")

def d_range(start, end):
    return [i for i in range(start - 1, min(end, n_days))]

depts = sorted(set(d for _, d, _ in grp_daily.keys()))
tier_labels = {0: '<P25', 1: 'P25-50', 2: 'P50-75', 3: 'P75+'}

lift_rows_7b = []
for period in ram_periods:
    ram_idx  = d_range(period['ram_start'], period['ram_end'])
    base_idx = d_range(period['base_start'], period['base_end'])
    if not ram_idx or not base_idx:
        continue
    for s in stores:
        for dept in depts:
            for tier in range(4):
                key = (s, dept, tier)
                if key not in grp_daily:
                    continue
                d = grp_daily[key]
                base_avg = d[base_idx].mean()
                ram_avg  = d[ram_idx].mean()
                lift = (ram_avg / base_avg - 1) * 100 if base_avg > 1 else 0
                lift_rows_7b.append({'year': period['year'], 'store': s,
                                     'dept': dept, 'tier': tier, 'lift': lift})

df_ram = pd.DataFrame(lift_rows_7b)

# ramadan_sensitive score
score_rows = []
for s in stores:
    sd = df_ram[df_ram['store'] == s]
    if len(sd) == 0:
        continue
    foods_p75 = sd[(sd['dept'].str.startswith('FOODS')) & (sd['tier'] == 3)]['lift'].mean()
    hh_p75 = sd[(sd['dept'].str.startswith('HOUSEHOLD')) & (sd['tier'] == 3)]['lift'].mean()
    foods_all = sd[sd['dept'].str.startswith('FOODS')]['lift'].mean()
    overall = sd['lift'].mean()
    score = max(0, foods_p75) * 0.4 + max(0, hh_p75) * 0.3 + max(0, foods_all) * 0.3
    score_rows.append({'store': s, 'foods_p75': foods_p75, 'hh_p75': hh_p75,
                       'foods_all': foods_all, 'overall': overall, 'ramadan_sensitive': score})
df_score = pd.DataFrame(score_rows).set_index('store').sort_values('ramadan_sensitive', ascending=False)
print("\n  Ramadan Sensitivity Score:")
print(df_score.round(2).to_string())

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(22, 16))

avg_sd = df_ram.groupby(['store','dept'])['lift'].mean().reset_index()
pv1 = avg_sd.pivot(index='dept', columns='store', values='lift')
sns.heatmap(pv1, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=axes[0,0], linewidths=0.5)
axes[0,0].set_title('Ramadan Avg Lift % — Store × Dept', fontsize=12)

foods_ram = df_ram[df_ram['dept'].str.startswith('FOODS')]
avg_st = foods_ram.groupby(['store','tier'])['lift'].mean().reset_index()
avg_st['tier_label'] = avg_st['tier'].map(tier_labels)
pv2 = avg_st.pivot(index='store', columns='tier_label', values='lift')
pv2 = pv2[['<P25','P25-50','P50-75','P75+']]
sns.heatmap(pv2, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=axes[0,1], linewidths=0.5)
axes[0,1].set_title('FOODS Ramadan Lift % by Price Tier × Store', fontsize=12)

# Eid spike
top3 = df_score.head(3).index.tolist()
ax3 = axes[1,0]
for s in top3:
    if s not in store_all:
        continue
    daily = store_all[s]
    avg_window = np.zeros(15)
    n_years = 0
    for _, ed in eid_dates.iterrows():
        dn = ed['d_num']
        for offset in range(-7, 8):
            idx = dn - 1 + offset
            if 0 <= idx < n_days:
                avg_window[offset + 7] += daily[idx]
        n_years += 1
    avg_window /= max(n_years, 1)
    ax3.plot(range(-7, 8), avg_window, marker='o', ms=4, label=s)
ax3.axvline(0, color='red', ls='--', alpha=0.7, label='Eid al-Fitr')
ax3.set_xlabel('Days from Eid al-Fitr'); ax3.set_ylabel('Avg Daily FOODS Sales')
ax3.set_title('Eid al-Fitr Spike — FOODS (avg across years)', fontsize=12)
ax3.legend(fontsize=9)

colors = ['teal' if v > 5 else 'gray' for v in df_score['ramadan_sensitive']]
axes[1,1].barh(df_score.index, df_score['ramadan_sensitive'], color=colors, edgecolor='k')
axes[1,1].set_xlabel('Ramadan Sensitive Score')
axes[1,1].set_title('Ramadan Sensitivity Ranking', fontsize=12)
axes[1,1].axvline(5, color='red', ls='--', alpha=0.5, label='threshold=5')
axes[1,1].legend()

plt.suptitle('Step 7b: Ramadan Deep Lift Analysis', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / '27_ramadan_deep_lift.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 27_ramadan_deep_lift.png")

# FOODS_2 & HOUSEHOLD P75+ detail
print("\n  FOODS_2 / FOODS_3 / HOUSEHOLD P75+ Ramadan Lift:")
for dept_check in ['FOODS_2', 'FOODS_3', 'HOUSEHOLD_1', 'HOUSEHOLD_2']:
    sub = df_ram[(df_ram['dept'] == dept_check) & (df_ram['tier'] == 3)]
    if len(sub) == 0:
        continue
    store_avg = sub.groupby('store')['lift'].mean().sort_values(ascending=False)
    sig = store_avg[store_avg > 5]
    print(f"    {dept_check} P75+: {len(sig)} stores with >5% lift")
    for s, v in sig.items():
        print(f"      {s}: +{v:.1f}%")

# ============================================================
# Phase 4: Analysis 7c — Price Profiling & Redundancy
# ============================================================
print("\n=== Step 7c: Price Tier Profiling & Redundancy Check ===")

df_ps = pd.DataFrame([
    {'store': s, 'dept': d, 'premium_share': (prem_share_num[(s,d)] / prem_share_den[(s,d)] * 100)
     if prem_share_den.get((s,d), 0) > 0 else 0.0}
    for s in stores for d in depts if (s, d) in prem_share_den
])

store_avg_ps = df_ps.groupby('store')['premium_share'].mean()
lux_vals = pd.Series(luxury_index)
pb_vals = pd.Series(pb_ratio)

# Correlations
r_lux, p_lux = sp_stats.pearsonr(lux_vals[stores], store_avg_ps[stores])
r_pb, p_pb = sp_stats.pearsonr([pb_ratio[s] for s in stores], store_avg_ps[stores].values)
ev_lift_store = df_lift.groupby('store')['lift'].mean()
r_ev, p_ev = sp_stats.pearsonr(store_avg_ps[stores].values, ev_lift_store[stores].values)

print(f"\n  Redundancy Check:")
print(f"    premium_share vs luxury_index: r={r_lux:.3f} (p={p_lux:.3f})")
print(f"    premium_share vs pb_ratio:     r={r_pb:.3f} (p={p_pb:.3f})")
print(f"    premium_share vs event_lift:   r={r_ev:.3f} (p={p_ev:.3f})")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(22, 16))

pv = df_ps.pivot(index='dept', columns='store', values='premium_share')
sns.heatmap(pv, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,0], linewidths=0.5)
axes[0,0].set_title('store_dept_premium_share (%) — Z>2.0 items', fontsize=12)

axes[0,1].scatter(lux_vals[stores], store_avg_ps[stores], s=120, c='darkorange', edgecolors='k')
for s in stores:
    axes[0,1].annotate(s, (lux_vals[s], store_avg_ps[s]), fontsize=9, xytext=(5,5),
                       textcoords='offset points')
axes[0,1].set_xlabel('Luxury Index'); axes[0,1].set_ylabel('Avg Premium Share %')
axes[0,1].set_title(f'Premium Share vs Luxury Index (r={r_lux:.3f})', fontsize=12)

axes[1,0].scatter([pb_ratio[s] for s in stores], store_avg_ps[stores].values,
                   s=120, c='steelblue', edgecolors='k')
for s in stores:
    axes[1,0].annotate(s, (pb_ratio[s], store_avg_ps[s]), fontsize=9, xytext=(5,5),
                       textcoords='offset points')
axes[1,0].set_xlabel('PB Ratio (P20 share) %'); axes[1,0].set_ylabel('Premium Share %')
axes[1,0].set_title(f'Redundancy: Premium Share vs PB Ratio (r={r_pb:.3f})', fontsize=12)

axes[1,1].scatter(store_avg_ps[stores], ev_lift_store[stores], s=120, c='green', edgecolors='k')
for s in stores:
    axes[1,1].annotate(s, (store_avg_ps[s], ev_lift_store[s]), fontsize=9, xytext=(5,5),
                       textcoords='offset points')
axes[1,1].set_xlabel('Avg Premium Share %'); axes[1,1].set_ylabel('Avg Event Lift %')
axes[1,1].set_title(f'Premium Share vs Event Lift (r={r_ev:.3f})', fontsize=12)

plt.suptitle('Step 7c: Price Tier Profiling & Feature Redundancy', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / '28_price_profiling.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 28_price_profiling.png")

# Anomaly detection
lux_med = lux_vals.median()
ps_med = store_avg_ps.median()
print("\n  嗜好性シグナル (低所得 × 高 premium):")
for s in stores:
    if lux_vals[s] < lux_med and store_avg_ps[s] > ps_med:
        print(f"    {s}: luxury={lux_vals[s]:.4f} (< med) but premium={store_avg_ps[s]:.2f}% (> med)")

# Redundancy conclusion
print(f"\n  === 特徴量の結論 ===")
if abs(r_pb) > 0.8:
    print(f"  pb_ratio と高相関 (r={r_pb:.3f}) → store_dept_premium_share で pb_ratio を置換推奨")
elif abs(r_pb) < 0.5:
    print(f"  pb_ratio と低相関 (r={r_pb:.3f}) → store_dept_premium_share は独立情報 → 両方投入")
else:
    print(f"  pb_ratio と中相関 (r={r_pb:.3f}) → 両方残して CV で判断")

if abs(r_lux) > 0.8:
    print(f"  luxury_index と高相関 (r={r_lux:.3f}) → luxury_affinity_score との統合を検討")
else:
    print(f"  luxury_index と中〜低相関 (r={r_lux:.3f}) → premium_share は新情報を含む")

total_time = time.time() - t0
print(f"\n=== 全分析完了 ({total_time:.0f}s) ===")
print(f"  Figures: {FIG_DIR}/26_event_universality.png")
print(f"           {FIG_DIR}/27_ramadan_deep_lift.png")
print(f"           {FIG_DIR}/28_price_profiling.png")
