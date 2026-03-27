import pandas as pd, numpy as np, gc

print("Loading data...")
prices = pd.read_csv('sell_prices.csv')
calendar = pd.read_csv('calendar.csv', parse_dates=['date'])
d_cols = [c for c in pd.read_csv('sales_train_evaluation.csv', nrows=0).columns if c.startswith('d_')]
d_to_wk = dict(zip(calendar['d'], calendar['wm_yr_wk']))

price_lookup = {}
for _, r in prices.iterrows():
    price_lookup[(r['item_id'], r['store_id'], r['wm_yr_wk'])] = r['sell_price']
item_avg = prices.groupby('item_id')['sell_price'].mean().to_dict()

wk_dcols = {}
for d in d_cols:
    wk = d_to_wk.get(d)
    if wk:
        wk_dcols.setdefault(wk, []).append(d)
wk_list = sorted(wk_dcols.keys())

print("Streaming sales...")
agg = {}
n_items = 0
reader = pd.read_csv('sales_train_evaluation.csv', chunksize=2000)
for chunk in reader:
    for _, row in chunk.iterrows():
        item, store, dept = row['item_id'], row['store_id'], row['dept_id']
        avg_p = item_avg.get(item, 0)
        if avg_p <= 0:
            continue
        ptier = '$0-2' if avg_p < 2 else ('$2-5' if avg_p < 5 else ('$5-10' if avg_p < 10 else '$10+'))
        normal_daily = row[d_cols].values.astype(float).mean()
        if normal_daily < 0.01:
            continue

        prev_price = None
        weeks_since = 99
        discount_depth = 0
        for wk in wk_list:
            cols = wk_dcols.get(wk, [])
            if not cols:
                continue
            price = price_lookup.get((item, store, wk))
            if price is None or price <= 0:
                continue
            ws = sum(float(row[c]) for c in cols)
            if prev_price is not None and prev_price > 0:
                pct = (price - prev_price) / prev_price
                if pct < -0.03:
                    weeks_since = 0
                    discount_depth = -pct
                elif weeks_since < 99:
                    weeks_since += 1
            lift = ws / (normal_daily * len(cols)) if normal_daily > 0 else 1
            dbin = '0-5%' if discount_depth < 0.05 else ('5-15%' if discount_depth < 0.15 else ('15-30%' if discount_depth < 0.30 else '30%+'))
            wbin = 'w0' if weeks_since == 0 else ('w1' if weeks_since == 1 else ('w2-3' if weeks_since <= 3 else ('w4-8' if weeks_since <= 8 else 'norm')))
            k = (dbin, wbin, ptier, dept)
            if k not in agg:
                agg[k] = [0., 0]
            agg[k][0] += lift
            agg[k][1] += 1
            prev_price = price
        n_items += 1
    del chunk
    gc.collect()
    if n_items > 15000:
        break

print(f"Items: {n_items}")
dbins = ['0-5%', '5-15%', '15-30%', '30%+']
wbins = ['w0', 'w1', 'w2-3', 'w4-8', 'norm']

print(f"\n{'='*70}")
print("1. Price tier x Discount depth x Freshness")
print(f"{'='*70}")
for pt in ['$0-2', '$2-5', '$5-10', '$10+']:
    print(f"\n  {pt}:")
    print(f"  {'Depth':>8}", end='')
    for w in wbins:
        print(f" {w:>8}", end='')
    print()
    for db in dbins:
        print(f"  {db:>8}", end='')
        for wb in wbins:
            s = 0
            n = 0
            for dept in set(k[3] for k in agg):
                k2 = (db, wb, pt, dept)
                if k2 in agg:
                    s += agg[k2][0]
                    n += agg[k2][1]
            if n > 10:
                print(f" {s/n:>8.3f}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        print()

print(f"\n{'='*70}")
print("2. Freshness: w0 / w4-8")
print(f"{'='*70}")
for pt in ['$0-2', '$2-5', '$5-10', '$10+']:
    print(f"  {pt}:")
    for db in ['5-15%', '15-30%', '30%+']:
        w0s = 0; w0n = 0; w4s = 0; w4n = 0
        for dept in set(k[3] for k in agg):
            k0 = (db, 'w0', pt, dept)
            k4 = (db, 'w4-8', pt, dept)
            if k0 in agg:
                w0s += agg[k0][0]; w0n += agg[k0][1]
            if k4 in agg:
                w4s += agg[k4][0]; w4n += agg[k4][1]
        w0 = w0s / w0n if w0n > 10 else 0
        w4 = w4s / w4n if w4n > 10 else 0
        if w4 > 0:
            print(f"    {db:>8}: w0={w0:.3f} w4-8={w4:.3f} freshness={w0/w4:.3f}")
        else:
            print(f"    {db:>8}: N/A")

print(f"\n{'='*70}")
print("3. FOODS vs NON_FOODS")
print(f"{'='*70}")
for depts, name in [(['FOODS_1', 'FOODS_2', 'FOODS_3'], 'FOODS'),
                     (['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2'], 'NON_FOODS')]:
    print(f"\n  {name}:")
    print(f"  {'Depth':>8}", end='')
    for w in wbins:
        print(f" {w:>8}", end='')
    print()
    for db in dbins:
        print(f"  {db:>8}", end='')
        for wb in wbins:
            s = 0; n = 0
            for dept in depts:
                for pt in ['$0-2', '$2-5', '$5-10', '$10+']:
                    k2 = (db, wb, pt, dept)
                    if k2 in agg:
                        s += agg[k2][0]; n += agg[k2][1]
            if n > 10:
                print(f" {s/n:>8.3f}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        print()

print("\nDone")
