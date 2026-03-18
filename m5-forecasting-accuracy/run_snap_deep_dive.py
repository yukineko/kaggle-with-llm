"""SNAP Deep Dive — チャンク処理版 (メモリ1GB以内)"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

BASE_DIR = 'm5-forecasting-accuracy'
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
CHUNK_SIZE = 1000  # rows of wide-format sales

def run_snap_deep_dive():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading calendar & prices...")
    calendar = pd.read_csv(os.path.join(DATA_DIR, 'calendar.csv'))
    prices = pd.read_csv(os.path.join(DATA_DIR, 'sell_prices.csv'))

    d_cols = [c for c in pd.read_csv(os.path.join(DATA_DIR, 'sales_train_evaluation.csv'), nrows=0).columns
              if c.startswith('d_')]

    # SNAP flag per day per state: d -> {CA: 0/1, TX: 0/1, WI: 0/1}
    snap_ca = dict(zip(calendar['d'], calendar['snap_CA']))
    snap_tx = dict(zip(calendar['d'], calendar['snap_TX']))
    snap_wi = dict(zip(calendar['d'], calendar['snap_WI']))

    # === Accumulators ===
    # Store-level: store_id -> [snap_sum, snap_cnt, nosnap_sum, nosnap_cnt]
    store_snap_agg = {}
    # Store×cat: (store_id, cat_id) -> total_sales
    store_cat_agg = {}
    # Item-level (FOODS & HOUSEHOLD): (cat_id, item_id) -> [snap_sum, snap_cnt, nosnap_sum, nosnap_cnt]
    item_snap_agg = {}

    print("Streaming sales chunks...")
    reader = pd.read_csv(os.path.join(DATA_DIR, 'sales_train_evaluation.csv'), chunksize=CHUNK_SIZE)

    for chunk_idx, chunk in enumerate(reader):
        for _, row in chunk.iterrows():
            store_id = row['store_id']
            state_id = row['state_id']
            cat_id = row['cat_id']
            item_id = row['item_id']

            if state_id == 'CA':
                snap_lookup = snap_ca
            elif state_id == 'TX':
                snap_lookup = snap_tx
            else:
                snap_lookup = snap_wi

            total_sales = 0.0
            for d in d_cols:
                s = float(row[d])
                total_sales += s
                is_snap = snap_lookup.get(d, 0)

                # Store-level SNAP lift
                if store_id not in store_snap_agg:
                    store_snap_agg[store_id] = [0.0, 0, 0.0, 0]
                if is_snap:
                    store_snap_agg[store_id][0] += s
                    store_snap_agg[store_id][1] += 1
                else:
                    store_snap_agg[store_id][2] += s
                    store_snap_agg[store_id][3] += 1

                # Item-level SNAP lift (FOODS & HOUSEHOLD only)
                if cat_id in ['FOODS', 'HOUSEHOLD']:
                    key = (cat_id, item_id)
                    if key not in item_snap_agg:
                        item_snap_agg[key] = [0.0, 0, 0.0, 0]
                    if is_snap:
                        item_snap_agg[key][0] += s
                        item_snap_agg[key][1] += 1
                    else:
                        item_snap_agg[key][2] += s
                        item_snap_agg[key][3] += 1

            # Store×cat total
            key = (store_id, cat_id)
            store_cat_agg[key] = store_cat_agg.get(key, 0.0) + total_sales

        if (chunk_idx + 1) % 5 == 0:
            print(f"  chunk {chunk_idx + 1} ({(chunk_idx + 1) * CHUNK_SIZE} rows)")
        del chunk
        gc.collect()

    print(f"Done streaming. Stores: {len(store_snap_agg)}, Items (F/H): {len(item_snap_agg)}")

    # === Compute Store SNAP Lift ===
    store_rows = []
    for sid, (ss, sc, ns, nc) in store_snap_agg.items():
        snap_avg = ss / sc if sc > 0 else 0
        nosnap_avg = ns / nc if nc > 0 else 0
        lift = snap_avg / nosnap_avg if nosnap_avg > 0 else 1.0
        store_rows.append({'store_id': sid, 'lift': lift})
    store_lift = pd.DataFrame(store_rows).set_index('store_id')

    # === Compute HOBBIES Ratio ===
    store_total = {}
    store_hobbies = {}
    for (sid, cat), total in store_cat_agg.items():
        store_total[sid] = store_total.get(sid, 0.0) + total
        if cat == 'HOBBIES':
            store_hobbies[sid] = store_hobbies.get(sid, 0.0) + total
    hobbies_ratio = {sid: store_hobbies.get(sid, 0.0) / t for sid, t in store_total.items() if t > 0}
    store_lift['hobbies_ratio'] = store_lift.index.map(hobbies_ratio)
    store_stats = store_lift.dropna()

    # === Plot 1: Store SNAP Lift vs HOBBIES Ratio ===
    plt.figure(figsize=(10, 6))
    sns.regplot(data=store_stats, x='lift', y='hobbies_ratio')
    for sid in store_stats.index:
        plt.annotate(sid, (store_stats.loc[sid, 'lift'], store_stats.loc[sid, 'hobbies_ratio']),
                     fontsize=8)
    plt.title('Store Level: SNAP Lift vs HOBBIES Sales Ratio')
    plt.xlabel('SNAP Lift (Mean Sales on SNAP days / Non-SNAP days)')
    plt.ylabel('HOBBIES Sales Ratio')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, '32_store_snap_vs_hobbies.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, '32_store_snap_vs_hobbies.png')}")

    # === Compute Item SNAP Lift (FOODS & HOUSEHOLD) ===
    item_rows = []
    for (cat, iid), (ss, sc, ns, nc) in item_snap_agg.items():
        snap_avg = ss / sc if sc > 0 else 0
        nosnap_avg = ns / nc if nc > 0 else 0
        lift = (snap_avg + 1e-6) / (nosnap_avg + 1e-6)
        item_rows.append({'cat_id': cat, 'item_id': iid, 'lift': lift})
    item_lift = pd.DataFrame(item_rows)

    # Get median price per item
    item_prices = prices.groupby('item_id')['sell_price'].median().reset_index()
    item_stats = item_lift.merge(item_prices, on='item_id').dropna()

    # === Plot 2: FOODS SNAP Lift vs Price ===
    foods_stats = item_stats[item_stats['cat_id'] == 'FOODS']
    plt.figure(figsize=(10, 6))
    plt.scatter(foods_stats['sell_price'], foods_stats['lift'], alpha=0.3, s=10)
    plt.xscale('log')
    plt.title('FOODS Items: SNAP Lift vs Median Price')
    plt.xlabel('Median Sell Price (Log Scale)')
    plt.ylabel('SNAP Lift')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, '33_foods_snap_lift_vs_price.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, '33_foods_snap_lift_vs_price.png')}")

    # === Plot 3: HOUSEHOLD SNAP Lift vs Price ===
    hh_stats = item_stats[item_stats['cat_id'] == 'HOUSEHOLD']
    plt.figure(figsize=(10, 6))
    plt.scatter(hh_stats['sell_price'], hh_stats['lift'], alpha=0.3, s=10, color='orange')
    plt.xscale('log')
    plt.title('HOUSEHOLD Items: SNAP Lift vs Median Price')
    plt.xlabel('Median Sell Price (Log Scale)')
    plt.ylabel('SNAP Lift')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, '34_household_snap_lift_vs_price.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, '34_household_snap_lift_vs_price.png')}")

    # === Top SNAP Sensitive Items ===
    print("\nTop 20 SNAP-sensitive FOODS items:")
    print(foods_stats.sort_values('lift', ascending=False).head(20).to_string())

    print("\nTop 20 SNAP-sensitive HOUSEHOLD items:")
    print(hh_stats.sort_values('lift', ascending=False).head(20).to_string())

    # === Correlations ===
    corr = store_stats.corr().iloc[0, 1]
    print(f"\nCorrelation (Store SNAP Lift vs HOBBIES Ratio): {corr:.4f}")

    foods_corr = foods_stats.corr(numeric_only=True).loc['lift', 'sell_price']
    print(f"Correlation (FOODS Item SNAP Lift vs Price): {foods_corr:.4f}")

    hh_corr = hh_stats.corr(numeric_only=True).loc['lift', 'sell_price']
    print(f"Correlation (HOUSEHOLD Item SNAP Lift vs Price): {hh_corr:.4f}")

if __name__ == "__main__":
    run_snap_deep_dive()
