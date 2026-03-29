#!/usr/bin/env python3
"""
create_cv_folds.py

Input:
 - metadata CSV (typically series-level rows). Required columns:
   patient_id, study_id, series_id, labeled (0/1), num_slices,
   slice_thickness (float), vendor (str), contrast (0/1), label_voxel_count (int, optional)

Output:
 - folds.csv: patient_id, fold, aggregated info
 - stats_report.txt: basic per-fold distribution statistics

Note: greedy assignment + local swaps tries to balance 'strat_key' distribution,
with priority on patients, then series, then image workload across folds.
"""
import argparse
import random
import pandas as pd
import numpy as np
from collections import Counter

RANDOM_SEED = 42

def bucket_thickness(x):
    if pd.isna(x):
        return "unk"
    x = float(x)
    if x <= 1.0:
        return "<=1.0"
    if x <= 2.0:
        return "1.0-2.0"
    return ">2.0"

def make_patient_table(df):
    """
    Build a patient-level table with stable column names used by the splitter.

    This table keeps both legacy labeled fields and workload fields.
    Fold balancing itself prioritizes patient count, series count, and image count.
    """
    import pandas as pd

    if 'labeled' not in df.columns:
        df = df.copy()
        df['labeled'] = 1
    if 'vendor' not in df.columns:
        df = df.copy()
        df['vendor'] = 'unk'
    if 'slice_thickness' not in df.columns:
        df = df.copy()
        df['slice_thickness'] = np.nan

    # Per-series aggregation: determine whether each series has any labels.
    series_agg = df.groupby(['patient_id', 'series_id'], dropna=False).agg(
        series_labeled=('labeled', 'max'),
        series_num_images=('num_slices', 'count'),
        series_slice_thickness=('slice_thickness', 'median'),
        series_vendor=('vendor', lambda x: x.mode().iloc[0] if len(x.mode())>0 else "unk")
    ).reset_index()

    # Per-patient aggregation.
    ag_series = series_agg.groupby('patient_id').agg(
        n_series=('series_id', 'nunique'),
        n_labeled_series=('series_labeled', 'sum'),
        # Median of per-series thickness as patient-level thickness.
        median_thickness=('series_slice_thickness', 'median'),
        vendor_mode=('series_vendor', lambda x: x.mode().iloc[0] if len(x.mode())>0 else "unk")
    ).reset_index()

    # Totals and other patient-level fields from the original table.
    ag_images = df.groupby('patient_id').agg(
        n_studies=('study_id','nunique'),
        total_images=('num_slices','sum'),
        labeled_any_from_images=('labeled','max'),
        total_label_voxels=('label_voxel_count', 'sum') if 'label_voxel_count' in df.columns else pd.NamedAgg(column='patient_id', aggfunc=lambda x: 0)
    ).reset_index()

    # Merge patient-level views.
    ag = pd.merge(ag_images, ag_series, on='patient_id', how='outer').fillna(0)

    # Standardize column names expected downstream.
    ag = ag.rename(columns={
        'labeled_any_from_images': 'labeled_any',
        'total_label_voxels': 'total_label_voxels',
        'n_series': 'n_series',
        'n_labeled_series': 'n_labeled_series',
        'total_images': 'total_images',
        'median_thickness': 'median_thickness',
        'vendor_mode': 'vendor_mode'
    })

    # Optional pathology columns mapping (safe extraction/sum).
    def _sum_col_or_zero(col):
        if col in df.columns:
            s = df.groupby('patient_id')[col].sum()
            return ag['patient_id'].map(s).fillna(0).astype(float)
        return pd.Series(0.0, index=ag.index)

    ag['total_pathology_size'] = _sum_col_or_zero('pathology_size') if 'pathology_size' in df.columns else _sum_col_or_zero('pathology_size_mm3')
    ag['total_num_pathologies'] = _sum_col_or_zero('num_pathologies')

    # Ensure types.
    ag['n_series'] = ag['n_series'].astype(int)
    ag['n_labeled_series'] = ag['n_labeled_series'].astype(int)
    ag['labeled_any'] = ag['labeled_any'].astype(int)

    # Build thickness bucket and stratification key.
    ag['median_thickness'] = ag['median_thickness'].replace({0: pd.NA}).astype(float)
    ag['thickness_bucket'] = ag['median_thickness'].apply(bucket_thickness)
    ag['strat_key'] = ag['thickness_bucket'].astype(str) + "_" + ag['vendor_mode'].astype(str)

    # Return expected columns (plus any extra derived fields).
    return ag

def _objective(folds, avg_images, avg_series, weights):
    obj = 0.0
    for f in folds:
        di = (f['total_images'] - avg_images) / (avg_images + 1e-12)
        ds = (f['total_series'] - avg_series) / (avg_series + 1e-12)
        obj += weights.get('images', 1.0) * (di * di) + weights.get('series', 1.0) * (ds * ds)
        # Small stratification smoothness penalty.
        if f['counts']:
            vals = np.array(list(f['counts'].values()), dtype=float)
            obj += 0.01 * float(((vals - vals.mean())**2).sum())

        # Penalize patient count imbalance explicitly.
        target_patients = f.get('target_patients', None)
        if target_patients is not None:
            dp = (len(f['patients']) - target_patients) / (target_patients + 1e-12)
            obj += weights.get('patients', 4.0) * (dp * dp)

    return float(obj)


def _target_patient_counts(n_patients, n_folds):
    """Return near-uniform per-fold patient targets (sum equals n_patients)."""
    base = n_patients // n_folds
    rem = n_patients % n_folds
    return [base + 1 if i < rem else base for i in range(n_folds)]


def greedy_assign_folds(pat_df, n_folds, strat_col='strat_key', seed=RANDOM_SEED, weights=None, swap_iters=2000):
    """
    Greedy assignment + local swap optimization with tunable weights for:
      - slices (total_slices) -- backed by total_images/num_slices if needed
      - labeled_patients (count of patients with any label)
      - labeled_series (sum of labeled series per fold)
      - size (total_pathology_size)
      - num (total_num_pathologies)
    weights: dict keys: slices, labeled_patients, labeled_series, size, num
    """
    # Ensure expected columns exist (compatibility with per-image or per-series metadata).
    if 'total_slices' not in pat_df.columns:
        if 'total_images' in pat_df.columns:
            pat_df['total_slices'] = pat_df['total_images']
        elif 'num_slices' in pat_df.columns:
            pat_df['total_slices'] = pat_df['num_slices']
        else:
            pat_df['total_slices'] = 0

    if 'n_series' not in pat_df.columns:
        pat_df['n_series'] = 1

    random.seed(seed); np.random.seed(seed)
    if weights is None:
        weights = {
            'patients': 8.0,
            'series': 3.0,
            'images': 2.0,
        }

    patients = pat_df.to_dict('records')
    # Hard cases first: high image and series workload.
    patients.sort(
        key=lambda r: (
            float(r.get('total_slices', 0)),
            float(r.get('n_series', 0)),
        ),
        reverse=True,
    )

    if len(patients) < n_folds:
        raise ValueError(f"n_folds={n_folds} is larger than number of patients={len(patients)}")

    total_images = float(pat_df['total_slices'].sum())
    total_series = float(pat_df['n_series'].sum())

    avg_images = float(total_images) / n_folds if n_folds else 0.0
    avg_series = float(total_series) / n_folds if n_folds else 0.0

    target_counts = _target_patient_counts(len(patients), n_folds)

    folds = []
    for i in range(n_folds):
        folds.append({
            'patients': [],
            'target_patients': target_counts[i],
            'total_images': 0.0,
            'total_series': 0.0,
            'counts': Counter(),
        })

    # Constrained greedy assignment: never exceed target patient count in any fold.
    for p in patients:
        best_fold = None; best_score = None
        for i, f in enumerate(folds):
            if len(f['patients']) >= f['target_patients']:
                continue

            future_images = f['total_images'] + float(p.get('total_slices', 0))
            future_series = f['total_series'] + float(p.get('n_series', 0))

            di = abs(future_images - avg_images) / (avg_images + 1e-12)
            ds = abs(future_series - avg_series) / (avg_series + 1e-12)

            strat_pen = (f['counts'][p.get(strat_col)] + 1)

            # Soft preference to fill folds toward target count in a controlled way.
            future_count = len(f['patients']) + 1
            count_util = future_count / (f['target_patients'] + 1e-12)

            score = (
                weights.get('images', 1.0) * di
                + weights.get('series', 1.0) * ds
                + 0.1 * strat_pen
                + 0.1 * count_util
            )

            if best_score is None or score < best_score:
                best_score = score; best_fold = i

        if best_fold is None:
            raise RuntimeError("No feasible fold found during constrained assignment")

        # Assign to best feasible fold.
        folds[best_fold]['patients'].append(p['patient_id'])
        folds[best_fold]['total_images'] += float(p.get('total_slices', 0))
        folds[best_fold]['total_series'] += float(p.get('n_series', 0))
        folds[best_fold]['counts'][p.get(strat_col)] += 1

    mapping = {}
    for i,f in enumerate(folds):
        for pid in f['patients']:
            mapping[pid] = i

    # Local swap optimization (patient counts remain fixed by design).
    if swap_iters and swap_iters > 0:
        current_obj = _objective(folds, avg_images, avg_series, weights)
        patient_to_record = {r['patient_id']: r for r in patients}
        all_patient_ids = [r['patient_id'] for r in patients]
        for it in range(swap_iters):
            a, b = random.sample(all_patient_ids, 2)
            fa = mapping[a]; fb = mapping[b]
            if fa == fb:
                continue

            ra = patient_to_record[a]; rb = patient_to_record[b]
            f_a = folds[fa]; f_b = folds[fb]

            # Save old states.
            old_fa = f_a.copy(); old_fb = f_b.copy()

            # Apply swap.
            f_a['total_images'] = f_a['total_images'] - ra.get('total_slices',0) + rb.get('total_slices',0)
            f_b['total_images'] = f_b['total_images'] - rb.get('total_slices',0) + ra.get('total_slices',0)
            f_a['total_series'] = f_a['total_series'] - ra.get('n_series', 0) + rb.get('n_series', 0)
            f_b['total_series'] = f_b['total_series'] - rb.get('n_series', 0) + ra.get('n_series', 0)
            f_a['counts'][ra.get(strat_col)] -= 1; f_a['counts'][rb.get(strat_col)] += 1
            f_b['counts'][rb.get(strat_col)] -= 1; f_b['counts'][ra.get(strat_col)] += 1

            new_obj = _objective(folds, avg_images, avg_series, weights)
            if new_obj < current_obj:
                mapping[a], mapping[b] = fb, fa
                try:
                    f_a['patients'].remove(a); f_a['patients'].append(b)
                    f_b['patients'].remove(b); f_b['patients'].append(a)
                except Exception:
                    pass
                current_obj = new_obj
            else:
                f_a.update(old_fa); f_b.update(old_fb)

    # Rebuild final folds from mapping.
    final_folds = []
    for i in range(n_folds):
        pids = [pid for pid,foldid in mapping.items() if foldid==i]
        img=0.0; ser=0.0; counts=Counter()
        for pid in pids:
            rec = next((r for r in patients if r['patient_id']==pid), None)
            if not rec: continue
            img += float(rec.get('total_slices',0))
            ser += float(rec.get('n_series', 0))
            counts[rec.get(strat_col)] += 1
        final_folds.append({'patients': pids, 'total_images': img, 'total_series': ser, 'counts': counts})
    return mapping, final_folds

def report_stats(pat_df, mapping, n_folds, out_prefix):
    pat_df['fold'] = pat_df['patient_id'].map(mapping)
    # Choose images column flexibly.
    if 'total_slices' in pat_df.columns:
        images_col = 'total_slices'
    elif 'total_images' in pat_df.columns:
        images_col = 'total_images'
    else:
        images_col = 'num_slices'
    series_col = 'n_series' if 'n_series' in pat_df.columns else None

    with open(out_prefix + "_stats.txt", 'w') as fh:
        fh.write(f"Fold counts (patients per fold):\n")
        for i in range(n_folds):
            sub = pat_df[pat_df['fold']==i]
            total_images = sub[images_col].sum() if images_col in sub.columns else 0
            total_series = sub[series_col].sum() if series_col and series_col in sub.columns else 'N/A'
            fh.write(f"Fold {i}: patients={len(sub)}, total_series={int(total_series) if total_series != 'N/A' else total_series}, total_images={int(total_images)}\n")
        fh.write("\nPer-fold thickness bucket distribution:\n")
        for i in range(n_folds):
            sub = pat_df[pat_df['fold']==i]
            cnt = sub['thickness_bucket'].value_counts().to_dict()
            fh.write(f"Fold {i}: {cnt}\n")
    # Save mapping CSV.
    pat_df.to_csv(out_prefix + "_folds.csv", index=False)
    print("Saved:", out_prefix + "_folds.csv", out_prefix + "_stats.txt")

def main(args):
    df = pd.read_csv(args.input_csv)
    # Sanity checks.
    required = {'patient_id','study_id','series_id','num_slices'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns in metadata CSV: " + ", ".join(missing))
    pat_df = make_patient_table(df)
    weights = {
        'patients': args.weight_patients,
        'series': args.weight_series,
        'images': args.weight_images,
    }
    mapping, folds = greedy_assign_folds(
        pat_df,
        args.n_folds,
        strat_col='strat_key',
        seed=args.seed,
        weights=weights,
        swap_iters=args.swap_iters,
    )
    report_stats(pat_df, mapping, args.n_folds, args.output_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create patient-level CV folds with tunable weighting")
    parser.add_argument("--input-csv", required=True, help="input metadata CSV")
    parser.add_argument("--n-folds", type=int, default=2, help="number of folds")
    parser.add_argument("--output-prefix", default="cv", help="prefix for output files")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    parser.add_argument("--weight-patients", type=float, default=8.0, help="weight for patient-count balancing")
    parser.add_argument("--weight-series", type=float, default=3.0, help="weight for total-series balancing")
    parser.add_argument("--weight-images", type=float, default=2.0, help="weight for total-image balancing")
    parser.add_argument("--swap-iters", type=int, default=2000, help="number of local swap iterations for improvement")
    args = parser.parse_args()
    main(args)