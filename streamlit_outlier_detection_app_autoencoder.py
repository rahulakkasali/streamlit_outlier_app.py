# outlier_detector_cli.py

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
import os
import tempfile


def preprocess(df, feature_cols):
    sub = df[feature_cols].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sub = sub.dropna(axis=1, how='all')
    sub = sub.dropna()
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        raise ValueError('No valid numeric data after preprocessing')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sub)
    return sub, scaled, scaler


def detect_outliers(df, feature_cols, pct=95, random_state=42, n_estimators=100, contamination='auto'):
    sub_df, scaled, scaler = preprocess(df, feature_cols)
    if sub_df.shape[0] < 5:
        raise ValueError('Not enough rows after preprocessing (need at least 5)')
    cleaned_index = sub_df.index.to_list()
    idx_train, idx_test = train_test_split(cleaned_index, test_size=0.2, random_state=random_state)
    index_positions = {idx: i for i, idx in enumerate(cleaned_index)}
    train_pos = [index_positions[i] for i in idx_train]
    test_pos = [index_positions[i] for i in idx_test]
    X_train = scaled[train_pos]
    X_test = scaled[test_pos]
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    clf.fit(X_train)
    scores = clf.score_samples(X_test)
    anomaly_score = -scores
    results_df = sub_df.loc[idx_test].copy()
    results_df = results_df.reset_index(drop=False).rename(columns={'index': 'orig_index'})
    results_df['anomaly_score'] = anomaly_score
    threshold = float(np.percentile(results_df['anomaly_score'], pct))
    results_df['is_outlier'] = results_df['anomaly_score'] > threshold
    return sub_df, results_df, threshold, scaler


def save_outputs(df, results_df, threshold, out_prefix):
    out_dir = os.path.dirname(out_prefix) or '.'
    os.makedirs(out_dir, exist_ok=True)
    hist_path = f"{out_prefix}_anomaly_hist.png"
    plt.figure(figsize=(8,4))
    plt.hist(results_df['anomaly_score'], bins=min(100, max(10, len(results_df)//2)))
    plt.axvline(threshold, color='red')
    plt.title('Anomaly score (test set)')
    plt.xlabel('anomaly_score')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    scatter_path = None
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2 and results_df.shape[0] >= 2:
        scatter_path = f"{out_prefix}_scatter.png"
        x = numeric_cols[0]
        y = numeric_cols[1]
        plt.figure(figsize=(6,6))
        normal = results_df[~results_df['is_outlier']]
        outl = results_df[results_df['is_outlier']]
        plt.scatter(normal[x], normal[y], label='Normal', alpha=0.6)
        plt.scatter(outl[x], outl[y], label='Outlier', alpha=0.9)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
    cleaned = df.loc[results_df.loc[~results_df['is_outlier'], 'orig_index']]
    cleaned_path = f"{out_prefix}_cleaned.csv"
    cleaned.to_csv(cleaned_path, index=False)
    results_path = f"{out_prefix}_results.csv"
    results_df.to_csv(results_path, index=False)
    return {'hist': hist_path, 'scatter': scatter_path, 'cleaned_csv': cleaned_path, 'results_csv': results_path}


def generate_sample_csv(path, n=500, n_outliers=10, random_state=42):
    rng = np.random.RandomState(random_state)
    X1 = rng.normal(0, 1, size=(n,))
    X2 = rng.normal(0, 1, size=(n,))
    df = pd.DataFrame({'feat1': X1, 'feat2': X2})
    out_idx = rng.choice(n, size=n_outliers, replace=False)
    df.loc[out_idx, 'feat1'] = df.loc[out_idx, 'feat1'] + rng.normal(8, 1, size=n_outliers)
    df.loc[out_idx, 'feat2'] = df.loc[out_idx, 'feat2'] + rng.normal(8, 1, size=n_outliers)
    df.to_csv(path, index=False)
    return path


def run_internal_tests():
    print('Running internal tests...')
    tmpdir = tempfile.mkdtemp(prefix='outlier_test_')
    # Test 1: basic synthetic dataset
    test_path1 = os.path.join(tmpdir, 'test_sample1.csv')
    generate_sample_csv(test_path1, n=200, n_outliers=8, random_state=1)
    df1 = pd.read_csv(test_path1)
    feature_cols1 = [c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c])]
    sub_df1, results_df1, threshold1, scaler1 = detect_outliers(df1, feature_cols1, pct=90, random_state=1)
    detected1 = int(results_df1['is_outlier'].sum())
    print(f'Test1 detected outliers: {detected1} (expected >0)')
    if detected1 == 0:
        raise AssertionError('Test1 failed: expected at least 1 outlier detected')
    # Test 2: different seed and contamination sensitivity
    test_path2 = os.path.join(tmpdir, 'test_sample2.csv')
    generate_sample_csv(test_path2, n=300, n_outliers=15, random_state=7)
    df2 = pd.read_csv(test_path2)
    feature_cols2 = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    sub_df2, results_df2, threshold2, scaler2 = detect_outliers(df2, feature_cols2, pct=95, random_state=7)
    detected2 = int(results_df2['is_outlier'].sum())
    print(f'Test2 detected outliers: {detected2} (expected >=1)')
    if detected2 == 0:
        raise AssertionError('Test2 failed: expected at least 1 outlier detected')
    print('All internal tests passed.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--out', type=str, default='outlier_output')
    parser.add_argument('--pct', type=int, default=95)
    parser.add_argument('--features', type=str, default=None)
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args()
    if args.run_tests:
        run_internal_tests()
        return
    if args.csv is None:
        sample_path = 'sample_data.csv'
        generate_sample_csv(sample_path)
        csv_path = sample_path
    else:
        csv_path = args.csv
    df = pd.read_csv(csv_path)
    if args.features:
        feature_cols = [c.strip() for c in args.features.split(',') if c.strip() in df.columns]
    else:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) == 0:
        raise SystemExit('No numeric features found or specified')
    sub_df, results_df, threshold, scaler = detect_outliers(df, feature_cols, pct=args.pct)
    paths = save_outputs(sub_df, results_df, threshold, args.out)
    print('Saved outputs:')
    for k, v in paths.items():
        print(k, v)


if __name__ == '__main__':
    main()
