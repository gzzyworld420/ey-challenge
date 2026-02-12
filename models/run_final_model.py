#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Final Water Quality Model
Uses geographic IDW (despite location mismatch) + calibrated post-processing.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
SEEDS = [42, 123, 456, 789, 2024]


def engineer_features(wq_df, ls_df, tc_df):
    df = pd.DataFrame()
    lat, lon = wq_df['Latitude'].values, wq_df['Longitude'].values
    eps = 1e-10

    # Spatial
    df['Latitude'], df['Longitude'] = lat, lon
    df['lat_lon'] = lat * lon
    df['lat_sq'], df['lon_sq'] = lat**2, lon**2

    # Temporal
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    df['month'] = dates.dt.month
    df['year'] = dates.dt.year
    df['doy'] = dates.dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Landsat
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['nir'], df['green'], df['swir16'], df['swir22'] = nir, green, swir16, swir22
    df['NDMI'], df['MNDWI'] = ls_df['NDMI'].values, ls_df['MNDWI'].values
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['turbidity'] = nir / (green + eps)
    df['nd_nir_swir'] = (nir - swir22) / (nir + swir22 + eps)
    df['band_sum'] = nir + green + swir16 + swir22

    df['pet'] = tc_df['pet'].values
    df['pet_ndmi'] = df['pet'] * df['NDMI']
    df['lat_ndmi'] = lat * df['NDMI']

    return df


def add_idw_features(X_df, train_wq, k_values=[5, 10, 20], powers=[1, 2]):
    """Add IDW features based on geographic proximity."""
    coords = X_df[['Latitude', 'Longitude']].values
    loc_stats = train_wq.groupby(['Latitude', 'Longitude'])[TARGET_COLS].agg(['mean', 'median']).reset_index()
    loc_coords = loc_stats[['Latitude', 'Longitude']].values

    tree = cKDTree(loc_coords)
    max_k = min(max(k_values), len(loc_coords))
    dists, idxs = tree.query(coords, k=max_k)
    if len(dists.shape) == 1:
        dists, idxs = dists.reshape(-1, 1), idxs.reshape(-1, 1)

    new_feats = pd.DataFrame(index=range(len(X_df)))
    new_feats['dist_nearest'] = dists[:, 0]
    new_feats['dist_5th'] = dists[:, min(4, max_k-1)]

    for target in TARGET_COLS:
        ts = target[:2]
        for stat in ['mean', 'median']:
            vals = loc_stats[(target, stat)].values
            for k in k_values:
                for power in powers:
                    k_use = min(k, max_k)
                    w = 1.0 / (dists[:, :k_use] ** power + 1e-6)
                    w = w / w.sum(axis=1, keepdims=True)
                    new_feats[f'idw_{ts}_{stat[:3]}_k{k}_p{power}'] = np.sum(w * vals[idxs[:, :k_use]], axis=1)
        new_feats[f'nn1_{ts}'] = loc_stats[(target, 'mean')].values[idxs[:, 0]]

    return new_feats


def train_ensemble(X_train, y_train, X_val):
    preds = []

    # XGBoost
    for seed in SEEDS:
        for cfg in [
            dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
                 colsample_bytree=0.5, min_child_weight=15, reg_alpha=1.0, reg_lambda=5.0),
            dict(n_estimators=800, max_depth=3, learning_rate=0.01, subsample=0.6,
                 colsample_bytree=0.4, min_child_weight=20, reg_alpha=2.0, reg_lambda=8.0),
        ]:
            m = xgb.XGBRegressor(**cfg, random_state=seed, n_jobs=-1, verbosity=0)
            m.fit(X_train, y_train)
            preds.append(m.predict(X_val))

    # LightGBM
    if HAS_LIGHTGBM:
        for seed in SEEDS[:3]:
            m = lgb.LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.5, min_child_samples=15,
                reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1
            )
            m.fit(X_train, y_train)
            preds.append(m.predict(X_val))

    # ExtraTrees
    for seed in SEEDS[:3]:
        m = ExtraTreesRegressor(n_estimators=400, max_depth=12, min_samples_leaf=8,
                                max_features=0.5, random_state=seed, n_jobs=-1)
        m.fit(X_train, y_train)
        preds.append(m.predict(X_val))

    # RandomForest
    for seed in SEEDS[:2]:
        m = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=10,
                                  max_features=0.5, random_state=seed, n_jobs=-1)
        m.fit(X_train, y_train)
        preds.append(m.predict(X_val))

    return np.mean(preds, axis=0), len(preds)


def calibrate_predictions(pred, train_values, target_name):
    """Post-hoc calibration to better match training distribution."""
    train_mean = train_values.mean()
    train_std = train_values.std()
    pred_mean = pred.mean()
    pred_std = pred.std()

    # Standardize and rescale
    if target_name == 'Dissolved Reactive Phosphorus':
        # DRP is highly skewed - use different calibration
        train_median = np.median(train_values)
        pred_median = np.median(pred)
        # Scale to match median better, preserve some variance
        calibrated = pred * (train_median / pred_median) * 0.8 + train_mean * 0.2
    else:
        # For TA and EC, scale towards training mean
        calibrated = (pred - pred_mean) / (pred_std + 1e-6) * train_std * 0.8 + train_mean * 0.95

    # Clip to reasonable range
    calibrated = np.clip(calibrated, train_values.min() * 0.3, train_values.max() * 1.1)

    return calibrated


def main():
    print('=' * 70)
    print('EY Challenge 2026 - Final Model (IDW + Calibration)')
    print('=' * 70)

    # Load
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub = pd.read_csv('submission_template.csv')

    print(f'\nTraining: {len(wq)}, Validation: {len(sub)}')

    # Features
    print('\n1. Engineering features...')
    X_train = engineer_features(wq, ls_train, tc_train)
    X_val = engineer_features(sub, ls_val, tc_val)

    # Impute
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    # IDW
    print('2. Adding IDW features...')
    idw_train = add_idw_features(X_train, wq)
    idw_val = add_idw_features(X_val, wq)

    X_train = pd.concat([X_train, idw_train], axis=1)
    X_val = pd.concat([X_val, idw_val], axis=1)
    print(f'   Features: {X_train.shape[1]}')

    # Clean & scale
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Target configs
    configs = {
        'Total Alkalinity': (np.log1p, np.expm1),
        'Electrical Conductance': (np.log1p, np.expm1),
        'Dissolved Reactive Phosphorus': (np.log1p, np.expm1),
    }

    # CV
    print('\n3. Cross-validation...')
    loc_groups = wq['Latitude'].round(1).astype(str) + '_' + wq['Longitude'].round(1).astype(str)
    for target, (tf, _) in configs.items():
        y = tf(wq[target].values)
        kf = GroupKFold(n_splits=5)
        scores = []
        for tr, val in kf.split(X_train_s, y, loc_groups):
            m = xgb.XGBRegressor(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
            m.fit(X_train_s[tr], y[tr])
            scores.append(np.sqrt(np.mean((y[val] - m.predict(X_train_s[val]))**2)))
        print(f'   {target}: CV RMSE = {np.mean(scores):.4f}')

    # Train & predict
    print('\n4. Training...')
    predictions = {}

    for target, (transform, inv_transform) in configs.items():
        print(f'\n   {target}:')
        y_train = transform(wq[target].values)

        pred_raw, n = train_ensemble(X_train_s, y_train, X_val_s)
        pred_raw = inv_transform(pred_raw)

        # Calibrate
        pred_cal = calibrate_predictions(pred_raw, wq[target].values, target)

        predictions[target] = pred_cal
        print(f'      Raw: mean={pred_raw.mean():.1f}, range=[{pred_raw.min():.1f}, {pred_raw.max():.1f}]')
        print(f'      Cal: mean={pred_cal.mean():.1f}, range=[{pred_cal.min():.1f}, {pred_cal.max():.1f}]')
        print(f'      Train: mean={wq[target].mean():.1f}, range=[{wq[target].min():.1f}, {wq[target].max():.1f}]')

    # Save
    submission = pd.DataFrame({
        'Latitude': sub['Latitude'],
        'Longitude': sub['Longitude'],
        'Sample Date': sub['Sample Date'],
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })
    submission = submission[['Latitude', 'Longitude', 'Sample Date'] + TARGET_COLS]
    submission.to_csv('submission_final.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE! Saved to submission_final.csv')
    print('=' * 70)

    # Also save uncalibrated version
    predictions_raw = {}
    for target, (transform, inv_transform) in configs.items():
        y_train = transform(wq[target].values)
        pred_raw, _ = train_ensemble(X_train_s, y_train, X_val_s)
        predictions_raw[target] = np.clip(inv_transform(pred_raw), wq[target].min()*0.5, wq[target].max()*1.1)

    submission_raw = submission.copy()
    for col in TARGET_COLS:
        submission_raw[col] = predictions_raw[col]
    submission_raw.to_csv('submission_final_raw.csv', index=False)
    print('Also saved: submission_final_raw.csv (uncalibrated)')


if __name__ == '__main__':
    main()
