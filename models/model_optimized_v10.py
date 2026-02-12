#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v10.0
Optimized based on what works: Simple LightGBM with coordinates.

Key learnings:
- Simple LightGBM gave best score (0.274)
- Coordinates are essential
- Overfitting is the main problem

Strategy:
- Use minimal but effective features
- Very strong regularization
- Multiple simple models averaged
- Focus on spatial patterns in coordinates
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']


def create_optimized_features(wq_df, ls_df, tc_df, train_wq=None, for_target=None):
    """Create optimized features based on learnings."""
    df = pd.DataFrame()
    eps = 1e-10

    # === COORDINATES (ESSENTIAL) ===
    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values
    df['Latitude'] = lat
    df['Longitude'] = lon

    # Polynomial coordinates (captures non-linear spatial patterns)
    df['lat_sq'] = lat ** 2
    df['lon_sq'] = lon ** 2
    df['lat_lon'] = lat * lon
    df['lat_cube'] = lat ** 3
    df['lon_cube'] = lon ** 3

    # Distance features
    df['dist_origin'] = np.sqrt(lat**2 + lon**2)
    df['lat_lon_sum'] = lat + lon
    df['lat_lon_diff'] = lat - lon

    # === TEMPORAL (cyclic) ===
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    month = dates.dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    doy = dates.dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    # === SPECTRAL (only proven useful) ===
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values

    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['turbidity'] = nir / (green + eps)

    # Reflectance stats (can indicate water clarity)
    bands = np.column_stack([nir, green, swir16, swir22])
    df['reflect_mean'] = np.nanmean(bands, axis=1)
    df['reflect_std'] = np.nanstd(bands, axis=1)

    # === CLIMATE ===
    df['pet'] = tc_df['pet'].values

    # === SPATIAL FEATURES FROM TRAINING (if target provided) ===
    if train_wq is not None and for_target is not None:
        coords = np.column_stack([lat, lon])
        loc_means = train_wq.groupby(['Latitude', 'Longitude'])[for_target].mean().reset_index()
        train_coords = loc_means[['Latitude', 'Longitude']].values
        train_values = loc_means[for_target].values

        tree = cKDTree(train_coords)
        dists, idxs = tree.query(coords, k=min(10, len(train_coords)))

        if len(dists.shape) == 1:
            dists = dists.reshape(-1, 1)
            idxs = idxs.reshape(-1, 1)

        dists = np.maximum(dists, 1e-10)

        # IDW features (different powers)
        for k in [3, 5, 10]:
            k_use = min(k, dists.shape[1])
            for power in [1, 2]:
                w = 1.0 / (dists[:, :k_use] ** power)
                w = w / w.sum(axis=1, keepdims=True)
                idw = np.sum(w * train_values[idxs[:, :k_use]], axis=1)
                df[f'idw_k{k}_p{power}'] = idw

        # Nearest neighbor
        df['nn1'] = train_values[idxs[:, 0]]
        df['dist_nearest'] = dists[:, 0]

    return df


def train_lightgbm_simple(X_train, y_train, X_val, seed=42, n_estimators=100):
    """Train simple LightGBM with strong regularization."""
    if not HAS_LIGHTGBM:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=3, learning_rate=0.05,
            subsample=0.6, colsample_bytree=0.6, min_child_weight=30,
            reg_alpha=2.0, reg_lambda=10.0, random_state=seed, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train)
        return model.predict(X_val)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=3,  # Very shallow
        num_leaves=8,  # Very few leaves
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        min_child_samples=50,  # High min samples
        reg_alpha=2.0,  # Strong L1
        reg_lambda=10.0,  # Strong L2
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_val)


def compute_cv_r2(X, y, groups, model_fn, n_splits=5):
    """Compute cross-validation R² with spatial groups."""
    from sklearn.model_selection import GroupKFold
    kf = GroupKFold(n_splits=n_splits)

    r2_scores = []
    for train_idx, val_idx in kf.split(X, y, groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        pred = model_fn(X_tr, y_tr, X_va)

        # R² calculation
        ss_res = np.sum((y_va - pred) ** 2)
        ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)

    return np.mean(r2_scores), np.std(r2_scores)


def main():
    print('=' * 70)
    print('EY Water Quality Prediction v10.0 - Optimized Simple Model')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Training: {len(wq)}, Validation: {len(sub_template)}')

    # Location groups for CV
    location_groups = wq['Latitude'].round(2).astype(str) + '_' + wq['Longitude'].round(2).astype(str)

    # Target configs
    target_configs = {
        'Total Alkalinity': {'transform': np.log1p, 'inverse': np.expm1, 'clip': (0, 500)},
        'Electrical Conductance': {'transform': np.log1p, 'inverse': np.expm1, 'clip': (0, 2000)},
        'Dissolved Reactive Phosphorus': {
            'transform': lambda x: np.log1p(x + 1),
            'inverse': lambda x: np.clip(np.expm1(x) - 1, 0, None),
            'clip': (0, 300)
        },
    }

    predictions = {}

    for target in TARGET_COLS:
        print(f'\n2. Processing {target}...')
        cfg = target_configs[target]

        # Create features with spatial info for this target
        X_train_raw = create_optimized_features(wq, ls_train, tc_train, train_wq=wq, for_target=target)
        X_val_raw = create_optimized_features(sub_template, ls_val, tc_val, train_wq=wq, for_target=target)

        # Fill NaN
        X_train = X_train_raw.fillna(X_train_raw.median())
        X_val = X_val_raw.fillna(X_train_raw.median())

        print(f'   Features: {X_train.shape[1]}')

        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Prepare target
        y_raw = wq[target].values
        y_train = cfg['transform'](y_raw)

        # CV evaluation
        cv_r2, cv_std = compute_cv_r2(
            X_train_scaled, y_train, location_groups,
            lambda X, y, X_v: train_lightgbm_simple(X, y, X_v, seed=42, n_estimators=100)
        )
        print(f'   Spatial CV R² (log): {cv_r2:.4f} +/- {cv_std:.4f}')

        # Train multiple models with different seeds
        print('   Training ensemble...')
        all_preds = []
        for seed in [42, 123, 456, 789, 1024]:
            for n_est in [80, 100, 120]:
                pred_log = train_lightgbm_simple(X_train_scaled, y_train, X_val_scaled, seed=seed, n_estimators=n_est)
                pred = cfg['inverse'](pred_log)
                all_preds.append(pred)

        # Also add pure IDW prediction
        tree = cKDTree(wq[['Latitude', 'Longitude']].values)
        val_coords = sub_template[['Latitude', 'Longitude']].values
        dists, idxs = tree.query(val_coords, k=15)
        dists = np.maximum(dists, 1e-10)
        weights = 1 / dists ** 2
        weights = weights / weights.sum(axis=1, keepdims=True)
        idw_pred = np.sum(weights * y_raw[idxs], axis=1)
        all_preds.append(idw_pred)

        # Combine predictions
        final_pred = np.median(all_preds, axis=0)  # Use median for robustness
        final_pred = np.clip(final_pred, cfg['clip'][0], cfg['clip'][1])

        predictions[target] = final_pred
        print(f'   Final: mean={final_pred.mean():.2f}, std={final_pred.std():.2f}')
        print(f'   Range: [{final_pred.min():.2f}, {final_pred.max():.2f}]')

    # Create submission
    print('\n3. Creating submission...')
    submission = pd.DataFrame({
        'Latitude': sub_template['Latitude'],
        'Longitude': sub_template['Longitude'],
        'Sample Date': sub_template['Sample Date'],
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })

    submission.to_csv('submission_v10.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE! Submission: submission_v10.csv')
    print('=' * 70)
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, median={submission[col].median():.2f}')


if __name__ == '__main__':
    main()
