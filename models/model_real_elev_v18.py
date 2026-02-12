#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v18.0
Using REAL elevation data from Open-Elevation API.

Elevation correlations found:
- Total Alkalinity: 0.34 (moderate)
- Dissolved Reactive Phosphorus: 0.29 (moderate)
- Electrical Conductance: 0.12 (low)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import os
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
CACHE_FILE = 'external_data_cache.json'


def load_elevation_cache():
    """Load cached elevation data."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            return cache.get('elevation', {})
    return {}


def get_elevation(lat, lon, cache):
    """Get elevation from cache or estimate."""
    key = f"{lat:.4f},{lon:.4f}"
    if key in cache:
        return cache[key]
    # Fallback estimate
    return 200 + max(0, 30 - lon) * 80 + max(0, lat + 34) * 50


def create_features(wq_df, ls_df, tc_df, elev_cache, train_wq=None, target=None):
    """Create features with real elevation data."""
    df = pd.DataFrame()
    eps = 1e-10
    n = len(wq_df)

    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values

    # === COORDINATES ===
    df['Latitude'] = lat
    df['Longitude'] = lon
    df['lat_sq'] = lat ** 2
    df['lon_sq'] = lon ** 2
    df['lat_lon'] = lat * lon

    # === REAL ELEVATION ===
    elevations = np.array([get_elevation(la, lo, elev_cache) for la, lo in zip(lat, lon)])
    df['elevation'] = elevations
    df['elevation_log'] = np.log1p(elevations)
    df['elevation_sq'] = elevations ** 2

    # Elevation categories
    df['is_coastal'] = (elevations < 100).astype(float)
    df['is_lowland'] = ((elevations >= 100) & (elevations < 500)).astype(float)
    df['is_midland'] = ((elevations >= 500) & (elevations < 1000)).astype(float)
    df['is_highland'] = (elevations >= 1000).astype(float)

    # Distance to coast
    df['dist_coast'] = np.array([min(max(0, 30 - lo) * 111, max(0, la + 34) * 111)
                                  for la, lo in zip(lat, lon)])

    # === TEMPORAL ===
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    month = dates.dt.month
    df['month'] = month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['is_wet_season'] = month.isin([10, 11, 12, 1, 2, 3]).astype(float)

    # === SPECTRAL ===
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['turbidity'] = nir / (green + eps)

    # === CLIMATE ===
    pet = tc_df['pet'].values.astype(float)
    df['pet'] = pet
    df['pet_log'] = np.log1p(np.maximum(pet, 0))

    # === INTERACTIONS WITH ELEVATION ===
    df['elev_pet'] = elevations * pet
    df['elev_ndmi'] = elevations * df['NDMI']
    df['elev_turbidity'] = elevations * df['turbidity']
    df['elev_wet'] = elevations * df['is_wet_season']
    df['elev_lat'] = elevations * lat
    df['elev_lon'] = elevations * lon

    # === SPATIAL IDW ===
    if train_wq is not None and target is not None:
        train_coords = train_wq[['Latitude', 'Longitude']].values
        train_values = train_wq[target].values
        val_coords = np.column_stack([lat, lon])

        tree = cKDTree(train_coords)

        for k in [10, 30, 50]:
            dists, idxs = tree.query(val_coords, k=min(k, len(train_coords)))
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / dists
            weights = weights / weights.sum(axis=1, keepdims=True)
            df[f'idw_k{k}'] = np.sum(weights * train_values[idxs], axis=1)

        df['dist_nearest'] = tree.query(val_coords, k=1)[0]
        _, nn_idx = tree.query(val_coords, k=1)
        df['nn_value'] = train_values[nn_idx]

    return df


def train_ensemble(X_train, y_train, X_val, seeds=[42, 123, 456]):
    """Train ensemble."""
    all_preds = []

    for seed in seeds:
        xgb1 = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
            reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbosity=0
        )
        xgb1.fit(X_train, y_train)
        all_preds.append(xgb1.predict(X_val))

        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=25,
                reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            all_preds.append(lgb_model.predict(X_val))

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            max_features=0.5, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        all_preds.append(rf.predict(X_val))

    return np.mean(all_preds, axis=0)


def main():
    print('=' * 70)
    print('EY Water Quality v18.0 - Real Elevation Data')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    elev_cache = load_elevation_cache()
    print(f'   Elevation cache: {len(elev_cache)} records')

    location_groups = wq['Latitude'].round(2).astype(str) + '_' + wq['Longitude'].round(2).astype(str)

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

        X_train_raw = create_features(wq, ls_train, tc_train, elev_cache, wq, target)
        X_val_raw = create_features(sub_template, ls_val, tc_val, elev_cache, wq, target)

        print(f'   Features: {X_train_raw.shape[1]}')

        # Preprocess
        X_train = X_train_raw.fillna(X_train_raw.median())
        X_val = X_val_raw.fillna(X_train_raw.median())
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        y_raw = wq[target].values
        y_train = cfg['transform'](y_raw)

        # CV
        print('   Cross-validation...')
        kf = GroupKFold(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train_scaled, y_train, location_groups):
            X_tr, X_va = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]
            model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      min_child_weight=15, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            ss_res = np.sum((y_va - pred) ** 2)
            ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
            cv_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        print(f'   CV R²: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}')

        # Train
        print('   Training ensemble...')
        ml_pred = cfg['inverse'](train_ensemble(X_train_scaled, y_train, X_val_scaled))
        idw_pred = X_val_raw['idw_k50'].values

        # Blend
        final_pred = 0.4 * ml_pred + 0.6 * idw_pred
        final_pred = np.clip(final_pred, cfg['clip'][0], cfg['clip'][1])

        predictions[target] = final_pred
        print(f'   Final: mean={final_pred.mean():.2f}, std={final_pred.std():.2f}')

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
    submission.to_csv('submission_v18.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print('\nSubmission: submission_v18.csv')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}')


if __name__ == '__main__':
    main()
