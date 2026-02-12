#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v5.0
Focus on spectral features for out-of-distribution prediction.
Validation locations are completely different from training!
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
SEEDS = [42, 123, 456, 789, 2024]


def engineer_spectral_features(wq_df, ls_df, tc_df):
    """Focus on spectral and climate features that generalize to new locations."""
    df = pd.DataFrame()

    # ============ SPATIAL (will help for regional patterns) ============
    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values

    df['Latitude'] = lat
    df['Longitude'] = lon

    # Regional encoding
    df['lat_bin'] = np.floor(lat).astype(int)
    df['lon_bin'] = np.floor(lon).astype(int)

    # Distance from key reference points
    df['dist_johannesburg'] = np.sqrt((lat + 26.2)**2 + (lon - 28.0)**2)
    df['dist_cape_town'] = np.sqrt((lat + 33.9)**2 + (lon - 18.4)**2)
    df['dist_durban'] = np.sqrt((lat + 29.9)**2 + (lon - 31.0)**2)

    # ============ TEMPORAL ============
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    df['month'] = dates.dt.month
    df['year'] = dates.dt.year
    df['doy'] = dates.dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

    # Seasons (Southern Hemisphere)
    df['is_summer'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_autumn'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_winter'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_spring'] = df['month'].isin([9, 10, 11]).astype(int)

    # ============ LANDSAT SPECTRAL (KEY FEATURES) ============
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)
    eps = 1e-10

    # Raw bands
    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22

    # Log of bands
    df['nir_log'] = np.log1p(nir)
    df['green_log'] = np.log1p(green)
    df['swir16_log'] = np.log1p(swir16)
    df['swir22_log'] = np.log1p(swir22)

    # Provided indices
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values

    # Water quality indices
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['AWEI'] = 4 * green - 0.25 * nir - 2.75 * swir16 - 2.75 * swir22
    df['AWEI_sh'] = green + 2.5 * nir - 1.5 * (swir16 + swir22)

    # Turbidity/sediment related
    df['turbidity'] = nir / (green + eps)
    df['sediment'] = (swir16 - green) / (swir16 + green + eps)

    # All band ratios
    df['nir_green'] = nir / (green + eps)
    df['nir_swir16'] = nir / (swir16 + eps)
    df['nir_swir22'] = nir / (swir22 + eps)
    df['green_swir16'] = green / (swir16 + eps)
    df['green_swir22'] = green / (swir22 + eps)
    df['swir16_swir22'] = swir16 / (swir22 + eps)
    df['swir22_swir16'] = swir22 / (swir16 + eps)

    # Normalized differences
    df['nd_nir_swir16'] = (nir - swir16) / (nir + swir16 + eps)
    df['nd_nir_swir22'] = (nir - swir22) / (nir + swir22 + eps)
    df['nd_green_swir16'] = (green - swir16) / (green + swir16 + eps)
    df['nd_green_swir22'] = (green - swir22) / (green + swir22 + eps)
    df['nd_swir16_swir22'] = (swir16 - swir22) / (swir16 + swir22 + eps)

    # Band statistics
    bands = np.column_stack([nir, green, swir16, swir22])
    df['band_sum'] = np.nansum(bands, axis=1)
    df['band_mean'] = np.nanmean(bands, axis=1)
    df['band_std'] = np.nanstd(bands, axis=1)
    df['band_max'] = np.nanmax(bands, axis=1)
    df['band_min'] = np.nanmin(bands, axis=1)
    df['band_range'] = df['band_max'] - df['band_min']
    df['band_cv'] = df['band_std'] / (df['band_mean'] + eps)

    # Differences
    df['nir_minus_green'] = nir - green
    df['swir_diff'] = swir16 - swir22
    df['nir_minus_swir'] = nir - swir16

    # ============ TERRACLIMATE ============
    pet = tc_df['pet'].values
    df['pet'] = pet
    df['pet_log'] = np.log1p(pet)

    # ============ INTERACTIONS ============
    # Spectral-climate
    df['pet_ndmi'] = pet * df['NDMI']
    df['pet_mndwi'] = pet * df['MNDWI']
    df['pet_ndwi'] = pet * df['NDWI']
    df['pet_turbidity'] = pet * df['turbidity']

    # Spatial-spectral
    df['lat_ndmi'] = lat * df['NDMI']
    df['lon_ndmi'] = lon * df['NDMI']
    df['lat_pet'] = lat * pet
    df['lat_turbidity'] = lat * df['turbidity']

    # Temporal-spectral
    df['month_ndmi'] = df['month'] * df['NDMI']
    df['month_pet'] = df['month'] * pet

    return df


def train_ensemble(X_train, y_train, X_val):
    """Train ensemble optimized for generalization."""
    preds = []

    # XGBoost - multiple configs
    xgb_configs = [
        dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
             colsample_bytree=0.5, min_child_weight=15, reg_alpha=1.0, reg_lambda=5.0),
        dict(n_estimators=800, max_depth=3, learning_rate=0.01, subsample=0.6,
             colsample_bytree=0.4, min_child_weight=20, reg_alpha=2.0, reg_lambda=10.0),
        dict(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.75,
             colsample_bytree=0.6, min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0),
    ]

    for cfg in xgb_configs:
        for seed in SEEDS:
            model = xgb.XGBRegressor(**cfg, random_state=seed, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)
            preds.append(model.predict(X_val))

    # LightGBM
    if HAS_LIGHTGBM:
        for seed in SEEDS[:3]:
            model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.5, min_child_samples=20,
                reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1
            )
            model.fit(X_train, y_train)
            preds.append(model.predict(X_val))

    # GradientBoosting
    for seed in SEEDS[:2]:
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=15, random_state=seed
        )
        model.fit(X_train, y_train)
        preds.append(model.predict(X_val))

    # ExtraTrees
    for seed in SEEDS[:3]:
        model = ExtraTreesRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=10,
            max_features=0.5, random_state=seed, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds.append(model.predict(X_val))

    # RandomForest
    for seed in SEEDS[:3]:
        model = RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features=0.5, random_state=seed, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds.append(model.predict(X_val))

    return np.mean(preds, axis=0), len(preds)


def main():
    print('=' * 70)
    print('EY Challenge 2026 - Water Quality v5.0 (Spectral Focus)')
    print('=' * 70)
    print('\nNOTE: Validation locations are DIFFERENT from training!')
    print('      Model must GENERALIZE, not interpolate.')

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Training: {wq.shape[0]}, Validation: {sub_template.shape[0]}')

    # Features
    print('\n2. Engineering spectral features...')
    X_train = engineer_spectral_features(wq, ls_train, tc_train)
    X_val = engineer_spectral_features(sub_template, ls_val, tc_val)
    print(f'   Features: {X_train.shape[1]}')

    # Impute
    print('\n3. Imputing missing values...')
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    # Clean
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

    # Scale
    print('\n4. Scaling...')
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Target configs
    configs = {
        'Total Alkalinity': (np.log1p, np.expm1, 5, 350),
        'Electrical Conductance': (np.log1p, np.expm1, 15, 1500),
        'Dissolved Reactive Phosphorus': (np.log1p, np.expm1, 5, 195),
    }

    # CV
    print('\n5. Cross-validation...')
    location_groups = (wq['Latitude'].round(1).astype(str) + '_' +
                       wq['Longitude'].round(1).astype(str))

    for target, (tf, _, _, _) in configs.items():
        y = tf(wq[target].values)
        kf = GroupKFold(n_splits=5)
        scores = []
        for tr, val in kf.split(X_train_scaled, y, location_groups):
            m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 random_state=42, verbosity=0)
            m.fit(X_train_scaled[tr], y[tr])
            scores.append(np.sqrt(np.mean((y[val] - m.predict(X_train_scaled[val]))**2)))
        print(f'   {target}: CV RMSE = {np.mean(scores):.4f}')

    # Train and predict
    print('\n6. Training final models...')
    predictions = {}

    for target, (transform, inv_transform, clip_min, clip_max) in configs.items():
        print(f'\n   {target}:')
        y_train = transform(wq[target].values)

        pred, n = train_ensemble(X_train_scaled, y_train, X_val_scaled)
        pred = inv_transform(pred)
        pred = np.clip(pred, clip_min, clip_max)

        predictions[target] = pred
        print(f'      Models: {n}, Range: [{pred.min():.1f}, {pred.max():.1f}]')
        print(f'      Training range: [{wq[target].min():.1f}, {wq[target].max():.1f}]')

    # Save
    print('\n7. Saving submission...')
    submission = pd.DataFrame({
        'Latitude': sub_template['Latitude'],
        'Longitude': sub_template['Longitude'],
        'Sample Date': sub_template['Sample Date'],
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })
    submission = submission[['Latitude', 'Longitude', 'Sample Date'] + TARGET_COLS]
    submission.to_csv('submission_v5.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE! Saved to submission_v5.csv')
    print('=' * 70)

    print('\nPrediction summary:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.1f}, '
              f'median={submission[col].median():.1f}')

    print('\nTraining reference:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={wq[col].mean():.1f}, '
              f'median={wq[col].median():.1f}')


if __name__ == '__main__':
    main()
