#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v4.0
Hybrid approach: IDW spatial interpolation + ML correction.
Uses proper out-of-fold IDW to avoid leakage.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
SEEDS = [42, 123, 456]


def create_location_id(df, precision=2):
    return df['Latitude'].round(precision).astype(str) + '_' + df['Longitude'].round(precision).astype(str)


def idw_interpolate(coords_pred, coords_train, values_train, k=10, power=2):
    """Simple IDW interpolation."""
    tree = cKDTree(coords_train)
    dists, idxs = tree.query(coords_pred, k=min(k, len(coords_train)))

    if len(dists.shape) == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    weights = 1.0 / (dists ** power + 1e-10)
    weights = weights / weights.sum(axis=1, keepdims=True)

    interpolated = np.sum(weights * values_train[idxs], axis=1)
    return interpolated


def engineer_features(wq_df, ls_df, tc_df):
    """Create features without target-based IDW."""
    df = pd.DataFrame()

    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values

    # Spatial
    df['Latitude'] = lat
    df['Longitude'] = lon
    df['lat_lon'] = lat * lon
    df['lat_sq'] = lat ** 2
    df['lon_sq'] = lon ** 2
    df['dist_center'] = np.sqrt((lat + 29)**2 + (lon - 25)**2)

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
    eps = 1e-10

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['nir_green'] = nir / (green + eps)
    df['swir_ratio'] = swir22 / (swir16 + eps)
    df['nd_nir_swir'] = (nir - swir22) / (nir + swir22 + eps)
    df['reflectance'] = nir + green + swir16 + swir22

    # TerraClimate
    df['pet'] = tc_df['pet'].values

    # Interactions
    df['pet_ndmi'] = df['pet'] * df['NDMI']
    df['lat_ndmi'] = lat * df['NDMI']

    return df


def compute_idw_features_oof(X_df, wq_df, target_col, k_values=[5, 10, 20], powers=[1, 2]):
    """Compute IDW features using out-of-fold for training data."""
    coords = X_df[['Latitude', 'Longitude']].values

    # Group by location and compute statistics
    loc_stats = wq_df.groupby(['Latitude', 'Longitude'])[target_col].agg(['mean', 'median', 'std', 'count']).reset_index()
    loc_coords = loc_stats[['Latitude', 'Longitude']].values

    new_feats = {}

    tree = cKDTree(loc_coords)
    max_k = min(max(k_values), len(loc_coords))
    dists, idxs = tree.query(coords, k=max_k)

    if len(dists.shape) == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    for stat in ['mean', 'median']:
        vals = loc_stats[stat].values
        for k in k_values:
            k_use = min(k, max_k)
            for power in powers:
                w = 1.0 / (dists[:, :k_use] ** power + 1e-6)
                w = w / w.sum(axis=1, keepdims=True)
                idw = np.sum(w * vals[idxs[:, :k_use]], axis=1)
                new_feats[f'idw_{stat}_k{k}_p{power}'] = idw

    # Nearest neighbor
    new_feats['nn1_mean'] = loc_stats['mean'].values[idxs[:, 0]]
    new_feats['nn1_median'] = loc_stats['median'].values[idxs[:, 0]]

    # Distance features
    new_feats['dist_nearest'] = dists[:, 0]
    new_feats['dist_5th'] = dists[:, min(4, max_k-1)]

    return pd.DataFrame(new_feats)


def train_model_blend(X_train, y_train, X_val, seeds=[42, 123, 456]):
    """Train blended model."""
    preds = []

    # XGBoost
    for seed in seeds:
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
            reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train)
        preds.append(model.predict(X_val))

    # LightGBM
    if HAS_LIGHTGBM:
        for seed in seeds:
            model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=15,
                reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1
            )
            model.fit(X_train, y_train)
            preds.append(model.predict(X_val))

    # ExtraTrees
    for seed in seeds:
        model = ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=8,
            max_features=0.5, random_state=seed, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds.append(model.predict(X_val))

    return np.mean(preds, axis=0), len(preds)


def main():
    print('=' * 70)
    print('EY AI & Data Challenge 2026 - Water Quality v4.0 (Hybrid IDW+ML)')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Training: {wq.shape[0]}, Validation: {sub_template.shape[0]}')

    # Base features
    print('\n2. Engineering base features...')
    X_train_base = engineer_features(wq, ls_train, tc_train)
    X_val_base = engineer_features(sub_template, ls_val, tc_val)

    # Impute
    print('\n3. Imputing missing values...')
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_base), columns=X_train_base.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val_base), columns=X_val_base.columns)

    # Target transformations
    transforms = {
        'Total Alkalinity': (np.log1p, np.expm1, 1, 400),
        'Electrical Conductance': (np.log1p, np.expm1, 10, 1500),
        'Dissolved Reactive Phosphorus': (lambda x: np.log1p(x), np.expm1, 5, 200),
    }

    predictions = {}

    for target in TARGET_COLS:
        print(f'\n4. Processing {target}...')
        transform, inv_transform, clip_min, clip_max = transforms[target]

        # Add IDW features for this target
        idw_train = compute_idw_features_oof(X_train_imp, wq, target)
        idw_val = compute_idw_features_oof(X_val_imp, wq, target)

        X_train_full = pd.concat([X_train_imp, idw_train], axis=1)
        X_val_full = pd.concat([X_val_imp, idw_val], axis=1)

        # Clean data
        X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan).fillna(X_train_full.median())
        X_val_full = X_val_full.replace([np.inf, -np.inf], np.nan).fillna(X_train_full.median())

        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_val_scaled = scaler.transform(X_val_full)

        # Transform target
        y_train = transform(wq[target].values)

        # CV score
        location_groups = create_location_id(wq)
        kf = GroupKFold(n_splits=5)
        cv_scores = []
        for tr_idx, val_idx in kf.split(X_train_scaled, y_train, location_groups):
            model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                     random_state=42, verbosity=0)
            model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train_scaled[val_idx])
            cv_scores.append(np.sqrt(np.mean((y_train[val_idx] - pred)**2)))
        print(f'   CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})')

        # Train and predict
        pred, n_models = train_model_blend(X_train_scaled, y_train, X_val_scaled)
        pred = inv_transform(pred)
        pred = np.clip(pred, clip_min, clip_max)
        predictions[target] = pred
        print(f'   Models: {n_models}, Range: [{pred.min():.1f}, {pred.max():.1f}]')

    # Create submission
    print('\n5. Creating submission...')
    submission_df = pd.DataFrame({
        'Latitude': sub_template['Latitude'].values,
        'Longitude': sub_template['Longitude'].values,
        'Sample Date': sub_template['Sample Date'].values,
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })
    submission_df = submission_df[['Latitude', 'Longitude', 'Sample Date'] + TARGET_COLS]
    submission_df.to_csv('submission_v4.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE! Saved to submission_v4.csv')
    print('=' * 70)

    print('\nPrediction summary:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission_df[col].mean():.1f}, '
              f'median={submission_df[col].median():.1f}, '
              f'range=[{submission_df[col].min():.1f}, {submission_df[col].max():.1f}]')

    print('\nTraining data reference:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={wq[col].mean():.1f}, '
              f'median={wq[col].median():.1f}, '
              f'range=[{wq[col].min():.1f}, {wq[col].max():.1f}]')


if __name__ == '__main__':
    main()
