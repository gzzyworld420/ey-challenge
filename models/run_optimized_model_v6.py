#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v6.0
Feature-space similarity: Find training points with similar spectral signatures.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
SEEDS = [42, 123, 456, 789, 2024]


def engineer_features(wq_df, ls_df, tc_df):
    """Create features."""
    df = pd.DataFrame()

    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values
    eps = 1e-10

    # Spatial
    df['Latitude'] = lat
    df['Longitude'] = lon
    df['lat_lon'] = lat * lon

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

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['turbidity'] = nir / (green + eps)
    df['nir_swir'] = nir / (swir16 + eps)
    df['band_sum'] = nir + green + swir16 + swir22

    # TerraClimate
    df['pet'] = tc_df['pet'].values

    return df


def get_spectral_features(ls_df, tc_df):
    """Get key spectral features for similarity matching."""
    eps = 1e-10
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    feats = np.column_stack([
        ls_df['NDMI'].values,
        ls_df['MNDWI'].values,
        (green - nir) / (green + nir + eps),  # NDWI
        nir / (green + eps),  # turbidity
        tc_df['pet'].values,
    ])
    return feats


def add_feature_space_neighbors(X_train, X_val, y_train_dict, spectral_train, spectral_val, k=20):
    """Add features based on similar training points in spectral space."""
    # Normalize spectral features
    scaler = StandardScaler()
    spec_train_norm = scaler.fit_transform(np.nan_to_num(spectral_train, nan=0))
    spec_val_norm = scaler.transform(np.nan_to_num(spectral_val, nan=0))

    # Build KD-tree in spectral space
    tree = cKDTree(spec_train_norm)
    dists, idxs = tree.query(spec_val_norm, k=k)

    new_feats = {}

    for target, y in y_train_dict.items():
        short = target[:2]
        for ki in [5, 10, 20]:
            ki_use = min(ki, k)
            weights = 1.0 / (dists[:, :ki_use] + 0.01)
            weights = weights / weights.sum(axis=1, keepdims=True)

            # Weighted mean of similar training targets
            new_feats[f'spec_nn_{short}_k{ki}'] = np.sum(weights * y[idxs[:, :ki_use]], axis=1)

        # Nearest neighbor
        new_feats[f'spec_nn1_{short}'] = y[idxs[:, 0]]

    # Distance to nearest spectral neighbor
    new_feats['spec_dist_nearest'] = dists[:, 0]
    new_feats['spec_dist_mean5'] = np.mean(dists[:, :5], axis=1)

    return pd.DataFrame(new_feats)


def train_ensemble(X_train, y_train, X_val):
    """Train ensemble."""
    preds = []

    # XGBoost
    for seed in SEEDS:
        for cfg in [
            dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
                 colsample_bytree=0.5, min_child_weight=15, reg_alpha=1.0, reg_lambda=5.0),
            dict(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.75,
                 colsample_bytree=0.6, min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0),
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

    return np.mean(preds, axis=0), len(preds)


def main():
    print('=' * 70)
    print('EY Challenge 2026 - Water Quality v6.0 (Spectral Similarity)')
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
    print('\n2. Engineering features...')
    X_train_base = engineer_features(wq, ls_train, tc_train)
    X_val_base = engineer_features(sub_template, ls_val, tc_val)

    # Impute base features
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    X_train_base = pd.DataFrame(imputer.fit_transform(X_train_base), columns=X_train_base.columns)
    X_val_base = pd.DataFrame(imputer.transform(X_val_base), columns=X_val_base.columns)

    # Spectral features for similarity
    spectral_train = get_spectral_features(ls_train, tc_train)
    spectral_val = get_spectral_features(ls_val, tc_val)

    # Impute spectral
    spectral_train = np.nan_to_num(spectral_train, nan=0)
    spectral_val = np.nan_to_num(spectral_val, nan=0)

    # Transforms
    configs = {
        'Total Alkalinity': (np.log1p, np.expm1, 5, 360),
        'Electrical Conductance': (np.log1p, np.expm1, 15, 1500),
        'Dissolved Reactive Phosphorus': (np.log1p, np.expm1, 5, 195),
    }

    # Prepare target dict
    y_train_dict = {t: wq[t].values for t in TARGET_COLS}

    # Add spectral-similarity features for validation
    print('\n3. Adding spectral similarity features...')
    spec_feats_val = add_feature_space_neighbors(
        X_train_base, X_val_base, y_train_dict,
        spectral_train, spectral_val, k=30
    )

    # For training, we need to be careful about leakage
    # Use leave-one-out or different samples
    print('   Computing OOF spectral features for training...')
    spec_feats_train = pd.DataFrame(index=range(len(X_train_base)))

    for target in TARGET_COLS:
        short = target[:2]
        y = y_train_dict[target]

        # For each training sample, find neighbors excluding itself
        scaler = StandardScaler()
        spec_norm = scaler.fit_transform(spectral_train)
        tree = cKDTree(spec_norm)

        for ki in [5, 10, 20]:
            vals = []
            for i in range(len(spec_norm)):
                dists, idxs = tree.query(spec_norm[i], k=ki+1)
                # Exclude self (first neighbor)
                dists, idxs = dists[1:ki+1], idxs[1:ki+1]
                w = 1.0 / (dists + 0.01)
                w = w / w.sum()
                vals.append(np.sum(w * y[idxs]))
            spec_feats_train[f'spec_nn_{short}_k{ki}'] = vals

        # NN1 (second closest, since first is self)
        nn1_vals = []
        for i in range(len(spec_norm)):
            _, idxs = tree.query(spec_norm[i], k=2)
            nn1_vals.append(y[idxs[1]])
        spec_feats_train[f'spec_nn1_{short}'] = nn1_vals

    # Distance features for training
    spec_feats_train['spec_dist_nearest'] = 0.0  # placeholder
    spec_feats_train['spec_dist_mean5'] = 0.0

    # Combine
    X_train_full = pd.concat([X_train_base, spec_feats_train], axis=1)
    X_val_full = pd.concat([X_val_base, spec_feats_val], axis=1)

    print(f'   Total features: {X_train_full.shape[1]}')

    # Clean
    X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan).fillna(X_train_full.median())
    X_val_full = X_val_full.replace([np.inf, -np.inf], np.nan).fillna(X_train_full.median())

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_val_scaled = scaler.transform(X_val_full)

    # CV
    print('\n4. Cross-validation...')
    loc_groups = wq['Latitude'].round(1).astype(str) + '_' + wq['Longitude'].round(1).astype(str)

    for target, (tf, _, _, _) in configs.items():
        y = tf(wq[target].values)
        kf = GroupKFold(n_splits=5)
        scores = []
        for tr, val in kf.split(X_train_scaled, y, loc_groups):
            m = xgb.XGBRegressor(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
            m.fit(X_train_scaled[tr], y[tr])
            scores.append(np.sqrt(np.mean((y[val] - m.predict(X_train_scaled[val]))**2)))
        print(f'   {target}: CV RMSE = {np.mean(scores):.4f}')

    # Train and predict
    print('\n5. Training final models...')
    predictions = {}

    for target, (transform, inv_transform, clip_min, clip_max) in configs.items():
        print(f'\n   {target}:')
        y_train = transform(wq[target].values)

        pred, n = train_ensemble(X_train_scaled, y_train, X_val_scaled)
        pred = inv_transform(pred)
        pred = np.clip(pred, clip_min, clip_max)

        predictions[target] = pred
        print(f'      Models: {n}, Pred range: [{pred.min():.1f}, {pred.max():.1f}]')
        print(f'      Train range: [{wq[target].min():.1f}, {wq[target].max():.1f}]')

    # Save
    print('\n6. Saving...')
    submission = pd.DataFrame({
        'Latitude': sub_template['Latitude'],
        'Longitude': sub_template['Longitude'],
        'Sample Date': sub_template['Sample Date'],
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })
    submission = submission[['Latitude', 'Longitude', 'Sample Date'] + TARGET_COLS]
    submission.to_csv('submission_v6.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE! Saved to submission_v6.csv')
    print('=' * 70)

    for col in TARGET_COLS:
        print(f'{col}: pred_mean={submission[col].mean():.1f}, train_mean={wq[col].mean():.1f}')


if __name__ == '__main__':
    main()
