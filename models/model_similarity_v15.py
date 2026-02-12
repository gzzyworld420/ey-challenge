#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v15.0
ENVIRONMENTAL SIMILARITY + SPATIAL INTERPOLATION

Key insight:
- Pure spatial interpolation overfits (training locations != validation locations)
- Pure environmental features underfit (lose too much predictive power)
- Solution: Find training samples that are ENVIRONMENTALLY SIMILAR to validation

Strategy:
1. Create environmental characteristics for each location
2. For validation points, find training points with similar environments
3. Use weighted interpolation based on environmental similarity
4. Combine with ML model that uses environmental features
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']


def estimate_elevation(lat, lon):
    """Estimate elevation based on SA geography."""
    dist_east_coast = max(0, 30 - lon)
    dist_south_coast = max(0, lat + 34)
    base_elev = 200 + dist_east_coast * 80 + dist_south_coast * 50
    if lon < 30 and lat > -32:
        base_elev += 300
    return min(base_elev, 2000)


def create_environment_vector(lat, lon, month, pet, ndmi, mndwi, turbidity):
    """
    Create an environmental characteristic vector for a location/time.
    This vector represents the "type" of water body environment.
    """
    # Elevation and topography
    elevation = estimate_elevation(lat, lon)

    # Distance to coast (major factor)
    dist_coast = min(max(0, 30 - lon) * 111, max(0, lat + 34) * 111)

    # Climate zone (based on lat/lon)
    is_western = 1 if lon < 22 else 0
    is_eastern = 1 if lon > 28 else 0
    is_subtropical = 1 if lat > -30 else 0

    # Seasonal
    is_wet_season = 1 if month in [10, 11, 12, 1, 2, 3] else 0

    # Fill NaN with medians
    ndmi = ndmi if not np.isnan(ndmi) else 0
    mndwi = mndwi if not np.isnan(mndwi) else 0
    turbidity = turbidity if not np.isnan(turbidity) else 1
    pet = pet if not np.isnan(pet) else 100

    return np.array([
        elevation / 1000,           # Normalized elevation
        dist_coast / 500,           # Normalized distance to coast
        is_western,
        is_eastern,
        is_subtropical,
        is_wet_season,
        pet / 150,                  # Normalized PET
        ndmi,                       # Already normalized (-1 to 1)
        mndwi,                      # Already normalized
        turbidity / 3,              # Normalized turbidity
    ])


def find_similar_training_samples(val_env, train_envs, train_values, k=50):
    """
    Find training samples with similar environmental characteristics.
    Returns weighted prediction based on environmental similarity.
    """
    # Compute distances in environmental space
    distances = cdist([val_env], train_envs, metric='euclidean')[0]

    # Get k nearest in environmental space
    k_use = min(k, len(train_envs))
    nearest_idx = np.argsort(distances)[:k_use]
    nearest_dist = distances[nearest_idx]

    # Compute weights (inverse distance in environmental space)
    weights = 1.0 / (nearest_dist + 0.1)
    weights = weights / weights.sum()

    # Weighted prediction
    prediction = np.sum(weights * train_values[nearest_idx])

    return prediction


def create_full_features(wq_df, ls_df, tc_df, train_wq=None, train_ls=None, train_tc=None, target=None):
    """Create comprehensive features combining environmental and spatial."""
    df = pd.DataFrame()
    eps = 1e-10
    n = len(wq_df)

    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values

    # === COORDINATES (still important) ===
    df['Latitude'] = lat
    df['Longitude'] = lon

    # === ENVIRONMENTAL FEATURES ===
    elevations = np.array([estimate_elevation(la, lo) for la, lo in zip(lat, lon)])
    df['elevation'] = elevations
    df['dist_coast'] = np.array([min(max(0, 30 - lo) * 111, max(0, la + 34) * 111)
                                  for la, lo in zip(lat, lon)])

    # Regional
    df['is_western'] = (lon < 22).astype(float)
    df['is_eastern'] = (lon > 28).astype(float)
    df['is_subtropical'] = (lat > -30).astype(float)

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

    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['turbidity'] = nir / (green + eps)

    # === CLIMATE ===
    pet = tc_df['pet'].values.astype(float)
    df['pet'] = pet

    # === ENVIRONMENTAL SIMILARITY-BASED FEATURES ===
    if train_wq is not None and target is not None:
        print(f'      Computing environmental similarity features for {target}...')

        # Create environment vectors for training
        train_dates = pd.to_datetime(train_wq['Sample Date'], format='%d-%m-%Y')
        train_months = train_dates.dt.month.values

        train_nir = train_ls['nir'].values.astype(float)
        train_green = train_ls['green'].values.astype(float)
        train_turbidity = train_nir / (train_green + eps)

        train_envs = np.array([
            create_environment_vector(
                train_wq['Latitude'].values[i],
                train_wq['Longitude'].values[i],
                train_months[i],
                train_tc['pet'].values[i],
                train_ls['NDMI'].values[i],
                train_ls['MNDWI'].values[i],
                train_turbidity[i]
            )
            for i in range(len(train_wq))
        ])

        train_values = train_wq[target].values

        # Create environment vectors for current data
        current_dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
        current_months = current_dates.dt.month.values
        current_turbidity = nir / (green + eps)

        # Environmental similarity predictions
        env_preds = []
        env_preds_k10 = []
        env_preds_k100 = []

        for i in range(n):
            val_env = create_environment_vector(
                lat[i], lon[i],
                current_months[i],
                pet[i],
                ls_df['NDMI'].values[i],
                ls_df['MNDWI'].values[i],
                current_turbidity[i]
            )

            # Different k values
            env_preds.append(find_similar_training_samples(val_env, train_envs, train_values, k=30))
            env_preds_k10.append(find_similar_training_samples(val_env, train_envs, train_values, k=10))
            env_preds_k100.append(find_similar_training_samples(val_env, train_envs, train_values, k=100))

        df['env_sim_pred'] = env_preds
        df['env_sim_pred_k10'] = env_preds_k10
        df['env_sim_pred_k100'] = env_preds_k100

        # Also add spatial IDW for comparison
        train_coords = train_wq[['Latitude', 'Longitude']].values
        val_coords = np.column_stack([lat, lon])

        tree = cKDTree(train_coords)
        dists, idxs = tree.query(val_coords, k=min(30, len(train_coords)))
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        weights = weights / weights.sum(axis=1, keepdims=True)
        spatial_idw = np.sum(weights * train_values[idxs], axis=1)
        df['spatial_idw'] = spatial_idw

    # === INTERACTIONS ===
    df['elev_pet'] = df['elevation'] * pet
    df['coast_ndmi'] = df['dist_coast'] * df['NDMI']
    df['wet_turbidity'] = df['is_wet_season'] * df['turbidity']

    return df


def main():
    print('=' * 70)
    print('EY Water Quality v15.0 - Environmental Similarity')
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

    # Location groups
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

        # Create features with environmental similarity
        print('   Creating features...')
        X_train_raw = create_full_features(wq, ls_train, tc_train, wq, ls_train, tc_train, target)
        X_val_raw = create_full_features(sub_template, ls_val, tc_val, wq, ls_train, tc_train, target)

        print(f'   Features: {X_train_raw.shape[1]}')

        # Preprocess
        X_train = X_train_raw.fillna(X_train_raw.median())
        X_val = X_val_raw.fillna(X_train_raw.median())
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        y_raw = wq[target].values
        y_train = cfg['transform'](y_raw)

        # Cross-validation
        print('   Cross-validation...')
        kf = GroupKFold(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in kf.split(X_train_scaled, y_train, location_groups):
            X_tr, X_va = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_child_weight=15, reg_alpha=0.5, reg_lambda=3.0,
                random_state=42, n_jobs=-1, verbosity=0
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)

            ss_res = np.sum((y_va - pred) ** 2)
            ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            cv_scores.append(r2)

        print(f'   CV R²: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}')

        # Train ensemble
        print('   Training ensemble...')
        all_preds = []

        for seed in [42, 123, 456]:
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_child_weight=15, reg_alpha=0.5, reg_lambda=3.0,
                random_state=seed, n_jobs=-1, verbosity=0
            )
            model.fit(X_train_scaled, y_train)
            all_preds.append(cfg['inverse'](model.predict(X_val_scaled)))

            if HAS_LIGHTGBM:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    min_child_samples=20, reg_alpha=0.5, reg_lambda=3.0,
                    random_state=seed, n_jobs=-1, verbose=-1
                )
                lgb_model.fit(X_train_scaled, y_train)
                all_preds.append(cfg['inverse'](lgb_model.predict(X_val_scaled)))

        # Also use direct environmental similarity prediction
        env_sim_pred = X_val_raw['env_sim_pred'].values
        all_preds.append(env_sim_pred)

        # Blend
        ml_pred = np.mean(all_preds[:-1], axis=0)
        final_pred = 0.6 * ml_pred + 0.4 * env_sim_pred
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

    submission.to_csv('submission_v15.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print('\nSubmission: submission_v15.csv')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}')


if __name__ == '__main__':
    main()
