#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v8.0
Regional focus + Spatial interpolation approach.

Key insights:
1. Validation locations have ZERO overlap with training
2. Spectral features have weak correlations with targets
3. High spatial autocorrelation - nearby locations have similar values
4. Only ~12% of training data is from the validation region

Strategy:
- Use regional training data + global data with weighting
- Improved spatial interpolation features
- Ensemble combining ML with spatial kriging-like approaches
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']


def spatial_idw_prediction(val_coords, train_coords, train_values, k=10, power=2):
    """
    Inverse Distance Weighting interpolation.
    Returns interpolated values for validation points.
    """
    tree = cKDTree(train_coords)
    dists, idxs = tree.query(val_coords, k=min(k, len(train_coords)))

    if len(dists.shape) == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    # Handle zero distances
    dists = np.maximum(dists, 1e-10)

    # IDW weights
    weights = 1.0 / (dists ** power)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Interpolate
    pred = np.sum(weights * train_values[idxs], axis=1)

    return pred


def create_features(wq_df, ls_df, tc_df, is_train=True, train_wq=None):
    """Create features for modeling."""
    df = pd.DataFrame()
    eps = 1e-10

    # === COORDINATES ===
    df['Latitude'] = wq_df['Latitude'].values
    df['Longitude'] = wq_df['Longitude'].values

    # Relative to validation region center
    val_center_lat = -33.0
    val_center_lon = 26.4
    df['lat_rel'] = df['Latitude'] - val_center_lat
    df['lon_rel'] = df['Longitude'] - val_center_lon
    df['dist_center'] = np.sqrt(df['lat_rel']**2 + df['lon_rel']**2)

    # === TEMPORAL ===
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    df['month'] = dates.dt.month
    df['day_of_year'] = dates.dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Seasons
    df['is_summer'] = df['month'].isin([12, 1, 2]).astype(float)
    df['is_autumn'] = df['month'].isin([3, 4, 5]).astype(float)
    df['is_winter'] = df['month'].isin([6, 7, 8]).astype(float)
    df['is_spring'] = df['month'].isin([9, 10, 11]).astype(float)

    # === SPECTRAL FEATURES ===
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

    # Log transforms
    df['nir_log'] = np.log1p(np.maximum(nir, 0))
    df['green_log'] = np.log1p(np.maximum(green, 0))
    df['swir16_log'] = np.log1p(np.maximum(swir16, 0))
    df['swir22_log'] = np.log1p(np.maximum(swir22, 0))

    # Key indices
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['turbidity'] = nir / (green + eps)
    df['sediment'] = (swir16 - green) / (swir16 + green + eps)

    # Ratios
    df['swir_ratio'] = swir22 / (swir16 + eps)
    df['nir_swir16'] = nir / (swir16 + eps)
    df['green_swir16'] = green / (swir16 + eps)

    # Normalized differences
    df['nd_nir_swir16'] = (nir - swir16) / (nir + swir16 + eps)
    df['nd_nir_swir22'] = (nir - swir22) / (nir + swir22 + eps)

    # Statistics
    bands = np.column_stack([nir, green, swir16, swir22])
    df['reflect_mean'] = np.nanmean(bands, axis=1)
    df['reflect_std'] = np.nanstd(bands, axis=1)
    df['reflect_range'] = np.nanmax(bands, axis=1) - np.nanmin(bands, axis=1)

    # === TERRACLIMATE ===
    pet = tc_df['pet'].values.astype(float)
    df['pet'] = pet
    df['pet_log'] = np.log1p(np.maximum(pet, 0))

    # === INTERACTIONS ===
    df['pet_ndmi'] = pet * df['NDMI']
    df['lat_ndmi'] = df['lat_rel'] * df['NDMI']
    df['lon_ndmi'] = df['lon_rel'] * df['NDMI']
    df['summer_ndmi'] = df['is_summer'] * df['NDMI']
    df['winter_ndmi'] = df['is_winter'] * df['NDMI']

    return df


def create_spatial_features(X_df, train_wq, target_name, k_values=[3, 5, 10, 20]):
    """Create spatial interpolation features for a specific target."""
    coords = X_df[['Latitude', 'Longitude']].values

    # Group training data by location
    loc_stats = train_wq.groupby(['Latitude', 'Longitude'])[target_name].agg(
        ['mean', 'median', 'std', 'min', 'max', 'count']
    ).reset_index()

    train_coords = loc_stats[['Latitude', 'Longitude']].values
    tree = cKDTree(train_coords)

    max_k = min(max(k_values), len(train_coords))
    dists, idxs = tree.query(coords, k=max_k)

    if len(dists.shape) == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    features = {}

    # Distance features
    features['dist_nearest'] = dists[:, 0]
    features['dist_5th'] = dists[:, min(4, max_k-1)] if max_k >= 5 else dists[:, -1]

    # IDW predictions with different k and power
    for stat in ['mean', 'median']:
        values = loc_stats[stat].values

        for k in k_values:
            k_use = min(k, max_k)
            for power in [1, 2, 3]:
                d = np.maximum(dists[:, :k_use], 1e-10)
                w = 1.0 / (d ** power)
                w = w / w.sum(axis=1, keepdims=True)
                idw = np.sum(w * values[idxs[:, :k_use]], axis=1)
                features[f'idw_{stat[:3]}_k{k}_p{power}'] = idw

    # Nearest neighbor features
    features['nn1_mean'] = loc_stats['mean'].values[idxs[:, 0]]
    features['nn1_median'] = loc_stats['median'].values[idxs[:, 0]]

    if max_k >= 3:
        features['nn3_mean'] = np.mean(
            loc_stats['mean'].values[idxs[:, :min(3, max_k)]], axis=1
        )

    # Local variability
    features['local_std'] = loc_stats['std'].values[idxs[:, 0]]
    features['local_range'] = (
        loc_stats['max'].values[idxs[:, 0]] - loc_stats['min'].values[idxs[:, 0]]
    )

    return pd.DataFrame(features)


def train_ensemble(X_train, y_train, X_val, seeds=[42, 123, 456]):
    """Train diverse ensemble."""
    all_preds = []

    for seed in seeds:
        # XGBoost
        for cfg in [
            dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
                 colsample_bytree=0.5, min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0),
            dict(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.75,
                 colsample_bytree=0.6, min_child_weight=8),
        ]:
            model = xgb.XGBRegressor(**cfg, random_state=seed, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)
            all_preds.append(model.predict(X_val))

        # LightGBM
        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.5, min_child_samples=15,
                reg_alpha=0.5, reg_lambda=2.0, random_state=seed, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            all_preds.append(lgb_model.predict(X_val))

        # CatBoost
        if HAS_CATBOOST:
            cat_model = CatBoostRegressor(
                iterations=500, depth=5, learning_rate=0.02,
                l2_leaf_reg=3.0, random_seed=seed, verbose=0
            )
            cat_model.fit(X_train, y_train)
            all_preds.append(cat_model.predict(X_val))

        # ExtraTrees
        et = ExtraTreesRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=5,
            max_features=0.5, random_state=seed, n_jobs=-1
        )
        et.fit(X_train, y_train)
        all_preds.append(et.predict(X_val))

    return np.mean(all_preds, axis=0)


def main():
    print('=' * 70)
    print('EY Water Quality Prediction v8.0 - Regional + Spatial Focus')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Total training: {len(wq)}')

    # Identify regional training data
    val_lat_min, val_lat_max = sub_template['Latitude'].min(), sub_template['Latitude'].max()
    val_lon_min, val_lon_max = sub_template['Longitude'].min(), sub_template['Longitude'].max()

    margin = 3.0
    regional_mask = (
        (wq['Latitude'] >= val_lat_min - margin) &
        (wq['Latitude'] <= val_lat_max + margin) &
        (wq['Longitude'] >= val_lon_min - margin) &
        (wq['Longitude'] <= val_lon_max + margin)
    )
    print(f'   Regional training: {regional_mask.sum()}')

    # Create base features
    print('\n2. Creating features...')
    X_train_base = create_features(wq, ls_train, tc_train)
    X_val_base = create_features(sub_template, ls_val, tc_val)
    print(f'   Base features: {X_train_base.shape[1]}')

    # Impute
    print('\n3. Imputing missing values...')
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_base),
        columns=X_train_base.columns
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val_base),
        columns=X_val_base.columns
    )

    # Handle inf/nan
    X_train_imp = X_train_imp.replace([np.inf, -np.inf], np.nan).fillna(X_train_imp.median())
    X_val_imp = X_val_imp.replace([np.inf, -np.inf], np.nan).fillna(X_train_imp.median())

    # Target configs
    target_configs = {
        'Total Alkalinity': {'transform': np.log1p, 'inverse': np.expm1, 'clip': (0, 500)},
        'Electrical Conductance': {'transform': np.log1p, 'inverse': np.expm1, 'clip': (0, 2000)},
        'Dissolved Reactive Phosphorus': {
            'transform': lambda x: np.log1p(x + 1),
            'inverse': lambda x: np.expm1(x) - 1,
            'clip': (0, 300)
        },
    }

    predictions = {}

    for target in TARGET_COLS:
        print(f'\n4. Processing {target}...')
        cfg = target_configs[target]

        # Add spatial features for this target
        print('   Adding spatial features...')
        spatial_train = create_spatial_features(X_train_imp, wq, target)
        spatial_val = create_spatial_features(X_val_imp, wq, target)

        X_train_full = pd.concat([X_train_imp.reset_index(drop=True),
                                   spatial_train.reset_index(drop=True)], axis=1)
        X_val_full = pd.concat([X_val_imp.reset_index(drop=True),
                                 spatial_val.reset_index(drop=True)], axis=1)

        print(f'   Total features: {X_train_full.shape[1]}')

        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_val_scaled = scaler.transform(X_val_full)

        # Prepare targets
        y_train = cfg['transform'](wq[target].values)

        # Compute pure IDW prediction for reference
        train_coords = wq[['Latitude', 'Longitude']].values
        val_coords = sub_template[['Latitude', 'Longitude']].values
        idw_pred = spatial_idw_prediction(
            val_coords, train_coords, wq[target].values, k=15, power=2
        )
        print(f'   IDW pred range: [{idw_pred.min():.2f}, {idw_pred.max():.2f}]')

        # Train ML ensemble
        print('   Training ensemble...')
        ml_pred_transformed = train_ensemble(X_train_scaled, y_train, X_val_scaled)
        ml_pred = cfg['inverse'](ml_pred_transformed)
        print(f'   ML pred range: [{ml_pred.min():.2f}, {ml_pred.max():.2f}]')

        # Blend ML and IDW predictions
        # Give more weight to IDW since spatial correlation is high
        blend_weight = 0.5  # weight for ML, (1-weight) for IDW
        final_pred = blend_weight * ml_pred + (1 - blend_weight) * idw_pred
        final_pred = np.clip(final_pred, cfg['clip'][0], cfg['clip'][1])

        predictions[target] = final_pred
        print(f'   Final range: [{final_pred.min():.2f}, {final_pred.max():.2f}], mean={final_pred.mean():.2f}')

    # Create submission
    print('\n5. Creating submission...')
    submission = pd.DataFrame({
        'Latitude': sub_template['Latitude'],
        'Longitude': sub_template['Longitude'],
        'Sample Date': sub_template['Sample Date'],
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })

    submission.to_csv('submission_v8.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print('\nSubmission: submission_v8.csv')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}')


if __name__ == '__main__':
    main()
