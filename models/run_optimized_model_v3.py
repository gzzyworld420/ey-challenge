#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v3.0
More robust model - less overfitting, no target leakage.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Constants
TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
SEEDS = [42, 123, 456, 789, 2024]


def create_location_id(df, precision=2):
    """Create unique location identifier."""
    return df['Latitude'].round(precision).astype(str) + '_' + df['Longitude'].round(precision).astype(str)


def engineer_features(wq_df, ls_df, tc_df):
    """Create robust features without target leakage."""
    df = pd.DataFrame()

    # ============ SPATIAL FEATURES ============
    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values

    df['Latitude'] = lat
    df['Longitude'] = lon
    df['lat_lon_interact'] = lat * lon
    df['lat_sq'] = lat ** 2
    df['lon_sq'] = lon ** 2

    # Distance from center of South Africa (approximate)
    center_lat, center_lon = -29.0, 25.0
    df['dist_center'] = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)

    # Coastal proximity (west coast approximation)
    df['dist_west_coast'] = np.abs(lon - 18)
    df['dist_east_coast'] = np.abs(lon - 32)

    # ============ TEMPORAL FEATURES ============
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    df['month'] = dates.dt.month
    df['year'] = dates.dt.year
    df['day_of_year'] = dates.dt.dayofyear

    # Cyclic encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Season (Southern Hemisphere)
    df['is_wet_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    df['is_dry_season'] = df['month'].isin([4, 5, 6, 7, 8, 9]).astype(int)

    # ============ LANDSAT BANDS ============
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22

    # ============ SPECTRAL INDICES ============
    eps = 1e-10

    # Provided indices
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values

    # Water indices
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['AWEI'] = 4 * green - 0.25 * nir - 2.75 * swir16 - 2.75 * swir22

    # Key ratios
    df['nir_green_ratio'] = nir / (green + eps)
    df['swir_ratio'] = swir22 / (swir16 + eps)
    df['nir_swir_ratio'] = nir / (swir16 + eps)

    # Normalized differences
    df['nd_nir_swir22'] = (nir - swir22) / (nir + swir22 + eps)
    df['nd_green_swir16'] = (green - swir16) / (green + swir16 + eps)

    # Band statistics
    all_bands = np.column_stack([nir, green, swir16, swir22])
    df['total_reflectance'] = np.nansum(all_bands, axis=1)
    df['mean_reflectance'] = np.nanmean(all_bands, axis=1)
    df['std_reflectance'] = np.nanstd(all_bands, axis=1)

    # ============ TERRACLIMATE ============
    pet = tc_df['pet'].values
    df['pet'] = pet

    # ============ KEY INTERACTIONS ============
    df['pet_ndmi'] = pet * df['NDMI']
    df['lat_ndmi'] = lat * df['NDMI']
    df['lat_pet'] = lat * pet
    df['month_pet'] = df['month'] * pet

    return df


def add_spatial_neighbor_features(X_df, train_wq, train_X, k_neighbors=10):
    """Add spatial features based on neighboring FEATURES (not targets)."""
    coords = X_df[['Latitude', 'Longitude']].values
    train_coords = train_wq[['Latitude', 'Longitude']].values

    tree = cKDTree(train_coords)
    dists, idxs = tree.query(coords, k=min(k_neighbors, len(train_coords)))

    new_feats = pd.DataFrame(index=range(len(X_df)))

    # Distance features only
    new_feats['dist_nearest_train'] = dists[:, 0]
    new_feats['dist_5th_train'] = dists[:, min(4, dists.shape[1]-1)]
    new_feats['dist_mean_5'] = np.mean(dists[:, :min(5, dists.shape[1])], axis=1)

    # Density feature
    new_feats['local_density'] = 1.0 / (np.mean(dists[:, :5], axis=1) + 0.01)

    return new_feats


def get_cv_predictions(X, y, model_class, model_params, n_splits=5, groups=None):
    """Get out-of-fold predictions for blending."""
    oof_preds = np.zeros(len(y))

    if groups is not None:
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, y, groups))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X))

    for train_idx, val_idx in splits:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    rmse = np.sqrt(np.mean((y - oof_preds) ** 2))
    return oof_preds, rmse


def train_and_predict(X_train, y_train, X_val, seed=42):
    """Train diverse models and return weighted predictions."""
    all_preds = []
    weights = []

    # 1. XGBoost - conservative settings
    xgb_params = [
        dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.7,
             colsample_bytree=0.6, min_child_weight=20, reg_alpha=2.0, reg_lambda=10.0),
        dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.65,
             colsample_bytree=0.5, min_child_weight=25, reg_alpha=3.0, reg_lambda=15.0),
    ]

    for params in xgb_params:
        for s in [seed, seed+100, seed+200]:
            model = xgb.XGBRegressor(**params, random_state=s, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)
            all_preds.append(model.predict(X_val))
            weights.append(1.0)

    # 2. LightGBM
    if HAS_LIGHTGBM:
        lgb_params = dict(
            n_estimators=400, max_depth=4, learning_rate=0.025,
            subsample=0.7, colsample_bytree=0.6, min_child_samples=30,
            reg_alpha=2.0, reg_lambda=10.0, verbose=-1
        )
        for s in [seed, seed+100]:
            model = lgb.LGBMRegressor(**lgb_params, random_state=s, n_jobs=-1)
            model.fit(X_train, y_train)
            all_preds.append(model.predict(X_val))
            weights.append(1.0)

    # 3. HistGradientBoosting (scikit-learn's fast GB)
    for s in [seed, seed+100]:
        model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=4, learning_rate=0.03,
            min_samples_leaf=30, l2_regularization=5.0, random_state=s
        )
        model.fit(X_train, y_train)
        all_preds.append(model.predict(X_val))
        weights.append(0.8)

    # 4. ExtraTrees - very regularized
    for s in [seed, seed+100]:
        model = ExtraTreesRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            max_features=0.4, random_state=s, n_jobs=-1
        )
        model.fit(X_train, y_train)
        all_preds.append(model.predict(X_val))
        weights.append(0.7)

    # 5. KNN - spatial interpolation baseline
    for k in [5, 10, 15]:
        model = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
        model.fit(X_train, y_train)
        all_preds.append(model.predict(X_val))
        weights.append(0.5)

    # Weighted average
    weights = np.array(weights)
    weights = weights / weights.sum()

    final_pred = np.zeros(len(X_val))
    for pred, w in zip(all_preds, weights):
        final_pred += w * pred

    return final_pred, len(all_preds)


def main():
    print('=' * 70)
    print('EY AI & Data Challenge 2026 - Water Quality Prediction v3.0')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Training samples: {wq.shape[0]}')
    print(f'   Validation samples: {sub_template.shape[0]}')

    # Build features
    print('\n2. Engineering features (simplified)...')
    X_train_base = engineer_features(wq, ls_train, tc_train)
    X_val_base = engineer_features(sub_template, ls_val, tc_val)
    print(f'   Base features: {X_train_base.shape[1]}')

    # Impute missing values
    print('\n3. Imputing missing values...')
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_base), columns=X_train_base.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val_base), columns=X_val_base.columns)

    # Add spatial neighbor features (without target leakage)
    print('\n4. Adding spatial neighbor features...')
    spatial_train = add_spatial_neighbor_features(X_train_imp, wq, X_train_imp)
    spatial_val = add_spatial_neighbor_features(X_val_imp, wq, X_train_imp)

    X_train_full = pd.concat([X_train_imp, spatial_train], axis=1)
    X_val_full = pd.concat([X_val_imp, spatial_val], axis=1)
    print(f'   Total features: {X_train_full.shape[1]}')

    # Handle NaN/Inf
    X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan).fillna(X_train_full.median())
    X_val_full = X_val_full.replace([np.inf, -np.inf], np.nan).fillna(X_train_full.median())

    # Scale features
    print('\n5. Scaling features...')
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_val_scaled = scaler.transform(X_val_full)

    # Location groups for CV
    location_groups = create_location_id(wq)

    # Target configurations
    target_configs = {
        'Total Alkalinity': {
            'transform': lambda x: np.log1p(x),
            'inv_transform': lambda x: np.expm1(x),
            'clip_min': 1,
            'clip_max': 500,
        },
        'Electrical Conductance': {
            'transform': lambda x: np.log1p(x),
            'inv_transform': lambda x: np.expm1(x),
            'clip_min': 10,
            'clip_max': 2000,
        },
        'Dissolved Reactive Phosphorus': {
            'transform': lambda x: np.sqrt(x + 1),
            'inv_transform': lambda x: np.clip(x, 0, None)**2 - 1,
            'clip_min': 0,
            'clip_max': 500,
        },
    }

    # Cross-validation
    print('\n6. Cross-validation (Spatial GroupKFold)...')
    for target_name, cfg in target_configs.items():
        y = cfg['transform'](wq[target_name].values)
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_child_weight=20, reg_lambda=10, random_state=42, verbosity=0
        )
        _, rmse = get_cv_predictions(X_train_scaled, y, type(model), model.get_params(),
                                      n_splits=5, groups=location_groups)
        print(f'   {target_name}: CV RMSE = {rmse:.4f}')

    # Train final models
    print('\n7. Training final ensemble...')
    predictions = {}

    for target_name, cfg in target_configs.items():
        print(f'\n   {target_name}:')
        y_train = cfg['transform'](wq[target_name].values)

        pred, n_models = train_and_predict(X_train_scaled, y_train, X_val_scaled, seed=42)

        # Inverse transform and clip
        pred = cfg['inv_transform'](pred)
        pred = np.clip(pred, cfg['clip_min'], cfg['clip_max'])

        predictions[target_name] = pred
        print(f'      Models: {n_models}, Range: [{pred.min():.2f}, {pred.max():.2f}]')

    # Create submission
    print('\n8. Creating submission file...')
    submission_df = pd.DataFrame({
        'Latitude': sub_template['Latitude'].values,
        'Longitude': sub_template['Longitude'].values,
        'Sample Date': sub_template['Sample Date'].values,
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })

    submission_df = submission_df[['Latitude', 'Longitude', 'Sample Date',
                                   'Total Alkalinity', 'Electrical Conductance',
                                   'Dissolved Reactive Phosphorus']]

    submission_df.to_csv('submission_v3.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print(f'\nSubmission saved to: submission_v3.csv')
    print('\nPrediction summary:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission_df[col].mean():.2f}, '
              f'median={submission_df[col].median():.2f}, '
              f'range=[{submission_df[col].min():.2f}, {submission_df[col].max():.2f}]')

    # Compare with training data statistics
    print('\nTraining data statistics:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={wq[col].mean():.2f}, '
              f'median={wq[col].median():.2f}, '
              f'range=[{wq[col].min():.2f}, {wq[col].max():.2f}]')


if __name__ == '__main__':
    main()
