#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Improved Water Quality Prediction v2.0
Significantly improved model with better validation, features, and ensemble.

Key improvements:
- Spatial cross-validation (GroupKFold by location)
- More diverse models: LightGBM, CatBoost, Stacking
- Extended feature engineering (100+ features)
- Better target transformations
- Temporal and location-based features
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# Try to import optional libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Note: LightGBM not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Note: CatBoost not installed. Install with: pip install catboost")

# Constants
TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
TARGET_SHORT = {'Total Alkalinity': 'TA', 'Electrical Conductance': 'EC', 'Dissolved Reactive Phosphorus': 'DRP'}
SEEDS = [42, 123, 456, 789, 2024]


def create_location_id(df):
    """Create unique location identifier for spatial grouping."""
    return df['Latitude'].round(2).astype(str) + '_' + df['Longitude'].round(2).astype(str)


def engineer_base_features(wq_df, ls_df, tc_df):
    """Create comprehensive base features from satellite and climate data."""
    df = pd.DataFrame()

    # ============ SPATIAL FEATURES ============
    df['Latitude'] = wq_df['Latitude'].values
    df['Longitude'] = wq_df['Longitude'].values

    # Distance from reference points
    df['dist_coast_approx'] = np.sqrt((df['Latitude'] + 30)**2 + (df['Longitude'] - 25)**2)
    df['dist_inland'] = np.sqrt((df['Latitude'] + 26)**2 + (df['Longitude'] - 28)**2)

    # Coordinate interactions and polynomials
    df['lat_lon_interact'] = df['Latitude'] * df['Longitude']
    df['lat_sq'] = df['Latitude'] ** 2
    df['lon_sq'] = df['Longitude'] ** 2
    df['lat_cube'] = df['Latitude'] ** 3
    df['lon_cube'] = df['Longitude'] ** 3
    df['lat_lon_sq'] = df['Latitude'] * df['Longitude']**2
    df['lat_sq_lon'] = df['Latitude']**2 * df['Longitude']

    # Polar coordinates
    df['radius'] = np.sqrt(df['Latitude']**2 + df['Longitude']**2)
    df['angle'] = np.arctan2(df['Latitude'], df['Longitude'])

    # ============ TEMPORAL FEATURES ============
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    df['month'] = dates.dt.month
    df['year'] = dates.dt.year
    df['day_of_year'] = dates.dt.dayofyear
    df['week_of_year'] = dates.dt.isocalendar().week.astype(int)
    df['quarter'] = dates.dt.quarter

    # Cyclic encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Season (Southern Hemisphere)
    df['season'] = ((df['month'] % 12) // 3)
    df['is_summer'] = (df['season'] == 0).astype(int)
    df['is_winter'] = (df['season'] == 2).astype(int)
    df['is_wet_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)

    # ============ RAW LANDSAT BANDS ============
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22

    # Log transforms
    df['nir_log'] = np.log1p(nir)
    df['green_log'] = np.log1p(green)
    df['swir16_log'] = np.log1p(swir16)
    df['swir22_log'] = np.log1p(swir22)

    # ============ SPECTRAL INDICES ============
    eps = 1e-10

    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values

    # Water indices
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['AWEI'] = 4 * green - 0.25 * nir - 2.75 * swir16 - 2.75 * swir22
    df['AWEI_sh'] = green + 2.5 * nir - 1.5 * (swir16 + swir22)

    # Turbidity and sediment
    df['turbidity_index'] = nir / (green + eps)
    df['sediment_index'] = (swir16 - green) / (swir16 + green + eps)

    # ============ BAND RATIOS ============
    df['nir_green_ratio'] = nir / (green + eps)
    df['green_nir_ratio'] = green / (nir + eps)
    df['swir22_nir_ratio'] = swir22 / (nir + eps)
    df['swir16_green_ratio'] = swir16 / (green + eps)
    df['swir22_green_ratio'] = swir22 / (green + eps)
    df['swir22_swir16_ratio'] = swir22 / (swir16 + eps)
    df['nir_swir16_ratio'] = nir / (swir16 + eps)
    df['nir_swir22_ratio'] = nir / (swir22 + eps)
    df['green_swir16_ratio'] = green / (swir16 + eps)
    df['green_swir22_ratio'] = green / (swir22 + eps)

    # ============ NORMALIZED DIFFERENCES ============
    df['nd_nir_swir22'] = (nir - swir22) / (nir + swir22 + eps)
    df['nd_nir_swir16'] = (nir - swir16) / (nir + swir16 + eps)
    df['nd_green_swir16'] = (green - swir16) / (green + swir16 + eps)
    df['nd_green_swir22'] = (green - swir22) / (green + swir22 + eps)
    df['nd_swir16_swir22'] = (swir16 - swir22) / (swir16 + swir22 + eps)

    # ============ BAND STATISTICS ============
    all_bands = np.column_stack([nir, green, swir16, swir22])
    df['total_reflectance'] = np.nansum(all_bands, axis=1)
    df['mean_reflectance'] = np.nanmean(all_bands, axis=1)
    df['std_reflectance'] = np.nanstd(all_bands, axis=1)
    df['max_reflectance'] = np.nanmax(all_bands, axis=1)
    df['min_reflectance'] = np.nanmin(all_bands, axis=1)
    df['range_reflectance'] = df['max_reflectance'] - df['min_reflectance']
    df['cv_reflectance'] = df['std_reflectance'] / (df['mean_reflectance'] + eps)

    # Band differences
    df['swir_diff'] = swir16 - swir22
    df['nir_minus_green'] = nir - green
    df['nir_minus_swir16'] = nir - swir16
    df['green_minus_swir22'] = green - swir22

    # ============ TERRACLIMATE ============
    pet = tc_df['pet'].values
    df['pet'] = pet
    df['pet_log'] = np.log1p(pet)
    df['pet_sq'] = pet ** 2

    # ============ INTERACTION FEATURES ============
    df['pet_ndmi'] = pet * df['NDMI']
    df['pet_mndwi'] = pet * df['MNDWI']
    df['pet_ndwi'] = pet * df['NDWI']
    df['pet_nir'] = pet * nir
    df['pet_turbidity'] = pet * df['turbidity_index']

    df['lat_ndmi'] = df['Latitude'] * df['NDMI']
    df['lon_ndmi'] = df['Longitude'] * df['NDMI']
    df['lat_mndwi'] = df['Latitude'] * df['MNDWI']
    df['lon_mndwi'] = df['Longitude'] * df['MNDWI']
    df['lat_pet'] = df['Latitude'] * pet
    df['lon_pet'] = df['Longitude'] * pet
    df['lat_nir'] = df['Latitude'] * nir
    df['lon_nir'] = df['Longitude'] * nir

    df['month_ndmi'] = df['month'] * df['NDMI']
    df['month_pet'] = df['month'] * pet
    df['season_ndmi'] = df['season'] * df['NDMI']
    df['wet_season_pet'] = df['is_wet_season'] * pet

    df['lat_lon_ndmi'] = df['Latitude'] * df['Longitude'] * df['NDMI']
    df['lat_pet_ndmi'] = df['Latitude'] * pet * df['NDMI']

    return df


def add_spatial_target_features(X_df, train_wq, k_values=[3, 5, 10, 15, 20, 30], powers=[1, 2, 3]):
    """Add comprehensive IDW features based on nearest training locations."""
    coords = X_df[['Latitude', 'Longitude']].values

    loc_stats = train_wq.groupby(['Latitude', 'Longitude'])[TARGET_COLS].agg(
        ['mean', 'median', 'std', 'min', 'max', 'count']
    ).reset_index()
    loc_coords = loc_stats[['Latitude', 'Longitude']].values

    tree = cKDTree(loc_coords)
    max_k = min(max(k_values), len(loc_coords))
    dists, idxs = tree.query(coords, k=max_k)

    if len(dists.shape) == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    new_feats = pd.DataFrame(index=range(len(X_df)))

    # Distance features
    new_feats['dist_nearest'] = dists[:, 0]
    new_feats['dist_nearest_log'] = np.log1p(dists[:, 0])
    if max_k >= 5:
        new_feats['dist_5th'] = dists[:, min(4, max_k-1)]
        new_feats['dist_mean_5'] = np.mean(dists[:, :min(5, max_k)], axis=1)
    if max_k >= 10:
        new_feats['dist_10th'] = dists[:, min(9, max_k-1)]
        new_feats['dist_mean_10'] = np.mean(dists[:, :min(10, max_k)], axis=1)

    for target in TARGET_COLS:
        ts = TARGET_SHORT[target]

        for stat in ['mean', 'median', 'std']:
            try:
                vals = loc_stats[(target, stat)].values
            except:
                continue

            for k in k_values:
                k_use = min(k, max_k)
                for power in powers:
                    w = 1.0 / (dists[:, :k_use] ** power + 1e-6)
                    w = w / w.sum(axis=1, keepdims=True)
                    idw = np.sum(w * vals[idxs[:, :k_use]], axis=1)
                    new_feats[f'idw_{ts}_{stat[:3]}_k{k}_p{power}'] = idw

        # Nearest neighbor values
        try:
            vals_mean = loc_stats[(target, 'mean')].values
            vals_median = loc_stats[(target, 'median')].values
            new_feats[f'nn1_{ts}_mean'] = vals_mean[idxs[:, 0]]
            new_feats[f'nn1_{ts}_median'] = vals_median[idxs[:, 0]]
            if max_k >= 3:
                new_feats[f'nn3_{ts}_mean'] = np.mean(vals_mean[idxs[:, :min(3, max_k)]], axis=1)
        except:
            pass

        # Local variability
        try:
            vals_std = loc_stats[(target, 'std')].values
            new_feats[f'local_std_{ts}'] = vals_std[idxs[:, 0]]
        except:
            pass

    return new_feats


def add_temporal_location_features(X_df, train_wq):
    """Add features based on temporal patterns at nearby locations."""
    new_feats = pd.DataFrame(index=range(len(X_df)))

    train_wq = train_wq.copy()
    train_wq['loc_id'] = create_location_id(train_wq)

    X_df = X_df.copy()
    X_df['loc_id'] = create_location_id(X_df)

    for target in TARGET_COLS:
        ts = TARGET_SHORT[target]
        loc_target_stats = train_wq.groupby('loc_id')[target].agg(['mean', 'std', 'median', 'min', 'max'])

        for stat in ['mean', 'std', 'median', 'min', 'max']:
            mapped = X_df['loc_id'].map(loc_target_stats[stat])
            fill_val = train_wq[target].mean() if stat == 'mean' else (train_wq[target].median() if stat == 'median' else 0)
            new_feats[f'loc_{ts}_{stat}'] = mapped.fillna(fill_val)

    return new_feats


def get_cv_score(X, y, model, n_splits=5, groups=None):
    """Get cross-validation score."""
    if groups is not None:
        kf = GroupKFold(n_splits=n_splits)
        splits = kf.split(X, y, groups)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = kf.split(X)

    scores = []
    for train_idx, val_idx in splits:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - pred) ** 2))
        scores.append(rmse)

    return np.mean(scores), np.std(scores)


def train_stacking_model(X_train, y_train, seed=42):
    """Create a stacking ensemble with diverse base models."""
    base_models = [
        ('xgb1', xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.5, min_child_weight=15,
            reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbosity=0
        )),
        ('xgb2', xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
            random_state=seed+1, n_jobs=-1, verbosity=0
        )),
        ('et', ExtraTreesRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            max_features=0.6, random_state=seed, n_jobs=-1
        )),
        ('rf', RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=8,
            max_features=0.5, random_state=seed, n_jobs=-1
        )),
        ('knn', KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)),
    ]

    if HAS_LIGHTGBM:
        base_models.append(('lgb', lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.5, min_child_samples=20,
            reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1
        )))

    if HAS_CATBOOST:
        base_models.append(('cat', CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.03,
            l2_leaf_reg=5.0, random_seed=seed, verbose=0
        )))

    meta_learner = Ridge(alpha=1.0)

    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    return stacking


def main():
    print('=' * 70)
    print('EY AI & Data Challenge 2026 - Water Quality Prediction v2.0')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')

    try:
        ls_val = pd.read_csv('landsat_features_validation.csv')
        tc_val = pd.read_csv('terraclimate_features_validation.csv')
        sub_template = pd.read_csv('submission_template.csv')
        HAS_VALIDATION = True
        print(f'   Training samples: {wq.shape[0]}')
        print(f'   Validation samples: {sub_template.shape[0]}')
    except FileNotFoundError:
        HAS_VALIDATION = False
        print(f'   Training samples: {wq.shape[0]}')
        print('   Validation files not found - will only do CV evaluation')

    # Build features
    print('\n2. Engineering features...')
    X_train_base = engineer_base_features(wq, ls_train, tc_train)
    print(f'   Base features: {X_train_base.shape[1]}')

    if HAS_VALIDATION:
        X_val_base = engineer_base_features(sub_template, ls_val, tc_val)

    # Impute missing values
    print('\n3. Imputing missing values (KNN)...')
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_base), columns=X_train_base.columns)

    if HAS_VALIDATION:
        X_val_imp = pd.DataFrame(imputer.transform(X_val_base), columns=X_val_base.columns)

    # Add spatial target features
    print('\n4. Adding spatial target features (IDW)...')
    spatial_feats_train = add_spatial_target_features(X_train_imp, wq)
    X_train_full = pd.concat([X_train_imp, spatial_feats_train], axis=1)

    if HAS_VALIDATION:
        spatial_feats_val = add_spatial_target_features(X_val_imp, wq)
        X_val_full = pd.concat([X_val_imp, spatial_feats_val], axis=1)

    # Add temporal-location features
    print('\n5. Adding temporal-location features...')
    temporal_feats_train = add_temporal_location_features(X_train_imp, wq)
    X_train_full = pd.concat([X_train_full, temporal_feats_train], axis=1)

    if HAS_VALIDATION:
        temporal_feats_val = add_temporal_location_features(X_val_imp, wq)
        X_val_full = pd.concat([X_val_full, temporal_feats_val], axis=1)

    print(f'   Total features: {X_train_full.shape[1]}')

    # Handle NaN/Inf
    X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
    X_train_full = X_train_full.fillna(X_train_full.median())

    if HAS_VALIDATION:
        X_val_full = X_val_full.replace([np.inf, -np.inf], np.nan)
        X_val_full = X_val_full.fillna(X_train_full.median())

    # Scale features
    print('\n6. Scaling features...')
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)

    if HAS_VALIDATION:
        X_val_scaled = scaler.transform(X_val_full)

    # Location groups for spatial CV
    location_groups = create_location_id(wq)

    # Target configurations - IMPROVED TRANSFORMATIONS
    target_configs = {
        'Total Alkalinity': {
            'y_train': np.log1p(wq['Total Alkalinity'].values),
            'inv_transform': np.expm1,
            'clip_min': 0,
        },
        'Electrical Conductance': {
            'y_train': np.log1p(wq['Electrical Conductance'].values),
            'inv_transform': np.expm1,
            'clip_min': 0,
        },
        'Dissolved Reactive Phosphorus': {
            'y_train': np.log1p(wq['Dissolved Reactive Phosphorus'].values + 1),
            'inv_transform': lambda x: np.expm1(x) - 1,
            'clip_min': 0,
        },
    }

    # Cross-validation evaluation
    print('\n7. Cross-validation evaluation (Spatial GroupKFold)...')
    cv_results = {}

    for target_name, cfg in target_configs.items():
        y_train = cfg['y_train']
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0
        )
        mean_rmse, std_rmse = get_cv_score(
            X_train_scaled, y_train, model,
            n_splits=5, groups=location_groups
        )
        cv_results[target_name] = (mean_rmse, std_rmse)
        print(f'   {target_name}: RMSE = {mean_rmse:.4f} (+/- {std_rmse:.4f})')

    # Train final models
    print('\n8. Training final ensemble models...')
    predictions = {}

    for target_name, cfg in target_configs.items():
        print(f'\n   {target_name}:')
        y_train = cfg['y_train']
        inv_func = cfg['inv_transform']
        clip_min = cfg['clip_min']
        all_preds = []

        # Stacking ensemble
        print('      Training stacking ensemble...')
        for seed in SEEDS[:3]:
            stacking = train_stacking_model(X_train_scaled, y_train, seed=seed)
            stacking.fit(X_train_scaled, y_train)
            if HAS_VALIDATION:
                all_preds.append(inv_func(stacking.predict(X_val_scaled)))

        # XGBoost variants
        print('      Training XGBoost variants...')
        xgb_configs = [
            dict(n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.7,
                 colsample_bytree=0.5, min_child_weight=15, reg_alpha=1.0, reg_lambda=5.0),
            dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.75,
                 colsample_bytree=0.6, min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0),
            dict(n_estimators=1000, max_depth=3, learning_rate=0.01, subsample=0.6,
                 colsample_bytree=0.4, min_child_weight=20, reg_alpha=2.0, reg_lambda=8.0),
        ]

        for params in xgb_configs:
            for seed in SEEDS:
                model = xgb.XGBRegressor(**params, random_state=seed, n_jobs=-1, verbosity=0)
                model.fit(X_train_scaled, y_train)
                if HAS_VALIDATION:
                    all_preds.append(inv_func(model.predict(X_val_scaled)))

        # LightGBM
        if HAS_LIGHTGBM:
            print('      Training LightGBM...')
            for seed in SEEDS[:3]:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=600, max_depth=5, learning_rate=0.025,
                    subsample=0.7, colsample_bytree=0.5, min_child_samples=15,
                    reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1
                )
                lgb_model.fit(X_train_scaled, y_train)
                if HAS_VALIDATION:
                    all_preds.append(inv_func(lgb_model.predict(X_val_scaled)))

        # CatBoost
        if HAS_CATBOOST:
            print('      Training CatBoost...')
            for seed in SEEDS[:3]:
                cat_model = CatBoostRegressor(
                    iterations=600, depth=5, learning_rate=0.025,
                    l2_leaf_reg=5.0, random_seed=seed, verbose=0
                )
                cat_model.fit(X_train_scaled, y_train)
                if HAS_VALIDATION:
                    all_preds.append(inv_func(cat_model.predict(X_val_scaled)))

        # Tree ensembles
        print('      Training tree ensembles...')
        for seed in SEEDS[:3]:
            et = ExtraTreesRegressor(
                n_estimators=500, max_depth=15, min_samples_leaf=5,
                max_features=0.5, random_state=seed, n_jobs=-1
            )
            et.fit(X_train_scaled, y_train)
            if HAS_VALIDATION:
                all_preds.append(inv_func(et.predict(X_val_scaled)))

            rf = RandomForestRegressor(
                n_estimators=500, max_depth=12, min_samples_leaf=8,
                max_features=0.5, random_state=seed, n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            if HAS_VALIDATION:
                all_preds.append(inv_func(rf.predict(X_val_scaled)))

        # Combine predictions
        if HAS_VALIDATION and len(all_preds) > 0:
            final_pred = np.clip(np.mean(all_preds, axis=0), clip_min, None)
            predictions[target_name] = final_pred
            print(f'      Ensemble size: {len(all_preds)}, Range: [{final_pred.min():.2f}, {final_pred.max():.2f}]')

    # Create submission
    if HAS_VALIDATION:
        print('\n9. Creating submission file...')
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

        submission_df.to_csv('submission.csv', index=False)

        print('\n' + '=' * 70)
        print('DONE!')
        print('=' * 70)
        print(f'\nSubmission saved to: submission.csv')
        print(f'Shape: {submission_df.shape}')
        print('\nPrediction summary:')
        for col in TARGET_COLS:
            print(f'  {col}: mean={submission_df[col].mean():.2f}, '
                  f'median={submission_df[col].median():.2f}, '
                  f'range=[{submission_df[col].min():.2f}, {submission_df[col].max():.2f}]')

        print('\nUpload submission.csv to the challenge platform!')
    else:
        print('\n' + '=' * 70)
        print('CV EVALUATION COMPLETE')
        print('=' * 70)
        print('\nTo generate predictions, add:')
        print('  - landsat_features_validation.csv')
        print('  - terraclimate_features_validation.csv')
        print('  - submission_template.csv')


if __name__ == '__main__':
    main()
