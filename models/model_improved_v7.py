#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v7.0
Focus: Spatial generalization to completely new locations.

Key insight: Validation locations have ZERO overlap with training.
This means IDW/spatial features based on training targets cause overfitting.
We need features that generalize based on physical relationships.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, BayesianRidge, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
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


def create_physical_features(wq_df, ls_df, tc_df):
    """
    Create features based on physical/spectral relationships.
    These should generalize to new locations.
    """
    df = pd.DataFrame()
    eps = 1e-10

    # === SPATIAL FEATURES (relative, not absolute) ===
    df['Latitude'] = wq_df['Latitude'].values
    df['Longitude'] = wq_df['Longitude'].values

    # Normalized/relative spatial (within validation region)
    lat_center, lon_center = -33.0, 26.5  # Approximate center of validation region
    df['lat_from_center'] = df['Latitude'] - lat_center
    df['lon_from_center'] = df['Longitude'] - lon_center
    df['dist_from_center'] = np.sqrt(df['lat_from_center']**2 + df['lon_from_center']**2)

    # Distance from coast (approximate for South Africa eastern coast)
    # Coast runs roughly along certain coordinates
    df['dist_coast'] = np.abs(df['Longitude'] - 28.5) + np.abs(df['Latitude'] + 33) * 0.5

    # Elevation proxy (latitude in SA correlates with inland plateau)
    df['elevation_proxy'] = -df['Latitude'] - 25  # Higher values = more inland/elevated

    # === TEMPORAL FEATURES ===
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    df['month'] = dates.dt.month
    df['day_of_year'] = dates.dt.dayofyear
    df['year'] = dates.dt.year

    # Cyclic encoding (important for seasonal patterns)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Season (Southern Hemisphere)
    df['season'] = ((df['month'] % 12) // 3)  # 0=summer, 1=autumn, 2=winter, 3=spring
    df['is_wet_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(float)
    df['is_dry_season'] = df['month'].isin([4, 5, 6, 7, 8, 9]).astype(float)

    # === RAW SPECTRAL BANDS ===
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22

    # Log transforms (handle skewness)
    df['nir_log'] = np.log1p(np.maximum(nir, 0))
    df['green_log'] = np.log1p(np.maximum(green, 0))
    df['swir16_log'] = np.log1p(np.maximum(swir16, 0))
    df['swir22_log'] = np.log1p(np.maximum(swir22, 0))

    # === SPECTRAL INDICES (physical meaning for water quality) ===
    # Existing indices
    df['NDMI'] = ls_df['NDMI'].values  # Moisture
    df['MNDWI'] = ls_df['MNDWI'].values  # Water

    # NDWI - Normalized Difference Water Index
    df['NDWI'] = (green - nir) / (green + nir + eps)

    # Water Ratio Index
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)

    # AWEI - Automated Water Extraction Index
    df['AWEI'] = 4 * green - 0.25 * nir - 2.75 * swir16 - 2.75 * swir22
    df['AWEI_sh'] = green + 2.5 * nir - 1.5 * (swir16 + swir22)  # Shadow version

    # Turbidity indicators
    df['turbidity_nir_green'] = nir / (green + eps)
    df['turbidity_swir_green'] = swir16 / (green + eps)

    # Sediment/suspended solids
    df['sediment_idx'] = (swir16 - green) / (swir16 + green + eps)
    df['suspended_idx'] = (nir - swir16) / (nir + swir16 + eps)

    # Chlorophyll proxy (algae/organic matter)
    df['chl_proxy'] = (green - swir16) / (green + swir16 + eps)

    # === BAND RATIOS (important for mineral content) ===
    df['nir_green'] = nir / (green + eps)
    df['green_nir'] = green / (nir + eps)
    df['swir16_nir'] = swir16 / (nir + eps)
    df['swir22_nir'] = swir22 / (nir + eps)
    df['swir16_green'] = swir16 / (green + eps)
    df['swir22_green'] = swir22 / (green + eps)
    df['swir22_swir16'] = swir22 / (swir16 + eps)
    df['nir_swir_mean'] = nir / ((swir16 + swir22) / 2 + eps)

    # === NORMALIZED DIFFERENCES ===
    df['nd_nir_swir16'] = (nir - swir16) / (nir + swir16 + eps)
    df['nd_nir_swir22'] = (nir - swir22) / (nir + swir22 + eps)
    df['nd_swir16_swir22'] = (swir16 - swir22) / (swir16 + swir22 + eps)
    df['nd_green_swir16'] = (green - swir16) / (green + swir16 + eps)
    df['nd_green_swir22'] = (green - swir22) / (green + swir22 + eps)

    # === BAND STATISTICS ===
    bands = np.column_stack([nir, green, swir16, swir22])
    df['reflectance_mean'] = np.nanmean(bands, axis=1)
    df['reflectance_std'] = np.nanstd(bands, axis=1)
    df['reflectance_max'] = np.nanmax(bands, axis=1)
    df['reflectance_min'] = np.nanmin(bands, axis=1)
    df['reflectance_range'] = df['reflectance_max'] - df['reflectance_min']
    df['reflectance_cv'] = df['reflectance_std'] / (df['reflectance_mean'] + eps)
    df['reflectance_sum'] = np.nansum(bands, axis=1)

    # === TERRACLIMATE ===
    pet = tc_df['pet'].values.astype(float)
    df['pet'] = pet
    df['pet_log'] = np.log1p(np.maximum(pet, 0))

    # PET indicates evapotranspiration - relates to water concentration
    df['pet_normalized'] = pet / (pet.mean() + eps)

    # === KEY INTERACTIONS (physically meaningful) ===
    # Climate-spectral interactions
    df['pet_ndmi'] = pet * df['NDMI']
    df['pet_mndwi'] = pet * df['MNDWI']
    df['pet_ndwi'] = pet * df['NDWI']
    df['pet_turbidity'] = pet * df['turbidity_nir_green']

    # Season-spectral interactions (water quality varies seasonally)
    df['wet_ndmi'] = df['is_wet_season'] * df['NDMI']
    df['wet_turbidity'] = df['is_wet_season'] * df['turbidity_nir_green']
    df['dry_ndmi'] = df['is_dry_season'] * df['NDMI']

    # Spatial-spectral interactions
    df['lat_ndmi'] = df['lat_from_center'] * df['NDMI']
    df['lon_ndmi'] = df['lon_from_center'] * df['NDMI']
    df['dist_ndmi'] = df['dist_from_center'] * df['NDMI']

    # === POLYNOMIAL FEATURES for key indices ===
    df['NDMI_sq'] = df['NDMI'] ** 2
    df['MNDWI_sq'] = df['MNDWI'] ** 2
    df['NDWI_sq'] = df['NDWI'] ** 2
    df['turbidity_sq'] = df['turbidity_nir_green'] ** 2

    return df


def train_with_stacking(X_train, y_train, X_val, n_folds=5, seed=42):
    """
    Train multiple models and stack predictions.
    Use out-of-fold predictions to avoid overfitting.
    """
    np.random.seed(seed)

    # Define diverse base models
    models = []

    # XGBoost variants
    models.append(('xgb1', xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.5, min_child_weight=10,
        reg_alpha=0.5, reg_lambda=2.0, random_state=seed, n_jobs=-1, verbosity=0
    )))
    models.append(('xgb2', xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed+1, n_jobs=-1, verbosity=0
    )))
    models.append(('xgb3', xgb.XGBRegressor(
        n_estimators=800, max_depth=3, learning_rate=0.01,
        subsample=0.6, colsample_bytree=0.4, min_child_weight=15,
        reg_alpha=1.0, reg_lambda=5.0, random_state=seed+2, n_jobs=-1, verbosity=0
    )))

    # LightGBM
    if HAS_LIGHTGBM:
        models.append(('lgb1', lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.5, min_child_samples=15,
            reg_alpha=0.5, reg_lambda=2.0, random_state=seed, n_jobs=-1, verbose=-1
        )))
        models.append(('lgb2', lgb.LGBMRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_samples=10,
            random_state=seed+1, n_jobs=-1, verbose=-1
        )))

    # CatBoost
    if HAS_CATBOOST:
        models.append(('cat1', CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.02,
            l2_leaf_reg=3.0, random_seed=seed, verbose=0
        )))
        models.append(('cat2', CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05,
            l2_leaf_reg=1.0, random_seed=seed+1, verbose=0
        )))

    # Tree ensembles
    models.append(('et', ExtraTreesRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=3,
        max_features=0.5, random_state=seed, n_jobs=-1
    )))
    models.append(('rf', RandomForestRegressor(
        n_estimators=500, max_depth=12, min_samples_leaf=5,
        max_features=0.5, random_state=seed, n_jobs=-1
    )))

    # Gradient boosting
    models.append(('hgb', HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=seed
    )))

    # Train all models and collect predictions
    all_train_oof = []
    all_val_preds = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for name, model in models:
        # Out-of-fold predictions for training set
        oof_preds = np.zeros(len(X_train))
        val_preds_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            model_clone = clone_model(model)
            model_clone.fit(X_tr, y_tr)

            oof_preds[val_idx] = model_clone.predict(X_va)
            val_preds_list.append(model_clone.predict(X_val))

        all_train_oof.append(oof_preds)
        all_val_preds.append(np.mean(val_preds_list, axis=0))

    # Stack with Ridge meta-learner
    train_meta = np.column_stack(all_train_oof)
    val_meta = np.column_stack(all_val_preds)

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(train_meta, y_train)

    stacked_pred = meta_model.predict(val_meta)

    # Also compute simple average
    simple_avg = np.mean(all_val_preds, axis=0)

    # Weighted combination
    final_pred = 0.6 * stacked_pred + 0.4 * simple_avg

    return final_pred


def clone_model(model):
    """Clone a sklearn-compatible model."""
    from sklearn.base import clone
    return clone(model)


def train_simple_ensemble(X_train, y_train, X_val, seeds=[42, 123, 456, 789, 2024]):
    """
    Simple but robust ensemble training.
    Focus on diversity and regularization.
    """
    all_preds = []

    for seed in seeds:
        # XGBoost - main workhorse
        for config in [
            dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
                 colsample_bytree=0.5, min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0),
            dict(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.75,
                 colsample_bytree=0.6, min_child_weight=8, reg_alpha=0.3, reg_lambda=1.5),
            dict(n_estimators=800, max_depth=3, learning_rate=0.01, subsample=0.6,
                 colsample_bytree=0.4, min_child_weight=15, reg_alpha=1.0, reg_lambda=3.0),
        ]:
            model = xgb.XGBRegressor(**config, random_state=seed, n_jobs=-1, verbosity=0)
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

        # RandomForest
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=8,
            max_features=0.4, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        all_preds.append(rf.predict(X_val))

    return np.mean(all_preds, axis=0)


def get_spatial_cv_score(X, y, model, groups, n_splits=5):
    """Spatial cross-validation using location groups."""
    kf = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in kf.split(X, y, groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model_clone = clone_model(model)
        model_clone.fit(X_tr, y_tr)
        pred = model_clone.predict(X_va)

        # RMSE
        rmse = np.sqrt(np.mean((y_va - pred) ** 2))
        scores.append(rmse)

    return np.mean(scores), np.std(scores)


def main():
    print('=' * 70)
    print('EY Water Quality Prediction v7.0 - Spatial Generalization Focus')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Training: {len(wq)} samples, {wq.groupby(["Latitude", "Longitude"]).ngroups} locations')
    print(f'   Validation: {len(sub_template)} samples, {sub_template.groupby(["Latitude", "Longitude"]).ngroups} locations')

    # Feature engineering
    print('\n2. Creating physical features...')
    X_train_raw = create_physical_features(wq, ls_train, tc_train)
    X_val_raw = create_physical_features(sub_template, ls_val, tc_val)
    print(f'   Features created: {X_train_raw.shape[1]}')

    # Impute missing values
    print('\n3. Imputing missing values...')
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X_train_raw.columns
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val_raw),
        columns=X_val_raw.columns
    )

    # Handle infinities
    X_train_imp = X_train_imp.replace([np.inf, -np.inf], np.nan)
    X_train_imp = X_train_imp.fillna(X_train_imp.median())
    X_val_imp = X_val_imp.replace([np.inf, -np.inf], np.nan)
    X_val_imp = X_val_imp.fillna(X_train_imp.median())

    # Scale features
    print('\n4. Scaling features...')
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    # Location groups for spatial CV
    location_groups = (wq['Latitude'].round(2).astype(str) + '_' +
                       wq['Longitude'].round(2).astype(str))

    # Train models for each target
    print('\n5. Training models...')
    predictions = {}

    target_configs = {
        'Total Alkalinity': {
            'transform': lambda x: np.log1p(x),
            'inverse': lambda x: np.expm1(x),
            'clip_min': 0,
            'clip_max': 500,
        },
        'Electrical Conductance': {
            'transform': lambda x: np.log1p(x),
            'inverse': lambda x: np.expm1(x),
            'clip_min': 0,
            'clip_max': 2000,
        },
        'Dissolved Reactive Phosphorus': {
            'transform': lambda x: np.log1p(x + 1),
            'inverse': lambda x: np.expm1(x) - 1,
            'clip_min': 0,
            'clip_max': 300,
        },
    }

    for target in TARGET_COLS:
        print(f'\n   {target}:')
        cfg = target_configs[target]

        y_raw = wq[target].values
        y_train = cfg['transform'](y_raw)

        # Quick CV check
        ref_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0
        )
        cv_mean, cv_std = get_spatial_cv_score(
            X_train_scaled, y_train, ref_model, location_groups
        )
        print(f'      CV RMSE (log): {cv_mean:.4f} +/- {cv_std:.4f}')

        # Train ensemble
        print('      Training ensemble...')
        pred_transformed = train_simple_ensemble(
            X_train_scaled, y_train, X_val_scaled
        )

        # Inverse transform and clip
        pred_final = cfg['inverse'](pred_transformed)
        pred_final = np.clip(pred_final, cfg['clip_min'], cfg['clip_max'])

        predictions[target] = pred_final
        print(f'      Range: [{pred_final.min():.2f}, {pred_final.max():.2f}], Mean: {pred_final.mean():.2f}')

    # Create submission
    print('\n6. Creating submission...')
    submission = pd.DataFrame({
        'Latitude': sub_template['Latitude'],
        'Longitude': sub_template['Longitude'],
        'Sample Date': sub_template['Sample Date'],
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })

    submission.to_csv('submission_v7.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print(f'\nSubmission saved: submission_v7.csv')
    print('\nPrediction Summary:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, '
              f'median={submission[col].median():.2f}, '
              f'std={submission[col].std():.2f}')


if __name__ == '__main__':
    main()
