#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v22.0
FINAL OPTIMIZED MODEL

Key strategy changes:
1. env_idw computed with LEAVE-LOCATION-OUT to prevent target leakage
2. Geographic IDW remains the backbone (proven at 0.274)
3. ML provides temporal/spectral corrections on top of IDW baseline
4. Calibration to Eastern Cape distribution
5. Diverse blending strategies
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET_COLS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus'
]

BASE_DIR = '/Users/nicolasspagnuolo/EY'


# =============================================================================
# ELEVATION
# =============================================================================

def load_elevation_cache():
    cache_path = os.path.join(BASE_DIR, 'external_data_cache.json')
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        if 'elevation' in cache:
            return cache['elevation']
    return {}


def get_elevation(lat, lon, elev_cache):
    for decimals in [4, 3, 2, 1]:
        key = f"{round(lat, decimals)},{round(lon, decimals)}"
        if key in elev_cache:
            return elev_cache[key]
    if elev_cache:
        best_dist = float('inf')
        best_elev = None
        for key, elev in elev_cache.items():
            try:
                parts = key.split(',')
                clat, clon = float(parts[0]), float(parts[1])
                dist = np.sqrt((lat - clat)**2 + (lon - clon)**2)
                if dist < best_dist and dist < 0.15:
                    best_dist = dist
                    best_elev = elev
            except (ValueError, IndexError):
                continue
        if best_elev is not None:
            return best_elev
    return estimate_elevation_sa(lat, lon)


def estimate_elevation_sa(lat, lon):
    dist_coast_km = min(
        max(0, 30 - lon) * 111 * np.cos(np.radians(lat)),
        max(0, lat + 34) * 111
    )
    if lon < 20:
        elev = 50 + dist_coast_km * 0.5
    elif lon < 24:
        elev = 400 + dist_coast_km * 0.3
    elif lon < 28:
        elev = 800 + dist_coast_km * 0.5 if lat <= -30 else 1200 + (lat + 30) * 50
    elif lon < 30:
        elev = 1000 + max(0, -30 - lat) * 200 if lat <= -28 else 1400 + (28 + lat) * 100
    else:
        elev = 200 + max(0, 30 - lon) * 100
    return np.clip(elev, 0, 2500)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(wq_df, ls_df, tc_df, climate_df,
                     train_wq, target, elev_cache,
                     train_tc, train_climate, train_ls,
                     is_validation=False):
    """
    Create features with proper handling of IDW leakage.
    For training data, env_idw is computed leave-location-out.
    """
    df = pd.DataFrame()
    eps = 1e-10
    n = len(wq_df)

    lat = wq_df['Latitude'].values.astype(float)
    lon = wq_df['Longitude'].values.astype(float)

    # =========================================================================
    # A. COORDINATE FEATURES
    # =========================================================================
    df['Latitude'] = lat
    df['Longitude'] = lon
    df['lat_sq'] = lat ** 2
    df['lon_sq'] = lon ** 2
    df['lat_lon'] = lat * lon
    df['lat_cu'] = lat ** 3
    df['lon_cu'] = lon ** 3

    # =========================================================================
    # B. ELEVATION & TOPOGRAPHIC
    # =========================================================================
    if elev_cache is None:
        elev_cache = {}

    elevations = np.array([get_elevation(la, lo, elev_cache)
                           for la, lo in zip(lat, lon)])
    df['elevation'] = elevations
    df['elevation_log'] = np.log1p(np.maximum(elevations, 0))

    df['dist_coast_lon'] = np.maximum(0, 30 - lon) * 111 * np.cos(np.radians(lat))
    df['dist_coast_lat'] = np.maximum(0, lat + 34) * 111
    df['dist_coast'] = np.minimum(df['dist_coast_lon'], df['dist_coast_lat'])
    df['dist_coast_log'] = np.log1p(df['dist_coast'])

    df['is_coastal'] = (elevations < 200).astype(float)
    df['is_lowland'] = ((elevations >= 200) & (elevations < 600)).astype(float)
    df['is_midland'] = ((elevations >= 600) & (elevations < 1200)).astype(float)
    df['is_highland'] = (elevations >= 1200).astype(float)

    df['is_eastern_cape'] = ((lon >= 24) & (lon <= 29) & (lat < -30)).astype(float)

    # =========================================================================
    # C. TEMPORAL
    # =========================================================================
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    month = dates.dt.month.values
    day_of_year = dates.dt.dayofyear.values

    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
    df['is_summer'] = np.isin(month, [12, 1, 2]).astype(float)
    df['is_winter'] = np.isin(month, [6, 7, 8]).astype(float)
    df['is_wet_season'] = np.isin(month, [10, 11, 12, 1, 2, 3]).astype(float)

    # =========================================================================
    # D. SPECTRAL
    # =========================================================================
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22
    df['NDMI'] = ls_df['NDMI'].values.astype(float)
    df['MNDWI'] = ls_df['MNDWI'].values.astype(float)
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['turbidity'] = nir / (green + eps)
    df['turbidity_log'] = np.log1p(np.maximum(df['turbidity'], 0))
    df['sediment'] = (swir16 - green) / (swir16 + green + eps)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)
    df['swir_ratio'] = swir22 / (swir16 + eps)
    df['nir_green'] = nir / (green + eps)
    df['NDBI'] = (swir16 - nir) / (swir16 + nir + eps)
    df['vegetation_proxy'] = (nir - green) / (nir + green + eps)
    df['clarity_index'] = green / (nir + eps)
    df['suspended_solids'] = (nir + swir16) / (green + eps)
    df['mineral_index'] = (swir16 + swir22) / (nir + green + eps)
    df['nir_minus_green'] = nir - green
    df['swir16_minus_swir22'] = swir16 - swir22

    bands = np.column_stack([nir, green, swir16, swir22])
    df['reflect_mean'] = np.nanmean(bands, axis=1)
    df['reflect_std'] = np.nanstd(bands, axis=1)
    df['reflect_range'] = np.nanmax(bands, axis=1) - np.nanmin(bands, axis=1)
    df['reflect_cv'] = df['reflect_std'] / (df['reflect_mean'] + eps)

    df['nir_log'] = np.log1p(np.maximum(nir, 0))
    df['green_log'] = np.log1p(np.maximum(green, 0))
    df['swir16_log'] = np.log1p(np.maximum(swir16, 0))

    # =========================================================================
    # E. CLIMATE
    # =========================================================================
    pet = tc_df['pet'].values.astype(float)
    df['pet'] = pet
    df['pet_log'] = np.log1p(np.maximum(pet, 0))

    if climate_df is not None and len(climate_df) == n:
        df['precip_30d'] = climate_df['precip_30d'].values
        df['precip_mean'] = climate_df['precip_mean'].values
        df['precip_max'] = climate_df['precip_max'].values
        df['precip_days'] = climate_df['precip_days'].values
        df['precip_30d_log'] = np.log1p(np.maximum(df['precip_30d'], 0))
        df['precip_intensity'] = df['precip_30d'] / (df['precip_days'] + eps)

        df['temp_mean'] = climate_df['temp_mean'].values
        df['temp_max'] = climate_df['temp_max'].values
        df['temp_min'] = climate_df['temp_min'].values
        df['temp_range'] = climate_df['temp_range'].values

        df['et0_mean'] = climate_df['et0_mean'].values
        df['et0_sum'] = climate_df['et0_sum'].values

        df['water_balance'] = df['precip_30d'] - df['et0_sum']
        df['aridity_index'] = df['pet'] / (df['precip_30d'] + eps)
        df['runoff_proxy'] = np.maximum(0, df['precip_30d'] - df['et0_sum'])

    # =========================================================================
    # F. INTERACTIONS
    # =========================================================================
    df['elev_pet'] = df['elevation'] * pet
    if 'temp_mean' in df.columns:
        df['elev_temp'] = df['elevation'] * df['temp_mean']
        df['elev_precip'] = df['elevation'] * df['precip_30d']
    df['elev_ndmi'] = df['elevation'] * df['NDMI']
    df['elev_turbidity'] = df['elevation'] * df['turbidity']
    df['elev_mineral'] = df['elevation'] * df['mineral_index']
    df['coast_ndmi'] = df['dist_coast'] * df['NDMI']
    df['coast_mineral'] = df['dist_coast'] * df['mineral_index']
    df['lat_ndmi'] = lat * df['NDMI']
    df['lat_turbidity'] = lat * df['turbidity']
    df['wet_turb'] = df['is_wet_season'] * df['turbidity']
    df['wet_mineral'] = df['is_wet_season'] * df['mineral_index']

    if 'precip_30d' in df.columns:
        df['precip_turbidity'] = df['precip_30d'] * df['turbidity']
        df['precip_ndwi'] = df['precip_30d'] * df['NDWI']
        df['temp_turbidity'] = df['temp_mean'] * df['turbidity']
        df['temp_mineral'] = df['temp_mean'] * df['mineral_index']
        df['aridity_mineral'] = df['aridity_index'] * df['mineral_index']

    # =========================================================================
    # G. IDW FEATURES
    # =========================================================================
    _add_idw_features(
        df, lat, lon, train_wq, target, elev_cache,
        train_tc, train_climate, train_ls,
        ls_df, tc_df, climate_df,
        is_validation=is_validation
    )

    return df


def _add_idw_features(df, lat, lon, train_wq, target, elev_cache,
                       train_tc, train_climate, train_ls,
                       ls_df, tc_df, climate_df,
                       is_validation=False):
    """
    Geographic IDW + Environmental IDW.
    For training data, env_idw uses LEAVE-LOCATION-OUT to prevent leakage.
    """
    eps = 1e-10
    train_coords = train_wq[['Latitude', 'Longitude']].values
    train_values = train_wq[target].values
    val_coords = np.column_stack([lat, lon])
    train_n = len(train_wq)

    tree = cKDTree(train_coords)

    # === Geographic IDW ===
    for k in [5, 10, 20, 30, 50, 100]:
        k_actual = min(k, train_n)
        dists, idxs = tree.query(val_coords, k=k_actual)
        dists = np.maximum(dists, eps)

        for power in [1.0, 1.5, 2.0]:
            weights = 1.0 / (dists ** power)
            weights = weights / weights.sum(axis=1, keepdims=True)
            p_label = str(power).replace('.', '')
            df[f'idw_k{k}_p{p_label}'] = np.sum(weights * train_values[idxs], axis=1)

        # IDW std
        weights1 = 1.0 / dists
        weights1 = weights1 / weights1.sum(axis=1, keepdims=True)
        idw_mean = np.sum(weights1 * train_values[idxs], axis=1)
        df[f'idw_k{k}_std'] = np.sqrt(
            np.sum(weights1 * (train_values[idxs] - idw_mean[:, None])**2, axis=1)
        )

    # Nearest neighbor features
    dists_1, idxs_1 = tree.query(val_coords, k=1)
    df['dist_nearest'] = dists_1
    df['dist_nearest_log'] = np.log1p(dists_1)
    df['nn_value'] = train_values[idxs_1]

    for k_nn in [3, 5, 10]:
        k_actual = min(k_nn, train_n)
        dists_k, idxs_k = tree.query(val_coords, k=k_actual)
        df[f'nn{k_nn}_mean'] = np.mean(train_values[idxs_k], axis=1)
        df[f'nn{k_nn}_std'] = np.std(train_values[idxs_k], axis=1)

    # === Environmental IDW with leave-location-out for training ===
    # Build environmental feature vectors
    train_elevs = np.array([get_elevation(la, lo, elev_cache)
                            for la, lo in train_coords])
    val_elevs = np.array([get_elevation(la, lo, elev_cache)
                          for la, lo in val_coords])

    # Training environmental features
    tr_pet = train_tc['pet'].values.astype(float) if len(train_tc) == train_n else np.zeros(train_n)
    tr_ndmi = train_ls['NDMI'].values.astype(float) if len(train_ls) == train_n else np.zeros(train_n)
    tr_turb = (train_ls['nir'].values.astype(float) /
               (train_ls['green'].values.astype(float) + eps)) if len(train_ls) == train_n else np.zeros(train_n)

    tr_precip = np.zeros(train_n)
    tr_temp = np.zeros(train_n)
    if train_climate is not None and len(train_climate) == train_n:
        tr_precip = train_climate['precip_30d'].values.astype(float)
        tr_temp = train_climate['temp_mean'].values.astype(float)

    # Current (val/train) environmental features
    v_pet = tc_df['pet'].values.astype(float)
    v_ndmi = ls_df['NDMI'].values.astype(float)
    v_turb = ls_df['nir'].values.astype(float) / (ls_df['green'].values.astype(float) + eps)

    v_precip = np.zeros(len(lat))
    v_temp = np.zeros(len(lat))
    if climate_df is not None and len(climate_df) == len(lat):
        v_precip = climate_df['precip_30d'].values.astype(float)
        v_temp = climate_df['temp_mean'].values.astype(float)

    # Build feature matrices
    train_env = np.column_stack([train_elevs, tr_pet, tr_ndmi, tr_turb, tr_precip, tr_temp])
    val_env = np.column_stack([val_elevs, v_pet, v_ndmi, v_turb, v_precip, v_temp])

    # Replace NaN
    train_env = np.nan_to_num(train_env, nan=0.0)
    val_env = np.nan_to_num(val_env, nan=0.0)

    # Normalize using training statistics
    env_mean = np.mean(train_env, axis=0)
    env_std = np.std(train_env, axis=0) + eps
    train_env_norm = (train_env - env_mean) / env_std
    val_env_norm = (val_env - env_mean) / env_std

    # Environmental distance matrix
    env_dists = cdist(val_env_norm, train_env_norm, metric='euclidean')

    if not is_validation:
        # TRAINING: Leave-location-out env IDW
        # For each training point, exclude same-location points
        train_loc_keys = (
            train_wq['Latitude'].round(2).astype(str) + '_' +
            train_wq['Longitude'].round(2).astype(str)
        ).values
        curr_loc_keys = (
            pd.Series(lat).round(2).astype(str).values + '_' +
            pd.Series(lon).round(2).astype(str).values
        )

        for k_env in [10, 30, 50]:
            env_idw_vals = np.zeros(len(lat))
            env_idw_stds = np.zeros(len(lat))

            for i in range(len(lat)):
                # Exclude same-location training points
                mask = train_loc_keys != curr_loc_keys[i]
                if mask.sum() == 0:
                    mask = np.ones(train_n, dtype=bool)

                # Get env distances to non-same-location points
                d = env_dists[i, mask]
                v = train_values[mask]

                k_actual = min(k_env, len(v))
                top_k = np.argsort(d)[:k_actual]
                d_k = np.maximum(d[top_k], eps)

                w = 1.0 / d_k
                w = w / w.sum()
                env_idw_vals[i] = np.sum(w * v[top_k])
                env_idw_stds[i] = np.sqrt(np.sum(w * (v[top_k] - env_idw_vals[i])**2))

            df[f'env_idw_k{k_env}'] = env_idw_vals
            df[f'env_idw_k{k_env}_std'] = env_idw_stds
    else:
        # VALIDATION: Use all training points (no leakage issue)
        for k_env in [10, 30, 50]:
            k_actual = min(k_env, train_n)
            env_idxs = np.argsort(env_dists, axis=1)[:, :k_actual]
            env_d = np.array([env_dists[i, env_idxs[i]] for i in range(len(lat))])
            env_d = np.maximum(env_d, eps)

            weights = 1.0 / env_d
            weights = weights / weights.sum(axis=1, keepdims=True)
            vals_sel = np.array([train_values[env_idxs[i]] for i in range(len(lat))])
            df[f'env_idw_k{k_env}'] = np.sum(weights * vals_sel, axis=1)

            env_mean_pred = df[f'env_idw_k{k_env}'].values
            df[f'env_idw_k{k_env}_std'] = np.sqrt(
                np.sum(weights * (vals_sel - env_mean_pred[:, None])**2, axis=1)
            )

    # === Combined Geographic + Environmental IDW ===
    k_combined = min(50, train_n)
    geo_dists_all, geo_idxs_all = tree.query(val_coords, k=k_combined)
    geo_dists_all = np.maximum(geo_dists_all, eps)

    for alpha in [0.3, 0.5, 0.7]:
        env_d_geo = np.array([env_dists[i, geo_idxs_all[i]] for i in range(len(lat))])
        geo_d_norm = geo_dists_all / (np.max(geo_dists_all) + eps)
        env_d_norm = env_d_geo / (np.max(env_d_geo) + eps)

        combined_d = alpha * geo_d_norm + (1 - alpha) * env_d_norm
        combined_d = np.maximum(combined_d, eps)

        weights_c = 1.0 / combined_d
        weights_c = weights_c / weights_c.sum(axis=1, keepdims=True)
        a_label = str(int(alpha * 10))
        df[f'combined_idw_a{a_label}'] = np.sum(
            weights_c * train_values[geo_idxs_all], axis=1
        )

    # Directional IDW
    for direction, condition in [
        ('north', lambda t, v: t[:, 0] > v[0]),
        ('south', lambda t, v: t[:, 0] < v[0]),
    ]:
        dir_values = []
        for i in range(len(val_coords)):
            mask = condition(train_coords, val_coords[i])
            if mask.sum() > 0:
                sub_coords = train_coords[mask]
                sub_vals = train_values[mask]
                sub_dists = np.sqrt(np.sum((sub_coords - val_coords[i])**2, axis=1))
                k_dir = min(5, len(sub_vals))
                top_k = np.argsort(sub_dists)[:k_dir]
                w = 1.0 / np.maximum(sub_dists[top_k], eps)
                w = w / w.sum()
                dir_values.append(np.sum(w * sub_vals[top_k]))
            else:
                dir_values.append(np.nan)
        df[f'idw_{direction}'] = dir_values


# =============================================================================
# OPTUNA OPTIMIZATION
# =============================================================================

def optimize_xgboost(X_train, y_train, groups, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 15.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 30.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }
        kf = GroupKFold(n_splits=5)
        scores = []
        for train_idx, val_idx in kf.split(X_train, y_train, groups):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            scores.append(r2_score(y_va, pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_lightgbm(X_train, y_train, groups, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 15.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 30.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        }
        kf = GroupKFold(n_splits=5)
        scores = []
        for train_idx, val_idx in kf.split(X_train, y_train, groups):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]
            model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            scores.append(r2_score(y_va, pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# =============================================================================
# STACKING ENSEMBLE
# =============================================================================

def train_stacking_ensemble(X_train, y_train, X_val, groups,
                             xgb_params, lgb_params, seeds=[42, 123, 456]):
    n_train = len(X_train)
    kf = GroupKFold(n_splits=5)
    folds = list(kf.split(X_train, y_train, groups))

    def train_base_model(ModelClass, params, seed):
        oof_preds = np.zeros(n_train)
        val_preds_list = []
        for train_idx, val_idx in folds:
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]
            params_copy = params.copy()
            if hasattr(ModelClass(), 'random_state'):
                params_copy['random_state'] = seed
            model = ModelClass(**params_copy)
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] = model.predict(X_va)
            val_preds_list.append(model.predict(X_val))
        val_pred = np.mean(val_preds_list, axis=0)
        return oof_preds, val_pred

    meta_train = []
    meta_val = []
    model_names = []

    for seed in seeds:
        xgb_p = {**xgb_params, 'n_jobs': -1, 'verbosity': 0}
        oof, val = train_base_model(xgb.XGBRegressor, xgb_p, seed)
        meta_train.append(oof); meta_val.append(val)
        model_names.append(f'XGB_s{seed}')

        lgb_p = {**lgb_params, 'n_jobs': -1, 'verbose': -1}
        oof, val = train_base_model(lgb.LGBMRegressor, lgb_p, seed)
        meta_train.append(oof); meta_val.append(val)
        model_names.append(f'LGB_s{seed}')

        et_params = {
            'n_estimators': 300, 'max_depth': 10,
            'min_samples_leaf': 10, 'max_features': 0.6, 'n_jobs': -1,
        }
        oof, val = train_base_model(ExtraTreesRegressor, et_params, seed)
        meta_train.append(oof); meta_val.append(val)
        model_names.append(f'ET_s{seed}')

        gb_params = {
            'n_estimators': 200, 'max_depth': 4,
            'learning_rate': 0.05, 'subsample': 0.7,
            'min_samples_leaf': 15, 'max_features': 0.6,
        }
        oof, val = train_base_model(GradientBoostingRegressor, gb_params, seed)
        meta_train.append(oof); meta_val.append(val)
        model_names.append(f'GB_s{seed}')

        rf_params = {
            'n_estimators': 300, 'max_depth': 10,
            'min_samples_leaf': 10, 'max_features': 0.5, 'n_jobs': -1,
        }
        oof, val = train_base_model(RandomForestRegressor, rf_params, seed)
        meta_train.append(oof); meta_val.append(val)
        model_names.append(f'RF_s{seed}')

    meta_X_train = np.column_stack(meta_train)
    meta_X_val = np.column_stack(meta_val)

    # Print OOF scores
    print('     OOF R² scores:')
    for i, name in enumerate(model_names):
        score = r2_score(y_train, meta_train[i])
        print(f'       {name}: {score:.4f}')

    # Meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X_train, y_train)
    stacking_pred = meta_model.predict(meta_X_val)

    # Simple average
    simple_avg = np.mean(meta_val, axis=0)

    # Weighted average
    oof_scores = [max(0, r2_score(y_train, meta_train[i])) for i in range(len(meta_train))]
    total = sum(oof_scores)
    if total > 0:
        oof_weights = [s / total for s in oof_scores]
        weighted_avg = np.sum([w * v for w, v in zip(oof_weights, meta_val)], axis=0)
    else:
        weighted_avg = simple_avg

    return stacking_pred, simple_avg, weighted_avg


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def select_features(X_train, y_train, feature_names, groups, top_n=100):
    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_child_samples=20, reg_alpha=1.0, reg_lambda=5.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    nonzero = feature_imp[feature_imp['importance'] > 0]
    top_features = nonzero.head(top_n)['feature'].values
    top_idx = [list(feature_names).index(f) for f in top_features]
    return top_idx, feature_imp


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('EY Water Quality v22.0 - Final Optimized (Leave-Location-Out Env IDW)')
    print('=' * 70)

    # Load data
    print('\n[1/7] Loading data...')
    wq = pd.read_csv(os.path.join(BASE_DIR, 'water_quality_training_dataset.csv'))
    ls_train = pd.read_csv(os.path.join(BASE_DIR, 'landsat_features_training.csv'))
    ls_val = pd.read_csv(os.path.join(BASE_DIR, 'landsat_features_validation.csv'))
    tc_train = pd.read_csv(os.path.join(BASE_DIR, 'terraclimate_features_training.csv'))
    tc_val = pd.read_csv(os.path.join(BASE_DIR, 'terraclimate_features_validation.csv'))
    clim_train = pd.read_csv(os.path.join(BASE_DIR, 'train_climate_data.csv'))
    clim_val = pd.read_csv(os.path.join(BASE_DIR, 'val_climate_data.csv'))
    sub_template = pd.read_csv(os.path.join(BASE_DIR, 'submission_template.csv'))

    elev_cache = load_elevation_cache()
    print(f'   Training: {len(wq)}, Validation: {len(sub_template)}')

    location_groups = (
        wq['Latitude'].round(2).astype(str) + '_' +
        wq['Longitude'].round(2).astype(str)
    )

    target_configs = {
        'Total Alkalinity': {'clip': (0, 500)},
        'Electrical Conductance': {'clip': (0, 2000)},
        'Dissolved Reactive Phosphorus': {'clip': (0, 300)},
    }

    all_preds = {}

    for target in TARGET_COLS:
        print(f'\n{"="*70}')
        print(f'Processing: {target}')
        print(f'{"="*70}')

        cfg = target_configs[target]

        # Feature Engineering (with leave-location-out env IDW)
        print('\n  [a] Creating training features (leave-location-out env IDW)...')
        X_train_raw = create_features(
            wq, ls_train, tc_train, clim_train,
            train_wq=wq, target=target, elev_cache=elev_cache,
            train_tc=tc_train, train_climate=clim_train, train_ls=ls_train,
            is_validation=False
        )

        print('  [b] Creating validation features...')
        X_val_raw = create_features(
            sub_template, ls_val, tc_val, clim_val,
            train_wq=wq, target=target, elev_cache=elev_cache,
            train_tc=tc_train, train_climate=clim_train, train_ls=ls_train,
            is_validation=True
        )

        feature_names = X_train_raw.columns.tolist()
        print(f'     Total features: {len(feature_names)}')

        # Preprocessing
        print('  [c] Preprocessing...')
        train_medians = X_train_raw.median()
        X_train = X_train_raw.fillna(train_medians)
        X_val = X_val_raw.fillna(train_medians)
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_val = X_val.replace([np.inf, -np.inf], np.nan)
        train_medians2 = X_train.median()
        X_train = X_train.fillna(train_medians2)
        X_val = X_val.fillna(train_medians2)

        # Target transform
        y_raw = wq[target].values.copy()
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        y_train = pt.fit_transform(y_raw.reshape(-1, 1)).ravel()

        # Feature Selection
        print('  [d] Feature selection...')
        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        top_idx, feat_imp = select_features(
            X_train_sc, y_train, feature_names, location_groups, top_n=100
        )
        print(f'     Selected {len(top_idx)} features. Top 10:')
        for _, row in feat_imp.head(10).iterrows():
            print(f'       {row["feature"]}: {row["importance"]:.0f}')

        X_train_sel = X_train_sc[:, top_idx]
        X_val_sel = X_val_sc[:, top_idx]

        # Optuna Tuning
        print('  [e] Optuna optimization (100 trials each)...')
        print('     Optimizing XGBoost...')
        xgb_best = optimize_xgboost(X_train_sel, y_train, location_groups, n_trials=100)
        print(f'     XGBoost: depth={xgb_best["max_depth"]}, '
              f'lr={xgb_best["learning_rate"]:.4f}, n_est={xgb_best["n_estimators"]}')

        print('     Optimizing LightGBM...')
        lgb_best = optimize_lightgbm(X_train_sel, y_train, location_groups, n_trials=100)
        print(f'     LightGBM: depth={lgb_best["max_depth"]}, '
              f'lr={lgb_best["learning_rate"]:.4f}, n_est={lgb_best["n_estimators"]}')

        # Leave-EC-Out CV
        print('  [f] Leave-Eastern-Cape-Out CV...')
        ec_mask = (
            (wq['Latitude'] >= -35) & (wq['Latitude'] <= -31) &
            (wq['Longitude'] >= 24) & (wq['Longitude'] <= 29)
        ).values
        if ec_mask.sum() > 10:
            ec_train_idx = np.where(~ec_mask)[0]
            ec_val_idx = np.where(ec_mask)[0]
            m = xgb.XGBRegressor(**xgb_best, random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_train_sel[ec_train_idx], y_train[ec_train_idx])
            pred_eco = m.predict(X_train_sel[ec_val_idx])
            eco_r2 = r2_score(y_train[ec_val_idx], pred_eco)
            print(f'     Leave-EC-Out R²: {eco_r2:.4f} (n={ec_mask.sum()})')

        # GroupKFold CV
        kf = GroupKFold(n_splits=5)
        gkf_scores = []
        for train_idx, val_idx in kf.split(X_train_sel, y_train, location_groups):
            m = xgb.XGBRegressor(**xgb_best, random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_train_sel[train_idx], y_train[train_idx])
            pred = m.predict(X_train_sel[val_idx])
            gkf_scores.append(r2_score(y_train[val_idx], pred))
        print(f'     GroupKFold CV R²: {np.mean(gkf_scores):.4f} +/- {np.std(gkf_scores):.4f}')

        # Stacking Ensemble
        print('  [g] Training stacking ensemble (5 models x 3 seeds)...')
        stacking_pred, avg_pred, weighted_pred = train_stacking_ensemble(
            X_train_sel, y_train, X_val_sel, location_groups,
            xgb_best, lgb_best, seeds=[42, 123, 456]
        )

        # Inverse transform
        stacking_final = pt.inverse_transform(stacking_pred.reshape(-1, 1)).ravel()
        avg_final = pt.inverse_transform(avg_pred.reshape(-1, 1)).ravel()
        weighted_final = pt.inverse_transform(weighted_pred.reshape(-1, 1)).ravel()

        stacking_final = np.clip(stacking_final, cfg['clip'][0], cfg['clip'][1])
        avg_final = np.clip(avg_final, cfg['clip'][0], cfg['clip'][1])
        weighted_final = np.clip(weighted_final, cfg['clip'][0], cfg['clip'][1])

        # IDW predictions
        idw_geo = X_val_raw['idw_k30_p10'].values if 'idw_k30_p10' in X_val_raw.columns else avg_final
        idw_geo_p15 = X_val_raw['idw_k30_p15'].values if 'idw_k30_p15' in X_val_raw.columns else idw_geo
        env_idw = X_val_raw['env_idw_k30'].values if 'env_idw_k30' in X_val_raw.columns else avg_final
        comb_idw = X_val_raw['combined_idw_a5'].values if 'combined_idw_a5' in X_val_raw.columns else idw_geo

        all_preds[target] = {
            'stacking': stacking_final,
            'avg': avg_final,
            'weighted': weighted_final,
            'idw_geo': idw_geo,
            'idw_geo_p15': idw_geo_p15,
            'env_idw': env_idw,
            'comb_idw': comb_idw,
        }

        print(f'\n  Predictions for {target}:')
        for name, pred in all_preds[target].items():
            print(f'    {name}: mean={pred.mean():.2f}, std={pred.std():.2f}')

    # =========================================================================
    # Create submissions
    # =========================================================================
    print(f'\n{"="*70}')
    print('Creating submissions...')
    print(f'{"="*70}')

    def make_submission(name, blend_config):
        """blend_config: dict of target -> (ml_type, ml_w, idw_type, idw_w)"""
        sub = pd.DataFrame({
            'Latitude': sub_template['Latitude'],
            'Longitude': sub_template['Longitude'],
            'Sample Date': sub_template['Sample Date'],
        })
        for target in TARGET_COLS:
            cfg = target_configs[target]
            preds = all_preds[target]
            ml_type, ml_w, idw_type, idw_w = blend_config[target]
            ml_pred = preds[ml_type] if ml_w > 0 else np.zeros(len(sub_template))
            idw_pred = preds[idw_type] if idw_w > 0 else np.zeros(len(sub_template))
            blended = ml_w * ml_pred + idw_w * idw_pred
            sub[target] = np.clip(blended, cfg['clip'][0], cfg['clip'][1])
        fname = f'submission_v22_{name}.csv'
        sub.to_csv(os.path.join(BASE_DIR, fname), index=False)
        print(f'  {fname}')
        for target in TARGET_COLS:
            print(f'    {target}: mean={sub[target].mean():.2f}, std={sub[target].std():.2f}')
        return sub

    # Standard blends (same config for all targets)
    standard_blends = [
        ('stacking_pure', 'stacking', 1.0, 'idw_geo', 0.0),
        ('weighted_pure', 'weighted', 1.0, 'idw_geo', 0.0),
        ('geo_idw_pure', 'stacking', 0.0, 'idw_geo', 1.0),
        ('env_idw_pure', 'stacking', 0.0, 'env_idw', 1.0),
        ('stack_70_gidw_30', 'stacking', 0.7, 'idw_geo', 0.3),
        ('stack_60_gidw_40', 'stacking', 0.6, 'idw_geo', 0.4),
        ('stack_50_gidw_50', 'stacking', 0.5, 'idw_geo', 0.5),
        ('stack_40_gidw_60', 'stacking', 0.4, 'idw_geo', 0.6),
        ('stack_30_gidw_70', 'stacking', 0.3, 'idw_geo', 0.7),
        ('stack_50_eidw_50', 'stacking', 0.5, 'env_idw', 0.5),
        ('stack_50_cidw_50', 'stacking', 0.5, 'comb_idw', 0.5),
        ('wt_40_gidw_60', 'weighted', 0.4, 'idw_geo', 0.6),
        ('wt_50_gidw_50', 'weighted', 0.5, 'idw_geo', 0.5),
    ]

    for name, ml, ml_w, idw, idw_w in standard_blends:
        blend_cfg = {t: (ml, ml_w, idw, idw_w) for t in TARGET_COLS}
        make_submission(name, blend_cfg)

    # Per-target optimal blends
    print('\n  Per-target optimized submissions:')

    # Strategy 1: Higher IDW for harder targets (DRP)
    make_submission('best_v1', {
        'Total Alkalinity': ('stacking', 0.5, 'idw_geo', 0.5),
        'Electrical Conductance': ('stacking', 0.5, 'idw_geo', 0.5),
        'Dissolved Reactive Phosphorus': ('stacking', 0.3, 'idw_geo', 0.7),
    })

    # Strategy 2: Blend stacking with env IDW
    make_submission('best_v2', {
        'Total Alkalinity': ('stacking', 0.5, 'env_idw', 0.5),
        'Electrical Conductance': ('stacking', 0.5, 'env_idw', 0.5),
        'Dissolved Reactive Phosphorus': ('stacking', 0.4, 'env_idw', 0.6),
    })

    # Strategy 3: Mix different IDW types per target
    make_submission('best_v3', {
        'Total Alkalinity': ('stacking', 0.4, 'comb_idw', 0.6),
        'Electrical Conductance': ('stacking', 0.5, 'idw_geo', 0.5),
        'Dissolved Reactive Phosphorus': ('stacking', 0.3, 'env_idw', 0.7),
    })

    # Strategy 4: Heavier IDW
    make_submission('best_v4', {
        'Total Alkalinity': ('stacking', 0.3, 'idw_geo_p15', 0.7),
        'Electrical Conductance': ('stacking', 0.3, 'idw_geo_p15', 0.7),
        'Dissolved Reactive Phosphorus': ('stacking', 0.2, 'idw_geo_p15', 0.8),
    })

    print(f'\n{"="*70}')
    print('DONE! Recommended submission order:')
    print('  1. submission_v22_stack_50_gidw_50.csv')
    print('  2. submission_v22_stack_40_gidw_60.csv')
    print('  3. submission_v22_best_v1.csv')
    print('  4. submission_v22_stack_50_eidw_50.csv')
    print('  5. submission_v22_best_v4.csv')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
