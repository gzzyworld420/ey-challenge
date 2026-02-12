#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v20.0
ADVANCED COMPREHENSIVE APPROACH

Implements ALL improvement strategies:
1. Advanced Feature Engineering (spectral, climate, elevation, interactions)
2. Optuna Hyperparameter Optimization per target
3. Target-specific models with Yeo-Johnson transforms
4. Stacking Ensemble (XGBoost + LightGBM + ExtraTrees + Ridge)
5. Environmentally-weighted IDW
6. Spatial CV mimicking geographic split
7. SHAP-inspired feature selection
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
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
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
# 1. ELEVATION DATA
# =============================================================================

def load_elevation_cache():
    """Load real elevation data from API cache."""
    cache_path = os.path.join(BASE_DIR, 'external_data_cache.json')
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        if 'elevation' in cache:
            return cache['elevation']
    return {}


def get_elevation(lat, lon, elev_cache):
    """Get elevation from cache with fuzzy matching, fallback to estimation."""
    # Try exact match with various rounding levels
    for decimals in [4, 3, 2, 1]:
        key = f"{round(lat, decimals)},{round(lon, decimals)}"
        if key in elev_cache:
            return elev_cache[key]

    # Fallback: find closest cached point
    if elev_cache:
        best_dist = float('inf')
        best_elev = None
        for key, elev in elev_cache.items():
            try:
                parts = key.split(',')
                clat, clon = float(parts[0]), float(parts[1])
                dist = np.sqrt((lat - clat)**2 + (lon - clon)**2)
                if dist < best_dist and dist < 0.1:  # Within ~11km
                    best_dist = dist
                    best_elev = elev
            except (ValueError, IndexError):
                continue
        if best_elev is not None:
            return best_elev

    # Fallback: estimate from South African geography
    return estimate_elevation_sa(lat, lon)


def estimate_elevation_sa(lat, lon):
    """Estimate elevation based on South African topography."""
    # Great Escarpment model
    dist_coast_km = min(
        max(0, 30 - lon) * 111 * np.cos(np.radians(lat)),
        max(0, lat + 34) * 111
    )

    # Base elevation model
    if lon < 20:  # Western Cape coastal
        elev = 50 + dist_coast_km * 0.5
    elif lon < 24:  # Karoo
        elev = 400 + dist_coast_km * 0.3
    elif lon < 28:  # Interior plateau
        if lat > -30:
            elev = 1200 + (lat + 30) * 50
        else:
            elev = 800 + dist_coast_km * 0.5
    elif lon < 30:  # Highveld / Drakensberg
        if lat > -28:
            elev = 1400 + (28 + lat) * 100
        else:
            elev = 1000 + max(0, -30 - lat) * 200
    else:  # Eastern lowveld / KZN coast
        elev = 200 + max(0, 30 - lon) * 100

    return np.clip(elev, 0, 2500)


# =============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# =============================================================================

def create_advanced_features(wq_df, ls_df, tc_df, climate_df,
                              train_wq=None, target=None,
                              train_ls=None, elev_cache=None):
    """Create comprehensive feature set with all improvement strategies."""
    df = pd.DataFrame()
    eps = 1e-10
    n = len(wq_df)

    lat = wq_df['Latitude'].values.astype(float)
    lon = wq_df['Longitude'].values.astype(float)

    # =========================================================================
    # A. COORDINATE FEATURES (polynomial + geographic)
    # =========================================================================
    df['Latitude'] = lat
    df['Longitude'] = lon
    df['lat_sq'] = lat ** 2
    df['lon_sq'] = lon ** 2
    df['lat_lon'] = lat * lon
    df['lat_cu'] = lat ** 3
    df['lon_cu'] = lon ** 3
    df['lat_sq_lon'] = lat ** 2 * lon
    df['lat_lon_sq'] = lat * lon ** 2

    # Radial distance from centroid of SA
    sa_center_lat, sa_center_lon = -29.0, 25.0
    df['dist_center'] = np.sqrt((lat - sa_center_lat)**2 + (lon - sa_center_lon)**2)

    # Distance to validation region centroid (Eastern Cape)
    ec_lat, ec_lon = -33.0, 26.5
    df['dist_ec'] = np.sqrt((lat - ec_lat)**2 + (lon - ec_lon)**2)

    # =========================================================================
    # B. ELEVATION / TOPOGRAPHIC FEATURES
    # =========================================================================
    if elev_cache is None:
        elev_cache = {}

    elevations = np.array([get_elevation(la, lo, elev_cache)
                           for la, lo in zip(lat, lon)])
    df['elevation'] = elevations
    df['elevation_log'] = np.log1p(np.maximum(elevations, 0))
    df['elevation_sq'] = elevations ** 2

    # Distance to coast (improved)
    df['dist_coast_lon'] = np.maximum(0, 30 - lon) * 111 * np.cos(np.radians(lat))
    df['dist_coast_lat'] = np.maximum(0, lat + 34) * 111
    df['dist_coast'] = np.minimum(df['dist_coast_lon'], df['dist_coast_lat'])
    df['dist_coast_log'] = np.log1p(df['dist_coast'])

    # Slope proxy (elevation gradient)
    # Use nearby elevation differences as proxy
    df['elev_lat_grad'] = elevations * np.abs(lat)
    df['elev_lon_grad'] = elevations * lon

    # Elevation categories
    df['is_coastal'] = (elevations < 200).astype(float)
    df['is_lowland'] = ((elevations >= 200) & (elevations < 600)).astype(float)
    df['is_midland'] = ((elevations >= 600) & (elevations < 1200)).astype(float)
    df['is_highland'] = (elevations >= 1200).astype(float)

    # Regional flags
    df['is_western_cape'] = ((lon < 22) & (lat < -32)).astype(float)
    df['is_eastern_cape'] = ((lon >= 24) & (lon <= 29) & (lat < -30)).astype(float)
    df['is_highveld'] = ((lat > -30) & (lat < -25) & (lon > 25) & (lon < 30)).astype(float)
    df['is_kwazulu'] = ((lon > 29) & (lat > -31)).astype(float)

    # =========================================================================
    # C. TEMPORAL FEATURES
    # =========================================================================
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    month = dates.dt.month.values
    day_of_year = dates.dt.dayofyear.values
    year = dates.dt.year.values

    # Cyclic encoding
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    # Season encoding
    df['is_summer'] = np.isin(month, [12, 1, 2]).astype(float)  # DJF
    df['is_autumn'] = np.isin(month, [3, 4, 5]).astype(float)   # MAM
    df['is_winter'] = np.isin(month, [6, 7, 8]).astype(float)   # JJA
    df['is_spring'] = np.isin(month, [9, 10, 11]).astype(float) # SON
    df['is_wet_season'] = np.isin(month, [10, 11, 12, 1, 2, 3]).astype(float)

    # Year as feature (water quality may have trends)
    df['year'] = year
    df['year_norm'] = (year - 2011) / 4.0  # Normalize to ~[0, 1]

    # =========================================================================
    # D. SPECTRAL FEATURES (Advanced)
    # =========================================================================
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)
    ndmi = ls_df['NDMI'].values.astype(float)
    mndwi = ls_df['MNDWI'].values.astype(float)

    # Raw bands (scaled)
    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22

    # Pre-computed indices
    df['NDMI'] = ndmi
    df['MNDWI'] = mndwi

    # --- Additional Spectral Indices ---

    # NDWI (Normalized Difference Water Index) - McFeeters
    df['NDWI'] = (green - nir) / (green + nir + eps)

    # Turbidity proxies
    df['turbidity'] = nir / (green + eps)
    df['turbidity_log'] = np.log1p(np.maximum(df['turbidity'], 0))

    # Sediment index
    df['sediment'] = (swir16 - green) / (swir16 + green + eps)

    # Water Ratio Index (WRI)
    df['WRI'] = (green + nir) / (swir16 + swir22 + eps)

    # Automated Water Extraction Index (AWEI)
    # Simplified version with available bands
    df['AWEI_sh'] = 4 * (green - swir16) - (0.25 * nir + 2.75 * swir22)
    df['AWEI_nsh'] = nir + 2.5 * green - 1.5 * (swir16 + swir22)

    # SWIR ratio (mineral/clay indicator)
    df['swir_ratio'] = swir22 / (swir16 + eps)

    # NIR/Green ratio (vegetation/turbidity)
    df['nir_green'] = nir / (green + eps)

    # Band differences (absolute values carry information)
    df['nir_minus_green'] = nir - green
    df['swir16_minus_swir22'] = swir16 - swir22
    df['nir_minus_swir16'] = nir - swir16
    df['nir_minus_swir22'] = nir - swir22

    # NDBI proxy (built-up index, using SWIR and NIR)
    df['NDBI'] = (swir16 - nir) / (swir16 + nir + eps)

    # Vegetation-like index (proxy for NDVI using available bands)
    # Since we don't have Red, use Green as proxy
    df['vegetation_proxy'] = (nir - green) / (nir + green + eps)

    # SAVI approximation (Soil Adjusted Vegetation Index)
    L = 0.5
    df['SAVI_proxy'] = ((nir - green) / (nir + green + L + eps)) * (1 + L)

    # Enhanced water turbidity indicators
    df['clarity_index'] = green / (nir + eps)  # Clear water: high green, low NIR
    df['suspended_solids'] = (nir + swir16) / (green + eps)

    # Mineral content indicator (relevant for alkalinity/conductance)
    df['mineral_index'] = (swir16 + swir22) / (nir + green + eps)

    # Reflectance statistics
    bands = np.column_stack([nir, green, swir16, swir22])
    df['reflect_mean'] = np.nanmean(bands, axis=1)
    df['reflect_std'] = np.nanstd(bands, axis=1)
    df['reflect_range'] = np.nanmax(bands, axis=1) - np.nanmin(bands, axis=1)
    df['reflect_cv'] = df['reflect_std'] / (df['reflect_mean'] + eps)
    df['reflect_max'] = np.nanmax(bands, axis=1)
    df['reflect_min'] = np.nanmin(bands, axis=1)
    df['reflect_skew'] = (df['reflect_mean'] - np.nanmedian(bands, axis=1)) / (df['reflect_std'] + eps)

    # Log-transformed bands
    df['nir_log'] = np.log1p(np.maximum(nir, 0))
    df['green_log'] = np.log1p(np.maximum(green, 0))
    df['swir16_log'] = np.log1p(np.maximum(swir16, 0))
    df['swir22_log'] = np.log1p(np.maximum(swir22, 0))

    # =========================================================================
    # E. CLIMATE FEATURES (Extended)
    # =========================================================================
    pet = tc_df['pet'].values.astype(float)
    df['pet'] = pet
    df['pet_log'] = np.log1p(np.maximum(pet, 0))

    # Extended climate data
    if climate_df is not None and len(climate_df) == n:
        # Precipitation features
        df['precip_30d'] = climate_df['precip_30d'].values
        df['precip_mean'] = climate_df['precip_mean'].values
        df['precip_max'] = climate_df['precip_max'].values
        df['precip_days'] = climate_df['precip_days'].values
        df['precip_30d_log'] = np.log1p(np.maximum(df['precip_30d'], 0))

        # Precipitation intensity
        df['precip_intensity'] = df['precip_30d'] / (df['precip_days'] + eps)
        df['precip_concentration'] = df['precip_max'] / (df['precip_30d'] + eps)

        # Temperature features
        df['temp_mean'] = climate_df['temp_mean'].values
        df['temp_max'] = climate_df['temp_max'].values
        df['temp_min'] = climate_df['temp_min'].values
        df['temp_range'] = climate_df['temp_range'].values
        df['temp_mean_sq'] = df['temp_mean'] ** 2

        # ET0 features
        df['et0_mean'] = climate_df['et0_mean'].values
        df['et0_sum'] = climate_df['et0_sum'].values

        # Water balance proxy
        df['water_balance'] = df['precip_30d'] - df['et0_sum']
        df['water_balance_ratio'] = df['precip_30d'] / (df['et0_sum'] + eps)
        df['aridity_index'] = df['pet'] / (df['precip_30d'] + eps)

        # Drought indicators
        df['is_dry'] = (df['precip_30d'] < 10).astype(float)
        df['is_very_dry'] = (df['precip_30d'] < 5).astype(float)
        df['dry_and_hot'] = df['is_dry'] * df['temp_mean']

        # Runoff proxy
        df['runoff_proxy'] = np.maximum(0, df['precip_30d'] - df['et0_sum'])
        df['runoff_proxy_log'] = np.log1p(df['runoff_proxy'])

    # =========================================================================
    # F. INTERACTION FEATURES
    # =========================================================================

    # Elevation × Climate
    df['elev_pet'] = df['elevation'] * pet
    df['elev_temp'] = df['elevation'] * df.get('temp_mean', pd.Series(np.zeros(n)))
    df['elev_precip'] = df['elevation'] * df.get('precip_30d', pd.Series(np.zeros(n)))

    # Elevation × Spectral
    df['elev_ndmi'] = df['elevation'] * df['NDMI']
    df['elev_turbidity'] = df['elevation'] * df['turbidity']
    df['elev_mineral'] = df['elevation'] * df['mineral_index']

    # Coast × Spectral
    df['coast_ndmi'] = df['dist_coast'] * df['NDMI']
    df['coast_turbidity'] = df['dist_coast'] * df['turbidity']
    df['coast_mineral'] = df['dist_coast'] * df['mineral_index']

    # Lat/Lon × Spectral
    df['lat_ndmi'] = lat * df['NDMI']
    df['lon_ndmi'] = lon * df['NDMI']
    df['lat_turbidity'] = lat * df['turbidity']

    # Season × Spectral
    df['wet_turb'] = df['is_wet_season'] * df['turbidity']
    df['wet_ndwi'] = df['is_wet_season'] * df['NDWI']
    df['wet_sediment'] = df['is_wet_season'] * df['sediment']
    df['wet_mineral'] = df['is_wet_season'] * df['mineral_index']

    # Season × Climate
    df['wet_precip'] = df['is_wet_season'] * df.get('precip_30d', pd.Series(np.zeros(n)))
    df['wet_temp'] = df['is_wet_season'] * df.get('temp_mean', pd.Series(np.zeros(n)))

    # Climate × Spectral (key interactions)
    if 'precip_30d' in df.columns:
        df['precip_turbidity'] = df['precip_30d'] * df['turbidity']
        df['precip_sediment'] = df['precip_30d'] * df['sediment']
        df['precip_ndwi'] = df['precip_30d'] * df['NDWI']
        df['temp_turbidity'] = df['temp_mean'] * df['turbidity']
        df['temp_mineral'] = df['temp_mean'] * df['mineral_index']
        df['aridity_mineral'] = df['aridity_index'] * df['mineral_index']
        df['water_bal_ndwi'] = df['water_balance'] * df['NDWI']

    # =========================================================================
    # G. LOCATION-BASED TEMPORAL STATISTICS
    # =========================================================================
    # For training points with multiple observations at same location,
    # compute temporal statistics of targets and features
    if train_wq is not None and target is not None:
        _add_location_statistics(df, wq_df, train_wq, ls_df, train_ls, target)

    # =========================================================================
    # H. SPATIAL IDW FEATURES (Enhanced)
    # =========================================================================
    if train_wq is not None and target is not None:
        _add_idw_features(df, lat, lon, train_wq, target, elev_cache)

    return df


def _add_location_statistics(df, wq_df, train_wq, ls_df, train_ls, target):
    """Add temporal statistics per location from training data."""
    n = len(wq_df)

    # Build location-based stats from training data
    train_locations = train_wq.groupby(
        [train_wq['Latitude'].round(2), train_wq['Longitude'].round(2)]
    )

    # Stats per location: mean, std, median, count of target
    loc_stats = train_locations[target].agg(['mean', 'std', 'median', 'count'])
    loc_stats.columns = ['loc_target_mean', 'loc_target_std',
                         'loc_target_median', 'loc_target_count']
    loc_stats['loc_target_std'] = loc_stats['loc_target_std'].fillna(0)

    # For each point in df, find matching location stats
    lat_r = wq_df['Latitude'].round(2)
    lon_r = wq_df['Longitude'].round(2)

    loc_means = []
    loc_stds = []
    loc_medians = []
    loc_counts = []

    for i in range(n):
        key = (lat_r.iloc[i], lon_r.iloc[i])
        if key in loc_stats.index:
            loc_means.append(loc_stats.loc[key, 'loc_target_mean'])
            loc_stds.append(loc_stats.loc[key, 'loc_target_std'])
            loc_medians.append(loc_stats.loc[key, 'loc_target_median'])
            loc_counts.append(loc_stats.loc[key, 'loc_target_count'])
        else:
            loc_means.append(np.nan)
            loc_stds.append(np.nan)
            loc_medians.append(np.nan)
            loc_counts.append(0)

    df['loc_target_mean'] = loc_means
    df['loc_target_std'] = loc_stds
    df['loc_target_median'] = loc_medians
    df['loc_target_count'] = loc_counts


def _add_idw_features(df, lat, lon, train_wq, target, elev_cache):
    """Add enhanced IDW features with environmental weighting."""
    train_coords = train_wq[['Latitude', 'Longitude']].values
    train_values = train_wq[target].values
    val_coords = np.column_stack([lat, lon])

    tree = cKDTree(train_coords)

    # === Standard Geographic IDW ===
    for k in [5, 10, 20, 30, 50]:
        k_actual = min(k, len(train_coords))
        dists, idxs = tree.query(val_coords, k=k_actual)
        dists = np.maximum(dists, 1e-10)

        # Power = 1 (linear)
        weights = 1.0 / dists
        weights = weights / weights.sum(axis=1, keepdims=True)
        df[f'idw_k{k}'] = np.sum(weights * train_values[idxs], axis=1)

        # Power = 2 (quadratic)
        weights2 = 1.0 / (dists ** 2)
        weights2 = weights2 / weights2.sum(axis=1, keepdims=True)
        df[f'idw_k{k}_p2'] = np.sum(weights2 * train_values[idxs], axis=1)

        # Power = 1.5
        weights15 = 1.0 / (dists ** 1.5)
        weights15 = weights15 / weights15.sum(axis=1, keepdims=True)
        df[f'idw_k{k}_p15'] = np.sum(weights15 * train_values[idxs], axis=1)

        # IDW std (uncertainty measure)
        idw_mean = df[f'idw_k{k}'].values
        df[f'idw_k{k}_std'] = np.sqrt(
            np.sum(weights * (train_values[idxs] - idw_mean[:, None])**2, axis=1)
        )

    # === Nearest neighbor features ===
    dists_1, idxs_1 = tree.query(val_coords, k=1)
    df['dist_nearest'] = dists_1
    df['dist_nearest_log'] = np.log1p(dists_1)
    df['nn_value'] = train_values[idxs_1]

    # Multiple nearest neighbors
    dists_3, idxs_3 = tree.query(val_coords, k=min(3, len(train_coords)))
    df['nn3_mean'] = np.mean(train_values[idxs_3], axis=1)
    df['nn3_std'] = np.std(train_values[idxs_3], axis=1)
    df['dist_3rd'] = dists_3[:, -1] if dists_3.ndim > 1 else dists_3

    dists_5, idxs_5 = tree.query(val_coords, k=min(5, len(train_coords)))
    df['nn5_mean'] = np.mean(train_values[idxs_5], axis=1)
    df['nn5_std'] = np.std(train_values[idxs_5], axis=1)

    # === Elevation-weighted IDW ===
    if elev_cache:
        train_elevs = np.array([
            get_elevation(la, lo, elev_cache)
            for la, lo in train_coords
        ])
        val_elevs = np.array([
            get_elevation(la, lo, elev_cache)
            for la, lo in val_coords
        ])

        k_env = min(30, len(train_coords))
        dists_env, idxs_env = tree.query(val_coords, k=k_env)
        dists_env = np.maximum(dists_env, 1e-10)

        # Combined distance: geographic + elevation difference
        elev_diffs = np.abs(val_elevs[:, None] - train_elevs[idxs_env])
        elev_diffs_norm = elev_diffs / (np.max(elev_diffs) + eps if (eps := 1e-10) else 1)
        geo_dists_norm = dists_env / (np.max(dists_env) + 1e-10)

        # Weighted combination
        combined_dist = 0.7 * geo_dists_norm + 0.3 * elev_diffs_norm
        combined_dist = np.maximum(combined_dist, 1e-10)

        weights_env = 1.0 / combined_dist
        weights_env = weights_env / weights_env.sum(axis=1, keepdims=True)
        df['idw_elev_weighted'] = np.sum(
            weights_env * train_values[idxs_env], axis=1
        )

    # === Directional IDW (N/S/E/W neighbors) ===
    # Find nearest neighbors in different directions
    for direction, condition in [
        ('north', lambda t, v: t[:, 0] > v[0]),
        ('south', lambda t, v: t[:, 0] < v[0]),
        ('east', lambda t, v: t[:, 1] > v[1]),
        ('west', lambda t, v: t[:, 1] < v[1]),
    ]:
        dir_values = []
        for i in range(len(val_coords)):
            mask = condition(train_coords, val_coords[i])
            if mask.sum() > 0:
                sub_coords = train_coords[mask]
                sub_vals = train_values[mask]
                sub_dists = np.sqrt(np.sum(
                    (sub_coords - val_coords[i])**2, axis=1
                ))
                k_dir = min(5, len(sub_vals))
                top_k = np.argsort(sub_dists)[:k_dir]
                w = 1.0 / np.maximum(sub_dists[top_k], 1e-10)
                w = w / w.sum()
                dir_values.append(np.sum(w * sub_vals[top_k]))
            else:
                dir_values.append(np.nan)
        df[f'idw_{direction}'] = dir_values

    # IDW range (max - min of nearby predictions)
    df['idw_range'] = df['idw_k5_p2'] - df['idw_k50']
    df['idw_agreement'] = df['idw_k5_p2'] / (df['idw_k50'] + 1e-10)


# =============================================================================
# 3. OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

def optimize_xgboost(X_train, y_train, groups, n_trials=50):
    """Optimize XGBoost hyperparameters with Optuna."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 20.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        kf = GroupKFold(n_splits=5)
        scores = []
        for train_idx, val_idx in kf.split(X_train, y_train, groups):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBRegressor(
                **params, random_state=42, n_jobs=-1, verbosity=0
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            scores.append(r2_score(y_va, pred))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_lightgbm(X_train, y_train, groups, n_trials=50):
    """Optimize LightGBM hyperparameters with Optuna."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 20.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        }

        kf = GroupKFold(n_splits=5)
        scores = []
        for train_idx, val_idx in kf.split(X_train, y_train, groups):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            model = lgb.LGBMRegressor(
                **params, random_state=42, n_jobs=-1, verbose=-1
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            scores.append(r2_score(y_va, pred))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# =============================================================================
# 4. STACKING ENSEMBLE
# =============================================================================

def train_stacking_ensemble(X_train, y_train, X_val, groups,
                             xgb_params, lgb_params, seeds=[42, 123, 456]):
    """
    Two-level stacking ensemble.
    Level 1: XGBoost, LightGBM, ExtraTrees, GradientBoosting
    Level 2: Ridge meta-learner trained on OOF predictions
    """
    n_train = len(X_train)
    n_val = len(X_val)

    # Level 1: Generate OOF predictions
    kf = GroupKFold(n_splits=5)
    folds = list(kf.split(X_train, y_train, groups))

    def train_base_model(ModelClass, params, seed):
        oof_preds = np.zeros(n_train)
        val_preds_list = []

        for train_idx, val_idx in folds:
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            if 'random_state' in params or hasattr(ModelClass(), 'random_state'):
                params_copy = {**params, 'random_state': seed}
            else:
                params_copy = params.copy()

            model = ModelClass(**params_copy)

            if isinstance(model, (lgb.LGBMRegressor,)):
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr)

            oof_preds[val_idx] = model.predict(X_va)
            val_preds_list.append(model.predict(X_val))

        val_pred = np.mean(val_preds_list, axis=0)
        return oof_preds, val_pred

    meta_train = []
    meta_val = []

    for seed in seeds:
        # XGBoost
        xgb_p = {**xgb_params, 'n_jobs': -1, 'verbosity': 0}
        oof, val = train_base_model(xgb.XGBRegressor, xgb_p, seed)
        meta_train.append(oof)
        meta_val.append(val)

        # LightGBM
        lgb_p = {**lgb_params, 'n_jobs': -1, 'verbose': -1}
        oof, val = train_base_model(lgb.LGBMRegressor, lgb_p, seed)
        meta_train.append(oof)
        meta_val.append(val)

        # ExtraTrees
        et_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_leaf': 10,
            'max_features': 0.6,
            'n_jobs': -1,
        }
        oof, val = train_base_model(ExtraTreesRegressor, et_params, seed)
        meta_train.append(oof)
        meta_val.append(val)

        # GradientBoosting (slower but different bias)
        gb_params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'min_samples_leaf': 15,
            'max_features': 0.6,
        }
        oof, val = train_base_model(GradientBoostingRegressor, gb_params, seed)
        meta_train.append(oof)
        meta_val.append(val)

    # Stack: combine OOF predictions as features
    meta_X_train = np.column_stack(meta_train)
    meta_X_val = np.column_stack(meta_val)

    # Level 2: Ridge meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X_train, y_train)

    stacking_pred = meta_model.predict(meta_X_val)

    # Also compute simple average for blending
    simple_avg = np.mean(meta_val, axis=0)

    # Return both for comparison
    return stacking_pred, simple_avg, meta_model.coef_


# =============================================================================
# 5. SPATIAL CROSS-VALIDATION
# =============================================================================

def spatial_cv_score(X_train, y_train, train_wq, model_fn, n_splits=5):
    """
    Spatial CV that groups by geographic region to simulate
    the train/validation geographic split.
    """
    lats = train_wq['Latitude'].values
    lons = train_wq['Longitude'].values

    # Create spatial groups based on geographic grid
    lat_bins = np.digitize(lats, np.linspace(lats.min(), lats.max(), n_splits + 1))
    lon_bins = np.digitize(lons, np.linspace(lons.min(), lons.max(), n_splits + 1))
    spatial_groups = lat_bins * 100 + lon_bins

    kf = GroupKFold(n_splits=min(n_splits, len(np.unique(spatial_groups))))
    scores = []

    for train_idx, val_idx in kf.split(X_train, y_train, spatial_groups):
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        scores.append(r2_score(y_va, pred))

    return np.mean(scores), np.std(scores)


# =============================================================================
# 6. TARGET-SPECIFIC TRANSFORMS
# =============================================================================

def get_target_config(target):
    """Get target-specific configuration."""
    configs = {
        'Total Alkalinity': {
            'clip': (0, 500),
            'use_yeo_johnson': True,
        },
        'Electrical Conductance': {
            'clip': (0, 2000),
            'use_yeo_johnson': True,
        },
        'Dissolved Reactive Phosphorus': {
            'clip': (0, 300),
            'use_yeo_johnson': True,
        },
    }
    return configs[target]


# =============================================================================
# 7. FEATURE IMPORTANCE AND SELECTION
# =============================================================================

def select_features(X_train, y_train, feature_names, groups, top_n=80):
    """Select top features using LightGBM importance."""
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

    # Keep top N features
    top_features = feature_imp.head(top_n)['feature'].values
    top_idx = [list(feature_names).index(f) for f in top_features]

    return top_idx, feature_imp


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('EY Water Quality v20.0 - Advanced Comprehensive Approach')
    print('=' * 70)

    # =========================================================================
    # Load data
    # =========================================================================
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
    print(f'   Elevation cache: {len(elev_cache)} entries')

    # Location groups for CV
    location_groups = (
        wq['Latitude'].round(2).astype(str) + '_' +
        wq['Longitude'].round(2).astype(str)
    )

    # =========================================================================
    # Process each target
    # =========================================================================
    predictions = {}
    predictions_stacking = {}
    predictions_avg = {}
    predictions_idw = {}

    for target in TARGET_COLS:
        print(f'\n{"="*70}')
        print(f'[2/7] Processing: {target}')
        print(f'{"="*70}')

        cfg = get_target_config(target)

        # =====================================================================
        # Feature Engineering
        # =====================================================================
        print('\n  [a] Creating features...')
        X_train_raw = create_advanced_features(
            wq, ls_train, tc_train, clim_train,
            train_wq=wq, target=target,
            train_ls=ls_train, elev_cache=elev_cache
        )
        X_val_raw = create_advanced_features(
            sub_template, ls_val, tc_val, clim_val,
            train_wq=wq, target=target,
            train_ls=ls_train, elev_cache=elev_cache
        )

        feature_names = X_train_raw.columns.tolist()
        print(f'     Total features: {len(feature_names)}')

        # =====================================================================
        # Preprocessing
        # =====================================================================
        print('  [b] Preprocessing...')

        # Handle missing values
        train_medians = X_train_raw.median()
        X_train = X_train_raw.fillna(train_medians)
        X_val = X_val_raw.fillna(train_medians)

        # Handle infinities
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_val = X_val.replace([np.inf, -np.inf], np.nan)
        train_medians2 = X_train.median()
        X_train = X_train.fillna(train_medians2)
        X_val = X_val.fillna(train_medians2)

        # Target transform
        y_raw = wq[target].values.copy()

        if cfg['use_yeo_johnson']:
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            y_train = pt.fit_transform(y_raw.reshape(-1, 1)).ravel()
        else:
            y_train = np.log1p(y_raw)

        # =====================================================================
        # Feature Selection
        # =====================================================================
        print('  [c] Feature selection...')
        scaler_fs = RobustScaler()
        X_train_sc = scaler_fs.fit_transform(X_train)
        X_val_sc = scaler_fs.transform(X_val)

        top_idx, feat_imp = select_features(
            X_train_sc, y_train, feature_names, location_groups, top_n=80
        )

        print(f'     Top 15 features:')
        for _, row in feat_imp.head(15).iterrows():
            print(f'       {row["feature"]}: {row["importance"]:.0f}')

        X_train_sel = X_train_sc[:, top_idx]
        X_val_sel = X_val_sc[:, top_idx]
        selected_features = [feature_names[i] for i in top_idx]

        # =====================================================================
        # Hyperparameter Optimization
        # =====================================================================
        print('  [d] Optuna hyperparameter optimization...')
        print('     Optimizing XGBoost...')
        xgb_best = optimize_xgboost(
            X_train_sel, y_train, location_groups, n_trials=60
        )
        print(f'     XGBoost best: depth={xgb_best["max_depth"]}, '
              f'lr={xgb_best["learning_rate"]:.4f}, '
              f'n_est={xgb_best["n_estimators"]}')

        print('     Optimizing LightGBM...')
        lgb_best = optimize_lightgbm(
            X_train_sel, y_train, location_groups, n_trials=60
        )
        print(f'     LightGBM best: depth={lgb_best["max_depth"]}, '
              f'lr={lgb_best["learning_rate"]:.4f}, '
              f'n_est={lgb_best["n_estimators"]}')

        # =====================================================================
        # Stacking Ensemble
        # =====================================================================
        print('  [e] Training stacking ensemble...')
        stacking_pred, avg_pred, meta_coefs = train_stacking_ensemble(
            X_train_sel, y_train, X_val_sel, location_groups,
            xgb_best, lgb_best, seeds=[42, 123, 456]
        )
        print(f'     Meta-learner coefficients: {meta_coefs}')

        # =====================================================================
        # Spatial CV Score
        # =====================================================================
        print('  [f] Spatial CV evaluation...')
        def make_model():
            return xgb.XGBRegressor(
                **xgb_best, random_state=42, n_jobs=-1, verbosity=0
            )

        cv_mean, cv_std = spatial_cv_score(
            X_train_sel, y_train, wq, make_model, n_splits=5
        )
        print(f'     Spatial CV R²: {cv_mean:.4f} +/- {cv_std:.4f}')

        # Standard GroupKFold CV
        kf = GroupKFold(n_splits=5)
        gkf_scores = []
        for train_idx, val_idx in kf.split(X_train_sel, y_train, location_groups):
            X_tr, X_va = X_train_sel[train_idx], X_train_sel[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]
            m = xgb.XGBRegressor(**xgb_best, random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_va)
            gkf_scores.append(r2_score(y_va, pred))
        print(f'     GroupKFold CV R²: {np.mean(gkf_scores):.4f} +/- {np.std(gkf_scores):.4f}')

        # =====================================================================
        # Inverse Transform
        # =====================================================================
        if cfg['use_yeo_johnson']:
            stacking_final = pt.inverse_transform(
                stacking_pred.reshape(-1, 1)
            ).ravel()
            avg_final = pt.inverse_transform(
                avg_pred.reshape(-1, 1)
            ).ravel()
        else:
            stacking_final = np.expm1(stacking_pred)
            avg_final = np.expm1(avg_pred)

        # Clip
        stacking_final = np.clip(stacking_final, cfg['clip'][0], cfg['clip'][1])
        avg_final = np.clip(avg_final, cfg['clip'][0], cfg['clip'][1])

        # IDW prediction (best performing k)
        idw_pred = X_val_raw['idw_k30'].values if 'idw_k30' in X_val_raw.columns else avg_final

        predictions_stacking[target] = stacking_final
        predictions_avg[target] = avg_final
        predictions_idw[target] = idw_pred

        print(f'\n  Results for {target}:')
        print(f'    Stacking: mean={stacking_final.mean():.2f}, std={stacking_final.std():.2f}')
        print(f'    Average:  mean={avg_final.mean():.2f}, std={avg_final.std():.2f}')
        print(f'    IDW:      mean={idw_pred.mean():.2f}, std={idw_pred.std():.2f}')

    # =========================================================================
    # Create blended submissions at various ML/IDW ratios
    # =========================================================================
    print(f'\n{"="*70}')
    print('[7/7] Creating submissions...')
    print(f'{"="*70}')

    blend_ratios = [
        (1.0, 0.0, 'pure_ml'),
        (0.7, 0.3, 'ml70_idw30'),
        (0.5, 0.5, 'ml50_idw50'),
        (0.4, 0.6, 'ml40_idw60'),
        (0.3, 0.7, 'ml30_idw70'),
        (0.0, 1.0, 'pure_idw'),
    ]

    for ml_w, idw_w, name in blend_ratios:
        sub = pd.DataFrame({
            'Latitude': sub_template['Latitude'],
            'Longitude': sub_template['Longitude'],
            'Sample Date': sub_template['Sample Date'],
        })
        for target in TARGET_COLS:
            cfg = get_target_config(target)
            ml_pred = predictions_stacking[target]
            idw_pred = predictions_idw[target]
            blended = ml_w * ml_pred + idw_w * idw_pred
            blended = np.clip(blended, cfg['clip'][0], cfg['clip'][1])
            sub[target] = blended

        fname = f'submission_v20_{name}.csv'
        sub.to_csv(os.path.join(BASE_DIR, fname), index=False)
        print(f'  {fname}')
        for target in TARGET_COLS:
            print(f'    {target}: mean={sub[target].mean():.2f}, std={sub[target].std():.2f}')

    # Also create stacking-only and average-only submissions
    for pred_dict, label in [
        (predictions_stacking, 'stacking'),
        (predictions_avg, 'avg_ensemble'),
    ]:
        sub = pd.DataFrame({
            'Latitude': sub_template['Latitude'],
            'Longitude': sub_template['Longitude'],
            'Sample Date': sub_template['Sample Date'],
        })
        for target in TARGET_COLS:
            cfg = get_target_config(target)
            sub[target] = np.clip(pred_dict[target], cfg['clip'][0], cfg['clip'][1])
        fname = f'submission_v20_{label}.csv'
        sub.to_csv(os.path.join(BASE_DIR, fname), index=False)
        print(f'  {fname}')

    # =========================================================================
    # Summary
    # =========================================================================
    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    print('\nAll submission files:')
    for ml_w, idw_w, name in blend_ratios:
        print(f'  submission_v20_{name}.csv  (ML={ml_w:.0%}, IDW={idw_w:.0%})')
    print(f'  submission_v20_stacking.csv  (stacking meta-learner)')
    print(f'  submission_v20_avg_ensemble.csv  (simple average)')

    print('\nRecommendation: Try submission_v20_stacking.csv first,')
    print('then submission_v20_ml50_idw50.csv and submission_v20_ml70_idw30.csv')
    print('to find optimal ML/IDW blend ratio.')
    print(f'\n{"="*70}')


if __name__ == '__main__':
    main()
