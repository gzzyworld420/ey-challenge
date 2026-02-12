#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v16.0
REAL EXTERNAL DATA from public APIs.

Data sources:
1. Open-Elevation API - Real elevation data
2. Open-Meteo Archive API - Historical climate (precipitation, temperature)

These real environmental data should help generalize to new regions.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
import time
import json
import os
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
CACHE_FILE = 'external_data_cache.json'


def load_cache():
    """Load cached external data."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {'elevation': {}, 'climate': {}}


def save_cache(cache):
    """Save external data cache."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def get_elevation(lat, lon, cache):
    """Get elevation from API or cache."""
    key = f"{lat:.4f},{lon:.4f}"
    if key in cache['elevation']:
        return cache['elevation'][key]

    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            elev = data['results'][0]['elevation']
            cache['elevation'][key] = elev
            return elev
    except:
        pass

    # Fallback estimate
    dist_east = max(0, 30 - lon)
    dist_south = max(0, lat + 34)
    elev = 200 + dist_east * 80 + dist_south * 50
    cache['elevation'][key] = elev
    return elev


def get_climate_data(lat, lon, date_str, cache):
    """Get climate data from Open-Meteo Archive API."""
    # Convert date
    try:
        date = datetime.strptime(date_str, '%d-%m-%Y')
    except:
        return None, None

    # Get 30-day window before sample date
    start_date = (date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = date.strftime('%Y-%m-%d')

    key = f"{lat:.2f},{lon:.2f},{start_date},{end_date}"
    if key in cache['climate']:
        return cache['climate'][key]

    try:
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}"
               f"&start_date={start_date}&end_date={end_date}"
               f"&daily=precipitation_sum,temperature_2m_mean,et0_fao_evapotranspiration")

        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            daily = data.get('daily', {})

            precip = daily.get('precipitation_sum', [])
            temp = daily.get('temperature_2m_mean', [])
            et0 = daily.get('et0_fao_evapotranspiration', [])

            result = {
                'precip_30d': np.nansum(precip) if precip else 0,
                'precip_mean': np.nanmean(precip) if precip else 0,
                'temp_mean': np.nanmean(temp) if temp else 20,
                'temp_std': np.nanstd(temp) if temp and len(temp) > 1 else 0,
                'et0_mean': np.nanmean(et0) if et0 else 3,
            }
            cache['climate'][key] = result
            return result
    except:
        pass

    # Fallback
    return {'precip_30d': 50, 'precip_mean': 1.5, 'temp_mean': 20, 'temp_std': 3, 'et0_mean': 3}


def fetch_all_external_data(df, cache, desc=""):
    """Fetch external data for all rows in dataframe."""
    n = len(df)
    elevations = []
    climate_data = []

    unique_locs = df[['Latitude', 'Longitude']].drop_duplicates()
    print(f'      {desc}: Fetching data for {len(unique_locs)} unique locations...')

    # Batch elevation requests
    loc_elevations = {}
    for i, (_, row) in enumerate(unique_locs.iterrows()):
        lat, lon = row['Latitude'], row['Longitude']
        key = f"{lat:.4f},{lon:.4f}"

        if key not in cache['elevation']:
            elev = get_elevation(lat, lon, cache)
            time.sleep(0.3)  # Rate limiting

            if i % 20 == 0:
                print(f'         Elevation: {i}/{len(unique_locs)}')
                save_cache(cache)  # Periodic save

        loc_elevations[key] = cache['elevation'].get(key, 1000)

    # Fetch climate data (sample-specific due to dates)
    sample_climate = []
    for i in range(n):
        lat = df['Latitude'].values[i]
        lon = df['Longitude'].values[i]
        date_str = df['Sample Date'].values[i]

        climate = get_climate_data(lat, lon, date_str, cache)

        if climate is None:
            climate = {'precip_30d': 50, 'precip_mean': 1.5, 'temp_mean': 20, 'temp_std': 3, 'et0_mean': 3}

        sample_climate.append(climate)

        if i % 50 == 0:
            print(f'         Climate: {i}/{n}')
            save_cache(cache)

        time.sleep(0.1)

    # Map elevations to all rows
    for i in range(n):
        lat = df['Latitude'].values[i]
        lon = df['Longitude'].values[i]
        key = f"{lat:.4f},{lon:.4f}"
        elevations.append(loc_elevations.get(key, 1000))

    save_cache(cache)

    return np.array(elevations), sample_climate


def create_features(wq_df, ls_df, tc_df, elevations, climate_data, train_wq=None, target=None):
    """Create features using real external data."""
    df = pd.DataFrame()
    eps = 1e-10
    n = len(wq_df)

    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values

    # === COORDINATES ===
    df['Latitude'] = lat
    df['Longitude'] = lon

    # === REAL ELEVATION ===
    df['elevation'] = elevations
    df['elevation_log'] = np.log1p(elevations)
    df['is_lowland'] = (elevations < 500).astype(float)
    df['is_highland'] = (elevations >= 1000).astype(float)

    # === REAL CLIMATE ===
    df['precip_30d'] = [c['precip_30d'] for c in climate_data]
    df['precip_mean'] = [c['precip_mean'] for c in climate_data]
    df['temp_mean'] = [c['temp_mean'] for c in climate_data]
    df['temp_std'] = [c['temp_std'] for c in climate_data]
    df['et0_mean'] = [c['et0_mean'] for c in climate_data]

    # Climate derived
    df['precip_log'] = np.log1p(df['precip_30d'])
    df['aridity'] = df['et0_mean'] / (df['precip_mean'] + 0.1)
    df['temp_precip'] = df['temp_mean'] * df['precip_mean']

    # === TOPOGRAPHIC ===
    df['dist_coast'] = np.array([min(max(0, 30 - lo) * 111, max(0, la + 34) * 111)
                                  for la, lo in zip(lat, lon)])

    # === TEMPORAL ===
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    month = dates.dt.month
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

    # === TERRACLIMATE PET (original) ===
    df['pet'] = tc_df['pet'].values

    # === INTERACTIONS ===
    df['elev_precip'] = df['elevation'] * df['precip_30d']
    df['elev_temp'] = df['elevation'] * df['temp_mean']
    df['precip_ndmi'] = df['precip_30d'] * df['NDMI']
    df['temp_turbidity'] = df['temp_mean'] * df['turbidity']

    # === SPATIAL FEATURES (if training data provided) ===
    if train_wq is not None and target is not None:
        train_coords = train_wq[['Latitude', 'Longitude']].values
        train_values = train_wq[target].values
        val_coords = np.column_stack([lat, lon])

        tree = cKDTree(train_coords)
        dists, idxs = tree.query(val_coords, k=min(30, len(train_coords)))
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        weights = weights / weights.sum(axis=1, keepdims=True)
        df['spatial_idw'] = np.sum(weights * train_values[idxs], axis=1)

    return df


def main():
    print('=' * 70)
    print('EY Water Quality v16.0 - Real External Data')
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

    # Load/create cache
    cache = load_cache()
    print(f'   Cache: {len(cache["elevation"])} elevations, {len(cache["climate"])} climate records')

    # Fetch external data
    print('\n2. Fetching external data (this may take a while)...')
    print('   Training data:')
    train_elevations, train_climate = fetch_all_external_data(wq, cache, "Training")
    print('   Validation data:')
    val_elevations, val_climate = fetch_all_external_data(sub_template, cache, "Validation")

    save_cache(cache)
    print(f'   Final cache: {len(cache["elevation"])} elevations, {len(cache["climate"])} climate records')

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
        print(f'\n3. Processing {target}...')
        cfg = target_configs[target]

        # Create features
        X_train_raw = create_features(wq, ls_train, tc_train, train_elevations, train_climate, wq, target)
        X_val_raw = create_features(sub_template, ls_val, tc_val, val_elevations, val_climate, wq, target)

        print(f'   Features: {X_train_raw.shape[1]}')

        # Preprocess
        X_train = X_train_raw.fillna(X_train_raw.median())
        X_val = X_val_raw.fillna(X_train_raw.median())
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        y_raw = wq[target].values
        y_train = cfg['transform'](y_raw)

        # CV
        print('   Cross-validation...')
        kf = GroupKFold(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train_scaled, y_train, location_groups):
            X_tr, X_va = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]
            model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      min_child_weight=15, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            ss_res = np.sum((y_va - pred) ** 2)
            ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
            cv_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        print(f'   CV R²: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}')

        # Train ensemble
        print('   Training ensemble...')
        all_preds = []
        for seed in [42, 123, 456]:
            model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      min_child_weight=15, random_state=seed, n_jobs=-1, verbosity=0)
            model.fit(X_train_scaled, y_train)
            all_preds.append(cfg['inverse'](model.predict(X_val_scaled)))

            if HAS_LIGHTGBM:
                lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                               min_child_samples=20, random_state=seed, n_jobs=-1, verbose=-1)
                lgb_model.fit(X_train_scaled, y_train)
                all_preds.append(cfg['inverse'](lgb_model.predict(X_val_scaled)))

        # Blend with spatial IDW
        ml_pred = np.mean(all_preds, axis=0)
        spatial_pred = X_val_raw['spatial_idw'].values
        final_pred = 0.5 * ml_pred + 0.5 * spatial_pred
        final_pred = np.clip(final_pred, cfg['clip'][0], cfg['clip'][1])

        predictions[target] = final_pred
        print(f'   Final: mean={final_pred.mean():.2f}, std={final_pred.std():.2f}')

    # Create submission
    print('\n4. Creating submission...')
    submission = pd.DataFrame({
        'Latitude': sub_template['Latitude'],
        'Longitude': sub_template['Longitude'],
        'Sample Date': sub_template['Sample Date'],
        'Total Alkalinity': predictions['Total Alkalinity'],
        'Electrical Conductance': predictions['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': predictions['Dissolved Reactive Phosphorus'],
    })
    submission.to_csv('submission_v16.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print('\nSubmission: submission_v16.csv')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}')


if __name__ == '__main__':
    main()
