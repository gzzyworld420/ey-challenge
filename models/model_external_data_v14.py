#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v14.0
EXTERNAL DATA APPROACH - Features that generalize across regions.

Key insight from challenge description:
- Validation locations are from DIFFERENT REGIONS not in training
- We are ENCOURAGED to use external public datasets
- Landsat and TerraClimate are just "starting points"

Strategy:
1. Download elevation data (SRTM/open-elevation API)
2. Get watershed/catchment characteristics
3. Land use/land cover data
4. Climate variables beyond PET
5. Soil type/geology proxies

These physical characteristics should generalize across regions.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
import time
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']


def get_elevation_batch(coords, batch_size=100):
    """Get elevation data from Open-Elevation API."""
    elevations = []

    for i in range(0, len(coords), batch_size):
        batch = coords[i:i+batch_size]
        locations = "|".join([f"{lat},{lon}" for lat, lon in batch])

        try:
            # Try open-elevation API
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                for result in data.get('results', []):
                    elevations.append(result.get('elevation', np.nan))
            else:
                # Fallback: estimate from latitude (rough proxy for SA)
                for lat, lon in batch:
                    # South Africa elevation roughly correlates with distance from coast
                    # and latitude (higher inland plateau)
                    est_elev = 1000 + (lon - 20) * 50 - (lat + 30) * 30
                    elevations.append(max(0, est_elev))

        except Exception as e:
            print(f"   Elevation API error: {e}, using estimates")
            for lat, lon in batch:
                est_elev = 1000 + (lon - 20) * 50 - (lat + 30) * 30
                elevations.append(max(0, est_elev))

        time.sleep(0.5)  # Rate limiting

    return np.array(elevations)


def estimate_elevation_from_coords(lat, lon):
    """
    Estimate elevation based on South African geography.
    - Eastern escarpment rises steeply
    - Interior plateau ~1000-1500m
    - Coastal areas lower
    """
    # Distance from coast (approximation)
    # East coast roughly at lon=30-32, south coast at lat=-34
    dist_east_coast = max(0, 30 - lon)
    dist_south_coast = max(0, lat + 34)

    # Base elevation increases inland
    base_elev = 200 + dist_east_coast * 80 + dist_south_coast * 50

    # Adjust for Great Escarpment (Drakensberg region)
    if lon < 30 and lat > -32:
        base_elev += 300

    return min(base_elev, 2000)


def get_slope_aspect(lat, lon, elevation, neighbors_lat, neighbors_lon, neighbors_elev):
    """Estimate slope and aspect from nearby elevation differences."""
    if len(neighbors_elev) < 3:
        return 0, 0

    # Simple gradient estimation
    dlat = neighbors_lat - lat
    dlon = neighbors_lon - lon
    delev = neighbors_elev - elevation

    # Avoid division by zero
    dist = np.sqrt(dlat**2 + dlon**2) * 111  # Convert to km (roughly)
    dist = np.maximum(dist, 0.1)

    slopes = np.abs(delev) / dist
    mean_slope = np.mean(slopes)

    # Aspect: direction of steepest descent
    if np.sum(np.abs(delev)) > 0:
        aspect = np.arctan2(np.mean(dlat * np.sign(delev)),
                           np.mean(dlon * np.sign(delev)))
    else:
        aspect = 0

    return mean_slope, aspect


def create_environmental_features(wq_df, ls_df, tc_df, all_coords=None, all_elevations=None):
    """
    Create features based on environmental characteristics.
    These should generalize across different regions.
    """
    df = pd.DataFrame()
    eps = 1e-10
    n = len(wq_df)

    lat = wq_df['Latitude'].values
    lon = wq_df['Longitude'].values

    # === BASIC COORDINATES ===
    df['Latitude'] = lat
    df['Longitude'] = lon

    # === ELEVATION FEATURES ===
    print('      Getting elevation data...')
    if all_elevations is not None:
        elevations = all_elevations
    else:
        elevations = np.array([estimate_elevation_from_coords(la, lo)
                               for la, lo in zip(lat, lon)])

    df['elevation'] = elevations
    df['elevation_log'] = np.log1p(elevations)

    # Elevation categories (influences water chemistry)
    df['is_lowland'] = (elevations < 500).astype(float)
    df['is_midland'] = ((elevations >= 500) & (elevations < 1000)).astype(float)
    df['is_highland'] = (elevations >= 1000).astype(float)

    # === TOPOGRAPHIC FEATURES ===
    # Distance to coast (major factor in water chemistry)
    # South Africa: East coast ~lon 30, South coast ~lat -34
    df['dist_east_coast'] = np.maximum(0, 30 - lon)
    df['dist_south_coast'] = np.maximum(0, lat + 34)
    df['dist_coast_min'] = np.minimum(df['dist_east_coast'] * 111,
                                       df['dist_south_coast'] * 111)  # km

    # Continentality index (distance from ocean)
    df['continentality'] = df['dist_coast_min'] / 500  # Normalized

    # === TEMPORAL FEATURES ===
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    month = dates.dt.month
    doy = dates.dt.dayofyear

    df['month'] = month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    # Seasons (Southern Hemisphere)
    df['is_summer'] = month.isin([12, 1, 2]).astype(float)  # Wet season
    df['is_winter'] = month.isin([6, 7, 8]).astype(float)   # Dry season
    df['is_wet_season'] = month.isin([10, 11, 12, 1, 2, 3]).astype(float)

    # === SPECTRAL FEATURES (water-related) ===
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    df['nir'] = nir
    df['green'] = green
    df['swir16'] = swir16
    df['swir22'] = swir22

    # Water indices (generalize across regions)
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values
    df['NDWI'] = (green - nir) / (green + nir + eps)

    # Water quality indicators from spectral
    df['turbidity'] = nir / (green + eps)
    df['sediment_load'] = (swir16 - green) / (swir16 + green + eps)
    df['organic_matter'] = (green - swir22) / (green + swir22 + eps)

    # Reflectance characteristics
    bands = np.column_stack([nir, green, swir16, swir22])
    df['brightness'] = np.nanmean(bands, axis=1)
    df['spectral_variability'] = np.nanstd(bands, axis=1)

    # === CLIMATE FEATURES ===
    pet = tc_df['pet'].values.astype(float)
    df['pet'] = pet
    df['pet_log'] = np.log1p(np.maximum(pet, 0))

    # Climate zone indicators
    df['pet_low'] = (pet < 80).astype(float)
    df['pet_medium'] = ((pet >= 80) & (pet < 120)).astype(float)
    df['pet_high'] = (pet >= 120).astype(float)

    # === REGIONAL CHARACTERISTICS ===
    # South African geographic regions (important for water chemistry)
    # Western Cape: drier, different geology
    df['is_western_region'] = (lon < 22).astype(float)
    # Eastern Cape/KZN: wetter, different geology
    df['is_eastern_region'] = (lon > 28).astype(float)
    # Central plateau
    df['is_central_region'] = ((lon >= 22) & (lon <= 28)).astype(float)

    # Latitude bands
    df['is_subtropical'] = (lat > -30).astype(float)
    df['is_temperate'] = (lat <= -30).astype(float)

    # === INTERACTION FEATURES ===
    # Elevation-climate interactions
    df['elev_pet'] = df['elevation'] * pet
    df['elev_wet_season'] = df['elevation'] * df['is_wet_season']

    # Location-spectral interactions
    df['continentality_ndmi'] = df['continentality'] * df['NDMI']
    df['elev_turbidity'] = df['elevation'] * df['turbidity']

    # Season-spectral interactions
    df['wet_season_ndmi'] = df['is_wet_season'] * df['NDMI']
    df['summer_turbidity'] = df['is_summer'] * df['turbidity']

    # Region-climate interactions
    df['eastern_pet'] = df['is_eastern_region'] * pet
    df['western_pet'] = df['is_western_region'] * pet

    return df


def train_model_ensemble(X_train, y_train, X_val, seeds=[42, 123, 456]):
    """Train ensemble of models with strong regularization."""
    all_preds = []

    for seed in seeds:
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=20,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        all_preds.append(xgb_model.predict(X_val))

        # LightGBM
        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.6,
                min_child_samples=30,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=seed,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            all_preds.append(lgb_model.predict(X_val))

        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        all_preds.append(rf_model.predict(X_val))

    return np.mean(all_preds, axis=0)


def get_feature_importance(X_train, y_train, feature_names):
    """Get feature importance from XGBoost."""
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance


def main():
    print('=' * 70)
    print('EY Water Quality v14.0 - External Data / Environmental Features')
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

    # Create environmental features
    print('\n2. Creating environmental features...')
    print('   Processing training data...')
    X_train_raw = create_environmental_features(wq, ls_train, tc_train)
    print('   Processing validation data...')
    X_val_raw = create_environmental_features(sub_template, ls_val, tc_val)

    print(f'   Total features: {X_train_raw.shape[1]}')

    # Handle missing values
    print('\n3. Preprocessing...')
    X_train = X_train_raw.fillna(X_train_raw.median())
    X_val = X_val_raw.fillna(X_train_raw.median())

    # Handle infinities
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    feature_names = X_train.columns.tolist()

    # Location groups for CV
    location_groups = wq['Latitude'].round(2).astype(str) + '_' + wq['Longitude'].round(2).astype(str)

    # Target configs
    target_configs = {
        'Total Alkalinity': {
            'transform': np.log1p,
            'inverse': np.expm1,
            'clip': (0, 500)
        },
        'Electrical Conductance': {
            'transform': np.log1p,
            'inverse': np.expm1,
            'clip': (0, 2000)
        },
        'Dissolved Reactive Phosphorus': {
            'transform': lambda x: np.log1p(x + 1),
            'inverse': lambda x: np.clip(np.expm1(x) - 1, 0, None),
            'clip': (0, 300)
        },
    }

    predictions = {}
    feature_importances = {}

    for target in TARGET_COLS:
        print(f'\n4. Processing {target}...')
        cfg = target_configs[target]

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
                min_child_weight=20, reg_alpha=1.0, reg_lambda=5.0,
                random_state=42, n_jobs=-1, verbosity=0
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)

            # R² on transformed scale
            ss_res = np.sum((y_va - pred) ** 2)
            ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            cv_scores.append(r2)

        print(f'   CV R² (log scale): {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}')

        # Feature importance
        importance = get_feature_importance(X_train_scaled, y_train, feature_names)
        feature_importances[target] = importance
        print(f'   Top 5 features: {importance.head(5)["feature"].tolist()}')

        # Train final model
        print('   Training ensemble...')
        pred_transformed = train_model_ensemble(X_train_scaled, y_train, X_val_scaled)
        pred_final = cfg['inverse'](pred_transformed)
        pred_final = np.clip(pred_final, cfg['clip'][0], cfg['clip'][1])

        predictions[target] = pred_final
        print(f'   Predictions: mean={pred_final.mean():.2f}, std={pred_final.std():.2f}')

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

    submission.to_csv('submission_v14.csv', index=False)

    # Save feature importance
    print('\n6. Feature Importance Analysis...')
    for target in TARGET_COLS:
        print(f'\n{target} - Top 10 features:')
        print(feature_importances[target].head(10).to_string(index=False))

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print('\nSubmission: submission_v14.csv')
    print('\nPrediction Summary:')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}')


if __name__ == '__main__':
    main()
