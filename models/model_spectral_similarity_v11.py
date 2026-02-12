#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v11.0
Spectral-Spatial Similarity Interpolation.

Key insight:
- LOO-IDW within training gives R² ~0.76 (similar to leader!)
- But validation locations are FAR from training (~0.8-2.4 degrees)
- Need to find training samples that are SIMILAR to validation
  not just geographically close

Strategy:
- Combine spatial distance with spectral similarity
- Find training samples with similar spectral signatures
- Use weighted interpolation based on both distance AND similarity
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']


def get_spectral_features(ls_df):
    """Extract key spectral features for similarity matching."""
    eps = 1e-10
    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)
    swir22 = ls_df['swir22'].values.astype(float)

    features = {
        'NDMI': ls_df['NDMI'].values,
        'MNDWI': ls_df['MNDWI'].values,
        'NDWI': (green - nir) / (green + nir + eps),
        'turbidity': nir / (green + eps),
        'swir_ratio': swir22 / (swir16 + eps),
        'reflectance_mean': np.nanmean([nir, green, swir16, swir22], axis=0),
    }

    return pd.DataFrame(features)


def spectral_spatial_interpolation(train_coords, train_spectral, train_values,
                                    val_coords, val_spectral,
                                    k_spatial=20, k_spectral=50,
                                    spatial_weight=0.5, spectral_weight=0.5):
    """
    Interpolation using both spatial distance and spectral similarity.
    """
    n_val = len(val_coords)

    # Handle NaN in spectral features
    train_spectral_clean = train_spectral.fillna(train_spectral.median())
    val_spectral_clean = val_spectral.fillna(train_spectral.median())

    # Normalize spectral features
    scaler = StandardScaler()
    train_spectral_scaled = scaler.fit_transform(train_spectral_clean)
    val_spectral_scaled = scaler.transform(val_spectral_clean)

    # Spatial nearest neighbors
    spatial_tree = cKDTree(train_coords)
    spatial_dists, spatial_idxs = spatial_tree.query(val_coords, k=min(k_spatial, len(train_coords)))

    # Spectral nearest neighbors
    spectral_nn = NearestNeighbors(n_neighbors=min(k_spectral, len(train_coords)), metric='euclidean')
    spectral_nn.fit(train_spectral_scaled)
    spectral_dists, spectral_idxs = spectral_nn.kneighbors(val_spectral_scaled)

    predictions = []

    for i in range(n_val):
        # Get candidates from both spatial and spectral neighbors
        candidates = set(spatial_idxs[i].tolist() + spectral_idxs[i].tolist())
        candidates = list(candidates)

        if len(candidates) == 0:
            predictions.append(np.mean(train_values))
            continue

        # Compute combined weights for candidates
        weights = []
        values = []

        for idx in candidates:
            # Spatial distance
            sp_dist = np.linalg.norm(val_coords[i] - train_coords[idx])
            sp_weight = 1.0 / (sp_dist ** 2 + 0.1)

            # Spectral distance
            spec_dist = np.linalg.norm(val_spectral_scaled[i] - train_spectral_scaled[idx])
            spec_weight = 1.0 / (spec_dist ** 2 + 0.1)

            # Combined weight
            combined = spatial_weight * sp_weight + spectral_weight * spec_weight
            weights.append(combined)
            values.append(train_values[idx])

        weights = np.array(weights)
        values = np.array(values)
        weights = weights / weights.sum()

        pred = np.sum(weights * values)
        predictions.append(pred)

    return np.array(predictions)


def adaptive_idw(train_coords, train_values, val_coords, k=30, power=2):
    """
    Adaptive IDW that adjusts power based on distance distribution.
    """
    tree = cKDTree(train_coords)
    dists, idxs = tree.query(val_coords, k=min(k, len(train_coords)))

    if len(dists.shape) == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    predictions = []

    for i in range(len(val_coords)):
        d = np.maximum(dists[i], 1e-10)
        v = train_values[idxs[i]]

        # Adaptive power based on distance spread
        dist_spread = d.max() / d.min() if d.min() > 0 else 1
        adaptive_power = power * (1 + np.log1p(dist_spread) / 5)
        adaptive_power = min(adaptive_power, 4)  # Cap at 4

        w = 1.0 / (d ** adaptive_power)
        w = w / w.sum()

        pred = np.sum(w * v)
        predictions.append(pred)

    return np.array(predictions)


def train_residual_model(train_coords, train_spectral, train_values, train_idw_pred,
                         val_coords, val_spectral, val_idw_pred, seed=42):
    """Train a model to predict residuals from IDW."""
    residuals = train_values - train_idw_pred

    # Features: coordinates + spectral
    X_train = pd.concat([
        pd.DataFrame(train_coords, columns=['lat', 'lon']),
        train_spectral.reset_index(drop=True)
    ], axis=1).fillna(0)

    X_val = pd.concat([
        pd.DataFrame(val_coords, columns=['lat', 'lon']),
        val_spectral.reset_index(drop=True)
    ], axis=1).fillna(0)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Simple model
    if HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            min_child_samples=100, reg_alpha=5.0, reg_lambda=10.0,
            random_state=seed, n_jobs=-1, verbose=-1
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            min_child_weight=100, reg_alpha=5.0, reg_lambda=10.0,
            random_state=seed, n_jobs=-1, verbosity=0
        )

    model.fit(X_train_scaled, residuals)
    residual_pred = model.predict(X_val_scaled)

    return residual_pred


def main():
    print('=' * 70)
    print('EY Water Quality v11.0 - Spectral-Spatial Similarity')
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

    # Coordinates
    train_coords = wq[['Latitude', 'Longitude']].values
    val_coords = sub_template[['Latitude', 'Longitude']].values

    # Spectral features
    train_spectral = get_spectral_features(ls_train)
    val_spectral = get_spectral_features(ls_val)

    print(f'   Spectral features: {train_spectral.shape[1]}')

    target_configs = {
        'Total Alkalinity': {'clip': (0, 500)},
        'Electrical Conductance': {'clip': (0, 2000)},
        'Dissolved Reactive Phosphorus': {'clip': (0, 300)},
    }

    predictions = {}

    for target in TARGET_COLS:
        print(f'\n2. Processing {target}...')
        cfg = target_configs[target]
        y = wq[target].values

        # Method 1: Pure IDW
        print('   Computing IDW...')
        idw_pred = adaptive_idw(train_coords, y, val_coords, k=30, power=2)
        print(f'   IDW: [{idw_pred.min():.2f}, {idw_pred.max():.2f}]')

        # Method 2: Spectral-Spatial Interpolation
        print('   Computing Spectral-Spatial interpolation...')
        ss_pred = spectral_spatial_interpolation(
            train_coords, train_spectral, y,
            val_coords, val_spectral,
            k_spatial=30, k_spectral=100,
            spatial_weight=0.7, spectral_weight=0.3
        )
        print(f'   SS: [{ss_pred.min():.2f}, {ss_pred.max():.2f}]')

        # Method 3: IDW + Residual model
        print('   Training residual model...')
        # Get IDW predictions for training (LOO approximation)
        train_tree = cKDTree(train_coords)
        train_idw = []
        for i in range(len(train_coords)):
            dists, idxs = train_tree.query(train_coords[i], k=21)
            # Skip self
            dists = dists[1:]
            idxs = idxs[1:]
            dists = np.maximum(dists, 1e-10)
            w = 1.0 / (dists ** 2)
            w = w / w.sum()
            train_idw.append(np.sum(w * y[idxs]))
        train_idw = np.array(train_idw)

        residual_pred = train_residual_model(
            train_coords, train_spectral, y, train_idw,
            val_coords, val_spectral, idw_pred
        )
        adjusted_pred = idw_pred + 0.3 * residual_pred  # Light adjustment
        print(f'   Adjusted: [{adjusted_pred.min():.2f}, {adjusted_pred.max():.2f}]')

        # Blend all methods
        # Emphasis on IDW since it works well for this problem
        final_pred = 0.5 * idw_pred + 0.3 * ss_pred + 0.2 * adjusted_pred
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

    submission.to_csv('submission_v11.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE! Submission: submission_v11.csv')
    print('=' * 70)
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, median={submission[col].median():.2f}')


if __name__ == '__main__':
    main()
