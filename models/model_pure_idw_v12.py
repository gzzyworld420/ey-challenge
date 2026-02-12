#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v12.0
Pure IDW Interpolation - Back to Basics.

Key insight:
- LOO-IDW within training gives R² ~0.76 for TA and EC
- This is similar to the leader's score!
- Maybe the answer is pure spatial interpolation without ML

Strategy:
- Pure IDW with optimized parameters
- No ML at all - let the spatial structure do the work
- Try different k values and power parameters
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']


def idw_interpolation(train_coords, train_values, pred_coords, k=20, power=2):
    """Simple IDW interpolation."""
    tree = cKDTree(train_coords)
    dists, idxs = tree.query(pred_coords, k=min(k, len(train_coords)))

    if len(dists.shape) == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)

    # IDW weights
    weights = 1.0 / (dists ** power)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Interpolate
    predictions = np.sum(weights * train_values[idxs], axis=1)

    return predictions


def loo_cv_idw(train_coords, train_values, k=20, power=2):
    """Leave-one-out cross-validation for IDW."""
    tree = cKDTree(train_coords)
    n = len(train_coords)

    predictions = np.zeros(n)

    for i in range(n):
        # Get k+1 neighbors (including self) then exclude self
        dists, idxs = tree.query(train_coords[i], k=min(k+1, n))

        # Remove self (first element)
        mask = idxs != i
        dists = dists[mask][:k]
        idxs = idxs[mask][:k]

        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / (dists ** power)
        weights = weights / weights.sum()

        predictions[i] = np.sum(weights * train_values[idxs])

    return predictions


def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def optimize_idw_params(train_coords, train_values, k_range=[5, 10, 15, 20, 30, 50],
                        power_range=[1, 1.5, 2, 2.5, 3]):
    """Find optimal k and power for IDW."""
    best_r2 = -np.inf
    best_k = 20
    best_power = 2

    for k in k_range:
        for power in power_range:
            preds = loo_cv_idw(train_coords, train_values, k=k, power=power)
            r2 = compute_r2(train_values, preds)
            if r2 > best_r2:
                best_r2 = r2
                best_k = k
                best_power = power

    return best_k, best_power, best_r2


def main():
    print('=' * 70)
    print('EY Water Quality v12.0 - Pure IDW Interpolation')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Training: {len(wq)}, Validation: {len(sub_template)}')

    # Coordinates
    train_coords = wq[['Latitude', 'Longitude']].values
    val_coords = sub_template[['Latitude', 'Longitude']].values

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

        # Optimize IDW parameters
        print('   Optimizing IDW parameters...')
        best_k, best_power, best_r2 = optimize_idw_params(train_coords, y)
        print(f'   Best params: k={best_k}, power={best_power}, LOO-R²={best_r2:.4f}')

        # Predict with optimized parameters
        pred = idw_interpolation(train_coords, y, val_coords, k=best_k, power=best_power)
        pred = np.clip(pred, cfg['clip'][0], cfg['clip'][1])

        # Also try some variations and blend
        pred_k10 = idw_interpolation(train_coords, y, val_coords, k=10, power=2)
        pred_k30 = idw_interpolation(train_coords, y, val_coords, k=30, power=2)
        pred_k50 = idw_interpolation(train_coords, y, val_coords, k=50, power=2)
        pred_p3 = idw_interpolation(train_coords, y, val_coords, k=20, power=3)

        # Blend predictions
        final_pred = 0.4 * pred + 0.15 * pred_k10 + 0.15 * pred_k30 + 0.15 * pred_k50 + 0.15 * pred_p3
        final_pred = np.clip(final_pred, cfg['clip'][0], cfg['clip'][1])

        predictions[target] = final_pred
        print(f'   Final: mean={final_pred.mean():.2f}, std={final_pred.std():.2f}')
        print(f'   Range: [{final_pred.min():.2f}, {final_pred.max():.2f}]')

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

    submission.to_csv('submission_v12.csv', index=False)

    # Also save a version with just the best IDW
    submission_pure = submission.copy()
    submission_pure.to_csv('submission_v12_pure.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print('\nSubmissions:')
    print('  - submission_v12.csv (blended IDW)')
    print('  - submission_v12_pure.csv (same as blended)')

    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, median={submission[col].median():.2f}')


if __name__ == '__main__':
    main()
