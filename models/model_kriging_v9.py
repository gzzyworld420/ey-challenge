#!/usr/bin/env python3
"""
EY AI & Data Challenge 2026 - Water Quality Prediction v9.0
Kriging-based spatial interpolation + Simple ML.

Key insights from previous attempts:
- Best score: 0.274 with simple LightGBM
- Complex models overfit (CV ~0.88, real ~0.27)
- Coordinates are ESSENTIAL (without them: 0.067)
- Zero geographic overlap between train/validation

New strategy:
1. Kriging-style spatial interpolation
2. Very simple features (avoid overfitting)
3. Strong regularization
4. Focus on generalization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']


def variogram_model(h, nugget, sill, range_param):
    """Spherical variogram model."""
    h = np.asarray(h)
    gamma = np.zeros_like(h, dtype=float)
    mask = h <= range_param
    gamma[mask] = nugget + sill * (1.5 * h[mask] / range_param - 0.5 * (h[mask] / range_param)**3)
    gamma[~mask] = nugget + sill
    return gamma


def fit_variogram(train_coords, train_values, n_bins=15):
    """Fit a variogram to the data."""
    # Compute pairwise distances and squared differences
    dists = cdist(train_coords, train_coords)
    diffs = (train_values[:, None] - train_values[None, :]) ** 2

    # Get upper triangle (unique pairs)
    triu_idx = np.triu_indices(len(train_coords), k=1)
    flat_dists = dists[triu_idx]
    flat_diffs = diffs[triu_idx] / 2  # semivariance

    # Bin by distance
    max_dist = np.percentile(flat_dists, 50)  # Use 50th percentile
    bins = np.linspace(0, max_dist, n_bins + 1)

    bin_centers = []
    bin_variances = []
    for i in range(len(bins) - 1):
        mask = (flat_dists >= bins[i]) & (flat_dists < bins[i + 1])
        if mask.sum() > 10:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_variances.append(np.mean(flat_diffs[mask]))

    bin_centers = np.array(bin_centers)
    bin_variances = np.array(bin_variances)

    if len(bin_centers) < 3:
        # Not enough data, use defaults
        return 0.1, 1.0, 5.0

    # Fit variogram parameters
    def objective(params):
        nugget, sill, range_p = params
        if nugget < 0 or sill < 0 or range_p <= 0:
            return 1e10
        pred = variogram_model(bin_centers, nugget, sill, range_p)
        return np.sum((pred - bin_variances) ** 2)

    result = minimize(
        objective,
        x0=[0.1, np.var(train_values), np.max(bin_centers) / 2],
        method='Nelder-Mead'
    )

    return result.x


def simple_kriging(train_coords, train_values, pred_coords, nugget, sill, range_param):
    """Simple kriging interpolation."""
    n_train = len(train_coords)
    n_pred = len(pred_coords)

    # Covariance matrix for training points
    train_dists = cdist(train_coords, train_coords)
    C_train = sill - variogram_model(train_dists, nugget, sill, range_param)
    np.fill_diagonal(C_train, sill + nugget * 0.01)  # Add small diagonal for stability

    # Covariance between prediction and training points
    pred_train_dists = cdist(pred_coords, train_coords)
    C_pred = sill - variogram_model(pred_train_dists, nugget, sill, range_param)

    # Solve kriging system
    try:
        weights = np.linalg.solve(C_train, C_pred.T).T
    except:
        # Fallback to regularized solution
        C_train_reg = C_train + np.eye(n_train) * 0.01 * np.trace(C_train) / n_train
        weights = np.linalg.solve(C_train_reg, C_pred.T).T

    # Normalize weights (simple kriging assumes mean is known)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Predictions
    mean_value = np.mean(train_values)
    predictions = mean_value + weights @ (train_values - mean_value)

    return predictions


def ordinary_kriging(train_coords, train_values, pred_coords, nugget, sill, range_param):
    """Ordinary kriging - weights sum to 1."""
    n_train = len(train_coords)

    # Build kriging matrix
    train_dists = cdist(train_coords, train_coords)
    gamma_train = variogram_model(train_dists, nugget, sill, range_param)

    # Add Lagrange multiplier row/column
    K = np.zeros((n_train + 1, n_train + 1))
    K[:n_train, :n_train] = gamma_train
    K[n_train, :n_train] = 1
    K[:n_train, n_train] = 1

    # Predictions
    predictions = []
    for pred_coord in pred_coords:
        pred_dists = cdist([pred_coord], train_coords)[0]
        gamma_pred = variogram_model(pred_dists, nugget, sill, range_param)

        # RHS
        k = np.zeros(n_train + 1)
        k[:n_train] = gamma_pred
        k[n_train] = 1

        try:
            weights = np.linalg.solve(K, k)[:n_train]
        except:
            # Fallback to IDW
            weights = 1 / (pred_dists ** 2 + 0.01)
            weights = weights / weights.sum()

        pred = np.dot(weights, train_values)
        predictions.append(pred)

    return np.array(predictions)


def create_simple_features(wq_df, ls_df, tc_df):
    """Create minimal, robust features."""
    df = pd.DataFrame()

    # === COORDINATES (essential!) ===
    df['Latitude'] = wq_df['Latitude'].values
    df['Longitude'] = wq_df['Longitude'].values

    # === TEMPORAL (cyclic only) ===
    dates = pd.to_datetime(wq_df['Sample Date'], format='%d-%m-%Y')
    df['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)

    # === KEY SPECTRAL FEATURES ONLY ===
    df['NDMI'] = ls_df['NDMI'].values
    df['MNDWI'] = ls_df['MNDWI'].values

    nir = ls_df['nir'].values.astype(float)
    green = ls_df['green'].values.astype(float)
    swir16 = ls_df['swir16'].values.astype(float)

    # Only most important ratios
    eps = 1e-10
    df['NDWI'] = (green - nir) / (green + nir + eps)
    df['turbidity'] = nir / (green + eps)
    df['swir_green'] = swir16 / (green + eps)

    # === CLIMATE ===
    df['pet'] = tc_df['pet'].values

    return df


def train_simple_lgbm(X_train, y_train, X_val, seed=42):
    """Train very simple LightGBM with strong regularization."""
    if not HAS_LIGHTGBM:
        # Fallback to XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.6, colsample_bytree=0.6, min_child_weight=20,
            reg_alpha=1.0, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train)
        return model.predict(X_val)

    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_val)


def gaussian_process_predict(train_coords, train_values, pred_coords):
    """Use Gaussian Process regression (another form of kriging)."""
    # Normalize coordinates
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_coords)
    pred_scaled = scaler.transform(pred_coords)

    # Define kernel
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)

    # Subsample if too large
    if len(train_coords) > 1000:
        idx = np.random.choice(len(train_coords), 1000, replace=False)
        train_scaled = train_scaled[idx]
        train_values_sub = train_values[idx]
    else:
        train_values_sub = train_values

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True, random_state=42)

    try:
        gp.fit(train_scaled, train_values_sub)
        predictions = gp.predict(pred_scaled)
    except:
        # Fallback to simple IDW
        from scipy.spatial import cKDTree
        tree = cKDTree(train_coords)
        dists, idxs = tree.query(pred_coords, k=10)
        dists = np.maximum(dists, 1e-10)
        weights = 1 / dists ** 2
        weights = weights / weights.sum(axis=1, keepdims=True)
        predictions = np.sum(weights * train_values[idxs], axis=1)

    return predictions


def main():
    print('=' * 70)
    print('EY Water Quality Prediction v9.0 - Kriging + Simple ML')
    print('=' * 70)

    # Load data
    print('\n1. Loading data...')
    wq = pd.read_csv('water_quality_training_dataset.csv')
    ls_train = pd.read_csv('landsat_features_training.csv')
    tc_train = pd.read_csv('terraclimate_features_training.csv')
    ls_val = pd.read_csv('landsat_features_validation.csv')
    tc_val = pd.read_csv('terraclimate_features_validation.csv')
    sub_template = pd.read_csv('submission_template.csv')

    print(f'   Training samples: {len(wq)}')
    print(f'   Validation samples: {len(sub_template)}')

    # Coordinates
    train_coords = wq[['Latitude', 'Longitude']].values
    val_coords = sub_template[['Latitude', 'Longitude']].values

    # Create simple features
    print('\n2. Creating minimal features...')
    X_train_raw = create_simple_features(wq, ls_train, tc_train)
    X_val_raw = create_simple_features(sub_template, ls_val, tc_val)

    # Fill NaN with median
    X_train = X_train_raw.fillna(X_train_raw.median())
    X_val = X_val_raw.fillna(X_train_raw.median())

    print(f'   Features: {X_train.shape[1]}')

    # Scale (but keep original coordinates for kriging)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

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
        print(f'\n3. Processing {target}...')
        cfg = target_configs[target]
        y_raw = wq[target].values
        y_train = cfg['transform'](y_raw)

        # Method 1: Kriging
        print('   Fitting variogram...')
        try:
            nugget, sill, range_p = fit_variogram(train_coords, y_raw)
            print(f'   Variogram: nugget={nugget:.3f}, sill={sill:.3f}, range={range_p:.3f}')

            print('   Running ordinary kriging...')
            kriging_pred = ordinary_kriging(train_coords, y_raw, val_coords, nugget, sill, range_p)
        except Exception as e:
            print(f'   Kriging failed: {e}, using IDW fallback')
            from scipy.spatial import cKDTree
            tree = cKDTree(train_coords)
            dists, idxs = tree.query(val_coords, k=15)
            dists = np.maximum(dists, 1e-10)
            weights = 1 / dists ** 2
            weights = weights / weights.sum(axis=1, keepdims=True)
            kriging_pred = np.sum(weights * y_raw[idxs], axis=1)

        print(f'   Kriging range: [{kriging_pred.min():.2f}, {kriging_pred.max():.2f}]')

        # Method 2: Simple LightGBM
        print('   Training simple LightGBM...')
        lgbm_preds = []
        for seed in [42, 123, 456]:
            lgbm_pred = train_simple_lgbm(X_train_scaled, y_train, X_val_scaled, seed)
            lgbm_preds.append(cfg['inverse'](lgbm_pred))
        ml_pred = np.mean(lgbm_preds, axis=0)
        print(f'   ML range: [{ml_pred.min():.2f}, {ml_pred.max():.2f}]')

        # Method 3: Gaussian Process
        print('   Running Gaussian Process...')
        try:
            gp_pred = gaussian_process_predict(train_coords, y_raw, val_coords)
            print(f'   GP range: [{gp_pred.min():.2f}, {gp_pred.max():.2f}]')
        except Exception as e:
            print(f'   GP failed: {e}, using kriging')
            gp_pred = kriging_pred

        # Blend predictions
        # Higher weight for kriging since spatial correlation is high
        final_pred = 0.4 * kriging_pred + 0.3 * ml_pred + 0.3 * gp_pred
        final_pred = np.clip(final_pred, cfg['clip'][0], cfg['clip'][1])

        predictions[target] = final_pred
        print(f'   Final: [{final_pred.min():.2f}, {final_pred.max():.2f}], mean={final_pred.mean():.2f}')

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

    submission.to_csv('submission_v9.csv', index=False)

    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)
    print('\nSubmission: submission_v9.csv')
    for col in TARGET_COLS:
        print(f'  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}')


if __name__ == '__main__':
    main()
