"""
Train win probability prediction models.
Compares Logistic Regression (baseline) with XGBoost (more complex).
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import logging

from features import get_matches_with_features, prepare_feature_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = 'models/saved_model.pkl'


def train_models():
    """
    Train and compare Logistic Regression and XGBoost models using time-series cross-validation.
    
    Why TimeSeriesSplit over single train/test split:
    - Small dataset (45 matches) means a single 80/20 split gives only 9 test matches
    - Single flipped prediction changes accuracy by 11%, making the metric too noisy
    - TimeSeriesSplit with 5 folds provides more reliable performance estimates
    - Preserves chronological order (train on past, validate on future) to prevent data leakage
    
    Why compare these two models:
    - Logistic Regression: Simple, interpretable baseline. Shows what we can achieve
      with a linear model. Good for understanding feature importance through coefficients.
    - XGBoost: More powerful tree-based model that can capture non-linear relationships
      and feature interactions. Often performs better on complex real-world data.
    
    Comparing both helps us understand if the extra complexity of XGBoost is justified
    by improved performance. If Logistic Regression performs similarly, we might prefer
    it for simplicity and interpretability.
    
    Manual hyperparameters (no tuning framework):
    - Logistic Regression: C=1.0 (regularization strength), max_iter=1000
    - XGBoost: n_estimators=100, max_depth=3, learning_rate=0.1
      These are reasonable defaults that work well for many problems.
    
    Process:
    1. Cross-validate both models with TimeSeriesSplit (5 folds)
    2. Report mean ± std accuracy for each model
    3. Fit final chosen model on FULL dataset
    4. Save final model for production use
    """
    logger.info("Starting model training...")
    
    # Load and prepare data
    df = get_matches_with_features()
    X, y, feature_names = prepare_feature_matrix(df)
    
    logger.info(f"Total dataset size: {len(df)} matches")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Use TimeSeriesSplit for cross-validation
    # Why: Preserves chronological order, prevents data leakage from future matches
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Cross-validate Logistic Regression
    logger.info("\n=== Cross-Validating Logistic Regression ===")
    lr_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Scale features per-fold to avoid leakage
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_val_scaled = scaler_fold.transform(X_val_fold)
        
        lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train_fold)
        val_acc = accuracy_score(y_val_fold, lr_model.predict(X_val_scaled))
        lr_scores.append(val_acc)
        logger.info(f"Fold {fold + 1}: {len(train_idx)} train, {len(val_idx)} val - Accuracy: {val_acc:.4f}")
    
    lr_mean = np.mean(lr_scores)
    lr_std = np.std(lr_scores)
    logger.info(f"Logistic Regression: {lr_mean:.2f} ± {lr_std:.2f}")
    
    # Cross-validate XGBoost
    logger.info("\n=== Cross-Validating XGBoost ===")
    xgb_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, base_score=0.5, use_label_encoder=False, eval_metric='logloss'
        )
        xgb_model.fit(X_train_fold, y_train_fold)
        val_acc = accuracy_score(y_val_fold, xgb_model.predict(X_val_fold))
        xgb_scores.append(val_acc)
        logger.info(f"Fold {fold + 1}: {len(train_idx)} train, {len(val_idx)} val - Accuracy: {val_acc:.4f}")
    
    xgb_mean = np.mean(xgb_scores)
    xgb_std = np.std(xgb_scores)
    logger.info(f"XGBoost: {xgb_mean:.2f} ± {xgb_std:.2f}")
    
    # Compare models
    logger.info("\n=== Model Comparison ===")
    logger.info(f"Logistic Regression: {lr_mean:.2f} ± {lr_std:.2f}")
    logger.info(f"XGBoost: {xgb_mean:.2f} ± {xgb_std:.2f}")
    
    # Select the better model based on mean cross-validation accuracy
    if xgb_mean > lr_mean:
        best_model_name = "XGBoost"
        logger.info(f"Selected model: {best_model_name}")
    else:
        best_model_name = "Logistic Regression"
        logger.info(f"Selected model: {best_model_name}")
    
    # Fit final model on FULL dataset for production use
    logger.info(f"\n=== Fitting final {best_model_name} model on full dataset ===")
    if best_model_name == "Logistic Regression":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        best_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        best_model.fit(X_scaled, y)
    else:
        scaler = None
        best_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, base_score=0.5, use_label_encoder=False, eval_metric='logloss'
        )
        best_model.fit(X, y)
    
    logger.info(f"Final model trained on all {len(df)} matches")
    
    # Save the best model along with necessary preprocessing
    # Why: Need to save scaler if using Logistic Regression, and feature names
    model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'feature_names': feature_names,
        'lr_cv_mean': lr_mean,
        'lr_cv_std': lr_std,
        'xgb_cv_mean': xgb_mean,
        'xgb_cv_std': xgb_std
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {MODEL_PATH}")
    
    # Return full dataset for evaluation
    return X, y, best_model, best_model_name, scaler


if __name__ == '__main__':
    train_models()
