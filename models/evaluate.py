"""
Evaluate trained model and generate SHAP explanations.
Includes accuracy, confusion matrix, and SHAP plots.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = 'models/saved_model.pkl'


def evaluate_model():
    """
    Load saved model and report cross-validation performance with SHAP explanations.
    
    Evaluates:
    - Cross-validation accuracy (mean ± std) from training
    - SHAP plots for model interpretability
    
    Why cross-validation metrics:
    - Final model is trained on FULL dataset, so there's no separate test set
    - Cross-validation provides reliable performance estimates without data leakage
    - TimeSeriesSplit preserves chronological order (train on past, validate on future)
    
    Why SHAP matters:
    - Makes complex models interpretable by showing feature contributions
    - Helps build trust by explaining "why" the model makes predictions
    """
    logger.info("Loading saved model...")
    
    # Load model data
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    model_name = model_data['model_name']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Cross-validation metrics from training
    lr_cv_mean = model_data.get('lr_cv_mean')
    lr_cv_std = model_data.get('lr_cv_std')
    xgb_cv_mean = model_data.get('xgb_cv_mean')
    xgb_cv_std = model_data.get('xgb_cv_std')
    
    logger.info(f"Loaded model: {model_name}")
    
    # Report cross-validation performance
    logger.info(f"\n=== Cross-Validation Performance ===")
    if lr_cv_mean is not None:
        logger.info(f"Logistic Regression: {lr_cv_mean:.2f} ± {lr_cv_std:.2f}")
    if xgb_cv_mean is not None:
        logger.info(f"XGBoost: {xgb_cv_mean:.2f} ± {xgb_cv_std:.2f}")
    logger.info(f"\nSelected model for production: {model_name}")
    logger.info(f"Note: Final model trained on full dataset (no separate test set)")
    
    # Load full dataset for SHAP analysis
    from features import get_matches_with_features, prepare_feature_matrix
    
    df = get_matches_with_features()
    X, y, _ = prepare_feature_matrix(df)
    
    # Apply scaling if using Logistic Regression
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Generate SHAP plots for interpretability
    if model_name == "XGBoost":
        # SHAP works best with tree-based models
        generate_shap_plots(model, X_scaled, feature_names)
    else:
        logger.info("Generating feature importance for Logistic Regression...")
        # For Logistic Regression, show feature importance via coefficients
        plot_logistic_coefficients(model, feature_names)
    
    return model, X_scaled, y, feature_names


def plot_confusion_matrix(cm, model_name):
    """
    Plot confusion matrix as a heatmap.
    
    Why visualize: Easier to understand than raw numbers.
    Shows at a glance where the model is making mistakes.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Team 1 Lost', 'Team 1 Won'],
                yticklabels=['Team 1 Lost', 'Team 1 Won'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    logger.info("Confusion matrix saved to models/confusion_matrix.png")
    plt.close()


def plot_logistic_coefficients(model, feature_names):
    """
    Plot Logistic Regression coefficients as feature importance.
    
    Why: For linear models, coefficients show how each feature affects prediction.
    Positive coefficient = increases probability of Team 1 winning.
    Negative coefficient = decreases probability of Team 1 winning.
    """
    coefficients = model.coef_[0]
    
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    importance_df = importance_df.sort_values('Coefficient', ascending=False)
    
    colors = ['green' if x > 0 else 'red' for x in importance_df['Coefficient']]
    plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Logistic Regression Feature Importance')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150)
    logger.info("Feature importance plot saved to models/feature_importance.png")
    plt.close()


def generate_shap_plots(model, X_test, feature_names):
    """
    Generate SHAP summary plot and force plot for model explanation.
    
    What is SHAP?
    SHAP (SHapley Additive exPlanations) is a method to explain machine learning predictions.
    It's based on game theory and assigns each feature an importance value for a particular prediction.
    
    What the SHAP summary plot shows:
    - Each dot is a single prediction (one match)
    - X-axis: SHAP value - how much each feature pushed the prediction higher or lower
      - Positive SHAP = feature increased probability of Team 1 winning
      - Negative SHAP = feature decreased probability of Team 1 winning
    - Y-axis: Features ordered by importance
    - Color: Feature value (red = high, blue = low)
    
    What the SHAP force plot shows:
    - For a single prediction, it shows how each feature contributed to the final prediction
    - Base value: The average prediction across all matches
    - Features pushing the prediction higher (red, to the right)
    - Features pushing the prediction lower (blue, to the left)
    - The final prediction is the sum of base value + all SHAP values
    
    Why SHAP matters:
    - Makes complex models like XGBoost interpretable
    - Helps build trust by showing "why" the model made a prediction
    - Can reveal if model is using sensible features or learning spurious correlations
    """
    logger.info("Generating SHAP plots...")
    
    # Create SHAP explainer
    # Why TreeExplainer: Optimized for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)
    
    # Plot 1: SHAP summary plot
    # Shows overall feature importance and how features affect predictions
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Win Probability')
    plt.tight_layout()
    plt.savefig('models/shap_summary_plot.png', dpi=150, bbox_inches='tight')
    logger.info("SHAP summary plot saved to models/shap_summary_plot.png")
    plt.close()
    
    # Plot 2: SHAP force plot for a single prediction
    # Shows how each feature contributed to one specific match prediction
    # We'll use the first test sample as an example
    sample_idx = 0
    
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[sample_idx], 
        X_test[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - Prediction for Match #{sample_idx}')
    plt.tight_layout()
    plt.savefig('models/shap_force_plot.png', dpi=150, bbox_inches='tight')
    logger.info("SHAP force plot saved to models/shap_force_plot.png")
    plt.close()
    
    logger.info("SHAP plots generated successfully")


if __name__ == '__main__':
    evaluate_model()
