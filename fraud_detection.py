"""
Fraud Detection Model for Financial Transactions
Dataset: Credit Card Fraud Detection

This script builds and evaluates ML models for detecting fraudulent transactions.
Includes preprocessing, hyperparameter tuning, and comprehensive evaluation.
"""

import os
import pandas as pd
import numpy as np

# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import clone

# imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint
from joblib import parallel_backend

# xgboost
import xgboost as xgb

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.dpi"] = 140


# Setup paths
DATA_PATH = r"C:\Users\Dell\Desktop\Leonard & Priya Group WOrk\creditcard.csv"
FIG_DIR = "figures"
MODEL_DIR = "models"
CACHE_DIR = "skcache"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

print("Columns:", list(df.columns))
print("\nDataset info:")
print(df.info())
print("\nMissing values check:")
print(df.isnull().sum())


# EDA: Visualizations before modeling
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

target_col = "Class"
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# Class distribution
print("\nClass distribution:")
print(y.value_counts())
print("\nClass proportions:")
print(y.value_counts(normalize=True))

# Plot 1: Class imbalance visualization
plt.figure(figsize=(8, 5))
y.value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Class Distribution (0=Legitimate, 1=Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=220, bbox_inches="tight")
plt.show()

# Plot 2: Transaction amount distribution by class
if 'Amount' in df.columns:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, label='Legitimate', color='green')
    plt.hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, label='Fraud', color='red')
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.title("Amount Distribution by Class")
    plt.legend()
    plt.xlim([0, df['Amount'].quantile(0.99)])  
    # remove extreme outliers for clarity
    
    plt.subplot(1, 2, 2)
    plt.boxplot([df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']], 
                labels=['Legitimate', 'Fraud'])
    plt.ylabel("Transaction Amount")
    plt.title("Amount Comparison (Boxplot)")
    plt.ylim([0, df['Amount'].quantile(0.99)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "amount_analysis.png"), dpi=220, bbox_inches="tight")
    plt.show()

# Plot 3: Time distribution (if Time column exists)
if 'Time' in df.columns:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df[df['Class']==0]['Time'], bins=50, alpha=0.7, label='Legitimate', color='green')
    plt.hist(df[df['Class']==1]['Time'], bins=50, alpha=0.7, label='Fraud', color='red')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.title("Transaction Time Distribution by Class")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    df.groupby('Class')['Time'].plot(kind='kde', legend=True)
    plt.xlabel("Time (seconds)")
    plt.title("Time Density Plot")
    plt.legend(['Legitimate', 'Fraud'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "time_analysis.png"), dpi=220, bbox_inches="tight")
    plt.show()

# Plot 4: Correlation heatmap (sample features to avoid overcrowding)
# Only show correlations with target and among first 10 features
print("\nGenerating correlation heatmap...")
corr_features = list(X.columns[:10]) + [target_col]
corr_df = df[corr_features]

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df.corr(), annot=False, cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Correlation'})
plt.title("Feature Correlation Heatmap (Sample)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=220, bbox_inches="tight")
plt.show()

# Summary statistics
print("\nSummary statistics for legitimate transactions:")
print(df[df['Class']==0].describe())
print("\nSummary statistics for fraudulent transactions:")
print(df[df['Class']==1].describe())


# Prepare features
# Reduce memory usage
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        X[c] = X[c].astype("float32")

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"\nFeature breakdown: {len(num_cols)} numeric, {len(cat_cols)} categorical")


# Train/test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain set:", X_train.shape, " Test set:", X_test.shape)


# Preprocessing pipeline
numeric_transformer = StandardScaler()
preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, list(X_train.columns))],
    remainder="drop"
)


# Helper functions
def print_full_report(y_true, y_pred, y_proba=None):
    """Print evaluation metrics including confusion matrix and AUC scores."""
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            print(f"ROC AUC: {auc:.4f}")
            print(f"PR AUC:  {ap:.4f}")
        except Exception as e:
            print("AUC calculation skipped:", e)

def tune_threshold(y_true, y_prob):
    """Find optimal classification threshold by maximizing F1 score."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    ix = np.nanargmax(f1)
    best_thr = thr[ix] if ix < len(thr) else 0.5
    return best_thr, float(prec[ix]), float(rec[ix])


# Create tuning subset (faster hyperparameter search)
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, train_size=0.30, stratify=y_train, random_state=42
)

# Model 1: Logistic Regression with SMOTE
print("\n" + "="*70)
print("TRAINING MODEL 1: LOGISTIC REGRESSION")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_pipe = ImbPipeline(
    steps=[
        ("pre", preprocess),
        ("smote", SMOTE(random_state=42)),  
        # Handle imbalance with synthetic samples
        ("clf", LogisticRegression(max_iter=300, solver="lbfgs"))
    ],
    memory=CACHE_DIR
)

log_grid = {"clf__C": [0.1, 1.0, 10.0]}  
# Regularization parameter

log_search = GridSearchCV(
    estimator=log_pipe,
    param_grid=log_grid,
    scoring="average_precision",  
    # PR AUC is better for imbalanced data
    n_jobs=1,
    cv=cv,
    refit=True,
    verbose=2
)

print("Running grid search on tuning subset...")
with parallel_backend("threading"):
    log_search.fit(X_tune, y_tune)
best_log = log_search.best_estimator_
print("Best params:", log_search.best_params_)

print("Refitting on full training data...")
best_log.fit(X_train, y_train)


# Model 2: Random Forest with class balancing
print("\n" + "="*70)
print("TRAINING MODEL 2: RANDOM FOREST")
print("="*70)

cv_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rf_pipe = ImbPipeline(
    steps=[
        ("pre", preprocess),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",  
            # Handle imbalance via weighting
            bootstrap=True,
            max_samples=0.4  
            # Subsample for diversity and speed
        ))
    ],
    memory=CACHE_DIR
)

rf_dist = {
    "clf__n_estimators": randint(120, 221),
    "clf__max_depth": [8, 12, 16],
    "clf__min_samples_split": randint(2, 21),
    "clf__max_features": ["sqrt"]
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipe,
    param_distributions=rf_dist,
    n_iter=10,
    scoring="average_precision",
    n_jobs=1,
    cv=cv_fast,
    refit=True,
    random_state=42,
    verbose=2
)

print("Running randomized search on tuning subset...")
with parallel_backend("threading"):
    rf_search.fit(X_tune, y_tune)
best_rf = rf_search.best_estimator_
print("Best params:", rf_search.best_params_)

print("Refitting on full training data...")
best_rf.fit(X_train, y_train)


# Model 3: XGBoost with early stopping
print("\n" + "="*70)
print("TRAINING MODEL 3: XGBOOST")
print("="*70)

# Calculate weight for positive class
pos = int(y_train.sum())
neg = int(y_train.shape[0] - pos)
spw = float(neg) / float(pos) if pos > 0 else 1.0
print(f"Positive class weight: {spw:.2f}")

# Split for early stopping validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

# Preprocess for XGBoost
pre_xgb = clone(preprocess)
pre_xgb.fit(X_tr)
X_tr_s = pre_xgb.transform(X_tr)
X_val_s = pre_xgb.transform(X_val)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_tr_s, label=y_tr.values if hasattr(y_tr, "values") else y_tr)
dval   = xgb.DMatrix(X_val_s, label=y_val.values if hasattr(y_val, "values") else y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "eta": 0.08,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": spw,
    "seed": 42
}

print("Training with early stopping...")
booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=False
)

print(f"Best iteration: {booster.best_iteration}")

# Wrapper to make XGBoost compatible with sklearn interface
class XGBBoosterWrapper:
    def __init__(self, booster, preprocessor, threshold=0.5):
        self.booster = booster
        self.pre = preprocessor
        self.threshold = threshold

    def predict_proba(self, X):
        Xs = self.pre.transform(X)
        d = xgb.DMatrix(Xs)
        p = self.booster.predict(d)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

best_xgb = XGBBoosterWrapper(booster, pre_xgb)


# Feature importance for Random Forest
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

try:
    # Get feature importances from Random Forest
    rf_model = best_rf.named_steps['clf']
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=220, bbox_inches="tight")
    plt.show()
    
    print("\nTop 10 most important features:")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


# Evaluate all models
print("\n" + "="*70)
print("MODEL EVALUATION ON TEST SET")
print("="*70)

models = {
    "LogisticRegression": best_log,
    "RandomForest": best_rf,
    "XGBoost": best_xgb
}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)
    
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    print_full_report(y_test, y_pred, y_proba)

    # Confusion Matrix
    try:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"cm_{name}.png"), dpi=220, bbox_inches="tight")
        plt.show()
    except Exception:
        pass

    # ROC and PR curves
    if y_proba is not None:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(FIG_DIR, f"roc_{name}.png"), dpi=220, bbox_inches="tight")
        plt.show()

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(rec, prec, linewidth=2, label=f'{name} (AP = {average_precision_score(y_test, y_proba):.3f})')
        baseline = y_test.sum() / len(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline (AP = {baseline:.3f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(FIG_DIR, f"pr_{name}.png"), dpi=220, bbox_inches="tight")
        plt.show()

        # Find optimal threshold
        thr, p, r = tune_threshold(y_test, y_proba)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        print(f"\nOptimal threshold: {thr:.4f}")
        print(f"  Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")


# Model comparison and selection
print("\n" + "="*70)
print("FINAL MODEL COMPARISON")
print("="*70)

scores = {}
for n, m in models.items():
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(X_test)[:, 1]
        scores[n] = average_precision_score(y_test, proba)

# Display scores
print("\nPR AUC Scores (higher is better):")
for model_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name}: {score:.4f}")

# Save best model
winner = max(scores, key=scores.get)
final_estimator = models[winner]
out_path = os.path.join(MODEL_DIR, f"fraud_best_model_{winner}.joblib")
joblib.dump(final_estimator, out_path)

print(f"\nBest model: {winner}")
print(f"Saved to: {out_path}")
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"All visualizations saved to '{FIG_DIR}/' directory")
print(f"Best model saved to '{MODEL_DIR}/' directory")