import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, precision_recall_curve, classification_report
)

def train_hyper_optimized_model(df_agg, model_dir):
    print("🤖 Step 3: Training Hyper-Optimized AI Ensemble...")
    features = ['ppi_mean', 'ppi_max', 'ppi_std', 'ppi_skew', 'interaction_count', 'degree_max', 'degree_mean']
    X = df_agg[features].values
    y = df_agg['is_target'].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # MLP: Deep 3-layer architecture
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu', 
        solver='adam', 
        alpha=0.001, 
        learning_rate='adaptive', 
        max_iter=500, 
        random_state=42
    ).fit(X_scaled, y)
    
    mlp_probs = mlp.predict_proba(X_scaled)[:, 1].reshape(-1, 1)

    # Random Forest: Entropy-based decision making
    rf = RandomForestClassifier(
        n_estimators=500, 
        criterion='entropy', 
        max_features='log2', 
        class_weight='balanced_subsample', 
        n_jobs=-1, 
        random_state=42
    ).fit(mlp_probs, y)

    final_probs = rf.predict_proba(mlp_probs)[:, 1]

    # Threshold Tuning for Maximum Precision (Targeting 90%)
    precisions, recalls, thresholds = precision_recall_curve(y, final_probs)
    opt_threshold = thresholds[np.where(precisions >= 0.90)[0][0]] if any(precisions >= 0.90) else 0.5

    y_pred = (final_probs >= opt_threshold).astype(int)

    # --- ENHANCED METRICS REPORT ---
    print("\n" + "="*55)
    print(f"🏆 DETAILED BIOMEDICAL CLASSIFICATION REPORT")
    print("-" * 55)
    print(f"Overall Accuracy:  {accuracy_score(y, y_pred):.2%}")
    print(f"System F1-Score:   {f1_score(y, y_pred):.2%}")
    print(f"Optimal Threshold: {opt_threshold:.4f}")
    print("-" * 55)
    
    # Class-wise details (0 = Non-target, 1 = Known target)
    print("CLASS-WISE PERFORMANCE:")
    print(classification_report(y, y_pred, target_names=['Non-Target (0)', 'Known Target (1)']))
    print("="*55)

    # Save components
    joblib.dump(mlp, model_dir / "hyper_mlp.pkl")
    joblib.dump(rf, model_dir / "hyper_rf.pkl")
    joblib.dump(scaler, model_dir / "hyper_scaler.pkl")
    joblib.dump(opt_threshold, model_dir / "hyper_threshold.pkl")
    
    return opt_threshold, final_probs