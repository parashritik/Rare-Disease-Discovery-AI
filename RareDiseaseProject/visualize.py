import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from utils import MOD_DIR, OUT_DIR

RESULT_FILE = OUT_DIR / "optimized_capstone_results.csv"

def generate_plots():
    if not RESULT_FILE.exists():
        print("❌ Error: Result file not found. Run main.py first!")
        return

    print("📈 Generating Capstone Visualizations...")
    df = pd.read_csv(RESULT_FILE)
    y_true = df['is_target']
    y_scores = df['discovery_score']
    opt_threshold = joblib.load(MOD_DIR / "hyper_threshold.pkl")
    y_pred = (y_scores >= opt_threshold).astype(int)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title(f"Confusion Matrix (Threshold: {opt_threshold:.3f})")

    # P-R Curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    ax[1].plot(recalls, precisions, color='darkorange', lw=2)
    ax[1].axhline(y=0.90, color='red', linestyle='--', label='90% Precision')
    ax[1].set_title("Precision-Recall Curve")
    ax[1].legend()

    plt.savefig(OUT_DIR / "model_performance_summary.png")
    plt.show()

if __name__ == "__main__":
    generate_plots()