import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def prepare_ml_dataset(master_df):
    """
    Convert master table into ML-ready dataset.
    """

    df = master_df.copy()

    # Only keep rows where drug exists or not
    df["has_drug"] = df["drug_name"].notna().astype(int)

    # Feature engineering
    features = (
        df.groupby(["orpha_code", "drug_name"])
        .agg(
            gene_support=("gene_symbol", "nunique")
        )
        .reset_index()
    )

    # Label: 1 if real drug association exists
    features["label"] = features["drug_name"].notna().astype(int)

    # Remove rows with no drug_name (NaN groups)
    features = features.dropna(subset=["drug_name"])

    return features


def train_random_forest(features_df):
    """
    Train Random Forest model.
    """

    X = features_df[["gene_support"]]
    y = features_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== RANDOM FOREST RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model
