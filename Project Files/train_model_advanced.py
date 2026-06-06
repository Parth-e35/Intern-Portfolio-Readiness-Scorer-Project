import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE


DATA_PATH = "data/final_portfolio_dataset.csv"
MODEL_PATH = "models/best_readiness_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURE_PATH = "models/model_features.json"
METRICS_PATH = "models/model_metrics.json"


def safe_divide(a, b):
    return np.where(b == 0, 0, a / b)


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)

    print("\nTarget Distribution:")
    print(df["readiness_category"].value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def clean_dataset(df):
    df = df.copy()

    df = df.dropna(subset=["readiness_category"])

    text_columns = [
        "candidate_name",
        "resume_file",
        "email",
        "github_link",
        "github_username",
        "linkedin_link",
        "portfolio_link",
        "github_name",
        "github_bio",
        "languages_used",
        "recommendations"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Not Available")

    return df


def preprocess_and_engineer_features(df):
    df = clean_dataset(df)

    base_numeric_columns = [
        "followers",
        "total_repositories_analyzed",
        "total_stars",
        "total_forks",
        "readme_count",
        "description_count",
        "homepage_count",
        "recent_repo_count",
        "language_count"
    ]

    for col in base_numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["readme_ratio"] = safe_divide(
        df["readme_count"],
        df["total_repositories_analyzed"]
    )

    df["description_ratio"] = safe_divide(
        df["description_count"],
        df["total_repositories_analyzed"]
    )

    df["recent_activity_ratio"] = safe_divide(
        df["recent_repo_count"],
        df["total_repositories_analyzed"]
    )

    df["avg_stars_per_repo"] = safe_divide(
        df["total_stars"],
        df["total_repositories_analyzed"]
    )

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # Final reduced feature set
    # Score-based columns are intentionally excluded to prevent leakage.
    features = [
        "total_repositories_analyzed",
        "language_count",
        "readme_count",
        "description_count",
        "homepage_count",
        "recent_repo_count",
        "total_stars",
        "total_forks",
        "followers",
        "readme_ratio",
        "description_ratio",
        "recent_activity_ratio",
        "avg_stars_per_repo"
    ]

    X = df[features]
    y = df["readiness_category"]

    return X, y, features


def train_multiple_models(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nEncoded Classes:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(index, "=", class_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\nBefore SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"{label_encoder.inverse_transform([u])[0]}: {c}")

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"{label_encoder.inverse_transform([u])[0]}: {c}")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=42
            ))
        ])
    }

    results = []
    best_model = None
    best_model_name = ""
    best_f1 = 0

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        model.fit(X_train_resampled, y_train_resampled)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        cv_scores = cross_val_score(
            model,
            X,
            y_encoded,
            cv=5,
            scoring="f1_weighted"
        )

        result = {
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4)
        }

        results.append(result)

        print(result)

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

    return best_model, best_model_name, label_encoder, results


def save_artifacts(best_model, best_model_name, label_encoder, features, results):
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    with open(FEATURE_PATH, "w") as f:
        json.dump(features, f, indent=4)

    metrics = {
        "best_model": best_model_name,
        "features_used": features,
        "results": results
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nBest Model:", best_model_name)
    print("Features Used:", features)
    print("Saved model:", MODEL_PATH)
    print("Saved label encoder:", ENCODER_PATH)
    print("Saved features:", FEATURE_PATH)
    print("Saved metrics:", METRICS_PATH)


def main():
    df = load_dataset()
    X, y, features = preprocess_and_engineer_features(df)
    best_model, best_model_name, label_encoder, results = train_multiple_models(X, y)
    save_artifacts(best_model, best_model_name, label_encoder, features, results)


if __name__ == "__main__":
    main()