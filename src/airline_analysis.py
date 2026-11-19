"""Airline recommendation and satisfaction modelling with train-only preprocessing."""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy import stats
from scipy.sparse import csr_matrix, hstack
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier

RATING_COLS = [
    "Seat Comfort", "Cabin Staff Service", "Food & Beverages",
    "Ground Service", "Inflight Entertainment", "Wifi & Connectivity",
    "Value For Money",
]
CAT_COLS = ["Type Of Traveller", "Seat Type", "Verified"]
RANDOM_STATE = 42


def download_nltk_resources():
    for resource in ("punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    df["Recommended"] = df["Recommended"].astype(str).str.lower().map({"yes": 1, "no": 0})
    df["Overall_Rating"] = pd.to_numeric(df["Overall_Rating"], errors="coerce")
    df = df.dropna(subset=["Recommended", "Overall_Rating"]).reset_index(drop=True)
    df["Full_Review"] = (
        df["Review_Title"].fillna("") + " " + df["Review"].fillna("")
    ).str.strip().str.replace(r"\s+", " ", regex=True)
    df[RATING_COLS] = df[RATING_COLS].fillna(0)
    df[CAT_COLS] = df[CAT_COLS].fillna("Unknown")
    return df


def build_text_preprocessor():
    lemmatizer = WordNetLemmatizer()
    airline_stopwords = set(stopwords.words("english")).union({
        "flight", "airline", "plane", "aircraft", "boarding", "gate",
        "airport", "check", "bag", "luggage", "seat", "would", "could",
        "should", "also", "like", "really", "get", "got", "one", "two",
    })

    def wordnet_pos(tag):
        return "a" if tag.startswith("J") else "v" if tag.startswith("V") else "r" if tag.startswith("R") else "n"

    def preprocess(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+|www\S+", "", text.lower())
        tokens = word_tokenize(re.sub(r"[^a-zA-Z\s]", "", text))
        tokens = [token for token in tokens if token not in airline_stopwords and len(token) > 2]
        return " ".join(lemmatizer.lemmatize(token, wordnet_pos(tag)) for token, tag in pos_tag(tokens))

    return preprocess


def sentiment_frame(series):
    analyzer = SentimentIntensityAnalyzer()
    scores = series.fillna("").map(analyzer.polarity_scores)
    return pd.DataFrame(scores.tolist(), index=series.index)[["compound", "pos", "neg", "neu"]]


def make_features(train, test):
    print("Preprocessing review text...")
    preprocess = build_text_preprocessor()
    train_text = train["Full_Review"].map(preprocess)
    test_text = test["Full_Review"].map(preprocess)

    encoder = OneHotEncoder(handle_unknown="ignore", drop="first")
    train_cat = encoder.fit_transform(train[CAT_COLS])
    test_cat = encoder.transform(test[CAT_COLS])
    train_structured = hstack([csr_matrix(train[RATING_COLS].values), train_cat]).tocsr()
    test_structured = hstack([csr_matrix(test[RATING_COLS].values), test_cat]).tocsr()

    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5, max_df=0.8, stop_words="english")
    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)
    train_vader = csr_matrix(sentiment_frame(train_text).values)
    test_vader = csr_matrix(sentiment_frame(test_text).values)
    train_text_features = hstack([train_tfidf, train_vader]).tocsr()
    test_text_features = hstack([test_tfidf, test_vader]).tocsr()

    return {
        "Structured-Only": (train_structured, test_structured),
        "Text-Only": (train_text_features, test_text_features),
        "Combined": (
            hstack([train_structured, train_text_features]).tocsr(),
            hstack([test_structured, test_text_features]).tocsr(),
        ),
    }


def evaluate_classification(features, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    rows, predictions = [], {}
    for config, (X_train, X_test) in features.items():
        for name, template in models.items():
            model = clone(template).fit(X_train, y_train)
            predicted = model.predict(X_test)
            probability = model.predict_proba(X_test)[:, 1]
            rows.append({
                "Configuration": config, "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision": precision_score(y_test, predicted),
                "Recall": recall_score(y_test, predicted),
                "F1": f1_score(y_test, predicted),
                "ROC-AUC": roc_auc_score(y_test, probability),
            })
            predictions[(config, name)] = predicted
    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print("\nClassification results\n", results.round(4).to_string(index=False))

    best = results.iloc[0]
    predicted = predictions[(best["Configuration"], best["Model"])]
    cm = confusion_matrix(y_test, predicted)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{best['Model']} • {best['Configuration']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    return results, predictions


def evaluate_regression(features, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=10000),
    }
    rows, predictions = [], {}
    for config, (X_train, X_test) in features.items():
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        for name, template in models.items():
            model = clone(template).fit(X_train, y_train)
            predicted = model.predict(X_test)
            rows.append({
                "Configuration": config, "Model": name,
                "MAE": mean_absolute_error(y_test, predicted),
                "R2": r2_score(y_test, predicted),
            })
            predictions[(config, name)] = predicted
    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    print("\nRegression results\n", results.round(4).to_string(index=False))

    best = results.iloc[0]
    predicted = predictions[(best["Configuration"], best["Model"])]
    plt.scatter(y_test, predicted, alpha=0.4)
    low, high = min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())
    plt.plot([low, high], [low, high], "r--")
    plt.xlabel("Actual rating")
    plt.ylabel("Predicted rating")
    plt.title(f"{best['Model']} • {best['Configuration']}")
    plt.tight_layout()
    plt.show()
    return results, predictions


def compare_service_correlations(df):
    for target in ("Recommended", "Overall_Rating"):
        print(f"\nService correlations with {target}")
        for column in RATING_COLS:
            correlation, p_value = stats.pearsonr(df[column], df[target])
            print(f"{column}: r={correlation:.3f}, p={p_value:.4e}")


def paired_comparisons(clf_predictions, reg_predictions, y_class_test, y_reg_test):
    combined = clf_predictions[("Combined", "XGBoost")]
    for config in ("Structured-Only", "Text-Only"):
        other = clf_predictions[(config, "XGBoost")]
        _, p_value = stats.ttest_rel(combined == y_class_test.to_numpy(), other == y_class_test.to_numpy())
        print(f"XGBoost Combined vs {config}: paired p={p_value:.4e}")
    combined = reg_predictions[("Combined", "Ridge Regression")]
    for config in ("Structured-Only", "Text-Only"):
        other = reg_predictions[(config, "Ridge Regression")]
        _, p_value = stats.ttest_rel(np.abs(y_class_test.to_numpy() * 0 + y_reg_test.to_numpy() - combined), np.abs(y_reg_test.to_numpy() - other))
        print(f"Ridge Combined vs {config}: paired absolute-error p={p_value:.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/airline_reviews.csv")
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}. See data/README.md")

    download_nltk_resources()
    df = load_and_clean(args.data)
    train, test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df["Recommended"],
    )
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    features = make_features(train, test)
    y_class_train, y_class_test = train["Recommended"].astype(int), test["Recommended"].astype(int)
    y_reg_train, y_reg_test = train["Overall_Rating"].astype(float), test["Overall_Rating"].astype(float)
    clf_results, clf_predictions = evaluate_classification(features, y_class_train, y_class_test)
    reg_results, reg_predictions = evaluate_regression(features, y_reg_train, y_reg_test)
    compare_service_correlations(df)
    paired_comparisons(clf_predictions, reg_predictions, y_class_test, y_reg_test)

    output = Path("results")
    output.mkdir(exist_ok=True)
    clf_results.to_csv(output / "classification_metrics.csv", index=False)
    reg_results.to_csv(output / "regression_metrics.csv", index=False)
    print(f"Metrics saved to {output.resolve()}")


if __name__ == "__main__":
    main()
