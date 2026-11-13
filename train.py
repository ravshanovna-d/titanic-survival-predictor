"""Train the Titanic survival model, evaluate it, and export it for the web demo.

Usage:  python train.py
Outputs: web/model.onnx, web/meta.json, web/assets/*.png
"""
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from export_utils import check_onnx, examples, export_onnx, feature_schema, write_meta
from features import FamilyFeatures

ROOT = Path(__file__).parent
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
SEED = 42

FEATURES = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "Title"]
TITLES = {"Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
          "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}          # everything else -> "Rare"
LABELS = {"Pclass": "Ticket class", "Sex": "Sex", "Age": "Age", "Fare": "Ticket fare",
          "SibSp": "Siblings / spouses aboard", "Parch": "Parents / children aboard",
          "Embarked": "Port of embarkation", "Title": "Title"}
HELPS = {"Age": "years", "Fare": "£ in 1912 (≈ 7–30 for 3rd class, 30–500 for 1st)"}
OPTION_LABELS = {"Title": {"Rare": "Other (Dr, Rev, Col, Lady, Sir…)",
                           "Master": "Master (boy)", "Miss": "Miss", "Mrs": "Mrs", "Mr": "Mr"}}


def fetch(url: str, dest: Path, tries: int = 4) -> Path:
    """Download once into data/ (retries for flaky connections)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(exist_ok=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return dest
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(fetch(DATA_URL, ROOT / "data" / "titanic.csv"))
    # "Braund, Mr. Owen Harris" -> "Mr"; rare titles are grouped together.
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].map(TITLES).fillna("Rare")
    df["Pclass"] = df["Pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
    for c in ["Age", "Fare", "SibSp", "Parch"]:
        df[c] = df[c].astype(float)
    return df


class AgeByTitle:
    """Fill missing ages with the median age of the passenger's title (learned on training data only).
    Titles carry age information: 'Master' is a boy, 'Miss' is often a young woman."""

    def fit(self, df):
        self.medians_ = df.groupby("Title")["Age"].median()
        return self

    def transform(self, df):
        df = df.copy()
        df["Age"] = df["Age"].fillna(df["Title"].map(self.medians_))
        return df


def make_prep(scale: bool) -> ColumnTransformer:
    """Scaling only matters for the linear model; trees get the raw engineered values."""
    num = StandardScaler() if scale else "passthrough"
    family = Pipeline([("fe", FamilyFeatures()), ("sc", StandardScaler())]) if scale else FamilyFeatures()
    return ColumnTransformer([
        ("age", num, ["Age"]),
        ("family", family, ["SibSp", "Parch", "Fare"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Pclass", "Sex", "Embarked", "Title"]),
    ])


def build_models():
    return {
        "Logistic Regression": Pipeline([("prep", make_prep(True)), ("model", LogisticRegression(max_iter=2000, C=0.5))]),
        "Random Forest": Pipeline([("prep", make_prep(False)), ("model", RandomForestClassifier(
            n_estimators=400, max_depth=7, min_samples_leaf=3, random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", make_prep(False)), ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED))]),
    }


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"{len(df)} passengers, {df['Survived'].mean():.1%} survived, {df['Age'].isna().sum()} missing ages")

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Survived"], random_state=SEED)
    ages = AgeByTitle().fit(train_df)
    train_df, test_df = ages.transform(train_df), ages.transform(test_df)
    X_train, y_train = train_df[FEATURES], train_df["Survived"]
    X_test, y_test = test_df[FEATURES], test_df["Survived"]

    # 1) Compare models with 5x3 repeated stratified CV on the training set
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    models = build_models()
    rows = []
    for name, model in models.items():
        r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=1,  # custom transformer: keep in-process
                           scoring={"ROC-AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy"})
        rows.append({"model": name, **{k: float(np.mean(r[f"test_{k}"])) for k in ("ROC-AUC", "F1", "Accuracy")}})
        print(f"{name:22s} ROC-AUC={rows[-1]['ROC-AUC']:.3f}  F1={rows[-1]['F1']:.3f}  Acc={rows[-1]['Accuracy']:.3f}")
    comparison = sorted(rows, key=lambda r: -r["ROC-AUC"])
    # Prefer the simplest model (dict order = simplest first) within 0.005 ROC-AUC of the top score.
    top = comparison[0]["ROC-AUC"]
    best_name = next(r["model"] for r in rows if r["ROC-AUC"] >= top - 0.005)
    best = models[best_name].fit(X_train, y_train)

    # 2) Final evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
    print(f"Test ({best_name}):", {k: round(v, 3) for k, v in test_metrics.items()})

    # 3) Figures
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Died", "Survived"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "roc_curve.png", dpi=130); plt.close(fig)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
    imp = pd.Series(imp.importances_mean, index=[LABELS[c] for c in X_test.columns]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Most important inputs")
    fig.tight_layout(); fig.savefig(ASSETS / "feature_importance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", order=["1st", "2nd", "3rd"],
                palette=["#4f46e5", "#f59e0b"], errorbar=None, ax=ax)
    ax.set_xlabel("Ticket class"); ax.set_ylabel("Survival rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Survival rate by class and sex")
    fig.tight_layout(); fig.savefig(ASSETS / "survival_by_class_sex.png", dpi=130); plt.close(fig)

    fam = df.assign(family_size=(df["SibSp"] + df["Parch"] + 1).clip(upper=8))
    rate = fam.groupby("family_size")["Survived"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot.bar(ax=ax, color="#4f46e5", rot=0)
    ax.set_xticklabels([str(int(i)) if i < 8 else "8+" for i in rate.index])
    ax.set_xlabel("Family size aboard (incl. passenger)"); ax.set_ylabel("Survival rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Survival by family size (engineered feature)")
    fig.tight_layout(); fig.savefig(ASSETS / "survival_by_family_size.png", dpi=130); plt.close(fig)

    # 4) Export for the browser and verify it matches scikit-learn
    export_onnx(best, X_train, WEB / "model.onnx")
    err = check_onnx(WEB / "model.onnx", best, X_test)
    print(f"ONNX check OK (max diff {err:.2e})")

    write_meta(WEB / "meta.json", {
        "title": "Titanic Survival Predictor",
        "subtitle": ("Would you have survived the Titanic? Enter a passenger profile and the model estimates "
                     "the chance of survival, learned from the real 1912 passenger list."),
        "tags": ["Classification", "scikit-learn", best_name, "Feature engineering", "Cross-validation", "ONNX"],
        "task": "classification",
        "classes": [0, 1], "positive_class": 1, "positive_is_good": True,
        "output_label": "Estimated chance of survival",
        "verdicts": ["Unlikely to survive", "Could go either way", "Likely to survive"],
        "thresholds": [0.35, 0.65],
        "result_note": "Try changing Sex, Ticket class or Title — “women and children first” shows up clearly.",
        "features": feature_schema(X_train, LABELS, HELPS, OPTION_LABELS),
        "examples": examples(X_test),
        "model_name": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metric_formats": {k: "percent" for k in test_metrics if k != "ROC-AUC"},
        "metrics_note": f"Scores on a held-out test set of {len(y_test)} passengers the model never saw during training.",
        "figures": [
            {"src": "assets/survival_by_class_sex.png", "caption": "Women and 1st-class passengers survived far more often."},
            {"src": "assets/survival_by_family_size.png", "caption": "Travelling alone or in a big family lowered the odds."},
            {"src": "assets/feature_importance.png", "caption": "What drives the prediction the most."},
            {"src": "assets/roc_curve.png", "caption": "ROC curve on the test set."},
            {"src": "assets/confusion_matrix.png", "caption": "Correct and wrong predictions on the test set."},
        ],
        "steps": [
            f"Loaded {len(df)} passengers and extracted the title (Mr, Mrs, Miss, Master, rare) from each name.",
            "Filled missing ages with the median age of the same title, learned on the training split only.",
            "Engineered family size, travelling-alone flag and log fare per person inside the pipeline "
            "(a custom transformer with its own ONNX converter, so the browser computes the same features).",
            "Compared Logistic Regression, Random Forest and Gradient Boosting with 5×3 repeated stratified CV.",
            f"Picked {best_name} by ROC-AUC and evaluated it once on a held-out 20% test set.",
            "Exported the whole pipeline to ONNX so it runs in the browser — no server needed.",
        ],
        "dataset": {"name": "Titanic passengers (Kaggle training set)", "url": "https://www.kaggle.com/competitions/titanic/data", "rows": len(df)},
        "repo_url": "https://github.com/ravshanovna-d/titanic-survival-predictor",
        "notebook_url": "https://github.com/ravshanovna-d/titanic-survival-predictor/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    })
    print("Wrote web/model.onnx and web/meta.json")


if __name__ == "__main__":
    main()
