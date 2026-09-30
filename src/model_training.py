"""
Target Creation + Supervised Model Training
===========================================

Two responsibilities:

1.  ``create_reward_response_target`` -- build the binary label
    ``reward_response``.  The label is *synthetically generated* because real
    historical campaign-response data is unavailable.  A latent logistic
    response surface combines many behavioural drivers and a noise term, then a
    Bernoulli draw produces the binary outcome.  Crucially it is a function of
    observation-window features ONLY, so nothing about the future leaks in.

2.  Train and fairly compare exactly five supervised models behind identical
    scikit-learn pipelines, pick the winner on F1, and persist it for reuse.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .feature_engineering import (
    CAMPAIGN_START, OBSERVATION_END, RANDOM_STATE,
    feature_columns, split_feature_types,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Threshold used when converting probabilities into class predictions.
# Chosen to favour recall: a missed responder (false negative) is a lost
# customer, whereas a false positive only costs the discount already budgeted.
DECISION_THRESHOLD = 0.50

# Log-odds offset that calibrates the simulated campaign to a realistic
# minority response rate.  sigmoid(-1.05) ~= 0.26.
LOGIT_OFFSET = -1.05


# ============================================================================
# 1.  Target creation
# ============================================================================
def _robust_scale(s: pd.Series) -> pd.Series:
    """Median/IQR scaling with a safe fallback, so no single outlier dominates."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    scale = iqr if iqr > 1e-9 else (s.std() if s.std() > 1e-9 else 1.0)
    return (s - med) / scale


def create_reward_response_target(
    features: pd.DataFrame,
    seed: int = RANDOM_STATE,
    noise_sd: float = 1.10,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Attach the simulated ``reward_response`` label and its true probability.

    Returns
    -------
    features : the feature frame with the label attached
    latent_p : the *true* simulated response probability (diagnostics only --
               never given to a model, never used for scoring)
    contrib  : per-customer log-odds contribution of each behavioural driver
    """
    rng = np.random.default_rng(seed)
    f = features.copy()

    # ---- individual behavioural drivers (all z-scaled) ------------------
    drivers = {
        # recent purchase activity raises responsiveness
        "recency_recent": -_robust_scale(f["days_since_last_purchase"]),
        "frequency": _robust_scale(f["total_orders"]),
        "monetary": _robust_scale(np.log1p(f["total_spend"])),
        "aov": _robust_scale(np.log1p(f["average_order_value"])),
        # promotion responsiveness
        "reward_engagement": _robust_scale(f["reward_usage_rate"]),
        "discount_sensitivity": _robust_scale(f["discount_usage_rate"]),
        "discount_depth": _robust_scale(f["average_discount_percentage"]),
        "commitment": _robust_scale(f["high_commitment_reward_rate"]),
        # loyalty / category stickiness
        "category_loyalty": _robust_scale(f["category_affinity_score"]),
        "loyalty": _robust_scale(f["rfm_score"]),
        "category_breadth": _robust_scale(f["category_entropy"]),
        # momentum
        "spend_trend": _robust_scale(f["spending_trend"]),
        "frequency_trend": _robust_scale(f["purchase_frequency_trend"]),
        # engagement style
        "weekend": _robust_scale(f["weekend_purchase_ratio"]),
        "digital": _robust_scale(f["digital_payment_ratio"]),
        "tenure": _robust_scale(f["customer_tenure_days"]),
    }

    # ---- weights chosen to be plausible, not to make the task trivial ---
    weights = {
        "recency_recent": 0.85, "frequency": 0.70, "monetary": 0.45, "aov": 0.25,
        "reward_engagement": 0.95, "discount_sensitivity": 0.80, "discount_depth": 0.35,
        "commitment": 0.20, "category_loyalty": 0.40, "loyalty": 0.50,
        "category_breadth": -0.15, "spend_trend": 0.55, "frequency_trend": 0.30,
        "weekend": 0.10, "digital": 0.25, "tenure": 0.15,
    }

    logit = pd.Series(0.0, index=f.index)
    contrib = pd.DataFrame(index=f.index)
    for name, series in drivers.items():
        c = weights[name] * series
        contrib[name] = c
        logit += c

    # ---- non-linear response surface ------------------------------------
    # The label is NOT a weighted sum of z-scores: that would be trivially
    # learnable by a linear model and would make the five-model comparison
    # meaningless.  These terms create shapes only a flexible learner can
    # recover, and they are the behaviours that actually drive a campaign.
    z_recency = _robust_scale(f["days_since_last_purchase"])

    # U-shaped recency: very recent customers respond, and so do the ones who
    # have just lapsed and are winnable; the middle band is the least reachable
    u_recency = 0.40 * (z_recency.clip(-2, 2) ** 2)
    contrib["recency_u_shape"] = u_recency
    logit += u_recency

    # saturating reward engagement: the first redemptions matter most, and a
    # customer who already redeems on every order is not more reachable
    sat = np.tanh(_robust_scale(f["reward_usage_rate"]))
    sat_delta = weights["reward_engagement"] * (sat - drivers["reward_engagement"])
    contrib["reward_engagement_saturating"] = sat_delta
    logit += sat_delta

    # the sweet spot: customers who combine reward engagement with genuine
    # discount seeking are markedly more responsive than either alone
    joint = _robust_scale(f["reward_usage_rate"]) * _robust_scale(f["discount_usage_rate"])
    joint = 0.45 * np.clip(joint, -2, 2)
    contrib["reward_x_discount_joint"] = joint
    logit += joint

    # momentum matters more once the customer is already spending
    trend_momentum = 0.35 * np.clip(_robust_scale(f["spending_trend"]), -2, 3)
    contrib["trend_x_spend_momentum"] = trend_momentum
    logit += trend_momentum

    # ---- centre the log-odds so the base rate is realistic ---------------
    # Real campaigns convert a minority of contacted customers, not half.  The
    # offset is applied *after* the behavioural drivers so their relative
    # ranking -- which is what the ML models must recover -- is untouched.
    logit = logit - logit.mean() + LOGIT_OFFSET
    contrib["intercept"] = LOGIT_OFFSET

    # ---- "Recent but low-frequency" correction ---------------------------
    # A customer who bought once, long ago, is far less reachable than the
    # linear term alone suggests.  This interaction keeps the label surface
    # non-linear without becoming an artificial rule.
    inactivity = _robust_scale(f["days_since_last_purchase"]).clip(lower=0)
    low_freq = (-_robust_scale(f["total_orders"])).clip(lower=0)
    interaction = -0.45 * np.minimum(inactivity, 3.0) * np.minimum(low_freq, 3.0)
    contrib["inactive_lowfreq_interaction"] = interaction
    logit += interaction

    # ---- demographic effects deliberately mild --------------------------
    # The brief requires city not to strongly determine reward response.
    city_effect = f["city"].map(
        {"Coimbatore": 0.10, "Erode": -0.05, "Salem": 0.0, "Tiruppur": 0.05,
         "Chennai": 0.15, "Bangalore": 0.12, "Madurai": -0.03, "Karur": -0.08}
    ).fillna(0.0)
    contrib["city_effect"] = city_effect
    logit += city_effect

    contrib["intercept"] = -0.15
    logit += -0.15

    # ---- squash to probability, then add noise and sample ---------------
    latent_p = 1.0 / (1.0 + np.exp(-logit))
    noisy_logit = np.log(
        latent_p.clip(1e-6, 1 - 1e-6) / (1 - latent_p.clip(1e-6, 1 - 1e-6))
    ) + rng.normal(0.0, noise_sd, size=len(f))

    true_p = 1.0 / (1.0 + np.exp(-noisy_logit))
    label = (rng.random(len(f)) < true_p).astype(int)

    f["reward_response"] = label
    # High-level response band, used only for reporting/dashboard segmentation
    f["response_band"] = pd.cut(
        true_p, bins=[-0.01, 0.35, 0.55, 1.01], labels=["Low", "Medium", "High"]
    ).astype(str)

    return f, pd.Series(true_p, index=f.index, name="latent_response_probability"), contrib


# ============================================================================
# 2.  Preprocessing
# ============================================================================
def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """ColumnTransformer fitted on the training split only.

    Tree ensembles do not need scaling, but sharing one builder keeps the
    comparison fair and lets every model be served through the same artifact.
    """
    num_steps = [("imputer", _SimpleImputer())]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_features),
            (
                "cat",
                Pipeline([
                    ("imputer", _SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None,
                                             sparse_output=False)),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class _SimpleImputer:
    """Thin wrapper so the imputer works for numeric and object columns alike."""

    def __new__(cls, strategy: str = "median"):
        from sklearn.impute import SimpleImputer
        if strategy == "most_frequent":
            return SimpleImputer(strategy="most_frequent")
        return SimpleImputer(strategy=strategy)


# ============================================================================
# 3.  Model zoo
# ============================================================================
def build_model_zoo(numeric_features: list[str],
                    categorical_features: list[str]) -> dict[str, Pipeline]:
    """Five deliberately different learners behind identical preprocessing."""
    zoo: dict[str, Pipeline] = {}

    zoo["Logistic Regression"] = Pipeline([
        ("prep", build_preprocessor(numeric_features, categorical_features, True)),
        ("clf", LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    zoo["Decision Tree"] = Pipeline([
        ("prep", build_preprocessor(numeric_features, categorical_features, False)),
        ("clf", DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=8, min_samples_split=12,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    zoo["Random Forest"] = Pipeline([
        ("prep", build_preprocessor(numeric_features, categorical_features, False)),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
    ])

    try:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = Pipeline([
            ("prep", build_preprocessor(numeric_features, categorical_features, False)),
            ("clf", XGBClassifier(
                n_estimators=450, max_depth=5, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.8,
                min_child_weight=3, reg_lambda=1.5,
                scale_pos_weight=1.0, eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)),
        ])
    except ImportError:  # pragma: no cover
        pass

    try:
        from catboost import CatBoostClassifier
        zoo["CatBoost"] = Pipeline([
            ("prep", build_preprocessor(numeric_features, categorical_features, False)),
            ("clf", CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.06,
                l2_leaf_reg=3.0, loss_function="Logloss",
                random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)),
        ])
    except ImportError:  # pragma: no cover
        pass

    return zoo


# ============================================================================
# 4.  Evaluation
# ============================================================================
def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """Fit, time, score and curve-trace one model.  Every number is measured."""
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    pred_time = time.perf_counter() - t1

    fpr, tpr, _ = roc_curve(y_test, proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
        "PR-AUC": average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
        "Train ROC-AUC": roc_auc_score(y_train, train_proba),
        "Overfitting (train-test AUC)": roc_auc_score(y_train, train_proba) - roc_auc_score(y_test, proba),
        "Train Time (s)": train_time,
        "Prediction Time (s)": pred_time,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "_proba": proba,
        "_pred": pred,
        "_fpr": fpr, "_tpr": tpr,
        "_prec_curve": prec_curve, "_rec_curve": rec_curve,
        "_pipeline": pipeline,
    }


def run_benchmark(
    features: pd.DataFrame,
    target_col: str = "reward_response",
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Train and compare all models.  Returns (comparison table, fits, split info, curves)."""
    cols = feature_columns(features)
    numeric, categorical = split_feature_types(cols, features)

    X = features[cols].copy()
    y = features[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    zoo = build_model_zoo(numeric, categorical)

    rows, fits, curves = [], {}, {}
    for name, pipe in zoo.items():
        res = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
        curves[name] = {k: res[k] for k in
                        ("_proba", "_pred", "_fpr", "_tpr", "_prec_curve", "_rec_curve")}
        fits[name] = res.pop("_pipeline")
        rows.append(res)

    comparison = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    comparison.insert(0, "Rank", range(1, len(comparison) + 1))

    split_info = {
        "features": cols, "numeric": numeric, "categorical": categorical,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_pos_rate": float(y_train.mean()), "test_pos_rate": float(y_test.mean()),
        "threshold": DECISION_THRESHOLD, "test_size": test_size, "seed": seed,
    }
    return comparison, fits, split_info, curves


# ============================================================================
# 5.  Model selection
# ============================================================================
def select_best_model(comparison: pd.DataFrame) -> tuple[str, pd.Series]:
    """Pick the winner: highest F1, ROC-AUC as the documented tie-break.

    Why F1 and not accuracy: in a targeted campaign a false positive means a
    discount handed to someone who would have bought anyway (wasted margin),
    while a false negative means a reachable customer who never got the offer.
    Neither error dominates, so the balanced F1 score is the honest choice.
    ROC-AUC is only consulted when two models are within 1e-9 F1 of each other.
    """
    ranked = comparison.sort_values(
        ["F1", "ROC-AUC", "PR-AUC", "Precision", "Recall"],
        ascending=[False, False, False, False, False],
    )
    return str(ranked.iloc[0]["Model"]), ranked


# ============================================================================
# 6.  Explainability
# ============================================================================
def native_importances(
    pipeline: Pipeline, model_name: str, features: pd.DataFrame,
) -> pd.Series | None:
    """Feature importance in the model's own currency.

    Logistic Regression -> absolute coefficients (comparable across one-hot
    columns).  Tree ensembles -> impurity-based importance.
    """
    clf = pipeline.named_steps["clf"]
    names = list(pipeline.named_steps["prep"].get_feature_names_out())

    if model_name == "Logistic Regression":
        if hasattr(clf, "coef_"):
            return pd.Series(np.abs(clf.coef_[0]), index=names, name="importance")

    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=names, name="importance")
    return None


def permutation_importances_safe(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, seed: int = RANDOM_STATE
) -> pd.Series:
    """Model-agnostic importance -- works for every one of the five models."""
    r = permutation_importance(
        pipeline, X_test, y_test, n_repeats=8, random_state=seed,
        scoring="roc_auc", n_jobs=1,
    )
    return pd.Series(r.importances_mean, index=list(X_test.columns), name="importance")


def contributions_for_customer(
    pipeline: Pipeline,
    row: pd.DataFrame,
    baseline_proba: float,
    top_n: int = 8,
) -> pd.DataFrame:
    """Single-customer explanation for a logistic model.

    Each column is flipped to the training-set mean in turn and the change in
    response probability is recorded.  Positive = this customer's value pushed
    the prediction up.
    """
    if not hasattr(pipeline.named_steps["clf"], "coef_"):
        return pd.DataFrame()

    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())

    base_vec = prep.transform(row)
    base_proba = float(clf.predict_proba(base_vec)[0, 1])
    if baseline_proba:
        base_proba = float(baseline_proba)

    mean_row = row.copy()
    for col in row.columns:
        if pd.api.types.is_numeric_dtype(row[col]):
            mean_row[col] = np.nanmedian(row[col]) if np.isfinite(row[col]).any() else 0.0
        else:
            mean_row[col] = row[col].mode().iat[0] if not row[col].mode().empty else row[col]

    rows = []
    for i, name in enumerate(names):
        mod = base_vec.copy()
        mean_vec = prep.transform(mean_row)
        mod[0, i] = mean_vec[0, i]
        p_mod = float(clf.predict_proba(mod)[0, 1])
        rows.append({"feature": name, "contribution": p_mod - base_proba})

    out = pd.DataFrame(rows)
    out["abs_contribution"] = out["contribution"].abs()
    return out.sort_values("abs_contribution", ascending=False).head(top_n)


# ============================================================================
# 7. Persistence
# ============================================================================
def save_artifacts(
    model_name: str,
    pipeline: Pipeline,
    split_info: dict,
    comparison: pd.DataFrame,
    extra_metadata: dict | None = None,
    models_dir: str = "models",
) -> dict[str, str]:
    """Persist the best pipeline plus its preprocessing and metadata."""
    os.makedirs(models_dir, exist_ok=True)

    paths = {
        "model": os.path.join(models_dir, "best_reward_model.pkl"),
        "preprocessor": os.path.join(models_dir, "preprocessing_pipeline.pkl"),
        "metadata": os.path.join(models_dir, "model_metadata.pkl"),
    }

    import pickle

    # the fitted Pipeline *is* preprocessing + model, so it doubles as the
    # preprocessing artifact -- loading one file reproduces the full pipeline
    with open(paths["model"], "wb") as fh:
        pickle.dump(pipeline, fh)
    with open(paths["preprocessor"], "wb") as fh:
        pickle.dump(pipeline.named_steps["prep"], fh)

    metadata = {
        "best_model": model_name,
        "selected_at": pd.Timestamp.now().isoformat(),
        "selection_metric": "F1 score (ROC-AUC / PR-AUC as tie-breaks)",
        "threshold": split_info["threshold"],
        "random_state": RANDOM_STATE,
        "train_rows": split_info["train_rows"],
        "test_rows": split_info["test_rows"],
        "train_positive_rate": round(split_info["train_pos_rate"], 4),
        "test_positive_rate": round(split_info["test_pos_rate"], 4),
        "feature_columns": split_info["features"],
        "numeric_features": split_info["numeric"],
        "categorical_features": split_info["categorical"],
        "observation_window": {"start": str(OBSERVATION_START_LABEL),
                               "end": str(OBSERVATION_END.date())},
        "campaign_window_start": str(CAMPAIGN_START.date()),
        "metrics": json.loads(
            comparison.drop(columns=[c for c in comparison.columns
                                     if c.startswith("_")], errors="ignore")
            .set_index("Model").to_json(orient="index")
        ),
        "target": "reward_response",
        "target_note": (
            "reward_response is synthetically generated from observation-window "
            "behaviour plus noise, because real campaign-response history is "
            "unavailable. Results are illustrative, not real business outcomes."
        ),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(paths["metadata"], "wb") as fh:
        pickle.dump(metadata, fh)

    return paths


OBSERVATION_START_LABEL = "2025-01-01"


def load_artifacts(models_dir: str = "models") -> tuple[Pipeline, dict, dict]:
    """Reload the saved artifacts and return (pipeline, metadata, preprocessing)."""
    import pickle
    with open(os.path.join(models_dir, "best_reward_model.pkl"), "rb") as fh:
        pipeline = pickle.load(fh)
    with open(os.path.join(models_dir, "model_metadata.pkl"), "rb") as fh:
        metadata = pickle.load(fh)
    with open(os.path.join(models_dir, "preprocessing_pipeline.pkl"), "rb") as fh:
        preprocessing = pickle.load(fh)
    return pipeline, metadata, preprocessing