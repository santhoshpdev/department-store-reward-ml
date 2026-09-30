"""
Feature Engineering
===================
Turns a raw transaction log into a customer-level modelling table.

Leakage control
---------------
Every feature is computed inside an explicit *observation window*.  The window
end is passed in by the caller and the module asserts that no row outside the
window survives into the feature frame.  Campaign-window data must never reach
this code path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RANDOM_STATE = 42

OBSERVATION_START = pd.Timestamp("2025-01-01")
OBSERVATION_END = pd.Timestamp("2025-09-30")
CAMPAIGN_START = pd.Timestamp("2025-10-01")

RECENT_WINDOW_DAYS = 90
PREVIOUS_WINDOW_DAYS = 90
EARLY_WINDOW_DAYS = 90

# Columns from customers.csv that are ground truth and must never become features
PERSONA_COLUMNS = ["customer_persona"]

# Categories that get an explicit per-category ratio feature.
# Only a curated subset is promoted to a model column: with a median of 6
# observation orders a customer rarely touches more than 3-4 categories, so
# most of these columns would be sparse zeros.  The *full* customer x category
# table is always returned by `build_category_profile` for analysis and for
# the reward engine, which is where the per-category detail actually belongs.
FEATURED_CATEGORIES = [
    "Groceries", "Baby Care", "Electronics", "Personal Care",
    "Clothing", "Beauty & Cosmetics",
]


def _slug(category: str) -> str:
    return (
        category.lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace(",", "")
        .replace("-", "_")
    )


# Columns computed during the walkthrough but excluded from the modelling
# frame because each is a deterministic function of a retained column.
REDUNDANT_FEATURES = [
    "max_order_value",                 # bounded above by average_order_value x total_orders
    "std_order_value",                 # dispersion, weak at 6 median orders
    "total_quantity",                  # == average_quantity x total_orders
    "total_discount_amount",           # == average_discount_percentage x subtotal
    "spend_per_month",                 # == total_spend / constant
    "std_days_between_orders",         # gap volatility, noisy on short histories
    "max_days_between_orders",         # dominated by average_days_between_orders
    "early_90_day_spend",              # the two 90-day windows carry the trend
    "max_discount_percentage",         # average_discount_percentage is sufficient
    "total_reward_orders",             # == reward_usage_rate x total_orders
    "unique_sub_categories",           # subsumed by unique_categories / unique_brands
    "unique_payment_methods",          # very low variance, no business signal
    "preferred_category_spend",        # == total_spend x spend share of top category
    "secondary_affinity_score",        # only the preferred category drives the reward
    "tenure_months",                   # == customer_tenure_days / 30.44
    "frequency_per_tenure_month",      # == orders_per_month for a fixed window
    "average_order_value_index",       # == average_order_value / global mean
    "last_order_dayofweek",            # weekend_purchase_ratio already captures this
    "rfm_cell",                        # encoded duplicate of R/F/M scores
    "customer_since",                  # identifier, not a feature
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load the three raw CSVs.

    `keep_default_na=False` is essential: the literal string ``"None"`` is a
    valid ``reward_type`` value but pandas would otherwise coerce it to NaN.
    """
    # `None` is a real category in reward_type, so suppress default NA coercion
    na_filter = dict(keep_default_na=False, na_values=[""])

    transactions = pd.read_csv(
        os.path.join(data_dir, "transactions.csv"),
        parse_dates=["transaction_date"],
        **na_filter,
    )
    customers = pd.read_csv(
        os.path.join(data_dir, "customers.csv"),
        parse_dates=["customer_since"],
        **na_filter,
    )
    products = pd.read_csv(
        os.path.join(data_dir, "products.csv"),
        **na_filter,
    )
    return {"transactions": transactions, "customers": customers, "products": products}


def observation_frame(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    start: pd.Timestamp = OBSERVATION_START,
    end: pd.Timestamp = OBSERVATION_END,
) -> pd.DataFrame:
    """Filter transactions to the observation window and enrich with product data."""
    mask = transactions["transaction_date"].between(start, end)
    obs = transactions.loc[mask].merge(
        products[["product_id", "product_name", "category", "sub_category",
                  "brand", "profit_margin"]],
        on="product_id",
        how="left",
    )
    obs = obs.sort_values(["customer_id", "transaction_date"]).reset_index(drop=True)
    obs["gross_profit"] = (obs["total_amount"] * obs["profit_margin"]).round(2)
    obs["is_weekend"] = obs["transaction_date"].dt.dayofweek.isin([5, 6]).astype(int)
    obs["reward_used_bool"] = obs["reward_used"].astype(bool)
    obs["discount_amt"] = obs["discount_amount"]
    return obs


# ----------------------------------------------------------------------------
# RFM
# ----------------------------------------------------------------------------
def _quintile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """1-5 quintile score with stable tie handling.

    Ranks are rescaled to the unit interval before binning, so the lowest
    ranked customer scores 1 and the highest ranked scores 5 regardless of
    whether a *high* or a *low* raw value is desirable.
    """
    s = series.dropna()
    if s.empty or s.nunique() <= 1:
        return pd.Series(3.0, index=series.index, dtype=float)

    ranks = series.rank(method="first", ascending=ascending)
    n = ranks.notna().sum()
    if n <= 1:
        return pd.Series(3.0, index=series.index, dtype=float)

    pct = (ranks - 1) / (n - 1)
    scores = np.ceil(pct * 5).clip(1, 5)
    return scores.astype(float)


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach R/F/M quintile scores, an RFM code and a business segment."""
    out = df.copy()
    out["R_score"] = _quintile(out["days_since_last_purchase"], ascending=False)
    out["F_score"] = _quintile(out["total_orders"], ascending=True)
    out["M_score"] = _quintile(out["total_spend"], ascending=True)
    out["rfm_score"] = (
        out["R_score"] + out["F_score"] + out["M_score"]
    ).astype(int)
    out["rfm_cell"] = (
        out["R_score"].astype(int).astype(str)
        + out["F_score"].astype(int).astype(str)
        + out["M_score"].astype(int).astype(str)
    )
    out["rfm_segment"] = assign_rfm_segment(out)
    return out


def assign_rfm_segment(df: pd.DataFrame) -> pd.Series:
    """Classic RFM business segments derived from the calculated scores.

    Rules (R and F only, M used to split the two "cannot lose" groups):
        R >= 4, F >= 4, M >= 4            -> Champions
        R >= 3, F >= 3                     -> Loyal Customers
        R >= 4, F <= 2                     -> Potential Loyalists
        R == 5, F == 1, M <= 3             -> New Customers
        R == 5, F == 1, M >= 4             -> Promising
        R == 2, F >= 3, M >= 4             -> Cannot Lose Them
        R <= 2, F >= 3                     -> At Risk
        R == 3, F <= 2                     -> Needs Attention
        R <= 2, F <= 2                     -> Hibernating / Lost
    """
    r, f, m = df["R_score"].astype(int), df["F_score"].astype(int), df["M_score"].astype(int)

    seg = pd.Series("Lost", index=df.index, dtype=object)

    seg[(r >= 4) & (f >= 4) & (m >= 4)] = "Champions"
    seg[(r >= 3) & (f >= 3) & (seg == "Lost")] = "Loyal Customers"
    seg[(r >= 4) & (f <= 2) & (seg == "Lost")] = "Potential Loyalists"
    seg[(r == 5) & (f == 1) & (m <= 3)] = "New Customers"
    seg[(r == 5) & (f == 1) & (m >= 4)] = "Promising"
    seg[(r <= 2) & (f >= 3) & (m >= 4)] = "Cannot Lose Them"
    seg[(r <= 2) & (f >= 3) & (seg == "Lost")] = "At Risk"
    seg[(r == 3) & (f <= 2) & (seg == "Lost")] = "Needs Attention"
    seg[(r <= 2) & (f <= 2) & (m <= 2)] = "Hibernating"
    return seg


# ----------------------------------------------------------------------------
# Category affinity
# ----------------------------------------------------------------------------
def build_category_profile(
    obs: pd.DataFrame,
) -> pd.DataFrame:
    """Long-format customer x category profile.

    Columns:
        category_purchase_count   orders containing that category
        category_spend            net spend on that category
        category_purchase_ratio   share of the customer's orders
        category_spend_ratio      share of the customer's net spend
        category_affinity_score   0.5 * purchase_ratio + 0.5 * spend_ratio
        category_avg_order_value  mean order value for that category
    """
    grp = (
        obs.groupby(["customer_id", "category"], observed=True)
        .agg(
            category_purchase_count=("transaction_id", "count"),
            category_spend=("total_amount", "sum"),
            category_units=("quantity", "sum"),
        )
        .reset_index()
    )

    totals = obs.groupby("customer_id", observed=True).agg(
        _total_orders=("transaction_id", "count"),
        _total_spend=("total_amount", "sum"),
    )
    grp = grp.join(totals, on="customer_id")

    grp["category_purchase_ratio"] = (
        grp["category_purchase_count"] / grp["_total_orders"]
    ).clip(0, 1)
    grp["category_spend_ratio"] = (
        grp["category_spend"] / grp["_total_spend"].replace(0, np.nan)
    ).fillna(0.0).clip(0, 1)
    grp["category_affinity_score"] = (
        0.5 * grp["category_purchase_ratio"] + 0.5 * grp["category_spend_ratio"]
    )
    grp["category_avg_order_value"] = (
        grp["category_spend"] / grp["category_purchase_count"].replace(0, np.nan)
    ).fillna(0.0)

    return grp.drop(columns=["_total_orders", "_total_spend"])


def attach_affinity_features(
    features: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Add preferred category, affinity score, diversity and per-category ratios."""
    out = features.copy()

    ranked = profile.sort_values(
        ["customer_id", "category_affinity_score", "category_spend", "category"],
        ascending=[True, False, False, True],
    )
    top = ranked.groupby("customer_id", observed=True).head(1).set_index("customer_id")
    out["preferred_category"] = top["category"]
    out["category_affinity_score"] = top["category_affinity_score"]
    out["preferred_category_orders"] = top["category_purchase_count"]
    out["preferred_category_spend"] = top["category_spend"]

    # second choice, used as a fallback reward category when affinity is weak
    second = ranked.groupby("customer_id", observed=True).nth(1).set_index("customer_id")
    out["secondary_category"] = second["category"].reindex(out.index).fillna("Other")

    # diversity of the category basket (1.0 = spread across all categories)
    ent = profile.groupby("customer_id", observed=True)["category_purchase_ratio"].apply(
        lambda s: float(-(s * np.log(s.clip(lower=1e-12))).sum())
    )
    out["category_entropy"] = ent / np.log(len(FEATURED_CATEGORIES))

    # per-category purchase ratios, one column per category
    wide = profile.pivot_table(
        index="customer_id", columns="category", values="category_purchase_ratio",
        aggfunc="sum", fill_value=0.0,
    )
    for cat in FEATURED_CATEGORIES:
        col = f"cat_ratio_{_slug(cat)}"
        if cat in wide.columns:
            out[col] = wide[cat].reindex(out.index).fillna(0.0).to_numpy()
        else:
            out[col] = 0.0

    out["category_entropy"] = out["category_entropy"].fillna(0.0)
    out["category_affinity_score"] = out["category_affinity_score"].fillna(0.0)
    return out


# ----------------------------------------------------------------------------
# Main feature builder
# ----------------------------------------------------------------------------
def build_customer_features(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    start: pd.Timestamp = OBSERVATION_START,
    end: pd.Timestamp = OBSERVATION_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the customer-level ML dataset.

    Returns
    -------
    features   : one row per customer, all behaviour + demographic features
    profile    : long-format customer x category affinity table
    """
    obs = observation_frame(transactions, products, start, end)

    # hard leakage guard -------------------------------------------------
    assert obs["transaction_date"].max() <= end, "observation window violation"
    assert obs["transaction_date"].min() >= start, "observation window violation"

    g = obs.groupby("customer_id", observed=True)

    feats = pd.DataFrame(index=g.size().index)
    feats.index.name = "customer_id"

    # ---- volume / monetary --------------------------------------------
    feats["total_orders"] = g.size()
    feats["total_spend"] = g["total_amount"].sum()
    feats["average_order_value"] = g["total_amount"].mean()
    feats["median_order_value"] = g["total_amount"].median()
    feats["max_order_value"] = g["total_amount"].max()
    feats["std_order_value"] = g["total_amount"].std().fillna(0.0)
    feats["total_quantity"] = g["quantity"].sum()
    feats["average_quantity"] = g["quantity"].mean()
    feats["average_unit_price"] = g["unit_price"].mean()
    feats["total_discount_amount"] = g["discount_amt"].sum()
    feats["gross_profit"] = g["gross_profit"].sum()

    obs_months = max(1.0, (end - start).days / 30.44)
    feats["orders_per_month"] = feats["total_orders"] / obs_months
    feats["spend_per_month"] = feats["total_spend"] / obs_months

    # ---- recency / temporal -------------------------------------------
    last = g["transaction_date"].max()
    first = g["transaction_date"].min()
    feats["days_since_last_purchase"] = (end - last).dt.days
    feats["days_since_first_purchase"] = (last - first).dt.days
    feats["weekend_purchase_ratio"] = g["is_weekend"].mean()
    feats["last_order_dayofweek"] = last.dt.dayofweek
    feats["last_order_month"] = last.dt.month

    # inter-purchase gaps
    gaps = (
        obs.sort_values(["customer_id", "transaction_date"])
        .groupby("customer_id", observed=True)["transaction_date"]
        .diff()
        .dt.days
    )
    gap_stats = gaps.groupby(level=0).agg(["mean", "std", "max"])
    feats["average_days_between_orders"] = gap_stats["mean"].reindex(feats.index).fillna(0.0)
    feats["std_days_between_orders"] = gap_stats["std"].reindex(feats.index).fillna(0.0)
    feats["max_days_between_orders"] = gap_stats["max"].reindex(feats.index).fillna(0.0)

    # ---- 90-day window comparison -------------------------------------
    win_recent = obs[obs["transaction_date"] > end - pd.Timedelta(days=RECENT_WINDOW_DAYS)]
    win_prev = obs[
        (obs["transaction_date"] > end - pd.Timedelta(days=RECENT_WINDOW_DAYS + PREVIOUS_WINDOW_DAYS))
        & (obs["transaction_date"] <= end - pd.Timedelta(days=RECENT_WINDOW_DAYS))
    ]
    win_early = obs[obs["transaction_date"] <= end - pd.Timedelta(days=RECENT_WINDOW_DAYS + PREVIOUS_WINDOW_DAYS)]

    recent_g = win_recent.groupby("customer_id", observed=True)
    prev_g = win_prev.groupby("customer_id", observed=True)
    early_g = win_early.groupby("customer_id", observed=True)

    feats["recent_90_day_spend"] = recent_g["total_amount"].sum().reindex(feats.index).fillna(0.0)
    feats["recent_90_day_orders"] = recent_g.size().reindex(feats.index).fillna(0)
    feats["previous_90_day_spend"] = prev_g["total_amount"].sum().reindex(feats.index).fillna(0.0)
    feats["previous_90_day_orders"] = prev_g.size().reindex(feats.index).fillna(0)
    feats["early_90_day_spend"] = early_g["total_amount"].sum().reindex(feats.index).fillna(0.0)

    feats["recent_30_day_spend"] = (
        obs[obs["transaction_date"] > end - pd.Timedelta(days=30)]
        .groupby("customer_id", observed=True)["total_amount"]
        .sum()
        .reindex(feats.index)
        .fillna(0.0)
    )

    denom = feats["previous_90_day_spend"].replace(0, np.nan)
    feats["spending_trend"] = (
        (feats["recent_90_day_spend"] - feats["previous_90_day_spend"]) / denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 10)

    denom_o = feats["previous_90_day_orders"].replace(0, np.nan)
    feats["purchase_frequency_trend"] = (
        (feats["recent_90_day_orders"] - feats["previous_90_day_orders"]) / denom_o
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 10)

    # ---- discount & reward engagement ---------------------------------
    feats["average_discount_percentage"] = g["discount_percentage"].mean()
    feats["max_discount_percentage"] = g["discount_percentage"].max()
    obs = obs.assign(_has_disc=(obs["discount_amt"] > 0).astype(int))
    feats["discount_usage_rate"] = (
        obs.groupby("customer_id", observed=True)["_has_disc"].mean()
    ).reindex(feats.index).fillna(0.0)

    reward_only = obs[obs["reward_used_bool"]]
    rg = reward_only.groupby("customer_id", observed=True)
    feats["total_reward_orders"] = rg.size().reindex(feats.index).fillna(0)
    feats["reward_usage_rate"] = (feats["total_reward_orders"] / feats["total_orders"]).fillna(0.0)
    feats["reward_discount_amount"] = rg["discount_amt"].sum().reindex(feats.index).fillna(0.0)

    # BOGO and cashback are higher-commitment reward mechanics
    bg = reward_only[reward_only["reward_type"].isin(["BOGO", "Cashback"])]
    feats["high_commitment_reward_rate"] = (
        bg.groupby("customer_id", observed=True).size()
        .reindex(feats.index).fillna(0) / feats["total_orders"]
    ).fillna(0.0)

    # ---- basket / product diversity -----------------------------------
    feats["unique_products"] = g["product_id"].nunique()
    feats["unique_categories"] = g["category"].nunique()
    feats["unique_brands"] = g["brand"].nunique()
    feats["unique_sub_categories"] = g["sub_category"].nunique()
    feats["unique_payment_methods"] = g["payment_method"].nunique()
    digital = obs.assign(_digital=obs["payment_method"].isin(["UPI", "Credit Card", "Debit Card", "Net Banking"]).astype(int))
    feats["digital_payment_ratio"] = (
        digital.groupby("customer_id", observed=True)["_digital"].mean()
    ).reindex(feats.index).fillna(0.0)

    # top brand share within the customer's basket
    top_brand = (
        obs.groupby(["customer_id", "brand"], observed=True).size().rename("n").reset_index()
    )
    top_brand["share"] = top_brand["n"] / top_brand.groupby("customer_id")["n"].transform("sum")
    feats["top_brand_share"] = (
        top_brand.sort_values(["customer_id", "share"], ascending=[True, False])
        .groupby("customer_id", observed=True).head(1)
        .set_index("customer_id")["share"]
        .reindex(feats.index).fillna(0.0)
    )

    # ---- category affinity --------------------------------------------
    profile = build_category_profile(obs)
    feats = attach_affinity_features(feats, profile)

    # ---- demographics (persona excluded) --------------------------------
    demo = customers.set_index("customer_id")
    feats["age"] = demo["age"].reindex(feats.index)
    feats["gender"] = demo["gender"].reindex(feats.index)
    feats["city"] = demo["city"].reindex(feats.index)
    feats["customer_since"] = demo["customer_since"].reindex(feats.index)
    feats["customer_tenure_days"] = (end - feats["customer_since"]).dt.days.clip(lower=0)
    feats["tenure_months"] = feats["customer_tenure_days"] / 30.44

    def _age_band(a: float) -> str:
        if a < 25:
            return "Under 25"
        if a < 35:
            return "25-34"
        if a < 45:
            return "35-44"
        if a < 55:
            return "45-54"
        return "55+"

    feats["age_band"] = feats["age"].apply(_age_band)

    # ---- derived indices ----------------------------------------------
    feats["frequency_per_tenure_month"] = feats["total_orders"] / (
        feats["customer_tenure_days"].replace(0, np.nan) / 30.44
    ).fillna(0.0).clip(lower=1.0)
    feats["average_order_value_index"] = feats["average_order_value"] / feats["average_order_value"].mean()
    feats["loyalty_index"] = (
        feats["total_orders"].rank(pct=True) * 0.5
        + feats["total_spend"].rank(pct=True) * 0.5
    )
    feats["engagement_index"] = (
        feats["reward_usage_rate"] * 0.4
        + feats["discount_usage_rate"] * 0.3
        + feats["weekend_purchase_ratio"] * 0.3
    )
    feats["price_sensitivity_index"] = (
        feats["average_discount_percentage"] / 100.0 * 0.6
        + feats["discount_usage_rate"] * 0.4
    )
    feats["urgency_score"] = (
        feats["days_since_last_purchase"].rank(pct=True) * 0.5
        + feats["category_entropy"].rank(pct=True) * 0.5
    )

    # ---- RFM -----------------------------------------------------------
    feats = add_rfm_features(feats)

    feats = feats.reset_index()

    # ---- drop redundant / near-duplicate columns -------------------------
    # Every exclusion below is a deterministic function of a column that is
    # kept, so including it would add no information while diluting the
    # importance plots and slowing training.
    feats = feats.drop(columns=REDUNDANT_FEATURES, errors="ignore")
    num_cols = feats.select_dtypes(include=[np.number]).columns
    feats[num_cols] = feats[num_cols].astype(float)
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # forward/back fill the sparse ratio columns, then fill any remainder
    feats = feats.fillna(0.0)

    return feats, profile


# ----------------------------------------------------------------------------
# Feature column selection
# ----------------------------------------------------------------------------
# Columns that identify a customer, describe a *label*, or describe a segment.
# None of these may be used as a model input -- otherwise the model would be
# reading the answer instead of the customer's purchase behaviour.
NON_FEATURE_COLUMNS = [
    # identifiers
    "customer_id", "customer_since", "rfm_cell",
    # target
    "reward_response", "response_probability", "latent_response_probability",
    "response_band",
    # segment / cluster assignments
    "rfm_segment", "customer_persona", "cluster_id", "segment_label",
    # reward-engine outputs
    "recommended_reward", "reward_discount_pct", "reward_flat_amount",
    "reward_probability", "expected_campaign_value", "campaign_priority_score",
]


# Composite indices that ARE retained in the feature frame (the reward engine
# uses them) but are exact linear combinations of columns already in the model
# set.  Feeding them in as well would only dilute importance plots.
MODEL_EXCLUSIONS = [
    "loyalty_index",            # 0.5 * rank(total_orders) + 0.5 * rank(total_spend)
    "engagement_index",         # 0.4*reward_usage + 0.3*discount_usage + 0.3*weekend
    "price_sensitivity_index",  # 0.6*avg_discount + 0.4*discount_usage
    "urgency_score",            # 0.5 * rank(recency) + 0.5 * rank(category_entropy)
]


def feature_columns(df: pd.DataFrame, drop: list[str] | None = None) -> list[str]:
    """Return the modelling feature list.

    Numeric columns are all eligible.  Categorical columns are eligible only
    while they stay low-cardinality, so identifiers and free-text-like fields
    are skipped (city is kept -- 8 levels, and a legitimate weak signal).
    """
    drop = set(NON_FEATURE_COLUMNS) | set(MODEL_EXCLUSIONS) | set(drop or [])
    cols: list[str] = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        elif df[c].nunique(dropna=False) <= 20:
            cols.append(c)
    return cols


def split_feature_types(columns: list[str], df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition the feature list into (numeric, categorical)."""
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in columns if c not in numeric]
    return numeric, categorical