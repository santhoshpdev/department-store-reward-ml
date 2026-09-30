"""
Visualization
=============

Charts, architecture diagram and the HTML dashboards used in the notebook.

All plotting functions return the matplotlib Figure/Axes so the notebook can
tighten layout, annotate and save.  Colour is kept on a consistent categorical
palette so figures remain readable when compared side by side.
"""

from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.frameon": False,
})

PALETTE = ["#1f4e79", "#2e86ab", "#17b3a3", "#f37748", "#f0a202",
           "#c94c63", "#7d3c98", "#34495e", "#16a085", "#d35400",
           "#2c3e50", "#8e44ad"]

MODEL_COLOURS = {
    "Logistic Regression": "#1f4e79",
    "Decision Tree": "#8e44ad",
    "Random Forest": "#16a085",
    "XGBoost": "#d35400",
    "CatBoost": "#c94c63",
}

SEGMENT_ORDER = [
    "Champions", "Loyal Customers", "Cannot Lose Them", "Potential Loyalists",
    "Promising", "New Customers", "Needs Attention", "At Risk",
    "Hibernating", "Lost",
]

CURRENCY = "\u20b9"


def _money(v: float, decimals: int = 0) -> str:
    return f"{CURRENCY}{v:,.{decimals}f}"


def _save(fig, path: str | None):
    if path:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    return fig


# ============================================================================
# Model comparison
# ============================================================================
def metric_comparison_bars(comparison: pd.DataFrame, metric: str, ax=None, title: str | None = None):
    """Single-metric bar chart across the five models, sorted and value-labelled."""
    df = comparison.sort_values(metric, ascending=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.8))

    cols = [MODEL_COLOURS.get(m, "#7f8c8d") for m in df["Model"]]
    bars = ax.barh(df["Model"], df[metric], color=cols, edgecolor="white", linewidth=0.6)

    lo, hi = df[metric].min(), df[metric].max()
    pad = (hi - lo) * 0.28 or 0.01
    ax.set_xlim(max(0.0, lo - pad), hi + pad)
    for bar, v in zip(bars, df[metric]):
        ax.text(bar.get_width() + pad * 0.08, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_title(title or f"{metric} Comparison (test set)")
    ax.set_xlabel(metric)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    return ax


def model_benchmark_dashboard(comparison: pd.DataFrame, save_dir: str | None = None):
    """Four-panel metric comparison for the headline metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m in zip(axes.flat, metrics):
        metric_comparison_bars(comparison, m, ax=ax)
    fig.suptitle("Model Benchmark \u2014 Held-out Test Set Performance", fontsize=15, y=1.02)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/model_benchmark.png" if save_dir else None)


def roc_curves_plot(curves: dict, comparison: pd.DataFrame, save_dir: str | None = None):
    """All five ROC curves on one figure, each labelled with its real AUC."""
    fig, ax = plt.subplots(figsize=(8, 6.2))

    for name, c in curves.items():
        auc = float(comparison.loc[comparison["Model"] == name, "ROC-AUC"].iloc[0])
        ax.plot(c["_fpr"], c["_tpr"],
                label=f"{name} \u2014 {auc:.3f}",
                color=MODEL_COLOURS.get(name, "#7f8c8d"),
                linewidth=2.0)

    ax.plot([0, 1], [0, 1], ls="--", c="#bdc3c7", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves \u2014 All Models on the Held-out Test Set")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/roc_curves.png" if save_dir else None)


def pr_curves_plot(curves: dict, comparison: pd.DataFrame, save_dir: str | None = None):
    """All five precision-recall curves, labelled with real PR-AUC."""
    fig, ax = plt.subplots(figsize=(8, 6.2))

    for name, c in curves.items():
        ap = float(comparison.loc[comparison["Model"] == name, "PR-AUC"].iloc[0])
        ax.plot(c["_rec_curve"], c["_prec_curve"],
                label=f"{name} \u2014 {ap:.3f}",
                color=MODEL_COLOURS.get(name, "#7f8c8d"),
                linewidth=2.0)

    pos = comparison["PR-AUC"].mean()
    ax.axhline(pos, ls=":", c="#bdc3c7", lw=1, label=f"Mean PR-AUC ({pos:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves \u2014 Relevant to Reward Targeting")
    ax.set_ylim(-0.02, 1.04)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/pr_curves.png" if save_dir else None)


def confusion_matrix_grid(comparison: pd.DataFrame, save_dir: str | None = None):
    """Consistent 2x3 confusion-matrix grid, one panel per model."""
    names = list(comparison["Model"])
    rows = comparison.set_index("Model")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    max_val = max(rows.loc[n, ["TN", "FP", "FN", "TP"]].max() for n in names)

    for i, name in enumerate(names):
        cm = np.array([[rows.loc[name, "TN"], rows.loc[name, "FP"]],
                       [rows.loc[name, "FN"], rows.loc[name, "TP"]]])
        im = axes[i].imshow(cm, cmap="Blues", vmin=0, vmax=max_val * 1.15)
        axes[i].set_xticks([0, 1], ["Pred 0", "Pred 1"])
        axes[i].set_yticks([0, 1], ["Actual 0", "Actual 1"])
        axes[i].grid(visible=False)
        for (x, y), v in zip([(0, 0), (1, 0), (0, 1), (1, 1)], cm.ravel()):
            axes[i].text(x, y, f"{int(v):,}", ha="center", va="center",
                         fontsize=13, fontweight="bold",
                         color="white" if v > max_val * 0.55 else "#1a2b3c")
        axes[i].set_title(f"{name}\nF1 {rows.loc[name, 'F1']:.3f} "
                          f"\u00b7 Acc {rows.loc[name, 'Accuracy']:.3f}", fontsize=10)

    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.colorbar(im, ax=axes[:len(names)], shrink=0.55, label="Count")
    fig.suptitle("Confusion Matrices \u2014 All Five Models (Held-out Test Set)",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/confusion_matrices.png" if save_dir else None)


def train_time_plot(comparison: pd.DataFrame, save_dir: str | None = None):
    """Training and prediction time side by side (log-friendly scale)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comparison))
    w = 0.38
    ax.bar(x - w / 2, comparison["Train Time (s)"], w, label="Train time (s)",
           color="#1f4e79")
    ax.bar(x + w / 2, comparison["Prediction Time (s)"], w, label="Prediction time (s)",
           color="#f0a202")
    ax.set_xticks(x, comparison["Model"], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Seconds (log scale)")
    ax.set_title("Training vs Prediction Time")
    ax.legend()
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/train_time.png" if save_dir else None)


def overfitting_plot(comparison: pd.DataFrame, save_dir: str | None = None):
    """Train vs test ROC-AUC gap, the main overfitting diagnostic."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comparison))
    w = 0.38
    ax.bar(x - w / 2, comparison["Train ROC-AUC"], w, label="Train ROC-AUC", color="#2e86ab")
    ax.bar(x + w / 2, comparison["ROC-AUC"], w, label="Test ROC-AUC", color="#16a085")
    ax.set_xticks(x, comparison["Model"], rotation=20, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Overfitting Check \u2014 Train vs Test ROC-AUC")
    ax.legend()
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/overfitting.png" if save_dir else None)


def feature_importance_plot(perm_importance: pd.Series,
                            native_importance: pd.Series | None,
                            model_name: str,
                            save_dir: str | None = None,
                            top_n: int = 15):
    """Top-N importances, model-agnostic permutation on the left and the model's
    own importances on the right where they exist."""
    n_plots = 2 if native_importance is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(13, 5.6))

    top = perm_importance.sort_values(ascending=False).head(top_n)
    axes[0].barh(top.index[::-1], top.values[::-1], color="#1f4e79")
    axes[0].set_title(f"Permutation Importance \u2014 {model_name}")
    axes[0].set_xlabel("Mean drop in ROC-AUC when shuffled")
    for i, v in enumerate(top.values[::-1]):
        axes[0].text(v + top.max() * 0.01, i, f"{v:.4f}", va="center", fontsize=8)

    if native_importance is not None:
        nt = native_importance.sort_values(ascending=False).head(top_n)
        axes[1].barh(nt.index[::-1], nt.values[::-1], color="#c94c63")
        axes[1].set_title(f"Model's Own Importance \u2014 {model_name}")
        axes[1].set_xlabel("Importance (model currency)")
        for i, v in enumerate(nt.values[::-1]):
            axes[1].text(v + nt.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle(f"What Drives Reward Response \u2014 {model_name}", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/feature_importance.png" if save_dir else None)


# ============================================================================
# Business / analytics charts
# ============================================================================
def revenue_trend_chart(transactions: pd.DataFrame, save_dir: str | None = None):
    """Monthly revenue with a 3-month rolling trend."""
    tx = transactions.copy()
    tx["month"] = tx["transaction_date"].dt.to_period("M")
    m = tx.groupby("month").agg(revenue=("total_amount", "sum"),
                                orders=("transaction_id", "count")).reset_index()
    m["month"] = m["month"].dt.to_timestamp()
    m["revenue"] = m["revenue"] / 1e7  # scale to crores for readability

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(m["month"], m["revenue"], color="#1f4e79", width=22)
    ax.plot(m["month"], m["revenue"].rolling(3, min_periods=1).mean(),
            color="#f37748", marker="o", ms=5, lw=2, label="3-month rolling mean")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (\u20b9 crore)")
    ax.set_title("Monthly Revenue Trend (full 12-month history)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/revenue_trend.png" if save_dir else None)


def category_revenue_chart(transactions: pd.DataFrame, products: pd.DataFrame,
                           save_dir: str | None = None):
    """Revenue by category, horizontal bars, sorted."""
    tx = transactions.merge(products[["product_id", "category", "profit_margin"]],
                            on="product_id")
    g = tx.groupby("category").agg(
        revenue=("total_amount", "sum"),
        orders=("transaction_id", "count"),
        margin=("profit_margin", "mean"),
    ).reset_index().sort_values("revenue", ascending=False)
    g["revenue_cr"] = g["revenue"] / 1e7

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(g["category"], g["revenue_cr"],
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(g))])
    ax.set_xlabel("Revenue (\u20b9 crore)")
    ax.set_title("Category Revenue Contribution")
    for bar, v, o in zip(bars, g["revenue_cr"], g["orders"]):
        ax.text(bar.get_width() + v * 0.01, bar.get_y() + bar.get_height() / 2,
                f"\u20b9{v:.2f}cr \u00b7 {o:,} orders", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/category_revenue.png" if save_dir else None)


def rfm_segment_chart(features: pd.DataFrame, save_dir: str | None = None):
    """RFM segment mix with the segment names ordered sensibly."""
    counts = features["rfm_segment"].value_counts()
    counts = counts.reindex([s for s in SEGMENT_ORDER if s in counts.index])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
    bars = ax.bar(counts.index, counts.values, color=cols)
    ax.set_ylabel("Customers")
    ax.set_title("RFM Segment Distribution")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(counts) * 0.02,
                f"{int(v)}", ha="center", fontsize=9)
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/rfm_segments.png" if save_dir else None)


def segment_profile_chart(features: pd.DataFrame, save_dir: str | None = None):
    """For each RFM segment: spend, order frequency, recency and, when available,
    reward response probability.

    The response panel falls back to the RFM score if the model has not been
    scored yet, so this chart is usable before or after model training.
    """
    if "reward_probability" in features.columns:
        response_col, response_label = "reward_probability", "Avg reward response probability"
    elif "reward_response" in features.columns:
        response_col, response_label = "reward_response", "Avg reward response (0/1)"
    else:
        response_col, response_label = "rfm_score", "Average RFM score (max 15)"

    g = features.groupby("rfm_segment").agg(
        customers=("customer_id", "size"),
        avg_spend=("total_spend", "mean"),
        avg_orders=("total_orders", "mean"),
        avg_recency=("days_since_last_purchase", "mean"),
        avg_response=(response_col, "mean"),
    ).reindex(SEGMENT_ORDER).dropna().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = [("avg_spend", "Average total spend (\u20b9)"),
             ("avg_orders", "Average orders"),
             ("avg_recency", "Average recency (days)"),
             ("avg_response", response_label)]
    for ax, (col, title) in zip(axes.flat, specs):
        bars = ax.bar(g["rfm_segment"], g[col],
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(g))])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, g[col]):
            if col == "avg_spend":
                label = _money(v)
            elif col == "avg_response" and response_col != "rfm_score":
                label = f"{v:.2f}"
            else:
                label = f"{v:,.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, v, label,
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="x", visible=False)

    fig.suptitle("RFM Segments Are Actionable \u2014 Behaviour by Segment", fontsize=14, y=1.0)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/segment_profile.png" if save_dir else None)


def kmeans_elbow_plot(scores: pd.DataFrame, save_dir: str | None = None):
    """Elbow curve with the silhouette score overlaid."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(scores["k"], scores["inertia"], marker="o", color="#1f4e79", lw=2,
            label="Inertia (sum of squared distances)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia", color="#1f4e79")
    ax2 = ax.twinx()
    ax2.plot(scores["k"], scores["silhouette"], marker="s", color="#f37748", lw=2,
             label="Silhouette score")
    ax2.set_ylabel("Silhouette score", color="#f37748")
    ax2.grid(False)
    if "recommended" in scores.columns:
        rec = scores.loc[scores["recommended"], "k"]
        for k in rec:
            ax.axvline(k, ls="--", c="#16a085", lw=1.5, alpha=0.8)
            ax.text(k, ax.get_ylim()[1] * 0.98, f" k={int(k)}", color="#16a085",
                    fontsize=9, va="top", rotation=90)
    ax.set_title("K-Means \u2014 Elbow Curve and Silhouette Score")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/kmeans_elbow.png" if save_dir else None)


def kmeans_cluster_plot(clusters: pd.DataFrame, save_dir: str | None = None):
    """Scatter of the two most informative clustering dimensions, coloured by cluster."""
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    labels = clusters["cluster_label"].astype(str)
    for i, lab in enumerate(labels.unique()):
        sub = clusters[labels == lab]
        ax.scatter(sub["recency_days"], sub["total_spend"],
                   s=28, alpha=0.72, label=lab, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Total spend (\u20b9)")
    ax.set_title("K-Means Customer Clusters")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/kmeans_clusters.png" if save_dir else None)


def category_affinity_chart(category_summary: pd.DataFrame, save_dir: str | None = None):
    """Category-level customer count, revenue and reward response rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    specs = [("customer_count", "Customers preferring this category", PALETTE[0]),
             ("revenue", "Revenue (\u20b9 crore)", PALETTE[2]),
             ("reward_response_rate", "Reward response rate", PALETTE[4])]
    for ax, (col, title, colour) in zip(axes, specs):
        df = category_summary.sort_values(col, ascending=False)
        ax.bar(df["category"], df[col], color=colour)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        if col == "revenue":
            for i, v in enumerate(df[col]):
                ax.text(i, v, f"{v:.2f}", ha="center", fontsize=8)
        ax.grid(axis="x", visible=False)

    fig.suptitle("Product & Category Intelligence \u2014 Personalisation Grounding",
                 fontsize=14, y=1.03)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/category_affinity.png" if save_dir else None)


def reward_distribution_chart(recommendations: pd.DataFrame, save_dir: str | None = None):
    """How many customers receive each reward type."""
    counts = recommendations["recommended_reward"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
    bars = ax.bar(counts.index, counts.values, color=cols)
    ax.set_ylabel("Customers")
    ax.set_title("Personalised Reward Distribution \u2014 Offers Are NOT Identical")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{int(v)}",
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/reward_distribution.png" if save_dir else None)


def discount_vs_value_plot(recommendations: pd.DataFrame,
                           save_dir: str | None = None):
    """Where the engine actually spends: discount depth vs expected campaign value."""
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = recommendations["recommended_reward"].unique()
    for i, lab in enumerate(labels):
        sub = recommendations[recommendations["recommended_reward"] == lab]
        ax.scatter(sub["discount_pct"] + sub["discount_flat"] / 20.0,
                   sub["expected_campaign_value"],
                   s=26, alpha=0.65, color=PALETTE[i % len(PALETTE)], label=lab)
    ax.set_xlabel("Offer depth index (discount % and rupee value combined)")
    ax.set_ylabel("Expected campaign value (\u20b9)")
    ax.set_title("Reward Optimisation \u2014 Bigger Discount Does Not Mean Better Value")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/discount_vs_value.png" if save_dir else None)


def campaign_simulation_chart(sim: dict, save_dir: str | None = None):
    """Three strategies side by side on the same budget."""
    s = sim["summary"].set_index("Strategy")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col in zip(axes, ["Expected Responders", "Expected Revenue", "Expected Net Value"]):
        bars = ax.bar(s.index, s[col], color=PALETTE[: len(s)])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, s[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:,.0f}" if col != "Expected Responders" else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
        ax.grid(axis="x", visible=False)

    fig.suptitle(f"Campaign Simulation \u2014 {sim['note']}", fontsize=12, y=1.03)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/campaign_simulation.png" if save_dir else None)


def category_response_rate_chart(features: pd.DataFrame,
                                 transactions: pd.DataFrame,
                                 products: pd.DataFrame,
                                 save_dir: str | None = None):
    """Reward response rate per preferred category, with revenue context."""
    tx = transactions.merge(products[["product_id", "category"]], on="product_id")
    rev = tx.groupby("category")["total_amount"].sum() / 1e7
    g = features.groupby("preferred_category").agg(
        customers=("customer_id", "size"),
        response_rate=("reward_probability", "mean"),
    )
    g["revenue_cr"] = g.index.map(rev).fillna(0.0)
    g = g.sort_values("response_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(g.index, g["response_rate"], color=PALETTE[1])
    ax.set_ylabel("Average reward response probability")
    ax.set_title("Reward Response Rate by Preferred Category")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="x", visible=False)
    for bar, v, n, r in zip(bars, g["response_rate"], g["customers"], g["revenue_cr"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.2f}\n{n} cust\n\u20b9{r:.2f}cr",
                ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/category_response_rate.png" if save_dir else None)


def discount_sensitivity_chart(features: pd.DataFrame, save_dir: str | None = None):
    """Average order value vs average discount accepted -- the price-sensitivity profile."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    groups = pd.cut(features["price_sensitivity_index"], bins=5,
                    labels=["Very low", "Low", "Medium", "High", "Very high"])
    g = features.groupby(groups, observed=True).agg(
        customers=("customer_id", "size"),
        avg_aov=("average_order_value", "mean"),
        avg_disc=("average_discount_percentage", "mean"),
        response=("reward_probability", "mean"),
    )
    bars = ax.bar(g.index.astype(str), g["avg_aov"], color=PALETTE[0], label="Average order value")
    ax.set_ylabel("Average order value (\u20b9)", color=PALETTE[0])
    ax2 = ax.twinx()
    ax2.plot(g.index.astype(str), g["avg_disc"], marker="o", color=PALETTE[4], lw=2,
             label="Average discount accepted (%)")
    ax2.set_ylabel("Average discount accepted (%)", color=PALETTE[4])
    ax2.grid(False)
    ax.set_title("Discount Sensitivity Profile (by price-sensitivity index)")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/discount_sensitivity.png" if save_dir else None)


# ============================================================================
# Architecture diagram
# ============================================================================
def architecture_diagram(save_dir: str | None = None):
    """System architecture rendered as a labelled flow diagram."""
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)

    def box(x, y, w, h, text, fc, ec, fs=10, bold=False):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                             linewidth=1.6, zorder=3, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=4,
                color="#10202e")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#455a64", lw=1.8))
        if label:
            ax.text((x1 + x2) / 2 + 0.16, (y1 + y2) / 2, label, fontsize=8,
                    color="#37474f", va="center")

    # top layer
    box(0.6, 13.3, 8.8, 1.15, "RAW DATA  \u2014  customers.csv  \u00b7  products.csv  \u00b7  transactions.csv",
        "#d6e4f0", "#1f4e79", fs=11, bold=True)
    arrow(5, 13.3, 5, 12.55)
    box(1.8, 11.55, 6.4, 1.0, "DATA VALIDATION & LOADING", "#e8f1f8", "#2e86ab", bold=True)
    arrow(5, 11.55, 5, 10.7)
    box(0.9, 9.7, 8.2, 1.0, "FEATURE ENGINEERING  \u2014  50 features, observation window only",
        "#e8f1f8", "#2e86ab", bold=True)
    arrow(5, 9.7, 5, 8.95)

    # parallel tracks
    box(0.6, 7.95, 4.0, 1.0, "RFM ANALYSIS\nR / F / M + segments", "#e6f5f2", "#17b3a3", bold=True)
    box(5.4, 7.95, 4.0, 1.0, "PRODUCT AFFINITY\nper-category behaviour", "#e6f5f2", "#17b3a3", bold=True)
    arrow(2.6, 9.7, 2.6, 8.95)
    arrow(7.4, 9.7, 7.4, 8.95)
    arrow(2.6, 7.95, 2.6, 7.2)
    arrow(7.4, 7.95, 7.4, 7.2)

    box(1.2, 6.2, 3.4, 1.0, "K-MEANS\ncluster profiling", "#e6f5f2", "#17b3a3", bold=True)
    box(5.4, 6.2, 4.0, 1.0, "CUSTOMER ML DATASET\n1 row per customer", "#e8f1f8", "#2e86ab", bold=True)
    arrow(2.6, 7.2, 2.9, 7.2)
    arrow(7.4, 7.2, 7.4, 6.2 + 1.0)

    # model zoo
    arrow(7.4, 6.2, 7.4, 5.5)
    ax.text(5, 5.28, "SUPERVISED MODELS", fontsize=9, ha="center", color="#37474f")
    models = [
        ("Logistic\nRegression", 0.4, "#d6e4f0"),
        ("Decision\nTree", 2.3, "#e6d9f2"),
        ("Random\nForest", 4.2, "#d6f0ec"),
        ("XGBoost", 6.1, "#f7e0c8"),
        ("CatBoost", 8.0, "#f2d9df"),
    ]
    for label, x, col in models:
        box(x, 4.35, 1.65, 0.95, label, col, "#455a64", fs=8.5, bold=True)
        arrow(7.4, 5.45, x + 0.82, 5.32)

    arrow(5, 4.35, 5, 3.7)
    box(1.6, 2.75, 6.8, 0.95, "MODEL COMPARISON  \u2192  BEST MODEL (F1 selected)", "#d6e4f0", "#1f4e79", bold=True)
    arrow(5, 2.75, 5, 2.0)
    box(1.0, 1.1, 8.0, 0.9, "CAMPAIGN OUTPUTS\nresponse score \u00b7 reward optimisation \u00b7 priority ranking",
        "#e8f1f8", "#2e86ab", bold=True)

    ax.set_title("System Architecture \u2014 AI-Powered Customer Intelligence &\nPersonalised Reward Recommendation",
                 fontsize=13, pad=14)
    fig.tight_layout()
    return _save(fig, f"{save_dir}/figures/architecture.png" if save_dir else None)


# ============================================================================
# HTML / Styler dashboards
# ============================================================================
def kpi_card_grid(kpis: dict[str, str | float]) -> str:
    """Build the KPI-card HTML used by the executive dashboard."""
    parts = ['<div style="display:flex;flex-wrap:wrap;gap:12px;font-family:Segoe UI,Arial,sans-serif">']
    colours = ["#1f4e79", "#2e86ab", "#17b3a3", "#f37748",
               "#f0a202", "#c94c63", "#7d3c98", "#34495e"]
    for i, (label, value) in enumerate(kpis.items()):
        c = colours[i % len(colours)]
        parts.append(
            f'<div style="flex:1 1 190px;background:{c};color:#fff;border-radius:12px;'
            f'padding:16px 14px;box-shadow:0 2px 8px rgba(0,0,0,.14)">'
            f'<div style="font-size:11px;letter-spacing:.6px;opacity:.9;text-transform:uppercase">'
            f'{label}</div>'
            f'<div style="font-size:24px;font-weight:700;margin-top:6px">{value}</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def customer_profile_card(row: pd.Series) -> str:
    """Polished HTML customer intelligence card."""
    def _money(v):
        return f"{CURRENCY}{float(v):,.0f}"

    def _band(p):
        if p >= 0.75:
            return ("HIGH RESPONSE", "#16a085")
        if p >= 0.45:
            return ("MODERATE RESPONSE", "#f0a202")
        return ("LOW RESPONSE", "#c94c63")

    label, colour = _band(float(row.get("reward_probability", 0)))

    def _sec(title, rows_html):
        return (
            f'<div style="background:#fff;border-radius:10px;padding:14px;'
            f'box-shadow:0 1px 5px rgba(0,0,0,.10);border-left:4px solid {colour};min-width:230px">'
            f'<div style="font-size:11px;letter-spacing:.8px;text-transform:uppercase;'
            f'color:#607d8b;margin-bottom:8px;font-weight:700">{title}</div>'
            f'<div style="font-size:13px;line-height:1.75">{rows_html}</div></div>'
        )

    def _line(k, v):
        return (f'<div style="display:flex;justify-content:space-between;gap:14px">'
                f'<span style="color:#607d8b">{k}</span><b>{v}</b></div>')

    html = f"""
    <div style="font-family:Segoe UI,Arial,sans-serif;background:#eef3f7;padding:18px;border-radius:14px">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <div style="font-size:20px;font-weight:700;color:#1f4e79">
          CUSTOMER INTELLIGENCE PROFILE
        </div>
        <div style="background:{colour};color:#fff;border-radius:20px;padding:7px 16px;
                    font-weight:700;letter-spacing:.5px">{label}</div>
      </div>
      <div style="margin-top:6px;color:#455a64;font-size:13px">
        Customer <b>{row.get('customer_id','')}</b> &nbsp;\u00b7&nbsp;
        {row.get('gender','')} &nbsp;\u00b7&nbsp; age {int(row.get('age',0))} &nbsp;\u00b7&nbsp;
        {row.get('city','')} &nbsp;\u00b7&nbsp; tenure {int(row.get('customer_tenure_days',0))} days
      </div>

      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:14px">
        {_sec('Purchase Behaviour', ''.join([
            _line('Orders', int(row.get('total_orders', 0))),
            _line('Total spend', _money(row.get('total_spend', 0))),
            _line('Average order value', _money(row.get('average_order_value', 0))),
            _line('Recency', f"{int(row.get('days_since_last_purchase',0))} days"),
            _line('Discount usage', f"{row.get('discount_usage_rate',0):.0%}"),
            _line('Reward usage', f"{row.get('reward_usage_rate',0):.0%}"),
        ]))}

        {_sec('RFM & Segment', ''.join([
            _line('R / F / M', f"{int(row.get('R_score',0))} / {int(row.get('F_score',0))} / {int(row.get('M_score',0))}"),
            _line('RFM score', f"{int(row.get('rfm_score',0))} / 15"),
            _line('Segment', row.get('rfm_segment','')),
        ]))}

        {_sec('Product Affinity', ''.join([
            _line('Preferred category', row.get('preferred_category','')),
            _line('Affinity score', f"{row.get('category_affinity_score',0):.2f}"),
            _line('Category entropy', f"{row.get('category_entropy',0):.2f}"),
        ]))}

        {_sec('ML Prediction', ''.join([
            _line('Reward probability', f"{row.get('reward_probability',0):.1%}"),
            _line('Targetable', 'Yes' if row.get('targetable') else 'No'),
        ]))}
      </div>

      <div style="margin-top:14px;background:#fff;border-radius:10px;padding:14px;
                  box-shadow:0 1px 5px rgba(0,0,0,.10);border-left:4px solid {colour}">
        <div style="font-size:11px;letter-spacing:.8px;text-transform:uppercase;
                    color:#607d8b;font-weight:700;margin-bottom:8px">Recommended Reward</div>
        <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:baseline">
          <div style="font-size:26px;font-weight:800;color:{colour}">
            {row.get('recommended_reward','\u2014')}
          </div>
          <div style="font-size:13px;color:#455a64">
            Category <b>{row.get('reward_category', row.get('preferred_category',''))}</b> &nbsp;\u00b7&nbsp;
            Min. purchase <b>{_money(row.get('minimum_purchase',0))}</b> &nbsp;\u00b7&nbsp;
            Max discount <b>{_money(row.get('maximum_discount',0))}</b>
          </div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:10px;font-size:12px;color:#455a64">
          <div>Estimated reward cost <b>{_money(row.get('estimated_reward_cost',0))}</b></div>
          <div>Expected campaign value <b>{_money(row.get('expected_campaign_value',0))}</b></div>
          <div>Campaign priority <b>{float(row.get('campaign_priority_score',0)):.1f}</b>
               (rank {int(row.get('campaign_rank',0))})</div>
        </div>
      </div>
    </div>
    """
    return html


def style_table(df: pd.DataFrame, subset: list[str] | None = None,
                format_map: dict[str, str] | None = None,
                highlight: str | None = None, max_rows: int | None = None) -> object:
    """Return a pandas Styler with professional number formatting and conditional colour."""
    view = df if max_rows is None else df.head(max_rows)
    styler = view.style
    if format_map:
        styler = styler.format(format_map)
    if highlight and highlight in view.columns:
        styler = styler.background_gradient(
            subset=[highlight], cmap="YlGnBu", vmin=view[highlight].min(),
            vmax=view[highlight].max()
        )
    return styler.set_properties(**{
        "font-size": "12px", "padding": "5px 8px",
        "text-align": "center", "border-bottom": "1px solid #eceff1",
    }).set_table_styles([{
        "selector": "th",
        "props": [("background-color", "#1f4e79"), ("color", "white"),
                  ("padding", "7px 9px"), ("font-size", "12px")]
    }])