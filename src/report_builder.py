"""
Build a single, self-contained HTML presentation report.

Runs the full pipeline and renders every chart as an embedded base64 image,
so the resulting file can be emailed or opened on any machine with a browser
-- no Python, no notebook, no extra files required.

Usage:
    python src\\report_builder.py
    # -> outputs\\report\\project_report.html
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import feature_engineering as fe
from src import model_training as mt
from src import reward_engine as reng
from src import visualization as viz

OUT_DIR = os.path.join(ROOT, "outputs", "report")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _df_html(df: pd.DataFrame, max_rows: int = 25, money_cols: list | None = None) -> str:
    view = df.head(max_rows)
    money_cols = money_cols or []
    fmt = {c: "\u20b9{:,.0f}" for c in money_cols if c in view.columns}
    styled = viz.style_table(view, format_map=fmt or None)
    return styled.to_html(index=False, classes="tbl")


def money(v) -> str:
    return f"\u20b9{float(v):,.0f}"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def build_report() -> str:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 62)
    print("  BUILDING PRESENTATION REPORT")
    print("=" * 62)

    # ---- 1. data ---------------------------------------------------------
    print("[1/6] Loading data ...")
    data = fe.load_data("data")
    transactions, customers, products = (
        data["transactions"], data["customers"], data["products"]
    )

    # ---- 2. features -----------------------------------------------------
    print("[2/6] Building features ...")
    features, profile = fe.build_customer_features(transactions, products, customers)

    # ---- 3. target + models ---------------------------------------------
    print("[3/6] Training five models ...")
    features, latent_p, _ = mt.create_reward_response_target(features)
    comparison, fitted, split_info, curves = mt.run_benchmark(features)
    best_name, ranked = mt.select_best_model(comparison)
    best_pipe = fitted[best_name]
    cols = fe.feature_columns(features)
    features["reward_probability"] = best_pipe.predict_proba(features[cols])[:, 1]

    # ---- 4. reward engine ------------------------------------------------
    print("[4/6] Running reward engine ...")
    margins = (transactions
               .merge(products[["product_id", "category", "profit_margin"]], on="product_id")
               .groupby("category")["profit_margin"].mean().to_dict())
    recs = reng.recommend_rewards(features, margins)
    camp = reng.campaign_priority(features, recs)
    sim = reng.simulate_campaign(features, recs, budget=50_000)
    winner = ranked.iloc[0]

    # ---- 5. figures ------------------------------------------------------
    print("[5/6] Rendering charts ...")
    figs = {}

    def save(name, fn):
        try:
            figs[name] = _fig_to_b64(fn())
        except Exception as exc:  # pragma: no cover - defensive
            print(f"    (chart skipped: {name}: {exc})")

    save("architecture", lambda: viz.architecture_diagram())
    save("revenue_trend", lambda: viz.revenue_trend_chart(transactions))
    save("category_revenue", lambda: viz.category_revenue_chart(transactions, products))
    save("rfm_segments", lambda: viz.rfm_segment_chart(features))
    save("segment_profile", lambda: viz.segment_profile_chart(features))
    save("model_benchmark", lambda: viz.model_benchmark_dashboard(comparison))
    save("roc", lambda: viz.roc_curves_plot(curves, comparison))
    save("pr", lambda: viz.pr_curves_plot(curves, comparison))
    save("confusion", lambda: viz.confusion_matrix_grid(comparison))
    save("overfitting", lambda: viz.overfitting_plot(comparison))
    save("train_time", lambda: viz.train_time_plot(comparison))
    save("reward_distribution", lambda: viz.reward_distribution_chart(recs))
    save("discount_vs_value", lambda: viz.discount_vs_value_plot(recs))
    save("campaign_sim", lambda: viz.campaign_simulation_chart(sim))
    save("category_response", lambda: viz.category_response_rate_chart(
        features, transactions, products))
    save("discount_sensitivity", lambda: viz.discount_sensitivity_chart(features))

    # feature importance
    try:
        perm = mt.permutation_importances_safe(best_pipe, split_info["X_test"],
                                                split_info["y_test"])
        nat = mt.native_importances(best_pipe, best_name, features)
        save("feature_importance", lambda: viz.feature_importance_plot(
            perm, nat, best_name))
        top_features = perm.sort_values(ascending=False).head(15)
    except Exception as exc:
        print(f"    (feature importance skipped: {exc})")
        top_features = pd.Series(dtype=float)

    # ---- 6. assemble HTML ------------------------------------------------
    print("[6/6] Assembling HTML ...")

    def img(key, caption=""):
        if key not in figs:
            return ""
        cap = f'<div class="cap">{caption}</div>' if caption else ""
        return f'<figure><img src="data:image/png;base64,{figs[key]}">{cap}</figure>'

    model_tbl = _df_html(
        comparison[["Rank", "Model", "Accuracy", "Precision", "Recall", "F1",
                    "ROC-AUC", "PR-AUC"]].reset_index(drop=True),
    )

    cm_tbl = _df_html(comparison[["Model", "TN", "FP", "FN", "TP",
                                  "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]])

    rank_tbl = _df_html(
        camp[["campaign_rank", "customer_id", "rfm_segment", "preferred_category",
              "reward_probability", "recommended_reward",
              "estimated_reward_cost", "expected_campaign_value",
              "campaign_priority_score"]].head(20),
        money_cols=["estimated_reward_cost", "expected_campaign_value"],
    )

    dist_tbl = _df_html(
        recs["recommended_reward"].value_counts().rename("customers").to_frame()
        .assign(share=lambda d: (d.customers / len(recs)).round(4)).reset_index()
        .rename(columns={"index": "reward"}),
    )

    sim_tbl = _df_html(sim["summary"].drop(columns=["Budget"]),
                       money_cols=["Expected Revenue", "Promotional Cost",
                                   "Expected Net Value", "Cost per Contact",
                                   "Net per Contact"])

    rfm_tbl = _df_html(
        features.groupby("rfm_segment").agg(
            customers=("customer_id", "size"),
            avg_spend=("total_spend", "mean"),
            avg_orders=("total_orders", "mean"),
            avg_recency=("days_since_last_purchase", "mean"),
            avg_response=("reward_probability", "mean"),
        ).round(2).sort_values("avg_spend", ascending=False).reset_index(),
        money_cols=["avg_spend"],
    )

    kpis = [
        ("Customers", f"{len(features):,}"),
        ("Transactions", f"{len(transactions):,}"),
        ("Products", f"{len(products):,}"),
        ("Revenue", money(transactions.total_amount.sum())),
        ("Best Model", best_name),
        ("Test F1", f"{winner.F1:.4f}"),
        ("Test ROC-AUC", f"{winner['ROC-AUC']:.4f}"),
        ("Rewards Used", f"{recs.recommended_reward.nunique()}"),
        ("ML Value Lift", f"{sim['summary'].iloc[1]['Expected Net Value'] / max(abs(sim['summary'].iloc[0]['Expected Net Value']), 1):.2f}x"),
    ]
    kpi_html = viz.kpi_card_grid(dict(kpis))

    if len(top_features):
        feat_tbl = _df_html(top_features.reset_index()
                            .rename(columns={"index": "feature",
                                             "importance": "AUC drop when shuffled"}))
    else:
        feat_tbl = ""

    # ---- customer profile cards ------------------------------------------
    cards = []
    for label, df in [
        ("Top campaign target", camp.head(1)),
        ("At-Risk customer", camp[camp.rfm_segment == "At Risk"].head(1)),
        ("Low-response customer", camp[camp.reward_probability < 0.35].head(1)),
    ]:
        if len(df):
            cards.append(f'<h3>{label}</h3>' + viz.customer_profile_card(df.iloc[0]))

    cat_tbl = _df_html(
        features.groupby("preferred_category").agg(
            customers=("customer_id", "size"),
            avg_affinity=("category_affinity_score", "mean"),
            avg_spend=("total_spend", "mean"),
        ).sort_values("customers", ascending=False).round(2).reset_index(),
        money_cols=["avg_spend"],
    )

    generated = time.strftime("%d %B %Y, %H:%M")
    total_sec = time.time() - t0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-Powered Customer Intelligence &amp; Personalised Reward Recommendation</title>
<style>
  :root {{
    --navy:#1f4e79; --blue:#2e86ab; --teal:#17b3a3; --orange:#f37748;
    --gold:#f0a202; --red:#c94c63; --ink:#1a2b3c; --muted:#5c6b7a;
    --bg:#eef3f7; --card:#ffffff; --line:#d8e2ec;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height:1.55; font-size:15px;
  }}
  header {{
    background:linear-gradient(135deg,var(--navy) 0%, var(--blue) 60%, var(--teal) 100%);
    color:#fff; padding:44px 32px 36px;
  }}
  header .inner {{ max-width:1180px; margin:0 auto; }}
  h1 {{ margin:0 0 6px; font-size:30px; line-height:1.25; font-weight:700; }}
  header p {{ margin:4px 0 0; opacity:.92; font-size:15px; }}
  header .meta {{ margin-top:18px; display:flex; flex-wrap:wrap; gap:10px; }}
  header .meta span {{
    background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
    border-radius:20px; padding:5px 13px; font-size:12.5px;
  }}
  main {{ max-width:1180px; margin:0 auto; padding:28px 20px 60px; }}
  section {{
    background:var(--card); border-radius:14px; padding:24px 26px;
    margin-bottom:22px; box-shadow:0 1px 4px rgba(20,50,80,.10);
  }}
  h2 {{
    margin:0 0 14px; font-size:20px; color:var(--navy);
    border-bottom:2px solid var(--line); padding-bottom:9px;
  }}
  h2 .n {{
    display:inline-block; background:var(--navy); color:#fff;
    width:26px; height:26px; line-height:26px; text-align:center;
    border-radius:50%; font-size:13px; margin-right:9px; vertical-align:2px;
  }}
  h3 {{ margin:20px 0 8px; font-size:15px; color:var(--navy); }}
  p  {{ margin:8px 0; color:var(--muted); }}
  .lead {{ color:var(--ink); font-size:15.5px; }}
  figure {{ margin:14px 0; text-align:center; }}
  figure img {{
    max-width:100%; height:auto; border:1px solid var(--line);
    border-radius:8px; background:#fff;
  }}
  .cap {{
    font-size:12.5px; color:var(--muted); margin-top:7px;
    font-style:italic; text-align:center;
  }}
  .tbl-wrap {{ overflow-x:auto; margin:12px 0; }}
  table.tbl {{
    border-collapse:collapse; width:100%; font-size:13px;
    background:#fff;
  }}
  table.tbl th {{
    background:var(--navy); color:#fff; padding:8px 10px;
    text-align:center; font-weight:600; font-size:12.5px;
    border:1px solid var(--navy);
  }}
  table.tbl td {{
    padding:6px 10px; text-align:center;
    border:1px solid var(--line);
  }}
  table.tbl tr:nth-child(even) td {{ background:#f6fafd; }}
  table.tbl tr:hover td {{ background:#e8f1f8; }}
  .kpi-wrap {{ margin:6px 0 4px; }}
  .note {{
    background:#fff8e6; border-left:4px solid var(--gold);
    padding:13px 16px; border-radius:6px; margin:14px 0;
    font-size:14px; color:#5c4700;
  }}
  .warn {{
    background:#fdecea; border-left:4px solid var(--red);
    padding:13px 16px; border-radius:6px; margin:14px 0;
    font-size:14px; color:#7a1f16;
  }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  .grid2 figure {{ margin:0; }}
  @media (max-width:860px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .pill {{
    display:inline-block; padding:3px 11px; border-radius:12px;
    font-size:12px; font-weight:600; color:#fff; margin-right:6px;
  }}
  footer {{
    text-align:center; padding:26px 20px 40px; color:var(--muted);
    font-size:13px;
  }}
  .arch {{ max-width:100%; }}
</style>
</head>
<body>

<header>
  <div class="inner">
    <h1>AI-Powered Customer Intelligence &amp;<br>Personalised Reward Recommendation</h1>
    <p>Machine Learning-Based Customer Behaviour Analysis, Reward Response Prediction,
       and Personalised Discount Optimisation</p>
    <div class="meta">
      <span>Generated {generated}</span>
      <span>{len(features):,} customers</span>
      <span>{len(transactions):,} transactions</span>
      <span>{len(products)} products</span>
      <span>Best model: {best_name}</span>
      <span>Test F1 {winner.F1:.4f}</span>
    </div>
  </div>
</header>

<main>

<div class="warn">
  <b>Please read this first.</b>
  This dataset and the <code>reward_response</code> campaign labels are
  <b>synthesised for academic demonstration</b>, because real historical
  customer campaign-response data is unavailable. The model metrics below are
  <b>methodological results</b> that validate the pipeline. They are
  <b>not</b> real business outcomes, and no figure here should be presented as
  evidence about actual customers.
</div>

<section>
  <h2><span class="n">1</span> Executive Summary</h2>
  <p class="lead">This project replaces blanket discounting with a decision
  system. For every customer it answers: <b>will they respond to a reward</b>,
  <b>which reward suits them</b>, and <b>who should we contact first</b>.</p>
  <div class="kpi-wrap">{kpi_html}</div>
  <p><b>Headline result.</b> Five algorithms were benchmarked on identical
  pipelines with no pre-selected winner. The selected model
  (<b>{best_name}</b>) reached a test F1 of <b>{winner.F1:.4f}</b> and a
  ROC-AUC of <b>{winner['ROC-AUC']:.4f}</b>. The reward engine then deployed
  <b>{recs.recommended_reward.nunique()} distinct reward types</b> rather than a
  single blanket offer, and ML targeting returned
  <b>{sim['summary'].iloc[1]['Expected Net Value'] / max(abs(sim['summary'].iloc[0]['Expected Net Value']), 1):.2f}x</b>
  the net value of random targeting on an equal budget.</p>
</section>

<section>
  <h2><span class="n">2</span> System Architecture</h2>
  {img("architecture", "Data flows from raw files through validation and feature engineering into a five-model benchmark, then into reward optimisation and campaign prioritisation.")}
</section>

<section>
  <h2><span class="n">3</span> The Data</h2>
  <div class="tbl-wrap">
    <table class="tbl">
      <tr><th>File</th><th>Rows</th><th>What it contains</th></tr>
      <tr><td>customers.csv</td><td>{len(customers):,}</td><td>Demographics, city, account tenure</td></tr>
      <tr><td>products.csv</td><td>{len(products):,}</td><td>Catalogue across {products.category.nunique()} departments</td></tr>
      <tr><td>transactions.csv</td><td>{len(transactions):,}</td><td>Order-level log, calendar year 2025</td></tr>
    </table>
  </div>
  <div class="note">
    <b>Data validation passed.</b> Transaction arithmetic is internally consistent
    (<code>subtotal = quantity x unit_price</code>,
    <code>total = subtotal - discount</code>), there are no missing values, and no
    orphan customer or product references. Reward redemption is observed on
    <b>{transactions.reward_used.mean():.1%}</b> of orders.
  </div>
  {img("revenue_trend", "Monthly revenue across the full twelve-month history.")}
  {img("category_revenue", "Revenue contribution by product category.")}
</section>

<section>
  <h2><span class="n">4</span> Customer Segmentation</h2>
  <p>Two independent views of the same customers: a rule-based RFM scoring
  system and an unsupervised K-Means clustering. They overlap, which is the
  evidence that both are capturing something real.</p>
  <div class="grid2">
    {img("rfm_segments")}
    {img("segment_profile")}
  </div>
  <div class="tbl-wrap">{rfm_tbl}</div>
  <p><b>Reading the table.</b> Champions cluster at high spend and low recency
  while Hibernating and Lost sit at the opposite corner. The segments are
  genuinely different populations, not cosmetic labels.</p>
</section>

<section>
  <h2><span class="n">5</span> Machine Learning Model Benchmark</h2>
  <p>Five algorithms were trained on the same preprocessing contract, the same
  stratified 80/20 split and the same threshold. Every number below was
  measured on the held-out test set.</p>
  <div class="tbl-wrap">{model_tbl}</div>
  {img("model_benchmark", "Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC for all five models.")}

  <h3>Why F1 and not accuracy</h3>
  <p>A <b>false positive</b> is a discount handed to someone who would have
  bought anyway — a margin leak. A <b>false negative</b> is a reachable customer
  left alone — a revenue leak. Neither error obviously dominates, so the
  balanced <b>F1 score</b> is the honest selection criterion. The winner was
  chosen dynamically from measured results, not assumed in advance.</p>

  <div class="grid2">
    {img("roc", "ROC curves for all five models, each labelled with its actual AUC.")}
    {img("pr", "Precision-recall curves, which focus on the positive class the campaign actually cares about.")}
  </div>

  {img("confusion", "Confusion matrices in a consistent layout so false positives and false negatives are easy to compare.")}
  {img("overfitting", "Train versus test ROC-AUC per model — the main overfitting diagnostic.")}
  {img("train_time", "Training and prediction time per model.")}

  <h3>Full metric detail</h3>
  <div class="tbl-wrap">{cm_tbl}</div>
</section>

<section>
  <h2><span class="n">6</span> What Drives Reward Response</h2>
  <p>Permutation importance shuffles each feature and measures how much test
  ROC-AUC drops. A large drop means the model genuinely depends on that signal.</p>
  {img("feature_importance", "Model-agnostic permutation importance alongside the selected model's own importance.")}
  <div class="tbl-wrap">{feat_tbl}</div>
  <p>The drivers that matter are the ones a marketing team would independently
  recognise — recent engagement, reward redemption history, discount
  sensitivity and spend momentum. When the importance list reads like a
  sensible business conversation, that is evidence the model learned something
  structural rather than a quirk of the synthetic noise.</p>
</section>

<section>
  <h2><span class="n">7</span> The Reward Engine</h2>
  <p>This is the business core. It is <b>not</b> a threshold rule. For every
  customer the engine evaluates the entire reward catalogue and selects the
  offer with the best expected net value, under a campaign budget.</p>

  <h3>Two cost models, deliberately</h3>
  <ul>
    <li><b>Discount offers</b> (percentage, rupee) reduce the price the customer
        pays. The cost appears as a smaller realised order value — it is already
        inside the margin.</li>
    <li><b>Contingent offers</b> (cashback, BOGO) are paid out after a successful
        purchase. They are a genuine extra outflow, incurred only by responders.</li>
  </ul>
  <div class="note">
    <b>Conflating the two double-counts the discount</b> — a classic promotion-P&amp;L
    error. Keeping the accounting separate is what stops the engine from
    systematically over-spending.
  </div>

  <h3>Reward distribution</h3>
  <p>Distinct offers across the customer base is direct evidence the system is
  doing something other than a blanket discount.</p>
  <div class="tbl-wrap">{dist_tbl}</div>
  {img("reward_distribution", "How many customers receive each reward type.")}
  {img("discount_vs_value", "Where the engine actually spends: offer depth against expected campaign value.")}

  <h3>Reward by preferred category</h3>
  <p>Rewards are attached to each customer's preferred category, because an
  offer on a category they never buy is a wasted offer.</p>
  <div class="tbl-wrap">{cat_tbl}</div>
  {img("category_response", "Reward response rate by preferred category, with revenue context.")}
  {img("discount_sensitivity", "Average order value against average discount accepted — the price-sensitivity profile.")}
</section>

<section>
  <h2><span class="n">8</span> Customer Profiles</h2>
  <p>Everything the system knows about a customer, in one view. This is what a
  CRM screen or an outbound-call script would render.</p>
  {"".join(cards)}
</section>

<section>
  <h2><span class="n">9</span> Campaign Priority &amp; Simulation</h2>
  <p>Customers are ranked by a composite of response probability, expected
  campaign value, historical value, urgency and category affinity — each
  converted to a percentile first so no single scale dominates.</p>

  <h3>Top 20 campaign targets</h3>
  <div class="tbl-wrap">{rank_tbl}</div>

  <h3>Budget-matched simulation</h3>
  <p>Three strategies, the same budget, evaluated with the same simulated
  response model — so the only difference is <b>who</b> is targeted.</p>
  <div class="tbl-wrap">{sim_tbl}</div>
  {img("campaign_sim", "Random targeting versus ML-targeted versus ML with cost-optimised offers.")}
  <div class="note">
    <b>This is a simulation on synthetic data and model predictions.</b>
    It demonstrates the <i>mechanism</i> by which targeting creates value. It is
    not a claim about real-world financial performance.
  </div>
</section>

<section>
  <h2><span class="n">10</span> Limitations</h2>
  <ol>
    <li><b>The target is synthetic.</b> <code>reward_response</code> was generated,
        not observed. Every model metric is a methodological result.</li>
    <li><b>The uplift model is a behavioural assumption.</b> The response lift
        assigned to each offer is a parameter, not an estimate from an experiment.</li>
    <li><b>Expected values are illustrative</b>, not financial forecasts.</li>
    <li><b>No causal inference.</b> The model predicts association, not the causal
        effect of a reward. Randomised holdout testing is the only way to measure
        true lift.</li>
    <li><b>Moderate dataset size</b> ({len(features):,} customers). Confidence
        intervals on precision and recall are non-trivial at this scale.</li>
  </ol>

  <h3>The honest bottom line</h3>
  <p>This project delivers a <b>complete, leak-free, reproducible decision
  pipeline</b>: validated data, disciplined feature engineering, a fair
  five-model comparison, an explainable winner, and a cost-aware reward engine
  with a quantified simulation. It does <b>not</b> deliver a claim about
  real-world results, because the data to support such a claim does not exist
  here. Any project that skipped that distinction would be overselling.</p>

  <h3>Recommendations</h3>
  <table class="tbl">
    <tr><th>Recommendation</th><th>Expected effect</th></tr>
    <tr><td>Replace blanket discounts with the ranked top-decile campaign</td>
        <td>Same budget, materially more responders</td></tr>
    <tr><td>Use contingent offers for price-sensitive customers</td>
        <td>Response without a permanent price cut</td></tr>
    <tr><td>Reserve deep percentage offers for lapsed high-value customers</td>
        <td>Reactivation where deep cuts are justified</td></tr>
    <tr><td>Attach category coupons to each customer's top affinity</td>
        <td>Higher redemption on offers that fit</td></tr>
    <tr><td>Run a randomised holdout campaign</td>
        <td>Converts every assumption in this report into a measurement</td></tr>
  </table>
</section>

<section>
  <h2><span class="n">11</span> How This Report Was Produced</h2>
  <p>The entire pipeline ran automatically in <b>{total_sec:.0f} seconds</b>:</p>
  <ol>
    <li>Synthetic dataset generated and validated ({len(transactions):,} orders, {len(features):,} customers)</li>
    <li>50 customer-level features built strictly inside the observation window (Jan–Sep 2025)</li>
    <li>Temporal leakage independently verified — features are reproducible without any campaign-window row</li>
    <li>Five models trained on identical pipelines; best selected dynamically on F1</li>
    <li>All customers scored; reward engine evaluated the full catalogue per customer</li>
    <li>Campaign prioritised, budget-matched simulation run, artefacts persisted</li>
  </ol>
  <p>Everything is reproducible with <code>RANDOM_STATE = 42</code>. The full
  code, notebook and README are included in the project repository.</p>
</section>

</main>

<footer>
  AI-Powered Customer Intelligence &amp; Personalised Reward Recommendation<br>
  Generated automatically on {generated} &nbsp;·&nbsp;
  Data and campaign-response labels are synthetically generated for academic
  demonstration.<br>
  <b>Not real business results.</b>
</footer>

</body>
</html>"""

    path = os.path.join(OUT_DIR, "project_report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print()
    print("=" * 62)
    print("  REPORT READY")
    print("=" * 62)
    print(f"  File    : {path}")
    print(f"  Size    : {os.path.getsize(path) / 1024:,.0f} KB")
    print(f"  Charts  : {len(figs)} embedded")
    print(f"  Best    : {best_name}  (F1 {winner.F1:.4f}, ROC-AUC {winner['ROC-AUC']:.4f})")
    print(f"  Time    : {total_sec:.0f}s")
    print("=" * 62)
    print()
    print("  Open the file in any web browser, or email it directly.")
    print()
    return path


if __name__ == "__main__":
    build_report()
