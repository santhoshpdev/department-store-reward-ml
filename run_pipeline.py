"""
Run the full ML pipeline and print the results.

Called automatically by run_everything.bat, or run directly:

    python run_pipeline.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src import feature_engineering as fe
from src import model_training as mt
from src import reward_engine as reng


def main():
    t0 = time.time()
    print("=" * 62)
    print("  RUNNING THE ML PIPELINE")
    print("=" * 62)

    # ---- 1. load ---------------------------------------------------------
    data = fe.load_data("data")
    transactions = data["transactions"]
    print("\n[1/6] Loaded CSV files")
    print(f"      transactions : {transactions.shape}")
    print(f"      customers    : {data['customers'].shape}")
    print(f"      products     : {data['products'].shape}")

    # ---- 2. features -----------------------------------------------------
    print("\n[2/6] Building customer features (observation window only)...")
    features, profile = fe.build_customer_features(transactions, data["products"],
                                                    data["customers"])
    cols = fe.feature_columns(features)
    print(f"      feature frame  : {features.shape[0]:,} customers x {features.shape[1]} columns")
    print(f"      model features : {len(cols)}")
    print(f"      category table : {profile.shape[0]:,} rows")

    # ---- 3. leakage check ------------------------------------------------
    print("\n[3/6] Verifying there is no temporal data leakage...")
    campaign_rows = transactions[transactions.transaction_date >= fe.CAMPAIGN_START]
    recheck, _ = fe.build_customer_features(transactions, data["products"], data["customers"])
    same = bool((recheck.total_spend.values == features.total_spend.values).all())
    print(f"      campaign-window rows excluded : {len(campaign_rows):,}")
    print(f"      features identical without them : {same}")
    if not same:
        raise RuntimeError("LEAKAGE DETECTED - feature frame is not window-safe")

    # ---- 4. target + models ---------------------------------------------
    print("\n[4/6] Creating the target and training five models...")
    features, latent, _ = mt.create_reward_response_target(features)
    print(f"      response rate : {features.reward_response.mean():.1%}")
    comparison, fitted, split_info, curves = mt.run_benchmark(features)
    best_name, ranked = mt.select_best_model(comparison)
    best_pipe = fitted[best_name]
    winner = ranked.iloc[0]

    show = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print()
    print(comparison[show].round(4).to_string(index=False))
    print()
    print(f"      Best model (selected by F1) : {best_name}")
    print(f"      F1 score                    : {winner.F1:.4f}")
    print(f"      ROC-AUC                     : {winner['ROC-AUC']:.4f}")
    print(f"      PR-AUC                      : {winner['PR-AUC']:.4f}")

    # ---- 5. reward engine -------------------------------------------------
    print("\n[5/6] Scoring customers and running the reward engine...")
    features["reward_probability"] = best_pipe.predict_proba(features[cols])[:, 1]
    margins = (transactions
               .merge(data["products"][["product_id", "category", "profit_margin"]], on="product_id")
               .groupby("category")["profit_margin"].mean().to_dict())

    recs = reng.recommend_rewards(features, margins)
    camp = reng.campaign_priority(features, recs)
    sim = reng.simulate_campaign(features, recs, budget=50_000)

    print(f"      customers scored        : {len(features):,}")
    print(f"      targetable customers    : {int(recs.targetable.sum()):,}")
    print(f"      distinct rewards used   : {recs.recommended_reward.nunique()}")
    print(f"      expected reward cost    : {money(recs.loc[recs.targetable, 'estimated_reward_cost'].sum())}")
    print(f"      expected campaign value : {money(recs.loc[recs.targetable, 'expected_campaign_value'].sum())}")
    print()
    print("      REWARD DISTRIBUTION:")
    for name, n in recs.recommended_reward.value_counts().items():
        print(f"        {name:22s} {n:>5,}  ({n / len(recs):>5.1%})")
    print()
    print("      CAMPAIGN SIMULATION (Rs 50,000 budget):")
    s = sim["summary"]
    for _, r in s.iterrows():
        print(f"        {r['Strategy']:28s} {r['Customers Selected']:>5,} customers, "
              f"{r['Expected Responders']:>6.0f} responders, "
              f"net {money(r['Expected Net Value'])}")
    print()
    print("      TOP 5 CAMPAIGN TARGETS:")
    print(camp.head(5)[["campaign_rank", "customer_id", "rfm_segment",
                        "preferred_category", "reward_probability",
                        "recommended_reward", "campaign_priority_score"]]
          .round(3).to_string(index=False))

    # ---- 6. persist -------------------------------------------------------
    print("\n[6/6] Saving model artefacts...")
    paths = mt.save_artifacts(best_name, best_pipe, split_info, comparison,
                              models_dir=os.path.join(ROOT, "models"))
    for label, p in paths.items():
        print(f"      {label:14s} -> {p}")

    print()
    print("=" * 62)
    print("  PIPELINE COMPLETE")
    print(f"  Time: {time.time() - t0:.1f} seconds")
    print("=" * 62)
    print()
    print("  Reminder: reward_response is simulated. These are methodological")
    print("  results that validate the pipeline, not real business outcomes.")
    print()
    print("  Next: build the presentation report with")
    print("        python src\\report_builder.py")
    print()


def money(v):
    return f"\u20b9{float(v):,.0f}"


if __name__ == "__main__":
    main()
