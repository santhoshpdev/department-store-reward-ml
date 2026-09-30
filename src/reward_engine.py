"""
Reward Decision Engine
======================

The business heart of the project.  Given a customer's ML response
probability plus their behavioural profile, this module decides *which* offer to
send, *how much* discount to give, and *who to contact first*.

It is deliberately not a threshold rule.  For every customer it evaluates the
whole reward catalogue against three competing objectives and picks the offer
with the best expected net value:

    maximise   expected incremental profit
    subject to a discount budget the campaign can actually afford

That trade-off is what stops the engine from handing the biggest discount to
everybody.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generator import RUPEE

# ============================================================================
# Reward catalogue
# ============================================================================
# Each offer declares HOW it is paid for, because the two models need different
# accounting:
#   "discount"   -> the reward reduces the amount the customer pays.  The cost
#                   shows up as a reduced realised order value, NOT as a
#                   separate cash outflow.  Subtracting it twice would be a
#                   classic promotion-P&L error.
#   "contingent" -> cashback / BOGO are credited *after* a successful
#                   purchase, so they are a genuine extra outflow and are only
#                   incurred by people who actually respond.
REWARD_CATALOGUE: list[dict] = [
    # --- percentage style: scales with basket, suits high-value customers ---
    {"code": "PCT_05", "reward_type": "Percentage Discount", "reward_label": "5% OFF",
     "discount_pct": 5.0, "discount_flat": 0.0, "minimum_purchase": 0.0,
     "max_discount": 150.0, "min_purchase_rule": 0.0, "cost_model": "discount",
     "uplift": 0.06, "segment_affinity": ["Champions", "Loyal Customers", "Cannot Lose Them"]},

    {"code": "PCT_10", "reward_type": "Category Discount", "reward_label": "10% OFF",
     "discount_pct": 10.0, "discount_flat": 0.0, "minimum_purchase": 0.0,
     "max_discount": 400.0, "min_purchase_rule": 0.0, "cost_model": "discount",
     "uplift": 0.13, "segment_affinity": ["Loyal Customers", "Champions"]},

    {"code": "PCT_15", "reward_type": "Percentage Discount", "reward_label": "15% OFF",
     "discount_pct": 15.0, "discount_flat": 0.0, "minimum_purchase": 999.0,
     "max_discount": 750.0, "min_purchase_rule": 0.8, "cost_model": "discount",
     "uplift": 0.22, "segment_affinity": ["At Risk", "Needs Attention", "Hibernating"]},

    # --- fixed-rupee style: predictable cost, suits small baskets ----------
    {"code": "FLAT_100", "reward_type": "Fixed Discount", "reward_label": "Rs 100 OFF",
     "discount_pct": 0.0, "discount_flat": 100.0, "minimum_purchase": 400.0,
     "max_discount": 100.0, "min_purchase_rule": 0.35, "cost_model": "discount",
     "uplift": 0.09, "segment_affinity": ["Regular", "Potential Loyalists"]},

    {"code": "FLAT_200", "reward_type": "Fixed Discount", "reward_label": "Rs 200 OFF",
     "discount_pct": 0.0, "discount_flat": 200.0, "minimum_purchase": 799.0,
     "max_discount": 200.0, "min_purchase_rule": 0.45, "cost_model": "discount",
     "uplift": 0.15, "segment_affinity": ["Champions", "Loyal Customers"]},

    {"code": "FLAT_500", "reward_type": "Fixed Discount", "reward_label": "Rs 500 OFF",
     "discount_pct": 0.0, "discount_flat": 500.0, "minimum_purchase": 2499.0,
     "max_discount": 500.0, "min_purchase_rule": 0.55, "cost_model": "discount",
     "uplift": 0.24, "segment_affinity": ["Champions", "Cannot Lose Them"]},

    # --- value-back style: cheap contingent cost ---------------------------
    {"code": "CASHBACK", "reward_type": "Cashback", "reward_label": "Cashback",
     "discount_pct": 0.0, "discount_flat": 150.0, "minimum_purchase": 599.0,
     "max_discount": 150.0, "min_purchase_rule": 0.30, "cost_model": "contingent",
     "cost_weight": 0.35,
     "uplift": 0.11, "segment_affinity": ["Regular", "Potential Loyalists", "Promising"]},

    # --- engagement style: zero discount, pure retention play ---------------
    {"code": "BOGO", "reward_type": "BOGO", "reward_label": "Buy One Get One",
     "discount_pct": 0.0, "discount_flat": 0.0, "minimum_purchase": 699.0,
     "max_discount": 180.0, "min_purchase_rule": 0.40, "cost_model": "contingent",
     "cost_weight": 0.55,
     "uplift": 0.18, "segment_affinity": ["Price Sensitive", "Regular", "Occasional"]},
]

# Flat fallback value of a BOGO offer, as a share of the basket
BOGO_COST_SHARE = 0.14

# Categories where a BOGO makes commercial sense
BOGO_ELIGIBLE_CATEGORIES = {
    "Snacks & Beverages", "Groceries", "Stationery",
    "Personal Care", "Household Cleaning", "Health & Wellness",
}


def reward_catalogue_frame() -> pd.DataFrame:
    """The catalogue as a DataFrame for display in the notebook."""
    df = pd.DataFrame(REWARD_CATALOGUE).rename(
        columns={"reward_label": "Recommended Reward", "discount_pct": "Discount %",
                 "discount_flat": "Flat Amount", "minimum_purchase": "Minimum Purchase",
                 "max_discount": "Maximum Discount"}
    )
    df = df[["Recommended Reward", "reward_type", "Discount %", "Flat Amount",
             "Minimum Purchase", "Maximum Discount", "uplift", "cost_weight"]]
    df.columns = ["Recommended Reward", "Reward Type", "Discount %", "Flat Amount",
                  "Minimum Purchase", "Maximum Discount", "Expected Uplift", "Cost Weight"]
    return df


# ============================================================================
# Offer evaluation
# ============================================================================
def expected_response_probability(
    base_probability: float,
    offer: dict,
    affinity_score: float,
    discount_sensitivity: float,
    reward_usage_rate: float,
) -> float:
    """How much this specific offer lifts a specific customer's response.

    The uplift is *modulated*, never added blindly:
      - customers already saturated on rewards gain little,
      - discount-led customers respond most to value offers,
      - affinity determines whether the category actually matters to them.
    """
    affinity = float(np.clip(affinity_score, 0, 1))
    sensitivity = float(np.clip(discount_sensitivity, 0, 1))
    usage = float(np.clip(reward_usage_rate, 0, 1))

    # diminishing returns: someone redeeming every order is hard to move further
    saturation = 1.0 - 0.45 * usage

    # a discount-led customer responds to the rupee value of the offer
    value_strength = offer["discount_pct"] / 15.0 + offer["discount_flat"] / 500.0
    value_strength = float(np.clip(value_strength, 0, 1))
    discount_fit = 0.55 + 0.75 * sensitivity * value_strength

    # category affinity decides whether *this* category is worth rewarding
    category_fit = 0.70 + 0.45 * affinity

    p = base_probability + offer["uplift"] * saturation * discount_fit * category_fit
    return float(np.clip(p, 0.0, 0.97))


def offer_cost(
    offer: dict,
    order_value: float,
    reward_type_code: str,
    category: str,
) -> float:
    """Gross reward value of a contingent offer (cashback / BOGO).

    Returns the full-value figure; `cost_weight` scales it down afterwards to
    reflect partial redemption and the fact that these offers are only paid on
    a successful purchase.
    """
    pct = offer["discount_pct"]
    flat = offer["discount_flat"]

    if reward_type_code == "CASHBACK":
        return float(min(offer["max_discount"], order_value * 0.10, max(flat, 0.0)))

    if reward_type_code == "BOGO":
        return float(order_value * BOGO_COST_SHARE)

    return float(min(flat, order_value * 0.5)) if flat > 0 \
        else float(min(offer["max_discount"], order_value * pct / 100.0))


def reward_economics(
    offer: dict,
    order_value: float,
    profit_margin: float,
    response_probability: float,
) -> dict:
    """Correctly cost one offer against one customer.

    Returns realised order value, the gross reward value, the expected net
    campaign value and the expected promotional spend.

    Accounting
    ----------
    *discount* offers
        The discount reduces what the customer pays, so the revenue base is
        already smaller.  Net value = p x realised_value x margin.  Nothing is
        subtracted a second time.

    *contingent* offers (cashback, BOGO)
        These are extra cash paid out on a successful purchase, so they reduce
        the margin directly and are also only paid with probability p.
        Net value = p x (order_value x margin - gross_cost).

    Expected promotional spend is `p x gross_reward_value` in both cases,
    because that is what the finance team actually budgets for.
    """
    p = float(np.clip(response_probability, 0.0, 0.97))
    pct = offer["discount_pct"]
    flat = offer["discount_flat"]
    code = offer["code"]

    if offer.get("cost_model") == "contingent":
        gross = float(offer_cost(offer, order_value, code, "")) \
            * float(offer.get("cost_weight", 1.0))
        realised = float(order_value)
        net_if_convert = order_value * profit_margin - gross
    else:
        # realisation: cap the rupee cut-off by the basket, never let the
        # realised value go negative
        realised = float(max(0.0, order_value * (1 - pct / 100.0) - flat))
        gross = float(min(order_value - realised, offer["max_discount"]))
        net_if_convert = realised * profit_margin

    expected_cost = p * gross
    return {
        "realised_order_value": realised,
        "gross_reward_value": gross,
        "net_if_convert": float(net_if_convert),
        "expected_reward_cost": expected_cost,
        "expected_campaign_value": float(p * net_if_convert),
    }


def expected_campaign_value(
    response_probability: float,
    order_value: float,
    profit_margin: float,
    expected_reward_cost: float,
    discount_pct: float,
) -> float:
    """Backwards-compatible wrapper around `reward_economics`.

    Provided for the notebook walkthrough; the engine itself uses the richer
    `reward_economics` so that contingent offers are priced correctly.
    """
    p = float(np.clip(response_probability, 0.0, 0.97))
    effective_margin = max(0.0, float(profit_margin) - float(discount_pct) / 100.0)
    gross = p * order_value * effective_margin
    return float(gross - expected_reward_cost)


# ============================================================================
# Candidate generation
# ============================================================================
def _candidate_offers(
    row: pd.Series,
    preferred_category: str,
    order_value: float,
) -> list[dict]:
    """Filter the catalogue down to offers this customer can actually redeem.

    An offer survives if the customer's typical basket clears the reward's
    minimum purchase, which stops the engine from recommending a Rs 500 OFF to
    someone who usually spends Rs 600.
    """
    out = []
    for offer in REWARD_CATALOGUE:
        floor = max(offer["minimum_purchase"], order_value * offer["min_purchase_rule"])
        if order_value >= floor:
            out.append({**offer, "_effective_min_purchase": float(floor)})
    if not out:  # tiny basket -> always offer something redeemable
        cheapest = min(REWARD_CATALOGUE, key=lambda o: o["minimum_purchase"])
        out = [{**cheapest, "_effective_min_purchase": float(order_value)}]
    return out


def optimise_reward_for_customer(
    row: pd.Series,
    segment: str,
    margin_by_category: dict[str, float],
    probability_lookup: callable | None = None,
) -> list[dict]:
    """Evaluate every eligible reward for one customer.

    Returns a list of per-offer evaluation dicts, best-first by expected
    campaign value.  Each entry carries both the mechanics and the simulated
    economics, so the notebook can show a side-by-side trade-off table.
    """
    base_p = float(row["reward_probability"])
    affinity = float(row.get("category_affinity_score", 0.3))
    sensitivity = float(row.get("price_sensitivity_index", 0.3))
    usage = float(row.get("reward_usage_rate", 0.2))

    aov = float(row.get("average_order_value", 0.0))
    # a responder typically spends at least their usual basket
    order_value = aov * 1.15 if aov > 0 else 500.0

    category = str(row.get("preferred_category", "Groceries"))
    margin = float(margin_by_category.get(category, 0.22))

    candidates = _candidate_offers(row, category, order_value)

    evaluations = []
    for offer in candidates:
        p = expected_response_probability(base_p, offer, affinity, sensitivity, usage)

        # a segment-aligned offer is easier for the customer to act on
        if segment in offer["segment_affinity"]:
            p = float(np.clip(p * 1.08, 0, 0.97))

        econ = reward_economics(offer, order_value, margin, p)

        evaluations.append({
            "reward_code": offer["code"],
            "reward_type": offer["reward_type"],
            "recommended_reward": offer["reward_label"],
            "discount_pct": offer["discount_pct"],
            "discount_flat": offer["discount_flat"],
            "minimum_purchase": offer["_effective_min_purchase"],
            "maximum_discount": offer["max_discount"],
            "reward_category": category,
            "estimated_order_value": order_value,
            "expected_response_probability": p,
            "response_lift": p - base_p,
            "realised_order_value": econ["realised_order_value"],
            "gross_reward_value": econ["gross_reward_value"],
            "net_if_convert": econ["net_if_convert"],
            "estimated_reward_cost": econ["expected_reward_cost"],
            "expected_campaign_value": econ["expected_campaign_value"],
            "value_per_rupee": (
                econ["expected_campaign_value"] / econ["expected_reward_cost"]
                if econ["expected_reward_cost"] > 0 else np.inf
            ),
        })

    evaluations.sort(key=lambda e: e["expected_campaign_value"], reverse=True)
    return evaluations


# ============================================================================
# Vectorised engine
# ============================================================================
def recommend_rewards(
    scored: pd.DataFrame,
    margin_by_category: dict[str, float],
    budget: float | None = None,
    value_weight: float = 1.0,
    probability_weight: float = 0.0,
    urgency_weight: float = 0.0,
) -> pd.DataFrame:
    """Pick one reward per customer by expected value, under an optional budget.

    ``budget`` is a campaign-wide spend cap on expected reward cost.  When
    supplied, customers are ranked by objective score and offers are accepted
    greedily until the budget is exhausted -- so a large budget yields more
    (and richer) offers rather than richer offers for everyone.
    """
    rows = []
    for _, row in scored.iterrows():
        segment = str(row.get("rfm_segment", "Regular"))
        evals = optimise_reward_for_customer(row, segment, margin_by_category)
        rows.append(evals)

    best = pd.DataFrame([e[0] for e in rows])
    best.insert(0, "customer_id", scored["customer_id"].to_numpy())

    # ---- objective the business actually cares about ---------------------
    best["objective_score"] = (
        best["expected_campaign_value"] * value_weight
        + best["expected_response_probability"] * 1000.0 * probability_weight
        + scored["urgency_score"].to_numpy() * 100.0 * urgency_weight
    )

    # ---- exposure policy: not everyone gets an offer ---------------------
    # Customers the model is confident will not respond are excluded: a reward
    # to them is pure margin loss.  The cut-off is a business threshold, not a
    # model parameter.
    best["targetable"] = best["expected_response_probability"] >= 0.30

    if budget is not None and budget > 0:
        best = _apply_budget(best, budget)
    else:
        best["budget_accepted"] = best["targetable"]

    best = best.sort_values(
        ["budget_accepted", "expected_campaign_value"], ascending=[False, False]
    ).reset_index(drop=True)
    best["priority_rank"] = np.where(
        best["budget_accepted"], best["expected_campaign_value"].rank(ascending=False,
                                                                     method="first").astype(int), 0
    )
    return best


def _apply_budget(best: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Greedy acceptance under a campaign spend cap.

    Iterating in descending expected value is a simple, auditable knapsack
    relaxation: it spends the budget where it buys the most incremental value
    and stops when the next-best offer no longer fits.
    """
    order = best.sort_values("expected_campaign_value", ascending=False).index
    accepted = pd.Series(False, index=best.index)
    spent = 0.0
    for idx in order:
        if not best.at[idx, "targetable"]:
            continue
        cost = float(best.at[idx, "estimated_reward_cost"])
        if spent + cost <= budget:
            accepted.at[idx] = True
            spent += cost
    best["budget_accepted"] = accepted
    best["budget_spent"] = float(best.loc[accepted, "estimated_reward_cost"].sum())
    return best


# ============================================================================
# Campaign priority
# ============================================================================
def campaign_priority(
    scored: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Blend response, value, recency pressure and affinity into one score.

    All five inputs are converted to 0-1 percentiles first, so no single scale
    dominates the blend.
    """
    df = scored.merge(
        recommendations[[
            "customer_id", "recommended_reward", "reward_type", "reward_code",
            "discount_pct", "discount_flat", "minimum_purchase", "maximum_discount",
            "reward_category", "estimated_order_value",
            "expected_response_probability", "response_lift",
            "estimated_reward_cost", "expected_campaign_value",
            "targetable", "budget_accepted", "priority_rank",
        ]],
        on="customer_id", how="left",
    )

    df["p_response"] = df["reward_probability"].rank(pct=True)
    df["p_value"] = df["total_spend"].rank(pct=True)
    df["p_campaign_value"] = df["expected_campaign_value"].rank(pct=True)
    df["p_urgency"] = df["days_since_last_purchase"].rank(pct=True, ascending=False)
    df["p_affinity"] = df["category_affinity_score"].rank(pct=True)

    df["campaign_priority_score"] = (
        0.30 * df["p_response"]
        + 0.22 * df["p_campaign_value"]
        + 0.20 * df["p_value"]
        + 0.16 * df["p_urgency"]
        + 0.12 * df["p_affinity"]
    ) * 100.0

    df = df.sort_values("campaign_priority_score", ascending=False).reset_index(drop=True)
    df["campaign_rank"] = np.arange(1, len(df) + 1)
    return df


# ============================================================================
# Campaign simulation
# ============================================================================
def simulate_campaign(
    scored: pd.DataFrame,
    recommendations: pd.DataFrame,
    budget: float = 50_000.0,
    seed: int = 42,
) -> dict:
    """Compare random targeting against ML targeting on the same budget.

    Strategy A picks customers uniformly at random.
    Strategy B spends the identical budget on the customers the model ranks as
    most likely to respond.
    Strategy C applies the budget *per customer* so the engine can pick a
    cheaper, better-fitting offer for each one.

    All three are evaluated with the SAME simulated response model, so the only
    difference between them is *who gets targeted* and *what they are offered*.

    Reported figures:
        Expected Revenue  = sum over responders of the realised order value
        Promotional Cost  = sum over responders of the gross reward value
        Expected Net      = sum over responders of the net value if they
                            convert (the discount is already netted out of the
                            realised order value for discount-style offers)
    Outcomes are simulated from synthetic data and are illustrative only.
    """
    rng = np.random.default_rng(seed)

    work = scored[["customer_id", "reward_probability"]].merge(
        recommendations[[
            "customer_id", "gross_reward_value", "realised_order_value",
            "net_if_convert", "expected_response_probability",
            "expected_campaign_value", "estimated_reward_cost",
        ]],
        on="customer_id", how="left",
    )

    n = len(work)
    # spend needed to place an offer on each customer (budget is measured on
    # *expected* promotional cost, which is what finance actually forecasts)
    cost = work["estimated_reward_cost"].fillna(0.0).to_numpy()
    p_offer = work["expected_response_probability"].fillna(0.0).to_numpy()
    realised = work["realised_order_value"].fillna(0.0).to_numpy()
    gross_val = work["gross_reward_value"].fillna(0.0).to_numpy()
    net_conv = work["net_if_convert"].fillna(0.0).to_numpy()

    def evaluate(idx: np.ndarray, probs: np.ndarray, label: str) -> dict:
        idx = np.asarray(idx)
        if len(idx) == 0:
            return {
                "Strategy": label, "Customers Selected": 0, "Expected Responders": 0.0,
                "Response Rate": 0.0, "Expected Revenue": 0.0, "Promotional Cost": 0.0,
                "Expected Net Value": 0.0, "Cost per Contact": 0.0, "Net per Contact": 0.0,
            }

        responded = rng.random(len(idx)) < probs[idx]
        n_resp = float(responded.sum())
        revenue = float((responded * realised[idx]).sum())
        promo = float((responded * gross_val[idx]).sum())
        net = float((responded * net_conv[idx]).sum())
        spend = float(cost[idx].sum())

        return {
            "Strategy": label,
            "Customers Selected": int(len(idx)),
            "Expected Responders": round(n_resp, 1),
            "Response Rate": round(n_resp / len(idx), 4),
            "Expected Revenue": round(revenue, 2),
            "Promotional Cost": round(promo, 2),
            "Expected Net Value": round(net, 2),
            "Cost per Contact": round(spend / len(idx), 2),
            "Net per Contact": round(net / len(idx), 2),
        }

    # ---- Strategy A: random targeting under the same budget -------------
    rand_order = rng.permutation(n)
    rand_idx = rand_order[:int(min(np.searchsorted(np.cumsum(cost[rand_order]), budget) + 1, n))]
    a = evaluate(rand_idx, p_offer, "A. Random Targeting")

    # ---- Strategy B: ML-targeted, same total budget ----------------------
    ml_order = np.argsort(-p_offer)
    ml_idx = ml_order[:int(min(np.searchsorted(np.cumsum(cost[ml_order]), budget) + 1, n))]
    b = evaluate(ml_idx, p_offer, "B. ML-Targeted")

    # ---- Strategy C: ML-targeted with a per-customer offer budget --------
    cap = budget / max(1, min(len(ml_idx), n))
    c_idx = np.array([i for i in ml_order if cost[i] <= cap])[: min(len(ml_idx), n)]
    c = evaluate(c_idx, p_offer, "C. ML + Cost-Optimised")

    summary = pd.DataFrame([a, b, c])
    summary["Budget"] = budget
    summary["Net Value vs Random"] = (
        summary["Expected Net Value"] / max(abs(a["Expected Net Value"]), 1e-9)
    ).round(3)

    return {
        "summary": summary,
        "budget": budget,
        "total_customers": int(n),
        "note": (
            "Simulation only. Outcomes are drawn from the synthetic response "
            "model; they are not observed business results."
        ),
    }


# ============================================================================
# Reporting
# ============================================================================
def format_currency(v: float) -> str:
    return f"{RUPEE}{v:,.0f}"