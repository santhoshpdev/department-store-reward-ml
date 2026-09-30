# AI-Powered Customer Intelligence & Personalised Reward Recommendation

**Machine Learning-Based Customer Behavior Analysis, Reward Response Prediction, and Personalized Discount Optimization**

---

> **Honesty statement.** The dataset and the `reward_response` campaign labels are **synthetically generated** for academic demonstration, because real historical customer campaign-response data is unavailable. Nothing in this repository should be read as a real business result. The value of the project is its methodology: leak-free feature engineering, a fair five-model comparison, and a cost-aware reward decision engine.

---

## Table of contents

- [Overview](#overview)
- [The Business Problem](#the-business-problem)
- [Objectives](#objectives)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Dataset](#dataset)
- [Synthetic Data Methodology](#synthetic-data-methodology)
- [Feature Engineering](#feature-engineering)
- [RFM Segmentation](#rfm-segmentation)
- [K-Means Segmentation](#k-means-segmentation)
- [The Five Models](#the-five-models)
- [Evaluation & Model Selection](#evaluation--model-selection)
- [Reward Optimisation](#reward-optimisation)
- [Campaign Simulation](#campaign-simulation)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Future Scope](#future-scope)

---

## Overview

This project turns a department store's transaction history into a **decision
system**, not just a prediction.

Given 12 months of purchase data for 1,500 customers across 100 products, it:

1. builds 50 customer-level behavioural features inside a strict observation
   window;
2. trains and fairly benchmarks **five supervised classifiers** to estimate each
   customer's probability of responding to a reward;
3. profiles customers with **RFM scoring** and **K-Means clustering**;
4. runs a **reward optimisation engine** that evaluates the entire offer
   catalogue per customer against expected profit and a campaign budget;
5. ranks customers by campaign priority and simulates whether ML targeting
   actually beats random targeting.

The notebook is the primary demonstration artifact. The `src/` modules exist so
the logic is reusable and testable outside the notebook.

---

## The Business Problem

Reward programmes usually start with good intentions and end as blanket
discounting: everyone gets the same `10% OFF`, margin leaks away, and nobody
can say which offers actually changed behaviour.

The store needs to answer, **per customer**:

| Question | Answered by |
|---|---|
| Will this customer respond to a reward at all? | Supervised ML (5-model benchmark) |
| Which reward gives the best return for this person? | Reward optimisation engine |
| In which category should the offer sit? | Customer-product affinity |
| Who should we contact first, given a budget? | Campaign priority ranking |
| Is ML targeting worth it over the status quo? | Budget-matched campaign simulation |

---

## Objectives

- Build a realistic synthetic dataset with **latent behavioural structure**, not
  independent random columns.
- Engineer a meaningful, interpretable customer feature set (~50 features).
- Apply a proper **observation / campaign window split** to prevent temporal
  data leakage.
- Benchmark **Logistic Regression, Decision Tree, Random Forest, XGBoost and
  CatBoost** on identical pipelines, without pre-selecting a winner.
- Evaluate on Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC, plus
  training and prediction time.
- Select the best model **dynamically**, using F1 as the primary metric.
- Build a reward engine that considers response probability, customer value,
  RFM segment, category affinity, discount sensitivity and budget.
- Demonstrate explainability and produce a professional, presentation-ready
  notebook with dashboards.

---

## Architecture

```
RAW CUSTOMER + TRANSACTION + PRODUCT DATA
        |
        v
DATA VALIDATION
        |
        v
FEATURE ENGINEERING  (observation window only)
        |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
  RFM ANALYSIS        PRODUCT AFFINITY      K-MEANS CLUSTERS
        |                     |                     |
        +---------------------+---------------------+
                                |
                                v
                      CUSTOMER ML DATASET
                      (1 row per customer)
                                |
        +----------+------------+-----------+----------+
        |          |            |           |          |
        v          v            v           v          v
    Logistic    Decision      Random      XGBoost   CatBoost
    Regression   Tree        Forest
        +----------+------------+-----------+----------+
                                |
                                v
                       MODEL COMPARISON
                       (F1 selects the winner)
                                |
                                v
                     CUSTOMER RESPONSE SCORE
                                |
                                v
                     REWARD OPTIMISATION
                     (catalogue x customer x budget)
                                |
                                v
                     PERSONALISED OFFER
                                |
                                v
                     CAMPAIGN PRIORITY RANKING
                                |
                                v
                     BUSINESS SIMULATION
```

---

## Repository Layout

```
department_store_reward_ml/
├── data/
│   ├── customers.csv          # 1,500 customers
│   ├── products.csv           # 100 products across 12 categories
│   └── transactions.csv       # 9,800 orders, calendar year 2025
├── models/
│   ├── best_reward_model.pkl      # fitted Pipeline (preprocessing + model)
│   ├── preprocessing_pipeline.pkl # the preprocessing contract alone
│   └── model_metadata.pkl         # selection rule, metrics, feature list
├── notebooks/
│   └── customer_reward_ml.ipynb   # THE presentation artifact
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # synthetic data with latent behaviour
│   ├── feature_engineering.py # RFM, temporal, affinity features
│   ├── model_training.py      # target, 5 models, benchmark, persistence
│   ├── reward_engine.py       # catalogue, optimisation, ranking, simulation
│   └── visualization.py       # charts, architecture diagram, HTML dashboards
├── outputs/
│   ├── figures/               # all chart PNGs
│   └── reports/               # exported CSVs
└── README.md
```

---

## Dataset

| File | Shape | Key columns |
|---|---|---|
| `customers.csv` | 1,500 x 6 | `customer_id`, `age`, `gender`, `city`, `customer_since`, `customer_persona` |
| `products.csv` | 100 x 8 | `product_id`, `product_name`, `category`, `sub_category`, `brand`, `unit_price`, `profit_margin` |
| `transactions.csv` | 9,800 x 13 | `transaction_id`, `customer_id`, `transaction_date`, `product_id`, `quantity`, `unit_price`, `discount_percentage`, `discount_amount`, `subtotal`, `total_amount`, `payment_method`, `reward_used`, `reward_type` |

Internal arithmetic is consistent by construction:

```
subtotal        = quantity × unit_price
discount_amount = subtotal × discount_percentage / 100
total_amount    = subtotal − discount_amount
```

`RANDOM_STATE = 42` makes the entire project reproducible.

---

## Synthetic Data Methodology

No real customer dataset was used. Rather than drawing every column
independently — which produces data a model can learn in one line — the
generator gives each customer a **latent purchasing personality** that drives
their behaviour:

| Persona | Signature |
|---|---|
| **High-Value Loyal** | frequent, high-value, stable, category-focused, moderate reward use |
| **Regular** | moderate frequency and spend, consistent |
| **Occasional** | low frequency, low spend, long gaps |
| **Price-Sensitive** | high discount usage, strong reward response, low AOV |
| **At-Risk** | long history, recent decline, long recency |
| **New** | short tenure, few purchases |

A hidden **category affinity vector** is drawn per customer (a focused
Dirichlet), and each order is drawn from that distribution — so real category
preferences emerge in the transaction log rather than being assigned.

`customer_persona` is written to `customers.csv` as ground truth **for
validation only** and is excluded from every feature set. The model has to
rediscover behaviour from the transaction history alone.

Declining engagement is modelled two ways: a density *tilt* that pulls purchases
earlier in the window, and a *tail drop* that removes recent orders — together
producing the characteristic "historically active, recently gone quiet" signature
of an at-risk customer.

---

## Feature Engineering

Features are computed **only** from the observation window
(1 Jan – 30 Sep 2025). The campaign window (1 Oct – 2025) is never touched.

The notebook independently proves this: it rebuilds the feature frame after
removing all campaign-window rows and asserts the values are identical.

**50 model features** across six families:

| Family | Examples |
|---|---|
| RFM & value | `total_orders`, `total_spend`, `average_order_value`, `R_score`, `F_score`, `M_score`, `rfm_score` |
| Temporal momentum | `days_since_last_purchase`, `recent_90_day_spend`, `previous_90_day_spend`, `spending_trend`, `purchase_frequency_trend` |
| Discount & reward | `average_discount_percentage`, `discount_usage_rate`, `reward_usage_rate`, `high_commitment_reward_rate` |
| Basket diversity | `unique_products`, `unique_categories`, `unique_brands`, `category_entropy` |
| Category affinity | `preferred_category`, `category_affinity_score`, `cat_ratio_*` |
| Business & demo | `gross_profit`, `age`, `gender`, `city`, `age_band`, `urgency_score` |

Redundant and target-derived columns are explicitly excluded (documented in
`src/feature_engineering.py`), so the model sees 50 informative inputs rather
than a padded set.

---

## RFM Segmentation

Recency, Frequency and Monetary are each scored 1–5 by **percentile rank**
(the `total_spend` column contains raw rupees, while a derived proportion column
is used for the score, so no single unit scale dominates), giving an RFM score
out of 15 and a business segment.

Segments produced: **Champions, Loyal Customers, Cannot Lose Them, Potential
Loyalists, Promising, New Customers, Needs Attention, At Risk, Hibernating,
Lost.**

The segments are verified to be genuinely different populations, not cosmetic
labels — Champions cluster at high spend and low recency while Hibernating/Lost
sit at the opposite corner.

---

## K-Means Segmentation

An unsupervised component sits alongside the rule-based RFM view:

- Features are **standardised first** (K-Means is distance-based).
- `k` is chosen by **silhouette score** with an elbow plot, not assumed to be 5.
- Clusters are profiled on spend, orders, recency, affinity and predicted
  response, then given descriptive labels derived from their statistics.

The notebook cross-tabulates K-Means clusters against RFM segments to show where
they agree and where they differ — K-Means isolates value tiers, RFM encodes
recency direction.

---

## The Five Models

Every model is a scikit-learn `Pipeline`: preprocessing + estimator in one
object. This guarantees fair comparison and means a single pickle reproduces a
prediction end-to-end.

| # | Model | Role in the comparison |
|---|---|---|
| 1 | **Logistic Regression** | Interpretable linear baseline |
| 2 | **Decision Tree** | Explainable non-linear decision rules |
| 3 | **Random Forest** | Robust bagged ensemble |
| 4 | **XGBoost** | Gradient boosting on tabular data |
| 5 | **CatBoost** | Ordered boosting with categorical handling |

Preprocessing is fitted **on the training split only** — numerical imputation
(and scaling where required) plus one-hot encoding of categoricals.

---

## Evaluation & Model Selection

For every model, measured on the held-out test set:

| Metric | Why it matters here |
|---|---|
| Accuracy | Included for completeness — a trap with a minority class |
| Precision | Wasted offers / margin leakage |
| Recall | Missed customer opportunities |
| **F1** | Balanced treatment of both errors — **the selection criterion** |
| ROC-AUC | Ranking quality across all thresholds |
| PR-AUC | Performance on the positive class specifically |
| Train / Prediction time | Operational cost |

**Selection rule:** highest F1, ties broken by ROC-AUC, then PR-AUC, precision,
recall.

**Why F1 and not accuracy:** a false positive means a discount given to someone
who would have bought anyway (a margin leak); a false negative means a reachable
customer left alone (a revenue leak). Neither error obviously dominates, so the
balanced F1 is the honest criterion. The winner is selected *dynamically* from
measured results — never assumed in advance.

Additional diagnostics:

- **Overfitting check** — train vs test ROC-AUC gap per model.
- **ROC curves** and **precision-recall curves** for all five models on one
  figure, each labelled with its actual metric.
- **Confusion matrices** in a consistent grid so FP/FN are easy to compare.
- **Calibration plot** — decile-wise predicted probability vs observed response
  rate, because the reward engine treats the score as a real probability.

---

## Reward Optimisation

This is the business core, and it is deliberately **not** a threshold rule.

### The catalogue

| Reward | Type | Cost model | Suits |
|---|---|---|---|
| 5% OFF | Percentage | Reduces revenue | Champions, Loyal |
| 10% OFF | Category discount | Reduces revenue | Loyal, Champions |
| 15% OFF | Percentage | Reduces revenue | At Risk, Hibernating |
| Rs 100 OFF | Fixed | Reduces revenue | Regular, Potential Loyalists |
| Rs 200 OFF | Fixed | Reduces revenue | Champions, Loyal |
| Rs 500 OFF | Fixed | Reduces revenue | Champions, Cannot Lose Them |
| Cashback | Contingent | Paid on conversion | Regular, Promising |
| BOGO | Contingent | Paid on conversion | Price Sensitive, Occasional |

### Two cost models, deliberately

- **Discount offers** lower the price the customer pays. The cost appears as a
  smaller realised order value — it is *already* inside the margin.
- **Contingent offers** (cashback, BOGO) are paid out *after* a successful
  purchase, so they are a genuine extra outflow incurred only by responders.

Conflating the two double-counts the discount — a classic promotion-P&L error.

### Per-customer decision loop

For every candidate offer the engine computes:

```
P(response | offer)    base probability modulated by the offer
realised order value    basket after the discount
net_if_convert         realised_value x margin     (discount offers)
                       order_value x margin - payout (contingent offers)
expected campaign value = P(response) x net_if_convert
expected reward cost    = P(response) x gross reward value
```

The offer with the best **expected campaign value** wins — which is why the
engine does not always hand out the biggest discount. A customer the model
scores at ~0.99 makes every offer look equally good, so the engine's behaviour
is most visible on customers near the operating threshold.

### Exposure policy

Customers below a response-probability threshold are excluded: a reward to them
is pure margin loss. With an explicit budget, customers are ranked by objective
value and offers accepted greedily until the budget is exhausted, so a larger
budget buys **more** (not just richer) offers.

---

## Campaign Simulation

Three strategies, the **same budget**, evaluated with the same simulated
response model — so the only difference is *who* is targeted:

| Strategy | Description |
|---|---|
| A. Random | Contact customers at random until the budget is spent |
| B. ML-Targeted | Contact the highest predicted-responders first |
| C. ML + Cost-Optimised | ML targeting with a per-customer offer ceiling |

Reported figures: customers selected, expected responders, response rate,
expected revenue, promotional cost, expected net value, cost per contact, net
per contact.

> This is a **simulation on synthetic data and model predictions**. It is not a
> claim about real-world financial performance.

---

## How to Run

### Prerequisites

Python 3.10+ with: `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
`xgboost`, `catboost`, `nbformat`/`nbclient` (optional, for headless execution).

```bash
pip install pandas numpy scikit-learn matplotlib xgboost catboost
```

### 1. Generate the data

```bash
cd department_store_reward_ml
python src/data_generator.py
```

This writes `data/customers.csv`, `data/products.csv` and
`data/transactions.csv`.

### 2. Open the notebook

```bash
cd notebooks
jupyter notebook customer_reward_ml.ipynb
```

In Google Colab: upload the `department_store_reward_ml/` folder, open
`notebooks/customer_reward_ml.ipynb`, and run all cells. The notebook adds the
project root to `sys.path`, so it works from the `notebooks/` directory without
extra configuration.

### 3. Headless execution (optional)

```bash
pip install nbclient nbconvert
python -m nbconvert --to notebook --execute --inplace notebooks/customer_reward_ml.ipynb
```

### 4. Reuse the trained model

```python
import pandas as pd
from src import model_training as mt

pipeline, metadata, preprocessing = mt.load_artifacts('models')

new_data = pd.read_csv('new_customers.csv')          # same columns as the feature frame
proba = pipeline.predict_proba(new_data[metadata['feature_columns']])[:, 1]
```

The saved `Pipeline` contains preprocessing and the fitted estimator, so
`predict_proba` works directly — no retraining required.

### Expected run time

Roughly 30–60 seconds for a full notebook run on a laptop, dominated by
CatBoost and the permutation-importance calculation.

---

## Limitations

1. **The target is synthetic.** `reward_response` was generated, not observed.
   Every model metric is a *methodological* result. It demonstrates that the
   pipeline is sound, leak-free and well-evaluated; it is not evidence about
   real customers.
2. **The uplift model is an assumption.** The response lift assigned to each
   offer is a parameter of the simulator, not something estimated from an
   experiment.
3. **Expected values are illustrative**, not financial forecasts.
4. **No causal inference.** The model predicts association, not the causal
   effect of a reward. Randomised holdout testing is the only way to measure
   true lift.
5. **Moderate dataset size** (1,500 customers). Confidence intervals on
   precision/recall are non-trivial at this scale.
6. **SHAP is optional** and not installed by default; permutation importance
   plus the model's own importances are provided instead.

---

## Future Scope

| Enhancement | Why it would help |
|---|---|
| **Randomised holdout experiment** | Converts every assumption in this project into a measurement — the single highest-value next step |
| **Calibrated probabilities** (Platt / isotonic) | Improves the reward engine's expected-value estimates directly |
| **Time-based model rebuild** | Rolling retraining on recent windows to control drift |
| **Multi-objective offer selection** | Constrained optimisation over budget, coverage and margin simultaneously |
| **Lift modelling** (uplift / causal forests) | Predicts *incremental* response rather than raw response |
| **Category-level response models** | A separate model per department for finer personalisation |
| **Real campaign data ingestion** | The moment real labels exist, this pipeline consumes them unchanged |

---

## Project Standards

This project is built to demonstrate:

- **Data engineering** — synthetic generation with latent structure, validation, cleaning.
- **Data analytics** — EDA, RFM, product/category intelligence.
- **Machine learning** — K-Means plus five supervised models.
- **Model evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC/PR curves, calibration, overfitting checks.
- **Model interpretability** — permutation importance and model-native importance.
- **Business intelligence** — customer scoring, reward personalisation, reward optimisation, campaign prioritisation, budget-matched simulation.
- **Presentation** — executive dashboard, model benchmark dashboard, customer reward cards, architecture diagram, charts and tables.

Deliberately **out of scope**: neural networks, deep learning, LSTMs, CNNs,
transformers, reinforcement learning, multi-agent systems, LLMs inside the
prediction loop, backend APIs, frontend frameworks, authentication and
infrastructure. This is a tabular retail ML problem, and strong classical ML
plus analytics is the appropriate level of engineering for it.
