"""
Synthetic Data Generator
========================
Generates a behaviourally realistic synthetic dataset for a department store
customer-reward recommendation project.

Design goals
------------
1.  Customers are NOT drawn from one uniform distribution.  Every customer is
    assigned a *latent purchasing personality* (high-value loyal, regular,
    occasional, price-sensitive, at-risk, new) which drives:
        - how often they shop,
        - how much they spend,
        - which categories they prefer,
        - how strongly they react to discounts / rewards,
        - whether their activity is rising or declining.

2.  Category affinity is latent and per-customer.  A customer's product choice
    is drawn from a softmax over personal category-weights, so real category
    preferences emerge in the transaction log.

3.  One row of `transactions.csv` == one *order* (one product purchase event,
    possibly multiple units).  This keeps `transaction_id` unique per order and
    makes the order-level RFM features meaningful.

4.  `customer_persona` is ground truth for validation only.  It is written to
    `customers.csv` but is NEVER used as a model feature -- the model has to
    rediscover behaviour from transaction history alone.

5.  Arithmetic is internally consistent by construction:
        subtotal        = quantity * unit_price
        discount_amount = subtotal * discount_percentage / 100
        total_amount    = subtotal - discount_amount

6.  The log spans 12 months (Jan-Dec 2025).  Feature engineering is restricted
    to the observation window (Jan 1 - Sep 30 2025); Oct-Dec 2025 is the
    campaign/prediction window and must never leak into features.
"""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RANDOM_STATE = 42

CURRENCY = "INR"
RUPEE = "\u20b9"

# ----------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31")          # 12 months of purchase history
OBSERVATION_START = pd.Timestamp("2025-01-01")
OBSERVATION_END = pd.Timestamp("2025-09-30")    # feature window
CAMPAIGN_START = pd.Timestamp("2025-10-01")     # simulated campaign window


# ----------------------------------------------------------------------------
# Personas
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Persona:
    """Latent behavioural archetype used to drive transaction generation."""

    name: str
    weight: float
    orders_range: tuple[float, float]          # orders over the full 12 months
    qty_range: tuple[float, float]             # units-per-order multiplier
    discount_propensity: float                 # how likely to see / use a discount
    reward_usage_rate: tuple[float, float]     # prob. an order redeems a reward
    recency_decay: float                       # 0 = steady, 1 = strongly declining
    tenure_months_range: tuple[float, float]   # months since registration
    category_focus: float                      # 0 = broad basket, 1 = very focused


PERSONAS: list[Persona] = [
    Persona(
        name="High-Value Loyal", weight=0.16,
        orders_range=(13.0, 25.0), qty_range=(1.8, 3.2),
        discount_propensity=0.32, reward_usage_rate=(0.26, 0.46),
        recency_decay=0.10, tenure_months_range=(14.0, 48.0), category_focus=0.62,
    ),
    Persona(
        name="Regular", weight=0.28,
        orders_range=(5.0, 11.0), qty_range=(1.4, 2.6),
        discount_propensity=0.28, reward_usage_rate=(0.10, 0.26),
        recency_decay=0.25, tenure_months_range=(9.0, 40.0), category_focus=0.42,
    ),
    Persona(
        name="Occasional", weight=0.22,
        orders_range=(1.0, 4.0), qty_range=(1.2, 2.2),
        discount_propensity=0.20, reward_usage_rate=(0.03, 0.14),
        recency_decay=0.40, tenure_months_range=(7.0, 34.0), category_focus=0.50,
    ),
    Persona(
        name="Price-Sensitive", weight=0.19,
        orders_range=(4.0, 9.0), qty_range=(1.8, 3.4),
        discount_propensity=0.90, reward_usage_rate=(0.44, 0.72),
        recency_decay=0.20, tenure_months_range=(6.0, 34.0), category_focus=0.46,
    ),
    Persona(
        name="At-Risk", weight=0.10,
        orders_range=(7.0, 16.0), qty_range=(1.3, 2.4),
        discount_propensity=0.38, reward_usage_rate=(0.14, 0.32),
        recency_decay=0.90, tenure_months_range=(14.0, 48.0), category_focus=0.58,
    ),
    Persona(
        name="New", weight=0.05,
        orders_range=(1.0, 4.0), qty_range=(1.2, 2.2),
        discount_propensity=0.45, reward_usage_rate=(0.16, 0.36),
        recency_decay=0.15, tenure_months_range=(1.0, 4.0), category_focus=0.40,
    ),
]


# ----------------------------------------------------------------------------
# Product catalogue (exactly 100 products across 12 categories)
# ----------------------------------------------------------------------------
PRODUCT_CATALOGUE: dict[str, list[tuple[str, str, str, float, float]]] = {
    "Groceries": [
        ("Basmati Rice 5kg", "Staples", "Fortune", 520.0, 0.09),
        ("Sunflower Cooking Oil 1L", "Staples", "Tata", 145.0, 0.11),
        ("Sugar 1kg", "Staples", "Dhampure", 48.0, 0.08),
        ("Whole Wheat Flour 5kg", "Staples", "Aashirvaad", 285.0, 0.10),
        ("Toor Dal 1kg", "Pulses", "Tata", 165.0, 0.12),
        ("Chana Dal 1kg", "Pulses", "Tata", 132.0, 0.12),
        ("Urad Dal 1kg", "Pulses", "Aashirvaad", 148.0, 0.13),
        ("Red Lentils 1kg", "Pulses", "Fortune", 118.0, 0.12),
        ("Table Salt 1kg", "Spices", "Tata", 28.0, 0.22),
        ("Turmeric Powder 500g", "Spices", "Aashirvaad", 88.0, 0.20),
        ("Red Chilli Powder 200g", "Spices", "Tata", 72.0, 0.21),
        ("Garam Masala 100g", "Spices", "Everest", 95.0, 0.25),
    ],
    "Snacks & Beverages": [
        ("Potato Chips 100g", "Namkeen", "Lay's", 45.0, 0.30),
        ("Salted Peanuts 200g", "Namkeen", "Haldiram's", 65.0, 0.26),
        ("Digestive Biscuits 400g", "Biscuits", "Parle-G", 90.0, 0.22),
        ("Marie Gold Biscuits 500g", "Biscuits", "Britannia", 115.0, 0.20),
        ("Instant Noodles 70g", "Noodles", "Maggi", 34.0, 0.28),
        ("Orange Juice 1L", "Beverages", "Real", 120.0, 0.24),
        ("Green Tea Bags 25ct", "Beverages", "Tata Tea", 210.0, 0.32),
        ("Instant Coffee 500g", "Beverages", "Nescafe", 640.0, 0.18),
        ("Cola Soft Drink 750ml", "Beverages", "Coca-Cola", 45.0, 0.26),
        ("Energy Drink 250ml", "Beverages", "Red Bull", 105.0, 0.24),
    ],
    "Personal Care": [
        ("Herbal Shampoo 340ml", "Hair Care", "Himalaya", 245.0, 0.32),
        ("Anti-Dandruff Shampoo 360ml", "Hair Care", "Head & Shoulders", 385.0, 0.28),
        ("Coconut Hair Oil 500ml", "Hair Care", "Parachute", 275.0, 0.26),
        ("Body Wash 250ml", "Bath Care", "Dove", 210.0, 0.30),
        ("Beauty Soap 4x125g", "Bath Care", "Lifebuoy", 145.0, 0.24),
        ("Bamboo Toothpaste 150g", "Oral Care", "Colgate", 95.0, 0.30),
        ("Toothbrush Twin Pack", "Oral Care", "Oral-B", 210.0, 0.34),
        ("Sandal Deodorant 150ml", "Personal Care", "Godrej", 235.0, 0.33),
    ],
    "Beauty & Cosmetics": [
        ("Sandal Face Wash 100g", "Skin Care", "Clean & Clear", 145.0, 0.36),
        ("Vitamin C Face Serum 30ml", "Skin Care", "Minimalist", 399.0, 0.45),
        ("Sunscreen SPF50 50ml", "Skin Care", "Replika", 545.0, 0.44),
        ("Matte Lipstick 4.5g", "Makeup", "Maybelline", 289.0, 0.48),
        ("Waterproof Eyeliner 3.5g", "Makeup", "Maybelline", 325.0, 0.50),
        ("Compact Foundation 30g", "Makeup", "Lakme", 545.0, 0.44),
        ("Rose Water Toner 250ml", "Skin Care", "Minimalist", 399.0, 0.42),
        ("Hair Spa Mask 200g", "Hair Care", "Tresemme", 425.0, 0.40),
    ],
    "Baby Care": [
        ("Baby Diapers Small (M)", "Diapers", "Pampers", 845.0, 0.22),
        ("Baby Diapers Large (L)", "Diapers", "Pampers", 945.0, 0.22),
        ("Baby Wipes Wet Wipes (Pack)", "Wipes", "Johnson & Johnson", 165.0, 0.34),
        ("Baby Lotion 200ml", "Skin Care", "Johnson & Johnson", 285.0, 0.38),
        ("Baby Shampoo 200ml", "Bath Care", "Johnson & Johnson", 245.0, 0.36),
        ("Baby Powder 200g", "Skin Care", "Johnson's Baby", 165.0, 0.33),
        ("Baby Feeding Bottle 250ml", "Feeding", "Pigeon", 385.0, 0.30),
        ("Baby Pacifier (Pack of 2)", "Feeding", "Chicco", 325.0, 0.35),
    ],
    "Home & Kitchen": [
        ("Stainless Steel Cookware Set", "Cookware", "Prestige", 2450.0, 0.24),
        ("Non-stick Tawa 28cm", "Cookware", "Prestige", 985.0, 0.26),
        ("Pressure Cooker 5L", "Cookware", "Hawkins", 2450.0, 0.22),
        ("Mixer Grinder 750W", "Appliances", "Butterfly", 3250.0, 0.20),
        ("Electric Kettle 1.5L", "Appliances", "Butterfly", 1245.0, 0.23),
        ("Gas Stove 2 Burner", "Appliances", "Indane", 2245.0, 0.21),
        ("Cotton Bedsheet Set", "Home Textiles", "Bombay Dyeing", 1295.0, 0.32),
        ("Microfibre Cleaning Cloth (3pc)", "Home Care", "Scotch-Brite", 245.0, 0.30),
    ],
    "Household Cleaning": [
        ("Dishwash Liquid 750ml", "Dishwash", "Vim", 165.0, 0.26),
        ("Laundry Detergent 2kg", "Laundry", "Surf Excel", 645.0, 0.20),
        ("Floor Cleaner 1L", "Floor Care", "Lizol", 245.0, 0.28),
        ("Toilet Cleaner 500ml", "Bathroom", "Harpic", 185.0, 0.32),
        ("Glass Cleaner 500ml", "Surface", "Duzz", 145.0, 0.30),
        ("Hand Wash Refill 1L", "Hand Care", "Dettol", 265.0, 0.30),
        ("Mosquito Repellent Refill", "Repellents", "All Out", 145.0, 0.36),
        ("Garbage Bags Medium (30pc)", "Disposal", "Goodline", 210.0, 0.28),
    ],
    "Clothing": [
        ("Men Cotton T-Shirt", "Apparel", "Levi's", 645.0, 0.42),
        ("Women Printed Kurta", "Apparel", "Biba", 1245.0, 0.44),
        ("Kids Cotton Shorts", "Kids Wear", "Gini & Jony", 425.0, 0.45),
        ("Denim Jeans Regular Fit", "Apparel", "Lee Cooper", 1845.0, 0.40),
        ("Winter Sweater", "Apparel", "Allen Solly", 2245.0, 0.42),
        ("Saree Cotton Blend", "Ethnic Wear", "Biba", 2450.0, 0.43),
        ("Leather Formal Shoes", "Footwear", "Woodland", 4245.0, 0.36),
    ],
    "Footwear": [
        ("Sports Running Shoes", "Sports", "Nike", 3245.0, 0.38),
        ("Casual Sneakers", "Casual", "Puma", 2145.0, 0.40),
        ("Leather Formal Shoes", "Formal", "Woodland", 4245.0, 0.36),
        ("Women's Block Heel Sandals", "Casual", "Bata", 1845.0, 0.42),
        ("Flip Flops Slippers", "Casual", "Bata", 445.0, 0.44),
        ("Kids Sandals", "Kids", "Bata", 845.0, 0.43),
        ("Ankle Socks (Pack of 3)", "Accessories", "Puma", 345.0, 0.48),
    ],
    "Electronics": [
        ("Wired Earphones 3.5mm", "Audio", "boAt", 645.0, 0.42),
        ("Bluetooth Earbuds", "Audio", "OnePlus", 2495.0, 0.34),
        ("Power Bank 10000mAh", "Accessories", "Mi", 1245.0, 0.32),
        ("USB-C Fast Charging Cable", "Accessories", "AmazonBasics", 345.0, 0.44),
        ("Portable Bluetooth Speaker", "Audio", "JBL", 3245.0, 0.30),
        ("Smart Watch AMOLED", "Wearables", "Fire-Boltt", 2245.0, 0.34),
        ("LED Desk Lamp", "Home Tech", "Philips", 1245.0, 0.38),
        ("AA Rechargeable Batteries (4pc)", "Power", "Duracell", 845.0, 0.40),
    ],
    "Stationery": [
        ("A4 Notebook 200 Pages", "Notebooks", "ClassMate", 145.0, 0.30),
        ("Gel Pen Blue (Pack of 10)", "Writing", "Cello", 210.0, 0.34),
        ("School Exercise Book (5pc)", "Notebooks", "Navneet", 165.0, 0.28),
        ("Sticky Notes Assorted", "Office", "Post-it", 245.0, 0.36),
        ("Exam Pen Set (10pc)", "Writing", "Cello", 145.0, 0.34),
        ("Art Colour Box 12pc", "Art Supplies", "Camlin", 385.0, 0.32),
        ("Stapler Heavy Duty", "Office", "Kangaro", 645.0, 0.30),
        ("Geometry Box Complete", "Maths", "Apsara", 195.0, 0.30),
    ],
    "Health & Wellness": [
        ("Multivitamin Tablets (60ct)", "Supplements", "Centrum", 645.0, 0.34),
        ("Vitamin D3 Capsules (30ct)", "Supplements", "Dr. Best", 445.0, 0.36),
        ("Protein Powder Chocolate 500g", "Nutrition", "Muscle Blaze", 2245.0, 0.30),
        ("Hand Sanitizer 500ml", "Hygiene", "Purell", 245.0, 0.32),
        ("Face Mask Sheet (Pack of 5)", "Personal Care", "Glow & Lovely", 225.0, 0.40),
        ("Digital Thermometer", "Devices", "Omron", 845.0, 0.34),
        ("First Aid Kit Compact", "Devices", "Horizon", 545.0, 0.32),
        ("Herbal Tea Bags 25ct", "Wellness", "Patanjali", 165.0, 0.30),
    ],
}

CATEGORIES: list[str] = list(PRODUCT_CATALOGUE.keys())

CITIES = [
    "Coimbatore", "Erode", "Salem", "Tiruppur",
    "Chennai", "Bangalore", "Madurai", "Karur",
]
CITY_WEIGHTS = np.array([0.24, 0.11, 0.12, 0.13, 0.16, 0.12, 0.08, 0.04])

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
PAYMENT_WEIGHTS = np.array([0.46, 0.18, 0.15, 0.13, 0.08])

REWARD_TYPES = [
    "None", "Percentage Discount", "Fixed Discount",
    "Cashback", "Category Coupon", "BOGO",
]

# Groceries-like categories suit flat-rupee and BOGO style rewards
LOW_MARGIN_CATEGORIES = {"Groceries", "Snacks & Beverages", "Household Cleaning",
                         "Personal Care", "Health & Wellness"}

# Reference order-value ceiling (keeps basket sizes commercially realistic)
ORDER_VALUE_CEILING = 3500.0

# Promotion guard-rails
MAX_FLAT_REWARD_SHARE = 0.25     # a rupee reward may not exceed 25% of a basket
MAX_TOTAL_DISCOUNT_PCT = 45.0    # no order is ever discounted beyond 45%


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _zipf_weights(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    """Concentrated, realistic weight distribution over `n` items."""
    ranks = np.arange(1, n + 1, dtype=float)
    w = ranks ** (-alpha)
    return w / w.sum()


def _scale_orders_to_target(counts: dict[str, int], target: int) -> dict[str, int]:
    """Trim surplus orders so the total matches `target`, never dropping below 1.

    Uses a max-heap so surplus is always removed from the customers that have
    the most orders, which keeps the persona-driven shape of the distribution.
    """
    counts = dict(counts)
    total = sum(counts.values())
    if total <= target:
        return counts

    heap = [(-c, cid) for cid, c in counts.items() if c > 1]
    heapq.heapify(heap)

    while total > target and heap:
        _neg, cid = heapq.heappop(heap)
        counts[cid] -= 1
        total -= 1
        if counts[cid] > 1:
            heapq.heappush(heap, (-counts[cid], cid))
    return counts


# ----------------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------------
def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for c_idx, (category, items) in enumerate(PRODUCT_CATALOGUE.items(), start=1):
        for i, (name, sub_cat, brand, base_price, margin) in enumerate(items):
            price = round(max(10.0, base_price * float(rng.normal(1.0, 0.04))), 2)
            m = float(np.clip(rng.normal(margin, 0.025), 0.05, 0.55))
            rows.append(
                {
                    "product_id": f"P{c_idx:03d}{i + 1:03d}",
                    "product_name": name,
                    "category": category,
                    "sub_category": sub_cat,
                    "brand": brand,
                    "unit_price": price,
                    "profit_margin": round(m, 4),
                }
            )

    products = pd.DataFrame(rows)
    products["discount_ceiling"] = np.where(
        products["category"].isin(LOW_MARGIN_CATEGORIES),
        rng.choice([0, 5, 10], size=len(products), p=[0.45, 0.35, 0.20]),
        rng.choice([0, 5, 10, 15], size=len(products), p=[0.35, 0.35, 0.20, 0.10]),
    ).astype(int)
    return products


# ----------------------------------------------------------------------------
# Customers (+ hidden latent parameter table)
# ----------------------------------------------------------------------------
def build_customers(
    rng: np.random.Generator,
    n_customers: int,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (customers.csv frame, hidden latent parameter frame).

    The latent frame drives transaction generation and is *not* exported into
    any modelling dataset.  Only `latent_persona` is persisted separately, and
    purely so that the generated behaviour can be validated against ground
    truth in the notebook.
    """
    cat_list = products["category"].unique().tolist()

    weights = np.array([p.weight for p in PERSONAS])
    weights = weights / weights.sum()
    persona_idx = rng.choice(len(PERSONAS), size=n_customers, p=weights)

    cust_rows: list[dict] = []
    latent_rows: list[dict] = []

    max_back_months = (END_DATE - START_DATE).days / 30.44

    for i in range(n_customers):
        cid = f"C{i + 1:04d}"
        pidx = int(persona_idx[i])
        p = PERSONAS[pidx]

        age = int(np.clip(rng.normal(38, 13), 18, 75))
        gender = str(rng.choice(["Male", "Female"], p=[0.46, 0.54]))
        city = str(rng.choice(CITIES, p=CITY_WEIGHTS))

        months = float(rng.uniform(*p.tenure_months_range))
        months = float(np.clip(months, 1.0, max_back_months))
        join_date = END_DATE - pd.Timedelta(days=int(round(months * 30.44)))

        # ---- latent category affinity (focused Dirichlet) -------------------
        base = _zipf_weights(rng, len(cat_list), 0.30)
        concentration = float(np.clip(rng.normal(9.0 - 6.5 * p.category_focus, 2.5), 1.2, 30.0))
        cat_affinity = rng.dirichlet(concentration * base)

        cust_rows.append(
            {
                "customer_id": cid,
                "age": age,
                "gender": gender,
                "city": city,
                "customer_since": join_date.date().isoformat(),
                # ground truth for validation -- EXCLUDED from all model features
                "customer_persona": p.name,
            }
        )

        latent_rows.append(
            {
                "customer_id": cid,
                "_persona": p.name,
                "_persona_idx": pidx,
                "_n_orders": float(rng.uniform(*p.orders_range)),
                "_qty_mult": float(rng.uniform(*p.qty_range)),
                "_discount_propensity": float(
                    np.clip(rng.normal(p.discount_propensity, 0.10), 0.02, 0.98)
                ),
                "_reward_usage_rate": float(rng.uniform(*p.reward_usage_rate)),
                "_recency_decay": p.recency_decay,
                "_cat_affinity": cat_affinity,
            }
        )

    customers = pd.DataFrame(cust_rows)
    latent = pd.DataFrame(latent_rows)
    return customers, latent


# ----------------------------------------------------------------------------
# Transactions
# ----------------------------------------------------------------------------
def _order_dates(
    rng: np.random.Generator,
    n_orders: int,
    tenure_start: pd.Timestamp,
    recency_decay: float,
) -> list[pd.Timestamp]:
    """Sample distinct purchase dates with a persona-driven trend shape.

    Declining engagement is modelled in two complementary ways:

    1.  A density *tilt* pulls purchases towards the earlier part of the
        customer's active window.
    2.  A *tail drop* discards the most recent fraction of orders, which is
        what creates the characteristic "historically active, recently gone
        quiet" signature of an at-risk customer.

    At least one order is always retained so every customer stays reachable.
    """
    lo = max(tenure_start, START_DATE)
    hi = END_DATE
    span_days = max(1, (hi - lo).days)

    anchors = rng.uniform(0.04, 0.97, size=n_orders)
    # anchors ** tilt with tilt > 1 pulls purchases towards the EARLIER part of
    # the window (median date moves left), which is what declining engagement
    # looks like on a calendar.
    tilt = float(np.clip(1.0 + recency_decay * 0.55 + rng.normal(0.0, 0.10), 0.70, 1.60))
    anchors = anchors ** tilt

    offsets = np.clip((anchors * span_days).astype(int), 0, span_days)

    # de-duplicate dates (one order per day per customer)
    seen: set[int] = set()
    out: list[pd.Timestamp] = []
    for off in offsets:
        o = int(off)
        guard = 0
        while o in seen and guard < span_days:
            o = (o + 1) if o < span_days else (o - 1)
            guard += 1
        if o in seen:
            continue
        seen.add(o)
        out.append(lo + pd.Timedelta(days=o))

    out.sort()

    # --- tail drop: this is what makes declining engagement observable ------
    n_drop = int(np.floor(recency_decay * 0.45 * len(out)))
    n_drop = int(np.clip(n_drop, 0, max(0, len(out) - 1)))
    if n_drop:
        out = out[:-n_drop]

    return out


def _pick_reward(
    rng: np.random.Generator,
    reward_usage_rate: float,
    subtotal: float,
) -> tuple[str, float, float]:
    """Return (reward_type, reward_pct, reward_flat_amount).

    Flat-value rewards are only granted when the order can absorb them without
    an implausible effective discount.  A rupee-denominated reward that is too
    large for the basket is downgraded to a percentage reward instead -- this
    is how a real promotion engine would behave, and it keeps every observed
    effective discount commercially sensible.
    """
    if rng.random() >= reward_usage_rate:
        return "None", 0.0, 0.0

    # rupee rewards must never exceed this share of the basket
    max_flat = MAX_FLAT_REWARD_SHARE * subtotal

    channel = str(rng.choice(
        ["Percentage Discount", "Percentage Discount", "Fixed Discount",
         "Cashback", "BOGO", "Category Coupon"],
        p=[0.26, 0.18, 0.20, 0.20, 0.05, 0.11],
    ))

    if channel == "Percentage Discount":
        return channel, float(rng.choice([5, 10, 15], p=[0.45, 0.35, 0.20])), 0.0
    if channel == "Category Coupon":
        return channel, float(rng.choice([10, 15, 20], p=[0.4, 0.4, 0.2])), 0.0
    if channel == "BOGO":
        return channel, 0.0, 0.0

    ladder = [100, 200, 500] if channel == "Fixed Discount" else [50, 100, 200]
    weights = np.array([0.45, 0.35, 0.20]) if channel == "Fixed Discount" else np.array([0.4, 0.4, 0.2])
    affordable = [v for v in ladder if v <= max_flat]

    if affordable:
        return channel, 0.0, float(rng.choice(affordable, p=weights[: len(affordable)] / weights[: len(affordable)].sum()))

    # basket too small for a rupee reward -> downgrade to a percentage reward
    return "Percentage Discount", 10.0, 0.0


def generate_transactions(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    latent: pd.DataFrame,
    products: pd.DataFrame,
    target_rows: int = 9800,
) -> pd.DataFrame:
    """Generate a behaviourally realistic order-level transaction log."""
    prod_by_cat = {
        c: g.reset_index(drop=True) for c, g in products.groupby("category")
    }
    latent_by_id = latent.set_index("customer_id")
    cat_list = products["category"].unique().tolist()

    # ---- pass 1: plan order dates for every customer ----------------------
    plans: list[tuple[str, pd.Timestamp, pd.Series]] = []
    raw_counts: dict[str, int] = {}

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        lat = latent_by_id.loc[cid]
        dates = _order_dates(
            rng,
            max(1, int(round(lat["_n_orders"]))),
            pd.Timestamp(cust["customer_since"]),
            float(lat["_recency_decay"]),
        )
        raw_counts[cid] = len(dates)
        for d in dates:
            plans.append((cid, d, lat))

    # ---- pass 2: trim surplus so the row count hits the target ------------
    scaled = _scale_orders_to_target(raw_counts, target_rows)
    plans = _filter_plans_to_counts(plans, scaled)

    # ---- pass 3: materialise orders ---------------------------------------
    rows: list[dict] = []
    cat_names = np.array(cat_list, dtype=object)
    for cid, date, lat in plans:
        affinity = np.asarray(lat["_cat_affinity"], dtype=float)
        cat = str(rng.choice(cat_names, p=affinity / affinity.sum()))
        cat_df = prod_by_cat[cat]
        n_in_cat = len(cat_df)
        local_p = _zipf_weights(rng, n_in_cat, 0.45)
        prod = cat_df.iloc[int(rng.choice(n_in_cat, p=local_p))]

        unit_price = float(prod["unit_price"])
        # quantity respects a realistic order-value ceiling
        qty_cap = int(np.clip(ORDER_VALUE_CEILING / unit_price, 1, 6))
        qty = int(np.clip(rng.poisson(max(1.0, 1.6 * lat["_qty_mult"])), 1, qty_cap))
        subtotal = round(qty * unit_price, 2)

        reward_type, reward_pct, reward_flat = _pick_reward(
            rng, float(lat["_reward_usage_rate"]), subtotal
        )

        organic_pct = 0.0
        if rng.random() < lat["_discount_propensity"] * 0.55:
            organic_pct = float(rng.choice([5, 10], p=[0.6, 0.4]))

        pct_part = organic_pct + reward_pct
        total_discount = subtotal * pct_part / 100.0 + reward_flat
        total_discount = min(total_discount, MAX_TOTAL_DISCOUNT_PCT / 100.0 * subtotal)

        # percentage is the single source of truth -> arithmetic is exact
        discount_percentage = round(total_discount / subtotal * 100.0, 2)
        discount_amount = round(subtotal * discount_percentage / 100.0, 2)
        total_amount = round(subtotal - discount_amount, 2)

        rows.append(
            {
                "transaction_id": "",  # assigned after sorting
                "customer_id": cid,
                "transaction_date": date.date().isoformat(),
                "product_id": prod["product_id"],
                "quantity": qty,
                "unit_price": unit_price,
                "discount_percentage": discount_percentage,
                "discount_amount": discount_amount,
                "subtotal": subtotal,
                "total_amount": total_amount,
                "payment_method": str(rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)),
                "reward_used": int(reward_type != "None"),
                "reward_type": reward_type,
            }
        )
        if len(rows) >= target_rows:
            break

    tx = pd.DataFrame(rows)
    tx = tx.sort_values(["transaction_date", "customer_id"]).reset_index(drop=True)
    tx["transaction_id"] = [f"T{i + 1:06d}" for i in range(len(tx))]
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    return tx


def _filter_plans_to_counts(
    plans: list[tuple[str, pd.Timestamp, pd.Series]],
    scaled: dict[str, int],
) -> list[tuple[str, pd.Timestamp, pd.Series]]:
    """Keep at most `scaled[cid]` orders per customer, spread across their dates."""
    allowed = dict(scaled)
    kept: list[tuple[str, pd.Timestamp, pd.Series]] = []
    per_cust: dict[str, list[int]] = {}

    for cid, _date, _lat in plans:
        n = allowed.get(cid, 0)
        if n <= 0:
            continue
        idx = per_cust.get(cid, 0)
        per_cust[cid] = idx + 1
        if idx < n:
            kept.append((cid, _date, _lat))

    return kept


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def generate_dataset(
    output_dir: str,
    n_customers: int = 1500,
    target_transactions: int = 9800,
    seed: int = RANDOM_STATE,
    write_persona_reference: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate and persist customers / products / transactions CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    products = build_products(rng)
    customers, latent = build_customers(rng, n_customers, products)
    transactions = generate_transactions(rng, customers, latent, products, target_transactions)

    products.to_csv(os.path.join(output_dir, "products.csv"), index=False)
    customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    transactions.to_csv(
        os.path.join(output_dir, "transactions.csv"), index=False, date_format="%Y-%m-%d"
    )

    if write_persona_reference:
        ref = latent[["customer_id", "_persona"]].rename(columns={"_persona": "latent_persona"})
        ref.to_csv(os.path.join(output_dir, "persona_reference.csv"), index=False)

    return {"products": products, "customers": customers, "transactions": transactions,
            "_latent": latent}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")
    b = generate_dataset(data_dir)
    print("products    :", b["products"].shape)
    print("customers   :", b["customers"].shape)
    print("transactions:", b["transactions"].shape)