"""
Combined verification for every project stage.

Run:  python scripts\check_all.py

This is the script run_everything.bat's steps are built on. Run it any time
you want to confirm the whole project is healthy.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

PASS = []
FAIL = []


def check(label, cond, extra=""):
    cond = bool(cond)
    (PASS if cond else FAIL).append(label)
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("  " + extra if extra else ""))


def section(title):
    print()
    print("-" * 64)
    print("  " + title)
    print("-" * 64)


def main():
    t0 = time.time()
    print("=" * 64)
    print("  PROJECT VERIFICATION")
    print("=" * 64)

    # ------------------------------------------------------------------ data
    section("1. DATA FILES")
    need = ["data/customers.csv", "data/products.csv", "data/transactions.csv"]
    for f in need:
        check(f"exists: {f}", os.path.isfile(f),
              f"({os.path.getsize(f) / 1024:,.0f} KB)" if os.path.isfile(f) else "")

    t = pd.read_csv("data/transactions.csv", parse_dates=["transaction_date"],
                    keep_default_na=False, na_values=[""])
    c = pd.read_csv("data/customers.csv", parse_dates=["customer_since"])
    p = pd.read_csv("data/products.csv")

    check("transactions rows in 8,000-10,000", 8000 <= len(t) <= 10000, f"({len(t):,})")
    check("customers in 1,500-2,500", 1500 <= len(c) <= 2500, f"({len(c):,})")
    check("products in 60-100", 60 <= len(p) <= 100, f"({len(p):,})")
    check("no missing values", t.isna().sum().sum() == 0)
    check("transaction_id unique", t.transaction_id.is_unique)
    check("no orphan customer_id", t.customer_id.isin(c.customer_id).all())
    check("no orphan product_id", t.product_id.isin(p.product_id).all())
    check("subtotal = qty x price",
          np.allclose(t.subtotal, t.quantity * t.unit_price))
    check("discount_amount correct",
          np.allclose(t.discount_amount, t.subtotal * t.discount_percentage / 100, atol=0.011))
    check("total = subtotal - discount",
          np.allclose(t.total_amount, t.subtotal - t.discount_amount, atol=0.011))
    check("dates within 2025",
          t.transaction_date.min() >= pd.Timestamp("2025-01-01")
          and t.transaction_date.max() <= pd.Timestamp("2025-12-31"))

    # ---------------------------------------------------------------- models
    section("2. SAVED MODEL")
    for f in ["models/best_reward_model.pkl", "models/preprocessing_pipeline.pkl",
              "models/model_metadata.pkl"]:
        check(f"exists: {f}", os.path.isfile(f),
              f"({os.path.getsize(f) / 1024:,.1f} KB)" if os.path.isfile(f) else "")

    if os.path.isfile("models/model_metadata.pkl"):
        import pickle
        with open("models/model_metadata.pkl", "rb") as fh:
            meta = pickle.load(fh)
        check("metadata has best model", "best_model" in meta,
              f"({meta.get('best_model', '?')})")
        check("metadata has metrics", "metrics" in meta,
              f"({len(meta.get('metrics', {}))} models)")
        check("metadata records feature list", len(meta.get("feature_columns", [])) > 0,
              f"({len(meta.get('feature_columns', []))} features)")

    # ---------------------------------------------------------------- report
    section("3. OUTPUTS")
    for f in ["outputs/report/project_report.html"]:
        check(f"exists: {f}", os.path.isfile(f),
              f"({os.path.getsize(f) / 1024:,.0f} KB)" if os.path.isfile(f) else "")

    figs = [f for f in os.listdir("outputs/figures") if f.endswith(".png")] \
        if os.path.isdir("outputs/figures") else []
    check("figures generated", len(figs) >= 10, f"({len(figs)} PNGs)")

    reports = [f for f in os.listdir("outputs/reports") if f.endswith(".csv")] \
        if os.path.isdir("outputs/reports") else []
    check("report CSVs generated", len(reports) >= 4, f"({len(reports)} CSVs)")

    # ---------------------------------------------------------------- notebook
    section("4. NOTEBOOK")
    nb_path = "notebooks/customer_reward_ml.ipynb"
    check("notebook exists", os.path.isfile(nb_path),
          f"({os.path.getsize(nb_path) / 1024:,.0f} KB)" if os.path.isfile(nb_path) else "")

    if os.path.isfile(nb_path):
        import nbformat
        nb = nbformat.read(nb_path, as_version=4)
        n_code = sum(1 for x in nb.cells if x.cell_type == "code")
        n_md = sum(1 for x in nb.cells if x.cell_type == "markdown")
        n_out = sum(len(x.get("outputs", [])) for x in nb.cells if x.cell_type == "code")
        check("notebook has code cells", n_code >= 40, f"({n_code} code, {n_md} markdown)")
        check("notebook was executed", n_out > 50, f"({n_out} saved outputs)")
        errors = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for o in cell.get("outputs", []):
                if o.get("output_type") == "error":
                    errors.append(o.get("ename", "?"))
        check("no saved errors in notebook", len(errors) == 0,
              f"({len(errors)})" if errors else "")

    # ---------------------------------------------------------------- scripts
    section("5. ENTRY POINTS")
    for f in ["run_everything.bat", "run_pipeline.py", "README.md", "START_HERE.md",
              "src/report_builder.py"]:
        check(f"exists: {f}", os.path.isfile(f))

    for mod in ["data_generator", "feature_engineering", "model_training",
                "reward_engine", "visualization", "report_builder"]:
        check(f"src/{mod}.py importable", _importable(mod))

    # ---------------------------------------------------------------- summary
    print()
    print("=" * 64)
    print(f"  RESULT: {len(PASS)} passed, {len(FAIL)} failed "
          f"[{time.time() - t0:.1f}s]")
    print("=" * 64)
    if FAIL:
        print()
        print("  FAILED CHECKS:")
        for x in FAIL:
            print("    - " + x)
        print()
        print("  Fix: re-run  python src\\data_generator.py  then  run_everything.bat")
        return 1
    print()
    print("  Everything is in order. Open outputs\\report\\project_report.html")
    print("  to see the presentation.")
    print()
    return 0


def _importable(mod):
    try:
        __import__("src." + mod)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
