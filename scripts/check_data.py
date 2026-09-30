"""
STEP 2 — DATA VALIDATION
Run: python scripts\check_data.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

t = pd.read_csv('data/transactions.csv', parse_dates=['transaction_date'],
                keep_default_na=False, na_values=[''])
c = pd.read_csv('data/customers.csv', parse_dates=['customer_since'])
p = pd.read_csv('data/products.csv')

print('=' * 60)
print('  STEP 2: DATA VALIDATION')
print('=' * 60)

ok = 0; total = 0
def chk(label, cond):
    global ok, total
    total += 1; ok += bool(cond)
    print(('  [PASS] ' if cond else '  [FAIL] ') + label)

chk('transactions.csv loaded : {} rows'.format(len(t)), len(t) > 0)
chk('customers.csv loaded    : {} rows'.format(len(c)), len(c) > 0)
chk('products.csv loaded     : {} rows'.format(len(p)), len(p) > 0)
print()
chk('no missing values       : {}'.format(int(t.isna().sum().sum())), t.isna().sum().sum() == 0)
chk('transaction_id unique   : {}'.format(t.transaction_id.is_unique), t.transaction_id.is_unique)
chk('no orphan customer_id   : {}'.format((~t.customer_id.isin(c.customer_id)).sum()),
    t.customer_id.isin(c.customer_id).all())
chk('no orphan product_id    : {}'.format((~t.product_id.isin(p.product_id)).sum()),
    t.product_id.isin(p.product_id).all())
print()
chk('subtotal = qty * price  : {}'.format(np.allclose(t.subtotal, t.quantity*t.unit_price)),
    np.allclose(t.subtotal, t.quantity*t.unit_price))
chk('discount_amount correct : {}'.format(np.allclose(t.discount_amount, t.subtotal*t.discount_percentage/100, atol=0.011)),
    np.allclose(t.discount_amount, t.subtotal*t.discount_percentage/100, atol=0.011))
chk('total = subtotal - disc : {}'.format(np.allclose(t.total_amount, t.subtotal-t.discount_amount, atol=0.011)),
    np.allclose(t.total_amount, t.subtotal-t.discount_amount, atol=0.011))
print()
print('  DATE RANGE : {} -> {}'.format(t.transaction_date.min().date(), t.transaction_date.max().date()))
print('  REVENUE    : Rs {:,.0f}'.format(t.total_amount.sum()))
print('  AVERAGE    : Rs {:,.0f} per order'.format(t.total_amount.mean()))
print('  CATEGORIES : {}'.format(p.category.nunique()))
print('  REWARD USE : {:.1f}% of orders'.format(t.reward_used.mean()*100))
print()
print('=' * 60)
print('  RESULT: {}/{} checks passed'.format(ok, total))
print('=' * 60)
