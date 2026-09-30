"""
AI-Powered Customer Intelligence & Personalised Reward Recommendation
====================================================================

Modules
-------
data_generator       Synthetic department-store data with latent customer
                     behaviour, so a supervised reward model has something
                     realistic to learn from.
feature_engineering  Customer-level feature construction from the observation
                     window only, including RFM and category affinity.
model_training        Synthetic target creation, five-model benchmark, model
                     selection and artifact persistence.
reward_engine        Reward catalogue, per-customer offer optimisation, campaign
                     priority ranking and budget-matched simulation.
visualization        Charts, the architecture diagram and the HTML dashboards.
"""

__all__ = [
    "data_generator",
    "feature_engineering",
    "model_training",
    "reward_engine",
    "visualization",
]

__version__ = "1.0.0"