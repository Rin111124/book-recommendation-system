"""Reusable modules for the Book Recommendation System.

Avoid importing heavy submodules (matrix_factorization/Surprise, CF models)
at package import time to prevent unnecessary NumPy ABI-triggered imports.

Use explicit imports instead, e.g.::

    import src.data_preprocessing as data_preprocessing
    from src.collaborative_filtering import UserBasedCF

Optionally, helper accessors are provided for convenience.
"""

def load_data_preprocessing():  # lazy accessor
    from . import data_preprocessing
    return data_preprocessing

def load_cf():  # lazy accessor for collaborative filtering
    from . import collaborative_filtering
    return collaborative_filtering

__all__ = [
    "load_data_preprocessing",
    "load_cf",
]
