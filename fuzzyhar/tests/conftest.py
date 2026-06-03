# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rv_series() -> np.ndarray:
    """Serie sintetica de RV (positiva) para testes de propriedade."""
    rng = np.random.default_rng(42)
    T = 500
    rv = np.zeros(T)
    rv[0] = 1.0
    for t in range(1, T):
        rv[t] = 0.9 * rv[t - 1] + 0.1 * rng.gamma(2.0, 0.5)
    return np.maximum(rv, 1e-4)


@pytest.fixture
def synthetic_df(rv_series) -> pd.DataFrame:
    T = len(rv_series)
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=T, freq="D"),
        "Feature Origin": rv_series,
        "Target": np.roll(rv_series, -1),
    })
