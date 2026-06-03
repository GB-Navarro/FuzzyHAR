# -*- coding: utf-8 -*-
"""Testes de propriedade das metricas (nao exigem R/dados)."""
import numpy as np
import pytest

from fuzzyhar.metrics.implementations import (
    MAE, MSE, QLIKE, RMSE,
    qlike_report, qlike_har,
)


def test_mse_zero_on_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert MSE().aggregate(y, y) == pytest.approx(0.0, abs=1e-12)


def test_qlike_report_zero_on_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert QLIKE().aggregate(y, y) == pytest.approx(0.0, abs=1e-12)


def test_rmse_equals_sqrt_mse(rv_series):
    f = rv_series * 1.02
    assert RMSE().aggregate(rv_series, f) == pytest.approx(
        np.sqrt(MSE().aggregate(rv_series, f)), rel=1e-9
    )


def test_pointwise_shape_matches_input(rv_series):
    f = rv_series * 1.01
    for metric in (MSE(), QLIKE(), RMSE(), MAE()):
        assert metric.pointwise(rv_series, f).shape == rv_series.shape


def test_mse_aggregate_equals_mean_pointwise(rv_series):
    f = rv_series * 1.01
    m = MSE()
    assert m.aggregate(rv_series, f) == pytest.approx(m.pointwise(rv_series, f).mean(), rel=1e-9)


def test_qlike_pointwise_mean_matches_aggregate_without_nan(rv_series):
    """Sem NaN, mean(qlike_loss_t) deve bater com qlike_report."""
    f = rv_series * 1.03
    q = QLIKE()
    assert q.aggregate(rv_series, f) == pytest.approx(q.pointwise(rv_series, f).mean(), rel=1e-9)


def test_qlike_variants_are_distinct(rv_series):
    """Documenta R2: a QLIKE de reporte (cel. 52) e a do bloco HAR (cel. 47)
    NAO sao a mesma formula. Este teste falha se alguem tentar fundi-las."""
    f = rv_series * 1.03
    assert qlike_report(rv_series, f) != pytest.approx(qlike_har(rv_series, f), rel=1e-6)
