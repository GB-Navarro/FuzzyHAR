# -*- coding: utf-8 -*-
"""Integridade do registry e da config (nao exigem R/dados)."""
from fuzzyhar import config
from fuzzyhar.registry import MODELS, METRICS, TESTS, MCSSpec


def test_models_have_unique_prediction_cols():
    cols = [m.prediction_col for m in MODELS]
    assert len(cols) == len(set(cols))


def test_metrics_have_required_interface():
    for m in METRICS:
        assert hasattr(m, "name") and hasattr(m, "higher_is_better")
        assert callable(getattr(m, "aggregate")) and callable(getattr(m, "pointwise"))


def test_tests_reference_existing_metrics():
    metric_names = {m.name for m in METRICS}
    for t in TESTS:
        assert isinstance(t, MCSSpec)
        assert t.metric in metric_names


def test_config_values_match_notebook():
    """Trava os valores extraidos do notebook (mudaram => baseline invalido)."""
    assert config.SPLIT in ("40_60", "60_40", "70_30")
    assert config.rules_grid == [2, 3, 4, 5, 6]
    assert config.week_len == 7
    assert config.month_len == 30
    assert config.lamb == 0.99
    assert config.alfa == 1000
    assert config.ze == 1
    assert len(config.SYMBOLS) == 8
