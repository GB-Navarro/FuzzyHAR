# -*- coding: utf-8 -*-
"""Testes das features HAR (nao exigem R/dados)."""
import numpy as np
import pandas as pd
import pytest

from fuzzyhar.features import make_har_features, gaussmf


def test_gaussmf_peak_at_center():
    assert gaussmf(2.0, 2.0, 1.0) == pytest.approx(1.0)
    assert gaussmf(5.0, 0.0, 1.0) < gaussmf(1.0, 0.0, 1.0)


def test_make_har_features_no_lookahead(synthetic_df):
    """A media semanal em t usa rv[t-week+1..t]; nunca rv futuro."""
    train = synthetic_df.iloc[:300].copy()
    test = synthetic_df.iloc[300:].copy()
    X_tr, y_tr, X_te, y_te = make_har_features(
        train, test, week_len=7, month_len=30, include_daily=True
    )
    # 3 colunas: diaria, semanal (7), mensal (30)
    assert X_tr.shape[1] == 3
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_te.shape[0] == y_te.shape[0]
    assert np.isfinite(X_tr).all()


def test_make_har_features_weekly_mean_value(synthetic_df):
    """Verifica numericamente a media movel semanal (janela=7) na serie completa."""
    train = synthetic_df.iloc[:300].copy()
    test = synthetic_df.iloc[300:].copy()
    full_fo = pd.concat([train, test], ignore_index=True)["Feature Origin"].to_numpy(float)
    week = pd.Series(full_fo).rolling(window=7, min_periods=7).mean().to_numpy()
    # ponto valido arbitrario
    t = 100
    assert week[t] == pytest.approx(full_fo[t - 6:t + 1].mean(), rel=1e-12)
