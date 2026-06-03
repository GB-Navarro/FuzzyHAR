# -*- coding: utf-8 -*-
"""Implementacoes de metricas.

IMPORTANTE (risco R2 do plano): o notebook define `mse`/`qlike` de TRES formas
diferentes, em lugares diferentes, com formulas e tratamento de NaN distintos.
Elas NAO sao equivalentes e NAO podem ser fundidas sem mudar resultados:

  * Reporte (celula 52, tabelas MSE & QLIKE):
        qlike = mean(r - log(r) - 1), r = y/f, com dropna no par; eps=1e-12
        mse   = mean((y-f)^2), com dropna no par
  * Perda ponto-a-ponto (celula 56, entra no MCS):
        qlike_loss_t = r - log(r) - 1   (sem dropna)
        mse_loss_t   = (y-f)^2
  * Bloco HAR / graficos OOS (celula 47):
        qlike_har = mean(log(f_pos) + y_pos/f_pos), EPS=1e-12
        mse_har   = mean((y-f)^2)   (sem dropna)

As funcoes abaixo reproduzem cada variante VERBATIM. As classes `Metric`
registradas no `registry` usam as variantes de REPORTE (aggregate) e de PERDA
(pointwise), que sao as que alimentam as tabelas e o MCS — exatamente como hoje.

Estas funcoes NAO estao no caminho critico que gera os arquivos Excel atuais
(esses vem do codigo verbatim em evaluation/metrics_tables.py e evaluation/mcs.py).
Elas existem para (a) habilitar a adicao facil de novas metricas e (b) os testes
de propriedade. Sao copias fieis das mesmas formulas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Variantes de REPORTE (celula 52) — usam dropna par-a-par
# ---------------------------------------------------------------------------
def qlike_report(y_real, y_pred, eps: float = 1e-12) -> float:
    pair = pd.concat({"x": pd.Series(y_real).astype(float),
                      "h": pd.Series(y_pred).astype(float)}, axis=1).dropna()
    x = np.clip(pair["x"].to_numpy(), eps, None)
    h = np.clip(pair["h"].to_numpy(), eps, None)
    r = x / h
    return float(np.mean(r - np.log(r) - 1.0))


def mse_report(y_real, y_pred) -> float:
    pair = pd.concat({"y": pd.Series(y_real).astype(float),
                      "h": pd.Series(y_pred).astype(float)}, axis=1).dropna()
    diff = pair["y"].to_numpy() - pair["h"].to_numpy()
    return float(np.mean(diff ** 2))


# ---------------------------------------------------------------------------
# Variantes de PERDA ponto-a-ponto (celula 56) — entram no MCS
# ---------------------------------------------------------------------------
def qlike_loss_t(y_real: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(np.asarray(y_real).astype(float), eps, None)
    h = np.clip(np.asarray(y_pred).astype(float), eps, None)
    r = x / h
    return r - np.log(r) - 1.0


def mse_loss_t(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    d = np.asarray(y_real).astype(float) - np.asarray(y_pred).astype(float)
    return d ** 2


# ---------------------------------------------------------------------------
# Variantes do bloco HAR / graficos OOS (celula 47) — preservadas para fidelidade
# ---------------------------------------------------------------------------
_EPS_HAR = 1e-12


def qlike_har(y, f) -> float:
    y_pos = np.clip(np.asarray(y, float), _EPS_HAR, None)
    f_pos = np.clip(np.asarray(f, float), _EPS_HAR, None)
    return float(np.mean(np.log(f_pos) + y_pos / f_pos))


def mse_har(y, f) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2))


def rmse_raw(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2)))


def mae_raw(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


# ---------------------------------------------------------------------------
# Classes Metric (interface de extensao). aggregate = reporte; pointwise = perda.
# ---------------------------------------------------------------------------
class MSE:
    name = "MSE"
    higher_is_better = False

    def aggregate(self, y_true, y_pred) -> float:
        return mse_report(y_true, y_pred)

    def pointwise(self, y_true, y_pred) -> np.ndarray:
        return mse_loss_t(y_true, y_pred)


class QLIKE:
    name = "QLIKE"
    higher_is_better = False

    def aggregate(self, y_true, y_pred) -> float:
        return qlike_report(y_true, y_pred)

    def pointwise(self, y_true, y_pred) -> np.ndarray:
        return qlike_loss_t(y_true, y_pred)


class RMSE:
    name = "RMSE"
    higher_is_better = False

    def aggregate(self, y_true, y_pred) -> float:
        return rmse_raw(y_true, y_pred)

    def pointwise(self, y_true, y_pred) -> np.ndarray:
        return np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))


class MAE:
    name = "MAE"
    higher_is_better = False

    def aggregate(self, y_true, y_pred) -> float:
        return mae_raw(y_true, y_pred)

    def pointwise(self, y_true, y_pred) -> np.ndarray:
        return np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
