# -*- coding: utf-8 -*-
"""Contrato `Metric` — interface de uma metrica de avaliacao.

Qualquer classe que implemente estes atributos/metodos satisfaz o contrato
(duck typing estrutural via Protocol), sem precisar herdar de nada.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Metric(Protocol):
    name: str
    """Nome como aparece nos cabecalhos das tabelas. Ex.: "MSE", "QLIKE"."""

    higher_is_better: bool
    """False para metricas de erro (MSE, QLIKE, MAE, RMSE)."""

    def aggregate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Valor escalar (um numero por modelo) para a tabela de metricas."""
        ...

    def pointwise(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Perda ponto-a-ponto, shape (T,). Obrigatorio para testes estatisticos
        (MCS, Diebold-Mariano), que precisam da sequencia de perdas, nao da media."""
        ...
