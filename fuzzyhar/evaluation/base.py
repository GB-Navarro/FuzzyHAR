# -*- coding: utf-8 -*-
"""Contrato `ComparisonTest` — interface de um teste estatistico de comparacao."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

import pandas as pd


@dataclass
class TestResult:
    included: List[str]
    """Modelos que sobreviveram ao teste (ex.: modelos no MCS)."""

    excluded: List[str]
    """Modelos eliminados pelo teste."""

    pvalues: Dict[str, float] = field(default_factory=dict)
    """P-valores por modelo, quando o teste os produz. Vazio para o MCS."""

    raw: Any = None
    """Objeto nativo da biblioteca (ex.: arch.bootstrap.MCS), para diagnostico."""


@runtime_checkable
class ComparisonTest(Protocol):
    name: str
    """Nome do teste (aparece em pastas/arquivos de saida). Ex.: "MCS-75pct"."""

    metric: str
    """Nome da Metric usada para construir a matriz de perdas. Ex.: "QLIKE"."""

    def run(self, losses: pd.DataFrame, **kwargs) -> TestResult:
        """Executa o teste sobre a matriz T x k de perdas ponto-a-ponto."""
        ...
