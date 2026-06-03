# -*- coding: utf-8 -*-
"""Contratos de modelo.

Ha DOIS niveis de granularidade neste projeto, por razao historica:

1. `PipelineStep` (usado HOJE) — a arquitetura real do notebook executa cada
   familia de modelos em LOTE (um loop sobre todos os simbolos/horizontes),
   nesta ordem: Fuzzy HAR -> HAR -> merge -> metricas -> plots -> MCS.
   Para reproduzir os resultados EXATOS, o runner chama esses passos em lote
   (as funcoes `run_all_*` geradas verbatim do notebook). Cada passo e um
   `PipelineStep`. Adicionar um modelo novo nesse esquema = criar seu modulo
   com uma funcao `run_all_x()` e registrar um `PipelineStep` no registry.

2. `Forecaster` (interface ALVO, para modelos futuros de granularidade fina) —
   contrato (symbol, horizon, split) -> (df_train, df_test). E o destino de
   manutencao descrito no plano; modelos novos podem adota-lo e ser orquestrados
   por um runner generico DEPOIS de validados contra o baseline (ver plano, R4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class PipelineStep:
    """Um passo em lote do pipeline (granularidade real do notebook)."""
    name: str
    stage: int                 # ordem de execucao (stages menores rodam antes)
    run: Callable[[], object]  # funcao verbatim run_all_* (sem argumentos)
    description: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """Descricao declarativa de um modelo (para tabelas, plots, MCS e docs)."""
    name: str                  # ex.: "Fuzzy HAR"
    prediction_col: str        # coluna que o modelo adiciona, ex.: "HAR OOS prediction"
    color: str                 # cor hex para plots
    stage: int                 # 1 = le dados brutos; 2 = depende de saidas do stage 1


@runtime_checkable
class Forecaster(Protocol):
    """Interface ALVO de granularidade fina (ver docstring do modulo)."""
    name: str
    prediction_col: str
    color: str
    stage: int

    def run(self, symbol: str, horizon: int, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...
