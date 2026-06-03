# -*- coding: utf-8 -*-
"""Registry — o unico lugar para cadastrar modelos, metricas e testes.

Para ADICIONAR um componente novo, edite SOMENTE este arquivo:
  * novo modelo  -> crie seu modulo com uma funcao run_all_x() e adicione um
                    PipelineStep em `get_pipeline_steps()` + um ModelSpec em MODELS;
  * nova metrica -> implemente a classe em metrics/implementations.py e adicione a METRICS;
  * novo teste   -> adicione um MCSSpec (ou outro) em TESTS.

`MODELS`, `METRICS` e `TESTS` sao dados puros (nao importam rpy2/R), de modo que
podem ser inspecionados/testados sem o ambiente R. Os passos executaveis
(`get_pipeline_steps`) so importam os modulos pesados quando chamados.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fuzzyhar.models.base import ModelSpec, PipelineStep
from fuzzyhar.metrics.implementations import MAE, MSE, QLIKE, RMSE


# ════════════════════════════════════════════════════════════════════════════
# MODELOS (descricao declarativa — cores/colunas/stage)
# ════════════════════════════════════════════════════════════════════════════
MODELS: List[ModelSpec] = [
    ModelSpec("Fuzzy HAR", "Fuzzy HAR",                "#8c564b", stage=1),
    ModelSpec("HAR",       "HAR OOS prediction",       "#1f77b4", stage=1),
    ModelSpec("HAR-CJ",    "HAR-CJ OOS prediction",    "#ff7f0e", stage=1),
    ModelSpec("HAR-SJ",    "HAR-SJ OOS prediction",    "#2ca02c", stage=1),
    ModelSpec("HAR-TCJ",   "HAR-TCJ OOS prediction",   "#d62728", stage=1),
    ModelSpec("LHAR-TCJ",  "LHAR-TCJ OOS prediction",  "#9467bd", stage=1),
    ModelSpec("RV-FTS",    "RV-FTS",                   "#e377c2", stage=2),
]


# ════════════════════════════════════════════════════════════════════════════
# METRICAS (interface de extensao; aggregate=reporte, pointwise=perda)
# ════════════════════════════════════════════════════════════════════════════
METRICS = [MSE(), QLIKE(), RMSE(), MAE()]


# ════════════════════════════════════════════════════════════════════════════
# TESTES estatisticos (MCS) — parametros identicos ao notebook (celula 56)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MCSSpec:
    name: str
    metric: str          # "QLIKE" ou "MSE"
    confidence: float = 0.75
    B: int = 10000
    statistic: str = "max"
    seed: int = 123


TESTS: List[MCSSpec] = [
    MCSSpec("MCS-75pct-QLIKE", "QLIKE"),
    MCSSpec("MCS-75pct-MSE",   "MSE"),
]


# ════════════════════════════════════════════════════════════════════════════
# PASSOS DO PIPELINE (ordem EXATA do notebook — reproduz os resultados atuais)
# ════════════════════════════════════════════════════════════════════════════
def get_pipeline_steps(include_rvfts: bool = False) -> List[PipelineStep]:
    """Retorna os passos do pipeline na ordem de execucao do notebook.

    Imports sao locais (lazy) porque alguns modulos carregam rpy2 e definem
    a funcao R no import (ver models/r_gk.py).
    """
    from fuzzyhar.models.fuzzy_har import run_all_fuzzy
    from fuzzyhar.models.har import run_all_har
    from fuzzyhar.pipeline.merge import run_all_merge
    from fuzzyhar.evaluation.metrics_tables import run_all_metrics
    from fuzzyhar.pipeline.plots import run_all_plots
    from fuzzyhar.evaluation.mcs import run_all_mcs

    steps: List[PipelineStep] = [
        PipelineStep("Fuzzy HAR", 1, run_all_fuzzy,
                     "Treina/testa o Fuzzy HAR (GK em R) para todos os simbolos/horizontes."),
        PipelineStep("HAR variants", 1, run_all_har,
                     "Treina/testa HAR, HAR-CJ, HAR-SJ, HAR-TCJ, LHAR-TCJ (OOS)."),
        PipelineStep("Merge", 2, run_all_merge,
                     "Une previsoes classicas + Fuzzy por simbolo/horizonte."),
        PipelineStep("Metrics (MSE & QLIKE)", 3, run_all_metrics,
                     "Tabelas de MSE e QLIKE por horizonte."),
        PipelineStep("Plots", 3, run_all_plots,
                     "Graficos finais OOS (Actual vs HAR vs Fuzzy)."),
        PipelineStep("MCS", 4, run_all_mcs,
                     "Model Confidence Set (QLIKE e MSE), confianca 75%."),
    ]

    if include_rvfts:
        # RV-FTS (stage 2) depende das saidas de stage 1; roda apos os modelos HAR.
        from fuzzyhar.models.rvfts import run_all_rvfts
        steps.insert(2, PipelineStep("RV-FTS", 2, run_all_rvfts,
                     "Pipeline RV-FTS completo (rvfts_pipeline.py, inalterado)."))

    return steps
