# -*- coding: utf-8 -*-
"""Adaptador do RV-FTS (stage 2).

O RV-FTS vive em `fuzzyhar/models/rvfts_pipeline.py` (movido para dentro do pacote,
codigo INALTERADO). Este wrapper apenas chama a funcao existente
`rvfts_pipeline.run_full_pipeline(...)`, sem sobrescrever nenhum hiperparametro —
ou seja, herda EXATAMENTE os mesmos defaults do CLI
(`ew=756, cs=252, nclus=[2,3,4,6,9], rho=[...], fcm_seed=42`, MCS conf=0.75,
B=10000, statistic="max", seed=123). Por isso reproduz o `python rvfts_pipeline.py`.

RV-FTS depende das saidas de stage 1 (Fuzzy HAR e HAR) ja escritas em
`Others/Predictions/{split}/`; por isso roda como stage 2 e LE esses arquivos
do disco. Dado os mesmos arquivos de stage 1 + as mesmas sementes, o resultado e
deterministico.

Tres formas de uso:
  * run_rvfts_for_split(split) -> roda um split especifico.
  * run_all_rvfts()            -> roda o split atual (fuzzyhar.config.SPLIT);
                                  e o passo usado pelo runner integrado.
  * run_rvfts_all_splits()     -> roda 40_60, 60_40 e 70_30, reproduzindo
                                  EXATAMENTE `python rvfts_pipeline.py --splits 40_60 60_40 70_30`.
"""
from __future__ import annotations

from typing import Iterable, Optional

from fuzzyhar.models import rvfts_pipeline  # agora dentro do pacote (codigo inalterado)

from fuzzyhar.config import SPLIT


def run_rvfts_for_split(split: str):
    """Roda o pipeline RV-FTS completo para UM split (parametros canonicos do
    rvfts_pipeline.py — nao alterados). Identico a `... --splits {split}`."""
    return rvfts_pipeline.run_full_pipeline(split=split)


def run_all_rvfts():
    """Passo do runner integrado: roda o RV-FTS para o SPLIT atual, mantendo
    consistencia com o stage 1 (Fuzzy HAR/HAR) gerado para o mesmo split."""
    return run_rvfts_for_split(SPLIT)


def run_rvfts_all_splits(splits: Optional[Iterable[str]] = None) -> dict:
    """Roda o RV-FTS para varios splits, reproduzindo o CLI completo.

    Com `splits=None`, usa `rvfts_pipeline.VALID_SPLITS` = ['40_60','60_40','70_30'],
    ou seja, e o equivalente exato de:
        python rvfts_pipeline.py --splits 40_60 60_40 70_30
    (mesma funcao, mesmo loop, mesmos defaults). Le os arquivos de stage 1 que
    ja estao no disco para cada split.
    """
    if splits is None:
        splits = list(rvfts_pipeline.VALID_SPLITS)
    return {s: run_rvfts_for_split(s) for s in splits}
