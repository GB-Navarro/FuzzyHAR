# -*- coding: utf-8 -*-
"""Runner — orquestra o pipeline chamando os passos verbatim na ordem do notebook.

Reproduz EXATAMENTE o pipeline do notebook: Fuzzy HAR -> HAR -> Merge ->
Metricas -> Plots -> MCS. Cada passo e o codigo verbatim do notebook, apenas
encapsulado em uma funcao run_all_* (ver os modulos gerados).

Uso (com o working dir = pasta FuzzyHAR/, onde estao Inputs/Others/Results):

    from fuzzyhar.pipeline.runner import run_pipeline
    run_pipeline()                      # reproduz os 6 modelos do notebook
    run_pipeline(include_rvfts=True)    # tambem roda o pipeline RV-FTS (stage 2)
    run_pipeline(only=["MCS"])          # roda apenas um passo
"""
from __future__ import annotations

import time
from typing import Iterable, Optional

from fuzzyhar.config import SPLIT
from fuzzyhar.registry import get_pipeline_steps


def run_rvfts_all_splits(splits=None):
    """Conveniencia: roda o RV-FTS para os 3 splits (40_60, 60_40, 70_30),
    reproduzindo `python rvfts_pipeline.py --splits 40_60 60_40 70_30`.
    Import lazy (so carrega arch/rvfts_pipeline quando chamado)."""
    from fuzzyhar.models.rvfts import run_rvfts_all_splits as _impl
    return _impl(splits)


def run_pipeline(
    include_rvfts: bool = False,
    only: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> None:
    only_set = set(only) if only is not None else None
    steps = get_pipeline_steps(include_rvfts=include_rvfts)

    if verbose:
        print(f"[pipeline] SPLIT = {SPLIT}")
        print(f"[pipeline] passos: {[s.name for s in steps]}")

    for step in steps:
        if only_set is not None and step.name not in only_set:
            continue
        if verbose:
            print(f"\n[pipeline] >>> {step.name} (stage {step.stage})")
        t0 = time.time()
        step.run()
        if verbose:
            print(f"[pipeline] <<< {step.name} concluido em {time.time() - t0:.1f}s")

    if verbose:
        print("\n[pipeline] FIM.")


if __name__ == "__main__":
    run_pipeline()
