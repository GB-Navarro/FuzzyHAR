"""
mcs_elsm_comparison_test.py — MCS comparison test incluindo o e-LSM
===================================================================

Wrapper fino sobre ``mcs_comparison_test.py`` que **adiciona o e-LSM** ao
conjunto de modelos do teste de comparação do Model Confidence Set
(variantes arch/R, com/sem RV-FTS), para os splits 40_60 / 60_40 / 70_30 —
**sem alterar nenhum arquivo existente**.

A inclusão é feita estendendo, em tempo de execução, as listas de módulo
``ALL_MODELS`` e ``MODELS_NO_RVFTS`` de ``mcs_comparison_test`` com a coluna
``"e-LSM OOS prediction"`` (nome exportado por ``integracao_elsm.py``). As funções
``run_comparison`` / ``save_results`` leem essas listas no momento da chamada,
então passam a considerar o e-LSM automaticamente.

Uso:
    python mcs_elsm_comparison_test.py                     # 3 splits, sem R
    python mcs_elsm_comparison_test.py --splits 70_30
    python mcs_elsm_comparison_test.py --with-r            # inclui variantes R
"""

from __future__ import annotations

import argparse
from typing import List

import mcs_comparison_test as mct


ELSM_COL: str = "e-LSM OOS prediction"
SPLITS: List[str] = ["40_60", "60_40", "70_30"]


def _patch_models() -> None:
    if ELSM_COL not in mct.ALL_MODELS:
        mct.ALL_MODELS = list(mct.ALL_MODELS) + [ELSM_COL]
    mct.MODELS_NO_RVFTS = [m for m in mct.ALL_MODELS if m != "RV-FTS"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCS comparison test incluindo o e-LSM (reusa mcs_comparison_test)."
    )
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 30])
    parser.add_argument("--tickers", nargs="+", default=mct.TICKERS)
    parser.add_argument("--metrics", nargs="+", default=["QLIKE", "MSE"])
    parser.add_argument("--with-r", action="store_true",
                        help="Inclui as variantes R (requer Rscript). Default: pula.")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    _patch_models()

    print("=" * 70)
    print("MCS Comparison Test (com e-LSM)")
    print(f"  Splits:   {args.splits}")
    print(f"  Modelos:  {mct.ALL_MODELS}")
    print(f"  Skip R:   {not args.with_r}")
    print("=" * 70)

    results = mct.run_comparison(
        splits=args.splits,
        horizons=args.horizons,
        tickers=args.tickers,
        metrics=args.metrics,
        skip_r=not args.with_r,
        block_size=args.block_size,
        verbose=not args.quiet,
    )
    mct.save_results(results, skip_r=not args.with_r)


if __name__ == "__main__":
    main()
