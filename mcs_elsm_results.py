"""
mcs_elsm_results.py — Model Confidence Set incluindo o e-LSM, por split
=======================================================================

Roda o MCS (Model Confidence Set) sobre as planilhas de previsão produzidas por
``integracao_elsm.py``, **incluindo o e-LSM** no conjunto de modelos, para os
splits 40_60 / 60_40 / 70_30 — **sem alterar nenhum arquivo existente**.

Reutiliza integralmente as funções de ``mcs_rvfts_results.py`` (leitura,
construção de perdas, ``arch.bootstrap.MCS``, exportação). A única diferença é o
conjunto de modelos: acrescenta a coluna ``"e-LSM OOS prediction"`` (convenção de
nome dos modelos HAR, usada pela exportação de previsões).

Saída por split:
    Results/HAR_FuzzyHAR_RVFTS_Comparison/{split}/04_mcs/RV_t+{h}/MCS_*.xlsx

Uso:
    python mcs_elsm_results.py                                   # 3 splits, t+1/7/30
    python mcs_elsm_results.py --splits 70_30 --horizons 1
    python mcs_elsm_results.py --confidence 0.90
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import mcs_rvfts_results as mcr


SPLITS: List[str] = ["40_60", "60_40", "70_30"]
ELSM_COL: str = "e-LSM OOS prediction"

# Conjunto de modelos = os 7 originais + e-LSM (todos no formato OOS exportado)
MODELS: List[str] = list(mcr.DEFAULT_MODEL_SUBSET) + [ELSM_COL]
RESULTS_BASE = Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison"


def run_split(
    split: str,
    horizons: List[int],
    *,
    confidence: float = 0.75,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    verbose: bool = True,
) -> Dict[int, object]:
    results_root = RESULTS_BASE / split
    output_dir = results_root / "04_mcs"

    print("=" * 80)
    print(f"MCS (com e-LSM) | split={split} | root={results_root}")
    print(f"  Modelos ({len(MODELS)}): {MODELS}")
    print("=" * 80)

    outputs: Dict[int, object] = {}
    for horizon in horizons:
        try:
            outputs[horizon] = mcr.run_mcs_for_horizon(
                horizon=horizon,
                results_root=results_root,
                output_dir=output_dir,
                tickers=None,
                model_subset=MODELS,
                filter_cols=MODELS,      # filtra <=0 -> ffill também para o e-LSM
                confidence=confidence,
                B=B,
                statistic=statistic,
                seed=seed,
                verbose=verbose,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] split={split} t+{horizon}: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[ERRO] split={split} t+{horizon}: {e}")
            traceback.print_exc()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCS incluindo o e-LSM, por split (reusa mcs_rvfts_results)."
    )
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 30])
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--B", type=int, default=10000)
    parser.add_argument("--statistic", default="max")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    for split in args.splits:
        run_split(
            split,
            args.horizons,
            confidence=args.confidence,
            B=args.B,
            statistic=args.statistic,
            seed=args.seed,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
