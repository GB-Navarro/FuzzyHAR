"""
integracao_elsm.py — comparação HAR / Fuzzy HAR / RV-FTS / **e-LSM** por split
==============================================================================

Este script gera os resultados comparativos completos incluindo o **e-LSM**,
para os splits 40_60, 60_40 e 70_30, **sem alterar nenhum arquivo existente**.

Como funciona
-------------
- Reutiliza, por importação, todo o pipeline já validado em
  ``integracao_rvfts.py`` (carregamento dos dados do notebook, execução do
  RV-FTS, cálculo de MSE/QLIKE, exportação de planilhas/previsões/gráficos).
- Acrescenta o e-LSM (via ``elsm_model.add_elsm_to_dataframes``) a cada par
  (treino, teste), de forma **puramente aditiva**: HAR (5), Fuzzy HAR e RV-FTS
  permanecem inalterados; o e-LSM entra como mais um modelo.
- Torna o pipeline *split-aware* ajustando, **em tempo de execução**, as
  raízes de pasta que ``integracao_rvfts`` lê/escreve (atributos de módulo —
  não editamos o arquivo).

Convenção de nomes de coluna
----------------------------
O e-LSM usa ``"e-LSM prediction"`` (treino) e ``"e-LSM OOS prediction"`` (teste),
seguindo a convenção dos modelos HAR. Assim ``integracao_rvfts._insample_col`` e
``_oos_col`` já resolvem o nome automaticamente, sem qualquer alteração lá.

Saída
-----
``Results/HAR_FuzzyHAR_RVFTS_Comparison/{split}/...`` — exatamente o caminho que
``mcs_rvfts_results.py`` e ``mcs_comparison_test.py`` esperam.

Uso
---
    python integracao_elsm.py                      # 3 splits, grid completo
    python integracao_elsm.py --splits 70_30       # só um split
    python integracao_elsm.py --quick              # grids reduzidos (teste rápido)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import integracao_rvfts as base
import elsm_model
from elsm_model import DEFAULT_ELSM_GRID, add_elsm_to_dataframes, cfg_to_row


SPLITS: List[str] = ["40_60", "60_40", "70_30"]
ELSM_COLOR: str = "#2E7D32"   # verde (mesma cor do e-LSM no notebook)

_PATCHED = False


# ---------------------------------------------------------------------------
# Patches em tempo de execução (apenas dados de módulo — não editam arquivos)
# ---------------------------------------------------------------------------

def _patch_base_for_elsm() -> None:
    """Estende MODEL_ORDER / MODEL_COLORS / filtros de integracao_rvfts com e-LSM."""
    global _PATCHED
    if _PATCHED:
        return
    if "e-LSM" not in base.MODEL_ORDER:
        base.MODEL_ORDER = list(base.MODEL_ORDER) + ["e-LSM"]
    base.MODEL_COLORS = {**base.MODEL_COLORS, "e-LSM": ELSM_COLOR}
    if elsm_model.ELSM_OOS_COL not in base._FUZZY_PRED_COLS:
        base._FUZZY_PRED_COLS = list(base._FUZZY_PRED_COLS) + [elsm_model.ELSM_OOS_COL]
    _PATCHED = True


def _set_split_paths(split: str) -> None:
    """Aponta as raízes de leitura/escrita de integracao_rvfts para o split dado."""
    base.FUZZY_ROOT = Path("Others") / "Predictions" / split / "Fuzzy HAR"
    base.HAR_MODELS_ROOT = Path("Others") / "Predictions" / split / "HAR Models"
    base.OUTPUT_ROOT = Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison" / split


# ---------------------------------------------------------------------------
# Exportação dos hiperparâmetros escolhidos pelo grid search
# ---------------------------------------------------------------------------

def _save_elsm_hyperparams(
    elsm_info: Dict[int, Dict[str, Dict[str, Any]]],
    horizons: Iterable[int],
) -> None:
    for horizon in horizons:
        rows = [cfg_to_row(sym, horizon, info)
                for sym, info in elsm_info.get(horizon, {}).items()]
        if not rows:
            continue
        out_dir = base.OUTPUT_ROOT / f"RV_t+{horizon}" / "01_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"elsm_hyperparams_tplus{horizon}.xlsx"
        pd.DataFrame(rows).to_excel(path, index=False)
        print(f"[salvo] {path}")


# ---------------------------------------------------------------------------
# Pipeline por split
# ---------------------------------------------------------------------------

def run_split(
    split: str,
    *,
    symbols: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    elsm_grid: Optional[List[Dict[str, Any]]] = DEFAULT_ELSM_GRID,
    verbose: bool = True,
) -> Tuple[dict, dict, dict]:
    """Roda o pipeline completo (HAR + Fuzzy HAR + RV-FTS + e-LSM) para um split."""
    symbols = list(symbols) if symbols is not None else list(base.SYMBOLS)
    horizons = list(horizons) if horizons is not None else list(base.HORIZONS)

    _patch_base_for_elsm()
    _set_split_paths(split)

    print("#" * 70)
    print(f"# SPLIT {split}  |  in={base.FUZZY_ROOT}  ->  out={base.OUTPUT_ROOT}")
    print("#" * 70)

    # 1) HAR + Fuzzy HAR + RV-FTS (reuso integral de integracao_rvfts)
    print("\n[1/4] Carregando dados e executando RV-FTS (HAR + Fuzzy HAR já vêm do notebook)...")
    data = base.build_all_data(
        symbols=symbols, horizons=horizons,
        nclus_grid=nclus_grid, rho_grid=rho_grid, verbose=verbose,
    )

    # 2) e-LSM (aditivo)
    print("\n[2/4] Executando e-LSM (grid search no treino)...")
    elsm_info: Dict[int, Dict[str, Dict[str, Any]]] = {h: {} for h in horizons}
    for horizon in horizons:
        for sym in symbols:
            if sym not in data.get(horizon, {}):
                continue
            df_tr = data[horizon][sym]["train"]
            df_te = data[horizon][sym]["test"]
            try:
                df_tr, df_te, info = add_elsm_to_dataframes(
                    df_tr, df_te, horizon, grid=elsm_grid, verbose=verbose,
                )
                data[horizon][sym]["train"] = df_tr
                data[horizon][sym]["test"] = df_te
                elsm_info[horizon][sym] = info
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"[e-LSM ERRO] {sym} t+{horizon}: {e}")
                traceback.print_exc()

    # 3) Métricas + exportações (reuso integral de integracao_rvfts)
    print("\n[3/4] Calculando métricas e salvando planilhas/previsões/gráficos...")
    tables = base.compute_metrics_table(data, horizons=horizons, symbols=symbols)
    base.save_overview_spreadsheets(tables, horizons=horizons)
    base.save_metric_spreadsheets(tables, horizons=horizons)
    base.save_prediction_spreadsheets(data, horizons=horizons, symbols=symbols)
    base.save_all_plots(data, tables, horizons=horizons, symbols=symbols)

    # 4) Hiperparâmetros do grid search
    print("\n[4/4] Salvando hiperparâmetros do e-LSM escolhidos pelo grid...")
    _save_elsm_hyperparams(elsm_info, horizons)

    # Resumo no console
    for horizon in horizons:
        df = tables[horizon]["oos"]["MSE"]
        if df is None or df.empty:
            print(f"\n--- {split} | t+{horizon} | OOS | MSE: (sem dados) ---")
        else:
            print(f"\n--- {split} | t+{horizon} | OOS | MSE ---")
            print(df.to_string(float_format="{:.4e}".format))

    print(f"\nConcluído split {split}. Resultados em: {base.OUTPUT_ROOT.resolve()}")
    return data, tables, elsm_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparação HAR / Fuzzy HAR / RV-FTS / e-LSM por split."
    )
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                        help="Splits a processar. Ex.: --splits 40_60 60_40 70_30")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Horizontes. Default: usa integracao_rvfts.HORIZONS.")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Símbolos. Default: usa integracao_rvfts.SYMBOLS.")
    parser.add_argument("--quick", action="store_true",
                        help="Grids reduzidos (RV-FTS e e-LSM) para teste rápido.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    nclus_grid = rho_grid = elsm_grid = None
    if args.quick:
        nclus_grid = [2, 4]
        rho_grid = [0.05, 0.50, 0.95]
        elsm_grid = [dict(n_init_rules=6, sigma_max=0.12, tau_create=0.94)]

    for split in args.splits:
        run_split(
            split,
            symbols=args.symbols,
            horizons=args.horizons,
            nclus_grid=nclus_grid,
            rho_grid=rho_grid,
            elsm_grid=elsm_grid if args.quick else DEFAULT_ELSM_GRID,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
