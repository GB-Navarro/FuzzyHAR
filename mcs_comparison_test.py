"""
mcs_comparison_test.py
======================
Compara três implementações do MCS sobre as mesmas matrizes de perda:

  1. arch_notebook  — arch.bootstrap.MCS, filtra APENAS "Fuzzy HAR" (≤0 → NaN → ffill)
  2. arch_pipeline  — arch.bootstrap.MCS, filtra TODOS os modelos   (≤0 → NaN → ffill)
  3. r_mcs          — R MCSprocedure (pacote MCS), filtra TODOS os modelos

Para cada implementação, testa dois conjuntos de modelos:
  • no_rvfts   — Fuzzy HAR + 5 HAR models          (6 modelos)
  • with_rvfts — Fuzzy HAR + 5 HAR models + RV-FTS  (7 modelos)

Diferenças documentadas entre implementações
--------------------------------------------
  block_size : arch usa sqrt(T); R usa max(ar(x)$order) — tipicamente 1–5 para RV diário.
  min.k      : R por padrão para quando restam min.k=3 modelos; arch vai até 1.
               Este script usa min.k=1 para comparação justa.
  Statistic  : ambos usam T_max (arch: method='max'; R: statistic='Tmax').
  alpha      : 1 - confidence = 0.25 (confiança 75%) em todos.

Uso
---
    python mcs_comparison_test.py
    python mcs_comparison_test.py --splits 40_60 60_40 70_30
    python mcs_comparison_test.py --horizons 1 --skip-r
    python mcs_comparison_test.py --block-size 10   # força block_size em arch e R

Saída
-----
    Results/HAR_FuzzyHAR_RVFTS_Comparison/{split}/mcs_comparison/
        mcs_comparison_{split}.xlsx
            sheets: t+{h}_{METRIC}  (uma por horizonte × métrica)
            sheet:  summary          (share de moedas por variante × modelo)
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SPLITS:   List[str] = ["40_60"]
HORIZONS: List[int] = [1, 7, 30]
SYMBOLS:  List[str] = [
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "TRXUSDT", "XLMUSDT", "XRPUSDT",
]
TICKERS: List[str] = [s.replace("USDT", "") for s in SYMBOLS]

ALL_MODELS: List[str] = [
    "HAR OOS prediction",
    "HAR-CJ OOS prediction",
    "HAR-SJ OOS prediction",
    "HAR-TCJ OOS prediction",
    "LHAR-TCJ OOS prediction",
    "Fuzzy HAR",
    "RV-FTS",
]
MODELS_NO_RVFTS: List[str] = [m for m in ALL_MODELS if m != "RV-FTS"]

EPS: float = 1e-12

MCS_CONFIDENCE: float = 0.75
MCS_ALPHA:      float = 1.0 - MCS_CONFIDENCE
MCS_B:          int   = 10_000
MCS_SEED:       int   = 123

# Variantes a correr: (nome_legível, model_set_key, filter_style)
VARIANTS: List[Tuple[str, str, str]] = [
    ("arch_notebook__no_rvfts",   "no_rvfts",   "notebook"),
    ("arch_notebook__with_rvfts", "with_rvfts", "notebook"),
    ("arch_pipeline__no_rvfts",   "no_rvfts",   "pipeline"),
    ("arch_pipeline__with_rvfts", "with_rvfts", "pipeline"),
    ("r_mcs__no_rvfts",           "no_rvfts",   "pipeline"),   # R usa filtro pipeline
    ("r_mcs__with_rvfts",         "with_rvfts", "pipeline"),
]

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

def _pred_file(split: str, horizon: int, ticker: str) -> Path:
    return (
        Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison" / split
        / f"RV_t+{horizon}" / "02_predictions" / ticker
        / f"{ticker}_oos_tplus{horizon}.xlsx"
    )


def _output_dir(split: str) -> Path:
    return (
        Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison" / split / "mcs_comparison"
    )


# ---------------------------------------------------------------------------
# Funções de perda (idênticas ao pipeline)
# ---------------------------------------------------------------------------

def qlike_loss_t(y: np.ndarray, f: np.ndarray, eps: float = EPS) -> np.ndarray:
    y = np.clip(y.astype(float), eps, None)
    f = np.clip(f.astype(float), eps, None)
    r = y / f
    return r - np.log(r) - 1.0


def mse_loss_t(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (y.astype(float) - f.astype(float)) ** 2


# ---------------------------------------------------------------------------
# Filtros de previsão
# ---------------------------------------------------------------------------

def _filter_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas a coluna 'Fuzzy HAR' (≤0 → NaN → ffill) — como no notebook."""
    df = df.copy()
    col = "Fuzzy HAR"
    if col in df.columns:
        mask = df[col] <= 0.0
        if mask.any():
            df.loc[mask, col] = np.nan
            df[col] = df[col].ffill()
    return df


def _filter_pipeline(df: pd.DataFrame, model_cols: Sequence[str]) -> pd.DataFrame:
    """Filtra todos os modelos do model_set (≤0 → NaN → ffill) — como no pipeline."""
    df = df.copy()
    for col in model_cols:
        if col not in df.columns:
            continue
        mask = df[col] <= 0.0
        if mask.any():
            df.loc[mask, col] = np.nan
            df[col] = df[col].ffill()
    return df


# ---------------------------------------------------------------------------
# Construção da matriz de perdas
# ---------------------------------------------------------------------------

def build_loss_matrix(
    df: pd.DataFrame,
    metric: str,
    model_cols: Sequence[str],
    filter_style: str,  # "notebook" | "pipeline"
    real_col: str = "Target",
) -> Optional[pd.DataFrame]:
    """
    Aplica o filtro escolhido e devolve um DataFrame de perdas (T × k).
    Retorna None se não houver linhas válidas suficientes.
    """
    if real_col not in df.columns:
        return None

    present = [c for c in model_cols if c in df.columns]
    if len(present) < 2:
        return None

    if filter_style == "notebook":
        df_f = _filter_notebook(df)
    else:
        df_f = _filter_pipeline(df, present)

    y = df_f[real_col].values.astype(float)
    losses: Dict[str, np.ndarray] = {}
    for col in present:
        f = df_f[col].values.astype(float)
        if metric == "QLIKE":
            losses[col] = qlike_loss_t(y, f)
        else:
            losses[col] = mse_loss_t(y, f)

    idx = df_f["Date"] if "Date" in df_f.columns else df_f.index
    L = pd.DataFrame(losses, index=idx)
    L = L.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return L if not L.empty else None


# ---------------------------------------------------------------------------
# MCS via arch (Python)
# ---------------------------------------------------------------------------

def arch_mcs(
    losses_df: pd.DataFrame,
    *,
    alpha: float = MCS_ALPHA,
    B: int = MCS_B,
    seed: int = MCS_SEED,
    block_size: Optional[int] = None,
) -> List[str]:
    from arch.bootstrap import MCS as ArchMCS

    L = losses_df.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if L.shape[1] < 2 or L.shape[0] == 0:
        return list(L.columns)

    kwargs = dict(size=alpha, bootstrap="stationary", reps=B, method="max", seed=seed)
    if block_size is not None:
        kwargs["block_size"] = block_size

    mcs = ArchMCS(L, **kwargs)
    mcs.compute()
    return list(mcs.included)


# ---------------------------------------------------------------------------
# MCS via R (subprocess)
# ---------------------------------------------------------------------------

_R_SCRIPT = textwrap.dedent("""\
    args      <- commandArgs(trailingOnly=TRUE)
    loss_file <- args[1]
    out_file  <- args[2]
    alpha     <- as.numeric(args[3])
    B         <- as.integer(args[4])
    seed      <- as.integer(args[5])
    min_k     <- as.integer(args[6])
    k_arg     <- if (args[7] == "auto") NULL else as.integer(args[7])

    suppressPackageStartupMessages(library(MCS))

    L <- read.csv(loss_file, check.names=FALSE)
    if ("Date" %in% names(L)) L[["Date"]] <- NULL
    L <- as.matrix(L)
    storage.mode(L) <- "double"

    res <- MCSprocedure(
        Loss      = L,
        alpha     = alpha,
        B         = B,
        statistic = "Tmax",
        seed      = seed,
        min.k     = min_k,
        k         = k_arg,
        verbose   = FALSE
    )

    included <- rownames(res@show)
    write.csv(data.frame(model=included), out_file, row.names=FALSE, quote=TRUE)
""")


def r_mcs(
    losses_df: pd.DataFrame,
    *,
    alpha: float = MCS_ALPHA,
    B: int = MCS_B,
    seed: int = MCS_SEED,
    block_size: Optional[int] = None,
    min_k: int = 1,
) -> List[str]:
    """Roda MCSprocedure do R sobre losses_df via subprocess + Rscript."""
    L = losses_df.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if L.shape[1] < 2 or L.shape[0] == 0:
        return list(L.columns)

    with tempfile.TemporaryDirectory() as tmp:
        loss_csv  = Path(tmp) / "losses.csv"
        out_csv   = Path(tmp) / "included.csv"
        r_script  = Path(tmp) / "run_mcs.R"

        L.to_csv(loss_csv, index=False)
        r_script.write_text(_R_SCRIPT, encoding="utf-8")

        k_arg = str(block_size) if block_size is not None else "auto"
        cmd = [
            "Rscript", "--vanilla", str(r_script),
            str(loss_csv), str(out_csv),
            str(alpha), str(B), str(seed), str(min_k), k_arg,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            warnings.warn(
                f"[R MCS] Rscript retornou código {result.returncode}.\n"
                f"stderr: {result.stderr[:500]}"
            )
            return []

        if not out_csv.exists():
            warnings.warn("[R MCS] Arquivo de saída não criado pelo R.")
            return []

        df_out = pd.read_csv(out_csv)
        return list(df_out["model"].astype(str))


# ---------------------------------------------------------------------------
# Pipeline principal por (split, horizonte, ticker, métrica)
# ---------------------------------------------------------------------------

def run_comparison(
    splits:     Iterable[str] = SPLITS,
    horizons:   Iterable[int] = HORIZONS,
    tickers:    Iterable[str] = TICKERS,
    metrics:    Iterable[str] = ("QLIKE", "MSE"),
    *,
    skip_r:     bool = False,
    block_size: Optional[int] = None,
    verbose:    bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Executa todos os variantes do MCS e devolve resultados por split.

    Retorna: {split: DataFrame longo com colunas
              (Horizon, Metric, Ticker, Model, Variant, Included)}
    """
    splits   = list(splits)
    horizons = list(horizons)
    tickers  = list(tickers)
    metrics  = [m.upper() for m in metrics]

    all_results: Dict[str, List[dict]] = {s: [] for s in splits}

    for split in splits:
        print(f"\n{'='*70}\nSPLIT: {split}\n{'='*70}")

        for horizon in horizons:
            for ticker in tickers:
                fp = _pred_file(split, horizon, ticker)
                if not fp.exists():
                    if verbose:
                        print(f"  [SKIP] {ticker} t+{horizon}: arquivo não encontrado.")
                    continue

                df_raw = pd.read_excel(fp)
                if "Date" in df_raw.columns:
                    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
                for col in df_raw.columns:
                    if col != "Date":
                        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

                for metric in metrics:
                    if verbose:
                        print(f"\n  {ticker} t+{horizon} [{metric}]")

                    # ── pré-calcula as 4 matrizes de perda possíveis ──────────
                    L_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
                    for model_key, model_cols in [
                        ("no_rvfts",   MODELS_NO_RVFTS),
                        ("with_rvfts", ALL_MODELS),
                    ]:
                        for fstyle in ("notebook", "pipeline"):
                            key = (model_key, fstyle)
                            L_cache[key] = build_loss_matrix(
                                df_raw, metric, model_cols, fstyle
                            )

                    # ── corre cada variante ───────────────────────────────────
                    for variant_name, model_key, fstyle in VARIANTS:
                        if skip_r and variant_name.startswith("r_mcs"):
                            continue

                        L = L_cache[(model_key, fstyle)]
                        if L is None:
                            if verbose:
                                print(f"    [{variant_name}] sem dados — skip")
                            continue

                        try:
                            if variant_name.startswith("r_mcs"):
                                included = r_mcs(
                                    L, alpha=MCS_ALPHA, B=MCS_B,
                                    seed=MCS_SEED, block_size=block_size,
                                )
                            else:
                                included = arch_mcs(
                                    L, alpha=MCS_ALPHA, B=MCS_B,
                                    seed=MCS_SEED, block_size=block_size,
                                )
                            status = "ok"
                        except Exception as e:
                            warnings.warn(f"    [{variant_name}] erro: {e}")
                            included = []
                            status = f"error: {e}"

                        model_set = ALL_MODELS if model_key == "with_rvfts" else MODELS_NO_RVFTS
                        present   = [m for m in model_set if m in L.columns]

                        if verbose:
                            print(f"    [{variant_name}] included={included}")

                        for model in present:
                            all_results[split].append({
                                "Horizon":  f"t+{horizon}",
                                "Metric":   metric,
                                "Ticker":   ticker,
                                "Model":    model,
                                "Variant":  variant_name,
                                "Included": int(model in included),
                                "T":        len(L),
                                "Status":   status,
                            })

    return {s: pd.DataFrame(rows) for s, rows in all_results.items()}


# ---------------------------------------------------------------------------
# Exportação para Excel
# ---------------------------------------------------------------------------

def _pivot_sheet(
    df_long: pd.DataFrame,
    horizon: int,
    metric: str,
    variants: List[str],
) -> pd.DataFrame:
    """
    Cria tabela pivô: linhas = modelos, colunas = MultiIndex(Variant, Ticker).
    Célula = 1 (incluído) ou 0 (excluído) ou '—' (modelo não testado nesta variante).
    """
    sub = df_long[
        (df_long["Horizon"] == f"t+{horizon}") & (df_long["Metric"] == metric)
    ]
    if sub.empty:
        return pd.DataFrame()

    model_order = ALL_MODELS
    ticker_order = [t for t in TICKERS if t in sub["Ticker"].unique()]
    variant_order = [v for v in variants if v in sub["Variant"].unique()]

    cols = pd.MultiIndex.from_product(
        [variant_order, ticker_order], names=["Variant", "Ticker"]
    )
    pivot = pd.DataFrame("--", index=model_order, columns=cols, dtype=object)
    pivot.index.name = "Model"

    for _, row in sub.iterrows():
        v = row["Variant"]
        t = row["Ticker"]
        m = row["Model"]
        if m in pivot.index and v in variant_order and t in ticker_order:
            pivot.loc[m, (v, t)] = int(row["Included"])

    return pivot


def _summary_sheet(df_long: pd.DataFrame, variants: List[str]) -> pd.DataFrame:
    """
    Tabela de resumo: share de tickers onde o modelo foi incluído,
    por variante × horizonte × métrica.
    """
    rows = []
    for variant in variants:
        sub_v = df_long[df_long["Variant"] == variant]
        for horizon in HORIZONS:
            h_str = f"t+{horizon}"
            for metric in ("QLIKE", "MSE"):
                sub = sub_v[
                    (sub_v["Horizon"] == h_str) & (sub_v["Metric"] == metric)
                ]
                if sub.empty:
                    continue
                for model in sub["Model"].unique():
                    sub_m = sub[sub["Model"] == model]
                    n_tickers = sub_m["Ticker"].nunique()
                    share = sub_m["Included"].sum() / n_tickers if n_tickers else np.nan
                    rows.append({
                        "Variant": variant,
                        "Horizon": h_str,
                        "Metric":  metric,
                        "Model":   model,
                        "N_tickers": n_tickers,
                        "N_included": int(sub_m["Included"].sum()),
                        "Share":   round(share, 3),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    model_order = ALL_MODELS
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    return df.sort_values(["Variant", "Horizon", "Metric", "Model"]).reset_index(drop=True)


def save_results(
    results: Dict[str, pd.DataFrame],
    skip_r: bool = False,
) -> None:
    active_variants = [
        v for v, _, _ in VARIANTS
        if not (skip_r and v.startswith("r_mcs"))
    ]

    for split, df_long in results.items():
        if df_long.empty:
            print(f"[SKIP] {split}: sem dados para salvar.")
            continue

        out_dir = _output_dir(split)
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = out_dir / f"mcs_comparison_{split}.xlsx"

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for horizon in HORIZONS:
                for metric in ("QLIKE", "MSE"):
                    sheet_name = f"t+{horizon}_{metric}"
                    pivot = _pivot_sheet(df_long, horizon, metric, active_variants)
                    if not pivot.empty:
                        pivot.to_excel(writer, sheet_name=sheet_name)
                        print(f"  [sheet] {sheet_name}")

            summary = _summary_sheet(df_long, active_variants)
            if not summary.empty:
                summary.to_excel(writer, sheet_name="summary", index=False)
                print("  [sheet] summary")

            # Long format completo
            df_long.to_excel(writer, sheet_name="raw", index=False)
            print("  [sheet] raw")

        print(f"\n[salvo] {xlsx_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compara implementações do MCS (arch Python vs R MCSprocedure) "
            "sobre as mesmas matrizes de perda, com/sem RV-FTS."
        )
    )
    p.add_argument("--splits", nargs="+", default=["40_60"],
                   help="Splits a testar. Ex.: --splits 40_60 60_40 70_30")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 30])
    p.add_argument("--tickers", nargs="+", default=TICKERS,
                   help="Tickers a testar (sem 'USDT').")
    p.add_argument("--metrics", nargs="+", default=["QLIKE", "MSE"])
    p.add_argument("--skip-r", action="store_true",
                   help="Pula as variantes R (útil se Rscript não estiver disponível).")
    p.add_argument("--block-size", type=int, default=None,
                   help=(
                       "Força block_size em arch e k em R (padrão: auto — "
                       "arch usa sqrt(T), R usa max(ar.order))."
                   ))
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    print("=" * 70)
    print("MCS Comparison Test")
    print(f"  Splits:     {args.splits}")
    print(f"  Horizons:   {args.horizons}")
    print(f"  Tickers:    {args.tickers}")
    print(f"  Metrics:    {args.metrics}")
    print(f"  Skip R:     {args.skip_r}")
    print(f"  block_size: {args.block_size or 'auto'}")
    print(f"  confidence: {MCS_CONFIDENCE}  (alpha={MCS_ALPHA})")
    print(f"  B={MCS_B}, seed={MCS_SEED}")
    print("=" * 70)

    print("\nVariantes a correr:")
    for name, model_key, fstyle in VARIANTS:
        if args.skip_r and name.startswith("r_mcs"):
            print(f"  [skip]  {name}")
        else:
            n_models = len(ALL_MODELS if model_key == "with_rvfts" else MODELS_NO_RVFTS)
            print(f"  [run]   {name:<40} filter={fstyle}, models={n_models}")

    print("\nDiferenças entre implementações documentadas:")
    print("  block_size : arch=sqrt(T)~17-22  |  R=max(ar.order)~1-5  (para RV diario)")
    print("  min.k      : arch para em 1 modelo  |  R usa min.k=1 (forcado para comparacao justa)")
    print("  statistic  : T_max em ambos")
    if not args.skip_r:
        print("  Rscript    : chamado via subprocess")

    results = run_comparison(
        splits=args.splits,
        horizons=args.horizons,
        tickers=args.tickers,
        metrics=args.metrics,
        skip_r=args.skip_r,
        block_size=args.block_size,
        verbose=not args.quiet,
    )
    save_results(results, skip_r=args.skip_r)


if __name__ == "__main__":
    main()
