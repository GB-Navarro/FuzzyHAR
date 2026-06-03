# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Graficos finais OOS (Actual vs HAR vs Fuzzy). Codigo verbatim do notebook.'''
from fuzzyhar.config import SYMBOLS

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# GRAFICOS FINAIS OOS: ACTUAL vs HAR vs ALSM-HAR
# ============================================================
# Fonte dos dados: Others/Merged Predictions
# Saida dos graficos: Results

def _pick_oos_merged_file(merged_dir: Path, symbol: str) -> Path | None:
    with_fuzzy = merged_dir / f"{symbol}__oos_merged_with_fuzzy.xlsx"
    classic = merged_dir / f"{symbol}__oos_merged.xlsx"

    if with_fuzzy.exists():
        return with_fuzzy
    if classic.exists():
        return classic
    return None

def run_model_comparison_plots(
    symbols,
    horizon,
    merged_predictions_root=Path("Others") / "Merged Predictions",
    results_root=Path("Results"),
    sheet_name=0,
    date_cols_cand=("Data", "date", "Date", "DATA"),
    drop_cols=("retornos_diarios", "retorno_diario"),
):
    merged_predictions_root = Path(merged_predictions_root)
    output_dir = Path(results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    drop_cols = set(drop_cols)
    attachment_prefix_by_horizon = {1: "attachment_10_", 7: "attachment_15_", 30: "attachment_20_"}

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    generated = []
    symbols = list(symbols)

    if not symbols:
        print(f"[skip] Nenhum simbolo informado para RV_t+{horizon}.")
        return generated

    selected_symbol = next(
        (s for s in symbols if str(s).upper().startswith("BTC")),
        symbols[0],
    )
    print(f"[info] Gerando plot apenas para: {selected_symbol} (RV_t+{horizon})")

    for sym in [selected_symbol]:
        ticker = sym.replace("USDT", "")
        merged_dir = merged_predictions_root / f"RV_t+{horizon}" / sym / "out-of-sample"
        file_path = _pick_oos_merged_file(merged_dir=merged_dir, symbol=sym)

        if file_path is None:
            print(f"[skip] Arquivo OOS merged nao encontrado para {sym} em {merged_dir}")
            continue

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.copy()

        # mesma logica de deteccao de data
        dt = None
        for c in date_cols_cand:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce")
                if s.notna().mean() > 0.8:
                    dt = s
                    break
        if dt is None:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    dt = pd.to_datetime(df[c], errors="coerce")
                    break
                if df[c].dtype == object:
                    s = pd.to_datetime(df[c], errors="coerce")
                    if s.notna().mean() > 0.8:
                        dt = s
                        break
        if dt is None:
            print(f"[skip] {ticker}: nenhuma coluna de data valida encontrada.")
            continue

        df["__date__"] = dt
        df = df.dropna(subset=["__date__"]).sort_values("__date__")

        for c in list(df.columns):
            if str(c).lower() in drop_cols:
                df = df.drop(columns=c)

        cols = list(df.columns)
        lowered = {c: str(c).lower() for c in cols}

        def find_column_by_priority(priorities, exclude=None):
            exclude = [] if exclude is None else exclude
            for group in priorities:
                for c in cols:
                    lc = lowered[c]
                    if all(term in lc for term in group) and not any(ex in lc for ex in exclude):
                        return c
            return None

        target_col = find_column_by_priority(
            priorities=[("actual",), ("target",), ("realized",), ("rv",), ("real",)]
        )
        fuzzy_col = find_column_by_priority(
            priorities=[("fuzzy", "har"), ("fuzzy",)]
        )
        har_col = find_column_by_priority(
            priorities=[("har",)],
            exclude=["fuzzy"]
        )

        if target_col is None or fuzzy_col is None or har_col is None:
            print(
                f"[skip] {ticker}: colunas nao encontradas "
                f"(target={target_col}, har={har_col}, fuzzy={fuzzy_col})"
            )
            continue

        def replace_negatives_with_prev_positive(s):
            s = pd.to_numeric(s, errors="coerce")
            last_pos = s.where(s > 0).ffill()
            out = s.copy()
            out[s < 0] = last_pos[s < 0]
            if out.isna().any():
                next_pos = s.where(s > 0).bfill()
                out = out.fillna(next_pos)
            return out

        actual = replace_negatives_with_prev_positive(df[target_col])
        har = replace_negatives_with_prev_positive(df[har_col])
        fuzzy = replace_negatives_with_prev_positive(df[fuzzy_col])

        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(df["__date__"], actual, color="black", linewidth=1.6, label="Actual")
        ax.plot(df["__date__"], har, color="blue", linestyle="--", linewidth=1.5, label="HAR")
        ax.plot(df["__date__"], fuzzy, color="red", linestyle=":", linewidth=1.8, label="ALSM-HAR")

        ax.set_title(f"{ticker} - Out-of-sample")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))

        fig.tight_layout()

        attachment_prefix = attachment_prefix_by_horizon.get(horizon, "")
        png_path = output_dir / f"{attachment_prefix}{ticker}_series_estilo_artigo_RV_t+{horizon}.png"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        generated.append({
            "ticker": ticker,
            "source": file_path,
            "png": png_path,
        })

    print(f"Plots concluidos. Saidas salvas em: {output_dir}")
    return generated


def run_all_plots():
    # ---------------- EXECUCAO ----------------
    plots_t1 = run_model_comparison_plots(
        symbols=SYMBOLS,
        horizon=1,
        merged_predictions_root=Path("Others") / "Merged Predictions",
        results_root=Path("Results"),
    )

    plots_t7 = run_model_comparison_plots(
        symbols=SYMBOLS,
        horizon=7,
        merged_predictions_root=Path("Others") / "Merged Predictions",
        results_root=Path("Results"),
    )

    plots_t30 = run_model_comparison_plots(
        symbols=SYMBOLS,
        horizon=30,
        merged_predictions_root=Path("Others") / "Merged Predictions",
        results_root=Path("Results"),
    )
