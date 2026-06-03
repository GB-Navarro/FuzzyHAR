"""
integracao_rvfts.py
====================
Integração do modelo RV-FTS no pipeline FuzzyHAR.

Lê diretamente das pastas que o notebook produz:

    Others/Predictions/Fuzzy HAR/RV_t+{h}/{sym}/
        {sym}_tplus{h}_train_with_fuzzy.xlsx   → RV, Target, Fuzzy HAR
        {sym}_tplus{h}_test_with_fuzzy.xlsx

    Others/Predictions/HAR Models/RV_t+{h}/{sym}/
        insample_train/{sym}__{MODEL}__insample_train.xlsx
            → Data, Target, "{MODEL} train prediction"
        out-of-sample/{sym}__{MODEL}__oos_predictions.xlsx
            → Data, Target, "{MODEL} OOS prediction"

Não depende dos arquivos merged_with_fuzzy (gerados pela célula 50 do
notebook), eliminando o erro "Arquivo não encontrado".

Como usar
---------
    python integracao_rvfts.py

Correções vs. versão anterior
-----------------------------
1. Lê direto de Fuzzy HAR + HAR Models — sem depender de merged_with_fuzzy.
2. Alinhamento correto: RV-FTS[d] prevê Target[d] (mesma linha).
3. h>1: usa coluna Target (RV forward) em vez de agregação backward.
4. QLIKE conforme eq. (21) do paper: E[y/f − log(y/f) − 1].
5. ExcelWriter robusto: não tenta salvar se não há dados.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from rvfts_model import add_rvfts_to_dataframes, mse_paper, qlike_paper


# =====================================================================
# CONFIGURAÇÃO
# =====================================================================

SYMBOLS: List[str] = [
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "TRXUSDT", "XLMUSDT", "XRPUSDT",
]
HORIZONS: List[int] = [1, 7, 30]

# Raízes dos dados produzidos pelo notebook
FUZZY_ROOT     = Path("Others") / "Predictions" / "Fuzzy HAR"
HAR_MODELS_ROOT = Path("Others") / "Predictions" / "HAR Models"

# Pasta de saída desta integração
OUTPUT_ROOT = Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison"

# Modelos HAR a carregar (devem existir nas subpastas do notebook)
HAR_MODELS: List[str] = ["HAR", "HAR-CJ", "HAR-SJ", "HAR-TCJ", "LHAR-TCJ"]

# Hiperparâmetros do RV-FTS — versão "full" do paper (Section 3.6)
RVFTS_EW: int        = 756
RVFTS_CS: int        = 252
RVFTS_NCLUS: List[int]   = [2, 3, 4, 6, 9]
RVFTS_RHO:   List[float] = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
RVFTS_WINDOW: str    = "rolling"
RVFTS_FCM_SEED: int  = 42

# Para teste rápido, descomente:
# RVFTS_NCLUS = [2, 4]
# RVFTS_RHO   = [0.05, 0.50, 0.95]

# Ordem de exibição dos modelos nas tabelas
MODEL_ORDER: List[str] = HAR_MODELS + ["Fuzzy HAR", "RV-FTS"]

MODEL_COLORS: Dict[str, str] = {
    "HAR":       "#1f77b4",
    "HAR-CJ":    "#ff7f0e",
    "HAR-SJ":    "#2ca02c",
    "HAR-TCJ":   "#d62728",
    "LHAR-TCJ":  "#9467bd",
    "Fuzzy HAR": "#8c564b",
    "RV-FTS":    "#e377c2",
}


# =====================================================================
# FUNÇÕES DE CARREGAMENTO
# =====================================================================

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas recorrentes para um padrão único."""
    df = df.copy()
    if "Data" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Data": "Date"})
    if "target" in df.columns and "Target" not in df.columns:
        df = df.rename(columns={"target": "Target"})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _load_fuzzy_har(sym: str, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os arquivos do Fuzzy HAR (produzidos pela célula 44 do notebook).

    Retorna (df_train, df_test) com colunas:
        Date, RV, Target, Fuzzy HAR
    """
    base = FUZZY_ROOT / f"RV_t+{horizon}" / sym
    tr_path = base / f"{sym}_tplus{horizon}_train_with_fuzzy.xlsx"
    te_path = base / f"{sym}_tplus{horizon}_test_with_fuzzy.xlsx"

    if not tr_path.exists():
        raise FileNotFoundError(f"Fuzzy HAR treino não encontrado: {tr_path}")
    if not te_path.exists():
        raise FileNotFoundError(f"Fuzzy HAR teste não encontrado: {te_path}")

    df_tr = _norm(pd.read_excel(tr_path))
    df_te = _norm(pd.read_excel(te_path))
    return df_tr, df_te


def _load_har_model(sym: str, horizon: int, model: str
                    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Carrega as previsões de um modelo HAR individual (célula 47 do notebook).

    Arquivos esperados:
        HAR Models/RV_t+{h}/{sym}/insample_train/{sym}__{model}__insample_train.xlsx
        HAR Models/RV_t+{h}/{sym}/out-of-sample/{sym}__{model}__oos_predictions.xlsx

    Retorna (df_train, df_test) com as colunas de previsão, ou (None, None)
    se os arquivos não existirem.
    """
    base = HAR_MODELS_ROOT / f"RV_t+{horizon}" / sym
    tr_path  = base / "insample_train" / f"{sym}__{model}__insample_train.xlsx"
    oos_path = base / "out-of-sample"  / f"{sym}__{model}__oos_predictions.xlsx"

    df_tr  = _norm(pd.read_excel(tr_path))  if tr_path.exists()  else None
    df_oos = _norm(pd.read_excel(oos_path)) if oos_path.exists() else None
    return df_tr, df_oos


def _merge_har_into(
    df_base: pd.DataFrame,
    df_har:  Optional[pd.DataFrame],
    pred_col_in_har: str,
    pred_col_out: str,
) -> pd.DataFrame:
    """
    Faz left-join de df_har em df_base pela coluna Date, renomeando a
    coluna de previsão do HAR para pred_col_out.
    """
    if df_har is None or pred_col_in_har not in df_har.columns:
        return df_base

    right = df_har[["Date", pred_col_in_har]].rename(
        columns={pred_col_in_har: pred_col_out}
    )
    # Evita duplicar se já existir
    if pred_col_out in df_base.columns:
        df_base = df_base.drop(columns=[pred_col_out])

    if "Date" in df_base.columns:
        return df_base.merge(right, on="Date", how="left")
    else:
        # fallback: alinhamento posicional
        df_base = df_base.copy()
        vals = df_har[pred_col_in_har].values
        df_base[pred_col_out] = vals[:len(df_base)] if len(vals) >= len(df_base) \
            else np.concatenate([vals, np.full(len(df_base) - len(vals), np.nan)])
        return df_base


def load_full_data(sym: str, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega e consolida num único par (df_train, df_test) todas as previsões
    disponíveis para (sym, horizon): Fuzzy HAR + todos os modelos HAR.

    Colunas presentes no resultado (quando os arquivos existem):
        Date, RV, Target,
        Fuzzy HAR,
        HAR prediction,      HAR OOS prediction,
        HAR-CJ prediction,   HAR-CJ OOS prediction,
        ...
    """
    df_tr, df_te = _load_fuzzy_har(sym, horizon)

    for model in HAR_MODELS:
        df_har_tr, df_har_oos = _load_har_model(sym, horizon, model)

        # Coluna de treino: "{model} train prediction" → "{model} prediction"
        tr_col_src = f"{model} train prediction"
        tr_col_dst = f"{model} prediction"
        df_tr = _merge_har_into(df_tr, df_har_tr, tr_col_src, tr_col_dst)

        # Coluna OOS: "{model} OOS prediction" → mantém o nome
        oos_col = f"{model} OOS prediction"
        df_te = _merge_har_into(df_te, df_har_oos, oos_col, oos_col)

    return df_tr, df_te


# =====================================================================
# MODELOS A COMPARAR (derivados do que está disponível no par df_tr/df_te)
# =====================================================================

def _insample_col(model: str) -> str:
    if model == "Fuzzy HAR":
        return "Fuzzy HAR"
    if model == "RV-FTS":
        return "RV-FTS"
    return f"{model} prediction"


def _oos_col(model: str) -> str:
    if model == "Fuzzy HAR":
        return "Fuzzy HAR"
    if model == "RV-FTS":
        return "RV-FTS"
    return f"{model} OOS prediction"


# =====================================================================
# EXECUÇÃO DO RV-FTS
# =====================================================================

def build_all_data(
    symbols: Iterable[str] = SYMBOLS,
    horizons: Iterable[int] = HORIZONS,
    *,
    ew: int = RVFTS_EW,
    cs: int = RVFTS_CS,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    window: str = RVFTS_WINDOW,
    fcm_seed: int = RVFTS_FCM_SEED,
    verbose: bool = True,
) -> Dict[int, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Para cada (horizon × symbol):
      1. Carrega Fuzzy HAR + HAR Models das pastas do notebook.
      2. Roda o RV-FTS, adicionando a coluna 'RV-FTS'.

    Retorna:
        data[horizon][sym] = {"train": df_tr, "test": df_te}
    """
    if nclus_grid is None:
        nclus_grid = list(RVFTS_NCLUS)
    if rho_grid is None:
        rho_grid = list(RVFTS_RHO)

    symbols  = list(symbols)
    horizons = list(horizons)
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]] = {}
    total = len(horizons) * len(symbols)
    count = 0

    for horizon in horizons:
        data[horizon] = {}
        for sym in symbols:
            count += 1
            tag = f"[{count}/{total}] {sym} t+{horizon}"
            try:
                df_tr, df_te = load_full_data(sym, horizon)

                if verbose:
                    print(f"\n>>> {tag}")
                    tr_cols = [c for c in df_tr.columns if "prediction" in c.lower()
                               or c in ("Fuzzy HAR",)]
                    print(f"  Colunas de previsão carregadas: {tr_cols}")
                    print(f"  train={len(df_tr)} obs | test={len(df_te)} obs")

                df_tr, df_te, _info = add_rvfts_to_dataframes(
                    df_train=df_tr,
                    df_test=df_te,
                    horizon=horizon,
                    target_column="Target",
                    rv_column="RV",
                    ew=ew, cs=cs,
                    nclus_grid=nclus_grid,
                    rho_grid=rho_grid,
                    window=window,
                    fcm_seed=fcm_seed,
                    verbose=verbose,
                )
                data[horizon][sym] = {"train": df_tr, "test": df_te}

            except FileNotFoundError as e:
                print(f"[SKIP] {tag}: {e}")
            except Exception as e:
                import traceback
                print(f"[ERRO] {tag}: {e}")
                traceback.print_exc()

    return data


# =====================================================================
# MÉTRICAS
# =====================================================================

# Colunas de previsão que devem ter valores não-positivos substituídos
# por forward-fill antes do cálculo do QLIKE — mesma lista do notebook.
_FUZZY_PRED_COLS: List[str] = [
    "Fuzzy HAR",
    "HAR OOS prediction",
    "HAR-CJ OOS prediction",
    "HAR-SJ OOS prediction",
    "HAR-TCJ OOS prediction",
    "LHAR-TCJ OOS prediction",
    "RV-FTS",
]


def _apply_prediction_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica o mecanismo ``drop_negative_fuzzy`` do notebook (célula de métricas):
    para cada coluna de previsão presente no DataFrame, substitui valores
    <= 0 por NaN e aplica forward-fill.

    Isso evita que previsões nulas/negativas sejam clipadas para eps=1e-12
    dentro do QLIKE, o que tornaria r = y/f astronomicamente grande e
    explodiria a métrica.
    """
    df = df.copy()
    for col in _FUZZY_PRED_COLS:
        if col not in df.columns:
            continue
        neg_mask = df[col] <= 0.0
        if neg_mask.any():
            df.loc[neg_mask, col] = np.nan
            df[col] = df[col].ffill()
    return df


def compute_metrics_table(
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizons: Iterable[int] = HORIZONS,
    symbols:  Iterable[str] = SYMBOLS,
) -> Dict[int, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calcula MSE e QLIKE OOS para todos os modelos × símbolos × horizontes.

    Retorna:
        tables[horizon]["oos"] = {"MSE": df_mse, "QLIKE": df_qlike}
        Cada DataFrame: index=modelos, columns=tickers (sem "USDT").
    """
    horizons = list(horizons)
    symbols  = list(symbols)
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]] = {}

    for horizon in horizons:
        tables[horizon] = {"oos": {"MSE": {}, "QLIKE": {}}}

        for sym in symbols:
            if sym not in data.get(horizon, {}):
                continue

            ticker = sym.replace("USDT", "")
            df = data[horizon][sym]["test"]

            if "Target" not in df.columns:
                print(f"[WARN] {sym} t+{horizon}: sem coluna 'Target' — pulando.")
                continue

            # Replica o mecanismo drop_negative_fuzzy do notebook:
            # substitui previsões <= 0 por ffill antes de calcular QLIKE.
            df = _apply_prediction_filters(df)

            y = df["Target"].values.astype(float)

            for model in MODEL_ORDER:
                col = _oos_col(model)
                if col not in df.columns:
                    continue
                f = df[col].values.astype(float)
                tables[horizon]["oos"]["MSE"  ].setdefault(model, {})[ticker] = mse_paper(y, f)
                tables[horizon]["oos"]["QLIKE"].setdefault(model, {})[ticker] = qlike_paper(y, f)

        # Converter para DataFrames ordenados
        for metric in ("MSE", "QLIKE"):
            raw = tables[horizon]["oos"][metric]
            if not raw:
                tables[horizon]["oos"][metric] = pd.DataFrame()
                continue
            df_m = pd.DataFrame(raw).T
            ordered_models = [m for m in MODEL_ORDER if m in df_m.index]
            ordered_syms   = [s.replace("USDT", "") for s in symbols
                              if s.replace("USDT", "") in df_m.columns]
            df_m = df_m.loc[ordered_models, ordered_syms]
            df_m.index.name = "Model"
            tables[horizon]["oos"][metric] = df_m

    return tables


def build_long_metrics_table(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Tabela longa: (Horizon, Model, Symbol, MSE, QLIKE)."""
    rows = []
    for horizon in horizons:
        df_mse   = tables[horizon]["oos"]["MSE"]
        df_qlike = tables[horizon]["oos"]["QLIKE"]
        if df_mse.empty and df_qlike.empty:
            continue
        models = list(dict.fromkeys(list(df_mse.index) + list(df_qlike.index)))
        syms   = list(dict.fromkeys(list(df_mse.columns) + list(df_qlike.columns)))
        for model in models:
            for symbol in syms:
                mse = df_mse.loc[model, symbol] \
                    if (not df_mse.empty and model in df_mse.index and symbol in df_mse.columns) \
                    else np.nan
                ql  = df_qlike.loc[model, symbol] \
                    if (not df_qlike.empty and model in df_qlike.index and symbol in df_qlike.columns) \
                    else np.nan
                rows.append({"Horizon": f"t+{horizon}", "Model": model,
                             "Symbol": symbol, "MSE": mse, "QLIKE": ql})
    return pd.DataFrame(rows)


def _combine_metric_across_horizons(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    metric: str,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    frames = []
    for horizon in horizons:
        df = tables[horizon]["oos"][metric]
        if df is None or df.empty:
            continue
        block = df.copy()
        block.columns = pd.MultiIndex.from_product(
            [[f"t+{horizon}"], block.columns], names=["Horizon", "Symbol"])
        frames.append(block)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# =====================================================================
# EXPORTAÇÃO
# =====================================================================

def _safe_excel_writer(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """
    Salva múltiplos DataFrames num mesmo arquivo .xlsx.
    Ignora silenciosamente sheets vazios; não tenta gravar se todos
    estiverem vazios (evita IndexError do openpyxl).
    """
    non_empty = {name: df for name, df in sheets.items()
                 if df is not None and not df.empty}
    if not non_empty:
        print(f"  [skip] {path.name}: sem dados para salvar.")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in non_empty.items():
            df.to_excel(writer, sheet_name=name[:31])   # max 31 chars no Excel
    print(f"[salvo] {path}")


def _build_prediction_export_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    priority_cols = ["Date", "RV", "Target"]
    pred_cols = [_oos_col(m) if split == "oos" else _insample_col(m)
                 for m in MODEL_ORDER]
    selected = [c for c in priority_cols if c in df.columns]
    selected += [c for c in pred_cols if c in df.columns and c not in selected]
    return df.loc[:, selected].copy() if selected else df.copy()


def save_metric_spreadsheets(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizons: Iterable[int] = HORIZONS,
) -> None:
    for horizon in horizons:
        out_dir = OUTPUT_ROOT / f"RV_t+{horizon}" / "01_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        _safe_excel_writer(
            out_dir / f"metrics_summary_tplus{horizon}.xlsx",
            {f"OOS MSE": tables[horizon]["oos"]["MSE"],
             f"OOS QLIKE": tables[horizon]["oos"]["QLIKE"]},
        )
        for metric in ("MSE", "QLIKE"):
            df = tables[horizon]["oos"][metric]
            if df is None or df.empty:
                continue
            path = out_dir / f"{metric.lower()}_oos_tplus{horizon}.xlsx"
            df.to_excel(path)
            print(f"[salvo] {path}")


def save_overview_spreadsheets(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizons: Iterable[int] = HORIZONS,
) -> None:
    out_dir = OUTPUT_ROOT / "00_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = build_long_metrics_table(tables, horizons=horizons)
    if not long_df.empty:
        long_df.to_excel(out_dir / "all_metrics_long_format.xlsx", index=False)
        long_df.to_csv(out_dir / "all_metrics_long_format.csv", index=False)
        print(f"[salvo] {out_dir / 'all_metrics_long_format.xlsx'}")
        print(f"[salvo] {out_dir / 'all_metrics_long_format.csv'}")

    summary_sheets: Dict[str, pd.DataFrame] = {}
    for metric in ("MSE", "QLIKE"):
        df = _combine_metric_across_horizons(tables, metric, horizons=horizons)
        if not df.empty:
            summary_sheets[f"oos_{metric}"] = df

    _safe_excel_writer(out_dir / "all_metrics_by_split.xlsx", summary_sheets)


def save_prediction_spreadsheets(
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizons: Iterable[int] = HORIZONS,
    symbols:  Iterable[str] = SYMBOLS,
) -> None:
    for horizon in horizons:
        base_dir = OUTPUT_ROOT / f"RV_t+{horizon}" / "02_predictions"
        for sym in symbols:
            if sym not in data.get(horizon, {}):
                continue
            ticker = sym.replace("USDT", "")
            sym_dir = base_dir / ticker
            sym_dir.mkdir(parents=True, exist_ok=True)

            df_export = _build_prediction_export_df(data[horizon][sym]["test"], "oos")
            xlsx = sym_dir / f"{ticker}_oos_tplus{horizon}.xlsx"
            csv  = sym_dir / f"{ticker}_oos_tplus{horizon}.csv"
            df_export.to_excel(xlsx, index=False)
            df_export.to_csv(csv, index=False)
            print(f"[salvo] {xlsx}")
            print(f"[salvo] {csv}")


# =====================================================================
# GRÁFICOS
# =====================================================================

def _plot_timeseries_oos(
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizon: int,
    out_dir: Path,
    symbols: Iterable[str] = SYMBOLS,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly não instalado — pulando gráficos de séries.")
        return

    html_dir = out_dir / "html"
    png_dir  = out_dir / "png"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        if sym not in data.get(horizon, {}):
            continue
        ticker = sym.replace("USDT", "")
        df = data[horizon][sym]["test"]
        x  = df["Date"].tolist() if "Date" in df.columns else list(range(len(df)))

        fig = go.Figure()
        if "Target" in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df["Target"].tolist(),
                name="Actual", line=dict(color="black", width=1.2), opacity=0.85))

        for model in MODEL_ORDER:
            col = _oos_col(model)
            if col not in df.columns:
                continue
            fig.add_trace(go.Scatter(x=x, y=df[col].tolist(), name=model,
                line=dict(color=MODEL_COLORS.get(model), width=0.9), opacity=0.85))

        fig.update_layout(
            title=f"{ticker} — OOS — t+{horizon}",
            xaxis_title="Date", yaxis_title="Realized Volatility",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            template="plotly_white", width=1400, height=500,
        )
        stem = f"oos_ts_{ticker}_tplus{horizon}"
        fig.write_html(str(html_dir / f"{stem}.html"))
        print(f"[salvo] {html_dir / f'{stem}.html'}")
        try:
            fig.write_image(str(png_dir / f"{stem}.png"), scale=2)
            print(f"[salvo] {png_dir / f'{stem}.png'}")
        except Exception as e:
            print(f"[WARN] PNG não gerado ({stem}): {e}")


def _plot_metric_bars(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizon: int,
    out_dir: Path,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly não instalado — pulando gráficos de barras.")
        return

    html_dir = out_dir / "html"
    png_dir  = out_dir / "png"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("MSE", "QLIKE"):
        df = tables[horizon]["oos"][metric]
        if df is None or df.empty:
            continue
        fig = go.Figure()
        for model in df.index:
            fig.add_trace(go.Bar(
                name=model, x=list(df.columns),
                y=df.loc[model].values.astype(float),
                marker_color=MODEL_COLORS.get(model), opacity=0.85,
            ))
        fig.update_layout(
            barmode="group", title=f"{metric} OOS — t+{horizon}",
            xaxis_title="Symbol", yaxis_title=metric,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            template="plotly_white", width=1200, height=550,
        )
        stem = f"bar_{metric.lower()}_oos_tplus{horizon}"
        fig.write_html(str(html_dir / f"{stem}.html"))
        print(f"[salvo] {html_dir / f'{stem}.html'}")
        try:
            fig.write_image(str(png_dir / f"{stem}.png"), scale=2)
            print(f"[salvo] {png_dir / f'{stem}.png'}")
        except Exception as e:
            print(f"[WARN] PNG não gerado ({stem}): {e}")


def save_all_plots(
    data:   Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizons: Iterable[int] = HORIZONS,
    symbols:  Iterable[str] = SYMBOLS,
) -> None:
    for horizon in horizons:
        plot_root = OUTPUT_ROOT / f"RV_t+{horizon}" / "03_plots"
        _plot_timeseries_oos(data, horizon,
                             plot_root / "01_timeseries" / "out_of_sample",
                             symbols=symbols)
        _plot_metric_bars(tables, horizon, plot_root / "02_metric_bars")


# =====================================================================
# MAIN
# =====================================================================

def main(
    symbols: Iterable[str] = SYMBOLS,
    horizons: Iterable[int] = HORIZONS,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
]:
    """
    Pipeline completo. Retorna (data, tables).

    Pode ser chamado de dentro do notebook:
        from integracao_rvfts import main
        data, tables = main()
    """
    print("=" * 70)
    print("PASSO 1/3 — Carregando dados e executando RV-FTS...")
    print("=" * 70)
    data = build_all_data(symbols=symbols, horizons=horizons,
                          nclus_grid=nclus_grid, rho_grid=rho_grid,
                          verbose=verbose)

    print("\n" + "=" * 70)
    print("PASSO 2/3 — Calculando métricas...")
    print("=" * 70)
    tables = compute_metrics_table(data, horizons=horizons, symbols=symbols)

    for horizon in horizons:
        df = tables[horizon]["oos"]["MSE"]
        if df is None or df.empty:
            print(f"\n--- t+{horizon} | OOS | MSE: (sem dados) ---")
        else:
            print(f"\n--- t+{horizon} | OOS | MSE ---")
            print(df.to_string(float_format="{:.4e}".format))

    print("\n" + "=" * 70)
    print("PASSO 3/3 — Salvando planilhas, previsões e gráficos...")
    print("=" * 70)
    save_overview_spreadsheets(tables, horizons=horizons)
    save_metric_spreadsheets(tables, horizons=horizons)
    save_prediction_spreadsheets(data, horizons=horizons, symbols=symbols)
    save_all_plots(data, tables, horizons=horizons, symbols=symbols)

    print(f"\nConcluído. Resultados em: {OUTPUT_ROOT.resolve()}")
    return data, tables


if __name__ == "__main__":
    main()