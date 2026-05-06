"""
integracao_rvfts.py
====================
Integração do modelo RV-FTS no pipeline FuzzyHAR.

Gera métricas comparativas (MSE e QLIKE) e gráficos para todos os
modelos HAR, Fuzzy HAR e RV-FTS, separados por horizonte e por
in-sample / out-of-sample.

Pré-requisitos:
  - Notebook FuzzyHAR.ipynb executado até a célula 45 (arquivos em
    Others/Predictions/Fuzzy HAR/) e até a célula de merge (arquivos
    em Others/Merged Predictions/).
  - rvfts_model.py na mesma pasta deste script.

Como usar:
  Execute a partir da pasta raiz do projeto:
    python integracao_rvfts.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rvfts_model import add_rvfts_to_dataframes

# ============================================================
# CONFIGURAÇÃO
# ============================================================

SYMBOLS = [
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "TRXUSDT", "XLMUSDT", "XRPUSDT",
]
HORIZONS = [1, 7, 30]
EPS = 1e-12

# Fontes de dados
FUZZY_ROOT  = Path("Others") / "Predictions" / "Fuzzy HAR"
MERGED_ROOT = Path("Others") / "Merged Predictions"

# Pasta de saída para todos os resultados desta integração
OUTPUT_ROOT = Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison"

# Hiperparâmetros RV-FTS (versão "full" do paper)
RVFTS_EW        = 756
RVFTS_CS        = 252
RVFTS_NCLUS     = [2, 3, 4, 6, 9]
RVFTS_RHO       = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]

# Para testes rápidos, substitua por:
# RVFTS_NCLUS = [2, 4]
# RVFTS_RHO   = [0.05, 0.50, 0.95]

# Nomes padronizados dos modelos e colunas por split
MODELS_INSAMPLE: Dict[str, str] = {
    "HAR":      "HAR prediction",
    "HAR-CJ":   "HAR-CJ prediction",
    "HAR-SJ":   "HAR-SJ prediction",
    "HAR-TCJ":  "HAR-TCJ prediction",
    "LHAR-TCJ": "LHAR-TCJ prediction",
    "Fuzzy HAR": "Fuzzy HAR",
    "RV-FTS":   "RV-FTS",
}
MODELS_OOS: Dict[str, str] = {
    "HAR":      "HAR OOS prediction",
    "HAR-CJ":   "HAR-CJ OOS prediction",
    "HAR-SJ":   "HAR-SJ OOS prediction",
    "HAR-TCJ":  "HAR-TCJ OOS prediction",
    "LHAR-TCJ": "LHAR-TCJ OOS prediction",
    "Fuzzy HAR": "Fuzzy HAR",
    "RV-FTS":   "RV-FTS",
}
MODEL_ORDER = list(MODELS_INSAMPLE.keys())
SPLIT_FOLDERS = {
    "insample": "in_sample",
    "oos": "out_of_sample",
}

# Cores fixas por modelo para todos os gráficos
MODEL_COLORS: Dict[str, str] = {
    "HAR":       "#1f77b4",
    "HAR-CJ":    "#ff7f0e",
    "HAR-SJ":    "#2ca02c",
    "HAR-TCJ":   "#d62728",
    "LHAR-TCJ":  "#9467bd",
    "Fuzzy HAR": "#8c564b",
    "RV-FTS":    "#e377c2",
}


# ============================================================
# FUNÇÕES DE MÉTRICAS
# ============================================================

def _mse(y: np.ndarray, f: np.ndarray) -> float:
    mask = ~(np.isnan(y) | np.isnan(f))
    if mask.sum() < 2:
        return np.nan
    return float(np.mean((y[mask] - f[mask]) ** 2))


def _qlike(y: np.ndarray, f: np.ndarray) -> float:
    mask = ~(np.isnan(y) | np.isnan(f))
    if mask.sum() < 2:
        return np.nan
    yt = np.clip(y[mask], EPS, None)
    ft = np.clip(f[mask], EPS, None)
    r = yt / ft
    return float(np.mean(r - np.log(r) - 1.0))


# ============================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================

def _load_merged(sym: str, horizon: int, split: str) -> pd.DataFrame:
    """
    Carrega o arquivo merged_with_fuzzy para um símbolo/horizonte/split.
    split: 'insample' ou 'oos'
    """
    if split == "insample":
        path = (MERGED_ROOT / f"RV_t+{horizon}" / sym
                / "insample_train" / f"{sym}__insample_merged_with_fuzzy.xlsx")
    else:
        path = (MERGED_ROOT / f"RV_t+{horizon}" / sym
                / "out-of-sample" / f"{sym}__oos_merged_with_fuzzy.xlsx")

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_excel(path)
    if "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _load_rv_series(sym: str, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """
    Carrega as séries de RV diário (necessárias para o RV-FTS) dos
    arquivos de previsão do Fuzzy HAR.
    Retorna (rv_train, rv_test) alinhadas por data.
    """
    base = FUZZY_ROOT / f"RV_t+{horizon}" / sym
    tr_path = base / f"{sym}_tplus{horizon}_train_with_fuzzy.xlsx"
    te_path = base / f"{sym}_tplus{horizon}_test_with_fuzzy.xlsx"

    df_tr = pd.read_excel(tr_path)
    df_te = pd.read_excel(te_path)

    for df in (df_tr, df_te):
        if "Data" in df.columns:
            df.rename(columns={"Data": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    rv_tr = df_tr.set_index("Date")["RV"]
    rv_te = df_te.set_index("Date")["RV"]
    return rv_tr, rv_te


def _add_rv_to_merged(
    df_insample: pd.DataFrame,
    df_oos: pd.DataFrame,
    rv_train: pd.Series,
    rv_test: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adiciona a coluna RV (necessária para RV-FTS) aos DataFrames merged."""
    df_insample = df_insample.copy()
    df_oos = df_oos.copy()

    df_insample = df_insample.merge(
        rv_train.rename("RV").reset_index(), on="Date", how="left"
    )
    df_oos = df_oos.merge(
        rv_test.rename("RV").reset_index(), on="Date", how="left"
    )
    return df_insample, df_oos


# ============================================================
# PASSO PRINCIPAL: ADICIONAR RV-FTS E CALCULAR MÉTRICAS
# ============================================================

def build_all_data(
    verbose: bool = True,
) -> Dict[int, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Para cada horizonte e símbolo:
      1. Carrega os DataFrames merged (insample + OOS) com predições dos HAR e Fuzzy HAR.
      2. Adiciona coluna RV (necessária para RV-FTS).
      3. Executa add_rvfts_to_dataframes para inserir coluna 'RV-FTS'.

    Retorna:
        data[horizon][sym] = {"insample": df_in, "oos": df_oos}
    """
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]] = {}

    total = len(HORIZONS) * len(SYMBOLS)
    count = 0

    for horizon in HORIZONS:
        data[horizon] = {}
        for sym in SYMBOLS:
            count += 1
            tag = f"[{count}/{total}] {sym} t+{horizon}"

            try:
                df_in  = _load_merged(sym, horizon, "insample")
                df_oos = _load_merged(sym, horizon, "oos")
                rv_tr, rv_te = _load_rv_series(sym, horizon)
                df_in, df_oos = _add_rv_to_merged(df_in, df_oos, rv_tr, rv_te)

                if verbose:
                    print(f"\n>>> {tag}")

                df_in, df_oos, _ = add_rvfts_to_dataframes(
                    df_train=df_in,
                    df_test=df_oos,
                    horizon=horizon,
                    rv_column="RV",
                    ew=RVFTS_EW,
                    cs=RVFTS_CS,
                    nclus_grid=RVFTS_NCLUS,
                    rho_grid=RVFTS_RHO,
                    window="rolling",
                    verbose=verbose,
                )

                data[horizon][sym] = {"insample": df_in, "oos": df_oos}

            except FileNotFoundError as e:
                print(f"[SKIP] {tag}: {e}")
            except Exception as e:
                import traceback
                print(f"[ERRO] {tag}: {e}")
                traceback.print_exc()

    return data


# ============================================================
# CÁLCULO DE MÉTRICAS
# ============================================================

def compute_metrics_table(
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
) -> Dict[int, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calcula MSE e QLIKE para todos os modelos e horizontes (apenas out-of-sample).

    Retorna:
        tables[horizon]["oos"]  →  {"MSE": df_mse, "QLIKE": df_qlike}
        onde cada DataFrame tem índice=modelos e colunas=símbolos.
    """
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]] = {}

    for horizon in HORIZONS:
        tables[horizon] = {
            "oos": {"MSE": {}, "QLIKE": {}},
        }

        for sym in SYMBOLS:
            if sym not in data.get(horizon, {}):
                continue

            ticker = sym.replace("USDT", "")

            df = data[horizon][sym]["oos"]
            y  = df["Target"].values.astype(float)

            for model_name, col in MODELS_OOS.items():
                if col not in df.columns:
                    continue
                f = df[col].values.astype(float)
                tables[horizon]["oos"]["MSE"].setdefault(model_name, {})[ticker]   = _mse(y, f)
                tables[horizon]["oos"]["QLIKE"].setdefault(model_name, {})[ticker] = _qlike(y, f)

        # Converter dicts aninhados para DataFrames ordenados
        for metric in ("MSE", "QLIKE"):
            raw = tables[horizon]["oos"][metric]
            df_metric = pd.DataFrame(raw).T  # modelos × símbolos
            ordered_models = [m for m in MODEL_ORDER if m in df_metric.index]
            ordered_syms   = [s.replace("USDT", "") for s in SYMBOLS
                              if s.replace("USDT", "") in df_metric.columns]
            df_metric = df_metric.loc[ordered_models, ordered_syms]
            df_metric.index.name = "Model"
            tables[horizon]["oos"][metric] = df_metric

    return tables


def _build_prediction_export_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Seleciona as colunas mais úteis para auditoria das previsões."""
    col_map = MODELS_INSAMPLE if split == "insample" else MODELS_OOS

    selected_cols: List[str] = []
    for col in ["Date", "RV", "Target"]:
        if col in df.columns and col not in selected_cols:
            selected_cols.append(col)

    for col in col_map.values():
        if col in df.columns and col not in selected_cols:
            selected_cols.append(col)

    if not selected_cols:
        return df.copy()

    return df.loc[:, selected_cols].copy()


def build_long_metrics_table(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
) -> pd.DataFrame:
    """
    Constrói uma tabela longa com todas as combinações de horizonte, split,
    modelo e símbolo, reunindo MSE e QLIKE no mesmo arquivo.
    """
    rows = []

    for horizon in HORIZONS:
        df_mse   = tables[horizon]["oos"]["MSE"]
        df_qlike = tables[horizon]["oos"]["QLIKE"]

        models  = sorted(set(df_mse.index).union(df_qlike.index))
        symbols = sorted(set(df_mse.columns).union(df_qlike.columns))

        for model in models:
            for symbol in symbols:
                mse = df_mse.loc[model, symbol] if model in df_mse.index and symbol in df_mse.columns else np.nan
                qlike = (
                    df_qlike.loc[model, symbol]
                    if model in df_qlike.index and symbol in df_qlike.columns
                    else np.nan
                )
                rows.append({
                    "Horizon": f"t+{horizon}",
                    "Split": SPLIT_FOLDERS["oos"],
                    "Model": model,
                    "Symbol": symbol,
                    "MSE": mse,
                    "QLIKE": qlike,
                })

    return pd.DataFrame(rows)


def _combine_metric_across_horizons(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    split: str,
    metric: str,
) -> pd.DataFrame:
    """Combina um mesmo split/métrica em uma tabela única com MultiIndex."""
    frames = []

    for horizon in HORIZONS:
        df = tables[horizon][split][metric]
        if df.empty:
            continue
        block = df.copy()
        block.columns = pd.MultiIndex.from_product(
            [[f"t+{horizon}"], block.columns],
            names=["Horizon", "Symbol"],
        )
        frames.append(block)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1)


# ============================================================
# SALVAMENTO DE PLANILHAS
# ============================================================

def save_metric_spreadsheets(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
) -> None:
    """
    Salva uma planilha Excel por horizonte com abas para:
      - oos MSE / oos QLIKE
    Também salva arquivos individuais por métrica.
    """
    for horizon in HORIZONS:
        out_dir = OUTPUT_ROOT / f"RV_t+{horizon}" / "01_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Arquivo único com todas as abas
        combined_path = out_dir / f"metrics_summary_tplus{horizon}.xlsx"
        with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
            for metric in ("MSE", "QLIKE"):
                df = tables[horizon]["oos"][metric]
                df.to_excel(writer, sheet_name=f"Out-of-Sample {metric}")

        print(f"[salvo] {combined_path}")

        # Arquivos individuais por métrica
        split_label = SPLIT_FOLDERS["oos"]
        for metric in ("MSE", "QLIKE"):
            df = tables[horizon]["oos"][metric]
            path = out_dir / f"{metric.lower()}_{split_label}_tplus{horizon}.xlsx"
            df.to_excel(path)
            print(f"[salvo] {path}")


def save_overview_spreadsheets(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
) -> None:
    """
    Salva uma visão consolidada de todas as métricas em todos os horizontes.
    """
    out_dir = OUTPUT_ROOT / "00_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = build_long_metrics_table(tables)
    long_xlsx = out_dir / "all_metrics_long_format.xlsx"
    long_csv = out_dir / "all_metrics_long_format.csv"
    long_df.to_excel(long_xlsx, index=False)
    long_df.to_csv(long_csv, index=False)
    print(f"[salvo] {long_xlsx}")
    print(f"[salvo] {long_csv}")

    summary_xlsx = out_dir / "all_metrics_by_split.xlsx"
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        split_label = SPLIT_FOLDERS["oos"]
        for metric in ("MSE", "QLIKE"):
            df = _combine_metric_across_horizons(tables, "oos", metric)
            if not df.empty:
                df.to_excel(writer, sheet_name=f"{split_label}_{metric}")
    print(f"[salvo] {summary_xlsx}")


def save_prediction_spreadsheets(
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
) -> None:
    """
    Salva as tabelas de previsões por horizonte, símbolo e split para facilitar
    a auditoria das comparações entre HAR, Fuzzy HAR e RV-FTS.
    """
    for horizon in HORIZONS:
        base_dir = OUTPUT_ROOT / f"RV_t+{horizon}" / "02_predictions"
        base_dir.mkdir(parents=True, exist_ok=True)

        for sym in SYMBOLS:
            if sym not in data.get(horizon, {}):
                continue

            ticker = sym.replace("USDT", "")
            sym_dir = base_dir / ticker
            sym_dir.mkdir(parents=True, exist_ok=True)

            split_label = SPLIT_FOLDERS["oos"]
            df_export = _build_prediction_export_df(data[horizon][sym]["oos"], "oos")

            xlsx_path = sym_dir / f"{ticker}_{split_label}_tplus{horizon}.xlsx"
            csv_path  = sym_dir / f"{ticker}_{split_label}_tplus{horizon}.csv"

            df_export.to_excel(xlsx_path, index=False)
            df_export.to_csv(csv_path, index=False)

            print(f"[salvo] {xlsx_path}")
            print(f"[salvo] {csv_path}")


# ============================================================
# GRÁFICOS
# ============================================================

def _plot_timeseries_oos(
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizon: int,
    out_dir: Path,
) -> None:
    """
    Um gráfico por símbolo: série temporal OOS de Actual vs todos os modelos.
    Exporta HTML interativo e PNG estático em subpastas separadas.
    """
    syms = [s for s in SYMBOLS if s in data.get(horizon, {})]
    if not syms:
        return

    html_dir = out_dir / "html"
    png_dir  = out_dir / "png"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for sym in syms:
        ticker = sym.replace("USDT", "")
        df = data[horizon][sym]["oos"]

        x = df["Date"].tolist() if "Date" in df.columns else list(range(len(df)))

        fig = go.Figure()

        if "Target" in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df["Target"].tolist(),
                name="Actual",
                line=dict(color="black", width=1.2),
                opacity=0.8,
            ))

        for model_name, col in MODELS_OOS.items():
            if col not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=x, y=df[col].tolist(),
                name=model_name,
                line=dict(color=MODEL_COLORS.get(model_name), width=0.9),
                opacity=0.8,
            ))

        fig.update_layout(
            title=f"{ticker} — OOS Forecasts — t+{horizon}",
            xaxis_title="Date",
            yaxis_title="Realized Volatility",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            width=1400,
            height=500,
        )

        stem = f"oos_timeseries_{ticker}_tplus{horizon}"
        html_path = html_dir / f"{stem}.html"
        png_path  = png_dir  / f"{stem}.png"

        fig.write_html(str(html_path))
        fig.write_image(str(png_path), scale=2)
        print(f"[salvo] {html_path}")
        print(f"[salvo] {png_path}")



def _plot_metric_bars(
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizon: int,
    out_dir: Path,
) -> None:
    """
    Gráficos de barras agrupadas (out-of-sample):
    eixo x = símbolo, grupos de barras = modelos, para MSE e QLIKE.
    Exporta HTML interativo e PNG estático em subpastas separadas.
    """
    html_dir = out_dir / "html"
    png_dir  = out_dir / "png"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("MSE", "QLIKE"):
        df = tables[horizon]["oos"][metric]
        if df.empty:
            continue

        models = list(df.index)
        syms   = list(df.columns)

        fig = go.Figure()

        for model in models:
            vals = df.loc[model].values.astype(float)
            fig.add_trace(go.Bar(
                name=model,
                x=syms,
                y=vals,
                marker_color=MODEL_COLORS.get(model),
                opacity=0.85,
            ))

        fig.update_layout(
            barmode="group",
            title=f"{metric} by Symbol — OOS — t+{horizon}",
            xaxis_title="Symbol",
            yaxis_title=metric,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            width=1200,
            height=550,
        )

        stem = f"bar_{metric.lower()}_oos_tplus{horizon}"
        html_path = html_dir / f"{stem}.html"
        png_path  = png_dir  / f"{stem}.png"

        fig.write_html(str(html_path))
        fig.write_image(str(png_path), scale=2)
        print(f"[salvo] {html_path}")
        print(f"[salvo] {png_path}")


def save_all_plots(
    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    tables: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
) -> None:
    for horizon in HORIZONS:
        plot_root  = OUTPUT_ROOT / f"RV_t+{horizon}" / "03_plots"
        ts_oos_dir = plot_root / "01_timeseries" / "out_of_sample"
        bar_dir    = plot_root / "02_metric_bars"

        _plot_timeseries_oos(data, horizon, ts_oos_dir)
        _plot_metric_bars(tables, horizon, bar_dir)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PASSO 1/3 — Carregando dados e executando RV-FTS...")
    print("=" * 70)
    data = build_all_data(verbose=True)

    print("\n" + "=" * 70)
    print("PASSO 2/3 — Calculando métricas...")
    print("=" * 70)
    tables = compute_metrics_table(data)

    # Imprimir resumo no terminal
    for horizon in HORIZONS:
        print(f"\n--- t+{horizon} | OOS | MSE ---")
        print(tables[horizon]["oos"]["MSE"].to_string(float_format="{:.4e}".format))
    print()

    print("\n" + "=" * 70)
    print("PASSO 3/3 — Salvando planilhas, previsões e gráficos...")
    print("=" * 70)
    save_overview_spreadsheets(tables)
    save_metric_spreadsheets(tables)
    save_prediction_spreadsheets(data)
    save_all_plots(data, tables)

    print(f"\nConcluído. Resultados em: {OUTPUT_ROOT.resolve()}")
