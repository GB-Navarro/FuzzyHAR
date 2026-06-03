# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Tabelas de MSE & QLIKE por horizonte. Codigo verbatim do notebook.'''
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

from fuzzyhar.config import SYMBOLS

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
HORIZONS = [1, 7, 30]
_SYMBOLS_FROM_PARAMETERS = list(SYMBOLS)

# Salva todos os outputs desta celula diretamente em Results
RESULTS_FLAT_DIR = Path("Results")

# Origem dos arquivos mergeados (atualizada)
MERGED_PREDICTIONS_ROOT = Path("Others") / "Merged Predictions"

# Prefixos de nome por horizonte e metrica
ATTACHMENT_PREFIX_BY_HORIZON = {
    1: {"MSE": "attachment_6", "QLIKE": "attachment_7"},
    7: {"MSE": "attachment_11", "QLIKE": "attachment_12"},
    30: {"MSE": "attachment_16", "QLIKE": "attachment_17"},
}

def _build_metric_output_path(horizon: int, metric: str) -> Path:
    metric_u = metric.upper().strip()
    if horizon not in ATTACHMENT_PREFIX_BY_HORIZON:
        raise ValueError(f"Horizonte sem prefixo configurado: t+{horizon}")
    if metric_u not in ATTACHMENT_PREFIX_BY_HORIZON[horizon]:
        raise ValueError(f"Metrica sem prefixo configurado: {metric_u} em t+{horizon}")

    prefix = ATTACHMENT_PREFIX_BY_HORIZON[horizon][metric_u]
    return RESULTS_FLAT_DIR / f"{prefix}_{metric_u}_relativo_agrupado_por_modelo_tplus{horizon}.xlsx"

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def _read_excel_as_df(
    file_path: Union[str, Path],
    *,
    sheet: Union[int, str] = 0,
    header: Optional[int] = 0,
    index_col: Optional[Union[int, str]] = None,
    usecols: Optional[str] = None,
    engine: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_excel(
        file_path,
        sheet_name=sheet,
        header=header,
        index_col=index_col,
        usecols=usecols,
        engine=engine
    )

    for c in ["Data", "Date"]:
        if c in df.columns:
            df = df.drop(columns=c)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(axis=1, how="all")
    return df

def _auto_pick_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lower = {c: str(c).lower() for c in df.columns}
    real_keys = ("rv", "realized", "realizada", "real", "target")
    har_keys  = ("har",)

    real_col = None
    har_col  = None
    for c, lc in lower.items():
        if real_col is None and any(k in lc for k in real_keys):
            real_col = c
        if har_col is None and any(k in lc for k in har_keys):
            har_col = c
    return real_col, har_col

def _apply_optional_filters(
    df: pd.DataFrame,
    *,
    drop_negative_fuzzy: bool = False,
    fuzzy_col_name: Union[str, Sequence[str]] = "Fuzzy",
) -> pd.DataFrame:
    if not drop_negative_fuzzy:
        return df

    if isinstance(fuzzy_col_name, (str, bytes)):
        fuzzy_cols = [fuzzy_col_name]
    else:
        fuzzy_cols = list(fuzzy_col_name)

    fuzzy_cols = [c for c in fuzzy_cols if c in df.columns]
    if not fuzzy_cols:
        return df

    for c in fuzzy_cols:
        neg_mask = df[c] <= 0.00000000
        df.loc[neg_mask, c] = np.nan
        df[c] = df[c].ffill()

    return df

# ------------------------------------------------------------
# Metricas
# ------------------------------------------------------------
def qlike(y_real: pd.Series, y_pred: pd.Series, eps: float = 1e-12) -> float:
    pair = pd.concat({"x": y_real.astype(float), "h": y_pred.astype(float)}, axis=1).dropna()
    x = np.clip(pair["x"].to_numpy(), eps, None)
    h = np.clip(pair["h"].to_numpy(), eps, None)
    r = x / h
    return float(np.mean(r - np.log(r) - 1.0))

def mse(y_real: pd.Series, y_pred: pd.Series) -> float:
    pair = pd.concat({"y": y_real.astype(float), "h": y_pred.astype(float)}, axis=1).dropna()
    diff = pair["y"].to_numpy() - pair["h"].to_numpy()
    return float(np.mean(diff**2))

_METRIC_FUNCS: Dict[str, callable] = {
    "QLIKE": qlike,
    "MSE": mse,
}

# ------------------------------------------------------------
# Tabela por arquivo
# ------------------------------------------------------------
def metric_table_from_excel(
    file_path: Union[str, Path],
    *,
    metric: str = "QLIKE",
    sheet: Union[int, str] = 0,
    header: Optional[int] = 0,
    index_col: Optional[Union[int, str]] = None,
    usecols: Optional[str] = None,
    engine: Optional[str] = None,
    real_col: Optional[Union[str, int]] = None,
    har_col: Optional[Union[str, int]] = None,
    drop_negative_fuzzy: bool = False,
    fuzzy_col_name: Union[str, Sequence[str]] = "Fuzzy",
    eps: float = 1e-12,
) -> pd.DataFrame:
    metric = metric.upper().strip()
    if metric not in _METRIC_FUNCS:
        raise ValueError(f"Metrica invalida: '{metric}'. Use 'QLIKE' ou 'MSE'.")

    f_metric = _METRIC_FUNCS[metric]

    df = _read_excel_as_df(
        file_path, sheet=sheet, header=header, index_col=index_col,
        usecols=usecols, engine=engine
    )
    df = _apply_optional_filters(
        df,
        drop_negative_fuzzy=drop_negative_fuzzy,
        fuzzy_col_name=fuzzy_col_name,
    )

    if df.shape[1] < 2:
        raise ValueError("A planilha precisa ter ao menos 2 colunas (realizada e HAR).")

    cols_list = list(df.columns)
    if isinstance(real_col, int):
        real_col = cols_list[real_col]
    if isinstance(har_col, int):
        har_col = cols_list[har_col]

    if real_col is None or har_col is None:
        auto_real, auto_har = _auto_pick_columns(df)
        real_col = auto_real if real_col is None else real_col
        har_col  = auto_har  if har_col  is None else har_col

    if real_col is None or real_col not in df.columns:
        raise ValueError("Nao foi possivel identificar a coluna da serie realizada.")
    if har_col is None or har_col not in df.columns:
        raise ValueError("Nao foi possivel identificar a coluna HAR.")

    meta_cols = [c for c in ["retorno_diario", "retornos_diarios"] if c in df.columns]
    preds = df.drop(columns=[real_col] + meta_cols)

    rows = []
    for model_name in preds.columns:
        if metric == "QLIKE":
            val = f_metric(df[real_col], preds[model_name], eps=eps)
        else:
            val = f_metric(df[real_col], preds[model_name])
        rows.append({"Modelo": model_name, metric: val})

    out = pd.DataFrame(rows).set_index("Modelo")

    if har_col not in out.index:
        cand = [i for i in out.index if str(i).lower() == str(har_col).lower()]
        if cand:
            har_col = cand[0]
        else:
            raise ValueError("A coluna HAR nao apareceu no conjunto de modelos para normalizacao.")

    base = out.loc[har_col, metric]
    denom = base if base != 0 else 1e-12
    out[f"{metric}_norm"] = out[metric] / denom
    out["Rank"] = out[f"{metric}_norm"].rank(method="min")
    out = out.sort_values(f"{metric}_norm")
    return out

# ------------------------------------------------------------
# Agregacao final
# ------------------------------------------------------------
def metrics_to_excel(
    files: Iterable[Tuple[Union[str, Path], str]],
    *,
    metric: str = "QLIKE",
    out_path: Union[str, Path] = None,
    **common_kwargs,
) -> pd.DataFrame:
    files = list(files)
    if not files:
        raise ValueError("A lista 'files' esta vazia.")

    labels = [label for _, label in files]
    if len(labels) != len(set(labels)):
        raise ValueError("Existem rotulos repetidos em 'files'. Use nomes unicos.")

    metric = metric.upper().strip()
    if metric not in _METRIC_FUNCS:
        raise ValueError(f"Metrica invalida: '{metric}'. Use 'QLIKE' ou 'MSE'.")

    tables: Dict[str, pd.DataFrame] = {}
    for fp, label in files:
        t = metric_table_from_excel(fp, metric=metric, **common_kwargs)
        tables[label] = t[[f"{metric}_norm"]].rename(columns={f"{metric}_norm": label})

    agg = None
    for i, (_, label) in enumerate(files):
        agg = tables[label] if i == 0 else agg.join(tables[label], how="outer")

    agg = agg.sort_index()

    if out_path is None:
        out_path = f"{metric}_relativo_agrupado_por_modelo.xlsx"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_excel(out_path)
    print(f"Planilha salva em: {out_path}")

    return agg

# ------------------------------------------------------------
# Descoberta automatica dos merged_with_fuzzy do notebook
# ------------------------------------------------------------
def build_metric_file_list_for_horizon(
    horizon: int,
    symbols_to_run: Iterable[str] | None = None,
    prefer_with_fuzzy: bool = True,
) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []

    if symbols_to_run is None:
        symbols_to_run = _SYMBOLS_FROM_PARAMETERS

    for sym in symbols_to_run:
        merged_dir = MERGED_PREDICTIONS_ROOT / f"RV_t+{horizon}" / sym / "out-of-sample"

        fp_with_fuzzy = merged_dir / f"{sym}__oos_merged_with_fuzzy.xlsx"
        fp_classic = merged_dir / f"{sym}__oos_merged.xlsx"

        if prefer_with_fuzzy and fp_with_fuzzy.exists():
            files.append((fp_with_fuzzy, sym.replace("USDT", "")))
        elif fp_classic.exists():
            files.append((fp_classic, sym.replace("USDT", "")))
        else:
            print(f"Aviso: arquivo merged nao encontrado para {sym} | t+{horizon}")

    return files

def run_metrics_for_all_horizons(
    horizons: Iterable[int] = HORIZONS,
    symbols_to_run: Iterable[str] | None = None,
):
    common_kwargs = dict(
        sheet=0,
        header=0,
        index_col=None,
        real_col="Target",
        har_col="HAR OOS prediction",
        fuzzy_col_name=[
            "Fuzzy",
            "Fuzzy HAR",
            "Fuzzy_Har_Winsorized",
            "Fuzzy_Winsorized",
            "Fuzzy Lagged",
            "HAR OOS prediction",
            "HAR-CJ OOS prediction",
            "HAR-SJ OOS prediction",
            "HAR-TCJ OOS prediction",
            "LHAR-TCJ OOS prediction",
        ],
    )

    all_outputs = {}

    if symbols_to_run is None:
        symbols_to_run = _SYMBOLS_FROM_PARAMETERS

    RESULTS_FLAT_DIR.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        files = build_metric_file_list_for_horizon(
            horizon=h,
            symbols_to_run=symbols_to_run,
            prefer_with_fuzzy=True,
        )

        if not files:
            print(f"Aviso: nenhum arquivo de metricas encontrado para t+{h}.")
            continue

        out_qlike = _build_metric_output_path(horizon=h, metric="QLIKE")
        out_mse = _build_metric_output_path(horizon=h, metric="MSE")

        agg_qlike = metrics_to_excel(
            files,
            metric="QLIKE",
            out_path=out_qlike,
            drop_negative_fuzzy=True,
            **common_kwargs,
        )
        print(f"\n[QLIKE t+{h}]")
        print(agg_qlike.head())

        agg_mse = metrics_to_excel(
            files,
            metric="MSE",
            out_path=out_mse,
            drop_negative_fuzzy=True,
            **common_kwargs,
        )
        print(f"\n[MSE t+{h}]")
        print(agg_mse.head())

        all_outputs[h] = {
            "QLIKE": agg_qlike,
            "MSE": agg_mse,
            "files": files,
            "out_qlike": out_qlike,
            "out_mse": out_mse,
        }

    return all_outputs


def run_all_metrics():
    # EXECUCAO
    # ------------------------------------------------------------
    metrics_outputs = run_metrics_for_all_horizons(
        horizons=HORIZONS,
        symbols_to_run=_SYMBOLS_FROM_PARAMETERS,
    )
