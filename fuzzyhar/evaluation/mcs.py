# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Model Confidence Set (arch). Codigo verbatim do notebook.'''
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from fuzzyhar.config import SYMBOLS

# ============================================================
# UTILIDADES DE LEITURA / PRE-PROCESSAMENTO
# ============================================================

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
        engine=engine,
    )
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def _auto_pick_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lower = {c: str(c).lower() for c in df.columns}
    real_keys = ("rv", "realized", "realizada", "real", "target")
    har_keys = ("har",)
    real_col = None
    har_col = None
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
        df.loc[df[c] <= 0.0, c] = np.nan
        df[c] = df[c].ffill()
    return df


# ============================================================
# PERDAS POR TEMPO
# ============================================================

def qlike_loss_t(y_real: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(y_real.astype(float), eps, None)
    h = np.clip(y_pred.astype(float), eps, None)
    r = x / h
    return r - np.log(r) - 1.0


def mse_loss_t(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    d = y_real.astype(float) - y_pred.astype(float)
    return d ** 2


def losses_from_excel(
    file_path: Union[str, Path],
    *,
    metric: str = "QLIKE",
    sheet: Union[int, str] = 0,
    header: Optional[int] = 0,
    index_col: Optional[Union[int, str]] = None,
    usecols: Optional[str] = None,
    engine: Optional[str] = None,
    real_col: Optional[Union[str, int]] = None,
    drop_negative_fuzzy: bool = False,
    fuzzy_col_name: Union[str, Sequence[str]] = "Fuzzy",
    model_subset: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    metric = metric.upper().strip()
    if metric not in ("QLIKE", "MSE"):
        raise ValueError("metric deve ser 'QLIKE' ou 'MSE'.")

    df = _read_excel_as_df(
        file_path, sheet=sheet, header=header,
        index_col=index_col, usecols=usecols, engine=engine,
    )
    df = _apply_optional_filters(
        df, drop_negative_fuzzy=drop_negative_fuzzy,
        fuzzy_col_name=fuzzy_col_name,
    )

    cols_list = list(df.columns)
    if isinstance(real_col, int):
        real_col = cols_list[real_col]
    if real_col is None:
        auto_real, _ = _auto_pick_columns(df)
        real_col = auto_real
    if real_col is None or real_col not in df.columns:
        raise ValueError(f"Nao foi possivel identificar a coluna realizada. Arquivo: {file_path}")

    y = df[real_col].astype(float)
    preds = df.drop(columns=[real_col])

    if model_subset is not None:
        wanted = [m for m in model_subset if m in preds.columns]
        missing = [m for m in model_subset if m not in preds.columns]
        if len(wanted) < 2:
            raise ValueError(
                f"Precisa de ao menos 2 modelos. Encontrados: {wanted}. Ausentes: {missing}."
            )
        preds = preds[wanted]

    y_np = y.to_numpy()
    losses: Dict[str, np.ndarray] = {}
    for m in preds.columns:
        h_np = preds[m].astype(float).to_numpy()
        if metric == "QLIKE":
            losses[m] = qlike_loss_t(y_np, h_np, eps=eps)
        else:
            losses[m] = mse_loss_t(y_np, h_np)

    L = pd.DataFrame(losses, index=df.index)
    L = L.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return L


# ============================================================
# MCS VIA arch (SUBSTITUI rpy2 + pacote R MCS)
# ============================================================

def mcs_included_models(
    losses_df: pd.DataFrame,
    *,
    confidence: float = 0.70,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    verbose: bool = False,
) -> List[str]:
    """
    Roda o Model Confidence Set usando arch.bootstrap.MCS.

    Parametros
    ----------
    statistic : str
        "max" (equivale a Tmax do R) ou "R" (equivale a TR/Trange do R).
    """
    from arch.bootstrap import MCS as ArchMCS

    losses_df = (
        losses_df.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
    )

    if losses_df.shape[1] < 2:
        return list(losses_df.columns)
    if losses_df.shape[0] == 0:
        raise ValueError("Nao ha linhas validas para o MCS apos limpeza.")

    alpha = 1.0 - float(confidence)

    # Mapeia nomes de estatistica do R para arch
    stat_map = {
        "tmax": "max",
        "tr": "R",
        "trange": "R",
        "max": "max",
        "r": "R",
    }
    stat = stat_map.get(statistic.lower(), statistic)

    mcs = ArchMCS(losses_df, size=alpha, bootstrap="stationary", reps=B, method=stat, seed=seed)
    mcs.compute()

    # mcs.included contém os nomes dos modelos no Superior Set
    included = list(mcs.included)

    if verbose:
        print(f"  MCS alpha={alpha}, stat={stat}, B={B}")
        print(f"  Included: {included}")
        print(f"  Excluded: {list(mcs.excluded)}")

    return included


# ============================================================
# TABELA MCS POR MOEDA
# ============================================================

def mcs_table_by_coin(
    files: Iterable[Tuple[Union[str, Path], str]],
    *,
    metric: str = "QLIKE",
    confidence: float = 0.70,
    out_path: Optional[Union[str, Path]] = None,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    sheet: Union[int, str] = 0,
    header: Optional[int] = 0,
    index_col: Optional[Union[int, str]] = None,
    usecols: Optional[str] = None,
    engine: Optional[str] = None,
    real_col: Optional[Union[str, int]] = None,
    drop_negative_fuzzy: bool = False,
    fuzzy_col_name: Union[str, Sequence[str]] = "Fuzzy",
    model_subset: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
    verbose: bool = True,
) -> pd.DataFrame:
    files = list(files)
    if not files:
        raise ValueError("A lista 'files' esta vazia.")

    metric = metric.upper().strip()
    if metric not in ("QLIKE", "MSE"):
        raise ValueError("metric deve ser 'QLIKE' ou 'MSE'.")

    included_by_coin: Dict[str, List[str]] = {}
    all_models = set()

    for fp, coin in files:
        L = losses_from_excel(
            fp, metric=metric, sheet=sheet, header=header,
            index_col=index_col, usecols=usecols, engine=engine,
            real_col=real_col, drop_negative_fuzzy=drop_negative_fuzzy,
            fuzzy_col_name=fuzzy_col_name, model_subset=model_subset, eps=eps,
        )

        inc = mcs_included_models(
            L, confidence=confidence, B=B,
            statistic=statistic, seed=seed, verbose=False,
        )

        included_by_coin[coin] = inc
        all_models |= set(L.columns)

        if verbose:
            print(f"[{metric}] {coin} | MCS {int(confidence * 100)}% inclui {len(inc)} modelo(s): {inc}")

    all_models = sorted(all_models)
    coins = [coin for _, coin in files]
    mat = pd.DataFrame(0, index=all_models, columns=coins, dtype=int)

    for coin, inc_list in included_by_coin.items():
        for m in inc_list:
            if m in mat.index:
                mat.loc[m, coin] = 1

    mat["Total"] = mat[coins].sum(axis=1) / float(len(coins))
    mat = mat.sort_values(["Total"], ascending=False)

    if out_path is None:
        out_path = f"MCS_{metric}_{int(confidence * 100)}pct.xlsx"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat.to_excel(out_path, engine="openpyxl")

    if verbose:
        print(f"Tabela MCS salva em: {out_path.resolve()}")

    return mat


# ============================================================
# AUTO-DISCOVERY DOS ARQUIVOS OOS
# ============================================================

def _pick_oos_merged_file(merged_dir: Path, symbol: str) -> Optional[Path]:
    with_fuzzy = merged_dir / f"{symbol}__oos_merged_with_fuzzy.xlsx"
    classic = merged_dir / f"{symbol}__oos_merged.xlsx"
    if with_fuzzy.exists():
        return with_fuzzy
    if classic.exists():
        return classic
    return None


def build_files_from_pipeline(
    merged_predictions_root: Union[str, Path],
    horizon: int,
    tickers: Sequence[str],
    *,
    verbose: bool = True,
) -> List[Tuple[Path, str]]:
    merged_predictions_root = Path(merged_predictions_root)
    files: List[Tuple[Path, str]] = []
    for t in tickers:
        sym = str(t).upper()
        merged_dir = merged_predictions_root / f"RV_t+{horizon}" / sym / "out-of-sample"
        fp = _pick_oos_merged_file(merged_dir=merged_dir, symbol=sym)
        if fp is None:
            if verbose:
                print(f"[WARN] Nenhum merged OOS encontrado para {sym} em {merged_dir}")
            continue
        coin = sym.replace("USDT", "")
        files.append((fp, coin))
        if verbose:
            print(f"[OK] {sym} -> {fp.name}")
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo OOS valido encontrado em {merged_predictions_root} para RV_t+{horizon}."
        )
    return files


# ============================================================
# ORQUESTRACAO POR HORIZONTE
# ============================================================

def run_mcs_for_horizon(
    horizon: int,
    merged_predictions_root: Union[str, Path],
    tickers: Sequence[str],
    modelos_mcs: Sequence[str],
    *,
    verbose: bool = True,
    confidence: float = 0.70,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
):
    merged_predictions_root = Path(merged_predictions_root)
    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 80)
        print(f"Rodando horizonte RV_t+{horizon}")
        print("=" * 80)

    files = build_files_from_pipeline(
        merged_predictions_root=merged_predictions_root,
        horizon=horizon, tickers=tickers, verbose=verbose,
    )

    common_kwargs = dict(
        sheet=0, header=0, index_col=None,
        real_col="Target", drop_negative_fuzzy=True,
        fuzzy_col_name=["Fuzzy HAR"], model_subset=modelos_mcs,
    )

    attachment_prefix_by_horizon = {
        1: {"MSE": "attachment_8_", "QLIKE": "attachment_9_"},
        7: {"MSE": "attachment_13_", "QLIKE": "attachment_14_"},
        30: {"MSE": "attachment_18_", "QLIKE": "attachment_19_"},
    }

    prefix_q = attachment_prefix_by_horizon.get(horizon, {}).get("QLIKE", "")
    prefix_m = attachment_prefix_by_horizon.get(horizon, {}).get("MSE", "")

    mcs_qlike = mcs_table_by_coin(
        files, metric="QLIKE", confidence=confidence,
        out_path=results_dir / f"{prefix_q}MCS_QLIKE_{confidence}pct_confidence_restrito_RV_t+{horizon}_025.xlsx",
        B=B, statistic=statistic, seed=seed, **common_kwargs,
    )

    mcs_mse = mcs_table_by_coin(
        files, metric="MSE", confidence=confidence,
        out_path=results_dir / f"{prefix_m}MCS_MSE_{confidence}pct_confidence_restrito_RV_t+{horizon}_025.xlsx",
        B=B, statistic=statistic, seed=seed, **common_kwargs,
    )

    return {
        "horizon": horizon,
        "results_dir": results_dir,
        "files": files,
        "mcs_qlike": mcs_qlike,
        "mcs_mse": mcs_mse,
    }


def run_all_mcs():
    # CONFIGURACOES E EXECUCAO
    # ============================================================

    MERGED_PREDICTIONS_ROOT = Path("Others") / "Merged Predictions"
    HORIZONS = [1, 7, 30]

    if "SYMBOLS" in globals() and isinstance(SYMBOLS, (list, tuple)) and len(SYMBOLS) > 0:
        TICKERS = [str(s).upper() for s in SYMBOLS]
    else:
        TICKERS = [
            "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
            "ETHUSDT", "TRXUSDT", "XLMUSDT", "XRPUSDT",
        ]

    MODELOS_MCS = [
        "Fuzzy HAR",
        "HAR OOS prediction",
        "HAR-CJ OOS prediction",
        "HAR-SJ OOS prediction",
        "HAR-TCJ OOS prediction",
        "LHAR-TCJ OOS prediction",
    ]

    mcs_results = {}
    for h in HORIZONS:
        mcs_results[h] = run_mcs_for_horizon(
            horizon=h,
            merged_predictions_root=MERGED_PREDICTIONS_ROOT,
            tickers=TICKERS,
            modelos_mcs=MODELOS_MCS,
            verbose=True,
            confidence=0.75,
            B=10000,
            statistic="max",
            seed=123,
        )
