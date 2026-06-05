from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison"
DEFAULT_OUTPUT_SUBDIR = "04_mcs"
DEFAULT_HORIZONS = [1, 7, 30]
DEFAULT_MODEL_SUBSET = [
    "Fuzzy HAR",
    "HAR OOS prediction",
    "HAR-CJ OOS prediction",
    "HAR-SJ OOS prediction",
    "HAR-TCJ OOS prediction",
    "LHAR-TCJ OOS prediction",
    "RV-FTS",
]
DEFAULT_FILTER_COLS = list(DEFAULT_MODEL_SUBSET)


def _normalize_coin(value: str) -> str:
    coin = str(value).strip().upper()
    if coin.endswith("USDT"):
        coin = coin[:-4]
    return coin


def _confidence_label(confidence: float) -> str:
    pct = int(round(float(confidence) * 100))
    return f"{pct}pct"


def _read_prediction_df(file_path: Union[str, Path]) -> pd.DataFrame:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Formato nao suportado: {file_path}")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in df.columns:
        if col == "Date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _apply_prediction_filters(
    df: pd.DataFrame,
    *,
    drop_nonpositive: bool = True,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if not drop_nonpositive:
        return df

    df = df.copy()
    if columns is None:
        columns = DEFAULT_FILTER_COLS

    for col in columns:
        if col not in df.columns:
            continue
        neg_mask = df[col] <= 0.0
        if neg_mask.any():
            df.loc[neg_mask, col] = np.nan
            df[col] = df[col].ffill()

    return df


def qlike_loss_t(y_real: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(y_real.astype(float), eps, None)
    h = np.clip(y_pred.astype(float), eps, None)
    r = x / h
    return r - np.log(r) - 1.0


def mse_loss_t(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    d = y_real.astype(float) - y_pred.astype(float)
    return d ** 2


def losses_from_prediction_file(
    file_path: Union[str, Path],
    *,
    metric: str = "QLIKE",
    real_col: str = "Target",
    model_subset: Optional[Sequence[str]] = None,
    drop_nonpositive: bool = True,
    filter_cols: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    metric = metric.upper().strip()
    if metric not in ("QLIKE", "MSE"):
        raise ValueError("metric deve ser 'QLIKE' ou 'MSE'.")

    df = _read_prediction_df(file_path)
    df = _apply_prediction_filters(
        df,
        drop_nonpositive=drop_nonpositive,
        columns=filter_cols,
    )

    if real_col not in df.columns:
        raise ValueError(f"Coluna realizada '{real_col}' ausente em {file_path}.")

    if model_subset is None:
        excluded = {"Date", "RV", real_col}
        model_cols = [c for c in df.columns if c not in excluded]
    else:
        model_cols = [c for c in model_subset if c in df.columns]
        missing = [c for c in model_subset if c not in df.columns]
        if len(model_cols) < 2:
            raise ValueError(
                f"Precisa de ao menos 2 modelos presentes em {file_path}. "
                f"Encontrados: {model_cols}. Ausentes: {missing}."
            )

    y_np = df[real_col].to_numpy(dtype=float)
    losses: Dict[str, np.ndarray] = {}
    for model in model_cols:
        pred_np = df[model].to_numpy(dtype=float)
        if metric == "QLIKE":
            losses[model] = qlike_loss_t(y_np, pred_np, eps=eps)
        else:
            losses[model] = mse_loss_t(y_np, pred_np)

    index = df["Date"] if "Date" in df.columns else df.index
    losses_df = pd.DataFrame(losses, index=index)
    losses_df = losses_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if losses_df.empty:
        raise ValueError(
            f"Nao ha linhas validas para MCS apos limpeza em {file_path}."
        )

    return losses_df


def mcs_included_models(
    losses_df: pd.DataFrame,
    *,
    confidence: float = 0.75,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    verbose: bool = False,
) -> List[str]:
    try:
        from arch.bootstrap import MCS as ArchMCS
    except ImportError as exc:
        raise ImportError(
            "O pacote 'arch' nao esta instalado. Instale com: pip install arch"
        ) from exc

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
    stat_map = {
        "tmax": "max",
        "tr": "R",
        "trange": "R",
        "max": "max",
        "r": "R",
    }
    stat = stat_map.get(statistic.lower(), statistic)

    mcs = ArchMCS(
        losses_df,
        size=alpha,
        bootstrap="stationary",
        reps=B,
        method=stat,
        seed=seed,
    )
    mcs.compute()

    included = list(mcs.included)
    if verbose:
        print(f"  MCS alpha={alpha}, stat={stat}, B={B}")
        print(f"  Included: {included}")
        print(f"  Excluded: {list(mcs.excluded)}")

    return included


def mcs_table_by_coin(
    files: Iterable[Tuple[Union[str, Path], str]],
    *,
    metric: str = "QLIKE",
    confidence: float = 0.75,
    out_path: Optional[Union[str, Path]] = None,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    real_col: str = "Target",
    drop_nonpositive: bool = True,
    filter_cols: Optional[Sequence[str]] = None,
    model_subset: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
    verbose: bool = True,
) -> pd.DataFrame:
    files = list(files)
    if not files:
        raise ValueError("A lista de arquivos esta vazia.")

    metric = metric.upper().strip()
    if metric not in ("QLIKE", "MSE"):
        raise ValueError("metric deve ser 'QLIKE' ou 'MSE'.")

    included_by_coin: Dict[str, List[str]] = {}
    all_models = set()

    for file_path, coin in files:
        losses_df = losses_from_prediction_file(
            file_path,
            metric=metric,
            real_col=real_col,
            model_subset=model_subset,
            drop_nonpositive=drop_nonpositive,
            filter_cols=filter_cols,
            eps=eps,
        )
        included = mcs_included_models(
            losses_df,
            confidence=confidence,
            B=B,
            statistic=statistic,
            seed=seed,
            verbose=False,
        )
        included_by_coin[coin] = included
        all_models |= set(losses_df.columns)

        if verbose:
            print(
                f"[{metric}] {coin} | MCS {int(confidence * 100)}% inclui "
                f"{len(included)} modelo(s): {included}"
            )

    coins = [coin for _, coin in files]
    matrix = pd.DataFrame(0, index=sorted(all_models), columns=coins, dtype=int)

    for coin, included in included_by_coin.items():
        for model in included:
            if model in matrix.index:
                matrix.loc[model, coin] = 1

    matrix["Count"] = matrix[coins].sum(axis=1)
    matrix["Share"] = matrix["Count"] / float(len(coins))
    matrix = matrix.sort_values(["Share", "Count"], ascending=False)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_excel(out_path, engine="openpyxl")
        if verbose:
            print(f"Tabela MCS salva em: {out_path.resolve()}")

    return matrix


def _pick_prediction_file(predictions_root: Path, coin: str, horizon: int) -> Optional[Path]:
    base = predictions_root / coin
    xlsx = base / f"{coin}_oos_tplus{horizon}.xlsx"
    csv = base / f"{coin}_oos_tplus{horizon}.csv"
    if xlsx.exists():
        return xlsx
    if csv.exists():
        return csv
    return None


def discover_prediction_files(
    results_root: Union[str, Path],
    horizon: int,
    tickers: Optional[Sequence[str]] = None,
    *,
    verbose: bool = True,
) -> List[Tuple[Path, str]]:
    results_root = Path(results_root)
    predictions_root = results_root / f"RV_t+{horizon}" / "02_predictions"

    if not predictions_root.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {predictions_root}")

    if tickers:
        coins = [_normalize_coin(t) for t in tickers]
    else:
        coins = sorted([p.name for p in predictions_root.iterdir() if p.is_dir()])

    files: List[Tuple[Path, str]] = []
    for coin in coins:
        fp = _pick_prediction_file(predictions_root, coin, horizon)
        if fp is None:
            if verbose:
                print(f"[WARN] Arquivo OOS nao encontrado para {coin} em {predictions_root / coin}")
            continue
        files.append((fp, coin))
        if verbose:
            print(f"[OK] {coin} -> {fp.name}")

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo OOS valido encontrado em {predictions_root}."
        )

    return files


def save_horizon_summary(
    out_path: Union[str, Path],
    *,
    qlike_table: pd.DataFrame,
    mse_table: pd.DataFrame,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        qlike_table.to_excel(writer, sheet_name="QLIKE")
        mse_table.to_excel(writer, sheet_name="MSE")


def run_mcs_for_horizon(
    horizon: int,
    results_root: Union[str, Path],
    *,
    output_dir: Union[str, Path],
    tickers: Optional[Sequence[str]] = None,
    model_subset: Optional[Sequence[str]] = None,
    verbose: bool = True,
    confidence: float = 0.75,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    real_col: str = "Target",
    drop_nonpositive: bool = True,
    filter_cols: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> Dict[str, object]:
    if verbose:
        print("\n" + "=" * 80)
        print(f"Rodando MCS para RV_t+{horizon}")
        print("=" * 80)

    files = discover_prediction_files(
        results_root=results_root,
        horizon=horizon,
        tickers=tickers,
        verbose=verbose,
    )

    out_dir = Path(output_dir) / f"RV_t+{horizon}"
    conf_label = _confidence_label(confidence)

    qlike_path = out_dir / f"MCS_QLIKE_{conf_label}_RV_tplus{horizon}.xlsx"
    mse_path = out_dir / f"MCS_MSE_{conf_label}_RV_tplus{horizon}.xlsx"
    summary_path = out_dir / f"MCS_summary_{conf_label}_RV_tplus{horizon}.xlsx"

    common_kwargs = dict(
        confidence=confidence,
        B=B,
        statistic=statistic,
        seed=seed,
        real_col=real_col,
        drop_nonpositive=drop_nonpositive,
        filter_cols=filter_cols,
        model_subset=model_subset,
        eps=eps,
        verbose=verbose,
    )

    qlike_table = mcs_table_by_coin(
        files,
        metric="QLIKE",
        out_path=qlike_path,
        **common_kwargs,
    )
    mse_table = mcs_table_by_coin(
        files,
        metric="MSE",
        out_path=mse_path,
        **common_kwargs,
    )
    save_horizon_summary(
        summary_path,
        qlike_table=qlike_table,
        mse_table=mse_table,
    )

    if verbose:
        print(f"Resumo MCS salvo em: {summary_path.resolve()}")

    return {
        "horizon": horizon,
        "files": files,
        "qlike_path": qlike_path,
        "mse_path": mse_path,
        "summary_path": summary_path,
        "mcs_qlike": qlike_table,
        "mcs_mse": mse_table,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Roda o Model Confidence Set (MCS) diretamente nas planilhas de "
            "Results/HAR_FuzzyHAR_RVFTS_Comparison."
        )
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Raiz da pasta HAR_FuzzyHAR_RVFTS_Comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pasta de saida. Default: <results-root>/04_mcs",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Horizontes a processar. Ex.: --horizons 1 7 30",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers a processar. Aceita ADA ou ADAUSDT. Default: auto-discovery.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODEL_SUBSET),
        help="Subset de modelos a incluir no MCS.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.75,
        help="Nivel de confianca do MCS. Ex.: 0.75",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=10000,
        help="Numero de repeticoes bootstrap.",
    )
    parser.add_argument(
        "--statistic",
        default="max",
        help="Estatistica do MCS: max, R, tmax, tr ou trange.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Semente do bootstrap.",
    )
    parser.add_argument(
        "--real-col",
        default="Target",
        help="Nome da coluna realizada.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Epsilon numerico para QLIKE.",
    )
    parser.add_argument(
        "--no-drop-nonpositive",
        action="store_true",
        help="Nao substituir previsoes <= 0 por NaN + ffill antes do MCS.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduz a verbosidade.",
    )
    return parser


def main() -> Dict[int, Dict[str, object]]:
    parser = build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / DEFAULT_OUTPUT_SUBDIR
    verbose = not args.quiet

    if verbose:
        print(f"Results root: {results_root.resolve()}")
        print(f"Output dir:   {output_dir.resolve()}")

    outputs: Dict[int, Dict[str, object]] = {}
    for horizon in args.horizons:
        outputs[horizon] = run_mcs_for_horizon(
            horizon=horizon,
            results_root=results_root,
            output_dir=output_dir,
            tickers=args.tickers,
            model_subset=args.models,
            verbose=verbose,
            confidence=args.confidence,
            B=args.B,
            statistic=args.statistic,
            seed=args.seed,
            real_col=args.real_col,
            drop_nonpositive=not args.no_drop_nonpositive,
            filter_cols=DEFAULT_FILTER_COLS,
            eps=args.eps,
        )

    return outputs


if __name__ == "__main__":
    main()
