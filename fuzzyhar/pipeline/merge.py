# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Merge das previsoes (classicos + Fuzzy). Codigo verbatim do notebook.'''
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from fuzzyhar.config import (
    SYMBOLS, get_fuzzy_prediction_output_dir, get_prediction_output_dir,
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
HORIZONS = [1, 7, 30]

# Usa exclusivamente a lista definida em Parameters
_SYMBOLS_FROM_PARAMETERS = list(SYMBOLS)

# Novo destino centralizado para os arquivos mergeados
MERGED_PREDICTIONS_ROOT = Path("Others") / "Merged Predictions"

# ------------------------------------------------------------
# Helpers: parsing de filenames
# ------------------------------------------------------------
# Corrigido: insample_train (o .py original tinha "insample_rain")
_INSAMPLE_RE = re.compile(r"(.+)__([^_]+)__insample_train\.xlsx$")
_OOS_RE      = re.compile(r"(.+)__([^_]+)__oos_predictions\.xlsx$")

def _parse_insample_filename(path: Path) -> Tuple[str | None, str | None]:
    m = _INSAMPLE_RE.match(path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)  # base, modelo

def _parse_oos_filename(path: Path) -> Tuple[str | None, str | None]:
    m = _OOS_RE.match(path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)  # base, modelo

def _merged_output_dirs(horizon: int, symbol: str) -> dict[str, Path]:
    root = MERGED_PREDICTIONS_ROOT / f"RV_t+{horizon}" / symbol
    in_dir = root / "insample_train"
    oos_dir = root / "out-of-sample"
    in_dir.mkdir(parents=True, exist_ok=True)
    oos_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "in_dir": in_dir, "oos_dir": oos_dir}

# ------------------------------------------------------------
# Nucleo: merge generico por "moeda"
# ------------------------------------------------------------
def _merge_por_moeda(
    files: List[Path],
    parse_fn,
    is_oos: bool,
    out_dir: Path
) -> List[Path]:
    """
    Recebe uma lista de arquivos de previsao (INSAMPLE ou OOS),
    agrupa por moeda (base_name) e faz o merge horizontal das previsoes por modelo.
    Retorna a lista de caminhos de arquivos salvos.
    """
    if not files:
        return []

    buckets: Dict[str, List[Tuple[Path, str]]] = {}
    for f in files:
        base, modelo = parse_fn(f)
        if base is None:
            continue
        buckets.setdefault(base, []).append((f, modelo))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_files: List[Path] = []

    for base, items in buckets.items():
        merged = None
        keep_cols = ["Data", "retorno_diario", "Target"]

        for f, modelo in items:
            df = pd.read_excel(f)

            if "Date" in df.columns and "Data" not in df.columns:
                df = df.rename(columns={"Date": "Data"})
            if "target" in df.columns and "Target" not in df.columns:
                df = df.rename(columns={"target": "Target"})

            if "Data" in df.columns:
                df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

            # Descobre e renomeia a coluna de previsao
            if is_oos:
                desired = f"{modelo} OOS prediction"
                detected = [c for c in df.columns if c.endswith("OOS prediction")]
                if len(detected) == 1 and detected[0] != desired:
                    df = df.rename(columns={detected[0]: desired})
                pred_col = desired
            else:
                detected = [
                    c for c in df.columns
                    if c.endswith("prediction") and not c.endswith("OOS prediction")
                ]
                if len(detected) == 1:
                    df = df.rename(columns={detected[0]: f"{modelo} prediction"})
                pred_col = f"{modelo} prediction"

            cols = [c for c in keep_cols if c in df.columns] + [pred_col]
            cols = [c for c in cols if c in df.columns]
            df = df[cols].copy()

            if merged is None:
                merged = df.sort_values("Data")
            else:
                # mantem a logica do script original:
                # merge pela intersecao das chaves disponiveis
                on_cols = [
                    c for c in ["Data", "retorno_diario", "Target"]
                    if c in merged.columns and c in df.columns
                ]
                merged = pd.merge(merged, df, on=on_cols, how="outer")

        if merged is None or merged.empty:
            print(f"Aviso: {base}: nada para mesclar ({'OOS' if is_oos else 'IN'}).")
            continue

        merged = merged.sort_values("Data").reset_index(drop=True)
        if is_oos:
            out_file = out_dir / f"{base}__oos_merged.xlsx"
        else:
            out_file = out_dir / f"{base}__insample_merged.xlsx"

        merged.to_excel(out_file, index=False)
        out_files.append(out_file)
        print(f"MERGED {'OOS' if is_oos else 'IN'}: {out_file}")

    return out_files

# ------------------------------------------------------------
# Merge classico na estrutura atual do notebook
# ------------------------------------------------------------
def merge_outputs_classicos(
    horizons: Iterable[int] = HORIZONS,
    symbols_to_run: Iterable[str] | None = None,
) -> None:
    if symbols_to_run is None:
        symbols_to_run = _SYMBOLS_FROM_PARAMETERS

    MERGED_PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        horizon_root = get_prediction_output_dir(h)
        if not horizon_root.exists():
            print(f"Aviso: horizonte nao encontrado: {horizon_root}")
            continue

        print(f"\n================ Horizonte: RV_t+{h} ================")

        symbol_dirs = sorted([p for p in horizon_root.iterdir() if p.is_dir()])

        allowed = set(symbols_to_run)
        symbol_dirs = [p for p in symbol_dirs if p.name in allowed]

        for symbol_dir in symbol_dirs:
            symbol = symbol_dir.name
            merged_dirs = _merged_output_dirs(horizon=h, symbol=symbol)

            ins_dir = symbol_dir / "insample_train"
            oos_dir = symbol_dir / "out-of-sample"

            # ---------- IN-SAMPLE ----------
            if ins_dir.exists():
                ins_files = sorted(ins_dir.glob("*__*__insample_train.xlsx"))
                if ins_files:
                    _merge_por_moeda(
                        ins_files,
                        _parse_insample_filename,
                        is_oos=False,
                        out_dir=merged_dirs["in_dir"],
                    )
                else:
                    print(f"Aviso: nenhum *__insample_train.xlsx encontrado em {ins_dir}")
            else:
                print(f"Aviso: pasta nao encontrada: {ins_dir}")

            # ---------- OUT-OF-SAMPLE ----------
            if oos_dir.exists():
                oos_files = sorted(oos_dir.glob("*__*__oos_predictions.xlsx"))
                if oos_files:
                    _merge_por_moeda(
                        oos_files,
                        _parse_oos_filename,
                        is_oos=True,
                        out_dir=merged_dirs["oos_dir"],
                    )
                else:
                    print(f"Aviso: nenhum *__oos_predictions.xlsx encontrado em {oos_dir}")
            else:
                print(f"Aviso: pasta nao encontrada: {oos_dir}")

# ------------------------------------------------------------
# Ponte com os outputs do Fuzzy HAR do notebook
# Nao altera os arquivos classicos; so cria os merged_with_fuzzy
# ------------------------------------------------------------
def _standardize_fuzzy_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Date" in out.columns and "Data" not in out.columns:
        out = out.rename(columns={"Date": "Data"})
    if "target" in out.columns and "Target" not in out.columns:
        out = out.rename(columns={"target": "Target"})

    if "Data" in out.columns:
        out["Data"] = pd.to_datetime(out["Data"], errors="coerce")

    return out

def _pick_fuzzy_cols(df: pd.DataFrame) -> list[str]:
    priority = [
        "Fuzzy HAR",
        "Fuzzy",
        "Fuzzy_Har_Winsorized",
        "Fuzzy_Winsorized",
        "Fuzzy Lagged",
    ]

    candidates = [c for c in priority if c in df.columns]

    for c in df.columns:
        if "fuzzy" in str(c).lower() and c not in candidates:
            candidates.append(c)

    return candidates

def merge_classic_with_fuzzy_for_symbol(symbol: str, horizon: int) -> None:
    fuzzy_symbol_root = get_fuzzy_prediction_output_dir(horizon) / symbol
    merged_dirs = _merged_output_dirs(horizon=horizon, symbol=symbol)

    classic_in = merged_dirs["in_dir"] / f"{symbol}__insample_merged.xlsx"
    classic_oos = merged_dirs["oos_dir"] / f"{symbol}__oos_merged.xlsx"

    fuzzy_in = fuzzy_symbol_root / f"{symbol}_tplus{horizon}_train_with_fuzzy.xlsx"
    fuzzy_oos = fuzzy_symbol_root / f"{symbol}_tplus{horizon}_test_with_fuzzy.xlsx"

    out_in = merged_dirs["in_dir"] / f"{symbol}__insample_merged_with_fuzzy.xlsx"
    out_oos = merged_dirs["oos_dir"] / f"{symbol}__oos_merged_with_fuzzy.xlsx"

    # ---------- IN ----------
    if classic_in.exists() and fuzzy_in.exists():
        df_classic_in = _standardize_fuzzy_df(pd.read_excel(classic_in))
        df_fuzzy_in = _standardize_fuzzy_df(pd.read_excel(fuzzy_in))

        fuzzy_cols = _pick_fuzzy_cols(df_fuzzy_in)
        cols_to_keep = ["Data"] + [c for c in fuzzy_cols if c in df_fuzzy_in.columns]
        df_fuzzy_in = df_fuzzy_in[cols_to_keep].copy()

        merged_in = pd.merge(df_classic_in, df_fuzzy_in, on="Data", how="left")
        merged_in = merged_in.sort_values("Data").reset_index(drop=True)
        merged_in.to_excel(out_in, index=False)
        print(f"MERGED WITH FUZZY IN: {out_in}")
    else:
        print(f"Aviso: IN nao gerado para {symbol} | t+{horizon} (faltam arquivos).")

    # ---------- OOS ----------
    if classic_oos.exists() and fuzzy_oos.exists():
        df_classic_oos = _standardize_fuzzy_df(pd.read_excel(classic_oos))
        df_fuzzy_oos = _standardize_fuzzy_df(pd.read_excel(fuzzy_oos))

        fuzzy_cols = _pick_fuzzy_cols(df_fuzzy_oos)
        cols_to_keep = ["Data"] + [c for c in fuzzy_cols if c in df_fuzzy_oos.columns]
        df_fuzzy_oos = df_fuzzy_oos[cols_to_keep].copy()

        merged_oos = pd.merge(df_classic_oos, df_fuzzy_oos, on="Data", how="left")
        merged_oos = merged_oos.sort_values("Data").reset_index(drop=True)
        merged_oos.to_excel(out_oos, index=False)
        print(f"MERGED WITH FUZZY OOS: {out_oos}")
    else:
        print(f"Aviso: OOS nao gerado para {symbol} | t+{horizon} (faltam arquivos).")

def merge_classic_with_fuzzy_all(
    horizons: Iterable[int] = HORIZONS,
    symbols_to_run: Iterable[str] | None = None,
) -> None:
    if symbols_to_run is None:
        symbols_to_run = _SYMBOLS_FROM_PARAMETERS

    for h in horizons:
        for sym in symbols_to_run:
            try:
                merge_classic_with_fuzzy_for_symbol(sym, h)
            except Exception as e:
                print(f"Erro em {sym} | t+{h}: {e}")


def run_all_merge():
    # EXECUCAO
    # ------------------------------------------------------------
    merge_outputs_classicos(HORIZONS, symbols_to_run=_SYMBOLS_FROM_PARAMETERS)
    merge_classic_with_fuzzy_all(HORIZONS, symbols_to_run=_SYMBOLS_FROM_PARAMETERS)
