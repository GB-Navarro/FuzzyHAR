"""
rvfts_pipeline.py
=================
Pipeline consolidado RV-FTS (Realized Volatility Fuzzy Time Series).

Este script unifica em um único arquivo o que antes estava distribuído em:
    1) rvfts_model.py       — modelo RV-FTS (FCM + EWMA + CV rolante)
    2) integracao_rvfts.py  — carregamento, métricas, exportação e plots
    3) mcs_rvfts_results.py — Model Confidence Set sobre as previsões OOS

Referência: Iqbal, S. & Lyócsa, Š. (2026). "A Fuzzy Framework for Realized
Volatility Prediction: Empirical Evidence From Equity Markets". Journal of
Forecasting, 45, 1261–1291. doi:10.1002/for.70082

Mudanças vs. versão em três scripts
-----------------------------------
1.  Pasta de saída do MCS migrou de
        Results/HAR_FuzzyHAR_RVFTS_Comparison/04_mcs/RV_t+{h}/
    para
        Results/HAR_FuzzyHAR_RVFTS_Comparison/RV_t+{h}/04_mcs/
    (mantém a estrutura por horizonte que já é usada para 01_metrics,
     02_predictions e 03_plots).

2.  Novo passo final: gera um relatório Markdown
        Results/HAR_FuzzyHAR_RVFTS_Comparison/00_overview/
            rvfts_parameters_report.md
    listando, para cada moeda × horizonte:
      • parâmetros usados (EW, CS, grids de c e ρ, janela, semente);
      • parâmetros selecionados pela cross-validação rolante (distribuição
        empírica de c e ρ ao longo do tempo, valor modal e # de previsões
        válidas em train/test).

Estrutura de pastas produzida
-----------------------------
    Results/HAR_FuzzyHAR_RVFTS_Comparison/
        00_overview/
            all_metrics_long_format.{xlsx,csv}
            all_metrics_by_split.xlsx
            rvfts_parameters_report.md          ← NOVO
        RV_t+1/
            01_metrics/
                metrics_summary_tplus1.xlsx
                mse_oos_tplus1.xlsx
                qlike_oos_tplus1.xlsx
            02_predictions/
                ADA/
                    ADA_oos_tplus1.{xlsx,csv}
                ...
            03_plots/
                01_timeseries/out_of_sample/{html,png}/
                02_metric_bars/{html,png}/
            04_mcs/                              ← MOVIDO PARA CÁ
                MCS_QLIKE_75pct_RV_tplus1.xlsx
                MCS_MSE_75pct_RV_tplus1.xlsx
                MCS_summary_75pct_RV_tplus1.xlsx
        RV_t+7/  …
        RV_t+30/ …

Como usar
---------
    # Pipeline completo:
    python rvfts_pipeline.py

    # Apenas MCS, partindo de previsões já salvas em disco:
    python rvfts_pipeline.py --only-mcs --horizons 1 7 30

    # Pular MCS (gera apenas previsões / métricas / plots / relatório):
    python rvfts_pipeline.py --skip-mcs
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import (
    Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd


# =============================================================================
# 0.  CONFIGURAÇÃO GLOBAL
# =============================================================================

# --- Universo & horizontes ---------------------------------------------------
SYMBOLS: List[str] = [
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "TRXUSDT", "XLMUSDT", "XRPUSDT",
]
HORIZONS: List[int] = [1, 7, 30]

# --- Splits ------------------------------------------------------------------
DEFAULT_SPLIT: str = "40_60"
VALID_SPLITS:  List[str] = ["40_60", "60_40", "70_30"]

# --- Pastas (funções parametrizadas por split) ----------------------------
def _fuzzy_root(split: str) -> Path:
    return Path("Others") / "Predictions" / split / "Fuzzy HAR"

def _har_models_root(split: str) -> Path:
    return Path("Others") / "Predictions" / split / "HAR Models"

def _output_root(split: str) -> Path:
    return Path("Results") / "HAR_FuzzyHAR_RVFTS_Comparison" / split

# Aliases para o split padrão (retrocompatibilidade com uso direto das constantes)
FUZZY_ROOT      = _fuzzy_root(DEFAULT_SPLIT)
HAR_MODELS_ROOT = _har_models_root(DEFAULT_SPLIT)
OUTPUT_ROOT     = _output_root(DEFAULT_SPLIT)

# --- Modelos HAR a carregar e ordem de exibição ------------------------------
HAR_MODELS:  List[str] = ["HAR", "HAR-CJ", "HAR-SJ", "HAR-TCJ", "LHAR-TCJ"]
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

# --- Hiperparâmetros do RV-FTS — versão "full" do paper (Section 3.6) -------
RVFTS_EW: int            = 756
RVFTS_CS: int            = 252
RVFTS_NCLUS: List[int]   = [2, 3, 4, 6, 9]
RVFTS_RHO:   List[float] = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
RVFTS_WINDOW: str        = "rolling"
RVFTS_FCM_SEED: int      = 42

# --- MCS ---------------------------------------------------------------------
MCS_DEFAULT_MODELS: List[str] = [
    "Fuzzy HAR",
    "HAR OOS prediction",
    "HAR-CJ OOS prediction",
    "HAR-SJ OOS prediction",
    "HAR-TCJ OOS prediction",
    "LHAR-TCJ OOS prediction",
    "RV-FTS",
]
MCS_DEFAULT_FILTER_COLS: List[str] = list(MCS_DEFAULT_MODELS)

# --- Filtro de previsões para QLIKE (replica drop_negative_fuzzy do notebook)
_FUZZY_PRED_COLS: List[str] = [
    "Fuzzy HAR",
    "HAR OOS prediction",
    "HAR-CJ OOS prediction",
    "HAR-SJ OOS prediction",
    "HAR-TCJ OOS prediction",
    "LHAR-TCJ OOS prediction",
    "RV-FTS",
]

# --- Métricas ----------------------------------------------------------------
EPS: float = 1e-12


# =============================================================================
# 1.  FUZZY C-MEANS  (1-D, Bezdek; equivalente a e1071::cmeans com m=2)
# =============================================================================

def fcm_1d(
    x: np.ndarray,
    c: int,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuzzy C-Means clustering para dados unidimensionais.

    Returns
    -------
    centroids : array (c,)
    U         : array (n, c)  — matriz de pertinência γ_{ij}.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n <= c:
        # Caso degenerado: cada ponto vira seu próprio cluster
        centroids = x[:c].copy()
        U = np.eye(n, c)
        return centroids, U

    U = rng.rand(n, c).astype(np.float64)
    U /= U.sum(axis=1, keepdims=True)

    exp = 2.0 / (m - 1.0)

    for _ in range(max_iter):
        Um = U ** m
        denom = np.maximum(Um.sum(axis=0), 1e-300)
        centroids = (Um.T @ x) / denom

        dist = np.abs(x[:, None] - centroids[None, :])
        dist = np.maximum(dist, 1e-15)

        ratio = (dist[:, :, None] / dist[:, None, :]) ** exp
        U_new = 1.0 / ratio.sum(axis=2)

        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new

    return centroids, U


# =============================================================================
# 2.  MODELO RV-FTS  (com cross-validation rolante)
# =============================================================================

def rvfts_forecast(
    target_series: np.ndarray,
    h: int,
    ew: int = 756,
    cs: int = 252,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    window: str = "rolling",
    m: float = 2.0,
    cv_decay: float = 0.99,
    fcm_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Forecaster RV-FTS conforme Iqbal & Lyócsa (2026).

    Convenção de alinhamento
    ------------------------
    - target_series[t] é a quantidade a prever na linha t (forward h-day RV).
    - target_series[s] é observável ao final do dia s + h - 1.
    - predictions[t] é a previsão *para* target_series[t], feita ao final do
      dia t - 1 (sem look-ahead).

    Returns
    -------
    dict com: predictions, fuzzified, opt_c, opt_rho, pred_all, fuzz_all
    """
    if nclus_grid is None:
        nclus_grid = [2, 3, 4, 6, 9]
    if rho_grid is None:
        rho_grid = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
    if h < 1:
        raise ValueError(f"h deve ser >= 1; recebido h={h}.")
    if window not in ("rolling", "expanding"):
        raise ValueError(f"window deve ser 'rolling' ou 'expanding'; recebido '{window}'.")

    y = np.asarray(target_series, dtype=np.float64).copy()
    T = len(y)
    NC = len(nclus_grid)
    NL = len(rho_grid)

    min_T = ew + h + 1          # t_start + 1: primeiro t com ≥1 ponto de CV
    if T < min_T:
        raise ValueError(
            f"Série muito curta: T={T} < ew+h+1={min_T}. "
            f"Reduza ew ou forneça mais dados."
        )

    # ----------------------------------------------------------------------
    # Phase 1 — FCM por janela rolante para cada nº de clusters
    # ----------------------------------------------------------------------
    fuzz_all = np.full((NC, T), np.nan)
    t0 = time.time()
    if verbose:
        print(f"[RV-FTS] Phase 1/3: FCM × {NC} valores de c × ~{T-ew-h+1} janelas (h={h})")

    t_start = ew + h - 1

    for nc_idx, nc in enumerate(nclus_grid):
        if verbose:
            print(f"  c = {nc} ...", end=" ", flush=True)
        count = 0
        for t in range(t_start, T):
            end_excl = t - h + 1
            if end_excl <= 0:
                continue

            if window == "rolling":
                start_idx = max(0, end_excl - ew)
            else:  # expanding
                start_idx = 0

            win = y[start_idx:end_excl]
            mask = ~np.isnan(win)
            win = win[mask]

            if len(win) < max(nc + 1, 5):
                continue

            try:
                centroids, U = fcm_1d(win, nc, m=m, seed=fcm_seed)
                membership_last = U[-1]
                fuzz_all[nc_idx, t] = float(np.dot(membership_last, centroids))
                count += 1
            except Exception:
                continue

        if verbose:
            print(f"({count} janelas OK)")

    # ----------------------------------------------------------------------
    # Phase 2 — Recursão EWMA para cada (c, ρ) candidato
    # ----------------------------------------------------------------------
    pred_all = np.full((NC, NL, T), np.nan)
    if verbose:
        print(f"[RV-FTS] Phase 2/3: EWMA × {NC}×{NL} = {NC*NL} combinações")

    for nc_idx in range(NC):
        fuzz_nc = fuzz_all[nc_idx]
        valid_idx = np.where(~np.isnan(fuzz_nc))[0]
        if len(valid_idx) == 0:
            continue
        first = int(valid_idx[0])

        for nl_idx, rho in enumerate(rho_grid):
            preds = np.full(T, np.nan)
            preds[first] = fuzz_nc[first]   # init: 1ª previsão = ṽ inicial

            for t in range(first + 1, T):
                f_t = fuzz_nc[t]
                p_prev = preds[t - 1]

                if np.isnan(f_t) and np.isnan(p_prev):
                    continue
                elif np.isnan(f_t):
                    preds[t] = p_prev
                elif np.isnan(p_prev):
                    preds[t] = f_t
                else:
                    preds[t] = (1.0 - rho) * f_t + rho * p_prev

            pred_all[nc_idx, nl_idx] = preds

    # ----------------------------------------------------------------------
    # Phase 3 — CV rolante: seleciona (c, ρ) ótimos em cada t (sem look-ahead)
    # ----------------------------------------------------------------------
    final_preds = np.full(T, np.nan)
    final_fuzz  = np.full(T, np.nan)
    opt_c       = np.full(T, np.nan)
    opt_rho     = np.full(T, np.nan)

    if verbose:
        print(f"[RV-FTS] Phase 3/3: CV rolante (CS={cs}, decay={cv_decay}, h={h})")

    # Inicia o CV assim que o EWMA tem pelo menos 1 passo válido (t >= t_start).
    # O guarda `nv < 10` abaixo garante que CVs degenerados são ignorados.
    # cv_start usa max(0, ...) para tratar janelas menores no início da série,
    # evitando NaNs nas primeiras observações de teste quando n_train ≈ ew.
    first_cv = t_start
    for t in range(first_cv, T):
        cv_end   = t - h + 1                    # exclusive
        cv_start = max(0, t - h - cs + 1)       # usa janela menor no início

        cv_actual = y[cv_start:cv_end]

        best_loss = np.inf
        best_nc = 0
        best_nl = 0
        found = False

        for nc_idx in range(NC):
            for nl_idx in range(NL):
                cv_preds = pred_all[nc_idx, nl_idx, cv_start:cv_end]

                valid = ~(np.isnan(cv_actual) | np.isnan(cv_preds))
                nv = int(valid.sum())
                if nv < 10:
                    continue

                weights = cv_decay ** np.arange(nv - 1, -1, -1)
                errors  = (cv_actual[valid] - cv_preds[valid]) ** 2
                wloss   = float(np.average(errors, weights=weights))

                if wloss < best_loss:
                    best_loss = wloss
                    best_nc = nc_idx
                    best_nl = nl_idx
                    found = True

        if not found:
            continue

        final_preds[t] = pred_all[best_nc, best_nl, t]
        final_fuzz[t]  = fuzz_all[best_nc, t]
        opt_c[t]   = nclus_grid[best_nc]
        opt_rho[t] = rho_grid[best_nl]

    elapsed = time.time() - t0
    n_valid = int(np.sum(~np.isnan(final_preds)))
    if verbose:
        print(f"[RV-FTS] Concluído em {elapsed:.1f}s — {n_valid} previsões válidas.")

    return {
        "predictions": final_preds,
        "fuzzified":   final_fuzz,
        "opt_c":       opt_c,
        "opt_rho":     opt_rho,
        "pred_all":    pred_all,
        "fuzz_all":    fuzz_all,
    }


# =============================================================================
# 3.  MÉTRICAS  (eq. 20 e 21 do paper)
# =============================================================================

def mse_paper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE conforme eq. (20) do paper. Ignora NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return float("nan")
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def qlike_paper(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    """QLIKE conforme eq. (21): mean( y/f - log(y/f) - 1 ). Ignora NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return float("nan")
    yt = np.clip(y_true[mask], eps, None)
    fp = np.clip(y_pred[mask], eps, None)
    r = yt / fp
    return float(np.mean(r - np.log(r) - 1.0))


def qlike_loss_t(y_real: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> np.ndarray:
    """QLIKE ponto-a-ponto (vetor t), usado dentro do MCS."""
    x = np.clip(y_real.astype(float), eps, None)
    h = np.clip(y_pred.astype(float), eps, None)
    r = x / h
    return r - np.log(r) - 1.0


def mse_loss_t(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Erro quadrático ponto-a-ponto, usado dentro do MCS."""
    d = y_real.astype(float) - y_pred.astype(float)
    return d ** 2


# =============================================================================
# 4.  CARREGAMENTO DE DADOS  (Fuzzy HAR + HAR Models do notebook)
# =============================================================================

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


def _load_fuzzy_har(sym: str, horizon: int, split: str = DEFAULT_SPLIT,
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos do Fuzzy HAR (produzidos pelo notebook)."""
    base = _fuzzy_root(split) / f"RV_t+{horizon}" / sym
    tr_path = base / f"{sym}_tplus{horizon}_train_with_fuzzy.xlsx"
    te_path = base / f"{sym}_tplus{horizon}_test_with_fuzzy.xlsx"

    if not tr_path.exists():
        raise FileNotFoundError(f"Fuzzy HAR treino não encontrado: {tr_path}")
    if not te_path.exists():
        raise FileNotFoundError(f"Fuzzy HAR teste não encontrado: {te_path}")

    return _norm(pd.read_excel(tr_path)), _norm(pd.read_excel(te_path))


def _load_har_model(sym: str, horizon: int, model: str,
                    split: str = DEFAULT_SPLIT,
                    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Carrega previsões in-sample e OOS de um modelo HAR individual."""
    base = _har_models_root(split) / f"RV_t+{horizon}" / sym
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
    """Left-join de df_har em df_base por Date, renomeando a coluna de previsão."""
    if df_har is None or pred_col_in_har not in df_har.columns:
        return df_base

    right = df_har[["Date", pred_col_in_har]].rename(
        columns={pred_col_in_har: pred_col_out}
    )
    if pred_col_out in df_base.columns:
        df_base = df_base.drop(columns=[pred_col_out])

    if "Date" in df_base.columns:
        return df_base.merge(right, on="Date", how="left")
    else:
        df_base = df_base.copy()
        vals = df_har[pred_col_in_har].values
        df_base[pred_col_out] = vals[:len(df_base)] if len(vals) >= len(df_base) \
            else np.concatenate([vals, np.full(len(df_base) - len(vals), np.nan)])
        return df_base


def load_full_data(sym: str, horizon: int,
                   split: str = DEFAULT_SPLIT,
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida num único par (df_train, df_test) todas as previsões disponíveis
    para (sym, horizon): Fuzzy HAR + todos os modelos HAR.
    """
    df_tr, df_te = _load_fuzzy_har(sym, horizon, split=split)

    for model in HAR_MODELS:
        df_har_tr, df_har_oos = _load_har_model(sym, horizon, model, split=split)

        # Treino: "{model} train prediction" → "{model} prediction"
        df_tr = _merge_har_into(
            df_tr, df_har_tr,
            f"{model} train prediction",
            f"{model} prediction",
        )
        # OOS: mantém o nome "{model} OOS prediction"
        df_te = _merge_har_into(
            df_te, df_har_oos,
            f"{model} OOS prediction",
            f"{model} OOS prediction",
        )

    return df_tr, df_te


# =============================================================================
# 5.  INTEGRAÇÃO RV-FTS NOS DATAFRAMES DO PIPELINE
# =============================================================================

def add_rvfts_to_dataframes(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    horizon: int,
    target_column: str = "Target",
    rv_column: str = "RV",
    ew: int = 756,
    cs: int = 252,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    window: str = "rolling",
    fcm_seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Roda o RV-FTS sobre os dataframes do pipeline e adiciona a coluna 'RV-FTS'
    em ambos. Em cada linha d, RV-FTS[d] é a previsão feita ao final do dia
    d-1 para o valor de target_column[d].
    """
    df_train = df_train.copy()
    df_test  = df_test.copy()

    n_train = len(df_train)
    n_test  = len(df_test)

    use_target = (
        target_column in df_train.columns and target_column in df_test.columns
    )
    use_rv = (
        rv_column in df_train.columns and rv_column in df_test.columns
    )

    if use_target:
        series_full = np.concatenate([
            df_train[target_column].values.astype(float),
            df_test[target_column].values.astype(float),
        ])
        h_internal  = horizon
        used_column = target_column
    elif horizon == 1 and use_rv:
        # Para h=1 a coluna RV equivale ao target diário.
        series_full = np.concatenate([
            df_train[rv_column].values.astype(float),
            df_test[rv_column].values.astype(float),
        ])
        h_internal  = 1
        used_column = rv_column
    else:
        raise ValueError(
            f"Necessária a coluna '{target_column}' (RV forward de h-dias) ou "
            f"'{rv_column}' (apenas h=1). Disponíveis: "
            f"{list(df_train.columns)[:12]}"
        )

    # Adapta ew para garantir que o CV tenha ao menos nv_min erros válidos
    # no PRIMEIRO passo de teste (t = n_train).  A condição é:
    #   cv_end(t=n_train) > t_start  ∧  nv ≥ nv_min
    # onde cv_end = n_train - h + 1  e  t_start = ew_eff + h - 1.
    # Isolando ew_eff:  ew_eff ≤ n_train - 2h - nv_min + 2
    # Sem isso, as primeiras observações de teste ficam com previsão NaN
    # porque a janela CV cai antes do aquecimento do EWMA (ex: DOGE h=30).
    _NV_MIN = 10   # deve coincidir com o guarda `nv < 10` em rvfts_forecast
    ew_eff = min(ew, n_train - 2 * h_internal - _NV_MIN + 2)
    ew_eff = max(max(h_internal, 10), ew_eff)   # piso de segurança
    if ew_eff < ew and verbose:
        print(f"  [WARN] ew reduzido de {ew} para {ew_eff} "
              f"(n_train={n_train}, h={h_internal}; "
              f"garante nv>={_NV_MIN} no 1º passo de teste)")

    if verbose:
        print(f"\n[RV-FTS] horizon=t+{horizon} | input='{used_column}' "
              f"| h_interno={h_internal}")
        print(f"  Train: {n_train} obs | Test: {n_test} obs "
              f"| Total: {len(series_full)} obs")
        print(f"  EW={ew_eff} (pedido={ew}) | CS={cs} | window={window}")

    result = rvfts_forecast(
        target_series=series_full,
        h=h_internal,
        ew=ew_eff,
        cs=cs,
        nclus_grid=nclus_grid,
        rho_grid=rho_grid,
        window=window,
        fcm_seed=fcm_seed,
        verbose=verbose,
    )

    preds = result["predictions"]
    df_train["RV-FTS"] = preds[:n_train]
    df_test["RV-FTS"]  = preds[n_train:n_train + n_test]

    # Separa opt_c / opt_rho em train e test para o relatório.
    info = {
        "used_column":   used_column,
        "h_internal":    h_internal,
        "ew_used":       ew_eff,
        "opt_c":         result["opt_c"],
        "opt_rho":       result["opt_rho"],
        "opt_c_train":   result["opt_c"][:n_train],
        "opt_c_test":    result["opt_c"][n_train:n_train + n_test],
        "opt_rho_train": result["opt_rho"][:n_train],
        "opt_rho_test":  result["opt_rho"][n_train:n_train + n_test],
        "fuzzified":     result["fuzzified"],
        "n_valid_train": int(np.sum(~np.isnan(preds[:n_train]))),
        "n_valid_test":  int(np.sum(~np.isnan(preds[n_train:n_train + n_test]))),
    }

    if verbose:
        print(f"  → Previsões válidas: train={info['n_valid_train']}, "
              f"test={info['n_valid_test']}")

    return df_train, df_test, info


def build_all_data(
    symbols:  Iterable[str] = SYMBOLS,
    horizons: Iterable[int] = HORIZONS,
    *,
    split:      str = DEFAULT_SPLIT,
    ew:         int = RVFTS_EW,
    cs:         int = RVFTS_CS,
    nclus_grid: Optional[List[int]]   = None,
    rho_grid:   Optional[List[float]] = None,
    window:     str = RVFTS_WINDOW,
    fcm_seed:   int = RVFTS_FCM_SEED,
    verbose:    bool = True,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    Dict[int, Dict[str, Dict[str, Any]]],
]:
    """
    Para cada (horizon × symbol):
      1. Carrega Fuzzy HAR + HAR Models das pastas do notebook.
      2. Roda o RV-FTS, adicionando a coluna 'RV-FTS'.

    Retorna:
        data[horizon][sym] = {"train": df_tr, "test": df_te}
        info[horizon][sym] = dict (opt_c, opt_rho, etc.) — usado no relatório.
    """
    if nclus_grid is None:
        nclus_grid = list(RVFTS_NCLUS)
    if rho_grid is None:
        rho_grid = list(RVFTS_RHO)

    symbols  = list(symbols)
    horizons = list(horizons)

    data: Dict[int, Dict[str, Dict[str, pd.DataFrame]]] = {}
    info: Dict[int, Dict[str, Dict[str, Any]]] = {}

    total = len(horizons) * len(symbols)
    count = 0

    for horizon in horizons:
        data[horizon] = {}
        info[horizon] = {}

        for sym in symbols:
            count += 1
            tag = f"[{count}/{total}] {sym} t+{horizon}"
            try:
                df_tr, df_te = load_full_data(sym, horizon, split=split)

                if verbose:
                    print(f"\n>>> {tag}")
                    tr_cols = [c for c in df_tr.columns if "prediction" in c.lower()
                               or c in ("Fuzzy HAR",)]
                    print(f"  Colunas de previsão carregadas: {tr_cols}")
                    print(f"  train={len(df_tr)} obs | test={len(df_te)} obs")

                df_tr, df_te, sym_info = add_rvfts_to_dataframes(
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
                info[horizon][sym] = sym_info

            except FileNotFoundError as e:
                print(f"[SKIP] {tag}: {e}")
            except Exception as e:
                import traceback
                print(f"[ERRO] {tag}: {e}")
                traceback.print_exc()

    return data, info


# =============================================================================
# 6.  CÁLCULO DE MÉTRICAS  (MSE / QLIKE OOS por modelo × símbolo × horizonte)
# =============================================================================

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


def _apply_prediction_filters(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    drop_nonpositive: bool = True,
) -> pd.DataFrame:
    """
    Replica drop_negative_fuzzy do notebook: para cada coluna de previsão,
    substitui valores <= 0 por NaN e aplica forward-fill. Necessário para
    estabilizar o QLIKE.
    """
    if not drop_nonpositive:
        return df
    df = df.copy()
    cols = list(columns) if columns is not None else _FUZZY_PRED_COLS
    for col in cols:
        if col not in df.columns:
            continue
        neg_mask = df[col] <= 0.0
        if neg_mask.any():
            df.loc[neg_mask, col] = np.nan
            df[col] = df[col].ffill()
    return df


def compute_metrics_table(
    data:     Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    horizons: Iterable[int] = HORIZONS,
    symbols:  Iterable[str] = SYMBOLS,
) -> Dict[int, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calcula MSE e QLIKE OOS para todos os modelos × símbolos × horizontes.
    Retorna tables[horizon]["oos"] = {"MSE": df_mse, "QLIKE": df_qlike}.
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

            df = _apply_prediction_filters(df)
            y = df["Target"].values.astype(float)

            for model in MODEL_ORDER:
                col = _oos_col(model)
                if col not in df.columns:
                    continue
                f = df[col].values.astype(float)
                tables[horizon]["oos"]["MSE"  ].setdefault(model, {})[ticker] = mse_paper(y, f)
                tables[horizon]["oos"]["QLIKE"].setdefault(model, {})[ticker] = qlike_paper(y, f)

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


# =============================================================================
# 7.  EXPORTAÇÃO (planilhas + previsões)
# =============================================================================

def _safe_excel_writer(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """Salva múltiplos DataFrames num mesmo .xlsx; ignora vazios."""
    non_empty = {name: df for name, df in sheets.items()
                 if df is not None and not df.empty}
    if not non_empty:
        print(f"  [skip] {path.name}: sem dados para salvar.")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in non_empty.items():
            df.to_excel(writer, sheet_name=name[:31])
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
    split: str = DEFAULT_SPLIT,
) -> None:
    for horizon in horizons:
        out_dir = _output_root(split) / f"RV_t+{horizon}" / "01_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        _safe_excel_writer(
            out_dir / f"metrics_summary_tplus{horizon}.xlsx",
            {"OOS MSE":   tables[horizon]["oos"]["MSE"],
             "OOS QLIKE": tables[horizon]["oos"]["QLIKE"]},
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
    split: str = DEFAULT_SPLIT,
) -> None:
    out_dir = _output_root(split) / "00_overview"
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
    split: str = DEFAULT_SPLIT,
) -> None:
    for horizon in horizons:
        base_dir = _output_root(split) / f"RV_t+{horizon}" / "02_predictions"
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


# =============================================================================
# 8.  GRÁFICOS
# =============================================================================

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
    split: str = DEFAULT_SPLIT,
) -> None:
    for horizon in horizons:
        plot_root = _output_root(split) / f"RV_t+{horizon}" / "03_plots"
        _plot_timeseries_oos(data, horizon,
                             plot_root / "01_timeseries" / "out_of_sample",
                             symbols=symbols)
        _plot_metric_bars(tables, horizon, plot_root / "02_metric_bars")


# =============================================================================
# 9.  MODEL CONFIDENCE SET
# =============================================================================

def _confidence_label(confidence: float) -> str:
    pct = int(round(float(confidence) * 100))
    return f"{pct}pct"


def _normalize_coin(value: str) -> str:
    coin = str(value).strip().upper()
    if coin.endswith("USDT"):
        coin = coin[:-4]
    return coin


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


def losses_from_prediction_file(
    file_path: Union[str, Path],
    *,
    metric: str = "QLIKE",
    real_col: str = "Target",
    model_subset: Optional[Sequence[str]] = None,
    drop_nonpositive: bool = True,
    filter_cols: Optional[Sequence[str]] = None,
    eps: float = EPS,
) -> pd.DataFrame:
    metric = metric.upper().strip()
    if metric not in ("QLIKE", "MSE"):
        raise ValueError("metric deve ser 'QLIKE' ou 'MSE'.")

    df = _read_prediction_df(file_path)
    df = _apply_prediction_filters(
        df,
        columns=filter_cols if filter_cols is not None else MCS_DEFAULT_FILTER_COLS,
        drop_nonpositive=drop_nonpositive,
    )

    if real_col not in df.columns:
        raise ValueError(f"Coluna realizada '{real_col}' ausente em {file_path}.")

    if model_subset is None:
        excluded = {"Date", "RV", real_col}
        model_cols = [c for c in df.columns if c not in excluded]
    else:
        model_cols = [c for c in model_subset if c in df.columns]
        missing    = [c for c in model_subset if c not in df.columns]
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


def _ar_block_size(losses_df: pd.DataFrame, min_k: int = 3) -> int:
    """
    Replica a selecao de block_size do R MCSprocedure:
        k = max(apply(Loss, 2, function(x) ar(x)$order))
        if (k < min.k) k = min.k

    R usa Yule-Walker + AIC; aqui usamos ar_select_order (OLS + AIC) do
    statsmodels, que produz resultados equivalentes para series financeiras.
    """
    try:
        from statsmodels.tsa.ar_model import ar_select_order
    except ImportError:
        return min_k

    orders = []
    for col in losses_df.columns:
        x = losses_df[col].dropna().values.astype(float)
        n = len(x)
        max_lag = min(int(10 * np.log10(max(n, 10))), n // 4, 50)
        try:
            sel = ar_select_order(x, maxlag=max_lag, ic="aic", old_names=False)
            order = max(sel.ar_lags) if sel.ar_lags else 0
        except Exception:
            order = 0
        orders.append(order)

    k = max(orders) if orders else min_k
    return max(k, min_k)


def mcs_included_models(
    losses_df: pd.DataFrame,
    *,
    confidence: float = 0.75,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    block_size: Optional[int] = None,
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
        "tmax": "max", "tr": "R", "trange": "R",
        "max": "max", "r": "R",
    }
    stat = stat_map.get(statistic.lower(), statistic)

    if block_size is None:
        block_size = _ar_block_size(losses_df)

    mcs = ArchMCS(
        losses_df,
        size=alpha,
        bootstrap="stationary",
        block_size=block_size,
        reps=B,
        method=stat,
        seed=seed,
    )
    mcs.compute()

    included = list(mcs.included)
    if verbose:
        print(f"  MCS alpha={alpha}, stat={stat}, B={B}, block_size={block_size}")
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
    block_size: Optional[int] = None,
    real_col: str = "Target",
    drop_nonpositive: bool = True,
    filter_cols: Optional[Sequence[str]] = None,
    model_subset: Optional[Sequence[str]] = None,
    eps: float = EPS,
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
            block_size=block_size,
            verbose=False,
        )
        included_by_coin[coin] = included
        all_models |= set(losses_df.columns)
        if verbose:
            print(
                f"[{metric}] {coin} | MCS {int(confidence * 100)}% inclui "
                f"{len(included)} modelo(s): {included}"
            )

    coins  = [coin for _, coin in files]
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
    csv  = base / f"{coin}_oos_tplus{horizon}.csv"
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
    """Descobre os arquivos OOS já salvos por save_prediction_spreadsheets."""
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


def _save_horizon_summary(
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
    results_root: Union[str, Path, None] = None,
    *,
    split: str = DEFAULT_SPLIT,
    tickers:      Optional[Sequence[str]] = None,
    model_subset: Optional[Sequence[str]] = None,
    verbose: bool = True,
    confidence: float = 0.75,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    block_size: Optional[int] = None,
    real_col: str = "Target",
    drop_nonpositive: bool = True,
    filter_cols: Optional[Sequence[str]] = None,
    eps: float = EPS,
) -> Dict[str, object]:
    if results_root is None:
        results_root = _output_root(split)
    """
    Roda o MCS para todos os tickers em um dado horizonte.

    Mudança vs. versão antiga: a saída fica em
        <results_root>/RV_t+{horizon}/04_mcs/
    em vez de <output_dir>/RV_t+{horizon}/.
    """
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

    # ⇩⇩⇩ MUDANÇA DE PASTA: agora dentro de RV_t+{horizon}/04_mcs/  ⇩⇩⇩
    out_dir = Path(results_root) / f"RV_t+{horizon}" / "04_mcs"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_label = _confidence_label(confidence)

    qlike_path   = out_dir / f"MCS_QLIKE_{conf_label}_RV_tplus{horizon}.xlsx"
    mse_path     = out_dir / f"MCS_MSE_{conf_label}_RV_tplus{horizon}.xlsx"
    summary_path = out_dir / f"MCS_summary_{conf_label}_RV_tplus{horizon}.xlsx"

    common_kwargs = dict(
        confidence=confidence, B=B, statistic=statistic, seed=seed,
        block_size=block_size,
        real_col=real_col, drop_nonpositive=drop_nonpositive,
        filter_cols=filter_cols, model_subset=model_subset,
        eps=eps, verbose=verbose,
    )

    qlike_table = mcs_table_by_coin(files, metric="QLIKE", out_path=qlike_path,
                                    **common_kwargs)
    mse_table   = mcs_table_by_coin(files, metric="MSE",   out_path=mse_path,
                                    **common_kwargs)
    _save_horizon_summary(summary_path, qlike_table=qlike_table, mse_table=mse_table)

    if verbose:
        print(f"Resumo MCS salvo em: {summary_path.resolve()}")

    return {
        "horizon": horizon, "files": files,
        "qlike_path": qlike_path, "mse_path": mse_path,
        "summary_path": summary_path,
        "mcs_qlike": qlike_table, "mcs_mse": mse_table,
    }


def run_mcs_all_horizons(
    horizons: Iterable[int] = HORIZONS,
    results_root: Union[str, Path, None] = None,
    *,
    split: str = DEFAULT_SPLIT,
    tickers:      Optional[Sequence[str]] = None,
    model_subset: Optional[Sequence[str]] = None,
    confidence: float = 0.75,
    B: int = 10000,
    statistic: str = "max",
    seed: int = 123,
    block_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, object]]:
    if results_root is None:
        results_root = _output_root(split)
    out: Dict[int, Dict[str, object]] = {}
    for horizon in horizons:
        out[horizon] = run_mcs_for_horizon(
            horizon=horizon,
            results_root=results_root,
            split=split,
            tickers=tickers,
            model_subset=model_subset,
            confidence=confidence,
            B=B, statistic=statistic, seed=seed,
            block_size=block_size,
            filter_cols=MCS_DEFAULT_FILTER_COLS,
            verbose=verbose,
        )
    return out


# =============================================================================
# 10.  RELATÓRIO MARKDOWN — parâmetros usados e selecionados pelo modelo
# =============================================================================

def _format_distribution(values: np.ndarray, fmt: str = "{}") -> List[Tuple[str, int, float]]:
    """Conta a distribuição de valores não-NaN e retorna [(label, n, %), ...]
    ordenado por n decrescente."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return []

    # Arredonda para 6 casas para casar floats com pequeno ruído numérico
    keys = np.round(arr, 6)
    unique, counts = np.unique(keys, return_counts=True)
    total = float(arr.size)

    out = []
    for u, c in sorted(zip(unique, counts), key=lambda t: -t[1]):
        if abs(u - round(u)) < 1e-9:
            label = fmt.format(int(round(u)))
        else:
            label = fmt.format(u)
        out.append((label, int(c), 100.0 * c / total))
    return out


def _modal(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    keys = np.round(arr, 6)
    unique, counts = np.unique(keys, return_counts=True)
    return float(unique[int(np.argmax(counts))])


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return (
            "| " + " | ".join(headers) + " |\n"
            "| " + " | ".join(["---"] * len(headers)) + " |\n"
            "| " + " | ".join(["—"] * len(headers)) + " |\n"
        )
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def generate_parameters_report(
    info: Dict[int, Dict[str, Dict[str, Any]]],
    *,
    out_path: Optional[Path] = None,
    split: str = DEFAULT_SPLIT,
    horizons: Iterable[int] = HORIZONS,
    symbols:  Iterable[str]  = SYMBOLS,
    ew: int = RVFTS_EW,
    cs: int = RVFTS_CS,
    nclus_grid: List[int]    = RVFTS_NCLUS,
    rho_grid:   List[float]  = RVFTS_RHO,
    window:    str = RVFTS_WINDOW,
    fcm_seed:  int = RVFTS_FCM_SEED,
) -> Path:
    """
    Gera um relatório Markdown listando, para cada moeda × horizonte,
    os parâmetros usados pelo RV-FTS e os parâmetros que a CV rolante
    selecionou ao longo do tempo (distribuição empírica de c e ρ no
    train e no test).
    """
    if out_path is None:
        out_path = _output_root(split) / "00_overview" / "rvfts_parameters_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    horizons = list(horizons)
    symbols  = list(symbols)

    lines: List[str] = []
    lines.append("# Relatório de Parâmetros do RV-FTS\n")
    lines.append(
        "Este relatório documenta, para cada combinação **moeda × horizonte**, "
        "os hiperparâmetros usados pelo modelo `RV-FTS` (Iqbal & Lyócsa, 2026) "
        "e os valores que a cross-validação rolante selecionou ao longo do tempo.\n"
    )

    # ---------- Configuração global ----------
    lines.append("## Configuração global\n")
    lines.append(_md_table(
        ["Parâmetro", "Valor"],
        [
            ["EW (janela inicial de estimação)", ew],
            ["CS (janela de calibração / CV)",   cs],
            ["Grid de c (nº de clusters)",       str(nclus_grid)],
            ["Grid de ρ (memória EWMA)",         str(rho_grid)],
            ["Tipo de janela",                   window],
            ["Semente do FCM",                   fcm_seed],
            ["Expoente m do FCM",                "2 (paper)"],
            ["Decaimento da CV",                 "0.99 por passo"],
            ["Horizontes",                       str(horizons)],
            ["Moedas",                           str(symbols)],
        ],
    ))
    lines.append("")
    lines.append(
        "> A cada t, a CV rolante seleciona o par (c, ρ) com menor MSE "
        "ponderado exponencialmente sobre os últimos `CS` erros; "
        "portanto cada moeda/horizonte tem uma **série temporal** de (c, ρ) "
        "ótimos. As tabelas abaixo resumem essa distribuição empírica.\n"
    )

    # ---------- Por horizonte e moeda ----------
    for horizon in horizons:
        lines.append(f"## Horizonte t+{horizon}\n")
        if horizon not in info or not info[horizon]:
            lines.append("_Sem dados disponíveis para este horizonte._\n")
            continue

        # Tabela de overview para este horizonte
        overview_rows = []
        for sym in symbols:
            if sym not in info[horizon]:
                overview_rows.append([sym, "—", "—", "—", "—", "—"])
                continue
            si = info[horizon][sym]
            n_tr = si.get("n_valid_train", 0)
            n_te = si.get("n_valid_test", 0)
            modal_c   = _modal(si["opt_c"])
            modal_rho = _modal(si["opt_rho"])
            modal_c_str   = str(int(modal_c))   if modal_c   is not None else "—"
            modal_rho_str = f"{modal_rho:g}"     if modal_rho is not None else "—"
            overview_rows.append([
                sym, n_tr, n_te,
                modal_c_str, modal_rho_str,
                si.get("used_column", "—"),
            ])

        lines.append("### Visão geral\n")
        lines.append(_md_table(
            ["Moeda", "Previsões válidas (train)", "Previsões válidas (test)",
             "c modal", "ρ modal", "Coluna de entrada"],
            overview_rows,
        ))
        lines.append("")

        # Detalhe por moeda
        for sym in symbols:
            if sym not in info[horizon]:
                continue
            si = info[horizon][sym]
            ticker = sym.replace("USDT", "")
            lines.append(f"### {ticker} ({sym}) — t+{horizon}\n")

            n_tr = si.get("n_valid_train", 0)
            n_te = si.get("n_valid_test", 0)
            lines.append(
                f"- Coluna de entrada: `{si.get('used_column', '—')}`  \n"
                f"- h interno usado pelo forecaster: `{si.get('h_internal', '—')}`  \n"
                f"- Previsões válidas: **{n_tr}** em train, **{n_te}** em test  \n"
            )

            # Distribuição de c selecionado
            lines.append("**Distribuição do nº de clusters `c` selecionado**\n")
            for split_name, key in (("train", "opt_c_train"), ("test", "opt_c_test")):
                dist = _format_distribution(si[key])
                if not dist:
                    lines.append(f"_{split_name}: nenhuma seleção registrada._\n")
                    continue
                rows = [[d[0], d[1], f"{d[2]:.1f}%"] for d in dist]
                lines.append(f"_Split: {split_name}_\n")
                lines.append(_md_table(["c", "n", "% do tempo"], rows))
                lines.append("")

            # Distribuição de ρ selecionado
            lines.append("**Distribuição do parâmetro de memória `ρ` selecionado**\n")
            for split_name, key in (("train", "opt_rho_train"), ("test", "opt_rho_test")):
                dist = _format_distribution(si[key], fmt="{:g}")
                if not dist:
                    lines.append(f"_{split_name}: nenhuma seleção registrada._\n")
                    continue
                rows = [[d[0], d[1], f"{d[2]:.1f}%"] for d in dist]
                lines.append(f"_Split: {split_name}_\n")
                lines.append(_md_table(["ρ", "n", "% do tempo"], rows))
                lines.append("")

            lines.append("---\n")

    # ---------- Escreve no disco ----------
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[salvo] {out_path}")
    return out_path


# =============================================================================
# 11.  ORQUESTRADOR PRINCIPAL
# =============================================================================

def run_full_pipeline(
    symbols:  Iterable[str] = SYMBOLS,
    horizons: Iterable[int] = HORIZONS,
    *,
    split:    str = DEFAULT_SPLIT,
    nclus_grid: Optional[List[int]]   = None,
    rho_grid:   Optional[List[float]] = None,
    ew: int = RVFTS_EW,
    cs: int = RVFTS_CS,
    window:   str = RVFTS_WINDOW,
    fcm_seed: int = RVFTS_FCM_SEED,
    skip_mcs: bool = False,
    mcs_confidence: float = 0.75,
    mcs_B: int = 10000,
    mcs_statistic: str = "max",
    mcs_seed: int = 123,
    mcs_block_size: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    Dict[int, Dict[str, Dict[str, Any]]],
    Dict[int, Dict[str, object]],
]:
    """
    Executa o pipeline completo para um dado split:
      1) Carrega dados e roda RV-FTS  → `data`, `info`
      2) Calcula métricas             → `tables`
      3) Salva planilhas e gráficos
      4) Roda Model Confidence Set    → `mcs`
      5) Gera relatório Markdown de parâmetros

    Retorna (data, tables, info, mcs).
    """
    if nclus_grid is None:
        nclus_grid = list(RVFTS_NCLUS)
    if rho_grid is None:
        rho_grid = list(RVFTS_RHO)

    out_root = _output_root(split)

    print("=" * 70)
    print(f"SPLIT: {split}")
    print("PASSO 1/5 — Carregando dados e executando RV-FTS")
    print("=" * 70)
    data, info = build_all_data(
        symbols=symbols, horizons=horizons,
        split=split,
        ew=ew, cs=cs,
        nclus_grid=nclus_grid, rho_grid=rho_grid,
        window=window, fcm_seed=fcm_seed,
        verbose=verbose,
    )

    print("\n" + "=" * 70)
    print("PASSO 2/5 — Calculando métricas")
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
    print("PASSO 3/5 — Salvando planilhas, previsões e gráficos")
    print("=" * 70)
    save_overview_spreadsheets(tables, horizons=horizons, split=split)
    save_metric_spreadsheets(tables, horizons=horizons, split=split)
    save_prediction_spreadsheets(data, horizons=horizons, symbols=symbols, split=split)
    save_all_plots(data, tables, horizons=horizons, symbols=symbols, split=split)

    mcs_results: Dict[int, Dict[str, object]] = {}
    if skip_mcs:
        print("\n[INFO] Etapa do MCS pulada (--skip-mcs).")
    else:
        print("\n" + "=" * 70)
        print("PASSO 4/5 — Model Confidence Set")
        print("=" * 70)
        mcs_results = run_mcs_all_horizons(
            horizons=horizons,
            split=split,
            tickers=None,            # auto-discover por moeda salva no passo 3
            model_subset=MCS_DEFAULT_MODELS,
            confidence=mcs_confidence,
            B=mcs_B, statistic=mcs_statistic, seed=mcs_seed,
            block_size=mcs_block_size,
            verbose=verbose,
        )

    print("\n" + "=" * 70)
    print("PASSO 5/5 — Relatório Markdown de parâmetros")
    print("=" * 70)
    generate_parameters_report(
        info,
        split=split,
        horizons=horizons, symbols=symbols,
        ew=ew, cs=cs,
        nclus_grid=nclus_grid, rho_grid=rho_grid,
        window=window, fcm_seed=fcm_seed,
    )

    print(f"\nConcluído. Resultados em: {out_root.resolve()}")
    return data, tables, info, mcs_results


# =============================================================================
# 12.  CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pipeline consolidado RV-FTS: carrega previsões Fuzzy HAR + HAR, "
            "roda RV-FTS, calcula métricas, gera planilhas/plots, executa MCS "
            "e produz relatório Markdown de parâmetros."
        )
    )
    p.add_argument("--splits", nargs="+", default=[DEFAULT_SPLIT],
                   choices=VALID_SPLITS,
                   help=(
                       "Splits a processar. Cada split corresponde a uma razão "
                       "treino/teste diferente. Ex.: --splits 40_60 60_40 70_30. "
                       f"(default: {DEFAULT_SPLIT})"
                   ))
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS),
                   help="Símbolos a processar.")
    p.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS),
                   help="Horizontes a processar (ex.: --horizons 1 7 30).")
    p.add_argument("--ew", type=int, default=RVFTS_EW,
                   help="Janela inicial de estimação (default: 756).")
    p.add_argument("--cs", type=int, default=RVFTS_CS,
                   help="Janela de calibração / CV (default: 252).")
    p.add_argument("--nclus", nargs="+", type=int, default=list(RVFTS_NCLUS),
                   help="Grid de nº de clusters c.")
    p.add_argument("--rho", nargs="+", type=float, default=list(RVFTS_RHO),
                   help="Grid de rho (memoria EWMA).")
    p.add_argument("--fcm-seed", type=int, default=RVFTS_FCM_SEED,
                   help="Semente do FCM.")

    # MCS
    p.add_argument("--skip-mcs", action="store_true",
                   help="Pula a etapa do MCS.")
    p.add_argument("--only-mcs", action="store_true",
                   help="Roda apenas o MCS, partindo de previsões já em disco.")
    p.add_argument("--mcs-confidence", type=float, default=0.75,
                   help="Nível de confiança do MCS (default: 0.75).")
    p.add_argument("--mcs-B", type=int, default=10000,
                   help="Nº de réplicas bootstrap do MCS.")
    p.add_argument("--mcs-statistic", default="max",
                   help="Estatística do MCS: max | R | tmax | tr | trange.")
    p.add_argument("--mcs-seed", type=int, default=123,
                   help="Semente do bootstrap do MCS.")
    p.add_argument("--mcs-block-size", type=int, default=None,
                   help=(
                       "Tamanho do bloco do bootstrap estacionario do MCS. "
                       "Omitir (default) replica o metodo do R: "
                       "max(ar_order) por serie de perdas, com piso 3."
                   ))

    p.add_argument("--quiet", action="store_true", help="Reduz a verbosidade.")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    verbose = not args.quiet

    if args.only_mcs:
        # Não recarrega/recalcula nada — só roda o MCS sobre o que já está salvo.
        print("=" * 70)
        print("Modo --only-mcs: rodando apenas o MCS sobre previsões em disco.")
        print("=" * 70)
        for split in args.splits:
            print(f"\n>>> Split: {split}")
            run_mcs_all_horizons(
                horizons=args.horizons,
                split=split,
                tickers=None,
                model_subset=MCS_DEFAULT_MODELS,
                confidence=args.mcs_confidence,
                B=args.mcs_B,
                statistic=args.mcs_statistic,
                seed=args.mcs_seed,
                block_size=args.mcs_block_size,
                verbose=verbose,
            )
        return

    for split in args.splits:
        print(f"\n{'#' * 70}")
        print(f"# PROCESSANDO SPLIT: {split}")
        print(f"{'#' * 70}\n")
        run_full_pipeline(
            symbols=args.symbols,
            horizons=args.horizons,
            split=split,
            ew=args.ew, cs=args.cs,
            nclus_grid=args.nclus, rho_grid=args.rho,
            fcm_seed=args.fcm_seed,
            skip_mcs=args.skip_mcs,
            mcs_confidence=args.mcs_confidence,
            mcs_B=args.mcs_B,
            mcs_statistic=args.mcs_statistic,
            mcs_seed=args.mcs_seed,
            mcs_block_size=args.mcs_block_size,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
