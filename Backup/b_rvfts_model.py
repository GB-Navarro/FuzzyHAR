"""
rvfts_model.py — RV-FTS (Realized Volatility Fuzzy Time Series)
================================================================

Implementação em Python do modelo proposto em:
  Iqbal & Lyócsa (2026). "A Fuzzy Framework for Realized Volatility
  Prediction: Empirical Evidence From Equity Markets".
  Journal of Forecasting, 45, 1261-1291.

Adaptado para integração com o pipeline FuzzyHAR de volatilidade
realizada de criptomoedas (notebook FuzzyHAR.ipynb).

Algoritmo resumido:
  1. Janela rolante de tamanho EW sobre a série de RV
  2. FCM (Fuzzy C-Means) particiona a janela em c clusters
  3. Variância fuzzificada: ṽ_t = Σ γ_j · g_j  (memberships × centróides)
  4. Previsão via EWMA: RV̂_{t+1} = (1-ρ)·ṽ_t + ρ·RV̂_t
  5. Hiperparâmetros (c, ρ) selecionados por cross-validation rolante
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1.  FUZZY C-MEANS  (implementação própria para 1-D, sem dependência externa)
# ---------------------------------------------------------------------------

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

    Parâmetros
    ----------
    x        : array (n,)  — pontos de dados
    c        : int          — número de clusters
    m        : float        — expoente de fuzzificação (padrão 2)
    max_iter : int          — máximo de iterações
    tol      : float        — tolerância de convergência sobre U
    seed     : int          — semente aleatória

    Retorna
    -------
    centroids : array (c,)   — centróides dos clusters
    U         : array (n, c) — matriz de pertinência
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n <= c:
        # Caso degenerado: atribuir cada ponto a seu próprio cluster
        centroids = x[:c].copy()
        U = np.eye(n, c)
        return centroids, U

    # Inicializa pertinências aleatórias (cada linha soma 1)
    U = rng.rand(n, c).astype(np.float64)
    U /= U.sum(axis=1, keepdims=True)

    exp = 2.0 / (m - 1.0)

    for _ in range(max_iter):
        Um = U ** m
        denom = Um.sum(axis=0)
        denom = np.maximum(denom, 1e-300)
        centroids = (Um.T @ x) / denom          # (c,)

        # Distâncias |x_i - g_j|
        dist = np.abs(x[:, None] - centroids[None, :])   # (n, c)
        dist = np.maximum(dist, 1e-15)

        # Atualiza pertinências
        # U_ij = 1 / Σ_k (d_ij / d_ik)^(2/(m-1))
        ratio = (dist[:, :, None] / dist[:, None, :]) ** exp   # (n, c, c)
        U_new = 1.0 / ratio.sum(axis=2)                        # (n, c)

        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new

    return centroids, U


# ---------------------------------------------------------------------------
# 2.  MODELO RV-FTS  (com cross-validation rolante)
# ---------------------------------------------------------------------------

def rvfts_forecast(
    rv_series: np.ndarray,
    h: int = 1,
    ew: int = 756,
    cs: int = 252,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    window: str = "rolling",
    m: float = 2.0,
    cv_decay: float = 0.99,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Executa o modelo RV-FTS completo com cross-validation.

    Parâmetros
    ----------
    rv_series   : array (T,)  — série de volatilidade realizada
    h           : int          — horizonte de previsão (1, 7 ou 30)
    ew          : int          — tamanho da janela de estimação
    cs          : int          — tamanho da janela de cross-validation
    nclus_grid  : list[int]    — grid de número de clusters
    rho_grid    : list[float]  — grid de parâmetros de suavização EWMA
    window      : str          — 'rolling' ou 'expanding'
    m           : float        — expoente de fuzzificação do FCM
    cv_decay    : float        — fator de decaimento exponencial no CV
    verbose     : bool         — exibir progresso

    Retorna
    -------
    dict com chaves:
        'predictions'  — previsões finais (cross-validated), shape (T,)
        'fuzzified'    — variância fuzzificada, shape (T,)
        'opt_c'        — nº de clusters ótimo em cada t
        'opt_rho'      — ρ ótimo em cada t
    """
    if nclus_grid is None:
        nclus_grid = [2, 3, 4, 6, 9]
    if rho_grid is None:
        rho_grid = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]

    rv = np.asarray(rv_series, dtype=np.float64)
    T = len(rv)
    NC = len(nclus_grid)
    NL = len(rho_grid)

    # ------------------------------------------------------------------
    # Etapa 1: Calcular valores fuzzificados para cada nº de clusters
    # ------------------------------------------------------------------
    # fuzz_all[nc_idx, t] = ṽ_t com nc clusters
    fuzz_all = np.full((NC, T), np.nan)

    t0 = time.time()
    if verbose:
        print(f"[RV-FTS] Etapa 1/3: FCM para {NC} valores de c × {T-ew} janelas ...")

    for nc_idx, nc in enumerate(nclus_grid):
        if verbose:
            print(f"  c = {nc} ...", end=" ", flush=True)
        count = 0
        for t in range(ew, T):
            # Correção de look-ahead: em t, o último target conhecido é rv[t-h]
            # Mas para h=1 e usando RV diário: rv[t] é conhecido no fim do dia t
            # Seguimos a convenção do paper: janela vai até t (inclusivo)
            # e a previsão feita em t é para t+1
            end_idx = t + 1  # Python slice: rv[start:end_idx]

            if window == "rolling":
                start_idx = max(0, end_idx - ew)
            else:
                start_idx = 0

            win = rv[start_idx:end_idx]
            win = win[~np.isnan(win)]

            if len(win) < max(nc + 1, 5):
                continue

            try:
                centroids, U = fcm_1d(win, nc, m=m, seed=42 + t % 100)
                # Pertinência do último ponto da janela (= ponto mais recente)
                membership_last = U[-1]
                fuzz_all[nc_idx, t] = np.dot(membership_last, centroids)
                count += 1
            except Exception:
                continue

        if verbose:
            print(f"({count} janelas OK)")

    # ------------------------------------------------------------------
    # Etapa 2: Previsões EWMA para cada combinação (c, ρ)
    # ------------------------------------------------------------------
    # pred_all[nc_idx, nl_idx, t] = previsão em t (para t+1)
    pred_all = np.full((NC, NL, T), np.nan)

    if verbose:
        print(f"[RV-FTS] Etapa 2/3: EWMA para {NC}×{NL} combinações ...")

    for nc_idx in range(NC):
        fuzz_nc = fuzz_all[nc_idx]
        for nl_idx, rho in enumerate(rho_grid):
            preds = np.full(T, np.nan)

            # Primeiro valor válido: inicializa previsão como fuzzificado
            valid_idx = np.where(~np.isnan(fuzz_nc))[0]
            if len(valid_idx) == 0:
                continue
            first = valid_idx[0]
            preds[first] = fuzz_nc[first]

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

    # ------------------------------------------------------------------
    # Etapa 3: Cross-validation rolante para selecionar (c, ρ) ótimos
    # ------------------------------------------------------------------
    final_preds = np.full(T, np.nan)
    final_fuzz = np.full(T, np.nan)
    opt_c = np.full(T, np.nan)
    opt_rho = np.full(T, np.nan)

    if verbose:
        print(f"[RV-FTS] Etapa 3/3: Cross-validation (janela CV = {cs}) ...")

    first_cv = ew + cs
    for t in range(first_cv, T):
        best_loss = np.inf
        best_nc = 0
        best_nl = 0

        for nc_idx in range(NC):
            for nl_idx in range(NL):
                cv_preds = pred_all[nc_idx, nl_idx, t - cs : t]
                cv_actual = rv[t - cs : t]

                valid = ~(np.isnan(cv_actual) | np.isnan(cv_preds))
                nv = valid.sum()
                if nv < 10:
                    continue

                weights = cv_decay ** np.arange(nv - 1, -1, -1)
                errors = (cv_actual[valid] - cv_preds[valid]) ** 2
                wloss = np.average(errors, weights=weights)

                if wloss < best_loss:
                    best_loss = wloss
                    best_nc = nc_idx
                    best_nl = nl_idx

        final_preds[t] = pred_all[best_nc, best_nl, t]
        final_fuzz[t] = fuzz_all[best_nc, t]
        opt_c[t] = nclus_grid[best_nc]
        opt_rho[t] = rho_grid[best_nl]

    elapsed = time.time() - t0
    n_valid = np.sum(~np.isnan(final_preds))
    if verbose:
        print(f"[RV-FTS] Concluído em {elapsed:.1f}s — {n_valid} previsões válidas.")

    return {
        "predictions": final_preds,
        "fuzzified": final_fuzz,
        "opt_c": opt_c,
        "opt_rho": opt_rho,
    }


# ---------------------------------------------------------------------------
# 3.  INTEGRAÇÃO COM O NOTEBOOK FuzzyHAR
# ---------------------------------------------------------------------------

def add_rvfts_to_dataframes(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    horizon: int = 1,
    rv_column: str = "RV",
    ew: int = 756,
    cs: int = 252,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    window: str = "rolling",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executa RV-FTS e adiciona coluna 'RV-FTS' aos DataFrames de treino/teste.

    Segue a mesma convenção do ALSM ('Fuzzy HAR'):
      — pred[t] é a previsão feita no momento t para o valor no momento t+1.

    Parâmetros
    ----------
    df_train     : DataFrame de treino (com coluna rv_column)
    df_test      : DataFrame de teste
    horizon      : horizonte de previsão (1, 7 ou 30)
    rv_column    : nome da coluna de RV diário
    ew, cs       : janelas de estimação e CV
    nclus_grid   : grid de clusters
    rho_grid     : grid de ρ
    window       : 'rolling' ou 'expanding'
    verbose      : mostrar progresso

    Retorna
    -------
    (df_train, df_test, info)  com a coluna 'RV-FTS' adicionada.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Concatenar série completa (treino + teste)
    n_train = len(df_train)
    n_test = len(df_test)

    # Extrair a série de RV
    # Para h=1: usar RV diário
    # Para h>1: construir média rolante de h dias como proxy do h-day RV
    rv_train = df_train[rv_column].values.astype(float)
    rv_test = df_test[rv_column].values.astype(float)
    rv_full = np.concatenate([rv_train, rv_test])

    if horizon > 1:
        # Construir h-day rolling average RV (backward-looking)
        rv_series_h = pd.Series(rv_full).rolling(
            window=horizon, min_periods=horizon
        ).mean().values
    else:
        rv_series_h = rv_full.copy()

    T = len(rv_series_h)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[RV-FTS] Símbolo em processamento | horizon=t+{horizon}")
        print(f"  Treino: {n_train} obs | Teste: {n_test} obs | Total: {T}")
        print(f"  EW={ew} | CS={cs} | window={window}")
        print(f"  nclus_grid={nclus_grid}")
        print(f"  rho_grid={rho_grid}")
        print(f"{'='*60}")

    # Rodar o modelo
    result = rvfts_forecast(
        rv_series=rv_series_h,
        h=1,  # look-ahead já tratado pela construção da série
        ew=ew,
        cs=cs,
        nclus_grid=nclus_grid,
        rho_grid=rho_grid,
        window=window,
        verbose=verbose,
    )

    preds = result["predictions"]  # pred[t] prevê rv[t+1]

    # Separar previsões em treino e teste
    preds_train = preds[:n_train]
    preds_test = preds[n_train : n_train + n_test]

    df_train["RV-FTS"] = preds_train
    df_test["RV-FTS"] = preds_test

    info = {
        "opt_c": result["opt_c"],
        "opt_rho": result["opt_rho"],
        "fuzzified": result["fuzzified"],
        "n_valid_train": np.sum(~np.isnan(preds_train)),
        "n_valid_test": np.sum(~np.isnan(preds_test)),
    }

    if verbose:
        print(
            f"  → Previsões válidas: treino={info['n_valid_train']}, "
            f"teste={info['n_valid_test']}"
        )

    return df_train, df_test, info


def run_rvfts_pipeline(
    all_results: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    symbols: List[str],
    horizons: List[int] = [1, 7, 30],
    rv_column: str = "RV",
    ew: int = 756,
    cs: int = 252,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    window: str = "rolling",
    verbose: bool = True,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Executa RV-FTS para todos os símbolos e horizontes do pipeline,
    adicionando a coluna 'RV-FTS' aos DataFrames em all_results.

    Parâmetros
    ----------
    all_results : dict[horizon][symbol] = {"train": df_tr, "test": df_te}
                  (resultado do loop principal do notebook)
    symbols     : lista de símbolos (ex: ["BTCUSDT", "ETHUSDT", ...])
    horizons    : horizontes (ex: [1, 7, 30])
    ...demais parâmetros repassados a add_rvfts_to_dataframes

    Retorna
    -------
    rvfts_info : dict[horizon][symbol] = info dict
    """
    if nclus_grid is None:
        nclus_grid = [2, 3, 4, 6, 9]
    if rho_grid is None:
        rho_grid = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]

    rvfts_info: Dict[int, Dict[str, Any]] = {}
    total = len(horizons) * len(symbols)
    count = 0

    for horizon in horizons:
        rvfts_info[horizon] = {}
        for sym in symbols:
            count += 1
            if verbose:
                print(f"\n>>> [{count}/{total}] {sym} | t+{horizon}")

            if horizon not in all_results or sym not in all_results[horizon]:
                print(f"  [SKIP] Dados não encontrados para {sym} t+{horizon}")
                continue

            df_tr = all_results[horizon][sym]["train"]
            df_te = all_results[horizon][sym]["test"]

            # Verificar se a coluna RV existe
            if rv_column not in df_tr.columns:
                # Tentar alternativas
                alt_cols = ["Feature Origin", "RV", "rv"]
                found = False
                for alt in alt_cols:
                    if alt in df_tr.columns:
                        rv_column_used = alt
                        found = True
                        break
                if not found:
                    print(f"  [SKIP] Coluna de RV não encontrada. "
                          f"Disponíveis: {list(df_tr.columns[:10])}")
                    continue
            else:
                rv_column_used = rv_column

            try:
                df_tr, df_te, info = add_rvfts_to_dataframes(
                    df_train=df_tr,
                    df_test=df_te,
                    horizon=horizon,
                    rv_column=rv_column_used,
                    ew=ew,
                    cs=cs,
                    nclus_grid=nclus_grid,
                    rho_grid=rho_grid,
                    window=window,
                    verbose=verbose,
                )

                # Atualizar os DataFrames no all_results
                all_results[horizon][sym]["train"] = df_tr
                all_results[horizon][sym]["test"] = df_te
                rvfts_info[horizon][sym] = info

            except Exception as e:
                print(f"  [ERRO] {sym} t+{horizon}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return rvfts_info


# ---------------------------------------------------------------------------
# 4.  MÉTRICAS DE AVALIAÇÃO
# ---------------------------------------------------------------------------

def compute_rvfts_metrics(
    all_results: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    symbols: List[str],
    horizons: List[int] = [1, 7, 30],
    target_col: str = "target",
    pred_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calcula MSE e MAE para as previsões RV-FTS e (opcionalmente) outros modelos.

    Retorna um DataFrame com as métricas para comparação.
    """
    if pred_cols is None:
        pred_cols = ["Fuzzy HAR", "RV-FTS"]

    rows = []
    for horizon in horizons:
        for sym in symbols:
            if horizon not in all_results or sym not in all_results[horizon]:
                continue

            df_te = all_results[horizon][sym]["test"]
            if target_col not in df_te.columns:
                continue

            y_true = df_te[target_col].values.astype(float)

            for col in pred_cols:
                if col not in df_te.columns:
                    continue

                y_pred = df_te[col].values.astype(float)
                valid = ~(np.isnan(y_true) | np.isnan(y_pred))

                if valid.sum() < 10:
                    continue

                yt = y_true[valid]
                yp = y_pred[valid]

                mse = float(np.mean((yt - yp) ** 2))
                mae = float(np.mean(np.abs(yt - yp)))
                rmse = float(np.sqrt(mse))

                # QLIKE (com proteção numérica)
                eps = 1e-12
                yt_pos = np.maximum(yt, eps)
                yp_pos = np.maximum(yp, eps)
                qlike = float(np.mean(np.log(yp_pos) + yt_pos / yp_pos))

                rows.append({
                    "Symbol": sym.replace("USDT", ""),
                    "Horizon": f"t+{horizon}",
                    "Model": col,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "QLIKE": qlike,
                    "N_obs": int(valid.sum()),
                })

    return pd.DataFrame(rows)
