"""
rvfts_model.py — RV-FTS (Realized Volatility Fuzzy Time Series)
================================================================

Implementação em Python do modelo proposto em:
  Iqbal, S. & Lyócsa, Š. (2026). "A Fuzzy Framework for Realized Volatility
  Prediction: Empirical Evidence From Equity Markets". Journal of
  Forecasting, 45, 1261–1291. doi:10.1002/for.70082

Adaptado para integração com o pipeline FuzzyHAR de volatilidade
realizada de criptomoedas (notebook FuzzyHAR.ipynb).

Convenção de alinhamento (idêntica ao Fuzzy HAR do notebook)
------------------------------------------------------------
- ``target_series[t]`` é o valor a prever na linha ``t`` (RV de h dias
  forward, conforme eq. 1 do paper).
- ``target_series[s]`` é totalmente observável ao final do dia
  ``s + h - 1``.
- ``predictions[t]`` é a previsão *para* ``target_series[t]``, feita ao
  final do dia ``t - 1`` usando apenas dados observáveis até então.
- Logo, ``df.loc[d, "RV-FTS"] = predictions[d]`` casa diretamente com
  ``df.loc[d, "Target"]`` (o valor real na mesma linha) — pareando a
  convenção que o pipeline FuzzyHAR já utiliza para a coluna ``"Fuzzy HAR"``.

Algoritmo (Algorithm 1 do paper)
--------------------------------
1. Janela rolante de tamanho ``ew`` sobre a série de RV alvo.
2. FCM (fuzzy c-means) particiona a janela em ``c`` clusters.
3. Variância fuzzificada (eq. 7):
       ṽ_{t-1} = Σ γ_j(target[t-h]) · g_j
4. Previsão via EWMA (eq. 8):
       predictions[t] = (1-ρ) · ṽ_{t-1} + ρ · predictions[t-1]
5. Hiperparâmetros (c, ρ) selecionados em cada t via cross-validation
   rolante com pesos exponencialmente decrescentes (Section 3.6).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# 1.  FUZZY C-MEANS  (1-D, Bezdek; equivalente a e1071::cmeans com m=2)
# ===========================================================================

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

    Parameters
    ----------
    x        : array (n,)   — pontos de dados.
    c        : int           — número de clusters.
    m        : float         — expoente de fuzzificação (paper usa 2).
    max_iter : int           — máximo de iterações.
    tol      : float         — tolerância de convergência sobre U.
    seed     : int           — semente aleatória para a inicialização.

    Returns
    -------
    centroids : array (c,)    — centróides g_j.
    U         : array (n, c)  — matriz de pertinência γ_{ij}.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n <= c:
        # Caso degenerado — atribui cada ponto a seu próprio cluster
        centroids = x[:c].copy()
        U = np.eye(n, c)
        return centroids, U

    # Inicialização aleatória das pertinências (cada linha soma 1)
    U = rng.rand(n, c).astype(np.float64)
    U /= U.sum(axis=1, keepdims=True)

    exp = 2.0 / (m - 1.0)

    for _ in range(max_iter):
        # Atualiza centróides: g_j = Σ U_ij^m · x_i / Σ U_ij^m
        Um = U ** m
        denom = np.maximum(Um.sum(axis=0), 1e-300)
        centroids = (Um.T @ x) / denom            # shape (c,)

        # Distâncias |x_i − g_j|
        dist = np.abs(x[:, None] - centroids[None, :])   # (n, c)
        dist = np.maximum(dist, 1e-15)

        # Atualiza pertinências:  U_ij = 1 / Σ_k (d_ij/d_ik)^(2/(m-1))
        ratio = (dist[:, :, None] / dist[:, None, :]) ** exp   # (n, c, c)
        U_new = 1.0 / ratio.sum(axis=2)                        # (n, c)

        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new

    return centroids, U


# ===========================================================================
# 2.  MODELO RV-FTS  (com cross-validation rolante)
# ===========================================================================

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

    Convenção
    ---------
    - ``target_series[t]`` é a quantidade a prever na linha ``t``
      (tipicamente RV forward de h dias).
    - ``target_series[s]`` é observável ao final do dia ``s + h - 1``.
    - ``predictions[t]`` é a previsão *para* ``target_series[t]``,
      feita ao final do dia ``t - 1`` (sem look-ahead).

    Para ``h = 1``:
        FCM usa target[start : t]            (último ponto = target[t-1])
        ṽ_{t-1} = membership(target[t-1]) · centroids
        predictions[t] = (1-ρ)·ṽ_{t-1} + ρ·predictions[t-1]

    Para ``h > 1``:
        FCM usa target[start : t-h+1]        (último ponto = target[t-h])
        ṽ_{t-1} = membership(target[t-h]) · centroids
        predictions[t] = (1-ρ)·ṽ_{t-1} + ρ·predictions[t-1]

    Os hiperparâmetros (c, ρ) são selecionados a cada t por cross-validation
    rolante de tamanho ``cs`` com pesos exponenciais decrescentes
    (peso = ``cv_decay``^k), usando apenas erros de previsão sobre alvos
    fully observáveis até o final do dia ``t-1`` (sem look-ahead).

    Parameters
    ----------
    target_series : array (T,)  — série alvo (forward h-day RV).
    h             : int          — horizonte (≥ 1).
    ew            : int          — janela inicial de estimação (paper: 756).
    cs            : int          — janela de calibração / CV (paper: 252).
    nclus_grid    : list[int]    — grid de c (paper full: [2,3,4,6,9]).
    rho_grid      : list[float]  — grid de ρ (paper full: [0.025…0.975]).
    window        : "rolling" | "expanding".
    m             : float        — expoente de fuzzificação (paper: 2).
    cv_decay      : float        — decaimento exponencial no CV (R: 0.99).
    fcm_seed      : int          — semente da inicialização do FCM.
    verbose       : bool         — exibir progresso.

    Returns
    -------
    dict com chaves
        'predictions' : array (T,)  — previsão alinhada à linha t (NaN onde
                                       a CV ainda não pôde rodar).
        'fuzzified'   : array (T,)  — ṽ_{t-1} sob os params ótimos.
        'opt_c'       : array (T,)  — c selecionado em cada t.
        'opt_rho'     : array (T,)  — ρ selecionado em cada t.
        'pred_all'    : array (NC,NL,T) — todas as trajetórias EWMA
                                          candidatas (para diagnóstico).
        'fuzz_all'    : array (NC,T)    — ṽ candidatos por nº de clusters.
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

    # --------------------------------------------------------------
    # Phase 1 — FCM por janela rolante para cada nº de clusters
    # --------------------------------------------------------------
    # fuzz_all[nc_idx, t] = ṽ_{t-1} sob 'nc' clusters; usado para preds[t].
    fuzz_all = np.full((NC, T), np.nan)

    t0 = time.time()
    if verbose:
        print(f"[RV-FTS] Phase 1/3: FCM × {NC} valores de c × ~{T-ew-h+1} janelas (h={h})")

    # Primeiro t com janela FCM válida
    t_start = ew + h - 1

    for nc_idx, nc in enumerate(nclus_grid):
        if verbose:
            print(f"  c = {nc} ...", end=" ", flush=True)
        count = 0
        for t in range(t_start, T):
            end_excl = t - h + 1                 # exclusive end (Python slice)
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
                # O último ponto da janela é target[end_excl-1] = target[t-h],
                # que é o alvo mais recente fully observable em t-1.
                membership_last = U[-1]
                fuzz_all[nc_idx, t] = float(np.dot(membership_last, centroids))
                count += 1
            except Exception:
                continue

        if verbose:
            print(f"({count} janelas OK)")

    # --------------------------------------------------------------
    # Phase 2 — Recursão EWMA para cada (c, ρ) candidato
    # --------------------------------------------------------------
    # pred_all[nc_idx, nl_idx, t] = previsão para y[t] sob esses params.
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
            preds[first] = fuzz_nc[first]    # init: 1ª previsão = ṽ inicial

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

    # --------------------------------------------------------------
    # Phase 3 — Cross-validation rolante: seleciona (c, ρ) ótimos em cada t
    # --------------------------------------------------------------
    # No instante "fim do dia t-1" (quando preds[t] é gerado), os erros
    # observáveis são (y[s] - preds[s])² para s tal que y[s] é observável,
    # i.e. s + h - 1 ≤ t - 1  ⟺  s ≤ t - h.
    # Janela CV de tamanho cs:  s ∈ [t - h - cs + 1, t - h + 1)
    # Para h=1 isto equivale a [t-cs, t).
    final_preds = np.full(T, np.nan)
    final_fuzz = np.full(T, np.nan)
    opt_c = np.full(T, np.nan)
    opt_rho = np.full(T, np.nan)

    if verbose:
        print(f"[RV-FTS] Phase 3/3: CV rolante (CS={cs}, decay={cv_decay}, h={h})")

    # Inicia o CV assim que o EWMA tem pelo menos 1 passo válido (t >= t_start).
    # O guarda `nv < 10` abaixo garante que CVs degenerados são ignorados.
    # No paper (série única, sem split), first_cv = ew+cs+h-1 garantia nv=cs
    # para toda previsão. No nosso setup com split explícito, df_train pode ter
    # menos que ew+cs obs; iniciamos em t_start para que o teste inteiro seja
    # coberto (como o HAR faz com sua janela rolante de 730 obs).
    first_cv = t_start
    for t in range(first_cv, T):
        cv_end = t - h + 1                       # exclusive
        cv_start = max(0, t - h - cs + 1)        # usa janela menor no início

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
                errors = (cv_actual[valid] - cv_preds[valid]) ** 2
                wloss = float(np.average(errors, weights=weights))

                if wloss < best_loss:
                    best_loss = wloss
                    best_nc = nc_idx
                    best_nl = nl_idx
                    found = True

        if not found:
            continue

        # A previsão cross-validada para a linha t
        final_preds[t] = pred_all[best_nc, best_nl, t]
        final_fuzz[t] = fuzz_all[best_nc, t]
        opt_c[t] = nclus_grid[best_nc]
        opt_rho[t] = rho_grid[best_nl]

    elapsed = time.time() - t0
    n_valid = int(np.sum(~np.isnan(final_preds)))
    if verbose:
        print(f"[RV-FTS] Concluído em {elapsed:.1f}s — {n_valid} previsões válidas.")

    return {
        "predictions": final_preds,
        "fuzzified": final_fuzz,
        "opt_c": opt_c,
        "opt_rho": opt_rho,
        "pred_all": pred_all,
        "fuzz_all": fuzz_all,
    }


# ===========================================================================
# 3.  INTEGRAÇÃO COM O NOTEBOOK FuzzyHAR
# ===========================================================================

def add_rvfts_to_dataframes(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
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
    Roda o RV-FTS sobre os dataframes do pipeline FuzzyHAR e adiciona a
    coluna ``"RV-FTS"`` em ambos.

    Alinhamento (compatível com a coluna ``"Fuzzy HAR"`` do notebook)
    -----------------------------------------------------------------
    Em cada linha ``d`` o valor da coluna ``"RV-FTS"`` é a previsão
    feita ao final do dia ``d-1`` para o valor de ``target_column[d]``.
    Isso permite a comparação direta linha a linha em
    ``MSE(Target[d], RV-FTS[d])``.

    Coluna usada como entrada
    -------------------------
    Por padrão usa ``target_column`` (= ``"Target"``), que no pipeline
    do notebook já é o RV forward de ``h`` dias pré-computado nos
    inputs ``Inputs/RVs/RV_t+h/...``. Se ``target_column`` não estiver
    disponível e ``horizon == 1``, faz fallback para ``rv_column``
    (``"RV"`` — válido apenas para h=1, pois daí ``Target == RV``).

    A função **nunca** constrói uma média rolante backward internamente
    — isso evita o bug do alvo "errado" para h > 1 da versão anterior.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    n_train = len(df_train)
    n_test = len(df_test)

    # ---------- Escolha da série de entrada ----------
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
        h_internal = horizon
        used_column = target_column
    elif horizon == 1 and use_rv:
        # Para h=1 a coluna RV equivale ao target diário.
        series_full = np.concatenate([
            df_train[rv_column].values.astype(float),
            df_test[rv_column].values.astype(float),
        ])
        h_internal = 1
        used_column = rv_column
    else:
        raise ValueError(
            f"Necessária a coluna '{target_column}' (RV forward de h-dias) ou "
            f"'{rv_column}' (apenas h=1). Disponíveis: "
            f"{list(df_train.columns)[:12]}"
        )

    if verbose:
        print(f"\n[RV-FTS] horizon=t+{horizon} | input='{used_column}' "
              f"| h_interno={h_internal}")
        print(f"  Train: {n_train} obs | Test: {n_test} obs "
              f"| Total: {len(series_full)} obs")
        print(f"  EW={ew} | CS={cs} | window={window}")

    # ---------- Forecasting ----------
    result = rvfts_forecast(
        target_series=series_full,
        h=h_internal,
        ew=ew,
        cs=cs,
        nclus_grid=nclus_grid,
        rho_grid=rho_grid,
        window=window,
        fcm_seed=fcm_seed,
        verbose=verbose,
    )

    preds = result["predictions"]   # preds[t] já é a previsão para target[t]

    # Já alinhado por construção — basta atribuir nas colunas:
    df_train["RV-FTS"] = preds[:n_train]
    df_test["RV-FTS"] = preds[n_train:n_train + n_test]

    info = {
        "used_column": used_column,
        "h_internal": h_internal,
        "opt_c": result["opt_c"],
        "opt_rho": result["opt_rho"],
        "fuzzified": result["fuzzified"],
        "n_valid_train": int(np.sum(~np.isnan(preds[:n_train]))),
        "n_valid_test": int(np.sum(~np.isnan(preds[n_train:n_train + n_test]))),
    }

    if verbose:
        print(f"  → Previsões válidas: train={info['n_valid_train']}, "
              f"test={info['n_valid_test']}")

    return df_train, df_test, info


def run_rvfts_pipeline(
    all_results: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    symbols: List[str],
    horizons: List[int] = [1, 7, 30],
    target_column: str = "Target",
    rv_column: str = "RV",
    ew: int = 756,
    cs: int = 252,
    nclus_grid: Optional[List[int]] = None,
    rho_grid: Optional[List[float]] = None,
    window: str = "rolling",
    fcm_seed: int = 42,
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Executa o RV-FTS para todos os símbolos e horizontes do pipeline,
    adicionando a coluna ``"RV-FTS"`` aos DataFrames em ``all_results``.

    Estrutura esperada
    ------------------
    ``all_results[horizon][symbol] = {"train": df_tr, "test": df_te}``
    Cada DataFrame deve ter a coluna ``target_column`` (recomendado).
    """
    if nclus_grid is None:
        nclus_grid = [2, 3, 4, 6, 9]
    if rho_grid is None:
        rho_grid = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]

    rvfts_info: Dict[int, Dict[str, Any]] = {}
    total = len(horizons) * len(symbols)
    count = 0

    for horizon in horizons:
        rvfts_info[horizon] = {}
        for sym in symbols:
            count += 1
            tag = f"[{count}/{total}] {sym} t+{horizon}"
            if verbose:
                print(f"\n>>> {tag}")

            if horizon not in all_results or sym not in all_results[horizon]:
                print(f"  [SKIP] Dados não encontrados para {tag}")
                continue

            df_tr = all_results[horizon][sym]["train"]
            df_te = all_results[horizon][sym]["test"]

            try:
                df_tr, df_te, info = add_rvfts_to_dataframes(
                    df_train=df_tr,
                    df_test=df_te,
                    horizon=horizon,
                    target_column=target_column,
                    rv_column=rv_column,
                    ew=ew,
                    cs=cs,
                    nclus_grid=nclus_grid,
                    rho_grid=rho_grid,
                    window=window,
                    fcm_seed=fcm_seed,
                    verbose=verbose,
                )
                all_results[horizon][sym]["train"] = df_tr
                all_results[horizon][sym]["test"] = df_te
                rvfts_info[horizon][sym] = info

            except Exception as e:
                print(f"  [ERRO] {tag}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return rvfts_info


# ===========================================================================
# 4.  MÉTRICAS DE AVALIAÇÃO  (eq. 20 e 21 do paper)
# ===========================================================================

EPS = 1e-12


def mse_paper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE conforme eq. (20) do paper. Ignora NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return float("nan")
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def qlike_paper(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    """
    QLIKE conforme eq. (21) do paper:
        L = mean( y/f - log(y/f) - 1 )
    Ignora NaN; aplica clip mínimo `eps` para estabilidade numérica.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return float("nan")
    yt = np.clip(y_true[mask], eps, None)
    fp = np.clip(y_pred[mask], eps, None)
    r = yt / fp
    return float(np.mean(r - np.log(r) - 1.0))


def compute_rvfts_metrics(
    all_results: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
    symbols: List[str],
    horizons: List[int] = [1, 7, 30],
    target_col: str = "Target",
    pred_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calcula MSE, RMSE, MAE e QLIKE (paper eq. 21) para as previsões do
    RV-FTS e demais modelos especificados em ``pred_cols`` no conjunto
    out-of-sample (df_test). Retorna DataFrame longo para fácil análise.
    """
    if pred_cols is None:
        pred_cols = ["Fuzzy HAR", "RV-FTS"]

    rows = []
    for horizon in horizons:
        for sym in symbols:
            if horizon not in all_results or sym not in all_results[horizon]:
                continue

            df_te = all_results[horizon][sym].get("test")
            if df_te is None or target_col not in df_te.columns:
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
                qlike = qlike_paper(yt, yp)

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


# ===========================================================================
# 5.  TESTE DE SANIDADE  (executável via `python rvfts_model.py`)
# ===========================================================================

def _sanity_check() -> None:
    """
    Teste rápido: gera uma série persistente, roda o RV-FTS, e confirma
    que (a) o alinhamento está correto e (b) a previsão é razoável
    comparada à persistência ingênua para h=1.
    """
    print("=" * 70)
    print("RV-FTS — Teste de sanidade")
    print("=" * 70)

    np.random.seed(0)
    T = 1500
    rv = np.zeros(T)
    rv[0] = 1.0
    for t in range(1, T):
        rv[t] = 0.92 * rv[t - 1] + 0.08 * np.random.gamma(2, 1.0) \
                + 0.05 * np.random.randn()
    rv = np.maximum(rv, 0.01)

    # h = 1
    out = rvfts_forecast(
        target_series=rv,
        h=1,
        ew=400,
        cs=200,
        nclus_grid=[2, 3, 4],
        rho_grid=[0.05, 0.50, 0.90],
        verbose=False,
    )
    preds = out["predictions"]

    # Como o alinhamento agora é direto: preds[t] prevê rv[t]
    mask = ~np.isnan(preds)
    mse_rvfts = float(np.mean((preds[mask] - rv[mask]) ** 2))
    qlike_rvfts = qlike_paper(rv[mask], preds[mask])

    # Persistência ingênua: prediz rv[t] = rv[t-1]
    naive = np.full_like(rv, np.nan)
    naive[1:] = rv[:-1]
    mask_n = ~np.isnan(naive) & ~np.isnan(rv) & mask
    mse_naive = float(np.mean((naive[mask_n] - rv[mask_n]) ** 2))

    print(f"  T={T}, h=1, EW=400, CS=200")
    print(f"  Previsões válidas: {int(mask.sum())} / {T}")
    print(f"  MSE  RV-FTS:       {mse_rvfts:.6f}")
    print(f"  MSE  persistência: {mse_naive:.6f}")
    print(f"  QLIKE RV-FTS:      {qlike_rvfts:.6f}")
    if mse_rvfts < mse_naive * 1.5:
        print(f"  → MSE razoável (< 1.5x persistência). OK.")
    else:
        print(f"  ⚠ MSE muito acima da persistência — investigar.")

    # Verificação chave: corr(preds[t], rv[t]) >> corr(preds[t], rv[t+1])
    # Confirma que preds[t] está alinhado com rv[t] (a forma certa).
    m_now = mask & ~np.isnan(rv)
    corr_now = np.corrcoef(preds[m_now], rv[m_now])[0, 1]
    m_next = mask[:-1] & ~np.isnan(rv[1:])
    corr_next = np.corrcoef(preds[:-1][m_next], rv[1:][m_next])[0, 1]
    print(f"  corr(RV-FTS[t], rv[t])     = {corr_now:.4f}  (esperado: ALTO)")
    print(f"  corr(RV-FTS[t], rv[t+1])   = {corr_next:.4f}  (esperado: menor)")
    if corr_now > corr_next:
        print(f"  → Alinhamento OK: previsão casa com o alvo da mesma linha.")
    else:
        print(f"  ⚠ Alinhamento suspeito.")

    print()
    # h = 7  — testa que o forecaster lida com h>1 sem quebrar
    print("  Teste h=7 (multi-step) ...")
    rv7 = pd.Series(rv).rolling(7, min_periods=7).mean().shift(-6).values
    rv7 = rv7[~np.isnan(rv7)]
    out7 = rvfts_forecast(
        target_series=rv7,
        h=7,
        ew=400,
        cs=200,
        nclus_grid=[2, 3],
        rho_grid=[0.10, 0.50, 0.90],
        verbose=False,
    )
    preds7 = out7["predictions"]
    n_valid7 = int(np.sum(~np.isnan(preds7)))
    print(f"  → Previsões válidas h=7: {n_valid7} / {len(rv7)}")
    if n_valid7 > 100:
        print(f"  → h>1 funciona. OK.")
    else:
        print(f"  ⚠ Poucas previsões válidas para h=7.")

    print("\n" + "=" * 70)
    print("Teste concluído.")
    print("=" * 70)


if __name__ == "__main__":
    _sanity_check()
