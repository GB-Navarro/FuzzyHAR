"""
elsm_model.py — e-LSM (Evolving Level-Set Model)
================================================

Implementação do modelo **e-LSM** ("controlled e-RLSM with adaptive growth")
para integração no pipeline FuzzyHAR de previsão de volatilidade realizada (RV)
de criptomoedas, ao lado de HAR / Fuzzy HAR / RV-FTS.

Origem do modelo
----------------
O motor (classes ``EvolvingRules`` / ``eRLSM`` e os helpers ``build_d_from_tau``,
``gaussian_tau_vec``, ``kmeans_init_rules``, ``dlsm_fit``) é portado **VERBATIM**
da célula 74 (versão completa) do notebook ``Cópia_de_Level_Set_Algorithms.ipynb``.
Cada fórmula/linha aritmética é byte-a-byte idêntica ao notebook; a ÚNICA mudança
é de *plumbing*: os hiperparâmetros que a célula 74 lê como **globais de módulo**
passam a ser lidos de um objeto de configuração ``ELSMConfig`` injetado nos
construtores. A equivalência numérica é garantida pelo teste
``test_elsm_identical.py`` (diferença máxima ≤ 1e-12 vs. a célula 74).

Convenção de features/alvo/alinhamento (idêntica ao Fuzzy HAR)
-------------------------------------------------------------
As features são exatamente as do Fuzzy HAR — ``make_har_features`` (daily,
weekly e monthly RV, ``week_len=7``, ``month_len=30``) e alvo ``Target.shift(-1)``.
A coluna de previsão é atribuída posicionalmente ao ``df_test`` da mesma forma que
a célula 34 do FuzzyHAR faz com ``df_test["Fuzzy HAR"] = OS_har`` — garantindo que
``e-LSM[d]`` se alinhe com ``Target[d]`` linha a linha.

Normalização
------------
RV é da ordem de 1e-5…1e-1, enquanto os hiperparâmetros do e-LSM são calibrados
para dados em ``[0,1]``. Por isso X e y são normalizados (min-max, **estatísticas
de treino**), o modelo roda em escala normalizada, e as previsões são
**desnormalizadas** de volta à escala de RV antes de salvar/comparar.

Como usar
---------
    from elsm_model import add_elsm_to_dataframes
    df_tr, df_te, info = add_elsm_to_dataframes(df_tr, df_te, horizon=1)

Teste de sanidade:
    python elsm_model.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ===========================================================================
# 0.  CONSTANTES E CONFIGURAÇÃO
# ===========================================================================

EPS: float = 1e-12   # constante de módulo (igual à célula 74 do notebook)


@dataclass(frozen=True)
class ELSMConfig:
    """
    Hiperparâmetros do e-LSM. Os valores default são EXATAMENTE os da célula 74
    do notebook (versão completa "controlled e-RLSM"), exceto ``clip_yhat`` que
    fica como knob (a integração de RV usa ``clip_yhat=False``).
    """

    # --- estrutura inicial ---
    n_init_rules: int = 10
    sigma_scale: float = 1.10

    # --- gates de criação de regra ---
    tau_create: float = 0.94
    tau_err_gate: float = 0.90
    e_create: float = 0.0022

    # --- limites de forma das regras ---
    sigma_min: float = 0.008
    sigma_max: float = 0.12

    # --- núcleo RLS (e-RLSM) ---
    alpha: float = 1e4
    lam: float = 1.0          # fator de esquecimento (LAMBDA na célula 74)

    # --- dinâmica de confiabilidade ---
    beta: float = 0.985
    eta: float = 0.30

    # --- controle de merge ---
    merge_dist: float = 0.11
    merge_every: int = 12

    # --- freios suaves (sem teto rígido de regras) ---
    create_cooldown: int = 45
    tau_create_hard: float = 0.33
    err_gate_hard: float = 0.55
    growth_gamma: float = 0.035

    # --- saída / numérico ---
    clip_yhat: bool = True
    seed: int = 0


# Grid default para o grid search no treino (lista de dicts de override).
# Tunável — mantido enxuto para controlar o runtime.
DEFAULT_ELSM_GRID: List[Dict[str, Any]] = [
    dict(n_init_rules=n, sigma_max=sm, tau_create=tc)
    for n, sm, tc in product([6, 10], [0.12, 0.25], [0.5, 0.94])
]


# ===========================================================================
# 1.  HELPERS (PORTADOS VERBATIM DA CÉLULA 74)
# ===========================================================================

def safe_sum(x, eps: float = 1e-12) -> float:
    s = float(np.sum(x))
    return s if abs(s) > eps else eps


def build_d_from_tau(tau, eps: float = 1e-12) -> np.ndarray:
    tau = np.asarray(tau, float).ravel()
    s = safe_sum(tau, eps)
    d = np.empty(2 * tau.size, float)
    d[0::2] = (tau ** 2) / s
    d[1::2] = tau / s
    return d


def gaussian_tau_vec(x, centers, sigmas) -> np.ndarray:
    """
    Ativação Gaussiana média (vetorizada):
      tau_i = exp(-0.5 * mean(((x-c_i)/s_i)^2))
    """
    x = np.asarray(x, float).ravel()
    C = np.asarray(centers, float)
    S = np.maximum(np.asarray(sigmas, float), EPS)
    z = (x[None, :] - C) / S
    dist = np.mean(z * z, axis=1)
    return np.exp(-0.5 * dist)


def kmeans_init_rules(Xtr, n_rules, sigma_scale, seed, cfg: ELSMConfig):
    """Idêntico à célula 74; os limites de clip vêm de ``cfg`` (eram globais)."""
    n_rules = int(min(n_rules, len(Xtr)))
    km = KMeans(n_clusters=n_rules, n_init=10, random_state=seed)
    km.fit(Xtr)
    centers = km.cluster_centers_
    labels = km.labels_

    global_std = Xtr.std(axis=0) + 1e-3
    sigmas = np.zeros_like(centers)
    for i in range(n_rules):
        Xi = Xtr[labels == i]
        sigmas[i] = (Xi.std(axis=0) + 1e-3) if Xi.shape[0] >= 5 else global_std
    sigmas = np.clip(sigmas * float(sigma_scale), cfg.sigma_min, cfg.sigma_max)
    return centers, sigmas


def dlsm_fit(X, y, centers, sigmas) -> np.ndarray:
    D = []
    for x in X:
        tau = gaussian_tau_vec(x, centers, sigmas)
        D.append(build_d_from_tau(tau))
    D = np.vstack(D)
    u, *_ = np.linalg.lstsq(D, y, rcond=1e-12)
    return u


# ===========================================================================
# 2.  BASE DE REGRAS EVOLUTIVA (PORTADA VERBATIM DA CÉLULA 74)
# ===========================================================================

class EvolvingRules:
    def __init__(self, cfg: ELSMConfig, centers=None, sigmas=None):
        self.cfg = cfg
        self.c, self.s, self.n, self.r = [], [], [], []
        if centers is not None:
            for i in range(centers.shape[0]):
                self.c.append(centers[i].copy())
                self.s.append(sigmas[i].copy())
                self.n.append(1)
                self.r.append(1.0)

    def num_rules(self) -> int:
        return len(self.c)

    def raw_tau(self, x) -> np.ndarray:
        if self.num_rules() == 0:
            return np.array([])
        C = np.vstack(self.c)
        S = np.vstack(self.s)
        return gaussian_tau_vec(x, C, S)

    def effective_tau(self, x) -> np.ndarray:
        if self.num_rules() == 0:
            return np.array([])
        return self.raw_tau(x) * np.asarray(self.r, float)

    def winner_raw(self, x):
        if self.num_rules() == 0:
            return None, 0.0
        tau = self.raw_tau(x)
        i = int(np.argmax(tau))
        return i, float(tau[i])

    def add_rule(self, x):
        p = x.size
        self.c.append(x.copy())
        self.s.append(np.full(p, self.cfg.sigma_max))  # start wide
        self.n.append(1)
        self.r.append(1.0)

    def delete_rule(self, j):
        for arr in (self.c, self.s, self.n, self.r):
            del arr[j]

    def update_rule(self, i, x):
        ni = self.n[i]
        ci = self.c[i]
        ci_new = ci + (x - ci) / (ni + 1)

        si = self.s[i]
        diff1 = x - ci
        diff2 = x - ci_new
        var = (si ** 2 * ni + diff1 * diff2) / (ni + 1)
        si_new = np.sqrt(np.maximum(var, self.cfg.sigma_min ** 2))
        si_new = np.minimum(si_new, self.cfg.sigma_max)

        self.c[i], self.s[i], self.n[i] = ci_new, si_new, ni + 1

    def update_reliability(self, i, e_raw):
        self.r[i] = (self.cfg.beta * self.r[i]
                     + (1 - self.cfg.beta) * np.exp(-(e_raw * e_raw) / (2 * self.cfg.eta * self.cfg.eta)))

    def try_merge_winner_neighbor(self, i_win):
        """
        Merge barato (NÃO deleta aqui): escolhe o vizinho j mais próximo do
        vencedor; se dist < merge_dist, retorna o evento de merge.
        Returns (i_keep, j_remove, n_keep_before, n_remove_before) ou None.
        """
        if self.cfg.merge_dist is None or self.cfg.merge_dist <= 0 or self.num_rules() < 2:
            return None

        C = np.vstack(self.c)
        ci = C[i_win]
        d = np.linalg.norm(C - ci[None, :], axis=1)
        d[i_win] = np.inf
        j = int(np.argmin(d))
        if float(d[j]) >= self.cfg.merge_dist:
            return None

        ni, nj = self.n[i_win], self.n[j]
        w = ni + nj

        self.c[i_win] = (ni * self.c[i_win] + nj * self.c[j]) / w
        self.s[i_win] = np.minimum(
            np.maximum((ni * self.s[i_win] + nj * self.s[j]) / w, self.cfg.sigma_min),
            self.cfg.sigma_max,
        )
        self.n[i_win] = w
        self.r[i_win] = max(self.r[i_win], self.r[j])

        return (i_win, j, ni, nj)


# ===========================================================================
# 3.  e-RLSM (PORTADO VERBATIM DA CÉLULA 74)
# ===========================================================================

class eRLSM:
    def __init__(self, cfg: ELSMConfig, rules: EvolvingRules, u_init):
        self.cfg = cfg
        self.rules = rules
        self.u = u_init.astype(float).copy()
        self.P = cfg.alpha * np.eye(self.u.size, dtype=float)
        self._cooldown = 0
        self._step_id = 0

    def _ensure_dims(self):
        N = self.rules.num_rules()
        if self.u.size != 2 * N:
            raise RuntimeError(f"Dim mismatch: u={self.u.size}, expected {2 * N}")

    def expand_params_for_new_rule(self, init_w):
        old = self.u.size
        new = old + 2
        u_new = np.zeros((new,), float)
        u_new[:old] = self.u
        u_new[new - 2] = 0.0
        u_new[new - 1] = float(init_w)
        self.u = u_new

        P_new = np.zeros((new, new), float)
        P_new[:old, :old] = self.P
        P_new[old:, old:] = self.cfg.alpha * np.eye(2)
        self.P = P_new

    def apply_merge_event(self, i_keep, j_remove, n_keep_before, n_remove_before):
        ik = 2 * i_keep
        jk = 2 * j_remove
        w = n_keep_before + n_remove_before

        vk, wk = self.u[ik], self.u[ik + 1]
        vj, wj = self.u[jk], self.u[jk + 1]
        self.u[ik] = (n_keep_before * vk + n_remove_before * vj) / w
        self.u[ik + 1] = (n_keep_before * wk + n_remove_before * wj) / w

        del_idx = [jk, jk + 1]
        self.u = np.delete(self.u, del_idx)
        self.P = np.delete(np.delete(self.P, del_idx, axis=0), del_idx, axis=1)

        self.rules.delete_rule(j_remove)

    def _yhat_unclipped(self, x):
        tau_eff = self.rules.effective_tau(x)
        d = build_d_from_tau(tau_eff)
        return float(d @ self.u)

    def step(self, x, y):
        self._step_id += 1
        x = np.asarray(x, float).ravel()
        y = float(y)

        # ---- novelty signals ----
        i_win, tau_max = self.rules.winner_raw(x)
        if i_win is None:
            tau_max = 0.0

        # ---- prediction for creation decision ----
        yhat_unclipped = self._yhat_unclipped(x)
        e_create_raw = y - yhat_unclipped

        # ---- adaptive stricter creation as rules grow (soft brake) ----
        N = max(self.rules.num_rules(), 1)
        tau_create_eff = max(self.cfg.tau_create_hard,
                             self.cfg.tau_create - self.cfg.growth_gamma * np.log1p(N))
        err_gate_eff = max(self.cfg.err_gate_hard,
                           self.cfg.tau_err_gate - self.cfg.growth_gamma * np.log1p(N))

        create = False
        if self._cooldown <= 0:
            create = (tau_max < tau_create_eff) or \
                     ((abs(e_create_raw) > self.cfg.e_create) and (tau_max < err_gate_eff))

        if create:
            self.rules.add_rule(x)
            self.expand_params_for_new_rule(init_w=y)
            i_win = self.rules.num_rules() - 1
            yhat_unclipped = self._yhat_unclipped(x)
            self._cooldown = self.cfg.create_cooldown
        else:
            self._cooldown = max(self._cooldown - 1, 0)

        self._ensure_dims()

        # ---- reporting output (clipped only for display) ----
        yhat_out = float(np.clip(yhat_unclipped, 0.0, 1.0)) if self.cfg.clip_yhat else float(yhat_unclipped)

        # ---- learning residual MUST be raw ----
        e_raw = y - yhat_unclipped

        # ---- RLS update (use raw residual) ----
        tau_eff = self.rules.effective_tau(x)
        d = build_d_from_tau(tau_eff)
        Pd = self.P @ d
        denom = self.cfg.lam + float(d @ Pd)
        if abs(denom) < EPS:
            denom = EPS if denom >= 0 else -EPS
        self.P = (1.0 / self.cfg.lam) * (self.P - (1.0 / denom) * np.outer(Pd, Pd))
        self.u = self.u + (self.P @ d) * e_raw

        # ---- update winner stats ----
        if i_win is None:
            i_win = int(np.argmax(self.rules.raw_tau(x)))
        self.rules.update_rule(i_win, x)
        self.rules.update_reliability(i_win, e_raw)

        # ---- cheap merge (periodic) ----
        merged_event = None
        if (self.cfg.merge_dist is not None) and (self.cfg.merge_dist > 0) \
                and (self._step_id % self.cfg.merge_every == 0):
            merged_event = self.rules.try_merge_winner_neighbor(i_win)
            if merged_event is not None:
                ik, jr, nk, nr = merged_event
                self.apply_merge_event(ik, jr, nk, nr)

        self._ensure_dims()
        return yhat_out, bool(create), bool(merged_event)


# ===========================================================================
# 4.  FEATURES HAR (RÉPLICA FIEL DA CÉLULA 21 DO FuzzyHAR.ipynb)
# ===========================================================================

def make_har_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_col: str = "Feature Origin",
    rv_col: str = "RV",
    target_col: str = "Target",
    week_len: int = 7,
    month_len: int = 30,
    include_daily: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Réplica fiel de ``make_har_features`` (célula 21 do FuzzyHAR): concatena
    treino+teste, calcula médias móveis (daily/weekly/monthly) sobre a série base
    de RV, e gera o alvo deslocado um passo à frente (``Target.shift(-1)``).

    As linhas já chegam ordenadas por data (``_normalize_dates`` apenas converte
    a coluna Date), portanto não há reordenação.

    Retorna: X_train, y_train, X_test, y_test
    """
    if week_len < 2:
        raise ValueError("week_len deve ser >= 2.")
    if month_len < week_len:
        raise ValueError("month_len deve ser >= week_len.")

    # Coluna base de volatilidade (Feature Origin == RV nos arquivos do pipeline)
    base_col = feature_col if feature_col in train_df.columns else rv_col

    full = pd.concat([train_df, test_df], ignore_index=True)
    n_tr = len(train_df)
    n_full = len(full)

    s_all = pd.Series(full[base_col].to_numpy(dtype=float))

    feats_all = []
    if include_daily:
        feats_all.append(s_all.to_numpy())
    feats_all.append(s_all.rolling(window=week_len, min_periods=week_len).mean().to_numpy())
    feats_all.append(s_all.rolling(window=month_len, min_periods=month_len).mean().to_numpy())

    X_all = np.vstack(feats_all).T

    y_all_full = full[target_col].shift(-1)

    valid_all = (~np.isnan(X_all).any(axis=1)) & (~y_all_full.isna().to_numpy())

    idx_next = np.arange(n_full) + 1
    mask_y_in_train = idx_next < n_tr
    mask_y_in_test = (idx_next >= n_tr) & (idx_next < n_full)

    valid_train = valid_all & mask_y_in_train
    valid_test = valid_all & mask_y_in_test

    X_train = X_all[valid_train].astype(np.float64)
    y_train = y_all_full[valid_train].to_numpy(dtype=np.float64)
    X_test = X_all[valid_test].astype(np.float64)
    y_test = y_all_full[valid_test].to_numpy(dtype=np.float64)

    return X_train, y_train, X_test, y_test


# ===========================================================================
# 5.  EXECUÇÃO DO MOTOR + NORMALIZAÇÃO + GRID SEARCH
# ===========================================================================

def _run_engine(
    Xn_tr: np.ndarray, yn_tr: np.ndarray,
    Xn_te: np.ndarray, yn_te: np.ndarray,
    cfg: ELSMConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Inicializa (k-means + DLSM), evolui no stream de treino, depois no stream de
    teste — protocolo da célula 74 (evolução habilitada também no teste). A
    previsão de cada passo é feita ANTES do update RLS daquele passo (sem
    look-ahead). Opera em escala normalizada.
    """
    centers0, sigmas0 = kmeans_init_rules(Xn_tr, cfg.n_init_rules, cfg.sigma_scale, cfg.seed, cfg)
    u0 = dlsm_fit(Xn_tr, yn_tr, centers0, sigmas0)
    model = eRLSM(cfg, EvolvingRules(cfg, centers0, sigmas0), u0)

    yhat_tr = np.empty(len(Xn_tr), dtype=float)
    for k, (x, yy) in enumerate(zip(Xn_tr, yn_tr)):
        yhat_tr[k], _, _ = model.step(x, yy)

    created_test = 0
    merged_test = 0
    yhat_te = np.empty(len(Xn_te), dtype=float)
    for k, (x, yy) in enumerate(zip(Xn_te, yn_te)):
        yhat_te[k], did_c, did_m = model.step(x, yy)
        created_test += int(did_c)
        merged_test += int(did_m)

    diag = {
        "n_rules_final": int(model.rules.num_rules()),
        "created_test": int(created_test),
        "merged_test": int(merged_test),
    }
    return yhat_tr, yhat_te, diag


def _train_prequential_rmse(Xn_tr: np.ndarray, yn_tr: np.ndarray, cfg: ELSMConfig) -> float:
    """
    Critério de seleção do grid search: RMSE prequencial no TREINO (a previsão de
    cada passo é feita antes do update). Usa apenas o stream de treino — sem
    look-ahead sobre o teste.
    """
    centers0, sigmas0 = kmeans_init_rules(Xn_tr, cfg.n_init_rules, cfg.sigma_scale, cfg.seed, cfg)
    u0 = dlsm_fit(Xn_tr, yn_tr, centers0, sigmas0)
    model = eRLSM(cfg, EvolvingRules(cfg, centers0, sigmas0), u0)

    yhat = np.empty(len(Xn_tr), dtype=float)
    for k, (x, yy) in enumerate(zip(Xn_tr, yn_tr)):
        yhat[k], _, _ = model.step(x, yy)

    return float(np.sqrt(np.mean((yn_tr - yhat) ** 2)))


def select_elsm_hyperparams(
    Xn_tr: np.ndarray, yn_tr: np.ndarray,
    grid: List[Dict[str, Any]],
    base_cfg: ELSMConfig,
    *,
    verbose: bool = False,
) -> ELSMConfig:
    """Grid search no treino: devolve o ``ELSMConfig`` de menor RMSE prequencial."""
    best_cfg = base_cfg
    best_rmse = np.inf
    for combo in grid:
        cfg = replace(base_cfg, **combo)
        try:
            rmse = _train_prequential_rmse(Xn_tr, yn_tr, cfg)
        except Exception as e:
            if verbose:
                print(f"    [grid skip] {combo}: {e}")
            continue
        if verbose:
            print(f"    [grid] {combo} -> RMSE_train={rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_cfg = cfg
    if verbose:
        print(f"    [grid best] RMSE_train={best_rmse:.6f}")
    return best_cfg


def _minmax_fit_apply(X_train: np.ndarray, X_test: np.ndarray):
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_rng = np.maximum(x_max - x_min, 1e-12)
    Xn_tr = (X_train - x_min) / x_rng
    Xn_te = (X_test - x_min) / x_rng
    return Xn_tr, Xn_te


def _minmax_fit_y(y_train: np.ndarray):
    y_min = float(y_train.min())
    y_max = float(y_train.max())
    y_rng = max(y_max - y_min, 1e-12)
    return y_min, y_rng


def run_elsm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *,
    cfg: Optional[ELSMConfig] = None,
    grid: Optional[List[Dict[str, Any]]] = None,
    base_cfg: Optional[ELSMConfig] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Normaliza (min-max, stats de treino), opcionalmente seleciona hiperparâmetros
    via grid search no treino, roda o e-LSM, e desnormaliza as previsões.

    Os parâmetros encontrados no grid search são EXATAMENTE os usados para gerar
    as previsões finais (mesmo ``ELSMConfig`` passado ao motor), e ficam
    registrados em ``info["cfg"]``.

    Retorna: yhat_train_real, yhat_test_real, info
    """
    Xn_tr, Xn_te = _minmax_fit_apply(X_train, X_test)
    y_min, y_rng = _minmax_fit_y(y_train)
    yn_tr = (y_train - y_min) / y_rng
    yn_te = (y_test - y_min) / y_rng

    if cfg is None:
        base = base_cfg if base_cfg is not None else ELSMConfig()
        if grid:
            cfg = select_elsm_hyperparams(Xn_tr, yn_tr, grid, base, verbose=verbose)
        else:
            cfg = base

    yhat_tr_n, yhat_te_n, diag = _run_engine(Xn_tr, yn_tr, Xn_te, yn_te, cfg)

    yhat_tr_real = yhat_tr_n * y_rng + y_min
    yhat_te_real = yhat_te_n * y_rng + y_min

    info: Dict[str, Any] = {"cfg": cfg}
    info.update(diag)
    return yhat_tr_real, yhat_te_real, info


# ===========================================================================
# 6.  INTEGRAÇÃO COM O PIPELINE FuzzyHAR / integracao_rvfts
# ===========================================================================

# Nomes de coluna seguindo a convenção dos modelos HAR ("X prediction" /
# "X OOS prediction") para que integracao_rvfts._insample_col / _oos_col já
# resolvam automaticamente, sem nenhuma alteração naquele arquivo.
ELSM_TRAIN_COL: str = "e-LSM prediction"
ELSM_OOS_COL: str = "e-LSM OOS prediction"


def add_elsm_to_dataframes(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    horizon: int,
    *,
    target_column: str = "Target",
    feature_col: str = "Feature Origin",
    rv_column: str = "RV",
    week_len: int = 7,
    month_len: int = 30,
    grid: Optional[List[Dict[str, Any]]] = DEFAULT_ELSM_GRID,
    base_cfg: Optional[ELSMConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Roda o e-LSM sobre os dataframes do pipeline FuzzyHAR e adiciona as colunas
    ``"e-LSM prediction"`` (treino) e ``"e-LSM OOS prediction"`` (teste).

    O alinhamento é IDÊNTICO ao do Fuzzy HAR (célula 34): a previsão de teste é
    atribuída posicionalmente ao ``df_test`` (``e-LSM[d]`` casa com ``Target[d]``)
    e a de treino com offset ``month_len``.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    # base_cfg de produção: clip_yhat=False (não limitar picos de vol ao máx. de
    # treino); previsões <=0 são tratadas pelo filtro do pipeline.
    if base_cfg is None:
        base_cfg = ELSMConfig(clip_yhat=False)

    X_tr, y_tr, X_te, y_te = make_har_features(
        df_train, df_test,
        feature_col=feature_col, rv_col=rv_column, target_col=target_column,
        week_len=week_len, month_len=month_len, include_daily=True,
    )

    if verbose:
        print(f"\n[e-LSM] horizon=t+{horizon} | features HAR={X_tr.shape[1]} "
              f"| train={len(X_tr)} obs | test={len(X_te)} obs")

    t0 = time.time()
    yhat_tr_real, yhat_te_real, info = run_elsm(
        X_tr, y_tr, X_te, y_te,
        grid=grid, base_cfg=base_cfg, verbose=verbose,
    )
    info["elapsed_s"] = float(time.time() - t0)

    # ---- atribuição do teste (posicional, espelha a célula 34) ----
    n_test = len(df_test)
    if len(yhat_te_real) == n_test:
        df_test[ELSM_OOS_COL] = yhat_te_real
    else:
        col = np.full(n_test, np.nan)
        m = min(len(yhat_te_real), n_test)
        col[-m:] = yhat_te_real[-m:]
        df_test[ELSM_OOS_COL] = col

    # ---- atribuição do treino (offset month_len, espelha a célula 34) ----
    df_train[ELSM_TRAIN_COL] = np.nan
    loc = df_train.columns.get_loc(ELSM_TRAIN_COL)
    start = int(month_len)
    n = min(len(yhat_tr_real), max(len(df_train) - start, 0))
    if n > 0:
        df_train.iloc[start:start + n, loc] = yhat_tr_real[:n]

    cfg = info["cfg"]
    info["n_valid_test"] = int(np.sum(~np.isnan(df_test[ELSM_OOS_COL].to_numpy(dtype=float))))
    if verbose:
        print(f"  [e-LSM] cfg: n_init_rules={cfg.n_init_rules}, sigma_max={cfg.sigma_max}, "
              f"tau_create={cfg.tau_create} | regras finais={info['n_rules_final']} "
              f"| criadas(teste)={info['created_test']} | fundidas(teste)={info['merged_test']} "
              f"| {info['elapsed_s']:.1f}s")

    return df_train, df_test, info


def cfg_to_row(symbol: str, horizon: int, info: Dict[str, Any]) -> Dict[str, Any]:
    """Achata o ELSMConfig escolhido + diagnósticos numa linha para exportação."""
    cfg: ELSMConfig = info["cfg"]
    row = {"Symbol": symbol.replace("USDT", ""), "Horizon": f"t+{horizon}"}
    row.update(vars(cfg) if not hasattr(cfg, "__dataclass_fields__")
               else {f: getattr(cfg, f) for f in cfg.__dataclass_fields__})
    for k in ("n_rules_final", "created_test", "merged_test", "n_valid_test", "elapsed_s"):
        if k in info:
            row[k] = info[k]
    return row


# ===========================================================================
# 7.  TESTE DE SANIDADE  (executável via `python elsm_model.py`)
# ===========================================================================

def _sanity_check() -> None:
    print("=" * 70)
    print("e-LSM — Teste de sanidade")
    print("=" * 70)

    rng = np.random.RandomState(0)
    T = 1500
    rv = np.zeros(T)
    rv[0] = 1.0
    for t in range(1, T):
        rv[t] = 0.92 * rv[t - 1] + 0.08 * rng.gamma(2, 1.0) + 0.05 * rng.randn()
    rv = np.maximum(rv, 0.01)

    df = pd.DataFrame({"Date": pd.date_range("2018-01-01", periods=T),
                       "RV": rv, "Feature Origin": rv})
    df["Target"] = df["RV"].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)

    n_tr = 1000
    df_tr = df.iloc[:n_tr].reset_index(drop=True)
    df_te = df.iloc[n_tr:].reset_index(drop=True)

    df_tr, df_te, info = add_elsm_to_dataframes(
        df_tr, df_te, horizon=1, grid=DEFAULT_ELSM_GRID, verbose=True,
    )

    y = df_te["Target"].to_numpy(float)
    f = df_te[ELSM_OOS_COL].to_numpy(float)
    mask = ~(np.isnan(y) | np.isnan(f))
    mse_elsm = float(np.mean((y[mask] - f[mask]) ** 2))

    naive = df_te["RV"].to_numpy(float)
    mse_naive = float(np.mean((y[mask] - naive[mask]) ** 2))

    print(f"\n  Previsões válidas: {int(mask.sum())} / {len(y)}")
    print(f"  MSE e-LSM:        {mse_elsm:.6f}")
    print(f"  MSE persistência: {mse_naive:.6f}")
    print(f"  Regras finais:    {info['n_rules_final']}")
    if mse_elsm < mse_naive * 2.0:
        print("  → MSE razoável. OK.")
    else:
        print("  ⚠ MSE muito acima da persistência — investigar hiperparâmetros.")
    print("\n" + "=" * 70)
    print("Teste concluído.")
    print("=" * 70)


if __name__ == "__main__":
    _sanity_check()
