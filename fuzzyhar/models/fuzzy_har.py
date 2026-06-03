# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Pipeline Fuzzy HAR (clustering GK em R + ALSM). Codigo verbatim do notebook.'''
import os
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from fuzzyhar.config import (
    HORIZON_CONFIG, SPLIT, rules_grid, ze, lamb, alfa, week_len, month_len,
)
from fuzzyhar.features import make_har_features, gaussgranules
from fuzzyhar.models.r_gk import run_gk_clustering

def _resolve_input_filenames(symbol: str, split: str) -> tuple[str, str]:
    """Retorna (insample_filename, outofsample_filename) para o split dado."""
    if split == "40_60":
        return (
            f"{symbol}_5m_daily_insample.xlsx",
            f"{symbol}_5m_daily_outofsample.xlsx",
        )
    return (
        f"{symbol}_5m_daily__SPLIT_{split}__insample_train.xlsx",
        f"{symbol}_5m_daily__SPLIT_{split}__oos_predictions.xlsx",
    )


def _build_io_paths(symbol: str, base_input_dir: str, base_output_dir: str) -> dict[str, str]:
    insample_file, outsample_file = _resolve_input_filenames(symbol, SPLIT)

    insample_path = os.path.join(
        base_input_dir,
        "insample",
        insample_file,
    )
    outsample_path = os.path.join(
        base_input_dir,
        "out_of_sample",
        outsample_file,
    )

    symbol_out_dir = os.path.join(base_output_dir, symbol)
    os.makedirs(symbol_out_dir, exist_ok=True)

    return {
        "insample_path": insample_path,
        "outsample_path": outsample_path,
        "symbol_out_dir": symbol_out_dir,
    }

def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    df_name: str,
) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} estÃ¡ sem as colunas obrigatÃ³rias: {missing}. "
            f"Colunas disponÃ­veis: {list(df.columns)}"
        )

def _load_and_prepare_base_data(
    insample_path: str,
    outsample_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_excel(insample_path)
    df_test = pd.read_excel(outsample_path)

    # Mantém o comportamento das funções originais:
    # tenta renomear "Data" -> "Date" caso exista.
    if "Data" in df_train.columns:
        df_train.rename(columns={"Data": "Date"}, inplace=True)
    if "Data" in df_test.columns:
        df_test.rename(columns={"Data": "Date"}, inplace=True)

    # Normaliza coluna de alvo: aceita 'Target' (splits 60_40/70_30)
    # ou 'target' (split 40_60, formato original).
    if "Target" in df_train.columns and "target" not in df_train.columns:
        df_train["target"] = df_train["Target"]
    if "Target" in df_test.columns and "target" not in df_test.columns:
        df_test["target"] = df_test["Target"]

    _validate_required_columns(df_train, ["target", "RV"], "df_train")
    _validate_required_columns(df_test, ["target", "RV"], "df_test")

    df_train["Target"] = df_train["target"]
    df_test["Target"] = df_test["target"]

    df_train["Feature Origin"] = df_train["RV"]
    df_test["Feature Origin"] = df_test["RV"]

    return df_train, df_test


def _save_outputs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    symbol: str,
    horizon: int,
    symbol_out_dir: str,
) -> tuple[str, str]:
    train_out_path = os.path.join(
        symbol_out_dir,
        f"{symbol}_tplus{horizon}_train_with_fuzzy.xlsx",
    )
    test_out_path = os.path.join(
        symbol_out_dir,
        f"{symbol}_tplus{horizon}_test_with_fuzzy.xlsx",
    )

    df_train.to_excel(train_out_path, index=False)
    df_test.to_excel(test_out_path, index=False)

    return train_out_path, test_out_path

def train_model(
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    r: int,
    lamb: float,
    alfa: float,
    ze: float,                 # <- mudou: agora recebemos apenas ze
    centers: np.ndarray,
    sigmas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
      Treina um modelo ALSM com atualizaÃ§Ã£o online dos parÃ¢metros.
      Usa ativaÃ§Ãµes gaussianas e fator de confianÃ§a adaptativo psi.

      ParÃ¢metros do kernel (derivados internamente):
        kc Ã© constante (1/sqrt(2Ï€));
        kz = kc / (ze^3);
        zs = 1 / (2 * ze^2).
    """

    # --- kc fixo e derivaÃ§Ã£o de kz/zs a partir de ze (mantÃ©m compatibilidade conceitual) ---
    kc = 1.0 / np.sqrt(2.0 * np.pi)
    kz = kc / (ze ** 3)
    zs = 1.0 / (2.0 * (ze ** 2))

    ndatatrain = Xtrain.shape[0]
    OT = np.zeros(ndatatrain)
    norm = np.zeros(ndatatrain)

    P = alfa * np.eye(2 * r)
    u = np.zeros(2 * r)
    a = np.zeros(2 * r)

    # HistÃ³rico (inalterado no formato; agora registra ze, kz e zs calculados)
    hist: Dict[str, Any] = {
        "n": int(ndatatrain),
        "n_features": int(Xtrain.shape[1]),
        "params": {"r": r, "lamb": lamb, "alfa": alfa, "ze": ze, "kz": kz, "zs": zs},
        "y_true":      np.zeros(ndatatrain, dtype=float),
        "y_pred_pre":  np.zeros(ndatatrain, dtype=float),
        "error":       np.zeros(ndatatrain, dtype=float),
        "psi":         np.zeros(ndatatrain, dtype=float),
        "aTPa":        np.zeros(ndatatrain, dtype=float),
        "b":           np.zeros(ndatatrain, dtype=float),
        "sumd":        np.zeros(ndatatrain, dtype=float),
        "d_min":       np.zeros(ndatatrain, dtype=float),
        "d_max":       np.zeros(ndatatrain, dtype=float),
        "a_norm":      np.zeros(ndatatrain, dtype=float),
        "u_norm":      np.zeros(ndatatrain, dtype=float),
        "P_trace":     np.zeros(ndatatrain, dtype=float),
        "P_cond":      np.zeros(ndatatrain, dtype=float),
    }

    # -------- Salvaguardas numÃ©ricas (constantes) ----------
    EPS    = 1e-300  # piso para denominadores (~zero, mas nÃ£o zero) para evitar divisÃ£o por zero.
                     # Impacto: se valores normais, nunca Ã© usado; se sumdâ‰ˆ0, evita NaN e torna a atualizaÃ§Ã£o quase nula.
    BMIN   = 1e-300  # piso para o denominador b (evita divisÃ£o por um nÃºmero tÃ£o pequeno que causaria estouro).
    LMIN   = 1e-12   # piso mÃ­nimo para Î» na divisÃ£o de P (se alguÃ©m passar Î» muito pequeno/acidentalmente zero).
    ZSMIN  = 1e-12   # piso para zs (evita divisÃ£o por zero/underflow em exp(-sq/zs)).
    EXP_CLIP = -700  # limiar para cortar o expoente da exponencial (np.exp(-700) ~ 5e-305, abaixo disso vira 0 sob precisÃ£o dupla).

    st = time.process_time()

    for t in range(ndatatrain):
        x = Xtrain[t]

        # --- AtivaÃ§Ãµes gaussianas ---
        # Saneamento numÃ©rico: converte d para float e remove NaN/Inf por 0 (pertinÃªncia nula).
        # Justificativa: se gaussgranules retornar algo invÃ¡lido por escala extrema, evitamos contaminar o vetor 'a'.
        d = np.asarray(gaussgranules(x, centers, sigmas), dtype=float).ravel()
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # Soma das ativaÃ§Ãµes (denominador da normalizaÃ§Ã£o)
        sumd = float(np.sum(d))
        # Se sumd nÃ£o for finito ou for <= 0, usamos EPS (quase zero) para impedir divisÃ£o por zero.
        # Isso preserva as razÃµes (se existirem) e, quando tudo Ã© 0, gera um 'a' de magnitude minÃºscula,
        # levando a uma atualizaÃ§Ã£o praticamente nula (comportamento seguro).
        if (not np.isfinite(sumd)) or (sumd <= 0.0):
            sumd = EPS

        # Vetor de regressÃ£o 'a' com normalizaÃ§Ã£o segura por sumd
        for j in range(r):
            a[2 * j]     = d[j]**2 / sumd
            a[2 * j + 1] = d[j]    / sumd

        # PrediÃ§Ã£o antes de atualizar (para medir o erro)
        y_pre = float(np.dot(a, u))
        erro  = float(ytrain[t] - y_pre)
        sq    = float(np.abs(erro)**2)

        # --- Psi com proteÃ§Ã£o ---
        # zs_eff garante zs > 0; exp_arg Ã© cortado para evitar underflow extremo em np.exp.
        # Quando exp_arg < -700, definimos exp como 0.0 explicitamente (equivalente numÃ©rico).
        zs_eff  = max(float(zs), ZSMIN)
        exp_arg = -sq / zs_eff
        psi     = float(kz) * (0.0 if exp_arg < EXP_CLIP else np.exp(exp_arg))

        # --- Ganho RLS com proteÃ§Ã£o ---
        aTPa = float(np.dot(np.dot(a, P), a))
        b    = float(lamb + psi * aTPa)

        Pa = np.dot(P, a)

        # lambda_eff evita divisÃ£o por Î» muito pequeno; b Ã© verificado quanto a finitude e piso mÃ­nimo.
        lambda_eff = max(float(lamb), LMIN)
        if (not np.isfinite(b)) or (abs(b) < BMIN):
            # AtualizaÃ§Ã£o Ã© pulada nesta iteraÃ§Ã£o para evitar corromper P e u com nÃºmeros invÃ¡lidos.
            # Impacto: mÃ­nimo â€” apenas quando a configuraÃ§Ã£o numÃ©rica estÃ¡ patolÃ³gica; caso normal, nÃ£o dispara.
            pass
        else:
            P = (P - psi * np.outer(Pa, Pa) / b) / lambda_eff
            u = u + (psi * np.dot(P, a) * erro)

        # SaÃ­das pÃ³s-atualizaÃ§Ã£o
        OT[t]   = np.dot(a, u)
        norm[t] = np.linalg.norm(u)

        # Logs/histÃ³rico (inalterado)
        hist["y_true"][t]     = float(ytrain[t])
        hist["y_pred_pre"][t] = y_pre
        hist["error"][t]      = erro
        hist["psi"][t]        = float(psi)
        hist["aTPa"][t]       = aTPa
        hist["b"][t]          = float(b)
        hist["sumd"][t]       = float(np.sum(d))
        hist["d_min"][t]      = float(np.min(d))
        hist["d_max"][t]      = float(np.max(d))
        hist["a_norm"][t]     = float(np.linalg.norm(a))
        hist["u_norm"][t]     = float(norm[t])
        hist["P_trace"][t]    = float(np.trace(P))
        try:
            hist["P_cond"][t] = float(np.linalg.cond(P))
        except Exception:
            # Se P ficar singular/indefinido numericamente, registramos âˆž (evita crash do log).
            hist["P_cond"][t] = np.inf

    et = time.process_time()
    train_time = et - st

    # Fechamento do histÃ³rico
    hist["train_time_seconds"] = float(train_time)
    hist["u_final"] = u.copy()
    hist["P_final"] = P.copy()
    hist["OT"]      = OT.copy()
    hist["norm"]    = norm.copy()

    train_model._last_history = hist

    return OT, norm, u, P

def test_model(
    Xtest: np.ndarray,
    ytest: np.ndarray,
    r: int,
    lamb: float,
    P: np.ndarray,
    u: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray
) -> np.ndarray:

    """
      Realiza a prediÃ§Ã£o adaptativa com o modelo ALSM usando atualizaÃ§Ã£o online no teste.
    """

    ndatatest = Xtest.shape[0]
    OS = np.zeros(ndatatest)
    a  = np.zeros(2 * r)

    # HistÃ³rico compatÃ­vel com o do treino
    hist: Dict[str, Any] = {
        "n": int(ndatatest),
        "n_features": int(Xtest.shape[1]),
        "params": {"r": r, "lamb": lamb, "alfa": None, "ze": None, "kz": None, "zs": None},
        "y_true":      np.zeros(ndatatest, dtype=float),
        "y_pred_pre":  np.zeros(ndatatest, dtype=float),
        "error":       np.zeros(ndatatest, dtype=float),
        "psi":         np.zeros(ndatatest, dtype=float),
        "aTPa":        np.zeros(ndatatest, dtype=float),
        "b":           np.zeros(ndatatest, dtype=float),
        "sumd":        np.zeros(ndatatest, dtype=float),
        "d_min":       np.zeros(ndatatest, dtype=float),
        "d_max":       np.zeros(ndatatest, dtype=float),
        "a_norm":      np.zeros(ndatatest, dtype=float),
        "u_norm":      np.zeros(ndatatest, dtype=float),
        "P_trace":     np.zeros(ndatatest, dtype=float),
        "P_cond":      np.zeros(ndatatest, dtype=float),
    }

    # Salvaguardas numÃ©ricas do teste (anÃ¡logas Ã s do treino)
    EPS  = 1e-300   # piso para sumd
    BMIN = 1e-300   # piso para b
    LMIN = 1e-12    # piso para Î»

    st = time.process_time()

    for t in range(ndatatest):
        x = Xtest[t]

        # AtivaÃ§Ãµes gaussianas e saneamento
        d = np.asarray(gaussgranules(x, centers, sigmas), dtype=float).ravel()
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        sumd = float(np.sum(d))
        # Se todas as ativaÃ§Ãµes forem quase-zero (ou invÃ¡lidas), usa EPS para evitar divisÃ£o por zero.
        # Isso faz 'a' ficar de magnitude minÃºscula, provocando atualizaÃ§Ã£o desprezÃ­vel â€” seguro e coerente.
        if (not np.isfinite(sumd)) or (sumd <= 0.0):
            sumd = EPS

        # Vetor 'a' normalizado com proteÃ§Ã£o
        for j in range(r):
            a[2 * j]     = d[j]**2 / sumd
            a[2 * j + 1] = d[j]    / sumd

        # PrediÃ§Ã£o e erro (prÃ©-atualizaÃ§Ã£o)
        y_pre = float(np.dot(a, u))
        err   = float(ytest[t] - y_pre)

        # Ganho RLS com proteÃ§Ã£o
        aTPa = float(np.dot(np.dot(a, P), a))
        b    = float(lamb + aTPa)
        Pa   = np.dot(P, a)

        lambda_eff = max(float(lamb), LMIN)
        if (not np.isfinite(b)) or (abs(b) < BMIN):
            # Pula a atualizaÃ§Ã£o nesta iteraÃ§Ã£o (evita contaminar P/u com nÃºmeros ruins).
            pass
        else:
            P = (P - np.outer(Pa, Pa) / b) / lambda_eff
            u = u + (np.dot(P, a) * (ytest[t] - np.dot(a, u)))

        OS[t] = np.dot(a, u)

        # Logs da iteraÃ§Ã£o
        hist["y_true"][t]     = float(ytest[t])
        hist["y_pred_pre"][t] = y_pre
        hist["error"][t]      = err
        hist["psi"][t]        = 0.0            # no teste nÃ£o hÃ¡ psi (mantido por compatibilidade de log)
        hist["aTPa"][t]       = aTPa
        hist["b"][t]          = b
        hist["sumd"][t]       = float(np.sum(d))
        hist["d_min"][t]      = float(np.min(d))
        hist["d_max"][t]      = float(np.max(d))
        hist["a_norm"][t]     = float(np.linalg.norm(a))
        hist["u_norm"][t]     = float(np.linalg.norm(u))
        hist["P_trace"][t]    = float(np.trace(P))
        try:
            hist["P_cond"][t] = float(np.linalg.cond(P))
        except Exception:
            hist["P_cond"][t] = np.inf

    et = time.process_time()

    # Finaliza o histÃ³rico (mesmas chaves do treino)
    hist["train_time_seconds"] = float(et - st)
    hist["u_final"] = u.copy()
    hist["P_final"] = P.copy()
    hist["OT"]      = OS.copy()
    hist["norm"]    = hist["u_norm"].copy()

    test_model._last_history = hist

    return OS

def _rmse(y_true, y_pred):
    if y_true.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true, y_pred):
    if y_true.size == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)))

def grid_search_rules_features(
    df_train,
    df_test,
    rules_grid,
    *,
    lamb,
    alfa,
    ze,
    har_week_len: int = 7,
    har_month_len: int = 30,
    har_include_daily: bool = True,
    verbose: bool = True,
):

    ze = float(ze)

    wl = int(har_week_len)
    ml = max(wl, int(har_month_len))

    X_tr, y_tr, _, _ = make_har_features(
        df_train,
        df_test,
        week_len=wl,
        month_len=ml,
        include_daily=har_include_daily
    )

    if X_tr.size == 0 or X_tr.shape[1] == 0:
        raise RuntimeError("HAR: conjunto de treino vazio com as janelas fornecidas.")

    trials = []
    best = None

    for r_rules in rules_grid:
        try:
            centers_df, memberships_df, sigmas = run_gk_clustering(X_tr, int(r_rules))

            OT, norm, u, P = train_model(
                X_tr,
                y_tr,
                int(r_rules),
                lamb,
                alfa,
                ze,
                centers_df.to_numpy(),
                sigmas,
            )
        except Exception as e:
            if verbose:
                print(f"[skip] mode=har r={r_rules} ze={ze:.6f}: treino -> {e}")
            continue

        rmse_tr = _rmse(y_tr, OT)
        mae_tr = _mae(y_tr, OT)

        trial = {
            "feature_mode": "har",
            "r": int(r_rules),
            "delay_like": 3,
            "ze": ze,
            "n_train": int(X_tr.shape[0]),
            "n_features": int(X_tr.shape[1]),
            "rmse_train": rmse_tr,
            "mae_train": mae_tr,
            "OT": OT,
            "norm": norm,
            "u": u,
            "P": P,
            "centers": centers_df.to_numpy(),
            "sigmas": sigmas,
        }

        trials.append(trial)

        if verbose:
            print(
                f"[trial] mode=har r={r_rules:>2d} ze={ze:.6f} "
                f"| n={X_tr.shape[0]:>4d} "
                f"| RMSE={rmse_tr:.6f} | MAE={mae_tr:.6f}"
            )

        if best is None or trial["rmse_train"] < best["rmse_train"]:
            best = trial

    if best is None:
        raise RuntimeError("Nenhum experimento vÃ¡lido foi executado.")

    if verbose:
        print(
            f"[best] mode=har r={best['r']} ze={best['ze']:.6f} "
            f"| RMSE={best['rmse_train']:.6f}"
        )

    return {
        "best": {
            "feature_mode": best["feature_mode"],
            "r": best["r"],
            "delay_like": best["delay_like"],
            "ze": best["ze"],
            "rmse_train": best["rmse_train"],
            "mae_train": best["mae_train"],
            "u": best["u"],
            "P": best["P"],
            "centers": best["centers"],
            "sigmas": best["sigmas"],
            "n_train": best["n_train"],
            "n_features": best["n_features"],
        },
        "trials": trials,
    }

def _select_best_har_hyperparameters(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    symbol_out_dir: str,
    rules_grid: list[int],
    ze: float,
    lamb: float,
    alfa: float,
    week_len: int,
    month_len: int,
) -> tuple[int, float]:
    res_har = grid_search_rules_features(
        df_train=df_train,
        df_test=df_test,
        rules_grid=rules_grid,
        lamb=lamb,
        alfa=alfa,
        #ze_grid=[1],
        ze=1,
        har_week_len=week_len,
        har_month_len=month_len,
        har_include_daily=True,
        verbose=False,
    )

    best_rules_number_har = int(res_har["best"]["r"])
    best_ze_har = float(res_har["best"]["ze"])

    return best_rules_number_har, best_ze_har

def _run_best_har_block(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    best_rules_number_har: int,
    best_ze_har: float,
    lamb: float,
    alfa: float,
    week_len: int,
    month_len: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_har, y_train_har, X_test_har, y_test_har = make_har_features(
        train_df=df_train,
        test_df=df_test,
        week_len=week_len,
        month_len=month_len,
        include_daily=True,
    )

    centers_df_har, memberships_df_har, sigmas_har = run_gk_clustering(
        X_train_har,
        best_rules_number_har,
    )

    OT_har, norm_har, u_har, P_har = train_model(
        X_train_har,
        y_train_har,
        best_rules_number_har,
        lamb,
        alfa,
        best_ze_har,
        centers_df_har.to_numpy(),
        sigmas_har,
    )

    OS_har = test_model(
        X_test_har,
        y_test_har,
        best_rules_number_har,
        lamb,
        P_har,
        u_har,
        centers_df_har.to_numpy(),
        sigmas_har,
    )

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["Fuzzy HAR"] = np.nan
    start_idx_har = month_len

    df_train.loc[
        df_train.index[start_idx_har : start_idx_har + len(OT_har)],
        "Fuzzy HAR",
    ] = OT_har

    df_test["Fuzzy HAR"] = OS_har

    return df_train, df_test

def run_fuzzy_for_symbol(symbol: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roda apenas o pipeline fuzzy HAR para um sÃ­mbolo e horizonte.
    """
    if horizon not in HORIZON_CONFIG:
        raise ValueError(f"Horizonte invÃ¡lido: {horizon}. Use 1, 7 ou 30.")

    config = HORIZON_CONFIG[horizon]

    # 1) Paths
    paths = _build_io_paths(
        symbol=symbol,
        base_input_dir=config.base_input_dir,
        base_output_dir=config.base_output_dir,
    )

    # Garante que a pasta de saÃ­da do sÃ­mbolo exista
    os.makedirs(paths["symbol_out_dir"], exist_ok=True)

    # 2) Carrega e prepara os dados
    df_train, df_test = _load_and_prepare_base_data(
        insample_path=paths["insample_path"],
        outsample_path=paths["outsample_path"],
    )

    # 3) SeleÃ§Ã£o de hiperparÃ¢metros apenas para HAR
    best_rules_number_har, best_ze_har = _select_best_har_hyperparameters(
        df_train=df_train,
        df_test=df_test,
        symbol_out_dir=paths["symbol_out_dir"],
        rules_grid=rules_grid,
        ze=ze,
        lamb=lamb,
        alfa=alfa,
        week_len=week_len,
        month_len=month_len,
)

    # 4) Executa apenas o bloco HAR
    df_train, df_test = _run_best_har_block(
        df_train=df_train,
        df_test=df_test,
        best_rules_number_har=best_rules_number_har,
        best_ze_har=best_ze_har,
        lamb=lamb,
        alfa=alfa,
        week_len=week_len,
        month_len=month_len,
    )

    # 5) Salva outputs
    train_out_path, test_out_path = _save_outputs(
        df_train=df_train,
        df_test=df_test,
        symbol=symbol,
        horizon=horizon,
        symbol_out_dir=paths["symbol_out_dir"],
    )

    print(
        f"[{symbol} | t+{horizon} | HAR only] concluÃ­do.\n"
        f"Train: {train_out_path}\n"
        f"Test:  {test_out_path}\n"
        f"Best HAR: r={best_rules_number_har}, ze={best_ze_har}"
    )

    return df_train, df_test


def run_all_fuzzy():
    # Usa exclusivamente a lista definida em Parameters
    symbols = list(SYMBOLS)

    all_results = {}

    for horizon in [1, 7, 30]:
        all_results[horizon] = {}
        for sym in symbols:
            df_tr, df_te = run_fuzzy_for_symbol(sym, horizon)
            all_results[horizon][sym] = {"train": df_tr, "test": df_te}
