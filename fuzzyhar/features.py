# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Feature engineering e pre-processamento (funcoes puras).'''
import numpy as np
import pandas as pd
from typing import Tuple, Union

def _normalize_dates(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
      Description:
        Converte a coluna 'Date' do dataframe recebido como
        parÃ¢metro (assumindo que essa coluna existe) para datetime.

      Params:
        df (pd.DataFrame): DataFrame de entrada contendo a coluna 'Date',
          alÃ©m de 'Feature Origin' e 'Target'. Assume-se que as linhas jÃ¡
          chegam ordenadas por data.

      Return:
        df (pd.DataFrame): CÃ³pia do DataFrame original com a coluna 'Date'
          convertida para datetime (valores invÃ¡lidos viram NaT).
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

def _align_y_next(
    target: pd.Series
) -> pd.Series:

    """
      Description:
        Prepara o alvo para previsÃ£o one-step-ahead. Ao deslocar a sÃ©rie em -1,
        garantimos que as features calculadas em t sejam pareadas com o valor
        de Target observado em t+1 (o â€œprÃ³ximo perÃ­odoâ€). Isso evita vazamento
        de informaÃ§Ã£o e alinha corretamente X_t -> y_{t+1} no treinamento/teste.

      Params:
        target (pd.Series): SÃ©rie alvo original (Target_t) no mesmo Ã­ndice
          temporal das features.

      Return:
        (pd.Series): SÃ©rie alvo deslocada em -1 (Target_{t+1}); a Ãºltima
          posiÃ§Ã£o torna-se NaN por nÃ£o haver observaÃ§Ã£o futura correspondente.
    """

    return target.shift(-1)

def make_har_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    week_len: int = 5,
    month_len: int = 22,
    include_daily: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ConstrÃ³i as features HAR (Heterogeneous Autoregressive) a partir da coluna
    'Feature Origin' e gera o alvo deslocado em um passo Ã  frente:

        y(t) = Target_{t+1}

    Retorna:
        X_train, y_train, X_test, y_test
    """

    if week_len < 2:
        raise ValueError("week_len deve ser >= 2.")
    if month_len < week_len:
        raise ValueError("month_len deve ser >= week_len.")

    # 1. Normaliza datas e copia
    tr = _normalize_dates(train_df)
    te = _normalize_dates(test_df)

    # 2. Concatena treino + teste numa sÃ©rie temporal contÃ­nua
    full = pd.concat([tr, te], ignore_index=True)

    n_tr = len(tr)
    n_full = len(full)

    # 3. Extrai a sÃ©rie base de volatilidade
    fo_all = full["Feature Origin"].to_numpy(dtype=float)
    s_all = pd.Series(fo_all)

    # 4. Cria as colunas HAR sobre a sÃ©rie completa
    feats_all = []

    if include_daily:
        feats_all.append(s_all.to_numpy())

    week_col = (
        s_all.rolling(window=week_len, min_periods=week_len)
        .mean()
        .to_numpy()
    )
    feats_all.append(week_col)

    month_col = (
        s_all.rolling(window=month_len, min_periods=month_len)
        .mean()
        .to_numpy()
    )
    feats_all.append(month_col)

    X_all = np.vstack(feats_all).T

    # 5. Alvo deslocado em um passo Ã  frente
    y_all_full = _align_y_next(full["Target"])

    # 6. Linhas matematicamente vÃ¡lidas
    valid_all = (~np.isnan(X_all).any(axis=1)) & (~y_all_full.isna().to_numpy())

    # 7. Define treino/teste com base em onde cai o alvo t+1
    idx_next = np.arange(n_full) + 1
    mask_y_in_train = idx_next < n_tr
    mask_y_in_test = (idx_next >= n_tr) & (idx_next < n_full)

    valid_train = valid_all & mask_y_in_train
    valid_test = valid_all & mask_y_in_test

    # 8. Separa os conjuntos finais
    X_train = X_all[valid_train].astype(np.float64)
    y_train = y_all_full[valid_train].to_numpy(dtype=np.float64)

    X_test = X_all[valid_test].astype(np.float64)
    y_test = y_all_full[valid_test].to_numpy(dtype=np.float64)

    return X_train, y_train, X_test, y_test

def gaussmf(
    x: Union[float, np.ndarray],
    c: float,
    s: float
) -> Union[float, np.ndarray]:

    """
        Description:
            Calcula o grau de pertinÃªncia usando uma funÃ§Ã£o de pertinÃªncia gaussiana.

        Args:
            x (float | np.ndarray): Valor(es) de entrada para calcular o grau de pertinÃªncia.
            c (float): Centro da funÃ§Ã£o gaussiana.
            s (float): Desvio padrÃ£o da funÃ§Ã£o gaussiana.

        Return:
            float | np.ndarray: Grau de pertinÃªncia calculado para o(s) valor(es) de entrada.
    """

    aux = (x - c)/s  # Calcula a distÃ¢ncia normalizada do valor ao centro
    z = np.exp(-(aux*aux)/2)  # Aplica a fÃ³rmula da funÃ§Ã£o gaussiana

    return z

def gaussgranules(
    x: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray
) -> np.ndarray:

    """
        Description:
            Calcula as ativaÃ§Ãµes de mÃºltiplas regras fuzzy usando funÃ§Ãµes gaussianas.

        Args:
            x (np.ndarray): Vetor de entrada de dimensÃ£o d.
            centers (np.ndarray): Array de formato (N, d) com os centros das N regras.
            sigmas (np.ndarray): Array de formato (N, d) com os desvios padrÃ£o das N regras.

        Return:
            np.ndarray: Array de formato (N,) contendo o grau de ativaÃ§Ã£o para cada uma das N regras.
    """

    # Um centro c = (c_{1},c_{2},...,c_{n}) e um desvio padrÃ£o s = (s_{1},s_{2},...,s_{n}) definem uma regra fuzzy (chamemos r).
    # Quando aplicamos gaussmf(x,c,s) para x = (x_{1},x_{2},...,x_{n}) obtemos o grau de pertencimento do vetor x Ã  regra r.

    N, d = centers.shape  # N Ã© o nÃºmero de regras, d Ã© a dimensÃ£o representada pelo delay
    activations = []  # Lista para armazenar as ativaÃ§Ãµes de cada regra

    for i in range(N):
        mu = 1.0  # Inicializa o grau de pertinÃªncia como 1
        for j in range(d):
            # Multiplica o grau de pertinÃªncia pela ativaÃ§Ã£o da j-Ã©sima dimensÃ£o
            mu *= gaussmf(x[j], centers[i, j], sigmas[i, j]) # Multiplica o grau de pertinÃªncia da dimensÃ£o j de x Ã  regra i
        activations.append(mu)  # Adiciona o resultado Ã  lista de ativaÃ§Ãµes

    return np.array(activations)  # Converte a lista para um array numpy
