# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Ponte rpy2 para o clustering Gustafson-Kessel (roda em R).

A celula %%R do notebook definia gk_clustering no ambiente global do R.
Aqui o mesmo codigo R e definido UMA vez, no import deste modulo, via r(_GK_R_SRC),
preservando o set.seed(123) e o corpo R byte a byte. run_gk_clustering e verbatim.'''
import numpy as np
import pandas as pd
from typing import Tuple
from rpy2.robjects import conversion, globalenv, pandas2ri, r
from rpy2.robjects.conversion import localconverter

_GK_R_SRC = r'''
#' @title ClusterizaÃ§Ã£o Fuzzy com Gustafson-Kessel
#' @description
#' Executa o algoritmo de clusterizaÃ§Ã£o fuzzy Gustafsonâ€“Kessel (GK) com inicializaÃ§Ã£o por k-means++ e pertinÃªncias fuzzy aleatÃ³rias.
#' A funÃ§Ã£o tambÃ©m calcula, manualmente, as matrizes de dispersÃ£o (\eqn{\Sigma_j}) e os desvios padrÃ£o por dimensÃ£o para cada cluster.
#'
#' @param x Um data.frame ou matriz numÃ©rica com n amostras (linhas) e d variÃ¡veis (colunas).
#' @param k Um inteiro representando o nÃºmero de clusters (ou regras fuzzy).
#'
#' @return Uma lista com trÃªs elementos:
#' \describe{
#'   \item{\code{centros}}{Matriz \code{k x d} contendo os centros finais de cada cluster.}
#'   \item{\code{pertinencias}}{Matriz \code{n x k} com os graus de pertinÃªncia fuzzy das amostras aos clusters.}
#'   \item{\code{desvios}}{Lista de \code{k} matrizes \code{d x d}, cada uma representando a matriz de dispersÃ£o ponderada (\eqn{\Sigma_j}) de um cluster.}
#' }
#'
#' @details
#' A funÃ§Ã£o usa:
#' \itemize{
#'   \item \code{ppclust::gk} para a execuÃ§Ã£o do algoritmo GK;
#'   \item \code{inaparc::kmpp} para inicializaÃ§Ã£o dos centros;
#'   \item \code{inaparc::imembrand} para inicializaÃ§Ã£o da matriz de pertinÃªncia.
#' }
#' O expoente de fuzzificaÃ§Ã£o \code{m} Ã© fixado em 2. As matrizes de dispersÃ£o sÃ£o calculadas usando os pesos fuzzy elevados a \code{m}, conforme a definiÃ§Ã£o clÃ¡ssica.
#'

gk_clustering <- function(x, k) {
  # Carrega o pacote 'ppclust', que fornece implementaÃ§Ãµes de algoritmos de clusterizaÃ§Ã£o fuzzy,
  # como Fuzzy C-Means (FCM), Gustafsonâ€“Kessel (GK), entre outros.
  library(ppclust)

  # Carrega o pacote 'inaparc', que provÃª mÃ©todos de inicializaÃ§Ã£o de centros e pertinÃªncias,
  # como k-means++ (kmpp) e inicializaÃ§Ã£o fuzzy aleatÃ³ria (imembrand).
  library(inaparc)

  # Garante que k seja um escalar inteiro vÃ¡lido
  k <- as.integer(as.numeric(k[1]))

  # ----------- InicializaÃ§Ã£o dos clusters -----------

  # Semente para reprodutibilidade de resultados
  set.seed(123)

  # Inicializa os centros dos clusters com o algoritmo k-means++.
  # Retorna uma lista com vÃ¡rios componentes; o '$v' acessa a matriz dos centros inicializados (k x d).
  v <- kmpp(x, k = k)$v

  # Inicializa a matriz de pertinÃªncia fuzzy (n x k), onde cada linha soma 1.
  # Essa matriz define o grau de associaÃ§Ã£o inicial de cada amostra a cada cluster.
  u <- imembrand(nrow(x), k = k)$u

  # ----------- ExecuÃ§Ã£o do algoritmo Gustafson-Kessel -----------

  # Executa o algoritmo GK (fuzzy clustering com mÃ©tricas de covariÃ¢ncia adaptativas).
  # Argumentos principais:
  # - x: conjunto de dados
  # - centers: centros inicializados (matriz k x d)
  # - memberships: matriz de pertinÃªncia inicial (n x k)
  # - m: expoente de fuzzificaÃ§Ã£o (m > 1; m = 2 Ã© o padrÃ£o mais usado)
  # - dmetric: mÃ©trica de distÃ¢ncia ("sqeuclidean" = euclidiana ao quadrado)
  # - iter.max: nÃºmero mÃ¡ximo de iteraÃ§Ãµes
  # - con.val: critÃ©rio de convergÃªncia (parar se mudanÃ§a entre iteraÃ§Ãµes < 1e-9)
  gk.res <- gk(x, centers = v, memberships = u, m = 2,
              dmetric = "sqeuclidean", iter.max = 1000, con.val = 1e-9)

  # Exporta a matriz de centros finais (k x d) para uso posterior
  centros <- gk.res$v

  # Exporta a matriz final de pertinÃªncia fuzzy (n x k)
  pertinencias <- gk.res$u

  # ----------- CÃ¡lculo manual das matrizes de dispersÃ£o (Î£_j) e desvios padrÃ£o -----------

  # Converte os dados para uma matriz numÃ©rica (n x d), onde:
  # n = nÃºmero de amostras (linhas), d = nÃºmero de variÃ¡veis (lags ou dimensÃµes)
  X <- as.matrix(x)

  # Recupera a matriz de pertinÃªncia fuzzy (n x k), onde:
  # U[i, j] representa o grau de associaÃ§Ã£o da amostra i ao cluster j
  U <- gk.res$u

  # Recupera a matriz de centros dos clusters (k x d), onde:
  # V[j, ] contÃ©m as coordenadas do centro do cluster j
  V <- gk.res$v

  # Expoente de fuzzificaÃ§Ã£o usado tanto no algoritmo quanto neste cÃ¡lculo
  m <- 2

  # Define as dimensÃµes principais do problema
  n <- nrow(X)   # nÃºmero de amostras
  k <- ncol(U)   # nÃºmero de clusters
  d <- ncol(X)   # nÃºmero de variÃ¡veis (dimensÃµes)

  # Inicializa uma lista que armazenarÃ¡ a matriz de dispersÃ£o Î£_j de cada cluster j
  Sigma_list <- vector("list", k)

  # Loop principal: calcula Î£_j para cada cluster j
  for (j in 1:k) {

    # Inicializa o numerador e denominador da fÃ³rmula de Î£_j
    # Î£_j = ( âˆ‘_{i=1}^{n} (u_ij)^m * (x_i - v_j)(x_i - v_j)^T ) / âˆ‘_{i=1}^{n} (u_ij)^m
    num <- matrix(0, d, d)  # acumulador da soma ponderada das covariÃ¢ncias
    denom <- 0              # acumulador da soma dos pesos (u_ij^m)

    # Loop interno: percorre todas as amostras i
    for (i in 1:n) {

      # Calcula o grau de pertinÃªncia fuzzy elevado ao expoente m
      u_ij_m <- U[i, j]^m

      # Calcula o vetor de diferenÃ§a entre a amostra i e o centro do cluster j:
      # diff = x_i - v_j (vetor coluna d x 1)
      diff <- matrix(X[i, ] - V[j, ], ncol = 1)

      # Atualiza o numerador com a contribuiÃ§Ã£o da amostra i:
      # u_ij^m * (diff * diffáµ—) â†’ matriz d x d (produto externo)
      num <- num + u_ij_m * (diff %*% t(diff))

      # Atualiza o denominador com o peso u_ij^m
      denom <- denom + u_ij_m
    }

    # Finaliza o cÃ¡lculo da matriz de dispersÃ£o Î£_j:
    # divide o somatÃ³rio ponderado das covariÃ¢ncias pelo total de pesos
    Sigma_list[[j]] <- num / denom
  }

  # Exporta os desvios padrÃ£o com o nome 'desvios'
  desvios <- Sigma_list

  return(list(
    centros = centros,
    pertinencias = pertinencias,
    desvios = desvios
  ))

}
'''

# Define gk_clustering no ambiente R (equivalente a executar a celula %%R uma vez).
r(_GK_R_SRC)

def run_gk_clustering(
    X_train: pd.DataFrame,
    rules_number: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:

    """
      Description:
          Executa a funÃ§Ã£o R 'gk_clustering' para realizar clusterizaÃ§Ã£o fuzzy com o algoritmo
          Gustafsonâ€“Kessel. Retorna os centros dos clusters, as pertinÃªncias fuzzy e os desvios
          padrÃ£o por dimensÃ£o de cada cluster, extraÃ­dos das matrizes de covariÃ¢ncia.

      Args:
          X_train (pd.DataFrame): Conjunto de dados com colunas defasadas (ex: 'lag1', 'lag2', ...).
          rules_number (int): NÃºmero de clusters ou regras fuzzy (k).

      Return:
          Tuple contendo:
              - centers_df (pd.DataFrame): Matriz (k x d) com os centros finais dos clusters.
              - memberships_df (pd.DataFrame): Matriz (n x k) de pertinÃªncia fuzzy das amostras.
              - sigmas (np.ndarray): Matriz (k x d) com os desvios padrÃ£o por dimensÃ£o para cada cluster.
    """

    # Garante que Xtrain seja um DataFrame, mesmo que seja um array
    X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train

    # Se quiser, nomeie automaticamente as colunas como lag1, lag2, ..., lagN
    if X_train.columns.dtype == 'int':
        X_train.columns = [f"lag{i+1}" for i in range(X_train.shape[1])]

    # ConversÃ£o pandas -> R sem usar activate/deactivate (API antiga/depreciada)
    with localconverter(conversion.get_conversion() + pandas2ri.converter):
        globalenv["x"] = conversion.py2rpy(X_train)

    # Envia o nÃºmero de clusters/regras para o R como vetor inteiro, com o nome 'rules_number'
    globalenv["k"] = int(rules_number)

    # Executa a funÃ§Ã£o R declarada anteriormente
    r_result = r("gk_clustering(x, k[[1]])")

    centers = r_result.rx2("centros")
    memberships = r_result.rx2("pertinencias")
    desvios = r_result.rx2("desvios")

    n_rules = int(rules_number)
    n_features = int(X_train.shape[1])

    centers_np = np.asarray(centers, dtype=float)
    if centers_np.ndim == 1:
        centers_np = centers_np.reshape(n_rules, n_features)
    elif centers_np.ndim == 2 and centers_np.shape == (n_features, n_rules):
        centers_np = centers_np.T

    memberships_np = np.asarray(memberships, dtype=float)
    if memberships_np.ndim == 1:
        memberships_np = memberships_np.reshape(-1, n_rules)

    centers_df = pd.DataFrame(centers_np, columns=X_train.columns)
    memberships_df = pd.DataFrame(memberships_np)

    # Converte cada matriz R da lista para um array NumPy
    # Cada elemento da lista representa a matriz de covariÃ¢ncia (d x d) de um cluster
    sigmas = [np.array(dev) for dev in desvios]

    # Extrai os desvios padrÃ£o (raiz quadrada das variÃ¢ncias) de cada matriz de covariÃ¢ncia
    # Isso gera uma lista de vetores 1D, onde cada vetor contÃ©m os desvios padrÃ£o por dimensÃ£o
    sigmas = [np.sqrt(np.diag(S)) for S in sigmas]

    # Concatena a lista de vetores em um array 2D (n_clusters x n_features)
    # Isso facilita o acesso via sigmas[i, j] para a dimensÃ£o j do cluster i
    sigmas = np.array(sigmas)  # shape: (n_clusters, n_features)

    return centers_df, memberships_df, sigmas
