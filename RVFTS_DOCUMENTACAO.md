## 1. Parametros de controle

Os parametros de controle estao distribuidos entre o nucleo do modelo e o script de integracao.

### 1.1. Parametros centrais de `rvfts_forecast`

- `h`: horizonte da previsao. Deve ser inteiro maior ou igual a 1.
- `ew`: tamanho da janela inicial de estimacao fuzzy. Controla quanto historico minimo o modelo precisa antes de iniciar a fase fuzzy.
- `cs`: tamanho da janela de cross-validation rolante. Controla quantos erros passados sao usados para escolher `(c, rho)`.
- `nclus_grid`: grade de numeros de clusters fuzzy candidatos.
- `rho_grid`: grade de valores de suavizacao EWMA candidatos.
- `window`: forma da janela da etapa fuzzy. Aceita `rolling` ou `expanding`.
- `m`: expoente de fuzzificacao do FCM. No codigo, o default e `2.0`.
- `cv_decay`: fator de decaimento exponencial usado para ponderar os erros na cross-validation.
- `fcm_seed`: semente pseudoaleatoria da inicializacao do Fuzzy C-Means.
- `verbose`: controla a exibicao de progresso em tela.

### 1.2. Efeito pratico dos parametros principais

- `h` altera o alinhamento temporal do alvo e tambem desloca o instante em que cada observacao se torna utilizavel.
- `ew` maior torna o modelo mais exigente em historico e tende a atrasar o inicio das previsoes.
- `cs` maior torna a escolha de hiperparametros mais conservadora, mas tambem aumenta o periodo inicial sem previsao final.
- `nclus_grid` maior amplia a busca de estruturas fuzzy, mas aumenta o custo computacional.
- `rho_grid` maior amplia a busca do nivel de suavizacao EWMA, tambem com maior custo computacional.
- `window="rolling"` usa apenas a janela mais recente; `window="expanding"` usa todo o historico disponivel desde o inicio.
- `cv_decay` proximo de `1` reduz a diferenca de peso entre erros antigos e recentes; valores menores enfatizam observacoes mais recentes.
- `fcm_seed` afeta a inicializacao do FCM e ajuda na reprodutibilidade.

### 1.3. Parametros expostos na integracao

O script `integracao_rvfts.py` expoe diretamente:

- `symbols`
- `horizons`
- `ew`
- `cs`
- `nclus_grid`
- `rho_grid`
- `window`
- `fcm_seed`
- `verbose`

Os defaults da integracao sao:

- `RVFTS_EW = 756`
- `RVFTS_CS = 252`
- `RVFTS_NCLUS = [2, 3, 4, 6, 9]`
- `RVFTS_RHO = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]`
- `RVFTS_WINDOW = "rolling"`
- `RVFTS_FCM_SEED = 42`

### 1.4. Parametros nao expostos pela integracao

Embora existam em `rvfts_forecast`, os seguintes parametros nao sao passados explicitamente por `integracao_rvfts.py`:

- `m`
- `cv_decay`

No fluxo atual, isso significa que ambos ficam nos defaults definidos dentro do proprio `rvfts_model.py`:

- `m = 2.0`
- `cv_decay = 0.99`

## 2. Restricoes e pre-condicoes operacionais

Para o RV-FTS rodar corretamente neste projeto, algumas condicoes precisam ser satisfeitas:

- `h >= 1`
- `window` deve ser `rolling` ou `expanding`
- a serie completa precisa ter comprimento minimo `T >= ew + cs + h`
- os DataFrames precisam conter `Target`, ou `RV` apenas no caso `h=1`
- a serie precisa estar ordenada no tempo

Tambem e importante notar:

- o modelo aceita `NaN` na serie, mas a disponibilidade efetiva de previsoes depende da quantidade de pontos validos em cada janela;
- na fase de FCM, janelas com dados insuficientes nao geram valor fuzzificado;
- na cross-validation, uma combinacao candidata so entra na disputa se tiver pelo menos 10 observacoes validas na janela de avaliacao.

## 3. Como o modelo funciona

O RV-FTS foi implementado em tres fases.

### 3.1. Fase 1: Fuzzy C-Means em janela rolante

Para cada numero de clusters `c` no grid configurado, o modelo:

- toma uma janela temporal da serie alvo;
- executa `fcm_1d`, uma versao unidimensional de Fuzzy C-Means;
- obtem os centroides fuzzy;
- usa a pertinencia do ultimo ponto observavel da janela para calcular uma quantidade fuzzificada.

Em termos conceituais:

`v_tilde[t-1] = soma_j( pertinencia_j * centroide_j )`

A implementacao usa, por padrao:

- `ew = 756`
- `window = "rolling"`
- `m = 2.0`

O grid de clusters padrao e:

- `nclus_grid = [2, 3, 4, 6, 9]`

### 3.2. Fase 2: previsao por EWMA

Depois de gerar a serie fuzzificada para cada `c`, o modelo monta previsoes candidatas usando uma recursao EWMA:

`pred[t] = (1 - rho) * v_tilde[t-1] + rho * pred[t-1]`

O grid padrao de `rho` e:

- `rho_grid = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]`

Assim, para cada combinacao `(c, rho)`, o codigo gera uma trajetoria completa de previsoes candidatas.

### 3.3. Fase 3: cross-validation rolante

As previsoes exportadas nao sao escolhidas com um unico `(c, rho)` fixo para toda a amostra. O script seleciona os hiperparametros dinamicamente ao longo do tempo.

Para cada instante `t`, ele:

- olha para uma janela de cross-validation de tamanho `cs`;
- compara os erros das combinacoes candidatas;
- aplica pesos exponencialmente decrescentes;
- escolhe o par `(c, rho)` com menor perda ponderada.

Os parametros padrao sao:

- `cs = 252`
- `cv_decay = 0.99`

So depois dessa etapa a previsao final `RV-FTS[t]` e gravada.

## 4. Como treino e teste sao usados

O script de integracao nao cria um novo split de treino e teste. Ele usa exatamente os arquivos que o notebook ja salvou como:

- treino in-sample
- teste out-of-sample

O fluxo e o seguinte:

1. Carrega `df_train` e `df_test` do Fuzzy HAR.
2. Faz merge das previsoes dos modelos HAR no `train` e no `test`.
3. Concatena a serie alvo de treino e teste em uma serie temporal unica.
4. Roda o RV-FTS nessa serie completa, sem embaralhar datas e sem usar futuro.
5. Divide a previsao final de volta entre os conjuntos: as primeiras `n_train` linhas retornam para `df_train` e as linhas seguintes retornam para `df_test`.

Isso preserva o corte temporal original e evita look-ahead.

Na pratica, o comportamento e de pseudo-out-of-sample cronologico:

- no inicio do teste, o modelo usa apenas o historico anterior disponivel;
- ao longo do teste, ele pode usar observacoes passadas do proprio teste que ja se tornaram observaveis;
- ele nunca usa observacoes futuras para prever a linha atual.

## 5. Por que algumas datas do teste ficam sem previsao

Mesmo concatenando treino e teste, o RV-FTS nao consegue gerar previsoes desde a primeira linha da amostra completa. Ele precisa de historico minimo para:

- formar a janela fuzzy inicial;
- rodar a recursao EWMA;
- acumular erros suficientes para a cross-validation rolante.

Por isso, a previsao final so comeca a existir a partir de:

`first_cv = ew + cs + h - 1`

Se o treino terminar antes desse indice, o inicio do conjunto de teste ficara com `NaN` na coluna `RV-FTS`.

Formula pratica:

`NaNs no inicio do teste = max(0, ew + cs + h - 1 - n_train)`

Esses vazios sao esperados pelo desenho atual do algoritmo. Eles nao indicam erro de merge nem falha de exportacao.

