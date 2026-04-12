# FuzzyHAR - Guia Rápido de Uso

## O que precisa estar instalado

### Python
- Python 3.10 ou 3.11 (recomendado)
- Jupyter Notebook ou VS Code com suporte a notebooks

Pacotes Python usados no notebook:
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- statsmodels
- rpy2
- openpyxl

Instalação sugerida:

```bash
pip install -U numpy pandas matplotlib scipy scikit-learn statsmodels rpy2 openpyxl
```

### R
- R instalado no sistema (com Rscript disponível)

Pacotes R usados:
- inaparc
- ppclust
- MCS

Observação:
- O notebook já tenta instalar inaparc e ppclust automaticamente.
- O pacote MCS é instalado quando a etapa de MCS é executada.

## Como executar (ordem rápida)

Execute o notebook FuzzyHAR.ipynb nesta ordem:

1. Requirements & Imports
2. Auxiliary Functions
3. Parameters
4. Exploratory Data Analysis (opcional)
5. Fuzzy HAR Train & Test
6. HAR Models Train & Test
7. Results -> Merge
8. Results -> MSE & QLIKE
9. Results -> Graficos
10. Results -> MCS

## Parâmetros que você pode alterar

## Bloco Parameters
- SYMBOLS: lista de ativos a processar (ex.: ADAUSDT, BNBUSDT)
- rules_grid: grade de regras fuzzy testadas (ex.: [2, 3, 4, 5, 6])
- week_len: janela semanal do HAR
- month_len: janela mensal do HAR
- lamb: fator de esquecimento do ajuste online
- alfa: escala inicial da matriz P no treino
- ze: largura do kernel usada no ALSM-HAR


## Saídas geradas

O notebook gera:
- previsões in-sample e out-of-sample
- arquivos mergeados por ativo/horizonte
- tabelas de métricas (MSE e QLIKE)
- tabelas MCS
- gráficos finais de comparação