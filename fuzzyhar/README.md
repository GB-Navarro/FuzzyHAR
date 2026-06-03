# fuzzyhar/

Versao **modular e desacoplada** do pipeline FuzzyHAR. Todo o codigo novo vive
aqui; o codigo antigo/supersedido foi para [`../legacy/`](../legacy/).

> **Garantia de reprodutibilidade.** O corpo de todas as funcoes numericas foi
> **copiado verbatim** das celulas do `FuzzyHAR.ipynb` (via script de geracao).
> A refatoracao e estrutural: muda *onde* o codigo mora e *como* ele e importado,
> nunca *o que* ele calcula. O clustering Gustafson-Kessel continua rodando em R
> via rpy2, com o mesmo `set.seed(123)` (ver `models/r_gk.py`).
>
> Mesmo assim, a igualdade dos resultados **deve ser verificada** com o golden
> master (ver "Verificacao" abaixo) — esse e o unico jeito de provar a igualdade
> com dados reais.

## Estrutura

```
fuzzyhar/
├── config.py                 # FONTE UNICA: SPLIT, paths, SYMBOLS, hiperparametros
├── features.py               # gaussmf, gaussgranules, make_har_features, ... (puro)
├── metrics/
│   ├── base.py               # Protocol Metric
│   └── implementations.py    # MSE/QLIKE/RMSE/MAE + variantes distintas (ver R2)
├── models/
│   ├── base.py               # PipelineStep, ModelSpec, Protocol Forecaster
│   ├── r_gk.py               # ponte rpy2: define gk_clustering em R (set.seed=123)
│   ├── fuzzy_har.py          # Fuzzy HAR (verbatim) + run_all_fuzzy()
│   ├── har.py                # HAR e variantes (verbatim) + run_all_har()
│   ├── rvfts_pipeline.py     # RV-FTS completo (movido p/ ca, INALTERADO)
│   └── rvfts.py              # adaptador: run_all_rvfts() / run_rvfts_all_splits()
├── evaluation/
│   ├── base.py               # Protocol ComparisonTest + TestResult
│   ├── metrics_tables.py     # tabelas MSE & QLIKE (verbatim) + run_all_metrics()
│   └── mcs.py                # Model Confidence Set (verbatim) + run_all_mcs()
├── pipeline/
│   ├── merge.py              # merge de previsoes (verbatim) + run_all_merge()
│   ├── plots.py              # graficos OOS (verbatim) + run_all_plots()
│   └── runner.py             # orquestra os passos na ORDEM do notebook
├── registry.py               # UNICO lugar para cadastrar modelo/metrica/teste
├── scripts/
│   ├── freeze_baseline.py    # congela snapshot dos resultados (golden master)
│   └── compare_baseline.py   # compara dois snapshots VALOR A VALOR
└── tests/                    # testes de propriedade + golden master
```

## Como rodar (working dir = pasta `FuzzyHAR/`)

```python
from fuzzyhar.pipeline.runner import run_pipeline, run_rvfts_all_splits
run_pipeline()                    # reproduz os 6 modelos do notebook (split atual)
run_pipeline(include_rvfts=True)  # tambem roda o RV-FTS (stage 2) no split atual
run_pipeline(only=["MCS"])        # so um passo

# RV-FTS nos 3 splits (= python -m fuzzyhar.models.rvfts_pipeline --splits 40_60 60_40 70_30):
run_rvfts_all_splits()
```

O RV-FTS continua tendo CLI proprio, agora dentro do pacote:

```bash
python -m fuzzyhar.models.rvfts_pipeline --splits 40_60 60_40 70_30
```

Para mudar o split do pipeline do notebook, edite `SPLIT` em `config.py` (atual: `"70_30"`).

## Como adicionar (o desacoplamento na pratica)

* **Novo modelo:** crie `models/seu_modelo.py` com uma funcao `run_all_seu()` e
  registre um `PipelineStep` + `ModelSpec` em `registry.py`.
* **Nova metrica:** implemente a classe em `metrics/implementations.py` (com
  `aggregate` e `pointwise`) e adicione a `METRICS` em `registry.py`.
* **Novo teste:** adicione um `MCSSpec` (ou um novo tipo) em `TESTS`.

Em todos os casos, **so o `registry.py`** (e o modulo do componente) muda.

## Verificacao (OBRIGATORIA para confiar na igualdade)

```bash
# 1) com o codigo ATUAL (notebook), congele o baseline:
python -m fuzzyhar.scripts.freeze_baseline --label antes

# 2) rode a versao refatorada e regenere Others/ e Results/:
python -c "from fuzzyhar.pipeline.runner import run_pipeline; run_pipeline()"

# 3) congele de novo e compare valor a valor:
python -m fuzzyhar.scripts.freeze_baseline --label depois
python -m fuzzyhar.scripts.compare_baseline antes depois   # deve dizer "OK"

# testes de propriedade (rapidos, sem R):
pytest fuzzyhar/tests -v
```

A comparacao e por **valor** (tolerancia `rtol=1e-9`), nao por bytes de Excel
(que sempre diferem por timestamps embutidos).

> **Atencao (R):** os resultados so sao comparaveis no MESMO ambiente. Como o GK
> roda em R, fixe as versoes de `ppclust`/`inaparc` (e de `numpy`/`arch`) usadas
> ao congelar o baseline — salve `pip freeze` e `sessionInfo()` junto.
