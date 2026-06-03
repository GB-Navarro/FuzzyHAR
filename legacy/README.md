# legacy/

Scripts **antigos/supersedidos**, mantidos apenas para referencia historica.
Nao fazem parte do pipeline atual e **nao devem ser executados** para gerar
resultados — usar a versao em `fuzzyhar/` (ou os scripts oficiais na raiz).

| Arquivo | Substituido por |
|---|---|
| `integracao_rvfts.py` | `fuzzyhar/models/rvfts_pipeline.py` + `fuzzyhar/models/rvfts.py` |
| `mcs_rvfts_results.py` | `fuzzyhar/models/rvfts_pipeline.py` (MCS) + `fuzzyhar/evaluation/mcs.py` |
| `mcs_implementation_differences.md` | nota de diagnostico antiga (arch vs R) |
| `rvfts_model.py` | implementacao alternativa do RV-FTS, usada **apenas** por `integracao_rvfts.py` (ambos arquivados aqui). O pipeline vivo usa `fuzzyhar/models/rvfts_pipeline.py`. |

> Os scripts aqui sao **congelados** e nao sao executados; apos a reorganizacao,
> alguns imports internos (ex.: `from rvfts_model import ...`) so resolvem se
> rodados de dentro de `legacy/`.

Mantidos **na raiz** (ainda em uso): `mcs_comparison_test.py`, `FuzzyHAR.ipynb`.
