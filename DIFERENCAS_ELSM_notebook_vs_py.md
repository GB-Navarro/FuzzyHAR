# e-LSM: notebook vs. `elsm_model.py`

Comparação entre a implementação do e-LSM no notebook
`Cópia_de_Level_Set_Algorithms.ipynb` (célula 74, *"controlled e-RLSM"*) e no
módulo `elsm_model.py` usado pelo pipeline (`integracao_elsm.py`).

## Resumo

**O motor (algoritmo) é idêntico; só a camada em volta muda.** A equivalência
numérica do núcleo é garantida por `test_elsm_identical.py` (`max|Δ| ≤ 1e-12`).

## O que é IDÊNTICO — o motor

- Helpers `safe_sum`, `build_d_from_tau`, `gaussian_tau_vec`, `dlsm_fit`.
- Classes `EvolvingRules` e `eRLSM` (incluindo todo o `step()`: gates de criação,
  freios suaves, update RLS, merge periódico).
- Defaults do `ELSMConfig` == literais da célula 74 (TAU_CREATE=0.94,
  SIGMA_MAX=0.12, ALPHA=1e4, BETA=0.985, etc.).
- **Única mudança de motor:** hiperparâmetros que eram **globais de módulo** no
  notebook passam a campos de um `ELSMConfig` injetado (*plumbing*, não muda a
  matemática). Há ainda um clamp defensivo `n_rules = min(n_rules, len(Xtr))` em
  `kmeans_init_rules`, numericamente inócuo nos tamanhos reais.

## O que é DIFERENTE — a camada de aplicação

| Aspecto | Notebook (célula 74) | `elsm_model.py` (integração) |
|---------|----------------------|------------------------------|
| **Dados** | 5 séries sintéticas **Mackey-Glass** já normalizadas | **RV de cripto** → features **HAR** (daily/weekly/monthly) |
| **Normalização** | dados já normalizados; MSE em escala normalizada, sem desnormalizar | min-max com **stats de treino**; previsões **desnormalizadas** de volta à escala de RV |
| **Hiperparâmetros** | **um** conjunto fixo, igual p/ os 5 datasets | **grid search** por símbolo×horizonte (RMSE prequencial no treino) |
| **`clip_yhat`** | `True` (capa yhat em [0,1]) | produção força **`False`** (não capa picos de volatilidade) |
| **Saída** | só calcula MSE e imprime diagnósticos | grava colunas `e-LSM prediction` / `e-LSM OOS prediction`, alinhamento posicional estilo Fuzzy HAR (célula 34) |
| **Estrutura** | globais de módulo | dataclass `ELSMConfig` |
| **Protocolo treino→teste** | evolui no treino, depois no teste (previsão antes do update) | **idêntico** |

## Por que isso importa

O núcleo se comporta igual, **mas as previsões finais não são as que o notebook
produziria**, por três motivos materiais:

1. **Grid search** pode escolher config fora do notebook (ex.: `sigma_max=0.25`,
   `tau_create=0.5`, `n_init_rules=6`).
2. **`clip_yhat=False`** vs `True` — muda a saída quando a previsão normalizada
   sairia de [0,1] (relevante em picos de volatilidade).
3. **Normalização/desnormalização** aplicada ao RV (no notebook os dados já
   chegavam normalizados e o resultado ficava em escala normalizada).

**Conclusão:** mesmo algoritmo (controlled e-RLSM); aplicação diferente por
design — o notebook é um *benchmark* em Mackey-Glass; o `elsm_model.py` é a
integração no pipeline de previsão de volatilidade de cripto.
