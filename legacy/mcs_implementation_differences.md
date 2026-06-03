# Diferenças entre as implementações do MCS

## O que é o Model Confidence Set (MCS)?

O MCS (Hansen, Lunde & Nason, 2011) é um teste estatístico que elimina iterativamente
o modelo com maior perda esperada até que não seja mais possível rejeitar a hipótese
de que todos os modelos restantes são equivalentes. O conjunto que sobra é o
**Superior Set de Modelos** ao nível de confiança escolhido (aqui: 75%).

---

## O que é o `block_size`?

Os dados de RV são séries temporais — observações próximas no tempo são correlacionadas.
Um bootstrap padrão (reamostrar pontos individuais) destrói essa estrutura.

O **bootstrap estacionário** (Politis & Romano, 1994) reamostrar *blocos* de
observações consecutivas, preservando a dependência temporal. O `block_size`
define o comprimento médio desses blocos.

**Por que importa:**
- Bloco **pequeno** → bootstrap gera amostras muito variáveis → distribuição nula
  mais espalhada → valor crítico alto → difícil rejeitar H0 → MCS **maior** (inclui mais modelos).
- Bloco **grande** → distribuição nula mais concentrada → valor crítico baixo →
  fácil rejeitar H0 → MCS **menor** (inclui menos modelos).

Não existe um valor universalmente correto; para dados financeiros diários a
literatura usa tipicamente entre 5 e 20.

---

## Comparação das três implementações

| Aspecto | `arch` (notebook) | `arch` (pipeline) | R `MCSprocedure` |
|---|---|---|---|
| **Biblioteca** | `arch.bootstrap.MCS` | `arch.bootstrap.MCS` | pacote R `MCS` |
| **Estatística** | T_max | T_max | T_max |
| **Bootstrap** | Estacionário | Estacionário | Estacionário |
| **`block_size` padrão** | `sqrt(T)` ≈ 39 (T=1517) | `sqrt(T)` ≈ 39 | `max(ar.order)` ≈ 1–5 |
| **Réplicas (B)** | 10 000 | 10 000 | 10 000 |
| **Seed** | 123 | 123 | 123 |
| **Modelos testados** | Fuzzy HAR + 5 HARs | Fuzzy HAR + 5 HARs + RV-FTS | Fuzzy HAR + 5 HARs + RV-FTS |
| **Filtro previsões ≤ 0** | Apenas "Fuzzy HAR" | Todos os modelos | Todos os modelos |
| **Parada da eliminação** | 1 modelo restante | 1 modelo restante | `min.k=1` (forçado; padrão=3) |

---

## Consequência prática observada (BTC, t+1, QLIKE, split 40/60)

| Implementação | Modelos incluídos no MCS (de 6 ou 7) |
|---|---|
| arch / notebook filter / sem RV-FTS | 4 / 6 |
| arch / notebook filter / com RV-FTS | 5 / 7 |
| arch / pipeline filter / sem RV-FTS | 2 / 6 |
| arch / pipeline filter / com RV-FTS | 2 / 7 |
| R MCSprocedure / sem RV-FTS | 6 / 6 |
| R MCSprocedure / com RV-FTS | 7 / 7 |

O R inclui sistematicamente mais modelos porque usa blocos menores
(`ar.order` ≈ 1–3 vs `sqrt(T)` ≈ 39), gerando valores críticos mais altos.

---

## Qual usar?

| Objetivo | Recomendação |
|---|---|
| Replicar exatamente resultados de papers com R | R `MCSprocedure` |
| Pipeline Python sem dependência de R | `arch.bootstrap.MCS` |
| Comparação justa entre as duas | Fixar `block_size` igual em ambas (`--block-size 10`) |

Para publicação, o `block_size` deve ser **reportado e justificado** independentemente
da biblioteca escolhida.
