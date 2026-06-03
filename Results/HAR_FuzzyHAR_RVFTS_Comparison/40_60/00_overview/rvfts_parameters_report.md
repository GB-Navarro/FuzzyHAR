# Relatório de Parâmetros do RV-FTS

Este relatório documenta, para cada combinação **moeda × horizonte**, os hiperparâmetros usados pelo modelo `RV-FTS` (Iqbal & Lyócsa, 2026) e os valores que a cross-validação rolante selecionou ao longo do tempo.

## Configuração global

| Parâmetro | Valor |
| --- | --- |
| EW (janela inicial de estimação) | 756 |
| CS (janela de calibração / CV) | 252 |
| Grid de c (nº de clusters) | [2, 3, 4, 6, 9] |
| Grid de ρ (memória EWMA) | [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975] |
| Tipo de janela | rolling |
| Semente do FCM | 42 |
| Expoente m do FCM | 2 (paper) |
| Decaimento da CV | 0.99 por passo |
| Horizontes | [1, 7, 30] |
| Moedas | ['ADAUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETHUSDT', 'TRXUSDT', 'XLMUSDT', 'XRPUSDT'] |


> A cada t, a CV rolante seleciona o par (c, ρ) com menor MSE ponderado exponencialmente sobre os últimos `CS` erros; portanto cada moeda/horizonte tem uma **série temporal** de (c, ρ) ótimos. As tabelas abaixo resumem essa distribuição empírica.

## Horizonte t+1

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 190 | 1435 | 9 | 0.5 | Target |
| BNBUSDT | 245 | 1517 | 9 | 0.75 | Target |
| BTCUSDT | 245 | 1517 | 3 | 0.75 | Target |
| DOGEUSDT | 12 | 1168 | 9 | 0.5 | Target |
| ETHUSDT | 245 | 1517 | 9 | 0.75 | Target |
| TRXUSDT | 168 | 1402 | 9 | 0.9 | Target |
| XLMUSDT | 173 | 1408 | 9 | 0.75 | Target |
| XRPUSDT | 184 | 1424 | 9 | 0.25 | Target |


### ADA (ADAUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **190** em train, **1435** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 6 | 141 | 74.2% |
| 4 | 31 | 16.3% |
| 3 | 10 | 5.3% |
| 2 | 6 | 3.2% |
| 9 | 2 | 1.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 438 | 30.5% |
| 4 | 382 | 26.6% |
| 3 | 327 | 22.8% |
| 6 | 198 | 13.8% |
| 2 | 90 | 6.3% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 98 | 51.6% |
| 0.5 | 78 | 41.1% |
| 0.025 | 7 | 3.7% |
| 0.1 | 7 | 3.7% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.5 | 494 | 34.4% |
| 0.9 | 363 | 25.3% |
| 0.025 | 245 | 17.1% |
| 0.75 | 187 | 13.0% |
| 0.25 | 69 | 4.8% |
| 0.95 | 54 | 3.8% |
| 0.975 | 14 | 1.0% |
| 0.1 | 9 | 0.6% |


---

### BNB (BNBUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **245** em train, **1517** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 244 | 99.6% |
| 4 | 1 | 0.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 801 | 52.8% |
| 6 | 528 | 34.8% |
| 3 | 98 | 6.5% |
| 2 | 65 | 4.3% |
| 4 | 25 | 1.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 210 | 85.7% |
| 0.5 | 24 | 9.8% |
| 0.75 | 6 | 2.4% |
| 0.25 | 2 | 0.8% |
| 0.975 | 2 | 0.8% |
| 0.05 | 1 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 650 | 42.8% |
| 0.5 | 336 | 22.1% |
| 0.25 | 278 | 18.3% |
| 0.9 | 157 | 10.3% |
| 0.025 | 83 | 5.5% |
| 0.1 | 13 | 0.9% |


---

### BTC (BTCUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **245** em train, **1517** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 203 | 82.9% |
| 4 | 24 | 9.8% |
| 6 | 11 | 4.5% |
| 9 | 7 | 2.9% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 863 | 56.9% |
| 9 | 479 | 31.6% |
| 6 | 137 | 9.0% |
| 4 | 37 | 2.4% |
| 2 | 1 | 0.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 213 | 86.9% |
| 0.975 | 22 | 9.0% |
| 0.9 | 4 | 1.6% |
| 0.95 | 4 | 1.6% |
| 0.025 | 1 | 0.4% |
| 0.5 | 1 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 562 | 37.0% |
| 0.25 | 378 | 24.9% |
| 0.5 | 173 | 11.4% |
| 0.1 | 135 | 8.9% |
| 0.975 | 132 | 8.7% |
| 0.9 | 42 | 2.8% |
| 0.05 | 38 | 2.5% |
| 0.025 | 30 | 2.0% |
| 0.95 | 27 | 1.8% |


---

### DOGE (DOGEUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **12** em train, **1168** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 12 | 100.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 695 | 59.5% |
| 2 | 269 | 23.0% |
| 4 | 128 | 11.0% |
| 6 | 67 | 5.7% |
| 3 | 9 | 0.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 11 | 91.7% |
| 0.95 | 1 | 8.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.5 | 275 | 23.5% |
| 0.25 | 229 | 19.6% |
| 0.975 | 202 | 17.3% |
| 0.75 | 179 | 15.3% |
| 0.9 | 132 | 11.3% |
| 0.95 | 81 | 6.9% |
| 0.025 | 40 | 3.4% |
| 0.1 | 30 | 2.6% |


---

### ETH (ETHUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **245** em train, **1517** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 6 | 148 | 60.4% |
| 2 | 44 | 18.0% |
| 4 | 26 | 10.6% |
| 9 | 19 | 7.8% |
| 3 | 8 | 3.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 562 | 37.0% |
| 3 | 489 | 32.2% |
| 6 | 190 | 12.5% |
| 2 | 179 | 11.8% |
| 4 | 97 | 6.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 154 | 62.9% |
| 0.025 | 63 | 25.7% |
| 0.9 | 14 | 5.7% |
| 0.5 | 6 | 2.4% |
| 0.1 | 3 | 1.2% |
| 0.25 | 3 | 1.2% |
| 0.05 | 1 | 0.4% |
| 0.975 | 1 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 464 | 30.6% |
| 0.025 | 404 | 26.6% |
| 0.25 | 203 | 13.4% |
| 0.95 | 183 | 12.1% |
| 0.5 | 156 | 10.3% |
| 0.05 | 68 | 4.5% |
| 0.1 | 16 | 1.1% |
| 0.9 | 14 | 0.9% |
| 0.975 | 9 | 0.6% |


---

### TRX (TRXUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **168** em train, **1402** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 6 | 149 | 88.7% |
| 4 | 8 | 4.8% |
| 9 | 8 | 4.8% |
| 3 | 3 | 1.8% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 950 | 67.8% |
| 3 | 220 | 15.7% |
| 4 | 208 | 14.8% |
| 6 | 24 | 1.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 90 | 53.6% |
| 0.05 | 32 | 19.0% |
| 0.25 | 31 | 18.5% |
| 0.1 | 15 | 8.9% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 433 | 30.9% |
| 0.75 | 367 | 26.2% |
| 0.95 | 180 | 12.8% |
| 0.1 | 179 | 12.8% |
| 0.5 | 114 | 8.1% |
| 0.25 | 108 | 7.7% |
| 0.975 | 16 | 1.1% |
| 0.025 | 5 | 0.4% |


---

### XLM (XLMUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **173** em train, **1408** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 72 | 41.6% |
| 4 | 58 | 33.5% |
| 9 | 31 | 17.9% |
| 6 | 12 | 6.9% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 679 | 48.2% |
| 3 | 368 | 26.1% |
| 4 | 205 | 14.6% |
| 6 | 156 | 11.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 63 | 36.4% |
| 0.1 | 28 | 16.2% |
| 0.25 | 28 | 16.2% |
| 0.975 | 20 | 11.6% |
| 0.5 | 11 | 6.4% |
| 0.95 | 11 | 6.4% |
| 0.9 | 8 | 4.6% |
| 0.025 | 3 | 1.7% |
| 0.05 | 1 | 0.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 456 | 32.4% |
| 0.25 | 296 | 21.0% |
| 0.9 | 259 | 18.4% |
| 0.5 | 177 | 12.6% |
| 0.1 | 98 | 7.0% |
| 0.95 | 76 | 5.4% |
| 0.05 | 24 | 1.7% |
| 0.025 | 22 | 1.6% |


---

### XRP (XRPUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **184** em train, **1424** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 66 | 35.9% |
| 6 | 50 | 27.2% |
| 9 | 37 | 20.1% |
| 2 | 24 | 13.0% |
| 4 | 7 | 3.8% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 559 | 39.3% |
| 3 | 444 | 31.2% |
| 6 | 284 | 19.9% |
| 4 | 137 | 9.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 85 | 46.2% |
| 0.75 | 45 | 24.5% |
| 0.5 | 39 | 21.2% |
| 0.025 | 5 | 2.7% |
| 0.95 | 5 | 2.7% |
| 0.975 | 3 | 1.6% |
| 0.1 | 2 | 1.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 320 | 22.5% |
| 0.75 | 286 | 20.1% |
| 0.975 | 233 | 16.4% |
| 0.5 | 232 | 16.3% |
| 0.9 | 134 | 9.4% |
| 0.1 | 101 | 7.1% |
| 0.95 | 75 | 5.3% |
| 0.025 | 42 | 2.9% |
| 0.05 | 1 | 0.1% |


---

## Horizonte t+7

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 176 | 1431 | 2 | 0.975 | Target |
| BNBUSDT | 231 | 1513 | 9 | 0.975 | Target |
| BTCUSDT | 231 | 1513 | 9 | 0.975 | Target |
| DOGEUSDT | 0 | 1164 | 9 | 0.975 | Target |
| ETHUSDT | 231 | 1513 | 9 | 0.95 | Target |
| TRXUSDT | 154 | 1398 | 9 | 0.95 | Target |
| XLMUSDT | 158 | 1405 | 9 | 0.975 | Target |
| XRPUSDT | 169 | 1421 | 9 | 0.975 | Target |


### ADA (ADAUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **176** em train, **1431** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 113 | 64.2% |
| 3 | 32 | 18.2% |
| 6 | 22 | 12.5% |
| 4 | 9 | 5.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 508 | 35.5% |
| 2 | 504 | 35.2% |
| 4 | 240 | 16.8% |
| 6 | 151 | 10.6% |
| 3 | 28 | 2.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 132 | 75.0% |
| 0.025 | 35 | 19.9% |
| 0.9 | 7 | 4.0% |
| 0.75 | 1 | 0.6% |
| 0.95 | 1 | 0.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 545 | 38.1% |
| 0.95 | 423 | 29.6% |
| 0.025 | 352 | 24.6% |
| 0.9 | 102 | 7.1% |
| 0.75 | 4 | 0.3% |
| 0.25 | 2 | 0.1% |
| 0.5 | 2 | 0.1% |
| 0.1 | 1 | 0.1% |


---

### BNB (BNBUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **231** em train, **1513** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 166 | 71.9% |
| 3 | 62 | 26.8% |
| 4 | 3 | 1.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 730 | 48.2% |
| 4 | 302 | 20.0% |
| 3 | 197 | 13.0% |
| 6 | 148 | 9.8% |
| 2 | 136 | 9.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 133 | 57.6% |
| 0.95 | 79 | 34.2% |
| 0.9 | 12 | 5.2% |
| 0.025 | 4 | 1.7% |
| 0.75 | 3 | 1.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 570 | 37.7% |
| 0.025 | 417 | 27.6% |
| 0.95 | 291 | 19.2% |
| 0.9 | 81 | 5.4% |
| 0.5 | 47 | 3.1% |
| 0.75 | 45 | 3.0% |
| 0.05 | 43 | 2.8% |
| 0.1 | 11 | 0.7% |
| 0.25 | 8 | 0.5% |


---

### BTC (BTCUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **231** em train, **1513** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 198 | 85.7% |
| 4 | 22 | 9.5% |
| 6 | 8 | 3.5% |
| 9 | 3 | 1.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 640 | 42.3% |
| 3 | 340 | 22.5% |
| 2 | 311 | 20.6% |
| 6 | 165 | 10.9% |
| 4 | 57 | 3.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 209 | 90.5% |
| 0.9 | 11 | 4.8% |
| 0.025 | 5 | 2.2% |
| 0.95 | 3 | 1.3% |
| 0.75 | 2 | 0.9% |
| 0.5 | 1 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 955 | 63.1% |
| 0.025 | 160 | 10.6% |
| 0.9 | 157 | 10.4% |
| 0.95 | 155 | 10.2% |
| 0.75 | 64 | 4.2% |
| 0.1 | 8 | 0.5% |
| 0.5 | 6 | 0.4% |
| 0.25 | 5 | 0.3% |
| 0.05 | 3 | 0.2% |


---

### DOGE (DOGEUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **1164** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 466 | 40.0% |
| 3 | 311 | 26.7% |
| 2 | 261 | 22.4% |
| 4 | 117 | 10.1% |
| 6 | 9 | 0.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 440 | 37.8% |
| 0.95 | 376 | 32.3% |
| 0.75 | 221 | 19.0% |
| 0.9 | 104 | 8.9% |
| 0.025 | 20 | 1.7% |
| 0.5 | 2 | 0.2% |
| 0.1 | 1 | 0.1% |


---

### ETH (ETHUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **231** em train, **1513** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 215 | 93.1% |
| 3 | 8 | 3.5% |
| 9 | 7 | 3.0% |
| 4 | 1 | 0.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 771 | 51.0% |
| 2 | 319 | 21.1% |
| 3 | 259 | 17.1% |
| 4 | 162 | 10.7% |
| 6 | 2 | 0.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 161 | 69.7% |
| 0.9 | 41 | 17.7% |
| 0.975 | 25 | 10.8% |
| 0.025 | 4 | 1.7% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 506 | 33.4% |
| 0.95 | 499 | 33.0% |
| 0.025 | 432 | 28.6% |
| 0.9 | 50 | 3.3% |
| 0.75 | 15 | 1.0% |
| 0.1 | 6 | 0.4% |
| 0.5 | 3 | 0.2% |
| 0.05 | 1 | 0.1% |
| 0.25 | 1 | 0.1% |


---

### TRX (TRXUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **154** em train, **1398** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 149 | 96.8% |
| 9 | 5 | 3.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 922 | 66.0% |
| 2 | 293 | 21.0% |
| 3 | 103 | 7.4% |
| 6 | 78 | 5.6% |
| 4 | 2 | 0.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 99 | 64.3% |
| 0.975 | 45 | 29.2% |
| 0.95 | 9 | 5.8% |
| 0.025 | 1 | 0.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 461 | 33.0% |
| 0.975 | 413 | 29.5% |
| 0.025 | 139 | 9.9% |
| 0.9 | 109 | 7.8% |
| 0.5 | 100 | 7.2% |
| 0.1 | 76 | 5.4% |
| 0.25 | 63 | 4.5% |
| 0.75 | 36 | 2.6% |
| 0.05 | 1 | 0.1% |


---

### XLM (XLMUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **158** em train, **1405** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 54 | 34.2% |
| 6 | 44 | 27.8% |
| 2 | 43 | 27.2% |
| 9 | 17 | 10.8% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 397 | 28.3% |
| 2 | 343 | 24.4% |
| 3 | 257 | 18.3% |
| 4 | 248 | 17.7% |
| 6 | 160 | 11.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 92 | 58.2% |
| 0.95 | 49 | 31.0% |
| 0.9 | 10 | 6.3% |
| 0.025 | 7 | 4.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 390 | 27.8% |
| 0.025 | 389 | 27.7% |
| 0.95 | 331 | 23.6% |
| 0.9 | 149 | 10.6% |
| 0.1 | 79 | 5.6% |
| 0.75 | 35 | 2.5% |
| 0.05 | 24 | 1.7% |
| 0.5 | 6 | 0.4% |
| 0.25 | 2 | 0.1% |


---

### XRP (XRPUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **169** em train, **1421** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 68 | 40.2% |
| 9 | 42 | 24.9% |
| 3 | 29 | 17.2% |
| 6 | 25 | 14.8% |
| 4 | 5 | 3.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 452 | 31.8% |
| 2 | 354 | 24.9% |
| 3 | 306 | 21.5% |
| 4 | 171 | 12.0% |
| 6 | 138 | 9.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 73 | 43.2% |
| 0.025 | 39 | 23.1% |
| 0.95 | 23 | 13.6% |
| 0.75 | 19 | 11.2% |
| 0.9 | 10 | 5.9% |
| 0.25 | 4 | 2.4% |
| 0.5 | 1 | 0.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 987 | 69.5% |
| 0.95 | 300 | 21.1% |
| 0.025 | 99 | 7.0% |
| 0.9 | 33 | 2.3% |
| 0.75 | 2 | 0.1% |


---

## Horizonte t+30

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 121 | 1417 | 2 | 0.975 | Target |
| BNBUSDT | 176 | 1499 | 2 | 0.975 | Target |
| BTCUSDT | 176 | 1499 | 2 | 0.975 | Target |
| DOGEUSDT | 0 | 1150 | 2 | 0.025 | Target |
| ETHUSDT | 188 | 1517 | 2 | 0.975 | Target |
| TRXUSDT | 111 | 1402 | 9 | 0.975 | Target |
| XLMUSDT | 115 | 1409 | 2 | 0.025 | Target |
| XRPUSDT | 126 | 1425 | 2 | 0.975 | Target |


### ADA (ADAUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **121** em train, **1417** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 73 | 60.3% |
| 9 | 21 | 17.4% |
| 3 | 16 | 13.2% |
| 6 | 11 | 9.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 672 | 47.4% |
| 9 | 427 | 30.1% |
| 3 | 188 | 13.3% |
| 6 | 102 | 7.2% |
| 4 | 28 | 2.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 98 | 81.0% |
| 0.95 | 23 | 19.0% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 545 | 38.5% |
| 0.975 | 476 | 33.6% |
| 0.75 | 102 | 7.2% |
| 0.95 | 93 | 6.6% |
| 0.9 | 90 | 6.4% |
| 0.5 | 65 | 4.6% |
| 0.1 | 22 | 1.6% |
| 0.25 | 21 | 1.5% |
| 0.05 | 3 | 0.2% |


---

### BNB (BNBUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **176** em train, **1499** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 121 | 68.8% |
| 3 | 54 | 30.7% |
| 4 | 1 | 0.6% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 382 | 25.5% |
| 3 | 336 | 22.4% |
| 9 | 312 | 20.8% |
| 6 | 275 | 18.3% |
| 4 | 194 | 12.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 175 | 99.4% |
| 0.025 | 1 | 0.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 781 | 52.1% |
| 0.025 | 318 | 21.2% |
| 0.75 | 159 | 10.6% |
| 0.95 | 127 | 8.5% |
| 0.9 | 98 | 6.5% |
| 0.5 | 10 | 0.7% |
| 0.25 | 4 | 0.3% |
| 0.1 | 2 | 0.1% |


---

### BTC (BTCUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **176** em train, **1499** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 176 | 100.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 604 | 40.3% |
| 9 | 379 | 25.3% |
| 6 | 248 | 16.5% |
| 3 | 187 | 12.5% |
| 4 | 81 | 5.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 115 | 65.3% |
| 0.95 | 61 | 34.7% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 781 | 52.1% |
| 0.95 | 293 | 19.5% |
| 0.025 | 249 | 16.6% |
| 0.75 | 72 | 4.8% |
| 0.5 | 56 | 3.7% |
| 0.9 | 33 | 2.2% |
| 0.25 | 10 | 0.7% |
| 0.05 | 3 | 0.2% |
| 0.1 | 2 | 0.1% |


---

### DOGE (DOGEUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1150** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 439 | 38.2% |
| 4 | 248 | 21.6% |
| 3 | 212 | 18.4% |
| 9 | 148 | 12.9% |
| 6 | 103 | 9.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 439 | 38.2% |
| 0.975 | 233 | 20.3% |
| 0.9 | 186 | 16.2% |
| 0.95 | 178 | 15.5% |
| 0.75 | 57 | 5.0% |
| 0.25 | 28 | 2.4% |
| 0.5 | 13 | 1.1% |
| 0.1 | 10 | 0.9% |
| 0.05 | 6 | 0.5% |


---

### ETH (ETHUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **188** em train, **1517** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 181 | 96.3% |
| 6 | 7 | 3.7% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 543 | 35.8% |
| 9 | 484 | 31.9% |
| 4 | 276 | 18.2% |
| 3 | 149 | 9.8% |
| 6 | 65 | 4.3% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 81 | 43.1% |
| 0.975 | 72 | 38.3% |
| 0.025 | 35 | 18.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 546 | 36.0% |
| 0.025 | 363 | 23.9% |
| 0.95 | 265 | 17.5% |
| 0.75 | 192 | 12.7% |
| 0.25 | 49 | 3.2% |
| 0.9 | 49 | 3.2% |
| 0.05 | 19 | 1.3% |
| 0.5 | 19 | 1.3% |
| 0.1 | 15 | 1.0% |


---

### TRX (TRXUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **111** em train, **1402** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 95 | 85.6% |
| 3 | 16 | 14.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 521 | 37.2% |
| 2 | 404 | 28.8% |
| 6 | 372 | 26.5% |
| 3 | 96 | 6.8% |
| 4 | 9 | 0.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 87 | 78.4% |
| 0.025 | 18 | 16.2% |
| 0.5 | 2 | 1.8% |
| 0.95 | 2 | 1.8% |
| 0.75 | 1 | 0.9% |
| 0.9 | 1 | 0.9% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 462 | 33.0% |
| 0.975 | 422 | 30.1% |
| 0.95 | 329 | 23.5% |
| 0.75 | 118 | 8.4% |
| 0.05 | 27 | 1.9% |
| 0.9 | 20 | 1.4% |
| 0.5 | 14 | 1.0% |
| 0.25 | 6 | 0.4% |
| 0.1 | 4 | 0.3% |


---

### XLM (XLMUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **115** em train, **1409** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 58 | 50.4% |
| 4 | 31 | 27.0% |
| 3 | 17 | 14.8% |
| 6 | 9 | 7.8% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 833 | 59.1% |
| 6 | 220 | 15.6% |
| 3 | 187 | 13.3% |
| 9 | 137 | 9.7% |
| 4 | 32 | 2.3% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 44 | 38.3% |
| 0.75 | 39 | 33.9% |
| 0.025 | 27 | 23.5% |
| 0.9 | 5 | 4.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 453 | 32.2% |
| 0.75 | 350 | 24.8% |
| 0.95 | 297 | 21.1% |
| 0.975 | 113 | 8.0% |
| 0.9 | 107 | 7.6% |
| 0.25 | 41 | 2.9% |
| 0.5 | 30 | 2.1% |
| 0.1 | 13 | 0.9% |
| 0.05 | 5 | 0.4% |


---

### XRP (XRPUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **126** em train, **1425** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 101 | 80.2% |
| 4 | 16 | 12.7% |
| 6 | 6 | 4.8% |
| 3 | 3 | 2.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 491 | 34.5% |
| 3 | 459 | 32.2% |
| 9 | 321 | 22.5% |
| 4 | 117 | 8.2% |
| 6 | 37 | 2.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 41 | 32.5% |
| 0.95 | 22 | 17.5% |
| 0.25 | 20 | 15.9% |
| 0.5 | 17 | 13.5% |
| 0.75 | 10 | 7.9% |
| 0.9 | 9 | 7.1% |
| 0.025 | 6 | 4.8% |
| 0.1 | 1 | 0.8% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 701 | 49.2% |
| 0.025 | 379 | 26.6% |
| 0.9 | 139 | 9.8% |
| 0.75 | 109 | 7.6% |
| 0.95 | 74 | 5.2% |
| 0.05 | 10 | 0.7% |
| 0.5 | 8 | 0.6% |
| 0.25 | 4 | 0.3% |
| 0.1 | 1 | 0.1% |


---
