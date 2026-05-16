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
| ADAUSDT | 0 | 1383 | 9 | 0.5 | Target |
| BNBUSDT | 3 | 1517 | 9 | 0.75 | Target |
| BTCUSDT | 3 | 1517 | 3 | 0.75 | Target |
| DOGEUSDT | 0 | 938 | 9 | 0.5 | Target |
| ETHUSDT | 3 | 1517 | 9 | 0.75 | Target |
| TRXUSDT | 0 | 1328 | 9 | 0.9 | Target |
| XLMUSDT | 0 | 1339 | 9 | 0.75 | Target |
| XRPUSDT | 0 | 1366 | 9 | 0.75 | Target |


### ADA (ADAUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **0** em train, **1383** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 438 | 31.7% |
| 4 | 339 | 24.5% |
| 3 | 326 | 23.6% |
| 6 | 192 | 13.9% |
| 2 | 88 | 6.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.5 | 451 | 32.6% |
| 0.9 | 363 | 26.2% |
| 0.025 | 244 | 17.6% |
| 0.75 | 187 | 13.5% |
| 0.25 | 61 | 4.4% |
| 0.95 | 54 | 3.9% |
| 0.975 | 14 | 1.0% |
| 0.1 | 9 | 0.7% |


---

### BNB (BNBUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **3** em train, **1517** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 3 | 100.0% |


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
| 0.025 | 3 | 100.0% |


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
- Previsões válidas: **3** em train, **1517** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 3 | 100.0% |


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
| 0.75 | 3 | 100.0% |


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
- Previsões válidas: **0** em train, **938** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 631 | 67.3% |
| 2 | 261 | 27.8% |
| 6 | 35 | 3.7% |
| 4 | 7 | 0.7% |
| 3 | 4 | 0.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.5 | 274 | 29.2% |
| 0.25 | 229 | 24.4% |
| 0.75 | 176 | 18.8% |
| 0.9 | 128 | 13.6% |
| 0.95 | 58 | 6.2% |
| 0.025 | 40 | 4.3% |
| 0.1 | 30 | 3.2% |
| 0.975 | 3 | 0.3% |


---

### ETH (ETHUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **3** em train, **1517** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 6 | 3 | 100.0% |


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
| 0.75 | 3 | 100.0% |


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
- Previsões válidas: **0** em train, **1328** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 882 | 66.4% |
| 3 | 216 | 16.3% |
| 4 | 206 | 15.5% |
| 6 | 24 | 1.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 394 | 29.7% |
| 0.75 | 364 | 27.4% |
| 0.1 | 179 | 13.5% |
| 0.95 | 148 | 11.1% |
| 0.5 | 114 | 8.6% |
| 0.25 | 108 | 8.1% |
| 0.975 | 16 | 1.2% |
| 0.025 | 5 | 0.4% |


---

### XLM (XLMUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **0** em train, **1339** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 678 | 50.6% |
| 3 | 368 | 27.5% |
| 6 | 156 | 11.7% |
| 4 | 137 | 10.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 456 | 34.1% |
| 0.25 | 267 | 19.9% |
| 0.9 | 259 | 19.3% |
| 0.5 | 137 | 10.2% |
| 0.1 | 98 | 7.3% |
| 0.95 | 76 | 5.7% |
| 0.05 | 24 | 1.8% |
| 0.025 | 22 | 1.6% |


---

### XRP (XRPUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **0** em train, **1366** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 525 | 38.4% |
| 3 | 444 | 32.5% |
| 6 | 262 | 19.2% |
| 4 | 135 | 9.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 286 | 20.9% |
| 0.25 | 266 | 19.5% |
| 0.975 | 233 | 17.1% |
| 0.5 | 229 | 16.8% |
| 0.9 | 134 | 9.8% |
| 0.1 | 100 | 7.3% |
| 0.95 | 75 | 5.5% |
| 0.025 | 42 | 3.1% |
| 0.05 | 1 | 0.1% |


---

## Horizonte t+7

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 0 | 1371 | 9 | 0.975 | Target |
| BNBUSDT | 0 | 1508 | 9 | 0.975 | Target |
| BTCUSDT | 0 | 1508 | 9 | 0.975 | Target |
| DOGEUSDT | 0 | 926 | 9 | 0.95 | Target |
| ETHUSDT | 0 | 1508 | 9 | 0.975 | Target |
| TRXUSDT | 0 | 1316 | 9 | 0.95 | Target |
| XLMUSDT | 0 | 1327 | 9 | 0.975 | Target |
| XRPUSDT | 0 | 1354 | 9 | 0.975 | Target |


### ADA (ADAUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **1371** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 508 | 37.1% |
| 2 | 483 | 35.2% |
| 4 | 240 | 17.5% |
| 6 | 112 | 8.2% |
| 3 | 28 | 2.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 545 | 39.8% |
| 0.95 | 423 | 30.9% |
| 0.025 | 292 | 21.3% |
| 0.9 | 102 | 7.4% |
| 0.75 | 4 | 0.3% |
| 0.25 | 2 | 0.1% |
| 0.5 | 2 | 0.1% |
| 0.1 | 1 | 0.1% |


---

### BNB (BNBUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **1508** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 730 | 48.4% |
| 4 | 302 | 20.0% |
| 3 | 197 | 13.1% |
| 6 | 148 | 9.8% |
| 2 | 131 | 8.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 570 | 37.8% |
| 0.025 | 417 | 27.7% |
| 0.95 | 286 | 19.0% |
| 0.9 | 81 | 5.4% |
| 0.5 | 47 | 3.1% |
| 0.75 | 45 | 3.0% |
| 0.05 | 43 | 2.9% |
| 0.1 | 11 | 0.7% |
| 0.25 | 8 | 0.5% |


---

### BTC (BTCUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **1508** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 640 | 42.4% |
| 3 | 340 | 22.5% |
| 2 | 306 | 20.3% |
| 6 | 165 | 10.9% |
| 4 | 57 | 3.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 955 | 63.3% |
| 0.025 | 160 | 10.6% |
| 0.9 | 157 | 10.4% |
| 0.95 | 150 | 9.9% |
| 0.75 | 64 | 4.2% |
| 0.1 | 8 | 0.5% |
| 0.5 | 6 | 0.4% |
| 0.25 | 5 | 0.3% |
| 0.05 | 3 | 0.2% |


---

### DOGE (DOGEUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **926** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 453 | 48.9% |
| 2 | 246 | 26.6% |
| 3 | 157 | 17.0% |
| 4 | 56 | 6.0% |
| 6 | 14 | 1.5% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 374 | 40.4% |
| 0.75 | 226 | 24.4% |
| 0.975 | 202 | 21.8% |
| 0.9 | 101 | 10.9% |
| 0.025 | 20 | 2.2% |
| 0.5 | 3 | 0.3% |


---

### ETH (ETHUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **1508** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 771 | 51.1% |
| 2 | 314 | 20.8% |
| 3 | 259 | 17.2% |
| 4 | 162 | 10.7% |
| 6 | 2 | 0.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 501 | 33.2% |
| 0.95 | 499 | 33.1% |
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
- Previsões válidas: **0** em train, **1316** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 921 | 70.0% |
| 2 | 258 | 19.6% |
| 3 | 103 | 7.8% |
| 6 | 32 | 2.4% |
| 4 | 2 | 0.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 428 | 32.5% |
| 0.975 | 393 | 29.9% |
| 0.025 | 139 | 10.6% |
| 0.5 | 100 | 7.6% |
| 0.9 | 80 | 6.1% |
| 0.1 | 76 | 5.8% |
| 0.25 | 63 | 4.8% |
| 0.75 | 36 | 2.7% |
| 0.05 | 1 | 0.1% |


---

### XLM (XLMUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **1327** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 397 | 29.9% |
| 2 | 343 | 25.8% |
| 4 | 248 | 18.7% |
| 3 | 231 | 17.4% |
| 6 | 108 | 8.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 390 | 29.4% |
| 0.025 | 389 | 29.3% |
| 0.95 | 253 | 19.1% |
| 0.9 | 149 | 11.2% |
| 0.1 | 79 | 6.0% |
| 0.75 | 35 | 2.6% |
| 0.05 | 24 | 1.8% |
| 0.5 | 6 | 0.5% |
| 0.25 | 2 | 0.2% |


---

### XRP (XRPUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **0** em train, **1354** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 432 | 31.9% |
| 2 | 354 | 26.1% |
| 3 | 279 | 20.6% |
| 4 | 171 | 12.6% |
| 6 | 118 | 8.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 987 | 72.9% |
| 0.95 | 255 | 18.8% |
| 0.025 | 85 | 6.3% |
| 0.9 | 25 | 1.8% |
| 0.75 | 2 | 0.1% |


---

## Horizonte t+30

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 0 | 1325 | 2 | 0.025 | Target |
| BNBUSDT | 0 | 1462 | 2 | 0.975 | Target |
| BTCUSDT | 0 | 1462 | 2 | 0.975 | Target |
| DOGEUSDT | 0 | 880 | 2 | 0.975 | Target |
| ETHUSDT | 0 | 1492 | 2 | 0.975 | Target |
| TRXUSDT | 0 | 1300 | 9 | 0.025 | Target |
| XLMUSDT | 0 | 1311 | 2 | 0.025 | Target |
| XRPUSDT | 0 | 1338 | 2 | 0.975 | Target |


### ADA (ADAUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1325** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 598 | 45.1% |
| 9 | 409 | 30.9% |
| 3 | 188 | 14.2% |
| 6 | 102 | 7.7% |
| 4 | 28 | 2.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 545 | 41.1% |
| 0.975 | 476 | 35.9% |
| 0.95 | 79 | 6.0% |
| 0.75 | 77 | 5.8% |
| 0.9 | 55 | 4.2% |
| 0.5 | 54 | 4.1% |
| 0.1 | 21 | 1.6% |
| 0.25 | 15 | 1.1% |
| 0.05 | 3 | 0.2% |


---

### BNB (BNBUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1462** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 345 | 23.6% |
| 3 | 336 | 23.0% |
| 9 | 312 | 21.3% |
| 6 | 275 | 18.8% |
| 4 | 194 | 13.3% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 744 | 50.9% |
| 0.025 | 318 | 21.8% |
| 0.75 | 159 | 10.9% |
| 0.95 | 127 | 8.7% |
| 0.9 | 98 | 6.7% |
| 0.5 | 10 | 0.7% |
| 0.25 | 4 | 0.3% |
| 0.1 | 2 | 0.1% |


---

### BTC (BTCUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1462** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 567 | 38.8% |
| 9 | 379 | 25.9% |
| 6 | 248 | 17.0% |
| 3 | 187 | 12.8% |
| 4 | 81 | 5.5% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 781 | 53.4% |
| 0.95 | 256 | 17.5% |
| 0.025 | 249 | 17.0% |
| 0.75 | 72 | 4.9% |
| 0.5 | 56 | 3.8% |
| 0.9 | 33 | 2.3% |
| 0.25 | 10 | 0.7% |
| 0.05 | 3 | 0.2% |
| 0.1 | 2 | 0.1% |


---

### DOGE (DOGEUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **880** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 288 | 32.7% |
| 6 | 265 | 30.1% |
| 9 | 159 | 18.1% |
| 4 | 138 | 15.7% |
| 3 | 30 | 3.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 293 | 33.3% |
| 0.025 | 262 | 29.8% |
| 0.95 | 136 | 15.5% |
| 0.9 | 100 | 11.4% |
| 0.75 | 46 | 5.2% |
| 0.05 | 24 | 2.7% |
| 0.5 | 7 | 0.8% |
| 0.1 | 6 | 0.7% |
| 0.25 | 6 | 0.7% |


---

### ETH (ETHUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1492** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 518 | 34.7% |
| 9 | 484 | 32.4% |
| 4 | 276 | 18.5% |
| 3 | 149 | 10.0% |
| 6 | 65 | 4.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 546 | 36.6% |
| 0.025 | 363 | 24.3% |
| 0.95 | 265 | 17.8% |
| 0.75 | 167 | 11.2% |
| 0.25 | 49 | 3.3% |
| 0.9 | 49 | 3.3% |
| 0.05 | 19 | 1.3% |
| 0.5 | 19 | 1.3% |
| 0.1 | 15 | 1.0% |


---

### TRX (TRXUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1300** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 521 | 40.1% |
| 2 | 356 | 27.4% |
| 6 | 342 | 26.3% |
| 3 | 72 | 5.5% |
| 4 | 9 | 0.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 434 | 33.4% |
| 0.975 | 350 | 26.9% |
| 0.95 | 329 | 25.3% |
| 0.75 | 118 | 9.1% |
| 0.05 | 27 | 2.1% |
| 0.9 | 20 | 1.5% |
| 0.5 | 12 | 0.9% |
| 0.25 | 6 | 0.5% |
| 0.1 | 4 | 0.3% |


---

### XLM (XLMUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1311** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 735 | 56.1% |
| 6 | 220 | 16.8% |
| 3 | 187 | 14.3% |
| 9 | 137 | 10.5% |
| 4 | 32 | 2.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 355 | 27.1% |
| 0.75 | 350 | 26.7% |
| 0.95 | 297 | 22.7% |
| 0.975 | 113 | 8.6% |
| 0.9 | 107 | 8.2% |
| 0.25 | 41 | 3.1% |
| 0.5 | 30 | 2.3% |
| 0.1 | 13 | 1.0% |
| 0.05 | 5 | 0.4% |


---

### XRP (XRPUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **0** em train, **1338** em test  

**Distribuição do nº de clusters `c` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 473 | 35.4% |
| 3 | 444 | 33.2% |
| 9 | 267 | 20.0% |
| 4 | 117 | 8.7% |
| 6 | 37 | 2.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_train: nenhuma seleção registrada._

_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 686 | 51.3% |
| 0.025 | 327 | 24.4% |
| 0.9 | 139 | 10.4% |
| 0.75 | 108 | 8.1% |
| 0.95 | 74 | 5.5% |
| 0.5 | 2 | 0.1% |
| 0.05 | 1 | 0.1% |
| 0.25 | 1 | 0.1% |


---
