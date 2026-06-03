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
| ADAUSDT | 686 | 969 | 3 | 0.5 | Target |
| BNBUSDT | 768 | 1024 | 9 | 0.75 | Target |
| BTCUSDT | 768 | 1024 | 3 | 0.75 | Target |
| DOGEUSDT | 419 | 791 | 9 | 0.5 | Target |
| ETHUSDT | 768 | 1024 | 9 | 0.75 | Target |
| TRXUSDT | 653 | 947 | 9 | 0.9 | Target |
| XLMUSDT | 660 | 951 | 9 | 0.75 | Target |
| XRPUSDT | 676 | 962 | 9 | 0.25 | Target |


### ADA (ADAUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **686** em train, **969** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 374 | 54.5% |
| 4 | 208 | 30.3% |
| 6 | 79 | 11.5% |
| 9 | 17 | 2.5% |
| 2 | 8 | 1.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 428 | 44.2% |
| 4 | 203 | 20.9% |
| 6 | 168 | 17.3% |
| 2 | 88 | 9.1% |
| 3 | 82 | 8.5% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 253 | 36.9% |
| 0.5 | 208 | 30.3% |
| 0.25 | 129 | 18.8% |
| 0.1 | 46 | 6.7% |
| 0.75 | 29 | 4.2% |
| 0.025 | 12 | 1.7% |
| 0.05 | 6 | 0.9% |
| 0.975 | 2 | 0.3% |
| 0.95 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.5 | 319 | 32.9% |
| 0.025 | 244 | 25.2% |
| 0.75 | 158 | 16.3% |
| 0.9 | 110 | 11.4% |
| 0.25 | 61 | 6.3% |
| 0.95 | 54 | 5.6% |
| 0.975 | 14 | 1.4% |
| 0.1 | 9 | 0.9% |


---

### BNB (BNBUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **768** em train, **1024** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 6 | 382 | 49.7% |
| 9 | 333 | 43.4% |
| 3 | 51 | 6.6% |
| 4 | 2 | 0.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 742 | 72.5% |
| 6 | 146 | 14.3% |
| 2 | 65 | 6.3% |
| 3 | 47 | 4.6% |
| 4 | 24 | 2.3% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 264 | 34.4% |
| 0.025 | 255 | 33.2% |
| 0.5 | 185 | 24.1% |
| 0.25 | 52 | 6.8% |
| 0.975 | 10 | 1.3% |
| 0.05 | 1 | 0.1% |
| 0.1 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 386 | 37.7% |
| 0.25 | 248 | 24.2% |
| 0.5 | 183 | 17.9% |
| 0.9 | 157 | 15.3% |
| 0.025 | 37 | 3.6% |
| 0.1 | 13 | 1.3% |


---

### BTC (BTCUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **768** em train, **1024** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 642 | 83.6% |
| 6 | 102 | 13.3% |
| 9 | 19 | 2.5% |
| 4 | 4 | 0.5% |
| 2 | 1 | 0.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 469 | 45.8% |
| 3 | 424 | 41.4% |
| 6 | 94 | 9.2% |
| 4 | 37 | 3.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 553 | 72.0% |
| 0.5 | 90 | 11.7% |
| 0.25 | 54 | 7.0% |
| 0.9 | 43 | 5.6% |
| 0.975 | 12 | 1.6% |
| 0.95 | 8 | 1.0% |
| 0.025 | 4 | 0.5% |
| 0.1 | 3 | 0.4% |
| 0.05 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 325 | 31.7% |
| 0.75 | 218 | 21.3% |
| 0.1 | 132 | 12.9% |
| 0.975 | 132 | 12.9% |
| 0.5 | 86 | 8.4% |
| 0.9 | 40 | 3.9% |
| 0.05 | 37 | 3.6% |
| 0.025 | 27 | 2.6% |
| 0.95 | 27 | 2.6% |


---

### DOGE (DOGEUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **419** em train, **791** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 221 | 52.7% |
| 4 | 111 | 26.5% |
| 6 | 63 | 15.0% |
| 2 | 19 | 4.5% |
| 3 | 5 | 1.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 518 | 65.5% |
| 2 | 261 | 33.0% |
| 4 | 7 | 0.9% |
| 3 | 4 | 0.5% |
| 6 | 1 | 0.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 205 | 48.9% |
| 0.5 | 102 | 24.3% |
| 0.25 | 40 | 9.5% |
| 0.95 | 20 | 4.8% |
| 0.75 | 18 | 4.3% |
| 0.025 | 15 | 3.6% |
| 0.05 | 13 | 3.1% |
| 0.9 | 6 | 1.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 190 | 24.0% |
| 0.5 | 173 | 21.9% |
| 0.75 | 172 | 21.7% |
| 0.9 | 128 | 16.2% |
| 0.95 | 58 | 7.3% |
| 0.025 | 40 | 5.1% |
| 0.1 | 30 | 3.8% |


---

### ETH (ETHUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **768** em train, **1024** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 347 | 45.2% |
| 6 | 236 | 30.7% |
| 2 | 104 | 13.5% |
| 4 | 42 | 5.5% |
| 9 | 39 | 5.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 553 | 54.0% |
| 3 | 146 | 14.3% |
| 6 | 133 | 13.0% |
| 2 | 118 | 11.5% |
| 4 | 74 | 7.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 466 | 60.7% |
| 0.5 | 114 | 14.8% |
| 0.025 | 74 | 9.6% |
| 0.05 | 47 | 6.1% |
| 0.25 | 39 | 5.1% |
| 0.9 | 14 | 1.8% |
| 0.1 | 10 | 1.3% |
| 0.975 | 4 | 0.5% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 378 | 36.9% |
| 0.95 | 183 | 17.9% |
| 0.25 | 173 | 16.9% |
| 0.75 | 159 | 15.5% |
| 0.5 | 73 | 7.1% |
| 0.05 | 21 | 2.1% |
| 0.1 | 14 | 1.4% |
| 0.9 | 14 | 1.4% |
| 0.975 | 9 | 0.9% |


---

### TRX (TRXUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **653** em train, **947** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 270 | 41.3% |
| 4 | 206 | 31.5% |
| 6 | 166 | 25.4% |
| 3 | 11 | 1.7% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 723 | 76.3% |
| 3 | 209 | 22.1% |
| 4 | 15 | 1.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 283 | 43.3% |
| 0.75 | 107 | 16.4% |
| 0.025 | 87 | 13.3% |
| 0.95 | 52 | 8.0% |
| 0.1 | 41 | 6.3% |
| 0.5 | 31 | 4.7% |
| 0.25 | 25 | 3.8% |
| 0.975 | 23 | 3.5% |
| 0.05 | 4 | 0.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 262 | 27.7% |
| 0.1 | 179 | 18.9% |
| 0.9 | 150 | 15.8% |
| 0.95 | 135 | 14.3% |
| 0.25 | 108 | 11.4% |
| 0.5 | 108 | 11.4% |
| 0.025 | 5 | 0.5% |


---

### XLM (XLMUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **660** em train, **951** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 312 | 47.3% |
| 4 | 216 | 32.7% |
| 6 | 59 | 8.9% |
| 2 | 53 | 8.0% |
| 3 | 20 | 3.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 405 | 42.6% |
| 3 | 351 | 36.9% |
| 6 | 109 | 11.5% |
| 4 | 86 | 9.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 315 | 47.7% |
| 0.5 | 179 | 27.1% |
| 0.25 | 57 | 8.6% |
| 0.95 | 46 | 7.0% |
| 0.1 | 30 | 4.5% |
| 0.9 | 26 | 3.9% |
| 0.975 | 4 | 0.6% |
| 0.025 | 2 | 0.3% |
| 0.05 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 267 | 28.1% |
| 0.9 | 233 | 24.5% |
| 0.75 | 187 | 19.7% |
| 0.1 | 98 | 10.3% |
| 0.5 | 90 | 9.5% |
| 0.95 | 30 | 3.2% |
| 0.05 | 24 | 2.5% |
| 0.025 | 22 | 2.3% |


---

### XRP (XRPUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **676** em train, **962** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 333 | 49.3% |
| 9 | 209 | 30.9% |
| 6 | 104 | 15.4% |
| 2 | 22 | 3.3% |
| 4 | 8 | 1.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 451 | 46.9% |
| 6 | 234 | 24.3% |
| 3 | 142 | 14.8% |
| 4 | 135 | 14.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 220 | 32.5% |
| 0.5 | 182 | 26.9% |
| 0.25 | 140 | 20.7% |
| 0.975 | 50 | 7.4% |
| 0.95 | 40 | 5.9% |
| 0.9 | 38 | 5.6% |
| 0.025 | 4 | 0.6% |
| 0.1 | 2 | 0.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 233 | 24.2% |
| 0.25 | 226 | 23.5% |
| 0.75 | 145 | 15.1% |
| 0.1 | 100 | 10.4% |
| 0.9 | 96 | 10.0% |
| 0.5 | 84 | 8.7% |
| 0.025 | 42 | 4.4% |
| 0.95 | 35 | 3.6% |
| 0.05 | 1 | 0.1% |


---

## Horizonte t+7

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 671 | 966 | 2 | 0.975 | Target |
| BNBUSDT | 753 | 1021 | 9 | 0.975 | Target |
| BTCUSDT | 753 | 1021 | 9 | 0.975 | Target |
| DOGEUSDT | 404 | 788 | 9 | 0.975 | Target |
| ETHUSDT | 753 | 1021 | 9 | 0.95 | Target |
| TRXUSDT | 638 | 944 | 9 | 0.95 | Target |
| XLMUSDT | 644 | 949 | 2 | 0.975 | Target |
| XRPUSDT | 660 | 960 | 9 | 0.975 | Target |


### ADA (ADAUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **671** em train, **966** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 545 | 81.2% |
| 3 | 63 | 9.4% |
| 6 | 57 | 8.5% |
| 9 | 6 | 0.9% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 502 | 52.0% |
| 4 | 240 | 24.8% |
| 6 | 99 | 10.2% |
| 2 | 97 | 10.0% |
| 3 | 28 | 2.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 370 | 55.1% |
| 0.975 | 184 | 27.4% |
| 0.95 | 71 | 10.6% |
| 0.9 | 37 | 5.5% |
| 0.75 | 4 | 0.6% |
| 0.25 | 2 | 0.3% |
| 0.5 | 2 | 0.3% |
| 0.1 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 536 | 55.5% |
| 0.95 | 352 | 36.4% |
| 0.9 | 65 | 6.7% |
| 0.025 | 13 | 1.3% |


---

### BNB (BNBUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **753** em train, **1021** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 250 | 33.2% |
| 9 | 235 | 31.2% |
| 4 | 142 | 18.9% |
| 3 | 89 | 11.8% |
| 6 | 37 | 4.9% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 518 | 50.7% |
| 4 | 186 | 18.2% |
| 3 | 164 | 16.1% |
| 6 | 129 | 12.6% |
| 2 | 24 | 2.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 288 | 38.2% |
| 0.025 | 195 | 25.9% |
| 0.95 | 155 | 20.6% |
| 0.5 | 39 | 5.2% |
| 0.9 | 37 | 4.9% |
| 0.75 | 36 | 4.8% |
| 0.25 | 2 | 0.3% |
| 0.1 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 436 | 42.7% |
| 0.025 | 231 | 22.6% |
| 0.95 | 222 | 21.7% |
| 0.9 | 55 | 5.4% |
| 0.05 | 43 | 4.2% |
| 0.1 | 10 | 1.0% |
| 0.75 | 10 | 1.0% |
| 0.5 | 8 | 0.8% |
| 0.25 | 6 | 0.6% |


---

### BTC (BTCUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **753** em train, **1021** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 266 | 35.3% |
| 9 | 191 | 25.4% |
| 6 | 115 | 15.3% |
| 4 | 94 | 12.5% |
| 3 | 87 | 11.6% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 449 | 44.0% |
| 3 | 263 | 25.8% |
| 2 | 242 | 23.7% |
| 6 | 59 | 5.8% |
| 4 | 8 | 0.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 491 | 65.2% |
| 0.9 | 97 | 12.9% |
| 0.75 | 71 | 9.4% |
| 0.025 | 69 | 9.2% |
| 0.95 | 7 | 0.9% |
| 0.1 | 5 | 0.7% |
| 0.25 | 5 | 0.7% |
| 0.5 | 5 | 0.7% |
| 0.05 | 3 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 708 | 69.3% |
| 0.95 | 105 | 10.3% |
| 0.025 | 98 | 9.6% |
| 0.9 | 81 | 7.9% |
| 0.75 | 24 | 2.4% |
| 0.1 | 3 | 0.3% |
| 0.5 | 2 | 0.2% |


---

### DOGE (DOGEUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **404** em train, **788** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 242 | 59.9% |
| 4 | 77 | 19.1% |
| 9 | 51 | 12.6% |
| 2 | 34 | 8.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 420 | 53.3% |
| 2 | 246 | 31.2% |
| 3 | 82 | 10.4% |
| 4 | 26 | 3.3% |
| 6 | 14 | 1.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 377 | 93.3% |
| 0.75 | 15 | 3.7% |
| 0.025 | 8 | 2.0% |
| 0.9 | 3 | 0.7% |
| 0.95 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 374 | 47.5% |
| 0.75 | 226 | 28.7% |
| 0.9 | 101 | 12.8% |
| 0.975 | 64 | 8.1% |
| 0.025 | 20 | 2.5% |
| 0.5 | 3 | 0.4% |


---

### ETH (ETHUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **753** em train, **1021** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 247 | 32.8% |
| 9 | 247 | 32.8% |
| 3 | 129 | 17.1% |
| 4 | 129 | 17.1% |
| 6 | 1 | 0.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 531 | 52.0% |
| 2 | 277 | 27.1% |
| 3 | 171 | 16.7% |
| 4 | 41 | 4.0% |
| 6 | 1 | 0.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 394 | 52.3% |
| 0.95 | 220 | 29.2% |
| 0.975 | 106 | 14.1% |
| 0.9 | 33 | 4.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 455 | 44.6% |
| 0.95 | 448 | 43.9% |
| 0.9 | 50 | 4.9% |
| 0.025 | 42 | 4.1% |
| 0.75 | 15 | 1.5% |
| 0.1 | 6 | 0.6% |
| 0.5 | 3 | 0.3% |
| 0.05 | 1 | 0.1% |
| 0.25 | 1 | 0.1% |


---

### TRX (TRXUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **638** em train, **944** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 422 | 66.1% |
| 9 | 79 | 12.4% |
| 6 | 76 | 11.9% |
| 4 | 39 | 6.1% |
| 3 | 22 | 3.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 847 | 89.7% |
| 3 | 81 | 8.6% |
| 6 | 14 | 1.5% |
| 4 | 2 | 0.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 185 | 29.0% |
| 0.025 | 125 | 19.6% |
| 0.95 | 88 | 13.8% |
| 0.1 | 76 | 11.9% |
| 0.975 | 68 | 10.7% |
| 0.25 | 62 | 9.7% |
| 0.75 | 19 | 3.0% |
| 0.5 | 14 | 2.2% |
| 0.05 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 385 | 40.8% |
| 0.975 | 385 | 40.8% |
| 0.5 | 86 | 9.1% |
| 0.9 | 45 | 4.8% |
| 0.025 | 21 | 2.2% |
| 0.75 | 21 | 2.2% |
| 0.25 | 1 | 0.1% |


---

### XLM (XLMUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **644** em train, **949** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 399 | 62.0% |
| 6 | 97 | 15.1% |
| 3 | 89 | 13.8% |
| 9 | 44 | 6.8% |
| 4 | 15 | 2.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 365 | 38.5% |
| 4 | 233 | 24.6% |
| 3 | 189 | 19.9% |
| 6 | 108 | 11.4% |
| 2 | 54 | 5.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 267 | 41.5% |
| 0.95 | 219 | 34.0% |
| 0.975 | 112 | 17.4% |
| 0.9 | 32 | 5.0% |
| 0.75 | 7 | 1.1% |
| 0.5 | 3 | 0.5% |
| 0.1 | 2 | 0.3% |
| 0.05 | 1 | 0.2% |
| 0.25 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 384 | 40.5% |
| 0.95 | 170 | 17.9% |
| 0.025 | 132 | 13.9% |
| 0.9 | 129 | 13.6% |
| 0.1 | 77 | 8.1% |
| 0.75 | 29 | 3.1% |
| 0.05 | 23 | 2.4% |
| 0.5 | 4 | 0.4% |
| 0.25 | 1 | 0.1% |


---

### XRP (XRPUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **660** em train, **960** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 285 | 43.2% |
| 3 | 148 | 22.4% |
| 9 | 99 | 15.0% |
| 6 | 92 | 13.9% |
| 4 | 36 | 5.5% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 431 | 44.9% |
| 3 | 228 | 23.8% |
| 4 | 142 | 14.8% |
| 6 | 90 | 9.4% |
| 2 | 69 | 7.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 297 | 45.0% |
| 0.975 | 225 | 34.1% |
| 0.025 | 94 | 14.2% |
| 0.9 | 31 | 4.7% |
| 0.75 | 6 | 0.9% |
| 0.25 | 3 | 0.5% |
| 0.5 | 2 | 0.3% |
| 0.05 | 1 | 0.2% |
| 0.1 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 843 | 87.8% |
| 0.025 | 85 | 8.9% |
| 0.95 | 30 | 3.1% |
| 0.9 | 2 | 0.2% |


---

## Horizonte t+30

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 611 | 957 | 2 | 0.975 | Target |
| BNBUSDT | 693 | 1012 | 2 | 0.975 | Target |
| BTCUSDT | 693 | 1012 | 2 | 0.975 | Target |
| DOGEUSDT | 344 | 779 | 2 | 0.975 | Target |
| ETHUSDT | 693 | 1012 | 2 | 0.975 | Target |
| TRXUSDT | 578 | 935 | 9 | 0.975 | Target |
| XLMUSDT | 584 | 940 | 2 | 0.025 | Target |
| XRPUSDT | 601 | 950 | 2 | 0.975 | Target |


### ADA (ADAUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **611** em train, **957** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 416 | 68.1% |
| 3 | 71 | 11.6% |
| 9 | 61 | 10.0% |
| 6 | 39 | 6.4% |
| 4 | 24 | 3.9% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 388 | 40.5% |
| 2 | 311 | 32.5% |
| 3 | 137 | 14.3% |
| 6 | 93 | 9.7% |
| 4 | 28 | 2.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 238 | 39.0% |
| 0.025 | 228 | 37.3% |
| 0.75 | 53 | 8.7% |
| 0.9 | 42 | 6.9% |
| 0.95 | 23 | 3.8% |
| 0.5 | 19 | 3.1% |
| 0.25 | 7 | 1.1% |
| 0.1 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 381 | 39.8% |
| 0.025 | 316 | 33.0% |
| 0.95 | 78 | 8.2% |
| 0.75 | 50 | 5.2% |
| 0.9 | 48 | 5.0% |
| 0.5 | 46 | 4.8% |
| 0.1 | 21 | 2.2% |
| 0.25 | 14 | 1.5% |
| 0.05 | 3 | 0.3% |


---

### BNB (BNBUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **693** em train, **1012** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 463 | 66.8% |
| 4 | 96 | 13.9% |
| 3 | 69 | 10.0% |
| 9 | 61 | 8.8% |
| 6 | 4 | 0.6% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 307 | 30.3% |
| 6 | 271 | 26.8% |
| 9 | 251 | 24.8% |
| 4 | 98 | 9.7% |
| 2 | 85 | 8.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 565 | 81.5% |
| 0.025 | 115 | 16.6% |
| 0.95 | 7 | 1.0% |
| 0.9 | 5 | 0.7% |
| 0.25 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 421 | 41.6% |
| 0.025 | 203 | 20.1% |
| 0.75 | 159 | 15.7% |
| 0.95 | 121 | 12.0% |
| 0.9 | 93 | 9.2% |
| 0.5 | 10 | 1.0% |
| 0.25 | 3 | 0.3% |
| 0.1 | 2 | 0.2% |


---

### BTC (BTCUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **693** em train, **1012** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 479 | 69.1% |
| 6 | 153 | 22.1% |
| 4 | 58 | 8.4% |
| 9 | 2 | 0.3% |
| 3 | 1 | 0.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 379 | 37.5% |
| 2 | 320 | 31.6% |
| 3 | 187 | 18.5% |
| 6 | 102 | 10.1% |
| 4 | 24 | 2.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 385 | 55.6% |
| 0.95 | 165 | 23.8% |
| 0.025 | 133 | 19.2% |
| 0.5 | 6 | 0.9% |
| 0.75 | 2 | 0.3% |
| 0.25 | 1 | 0.1% |
| 0.9 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 537 | 53.1% |
| 0.95 | 184 | 18.2% |
| 0.025 | 116 | 11.5% |
| 0.75 | 72 | 7.1% |
| 0.5 | 56 | 5.5% |
| 0.9 | 32 | 3.2% |
| 0.25 | 10 | 1.0% |
| 0.05 | 3 | 0.3% |
| 0.1 | 2 | 0.2% |


---

### DOGE (DOGEUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **344** em train, **779** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 4 | 180 | 52.3% |
| 2 | 87 | 25.3% |
| 9 | 30 | 8.7% |
| 6 | 26 | 7.6% |
| 3 | 21 | 6.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 288 | 37.0% |
| 6 | 265 | 34.0% |
| 9 | 159 | 20.4% |
| 4 | 37 | 4.7% |
| 3 | 30 | 3.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 123 | 35.8% |
| 0.9 | 103 | 29.9% |
| 0.975 | 93 | 27.0% |
| 0.75 | 12 | 3.5% |
| 0.5 | 9 | 2.6% |
| 0.25 | 4 | 1.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 293 | 37.6% |
| 0.025 | 262 | 33.6% |
| 0.95 | 89 | 11.4% |
| 0.75 | 46 | 5.9% |
| 0.9 | 46 | 5.9% |
| 0.05 | 24 | 3.1% |
| 0.5 | 7 | 0.9% |
| 0.1 | 6 | 0.8% |
| 0.25 | 6 | 0.8% |


---

### ETH (ETHUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **693** em train, **1012** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 461 | 66.5% |
| 4 | 101 | 14.6% |
| 3 | 73 | 10.5% |
| 9 | 51 | 7.4% |
| 6 | 7 | 1.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 433 | 42.8% |
| 2 | 263 | 26.0% |
| 4 | 175 | 17.3% |
| 3 | 76 | 7.5% |
| 6 | 65 | 6.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 387 | 55.8% |
| 0.75 | 170 | 24.5% |
| 0.025 | 83 | 12.0% |
| 0.9 | 28 | 4.0% |
| 0.95 | 23 | 3.3% |
| 0.05 | 1 | 0.1% |
| 0.5 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 315 | 31.1% |
| 0.95 | 242 | 23.9% |
| 0.975 | 231 | 22.8% |
| 0.75 | 103 | 10.2% |
| 0.25 | 49 | 4.8% |
| 0.9 | 21 | 2.1% |
| 0.05 | 18 | 1.8% |
| 0.5 | 18 | 1.8% |
| 0.1 | 15 | 1.5% |


---

### TRX (TRXUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **578** em train, **935** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 380 | 65.7% |
| 6 | 129 | 22.3% |
| 3 | 65 | 11.2% |
| 4 | 4 | 0.7% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 521 | 55.7% |
| 6 | 243 | 26.0% |
| 2 | 119 | 12.7% |
| 3 | 47 | 5.0% |
| 4 | 5 | 0.5% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 281 | 48.6% |
| 0.975 | 228 | 39.4% |
| 0.95 | 37 | 6.4% |
| 0.05 | 22 | 3.8% |
| 0.5 | 4 | 0.7% |
| 0.9 | 3 | 0.5% |
| 0.1 | 2 | 0.3% |
| 0.75 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 294 | 31.4% |
| 0.975 | 281 | 30.1% |
| 0.025 | 199 | 21.3% |
| 0.75 | 118 | 12.6% |
| 0.9 | 18 | 1.9% |
| 0.5 | 12 | 1.3% |
| 0.25 | 6 | 0.6% |
| 0.05 | 5 | 0.5% |
| 0.1 | 2 | 0.2% |


---

### XLM (XLMUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **584** em train, **940** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 443 | 75.9% |
| 3 | 101 | 17.3% |
| 4 | 31 | 5.3% |
| 6 | 9 | 1.5% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 448 | 47.7% |
| 6 | 220 | 23.4% |
| 9 | 137 | 14.6% |
| 3 | 103 | 11.0% |
| 4 | 32 | 3.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 436 | 74.7% |
| 0.75 | 62 | 10.6% |
| 0.975 | 44 | 7.5% |
| 0.9 | 19 | 3.3% |
| 0.25 | 10 | 1.7% |
| 0.5 | 6 | 1.0% |
| 0.05 | 4 | 0.7% |
| 0.1 | 3 | 0.5% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 327 | 34.8% |
| 0.95 | 297 | 31.6% |
| 0.975 | 113 | 12.0% |
| 0.9 | 93 | 9.9% |
| 0.025 | 44 | 4.7% |
| 0.25 | 31 | 3.3% |
| 0.5 | 24 | 2.6% |
| 0.1 | 10 | 1.1% |
| 0.05 | 1 | 0.1% |


---

### XRP (XRPUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **601** em train, **950** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 421 | 70.0% |
| 9 | 132 | 22.0% |
| 4 | 24 | 4.0% |
| 3 | 18 | 3.0% |
| 6 | 6 | 1.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 444 | 46.7% |
| 9 | 189 | 19.9% |
| 2 | 171 | 18.0% |
| 4 | 109 | 11.5% |
| 6 | 37 | 3.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 373 | 62.1% |
| 0.975 | 96 | 16.0% |
| 0.75 | 37 | 6.2% |
| 0.5 | 25 | 4.2% |
| 0.25 | 24 | 4.0% |
| 0.95 | 22 | 3.7% |
| 0.9 | 12 | 2.0% |
| 0.05 | 10 | 1.7% |
| 0.1 | 2 | 0.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 646 | 68.0% |
| 0.9 | 136 | 14.3% |
| 0.75 | 82 | 8.6% |
| 0.95 | 74 | 7.8% |
| 0.025 | 12 | 1.3% |


---
