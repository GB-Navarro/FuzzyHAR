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
| ADAUSDT | 928 | 727 | 3 | 0.5 | Target |
| BNBUSDT | 1024 | 768 | 9 | 0.75 | Target |
| BTCUSDT | 1024 | 768 | 3 | 0.75 | Target |
| DOGEUSDT | 617 | 593 | 9 | 0.5 | Target |
| ETHUSDT | 1024 | 768 | 9 | 0.75 | Target |
| TRXUSDT | 890 | 710 | 9 | 0.9 | Target |
| XLMUSDT | 897 | 714 | 9 | 0.75 | Target |
| XRPUSDT | 916 | 722 | 9 | 0.25 | Target |


### ADA (ADAUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **928** em train, **727** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 4 | 385 | 41.5% |
| 3 | 374 | 40.3% |
| 6 | 104 | 11.2% |
| 9 | 57 | 6.1% |
| 2 | 8 | 0.9% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 388 | 53.4% |
| 6 | 143 | 19.7% |
| 2 | 88 | 12.1% |
| 3 | 82 | 11.3% |
| 4 | 26 | 3.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 253 | 27.3% |
| 0.025 | 220 | 23.7% |
| 0.5 | 218 | 23.5% |
| 0.25 | 130 | 14.0% |
| 0.75 | 52 | 5.6% |
| 0.1 | 46 | 5.0% |
| 0.05 | 6 | 0.6% |
| 0.975 | 2 | 0.2% |
| 0.95 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.5 | 309 | 42.5% |
| 0.75 | 135 | 18.6% |
| 0.9 | 110 | 15.1% |
| 0.25 | 60 | 8.3% |
| 0.95 | 54 | 7.4% |
| 0.025 | 36 | 5.0% |
| 0.975 | 14 | 1.9% |
| 0.1 | 9 | 1.2% |


---

### BNB (BNBUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **1024** em train, **768** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 531 | 51.9% |
| 6 | 411 | 40.1% |
| 3 | 80 | 7.8% |
| 4 | 2 | 0.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 544 | 70.8% |
| 6 | 117 | 15.2% |
| 2 | 65 | 8.5% |
| 4 | 24 | 3.1% |
| 3 | 18 | 2.3% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 284 | 27.7% |
| 0.75 | 275 | 26.9% |
| 0.5 | 188 | 18.4% |
| 0.25 | 186 | 18.2% |
| 0.9 | 78 | 7.6% |
| 0.975 | 10 | 1.0% |
| 0.1 | 2 | 0.2% |
| 0.05 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 375 | 48.8% |
| 0.5 | 180 | 23.4% |
| 0.25 | 114 | 14.8% |
| 0.9 | 79 | 10.3% |
| 0.1 | 12 | 1.6% |
| 0.025 | 8 | 1.0% |


---

### BTC (BTCUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **1024** em train, **768** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 842 | 82.2% |
| 6 | 158 | 15.4% |
| 9 | 19 | 1.9% |
| 4 | 4 | 0.4% |
| 2 | 1 | 0.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 469 | 61.1% |
| 3 | 224 | 29.2% |
| 6 | 38 | 4.9% |
| 4 | 37 | 4.8% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 611 | 59.7% |
| 0.25 | 173 | 16.9% |
| 0.5 | 91 | 8.9% |
| 0.9 | 74 | 7.2% |
| 0.1 | 48 | 4.7% |
| 0.975 | 12 | 1.2% |
| 0.95 | 8 | 0.8% |
| 0.025 | 4 | 0.4% |
| 0.05 | 3 | 0.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 206 | 26.8% |
| 0.75 | 160 | 20.8% |
| 0.975 | 132 | 17.2% |
| 0.1 | 87 | 11.3% |
| 0.5 | 85 | 11.1% |
| 0.05 | 35 | 4.6% |
| 0.025 | 27 | 3.5% |
| 0.95 | 27 | 3.5% |
| 0.9 | 9 | 1.2% |


---

### DOGE (DOGEUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **617** em train, **593** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 418 | 67.7% |
| 4 | 111 | 18.0% |
| 6 | 64 | 10.4% |
| 2 | 19 | 3.1% |
| 3 | 5 | 0.8% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 321 | 54.1% |
| 2 | 261 | 44.0% |
| 4 | 7 | 1.2% |
| 3 | 4 | 0.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 205 | 33.2% |
| 0.25 | 184 | 29.8% |
| 0.5 | 155 | 25.1% |
| 0.95 | 20 | 3.2% |
| 0.75 | 18 | 2.9% |
| 0.025 | 15 | 2.4% |
| 0.05 | 13 | 2.1% |
| 0.9 | 6 | 1.0% |
| 0.1 | 1 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 172 | 29.0% |
| 0.9 | 128 | 21.6% |
| 0.5 | 120 | 20.2% |
| 0.95 | 58 | 9.8% |
| 0.25 | 46 | 7.8% |
| 0.025 | 40 | 6.7% |
| 0.1 | 29 | 4.9% |


---

### ETH (ETHUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **1024** em train, **768** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 380 | 37.1% |
| 6 | 342 | 33.4% |
| 9 | 135 | 13.2% |
| 2 | 104 | 10.2% |
| 4 | 63 | 6.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 457 | 59.5% |
| 2 | 118 | 15.4% |
| 3 | 113 | 14.7% |
| 4 | 53 | 6.9% |
| 6 | 27 | 3.5% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 555 | 54.2% |
| 0.25 | 162 | 15.8% |
| 0.5 | 115 | 11.2% |
| 0.025 | 102 | 10.0% |
| 0.05 | 48 | 4.7% |
| 0.1 | 24 | 2.3% |
| 0.9 | 14 | 1.4% |
| 0.975 | 4 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 350 | 45.6% |
| 0.95 | 183 | 23.8% |
| 0.5 | 72 | 9.4% |
| 0.75 | 70 | 9.1% |
| 0.25 | 50 | 6.5% |
| 0.05 | 20 | 2.6% |
| 0.9 | 14 | 1.8% |
| 0.975 | 9 | 1.2% |


---

### TRX (TRXUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **890** em train, **710** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 298 | 33.5% |
| 3 | 220 | 24.7% |
| 4 | 206 | 23.1% |
| 6 | 166 | 18.7% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 695 | 97.9% |
| 4 | 15 | 2.1% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 285 | 32.0% |
| 0.1 | 220 | 24.7% |
| 0.75 | 111 | 12.5% |
| 0.025 | 92 | 10.3% |
| 0.5 | 52 | 5.8% |
| 0.95 | 52 | 5.8% |
| 0.25 | 51 | 5.7% |
| 0.975 | 23 | 2.6% |
| 0.05 | 4 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 258 | 36.3% |
| 0.9 | 148 | 20.8% |
| 0.95 | 135 | 19.0% |
| 0.5 | 87 | 12.3% |
| 0.25 | 82 | 11.5% |


---

### XLM (XLMUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **897** em train, **714** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 500 | 55.7% |
| 4 | 216 | 24.1% |
| 6 | 108 | 12.0% |
| 2 | 53 | 5.9% |
| 3 | 20 | 2.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 351 | 49.2% |
| 9 | 217 | 30.4% |
| 4 | 86 | 12.0% |
| 6 | 60 | 8.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 316 | 35.2% |
| 0.25 | 229 | 25.5% |
| 0.5 | 186 | 20.7% |
| 0.95 | 59 | 6.6% |
| 0.1 | 37 | 4.1% |
| 0.9 | 27 | 3.0% |
| 0.05 | 25 | 2.8% |
| 0.025 | 14 | 1.6% |
| 0.975 | 4 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.9 | 232 | 32.5% |
| 0.75 | 186 | 26.1% |
| 0.25 | 95 | 13.3% |
| 0.1 | 91 | 12.7% |
| 0.5 | 83 | 11.6% |
| 0.95 | 17 | 2.4% |
| 0.025 | 10 | 1.4% |


---

### XRP (XRPUSDT) — t+1

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `1`  
- Previsões válidas: **916** em train, **722** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 333 | 36.4% |
| 6 | 308 | 33.6% |
| 9 | 229 | 25.0% |
| 4 | 24 | 2.6% |
| 2 | 22 | 2.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 431 | 59.7% |
| 3 | 142 | 19.7% |
| 4 | 119 | 16.5% |
| 6 | 30 | 4.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.25 | 292 | 31.9% |
| 0.75 | 220 | 24.0% |
| 0.5 | 201 | 21.9% |
| 0.95 | 66 | 7.2% |
| 0.975 | 50 | 5.5% |
| 0.9 | 38 | 4.1% |
| 0.025 | 34 | 3.7% |
| 0.1 | 15 | 1.6% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 233 | 32.3% |
| 0.75 | 145 | 20.1% |
| 0.9 | 96 | 13.3% |
| 0.1 | 87 | 12.0% |
| 0.25 | 74 | 10.2% |
| 0.5 | 65 | 9.0% |
| 0.025 | 12 | 1.7% |
| 0.95 | 9 | 1.2% |
| 0.05 | 1 | 0.1% |


---

## Horizonte t+7

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 912 | 725 | 2 | 0.975 | Target |
| BNBUSDT | 1008 | 766 | 9 | 0.975 | Target |
| BTCUSDT | 1008 | 766 | 9 | 0.975 | Target |
| DOGEUSDT | 601 | 591 | 9 | 0.975 | Target |
| ETHUSDT | 1008 | 766 | 9 | 0.95 | Target |
| TRXUSDT | 874 | 708 | 9 | 0.95 | Target |
| XLMUSDT | 881 | 712 | 2 | 0.975 | Target |
| XRPUSDT | 900 | 720 | 9 | 0.975 | Target |


### ADA (ADAUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **912** em train, **725** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 607 | 66.6% |
| 9 | 99 | 10.9% |
| 3 | 86 | 9.4% |
| 6 | 74 | 8.1% |
| 4 | 46 | 5.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 409 | 56.4% |
| 4 | 194 | 26.8% |
| 6 | 82 | 11.3% |
| 2 | 35 | 4.8% |
| 3 | 5 | 0.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 370 | 40.6% |
| 0.975 | 331 | 36.3% |
| 0.95 | 165 | 18.1% |
| 0.9 | 37 | 4.1% |
| 0.75 | 4 | 0.4% |
| 0.25 | 2 | 0.2% |
| 0.5 | 2 | 0.2% |
| 0.1 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 389 | 53.7% |
| 0.95 | 258 | 35.6% |
| 0.9 | 65 | 9.0% |
| 0.025 | 13 | 1.8% |


---

### BNB (BNBUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **1008** em train, **766** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 312 | 31.0% |
| 2 | 263 | 26.1% |
| 4 | 212 | 21.0% |
| 6 | 118 | 11.7% |
| 3 | 103 | 10.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 441 | 57.6% |
| 3 | 150 | 19.6% |
| 4 | 116 | 15.1% |
| 6 | 48 | 6.3% |
| 2 | 11 | 1.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 434 | 43.1% |
| 0.95 | 227 | 22.5% |
| 0.025 | 199 | 19.7% |
| 0.9 | 63 | 6.2% |
| 0.5 | 42 | 4.2% |
| 0.75 | 39 | 3.9% |
| 0.25 | 3 | 0.3% |
| 0.1 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 290 | 37.9% |
| 0.025 | 227 | 29.6% |
| 0.95 | 150 | 19.6% |
| 0.05 | 43 | 5.6% |
| 0.9 | 29 | 3.8% |
| 0.1 | 10 | 1.3% |
| 0.75 | 7 | 0.9% |
| 0.25 | 5 | 0.7% |
| 0.5 | 5 | 0.7% |


---

### BTC (BTCUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **1008** em train, **766** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 336 | 33.3% |
| 3 | 252 | 25.0% |
| 9 | 191 | 18.9% |
| 6 | 135 | 13.4% |
| 4 | 94 | 9.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 449 | 58.6% |
| 2 | 172 | 22.5% |
| 3 | 98 | 12.8% |
| 6 | 39 | 5.1% |
| 4 | 8 | 1.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 647 | 64.2% |
| 0.9 | 112 | 11.1% |
| 0.75 | 95 | 9.4% |
| 0.025 | 74 | 7.3% |
| 0.95 | 60 | 6.0% |
| 0.5 | 7 | 0.7% |
| 0.1 | 5 | 0.5% |
| 0.25 | 5 | 0.5% |
| 0.05 | 3 | 0.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 552 | 72.1% |
| 0.025 | 93 | 12.1% |
| 0.9 | 66 | 8.6% |
| 0.95 | 52 | 6.8% |
| 0.1 | 3 | 0.4% |


---

### DOGE (DOGEUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **601** em train, **591** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 320 | 53.2% |
| 9 | 157 | 26.1% |
| 4 | 82 | 13.6% |
| 2 | 34 | 5.7% |
| 6 | 8 | 1.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 314 | 53.1% |
| 2 | 246 | 41.6% |
| 4 | 21 | 3.6% |
| 6 | 6 | 1.0% |
| 3 | 4 | 0.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 411 | 68.4% |
| 0.75 | 170 | 28.3% |
| 0.025 | 8 | 1.3% |
| 0.95 | 7 | 1.2% |
| 0.9 | 3 | 0.5% |
| 0.5 | 2 | 0.3% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 368 | 62.3% |
| 0.9 | 101 | 17.1% |
| 0.75 | 71 | 12.0% |
| 0.975 | 30 | 5.1% |
| 0.025 | 20 | 3.4% |
| 0.5 | 1 | 0.2% |


---

### ETH (ETHUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **1008** em train, **766** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 381 | 37.8% |
| 9 | 279 | 27.7% |
| 3 | 198 | 19.6% |
| 4 | 148 | 14.7% |
| 6 | 2 | 0.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 499 | 65.1% |
| 2 | 143 | 18.7% |
| 3 | 102 | 13.3% |
| 4 | 22 | 2.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 401 | 39.8% |
| 0.95 | 271 | 26.9% |
| 0.975 | 267 | 26.5% |
| 0.9 | 57 | 5.7% |
| 0.75 | 8 | 0.8% |
| 0.5 | 3 | 0.3% |
| 0.1 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 397 | 51.8% |
| 0.975 | 294 | 38.4% |
| 0.025 | 35 | 4.6% |
| 0.9 | 26 | 3.4% |
| 0.75 | 7 | 0.9% |
| 0.1 | 5 | 0.7% |
| 0.05 | 1 | 0.1% |
| 0.25 | 1 | 0.1% |


---

### TRX (TRXUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **874** em train, **708** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 422 | 48.3% |
| 9 | 229 | 26.2% |
| 3 | 97 | 11.1% |
| 6 | 85 | 9.7% |
| 4 | 41 | 4.7% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 697 | 98.4% |
| 3 | 6 | 0.8% |
| 6 | 5 | 0.7% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 199 | 22.8% |
| 0.9 | 189 | 21.6% |
| 0.975 | 177 | 20.3% |
| 0.025 | 137 | 15.7% |
| 0.1 | 76 | 8.7% |
| 0.25 | 62 | 7.1% |
| 0.75 | 19 | 2.2% |
| 0.5 | 14 | 1.6% |
| 0.05 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 276 | 39.0% |
| 0.95 | 274 | 38.7% |
| 0.5 | 86 | 12.1% |
| 0.9 | 41 | 5.8% |
| 0.75 | 21 | 3.0% |
| 0.025 | 9 | 1.3% |
| 0.25 | 1 | 0.1% |


---

### XLM (XLMUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **881** em train, **712** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 399 | 45.3% |
| 9 | 177 | 20.1% |
| 6 | 142 | 16.1% |
| 3 | 89 | 10.1% |
| 4 | 74 | 8.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 232 | 32.6% |
| 3 | 189 | 26.5% |
| 4 | 174 | 24.4% |
| 6 | 63 | 8.8% |
| 2 | 54 | 7.6% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 318 | 36.1% |
| 0.025 | 267 | 30.3% |
| 0.975 | 250 | 28.4% |
| 0.9 | 32 | 3.6% |
| 0.75 | 7 | 0.8% |
| 0.5 | 3 | 0.3% |
| 0.1 | 2 | 0.2% |
| 0.05 | 1 | 0.1% |
| 0.25 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 246 | 34.6% |
| 0.025 | 132 | 18.5% |
| 0.9 | 129 | 18.1% |
| 0.1 | 77 | 10.8% |
| 0.95 | 71 | 10.0% |
| 0.75 | 29 | 4.1% |
| 0.05 | 23 | 3.2% |
| 0.5 | 4 | 0.6% |
| 0.25 | 1 | 0.1% |


---

### XRP (XRPUSDT) — t+7

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `7`  
- Previsões válidas: **900** em train, **720** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 308 | 34.2% |
| 4 | 175 | 19.4% |
| 6 | 170 | 18.9% |
| 3 | 148 | 16.4% |
| 9 | 99 | 11.0% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 431 | 59.9% |
| 3 | 228 | 31.7% |
| 2 | 46 | 6.4% |
| 6 | 12 | 1.7% |
| 4 | 3 | 0.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 458 | 50.9% |
| 0.95 | 304 | 33.8% |
| 0.025 | 94 | 10.4% |
| 0.9 | 31 | 3.4% |
| 0.75 | 6 | 0.7% |
| 0.25 | 3 | 0.3% |
| 0.5 | 2 | 0.2% |
| 0.05 | 1 | 0.1% |
| 0.1 | 1 | 0.1% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 610 | 84.7% |
| 0.025 | 85 | 11.8% |
| 0.95 | 23 | 3.2% |
| 0.9 | 2 | 0.3% |


---

## Horizonte t+30

### Visão geral

| Moeda | Previsões válidas (train) | Previsões válidas (test) | c modal | ρ modal | Coluna de entrada |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | 850 | 718 | 2 | 0.975 | Target |
| BNBUSDT | 946 | 759 | 2 | 0.975 | Target |
| BTCUSDT | 946 | 759 | 2 | 0.975 | Target |
| DOGEUSDT | 538 | 585 | 2 | 0.975 | Target |
| ETHUSDT | 946 | 759 | 2 | 0.975 | Target |
| TRXUSDT | 811 | 702 | 9 | 0.975 | Target |
| XLMUSDT | 819 | 705 | 2 | 0.025 | Target |
| XRPUSDT | 838 | 713 | 2 | 0.975 | Target |


### ADA (ADAUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **850** em train, **718** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 607 | 71.4% |
| 9 | 109 | 12.8% |
| 3 | 71 | 8.4% |
| 6 | 39 | 4.6% |
| 4 | 24 | 2.8% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 340 | 47.4% |
| 3 | 137 | 19.1% |
| 2 | 120 | 16.7% |
| 6 | 93 | 13.0% |
| 4 | 28 | 3.9% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 398 | 46.8% |
| 0.975 | 292 | 34.4% |
| 0.75 | 53 | 6.2% |
| 0.9 | 42 | 4.9% |
| 0.95 | 35 | 4.1% |
| 0.5 | 19 | 2.2% |
| 0.25 | 7 | 0.8% |
| 0.05 | 2 | 0.2% |
| 0.1 | 2 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 327 | 45.5% |
| 0.025 | 146 | 20.3% |
| 0.95 | 66 | 9.2% |
| 0.75 | 50 | 7.0% |
| 0.9 | 48 | 6.7% |
| 0.5 | 46 | 6.4% |
| 0.1 | 20 | 2.8% |
| 0.25 | 14 | 1.9% |
| 0.05 | 1 | 0.1% |


---

### BNB (BNBUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **946** em train, **759** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 516 | 54.5% |
| 3 | 156 | 16.5% |
| 4 | 150 | 15.9% |
| 9 | 66 | 7.0% |
| 6 | 58 | 6.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 246 | 32.4% |
| 3 | 220 | 29.0% |
| 6 | 217 | 28.6% |
| 4 | 44 | 5.8% |
| 2 | 32 | 4.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 565 | 59.7% |
| 0.025 | 183 | 19.3% |
| 0.75 | 76 | 8.0% |
| 0.95 | 59 | 6.2% |
| 0.9 | 54 | 5.7% |
| 0.5 | 4 | 0.4% |
| 0.25 | 3 | 0.3% |
| 0.1 | 2 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 421 | 55.5% |
| 0.025 | 135 | 17.8% |
| 0.75 | 83 | 10.9% |
| 0.95 | 69 | 9.1% |
| 0.9 | 44 | 5.8% |
| 0.5 | 6 | 0.8% |
| 0.25 | 1 | 0.1% |


---

### BTC (BTCUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **946** em train, **759** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 648 | 68.5% |
| 6 | 153 | 16.2% |
| 3 | 85 | 9.0% |
| 4 | 58 | 6.1% |
| 9 | 2 | 0.2% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 379 | 49.9% |
| 2 | 151 | 19.9% |
| 3 | 103 | 13.6% |
| 6 | 102 | 13.4% |
| 4 | 24 | 3.2% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 461 | 48.7% |
| 0.95 | 221 | 23.4% |
| 0.025 | 174 | 18.4% |
| 0.75 | 42 | 4.4% |
| 0.5 | 24 | 2.5% |
| 0.9 | 22 | 2.3% |
| 0.25 | 2 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 461 | 60.7% |
| 0.95 | 128 | 16.9% |
| 0.025 | 75 | 9.9% |
| 0.5 | 38 | 5.0% |
| 0.75 | 32 | 4.2% |
| 0.9 | 11 | 1.4% |
| 0.25 | 9 | 1.2% |
| 0.05 | 3 | 0.4% |
| 0.1 | 2 | 0.3% |


---

### DOGE (DOGEUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **538** em train, **585** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 4 | 214 | 39.8% |
| 2 | 157 | 29.2% |
| 9 | 81 | 15.1% |
| 6 | 52 | 9.7% |
| 3 | 34 | 6.3% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 6 | 239 | 40.9% |
| 2 | 218 | 37.3% |
| 9 | 108 | 18.5% |
| 3 | 17 | 2.9% |
| 4 | 3 | 0.5% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 198 | 36.8% |
| 0.9 | 125 | 23.2% |
| 0.95 | 123 | 22.9% |
| 0.75 | 46 | 8.6% |
| 0.025 | 26 | 4.8% |
| 0.5 | 9 | 1.7% |
| 0.25 | 7 | 1.3% |
| 0.05 | 2 | 0.4% |
| 0.1 | 2 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 236 | 40.3% |
| 0.975 | 188 | 32.1% |
| 0.95 | 89 | 15.2% |
| 0.9 | 24 | 4.1% |
| 0.05 | 22 | 3.8% |
| 0.75 | 12 | 2.1% |
| 0.5 | 7 | 1.2% |
| 0.1 | 4 | 0.7% |
| 0.25 | 3 | 0.5% |


---

### ETH (ETHUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **946** em train, **759** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 603 | 63.7% |
| 4 | 174 | 18.4% |
| 3 | 111 | 11.7% |
| 9 | 51 | 5.4% |
| 6 | 7 | 0.7% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 433 | 57.0% |
| 2 | 121 | 15.9% |
| 4 | 102 | 13.4% |
| 6 | 65 | 8.6% |
| 3 | 38 | 5.0% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 452 | 47.8% |
| 0.75 | 208 | 22.0% |
| 0.025 | 156 | 16.5% |
| 0.95 | 44 | 4.7% |
| 0.9 | 42 | 4.4% |
| 0.25 | 31 | 3.3% |
| 0.1 | 6 | 0.6% |
| 0.5 | 5 | 0.5% |
| 0.05 | 2 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 242 | 31.9% |
| 0.95 | 221 | 29.1% |
| 0.975 | 166 | 21.9% |
| 0.75 | 65 | 8.6% |
| 0.25 | 18 | 2.4% |
| 0.05 | 17 | 2.2% |
| 0.5 | 14 | 1.8% |
| 0.1 | 9 | 1.2% |
| 0.9 | 7 | 0.9% |


---

### TRX (TRXUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **811** em train, **702** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 499 | 61.5% |
| 6 | 159 | 19.6% |
| 3 | 74 | 9.1% |
| 9 | 70 | 8.6% |
| 4 | 9 | 1.1% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 9 | 451 | 64.2% |
| 6 | 213 | 30.3% |
| 3 | 38 | 5.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 351 | 43.3% |
| 0.975 | 292 | 36.0% |
| 0.95 | 89 | 11.0% |
| 0.75 | 39 | 4.8% |
| 0.05 | 22 | 2.7% |
| 0.9 | 12 | 1.5% |
| 0.5 | 4 | 0.5% |
| 0.1 | 2 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.95 | 242 | 34.5% |
| 0.975 | 217 | 30.9% |
| 0.025 | 129 | 18.4% |
| 0.75 | 80 | 11.4% |
| 0.5 | 12 | 1.7% |
| 0.9 | 9 | 1.3% |
| 0.25 | 6 | 0.9% |
| 0.05 | 5 | 0.7% |
| 0.1 | 2 | 0.3% |


---

### XLM (XLMUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **819** em train, **705** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 443 | 54.1% |
| 3 | 202 | 24.7% |
| 6 | 125 | 15.3% |
| 4 | 46 | 5.6% |
| 9 | 3 | 0.4% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 448 | 63.5% |
| 9 | 134 | 19.0% |
| 6 | 104 | 14.8% |
| 4 | 17 | 2.4% |
| 3 | 2 | 0.3% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 436 | 53.2% |
| 0.75 | 164 | 20.0% |
| 0.95 | 81 | 9.9% |
| 0.9 | 66 | 8.1% |
| 0.975 | 44 | 5.4% |
| 0.5 | 11 | 1.3% |
| 0.25 | 10 | 1.2% |
| 0.05 | 4 | 0.5% |
| 0.1 | 3 | 0.4% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.75 | 225 | 31.9% |
| 0.95 | 216 | 30.6% |
| 0.975 | 113 | 16.0% |
| 0.9 | 46 | 6.5% |
| 0.025 | 44 | 6.2% |
| 0.25 | 31 | 4.4% |
| 0.5 | 19 | 2.7% |
| 0.1 | 10 | 1.4% |
| 0.05 | 1 | 0.1% |


---

### XRP (XRPUSDT) — t+30

- Coluna de entrada: `Target`  
- h interno usado pelo forecaster: `30`  
- Previsões válidas: **838** em train, **713** em test  

**Distribuição do nº de clusters `c` selecionado**

_Split: train_

| c | n | % do tempo |
| --- | --- | --- |
| 2 | 575 | 68.6% |
| 9 | 132 | 15.8% |
| 3 | 74 | 8.8% |
| 4 | 51 | 6.1% |
| 6 | 6 | 0.7% |


_Split: test_

| c | n | % do tempo |
| --- | --- | --- |
| 3 | 388 | 54.4% |
| 9 | 189 | 26.5% |
| 4 | 82 | 11.5% |
| 6 | 37 | 5.2% |
| 2 | 17 | 2.4% |


**Distribuição do parâmetro de memória `ρ` selecionado**

_Split: train_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.025 | 385 | 45.9% |
| 0.975 | 225 | 26.8% |
| 0.95 | 66 | 7.9% |
| 0.9 | 64 | 7.6% |
| 0.75 | 37 | 4.4% |
| 0.5 | 25 | 3.0% |
| 0.25 | 24 | 2.9% |
| 0.05 | 10 | 1.2% |
| 0.1 | 2 | 0.2% |


_Split: test_

| ρ | n | % do tempo |
| --- | --- | --- |
| 0.975 | 517 | 72.5% |
| 0.9 | 84 | 11.8% |
| 0.75 | 82 | 11.5% |
| 0.95 | 30 | 4.2% |


---
