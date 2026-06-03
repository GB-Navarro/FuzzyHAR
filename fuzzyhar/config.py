# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Fonte unica de configuracao (paths, universo de ativos, hiperparametros).'''
from dataclasses import dataclass
from pathlib import Path

# ===== celula 013: HorizonConfig + _build_default_horizon_config =====
@dataclass(frozen=True)
class HorizonConfig:
    horizon: int
    base_input_dir: str
    base_output_dir: str


def _build_default_horizon_config() -> dict[int, HorizonConfig]:
    """
    Monta o mapa de configuraÃ§Ã£o padrÃ£o por horizonte.
    """
    return {
        1: HorizonConfig(
            horizon=1,
            base_input_dir=input_RV_t1,
            base_output_dir=predictions_output_RV_t1,
        ),
        7: HorizonConfig(
            horizon=7,
            base_input_dir=input_RV_t7,
            base_output_dir=predictions_output_RV_t7,
        ),
        30: HorizonConfig(
            horizon=30,
            base_input_dir=input_RV_t30,
            base_output_dir=predictions_output_RV_t30,
        ),
    }

# ===== celula 037: Path Settings (SPLIT, diretorios, mapas por horizonte) =====
# ── PARÂMETRO DE SPLIT (única variável a mudar entre execuções) ─────────
# Opções disponíveis: "40_60" | "60_40" | "70_30"
SPLIT = "70_30"

# ----------------------------
# Caminhos por horizonte
# ----------------------------

main_input = Path("InputsNew") / "RVs" / SPLIT
main_output = Path("Results")
predictions_root = Path("Others") / "Predictions"

# Raiz dedicada para cada família de modelos (inclui subfolder do split)
predictions_fuzzy_root = predictions_root / SPLIT / "Fuzzy HAR"
predictions_har_models_root = predictions_root / SPLIT / "HAR Models"

eda_output_dir = main_output

input_RV_t1 = main_input / "RV_t+1"
input_RV_t7 = main_input / "RV_t+7"
input_RV_t30 = main_input / "RV_t+30"

# Fuzzy HAR (t+1, t+7, t+30)
predictions_output_fuzzy_RV_t1 = predictions_fuzzy_root / "RV_t+1"
predictions_output_fuzzy_RV_t7 = predictions_fuzzy_root / "RV_t+7"
predictions_output_fuzzy_RV_t30 = predictions_fuzzy_root / "RV_t+30"

# HAR Models (t+1, t+7, t+30)
predictions_output_har_RV_t1 = predictions_har_models_root / "RV_t+1"
predictions_output_har_RV_t7 = predictions_har_models_root / "RV_t+7"
predictions_output_har_RV_t30 = predictions_har_models_root / "RV_t+30"

# Mantido para compatibilidade com HORIZON_CONFIG (pipeline Fuzzy HAR)
predictions_output_RV_t1 = predictions_output_fuzzy_RV_t1
predictions_output_RV_t7 = predictions_output_fuzzy_RV_t7
predictions_output_RV_t30 = predictions_output_fuzzy_RV_t30

results_output_RV_t1 = main_output / "RV_t+1"
results_output_RV_t7 = main_output / "RV_t+7"
results_output_RV_t30 = main_output / "RV_t+30"

# Diretórios de saída dos modelos HAR (usados pelo bloco HAR Models)
PREDICTIONS_OUTPUT_BY_HORIZON = {
    1: predictions_output_har_RV_t1,
    7: predictions_output_har_RV_t7,
    30: predictions_output_har_RV_t30,
}

# Diretórios de saída do Fuzzy HAR
FUZZY_PREDICTIONS_OUTPUT_BY_HORIZON = {
    1: predictions_output_fuzzy_RV_t1,
    7: predictions_output_fuzzy_RV_t7,
    30: predictions_output_fuzzy_RV_t30,
}

INPUT_BY_HORIZON = {
    1: input_RV_t1,
    7: input_RV_t7,
    30: input_RV_t30,
}

RESULTS_BY_HORIZON = {
    1: results_output_RV_t1,
    7: results_output_RV_t7,
    30: results_output_RV_t30,
}

PAPER_PLOTS_OUTPUT_BY_HORIZON = {
    1: predictions_output_fuzzy_RV_t1 / "_paper_plots",
    7: predictions_output_fuzzy_RV_t7 / "_paper_plots",
    30: predictions_output_fuzzy_RV_t30 / "_paper_plots",
}

METRICS_OUTPUT_BY_HORIZON = {
    1: results_output_RV_t1 / "_comparison_metrics",
    7: results_output_RV_t7 / "_comparison_metrics",
    30: results_output_RV_t30 / "_comparison_metrics",
}

def _horizon_key_from_folder(horizon_folder: str) -> int:
    return int(str(horizon_folder).split("+")[-1])

def get_prediction_output_dir(horizon: int) -> Path:
    return PREDICTIONS_OUTPUT_BY_HORIZON[horizon]

def get_fuzzy_prediction_output_dir(horizon: int) -> Path:
    return FUZZY_PREDICTIONS_OUTPUT_BY_HORIZON[horizon]

def get_prediction_output_dir_from_folder(horizon_folder: str) -> Path:
    return get_prediction_output_dir(_horizon_key_from_folder(horizon_folder))

def get_fuzzy_prediction_output_dir_from_folder(horizon_folder: str) -> Path:
    return get_fuzzy_prediction_output_dir(_horizon_key_from_folder(horizon_folder))

def get_input_dir_from_folder(horizon_folder: str) -> Path:
    return INPUT_BY_HORIZON[_horizon_key_from_folder(horizon_folder)]


# ----------------------------
# Mapa de horizontes
# ----------------------------

HORIZON_CONFIG = _build_default_horizon_config()

# ===== celula 039: Parameters (SYMBOLS, grades, hiperparametros Fuzzy HAR) =====
# ----------------------------
# Universo de criptomoedas (FONTE ÃšNICA)
# ----------------------------

SYMBOLS = [
    "ADAUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "TRXUSDT",
    "XLMUSDT",
    "XRPUSDT",
]


# ----------------------------
# Grades de busca
# ----------------------------

rules_grid = [2, 3, 4, 5, 6]

# ----------------------------
# ParÃ¢metros do Fuzzy HAR
# ----------------------------

week_len = 7
month_len = 30
lamb = 0.99
alfa = 1000
ze = 1
