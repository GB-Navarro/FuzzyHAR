# -*- coding: utf-8 -*-
# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).
# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.

'''Modelos HAR classicos e variantes (OOS). Codigo verbatim do notebook.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from fuzzyhar.config import (
    SPLIT, SYMBOLS, get_input_dir_from_folder, get_prediction_output_dir_from_folder,
)

pd.set_option("display.max_columns", 120)

pd.set_option("display.width", 160)

np.seterr(all="ignore")



# ============================================================

# CONFIGURAÃ‡ÃƒO

# ============================================================



HORIZON_FOLDERS = ["RV_t+1", "RV_t+7", "RV_t+30"]



# Modelos a avaliar

MODELOS = ["HAR", "HAR-CJ", "HAR-TCJ", "LHAR-TCJ", "HAR-SJ"]

# MODELOS = ["HAR"]  # use isso se quiser testar sÃ³ HAR primeiro



# EstratÃ©gia OOS

SCHEME = "rolling"         # "expanding" ou "rolling"

ROLLING_WINDOW = 730       # usado apenas se SCHEME="rolling"

REESTIMAR_A_CADA_PASSO = True



# SeguranÃ§a numÃ©rica para QLIKE

EPS = 1e-12



# PadrÃµes de nomes esperados

def _get_file_suffixes(split: str) -> tuple[str, str]:
    if split == "40_60":
        return "_5m_daily_insample.xlsx", "_5m_daily_outofsample.xlsx"
    return (
        f"_5m_daily__SPLIT_{split}__insample_train.xlsx",
        f"_5m_daily__SPLIT_{split}__oos_predictions.xlsx",
    )

INSAMPLE_SUFFIX, OOS_SUFFIX = _get_file_suffixes(SPLIT)



# Regressoras esperadas por modelo

model_variables_dict = {

    "HAR":      ["RV_t-1", "RV_t-7", "RV_t-30"],

    "HAR-CJ":   ["C_d", "C_d_7d", "C_d_30d", "J_t"],

    "HAR-TCJ":  ["TC_d", "TC_d_7d", "TC_d_30d", "TJ_t"],

    "LHAR-TCJ": ["TC_d", "TC_d_7d", "TC_d_30d", "TJ_t", "exp_r-_d"],

    "HAR-SJ":   ["RV_t-7", "RV_t-30", "BPV_d", "exp_SJ+_t", "exp_SJ-_t"],

}



# ============================================================

# FUNÃ‡Ã•ES AUXILIARES

# ============================================================



def mse(y, f):

    return float(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2))



def mae(y, f):

    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))



def rmse(y, f):

    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2)))



def qlike(y, f):

    y_pos = np.clip(np.asarray(y, float), EPS, None)

    f_pos = np.clip(np.asarray(f, float), EPS, None)

    return float(np.mean(np.log(f_pos) + y_pos / f_pos))



def _save_ts_plot(x, series, title, ylabel, out_path):

    fig = plt.figure()

    plt.plot(x, series)

    plt.title(title)

    plt.xlabel("Data")

    plt.ylabel(ylabel)

    plt.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)



def _save_scatter(x, y, title, xlabel, ylabel, out_path):

    fig = plt.figure()

    plt.scatter(x, y, s=10)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)



def gerar_graficos_oos(test_df, y_test, yhat_test, rep_dir, base_name, model_name):

    plots_dir = rep_dir / "plots_oos"

    plots_dir.mkdir(parents=True, exist_ok=True)



    dates = test_df["Data"].values

    _save_ts_plot(

        dates, y_test,

        f"OOS â€” Target â€” {base_name} [{model_name}]",

        "Target",

        plots_dir / f"{base_name}__{model_name}__oos_target.png"

    )

    _save_ts_plot(

        dates, yhat_test,

        f"OOS â€” Prediction â€” {base_name} [{model_name}]",

        "Prediction",

        plots_dir / f"{base_name}__{model_name}__oos_prediction.png"

    )

    _save_scatter(

        y_test, yhat_test,

        f"OOS â€” Target vs Prediction â€” {base_name} [{model_name}]",

        "Target", "Prediction",

        plots_dir / f"{base_name}__{model_name}__oos_target_vs_pred.png"

    )

    _save_ts_plot(

        dates, np.asarray(y_test) - np.asarray(yhat_test),

        f"OOS â€” ResÃ­duos â€” {base_name} [{model_name}]",

        "ResÃ­duo",

        plots_dir / f"{base_name}__{model_name}__oos_residuos.png"

    )



def _standardize_input_df(df: pd.DataFrame, df_name: str) -> pd.DataFrame:

    """

    Padroniza nomes de colunas para o formato esperado por este cÃ³digo.

    """

    out = df.copy()



    # Data

    if "Data" not in out.columns and "Date" in out.columns:

        out = out.rename(columns={"Date": "Data"})



    # target

    if "target" not in out.columns and "Target" in out.columns:

        out = out.rename(columns={"Target": "target"})



    # retorno_diario Ã© Ãºtil para outputs/grÃ¡ficos; se nÃ£o existir, cria vazio

    if "retorno_diario" not in out.columns:

        out["retorno_diario"] = np.nan



    required = {"Data", "target"}

    missing = required - set(out.columns)

    if missing:

        raise ValueError(f"{df_name}: faltam colunas obrigatÃ³rias: {missing}")



    out["Data"] = pd.to_datetime(out["Data"], errors="coerce")

    out = out.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)



    return out



def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:

    """

    Deriva apenas colunas exponenciadas que o script original jÃ¡ sabia criar.

    """

    out = df.copy()



    if "exp_r-_d" not in out.columns and "r-_d" in out.columns:

        out["exp_r-_d"] = np.exp(out["r-_d"])



    if "exp_SJ+_t" not in out.columns and "SJ+_t" in out.columns:

        out["exp_SJ+_t"] = np.exp(out["SJ+_t"])



    if "exp_SJ-_t" not in out.columns and "SJ-_t" in out.columns:

        out["exp_SJ-_t"] = np.exp(out["SJ-_t"])



    return out



def _extract_symbol_from_filename(filename: str, suffix: str) -> str:

    if not filename.endswith(suffix):

        raise ValueError(f"Arquivo com nome inesperado: {filename}")

    return filename[:-len(suffix)]



def _build_output_paths(horizon_folder: str, symbol: str):

    """

    Exemplo:

    /content/drive/MyDrive/ProductionFuzzyHARPipeline/Others/Predictions_RV_t+1/BTCUSDT

    """

    symbol_root = (

        get_prediction_output_dir_from_folder(horizon_folder)

        / symbol

    )

    train_dir = symbol_root / "insample_train"

    oos_dir = symbol_root / "out-of-sample"



    train_dir.mkdir(parents=True, exist_ok=True)

    oos_dir.mkdir(parents=True, exist_ok=True)



    return {

        "symbol_root": symbol_root,

        "train_dir": train_dir,

        "oos_dir": oos_dir,

    }



# ============================================================

# NÃšCLEO: RODA OOS PARA UM PAR (TREINO + TESTE)

# ============================================================



def oos_modelos_por_par(

    train_file: str | Path,

    test_file: str | Path,

    modelos: list[str] | None = None,

    scheme: str = "expanding",

    rolling_window: int | None = None,

    reestimar_a_cada_passo: bool = True,

) -> None:

    """

    Roda os modelos OLS OOS usando:

      - train_file = insample

      - test_file  = out_of_sample



    Sem refazer split interno.

    """

    train_path = Path(train_file)

    test_path = Path(test_file)



    # Identifica horizonte e sÃ­mbolo

    # Esperado:

    # .../RV_t+1/insample/SYMBOL_5m_daily_insample.xlsx

    # .../RV_t+1/out_of_sample/SYMBOL_5m_daily_outofsample.xlsx

    horizon_folder = train_path.parent.parent.name

    if horizon_folder != test_path.parent.parent.name:

        raise ValueError(

            f"Treino e teste pertencem a horizontes diferentes: "

            f"{horizon_folder} vs {test_path.parent.parent.name}"

        )



    symbol_train = _extract_symbol_from_filename(train_path.name, INSAMPLE_SUFFIX)

    symbol_test = _extract_symbol_from_filename(test_path.name, OOS_SUFFIX)



    if symbol_train != symbol_test:

        raise ValueError(

            f"SÃ­mbolo do treino difere do teste: {symbol_train} vs {symbol_test}"

        )



    symbol = symbol_train

    base_name = symbol



    outputs = _build_output_paths(horizon_folder, symbol)



    # ---------- leitura ----------

    df_train = pd.read_excel(train_path)

    df_test = pd.read_excel(test_path)



    df_train = _standardize_input_df(df_train, f"Treino ({train_path.name})")

    df_test = _standardize_input_df(df_test, f"Teste ({test_path.name})")



    df_train = _ensure_derived_columns(df_train)

    df_test = _ensure_derived_columns(df_test)



    if modelos is None:

        modelos = list(model_variables_dict.keys())



    # ---------- roda modelo a modelo ----------

    for model_name in modelos:

        vars_needed = model_variables_dict[model_name]



        missing_cols = [

            c for c in vars_needed

            if c not in df_train.columns or c not in df_test.columns

        ]

        if missing_cols:

            print(

                f"âš ï¸ {symbol} | {horizon_folder} | {model_name}: "

                f"pulando (faltam colunas: {missing_cols})"

            )

            continue



        # filtra linhas vÃ¡lidas separadamente em treino e teste

        train_valid_mask = df_train[vars_needed].notna().all(axis=1) & df_train["target"].notna()

        test_valid_mask = df_test[vars_needed].notna().all(axis=1) & df_test["target"].notna()



        if not train_valid_mask.any():

            print(f"âš ï¸ {symbol} | {horizon_folder} | {model_name}: treino sem linhas vÃ¡lidas.")

            continue



        if not test_valid_mask.any():

            print(f"âš ï¸ {symbol} | {horizon_folder} | {model_name}: teste sem linhas vÃ¡lidas.")

            continue



        train_valid = df_train.loc[train_valid_mask].copy().reset_index(drop=True)

        test_valid = df_test.loc[test_valid_mask].copy().reset_index(drop=True)



        X_train0 = train_valid[vars_needed].copy()

        y_train0 = train_valid["target"].copy()



        X_test = test_valid[vars_needed].copy()

        y_test = test_valid["target"].copy()



        df_test_meta = test_valid[["Data", "retorno_diario"]].copy()

        test_dates = test_valid["Data"].values



        if X_train0.empty or X_test.empty:

            print(f"âš ï¸ {symbol} | {horizon_folder} | {model_name}: treino ou teste vazios.")

            continue



        # ---------- (1) Fit no treino inteiro ----------

        Xc_train = sm.add_constant(X_train0, has_constant="add")

        mdl_train = sm.OLS(y_train0, Xc_train).fit(

            cov_type="HAC",

            cov_kwds={"maxlags": 5},

            use_t=True

        )

        yhat_train = mdl_train.predict(Xc_train).values



        # ---------- salva treino ----------

        train_sheet = train_valid[["Data", "retorno_diario"]].copy()

        train_sheet["Target"] = y_train0.values

        train_sheet[f"{model_name} train prediction"] = yhat_train



        train_out = outputs["train_dir"] / f"{base_name}__{model_name}__insample_train.xlsx"

        train_sheet.to_excel(train_out, index=False)



        # ---------- (2) previsÃ£o OOS ----------

        yhat_oos = np.full(len(X_test), np.nan, dtype=float)



        if not reestimar_a_cada_passo:

            Xc_test = sm.add_constant(X_test, has_constant="add")

            yhat_oos = mdl_train.predict(Xc_test).values

        else:

            # histÃ³rico total elegÃ­vel = treino vÃ¡lido + teste vÃ¡lido

            # a cada data de teste d, usa apenas observaÃ§Ãµes com Data < d

            all_valid = pd.concat([train_valid, test_valid], axis=0, ignore_index=True)

            all_valid = all_valid.sort_values("Data").reset_index(drop=True)



            X_all = all_valid[vars_needed].copy()

            y_all = all_valid["target"].copy()

            data_values = all_valid["Data"].values



            scheme_l = (scheme or "expanding").lower()

            use_rolling = (

                (scheme_l == "rolling") and

                (rolling_window is not None) and

                (int(rolling_window) > 0)

            )

            roll_n = int(rolling_window) if use_rolling else None



            for i, d in enumerate(test_dates):

                hist_mask = (data_values < d)



                if not hist_mask.any():

                    x_row = X_test.iloc[i:i+1, :]

                    Xc_row = sm.add_constant(x_row, has_constant="add")

                    yhat_oos[i] = mdl_train.predict(Xc_row).item()

                    continue



                if use_rolling:

                    hist_idx = np.flatnonzero(hist_mask)

                    if len(hist_idx) > roll_n:

                        hist_idx = hist_idx[-roll_n:]

                    X_hist = X_all.iloc[hist_idx, :]

                    y_hist = y_all.iloc[hist_idx]

                else:

                    X_hist = X_all.loc[hist_mask, :]

                    y_hist = y_all.loc[hist_mask]



                if X_hist.empty:

                    x_row = X_test.iloc[i:i+1, :]

                    Xc_row = sm.add_constant(x_row, has_constant="add")

                    yhat_oos[i] = mdl_train.predict(Xc_row).item()

                    continue



                Xc_hist = sm.add_constant(X_hist, has_constant="add")

                mdl_hist = sm.OLS(y_hist, Xc_hist).fit(

                    cov_type="HAC",

                    cov_kwds={"maxlags": 5},

                    use_t=True

                )



                x_row = X_test.iloc[i:i+1, :]

                Xc_row = sm.add_constant(x_row, has_constant="add")

                yhat_oos[i] = mdl_hist.predict(Xc_row).item()



        # ---------- saÃ­das OOS ----------

        out_pred = outputs["oos_dir"] / f"{base_name}__{model_name}__oos_predictions.xlsx"



        saida = df_test_meta.copy()

        saida["Target"] = y_test.values

        saida[f"{model_name} OOS prediction"] = yhat_oos

        saida.to_excel(out_pred, index=False)



        print(

            f"âœ… {symbol} | {horizon_folder} | {model_name}: "

            f"OOS salvo em {out_pred}"

        )



# ============================================================

# DESCOBRE OS PARES INSAMPLE / OUT_OF_SAMPLE

# ============================================================



def encontrar_pares_por_horizonte(horizon_folder: str):

    base_input_dir = get_input_dir_from_folder(horizon_folder)

    ins_dir = base_input_dir / "insample"

    oos_dir = base_input_dir / "out_of_sample"



    if not ins_dir.exists():

        raise FileNotFoundError(f"Pasta nÃ£o encontrada: {ins_dir}")

    if not oos_dir.exists():

        raise FileNotFoundError(f"Pasta nÃ£o encontrada: {oos_dir}")



    ins_files = sorted(ins_dir.glob(f"*{INSAMPLE_SUFFIX}"))

    oos_files = sorted(oos_dir.glob(f"*{OOS_SUFFIX}"))



    ins_map = {

        _extract_symbol_from_filename(f.name, INSAMPLE_SUFFIX): f

        for f in ins_files

    }

    oos_map = {

        _extract_symbol_from_filename(f.name, OOS_SUFFIX): f

        for f in oos_files

    }



    common_symbols = sorted(set(ins_map) & set(oos_map))

    missing_in_oos = sorted(set(ins_map) - set(oos_map))

    missing_in_ins = sorted(set(oos_map) - set(ins_map))



    if missing_in_oos:

        print(f"âš ï¸ {horizon_folder}: sem arquivo de teste para: {missing_in_oos}")

    if missing_in_ins:

        print(f"âš ï¸ {horizon_folder}: sem arquivo de treino para: {missing_in_ins}")



    pairs = [(sym, ins_map[sym], oos_map[sym]) for sym in common_symbols]

    return pairs



def oos_varios_horizontes_split_ready(
    horizon_folders: list[str] | None = None,
    modelos: list[str] | None = None,
    scheme: str = "expanding",
    rolling_window: int | None = None,
    reestimar_a_cada_passo: bool = True,
    symbols_to_run: list[str] | None = None,
):
    if horizon_folders is None:
        horizon_folders = HORIZON_FOLDERS
    if symbols_to_run is None:
        symbols_to_run = list(SYMBOLS)

    allowed_symbols = set(symbols_to_run)

    total_pairs = 0
    for horizon_folder in horizon_folders:
        pairs = encontrar_pares_por_horizonte(horizon_folder)
        pairs = [p for p in pairs if p[0] in allowed_symbols]

        print(f"▶️ {horizon_folder}: encontrados {len(pairs)} par(es) treino/teste.")
        total_pairs += len(pairs)

        for symbol, train_file, test_file in pairs:
            try:
                oos_modelos_por_par(
                    train_file=train_file,
                    test_file=test_file,
                    modelos=modelos,
                    scheme=scheme,
                    rolling_window=rolling_window,
                    reestimar_a_cada_passo=reestimar_a_cada_passo,
                )
            except Exception as e:
                print(f"❌ Erro em {symbol} | {horizon_folder}: {e}")

    print(f"\n✅ Concluído. Total de pares processados: {total_pairs}")


def run_all_har():
    # EXECUÇÃO
    # ============================================================

    oos_varios_horizontes_split_ready(
        horizon_folders=HORIZON_FOLDERS,
        modelos=MODELOS,
        scheme=SCHEME,
        rolling_window=ROLLING_WINDOW,
        reestimar_a_cada_passo=REESTIMAR_A_CADA_PASSO,
        symbols_to_run=SYMBOLS,
    )
