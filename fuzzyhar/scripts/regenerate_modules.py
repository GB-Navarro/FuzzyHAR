# -*- coding: utf-8 -*-
"""
Regenera os modulos VERBATIM do pacote `fuzzyhar/` a partir das celulas do
FuzzyHAR.ipynb, preservando o codigo das funcoes BYTE A BYTE. Apenas:
  - move blocos de codigo para modulos,
  - adiciona cabecalhos de import (resolvendo variaveis livres via fuzzyhar.config),
  - separa o bloco "EXECUCAO" final de cada celula numa funcao run_all_*().

ATENCAO: SOBRESCREVE os modulos gerados (config.py, features.py, models/r_gk.py,
models/fuzzy_har.py, models/har.py, evaluation/metrics_tables.py, evaluation/mcs.py,
pipeline/merge.py, pipeline/plots.py). NAO toca nos arquivos escritos a mao
(base.py, registry.py, runner.py, implementations.py, rvfts.py, __init__.py, tests/).
Serve para re-verificar que os modulos batem com o notebook ou se o notebook mudar.

Uso (working dir = pasta FuzzyHAR/):  python -m fuzzyhar.scripts.regenerate_modules
"""
import json, io, os, textwrap, py_compile

# ROOT = pasta FuzzyHAR/ (este arquivo esta em FuzzyHAR/fuzzyhar/scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB   = os.path.join(ROOT, "FuzzyHAR.ipynb")
PKG  = os.path.join(ROOT, "fuzzyhar")

with io.open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)
CELLS = {i: "".join(c.get("source", []))
         for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"}

TQ = "'" * 3  # triple single quote, evita conflitos de aspas neste script


def strip_magics(src):
    out = []
    for ln in src.splitlines():
        s = ln.lstrip()
        if s.startswith("%") or s.startswith("!"):
            out.append("# [notebook-magic removido] " + ln.strip())
        else:
            out.append(ln)
    return "\n".join(out)


def split_driver(src):
    """Separa (defs+consts, driver) cortando no ULTIMO banner de comentario com 'EXECU'."""
    lines = src.splitlines()
    cut = None
    for i, ln in enumerate(lines):
        st = ln.strip()
        if st.startswith("#") and ("EXECU" in st.upper()):
            cut = i
    if cut is None:
        return src, ""
    j = cut
    while j - 1 >= 0:
        prev = lines[j - 1].strip()
        if prev.startswith("#") and (set(prev) <= set("#=- ")):
            j -= 1
        else:
            break
    body = "\n".join(lines[:j]).rstrip()
    driver = "\n".join(lines[cut:]).rstrip()
    return body, driver


def wrap_driver(driver, fn_name):
    if not driver.strip():
        return ""
    return "\n\ndef " + fn_name + "():\n" + textwrap.indent(driver, "    ") + "\n"


def write(relpath, text):
    path = os.path.join(PKG, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


GEN_HEADER = (
    "# -*- coding: utf-8 -*-\n"
    "# GERADO automaticamente a partir de FuzzyHAR.ipynb (codigo preservado verbatim).\n"
    "# NAO edite o corpo das funcoes: a fidelidade numerica depende disso.\n"
)


def cell(n):
    return strip_magics(CELLS[n])


def doc(text):
    return TQ + text + TQ


# --------------------------------------------------------------------------
# config.py  <- celulas 13, 37, 39
# --------------------------------------------------------------------------
cfg = [
    GEN_HEADER,
    doc("Fonte unica de configuracao (paths, universo de ativos, hiperparametros)."),
    "from dataclasses import dataclass",
    "from pathlib import Path",
    "",
    "# ===== celula 013: HorizonConfig + _build_default_horizon_config =====",
    cell(13),
    "",
    "# ===== celula 037: Path Settings (SPLIT, diretorios, mapas por horizonte) =====",
    cell(37),
    "",
    "# ===== celula 039: Parameters (SYMBOLS, grades, hiperparametros Fuzzy HAR) =====",
    cell(39),
]
write("config.py", "\n".join(cfg))

# --------------------------------------------------------------------------
# features.py <- 19,20,21,23,24
# --------------------------------------------------------------------------
feat = [
    GEN_HEADER,
    doc("Feature engineering e pre-processamento (funcoes puras)."),
    "import numpy as np",
    "import pandas as pd",
    "from typing import Tuple, Union",
    "",
    cell(19), "", cell(20), "", cell(21), "", cell(23), "", cell(24),
]
write("features.py", "\n".join(feat))

# --------------------------------------------------------------------------
# models/r_gk.py <- 25 (R) + 26
# --------------------------------------------------------------------------
r_src_lines = CELLS[25].splitlines()
while r_src_lines and (r_src_lines[0].strip() == "" or r_src_lines[0].strip().startswith("%%")):
    r_src_lines.pop(0)
r_src = "\n".join(r_src_lines)
rgk = [
    GEN_HEADER,
    doc("Ponte rpy2 para o clustering Gustafson-Kessel (roda em R).\n\n"
        "A celula %%R do notebook definia gk_clustering no ambiente global do R.\n"
        "Aqui o mesmo codigo R e definido UMA vez, no import deste modulo, via r(_GK_R_SRC),\n"
        "preservando o set.seed(123) e o corpo R byte a byte. run_gk_clustering e verbatim."),
    "import numpy as np",
    "import pandas as pd",
    "from typing import Tuple",
    "from rpy2.robjects import conversion, globalenv, pandas2ri, r",
    "from rpy2.robjects.conversion import localconverter",
    "",
    "_GK_R_SRC = r" + TQ,
    r_src,
    TQ,
    "",
    "# Define gk_clustering no ambiente R (equivalente a executar a celula %%R uma vez).",
    "r(_GK_R_SRC)",
    "",
    cell(26),
]
write("models/r_gk.py", "\n".join(rgk))

# --------------------------------------------------------------------------
# models/fuzzy_har.py <- 14,15,16,17 + 27,28,30,31,32,33,34,44 + driver 45
# --------------------------------------------------------------------------
fuzzy = [
    GEN_HEADER,
    doc("Pipeline Fuzzy HAR (clustering GK em R + ALSM). Codigo verbatim do notebook."),
    "import os",
    "import time",
    "import numpy as np",
    "import pandas as pd",
    "from typing import Any, Dict, Tuple",
    "",
    "from fuzzyhar.config import (",
    "    HORIZON_CONFIG, SPLIT, rules_grid, ze, lamb, alfa, week_len, month_len,",
    ")",
    "from fuzzyhar.features import make_har_features, gaussgranules",
    "from fuzzyhar.models.r_gk import run_gk_clustering",
    "",
    cell(14), "", cell(15), "", cell(16), "", cell(17), "",
    cell(27), "", cell(28), "", cell(30), "", cell(31), "",
    cell(32), "", cell(33), "", cell(34), "", cell(44),
    wrap_driver(cell(45), "run_all_fuzzy"),
]
write("models/fuzzy_har.py", "\n".join(fuzzy))

# --------------------------------------------------------------------------
# models/har.py <- 47
# --------------------------------------------------------------------------
body47, drv47 = split_driver(cell(47))
har = [
    GEN_HEADER,
    doc("Modelos HAR classicos e variantes (OOS). Codigo verbatim do notebook."),
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import statsmodels.api as sm",
    "from pathlib import Path",
    "",
    "from fuzzyhar.config import (",
    "    SPLIT, SYMBOLS, get_input_dir_from_folder, get_prediction_output_dir_from_folder,",
    ")",
    "",
    body47,
    wrap_driver(drv47, "run_all_har"),
]
write("models/har.py", "\n".join(har))

# --------------------------------------------------------------------------
# pipeline/merge.py <- 50
# --------------------------------------------------------------------------
body50, drv50 = split_driver(cell(50))
merge = [
    GEN_HEADER,
    doc("Merge das previsoes (classicos + Fuzzy). Codigo verbatim do notebook."),
    "import re",
    "import pandas as pd",
    "from pathlib import Path",
    "from typing import Dict, Iterable, List, Tuple",
    "",
    "from fuzzyhar.config import (",
    "    SYMBOLS, get_fuzzy_prediction_output_dir, get_prediction_output_dir,",
    ")",
    "",
    body50,
    wrap_driver(drv50, "run_all_merge"),
]
write("pipeline/merge.py", "\n".join(merge))

# --------------------------------------------------------------------------
# evaluation/metrics_tables.py <- 52
# --------------------------------------------------------------------------
body52, drv52 = split_driver(cell(52))
mt = [
    GEN_HEADER,
    doc("Tabelas de MSE & QLIKE por horizonte. Codigo verbatim do notebook."),
    "import numpy as np",
    "import pandas as pd",
    "from pathlib import Path",
    "from typing import Dict, Iterable, Optional, Sequence, Tuple, Union",
    "",
    "from fuzzyhar.config import SYMBOLS",
    "",
    body52,
    wrap_driver(drv52, "run_all_metrics"),
]
write("evaluation/metrics_tables.py", "\n".join(mt))

# --------------------------------------------------------------------------
# pipeline/plots.py <- 54
# --------------------------------------------------------------------------
body54, drv54 = split_driver(cell(54))
plots = [
    GEN_HEADER,
    doc("Graficos finais OOS (Actual vs HAR vs Fuzzy). Codigo verbatim do notebook."),
    "from fuzzyhar.config import SYMBOLS",
    "",
    body54,
    wrap_driver(drv54, "run_all_plots"),
]
write("pipeline/plots.py", "\n".join(plots))

# --------------------------------------------------------------------------
# evaluation/mcs.py <- 56
# --------------------------------------------------------------------------
body56, drv56 = split_driver(cell(56))
mcs = [
    GEN_HEADER,
    doc("Model Confidence Set (arch). Codigo verbatim do notebook."),
    "import numpy as np",
    "import pandas as pd",
    "from pathlib import Path",
    "from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union",
    "",
    "from fuzzyhar.config import SYMBOLS",
    "",
    body56,
    wrap_driver(drv56, "run_all_mcs"),
]
write("evaluation/mcs.py", "\n".join(mcs))

# --------------------------------------------------------------------------
# py_compile de tudo (sem importar -> nao exige rpy2/dados)
# --------------------------------------------------------------------------
print("== py_compile ==")
ok = True
for dirpath, _, files in os.walk(PKG):
    for fn in sorted(files):
        if fn.endswith(".py"):
            p = os.path.join(dirpath, fn)
            try:
                py_compile.compile(p, doraise=True)
                print("OK  ", os.path.relpath(p, ROOT))
            except py_compile.PyCompileError as e:
                ok = False
                print("FAIL", os.path.relpath(p, ROOT))
                print(e)
print("ALL OK" if ok else "ERRORS PRESENT")
