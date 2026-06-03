# -*- coding: utf-8 -*-
"""Congela um BASELINE dos resultados atuais (golden master).

Execute este script com o codigo ATUAL (antes de confiar na refatoracao) para
capturar um snapshot estavel, comparavel por VALOR (nao por bytes de Excel).
Depois, rode `compare_baseline.py` para verificar se a versao refatorada produz
exatamente os mesmos numeros.

Como usar (working dir = pasta FuzzyHAR/):

    python -m fuzzyhar.scripts.freeze_baseline --label antes
    # ... rode a refatoracao / regenere Others/ e Results/ ...
    python -m fuzzyhar.scripts.freeze_baseline --label depois
    python -m fuzzyhar.scripts.compare_baseline antes depois

O baseline normaliza todos os .xlsx/.csv de Others/ e Results/ para parquet,
com colunas e ordem estaveis, dentro de baseline/<label>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

SCAN_DIRS = ["Others", "Results"]
BASELINE_ROOT = Path("baseline")


def _read_tabular(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception as exc:  # arquivo nao tabular / corrompido -> ignora
        print(f"  [skip] {path}: {exc}")
    return None


def freeze(label: str) -> None:
    out_root = BASELINE_ROOT / label
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    n = 0
    for d in SCAN_DIRS:
        base = Path(d)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in (".xlsx", ".xls", ".csv"):
                continue
            df = _read_tabular(path)
            if df is None:
                continue
            # normaliza para comparacao por valor
            df = df.reset_index(drop=True)
            rel = path.as_posix().replace("/", "__")
            dest = out_root / (rel + ".parquet")
            try:
                df.to_parquet(dest, index=False)
            except Exception:
                # fallback: csv com float_format fixo
                dest = out_root / (rel + ".csv")
                df.to_csv(dest, index=False, float_format="%.12g")
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(df, index=True).values.tobytes()
            ).hexdigest()
            manifest[path.as_posix()] = {
                "rows": int(df.shape[0]),
                "cols": list(map(str, df.columns)),
                "value_hash": digest,
            }
            n += 1
    (out_root / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[freeze] {n} arquivos capturados em {out_root}")
    print(f"[freeze] manifest: {out_root / '_manifest.json'}")
    print("[freeze] DICA: salve tambem o ambiente:")
    print("         pip freeze > " + str(out_root / "requirements.lock"))
    print('         Rscript -e "writeLines(capture.output(sessionInfo()))" > '
          + str(out_root / "R_sessioninfo.txt"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Congela baseline dos resultados (golden master).")
    p.add_argument("--label", required=True, help='rotulo do snapshot, ex.: "antes" ou "depois"')
    args = p.parse_args(argv)
    freeze(args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
