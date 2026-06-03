# -*- coding: utf-8 -*-
"""Compara dois snapshots de baseline VALOR A VALOR.

    python -m fuzzyhar.scripts.compare_baseline antes depois

Saida: lista os arquivos cujos VALORES divergem (com a primeira coluna/linha
que difere), em vez de comparar bytes de Excel (que sempre diferem por causa de
timestamps embutidos). Codigo de saida != 0 se houver qualquer divergencia.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASELINE_ROOT = Path("baseline")
RTOL = 1e-9
ATOL = 1e-12


def _load(label: str, rel_key_file: str) -> pd.DataFrame:
    p = BASELINE_ROOT / label / rel_key_file
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def compare(label_a: str, label_b: str) -> int:
    man_a = json.loads((BASELINE_ROOT / label_a / "_manifest.json").read_text(encoding="utf-8"))
    man_b = json.loads((BASELINE_ROOT / label_b / "_manifest.json").read_text(encoding="utf-8"))

    keys_a, keys_b = set(man_a), set(man_b)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    diffs = 0

    for k in sorted(only_a):
        print(f"[FALTA em {label_b}] {k}")
        diffs += 1
    for k in sorted(only_b):
        print(f"[NOVO em {label_b}]  {k}")
        diffs += 1

    for k in sorted(keys_a & keys_b):
        if man_a[k]["value_hash"] == man_b[k]["value_hash"]:
            continue
        # hash difere -> investiga valor a valor
        rel = k.replace("/", "__")
        fa = _resolve(label_a, rel)
        fb = _resolve(label_b, rel)
        if fa is None or fb is None:
            print(f"[DIVERGE] {k} (hash difere; arquivo nao recarregavel)")
            diffs += 1
            continue
        da, db = _load(label_a, fa), _load(label_b, fb)
        try:
            pd.testing.assert_frame_equal(
                da.reset_index(drop=True), db.reset_index(drop=True),
                rtol=RTOL, atol=ATOL, check_like=True, check_dtype=False,
            )
            # hash diferente mas valores iguais sob tolerancia -> ok
        except AssertionError as exc:
            first = str(exc).splitlines()[0]
            print(f"[DIVERGE] {k}\n           {first}")
            diffs += 1

    if diffs == 0:
        print(f"OK — '{label_a}' e '{label_b}' identicos (rtol={RTOL}, atol={ATOL}).")
        return 0
    print(f"\n{diffs} divergencia(s) encontrada(s).")
    return 1


def _resolve(label: str, rel: str) -> str | None:
    for ext in (".parquet", ".csv"):
        if (BASELINE_ROOT / label / (rel + ext)).exists():
            return rel + ext
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("uso: python -m fuzzyhar.scripts.compare_baseline <label_a> <label_b>")
        return 2
    return compare(argv[0], argv[1])


if __name__ == "__main__":
    sys.exit(main())
