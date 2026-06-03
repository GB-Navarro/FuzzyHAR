# -*- coding: utf-8 -*-
"""Golden master: verifica que dois snapshots de baseline coincidem por VALOR.

Este teste so roda se existirem os snapshots `baseline/antes` e `baseline/depois`
(criados com fuzzyhar.scripts.freeze_baseline). Caso contrario e PULADO.

Fluxo recomendado:
    python -m fuzzyhar.scripts.freeze_baseline --label antes   # com codigo atual
    # ... rode a refatoracao e regenere Others/ e Results/ ...
    python -m fuzzyhar.scripts.freeze_baseline --label depois
    pytest fuzzyhar/tests/test_golden_master.py -v
"""
from pathlib import Path

import pytest

BASELINE = Path("baseline")
A, B = "antes", "depois"

pytestmark = pytest.mark.skipif(
    not ((BASELINE / A / "_manifest.json").exists()
         and (BASELINE / B / "_manifest.json").exists()),
    reason="baseline/antes e baseline/depois ainda nao foram congelados",
)


def test_refactor_matches_baseline():
    from fuzzyhar.scripts.compare_baseline import compare
    assert compare(A, B) == 0, "resultados divergiram do baseline (ver saida acima)"
