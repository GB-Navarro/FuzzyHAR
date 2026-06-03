# -*- coding: utf-8 -*-
"""
Pacote `fuzzyhar` — versao modular e desacoplada do pipeline FuzzyHAR.

Objetivo: reproduzir EXATAMENTE os resultados do notebook FuzzyHAR.ipynb,
com o codigo das funcoes preservado verbatim, porem organizado em modulos
importaveis e com pontos de extensao claros (ver `fuzzyhar.registry`).

Para rodar o pipeline completo (com o working dir = pasta FuzzyHAR/):

    from fuzzyhar.pipeline.runner import run_pipeline
    run_pipeline()

Para mudar o split, edite `SPLIT` em `fuzzyhar/config.py`.
"""

__all__ = ["config"]
