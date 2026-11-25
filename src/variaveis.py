"""Cálculo das variáveis explicativas (mesma lógica do 04_variaveis.ipynb)."""
from __future__ import annotations

import pandas as pd

from src.dados import CHAVES_RECORTE

VARIAVEIS_NUM = [
    "taxa_inad",
    "taxa_atraso_15_90",
    "taxa_problematico",
    "var_taxa_inad_3m",
    "var_taxa_inad_6m",
    "var_atraso_15_90_3m",
    "cresc_carteira_3m",
    "cresc_carteira_6m",
]
VARIAVEIS_CAT = ["uf", "cliente", "modalidade"]
VARIAVEIS = VARIAVEIS_NUM + VARIAVEIS_CAT


def criar_variaveis(painel: pd.DataFrame) -> pd.DataFrame:
    """Recebe o painel mensal e devolve uma cópia com as variáveis do modelo.

    Cada variável do mês t usa só dados até o mês t: os shift(h) com h positivo
    olham para meses anteriores do mesmo recorte, nunca para o futuro.
    """
    painel = painel.sort_values(CHAVES_RECORTE + ["data_base"]).reset_index(drop=True).copy()
    carteira = painel["carteira_ativa"]

    # Taxas do mês, em % da carteira ativa
    painel["taxa_inad"] = 100 * painel["carteira_inadimplencia"] / carteira
    painel["taxa_atraso_15_90"] = 100 * painel["vencido_de_15_ate_90_dias"] / carteira
    painel["taxa_problematico"] = 100 * painel["ativo_problematico"] / carteira

    grupos = painel.groupby(CHAVES_RECORTE, observed=True)
    for h in (3, 6):
        painel[f"var_taxa_inad_{h}m"] = painel["taxa_inad"] - grupos["taxa_inad"].shift(h)
        painel[f"cresc_carteira_{h}m"] = 100 * (carteira / grupos["carteira_ativa"].shift(h) - 1)

    painel["var_atraso_15_90_3m"] = (
        painel["taxa_atraso_15_90"] - grupos["taxa_atraso_15_90"].shift(3)
    )
    return painel
