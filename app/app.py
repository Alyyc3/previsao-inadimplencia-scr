"""App Streamlit: risco de alta da inadimplência por recorte (SCR.data, Banco Central).

Para rodar, a partir da pasta do projeto:
    streamlit run app/app.py

Os arquivos modelo.joblib e dados_app.parquet são gerados pelo 06_modelo_final.ipynb.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PASTA_APP = Path(__file__).parent

st.set_page_config(page_title="Risco de alta da inadimplência", layout="wide")


# ---------------------------------------------------------------------------
# Carregamento (com cache: o Streamlit roda o script inteiro a cada clique,
# e o cache evita reler os arquivos e refazer as previsões toda vez)
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    """Pipeline treinado no 06_modelo_final.ipynb (preparação + floresta aleatória)."""
    return joblib.load(PASTA_APP / "modelo.joblib")


@st.cache_data
def carregar_dados():
    """Histórico mensal dos 386 recortes, com as variáveis do modelo."""
    return pd.read_parquet(PASTA_APP / "dados_app.parquet")


modelo = carregar_modelo()
dados = carregar_dados()

# O Pipeline guarda os nomes das colunas usadas no fit
VARIAVEIS = list(modelo.feature_names_in_)


@st.cache_data
def pontuar():
    """Calcula a nota de risco dos meses que ainda não têm alvo (jul a dez/2025)."""
    a_pontuar = dados[dados["alvo"].isna()].dropna(subset=VARIAVEIS).copy()
    a_pontuar["nota_risco"] = modelo.predict_proba(a_pontuar[VARIAVEIS])[:, 1]
    return a_pontuar


notas = pontuar()

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("Risco de alta da inadimplência de crédito")
st.markdown(
    "Para cada combinação de **UF × tipo de cliente × modalidade**, o modelo estima o risco de a "
    "inadimplência ter uma **alta relevante nos 6 meses seguintes** "
    "(mais de 0,5 ponto percentual **e** mais de 20% acima da taxa atual). "
    "Dados: SCR.data, Banco Central do Brasil."
)

# ---------------------------------------------------------------------------
# Barra lateral: mês e filtros
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")

meses = sorted(notas["data_base"].unique(), reverse=True)
mes = st.sidebar.selectbox(
    "Mês-base da previsão",
    meses,
    format_func=lambda d: pd.Timestamp(d).strftime("%m/%Y"),
)
st.sidebar.caption("A previsão olha para os 6 meses seguintes ao mês-base.")

clientes = st.sidebar.multiselect("Tipo de cliente", sorted(dados["cliente"].unique()), placeholder="Todos")
modalidades = st.sidebar.multiselect("Modalidade", sorted(dados["modalidade"].unique()), placeholder="Todas")
ufs = st.sidebar.multiselect("UF", sorted(dados["uf"].unique()), placeholder="Todas")
st.sidebar.caption("Filtro vazio = todos.")

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
aba_ranking, aba_recorte, aba_modelo = st.tabs(
    ["Ranking de risco", "Evolução de um recorte", "Sobre o modelo"]
)

# --- Aba 1: ranking ---------------------------------------------------------
with aba_ranking:
    tabela = notas[notas["data_base"] == mes]
    if clientes:
        tabela = tabela[tabela["cliente"].isin(clientes)]
    if modalidades:
        tabela = tabela[tabela["modalidade"].isin(modalidades)]
    if ufs:
        tabela = tabela[tabela["uf"].isin(ufs)]

    st.subheader(f"Recortes por nota de risco: mês-base {pd.Timestamp(mes):%m/%Y}")
    st.info(
        "A **nota de risco** (0 a 1) serve para **ordenar** os recortes: quanto maior, mais risco "
        "em relação aos outros. Ela **não é uma probabilidade**: o modelo foi treinado dando mais "
        "peso aos casos de alta, então as notas ficam acima da frequência real.",
        icon="ℹ️",
    )

    if tabela.empty:
        st.warning("Nenhum recorte com esses filtros.")
    else:
        exibir = (
            tabela.sort_values("nota_risco", ascending=False)
            .assign(carteira_bi=lambda d: d["carteira_ativa"] / 1e9)
            # A nota vem primeiro para ficar visível mesmo em telas estreitas
            [["nota_risco", "uf", "cliente", "modalidade", "taxa_inad", "taxa_atraso_15_90",
              "var_taxa_inad_3m", "carteira_bi"]]
        )
        st.caption(f"{len(exibir)} recortes")
        st.dataframe(
            exibir,
            hide_index=True,
            width="stretch",
            column_config={
                "uf": "UF",
                "cliente": "Cliente",
                "modalidade": st.column_config.TextColumn("Modalidade", width="medium"),
                "taxa_inad": st.column_config.NumberColumn("Inadimplência (%)", format="%.2f"),
                "taxa_atraso_15_90": st.column_config.NumberColumn("Atraso 15-90 dias (%)", format="%.2f"),
                "var_taxa_inad_3m": st.column_config.NumberColumn("Var. inadimplência 3 meses (p.p.)", format="%.2f"),
                "carteira_bi": st.column_config.NumberColumn("Carteira (R$ bi)", format="%.1f"),
                "nota_risco": st.column_config.ProgressColumn(
                    "Nota de risco", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

# --- Aba 2: evolução de um recorte -----------------------------------------
with aba_recorte:
    st.subheader("Evolução de um recorte")
    col1, col2, col3 = st.columns(3)

    def posicao(lista, preferido):
        """Posição do valor preferido na lista (para abrir com um exemplo útil), ou 0."""
        return lista.index(preferido) if preferido in lista else 0

    # Cada escolha limita as opções seguintes às combinações que existem nos dados.
    # O exemplo inicial é PF / Empréstimos / SP, um dos maiores recortes.
    lista = sorted(dados["cliente"].unique())
    cliente = col1.selectbox("Tipo de cliente", lista, index=posicao(lista, "PF"))
    opcoes = dados[dados["cliente"] == cliente]
    lista = sorted(opcoes["modalidade"].unique())
    modalidade = col2.selectbox("Modalidade", lista, index=posicao(lista, "Empréstimos"))
    opcoes = opcoes[opcoes["modalidade"] == modalidade]
    lista = sorted(opcoes["uf"].unique())
    uf = col3.selectbox("UF", lista, index=posicao(lista, "SP"))

    historico = dados[
        (dados["cliente"] == cliente) & (dados["modalidade"] == modalidade) & (dados["uf"] == uf)
    ].sort_values("data_base")

    st.markdown("**Taxas mensais (% da carteira ativa)**")
    st.line_chart(
        historico.set_index("data_base")[["taxa_inad", "taxa_atraso_15_90"]].rename(
            columns={"taxa_inad": "Inadimplência", "taxa_atraso_15_90": "Atraso 15-90 dias"}
        )
    )

    notas_recorte = notas[
        (notas["cliente"] == cliente) & (notas["modalidade"] == modalidade) & (notas["uf"] == uf)
    ].sort_values("data_base")
    if not notas_recorte.empty:
        st.markdown("**Nota de risco por mês-base (jul a dez/2025)**")
        st.bar_chart(
            notas_recorte.assign(mes=notas_recorte["data_base"].dt.strftime("%m/%Y"))
            .set_index("mes")[["nota_risco"]]
            .rename(columns={"nota_risco": "Nota de risco"})
        )

# --- Aba 3: sobre o modelo -------------------------------------------------
with aba_modelo:
    st.subheader("Como o modelo foi construído")
    st.markdown(
        """
- **Recortes:** 386 combinações de UF × cliente × modalidade com 48 meses completos (2022 a 2025)
  e carteira ativa mediana de pelo menos R$ 100 milhões.
- **Alvo:** alta da inadimplência em 6 meses de mais de 0,5 ponto percentual **e** mais de 20%.
- **Variáveis:** taxas de inadimplência, de atraso de 15 a 90 dias e de ativo problemático; variações
  em 3 e 6 meses; crescimento da carteira; UF, tipo de cliente e modalidade.
- **Modelo:** floresta aleatória (scikit-learn), escolhida numa validação temporal.
  Para o app, foi treinada de novo com todos os meses que já têm alvo (jul/2022 a jun/2025).
"""
    )

    st.subheader("Resultados no teste (jan a jun/2025)")
    st.caption("Modelos treinados com dados até jun/2024 e avaliados em 2025 (05_modelo.ipynb).")
    resultados = pd.DataFrame(
        {
            "Modelo": [
                "Chute (proporção de positivos)",
                "Regra simples: atraso 15-90 dias",
                "Regressão logística",
                "Gradient boosting",
                "Floresta aleatória",
            ],
            "AUC": [0.500, 0.690, 0.678, 0.712, 0.770],
            "AP": [0.249, 0.369, 0.414, 0.462, 0.527],
        }
    )
    st.dataframe(resultados, hide_index=True, width="stretch")

    st.subheader("Limitações")
    st.markdown(
        """
- A proporção de altas subiu de 13% (treino) para 25% (teste): as notas ordenam bem o risco,
  mas não estão calibradas como probabilidades.
- Os resultados do teste foram vistos antes da escolha final do modelo.
- O modelo trabalha com dados agregados por recorte, não com operações individuais.
"""
    )

st.divider()
st.caption("Projeto de Alyce Abade · Dados: SCR.data, Banco Central do Brasil.")
