"""App Streamlit: risco de alta da inadimplência por recorte (SCR.data, Banco Central).

Para rodar, a partir da pasta do projeto:
    streamlit run app/app.py

Os arquivos modelo.joblib e dados_app.parquet são gerados pelo 06_modelo_final.ipynb.
O tema "Azul confiança" (cores, fonte, modo claro/escuro) fica em app/.streamlit/config.toml.
"""
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

PASTA_APP = Path(__file__).parent

st.set_page_config(
    page_title="Risco de alta da inadimplência",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",  # filtros visíveis logo de cara
)


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
# Funções de apoio
# ---------------------------------------------------------------------------
def br(valor, casas=2):
    """Formata um número no padrão brasileiro (vírgula decimal)."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def nome_curto(modalidade):
    """Tira o trecho entre parênteses dos nomes longos de modalidade."""
    return modalidade.split(" (")[0].strip()


def taxa_ponderada(df):
    """Inadimplência do conjunto: soma da carteira inadimplente ÷ soma da carteira ativa (%)."""
    inadimplente = (df["taxa_inad"] / 100 * df["carteira_ativa"]).sum()
    return 100 * inadimplente / df["carteira_ativa"].sum()


def aplicar_filtros(df):
    """Aplica os filtros da barra lateral (filtro vazio = todos)."""
    if clientes:
        df = df[df["cliente"].isin(clientes)]
    if modalidades:
        df = df[df["modalidade"].isin(modalidades)]
    if ufs:
        df = df[df["uf"].isin(ufs)]
    return df


NOME_CLIENTE = {"PF": "pessoas físicas", "PJ": "empresas"}


def descrever(linha):
    """Nome legível de um recorte, ex.: 'AM · PF · financiamentos rurais'."""
    return f"{linha['uf']} · {linha['cliente']} · {nome_curto(str(linha['modalidade'])).lower()}"


def quantos_abaixo(nota, mes_ref):
    """Quantos dos 386 recortes têm nota MENOR que esta (no mesmo mês-base)."""
    todas = notas.loc[notas["data_base"] == mes_ref, "nota_risco"]
    return int((todas < nota).sum()), len(todas)


def sinais_do_recorte(linha, mes_ref):
    """Compara um recorte com recortes parecidos para explicar a nota em linguagem simples.

    "Parecidos" = mesmo tipo de crédito e mesmo tipo de cliente, em outros estados.
    É uma comparação simples com o valor típico (mediana); o modelo combina essas
    informações de um jeito mais complexo, então isto ajuda a entender, mas não é a
    conta exata do modelo.
    """
    no_mes = dados[dados["data_base"] == mes_ref]
    pares = no_mes[
        (no_mes["modalidade"] == linha["modalidade"]) & (no_mes["cliente"] == linha["cliente"])
    ]
    if len(pares) >= 4:
        ref = (
            f"outros estados com {nome_curto(str(linha['modalidade'])).lower()} "
            f"para {NOME_CLIENTE[str(linha['cliente'])]}"
        )
    else:
        pares, ref = no_mes, "todos os recortes"

    alertas, tranquilos = [], []

    # 1. Nível do atraso curto (15 a 90 dias): o sinal que mais pesa no modelo
    atraso, atraso_tipico = linha["taxa_atraso_15_90"], pares["taxa_atraso_15_90"].median()
    if atraso > atraso_tipico * 1.1:
        alertas.append(
            f"O **atraso curto** (15 a 90 dias) está em **{br(atraso)}%**, acima do típico de {ref} "
            f"({br(atraso_tipico)}%). Atraso curto alto costuma vir antes de inadimplência."
        )
    elif atraso < atraso_tipico * 0.9:
        tranquilos.append(
            f"O **atraso curto** está em {br(atraso)}%, abaixo do típico de {ref} ({br(atraso_tipico)}%)."
        )

    # 2. Direção recente do atraso curto
    var_atraso = linha["var_atraso_15_90_3m"]
    if var_atraso > 0.05:
        alertas.append(f"O atraso curto **subiu {br(var_atraso)} ponto** nos últimos 3 meses.")
    elif var_atraso < -0.05:
        tranquilos.append(f"O atraso curto **caiu {br(-var_atraso)} ponto** nos últimos 3 meses.")

    # 3. Direção recente da inadimplência
    var_inad = linha["var_taxa_inad_3m"]
    if var_inad > 0.1:
        alertas.append(f"A inadimplência **já subiu {br(var_inad)} ponto** nos últimos 3 meses.")
    elif var_inad < -0.1:
        tranquilos.append(f"A inadimplência **caiu {br(-var_inad)} ponto** nos últimos 3 meses.")

    # 4. Ativo problemático (créditos com sinais de que podem não ser pagos)
    prob, prob_tipico = linha["taxa_problematico"], pares["taxa_problematico"].median()
    if prob > prob_tipico * 1.1:
        alertas.append(
            f"A parte de **crédito problemático** ({br(prob)}%) está acima do típico de {ref} "
            f"({br(prob_tipico)}%)."
        )
    elif prob < prob_tipico * 0.9:
        tranquilos.append(
            f"A parte de crédito problemático ({br(prob)}%) está abaixo do típico de {ref} "
            f"({br(prob_tipico)}%)."
        )
    return alertas, tranquilos


def explicar_recorte(linha, nota, mes_ref):
    """Mostra, em linguagem simples, por que um recorte tem a nota que tem."""
    abaixo, total = quantos_abaixo(nota, mes_ref)
    mediana = notas.loc[notas["data_base"] == mes_ref, "nota_risco"].median()
    nivel = "acima" if nota > mediana else "abaixo"
    st.markdown(
        f"**{descrever(linha)}** tem nota de risco **{br(nota)}**, maior que a de **{abaixo} dos "
        f"{total}** recortes em {mes_ref:%m/%Y} ({nivel} da nota típica, {br(mediana)})."
    )

    # O que precisaria acontecer para contar como "alta relevante" (a regra do alvo):
    # subir mais de 0,5 ponto E mais de 20%. Para taxas altas, os 20% pesam mais.
    taxa = linha["taxa_inad"]
    limite = max(taxa + 0.5, taxa * 1.2)
    prazo = mes_ref + pd.DateOffset(months=6) + pd.offsets.MonthEnd(0)
    regra = "a regra dos 20% a mais" if taxa * 0.2 > 0.5 else "a regra dos 0,5 ponto a mais"
    st.markdown(
        f"Para contar como **alta relevante**, a inadimplência deste recorte (hoje **{br(taxa)}%**) "
        f"precisaria passar de **{br(limite)}%** até {prazo:%m/%Y}. Neste caso, o que pesa é {regra}."
    )

    alertas, tranquilos = sinais_do_recorte(linha, mes_ref)
    if alertas and nota <= mediana:
        st.markdown(
            "Há sinais de alerta abaixo, mas a nota ficou baixa: o modelo não vê força suficiente "
            "para um salto desse tamanho em 6 meses."
        )
    elif not alertas and nota > mediana:
        st.markdown(
            "Mesmo sem sinais fortes nesta comparação simples, a nota ficou alta: o modelo também "
            "considera o tipo de crédito e de cliente, que pesam muito na previsão."
        )
    col_a, col_t = st.columns(2, gap="medium")
    with col_a:
        st.markdown(":orange[**:material/warning: Sinais de alerta**]")
        st.markdown("\n".join(f"- {s}" for s in alertas) if alertas else "- Nenhum sinal forte de alerta.")
    with col_t:
        st.markdown(":green[**:material/check_circle: Sinais tranquilizadores**]")
        st.markdown("\n".join(f"- {s}" for s in tranquilos) if tranquilos else "- Nenhum sinal claro de melhora.")
    st.caption(
        "Comparação simples com o valor típico (mediana). O modelo combina essas e outras informações "
        "(tipo de crédito, tipo de cliente, crescimento da carteira) de forma mais complexa."
    )


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("Risco de alta da inadimplência de crédito", icon=":material/monitoring:")
st.markdown(
    "Para cada combinação de **UF × tipo de cliente × modalidade**, o modelo estima o risco de a "
    "inadimplência ter uma **alta relevante nos 6 meses seguintes** "
    "(mais de 0,5 ponto percentual **e** mais de 20% acima da taxa atual)."
)

with st.expander("Primeira vez aqui? Entenda o app em 1 minuto", icon=":material/school:"):
    st.markdown(
        """
**O que é inadimplência?**
É a parte do dinheiro emprestado que está com o pagamento muito atrasado (em geral, mais de 90 dias).
Uma inadimplência de 4% quer dizer que, de cada R$ 100 emprestados, R$ 4 estão nessa situação.

**O que é um recorte?**
É um grupo de empréstimos com três características em comum:
- o **estado** (UF), como SP ou BA;
- o **tipo de cliente**: **PF** é pessoa física e **PJ** é empresa;
- a **modalidade**, que é o tipo de crédito, como empréstimo, financiamento imobiliário ou crédito rural.

Exemplo de recorte: *empréstimos para pessoas físicas em São Paulo*. O app acompanha 386 recortes.

**O que o modelo faz?**
Ele olha a história recente de cada recorte e estima se a inadimplência vai **subir de forma relevante
nos 6 meses seguintes**. "Relevante" quer dizer subir mais de 0,5 ponto percentual **e** mais de 20%.
- Passar de 2,0% para 2,6% **conta** como alta relevante (+0,6 ponto e +30%).
- Passar de 2,0% para 2,3% **não conta** (só +0,3 ponto).

**O que é a nota de risco?**
Um número de 0 a 1 para **comparar** recortes: quanto maior, mais o modelo vê sinais de uma alta pela frente.
Ela **não é uma chance em porcentagem**: uma nota 0,8 não quer dizer 80% de chance.

**Palavras que aparecem no app**
- **Atraso de 15 a 90 dias:** pagamentos atrasados há pouco tempo. Muitas vezes vêm antes da inadimplência,
  por isso funcionam como um sinal de alerta. Foi o sinal que mais ajudou o modelo.
- **Carteira ativa:** o total de dinheiro emprestado naquele recorte.
- **Ponto percentual (p.p.):** a diferença simples entre duas taxas. De 4% para 5% é +1 p.p.
- **Mês-base:** o mês em que a previsão é feita. A previsão olha para os 6 meses seguintes.

**De onde vêm os dados?**
Do SCR.data, uma base pública do Banco Central com informações de crédito de todo o sistema financeiro
do Brasil. O app usa dados de 2022 a 2025. O modelo é uma floresta aleatória (scikit-learn).
"""
    )

# ---------------------------------------------------------------------------
# Barra lateral: mês e filtros. Eles personalizam TODAS as abas (menos "Modelo").
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader(":blue[Personalize a análise]", icon=":material/tune:")

    meses = sorted(notas["data_base"].unique(), reverse=True)
    mes = pd.Timestamp(
        st.selectbox(
            "Mês-base da previsão",
            meses,
            format_func=lambda d: pd.Timestamp(d).strftime("%m/%Y"),
            key="mes_base",
        )
    )
    st.caption("A previsão olha para os 6 meses seguintes ao mês-base.")

    # Poucas opções (PF/PJ): pills deixam tudo visível, sem abrir lista
    clientes = st.pills(
        "Tipo de cliente", sorted(dados["cliente"].unique()), selection_mode="multi",
        key="f_cliente",
    )
    modalidades = st.multiselect(
        "Modalidade",
        sorted(dados["modalidade"].unique()),
        format_func=nome_curto,
        placeholder="Todas",
        key="f_modalidade",
    )
    ufs = st.multiselect("UF", sorted(dados["uf"].unique()), placeholder="Todas", key="f_uf")
    st.caption("Sem seleção = todos. Os resultados, o ranking e o recorte mudam conforme os filtros.")

# Versões filtradas dos dados: todas as abas de análise usam estas
dados_f = aplicar_filtros(dados)
notas_f = aplicar_filtros(notas)
n_recortes = len(notas_f[notas_f["data_base"] == mes])
filtrado = bool(clientes or modalidades or ufs)

if n_recortes == 0:
    st.warning(
        "Nenhum recorte combina com esses filtros. Remova algum filtro na barra lateral.",
        icon=":material/search_off:",
    )
    st.stop()  # interrompe o app aqui: não há dados para mostrar

# ---------------------------------------------------------------------------
# Abas (nomes curtos para caberem na tela do celular)
# ---------------------------------------------------------------------------
aba_resultados, aba_ranking, aba_recorte, aba_modelo = st.tabs(
    [
        ":material/lightbulb: Resultados",
        ":material/leaderboard: Ranking",
        ":material/timeline: Recorte",
        ":material/info: Modelo",
    ]
)

# --- Aba 0: principais resultados ------------------------------------------
# Os textos são montados a partir dos dados: se o mês-base mudar, os números mudam junto.
with aba_resultados:
    if filtrado:
        st.caption(
            f":material/tune: **Resultados personalizados:** {n_recortes} recortes que combinam com "
            f"os filtros da barra lateral, mês-base {mes:%m/%Y}."
        )
    else:
        st.caption(
            f"Resumo dos 386 recortes no mês-base {mes:%m/%Y}. Use a barra lateral para personalizar."
        )

    # Cálculos usados nos cartões e nos textos (sempre sobre os dados FILTRADOS) ----
    primeiro_mes = dados_f["data_base"].min()
    dados_mes = dados_f[dados_f["data_base"] == mes]
    dados_inicio = dados_f[dados_f["data_base"] == primeiro_mes]
    notas_mes = notas_f[notas_f["data_base"] == mes]
    mediana_geral = notas.loc[notas["data_base"] == mes, "nota_risco"].median()  # 386 recortes

    taxa_ini, taxa_fim = taxa_ponderada(dados_inicio), taxa_ponderada(dados_mes)
    por_cliente_ini = dados_inicio.groupby("cliente", observed=True).apply(taxa_ponderada, include_groups=False)
    por_cliente_fim = dados_mes.groupby("cliente", observed=True).apply(taxa_ponderada, include_groups=False)

    nota_modalidade = (
        notas_mes.groupby("modalidade", observed=True)["nota_risco"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    # Para o texto, prefere tipos de crédito com 5+ recortes (médias de 1 ou 2 recortes enganam).
    # Com filtros, pode não haver nenhum assim; aí usa todos.
    confiaveis = nota_modalidade[nota_modalidade["count"] >= 5]
    usa_confiaveis = len(confiaveis) >= 2
    base_texto = confiaveis if usa_confiaveis else nota_modalidade

    # Top 20 com todos os recortes; com filtros, um quarto dos recortes selecionados (no máximo 20).
    # Ex.: 22 recortes -> top 5. Assim o "top" continua sendo só o grupo de maior risco.
    k_top = min(20, max(1, n_recortes // 4))
    top = notas_mes.nlargest(k_top, "nota_risco")
    # Agrupa o top por tipo de crédito; se só um tipo foi filtrado, agrupa por estado
    # (ou por tipo de cliente), senão a frase seria óbvia ("13 de 13 são de empréstimos")
    if notas_mes["modalidade"].nunique() > 1:
        dim_top, rotulo_dim = "modalidade", lambda v: nome_curto(str(v)).lower()
    elif notas_mes["uf"].nunique() > 1:
        dim_top, rotulo_dim = "uf", str
    else:
        dim_top, rotulo_dim = "cliente", lambda v: {"PF": "pessoas físicas", "PJ": "empresas"}[str(v)]
    contagem_top = top[dim_top].value_counts()
    contagem_top = contagem_top[contagem_top > 0]
    lider, n_lider = contagem_top.index[0], int(contagem_top.iloc[0])

    rural_pf = dados_f[
        dados_f["modalidade"].str.startswith("Financiamentos rurais") & (dados_f["cliente"] == "PF")
    ]
    notas_rural = notas_mes[
        notas_mes["modalidade"].str.startswith("Financiamentos rurais") & (notas_mes["cliente"] == "PF")
    ]
    tem_rural = not rural_pf.empty and not notas_rural.empty
    if tem_rural:
        rural_ini = taxa_ponderada(rural_pf[rural_pf["data_base"] == primeiro_mes])
        rural_fim = taxa_ponderada(rural_pf[rural_pf["data_base"] == mes])

    # Resumo em 3 números: a mensagem principal antes dos detalhes ------------
    with st.grid(3, gap="medium", min_column_width=220):
        st.metric(
            "Inadimplência do grupo filtrado" if filtrado else "Inadimplência geral",
            f"{br(taxa_fim)}%",
            delta=f"{br(taxa_fim - taxa_ini)} p.p.",
            delta_color="inverse",
            delta_description=f"desde {primeiro_mes:%m/%Y}",
            icon=":material/trending_up:",
            help="Dinheiro com pagamento muito atrasado ÷ dinheiro emprestado, somando os recortes "
            "selecionados.",
            border=True,
            height="stretch",
        )
        if tem_rural:
            st.metric(
                "Crédito rural para pessoas físicas",
                f"{br(rural_fim)}%",
                delta=f"{br(rural_fim - rural_ini)} p.p.",
                delta_color="inverse",
                delta_description=f"desde {primeiro_mes:%m/%Y}",
                icon=":material/agriculture:",
                help="Inadimplência do crédito rural para pessoas físicas nos estados selecionados.",
                border=True,
                height="stretch",
            )
        else:
            st.metric(
                "Recortes analisados",
                n_recortes,
                icon=":material/grid_view:",
                help="Quantos grupos (estado × tipo de cliente × modalidade) entram nos filtros.",
                border=True,
                height="stretch",
            )
        # Mesma estrutura dos outros cartões: rótulo + número.
        # Se algum grupo se repete no topo (concentração), mostra quantos; senão, a maior nota.
        if n_lider >= 2:
            st.metric(
                f"Top {k_top} de risco: {rotulo_dim(lider)}",
                f"{n_lider} de {k_top}",
                icon=":material/leaderboard:",
                help=f"Dos {k_top} recortes com maior nota de risco, {n_lider} são de "
                f"{rotulo_dim(lider)}, o grupo mais frequente nessa lista.",
                border=True,
                height="stretch",
            )
        else:
            maior = notas_mes.loc[notas_mes["nota_risco"].idxmax()]
            st.metric(
                f"Maior nota: {maior['uf']} · {maior['cliente']} · {nome_curto(str(maior['modalidade'])).lower()}",
                br(maior["nota_risco"]),
                icon=":material/priority_high:",
                help="O recorte com mais sinais de alta da inadimplência entre os selecionados.",
                border=True,
                height="stretch",
            )

    # 1. Cenário: texto à esquerda, gráfico à direita -------------------------
    st.markdown("#### :blue[1. Como a inadimplência mudou desde 2022]")
    col_texto, col_grafico = st.columns([2, 3], gap="large")
    with col_texto:
        if not filtrado:
            quem = "os 386 recortes"
        elif n_recortes == 1:
            quem = "o recorte selecionado"
        else:
            quem = f"os {n_recortes} recortes selecionados"
        st.markdown(
            f"Somando {quem}, a inadimplência passou de **{br(taxa_ini)}%** em "
            f"{primeiro_mes:%m/%Y} para **{br(taxa_fim)}%** em {mes:%m/%Y}. Ou seja, de cada R$ 100 "
            f"emprestados, o valor com pagamento muito atrasado foi de R$ {br(taxa_ini)} para "
            f"R$ {br(taxa_fim)}."
        )
        nomes_cliente = {"PF": "Pessoas físicas", "PJ": "Empresas"}
        linhas_cliente = [
            f"- **{nomes_cliente[c]}:** de {br(por_cliente_ini[c])}% para {br(por_cliente_fim[c])}%."
            for c in ("PF", "PJ")
            if c in por_cliente_ini.index and c in por_cliente_fim.index
        ]
        if len(linhas_cliente) > 1:
            st.markdown("\n".join(linhas_cliente))
        if taxa_fim > taxa_ini:
            significado_1 = (
                "Uma parte maior do dinheiro emprestado está com pagamento muito atrasado do que em "
                f"{primeiro_mes:%m/%Y}: para quem empresta, fica mais difícil receber. Foi nesse cenário "
                "que as altas relevantes ficaram mais comuns: no conjunto dos 386 recortes, aconteceram "
                "em 11% dos casos com mês-base em 2023 e em 25% no 1º semestre de 2025."
            )
        else:
            significado_1 = (
                f"A inadimplência deste grupo está menor do que em {primeiro_mes:%m/%Y}: uma parte menor "
                "do dinheiro emprestado está com pagamento muito atrasado."
            )
        st.markdown(f":blue[**:material/lightbulb: O que isso significa:**] {significado_1}")

    evolucao = (
        dados_f[dados_f["data_base"] <= mes]
        .groupby(["data_base", "cliente"], observed=True)
        .apply(taxa_ponderada, include_groups=False)
        .rename("taxa")
        .reset_index()
    )
    evolucao["cliente"] = evolucao["cliente"].map({"PF": "Pessoa física", "PJ": "Empresa"})
    evolucao["taxa_txt"] = evolucao["taxa"].map(br) + "%"  # dica do gráfico com vírgula
    with col_grafico.container(border=True):
        st.markdown("**Inadimplência por tipo de cliente (% do dinheiro emprestado)**")
        grafico_evolucao = alt.Chart(evolucao).mark_line(strokeWidth=2.5).encode(
            x=alt.X("data_base:T", title=None, axis=alt.Axis(format="%m/%Y", tickCount=6)),
            y=alt.Y("taxa:Q", title="% da carteira ativa"),
            color=alt.Color("cliente:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip("data_base:T", title="Mês", format="%m/%Y"),
                alt.Tooltip("cliente:N", title="Cliente"),
                alt.Tooltip("taxa_txt:N", title="Inadimplência"),
            ],
        )
        st.altair_chart(
            grafico_evolucao,
            alt="Inadimplência por tipo de cliente desde 2022 nos recortes selecionados",
        )

    # 2. Onde está o maior risco: texto à esquerda, tabela à direita ----------
    st.markdown("#### :blue[2. Onde o modelo vê mais risco]")
    col_texto, col_grafico = st.columns([2, 3], gap="large")
    with col_texto:
        def juntar(itens):
            """Junta uma lista em português: 'a', 'a e b', 'a, b e c'."""
            return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]

        if k_top >= 3 and n_lider >= 2:
            partes = [f"{n} de **{rotulo_dim(valor)}**" for valor, n in contagem_top.items()]
            partes[0] = partes[0].replace(" de ", " são de ", 1)
            st.markdown(f"Entre os **{k_top} recortes com maior nota**, {juntar(partes)}.")
        elif k_top >= 3:
            # Nenhum grupo se repete: o risco está espalhado
            grupos = [f"**{rotulo_dim(valor)}**" for valor in contagem_top.index]
            st.markdown(
                f"Os **{k_top} recortes com maior nota** estão espalhados, sem um grupo dominante: "
                f"{juntar(grupos)}."
            )
        if len(base_texto) >= 2:
            m1, m2 = base_texto.index[0], base_texto.index[1]
            st.markdown(
                f"Olhando a média de cada tipo de crédito, os que mais preocupam são "
                f"**{nome_curto(m1).lower()}** (nota média {br(base_texto.loc[m1, 'mean'])}) e "
                f"**{nome_curto(m2).lower()}** ({br(base_texto.loc[m2, 'mean'])})."
            )
        else:
            m1 = base_texto.index[0]
            st.markdown(
                f"O tipo de crédito selecionado, **{nome_curto(m1).lower()}**, tem nota média "
                f"**{br(base_texto.loc[m1, 'mean'])}** (a mediana dos 386 recortes é {br(mediana_geral)})."
            )

    ranking_mod = nota_modalidade.reset_index()
    ranking_mod["nome"] = ranking_mod["modalidade"].astype(str).map(nome_curto)
    ranking_mod["mean"] = ranking_mod["mean"].round(2)
    with col_grafico.container(border=True):
        st.markdown("**Nota de risco média por tipo de crédito**")
        # Tabela com barras (em vez de gráfico): os nomes longos de modalidade aparecem inteiros
        st.dataframe(
            ranking_mod[["nome", "mean", "count"]],
            hide_index=True,
            column_config={
                "nome": st.column_config.TextColumn("Tipo de crédito", width="large"),
                "mean": st.column_config.ProgressColumn(
                    "Nota média", min_value=0.0, max_value=1.0, format="localized",
                    help="Média da nota de risco dos recortes desse tipo de crédito.",
                ),
                "count": st.column_config.NumberColumn(
                    "Recortes", help="Quantos recortes (estados × tipo de cliente) entram na média."
                ),
            },
            alt="Nota média de risco por tipo de crédito nos recortes selecionados",
        )
        st.caption(
            "**Como ler:** barra mais comprida = mais sinais de alta, na média dos estados. "
            "Médias com poucos recortes são menos confiáveis."
        )

    st.markdown(
        ":blue[**:material/lightbulb: O que isso significa:**] a nota mede **sinais de que a "
        "inadimplência vai subir**, e não o tamanho dela hoje. Por isso um tipo de crédito pode ter "
        "inadimplência baixa agora e mesmo assim nota alta: o modelo enxerga sinais, como o atraso "
        "curto subindo, de que a situação tende a piorar nos próximos 6 meses. O contrário também "
        "acontece: inadimplência alta, mas estável, pode ter nota baixa."
    )

    # Exemplo concreto: por que o recorte de maior risco tem nota alta -------
    maior = notas_mes.loc[notas_mes["nota_risco"].idxmax()]
    st.markdown("#### :blue[Por dentro de um exemplo: o recorte de maior risco]")
    with st.container(border=True):
        explicar_recorte(maior, maior["nota_risco"], mes)
    st.caption("Quer entender outro recorte? Escolha um na aba **Ranking** ou na aba **Recorte**.")

    # 3. Destaque: crédito rural para pessoa física --------------------------
    if tem_rural:
        st.markdown("#### :blue[3. Destaque: crédito rural para pessoas físicas]")
        acima_mediana = (notas_rural["nota_risco"] > mediana_geral).sum()
        texto_rural = (
            f"A inadimplência do crédito rural para pessoas físicas passou de **{br(rural_ini)}%** em "
            f"{primeiro_mes:%m/%Y} para **{br(rural_fim)}%** em {mes:%m/%Y}. Em {mes:%m/%Y}, "
            f"**{acima_mediana} dos {len(notas_rural)} estados** selecionados com esse tipo de crédito "
            f"têm nota acima da mediana dos 386 recortes, com nota típica de "
            f"**{br(notas_rural['nota_risco'].median())}**."
        )
        if len(notas_rural) >= 5 and acima_mediana >= len(notas_rural) / 2:
            texto_rural += (
                " Como o sinal aparece em muitos estados ao mesmo tempo, ele indica um movimento do "
                "**setor**, e não de um estado específico."
            )
        st.markdown(texto_rural)
        if rural_ini > 0 and rural_fim > rural_ini and len(notas_rural) >= 5:
            vezes = rural_fim / rural_ini
            st.markdown(
                f":blue[**:material/lightbulb: O que isso significa:**] a inadimplência desse grupo ficou "
                f"**{br(vezes, 1)} vezes maior** desde {primeiro_mes:%m/%Y}. Quando uma piora acontece em "
                "muitos estados ao mesmo tempo, é sinal de algo que afeta o setor como um todo, e não de "
                "um problema local. É por isso que esses recortes aparecem no topo do ranking."
            )

    # 4. Onde o risco é menor (só faz sentido comparando 3+ tipos de crédito) --
    if len(base_texto) >= 3:
        st.markdown("#### :blue[4. Onde o risco é menor]")
        menores = base_texto.sort_values("mean").head(2)
        st.markdown(
            "Os tipos de crédito com as menores notas médias são "
            + " e ".join(
                f"**{nome_curto(mod).lower()}** ({br(linha['mean'])})" for mod, linha in menores.iterrows()
            )
            + ". Nesses grupos, o modelo encontra poucos sinais de alta da inadimplência."
        )
        if usa_confiaveis:
            st.caption("Considerando só os tipos de crédito com pelo menos 5 recortes no mês.")
        st.markdown(
            ":blue[**:material/lightbulb: O que isso significa:**] nota baixa não quer dizer risco zero. "
            "Quer dizer que, olhando os dados até agora, o modelo não vê sinais de que a inadimplência "
            "desses grupos vá subir de forma relevante nos próximos 6 meses."
        )

    # 5. O que o modelo aprendeu (vale para o modelo como um todo) -----------
    st.markdown("#### :blue[O que o modelo aprendeu]")
    st.markdown(
        """
Ao analisar quais informações mais ajudaram nas previsões (no modelo avaliado em `05_modelo.ipynb`):

- **O atraso curto é o melhor sinal de alerta.** A taxa de atraso de 15 a 90 dias foi a informação
  mais usada pelo modelo. Faz sentido: quem atrasa hoje pode virar inadimplente nos meses seguintes.
- **O tipo de crédito e o tipo de cliente pesam muito.** Empréstimos, financiamentos imobiliários e
  crédito rural se comportam de formas diferentes, e pessoas físicas são diferentes de empresas.
- **O estado pesa pouco.** A UF foi a informação menos usada. O risco depende mais de *que tipo de
  crédito* é do que de *onde* ele está.
"""
    )

    # Como usar --------------------------------------------------------------
    st.info(
        "Use a nota para **priorizar onde olhar primeiro**: os recortes do topo merecem mais atenção. "
        "A nota não é uma certeza nem uma chance exata. Use a **barra lateral** para personalizar a "
        "análise; veja a lista completa na aba **Ranking** e a história de um recorte na aba **Recorte**.",
        title="Como usar esses resultados",
        icon=":material/tips_and_updates:",
    )

# --- Aba 1: ranking ---------------------------------------------------------
with aba_ranking:
    tabela = notas_f[notas_f["data_base"] == mes]

    if tabela.empty:
        st.warning("Nenhum recorte com esses filtros.", icon=":material/search_off:")
    else:
        # Cartões de resumo
        historico_filtro = dados_f
        serie_taxa = (
            historico_filtro.groupby("data_base").apply(taxa_ponderada, include_groups=False)
        )
        mes_anterior = mes - pd.DateOffset(months=6) + pd.offsets.MonthEnd(0)
        taxa_atual = serie_taxa.loc[mes]
        taxa_anterior = serie_taxa.loc[mes_anterior]

        topo = tabela.loc[tabela["nota_risco"].idxmax()]
        por_modalidade = tabela.groupby("modalidade", observed=True)["nota_risco"].mean()
        modalidade_topo = por_modalidade.idxmax()

        with st.grid(4, gap="medium", min_column_width=200):
            st.metric(
                "Recortes no filtro",
                len(tabela),
                icon=":material/grid_view:",
                help="Quantos grupos de empréstimos (estado × tipo de cliente × modalidade) "
                "entram nos filtros da barra lateral. Sem filtro = todos os 386.",
                border=True,
                height="stretch",
            )
            st.metric(
                "Inadimplência do filtro",
                f"{br(taxa_atual)}%",
                delta=f"{br(taxa_atual - taxa_anterior)} p.p.",
                delta_color="inverse",  # subir é ruim: vermelho
                delta_description="vs. 6 meses antes",
                chart_data=serie_taxa.loc[:mes].tolist(),
                chart_type="area",
                icon=":material/percent:",
                help="De todo o dinheiro emprestado nos recortes filtrados, quanto está com "
                "pagamento muito atrasado. A seta compara com 6 meses antes: vermelho = piorou, "
                "verde = melhorou. O gráfico mostra a evolução desde 2022.",
                border=True,
                height="stretch",
            )
            st.metric(
                "Maior nota de risco",
                br(topo["nota_risco"]),
                icon=":material/priority_high:",
                help=f"O recorte com mais sinais de alta da inadimplência: {topo['uf']} · "
                f"{topo['cliente']} · {topo['modalidade']}. Ele aparece no topo da tabela abaixo.",
                border=True,
                height="stretch",
            )
            # Mesmo formato dos outros cartões: o tipo de crédito vai no rótulo e a nota no número
            st.metric(
                f"Maior nota média: {nome_curto(modalidade_topo).lower()}",
                br(por_modalidade.max()),
                icon=":material/category:",
                help=f"{modalidade_topo}: é o tipo de crédito que, na média dos estados, mais "
                "preocupa o modelo.",
                border=True,
                height="stretch",
            )

        st.subheader(":blue[Recortes por nota de risco]", icon=":material/leaderboard:")
        st.markdown(
            "Cada linha é um recorte. **Os do topo têm mais sinais de alta da inadimplência nos 6 meses "
            f"depois de {mes:%m/%Y}.** Use os filtros da barra lateral para focar num tipo de cliente, "
            "modalidade ou estado."
        )
        st.info(
            "A nota (0 a 1) serve para **ordenar** os recortes: quanto maior, mais risco em relação "
            "aos outros. O modelo foi treinado dando mais peso aos casos de alta, então as notas "
            "ficam acima da frequência real.",
            title="A nota de risco não é uma probabilidade",
            icon=":material/info:",
        )

        # Posição entre os 386 recortes do mês (1 = maior nota), mesmo quando há filtro
        posicao_geral = (
            notas[notas["data_base"] == mes]["nota_risco"].rank(ascending=False, method="min").astype(int)
        )
        exibir = (
            tabela.sort_values("nota_risco", ascending=False)
            .assign(
                posicao=lambda d: posicao_geral.loc[d.index],
                carteira_bi=lambda d: d["carteira_ativa"] / 1e9,
                # Nome curto: tira o trecho entre parênteses que cortava a coluna
                modalidade=lambda d: d["modalidade"].astype(str).map(nome_curto),
            )
            # A nota vem primeiro para ficar visível mesmo em telas estreitas
            [["posicao", "nota_risco", "uf", "cliente", "modalidade", "taxa_inad", "taxa_atraso_15_90",
              "var_taxa_inad_3m", "carteira_bi"]]
            # Arredonda antes de exibir: com format="localized", o navegador mostra
            # vírgula decimal (3,18) para visitantes brasileiros, como no resto do app
            .round({"nota_risco": 2, "taxa_inad": 2, "taxa_atraso_15_90": 2,
                    "var_taxa_inad_3m": 2, "carteira_bi": 1})
        )
        st.dataframe(
            exibir,
            hide_index=True,
            # help= mostra uma explicação quando o mouse passa sobre o nome da coluna
            column_config={
                "posicao": st.column_config.NumberColumn(
                    "Posição", format="%dº",
                    help="Lugar do recorte entre os 386, do maior para o menor risco. 1º = maior nota.",
                ),
                "nota_risco": st.column_config.ProgressColumn(
                    "Nota de risco", min_value=0.0, max_value=1.0, format="localized",
                    help="De 0 a 1: quanto maior, mais sinais de alta da inadimplência nos "
                    "próximos 6 meses. Serve para comparar recortes, não é uma chance em %.",
                ),
                "uf": st.column_config.TextColumn("UF", help="Estado."),
                "cliente": st.column_config.TextColumn(
                    "Cliente", help="PF = pessoa física. PJ = pessoa jurídica (empresa)."
                ),
                "modalidade": st.column_config.TextColumn(
                    "Modalidade", width="medium", help="O tipo de crédito."
                ),
                "taxa_inad": st.column_config.NumberColumn(
                    "Inadimplência (%)", format="localized",
                    help="Parte do dinheiro emprestado com pagamento muito atrasado no mês-base.",
                ),
                "taxa_atraso_15_90": st.column_config.NumberColumn(
                    "Atraso 15-90 dias (%)", format="localized",
                    help="Parte do dinheiro emprestado com atraso curto (15 a 90 dias). "
                    "Costuma ser um sinal de alerta.",
                ),
                "var_taxa_inad_3m": st.column_config.NumberColumn(
                    "Var. inadimplência 3 meses (p.p.)", format="localized",
                    help="Quanto a inadimplência mudou nos últimos 3 meses, em pontos percentuais. "
                    "Positivo = subiu.",
                ),
                "carteira_bi": st.column_config.NumberColumn(
                    "Carteira (R$ bi)", format="localized",
                    help="Total de dinheiro emprestado no recorte, em bilhões de reais.",
                ),
            },
            alt=f"Recortes ordenados pela nota de risco no mês-base {mes:%m/%Y}",
        )

        st.subheader(":blue[Por que esse recorte tem essa nota?]", icon=":material/psychology_alt:")
        ordenada = tabela.sort_values("nota_risco", ascending=False)
        escolhido = st.selectbox(
            "Escolha um recorte da lista para ver a explicação",
            ordenada.index,
            format_func=lambda i: f"{posicao_geral.loc[i]}º · {descrever(ordenada.loc[i])} "
            f"(nota {br(ordenada.loc[i, 'nota_risco'])})",
            key="rk_explica",
        )
        with st.container(border=True):
            explicar_recorte(ordenada.loc[escolhido], ordenada.loc[escolhido, "nota_risco"], mes)

# --- Aba 2: evolução de um recorte -----------------------------------------
with aba_recorte:
    st.subheader(":blue[Evolução de um recorte]", icon=":material/timeline:")
    st.markdown(
        "Escolha um tipo de cliente, uma modalidade e um estado (as opções seguem os filtros da "
        "barra lateral) para ver **como a inadimplência "
        "desse grupo mudou mês a mês** desde 2022 e **qual nota de risco o modelo deu a ele**. "
        "O mês dos cartões é o mês-base escolhido na barra lateral."
    )

    def posicao(lista, preferido):
        """Posição do valor preferido na lista (para abrir com um exemplo útil), ou 0."""
        return lista.index(preferido) if preferido in lista else 0

    # Cada escolha limita as opções seguintes às combinações que existem nos dados.
    # O exemplo inicial é PF / Empréstimos / SP, um dos maiores recortes.
    col1, col2, col3 = st.columns([1, 2, 1])
    # As opções respeitam os filtros da barra lateral
    lista = sorted(dados_f["cliente"].unique())
    cliente = col1.segmented_control(
        "Tipo de cliente", lista, default=lista[posicao(lista, "PF")], required=True, width="stretch"
    )
    opcoes = dados_f[dados_f["cliente"] == cliente]
    lista = sorted(opcoes["modalidade"].unique())
    modalidade = col2.selectbox(
        "Modalidade", lista, index=posicao(lista, "Empréstimos"), format_func=nome_curto
    )
    opcoes = opcoes[opcoes["modalidade"] == modalidade]
    lista = sorted(opcoes["uf"].unique())
    uf = col3.selectbox("UF", lista, index=posicao(lista, "SP"))

    historico = dados[
        (dados["cliente"] == cliente) & (dados["modalidade"] == modalidade) & (dados["uf"] == uf)
    ].sort_values("data_base")
    notas_recorte = notas[
        (notas["cliente"] == cliente) & (notas["modalidade"] == modalidade) & (notas["uf"] == uf)
    ].sort_values("data_base")

    # Cartões do recorte no mês-base escolhido na barra lateral
    linha_mes = historico[historico["data_base"] == mes].iloc[0]
    nota_mes = notas_recorte.loc[notas_recorte["data_base"] == mes, "nota_risco"].iloc[0]
    mediana_mes = notas.loc[notas["data_base"] == mes, "nota_risco"].median()

    with st.grid(3, gap="medium", min_column_width=200):
        st.metric(
            f"Inadimplência em {mes:%m/%Y}",
            f"{br(linha_mes['taxa_inad'])}%",
            icon=":material/percent:",
            help="Parte do dinheiro emprestado neste recorte com pagamento muito atrasado.",
            border=True,
            height="stretch",
        )
        st.metric(
            f"Atraso 15-90 dias em {mes:%m/%Y}",
            f"{br(linha_mes['taxa_atraso_15_90'])}%",
            icon=":material/schedule:",
            help="Parte do dinheiro emprestado com atraso curto (15 a 90 dias), um sinal de alerta.",
            border=True,
            height="stretch",
        )
        st.metric(
            "Nota de risco",
            br(nota_mes),
            delta=br(nota_mes - mediana_mes),
            delta_color="inverse",
            delta_description="vs. mediana dos 386 recortes",
            icon=":material/priority_high:",
            help="A seta compara com a nota típica (mediana) de todos os recortes no mesmo mês. "
            "Vermelho = risco acima do típico. Verde = abaixo.",
            border=True,
            height="stretch",
        )

    # Explicação em linguagem simples: história do recorte + por que a nota é essa
    st.markdown("#### :blue[O que está acontecendo com este recorte]")
    with st.container(border=True):
        inicio_r = historico.iloc[0]
        var_6m = linha_mes["var_taxa_inad_6m"]
        if var_6m > 0.1:
            direcao = f"**subiu {br(var_6m)} ponto** nos últimos 6 meses"
        elif var_6m < -0.1:
            direcao = f"**caiu {br(-var_6m)} ponto** nos últimos 6 meses"
        else:
            direcao = "ficou **praticamente estável** nos últimos 6 meses"
        st.markdown(
            f"Desde {inicio_r['data_base']:%m/%Y}, a inadimplência deste recorte passou de "
            f"**{br(inicio_r['taxa_inad'])}%** para **{br(linha_mes['taxa_inad'])}%** "
            f"em {mes:%m/%Y}, e {direcao}."
        )
        explicar_recorte(linha_mes, nota_mes, mes)

    col_linhas, col_barras = st.columns([3, 2], gap="medium")

    with col_linhas.container(border=True):
        st.markdown("**Taxas mensais (% da carteira ativa)**")
        nomes_series = {"taxa_inad": "Inadimplência", "taxa_atraso_15_90": "Atraso 15-90 dias"}
        series = historico.melt(
            id_vars="data_base", value_vars=list(nomes_series), var_name="serie", value_name="taxa"
        )
        series["serie"] = series["serie"].map(nomes_series)
        # Texto já formatado com vírgula para a dica do gráfico (o Vega usaria ponto)
        series["taxa_txt"] = series["taxa"].map(br) + "%"

        # Faixa azul-clara destacando os meses que o modelo pontuou (jul a dez/2025)
        faixa = pd.DataFrame({"inicio": [notas["data_base"].min()], "fim": [notas["data_base"].max()]})
        destaque = alt.Chart(faixa).mark_rect(opacity=0.12, color="#3B82F6").encode(
            x="inicio:T", x2="fim:T"
        )
        linhas = alt.Chart(series).mark_line(strokeWidth=2).encode(
            x=alt.X("data_base:T", title=None, axis=alt.Axis(format="%m/%Y", tickCount=8)),
            y=alt.Y("taxa:Q", title="% da carteira ativa"),
            color=alt.Color(
                "serie:N",
                title=None,
                scale=alt.Scale(domain=list(nomes_series.values())),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("data_base:T", title="Mês", format="%m/%Y"),
                alt.Tooltip("serie:N", title="Série"),
                alt.Tooltip("taxa_txt:N", title="Taxa"),
            ],
        )
        st.altair_chart(
            destaque + linhas,
            alt=f"Inadimplência e atraso de 15 a 90 dias de {uf}, {cliente}, {nome_curto(modalidade)}, 2022 a 2025",
        )
        st.caption(
            "**Como ler:** a linha azul é a inadimplência e a laranja é o atraso de 15 a 90 dias. "
            "Uma linha subindo quer dizer mais pagamentos atrasados. A faixa azul-clara marca os meses "
            "(jul a dez/2025) para os quais o modelo fez previsões."
        )

    with col_barras.container(border=True):
        st.markdown("**Nota de risco por mês-base**")
        mediana_mensal = (
            notas.groupby("data_base")["nota_risco"].median().rename("mediana").reset_index()
        )
        comparacao = notas_recorte[["data_base", "nota_risco"]].merge(mediana_mensal, on="data_base")
        comparacao["mes"] = comparacao["data_base"].dt.strftime("%m/%Y")
        comparacao["nota_txt"] = comparacao["nota_risco"].map(br)
        comparacao["mediana_txt"] = comparacao["mediana"].map(br)

        base = alt.Chart(comparacao).encode(x=alt.X("mes:N", title=None, sort=None))
        barras = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            y=alt.Y("nota_risco:Q", title="Nota de risco", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("mes:N", title="Mês-base"),
                alt.Tooltip("nota_txt:N", title="Nota do recorte"),
                alt.Tooltip("mediana_txt:N", title="Mediana dos recortes"),
            ],
        )
        linha_mediana = base.mark_line(
            # Cinza-azulado neutro: a mediana é só uma referência (vermelho fica para "piorou").
            # #64748B tem contraste acima de 3:1 tanto no fundo claro quanto no escuro.
            color="#64748B", strokeDash=[4, 3], point=alt.OverlayMarkDef(color="#64748B")
        ).encode(
            y="mediana:Q"
        )
        st.altair_chart(
            barras + linha_mediana,
            alt=f"Nota de risco de {uf}, {cliente}, {nome_curto(modalidade)} comparada à mediana dos recortes",
        )
        st.caption(
            "**Como ler:** cada barra é a nota deste recorte num mês-base. A linha tracejada cinza "
            "é a nota típica (mediana) entre os 386 recortes. Barra acima da linha = risco acima do "
            "normal."
        )

# --- Aba 3: sobre o modelo -------------------------------------------------
with aba_modelo:
    st.subheader(":blue[Como o modelo funciona]", icon=":material/construction:")
    st.markdown(
        """
**Em palavras simples:** o modelo aprendeu com o passado. Para cada recorte e cada mês de 2022 a 2025,
ele viu como estavam os números (inadimplência, atrasos curtos, crescimento do crédito) e se, 6 meses
depois, a inadimplência subiu de forma relevante. Com isso, ele aprendeu quais combinações de sinais
costumam vir antes de uma alta.

O modelo é uma **floresta aleatória**: um conjunto de 300 "árvores de decisão". Cada árvore faz uma
sequência de perguntas simples, como *"o atraso de 15 a 90 dias está alto?"* ou *"a inadimplência
subiu nos últimos 3 meses?"*. A nota final combina as respostas de todas as árvores.
"""
    )

    with st.expander("Detalhes técnicos", icon=":material/code:"):
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

    st.subheader(":blue[O modelo é bom?]", icon=":material/fact_check:")
    st.markdown(
        """
Para saber, o modelo foi treinado só com dados até jun/2024 e depois **testado em 2025**, com dados
que ele nunca tinha visto. Como em 2025 já sabemos o que aconteceu, dá para conferir as previsões.

- **AUC:** pegue um recorte que teve alta e outro que não teve. Em quantas vezes o modelo dá nota maior
  para o que teve alta? Chutar acertaria 50% das vezes (0,5). **O modelo acerta 77% (0,77).**
- **AP:** mede se os recortes do topo da lista são mesmo os que tiveram alta. No teste, 25% dos casos
  tiveram alta, então chutar daria 0,25. **O modelo chegou a 0,53, mais que o dobro.**

A tabela compara o modelo com alternativas mais simples. Quanto maior a barra, melhor.
"""
    )
    st.caption("Modelos treinados com dados até jun/2024 e avaliados em jan a jun/2025 (05_modelo.ipynb).")
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
    st.dataframe(
        resultados,
        hide_index=True,
        column_config={
            "AUC": st.column_config.ProgressColumn("AUC", min_value=0.0, max_value=1.0, format="%.3f"),
            "AP": st.column_config.ProgressColumn("AP", min_value=0.0, max_value=1.0, format="%.3f"),
        },
        alt="AUC e AP de cada modelo no teste de 2025; a floresta aleatória tem os maiores valores",
    )

    st.subheader(":blue[O que o modelo não faz]", icon=":material/warning:")
    st.markdown(
        """
- **Não dá uma chance exata.** A nota serve para comparar recortes. Entre o período de treino e o de
  teste, as altas ficaram mais comuns (de 13% para 25% dos casos), e as notas não foram ajustadas para
  virar porcentagens.
- **Não olha para pessoas ou empresas específicas.** Ele trabalha com grupos (recortes), não com
  empréstimos individuais.
- **Não é uma avaliação totalmente "às cegas".** Os resultados do teste foram vistos antes da escolha
  final do modelo, o que pode deixá-los um pouco otimistas.
"""
    )

st.caption("Projeto de Alyce Abade · Dados: SCR.data, Banco Central do Brasil.")
