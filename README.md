# Previsão de alta da inadimplência de crédito com machine learning (SCR.data)

Projeto de ciência de dados com dados públicos do **Banco Central do Brasil**. A pergunta é:

> Para cada combinação de **UF × tipo de cliente (PF/PJ) × modalidade de crédito**, a inadimplência vai ter uma **alta relevante nos próximos 6 meses**?

Uma floresta aleatória treinada com dados de 2022 a 2024 e avaliada em 2025 alcançou **AUC de 0,770** e **precisão média (AP) de 0,527**, mais que o dobro da AP de um chute (0,249).

Depois, com os dados de 2026, veio um **teste de verdade**: o modelo do app, sem nenhum ajuste, foi conferido nos meses-base de jul/2025 a mar/2026, que ele nunca tinha visto. Resultado: **AUC de 0,780** e **AP de 0,512**, contra 0,196 de um chute.

**Veja o app funcionando:** https://previsao-inadimplencia-scr.streamlit.app

---

## Dados

- **Fonte:** SCR.data (Sistema de Informações de Crédito), Banco Central do Brasil. Arquivos anuais em `https://www.bcb.gov.br/pda/desig/scrdata_{ano}.zip`.
- **Período:** 2022 a 2025 para construir e avaliar o modelo (48 arquivos mensais, cerca de 690 MB compactados). Depois, **jan a set/2026** (9 arquivos, cerca de 136 MB) para o app e para o teste com dados novos.
- **Colunas usadas:** carteira ativa, carteira inadimplente, carteira a vencer, a vencer até 90 dias, vencido de 15 a 90 dias, vencido acima de 90 dias e ativo problemático.

Os dados brutos não estão no repositório. O código baixa os arquivos na primeira execução (veja [Como reproduzir](#como-reproduzir)).

## Abordagem

### 1. Painel mensal e filtro dos recortes (`02_painel_mensal.ipynb`)

Os dados mensais foram somados por **data × UF × cliente × modalidade**, formando 624 recortes. Ficaram no estudo **386 recortes**, os que têm:
- os **48 meses completos**, para evitar buracos na série;
- **carteira ativa mediana de pelo menos R$ 100 milhões**.

O motivo do segundo critério: entre R$ 1 milhão e R$ 100 milhões de carteira, um quarto dos recortes tem variação mensal típica da inadimplência acima de 2 pontos percentuais, o que é puro ruído. A partir de R$ 100 milhões, essa variação cai para cerca de 0,1 ponto. Abaixo de R$ 1 milhão, a taxa quase não varia, porque costuma ficar zerada. Os recortes descartados somam só **0,07% da carteira total**.

### 2. Alvo (`03_alvo.ipynb`)

**Alvo = 1** quando a taxa de inadimplência daqui a 6 meses estiver ao mesmo tempo:
- mais de **0,5 ponto percentual** acima da taxa atual, **e**
- mais de **20%** acima da taxa atual.

As duas condições se completam:
- só a regra em pontos percentuais trataria igual um recorte com 0,5% de inadimplência (onde +0,5 p.p. dobra a taxa) e um com 7%;
- só a regra relativa marcaria como alta relevante uma subida de 0,10% para 0,13%.

Resultado: **16.212 linhas com alvo, 16,4% positivas**. A proporção muda bastante com o tempo: 16,0% em 2022, 11,0% em 2023, 18,1% em 2024 e 24,9% em 2025.

### 3. Variáveis (`04_variaveis.ipynb`)

Cada variável do mês *t* usa **só dados até o mês *t***, para evitar vazamento de informação do futuro.

| Variável | Descrição |
|---|---|
| `taxa_inad` | inadimplência ÷ carteira ativa (%) |
| `taxa_atraso_15_90` | vencido de 15 a 90 dias ÷ carteira ativa (%) |
| `taxa_problematico` | ativo problemático ÷ carteira ativa (%) |
| `var_taxa_inad_3m`, `var_taxa_inad_6m` | variação da inadimplência em 3 e 6 meses (p.p.) |
| `var_atraso_15_90_3m` | variação do atraso de 15 a 90 dias em 3 meses (p.p.) |
| `cresc_carteira_3m`, `cresc_carteira_6m` | crescimento da carteira ativa em 3 e 6 meses (%) |
| `uf`, `cliente`, `modalidade` | categóricas |

### 4. Separação treino/teste por data

A separação **não é aleatória**. Ela simula o uso real do modelo em janeiro de 2025:

| Parte | Meses-base | Linhas | % alvo = 1 |
|---|---|---|---|
| Treino | jul/2022 a jun/2024 | 9.264 | 13,0% |
| Intervalo (fora) | jul/2024 a dez/2024 | 2.316 | 23,6% |
| Teste | jan/2025 a jun/2025 | 2.316 | 24,9% |

O intervalo de 6 meses existe porque, em janeiro de 2025, o alvo desses meses ainda não seria conhecido. Por exemplo, o alvo de outubro/2024 depende de abril/2025.

### 5. Modelos (`05_modelo.ipynb`)

Todos os modelos usam um `Pipeline` do scikit-learn com `ColumnTransformer`, `OneHotEncoder` nas categóricas e `StandardScaler` nas numéricas (este só na regressão logística). Os modelos usam `class_weight="balanced"` porque os positivos são minoria.

A escolha foi feita numa **validação temporal dentro do treino**: sub-treino de jul/2022 a jun/2023 e validação de jan a jun/2024. A floresta aleatória e o gradient boosting ficaram próximos (AUC 0,704 contra 0,723), mas o gradient boosting quase decorou o treino (AUC de 0,998 no treino contra 0,920 da floresta). Por isso a escolha foi a **floresta aleatória**.

## Resultados no teste (jan a jun/2025)

| Modelo | AUC | AP |
|---|---|---|
| Chute (proporção de positivos) | 0,500 | 0,249 |
| Regra simples: ordenar por `taxa_atraso_15_90` | 0,690 | 0,369 |
| Regressão logística | 0,678 | 0,414 |
| Gradient boosting | 0,712 | 0,462 |
| **Floresta aleatória** | **0,770** | **0,527** |

- **AUC:** probabilidade de o modelo dar nota maior a um recorte que teve alta do que a um que não teve. Vale 0,5 para um chute e 1,0 para um modelo perfeito.
- **AP (precisão média):** resume a precisão ao longo da lista ordenada por risco. Um chute vale a proporção de positivos.

### Importância das variáveis (floresta aleatória)

| Variável | Importância |
|---|---|
| `taxa_atraso_15_90` | 0,147 |
| `modalidade` | 0,136 |
| `taxa_problematico` | 0,135 |
| `taxa_inad` | 0,134 |
| `var_atraso_15_90_3m` | 0,123 |
| `cliente` | 0,109 |
| `var_taxa_inad_3m` | 0,060 |
| `var_taxa_inad_6m` | 0,057 |
| `cresc_carteira_6m` | 0,041 |
| `cresc_carteira_3m` | 0,033 |
| `uf` | 0,024 |

**O que a tabela mostra:**
- O **atraso curto (15 a 90 dias)** é o melhor sinal antecipado. Quem atrasa hoje tende a virar inadimplente (mais de 90 dias) nos meses seguintes.
- O **tipo de crédito** e o **tipo de cliente** pesam mais do que o **estado**.

## Teste de verdade com dados de 2026 (`07_dados_2026.ipynb`)

O Banco Central publicou os dados de jan a set/2026. O `07_dados_2026.ipynb` acrescenta esses meses aos mesmos 386 recortes e recalcula o alvo com a mesma regra. **O modelo não foi treinado de novo**: o `app/modelo.joblib` é o mesmo do passo 6, treinado com meses-base até jun/2025.

Com isso, os meses-base de **jul/2025 a mar/2026** viram um teste totalmente "cego": o resultado de 6 meses depois já é conhecido, e o modelo nunca viu esses meses nem foi escolhido olhando para eles.

| Período (meses-base) | Linhas | Positivos | AUC | AP |
|---|---|---|---|---|
| Chute | | | 0,500 | 0,196 |
| Regra simples: ordenar por `taxa_atraso_15_90` | | | 0,637 | 0,291 |
| **Floresta aleatória** (jul/2025 a mar/2026) | 3.472 | 19,6% | **0,780** | **0,512** |

O desempenho ficou no mesmo nível do teste de 2025 (AUC 0,770 e AP 0,527). Isso indica que o modelo **continua funcionando** com dados novos.

Mês a mês, dos **20 recortes com maior nota**, quantos tiveram alta relevante de fato:

| Mês-base | 07/25 | 08/25 | 09/25 | 10/25 | 11/25 | 12/25 | 01/26 | 02/26 | 03/26 |
|---|---|---|---|---|---|---|---|---|---|
| Altas no top 20 | 20 | 17 | 17 | 10 | 13 | 13 | 12 | 11 | 11 |
| % de altas entre todos | 17,1% | 19,7% | 20,0% | 20,2% | 22,0% | 21,8% | 20,2% | 18,1% | 16,9% |

Escolhendo 20 recortes ao acaso, seriam cerca de 4 altas por mês. O topo do ranking concentrou de 10 a 20.

**Um mês faltando:** o recorte SP · PJ · financiamentos de títulos e valores mobiliários não aparece no arquivo de mar/2026 do Banco Central. O mês ficou vazio, sem valor inventado. Por isso, esse recorte fica sem nota nos meses-base que dependem de mar/2026 (mar, jun e set/2026) e sem alvo em set/2025 e mar/2026.

## Limitações e transparência

- **O teste de 2025 foi visto antes da escolha final.** Os resultados no teste foram calculados numa análise exploratória antes de o modelo ser fixado. A escolha se apoia na validação temporal e no critério de sobreajuste, mas o número do teste não é uma avaliação totalmente "cega". O teste com dados de 2026 (seção anterior) é cego.
- **Mudança de regime:** a proporção de positivos sobe de 13% no treino para 25% no teste. O modelo ordena bem o risco, mas as probabilidades previstas não estão calibradas para o novo nível.
- **Validação pequena:** o sub-treino tem só 12 meses-base. Por isso, a diferença entre a floresta e o gradient boosting na validação é pouco conclusiva.
- **Sem ajuste de hiperparâmetros:** os modelos usam parâmetros fixos. Um ajuste exigiria uma validação temporal mais longa.
- **Importância por impureza:** a importância nativa das árvores tende a favorecer variáveis com muitos valores possíveis. Serve como visão geral, não como medida exata.
- **Dados agregados:** o modelo trabalha com recortes, não com operações ou clientes individuais.

## App no Streamlit

**Acesse:** https://previsao-inadimplencia-scr.streamlit.app (publicado no Streamlit Community Cloud; se o app estiver "dormindo", clique no botão para acordá-lo e aguarde cerca de 1 minuto).

O app (`app/app.py`) usa a floresta aleatória **treinada de novo com todos os meses que já têm alvo** (jul/2022 a jun/2025, 13.896 linhas; `06_modelo_final.ipynb`). Com ela, pontua os meses-base de **jul/2025 a set/2026**, que o modelo nunca viu. Para os de jul/2025 a mar/2026, o resultado de 6 meses depois já saiu, e o app mostra se a alta aconteceu de fato.

O app foi pensado para quem **não é da área**: os textos usam linguagem simples e explicam cada número.

### Como funciona

- **Barra lateral para personalizar:** escolha o **mês-base** (jul/2025 a set/2026) e filtre por **tipo de cliente** (PF/PJ), **modalidade** e **UF**. Os filtros mudam os resultados, o ranking e a aba do recorte. Os textos são **gerados a partir dos dados**, então os números se ajustam ao grupo escolhido.
- **Guia "Primeira vez aqui?":** explica, em 1 minuto, o que é inadimplência, recorte, nota de risco e ponto percentual.

### Abas

| Aba | O que mostra |
|---|---|
| **Resultados** | Resumo em 3 números e os principais achados, cada um com um **"O que isso significa"**. Inclui um exemplo detalhado do recorte de maior risco e, nos meses-base já conferidos, **quantos dos recortes do topo tiveram alta de fato**. |
| **Ranking** | Os recortes ordenados pela nota, com a **posição** de cada um entre os 386. Nos meses-base já conferidos, a coluna **Alta aconteceu?** mostra o resultado real. Escolhendo um recorte da lista, o app explica por que ele tem aquela nota. |
| **Recorte** | A história de um recorte desde 2022 (gráfico da inadimplência e do atraso curto) e a explicação da nota dele. |
| **Modelo** | Como o modelo funciona, se ele é bom (AUC e AP em palavras simples), o **teste de verdade** com dados de 2026 e o que o modelo não faz. |

### Como o app explica a nota de um recorte

Para cada recorte, o app mostra:
- **Posição:** por exemplo, "nota 0,95, maior que a de 385 dos 386 recortes".
- **O que precisaria acontecer:** a taxa que a inadimplência teria que passar em 6 meses para contar como alta relevante, pela regra do alvo (+0,5 ponto **e** +20%).
- **Sinais de alerta e sinais tranquilizadores:** comparação com outros estados que têm o mesmo tipo de crédito e de cliente, olhando o atraso curto, a direção recente da inadimplência e o crédito problemático.

Isso ajuda a entender casos que parecem contraditórios. Em dez/2025, **SP · PF · empréstimos** tem inadimplência alta (10,59%), mas nota baixa (0,18). Para contar como alta relevante, a taxa teria que passar de **12,71%** em 6 meses, um salto que o modelo não vê força para acontecer. Com os dados de 2026, deu para conferir: em jun/2026 a taxa estava em 11,38%, **sem alta relevante**, como o modelo indicava.

A comparação com outros estados é uma explicação simplificada. O modelo combina essas e outras informações de forma mais complexa.

### A nota de risco

A **nota de risco** (0 a 1) serve para **ordenar** os recortes. Ela **não é uma probabilidade**, porque o modelo foi treinado dando mais peso aos casos de alta e as notas não foram calibradas. Ela mede **sinais de que a inadimplência vai subir**, e não o tamanho da inadimplência hoje.

### O que o app mostra em set/2026 (sem filtros)

- A inadimplência dos 386 recortes passou de **2,38%** (01/2022) para **4,99%**.
- Dos 20 recortes de maior nota, **10 são de financiamento rural**. No crédito rural para pessoas físicas, a inadimplência ficou **14,5 vezes maior** desde 01/2022 (de 0,54% para 7,84%).
- Em dez/2025, os 10 recortes de maior nota eram todos de crédito rural para pessoas físicas. Seis meses depois, **5 deles tiveram alta relevante**.
- O sinal aparece em muitos estados ao mesmo tempo, o que aponta para um movimento do **setor**, e não de um estado específico.

### Visual

Tema **"Azul confiança"**, com barra lateral azul-marinho e modos claro e escuro. O tema fica em `app/.streamlit/config.toml`, e os contrastes de cor foram conferidos pelas regras de acessibilidade (WCAG).

## Estrutura do repositório

```
├── 01_exploracao_inicial.ipynb   # primeira leitura de um arquivo mensal
├── 02_painel_mensal.ipynb        # painel por recorte e filtro dos 386 recortes
├── 03_alvo.ipynb                 # definição do alvo
├── 04_variaveis.ipynb            # variáveis explicativas e separação por data
├── 05_modelo.ipynb               # treino, validação, teste e importância
├── 06_modelo_final.ipynb         # modelo final e arquivos do app
├── 07_dados_2026.ipynb           # meses de 2026 no app e teste com dados novos
├── app/
│   ├── app.py                    # app no Streamlit
│   ├── .streamlit/config.toml    # tema de cores (claro e escuro)
│   ├── modelo.joblib             # modelo treinado (gerado pelo 06)
│   └── dados_app.parquet         # histórico dos recortes (gerado pelo 06, atualizado pelo 07)
├── src/
│   ├── dados.py                  # download do SCR.data e montagem do painel
│   └── variaveis.py              # cálculo das variáveis do modelo
├── data/                         # gerada pelos notebooks (fora do Git)
│   ├── raw/                      # ZIPs anuais do BCB
│   └── processed/                # painéis e base do modelo em parquet
└── requirements.txt
```

## Como reproduzir

Requer Python 3.12. No Windows (PowerShell):

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
```

Depois rode os notebooks na ordem (`02` → `03` → `04` → `05` → `06` → `07`). O `02` baixa os ZIPs de 2022 a 2025 do Banco Central na primeira execução (cerca de 690 MB), e o `07` baixa o de 2026 (cerca de 136 MB).

Para abrir o app, rode na pasta do projeto o comando abaixo. O modelo e os dados do app já estão no repositório, então não é preciso rodar os notebooks antes.

```powershell
streamlit run app/app.py
```

## Tecnologias

Python, pandas, pyarrow, matplotlib, scikit-learn, Streamlit e Jupyter.

## Autoria

Alyce Abade
