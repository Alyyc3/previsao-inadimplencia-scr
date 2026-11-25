# Previsão de alta da inadimplência de crédito com machine learning (SCR.data)

Projeto de ciência de dados com dados públicos do **Banco Central do Brasil**. A pergunta é:

> Para cada combinação de **UF × tipo de cliente (PF/PJ) × modalidade de crédito**, a inadimplência vai ter uma **alta relevante nos próximos 6 meses**?

Uma floresta aleatória treinada com dados de 2022 a 2024 e avaliada em 2025 alcançou **AUC de 0,770** e **precisão média (AP) de 0,527**, mais que o dobro da AP de um chute (0,249).

---

## Dados

- **Fonte:** SCR.data (Sistema de Informações de Crédito), Banco Central do Brasil. Arquivos anuais em `https://www.bcb.gov.br/pda/desig/scrdata_{ano}.zip`.
- **Período:** 2022 a 2025, 48 arquivos mensais, cerca de 690 MB compactados.
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

## Limitações e transparência

- **O teste foi visto antes da escolha final.** Os resultados no teste foram calculados numa análise exploratória antes de o modelo ser fixado. A escolha se apoia na validação temporal e no critério de sobreajuste, mas o número do teste não é uma avaliação totalmente "cega".
- **Mudança de regime:** a proporção de positivos sobe de 13% no treino para 25% no teste. O modelo ordena bem o risco, mas as probabilidades previstas não estão calibradas para o novo nível.
- **Validação pequena:** o sub-treino tem só 12 meses-base. Por isso, a diferença entre a floresta e o gradient boosting na validação é pouco conclusiva.
- **Sem ajuste de hiperparâmetros:** os modelos usam parâmetros fixos. Um ajuste exigiria uma validação temporal mais longa.
- **Importância por impureza:** a importância nativa das árvores tende a favorecer variáveis com muitos valores possíveis. Serve como visão geral, não como medida exata.
- **Dados agregados:** o modelo trabalha com recortes, não com operações ou clientes individuais.

## App no Streamlit

O app (`app/app.py`) usa a floresta aleatória **treinada de novo com todos os meses que já têm alvo** (jul/2022 a jun/2025, 13.896 linhas; `06_modelo_final.ipynb`). Com ela, pontua os meses-base de **jul a dez/2025**, cujo futuro ainda não está nos dados.

O app foi pensado para quem **não é da área**: os textos usam linguagem simples e explicam cada número.

### Como funciona

- **Barra lateral para personalizar:** escolha o **mês-base** (jul a dez/2025) e filtre por **tipo de cliente** (PF/PJ), **modalidade** e **UF**. Os filtros mudam os resultados, o ranking e a aba do recorte. Os textos são **gerados a partir dos dados**, então os números se ajustam ao grupo escolhido.
- **Guia "Primeira vez aqui?":** explica, em 1 minuto, o que é inadimplência, recorte, nota de risco e ponto percentual.

### Abas

| Aba | O que mostra |
|---|---|
| **Resultados** | Resumo em 3 números e os principais achados, cada um com um **"O que isso significa"**. Inclui um exemplo detalhado do recorte de maior risco. |
| **Ranking** | Os recortes ordenados pela nota, com a **posição** de cada um entre os 386. Escolhendo um recorte da lista, o app explica por que ele tem aquela nota. |
| **Recorte** | A história de um recorte desde 2022 (gráfico da inadimplência e do atraso curto) e a explicação da nota dele. |
| **Modelo** | Como o modelo funciona, se ele é bom (AUC e AP em palavras simples) e o que ele não faz. |

### Como o app explica a nota de um recorte

Para cada recorte, o app mostra:
- **Posição:** por exemplo, "nota 0,95, maior que a de 385 dos 386 recortes".
- **O que precisaria acontecer:** a taxa que a inadimplência teria que passar em 6 meses para contar como alta relevante, pela regra do alvo (+0,5 ponto **e** +20%).
- **Sinais de alerta e sinais tranquilizadores:** comparação com outros estados que têm o mesmo tipo de crédito e de cliente, olhando o atraso curto, a direção recente da inadimplência e o crédito problemático.

Isso ajuda a entender casos que parecem contraditórios. Em dez/2025, **SP · PF · empréstimos** tem inadimplência alta (10,59%), mas nota baixa (0,18). Para contar como alta relevante, a taxa teria que passar de **12,71%** em 6 meses, um salto que o modelo não vê força para acontecer.

A comparação com outros estados é uma explicação simplificada. O modelo combina essas e outras informações de forma mais complexa.

### A nota de risco

A **nota de risco** (0 a 1) serve para **ordenar** os recortes. Ela **não é uma probabilidade**, porque o modelo foi treinado dando mais peso aos casos de alta e as notas não foram calibradas. Ela mede **sinais de que a inadimplência vai subir**, e não o tamanho da inadimplência hoje.

### O que o app mostra em dez/2025 (sem filtros)

- A inadimplência dos 386 recortes passou de **2,38%** (01/2022) para **4,09%**.
- Dos 20 recortes de maior nota, **12 são de financiamento rural**. No crédito rural para pessoas físicas, a inadimplência ficou **11,1 vezes maior** desde 01/2022 (de 0,54% para 6,02%).
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
├── app/
│   ├── app.py                    # app no Streamlit
│   ├── .streamlit/config.toml    # tema de cores (claro e escuro)
│   ├── modelo.joblib             # modelo treinado (gerado pelo 06)
│   └── dados_app.parquet         # histórico dos recortes (gerado pelo 06)
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

Depois rode os notebooks na ordem (`02` → `03` → `04` → `05` → `06`). O `02` baixa os ZIPs do Banco Central na primeira execução, cerca de 690 MB.

Para abrir o app, rode na pasta do projeto o comando abaixo. O modelo e os dados do app já estão no repositório, então não é preciso rodar os notebooks antes.

```powershell
streamlit run app/app.py
```

## Tecnologias

Python, pandas, pyarrow, matplotlib, scikit-learn, Streamlit e Jupyter.

## Autoria

Alyce Abade
