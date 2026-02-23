## Visão geral

Este é um **aplicativo web de Análise Exploratória de Dados (EDA)** construído com **Streamlit**. Ele permite que o usuário faça upload de um arquivo CSV e, a partir dele, obtenha estatísticas descritivas e visualizações interativas — tudo pelo navegador, sem escrever código.

---

## Estrutura dos arquivos

| Arquivo | Função |
|---|---|
| eda_main.py | Código principal do app (toda a lógica) |
| requirements.txt | Dependências Python (Streamlit, Pandas, Plotly, Seaborn, etc.) |
| setup.sh | Script de configuração do Streamlit para deploy no Heroku |
| Procfile | Comando de inicialização para o Heroku |
| README.md | Documentação do projeto |

---

## Detalhamento de eda_main.py

### 1. Classe `EDA` (linhas 15-82)

Encapsula todos os métodos de visualização. No construtor, separa automaticamente as colunas do DataFrame em **numéricas** (`num_vars`) e **categóricas** (`cat_vars`). Os métodos de gráficos são:

- **`box_plot`** — Boxplot interativo via Plotly, com suporte a variável x e hue (cor).
- **`violin`** — Gráfico violino (Seaborn), com opção de split.
- **`swarmplot`** — Swarm plot (Seaborn), mostra pontos individuais.
- **`histogram_num`** — Histograma interativo (Plotly) com filtro de range e violin marginal.
- **`scatter_plot`** — Gráfico de dispersão (Plotly) com hue e tamanho variável.
- **`bar_plot`** — Gráfico de barras (Plotly).
- **`line_plot`** — Gráfico de linhas (Plotly) com agrupamento.
- **`CountPlot`** — Contagem de frequências por categoria (Seaborn).
- **`heatmap_vars`** — Heatmap de tabela pivot entre 3 variáveis com função de agregação (média, soma, mediana).
- **`Corr`** — Heatmap de correlação (Pearson, Kendall ou Spearman).
- **`DistPlot`** — Gráfico de distribuição com rug plot (Seaborn).

### 2. Funções auxiliares de dados e estatísticas (linhas 84-175)

- **`get_data`** — Lê o CSV com cache do Streamlit para performance.
- **`get_stats`** — Retorna `describe()` separado para variáveis numéricas e categóricas.
- **`get_info`** — Gera tabela com tipos, contagem de NaN, percentual de NaN e valores únicos por coluna.
- **`pd_of_stats`** — Calcula média, desvio padrão, variância, curtose, assimetria e coeficiente de variação.
- **`pf_of_info`** — Calcula tipo, valores únicos, nº de zeros, % de zeros, nº de NaN e % de NaN.
- **`pd_of_stats_quantile`** — Calcula mín, Q1, mediana, Q3, máx, range e IQR.
- **`get_table_download_link`** — Gera link de download do DataFrame em CSV (codificado em base64).

### 3. Tratamento de valores nulos (linhas 98-148)

Duas funções para imputação de dados faltantes:

- **`input_null`** (numéricas) — Oferece preenchimento por: Média, Mediana, Moda, Forward Fill, Backward Fill, valor manual, ou remoção de linhas.
- **`input_null_cat`** (categóricas) — Oferece preenchimento por texto ou remoção de linhas.

Ambas exibem uma comparação antes/depois da imputação.

### 4. Funções de visualização da interface (linhas 190-332)

- **`plot_univariate`** — Renderiza gráficos para análise de **uma variável**: Histograma, BoxPlot ou Distribution Plot. Os controles (bins, range, hue) ficam na sidebar.
- **`plot_multivariate`** — Renderiza gráficos para análise de **múltiplas variáveis**: Correlation, Boxplot, Violin, Swarmplot, Heatmap, Histogram, Scatterplot, Countplot, Barplot e Lineplot. Cada tipo tem seus próprios controles na sidebar.

### 5. Função `main()` (linhas 334-420)

Orquestra todo o fluxo do aplicativo:

1. **Upload** — O usuário faz upload de um arquivo CSV.
2. **Info básica** — Mostra nº de observações, variáveis e % de valores faltantes.
3. **Menu lateral** com 3 opções:
   - **"View statistics"** — Exibe resumo numérico (`describe`), resumo categórico e tabela de valores faltantes.
   - **"Statistic univariate"** — O usuário seleciona uma variável; se numérica, mostra info, estatísticas descritivas, quantis e opções de gráfico; se categórica, mostra `describe` e gráfico de barras de frequência. Há também um explorador de variáveis categóricas na sidebar.
   - **"Statistic multivariate"** — Oferece 10 tipos de gráficos para explorar relações entre variáveis.

---

## Infraestrutura de deploy

- **setup.sh** — Cria os arquivos de configuração do Streamlit (`credentials.toml` e `config.toml`) com modo headless e CORS desabilitado, necessários para rodar no Heroku.
- **Procfile** — Define o comando `web: sh setup.sh && streamlit run eda_main.py`.

---

**Em resumo:** é uma ferramenta "no-code" para cientistas de dados fazerem análise exploratória rápida — basta subir um CSV e o app gera estatísticas descritivas, detecta valores faltantes (com opções de tratamento) e oferece uma variedade de gráficos interativos (Plotly + Seaborn), tudo via interface web Streamlit.
