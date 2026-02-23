## Plan: Modernização do Exploratory Data Analysis App

O projeto atual é um app Streamlit de 2020, com 420 linhas em um único arquivo, dependências desatualizadas (Streamlit 0.58, Pandas 1.0, NumPy 1.18), ~7 APIs removidas/deprecadas que impedem a execução em versões modernas, zero testes, zero type hints, e zero infraestrutura de engenharia. O objetivo é transformá-lo em um app profissional multi-page com profiling automático, suporte a múltiplos formatos, export de gráficos — mantendo Plotly + Seaborn — usando pip + requirements.txt e deploy no Streamlit Cloud.

---

### Fase 0 — Setup local e validação do estado atual

1. Criar o ambiente conda `eda-app` com Python 3.11+ e instalar as dependências atuais do [requirements.txt](Exploratory-Data-Analysis-App/requirements.txt)
2. Rodar `streamlit run eda_main.py` e documentar todos os erros e warnings (esperamos falhas por `np.object`, `st.cache`, `sns.distplot`, etc.)
3. Commit do estado original como baseline (tag `v0-legacy`)

### Fase 1 — Atualização de dependências e correção de breaking changes

4. Reduzir o [requirements.txt](Exploratory-Data-Analysis-App/requirements.txt) de 80 pacotes para as ~10 dependências diretas reais (o arquivo atual é um `pip freeze` que inclui boto3, jupyter, pywinpty, etc. — nada a ver com o app)
5. Atualizar para versões modernas: `streamlit>=1.40`, `pandas>=2.2`, `numpy>=2.1`, `seaborn>=0.13`, `plotly>=5.24`, `matplotlib>=3.9`, `scipy>=1.14`
6. Corrigir todas as APIs removidas/deprecadas em [eda_main.py](Exploratory-Data-Analysis-App/eda_main.py):
   - `np.object` → `"object"` (string) — 5 ocorrências nas linhas 21, 89, 91, 345, 393
   - `st.cache` → `st.cache_data` — 8 ocorrências; remover `allow_output_mutation`
   - `sns.distplot` → `sns.histplot(..., kde=True)` — linha 80
   - `df.fillna(method='ffill'/'bfill')` → `df.ffill()` / `df.bfill()` — linhas 110, 113
   - `st.pyplot()` sem argumentos → `st.pyplot(fig)` — 6 ocorrências
   - `df.corr()` → `df.select_dtypes('number').corr()` — linha 70
   - Duplicate `key` nos widgets `selectbox` → keys únicos — linhas 224-226, 231-233, 300-303
7. Rodar o app novamente e confirmar que todas as funcionalidades existentes funcionam sem erros
8. Commit: `fix: update all dependencies and fix deprecated APIs`

### Fase 2 — Reestruturação do código (engenharia)

9. Criar estrutura modular de projeto com `src/` layout:
   ```
   src/eda_app/
   ├── __init__.py
   ├── app.py                  # Entry point (main)
   ├── data/
   │   ├── loader.py           # get_data, parse CSV/Excel/Parquet
   │   └── preprocessing.py    # input_null, input_null_cat, transforms
   ├── stats/
   │   ├── descriptive.py      # pd_of_stats, pf_of_info, pd_of_stats_quantile, get_stats, get_info
   │   └── correlation.py      # Corr, heatmap_vars
   ├── visualization/
   │   ├── plots.py            # Classe EDA refatorada (métodos de gráfico)
   │   ├── univariate.py       # plot_univariate
   │   └── multivariate.py     # plot_multivariate
   └── components/
       ├── sidebar.py          # Lógica da sidebar
       └── download.py         # Export de gráficos e dados
   pages/
   ├── 1_📊_Overview.py
   ├── 2_📈_Univariate.py
   ├── 3_📉_Multivariate.py
   └── 4_📋_Profiling.py
   ```
10. Adicionar type hints em todas as funções e classes
11. Adicionar docstrings (Google style) em todas as funções e classes
12. Corrigir naming inconsistente: `CountPlot` → `count_plot`, `DistPlot` → `dist_plot`, `Corr` → `correlation_heatmap`, `pf_of_info` → `variable_info`, `pd_of_stats` → `descriptive_stats`
13. Substituir os if-chains longos em `plot_multivariate` por dispatch dict ou match/case
14. Mover funções `pretty()` e `map_func()` de dentro de `plot_multivariate` para nível de módulo
15. Substituir `type(col) != list` por `isinstance(col, list)` — linhas 122, 137
16. Remover backslash line continuations → usar parênteses
17. Adicionar `st.set_page_config()` como primeiro comando Streamlit (título, favicon, wide layout)
18. Implementar `st.session_state` para persistir transformações de dados entre reruns (ex.: imputação de nulos)
19. Adicionar tratamento de erros robusto: try/except em uploads, parsing, operações de dados; mensagens amigáveis via `st.error()`
20. Commit: `refactor: modular architecture with type hints and docstrings`

### Fase 3 — Novas funcionalidades

21. **Multi-page app**: Converter para Streamlit multi-page nativo (pasta `pages/`) com 4 páginas:
    - **Overview** — Upload, info básica, preview dos dados, missing values
    - **Univariate** — Análise de uma variável
    - **Multivariate** — Análise multivariada com todos os gráficos
    - **Profiling** — Relatório automático completo
22. **Profiling automático**: Ao subir o CSV, gerar um relatório estilo ydata-profiling com:
    - Distribuição de cada variável (histograma + stats)
    - Matriz de correlação
    - Detecção de outliers (IQR e Z-score)
    - Alertas automáticos (alta cardinalidade, muitos NaN, colunas constantes, alta correlação entre features)
    - Amostra dos dados
23. **Suporte a Excel e Parquet**: Aceitar `.xlsx` e `.parquet` além de `.csv` no uploader; adicionar `openpyxl` e `pyarrow` nas dependências
24. **Export de gráficos**: Botão de download em PNG/SVG para cada gráfico gerado:
    - Plotly: usar `fig.to_image()` (requer `kaleido`)
    - Seaborn/Matplotlib: usar `fig.savefig()` em buffer BytesIO
    - Exibir via `st.download_button()`
25. Substituir `get_table_download_link` (base64 hack) por `st.download_button()` nativo para export de dados
26. Adicionar componentes visuais modernos: `st.tabs()`, `st.columns()`, `st.expander()`, `st.metric()` para exibir KPIs
27. Commit: `feat: multi-page app, auto-profiling, Excel/Parquet support, chart export`

### Fase 4 — Infraestrutura e qualidade

28. Criar `pyproject.toml` com metadata do projeto, configuração do ruff, pytest, e mypy
29. Configurar **ruff** para linting + formatting (substituindo black/flake8/isort)
30. Criar `.pre-commit-config.yaml` com hooks: ruff, mypy, trailing-whitespace, check-yaml
31. Escrever testes com **pytest**:
    - `tests/test_loader.py` — testa carregamento de CSV, Excel, Parquet, arquivos inválidos
    - `tests/test_stats.py` — testa funções de estatística descritiva, quantis, correlação
    - `tests/test_preprocessing.py` — testa imputação de nulos
    - `tests/test_visualization.py` — testa que gráficos são gerados sem erros (smoke tests)
32. Criar **GitHub Actions** CI pipeline (`.github/workflows/ci.yml`):
    - Run ruff lint + format check
    - Run mypy
    - Run pytest
    - Trigger on push/PR
33. Atualizar `.gitignore` com: `__pycache__/`, `*.pyc`, `.env`, `.venv/`, `.ruff_cache/`, `.mypy_cache/`, `.pytest_cache/`, `*.egg-info/`, `.streamlit/secrets.toml`
34. Criar `.streamlit/config.toml` versionado no repo (substituindo o gerado por `setup.sh`)
35. Remover `setup.sh` e `Procfile` (desnecessários para Streamlit Cloud)
36. Commit: `chore: add pyproject.toml, CI, pre-commit, tests, ruff`

### Fase 5 — Polish e deploy

37. Reescrever o [README.md](Exploratory-Data-Analysis-App/README.md) com:
    - Badges (CI status, Python version, Streamlit)
    - Screenshots/GIFs do app
    - Instruções de setup local com `pip install -r requirements.txt`
    - Link para o app no Streamlit Cloud
    - Seção "Architecture" explicando a estrutura modular
38. Configurar deploy no **Streamlit Cloud** (conectar repo GitHub, definir entry point `src/eda_app/app.py`)
39. Rodar o app end-to-end com datasets reais (Titanic, Iris, housing) e validar todas as funcionalidades
40. Tag `v1.0.0` — release

---

### Verificação

- **Fase 0**: App roda (ou falha com erros documentados) no ambiente conda
- **Fase 1**: `streamlit run eda_main.py` funciona sem erros/warnings com dependências modernas
- **Fase 2**: `ruff check src/` passa; `mypy src/` sem erros; todas as funcionalidades existentes ainda funcionam
- **Fase 3**: Upload de CSV/Excel/Parquet funciona; profiling gera relatório; gráficos exportam; multi-page navega corretamente
- **Fase 4**: `pytest` passa; `pre-commit run --all-files` passa; GitHub Actions CI verde
- **Fase 5**: App acessível no Streamlit Cloud; README renderiza corretamente

### Decisões

- **Deploy**: Streamlit Cloud (gratuito, nativo, elimina setup.sh/Procfile)
- **Dependências**: pip + requirements.txt (simples, sem overhead de ferramenta extra)
- **Gráficos**: Manter Plotly + Seaborn/Matplotlib (aproveitar o melhor de cada)
- **Profiling**: Implementação própria (não usar ydata-profiling como dependência — mais leve e customizável, mostra habilidade de engenharia)
- **Python**: 3.11+ (melhor performance, type hints modernos, match/case disponível)
- **Linting**: ruff (substitui black + flake8 + isort em uma ferramenta só)
