# Análise Socioeconômica dos Estados do Sudeste do Brasil (2019–2023)

Este projeto tem como finalidade realizar uma análise dos dados socioeconômicos referentes aos estados da região Sudeste do Brasil, investigando as inter-relações entre os indicadores de renda, educação, saúde e emprego. A partir dessas variáveis, busca-se classificar os estados em diferentes níveis socioeconômicos claassificados em alto, médio e baixo, de modo a caracterizar o grau de desenvolvimento regional.
Ademais, a pesquisa visa identificar os fatores que exercem maior influência sobre a formação desse índice, possibilitando compreender os elementos determinantes das disparidades socioeconômicas e prever quais variáveis tendem a impactar de forma mais significativa o desempenho social e econômico das unidades federativas analisadas.

## Base de Dados
Os dados utilizados neste estudo foram obtidos a partir das bases públicas do IBGE, especificamente da PNADC, abrangendo o período de 2019 a 2023. As variáveis analisadas incluem renda média real, taxa de desocupação, escolaridade média, cobertura de saúde, condição de ocupação nos domicílios, entre outras.
O pré-processamento dos dados envolveu a seleção das informações relevantes, a padronização temporal anual e a remoção de valores ausentes, garantindo a consistência e a qualidade das análises subsequentes.

## Metodologia
O pipeline analítico foi desenvolvido em Python, utilizando bibliotecas como pandas, matplotlib, seaborn, scikit-learn, geopandas, entre outras. A coleta de dados foi realizada a partir de fontes públicas nacionais, seguida dos procedimentos necessários de pré-processamento para garantir a qualidade das informações.
Na fase de análise exploratória, foram empregadas matrizes de correlação para identificar as relações positivas e negativas entre as variáveis, permitindo uma compreensão inicial das interdependências do conjunto de dados.
Na etapa de modelagem estatística, foram testados dois modelos de classificação — Regressão Linear e Random Forest — com o objetivo de prever o índice socioeconômico dos estados. A avaliação de desempenho, por meio de métricas apropriadas, indicou que o modelo de Regressão Linear apresentou maior eficiência na previsão do índice.
Adicionalmente, foram utilizados diversos recursos de visualização, como gráficos de dispersão, heatmaps e histogramas, para facilitar a interpretação dos dados e a extração de insights relevantes.

## Resultados
O modelo de Regressão Linear apresentou um Erro Médio Absoluto (MAE) de 2,78%, indicando elevada precisão na previsão do índice socioeconômico. Além disso, identificou-se que a variável de maior relevância foi o rendimento médio real habitual do trabalho principal, a qual apresentou forte correlação com diversos fatores socioeconômicos, demonstrando sua importância como indicador preditivo do nível socioeconômico regional.

## Conclusão
Os resultados obtidos evidenciam que o rendimento médio real exerce influência significativa sobre os indicadores sociais, possibilitando a construção de um índice socioeconômico consistente e representativo para os estados da região Sudeste do Brasil.

## Tecnologias Utilizadas
- Python 3.14.0
- Python 3.11.9
- pandas, numpy, matplotlib, seaborn, scikit-learn, geopandas
- Jupyter Notebook  

## Como Reproduzir
1. Recomendado o uso da versão 3.11.9 do python
2. Clone o repositório: 
```bash
git clone https://github.com/rafaella-moreira/Desafio-Zetta.git
```
3. Abra a aba do notebook:
```bash
 cd Desafio-Zetta/notebooks
```
4. Execute o notebook principal:
```bash
jupyter notebook analisedados.ipynb
```
