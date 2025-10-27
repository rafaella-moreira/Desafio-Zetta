import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calcular_pesos_correlacao(database, variaveis, variavel_referencia):
    """
    Calcula os pesos das variáveis com base nas correlações absolutas
    em relação a uma variável de referência (por exemplo, o rendimento médio).

    Os pesos são normalizados para que a soma total seja igual a 1.

    Parâmetros:
    ----------
    database : pd.DataFrame
        Base de dados original.
    variaveis : list
        Lista com os nomes das variáveis numéricas que serão consideradas.
    variavel_referencia : str
        Nome da variável usada como referência (target) para o cálculo das correlações.

    Retorna:
    -------
    tuple
        (df_norm, pesos)
        - df_norm: DataFrame com variáveis normalizadas via MinMaxScaler.
        - pesos: pd.Series com os pesos normalizados (soma = 1).
    """

    # Copiar base para não alterar o DataFrame original
    df_norm = database.copy()

    # Normalizar as variáveis
    scaler = MinMaxScaler()
    df_norm[variaveis] = scaler.fit_transform(df_norm[variaveis])

    # Calcular correlações absolutas com a variável de referência
    corrs = df_norm[variaveis].corr()[variavel_referencia].abs()

    # Calcular pesos normalizados (soma = 1)
    pesos = corrs / corrs.sum()

    print("Pesos calculados com base nas correlações:")
    print(pesos)

    return df_norm, pesos

def calcular_classificacao_indice(df_norm, variaveis, pesos, coluna_estado='estado'):
    """
    Calcula o índice socioeconômico ponderado e classifica os registros em
    categorias ('baixo', 'médio', 'alto') com base em faixas do índice.

    Parâmetros:
    ----------
    df_norm : pd.DataFrame
        DataFrame já normalizado (geralmente vindo da função calcular_pesos_correlacao).
    variaveis : list
        Lista das variáveis utilizadas no cálculo do índice.
    pesos : pd.Series
        Pesos calculados com base nas correlações (deve ter o mesmo índice que as variáveis).
    coluna_estado : str, opcional
        Nome da coluna que contém o nome do estado ou unidade de análise (padrão: 'estado').

    Retorna:
    -------
    pd.DataFrame
        DataFrame contendo as colunas:
        - 'indice_socioeconomico': valor normalizado entre 0 e 1.
        - 'classificacao': categoria socioeconômica ('baixo', 'médio', 'alto').

    Exemplo:
    --------
    df_resultado = calcular_classificacao_indice(df_norm, variaveis, pesos, coluna_estado='Estado')
    """

    # Calcular o índice ponderado
    df_norm['indice_socioeconomico'] = (df_norm[variaveis] * pesos).sum(axis=1)

    # Normalizar o índice entre 0 e 1
    df_norm['indice_socioeconomico'] = (
        df_norm['indice_socioeconomico'] - df_norm['indice_socioeconomico'].min()
    ) / (
        df_norm['indice_socioeconomico'].max() - df_norm['indice_socioeconomico'].min()
    )

    # Classificar o índice em categorias (baixo, médio, alto)
    df_norm['classificacao'] = pd.cut(
        df_norm['indice_socioeconomico'],
        bins=[0, 0.33, 0.66, 1],
        labels=['baixo', 'médio', 'alto']
    )

    # Remover possíveis valores nulos resultantes
    df_norm.dropna(subset=['indice_socioeconomico', 'classificacao'], inplace=True)

    print("\nÍndice socioeconômico calculado:")
    print(df_norm[[coluna_estado, 'indice_socioeconomico', 'classificacao']]
          .sort_values(by='indice_socioeconomico', ascending=False))

    return df_norm

