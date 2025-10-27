import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preparar_dados_para_modelo(database, variavel_alvo, variaveis_exp, test_size=0.3, random_state=42):
    """
    Normaliza as variáveis explicativas e a variável alvo usando MinMaxScaler,
    e divide os dados em conjuntos de treino e teste.

    Parâmetros:
    database (pd.DataFrame): Base de dados original.
    variavel_alvo (str): Nome da variável dependente (target).
    variaveis_exp (list): Lista de variáveis independentes (features).
    test_size (float): Proporção do conjunto de teste (padrão: 0.3).
    random_state (int): Semente aleatória para reprodutibilidade (padrão: 42).

    Retorna:
    tuple: (X_train, X_test, y_train, y_test, df_norm)
    """
    # Copiar base para não alterar o original
    df_norm = database.copy()
    
    # Normalizar variáveis
    scaler = MinMaxScaler()
    df_norm[variaveis_exp + [variavel_alvo]] = scaler.fit_transform(df_norm[variaveis_exp + [variavel_alvo]])
    
    # Separar variáveis independentes e dependente
    X = df_norm[variaveis_exp]
    y = df_norm[variavel_alvo]
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, df_norm



def calcular_indice_socioeconomico(database, variaveis, variavel_referencia, test_size=0.3, random_state=42):
    """
    Calcula o índice socioeconômico a partir das variáveis fornecidas,
    normaliza os dados e divide em conjuntos de treino e teste.

    O índice é baseado nas correlações absolutas entre as variáveis explicativas
    e a variável de referência (geralmente o rendimento médio).

    Parâmetros:
    ----------
    database : pd.DataFrame
        Base de dados original.
    variaveis : list
        Lista com os nomes das variáveis utilizadas no cálculo.
    variavel_referencia : str
        Nome da variável usada como referência para o cálculo dos pesos (ex: rendimento médio).
    test_size : float, opcional
        Proporção do conjunto de teste (padrão: 0.3).
    random_state : int, opcional
        Semente aleatória para reprodutibilidade (padrão: 42).

    Retorna:
    -------
    tuple
        (X_train, X_test, y_train, y_test, df_norm, pesos)
    """

    # Copiar base para não alterar o DataFrame original
    df_norm = database.copy()

    # Normalizar as variáveis
    scaler = MinMaxScaler()
    df_norm[variaveis] = scaler.fit_transform(df_norm[variaveis])

    # Calcular as correlações absolutas com a variável de referência
    corrs = df_norm[variaveis].corr()[variavel_referencia].abs()

    # Calcular os pesos (cada variável recebe um peso proporcional à correlação)
    pesos = corrs / corrs.sum()

    # Calcular o índice socioeconômico ponderado
    df_norm["indice_socioeconomico"] = (df_norm[variaveis] * pesos).sum(axis=1)

    # Normalizar o índice socioeconômico entre 0 e 1
    df_norm["indice_socioeconomico"] = (
        (df_norm["indice_socioeconomico"] - df_norm["indice_socioeconomico"].min())
        / (df_norm["indice_socioeconomico"].max() - df_norm["indice_socioeconomico"].min())
    )

    # Separar variáveis independentes e dependente
    X = df_norm[variaveis]
    y = df_norm["indice_socioeconomico"]

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, df_norm, pesos
