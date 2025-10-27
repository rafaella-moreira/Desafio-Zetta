import pandas as pd

def preencher_media_por_estado(df, coluna_valor, coluna_estado):
    """
    Preenche valores nulos de uma coluna com a média agrupada por estado.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    coluna_valor (str): Nome da coluna onde os valores nulos serão preenchidos.
    coluna_estado (str): Nome da coluna que contém os estados.

    Retorna:
    pd.DataFrame: DataFrame com valores nulos preenchidos.
    """
    estados = df[coluna_estado].unique()
    
    for estado in estados:
        media_estado = df.loc[df[coluna_estado] == estado, coluna_valor].mean()
        df.loc[df[coluna_estado] == estado, coluna_valor] = df.loc[
            df[coluna_estado] == estado, coluna_valor
        ].fillna(media_estado)
    
    return df

# limpeza.py


def transformar_e_agregar_plano(df, coluna_estado="Estado", coluna_valor="Número de beneficiários de plano de saúde"):
    """
    Transforma uma tabela trimestral em anual, calculando a média anual por estado.

    Parâmetros:
    df (pd.DataFrame): DataFrame no formato trimestral.
    coluna_estado (str): Nome da coluna que contém os estados.
    coluna_valor (str): Nome da coluna com os valores a serem agregados.

    Retorna:
    pd.DataFrame: DataFrame agregado por estado e ano.
    """
    # Transformar a tabela de wide para long
    df_reformado = df.melt(id_vars=[coluna_estado], var_name="Trimestre", value_name=coluna_valor)
    
    # Extrair o ano da coluna "Trimestre"
    df_reformado["Ano"] = df_reformado["Trimestre"].str.split().str[1].astype(int)
    
    # Agrupar por estado e ano, calculando a média
    df_anual = df_reformado.groupby([coluna_estado, "Ano"], as_index=False)[coluna_valor].mean()
    
    return df_anual

