import pandas as pd

meses = ['202101', '202102', '202103', '202104', '202105', '202106', '202107']

modelos = ['01_072', 
           '02_055',
           '03_069',
           '04_068',
           '05_060']

dic_meses = {}
lista_dfs = []
for mes in meses:
    df = pd.DataFrame()
    for modelo in modelos:
        df_mod = pd.read_csv(f'00_Probabilidades_por_mes/pred_{mes}/pred_{modelo}.csv', sep='\t', names=['numero_de_cliente', 'foto_mes', modelo], header=0)
        if df.empty:
            df = df_mod
        else:
            df = pd.merge(df, df_mod[['numero_de_cliente', modelo]], on='numero_de_cliente', how='outer')
    df['prob_media'] = df[modelos].mean(axis=1)
    dic_meses[mes] = df    
    df_concat = df[['numero_de_cliente', 'foto_mes', 'prob_media']]
    lista_dfs.append(df_concat)
    
df_objetivo = pd.concat(lista_dfs, axis=0)
df_objetivo.to_csv('df_probas.csv')