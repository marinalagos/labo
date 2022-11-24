import pandas as pd
import os

dic_prob_corte = {'m1': 0.039743906,
                  'm2': 0.039939367,
                  'm3': 0.035208223,
                  #'m4': 0.030132797, 
                  'm4': 0.1 # se modificó arbitrariamente porque el número de envíos no resultaba razonable (muy grande)
                  }

df_resumen = pd.DataFrame()
for modelo in range(1,5):
    fp = f"00_Resultados_no_sincronizados\ZZ9442_semillerio_m{modelo}_1-20"
    resultados = os.listdir(fp)
    
    for r in resultados:
        seed = r[-21:-15]
        df = pd.read_csv(fp + '/' + r)
        df['bool'] = (df.prob >= dic_prob_corte[f'm{modelo}'])
        df = df[['numero_de_cliente','prob','rank','bool']]
        df = df.rename(columns={'prob': f'prob_m{modelo}_s{seed}',
                                'rank': f'rank_m{modelo}_s{seed}',
                                'bool': f'bool_m{modelo}_s{seed}'})
    
        if df_resumen.empty:
            df_resumen = df.copy()
        else:
            df_resumen = pd.merge(df_resumen, df, on='numero_de_cliente')
 
df_resumen = df_resumen.set_index('numero_de_cliente')            
bool_cols = [c for c in df_resumen.columns if c[:4] == 'bool']
df_bools = df_resumen[bool_cols]
df_kaggle = df_bools.any(axis='columns').replace({False:0, True:1}).rename('Predicted')
df_kaggle.to_csv('any_4x20.csv')

#m1
bool_cols = [c for c in df_resumen.columns if c[:7] == 'bool_m1']
df_bools = df_resumen[bool_cols]
df_m1 = df_bools.any(axis='columns').replace({False:0, True:1})
print(f'M1: {df_m1.sum()}')
#m2
bool_cols = [c for c in df_resumen.columns if c[:7] == 'bool_m2']
df_bools = df_resumen[bool_cols]
df_m2 = df_bools.any(axis='columns').replace({False:0, True:1})
print(f'M2: {df_m2.sum()}')
#m3
bool_cols = [c for c in df_resumen.columns if c[:7] == 'bool_m3']
df_bools = df_resumen[bool_cols]
df_m3 = df_bools.any(axis='columns').replace({False:0, True:1})
print(f'M3: {df_m3.sum()}')
#m4
bool_cols = [c for c in df_resumen.columns if c[:7] == 'bool_m4']
df_bools = df_resumen[bool_cols]
df_m4 = df_bools.any(axis='columns').replace({False:0, True:1})
print(f'M4: {df_m4.sum()}')