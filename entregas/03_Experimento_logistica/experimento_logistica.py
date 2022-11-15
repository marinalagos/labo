import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

lgbm = pd.read_csv('prediccion_LGBM.csv', sep='\t')
log = pd.read_excel('prediccion_logistica.xlsx')#, sep='\t', encoding='utf-8')
log.to_csv('log.csv')
log = pd.read_csv('log.csv', sep='\t')
log['numero_de_cliente'] = log.iloc[:, 0].str.split(',', expand=True)[1].astype(int)
log = log[['prob_glm', 'numero_de_cliente']]
df = pd.merge(lgbm, log, on = 'numero_de_cliente')
df['clase_glm'] = (df['prob_glm'] > 0.5)

cols_prob = []
for col in df.columns:
    if col[:3] == 'pro':
        cols_prob.append(col)
        
cols_prob.remove('prob_glm')

# ENSAMBLES ENTRE SEMILLERÍO
# promedio de las probabilidades de todas las semillas (E0)
df['mean_prob'] = df[cols_prob].mean(axis=1)
df['mean_prob_rank'] = df.mean_prob.rank(ascending=False)

# promedio de rankings de todas las semillas (E1)
cols_rank = []
for col in df.columns:
    if col[:3] == 'ran':
        cols_rank.append(col)
df['mean_rank'] = df[cols_rank].mean(axis=1)
df['mean_rank_rank'] = df.mean_rank.rank(ascending=False, pct=True)


# MERGE CON LAS CLASES POSTA
bajas = pd.read_csv('dataset_202107.csv.gz')
bajas['baja2'] = (bajas['clase_ternaria'] == 'BAJA+2')

df = pd.merge(df, bajas, on='numero_de_cliente')

# ENSAMBLES MIX LGBM Y GLM
# toma la proba máxima (EM0)
df['max_meanprob_log'] = df[["mean_prob", "prob_glm"]].max(axis=1)
# toma la proba promedio (EM1)
df['mean_meanprob_log'] = df[["mean_prob", "prob_glm"]].mean(axis=1)
# amplifica la probabilidad de lgbm si glm lo vota (suma 0.25) (EM2)
df['suma0p25_meanprob_log'] = df["mean_prob"] + 0.25* df["prob_glm"]
df['suma0p25_meanprob_log'][df.suma0p25_meanprob_log > 1] = 1
# amplifica la probabilidad de lgbm si glm lo vota (* 1.50) (EM3)
df['mult1p5_meanprob_log'] = df["mean_prob"] + 0.50* df["prob_glm"] *df["mean_prob"] 
df['mult1p5_meanprob_log'][df.mult1p5_meanprob_log > 1] = 1


false_positive_rate_E0, true_positive_rate_E0, threshold_E0 = roc_curve(df['baja2'], df['mean_prob'])
false_positive_rate_E1, true_positive_rate_E1, threshold_E1 = roc_curve(df['baja2'], df['mean_rank_rank'])


false_positive_rate_EM0, true_positive_rate_EM0, threshold_EM0 = roc_curve(df['baja2'], df['max_meanprob_log'])
false_positive_rate_EM1, true_positive_rate_EM1, threshold_EM1 = roc_curve(df['baja2'], df['mean_meanprob_log'])
false_positive_rate_EM2, true_positive_rate_EM2, threshold_EM2 = roc_curve(df['baja2'], df['suma0p25_meanprob_log'])
false_positive_rate_EM3, true_positive_rate_EM3, threshold_EM3 = roc_curve(df['baja2'], df['mult1p5_meanprob_log'])


plt.subplots(1, figsize=(6,6))
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

#for prob in cols_prob:
#    false_positive_rate_ind, true_positive_rate_ind, threshold_ind = roc_curve(df['baja2'], df[prob])
#    plt.plot(false_positive_rate_ind, true_positive_rate_ind, color='gray', alpha=0.05)

plt.plot(false_positive_rate_E0, true_positive_rate_E0, label='E0 - promedio prob (LGBM)', color='cyan', ls=':', alpha=0.5, lw=3) #ensamble promedio lgbm
plt.plot(false_positive_rate_E1, true_positive_rate_E1, label='E1 - promedio rank (LGBM)', color='m', ls=':', alpha=0.5, lw=3) #ensamble promedio de rankings lgbm

plt.plot(false_positive_rate_EM0, true_positive_rate_EM0, label='EM0 - max(LGBM, RL)', color='g', ls='-') #'max_meanprob_log'
plt.plot(false_positive_rate_EM1, true_positive_rate_EM1, label='EM1 - mean(LGBM, RL)', color='b', ls='-') #mean_meanprob_log
plt.plot(false_positive_rate_EM2, true_positive_rate_EM2, label='EM2 - amplificación (+0.25)', color='orange', ls='-') #suma0p25_meanprob_log
plt.plot(false_positive_rate_EM3, true_positive_rate_EM3, label='EM3 - amplificación (*1.5)', color='r', ls='-') #mult1p5_meanprob_log


plt.title('Curvas ROC - Ensambles LGBM - REG LOG')
plt.plot([0, 1], ls="--", c='gray')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.ylim(0,1)
#plt.xlim(0,1)
plt.legend()
plt.show()

envios_glm = df.clase_glm.sum()
df['mean_prob_enviosglm'] = (df.mean_prob_rank <= envios_glm)

df_bajas = df[df.baja2 == True]
cantidad_detectados_glm = df_bajas.clase_glm.sum()
cantidad_detectados_mean_prob = df_bajas.mean_prob_enviosglm.sum()
cant_bajas = len(df_bajas)
df_detectados_bajas = df_bajas[['mean_prob_enviosglm', 'clase_glm']]
detectados_ambos = len(df_detectados_bajas[(df_detectados_bajas.mean_prob_enviosglm == True) & (df_detectados_bajas.clase_glm == True)])
detectados_ninguno = len(df_detectados_bajas[(df_detectados_bajas.mean_prob_enviosglm == False) & (df_detectados_bajas.clase_glm == False)])
detectados_lgbm = len(df_detectados_bajas[(df_detectados_bajas.mean_prob_enviosglm == True) & (df_detectados_bajas.clase_glm == False)])
detectados_glm = len(df_detectados_bajas[(df_detectados_bajas.mean_prob_enviosglm == False) & (df_detectados_bajas.clase_glm == True)])

print(f'AUC E0: {roc_auc_score(df.baja2, df.mean_prob)}')
print(f'AUC E1: {roc_auc_score(df.baja2, df.mean_rank_rank)}')
print(f'AUC EM0: {roc_auc_score(df.baja2, df.max_meanprob_log)}')
print(f'AUC EM1: {roc_auc_score(df.baja2, df.mean_meanprob_log)}')
print(f'AUC EM2: {roc_auc_score(df.baja2, df.suma0p25_meanprob_log)}')
print(f'AUC EM3: {roc_auc_score(df.baja2, df.mult1p5_meanprob_log)}')