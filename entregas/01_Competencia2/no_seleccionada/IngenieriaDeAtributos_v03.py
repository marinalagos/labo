# -*- coding: utf-8 -*-

"""
Created on Sun Sep 11 14:19:05 2022

@author: marin
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('C:/Users/marin/Maestria/2022_2C/00_DMEyF/datasets/competencia2_2022.csv.gz')


dataset['atm_other'] = dataset.catm_trx_other > (dataset.catm_trx)
dataset['mes_periodo_anual'] = dataset.cliente_antiguedad%12
dataset['mes_periodo_anual'] = dataset.mes_periodo_anual.map({0:2,
                                                              1:3,
                                                              2:4,
                                                              3:5,
                                                              4:6,
                                                              5:7,
                                                              6:8,
                                                              7:9,
                                                              8:10,
                                                              9:11,
                                                              10:0,
                                                              11:1})


#creo un ctr_quarter que tenga en cuenta cuando los clientes hace 3 menos meses que estan
dataset['ctrx_quarter_normalizado'] = dataset['ctrx_quarter'].astype(int)
dataset.loc[dataset.cliente_antiguedad == 1, 'ctrx_quarter_normalizado'] = (dataset[dataset.cliente_antiguedad == 1]['ctrx_quarter']*5).astype(int)
dataset.loc[dataset.cliente_antiguedad == 2, 'ctrx_quarter_normalizado'] = (dataset[dataset.cliente_antiguedad == 2]['ctrx_quarter']*2).astype(int)
dataset.loc[dataset.cliente_antiguedad == 3, 'ctrx_quarter_normalizado'] = (dataset[dataset.cliente_antiguedad == 3]['ctrx_quarter']*1.2).astype(int)
dataset = dataset.drop(['ctrx_quarter'], axis=1)

#DATA DRIFTING
dataset.loc[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre==1), 'Visa_fultimo_cierre'] = 4
dataset.loc[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre==7), 'Visa_fultimo_cierre'] = 11
dataset.loc[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre==21), 'Visa_fultimo_cierre'] = 25
dataset.loc[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre==14), 'Visa_fultimo_cierre'] = 18
dataset.loc[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre==28), 'Visa_fultimo_cierre'] = 32
dataset.loc[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre==35), 'Visa_fultimo_cierre'] = 39
dataset.loc[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre>39), 'Visa_fultimo_cierre'] = dataset[(dataset.foto_mes==202105) & (dataset.Visa_fultimo_cierre>39)]['Visa_fultimo_cierre'] + 4

dataset.loc[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre==1), 'Master_fultimo_cierre'] = 4
dataset.loc[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre==7), 'Master_fultimo_cierre'] = 11
dataset.loc[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre==21), 'Master_fultimo_cierre'] = 25
dataset.loc[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre==14), 'Master_fultimo_cierre'] = 18
dataset.loc[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre==28), 'Master_fultimo_cierre'] = 32
dataset.loc[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre==35), 'Master_fultimo_cierre'] = 39
dataset.loc[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre>39), 'Master_fultimo_cierre'] = dataset[(dataset.foto_mes==202105) & (dataset.Master_fultimo_cierre>39)]['Master_fultimo_cierre'] + 4

# corrijo manualmente el drifting de  Visa_fultimo_cierre
# dapply[ Visa_fultimo_cierre== 1, Visa_fultimo_cierre :=  4 ]
# dapply[ Visa_fultimo_cierre== 7, Visa_fultimo_cierre := 11 ]
# dapply[ Visa_fultimo_cierre==21, Visa_fultimo_cierre := 25 ]
# dapply[ Visa_fultimo_cierre==14, Visa_fultimo_cierre := 18 ]
# dapply[ Visa_fultimo_cierre==28, Visa_fultimo_cierre := 32 ]
# dapply[ Visa_fultimo_cierre==35, Visa_fultimo_cierre := 39 ]
# dapply[ Visa_fultimo_cierre> 39, Visa_fultimo_cierre := Visa_fultimo_cierre + 4 ]

# corrijo manualmente el drifting de  Visa_fultimo_cierre
# dapply[ Master_fultimo_cierre== 1, Master_fultimo_cierre :=  4 ]
# dapply[ Master_fultimo_cierre== 7, Master_fultimo_cierre := 11 ]
# dapply[ Master_fultimo_cierre==21, Master_fultimo_cierre := 25 ]
# dapply[ Master_fultimo_cierre==14, Master_fultimo_cierre := 18 ]
# dapply[ Master_fultimo_cierre==28, Master_fultimo_cierre := 32 ]
# dapply[ Master_fultimo_cierre==35, Master_fultimo_cierre := 39 ]
# dapply[ Master_fultimo_cierre> 39, Master_fultimo_cierre := Master_fultimo_cierre + 4 ]



atributos_suma = ['msaldototal', 'msaldopesos', 'msaldodolares', 'mconsumopesos']

def sumar_columnas_manterner_nans(ds, columnas_a_sumar, columna_suma):
    # guardar una columna que diga si toda la row es na
    ds['bool_aux'] = True
    ds[columna_suma] = 0
    for col in columnas_a_sumar:
        ds.bool_aux = (ds[col].isna()) & (ds.bool_aux)
        ds[col] = ds[col].fillna(0)    
        ds[columna_suma] = ds[columna_suma] + ds[col]
    ds.at[ds.bool_aux, columna_suma] = np.nan
    ds = ds.drop(columnas_a_sumar, axis=1)
    ds = ds.drop('bool_aux', axis=1)
    return ds

def sumar_columnas(ds, columnas_a_sumar, columna_suma):
    # guardar una columna que diga si toda la row es na
    ds[columna_suma] = 0
    for col in columnas_a_sumar:
        ds[col] = ds[col].fillna(0)    
        ds[columna_suma] = ds[columna_suma] + ds[col]
    ds = ds.drop(columnas_a_sumar, axis=1)
    return ds

def sumar_columnas_conservar_originales(ds, columnas_a_sumar, columna_suma):
    # guardar una columna que diga si toda la row es na
    ds[columna_suma] = 0
    for col in columnas_a_sumar:
        ds[col] = ds[col].fillna(0)    
        ds[columna_suma] = ds[columna_suma] + ds[col]
    return ds

def sumar_columnas_status(ds, columnas_a_sumar, columna_suma):
    # guardar una columna que diga si toda la row es na
    ds[columna_suma] = 0
    for col in columnas_a_sumar:
        ds[col] = ds[col].fillna(20)    
        ds[columna_suma] = ds[columna_suma] + ds[col]
    ds = ds.drop(columnas_a_sumar, axis=1)
    return ds

dataset['mlimite_compra_max'] = dataset[['Visa_mlimitecompra', 'Master_mlimitecompra']].max(axis=1)
dataset['mfinanciacion_limite_max'] = dataset[['Visa_mfinanciacion_limite', 'Master_mfinanciacion_limite']].max(axis=1)

atributos_suma_TC = ['msaldototal', 'msaldopesos', 'msaldodolares',
                     'mconsumospesos', 'mconsumosdolares', 
                     'madelantopesos', 'madelantodolares',
                     'mpagado', 'mpagospesos', 'mpagosdolares',
                     'mconsumototal', 'cconsumos', 'cadelantosefectivo',
                     'mlimitecompra', 'mfinanciacion_limite']

for at in atributos_suma_TC:
    columnas_a_sumar = ['Visa_' + at, 'Master_' + at]
    dataset = sumar_columnas(dataset, columnas_a_sumar, at)
    
dataset['delinquency'] = (dataset.Master_delinquency.fillna(0) + dataset.Visa_delinquency.fillna(0)) > 0
dataset = dataset.drop(['Master_delinquency', 'Visa_delinquency'], axis=1)


dataset = sumar_columnas_status(dataset, ['Master_status', 'Visa_status'], 'status')

""" OTROS """

dataset = sumar_columnas(dataset, 
                         ['cseguro_vida', 
                          'cseguro_auto',
                          'cseguro_vivienda',
                          'cseguro_accidentes_personales'],
                          'cseguros')

dataset = sumar_columnas_conservar_originales(dataset, 
                                             ['cprestamos_personales', 
                                              'cprestamos_prendarios',
                                              'cprestamos_hipotecarios'],
                                              'cprestamos')

dataset = sumar_columnas_conservar_originales(dataset, 
                                             ['mprestamos_personales', 
                                              'mprestamos_prendarios',
                                              'mprestamos_hipotecarios'],
                                              'mprestamos')

dataset = sumar_columnas(dataset, 
                         ['ccuenta_debitos_automaticos', 
                          'ctarjeta_visa_debitos_automaticos',
                          'ctarjeta_master_debitos_automaticos',
                          'cpagodeservicios',
                          'cpagomiscuentas'],
                          'cdebitos_automaticos')

dataset = sumar_columnas(dataset, 
                         ['ccajeros_propios_descuentos',
                          'ctarjeta_visa_descuentos',
                          'ctarjeta_master_descuentos'],
                          'cdescuentos')

dataset = sumar_columnas(dataset, 
                         ['mcajeros_propios_descuentos',
                          'mtarjeta_visa_descuentos',
                          'mtarjeta_master_descuentos'],
                          'mdescuentos')

dataset = sumar_columnas(dataset, 
                         ['mcomisiones_mantenimiento',
                          'mcomisiones_otras'],
                          'mcomisiones')

dataset.to_csv('dataset2da_v03.csv')