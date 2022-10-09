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
dataset = dataset.drop('ctrx_quarter', axis=1)

dataset['mpayroll_sobre_edad'] = dataset.mpayroll/dataset.cliente_edad
dataset['payroll_bool'] = dataset.mpayroll > 0
dataset['master_bool'] = dataset.Master_status.isnull()
dataset['descuentos_trx'] = (dataset.ctarjeta_visa_descuentos + dataset.ctarjeta_master_descuentos)/(dataset.ctarjeta_visa_transacciones + dataset.ctarjeta_master_transacciones)
dataset['descuentos_trx'] = dataset['descuentos_trx'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

dataset['cociente_rentabilidad'] = dataset.mrentabilidad/dataset.mrentabilidad_annual
dataset['no_usa_HB'] = (dataset['thomebanking']==False | (dataset['chomebanking_transacciones'] == 0))
dataset['antiguedad_HB'] = dataset.cliente_antiguedad
dataset.loc[dataset.no_usa_HB == False, 'antiguedad_HB'] = 1000
dataset['cliente_adulto_mayor'] = (dataset.cliente_edad >= 65) & (dataset.active_quarter == True) & (dataset.no_usa_HB == True)

campos_trx = ['ctarjeta_debito_transacciones',
              'ctarjeta_visa_transacciones',
              'ctarjeta_master_transacciones',
              'cprestamos_personales', # SUPER * 90
              'cprestamos_prendarios', # SUPER * 90
              'cprestamos_hipotecarios', # SUPER * 90
              'cplazo_fijo', # SUPER * 60
              'cinversion1', # SUPER * 60
              'cinversion2', # SUPER * 60
              'cseguro_vida', # SUPER * 10
              'cseguro_auto', # SUPER * 10
              'cseguro_vivienda', # SUPER  * 10
              'cseguro_accidentes_personales', # SUPER * 10
              'ccaja_seguridad', ### HIPER 90
              'cpayroll_trx', ### HIPER 90
              'cpayroll2_trx', ### HIPER 90
              'ccuenta_debitos_automaticos', 
              'ctarjeta_visa_debitos_automaticos',
              'ctarjeta_master_debitos_automaticos',
              'cpagodeservicios',
              'cpagomiscuentas',
              'cforex',
              'cforex_buy',
              'cforex_sell',
              'ctransferencias_recibidas',
              'ctransferencias_emitidas',
              'cextraccion_autoservicio',
              'ccheques_depositados',
              'ccheques_emitidos',
              'ccallcenter_transacciones',
              'chomebanking_transacciones',
              'ccajas_transacciones',
              'ccajas_consultas',
              'ccajas_depositos',
              'ccajas_extracciones',
              'ccajas_otras',
              'catm_trx']

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

dataset = sumar_columnas_conservar_originales(dataset, ['Visa_mpagominimo', 'Master_mpagominimo'], 'mpagominimo')

dataset = sumar_columnas_conservar_originales(dataset,
                                              campos_trx,
                                              'conjunto_trx')


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
                         ['cseguro_vida', 
                          'cseguro_auto',
                          'cseguro_vivienda',
                          'cseguro_accidentes_personales'],
                          'cseguros')

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


#dataset.to_csv('dataset2da_v02.csv')

""" OTROS PARA PROBAR """

#dataset[ , mvr_Master_mlimitecompra:= Master_mlimitecompra / mv_mlimitecompra ]
#dataset[ , mvr_Visa_mlimitecompra  := Visa_mlimitecompra / mv_mlimitecompra ]
#dataset[ , mvr_msaldototal         := mv_msaldototal / mv_mlimitecompra ]
dataset['mvr_msaldototal'] = dataset.msaldototal/dataset.mlimitecompra
#dataset[ , mvr_msaldopesos         := mv_msaldopesos / mv_mlimitecompra ]
dataset['mvr_msaldopesos'] = dataset.msaldopesos/dataset.mlimitecompra
#dataset[ , mvr_msaldopesos2        := mv_msaldopesos / mv_msaldototal ]
dataset['mvr_msaldopesos2'] = dataset.msaldopesos/dataset.msaldototal
#dataset[ , mvr_msaldodolares       := mv_msaldodolares / mv_mlimitecompra ]
dataset['mvr_msaldolares'] = dataset.msaldodolares/dataset.mlimitecompra
#dataset[ , mvr_msaldodolares2      := mv_msaldodolares / mv_msaldototal ]
dataset['mvr_msaldolares2'] = dataset.msaldodolares/dataset.msaldototal
#dataset[ , mvr_mconsumospesos      := mv_mconsumospesos / mv_mlimitecompra ]
dataset['mvr_mconsumospesos'] = dataset.mconsumospesos/dataset.mlimitecompra
#dataset[ , mvr_mconsumosdolares    := mv_mconsumosdolares / mv_mlimitecompra ]
dataset['mvr_mconsumosdolares'] = dataset.mconsumosdolares/dataset.mlimitecompra
#dataset[ , mvr_madelantopesos      := mv_madelantopesos / mv_mlimitecompra ]
dataset['mvr_madelantopesos'] = dataset.madelantopesos/dataset.mlimitecompra
#dataset[ , mvr_madelantodolares    := mv_madelantodolares / mv_mlimitecompra ]
dataset['mvr_madelantodolares'] = dataset.madelantodolares/dataset.mlimitecompra
#dataset[ , mvr_mpagado             := mv_mpagado / mv_mlimitecompra ]
dataset['mvr_mpagado'] = dataset.mpagado/dataset.mlimitecompra
#dataset[ , mvr_mpagospesos         := mv_mpagospesos / mv_mlimitecompra ]
dataset['mvr_mpagospesos'] = dataset.mpagospesos/dataset.mlimitecompra
#dataset[ , mvr_mpagosdolares       := mv_mpagosdolares / mv_mlimitecompra ]
dataset['mvr_mpagosdolares'] = dataset.mpagosdolares/dataset.mlimitecompra
#dataset[ , mvr_mconsumototal       := mv_mconsumototal  / mv_mlimitecompra ]
dataset['mvr_mconsumototal'] = dataset.mconsumototal/dataset.mlimitecompra
#dataset[ , mvr_mpagominimo         := mv_mpagominimo  / mv_mlimitecompra ]
dataset['mvr_mpagominimo'] = (dataset.mpagominimo)/dataset.mlimitecompra


#dataset = dataset[(dataset.foto_mes == 202103) | (dataset.foto_mes == 202105)]
dataset.to_csv('dataset2da_v05.csv')
