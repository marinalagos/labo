# Limpiamos el entorno
rm(list = ls())
gc(verbose = FALSE)

#Arbol elemental con libreria  rpart
#Debe tener instaladas las librerias  data.table  ,  rpart  y  rpart.plot

#cargo las librerias que necesito
require("data.table")
require("rpart")
require("rpart.plot")

#Aqui se debe poner la carpeta de la materia de SU computadora local
setwd("C:/Users/marin/Maestria/2022_2C/00_DMEyF/") #setwd("D:\\gdrive\\UBA2022\\")  #Establezco el Working Directory

#cargo el dataset
dataset  <- fread("./datasets/dataset_v00.csv")

# Clases BAJAS+1 y BAJA+2 combinadas
dataset[, clase_binaria := ifelse(
  clase_ternaria == "CONTINUA",
  "noevento",
  "evento"
)]
dataset[,clase_ternaria := NULL]

dtrain  <- dataset[ foto_mes==202101 ]  #defino donde voy a entrenar
dapply  <- dataset[ foto_mes==202103 ]  #defino donde voy a aplicar el modelo


#genero el modelo,  aqui se construye el arbol
modelo  <- rpart(formula=   "clase_binaria ~ .",  #quiero predecir clase_ternaria a partir de el resto de las variables
                 data=      dtrain,  #los datos donde voy a entrenar
                 xval=      0,
                 cp=       -1,   
                 minsplit=  188,  
                 minbucket= 152,  
                 maxdepth=  7 )   


#grafico el arbol
prp(modelo, extra=101, digits=5, branch=1, type=4, varlen=0, faclen=0)


#aplico el modelo a los datos nuevos
prediccion  <- predict( object= modelo,
                        newdata= dapply,
                        type = "prob")

#prediccion es una matriz con TRES columnas, llamadas "BAJA+1", "BAJA+2"  y "CONTINUA"
#cada columna es el vector de probabilidades 

#agrego a dapply una columna nueva que es la probabilidad de BAJA+2
dapply[ , prob_evento := prediccion[, "evento"] ]

#solo le envio estimulo a los registros con probabilidad de BAJA+2 mayor  a  1/40
dapply[ , Predicted := as.numeric( prob_evento > 0.0601 ) ]

#genero el archivo para Kaggle
#primero creo la carpeta donde va el experimento
dir.create( "./exp/" )
dir.create( "./exp/KA2001" )

fwrite( dapply[ , list(numero_de_cliente, Predicted) ], #solo los campos para Kaggle
        file= "./exp/KA2001/K101_009.csv",
        sep=  "," )
