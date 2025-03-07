# permite manejar el sampling_total y el undersampling de la clase mayoritaria

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")




# Poner sus semillas
semillas <- c(732497,
              681979,
              281887,
              936659,
              692089)

#Parametros del script
PARAM  <- list()
PARAM$experimento <- "TS9330_202107"
PARAM$experimento <- "TS9330_202106"
PARAM$experimento <- "TS9330_202105"
PARAM$experimento <- "TS9330_202104"
PARAM$experimento <- "TS9330_202103"
PARAM$experimento <- "TS9330_202102"
PARAM$experimento <- "TS9330_202101"

PARAM$exp_input  <- "FE9260"



#[201901, 202105]

#ORIGINAL
#PARAM$future       <- c( 202107 )
#PARAM$final_train  <- c( 201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103)
#PARAM$train$training     <- c( 201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103)
#PARAM$train$validation   <- c( 202104 )
#PARAM$train$testing      <- c( 202105 )

#202107
PARAM$future       <- c( 202109 )
PARAM$final_train  <- c( 201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103, 202104, 202105)
PARAM$train$training     <- c( 201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103, 202104, 202105)
PARAM$train$validation   <- c( 202106 )
PARAM$train$testing      <- c( 202107 )

#202106
PARAM$future       <- c( 202108 )
PARAM$final_train  <- c( 201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103, 202104)
PARAM$train$training     <- c( 201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103, 202104)
PARAM$train$validation   <- c( 202105 )
PARAM$train$testing      <- c( 202106 )

#202105
PARAM$future       <- c( 202107 )
PARAM$final_train  <- c( 201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103)
PARAM$train$training     <- c( 201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102, 202103)
PARAM$train$validation   <- c( 202104 )
PARAM$train$testing      <- c( 202105 )

#202104
PARAM$future       <- c( 202106 )
PARAM$final_train  <- c( 201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102)
PARAM$train$training     <- c( 201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101, 202102)
PARAM$train$validation   <- c( 202103 )
PARAM$train$testing      <- c( 202104 )

#202103
PARAM$future       <- c( 202105 )
PARAM$final_train  <- c( 201903,201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101)
PARAM$train$training     <- c( 201903,201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012, 202101)
PARAM$train$validation   <- c( 202102 )
PARAM$train$testing      <- c( 202103 )

#202102
PARAM$future       <- c( 202104 )
PARAM$final_train  <- c( 201902,201903,201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012)
PARAM$train$training     <- c( 201902,201903,201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011, 202012)
PARAM$train$validation   <- c( 202101 )
PARAM$train$testing      <- c( 202102 )

#202101
PARAM$future       <- c( 202103 )
PARAM$final_train  <- c( 201901, 201902,201903,201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011)
PARAM$train$training     <- c( 201901, 201902,201903,201904,201905,201906,201907,201908,201909,201910,201911,201912, 202011)
PARAM$train$validation   <- c( 202012 )
PARAM$train$testing      <- c( 202101 )

PARAM$train$sampling_total  <- 1  # 1.0 significa que NO se hace sampling total,  0.3 es quedarse con el 30% de TODOS los registros
PARAM$train$undersampling_mayoritaria  <- 0.4   # 1.0 significa NO undersampling ,  0.1  es quedarse con el 10% de los CONTINUA

#Atencion, las semillas deben ser distintas
PARAM$train$semilla_sampling  <- 732497
PARAM$train$semilla_under     <- 936659
# FIN Parametros del script


#------------------------------------------------------------------------------

options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa

setwd( "~/buckets/b1/" )

#Aqui se debe poner la carpeta de la computadora local
#setwd("D:/economia_finanzas")  #Establezco el Working Directory

#cargo el dataset donde voy a entrenar
#esta en la carpeta del exp_input y siempre se llama  dataset.csv.gz
dataset_input  <- paste0( "./exp/", PARAM$exp_input, "/dataset.csv.gz" )
dataset  <- fread( dataset_input )


#creo la carpeta donde va el experimento
dir.create( paste0( "./exp/", PARAM$experimento, "/"), showWarnings = FALSE )
setwd(paste0( "./exp/", PARAM$experimento, "/"))   #Establezco el Working Directory DEL EXPERIMENTO


setorder( dataset, foto_mes, numero_de_cliente )

#grabo los datos del futuro
# aqui JAMAS se hace sampling
fwrite( dataset[ foto_mes %in% PARAM$future, ],
        file= "dataset_future.csv.gz",
        logical01= TRUE,
        sep= "," )

#grabo los datos donde voy a entrenar los Final Models
# aqui  JAMAS se hace sampling
fwrite( dataset[ foto_mes %in% PARAM$final_train, ],
        file= "dataset_train_final.csv.gz",
        logical01= TRUE,
        sep= "," )



#grabo los datos donde voy a hacer la optimizacion de hiperparametros
set.seed( PARAM$train$semilla_sampling )
dataset[ foto_mes %in% PARAM$train$training , azar_sampling := runif( nrow(dataset[foto_mes %in% PARAM$train$training ]) ) ]


set.seed( PARAM$train$semilla_under )
dataset[ foto_mes %in% PARAM$train$training , azar_under := runif( nrow(dataset[foto_mes %in% PARAM$train$training ]) ) ]

dataset[  , fold_train := 0L ]
dataset[ foto_mes %in% PARAM$train$training & 
         ( azar_sampling <= PARAM$train$sampling_total ) &
         ( azar_under <= PARAM$train$undersampling_mayoritaria | clase_ternaria %in% c( "BAJA+1", "BAJA+2" ) )
         , fold_train := 1L ]

#Se valida SIN sampling de ningun tipo
dataset[  , fold_validate := 0L ]
dataset[ foto_mes %in% PARAM$train$validation, fold_validate := 1L ]

#Se testea SIN sampling de ningun tipo
dataset[  , fold_test := 0L ]
dataset[ foto_mes %in% PARAM$train$testing, fold_test := 1L ]


fwrite( dataset[ fold_train + fold_validate + fold_test >= 1 , ],
        file= "dataset_training.csv.gz",
        logical01= TRUE,
        sep= "," )
