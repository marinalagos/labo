##
## Sobre como cocinar Pizzas
##
## ---------------------------
## Step 1: Cargando los datos y las librerías
## ---------------------------
##
## Success is a lousy teacher. It seduces smart people into thinking they can't 
## lose.
## --- Bill Gates

# Limpiamos el entorno
rm(list = ls())
gc(verbose = FALSE)

# Librerías necesarias
require("data.table")
require("rpart")
require("ROCR")
require("ggplot2")
require("lubridate")
require("lhs")
require("DiceKriging")
require("mlrMBO")

# Poner la carpeta de la materia de SU computadora local
setwd("C:/Users/marin/Maestria/2022_2C/00_DMEyF")
# Poner sus semillas
semillas <- c(838609,
              882019,
              124987,
              348431,
              819503)

# Cargamos el dataset
dataset <- fread("./datasets/competencia1_2022.csv")

# Nos quedamos solo con el 202101
dataset <- dataset[foto_mes == 202101]
# Creamos una clase binaria
dataset[, clase_binaria := ifelse(
                            clase_ternaria == "BAJA+2",
                                "evento",
                                "noevento"
                            )]
# Borramos el target viejo
dataset[, clase_ternaria := NULL]

# Seteamos nuestra primera semilla
set.seed(semillas[1])

## ---------------------------
## RESOLUCIÓN TAREA
## ---------------------------

ganancia <- function(probabilidades, clase) {
  return(sum(
    (probabilidades >= 0.025) * ifelse(clase == "evento", 78000, -2000))
  )
}

# FUNCIÓN modelo_rpart cambiando AUC por ganancia
modelo_rpart <- function(train, test, cp =  0, ms = 20, mb = 1, md = 10) {
  modelo <- rpart(clase_binaria ~ ., data = train,
                  xval = 0,
                  cp = cp,
                  minsplit = ms,
                  minbucket = mb,
                  maxdepth = md)
  
  test_prediccion <- predict(modelo, test, type = "prob")
  g = ganancia(test_prediccion[, "evento"], test$clase_binaria) / 0.3
  unlist(g)
}


# FUNCIÓN experimento_rpart -> cambios: no muestra (OK), ganacia en lugar de auc (OK)

# Una función auxiliar para los experimentos

experimento_rpart <- function(ds, semillas, cp = -1, ms = 20, mb = 1, md = 10) { #AGREGAR PARÁMETROS, CP=-1
  ganancias <- c()
  for (s in semillas) {
    set.seed(s)
    in_training <- caret::createDataPartition(ds$clase_binaria, p = 0.70,
                                              list = FALSE)
    train  <-  ds[in_training, ]
    test   <-  ds[-in_training, ]
    #train_sample <- tomar_muestra(train) # SIN MUESTREO
    g <- modelo_rpart(train, test, 
                      cp = cp, ms = ms, mb = mb, md = md)
    ganancias <- c(ganancias, g) #CAMBIAR POR LA GANACIA
  }
  mean(ganancias)
}

# MAIN

set.seed(semillas[1])
obj_fun_md_ms <- function(x) {
  experimento_rpart(dataset, semillas
                    , cp = -1
                    , md = x$maxdepth
                    , ms = x$minsplit
                    , mb = floor(x$minbucket * x$minsplit))
}

obj_fun <- makeSingleObjectiveFunction(
  minimize = FALSE,
  fn = obj_fun_md_ms,
  par.set = makeParamSet(
    makeIntegerParam("maxdepth",  lower = 4L, upper = 30L),
    makeIntegerParam("minsplit",  lower = 1L, upper = 300L),
    makeNumericParam("minbucket", lower=0, upper=1)
    # makeNumericParam <- para parámetros continuos
  ),
  # noisy = TRUE,
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 150L)
ctrl <- setMBOControlInfill(
  ctrl,
  crit = makeMBOInfillCritEI(),
  opt = "focussearch",
  # sacar parámetro opt.focussearch.points en próximas ejecuciones
  #opt.focussearch.points = 40 #20 antes!!
)

lrn <- makeMBOLearner(ctrl, obj_fun)

surr_km <- makeLearner("regr.km", predict.type = "se", covtype = "matern3_2")

run_md_ms <- mbo(obj_fun, learner = surr_km, control = ctrl, )
print(run_md_ms)


