require("data.table")

setwd( "~/buckets/b1/" )
# Cargamos el dataset
dataset <- fread("./datasets/competenciaFINAL_2022.csv.gz")

# Cargamos el dataset de probabilidades
ds_probas <- fread("./datasets/df_probas.csv", select=c("numero_de_cliente", "foto_mes", "prob_media"))
dataset <- merge(dataset, ds_probas, by = c('numero_de_cliente', 'foto_mes'), all=TRUE)

fwrite( dataset,
        "./datasets/competenciaFINAL_2022_PROBA.csv.gz",
        logical01= TRUE,
        sep= "," )