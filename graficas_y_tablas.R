library(dplyr)
library(readr)

df_svm <- read_csv("reporte_svm_general_data.csv")

reporte_limpio_na <- df_svm %>%
  filter(Dataset %in% c("Points_100", "Points_1000", "Points_2000", "WDBC", "Moons_100", "Moons_1000", "Moons_2000")) %>%
  
  # Primero pasamos los -1 a NA para que no estropeen las medias de otros casos
  mutate(
    Tiempo   = ifelse(Tiempo == -1, NA, Tiempo),
    Bar_iter = ifelse(Bar_iter == -1, NA, Bar_iter)
  ) %>%
  group_by(Dataset, Modelo, Nu) %>%
  summarise(
    Media_Funcion_Objetivo = mean(Funcion_Objetivo, na.rm = TRUE),
    Media_Tiempo           = mean(Tiempo, na.rm = TRUE),
    Media_Bar_iter         = mean(Bar_iter, na.rm = TRUE),
    Media_Accuracy_Train   = mean(Accuracy_Train, na.rm = TRUE),
    Media_Accuracy_Test    = mean(Accuracy_Test, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  
  # CORRECCIÓN ESTÉTICA: Convertimos los 'NaN' (bloques que fallaron siempre) en NA limpios
  mutate(
    Media_Tiempo   = ifelse(is.nan(Media_Tiempo), NA, Media_Tiempo),
    Media_Bar_iter = ifelse(is.nan(Media_Bar_iter), NA, Media_Bar_iter)
  ) %>%
  arrange(Dataset, Modelo, Nu)

print(reporte_limpio_na)
write_csv(reporte_limpio_na, "resumen_medias_svm_na.csv")