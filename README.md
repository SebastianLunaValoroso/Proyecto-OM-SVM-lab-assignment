# Proyecto OM SVM lab assignment
Autores: Unai Lema y Sebastián Luna

## Entrega
2 archivos separados:
- un pdf con el reporte (Máximo de 10 páginas)
- un zip con el resto de archivos (códigos, datasets, etc)

## Contenido del Report
- Una primera página indicando el nombre de los dos autores de la práctica.
- Los códigos de AMPL de la formulación del dual y primal
- Para cada Dataset (generado con el generador y dataset real):
  - Los resultados obtenidos en la optimización (indicar que solver hemos usado, tiempo de cálculo, iteraciones, etc)
  - Comprobación  que a partir de la solución dual recuperamos el plano del primal
  - Accuracy que obtenemos (usando otro conjunto de datos de test)
  - Si hemos probado diversos valores de nu, o datasets de tamaños diferentes, dar también:
    -  Una tabla de resultados según diferentes valores de nu o tamaño del dataset
    - Cualquier otro comentario que queramos añadir
- Para el dataset no separable:
  - Indicar cómo lo hemos obtenido
  - Resultados de optimización usando el Kernel Gaussiano
  - Comprobar si funciona mejor que sin usar ningún Kernel
  - Comprobar la accuracy usando un dataset de test

## Instrucciones
- Implementar las formulaciones cuadráticas del primal y dual del Support Vector Machine Classifier en AMPL.
- Aplicar nuestra implementación del SVM al dataset obtenido con el generador `gensvm` (de linux, windows o mac). Validar la accuracy del SVM con datos diferentes de los de train.
- Aplicar nuestra implementación del SVM a otro dataset de algún repositorio de internet. Validar la accuracy.
- Para todos los dataset probados, calcular el hiperplano separador del modelo SVM del dual y comprobar que coincide con el del  modelo SVM del primal.
- Encontrar un dataset NO SEPARABLE linealmente y clasificarlo con nuetra implementación del dual SVM con kernel RBF o Gaussiano (se pueden generar un dataset no separable linealmente usando la función `sklearn.datasets.make_swiss_roll()` del paquete de python `sklearn`)
- Comprueba la accuracy del SVM resultante

## Generar datos con gensvmdat (correspondiente ejecutable de linux, mac o windows)

Comando:

$ ./gensvmdat points_x.dat x seed

Dónde:
- `x` es el número de puntos
- `seed` es la semilla. Utilizaremos 77214914

El archivo `points_x.dat` generado contiene unos * en los vectores que se clasificarán incorrectamente, se tiene que quitar este *.