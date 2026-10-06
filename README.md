# Laboratorio de análisis de datos

Aplicación de análisis de datos desarrollada en Python. La interfaz principal utiliza **Streamlit** y permite cargar archivos CSV, explorar los datos, ejecutar análisis no supervisados y entrenar modelos de clasificación. La lógica de análisis se organiza en módulos reutilizables; los notebooks que quedan en el repositorio son material complementario y no son necesarios para ejecutar la aplicación.

## Requisitos

- Python instalado.
- Las dependencias indicadas en `requirements.txt`.

## Instalación y ejecución

Desde la carpeta raíz del proyecto, cree y active un entorno virtual e instale las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Inicie la aplicación con:

```powershell
python -m streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador. Para detenerla, presione `Ctrl+C` en la terminal.

## Uso de la aplicación

1. Ejecute `streamlit run app.py` desde la raíz del proyecto.
2. En la barra lateral, cargue un archivo CSV. El mismo conjunto de datos queda disponible en las páginas de la aplicación durante la sesión.
3. Seleccione una página desde la navegación:
   - **No supervisado**: vista previa, estadísticas descriptivas, conteo de nulos, correlación e histogramas de columnas numéricas; reducción dimensional con PCA, t-SNE o UMAP; agrupamiento con K-Means o HAC y método del codo.
   - **Supervisado - Clasificación**: seleccione la variable objetivo, la proporción de prueba y el algoritmo; entrene y consulte métricas, matriz de confusión e informe por clase.
   - **Supervisado - Predicción**: página reservada para regresión; el módulo todavía no está implementado.
4. La aplicación muestra los controles pertinentes para los parámetros del método elegido.

El archivo `data/drug200.csv` se incluye como conjunto de ejemplo. También puede cargar otro CSV desde la interfaz.

### Funciones disponibles en la interfaz Streamlit

En la página **No supervisado** se ofrecen:

- Exploración de las primeras filas y estadísticas descriptivas.
- Conteo de valores nulos y análisis visual de correlación y distribución para variables numéricas.
- PCA en dos o tres dimensiones, con varianza explicada y visualizaciones.
- t-SNE y UMAP en dos o tres dimensiones, con parámetros configurables.
- K-Means y HAC, selección del número de clústeres, método del codo y visualización de los grupos proyectados mediante PCA.

En **Supervisado - Clasificación** se ofrecen KNN, árbol de decisión, Random Forest, Gradient Boosting, AdaBoost y XGBoost. Se presentan *accuracy*, precisión macro, *recall* macro, F1 macro, matriz de confusión e informe por clase.

> La interfaz Streamlit no expone todos los métodos de manipulación y análisis del modelo MVC descritos más abajo. Esas operaciones siguen disponibles a través de la API Python de `DataModel`; las páginas de la aplicación muestran el subconjunto implementado en cada flujo.

### Pruebas

Desde la raíz del proyecto, ejecute:

```powershell
python -m pytest
```

Los notebooks de `notebooks/` son experimentos y material heredado de desarrollo; no son el punto de entrada de la aplicación Streamlit.

## Operaciones disponibles en el modelo MVC

Las siguientes tablas describen operaciones expuestas por el modelo/API. No todas tienen un control equivalente en la interfaz Streamlit actual.

### Consulta y estructura

| Procedimiento | Significado |
| --- | --- |
| Información | Resume la cantidad de filas, columnas y sus nombres. |
| Mostrar | Presenta el conjunto de datos cargado. |
| Primeras filas | Muestra los primeros registros para una inspección rápida. |
| Últimas filas | Muestra los registros finales. |
| Obtener fila | Recupera una fila según su posición numérica. |
| Obtener columna | Recupera una variable por su nombre. |
| Dimensiones | Devuelve la cantidad de filas y columnas $(n, p)$. |
| Nombres columnas | Lista las variables disponibles. |
| Tipos de datos | Indica si cada variable es numérica, texto, fecha, etc. |

### Limpieza y transformación

| Procedimiento | Significado |
| --- | --- |
| Contar nulos | Cuenta los valores ausentes por columna. |
| Eliminar nulos | Quita las filas que contienen al menos un valor ausente. |
| Reemplazar nulos | Sustituye los valores ausentes por el valor indicado. |
| Imputar nulos | Completa nulos con la mediana en variables numéricas y la moda en categóricas. |
| Detectar duplicados | Identifica registros repetidos. |
| Eliminar duplicados | Conserva una sola copia de cada registro repetido. |
| Convertir tipos | Cambia el tipo de una variable; en la interfaz actual convierte la columna elegida a `float64`. |
| Normalizar | Escala cada variable numérica al intervalo $[0, 1]$. |
| Estandarizar | Transforma variables numéricas para que tengan media $0$ y desviación estándar $1$. |
| Agregar columna | Crea una variable nueva; en la interfaz se agrega con valor constante `1`. |
| Eliminar columna | Borra una variable del conjunto de datos. |
| Seleccionar variables | Devuelve un subconjunto con las columnas seleccionadas. |

### Manipulación de registros

| Procedimiento | Significado |
| --- | --- |
| Ordenar | Reordena los registros ascendentemente según la columna elegida. |
| Filtrar | Conserva filas que cumplen una comparación: `==`, `!=`, `>`, `>=`, `<` o `<=`. |

### EDA y visualización

| Procedimiento | Significado |
| --- | --- |
| Estadísticas | Calcula conteos, media, desviación, cuartiles, mínimos y máximos; para categorías muestra frecuencia y moda. |
| Correlación | Calcula la relación lineal entre pares de variables numéricas, con valores entre $-1$ y $1$. |
| Visualizar datos | Muestra primeras filas, dimensiones y tipos de datos en una sola salida. |
| Frecuencias | Cuenta cuántas veces aparece cada valor en cada columna. |
| Distribución variables | Separa el resumen descriptivo de variables numéricas y categóricas. |
| Histogramas | Muestran la distribución de frecuencias de variables numéricas. |
| Boxplots | Resumen gráfico con mediana, cuartiles y posibles valores atípicos. |
| Scatterplots | Matriz de dispersión para observar relaciones entre pares de variables numéricas. |
| Mapa de calor | Representa visualmente la matriz de correlación. |
| Detectar outliers | Detecta valores atípicos con el rango intercuartílico: valores fuera de $[Q_1 - 1.5IQR, Q_3 + 1.5IQR]$. |

### Persistencia

| Procedimiento | Significado |
| --- | --- |
| Guardar CSV | Guarda el estado actual de los datos en `dataframe_guardado.csv`. |
| Exportar resultados | Exporta el estado resultante del EDA en `resultados_eda.csv`. |
| Restaurar datos | Recupera la copia original del último CSV cargado y descarta transformaciones hechas en memoria. |

### Reducción de dimensionalidad y clustering

Antes de estos métodos, la aplicación codifica variables categóricas mediante *one-hot encoding* y estandariza los atributos. Esto evita que la escala de una variable domine los resultados.

| Procedimiento | Significado |
| --- | --- |
| PCA | Análisis de Componentes Principales (ACP): crea componentes ortogonales que resumen la mayor cantidad posible de variabilidad. La API permite elegir el número de componentes. |
| PCA (variación) | Ejecuta PCA con `whiten=True` y el solucionador aleatorizado; sirve para comparar configuraciones. |
| HAC | Agrupamiento Jerárquico Aglomerativo: inicia con un grupo por observación y une los más similares hasta obtener el número de clústeres elegido. El enlace puede ser `ward`, `average`, `complete` o `single`. |
| HAC Dendrograma | Árbol que muestra el orden y la distancia a la que se fusionan grupos en HAC. |
| K-Means | Agrupa observaciones alrededor de $k$ centroides, minimizando la distancia interna de cada grupo. |
| K-Means Codo | Grafica la inercia para varios valores de $k$; el punto donde la mejora empieza a disminuir orienta la elección de clústeres. |
| t-SNE | Proyección no lineal que prioriza conservar vecindades locales; se utiliza principalmente para visualizar datos de muchas dimensiones. Su parámetro clave es la *perplexity*. |
| UMAP | Proyección no lineal que preserva estructura local y parte de la global. `n_neighbors` controla el tamaño del vecindario y `min_dist` la separación mínima en la proyección. |

Para HAC y K-Means se reporta el **silhouette score**, donde valores cercanos a $1$ sugieren grupos más separados. K-Means además reporta la **inercia**, suma de distancias cuadradas de cada observación a su centroide. En la interfaz Streamlit, la opción de método del codo está disponible en la página **No supervisado**.

## Aprendizaje supervisado: clasificación

En el aprendizaje supervisado, cada observación incluye atributos de entrada $X$ y una respuesta conocida $y$. El algoritmo aprende una función que relaciona ambos y luego la utiliza para predecir respuestas en observaciones nuevas. En clasificación, $y$ representa una categoría, como una clase binaria o una de varias clases.

Para estimar la capacidad de generalización, los datos se dividen en entrenamiento y prueba. El modelo aprende con el conjunto de entrenamiento y se evalúa con el conjunto de prueba, que se reserva para medir su desempeño. En clasificación, la división estratificada procura conservar la proporción de las clases en ambos conjuntos.

### Algoritmos

**K vecinos más cercanos (KNN).** Clasifica una observación según las clases de sus $k$ vecinos más cercanos. La distancia, comúnmente euclidiana, determina qué observaciones se consideran cercanas; por eso, la escala de las variables numéricas puede cambiar el resultado. Un $k$ pequeño puede ser sensible al ruido, mientras que uno grande puede suavizar demasiado las diferencias entre clases.

**Árbol de decisión (DT).** Divide repetidamente los datos mediante reglas sobre los atributos. En cada división busca separar las clases reduciendo una medida de impureza, como Gini o entropía. Es fácil de interpretar, pero un árbol demasiado profundo puede ajustarse demasiado a los datos de entrenamiento.

**Random Forest (RF).** Entrena varios árboles sobre muestras aleatorias del conjunto de entrenamiento y subconjuntos de atributos. Para clasificar, combina sus votos. Esta combinación suele reducir la variabilidad de un árbol individual, aunque hace menos directa la interpretación del resultado.

**AdaBoost.** Entrena una secuencia de clasificadores débiles. En cada iteración aumenta la atención sobre las observaciones que los clasificadores anteriores confundieron, y al final combina sus predicciones con pesos. Puede ser sensible a observaciones atípicas o etiquetas incorrectas.

**XGBoost.** Implementa árboles de *gradient boosting*: agrega árboles secuencialmente para corregir errores de la combinación actual. Optimiza una función objetivo que incorpora tanto la pérdida de predicción como regularización, con el fin de controlar la complejidad del modelo. Es distinto de `GradientBoostingClassifier` de scikit-learn, que también está disponible en este paquete como método adicional.

### Flujo de clasificación implementado

La clase `ClasificacionModelos` hereda de `SupervisadoBase`. Se construye con un DataFrame y el nombre de la columna objetivo. El parámetro `test_size` controla la proporción reservada para prueba; por defecto es $0.25$, y la división es reproducible mediante `random_state`. La interfaz Streamlit permite configurar `test_size` y seleccionar la variable objetivo.

La preparación de predictores forma parte de un `Pipeline` para aprender las transformaciones únicamente con los datos de entrenamiento. Las columnas numéricas reciben imputación por mediana y, si se solicita, estandarización. Las categóricas reciben imputación por moda y codificación *one-hot*. Las columnas predictoras pueden limitarse con `feature_columns`; los registros sin respuesta objetivo se rechazan.

Los métodos disponibles son `knn`, `decision_tree`, `random_forest`, `gradient_boosting`, `adaboost` y `xgboost`. Cada método permite configurar algunos hiperparámetros y devuelve el pipeline entrenado junto con sus métricas. XGBoost codifica internamente las etiquetas de texto y las devuelve en su forma original al predecir.

Las métricas de clasificación incluyen *accuracy*, precisión, *recall* y F1 macro, matriz de confusión y reporte por clase. El promedio macro otorga el mismo peso a cada clase, lo que resulta útil cuando las clases tienen tamaños distintos.

### Experimentación de clasificación

`comparar_basico()` ejecuta una configuración predeterminada de cada algoritmo principal. También se pueden cambiar los hiperparámetros al llamar cada método; por ejemplo, `knn(n_neighbors=3)` o `random_forest(n_estimators=100, max_depth=8)`. `adaboost_grid()` prueba combinaciones de parámetros mediante validación cruzada sobre los datos de entrenamiento y evalúa la mejor configuración en el conjunto de prueba.

Para comparar configuraciones, conviene mantener la misma división de datos y observar las métricas en conjunto, especialmente F1 macro y la matriz de confusión. La configuración con mejor resultado debe seleccionarse usando validación cruzada y evaluarse al final con el conjunto de prueba reservado. La clase ofrece búsqueda automatizada de hiperparámetros para AdaBoost; para los demás algoritmos, las variantes se ejecutan pasando sus parámetros a los métodos y sus resultados se comparan. La selección final y el análisis deben considerar también los errores por clase, no solo una métrica global.

La API puede utilizarse desde Python:

```python
from supervised import ClasificacionModelos

clasificador = ClasificacionModelos(
	df,
	target="columna_objetivo",
	test_size=0.25,
	random_state=42,
)
resultado = clasificador.knn(n_neighbors=5)
print(resultado["metricas"])
```

La página de predicción está reservada para desarrollar regresión más adelante; actualmente no hay algoritmos de regresión implementados.

### Métricas de los métodos no supervisados

Los métodos no supervisados devuelven métricas para ayudar a interpretar sus resultados.

| Técnica | Mensaje en consola | Interpretación simple |
| --- | --- | --- |
| PCA | `PCA - Varianza explicada acumulada: X%` | Indica cuánta información de los datos conservan las componentes calculadas. Un porcentaje más alto es mejor; como referencia, superar $70\%$ suele ser una representación razonable. |
| HAC | `HAC - Correlación cofenética: X` | Mide qué tan bien el dendrograma conserva las distancias originales entre observaciones. Va aproximadamente de $0$ a $1$; valores cercanos a $1$ son mejores. |
| K-Means | `K-Means - Silhouette score: X` | Mide si los grupos están compactos y separados. Va de $-1$ a $1$; cercano a $1$ es bueno, cercano a $0$ indica grupos mezclados y negativo es desfavorable. |
| t-SNE | `t-SNE - Trustworthiness: X` | Indica si los vecinos cercanos de los datos originales continúan cerca en el plano proyectado. Va de $0$ a $1$; valores cercanos a $1$ son mejores. |
| UMAP | `UMAP - Trustworthiness: X` | Mide la conservación de vecinos al reducir los datos. Va de $0$ a $1$; valores cercanos a $1$ son mejores. |

Las funciones también devuelven el modelo o resultado. En PCA y clustering se retorna un diccionario con el modelo y la métrica; en t-SNE y UMAP, el `DataFrame` del embedding conserva el modelo y `trustworthiness` en `resultado.attrs`.

## Estructura del proyecto

```text
app.py
pages/
	_shared.py
	unsupervised.py
	classification.py
	prediction.py
data/
	Archivos CSV de entrada
notebooks/
	Experimentacion_No_Supervisada.ipynb
	Pruebas_MVC.ipynb
	Version3_MVC.ipynb
	src/
		controller.py
		dataframe_desarrollado.py
		model.py
		view.py
		supervised/
			__init__.py
			_base.py
			Classification/
				__init__.py
				Clasificacion.py
		unsupervised/
			__init__.py
			_base.py
		clustering.py
		pca.py
tests/
	test_mvc.py
requirements.txt
```