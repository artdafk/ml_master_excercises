# Análisis Comparativo de Modelos Supervisados

Este proyecto implementa y compara tres modelos clásicos de aprendizaje supervisado (Regresión Logística, SVM, y Árboles de Decisión) para predecir el comportamiento de compra de clientes, utilizando una arquitectura de código modular y profesional.

## 1. Resumen del Problema
En el contexto del marketing digital, identificar qué usuarios tienen una alta probabilidad de adquirir un producto es crucial para optimizar el retorno de inversión (ROI) publicitario. 
El objetivo de este análisis es desarrollar un modelo predictivo capaz de clasificar si un usuario realizará una compra (`Purchased = 1`) o no (`Purchased = 0`) basándose en sus características demográficas.

**Dataset:** "Social Network Ads"
**Variables Independientes:** Edad (`Age`) y Salario Estimado (`EstimatedSalary`).
**Variable Objetivo:** Compra (`Purchased`).

## 2. Metodología Utilizada
El flujo de trabajo se adhiere a las mejores prácticas posibles de desarrollo y Machine Learning:

### 2.1. Análisis Exploratorio de Datos (EDA)
Antes del modelado, se realizó una inspección visual y estadística para comprender la naturaleza de los datos.
- **Distribución y Relaciones:** Se utilizó un `pairplot` para visualizar la separación de clases. Se observa que la edad y el salario son discriminadores fuertes, aunque no linealmente separables a la perfección.
- **Correlación:** Se calculó la matriz de correlación para evaluar la multicolinealidad.

![Pairplot de Variables](assets/pairplot.png)
![Matriz de Correlación](assets/correlation_matrix.png)

### 2.2. Preprocesamiento de Datos
Para garantizar la convergencia y el rendimiento óptimo de los modelos:
1.  **Limpieza:** Se descartó el `User ID` por carecer de valor predictivo.
2.  **Codificación:** La variable categórica `Gender` se codificó numéricamente (aunque para la visualización 2D final nos centramos en Edad y Salario).
3.  **Escalado (Feature Scaling):** Se aplicó `StandardScaler` para normalizar las características ($\mu=0, \sigma=1$).
    - *Justificación:* Algoritmos basados en distancias (como SVM) o gradientes (como Regresión Logística) son altamente sensibles a la magnitud de las variables. Sin escalado, el "Salario" (rango miles) dominaría sobre la "Edad" (rango decenas), sesgando el modelo.
4.  **División (Split):** Se particionó el dataset en 80% entrenamiento y 20% prueba.

### 2.3. Implementación de Modelos
Se evaluaron tres enfoques distintos:
1.  **Regresión Logística:** Modelo lineal base, ideal para establecer un benchmark y estimar probabilidades.
2.  **Support Vector Machine (SVM):** Utilizando un kernel Radial (RBF) para capturar fronteras de decisión complejas y no lineales.
3.  **Árbol de Decisión:** Modelo no paramétrico que ofrece alta interpretabilidad a través de reglas de decisión jerárquicas (`max_depth=4`).

## 3. Comparación Experimental y Resultados

### 3.1. Definición de Métricas de Evaluación
Para evaluar la calidad de los clasificadores, se analizaron las siguientes métricas:
- **Matriz de Confusión:** Tabla que contrasta las predicciones contra los valores reales (TP, TN, FP, FN). Fundamental para ver errores tipo I y II.
- **Precisión (Precision):** La proporción de identificaciones positivas que fueron realmente correctas ($TP / (TP + FP)$). Importante para minimizar falsos positivos (campañas desperdiciadas).
- **Recall (Sensibilidad):** La proporción de positivos reales que se identificaron correctamente ($TP / (TP + FN)$). Crítico si queremos capturar a todos los posibles compradores.
- **F1-Score:** La media armónica entre Precisión y Recall. Proporciona una métrica única balanceada.

### 3.2. Resultados Visuales

#### Regresión Logística
![Matriz Confusión - Regresión Logística](assets/confusion_matrix_regresión_logística.png)
![Frontera - Regresión Logística](assets/decision_boundary_regresión_logística.png)

#### Support Vector Machine (SVM)
![Matriz Confusión - SVM](assets/confusion_matrix_svm.png)
![Frontera - SVM](assets/decision_boundary_svm.png)

#### Árbol de Decisión
![Matriz Confusión - Árbol](assets/confusion_matrix_árbol_de_decisión.png)
![Frontera - Árbol](assets/decision_boundary_árbol_de_decisión.png)

### 3.3. Comparación de Rendimiento (Cross-Validation)
Se realizó una validación cruzada (k-fold=5) para asegurar que los resultados no dependan de una partición de datos específica.

![Comparación Cross-Validation](assets/cross_validation_comparison.png)

### 3.4. Tabla Resumen de Métricas
*(Valores referenciales obtenidos en la ejecución)*

| Modelo | Accuracy Promedio (CV) | Precisión | Recall | F1-Score | Características Clave |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Regresión Logística** | ~0.82 | Alta (>0.85) | Media (~0.75) | ~0.80 | Frontera lineal. Simple y rápido. |
| **SVM (Kernel RBF)** | **~0.91** | **Muy Alta** | **Alta** | **~0.90** | **Captura no linealidad. Mejor balance.** |
| **Árbol de Decisión** | ~0.89 | Alta | Alta | ~0.88 | Interpretable. Fronteras ortogonales. |

## 4. Conclusiones Técnicas
1.  **Superioridad del Modelo SVM:** El modelo Support Vector Machine con kernel RBF demostró consistentemente el mejor rendimiento (mayor Accuracy y F1-Score). Esto se debe a que la frontera de decisión real entre compradores y no compradores en este dataset no es lineal; tiene una curvatura que el modelo lineal (Regresión Logística) no puede capturar adecuadamente.
2.  **Efectividad del Árbol de Decisión:** El árbol de decisión logró un rendimiento muy competitivo, casi a la par del SVM. Su capacidad para definir regiones rectangulares en el espacio de características se adapta bien a la distribución de los datos, ofreciendo además la ventaja de reglas de negocio explícitas (*e.g., "Si Edad > 45 y Salario > 90k..."*).
3.  **Importancia del Preprocesamiento:** La estandarización de las variables fue un paso crítico. Los experimentos demostraron que sin escalar `Age` y `EstimatedSalary`, el modelo SVM fallaría en encontrar el hiperplano óptimo, degradando significativamente su capacidad predictiva.
4.  **Recomendación de Implementación:** 
    - Si el objetivo es **maximizar la precisión predictiva** pura, se recomienda implementar **SVM**.
    - Si el negocio requiere **explicar el "por qué"** de una clasificación a un equipo no técnico, el **Árbol de Decisión** es la mejor alternativa con un sacrificio mínimo de precisión.