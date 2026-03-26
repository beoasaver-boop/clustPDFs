# PDF Clustering with K-Means

Un análisis avanzado de clustering para documentos PDF utilizando procesamiento de texto y algoritmo K-Means con optimizaciones de rendimiento.

## 📋 Descripción

Este proyecto permite:
- Extraer texto de múltiples PDFs en paralelo
- Vectorizar documentos con TF-IDF
- Aplicar clustering K-Means para clasificar PDFs por similaridad
- Generar análisis estadísticos y visualizaciones
- Exportar resultados a Excel

## 🚀 Características

✅ **Extracción paralela** - Procesa múltiples PDFs simultáneamente  
✅ **Manejo robusto de errores** - PDFs dañados no rompen el proceso  
✅ **Análisis automático** - Estadísticas, palabras clave y métricas de calidad  
✅ **Visualizaciones** - Gráficos de distribución y proporciones  
✅ **Exportación** - Resultados en Excel con múltiples hojas  

## 📦 Instalación

```bash
pip install PyPDF2 pandas scikit-learn pdfplumber matplotlib seaborn openpyxl requests plotly
```

## 💻 Uso

```bash
python cluster_kmeans_pdf.py
```

El script solicitará:
1. Ruta de la carpeta con PDFs
2. Número de clusters deseados (default: 3)

## 📊 Funciones Principales

### `extract_text_from_pdfs(folder_path, use_parallel=True, max_workers=4)`
Extrae texto de todos los PDFs de una carpeta con procesamiento paralelo.

### `process_and_cluster_pdfs(folder_path, n_clusters=5, use_parallel=True)`
Pipeline completo: extracción → vectorización → clustering.

### `get_cluster_stats(df_clusters)`
Retorna cantidad y porcentaje de PDFs por cluster.

### `get_top_keywords_per_cluster(texts, clusters, vectorizer, n_keywords=5)`
Identifica palabras clave representativas para cada cluster.

### `evaluate_clustering(X_scaled, clusters, vectorizer)`
Calcula métricas de calidad: Silhouette Score y Davies-Bouldin Index.

### `export_results(df_clusters, stats, keywords, output_file='cluster_results.xlsx')`
Exporta resultados a archivo Excel con múltiples hojas.

### `plot_cluster_distribution(df_clusters)`
Gráfico de barras con distribución de PDFs por cluster.

### `plot_cluster_size_pie(df_clusters)`
Gráfico pastel mostrando proporciones de cada cluster.

### `get_cluster_summary(df_clusters, keywords_dict)`
Resumen conciso con documentos y palabras clave por cluster.

## 📈 Salida

El programa genera:
- **Estadísticas de clustering** - Cantidad y porcentaje por cluster
- **Resumen de clusters** - Documentos y palabras clave principales
- **Métricas de calidad**:
  - *Silhouette Score*: Entre -1 y 1 (cercano a 1 es mejor)
  - *Davies-Bouldin Index*: Valores menores son mejores
- **Archivo Excel** (`cluster_results.xlsx`) con múltiples hojas
- **Visualizaciones** - Gráficos de distribución y proporciones

## ⚙️ Parámetros Configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `n_clusters` | 5 | Número de clusters K-Means |
| `use_parallel` | True | Activar procesamiento paralelo |
| `max_workers` | 4 | Número máximo de workers en paralelo |
| `max_features` | 1000 | Máximo número de features TF-IDF |
| `max_df` | 0.8 | Máximo documento frequency |
| `min_df` | 2 | Mínimo documento frequency |

## 🔧 Optimizaciones

- **pdfplumber** en lugar de PyPDF2 para extracción más rápida
- **ThreadPoolExecutor** para procesamiento paralelo
- **List comprehension** + `join()` para concatenación eficiente
- **max_features limitado** en TF-IDF para mejor rendimiento
- **Escalado StandardScaler** para mejor convergencia K-Means
- **n_init=10** para mejor exploración de soluciones

## 📝 Ejemplo de Flujo

```python
# 1. Extrae PDFs
texts = extract_text_from_pdfs("./pdfs", use_parallel=True)

# 2. Vectoriza
vectorizer = TfidfVectorizer(...)
X_tfidf = vectorizer.fit_transform(texts)

# 3. Clusteriza
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_scaled)

# 4. Analiza
stats = get_cluster_stats(df_clusters)
keywords = get_top_keywords_per_cluster(texts, clusters, vectorizer)
metrics = evaluate_clustering(X_scaled, clusters, vectorizer)

# 5. Exporta y visualiza
export_results(df_clusters, stats, keywords)
plot_cluster_distribution(df_clusters)
```

## 🐛 Manejo de Errores

El programa:
- Valida existencia de carpeta y PDFs
- Captura excepciones de PDFs dañados sin interrumpir el proceso
- Filtra documentos vacíos automáticamente
- Verifica que haya al menos 1 PDF para procesar

## 📋 Requisitos

- Python 3.7+
- Librerías especificadas en `pip install`

## 📄 Licencia

Este proyecto es de código abierto.

## 👤 Autor

Proyecto de análisis de clustering para documentos PDF.

---

**Última actualización**: Marzo 2026
