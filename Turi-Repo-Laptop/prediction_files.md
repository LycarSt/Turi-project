# Sección final: decisiones, criterios y ubicación de artefactos (CONSOLIDADO)

Esta sección debe ser la única fuente de verdad para el pipeline — aquí se registran las decisiones, criterios de cierre y la ubicación de notebooks, modelos y datos. Cualquier cambio importante del pipeline debe reflejarse en este fichero.
1) ¿Qué debe añadirse al notebook de EDA para considerarlo completo?
   - Cabecera de metadatos: versión de Python, `pip freeze` y commit hash (automatizable con `git rev-parse --short HEAD`).
   - Celda de configuración al inicio con parámetros editables: `VOXEL_SIZE`, `OUTLIER_NB`, `OUTLIER_STD`, `SAMPLE_RATE`, `OUT_DIR`.
   - Guardado automático de estadísticas por frame en `reports/eda_summary.csv`.
   - Visualizaciones adicionales: heatmap XY (densidad), histograma de distancias radial, diagrama de caja (boxplot) de Z por escena.
   - Celdas de validación: asserts sobre shapes y existencia de archivos tras preprocesado.
2) ¿Cuándo se considera terminado el notebook? Criterios de aceptación
   - Incluye la celda de configuración y de metadatos. (Sí/No)
   - Ejecuta completamente desde cero en un entorno limpio usando `pip install -r requirements.txt`. (Sí/No)
   - Genera `reports/eda_summary.csv` y `data/preprocessed/` para el conjunto de prueba (p. ej. 10 frames). (Sí/No)
   - Contiene instrucciones claras para reproducir el preprocesado en batch (comando). (Sí/No)
   - Contiene tests mínimos que se ejecutan sin error (asserts). (Sí/No)
3) ¿Dónde guardar el notebook del modelo de ML?
   - Notebooks de experimentación y entrenamiento: `notebooks/model_training.ipynb`.
   - Notebook de producción / documentación del pipeline: `notebooks/model_pipeline_and_docs.ipynb` o mantener `prediction_files.md` como la fuente de verdad si prefieres texto plano.

4) ¿Dónde guardar los modelos y checkpoints?
   - Estructura recomendada en `models/`:
5) Pasos que faltan por hacer (resumido) y cómo implementarlos
    - a) Crear `tools/convert_to_kitti.py` (detallar abajo). Responsable: Dev. Entrega: script verificado.\
       - Lectura: `data/preprocessed/*.npy` y `data/labels/*`\
       - Salida: `dataset/kitti/velodyne/*.bin`, `dataset/kitti/label_2/*.txt`\
    - b) Implementar pipeline de entrenamiento (OpenPCDet o TorchPoints3D). Responsable: ML Engineer. Entrega: `training/README.md`, `configs/`, `scripts/train.sh`\
    - c) Evaluación automática: `scripts/evaluate.sh` que llama al framework y guarda `reports/metrics_*.json`\
   - d) Exportación: `tools/export_to_onnx.py` que carga checkpoint y genera ONNX/TorchScript; incluir test de inferencia mínima.\
   - e) Integración en robot/device: script `inference/run_inference.py` que carga `models/exports/model.onnx` y hace inferencia sobre `data/preprocessed/` o stream en vivo.\

6) Ejemplo rápido: `tools/convert_to_kitti.py` (qué debe hacer)
   - Recibir `--preprocessed-dir` y `--labels-dir` y `--out-dir`\
7) Buenas prácticas y gobernanza
   - Cada experimento debe incluir `metadata.json` con: commit, fecha, dataset hash (o lista de frames), parámetros de preprocesado y métricas finales.\
   - Mantener `prediction_files.md` como fuente de verdad (actualizar cuando cambie el pipeline).\
   - Añadir un pequeño CI que valide `scripts/list_pcds.py` y `preprocess/pcd_preprocess.py` (ejecutar sobre 2–3 archivos y comprobar salida) para evitar regresiones.

## Checklist final (usar para marcar progreso en el repo)
- [ ] `notebooks/model_pipeline_and_docs.ipynb` o `prediction_files.md` contienen toda la documentación final.
- [ ] `data/preprocessed/` poblado con frames representativos.
- [ ] `data/labels/` con anotaciones suficientes para baseline.
- [ ] `tools/convert_to_kitti.py` implementado.
- [ ] Baseline entrenado y checkpoint disponible en `models/`.
- [ ] Modelo exportado en `models/exports/`.
- [ ] Script de inferencia probado en `Pruebas OAK-D/` o `inference/`.

---

Si quieres, puedo crear ahora los esqueletos de `tools/convert_to_kitti.py` y `tools/export_to_onnx.py`, además de un pequeño CI local que ejecute el preprocesado en un par de PCDs como prueba automatizada. ¿Qué prefieres que haga ahora?
# Archivos utilizados para predicción (objetos / personas / obstáculos)

Este documento lista exclusivamente los ficheros y ubicaciones dentro del repositorio que usaremos en el pipeline de obtención del modelo de percepción (detección/segmentación sobre nubes de puntos), y describe el orden de ejecución recomendado para generar el modelo entrenado.

---

## Datos crudos
- `docs/Pruebas_posicion_2/*.pcd` — Carpeta con las nubes de puntos originales (frames). Fuente principal de los datos.

## Scripts de preparación y exploración (existentes en el repositorio)
- `scripts/list_pcds.py` — Lista recursiva de `.pcd` y genera un resumen por carpeta. Útil para inventario inicial.
- `preprocess/pcd_preprocess.py` — Script de preprocesado (lectura PCD, eliminación de outliers, voxel downsample, estimación de normales opcional). Guarda `.npy` y `.ply` en `data/preprocessed/`.
- `notebooks/pcd_explore.ipynb` — Notebook inicial para inspección visual y estadística rápida de PCDs (visualización con Open3D).
- `Codigo_Data/leerBag.py` — (Si aplicable) conversión de `.bag` a `.pcd` (usa si tienes datos en formato ROS bag).
- `Codigo_Data/visualizadorV2.py` — visualizador secuencial de `.pcd` (útil para inspección y QA manual).

## Scripts de detección presente en el repositorio (uso en inferencia / ejemplos)
- `Pruebas OAK-D/DeteccionPersonas.py` — ejemplo que usa OAK‑D para detección de personas (inferencia en dispositivo OAK‑D).
- `Pruebas OAK-D/AlertasFinal.py` y `Pruebas OAK-D/AlertasFinalConectado.py` — ejemplos/alertas integradas que pueden usarse como referencia del pipeline de inferencia.

Nota: en este repositorio no hay (a la fecha) un script de entrenamiento completo ya configurado (p.ej. OpenPCDet config). Es necesario usar un framework externo (OpenPCDet, TorchPoints3D o implementar entrenamiento PointNet++) para la etapa de entrenamiento. En las siguientes secciones indico el orden y los puntos donde integrar dichas herramientas.

---

## Estructura de salida recomendada
- `data/preprocessed/` — resultado del preprocesado (.npy por frame con xyz, opcional .ply para visualización). También se puede crear `data/labels/` con anotaciones por frame.

## Formatos intermedios sugeridos
- Para detección 3D: formato KITTI / OpenPCDet (.bin + label txt) o formato propio `.npy` + `.json` labels.
- Para segmentación: `.npy` con shape [N, 3] y `.npy` labels [N] o `.ply` con propiedad de label por vértice.

---

# Paso a paso del desarrollo y orden de ejecución

Estos pasos explican qué archivos ejecutar, en qué orden, y qué esperar en cada etapa para obtener finalmente un modelo entrenado y exportado listo para inferencia.

1) Inventario y validación de datos (comprobar cobertura y tamaño)
   - Archivo(s): `scripts/list_pcds.py`
   - Acción: ejecutar para confirmar número de frames y tamaños.
   - Comando:
     ```bash
     python3 scripts/list_pcds.py docs/Pruebas_posicion_2
     ```

2) Exploración rápida y muestreo (EDA)
   - Archivo(s): `notebooks/pcd_explore.ipynb`
   - Acción: abrir el notebook, ejecutar la celda que carga un ejemplo, inspeccionar histogramas de Z, densidad de puntos y visualización.
   - Objetivo: decidir parámetros de preprocesado (voxel size, outlier thresholds).

3) Preprocesado por frame (limpieza y reducción)
   - Archivo(s): `preprocess/pcd_preprocess.py`
   - Acción: aplicar StatisticalOutlierRemoval y voxel_down_sample con los parámetros definidos en EDA. Genera `.npy` y `.ply` en `data/preprocessed`.
   - Ejemplo de comando (procesar todo el folder):
     ```bash
     python3 preprocess/pcd_preprocess.py --input-folder docs/Pruebas_posicion_2 --out-dir data/preprocessed --voxel 0.02 --remove-outliers 20 1.0
     ```
   - Resultado: conjunto de archivos limpios listos para anotación o conversión.

4) Anotación / etiquetado
   - Archivo(s): (herramienta externa: CVAT 3D, LabelCloud o una solución custom basada en Open3D)
   - Acción: anotar objetos de interés (person, vehicle, obstacle, furniture, etc.) por frame.
   - Salida esperada: `data/labels/*.json` o `data/labels/*.txt` por frame siguiendo el formato elegido (KITTI/Custom).
   - Recomendación: anotar inicialmente un subset (p.ej. 500–1000 frames) para crear un baseline.

5) Conversión al formato del framework de entrenamiento
   - Archivo(s): script a crear `tools/convert_to_kitti.py` (no incluido actualmente)
   - Acción: convertir `data/preprocessed/*.npy` + `data/labels/*` al formato requerido por el modelo (ej. OpenPCDet requiere `.bin` y `label_2` txt con boxes/class).
   - Resultado: dataset listo para el entrenamiento con la librería elegida.

6) Entrenamiento del modelo (usar framework externo)
   - Archivo(s): fuera del repo (OpenPCDet / TorchPoints3D / implementación PointNet++)
   - Acciones generales:
     - Preparar `configs/` (anchors, voxel size, batch size, augmentations).
     - Ejecutar script de entrenamiento del framework elegido. Ejemplo genérico:
       ```bash
       # Ejemplo con OpenPCDet (instalado en entorno):
       python tools/train.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 8
       ```
   - Resultado: checkpoints en `output/` con modelos entrenados.

7) Evaluación y selección de checkpoint
   - Archivo(s): scripts de evaluación del framework (p. ej. `tools/test.py` en OpenPCDet) y notebooks para analizar métricas (mAP, IoU).
   - Acción: evaluar sobre split de validación, seleccionar mejor checkpoint.

8) Exportar modelo para despliegue
   - Archivo(s): export script del framework o `tools/export_to_onnx.py` (crear si necesario)
   - Acción: convertir checkpoint a ONNX o TorchScript.
   - Ejemplo de objetivo: `model.onnx`, `model.pt`.

9) Integración de inferencia en el repositorio
   - Archivo(s): `Pruebas OAK-D/DeteccionPersonas.py`, `Pruebas OAK-D/AlertasFinal.py`
   - Acción: adaptar/incluir un script de inferencia que cargue `model.onnx` o `model.pt` y procese nubes de puntos en tiempo real o desde archivos `.pcd`.
   - Flujo mínimo de inferencia recomendado:
     - Cargar PCD -> aplicar mismo preprocesado (voxel, normalización) -> convertir al tensor de entrada del modelo -> inferir -> postprocesar boxes/labels -> visualizar/loggear.

10) Pruebas en hardware y monitorización
   - Archivo(s): `Pruebas OAK-D/*` para pruebas en OAK, y scripts custom para medir latencia.
   - Acción: medir latencia, memoria y robustez en el entorno objetivo; iterar optimizaciones (quant, pruning) si es necesario.

---

# Orden de ejecución (resumen rápido)
1. `python3 scripts/list_pcds.py docs/Pruebas_posicion_2`
2. Abrir `notebooks/pcd_explore.ipynb` y ajustar parámetros (voxel, outlier).
3. `python3 preprocess/pcd_preprocess.py --input-folder docs/Pruebas_posicion_2 --out-dir data/preprocessed --voxel <VOXEL> --remove-outliers <NB> <STD>`
4. Anotar un subset con la herramienta elegida -> guardar en `data/labels/`.
5. Convertir a formato del framework: `tools/convert_to_kitti.py` (crear) o similar.
6. Entrenar con framework (OpenPCDet / TorchPoints3D): `python tools/train.py --cfg_file ...`
7. Evaluar con `python tools/test.py --ckpt ...` y seleccionar checkpoint.
8. Exportar a ONNX/TorchScript con script de exportación.
9. Integrar modelo exportado en `Pruebas OAK-D/` o script de inferencia propio.

---

Si quieres, creo ahora los scripts faltantes que referencio en los pasos 5 y 8 (`tools/convert_to_kitti.py` y `tools/export_to_onnx.py`) y/o preparo un ejemplo de entrenamiento minimal usando PointNet++ para que tengas un pipeline completo dentro del repo. ¿Qué prefieres que haga ahora?
