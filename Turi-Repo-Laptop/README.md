# Proyecto Unitree Go1 - Laboratorio de IA (UNI)

Este repositorio está centrado en controlar los códigos y ejemplos que se usan para el proyecto del **Unitree Go1** en el **Laboratorio de IA de la UNI**.

Para encontrar los aspectos de funcionamiento basico, configuraciones internas y de red del robot de manera detallada, sirvase de leer la guia elaborada por el equipo:

https://garnet-battery-1d1.notion.site/Manual-Robot-Dog-1b472c8d82bf80baa843ff48c5089a67?pvs=74


## Contenido del Repositorio

- **`example_walk.py`**: Código de ejemplo proporcionado con el SDK *Legged* del Unitree Go1.
- **`prueba_teclado.py`**: Personalización del código de ejemplo para control mediante teclado.
- **`prueba_control_manos.py`**: Personalización del código para control mediante gestos de la mano.
  - Este archivo debe ser usado en simultáneo con **`Mediapipe.py`** para habilitar el control mediante gestos manuales.

- **`leerBag.py`**: Codigo que convierte los "videos" de nubes de puntos (.bag) en "fotos" de nubes de puntos (.pcd)
- **`visualizadorV2.py`**: Codigo que convierte permite reproducir todas las "fotos" de nubes de puntos (.pcd) en una carpeta y visualizarlas de manera  secuencial
- **`build_map.rviz`**: Archivo rviz configurado, extraido del Go1, que permite visualizar data en vivo del lidar en un visualizador rviz