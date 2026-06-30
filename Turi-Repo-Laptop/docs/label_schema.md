# Esquema de etiquetas (label schema)

Este documento define el esquema de anotaciones que usaremos para las nubes de puntos del proyecto. Incluye clases, formato JSON por frame, reglas de QA y mapeo a formatos de entrenamiento como KITTI.

## Clases (clase recomendada inicial)
- PERSON: persona caminando/estática
- VEHICLE: vehículo (car, truck, motorcycle)
- OBSTACLE: objetos que bloquean el paso (pilares, vegetación grande, escombros)
- FURNITURE: mobiliario (sillas, mesas, estanterías)
- BACKGROUND: fondo/no etiquetado

Puedes extender con subclases si se necesita mayor granularidad.

## Formato JSON por frame
Cada frame (PCD) tendrá un fichero JSON en `data/labels/<frame>.json` con la siguiente estructura:

```json
{
  "frame": "1758208492.275396347.pcd",
  "timestamp": 1758208492.275396347,
  "sensor": "OAK-D" ,
  "annotations": [
    {
      "id": "obj_0001",
      "type": "PERSON",
      "bbox3d": {
        "center": [x, y, z],
        "dimensions": [dx, dy, dz],
        "rotation_y": ry
      },
      "score": 1.0,
      "attributes": {
        "occluded": false,
        "truncated": false
      }
    }
  ]
}
```

Campos obligatorios:
- `frame`: nombre del archivo .pcd
- `timestamp`: timestamp numérico
- `annotations`: lista de objetos
- Para cada objeto: `id`, `type` (una de las clases), `bbox3d` (center [x,y,z], dimensions [dx,dy,dz], rotation_y en radianes), `score` (opcional en anotaciones manuales puede ser 1.0), `attributes` (opcional)

Notas sobre coordenadas y convención:
- Usar el sistema de coordenadas del PCD tal como están (si transformas en preprocesado, documenta el cambio en metadata).\
- `rotation_y`: ángulo alrededor del eje vertical (Z) en radianes según convención del framework destino (documentar si cambias).\

## Mapeo a KITTI
- KITTI usa cajas 3D con formato específico; al convertir, deberás transformar `bbox3d` al formato KITTI (clase, trunc, occlusion, alpha, bbox2D, dimensions(l,h,w), location(x,y,z), rotation_y). Para campos que no tengas, usa -1 o 0 según la especificación.

## QA y reglas de anotación
- Anotar sólo objetos visibles con al menos N puntos (p.ej. >30 puntos). Si un objeto está parcialmente visible, marca `occluded=true`.
- No anotar objetos detrás del sensor o fuera del rango útil.
- Mantener nombres de `id` únicos por frame (p.ej. `obj_0001`).

## Ejemplos
- Un ejemplo de JSON está en: `docs/examples/label_example.json` (crear si es necesario).

## Validación automática
- Se proporciona `tools/validate_labels.py` que comprueba campos obligatorios, tipos y rangos básicos.
