#!/usr/bin/env python3
"""
Valida archivos JSON de etiquetas en `data/labels/` según el esquema definido en `docs/label_schema.md`.
Uso: python tools/validate_labels.py [--labels-dir data/labels]
"""
import json
from pathlib import Path
import argparse

VALID_CLASSES = {"PERSON","VEHICLE","OBSTACLE","FURNITURE","BACKGROUND"}

def validate_file(path: Path):
    errors = []
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return [f"JSON parse error: {e}"]

    if 'frame' not in data:
        errors.append('Missing "frame"')
    if 'timestamp' not in data:
        errors.append('Missing "timestamp"')
    if 'annotations' not in data:
        errors.append('Missing "annotations"')
    else:
        if not isinstance(data['annotations'], list):
            errors.append('"annotations" must be a list')
        else:
            for i,obj in enumerate(data['annotations']):
                if 'id' not in obj:
                    errors.append(f'annotation[{i}] missing id')
                if 'type' not in obj:
                    errors.append(f'annotation[{i}] missing type')
                else:
                    if obj['type'] not in VALID_CLASSES:
                        errors.append(f'annotation[{i}] invalid type: {obj.get("type")}')
                if 'bbox3d' not in obj:
                    errors.append(f'annotation[{i}] missing bbox3d')
                else:
                    bbox = obj['bbox3d']
                    if not all(k in bbox for k in ('center','dimensions','rotation_y')):
                        errors.append(f'annotation[{i}] bbox3d missing fields')
                    else:
                        c = bbox['center']
                        d = bbox['dimensions']
                        if not (isinstance(c,list) and len(c)==3):
                            errors.append(f'annotation[{i}] bbox3d.center invalid')
                        if not (isinstance(d,list) and len(d)==3):
                            errors.append(f'annotation[{i}] bbox3d.dimensions invalid')

    return errors

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels-dir', default='data/labels')
    args = p.parse_args()
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        print(f'Labels dir not found: {labels_dir} (no hay archivos para validar)')
        return

    files = list(labels_dir.glob('*.json'))
    if not files:
        print('No JSON label files found in', labels_dir)
        return

    total_errors = 0
    for f in sorted(files):
        errs = validate_file(f)
        if errs:
            total_errors += 1
            print(f'-- {f} ERRORS:')
            for e in errs:
                print('   -', e)
    if total_errors==0:
        print('All label files passed basic validation.')
    else:
        print(f'{total_errors} files with errors (see above).')

if __name__ == '__main__':
    main()
