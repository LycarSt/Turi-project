#!/usr/bin/env python3
"""
Lista recursivamente archivos .pcd en el repositorio y muestra conteo y tamaños.
Uso: python scripts/list_pcds.py [ruta_base]
"""
import sys
from pathlib import Path

def human_size(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def main(base):
    p = Path(base)
    if not p.exists():
        print(f"Ruta no existe: {p}")
        return
    pcds = list(p.rglob('*.pcd'))
    print(f"Buscando .pcd en: {p}")
    print(f"Encontrados: {len(pcds)} archivos .pcd")
    total = 0
    folders = {}
    for f in sorted(pcds):
        try:
            sz = f.stat().st_size
        except Exception:
            sz = 0
        total += sz
        folder = str(f.parent.relative_to(p))
        folders.setdefault(folder, 0)
        folders[folder] += 1
        print(f"- {str(f)}  ({human_size(sz)})")
    print("\nResumen por carpeta:")
    for k, v in sorted(folders.items(), key=lambda x: -x[1]):
        print(f"{k or '.'}: {v} pcd")
    print(f"\nTamaño total aproximado: {human_size(total)}")

if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(base)
