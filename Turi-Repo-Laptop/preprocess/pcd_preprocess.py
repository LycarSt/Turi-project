#!/usr/bin/env python3
"""
Preprocesado de archivos .pcd: carga, filtrado de outliers, voxel downsample, opcional cálculo de normales
Salida: .npy (xyz) y .ply opcional

Uso:
  python preprocess/pcd_preprocess.py --input path/to/file.pcd --out-dir out/ --voxel 0.02 --remove-outliers 20 1.0
  python preprocess/pcd_preprocess.py --input-folder docs/Pruebas_posicion_2 --out-dir data/preprocessed --voxel 0.02
"""
import argparse
from pathlib import Path
import numpy as np
import csv
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def human_size(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def preprocess_file(in_path: Path, out_dir: Path, voxel_size: float = 0.02, remove_outliers: tuple = None, estimate_normals: bool = False):
    try:
        import open3d as o3d
    except Exception as e:
        print("ERROR: open3d no está instalado. Instálalo con: pip install open3d")
        raise

    print(f"Procesando: {in_path}")
    pcd = o3d.io.read_point_cloud(str(in_path))
    print("Puntos originales:", np.asarray(pcd.points).shape[0])

    # Remove statistical outliers
    if remove_outliers:
        nb_neighbors_raw, std_ratio_raw = remove_outliers
        # ensure correct types for Open3D API (nb_neighbors must be int)
        nb_neighbors = int(nb_neighbors_raw)
        std_ratio = float(std_ratio_raw)
        print(f"Aplicando StatisticalOutlierRemoval nb_neighbors={nb_neighbors} std_ratio={std_ratio}")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        print("Puntos tras outlier removal:", np.asarray(pcd.points).shape[0])

    # Voxel downsample
    if voxel_size and voxel_size > 0:
        print(f"Aplicando voxel downsample size={voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
        print("Puntos tras voxel:", np.asarray(pcd.points).shape[0])

    # Estimate normals
    if estimate_normals:
        print("Estimando normales")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    out_dir.mkdir(parents=True, exist_ok=True)
    base = in_path.stem
    np_out = out_dir / (base + '.npy')
    ply_out = out_dir / (base + '.ply')

    xyz = np.asarray(pcd.points, dtype=np.float32)
    np.save(str(np_out), xyz)
    print(f"Guardado: {np_out}  ({human_size(np_out.stat().st_size) if np_out.exists() else '0B'})")

    #Guardar PLY para inspección
    o3d.io.write_point_cloud(str(ply_out), pcd)
    print(f"Guardado: {ply_out}  ({human_size(ply_out.stat().st_size) if ply_out.exists() else '0B'})")

    return {'n_points': xyz.shape[0], 'np_path': str(np_out), 'ply_path': str(ply_out)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Archivo .pcd a procesar')
    parser.add_argument('--input-folder', type=str, help='Procesar todos los .pcd en la carpeta (recursivo)')
    parser.add_argument('--out-dir', type=str, default='data/preprocessed')
    parser.add_argument('--voxel', type=float, default=0.02)
    parser.add_argument('--remove-outliers', type=float, nargs=2, metavar=('NB','STD'), help='Aplicar StatisticalOutlierRemoval: nb_neighbors std_ratio')
    parser.add_argument('--estimate-normals', action='store_true')
    parser.add_argument('--workers', type=int, default=1, help='Número de workers para procesamiento paralelo')
    parser.add_argument('--skip-existing', action='store_true', help='No procesar archivos si existe .npy en out-dir')
    parser.add_argument('--overwrite', action='store_true', help='Forzar sobreescritura de outputs y metadata')
    parser.add_argument('--metadata-csv', type=str, default='', help='Ruta para CSV de metadatos (append)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Signal handling for graceful shutdown
    stop_requested = False

    def _handle_sigint(signum, frame):
        nonlocal stop_requested
        print('\nInterrupción recibida, deteniendo tras las tareas en curso...')
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_sigint)

    remove_outliers_val = tuple(map(float, args.remove_outliers)) if args.remove_outliers else None

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print('Input file no existe:', in_path)
            return
        preprocess_file(in_path, out_dir, voxel_size=args.voxel, remove_outliers=remove_outliers_val, estimate_normals=args.estimate_normals)
        return

    if not args.input_folder:
        print("Provee --input o --input-folder")
        return

    base = Path(args.input_folder)
    pcds = sorted(base.rglob('*.pcd'))
    print(f"En folder {base}, encontrados {len(pcds)} pcd")

    # Prepare metadata CSV
    metadata_file = Path(args.metadata_csv) if args.metadata_csv else None
    if metadata_file:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = not metadata_file.exists() or args.overwrite
        if write_header:
            with open(metadata_file, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=['input', 'n_points', 'np_path', 'ply_path', 'status', 'error'])
                writer.writeheader()

    # Worker function wrapper
    def _process(path: Path):
        try:
            np_out = Path(args.out_dir) / (path.stem + '.npy')
            if np_out.exists() and args.skip_existing and not args.overwrite:
                return {'input': str(path), 'status': 'skipped', 'n_points': None, 'np_path': str(np_out), 'ply_path': None, 'error': ''}

            res = preprocess_file(path, Path(args.out_dir), voxel_size=args.voxel, remove_outliers=remove_outliers_val, estimate_normals=args.estimate_normals)
            return {'input': str(path), 'status': 'ok', 'n_points': res.get('n_points'), 'np_path': res.get('np_path'), 'ply_path': res.get('ply_path'), 'error': ''}
        except Exception as e:
            return {'input': str(path), 'status': 'error', 'n_points': None, 'np_path': '', 'ply_path': '', 'error': str(e)}

    workers = max(1, int(args.workers))
    print(f"Procesando en paralelo con {workers} workers (skip_existing={args.skip_existing} overwrite={args.overwrite})")

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_process, p): p for p in pcds}
        with (open(metadata_file, 'a', newline='') if metadata_file else open('/dev/null', 'w')) as mfh:
            writer = csv.DictWriter(mfh, fieldnames=['input', 'n_points', 'np_path', 'ply_path', 'status', 'error']) if metadata_file else None
            for fut in as_completed(futures):
                if stop_requested:
                    break
                out = fut.result()
                # write metadata
                if writer:
                    writer.writerow(out)
                    mfh.flush()
                # print a short log
                print(f"{out['input']} -> {out['status']} ({out.get('n_points')})")


if __name__ == '__main__':
    main()
