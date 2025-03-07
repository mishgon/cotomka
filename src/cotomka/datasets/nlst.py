from typing import Iterable, List, Tuple
from pathlib import Path
import warnings
from tqdm.auto import tqdm
from multiprocessing import Pool
import pydicom
import numpy as np

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.dicom import (
    get_series_uid, Plane, get_series_slice_plane, drop_duplicated_slices, order_series,
    get_series_image, get_series_voxel_spacing, get_series_orientation_matrix,
    to_canonical_orientation
)
from cotomka.utils.io import save_numpy, save_json, load_numpy, load_json


class NLST(Dataset):
    name = 'nlst'

    def _get_image(self, index: str) -> np.ndarray:
        return load_numpy(self.root_dir / index / 'image.npy.gz', decompress=True).astype('float32')

    def _get_voxel_spacing(self, index: str) -> Tuple[float, float, float]:
        return tuple(load_json(self.root_dir / index / 'voxel_spacing.json'))

    def prepare(self, src_dir: str | Path, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        patient_dirs = list(Path(src_dir).glob('NLST/*'))

        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(self._prepare, patient_dirs), total=len(patient_dirs)))

    def _prepare(self, patient_dir: Path) -> None:
        series_dir = max(_iterate_series_dirs(patient_dir), key=_estimate_series_len)

        series = _load_series(series_dir)

        # extract image, voxel spacing and orientation matrix from dicoms
        # drop non-axial series and series with invalid tags
        try:
            if get_series_slice_plane(series) != Plane.Axial:
                raise ValueError('Series is not axial')

            series = drop_duplicated_slices(series)
            series = order_series(series)

            series_uid = get_series_uid(series)
            image = get_series_image(series)
            voxel_spacing = get_series_voxel_spacing(series)
            om = get_series_orientation_matrix(series)
        except (AttributeError, ValueError, NotImplementedError) as e:
            warnings.warn(f'Series at {str(series_dir)} fails with {e.__class__.__name__}: {str(e)}')
            return

        image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, om)

        data_dir = self.root_dir / series_uid
        data_dir.mkdir()
        save_numpy(image.astype('int16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)
        save_json(voxel_spacing, data_dir / 'voxel_spacing.json')


def _iterate_series_dirs(patient_dir: Path) -> Iterable[Path]:
    for study_dir in patient_dir.iterdir():
        study_dir, = study_dir.iterdir()
        for path in study_dir.iterdir():
            if path.is_dir():
                yield path


def _estimate_series_len(series_dir: Path) -> int:
    return len(list(series_dir.glob('*.dcm')))


def _load_series(series_dir: Path) -> List[pydicom.FileDataset]:
    return list(map(pydicom.dcmread, series_dir.glob('*.dcm')))
