from typing import Tuple
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm
import gzip
import nibabel
import numpy as np

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation, is_diagonal
from cotomka.utils.io import save_numpy, save_json, load_numpy


class KiTS21(Dataset):
    name = 'kits21'

    def _load_mask(self, index: str) -> np.ndarray:
        return load_numpy(self.root_dir / index / 'mask.npy.gz', decompress=True)

    def prepare(self, src_dir: str | Path, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        src_dir = Path(src_dir)

        ids = [p.name for p in src_dir.glob('kits21/data/*') if p.is_dir()]

        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(partial(self._prepare, src_dir=src_dir), ids), total=len(ids)))
    
    def _prepare(self, id_: str, src_dir: Path):
        image_file, = src_dir.glob(f'kits21/data/{id_}/imaging.nii.gz')
        try:
            mask_file, = src_dir.glob(f'kits21/data/{id_}/aggregated_MAJ_seg.nii.gz')
        except ValueError:
            return

        image, affine = _load_nii_gz(image_file)
        mask, mask_affine = _load_nii_gz(mask_file)
        if not is_diagonal(affine[:3, :3]) or not is_diagonal(mask_affine[:3, :3]):
            return
        voxel_spacing = affine_to_voxel_spacing(affine)
        image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)
        mask, _ = to_canonical_orientation(mask, None, mask_affine)

        data_dir = self.root_dir / id_
        data_dir.mkdir()
        save_numpy(image.astype('int16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)
        save_json(voxel_spacing, data_dir / 'voxel_spacing.json')
        save_numpy(mask.astype('uint8'), data_dir / 'mask.npy.gz', compression=1, timestamp=0)


def _load_nii_gz(nii_gz_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    with gzip.GzipFile(filename=nii_gz_file) as nii_f:
        fh = nibabel.FileHolder(fileobj=nii_f)
        nii = nibabel.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        nii = nibabel.as_closest_canonical(nii)
        image = nii.get_fdata()
        affine = nii.affine

    return image, affine
