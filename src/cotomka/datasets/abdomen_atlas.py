from typing import Tuple
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from multiprocessing import Pool
import nibabel
import numpy as np

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation, is_diagonal
from cotomka.utils.io import save_numpy, save_json, load_numpy, load_json


class AbdomenAtlas(Dataset):
    name = 'abdomen_atlas'

    def _get_image(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / id / 'image.npy.gz', decompress=True).astype('float32')

    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        return tuple(load_json(self.root_dir / id / 'voxel_spacing.json'))

    def _get_mask(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / id / 'mask.npy.gz', decompress=True)

    def prepare(self, src_dir: str | Path, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        src_dir = Path(src_dir)

        ids = [path.name for path in src_dir.glob('uncompressed/BDMAP_*')]

        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(partial(self._prepare, src_dir=src_dir), ids), total=len(ids)))

    def _prepare(self, id: str, src_dir: Path) -> None:
        image, affine = _load_nii(src_dir / 'uncompressed' / id / 'ct.nii.gz')
        mask, mask_affine = _load_nii(src_dir / 'uncompressed' / id / 'combined_labels.nii.gz')
        if not is_diagonal(affine[:3, :3]) or not is_diagonal(mask_affine[:3, :3]):
            return
        voxel_spacing = affine_to_voxel_spacing(affine)
        image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)
        mask, _ = to_canonical_orientation(mask, None, mask_affine)

        data_dir = self.root_dir / id
        data_dir.mkdir()
        save_numpy(image.astype('int16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)
        save_json(voxel_spacing, data_dir / 'voxel_spacing.json')
        save_numpy(mask.astype('uint8'), data_dir / 'mask.npy.gz', compression=1, timestamp=0)


def _load_nii(nii_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    nii = nibabel.load(nii_file)
    nii = nibabel.as_closest_canonical(nii)
    image = nii.get_fdata()
    affine = nii.affine
    return image, affine
