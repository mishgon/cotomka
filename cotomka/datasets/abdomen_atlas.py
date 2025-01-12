from typing import Tuple
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool
import nibabel
import numpy as np


from cotomka.datasets.base import Dataset
from cotomka.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation, is_diagonal
from cotomka.utils.io import save_numpy, save_json, load_numpy


class AbdomenAtlas(Dataset):
    name = 'abdomen_atlas'

    def _load_mask(self, index: str) -> np.ndarray:
        return load_numpy(self.root_dir / index / 'mask.npy.gz', decompress=True)

    def prepare(self, src_dir: str | Path, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        src_dir = Path(src_dir)

        ids = [path.name for path in src_dir.glob('uncompressed/BDMAP_*')]

        def prepare(id_: str) -> None:
            image, affine = _load_nii(src_dir / 'uncompressed' / id_ / 'ct.nii.gz')
            mask, mask_affine = _load_nii(src_dir / 'uncompressed' / id_ / 'combined_labels.nii.gz')
            if not is_diagonal(affine[:3, :3]) or not is_diagonal(mask_affine[:3, :3]):
                return
            voxel_spacing = affine_to_voxel_spacing(affine)
            image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)
            mask, _ = to_canonical_orientation(mask, None, mask_affine)

            save_dirpath = self.root_dir / id_
            save_dirpath.mkdir()
            save_numpy(image.astype('int16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
            save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
            save_numpy(mask.astype('uint8'), save_dirpath / 'mask.npy.gz', compression=1, timestamp=0)
        
        with ThreadPool(num_workers) as p:
            _ = list(tqdm(p.imap(prepare, ids), total=len(ids)))


def _load_nii(nii_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    nii = nibabel.load(nii_file)
    nii = nibabel.as_closest_canonical(nii)
    image = nii.get_fdata()
    affine = nii.affine
    return image, affine
