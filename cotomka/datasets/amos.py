import zipfile
import gzip
from pathlib import Path
from typing import Tuple
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
import numpy as np
import nibabel

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation
from cotomka.utils.io import save_numpy, save_json, load_numpy


# ids that cause some runtime errors
DROP_IDS = (
    '5397', '5073', '5120', '5173', '5414', '5437', '5514', '5547', '5588',
    '5640', '5654', '5677', '5687', '5721', '5871', '5950', '6016'
)


class AMOSCTLabeledTrain(Dataset):
    name = 'amos_ct_labeled_train'

    def _load_mask(self, index: str) -> np.ndarray:
        return load_numpy(self.root_dir / index / 'mask.npy.gz', decompress=True)

    def prepare(self, src_dir: str | Path, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        src_dir = Path(src_dir)

        ids = []
        with zipfile.ZipFile(src_dir / 'amos22.zip') as zf:
            for name in zf.namelist():
                if name.startswith('amos22/imagesTr/amos_') and name.endswith('.nii.gz'):
                    ids.append(name[len('amos22/imagesTr/amos_'):-len('.nii.gz')])
        ids = [i for i in ids if int(i) <= 500 and i not in DROP_IDS]

        def prepare(id_: str) -> None:
            image, affine = _load_nii_from_zip(src_dir, 'amos22.zip', f'amos22/imagesTr/amos_{id_}.nii.gz')
            mask, mask_affine = _load_nii_from_zip(src_dir, 'amos22.zip', f'amos22/labelsTr/amos_{id_}.nii.gz')
            voxel_spacing = affine_to_voxel_spacing(affine)
            image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)
            mask, _ = to_canonical_orientation(mask, None, mask_affine)
 
            data_dir = self.root_dir / id_
            data_dir.mkdir()
            save_numpy(image.astype('int16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)
            save_json(voxel_spacing, data_dir / 'voxel_spacing.json')
            save_numpy(mask.astype('uint8'), data_dir / 'mask.npy.gz', compression=1, timestamp=0)

        with ThreadPool(num_workers) as p:
            _ = list(tqdm(p.imap(prepare, ids), total=len(ids)))


class AMOSCTVal(Dataset):
    name = 'amos_ct_val'

    def _load_mask(self, index: str) -> np.ndarray:
        return load_numpy(self.root_dir / index / 'mask.npy.gz', decompress=True)

    def prepare(self, src_dir: str | Path, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        src_dir = Path(src_dir)

        ids = []
        with zipfile.ZipFile(src_dir / 'amos22.zip') as zf:
            for name in zf.namelist():
                if name.startswith('amos22/imagesVa/amos_') and name.endswith('.nii.gz'):
                    ids.append(name[len('amos22/imagesVa/amos_'):-len('.nii.gz')])
        # int(i) <= 500 - CT ids
        ids = [id_ for id_ in ids if int(id_) <= 500 and id_ not in DROP_IDS]

        def prepare(id_: str) -> None:
            image, affine = _load_nii_from_zip(src_dir, 'amos22.zip', f'amos22/imagesVa/amos_{id_}.nii.gz')
            mask, mask_affine = _load_nii_from_zip(src_dir, 'amos22.zip', f'amos22/labelsVa/amos_{id_}.nii.gz')
            voxel_spacing = affine_to_voxel_spacing(affine)
            image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)
            mask, _ = to_canonical_orientation(mask, None, mask_affine)

            data_dir = self.root_dir / id_
            data_dir.mkdir()
            save_numpy(image.astype('int16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)
            save_json(voxel_spacing, data_dir / 'voxel_spacing.json')
            save_numpy(mask.astype('uint8'), data_dir / 'mask.npy.gz', compression=1, timestamp=0)

        with ThreadPool(num_workers) as p:
            _ = list(tqdm(p.imap(prepare, ids), total=len(ids)))


class AMOSCTUnlabeledTrain(Dataset):
    name = 'amos_ct_unlabeled_train'

    def prepare(self, src_dir: str | Path, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        src_dir = Path(src_dir)

        ids = []
        for filename in [
                'amos22_unlabeled_ct_5000_5399.zip',
                'amos22_unlabeled_ct_5400_5899.zip',
                'amos22_unlabeled_ct_5900_6199.zip',
                'amos22_unlabeled_ct_6200_6899.zip',
        ]:
            with zipfile.ZipFile(src_dir / filename) as zf:
                for name in zf.namelist():
                    if name.endswith('.nii.gz'):
                        ids.append(name.split('/')[1][len('amos_'):-len('.nii.gz')])
        ids = [i for i in ids if i not in DROP_IDS]

        def prepare(id_: str) -> None:
            if 5000 <= int(id_) < 5400:
                zip_file = 'amos22_unlabeled_ct_5000_5399.zip'
                image_file = f'amos_unlabeled_ct_5000_5399/amos_{id_}.nii.gz'
            elif 5400 <= int(id_) < 5900:
                zip_file = 'amos22_unlabeled_ct_5400_5899.zip'
                image_file = f'amos_unlabeled_ct_5400_5899/amos_{id_}.nii.gz'
            elif 5900 <= int(id_) < 6200:
                zip_file = 'amos22_unlabeled_ct_5900_6199.zip'
                image_file = f'amos22_unlabeled_ct_5900_6199/amos_{id_}.nii.gz'
            else:
                zip_file = 'amos22_unlabeled_ct_6200_6899.zip'
                image_file = f'amos22_unlabeled_6200_6899/amos_{id_}.nii.gz'

            image, affine = _load_nii_from_zip(src_dir, zip_file, image_file)
            voxel_spacing = affine_to_voxel_spacing(affine)
            image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)

            data_dir = self.root_dir / id_
            data_dir.mkdir()
            save_numpy(image.astype('int16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)
            save_json(voxel_spacing, data_dir / 'voxel_spacing.json')

        with ThreadPool(num_workers) as p:
            _ = list(tqdm(p.imap(prepare, ids), total=len(ids)))


def _load_nii_from_zip(src_dir: Path, zip_file: str, nii_file: str) -> Tuple[np.ndarray, np.ndarray]:
    with zipfile.Path(src_dir / zip_file, nii_file).open('rb') as nii_gz_f:
        with gzip.GzipFile(fileobj=nii_gz_f) as nii_f:
            fh = nibabel.FileHolder(fileobj=nii_f)
            nii = nibabel.Nifti1Image.from_file_map({'header': fh, 'image': fh})
            nii = nibabel.as_closest_canonical(nii)
            image = nii.get_fdata()
            affine = nii.affine

    return image, affine
