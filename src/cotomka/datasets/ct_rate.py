from typing import Tuple, Dict
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm
from functools import cached_property
import gzip
import nibabel
import math
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.dicom import orientation_matrix_to_slice_plane, Plane
from cotomka.utils.io import save_numpy, save_json, load_numpy


REPO_ID = 'ibrahimhamamci/CT-RATE'


class _CTRATE(Dataset):
    _split: str

    @cached_property
    def labels_df(self):
        return pd.read_csv(self.root_dir / 'labels.csv').set_index('VolumeName')

    @cached_property
    def metadata_df(self):
        return pd.read_csv(self.root_dir / 'metadata.csv').set_index('VolumeName')

    @cached_property
    def reports_df(self):
        return pd.read_csv(self.root_dir / 'reports.csv').set_index('VolumeName')

    @cached_property
    def ids(self) -> Tuple[str]:
        return tuple(sorted(file.name[:-len('.npy.gz')] for file in self.root_dir.glob('*.npy.gz')))

    def _get_image(self, index: str) -> np.ndarray:
        return load_numpy(self.root_dir / f'{index}.npy.gz', decompress=True).astype('float32')

    def _get_voxel_spacing(self, index: str) -> Tuple[float, float, float]:
        return (
            *map(float, self.metadata_df.loc[index, 'XYSpacing'][1:-1].split(', ')),
            float(self.metadata_df.loc[index, 'ZSpacing'])
        )

    def _get_study_data(self, index: str) -> str:
        return self.metadata_df.loc[index, 'StudyDate']

    def _get_patient_sex(self, index: str) -> str:
        return self.metadata_df.loc[index, 'PatientSex']

    def _get_patient_age(self, index: str) -> str:
        return self.metadata_df.loc[index, 'PatientAge']

    def _get_technique(self, index: str) -> str:
        return self.reports_df.loc[index, 'Technique_EN']

    def _get_findings(self, index: str) -> str:
        return self.reports_df.loc[index, 'Findings_EN']

    def _get_impression(self, index: str) -> str:
        return self.reports_df.loc[index, 'Impressions_EN']

    def _get_labels(self, index: str) -> Dict[str, int]:
        return self.labels_df.loc[index].to_dict()

    def prepare(self, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        load_dataset(REPO_ID, 'labels', split=self._split).to_pandas().to_csv(self.root_dir / 'labels.csv', index=False)
        load_dataset(REPO_ID, 'metadata', split=self._split).to_pandas().to_csv(self.root_dir / 'metadata.csv', index=False)
        load_dataset(REPO_ID, 'reports', split=self._split).to_pandas().to_csv(self.root_dir / 'reports.csv', index=False)

        index = self.labels_df.index
        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(self._prepare_image, index), total=len(index)))

    def _prepare_image(self, index: str) -> None:
        # drop images with undefined slice spacing
        if math.isnan(self.metadata_df.loc[index, 'ZSpacing']):
            return

        # drop series with non-canonical orientation
        image_orientation_patient = self.metadata_df.loc[index, 'ImageOrientationPatient']
        image_orientation_patient = np.array(list(map(float, image_orientation_patient[1:-1].split(', '))))
        row, col = image_orientation_patient.reshape(2, 3)
        orientation_matrix = np.stack([row, col, np.cross(row, col)])
        if not np.allclose(orientation_matrix, np.eye(3)):
            return

        image_file = _find_nii_gz(index)
        image = _load_nii_gz(image_file)
        image = image * self.metadata_df.loc[index, 'RescaleSlope']
        image = image + self.metadata_df.loc[index, 'RescaleIntercept']
        image = np.swapaxes(image, 0, 1)[:, :, ::-1].copy()
        save_numpy(image.astype('int16'), self.root_dir / f'{index}.npy.gz', compression=1, timestamp=0)


class CTRATETrain(_CTRATE):
    name = 'ct_rate_train'
    _split = 'train'


class CTRATEVal(_CTRATE):
    name = 'ct_rate_val'
    _split = 'validation'


def _load_nii_gz(nii_gz_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    with gzip.GzipFile(filename=nii_gz_file) as nii_f:
        fh = nibabel.FileHolder(fileobj=nii_f)
        nii = nibabel.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        nii = nibabel.as_closest_canonical(nii)
        image = nii.get_fdata()

    return image


def _find_nii_gz(volume_name: str) -> Path:
    folder_1, folder_2, folder_3, _ = volume_name.split('_')
    folder_2 = folder_1 + '_' + folder_2
    folder_3 = folder_2 + '_' + folder_3
    subfolder = f'dataset/{folder_1}/{folder_2}/{folder_3}'

    return Path(hf_hub_download(repo_id=REPO_ID, repo_type='dataset', subfolder=subfolder, filename=volume_name))
