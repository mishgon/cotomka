from pathlib import Path
from tqdm.auto import tqdm
import pydicom
import mdai
import numpy as np
import pandas as pd
from skimage.draw import polygon
import warnings

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.dicom import (
    get_series_uid, Plane, get_series_slice_plane, drop_duplicated_slices, order_series,
    get_series_image, get_series_voxel_spacing, get_series_orientation_matrix,
    to_canonical_orientation
)
from cotomka.utils.io import save_numpy, save_json, load_numpy


LABELS = [
    'Atelectasis',
    'Infectious cavity',
    'Infectious opacity',
    'Infectious TIB/micronodules',
    'Noninfectious nodule/mass',
    'Other noninfectious opacity'
]


class MIDRCRICORD1A(Dataset):
    name = 'midrc_ricord_1a'

    def _load_mask(self, index: str):
        return load_numpy(self.root_dir / index / 'mask.npy.gz', decompress=True)

    def prepare(self, src_dir: Path):
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        src_dir = Path(src_dir)

        anns: pd.DataFrame = mdai.common_utils.json_to_dataframe(
            src_dir / 'MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
        )['annotations']
        anns = anns[anns['scope'] == 'INSTANCE']  # drop study-level annotations, i.e. labels

        for series_uid, series_anns in tqdm(anns.groupby('SeriesInstanceUID')):
            # load series from DICOMs
            series = list(map(pydicom.dcmread, src_dir.glob(f'**/{series_uid}/*.dcm')))

            # extract image, voxel spacing and orientation matrix from dicoms
            # drop non-axial series and series with invalid tags
            series_uid = get_series_uid(series)
            try:
                if get_series_slice_plane(series) != Plane.Axial:
                    raise ValueError('Series is not axial')

                series = drop_duplicated_slices(series)
                series = order_series(series)

                image = get_series_image(series)
                voxel_spacing = get_series_voxel_spacing(series)
                om = get_series_orientation_matrix(series)

                sop_instance_uids = [i.SOPInstanceUID for i in series]
            except (AttributeError, ValueError, NotImplementedError) as e:
                warnings.warn(f'Series {series_uid} fails with {e.__class__.__name__}: {str(e)}')
                continue
            
            # to canonical orientation
            image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, om)

            # create multiclass mask from annotations
            mask = np.zeros(image.shape, dtype='uint8')
            for label, label_anns in series_anns.groupby('labelName'):
                for _, ann in label_anns.iterrows():
                    slice_index = sop_instance_uids.index(ann['SOPInstanceUID'])
                    if ann['data'] is None:
                        warnings.warn(f'{label} annotations for series {series_uid} contains None for slice {slice_index}.')
                        continue
                    vertices = np.array(ann['data']['vertices'])
                    mask[(*polygon(vertices[:, 1], vertices[:, 0], image.shape[:2]), slice_index)] = LABELS.index(label) + 1

            # to canonical orientation
            mask, _ = to_canonical_orientation(mask, None, om)

            save_dirpath = self.root_dir / series_uid
            save_dirpath.mkdir()
            save_numpy(image.astype('int16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
            save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
            save_numpy(mask, save_dirpath / 'mask.npy.gz', compression=1, timestamp=0)
