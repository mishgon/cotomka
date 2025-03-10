from enum import Enum
from typing import NamedTuple, Tuple, List
import warnings
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
import numpy as np

import pylidc as pl
from pylidc.Scan import ClusterError

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.dicom import (
    get_series_uid, Plane, get_series_slice_plane, drop_duplicated_slices, order_series,
    get_series_image, get_series_voxel_spacing, get_series_orientation_matrix,
    to_canonical_orientation
)
from cotomka.utils.io import save_numpy, save_json, load_numpy, load_json


class Calcification(Enum):
    Popcorn, Laminated, Solid, NonCentral, Central, Absent = 1, 2, 3, 4, 5, 6


class InternalStructure(Enum):
    SoftTissue, Fluid, Fat, Air = 1, 2, 3, 4


class Lobulation(Enum):
    NoLobulation, NearlyNoLobulation, MediumLobulation, NearMarkedLobulation, MarkedLobulation = 1, 2, 3, 4, 5


class Malignancy(Enum):
    HighlyUnlikely, ModeratelyUnlikely, Indeterminate, ModeratelySuspicious, HighlySuspicious = 1, 2, 3, 4, 5


class Sphericity(Enum):
    Linear, OvoidLinear, Ovoid, OvoidRound, Round = 1, 2, 3, 4, 5


class Spiculation(Enum):
    NoSpiculation, NearlyNoSpiculation, MediumSpiculation, NearMarkedSpiculation, MarkedSpiculation = 1, 2, 3, 4, 5


class Subtlety(Enum):
    ExtremelySubtle, ModeratelySubtle, FairlySubtle, ModeratelyObvious, Obvious = 1, 2, 3, 4, 5


class Texture(Enum):
    NonSolidGGO, NonSolidMixed, PartSolidMixed, SolidMixed, Solid = 1, 2, 3, 4, 5


class Nodule(NamedTuple):
    contours: List[Tuple[int, List[Tuple[int, int]]]]
    bbox: np.ndarray  # (2, 3)
    diameter_mm: float
    surface_area_mm2: float
    volume_mm3: float
    calcification: Calcification = None
    internal_structure: InternalStructure = None
    lobulation: Lobulation = None
    malignancy: Malignancy = None
    sphericity: Sphericity = None
    spiculation: Spiculation = None
    subtlety: Subtlety = None
    texture: Texture = None

    def to_json(self) -> dict:
        j = {
            'contours': self.contours,
            'bbox': self.bbox.tolist(),
            'diameter_mm': self.diameter_mm,
            'surface_area_mm2': self.surface_area_mm2,
            'volume_mm3': self.volume_mm3,
        }
        if self.calcification is not None:
            j['calcification'] = self.calcification.value
        if self.internal_structure is not None:
            j['internal_structure'] = self.internal_structure.value
        if self.lobulation is not None:
            j['lobulation'] = self.lobulation.value
        if self.malignancy is not None:
            j['malignancy'] = self.malignancy.value
        if self.sphericity is not None:
            j['sphericity'] = self.sphericity.value
        if self.spiculation is not None:
            j['spiculation'] = self.spiculation.value
        if self.subtlety is not None:
            j['subtlety'] = self.subtlety.value
        if self.texture is not None:
            j['texture'] = self.texture.value

        return j

    @classmethod
    def from_json(cls, j: dict) -> 'Nodule':
        kwargs = dict(
            contours=[(c[0], list(map(tuple, c[1]))) for c in j['contours']],
            bbox=np.array(j['bbox']),
            diameter_mm=j['diameter_mm'],
            surface_area_mm2=j['surface_area_mm2'],
            volume_mm3=j['volume_mm3'],
        )
        if 'calcification' in j:
            kwargs['calcification'] = Calcification(j['calcification'])
        if 'internal_structure' in j:
            kwargs['internal_structure'] = InternalStructure(j['internal_structure'])
        if 'lobulation' in j:
            kwargs['lobulation'] = Lobulation(j['lobulation'])
        if 'malignancy' in j:
            kwargs['malignancy'] = Malignancy(j['malignancy'])
        if 'sphericity' in j:
            kwargs['sphericity'] = Sphericity(j['sphericity'])
        if 'spiculation' in j:
            kwargs['spiculation'] = Spiculation(j['spiculation'])
        if 'subtlety' in j:
            kwargs['subtlety'] = Subtlety(j['subtlety'])
        if 'texture' in j:
            kwargs['texture'] = Texture(j['texture'])
        
        return cls(**kwargs)


class LIDC(Dataset):
    name = 'lidc'

    def _get_image(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / id / 'image.npy.gz', decompress=True).astype('float32')

    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        return tuple(load_json(self.root_dir / id / 'voxel_spacing.json'))

    def _get_nodules(self, id: str) -> List[List[Nodule]]:
        return [[Nodule.from_json(n) for n in nodule] for nodule in load_json(self.root_dir / id / 'nodules.json')]

    def prepare(self, num_workers: int = 1):
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        scans = pl.query(pl.Scan).all()

        with ThreadPool(num_workers) as p:
            _ = list(tqdm(p.imap(self._prepare, scans), total=len(scans)))

    def _prepare(self, scan: pl.Scan):
        # read series
        series = scan.load_all_dicom_images(verbose=False)

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
        except (AttributeError, ValueError, NotImplementedError) as e:
            warnings.warn(f'Series {series_uid} fails with {e.__class__.__name__}: {str(e)}')
            return

        # to canonical orientation
        image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, om)

        try:
            nodules = [
                [annotation_to_nodule(ann, image_size=image.shape) for ann in anns]
                for anns in scan.cluster_annotations()
            ]
        except ClusterError as e:
            warnings.warn(f'Series {series_uid} fails with {e.__class__.__name__}: {str(e)}')
            return

        data_dir = self.root_dir / series_uid
        data_dir.mkdir()
        save_numpy(image.astype('int16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)
        save_json(voxel_spacing, data_dir / 'voxel_spacing.json')
        save_json([[n.to_json() for n in nodule] for nodule in nodules], data_dir / 'nodules.json')


def annotation_to_nodule(annotation: pl.Annotation, image_size: Tuple[int, int, int]) -> Nodule:
    bbox = annotation.bbox()
    return Nodule(
        contours=[
            (int(image_size[2] - 1 - c.image_k_position), list(map(tuple, c.to_matrix(include_k=False).tolist())))
            for c in annotation.contours
        ],
        bbox=np.array([
            [bbox[0].start, bbox[1].start, image_size[2] - bbox[2].stop],
            [bbox[0].stop, bbox[1].stop, image_size[2] - bbox[2].start],
        ]),
        diameter_mm=float(annotation.diameter),
        surface_area_mm2=float(annotation.surface_area),
        volume_mm3=float(annotation.volume),
        calcification=_enum_or_none(Calcification, annotation.calcification),
        internal_structure=_enum_or_none(InternalStructure, annotation.internalStructure),
        lobulation=_enum_or_none(Lobulation, annotation.lobulation),
        malignancy=_enum_or_none(Malignancy, annotation.malignancy),
        sphericity=_enum_or_none(Sphericity, annotation.sphericity),
        spiculation=_enum_or_none(Spiculation, annotation.spiculation),
        subtlety=_enum_or_none(Subtlety, annotation.subtlety),
        texture=_enum_or_none(Texture, annotation.texture),
    )


def _enum_or_none(cls, value):
    try:
        return cls(value)
    except ValueError:
        return
