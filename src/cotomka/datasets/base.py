import yaml
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple
from pathlib import Path
import numpy as np

from cotomka.utils.io import load_numpy, load_json


class Dataset(ABC):
    name: str

    def __init__(self) -> None:
        self.root_dir = COTOMKA_ROOT_DIR / self.name

    @property
    def ids(self) -> Tuple[str]:
        if not self.root_dir.exists():
            raise OSError('Dataset is not prepared on this OS')

        return tuple(sorted(data_dir.name for data_dir in self.root_dir.iterdir()))

    @property
    def fields(self) -> Tuple[str]:
        return tuple(sorted(name[len('_get_'):] for name in dir(self) if name.startswith('_get_')))

    def get(self, id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.root_dir.exists():
            raise OSError('Dataset is not prepared on this OS')

        if fields is None:
            fields = self.fields

        missing_fields = list(set(fields) - set(self.fields))
        if missing_fields:
            raise ValueError(f'Dataset does not contain fields {missing_fields}')

        data = {f: getattr(self, f'_get_{f}')(id) for f in fields}
        data['id'] = id
        return data

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:
        raise NotImplementedError


def get_cotomka_root_dir() -> Path:
    config_file = Path('~/.config/cotomka/cotomka.yaml').expanduser()

    if not config_file.exists():
        raise FileNotFoundError('Please create a config file ~/.config/cotomka/cotomka.yaml (see README)')

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return Path(config['root_dir']).expanduser()


COTOMKA_ROOT_DIR = get_cotomka_root_dir()
