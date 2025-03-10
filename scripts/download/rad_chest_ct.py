import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
import pandas as pd


SAVE_DIR = Path('<PATH_TO_SAVE_DIR>')
TOKEN = '<YOUR_TOKEN>'
NUM_THREADS = 16


def download(name) -> None:
    os.system(f'wget https://zenodo.org/records/6406114/files/{name}.npz?token={TOKEN} -O /disk/data1/RAD-ChestCT/{name}.npz > /dev/null 2>&1')


def main():
    df = pd.read_csv(SAVE_DIR / 'Summary_3630.csv')
    names = list(df['NoteAcc_DEID'])

    with ThreadPool(NUM_THREADS) as p:
        _ = list(tqdm(p.imap(download, names), total=len(names)))


if __name__ == '__main__':
    main()
