from cotomka.datasets import *


def main():
    NLST().prepare(
        src_dir='/home/jovyan/misha/source_data/nlst',
        num_workers=16
    )


if __name__ == '__main__':
    main()
