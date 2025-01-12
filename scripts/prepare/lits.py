from cotomka.datasets import *


def main():
    LiTS().prepare(
        src_dir='/home/jovyan/misha/source_data/lits',
        num_workers=16
    )


if __name__ == '__main__':
    main()
