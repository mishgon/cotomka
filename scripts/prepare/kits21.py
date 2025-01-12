from cotomka.datasets import *


def main():
    KiTS21().prepare(
        src_dir='/home/jovyan/misha/source_data/kits21',
        num_workers=16
    )


if __name__ == '__main__':
    main()
