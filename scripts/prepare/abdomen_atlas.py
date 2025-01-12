from cotomka.datasets import *


def main():
    AbdomenAtlas().prepare(
        src_dir='/home/jovyan/misha/source_data/abdomen_atlas_1.0_mini',
        num_workers=16
    )


if __name__ == '__main__':
    main()
