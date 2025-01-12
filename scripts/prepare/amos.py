from cotomka.datasets import *


def main():
    AMOSCTLabeledTrain().prepare(
        src_dir='/home/jovyan/misha/source_data/amos',
        num_workers=16
    )
    AMOSCTVal().prepare(
        src_dir='/home/jovyan/misha/source_data/amos',
        num_workers=16
    )
    AMOSCTUnlabeledTrain().prepare(
        src_dir='/home/jovyan/misha/source_data/amos',
        num_workers=16
    )


if __name__ == '__main__':
    main()
