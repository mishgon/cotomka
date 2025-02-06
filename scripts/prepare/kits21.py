import argparse

from cotomka.datasets import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True)
    parser.add_argument('--num_workers', default=8, type=int)
    return parser.parse_args()


def main(args):
    KiTS21().prepare(
        src_dir=args.src_dir,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main(parse_args())
