import argparse

from cotomka.datasets import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=8, type=int)
    return parser.parse_args()


def main(args):
    CTRATETrain().prepare(num_workers=args.num_workers)
    CTRATEVal().prepare(num_workers=args.num_workers)


if __name__ == '__main__':
    main(parse_args())
