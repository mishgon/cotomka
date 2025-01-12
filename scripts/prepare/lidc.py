from cotomka.datasets import *


def main():
    LIDC().prepare(num_workers=16)


if __name__ == '__main__':
    main()
