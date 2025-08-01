import argparse

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    return parser.parse_args()


def main(args):
    snapshot_download(
        repo_id='ibrahimhamamci/CT-RATE',
        repo_type='dataset',
        revision='d8fe2952748813799042cec9459ba12a99caab77',
        etag_timeout=60,
        token=args.token,
        max_workers=32
    )


if __name__ == '__main__':
    main(parse_args())
