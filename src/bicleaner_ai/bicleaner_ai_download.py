#!/usr/bin/env python
from tempfile import NamedTemporaryFile
from argparse import ArgumentParser
import tarfile
import logging
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError
from requests import get


GITHUB_URL = "https://github.com/bitextor/bicleaner-ai-data/releases/latest/download"


def logging_setup(args):
    logger = logging.getLogger()
    logger.handlers = []
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)


def main():
    parser = ArgumentParser(
            description='Download Bicleaner AI models from the HuggingFace Hub or GitHub')
    parser.add_argument('src_lang', type=str,
                        help='Source language')
    parser.add_argument('trg_lang', type=str,
                        help='Target language')
    parser.add_argument('model_type', type=str, choices=['full', 'lite'],
                        help='Download lite or full model')
    parser.add_argument('download_path', type=str, nargs='?',
                        help='Path to download the model (only for old Github models)')
    parser.add_argument('-t', '--auth_token', default=None, type=str,
                        help='Authentication token for private models downloading')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    logging_setup(args)

    if not args.download_path and args.model_type == 'lite':
        raise Exception("Lite models need a download path")

    if args.model_type == 'full':
        #TODO fallback to github if does not exist?
        name = f'bitextor/bicleaner-ai-full-{args.src_lang}-{args.trg_lang}'
        logging.info(f'Downloading {name}')
        try:
            snapshot_download(name, use_auth_token=args.auth_token, etag_timeout=100)
            return
        except RepositoryNotFoundError:
            if not args.download_path:
                logging.error(f"Model repository {name} not found at HuggingFace," \
                              + "please provide a path to download old model from Github")
            logging.warning(f"Model repository {name} not found, trying Github old models...")

    # Download from github
    url = f'{GITHUB_URL}/{args.model_type}-{args.src_lang}-{args.trg_lang}.tgz'
    logging.info(f"Trying {url}")
    response = get(url, allow_redirects=True, stream=True)
    if response.status_code == 404:
        response.close()
        logging.warning(f"{args.src_lang}-{args.trg_lang} language pack does not exist" \
                        + f" trying {args.trg_lang}-{args.src_lang}...")
        url = f'{GITHUB_URL}/{args.model_type}-{args.trg_lang}-{args.src_lang}.tgz'
        response = get(url, allow_redirects=True, stream=True)

        if response.status_code == 404:
            response.close()
            logging.error(f"{args.trg_lang}-{args.src_lang} language pack does not exist")
            sys.exit(1)

    # Write the tgz to temp and extract to desired path
    with NamedTemporaryFile() as temp:
        logging.info("Downloading file")
        with open(temp.name, mode='wb') as f:
            f.writelines(response.iter_content(1024))
        response.close()

        logging.info(f"Extracting tar.gz file to {args.download_path}")
        with tarfile.open(temp.name) as f:
            f.extractall(args.download_path)


if __name__ == "__main__":
    main()
