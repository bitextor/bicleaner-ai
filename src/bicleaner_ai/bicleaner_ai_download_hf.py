#!/usr/bin/env python
from huggingface_hub import snapshot_download
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description='Download Bicleaner AI full models from the Hugging Face Hub')
    parser.add_argument('model', type=str, help='Hugging Face Bicleaner AI model identifier (e.g. "bitextor/bicleaner-ai-full-en-fr")')
    parser.add_argument('-t', '--auth_token', default=None, type=str, help='Authentication token for private models downloading')

    args = parser.parse_args()

    snapshot_download(args.model, use_auth_token=args.auth_token, etag_timeout=100)

if __name__ == "__main__":
    main()
