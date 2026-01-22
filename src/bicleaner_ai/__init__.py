#!/usr/bin/env python
from importlib.metadata import version, PackageNotFoundError

name = "bicleaner_ai"

try:
    __version__ = version(name)
except PackageNotFoundError:
    # Running via PYTHONPATH without pip install (e.g., Docker)
    __version__ = "3.4"
