#!/usr/bin/env python
from importlib.metadata import version, PackageNotFoundError

name = "bicleaner_ai"

try:
    __version__ = version(name)
except PackageNotFoundError:
    __version__ = "3.4"
