#!/usr/bin/env python
from importlib.metadata import version, PackageNotFoundError

name = "bicleaner_ai"

# FIX: Handle PackageNotFoundError for cases when module is loaded via
# PYTHONPATH without pip install (e.g., Docker inference service).
# Previously this would crash with an unhandled exception.
try:
    __version__ = version(name)
except PackageNotFoundError:
    __version__ = "3.4"  # Fallback version for Docker deployment
