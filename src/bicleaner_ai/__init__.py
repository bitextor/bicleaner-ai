#!/usr/bin/env python
import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
from importlib.metadata import version

name = "bicleaner_ai"
__version__ = version(name)
