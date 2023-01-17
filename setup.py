#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt") as rf:
    requirements = rf.read().splitlines()
    
setuptools.setup(
    name="bicleaner-ai",
    install_requires=requirements,
    extras_require={ "transliterate" : ["cyrtranslit==1.1"] },
    license="GNU General Public License v3.0",
    author="Prompsit Language Engineering",
    author_email="info@prompsit.com",
    maintainer="Jaume Zaragoza",
    maintainer_email="jzaragoza@prompsit.com",
    description="Parallel corpus classifier, indicating the likelihood of a pair of sentences being mutual translations or not (neural version)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bitextor/bicleaner-ai",
    packages=setuptools.find_packages(),
    package_data={"bicleaner_ai":["../requirements.txt"]},
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Filters"
    ],
    project_urls={
        "Bicleaner on GitHub": "https://github.com/bitextor/bicleaner",
        "Prompsit Language Engineering": "http://www.prompsit.com",
        "Paracrawl": "https://paracrawl.eu/"
    },
    entry_points={
        "console_scripts": [
            "bicleaner-ai-train=bicleaner_ai.bicleaner_ai_train:main",
            "bicleaner-ai-classify=bicleaner_ai.bicleaner_ai_classifier:main",
        ]
    },
    scripts=[
         "scripts/bicleaner-ai-download",
         "scripts/bicleaner-ai-download-hf",
     ]
)
