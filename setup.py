#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="hear_ced",
    version="0.0.1",
    description="Conssitent Ensemble Distillation Transformers for Audio tagging",
    author="Heinrich Dinkel",
    author_email="dinkelheinrich@xiaomi.com",
    url="https://github.com/Richermans/CED",
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    project_urls={
        "Source Code": "https://github.com/RicherMans/HEAR_CED",
    },
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[]
)
