#!/usr/bin/env bash
set -e

python --version
pip install --upgrade "pip>=23.1" "setuptools>=68.0.0" "wheel>=0.41.2"
# Turn off PEP 517 isolated builds so pip doesn't pull an ancient setuptools for sdists
export PIP_NO_BUILD_ISOLATION=1
pip install -r requirements.txt
