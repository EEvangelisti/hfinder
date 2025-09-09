#! /usr/bin/env bash

set -euo pipefail

python3 -m venv hfinder-venv
source hfinder-venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install pyyaml ultralytics tifffile scikit-image

deactivate

echo "✅ Virtual env created: hfinder-venv
To activate:  source hfinder-venv/bin/activate"
