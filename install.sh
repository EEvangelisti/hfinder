#!/usr/bin/env bash
set -euo pipefail

#python3 -m venv hfinder-venv
# shellcheck source=/dev/null
#source hfinder-venv/bin/activate

#python -m pip install --upgrade pip setuptools wheel
#python -m pip install pyyaml ultralytics tifffile scikit-image

#deactivate || true

echo "✅ HFinder is now installed"

SCRIPTS_DIR="$(pwd)/scripts"

read -rp "👉 Do you want to add '$SCRIPTS_DIR' to your PATH in ~/.bashrc? [y/N] " reply
if [[ "$reply" =~ ^[Yy]$ ]]; then
    if ! grep -qF "$SCRIPTS_DIR" "$HOME/.bashrc"; then
        echo "" >> "$HOME/.bashrc"
        echo "# Added by HFinder installer" >> "$HOME/.bashrc"
        echo "export PATH=\"$SCRIPTS_DIR:\$PATH\"" >> "$HOME/.bashrc"
        echo "✅ Added $SCRIPTS_DIR to your PATH in ~/.bashrc"
        echo "ℹ️  Run 'source ~/.bashrc' or open a new terminal to use 'hfinder' and other scripts"
    else
        echo "ℹ️  $SCRIPTS_DIR is already in your PATH (via ~/.bashrc)"
    fi
else
    echo "ℹ️  Skipped PATH modification. You can add it manually with:"
    echo "    export PATH=\"$SCRIPTS_DIR:\$PATH\""
fi
