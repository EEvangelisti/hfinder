#!/usr/bin/env bash
set -euo pipefail

python3 -m venv hfinder-venv
# shellcheck source=/dev/null
source hfinder-venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install pyyaml ultralytics tifffile scikit-image

echo "✅ HFinder is now installed"

# Check PyTorch installation
if python -c "import torch" 2>/dev/null; then
    echo "ℹ️  PyTorch is installed"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "✅ PyTorch has CUDA support"
    else
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "⚠️  PyTorch is CPU-only, but a CUDA GPU was detected."
            echo "👉 For best performance, install a CUDA-enabled PyTorch build, e.g.:"
            echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        else
            echo "ℹ️  CPU-only PyTorch is fine (no NVIDIA GPU detected)."
        fi
    fi
else
    # Very unlikely: ultralytics should have installed torch automatically
    echo "⚠️  PyTorch is not installed."
    echo "👉 Please install PyTorch manually following: https://pytorch.org/get-started/locally/"
fi

deactivate || true

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
