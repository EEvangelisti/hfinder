# Installation Guide

## Requirements
- Python **3.9+** (3.10–3.12 recommended)  
- Linux, macOS, or Windows  

We recommend using a **virtual environment** to avoid conflicts with system packages.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv hfinder
source hfinder/bin/activate  # On Windows: hfinder\Scripts\activate

# 2. Install HFinder
pip install .

# 3. Verify the installation
hfinder --help
annot2images --help
```



## GPU / CPU Support

HFinder depends on **PyTorch** through `ultralytics`.  
By default, `pip` will install the most suitable version for your system:

- 💻 If a **CUDA-compatible GPU** is detected → the **GPU build** of PyTorch will be installed.  
- 🖥️ Otherwise → the **CPU-only build** will be installed.  

### Forcing CPU-only

If you prefer a lighter installation (no CUDA libraries), run:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

That’s it! You’re ready to use HFinder.



## Uninstallation

To remove HFinder from your environment:
```bash
pip uninstall hfinder
```

