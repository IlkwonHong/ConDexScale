# Installation Guide

Environment: Python 3.11, CUDA 12.4, Ubuntu (Docker container)

---

## Quick Setup

```bash
# 0. Create and activate venv
python3.11 -m venv venv
source venv/bin/activate

# 1. Fix SSH submodule URLs (required before pip install)
git config --global url."https://github.com/".insteadOf "git@github.com:"

# 2. Pre-install wheel and chumpy (avoids build failure)
pip install wheel
pip install chumpy --no-build-isolation

# 3. Patch chumpy for Python 3.11 (inspect.getargspec removed)
sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' \
    venv/lib/python3.11/site-packages/chumpy/ch.py

# 4. Install dex-ycb-toolkit with __init__.py fix (see Known Issue #6)
git clone https://github.com/NVlabs/dex-ycb-toolkit.git /tmp/dex-ycb-toolkit
touch /tmp/dex-ycb-toolkit/dex_ycb_toolkit/__init__.py
pip install /tmp/dex-ycb-toolkit

# 5. Install PyTorch with matching CUDA version
#    (requirements.txt pins torch==1.13.1 which is outdated — use the version below)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 6. Install pytorch3d from source (not on PyPI)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 7. Install remaining dependencies
pip install tensorboard loguru pybullet mujoco==2.3.7 fire lmdb kornia hdbscan urdf-parser-py \
    pandas scipy scikit-learn open3d trimesh

# 8. Install the project and submodules
pip install -e .
pip install -e network/manopth
pip install -e thirdparty/pytorch_kinematics
pip install -e thirdparty/pytorch_volumetric
```

---

## Known Issues and Workarounds

### 1. SSH permission denied when cloning submodules

**Error:**
```
git@github.com: Permission denied (publickey).
fatal: clone of 'git@github.com:ywchao/bop_toolkit.git' failed
```

**Cause:** `dex-ycb-toolkit`'s `.gitmodules` uses SSH URLs (`git@github.com:...`), but no SSH key is configured in the container.

**Fix:**
```bash
git config --global url."https://github.com/".insteadOf "git@github.com:"
```

---

### 2. `chumpy` fails to build

**Error:**
```
ModuleNotFoundError: No module named 'pip'
error: invalid command 'bdist_wheel'
```

**Cause:** `chumpy 0.70` has an outdated `setup.py` that imports `pip` directly (removed in newer pip) and requires `wheel` to be installed explicitly.

**Fix:**
```bash
pip install wheel
pip install chumpy --no-build-isolation
```

---

### 3. `torchvision==0.14.1` not installable

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torchvision==0.14.1
```

**Cause:** `torchvision==0.14.1` (which pairs with torch 1.13.1) has been yanked/removed from PyPI.

**Fix:** Install from PyTorch's own index, matching the CUDA version on your system:
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

Available CUDA variants: `cu118`, `cu121`, `cu124`. Check `nvidia-smi` for your driver and use `nvcc --version` for the toolkit version. **The `nvcc` version determines which variant to use.**

---

### 4. `numpy==1.21.6` incompatible with Python 3.11

**Error:**
```
ERROR: Ignored the following versions that require a different python version:
    1.21.6 Requires-Python >=3.7,<3.11
```

**Cause:** NumPy 1.21.x only supports Python < 3.11. The pin in `requirements.txt` is stale.

**Fix:** Skip the pin — let pip resolve. NumPy 2.4.4 is installed automatically as a dependency and works correctly.

---

### 6. `dex_ycb_toolkit` installs as empty package from GitHub

**Error:**
```
ModuleNotFoundError: No module named 'dex_ycb_toolkit'
```
even after `pip install "git+https://github.com/NVlabs/dex-ycb-toolkit.git@master"` succeeds.

**Cause:** `dex_ycb_toolkit/` has no `__init__.py`, so `find_packages()` in `setup.py` skips it entirely. The wheel is built with only dist-info metadata and no Python files.

**Fix:** Clone the repo locally, add an empty `__init__.py`, then install from the local path:
```bash
git clone https://github.com/NVlabs/dex-ycb-toolkit.git /tmp/dex-ycb-toolkit
touch /tmp/dex-ycb-toolkit/dex_ycb_toolkit/__init__.py
pip install /tmp/dex-ycb-toolkit
```

---

### 7. `chumpy` crashes on import under Python 3.11

**Error (during `pip install -e network/manopth`):**
```
AttributeError: module 'inspect' has no attribute 'getargspec'. Did you mean: 'getargs'?
```

**Cause:** `inspect.getargspec` was removed in Python 3.11. `chumpy 0.70` still uses it in `ch.py`.

**Fix:** After installing chumpy, patch the installed file:
```bash
sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' \
    venv/lib/python3.11/site-packages/chumpy/ch.py
```

---

### 8. venv activation broken after folder rename

**Symptom:** After renaming the project directory, `source venv/bin/activate` appears to work but `which python` still points to `/usr/bin/python` (system Python). All subsequent `pip install` calls go to the system Python instead of the venv.

**Cause:** `venv/bin/activate` (and other scripts in `venv/bin/`) hardcode the absolute path of the venv at creation time. Renaming the parent directory breaks those paths.

**Fix:** Replace the old path in all affected scripts:
```bash
OLD="<old absolute path>/venv"
NEW="<new absolute path>/venv"
grep -rl "$OLD" venv/bin/ | xargs sed -i "s|$OLD|$NEW|g"
```
Then verify: `source venv/bin/activate && which python` should print `<new path>/venv/bin/python`.

---

### 5. `pytorch3d` not on PyPI / CUDA version mismatch

**Error (from PyPI):**
```
ERROR: No matching distribution found for pytorch3d
```

**Error (during source build):**
```
RuntimeError: The detected CUDA version (12.4) mismatches the version
used to compile PyTorch (13.0).
```

**Cause:**
- `pytorch3d` is not distributed on PyPI; it must be built from source.
- If pip auto-selects `torch+cu130` (latest) but your toolkit is CUDA 12.4, compilation fails.

**Fix:** Downgrade torch to a cu124 build first, then build pytorch3d:
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

The source build takes ~10–20 minutes as it compiles CUDA kernels.

---

## Installed Package Versions (Verified Working)

| Package | Version | Notes |
|---|---|---|
| torch | 2.6.0+cu124 | replaces pinned 1.13.1 |
| torchvision | 0.21.0+cu124 | replaces pinned 0.14.1 |
| pytorch3d | 0.7.9 | built from source |
| numpy | 2.4.4 | replaces pinned 1.21.6 |
| scipy | 1.17.1 | replaces pinned 1.10.1 |
| scikit-image | 0.26.0 | replaces pinned 0.20.0 |
| scikit-learn | 1.8.0 | replaces pinned 1.2.2 |
| open3d | 0.19.0 | replaces pinned 0.18.0 |
| pandas | 3.0.2 | replaces pinned 2.2.1 |
| trimesh | 4.11.5 | replaces pinned 3.21.5 |
| opencv-python | 4.13.0.92 | replaces pinned 4.7.0.72 |
| matplotlib | 3.10.8 | replaces pinned 3.8.2 |
| mujoco | 2.3.7 | matches pin |
| pybullet | 3.2.7 | matches pin |
| tensorboard | 2.20.0 | replaces pinned 2.11.2 |
| dex-ycb-toolkit | 1.0 | installed from GitHub |
| chumpy | 0.70 | requires special install (see above) |
| hdbscan | 0.8.42 | |
| kornia | 0.8.2 | |
| loguru | 0.7.3 | |
| lmdb | 2.2.0 | |
| fire | 0.7.1 | |
| urdf-parser-py | 0.0.4 | |
