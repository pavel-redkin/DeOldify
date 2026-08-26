# Upgrade DeOldify to Python 3.14

## Overview

There are **7 categories** of changes across **15 files** (3 in `deoldify/`, 8 in `fastai/`, 1 in `fid/`, and 3 config files). The vendored `fastai/` is the biggest source of required changes.

## Prerequisite: PyTorch 2.x Required

PyTorch 1.x is incompatible with Python 3.14:
- No binary wheels exist for PyTorch 1.x on Python 3.14
- CPython C-API breaks (removed/changed internal symbols)
- `distutils` removal in Python 3.12 breaks PyTorch 1.x build system

Minimum viable combination: **Python 3.14 + PyTorch 2.5+**

---

## Step 1: Critical — Remove `abstractproperty` import (Python 3.13 removal)

**File:** `fastai/imports/core.py:8`

`abstractproperty` was removed from `abc` in Python 3.13. The import will crash the entire project. Since `@abstractproperty` is never actually used as a decorator anywhere in the codebase, this is a clean removal.

```python
# FROM
from abc import abstractmethod, abstractproperty
# TO
from abc import abstractmethod
```

**Also:** Update `.pylintrc:362` to remove `abc.abstractproperty` from `property-classes`.

---

## Step 2: Critical — Replace `logging.warn()` (removed in Python 3.12)

**File:** `deoldify/filters.py:62`

`logging.warn()` was deprecated in Python 3.3 and removed as an alias in Python 3.12+.

```python
# FROM
logging.warn('Warning: render_factor was set too high...')
# TO
logging.warning('Warning: render_factor was set too high...')
```

---

## Step 3: Critical — Replace `torch.symeig()` (removed in PyTorch 2.x)

**File:** `deoldify/loss.py:89, 98, 102`

`torch.symeig` was deprecated in PyTorch 1.9 and removed in PyTorch 2.0. The replacement is `torch.linalg.eigh`.

Both return eigenvalues in ascending order, so the logic is equivalent.

**Line 89:**
```python
# FROM
eigvals, eigvects = torch.symeig(cov, eigenvectors=True)
# TO
eigvals, eigvects = torch.linalg.eigh(cov)
```

**Line 98:**
```python
# FROM
tr_cov_synth = torch.symeig(cov_synth, eigenvectors=True)[0].clamp(min=0).sum()
# TO
tr_cov_synth = torch.linalg.eigh(cov_synth)[0].clamp(min=0).sum()
```

**Lines 101-102:**
```python
# FROM
torch.symeig(cov_prod, eigenvectors=True)[0].clamp(min=0) + 1e-8
# TO
torch.linalg.eigh(cov_prod)[0].clamp(min=0) + 1e-8
```

**Also:** Update `.pylintrc:385` — remove `torch.symeig` from `generated-members` list.

---

## Step 4: Replace `pkg_resources` with `importlib.metadata`

**File:** `fastai/imports/core.py:29-30, 47`

`pkg_resources` is deprecated in favor of `importlib.metadata`.

**Lines 29-30:**
```python
# FROM
import pkg_resources
pkg_resources.require("fastprogress>=0.1.19")
# TO
import importlib.metadata as _importlib_metadata
try:
    _fp_ver = _importlib_metadata.version("fastprogress")
except _importlib_metadata.PackageNotFoundError:
    _fp_ver = "0.0.0"
from packaging.version import Version as _Version
if _Version(_fp_ver) < _Version("0.1.19"):
    raise ImportError("fastprogress>=0.1.19 is required")
```

**Lines 44-49 (`have_min_pkg_version`):**
```python
# FROM
def have_min_pkg_version(package, version):
    try:
        pkg_resources.require(f"{package}>={version}")
        return True
    except:
        return False
# TO
def have_min_pkg_version(package, version):
    try:
        import importlib.metadata as _im
        from packaging.version import Version as _V
        pkg_ver = _im.version(package)
        return _V(pkg_ver) >= _V(version)
    except Exception:
        return False
```

---

## Step 5: Add `weights_only=False` to all `torch.load()` calls

In PyTorch 2.x+, `torch.load` defaults to `weights_only=True` (or raises a FutureWarning). Since these files load pickled model state dicts with custom objects, they need `weights_only=False` explicitly.

| File | Line | Call |
|------|------|------|
| `fastai/basic_train.py` | 271 | `torch.load(source, map_location=device)` |
| `fastai/basic_train.py` | 322 | `torch.load(tmp_file)` |
| `fastai/basic_train.py` | 619 | `torch.load(source, map_location='cpu')` and `torch.load(source)` |
| `fastai/basic_data.py` | 277 | `torch.load(source, map_location='cpu')` and `torch.load(source)` |
| `fastai/data_block.py` | 583 | `torch.load(open(path/fn, 'rb'))` |
| `fastai/text/learner.py` | 69 | `torch.load(..., map_location=device)` |
| `fastai/text/learner.py` | 76 | `torch.load(wgts_fname, map_location=...)` |
| `fastai/vision/models/presnet.py` | 123 | `torch.load(model_urls[name])` |

**Pattern for each:**
```python
# FROM
torch.load(source, map_location=device)
# TO
torch.load(source, map_location=device, weights_only=False)
```

---

## Step 6: Fix `pretrained=True` for torchvision compatibility

In torchvision 0.13+, the `pretrained` parameter is deprecated. The new API uses `weights=` enum. Use try/except to support both old and new torchvision.

**File:** `fastai/vision/learner.py:54-56`
```python
# FROM
def create_body(arch:Callable, pretrained:bool=True, cut=None):
    model = arch(pretrained=pretrained)
# TO
def create_body(arch:Callable, pretrained:bool=True, cut=None):
    try:
        model = arch(pretrained=pretrained)
    except TypeError:
        model = arch(weights='IMAGENET1K_V1' if pretrained else None)
```

**File:** `fid/inception.py:83`
```python
# FROM
inception = models.inception_v3(pretrained=True)
# TO
try:
    inception = models.inception_v3(pretrained=True)
except TypeError:
    inception = models.inception_v3(weights='IMAGENET1K_V1')
```

---

## Step 7: Update configuration files

### environment.yml
```yaml
# FROM
- python=3.10
- pytorch::pytorch=1.11.0
# TO
- python=3.14
- pytorch::pytorch>=2.5.0
```

### requirements.txt
```
# FROM
--extra-index-url https://download.pytorch.org/whl/cu113
torch==1.11.0
torchvision==0.12.0
# TO
--extra-index-url https://download.pytorch.org/whl/cu124
torch>=2.5.0
torchvision>=0.20.0
torchaudio>=2.5.0
```

### setup.py
```python
# FROM
python_requires=">=3.6",
classifiers=[..., "Programming Language :: Python :: 3.6", "Programming Language :: Python :: 3.7"]
# TO
python_requires=">=3.10",
classifiers=[..., "Programming Language :: Python :: 3.10", "Programming Language :: Python :: 3.11",
              "Programming Language :: Python :: 3.12", "Programming Language :: Python :: 3.13",
              "Programming Language :: Python :: 3.14"]
```

### .pre-commit-config.yaml
```yaml
# FROM
language_version: python3.6
# TO
language_version: python3.14
```

---

## Summary of all files to modify

| # | File | Changes |
|---|------|---------|
| 1 | `fastai/imports/core.py` | Remove `abstractproperty`, replace `pkg_resources` |
| 2 | `deoldify/filters.py` | `logging.warn` → `logging.warning` |
| 3 | `deoldify/loss.py` | `torch.symeig` → `torch.linalg.eigh` (3 locations) |
| 4 | `fastai/basic_train.py` | Add `weights_only=False` to 4 `torch.load` calls |
| 5 | `fastai/basic_data.py` | Add `weights_only=False` to 2 `torch.load` calls |
| 6 | `fastai/data_block.py` | Add `weights_only=False` to 1 `torch.load` call |
| 7 | `fastai/text/learner.py` | Add `weights_only=False` to 2 `torch.load` calls |
| 8 | `fastai/vision/models/presnet.py` | Add `weights_only=False` to 1 `torch.load` call |
| 9 | `fastai/vision/learner.py` | `create_body` try/except for `pretrained` API |
| 10 | `fid/inception.py` | `inception_v3(pretrained=True)` try/except |
| 11 | `environment.yml` | Python 3.14, PyTorch 2.5+ |
| 12 | `requirements.txt` | PyTorch 2.5+, torchvision 0.20+, CUDA 12.4 |
| 13 | `setup.py` | `python_requires`, classifiers |
| 14 | `.pre-commit-config.yaml` | `language_version: python3.14` |
| 15 | `.pylintrc` | Remove `abc.abstractproperty` and `torch.symeig` |

---

## Risks and caveats

1. **Vendored fastai monkey-patching** — `DataLoader.__init__` override, `Tensor.__array__`, `Path.ls`, etc. These are fragile with PyTorch 2.x but are runtime issues, not syntax issues. They will need testing.
2. **`torchvision.models.inception` internal classes** — `fid/inception.py` subclasses `models.inception.InceptionA`, `InceptionC`, `InceptionE`. These internal APIs may have moved in newer torchvision versions.
3. **Pretrained weight compatibility** — Existing `.pth` weight files were saved with PyTorch 1.11. Loading them with `weights_only=False` on PyTorch 2.x should work (backward compatible), but hasn't been validated.
4. **`imgaug==0.2.6`** in `requirements-colab.txt` may not support Python 3.14.
