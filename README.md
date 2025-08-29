# n-LMF: Neural Local Mixing Function Training with batches

Training code for **neural local-mixing-function (n-LMF)** based local-hybrid DFT models.
Implements a Keras/TensorFlow pipeline with a custom `DFTLayer` to evaluate XC terms on molecular grids and fit model parameters across standard benchmark test sets. **DFT-D4** dispersion is included via an external executable. The code split training into batches contianing one reaction.

> **Data availability:** Preprocessed training/test data are **available upon request**.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Configuration](#configuration)
  - [Test-set paths](#test-set-paths)
  - [DFT-D4 executable path (important)](#dft-d4-executable-path-important)
- [Usage](#usage)
- [Notes & Tips](#notes--tips)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Features

- Custom **Keras `DFTLayer`** (e.g., PBE-x, B95/B97-c; optional range separation).
- **DFT-D4** dispersion via a system `dftd4` binary.
- Flexible **test-set factory** (e.g., `BH76`, `W417`, `ACONF`, `FracP`, …).
- Configurable model depth/width/activations and loss scaling.
- Export of weights in a **TurboMole-friendly** format.

---

## Project Structure

```
├── calc.py                        # Entry point: configure & launch training
└── opt/
    ├── trainer.py                 # Training loop & bookkeeping
    ├── model_creator.py           # Builds & compiles the Keras model
    ├── DFTlayer.py                # Custom Keras layer for XC evaluation
    ├── dft.py                     # XC formulas (PBE-x, B95/B97-c, RS options)
    ├── data_loading.py            # I/O for grid/NPY, reaction assembly, D4 inject
    ├── dispersion.py              # D4 integration helpers
    ├── testsetclass.py            # Base test-set class + D4 calls (update paths!)
    ├── testset_factory.py         # Instantiates supported test sets
    ├── custom_transform.py        # Optional feature transforms
    ├── auxiliary_functions.py     # Utilities (e.g., TurboMole export)
    └── <test set files>           # ACONF.py, BH76.py, W417.py, ...
```

---

## Requirements

- **Python 3.10+**
- **TensorFlow 2.x**
- `numpy`, `psutil`
- **DFT-D4** (Grimme group) available on your system path (a known absolute path)

---

## Installation

Using conda (example):

```bash
conda create -n nlmf python=3.10 -y
conda activate nlmf
pip install "tensorflow>=2.12" numpy psutil
# Install dftd4 separately so the 'dftd4' executable is available.
```

---

## Data

- The repository does **not** include training/test data.
- Preprocessed *.npy files are **available upon request**.
- Expected per-set directory structure should match the paths you configure in `calc.py`.

---

## Configuration

### Test-set paths

Edit `TESTSET_PATHS` in **`calc.py`** to point to your local directories:

```python
TESTSET_PATHS = {
    "W417":      "/path/to/w417/",
    "BH76_full": "/path/to/bh76_full/",
    "ACONF":     "/path/to/aconf/",
    "FracP":     "/path/to/fracp/",
    # ...
}
```

Also adjust:
- `TESTSET_NAMES` (which datasets to include)
- Model hyperparameters (layers/units/activations)
- Training schedule (`EPOCHS`, learning-rate, etc.)
- Loss scaling and penalties (e.g., `SCALE`, negativity penalty)

### DFT-D4 executable path (important)

In **`opt/testsetclass.py`**, **replace the hard-coded DFT-D4 path** with your local path . This file constructs and runs shell commands that invoke `dftd4`. 

---

## Usage

From the project root:

```bash
python calc.py
```

Artifacts (checkpoints, exported weights in TurboMole-friendly format, logs) are written to the working directory.

---

## Notes & Tips

- **Reproducibility:** Set Python/NumPy/TensorFlow seeds at the top of `calc.py` if you need exact runs.
- **Features:** Ensure your feature selection in `calc.py` matches the columns present in your data.

---

## Citation

If you use this code in academic work, please cite the associated paper(s).
TO add.

---

## License

MIT.

---

## Contact

For questions or **data requests**, please open an issue or contact the maintainers.
