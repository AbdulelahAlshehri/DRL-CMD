# DRL-CAMD

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Running Design Cases](#running-design-cases)
- [Citation](#citation)

## Overview
This repository introduces **DRL-CAMD**, a novel deep reinforcement learning (DRL) framework specifically developed for **Computational Molecular Design (CMD)**. The primary goal of DRL-CAMD is to design molecules with optimal properties while rigorously controlling uncertainties in property predictions.

**Key Features:**

- **Tailored Molecular Representation:** Ensures efficient and accurate molecular encoding for DRL models.
- **Gaussian Process Models:** Incorporates probabilistic property prediction models to estimate uncertainties reliably.
- **Smart Search Strategy:** Balances property optimization, constraint satisfaction, and uncertainty reduction.

By addressing the critical challenge of uncertainty in CMD, **DRL-CAMD** contributes towards greener and more reliable molecular design solutions applicable across various industrial scenarios.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AbdulelahAlshehri/DRL-CMD.git
cd DRL-CMD
pip install -r requirements.txt
```

**Important Note:**  
The property models used in DRL-CAMD are not provided within this repository due to their size and DIPPR data-sharing restrictions. A large subset of these property models and associated data is available separately in the following repository: [Pure-Component-Property-Estimation](https://github.com/PEESEgroup/Pure-Component-Property-Estimation).

Make sure to download and appropriately place the necessary property model files into the project directory structure as required.

## Quick Start

### Running Design Cases
Navigate to the project directory (`DRL-CMD`) and run the provided minimum working examples (MWEs) for specific molecular design applications as follows:

1. **Mercaptobenzothiazole Crystallization Solvent Design**
```bash
python mwe.py
```

2. **Organic Synthesis (DCM) Solvent Design**
```bash
python mwe1.py
```

3. **Emulsion Surfactant Design**
```bash
python mwe2.py
```

4. **Refrigerant Design**
```bash
python mwe3.py
```

Each script runs a self-contained demonstration case and generates candidate molecules tailored to the specified application domain. Ensure that all dependencies and external property models are correctly loaded before execution.

## Citation
If you utilize or adapt any models, datasets, or methods provided within DRL-CAMD, please cite the following reference:

```bibtex
@article{doi,
  author = {Alshehri, Abdulelah S. and Tantisujjatham, Bryan},
  title = {Uncertainty-aware Deep Reinforcement Learning Approach for Computational Molecular Design},
  journal = {Submitted to Industrial & Engineering Chemistry Research},
  volume = {n/a},
  number = {n/a},
  pages = {n/a},
  keywords = {},
  doi = {https://doi.org/},
  url = {},
  eprint = {},
  abstract = {Abstract}
}
```
