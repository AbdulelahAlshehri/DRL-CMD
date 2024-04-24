# DRL-CAMD

## Table of Contents

- [DRL-CAMD](#drl-camd)
  - [Table of Contents](#table-of-contents)
  - [Overview](#Overview)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Citation](#citation)

## Overview
This work introduces DRL-CMD, a novel deep reinforcement learning (DRL) framework for designing molecules with desired properties while carefully controlling uncertainties in property predictions.DRL-CMD utilizes a tailored molecular representation,  Gaussian Process property prediction models, and a sophisticated search method to address the critical challenge of uncertainty in computational molecular design (CMD). By optimizing properties, minimizing constraint violations, and reducing uncertainty, DRL-CMD aims to create greener and more reliable molecular solutions for diverse industrial applications.
## Installation

`git clone https://github.com/PEESEgroup/DRL-CMD`

`pip install -r requirements.txt`

## Quick Start
To generate candidates for the following cases, run:

* **Mercaptobenzothiazole Crystallization Solvent Design:** `mwe.py`
* **Organic Synthesis (DCM) Solvent Design:** `mwe1.py`
* **Emulsion Surfactant Design:** `mwe2.py`
* **Refrigerant Design:** `mwe3.py`

**Important:** Our property models are not included in this repository due to their size and DIPPR data-sharing limitations. For a large subset of our property data, please refer to the following repository: [Pure-Component-Property-Estimation](https://github.com/PEESEgroup/Pure-Component-Property-Estimation)

## Citation 
Any alterations to the models, datasets, or functions included with DRL-CMD must be properly attributed according to the following citation.
```
@article{doi,
  author = {Alshehri, Abdulelah S. and Tantisujjatham, Bryan and You, Fengqi},
  title = {Uncertainty-aware Deep Reinforcement Learning Approach for Computational Molecular Design},
  journal = {Submitted to AIChE Journal},
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
