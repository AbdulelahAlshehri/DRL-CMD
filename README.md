# DRL-CAMD

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Running Design Cases](#running-design-cases)
  - [Creating Custom Design Cases](#creating-custom-design-cases)
- [Citation](#citation)

---

## Overview
**DRL-CAMD** (Deep Reinforcement Learning for Computational Molecular Design) is a state-of-the-art framework leveraging **Deep Reinforcement Learning (DRL)** to efficiently design molecules with desired properties while rigorously managing prediction uncertainties.

### Main Features:
- **Custom Molecular Representations**: Optimized encoding for molecular search efficiency.
- **Gaussian Process (GP) Models**: Robust uncertainty quantification for predicted molecular properties.
- **Advanced DRL Strategy**: Simultaneously optimizes property values, constraint adherence, and uncertainty minimization.

The framework is designed for diverse industrial applications including solvent design, refrigerants, surfactants, and absorbents.

---

## Installation

Clone the repository and install dependencies using the following commands:

```bash
git clone https://github.com/PEESEgroup/DRL-CMD.git
cd DRL-CMD
pip install -r requirements.txt
```

**Note:** Property prediction models are not included due to DIPPR data-sharing policies. To acquire property data/models, visit:

[Pure-Component-Property-Estimation](https://github.com/PEESEgroup/Pure-Component-Property-Estimation)

Download and integrate these models as instructed in their repository before running DRL-CAMD examples.

---

## Quick Start

### Running Design Cases
The repository includes ready-to-use scripts (`mwe.py`, `mwe1.py`, etc.) demonstrating various molecular design cases. Run these examples from the project root directory as follows:

1. **Mercaptobenzothiazole Crystallization Solvent Design:**
```bash
python mwe.py
```

2. **Organic Synthesis (DCM) Solvent Design:**
```bash
python mwe1.py
```

3. **Emulsion Surfactant Design:**
```bash
python mwe2.py
```

4. **Refrigerant Design:**
```bash
python mwe3.py
```

These scripts run DRL-based molecular searches and generate candidate molecules adhering to pre-defined constraints.

---

### Creating Custom Design Cases

To design a custom molecular search scenario, you must configure two main files:

#### Step 1: Editing the `config/case.yml` file

Define your custom molecular case in the YAML file by specifying constraints and groups as shown below:

**Example case configuration:**

```yaml
# Example: MBT Crystallization Case
mbt:
  name: "MBT"
  building_blocks: [1,2,3,4,15,..] #first order groups
  constraints:
    NUM_GROUPS:
      min: Lower bound
      max: Upper Bound
    NUM_REPEAT_GROUPS:
      min: Lower bound
      max: Upper Bound
    NUM_FUNC_GROUPS: 
      min: Lower bound
      max: Upper Bound
    MOLECULAR_WEIGHT: 
      min: Lower bound
      max: Upper Bound
    FLASH_POINT:
      min: Lower bound
      max: Upper Bound
    MELTING_POINT:
      min: Lower bound
      max: Upper Bound
    BOILING_POINT:
      min: Lower bound
      max: Upper Bound
    LC50:
      min: Lower bound
      max: Upper Bound
    HSP:
      min: Lower bound
      max: Upper Bound
  num_rings: all
```

- Modify the constraints to fit your specific requirements (e.g., boiling point, melting point).
- Adjust the list of building blocks based on your molecular groups database.

#### Step 2: Modifying the `mwe.py` script to run your case

Edit your Python script (`mwe.py`) to reflect the custom case you defined in the YAML file:

```python

warnings.filterwarnings('ignore')

# Define your logging directory
log_dir = "custom_case_logs/"

# Update this line to reflect your custom case name from the YAML file
parse = ParseData('-c mbt -r 2 -sp'.split())

# Load case data
cs = CaseSuite(parse, DataSet.instance())
rs = RunSettings()

# Replace 'CUSTOM' with your custom case ID from YAML configuration
case_data = cs.load_case_data()['CUSTOM']
case = CaseInstance(Case(case_data, DataSet.instance()), rs)

# Initialize the environment
env = Monitor(ActionMasker(MolecularSearchEnv(case), mask_fn), log_dir)

# Train the DRL model
model = MaskablePPO('MultiInputPolicy', env, verbose=1,
                    tensorboard_log=log_dir, n_steps=10)

# Adjust training steps as needed
model.learn(10000, tb_log_name="custom_case_run")

# Optional: Inspect the final state (uncomment the following line)
# env.state.show()
```

#### Step 3: Place your property models as named in the yml file above (e.g.,BOILING_POINT) in `models` folder to load property models


#### Explanation of Key Modifications:

- Change the argument passed to `ParseData()` to match the identifier used in your YAML file (e.g., `-c mbt` or `-c surfactant`).
- Customize the `log_dir` to store training outputs and TensorBoard logs appropriately.
- Adjust `n_steps` and total training steps (`model.learn`) based on your problem complexity.

**To run your customized script:**

```bash
python mwe.py
```

Your custom-designed molecular candidates will be generated and logged accordingly.

---

## Citation
When using or adapting any component or idea from DRL-CAMD, please cite:

```bibtex
@article{doi:10.1021/acs.iecr.4c04993,
author = {Alshehri, Abdulelah S. and Tantisujjatham, Bryan and Alrashed, Maher M.},
title = {Uncertainty-Aware Deep Reinforcement Learning Approach for Computational Molecular Design},
journal = {Industrial \& Engineering Chemistry Research},
volume = {0},
number = {0},
pages = {null},
year = {0},
doi = {10.1021/acs.iecr.4c04993},

URL = {https://doi.org/10.1021/acs.iecr.4c04993},
eprint = {https://doi.org/10.1021/acs.iecr.4c04993}
}
```


**Note:**  Commercial use is strictly prohibited in accordance with our license agreement, as well as the usage terms of several data sources and tools integrated into this framework.
