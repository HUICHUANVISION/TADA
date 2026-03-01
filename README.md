# TADA: Trustworthy Automated Data Augmentation

## Overview

This repository contains the official implementation of **TADA** (Trustworthy Automated Data Augmentation), a robust automated data augmentation framework for machine learning.

## Paper

The primary reference for this work is the published paper:

> **TADA: Trustworthy Automated Data Augmentation**
> 
> [Download Paper (PDF)](TADA_Paper.pdf)

If you use this work in your research, please cite our paper.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running TADA Experiments

```bash
python core/run_TADA_experiment.py
```

### Running HDP Experiments

```bash
python core/run_HDP_experiment.py
```

## Repository Structure

```
TADA_Public_Release/
├── TADA_Paper.pdf          # Main paper
├── core/                   # Core implementation
│   ├── TADA_GA.py         # Genetic Algorithm for augmentation
│   ├── run_TADA_experiment.py
│   └── run_HDP_experiment.py
├── README.md
└── .gitignore
```

## License

For academic use only. Please contact the authors for commercial licensing.

## Contact

For questions or issues, please refer to the paper for author contact information.
