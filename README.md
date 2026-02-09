# GOPH 547 — Lab 1: Gravity Forward Modelling
* **Semester: W2026**

* **Instructor: B. Karchewski**

* **Author: Chigozie Ikechukwu**

* **Repository: https://github.com/ChigozieIkechukwu/goph547-w2026-lab01-stCI**

This repository contains solutions for GOPH 547 Lab 1A/1B: gravitational potential and gravity effects from point masses and a distributed mass anomaly.

## Repository Structure

goph547-w2026-lab01/
├── src/goph547lab01/
│ ├── init.py
│ └── gravity.py
├── tests/
│ └── test_gravity_functions.py
├── examples/
│ ├── driver_single_mass.py
│ ├── driver_multi_mass.py
│ └── driver_mass_anomaly.py
├── anomaly_data.mat
├── pyproject.toml
└── README.md



## Setup (Windows PowerShell)
From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install pytest

## Running Tests

Unit tests for the gravity functions (Lab 1A):

pytest

## Usage
* **Part A: Single Point Mass**
python examples/driver_single_mass.py


Generates contour plots of gravitational potential and vertical gravity
component for different elevations and grid spacings.

* **Part B: Multiple Point Masses**
python examples/driver_multi_mass.py

Generates multiple random mass configurations with identical total mass and
barycentre, saves them as .mat files, and produces gravity and potential
plots.

* **Part B: Distributed Mass Anomaly**
python examples/driver_mass_anomaly.py

Loads anomaly_data.mat, computes physical properties of the anomaly, and
generates density cross-sections and gravity-effect plots.

## Dependencies

Python ≥ 3.8

numpy

scipy

matplotlib

pytest (development)

**Author**

Chigozie Ikechukwu