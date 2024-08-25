# QATCH Technologies - QModel
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/github/actions/workflow/status/username/repository/build.yml?branch=main)
![Code Coverage](https://img.shields.io/codecov/c/github/username/repository)
![Last Commit](https://img.shields.io/github/last-commit/username/repository)

## Overview

This repository serves as a collection of machine learning pipelines for accurately predicting fill location in sensory equipment.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Computer vision for categorization of runs.
- Gradient boosted trees for point-wise classification of runs.
- Overall system validation testing on a per model basis.
  
## Installation
While there is no installation, running all models can be done by running the `QTest.py` script which will benchmark testing against a set of randomly chosen runs from `QATCH-ML/content/`.
```bash
$ git clone https://github.com/QATCH-Technologies/QATCH-ML.git
```
```bash
$ python3 QModel/QTest.py
```
### Prerequisites

- Python 3.8+
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `tensoflow`

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```
