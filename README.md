
# CASadi Extreme Value Analysis

## Introduction

This package provides a simple entry point to the 1-dimensional extreme value analysis as explained in the standard extreme value reference: [Stuart Coles (2001): An Introduction to Statistical Modeling of Extreme Values](https://link.springer.com/book/10.1007/978-1-4471-3675-0).

In short, this covers:
- Two models:
    - The Block Maxima model, based on the general extreme value distribution
    - The Peaks-Over-Threshold model, based on the generalized Pareto distribution
- Maximum likelihood solver based on [casadi](https://web.casadi.org/), a modern, open-source, and reliable tool for non-linear optimization and algorithmic differentiation.


## Installation

Create the virtual environment:

```
conda create --name caseva_env python=3.11.2
```

Navigate to the root of the directory, and install locally:

```
pip install -e .
```

For the additional dependencies for running the tutorial notebooks (for instance, for fetching data from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)), run:

```
pip install -e .[tutorial]
```

## Example use

```python
import numpy as np
import pandas as pd
from caseva.models import BlockMaximaModel

# Load data
data = pd.read_csv("<path-to-your-dataset>")
extremes = data["<data-variable-name>"]

# Fit model
model = BlockMaximaModel(max_optim_restarts=5)
model.fit(data=extremes)

# Evaluate
model.diagnostic_plot()
```

## Tutorials

#### References mentioned in the code

Coles (2001): An Introduction to Statistical Modeling of Extreme Values. Springer Series in Statistics.

Hosking & Wallis (1987): Parameter and Quantile Estimation for the Generalized Pareto Distribution. Technometrics, August 1987, Vol. 29 No. 3.