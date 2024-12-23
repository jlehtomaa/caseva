
# CASadi Extreme Value Analysis

## Introduction

This package provides a simple entry point to the 1-dimensional extreme value analysis as explained in the standard extreme value reference: [Stuart Coles (2001): An Introduction to Statistical Modeling of Extreme Values](https://link.springer.com/book/10.1007/978-1-4471-3675-0).

In short, this package features:
- Two models:
    - The Block Maxima model, based on the general extreme value distribution
    - The Peaks-Over-Threshold model, based on the generalized Pareto distribution
- Maximum likelihood solver based on [casadi](https://web.casadi.org/), a modern, open-source, and reliable tool for non-linear optimization and algorithmic differentiation.


## Installation

Create the virtual environment:

```
conda create --name caseva_env python=3.11.2
```

Activate the environment:

```
conda activate caseva_env
```

Navigate to the root of the directory, and install the package locally:

```
pip install -e .
```

For the additional dependencies required for running all the tutorial notebooks (for instance, for fetching data from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)), run:

```
pip install -e .[tutorial]
```

## Quickstart

```python
import numpy as np
import pandas as pd
from caseva.models import BlockMaximaModel

# Load data (annual block maxima)
data = pd.read_csv("<path-to-your-dataset>")
extremes = data["<data-variable-name>"]

# Fit model
model = BlockMaximaModel(max_optim_restarts=5)
model.fit(data=extremes)

# Evaluate
model.diagnostic_plot()

# Return level inference
model.return_level(return_period=[10, 100, 200])
```

## Tutorials

The `notebooks` folder contains the following tutorials:

- `01_tutorial.ipynb`. A first tutorial replicating the relevant numerical examples in [Coles (2001)](https://link.springer.com/book/10.1007/978-1-4471-3675-0).

- `02_wind_storm_application.ipynb`. A real-world example where we first fetch [reanalysis](https://www.ecmwf.int/en/about/media-centre/focus/2023/fact-sheet-reanalysis) data on European winterstorm wind fields and derive a return period map for the entire area. Obtaining the data requires registering as a user to the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).


#### References cited in the code

Coles (2001): An Introduction to Statistical Modeling of Extreme Values. Springer Series in Statistics.

Hosking & Wallis (1987): Parameter and Quantile Estimation for the Generalized Pareto Distribution. Technometrics, August 1987, Vol. 29 No. 3.

