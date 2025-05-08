# Generating the datasets

Explanation of the simulation code used in the [original paper](https://arxiv.org/abs/2006.11287).

## Overview

Generate n-body dataset for particles under different potentials.
TODO: WRITE THE POTENTIALS

- `simulate.py`: simulation script from the original paper.
- `generate_data.py`: script which produces datasets from `simulate.py`.

## Usage

From the terminal,

```bash 
python3 simulations/generate_data.py --sim r2 --save
```

The dataset will save in the `datasets/` folder.

## Dataset

To load the data,

```python
from generate_data import load_data
X, y = load_data('datasets/file_name.pt') 
```

The data, `X`
- of shape (num_datapoints, num_particles, params)
- params contains (x, y, dx, dy, charge, mass), if in 2D

The target variables, `y`
- of shape (num_datapoints, num_particles, 2)
- last dimension is acceleration in x and y
