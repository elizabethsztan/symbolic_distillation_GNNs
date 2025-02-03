# Particle Simulation System Documentation (MADE USING CLAUDE)

Explanation of the simulation code used in the **original paper**.

## Overview

The system simulates particles interacting through various potential functions, supporting different physical scenarios from gravitational systems to charged particles and spring-like connections.

## Core Components

### 1. Class Structure

The main class `SimulationDataset` initializes with the following parameters:
- `sim`: Simulation type ('r2', 'charge', 'string', etc.)
- `n`: Number of particles
- `dim`: Dimensions (2D or 3D)
- `dt`: Time step size
- `nt`: Number of time steps
- `extra_potential`: Optional additional potential field

### 2. Potential Functions

The system supports multiple interaction types through `get_potential()`:
- 'r2': Inverse square law (gravitational-like)
- 'r1': Logarithmic potential
- 'spring': Harmonic oscillator
- 'charge': Coulomb-like interaction
- 'string': Connected particles with vertical force
- 'string_ball': String simulation with barrier

### 3. Simulation Core

The `simulate()` method handles the core simulation logic:
```python
def simulate(self, ns, key=0):
    # ns: Number of simulations
    # key: Random seed
```

#### Key Components:

a) Force Calculation
```python
def total_potential(xt):
    # Calculates total potential energy 
    # by summing pairwise interactions
```

b) Dynamics
```python
def acceleration(xt):
    # Uses F = ma to derive acceleration from forces
```

c) ODE Integration
```python
def odefunc(y, t):
    # Defines differential equations for particle motion
```

### 4. Data Structure

Each particle state contains:
- Position coordinates (first `dim` values)
- Velocity coordinates (next `dim` values)
- Additional parameters (e.g., mass/charge) (last `params` values)

### 5. Visualization

The `plot()` method provides visualization options:
```python
def plot(self, i, animate=False, plot_size=True, s_size=1):
```
Features:
- Static or animated trajectories
- Color-coded particles
- Mass-scaled markers
- Motion animation

## Technical Implementation

The code leverages JAX for efficient computation:
- JIT compilation (`@jit`)
- Automatic differentiation for force calculation
- Vectorized operations (`vmap`)
- ODE integration (`odeint`)

## Initial Conditions

Initial states are generated randomly with specific constraints based on simulation type:
```python
def make_sim(key):
    # Sets random initial positions and velocities
    # Handles special cases for different simulation types
```

## Usage

1. Create a simulation instance:
```python
sim = SimulationDataset(sim='r2', n=5, dim=2)
```

2. Run simulations:
```python
sim.simulate(ns=10)  # Run 10 different simulations
```

3. Visualize results:
```python
sim.plot(i=0)  # Plot first simulation
```

## Dependencies
- JAX: Core computation
- Matplotlib: Visualization
- NumPy: Array operations
- Celluloid: Animation support
