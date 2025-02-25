# Generates and saves datasets from the simulate script

import numpy as np
import simulate
from simulate import SimulationDataset
import torch


def generate_data(ns = 10000, sim = 'r1', n = 3, dim = 2, nt = 1000, save = False):
    """
    Generate dataset using the simulate script from the original paper

    TODO: write the rest of this 

    Args:
        ns (int): number of simulations to run
        sim (str): _description_. Defaults to 'r1'.
        n (int): _description_. Defaults to 3.
        dim (int): _description_. Defaults to 2.
        nt (int): _description_. Defaults to 1000.
        save (bool): _description_. Defaults to False.

    Returns:
        X, y (tuple) containing:
            - X (torch tensor):
            - y (torch tensor)
    """

    n_set = [4, 8]
    sim_sets = [
    {'sim': 'r1', 'dt': [5e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'r2', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'spring', 'dt': [1e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'string', 'dt': [1e-2], 'nt': [1000], 'n': [30], 'dim': [2]},
    {'sim': 'charge', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'superposition', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'damped', 'dt': [2e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    {'sim': 'discontinuous', 'dt': [1e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
    ]
    
    dt = [ss['dt'][0] for ss in sim_sets if ss['sim'] == sim][0]
    title = '{}_n={}_dim={}_nt={}_dt={}'.format(sim, n, dim, nt, dt)

    s = SimulationDataset(sim, n=n, dim=dim, nt=nt//2, dt=dt)
    s.simulate(ns)

    accel_data = s.get_acceleration()

    X = torch.from_numpy(np.concatenate([s.data[:, i] for i in range(0, s.data.shape[1], 5)]))
    y = torch.from_numpy(np.concatenate([accel_data[:, i] for i in range(0, s.data.shape[1], 5)]))

    if save: 
        torch.save({
        'X': X,
        'y': y,
        'X_shape': X.shape,
        'y_shape': y.shape
        }, f"datasets/{title}.pt")

    return X, y

def load_data(path):
    data = torch.load(f"{path}.pt")
    return data['X'], data['y']
