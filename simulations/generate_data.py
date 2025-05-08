# Generates and saves datasets from the simulate script
import numpy as np
import simulate
from simulate import SimulationDataset
import argparse
import torch
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))


def generate_data(sim = 'r1', save = True):
    """
    Generate dataset using the simulate script from the original paper

    Returns:
        X, y (tuple) containing:
            - X (torch.tensor): dataset of shape [num_datapoints, num_particles, params]
            - y (torch tensor): acceleration data of shape [num_datapoints, num_particles, num_dimension]
    """

    n_set = [4, 8] #could do 8 particles too but just working with 4 for now
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

    sim_config = next(s for s in sim_sets if s['sim'] == sim)
    n = sim_config['n'][0] #number of particles 'nodes'
    dim = sim_config['dim'][0] #dimension
    nt = sim_config['nt'][0] #number of timesteps
    dt = sim_config['dt'][0] #discretised time
    ns = 10_000 #number of simulations to run

    title = '{}_n={}_dim={}_nt={}_dt={}'.format(sim, n, dim, nt, dt)
    print('Running on', title)

    s = SimulationDataset(sim, n=n, dim=dim, nt=nt//2, dt=dt)
    s.simulate(ns)

    accel_data = s.get_acceleration()

    #take every 5th datapoint from each simulation
    #nearby datapoints are correlated 
    X = torch.from_numpy(np.concatenate([s.data[:, i] for i in range(0, s.data.shape[1], 5)]))

    #acceleration data (target)
    y = torch.from_numpy(np.concatenate([accel_data[:, i] for i in range(0, s.data.shape[1], 5)]))

    if save: 
        save_path = os.path.join(script_dir, f"../datasets/{title}.pt")
        torch.save({
        'X': X,
        'y': y,
        'X_shape': X.shape,
        'y_shape': y.shape
        }, f"{save_path}")

    return X, y

def load_data(path):
    data = torch.load(f"{path}.pt")
    return data['X'], data['y']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, required=True)
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()

    _,_ = generate_data (sim = args.sim, save = args.save)

if __name__ == "__main__":
    main()
