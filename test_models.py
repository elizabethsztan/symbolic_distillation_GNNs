from model import *
from utils import *
import json
import os

seed = 290402 
"""
This script runs the test set through all the trained model variations to get prediction losses (used in the report).
We assume that all the models (standard, L1, bottleneck, KL, pruning) have been trained on all the simulations 
(charge, r1, r2, spring) for 100 epochs.
"""

data_paths = ['datasets/charge_n=4_dim=2_nt=1000_dt=0.001.pt', 'datasets/r1_n=4_dim=2_nt=1000_dt=0.005.pt', 'datasets/r2_n=4_dim=2_nt=1000_dt=0.01.pt', 'datasets/spring_n=4_dim=2_nt=1000_dt=0.01.pt']
simulation_names = ['charge', 'r1', 'r2', 'spring']
model_types = ['standard', 'bottleneck', 'L1', 'KL', 'pruning']

#save the prediction losses here
metrics = {}

for i, data_path in enumerate(data_paths):
    #iterate thru all datasets
    _, _, test_data = load_and_process(data_path, seed)
    sim_name = simulation_names[i]
    metrics[sim_name] = {}
    
    #iterate thru all model types
    for model_type in model_types:
        print(f"Testing {sim_name} - {model_type}")
        model = load_model(simulation_names[i], model_type, 100)
        prediction_loss = test(model, test_data)
        print('Test set loss: ', prediction_loss)
        metrics[sim_name][model_type] = prediction_loss

save_path = 'model_weights'
os.makedirs(save_path, exist_ok=True)

#save results
with open(f'{save_path}/test_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nResults saved to 'test_results.json'")
print("\nSummary:")
for sim_name in simulation_names:
    print(f"\n{sim_name}:")
    for model_type in model_types:
        print(f"  {model_type}: {metrics[sim_name][model_type]:.4f}")