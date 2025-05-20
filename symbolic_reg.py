from plot_linear_rep import get_message_features
from model import load_model
from utils import load_data
import numpy as np
import os
from pysr import PySRRegressor

def get_pysr_variables(model, input_data):
    sr_vars = ['dy', 'dx', 'r', 'm1', 'm2'] #you can change this if you want to include others in the regression

    print(f'Variables considered in the SR are {sr_vars}.')

    #need to add option for KL too. choosing the best message now becomes different 
    if model.model_type_ != 'KL':
        print('Getting messages.')
        df, msg_array = get_message_features(model, input_data)
        msg_importance = msg_array.std(axis=0) #computes stds for each msg_dim over all datapoints
        most_important = np.argsort(msg_importance)[-1:]
    target_message = df[f'e{most_important[0]}'].to_numpy().reshape(-1, 1)
    variables = df[sr_vars].to_numpy()

    return target_message, variables

def perform_sr(target_message, variables, num_points = 5000):
    np.random.seed(290402)
    #get a smaller random subset of points because we have too many edge messages
    idx = np.random.choice(len(target_message), size=num_points, replace=False)
    target_subset = target_message[idx]
    variables_subset = variables[idx]

    print(f'Performing SR on the messages.')
    #set up the regressor
    regressor = PySRRegressor(
        maxsize=20,
        niterations=40,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "inv(x) = 1/x"
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )

    regressor.fit(variables_subset, target_subset)

    print(f'The best equation found was {regressor.get_best()['equation']}')

    return regressor

def main():
    model = load_model(dataset_name='spring', model_type='bottleneck', num_epoch='100')
    _, _, test_data = load_data('spring')
    target_message, variables = get_pysr_variables(model, test_data)
    regressor = perform_sr(target_message, variables)