import pysr
from plot_linear_rep import get_message_features
from model import load_model
from utils import load_data
import numpy as np
# import os
# import sympy
from pysr import PySRRegressor
print("PySR imported successfully without warnings!")

def get_pysr_variables(model, input_data):
    sr_vars = ['dx', 'dy', 'r', 'm1', 'm2'] #you can change this if you want to include others in the regression

    print(f'Variables considered in the SR are {sr_vars}.')

    #getting message for KL is different 
    print('Getting messages.')
    if model.model_type_ != 'KL':
        df, msg_array = get_message_features(model, input_data)
        msg_importance = msg_array.std(axis=0) #computes stds for each msg_dim over all datapoints
        most_important = np.argsort(msg_importance)[-1:]
    else: 
        df, msg_array = get_message_features(model, input_data)
        logvar_array = msg_array[1]
        msg_array = msg_array[0]
        KL_div =  (np.exp(logvar_array) + msg_array**2 - logvar_array)/2
        KL_mean = KL_div.mean(axis=0)
        most_important = np.argsort(KL_mean)[-1:]
    target_message = df[f'e{most_important[0]}'].to_numpy().reshape(-1, 1)
    variables = df[sr_vars].to_numpy()

    return target_message, variables

def perform_sr(target_message, variables, num_points = 10_000):
    np.random.seed(290402)
    #get a smaller random subset of points because we have too many edge messages
    idx = np.random.choice(len(target_message), size=num_points, replace=False)
    target_subset = target_message[idx]
    variables_subset = variables[idx]

    print(f'Performing SR on the messages.')
    #set up the regressor
    regressor = PySRRegressor(
        maxsize=20,
        niterations=2000,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "inv(x) = 1/x"
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
    )

    regressor.fit(variables_subset, target_subset)

    best_eq = regressor.get_best()['equation']
    print(f'The best equation found was {best_eq}')


    return regressor

def main():
    model = load_model(dataset_name='spring', model_type='bottleneck', num_epoch='100')
    _, _, test_data = load_data('spring')
    target_message, variables = get_pysr_variables(model, test_data)
    regressor = perform_sr(target_message, variables)
    # print(regressor.get_best()['equation'])

if __name__ == "__main__":
    main()
