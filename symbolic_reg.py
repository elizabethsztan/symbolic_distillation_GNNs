import pysr
from plot_linear_rep import get_message_features
from model import load_model
from utils import load_data
import numpy as np
from pysr import PySRRegressor
import sympy
print("PySR imported successfully!")

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

def perform_sr(target_message, variables, num_points = 5_000, niterations = 1000):
    np.random.seed(290402)
    #get a smaller random subset of points because we have too many edge messages
    idx = np.random.choice(len(target_message), size=num_points, replace=False)
    target_subset = target_message[idx]
    variables_subset = variables[idx]

    print(f'Performing SR on the messages.')
    #set up the regressor
    regressor = PySRRegressor(
        maxsize=20,
        niterations=niterations,
        binary_operators=["+", "*", ">", "<", "cond"],
        unary_operators=[
            "inv(x) = 1/x",
            "exp",
            "log"
        ],
        extra_sympy_mappings={
            "inv": lambda x: 1 / x
        },
        constraints={'exp': (1), 'log': (1)},
        complexity_of_operators={"exp": 3, "log": 3, "^": 3},
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        parsimony=0.05,
        batching=True
    )


    regressor.fit(variables_subset, target_subset)

    # best_eq = regressor.get_best()['equation']
    # print(f'The best equation found was {best_eq}')

    return regressor

def main():
    model = load_model(dataset_name='spring', model_type='L1', num_epoch='100')
    _, _, test_data = load_data('spring')
    target_message, variables = get_pysr_variables(model, test_data)
    regressor = perform_sr(target_message, variables, niterations=1000)
    print(regressor.get_best()['equation'])

if __name__ == "__main__":
    main()
