import pysr
from plot_linear_rep import get_message_features
from model import load_model
from utils import load_data
import numpy as np
from pysr import PySRRegressor
import argparse
import os
import pickle
import json
print("PySR imported successfully!")

def get_pysr_variables(model, input_data):
    """
    Extract target messages and variables for symbolic regression from a trained model.
    
    Args:
        model: Trained model with model_type_ attribute (supports 'KL' and other types)
        input_data: Input data for message feature extraction
        
    Returns:
        tuple: Contains:
            - (target_message1, target_message2): Tuple of the two most important messages
            - variables: NumPy array of input variables (dx, dy, bd, m1, m2, q1, q2)
            - variable_names: List of variable names for symbolic regression
            
    Notes:
        - For KL models, uses KL divergence to determine message importance
        - For other models, uses standard deviation of messages across datapoints
        - Variables 'bd' is recast as 'r' in the variable names
    """
    sr_vars = ['dx', 'dy', 'bd' , 'm1', 'm2', 'q1', 'q2'] #you can change this if you want to include others in the regression
    variable_names = ['dx', 'dy','r', 'm_one', 'm_two', 'q_one', 'q_two'] #recasted bd as r 
    print(f'Variables considered in the SR are {sr_vars}.')

    #getting message for KL is different 
    print('Getting messages.')
    if model.model_type_ != 'KL':
        df, msg_array = get_message_features(model, input_data)
        msg_importance = msg_array.std(axis=0) #computes stds for each msg_dim over all datapoints
        most_important1 = np.argsort(msg_importance)[-1:]
        most_important2 = np.argsort(msg_importance)[-2:]
    else: 
        df, msg_array = get_message_features(model, input_data)
        logvar_array = msg_array[1]
        msg_array = msg_array[0]
        KL_div =  (np.exp(logvar_array) + msg_array**2 - logvar_array)/2
        KL_mean = KL_div.mean(axis=0)
        most_important1 = np.argsort(KL_mean.values)[-1:]
        most_important2 = np.argsort(KL_mean.values)[-2:]
    target_message1 = df[f'e{most_important1[0]}'].to_numpy().reshape(-1, 1)

    target_message2 = df[f'e{most_important2[0]}'].to_numpy().reshape(-1, 1)

    variables = df[sr_vars].to_numpy()

    return (target_message1, target_message2), variables, variable_names

def perform_sr(target_message, variables, num_points = 5_000, niterations = 1000, dataset_name = 'charge', save_path = None, message_number = 'run', variable_names = None):
    """
    Perform symbolic regression on target messages using PySR.
    
    Args:
        target_message: Target message array for regression
        variables: Input variables array
        num_points (int): Number of random data points to use (default: 5000)
        niterations (int): Number of PySR iterations (default: 1000)
        dataset_name (str): Name of dataset for configuration (default: 'charge')
        save_path (str): Directory to save PySR output (default: None)
        message_number (str): Identifier for the regression run (default: 'run')
        variable_names (list): Names of input variables (default: None)
        
    Returns:
        PySRRegressor: Fitted symbolic regression model
        
    Notes:
        - Uses random seed 290402 for reproducibility
        - Supports binary operators: +, *
        - Supports unary operators: inv, exp, log
        - Uses absolute error loss function
        - Configuration varies by dataset (special handling for 'charge' dataset)
    """
    
    np.random.seed(290402)
    #get a smaller random subset of points because we have too many edge messages
    idx = np.random.choice(len(target_message), size=num_points, replace=False)
    target_subset = target_message[idx]
    variables_subset = variables[idx]

    #set up the hyperparams for SR
    config = {'parsimony': 0.05, 
              'complexity_of_constants': 1, 
              'maxsize': 23}

    if dataset_name == 'charge':
        config['maxsize']= 25
        config['parsimony'] = 0.05

    print(f'Performing SR on the messages.')
    #set up the regressor
    regressor = PySRRegressor(
        maxsize=config['maxsize'],
        niterations=niterations,
        binary_operators=["+", "*"],
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
        complexity_of_constants=config['complexity_of_constants'],
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        parsimony=config['parsimony'],
        batching=True, 
        output_directory = save_path, 
        run_id = message_number
    )

    #perform SR
    regressor.fit(variables_subset, target_subset, variable_names = variable_names)

    return regressor

def main():
    """
    Main function to run symbolic regression on GNN message features.
    
    Command line arguments:
        --dataset_name (str): Name of the dataset to process
        --model_type (str): Type of model to load
        --num_points (int): Number of data points for SR (default: 6000)
        --niterations (int): Number of SR iterations (default: 1000)
        --num_epoch (str): Number of training epochs for model (default: 100)
        --save: Flag to enable saving results
        
    Workflow:
        1. Loads test data and trained model
        2. Extracts target messages and variables using get_pysr_variables
        3. Performs symbolic regression on both most important messages
        4. Saves regression results and metrics to JSON files
        
    Output:
        - Creates directory structure under pysr_objects/
        - Saves SR models and metrics for both target messages
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument('--num_points', type=int, default = 6_000)
    parser.add_argument('--niterations', type=int, default = 1_000)
    parser.add_argument("--num_epoch", type=str, default = 100)
    parser.add_argument('--save', action='store_true')
    
    args = parser.parse_args()
    _, _, test_data = load_data(args.dataset_name)
    save_path = f'pysr_objects/{args.dataset_name}/nit_{args.niterations}/{args.model_type}'
    os.makedirs(save_path, exist_ok=True)
    model = load_model(dataset_name=args.dataset_name, model_type=args.model_type, num_epoch=args.num_epoch)
    target_messages, variables, names = get_pysr_variables(model, test_data)
    target_message1, target_message2 = target_messages

    os.makedirs(f'{save_path}/message1', exist_ok=True)
    os.makedirs(f'{save_path}/message2', exist_ok=True)

    print('Performing SR on target message 1.')
    regressor = perform_sr(target_message1, variables, num_points = args.num_points,
                niterations=args.niterations, dataset_name = args.dataset_name, save_path=save_path, message_number='message1', variable_names=names)

    metrics = {'best_eqn': regressor.get_best()['equation'], 
            'num_points': args.num_points,
            'niterations': args.niterations,
            'GNN_epochs': args.num_epoch
            }
    
    with open(f'{save_path}/message_1_sr_metrics.json', 'w') as f:
        json.dump(metrics, f, indent = 2)

    print('Performing SR on target message 2.')
    regressor = perform_sr(target_message2, variables, num_points = args.num_points,
                niterations=args.niterations, dataset_name = args.dataset_name, save_path=save_path, message_number='message2', variable_names=names)

    metrics = {'best_eqn': regressor.get_best()['equation'], 
            'num_points': args.num_points,
            'niterations': args.niterations,
            'GNN_epochs': args.num_epoch
            }
    
    with open(f'{save_path}/message_2_sr_metrics.json', 'w') as f:
        json.dump(metrics, f, indent = 2)



if __name__ == "__main__":
    main()
