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
    sr_vars = ['dx', 'dy', 'r', 'm1', 'm2', 'q1', 'q2'] #you can change this if you want to include others in the regression

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
        most_important = np.argsort(KL_mean.values)[-1:]
    # print(f"Model type: {model.model_type_}")
    # print(f"most_important: {most_important}")
    # print(f"most_important[0]: {most_important[0]}")
    # print(f"Type of most_important[0]: {type(most_important[0])}")
    # print(f"Column name to access: 'e{most_important[0]}'")
    # print(f"DataFrame shape: {df.shape}")
    # print(f"DataFrame columns: {list(df.columns)}")
    # print(f"Does column exist? {'e' + str(most_important[0]) in df.columns}")
    # print(f"Type of df: {type(df)}")
    target_message = df[f'e{most_important[0]}'].to_numpy().reshape(-1, 1)
    variables = df[sr_vars].to_numpy()

    return target_message, variables

def perform_sr(target_message, variables, num_points = 5_000, niterations = 1000, dataset_name = 'charge', save_path = None, model_type = 'run'):
    np.random.seed(290402)
    #get a smaller random subset of points because we have too many edge messages
    idx = np.random.choice(len(target_message), size=num_points, replace=False)
    target_subset = target_message[idx]
    variables_subset = variables[idx]

    config = {'parsimony': 0.01, 
              'complexity_of_constants': 1}
    
    if dataset_name == 'r2':
        config['parsimony'] = 0.01
        config['complexity_of_constants'] = 2.5

    elif dataset_name == 'charge' or dataset_name == 'spring':
        config['parsimony'] = 0.05
        config['complexity_of_constants'] = 1.5

    print(f'Performing SR on the messages.')
    #set up the regressor
    regressor = PySRRegressor(
        maxsize=20,
        niterations=niterations,
        # binary_operators=["+", "*", ">", "<", "cond"],
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
        # complexity_of_operators={"exp": 3, "log": 3, "^": 3, "cond": 3},
        complexity_of_operators={"exp": 3, "log": 3, "^": 3},
        complexity_of_constants=config['complexity_of_constants'],
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        parsimony=config['parsimony'],
        batching=True, 
        output_directory = save_path, 
        run_id = model_type
    )


    regressor.fit(variables_subset, target_subset)

    # best_eq = regressor.get_best()['equation']
    # print(f'The best equation found was {best_eq}')

    return regressor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument('--num_points', type=int, default = 6_000)
    parser.add_argument('--niterations', type=int, default = 1_000)
    parser.add_argument("--num_epoch", type=str, default = 100)
    parser.add_argument('--save', action='store_true')
    
    args = parser.parse_args()
    _, _, test_data = load_data(args.dataset_name)
    save_path = f'pysr_objects/{args.dataset_name}/nit_{args.niterations}'
    os.makedirs(save_path, exist_ok=True)
    if args.model_type == 'all':
        model_types = ['standard', 'L1', 'bottleneck', 'KL', 'pruning']

        for model_type in model_types:
            print(f'Running SR on {model_type} model.')
            model = load_model(dataset_name=args.dataset_name, model_type=model_type, num_epoch=args.num_epoch)
    
            target_message, variables = get_pysr_variables(model, test_data)

            regressor = perform_sr(target_message, variables, num_points = args.num_points,
                        niterations=args.niterations, dataset_name = args.dataset_name, save_path=save_path, model_type=model_type)

            metrics = {'best_eqn': regressor.get_best()['equation'], 
                    'num_points': args.num_points,
                    'niterations': args.niterations,
                    'GNN_epochs': args.num_epoch
                    }
            
            with open(f'{save_path}/{model_type}_sr_metrics.json', 'w') as f:
                json.dump(metrics, f, indent = 2)

            print(regressor.get_best()['equation'])
    else:
        model = load_model(dataset_name=args.dataset_name, model_type=args.model_type, num_epoch=args.num_epoch)
        target_message, variables = get_pysr_variables(model, test_data)

        regressor = perform_sr(target_message, variables, num_points = args.num_points,
                    niterations=args.niterations, dataset_name = args.dataset_name, save_path=save_path, model_type=args.model_type)

        metrics = {'best_eqn': regressor.get_best()['equation'], 
                'num_points': args.num_points,
                'niterations': args.niterations,
                'GNN_epochs': args.num_epoch
                }
        
        with open(f'{save_path}/{args.model_type}_sr_metrics.json', 'w') as f:
            json.dump(metrics, f, indent = 2)

        print(regressor.get_best()['equation'])


if __name__ == "__main__":
    main()
