import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from model import get_edge_index
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
from model import load_model, get_edge_index
from pruning_experiments import load_pruning_models
from utils import load_data
import argparse
from scipy.optimize import minimize
import json

def get_message_features(model, input_data):
    """
    Extract message features from a trained GNN model for analysis.
    
    Args:
        model: Trained PyTorch geometric model with edge_model attribute
        input_data (tuple): Tuple containing (X_test, y_test) where:
            - X_test: Input features of shape [n_samples, n_nodes, n_features]
            - y_test: Target values (accelerations) [n_samples, n_nodes, n_dim]
    
    Returns:
        tuple: (df, msg_array) where:
            - df: pandas DataFrame containing node features, messages, and computed distances
            - msg_array: numpy array of message features, or list [messages, logvar] for KL models
    """
    model.eval()

    X_test, y_test = input_data
    edge_index = get_edge_index(X_test.shape[1])
    
    dataset = DataLoader(
        [Data(x=X_test[i], edge_index=edge_index, y=y_test[i]) for i in range(len(X_test))],
        batch_size=len(X_test),
        shuffle=False)
    
    all_message_info = []
    
    with torch.no_grad():
        for datapoint in dataset:
            #datapoint.x is of shape [num_nodes, num_features]
            source_nodes = datapoint.x[datapoint.edge_index[0]]  #source nodes. shape [num_edges*2, num_features]. features of source nodes of all edges
            target_nodes = datapoint.x[datapoint.edge_index[1]]  #target nodes
            
            #get output from the edge_model MLP
            x = torch.cat((source_nodes, target_nodes), dim=1)
            messages = model.edge_model(x)
            if model.model_type_ == 'KL':
                logvar = messages[:,1::2] #for KL model, take means and logvar of the messages
                messages = messages[:, 0::2] 
                message_info = torch.cat((source_nodes, target_nodes, messages, logvar), dim=1)
            else:
                message_info = torch.cat((source_nodes, target_nodes, messages), dim=1)

            all_message_info.append(message_info) #collect message info for all edges in each graph
    
    #combine message info
    message_info = torch.cat(all_message_info, dim=0) #shape [num_edges, num_features*2 + msg_dim]
    message_info = message_info.numpy()
    
    #create a dataframe to store node info
    node_info = ['x', 'y', 'nu_x', 'nu_y', 'q', 'm'] #only in 2d for now
    source_cols = [f'{f}1' for f in node_info]
    target_cols = [f'{f}2' for f in node_info]

    msg_dim = messages.shape[-1] #this is 100 unless bottleneck
    message_cols = [f'e{i}' for i in range(msg_dim)] 

    if model.model_type_ == 'KL':
        logvar_cols = [f'logvar{i}' for i in range(msg_dim)]
        columns = source_cols + target_cols + message_cols + logvar_cols
    
    else:
        columns = source_cols + target_cols + message_cols
    
    #make a dataframe with all the relevent information 
    df = pd.DataFrame(message_info, columns=columns)
    
    #calculate distances
    df['dx'] = df.x1 - df.x2
    df['dy'] = df.y1 - df.y2
    df['r'] = np.sqrt(df.dx**2 + df.dy**2)
    df['bd'] = df.r + 1e-2 #this is the displacement 
    
    #calculate relative velocities 
    df['dnu_x'] = df.nu_x1 - df.nu_x2
    df['dnu_y'] = df.nu_y1 - df.nu_y2
    df['dnu'] = np.sqrt(df.dnu_x**2 + df.dnu_y**2)
    
    #extract just the message features
    msg_array = df[message_cols].values #of shape [num_datapoints*(2*num_edges), msg_dim]
    
    if model.model_type_ == 'KL':
        logvar_array = df[logvar_cols]
        return df, [msg_array, logvar_array]
    else:
        return df, msg_array

def fit_messages(df, msg_array, sim='spring', dim=2, robust = True):
    """
    Fit linear combinations of expected physical forces to explain message features.
    
    Args:
        df (pd.DataFrame): DataFrame containing node features and computed distances
        msg_array (np.array or list): Message features array, or list [messages, logvar] for KL models
        sim (str): Simulation type - 'spring', 'r1', or 'r2' for different force laws
        dim (int): Number of most important message dimensions to analyse (default: 2)
        robust (bool): Whether to use robust regression (default: True). For robust regression, 
        we mask the 10% worst outliers when doing the linear regression and calculating R² scores
    
    Returns:
        tuple: (r2_scores, lin_combos, params1, params2, msgs_to_compare) where:
            - r2_scores: Tuple of R² scores for each message dimension
            - lin_combos: Tuple of linear combination predictions for each dimension
            - params1, params2: Lists of fitted parameters [bias, coeff_x, coeff_y] for each dimension
            - msgs_to_compare: Normalised message features used in fitting
    """

    if 'logvar0' in df.columns: #this means that we are doing the KL variation 
        logvar_array = msg_array[1]
        msg_array = msg_array[0]
        #in KL variation, important features have highest KL div
        KL_div =  (np.exp(logvar_array) + msg_array**2 - logvar_array)/2
        KL_mean = KL_div.mean(axis=0)
        most_important = np.argsort(KL_mean)[-dim:]
        msgs_to_compare = msg_array[:, most_important]

    else:
        #find important features
        msg_importance = msg_array.std(axis=0) #computes stds for each msg_dim over all datapoints
        most_important = np.argsort(msg_importance)[-dim:] #msg of highest std
        msgs_to_compare = msg_array[:, most_important]
    
    #normalise the message elements
    msgs_to_compare = (msgs_to_compare - np.average(msgs_to_compare, axis=0)) / np.std(msgs_to_compare, axis=0)

    dir_cols = ['dx', 'dy']
    bd_array = np.array(df['bd'])
    dir_array = np.array(df[dir_cols])
    #true force for spring (returns force for each edge x and y direction)
    #calculate forces based on simulation
    if sim == 'spring':
        expected_forces = -(bd_array - 1)[:, np.newaxis] * dir_array / bd_array[:, np.newaxis]
    elif sim == 'r2':
        m1 = df.m1.values
        m2 = df.m2.values
        expected_forces =  -m1[:, np.newaxis]*m2[:, np.newaxis]*dir_array / (bd_array[:, np.newaxis]**3)
    elif sim == 'r1':
        m1 = df.m1.values
        m2 = df.m2.values
        expected_forces = -m1[:, np.newaxis]*m2[:, np.newaxis]*dir_array / (bd_array[:, np.newaxis]**2)
    elif sim == 'charge':
        q1 = df.q1.values
        q2 = df.q2.values
        expected_forces =  q1[:, np.newaxis]*q2[:, np.newaxis]*dir_array / (bd_array[:, np.newaxis]**3)

    else:
        raise ValueError(f"Unknown simulation type: {sim}")

    #fit linear model with bias: msg1 = a0 + a1 * Fx + a2 * Fy
    #not robust to outliers. just uses SKLearn LinearRegression
    def linear_reg (expected_forces, msgs_to_compare):
        """
        Perform standard linear regression to fit forces to messages.
        
        Args:
            expected_forces (np.array): Expected force values [n_samples, 2]
            msgs_to_compare (np.array): Message features to fit [n_samples, 2]
        
        Returns:
            tuple: (lin_combo, params, biases) - predictions, coefficients, and intercepts
        """
        reg = LinearRegression()
        reg.fit(expected_forces, msgs_to_compare)
        lin_combo = reg.predict(expected_forces) 

        #return back the parameters too
        params = reg.coef_
        biases = reg.intercept_  

        return lin_combo, params, biases
    
    #masks residual outliers
    def percentile_sum(x):
        x = x.ravel()
        bot = x.min()
        top = np.percentile(x, 90)
        msk = (x >= bot) & (x <= top)
        frac_good = (msk).sum() / len(x)
        return x[msk].sum() / frac_good
    
    def percentile_mask(x):
        bot = x.min()
        top = np.percentile(x, 90)
        mask = (x >= bot) & (x <= top)
        return mask
    
    #linear reg
    def robust_linear_reg(expected_forces, msgs_to_compare):
        """
        Perform robust linear regression using percentile-based loss function.
        
        Args:
            expected_forces (np.array): Expected force values [n_samples, 2]
            msgs_to_compare (np.array): Message features to fit [n_samples, 2]
        
        Returns:
            tuple: (lin_combo, params, biases) - predictions, coefficients, and intercepts
        """
        
        def linear_transformation_2d(alpha):
            """
            Objective function for 2D linear transformation optimisation.
            
            Args:
                alpha (np.array): Parameters [a00, a01, bias0, a10, a11, bias1]
            
            Returns:
                float: Robust loss score
            """
            #first target: msgs_to_compare[:, 0]
            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
            #second target: msgs_to_compare[:, 1]  
            lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]
            
            score = (
                percentile_sum(np.square(msgs_to_compare[:, 0] - lincomb1)) +
                percentile_sum(np.square(msgs_to_compare[:, 1] - lincomb2))
            ) / 2.0
            
            return score
        
        def get_predictions_2d(alpha):
            """
            Get predictions from optimised parameters.
            
            Args:
                alpha (np.array): Optimised parameters [a00, a01, bias0, a10, a11, bias1]
            
            Returns:
                np.array: Predicted values [n_samples, 2]
            """
            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
            lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]
            return np.column_stack([lincomb1, lincomb2])
        
        #initialise parameters just at zero
        initial_params = np.ones(6)
        
        #optimise
        min_result = minimize(linear_transformation_2d, initial_params, method='Powell')
        print('robust score: ', min_result.fun/len(msgs_to_compare)) #lower is better
        
        #get params
        alpha = min_result.x
        params = np.array([[alpha[0], alpha[1]], 
                        [alpha[3], alpha[4]]]) 
        biases = np.array([alpha[2], alpha[5]])  
        
        #get linear combinations
        lin_combo = get_predictions_2d(alpha)
        
        return lin_combo, params, biases

    if robust:
        #this is the metholodology used originally in the paper but not mentioned
        lin_combo, params, biases = robust_linear_reg(expected_forces, msgs_to_compare)
        lin_combo1 = lin_combo[:, 0]
        lin_combo2 = lin_combo[:, 1]

        a0_1, a1_1, a2_1 = biases[0], params[0,0], params[0,1] #msg1 params
        a0_2, a1_2, a2_2 = biases[1], params[1,0], params[1,1] #msg2 params

        residuals1 = np.square(msgs_to_compare[:, 0] - lin_combo1)
        residuals2 = np.square(msgs_to_compare[:, 1] - lin_combo2)

        mask1 = percentile_mask(residuals1)
        mask2 = percentile_mask(residuals2)

        lin_combo1 = lin_combo1[mask1]
        lin_combo2 = lin_combo2[mask2]
        msgs_to_compare1 = msgs_to_compare[:,0][mask1]
        msgs_to_compare2 = msgs_to_compare[:,1][mask2]

        msg1_r2 = r2_score(msgs_to_compare1, lin_combo1)
        msg2_r2 = r2_score(msgs_to_compare2, lin_combo2)

        #make sure they have the same length by truncating to the shorter one
        if len(msgs_to_compare1)!=len(msgs_to_compare2):
            min_length = min(len(msgs_to_compare1), len(msgs_to_compare2))
            msgs_to_compare1 = msgs_to_compare1[:min_length]
            msgs_to_compare2 = msgs_to_compare2[:min_length]
            lin_combo1 = lin_combo1[:min_length]
            lin_combo2 = lin_combo1[:min_length]

        msgs_to_compare = np.column_stack([msgs_to_compare1, msgs_to_compare2])

    else:
        #does not mask outliers. produces worse fits and r2 scores obviously
        lin_combo, params, biases = linear_reg(expected_forces, msgs_to_compare)
        lin_combo1 = lin_combo[:, 0]
        lin_combo2 = lin_combo[:, 1]

        a0_1, a1_1, a2_1 = biases[0], params[0,0], params[0,1] #msg1 params
        a0_2, a1_2, a2_2 = biases[1], params[1,0], params[1,1] #msg2 params
        
        #calc r2 scores for both messages
        msg1_r2 = r2_score(msgs_to_compare[:, 0], lin_combo1)
        msg2_r2 = r2_score(msgs_to_compare[:, 1], lin_combo2)
    
    return (msg1_r2, msg2_r2), (lin_combo1, lin_combo2), [a0_1, a1_1, a2_1], [a0_2, a1_2, a2_2], msgs_to_compare

def plot_force_components(r2_scores, lin_combos, msgs_to_compare, params, save_path, epochs):
    """
    Create scatter plots comparing linear combinations of forces vs. actual message components.
    
    Args:
        r2_scores (tuple): R² scores for each message dimension
        lin_combos (tuple): Linear combination predictions for each dimension  
        msgs_to_compare (np.array): Actual normalized message features [n_samples, 2]
        params (tuple): Parameter lists [params1, params2] for each dimension
        save_path (str): Directory path to save the plot
        epochs (str): Epoch number for filename
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    for i in range(2):
        x = lin_combos[i]
        y = msgs_to_compare[:, i]
        r2 = r2_scores[i]
        
        #plot points
        ax[i].scatter(x, y, alpha=0.1, s=0.2, edgecolors = 'black', facecolors='grey')

        ax[i].set_xlim(-1,1)
        ax[i].set_ylim(-1,1)
        
        #y = x line 
        line_x = np.linspace(-1, 1, 100)
        line_y = line_x
        ax[i].plot(line_x, line_y, color='black', lw =0.5)

        title = f"{params[i][1]:.2g} Fx + {params[i][2]:.2g} Fy + {params[i][0]:.2g}"
        ax[i].grid(True)
        ax[i].set_xlabel("Linear combination of forces", fontsize = 12)
        ax[i].set_ylabel(f"Message element {i+1}", fontsize = 12)
        ax[i].set_title(f"R2 Score {r2:.5g}\n{title}", fontsize = 12)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/epoch_{epochs}.png', dpi = 300)
    return fig

def plot_linear_representation (model, input_data, sim='spring', model_type = 'L1', epochs = 30):
    """
    Generate complete linear representation analysis and visualisation for a trained model.
    
    Args:
        model: Trained PyTorch geometric model
        input_data (tuple): Tuple of (X_test, y_test) test data
        sim (str): Simulation type 
        model_type (str): Model type identifier for saving plots
        epochs (str/int): Epoch number for filenames
    
    Returns:
        tuple: (r2_scores, fig) where:
            - r2_scores: Tuple of R² scores for both message dimensions
            - fig: matplotlib figure object of the generated plots
    """

    #get message features
    print('Extracting message features.')
    df, msg_array = get_message_features(model, input_data)
    
    #fit the model using linear regression
    print('Fitting the forces to the two most important messages - robust.')
    r2_scores, lin_combos, params1, params2 ,msgs_to_compare = fit_messages(df, msg_array, sim, robust=True)
    
    #plot linear representation of messages
    save_path = f'linrepr_plots/{sim}/{model_type}'
    os.makedirs(save_path, exist_ok=True)
    fig = plot_force_components(r2_scores, lin_combos, msgs_to_compare, (params1, params2), save_path, epochs)
    
    print(f"ROBUST Message 1 R2 Score: {r2_scores[0]:.5g}")
    print(f"ROBUST Message 2 R2 Score: {r2_scores[1]:.5g}")

    print('Fitting the forces to the two most important messages - including outliers.')
    r2_scores_w_outliers, lin_combos, params1, params2 ,msgs_to_compare = fit_messages(df, msg_array, sim, robust=False)

    save_path = f'linrepr_plots/{sim}/{model_type}/with_outliers'
    os.makedirs(save_path, exist_ok=True)
    fig = plot_force_components(r2_scores_w_outliers, lin_combos, msgs_to_compare, (params1, params2), save_path, epochs)
    
    return (r2_scores, r2_scores_w_outliers), fig

def pruning_r2_scores(input_data, sim='charge', num_epoch = 100):
    schedules = ['exp', 'linear', 'cosine']
    end_epoch_fracs = [0.65, 0.75, 0.85]

    save_path = f'linrepr_plots/pruning_experiments/{sim}'
    os.makedirs(save_path, exist_ok=True)
    r2_scores = {}
    r2_scores_w_outliers = {}
    for schedule in schedules:
        for frac in end_epoch_fracs:
            model = load_pruning_models(dataset_name=sim, pruning_schedule=schedule, end_epoch_frac=frac, num_epoch = num_epoch)
            print(f'Extracting message features for {schedule} schedule and {frac} end_epoch_frac.')
            df, msg_array = get_message_features(model, input_data)

            print('Fitting the forces to the two most important messages - robust.')
            r2_score, lin_combos,params1, params2 ,msg_to_compare = fit_messages(df, msg_array, sim, robust = True)
            new_save_path = f'{save_path}/{schedule}/{frac}'
            os.makedirs(new_save_path, exist_ok=True)
            print('Plotting.')
            plot_force_components(r2_score, lin_combos, msg_to_compare, (params1, params2), new_save_path, num_epoch)

            r2_scores[f'{schedule}_{frac}'] = {"message_1_r2":r2_score[0], "message_2_r2":r2_score[1]}
            print(f"R2 Scores (robust): {r2_score[0]}, {r2_score[1]}")

            print('Fitting the forces to the two most important messages - with outliers.')
            r2_score, lin_combos,params1, params2 ,msg_to_compare = fit_messages(df, msg_array, sim, robust = False)
            new_save_path = f'{save_path}/{schedule}/{frac}/with_outliers'
            os.makedirs(new_save_path, exist_ok=True)
            print('Plotting.')
            plot_force_components(r2_score, lin_combos, msg_to_compare, (params1, params2), new_save_path, num_epoch)

            r2_scores_w_outliers[f'{schedule}_{frac}'] = {"message_1_r2":r2_score[0], "message_2_r2":r2_score[1]}
            print(f"R2 Scores (for all data): {r2_score[0]}, {r2_score[1]}")

    
    print('Finished and saving R2 scores.')


    #save the r2 scores in a .JSON
    results_file = f'{save_path}/r2_scores_epoch_{num_epoch}.json'
    with open(results_file, 'w') as f:
        json.dump(r2_scores, f, indent=2)

    results_file = f'{save_path}/r2_scores_w_outliers_epoch_{num_epoch}.json'
    with open(results_file, 'w') as f:
        json.dump(r2_scores_w_outliers, f, indent=2)

def main():
    """
    Main function to run linear representation analysis from command line arguments.
    
    Supports analysing either a single model type or all model types ['standard', 'bottleneck', 'L1', 'KL'] for a system.
    When analysing all models, saves R² scores to a JSON file for comparison.
    
    Command line arguments:
        --dataset_name: Name of the dataset/simulation type
        --model_type: Model type to analyse, or 'all' for all model types
        --num_epoch: Epoch number of the trained model to load
        --cutoff: Optional limit on number of test samples to use (default: 0, use all)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--num_epoch", type=str, required=True)
    parser.add_argument("--cutoff", type=int, default=0)

    args = parser.parse_args()
   
   #load the test data
    _, _, test_data = load_data(args.dataset_name)
    X_test, y_test = test_data

    cutoff = args.cutoff #option to use a smaller subset of the test set 
    if cutoff != 0:
        X_test = X_test[:cutoff]
        y_test = y_test[:cutoff]

    #if you want to plot all the model types in the system...
    if args.model_type == 'all':
        model_types = ['standard', 'bottleneck', 'L1', 'KL', 'pruning']
        all_results = {}
        for model_type in model_types:
            model = load_model(dataset_name=args.dataset_name, model_type=model_type, num_epoch=args.num_epoch)
            print(f"\nProcessing model type: {model_type}")
            r2_scores_tuple, fig = plot_linear_representation(model, (X_test, y_test), sim=args.dataset_name, model_type=model_type, epochs=args.num_epoch)
            r2_scores = r2_scores_tuple[0]
            r2_scores_w_outliers = r2_scores_tuple[1]
            # Store results
            all_results[model_type] = {
                'robust message_1_r2': float(r2_scores[0]),
                'robust message_2_r2': float(r2_scores[1]),
                'message_1_r2': float(r2_scores_w_outliers[0]),
                'message_2_r2': float(r2_scores_w_outliers[1])
            }

        save_path = f'linrepr_plots/{args.dataset_name}'
        os.makedirs(save_path, exist_ok=True)

        #save the r2 scores in a .JSON
        results_file = f'{save_path}/r2_scores_epoch_{args.num_epoch}.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    #just get the r2 scores for the pruning experiments
    elif args.model_type == 'pruning_experiments':
        pruning_r2_scores((X_test, y_test), sim=args.dataset_name, num_epoch=args.num_epoch)
    
    #if you want to plot only a single model type
    else:
        model = load_model(dataset_name=args.dataset_name, model_type=args.model_type, num_epoch=args.num_epoch)
        plot_linear_representation(model, (X_test, y_test), sim=args.dataset_name, model_type=args.model_type, epochs = args.num_epoch)  
        


if __name__ == "__main__":
    main()


