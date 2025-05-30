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
from utils import load_data
import argparse
from scipy.optimize import minimize

def get_message_features(model, input_data):
    model.eval()

    X_test, y_test = input_data
    edge_index = get_edge_index(X_test.shape[1])

    # test_idxes = np.random.randint(0, len(X_test), 1000)
    # dataset = DataLoader(
    #     [Data(x=X_test[i], edge_index=edge_index, y=y_test[i]) for i in test_idxes],
    #     batch_size=len(X_test),
    #     shuffle=False)
    
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

    if 'logvar0' in df.columns: #this means that we are doing the KL variation 
        logvar_array = msg_array[1]
        msg_array = msg_array[0]
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
        print('R1 FORCE')
    else:
        raise ValueError(f"Unknown simulation type: {sim}")

    #fit linear model with bias: msg1 = a0 + a1 * Fx + a2 * Fy
    def linear_reg (expected_forces, msgs_to_compare):
        reg = LinearRegression()
        reg.fit(expected_forces, msgs_to_compare)
        lin_combo = reg.predict(expected_forces) 

        #return back the parameters too
        params = reg.coef_
        biases = reg.intercept_  


        return lin_combo, params, biases
    
    def robust_linear_reg(expected_forces, msgs_to_compare):
    
        def percentile_sum(x):
            x = x.ravel()
            bot = x.min()
            top = np.percentile(x, 90)
            msk = (x >= bot) & (x <= top)
            frac_good = (msk).sum() / len(x)
            return x[msk].sum() / frac_good
        
        def linear_transformation_2d(alpha):
            """
            Exactly matching your original implementation structure.
            alpha = [a00, a01, bias0, a10, a11, bias1]
            """
            # First target: msgs_to_compare[:, 0]
            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
            # Second target: msgs_to_compare[:, 1]  
            lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]
            
            score = (
                percentile_sum(np.square(msgs_to_compare[:, 0] - lincomb1)) +
                percentile_sum(np.square(msgs_to_compare[:, 1] - lincomb2))
            ) / 2.0
            
            return score
        
        def get_predictions_2d(alpha):
            """Get predictions from optimized parameters"""
            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
            lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]
            return np.column_stack([lincomb1, lincomb2])
        
        # Initialize parameters: [a00, a01, bias0, a10, a11, bias1]
        initial_params = np.ones(6)
        
        # Optimize using Powell method
        min_result = minimize(linear_transformation_2d, initial_params, method='Powell')
        print('robust score: ', min_result.fun/len(msgs_to_compare))
        
        # Extract optimized parameters
        alpha = min_result.x
        
        # Reshape to match sklearn format
        params = np.array([[alpha[0], alpha[1]], 
                        [alpha[3], alpha[4]]]) 
        biases = np.array([alpha[2], alpha[5]])  
        
        # Get final predictions
        lin_combo = get_predictions_2d(alpha)
        
        return lin_combo, params, biases

    def robust_r2_score(y_true, y_pred):
        """
        Calculate R² using only the best-fitting 90% of points (from min to 90th percentile of residuals)
        This matches the robust regression approach.
        """
        residuals = np.square(y_true - y_pred)
        
        # Use same percentile logic as robust regression
        bot = residuals.min()
        top = np.percentile(residuals, 90)
        mask = (residuals >= bot) & (residuals <= top)
        
        # Calculate R² only on the masked (good) points
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        
        # Standard R² formula on masked data
        ss_res = np.sum((y_true_masked - y_pred_masked) ** 2)
        ss_tot = np.sum((y_true_masked - np.mean(y_true_masked)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)

    if robust:
    
        lin_combo, params, biases = robust_linear_reg(expected_forces, msgs_to_compare)
        lin_combo1 = lin_combo[:, 0]
        lin_combo2 = lin_combo[:, 1]

        a0_1, a1_1, a2_1 = biases[0], params[0,0], params[0,1] #msg1 params
        a0_2, a1_2, a2_2 = biases[1], params[1,0], params[1,1] #msg2 params

        msg1_r2 = robust_r2_score(msgs_to_compare[:, 0], lin_combo1)
        msg2_r2 = robust_r2_score(msgs_to_compare[:, 1], lin_combo2)

    else:
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
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    for i in range(2):
        x = lin_combos[i]
        y = msgs_to_compare[:, i]
        r2 = r2_scores[i]
        
        #plot points
        ax[i].scatter(x, y, alpha=0.1, s=0.1, color='blue')
        
        # #set limits
        # xlim = np.array([np.percentile(x, q) for q in [10, 90]])
        # ylim = np.array([np.percentile(y, q) for q in [10, 90]])
        
        # #add padding to the limits (same as in colab)
        # xlim[0], xlim[1] = xlim[0] - (xlim[1] - xlim[0])*0.05, xlim[1] + (xlim[1] - xlim[0])*0.05
        # ylim[0], ylim[1] = ylim[0] - (ylim[1] - ylim[0])*0.05, ylim[1] + (ylim[1] - ylim[0])*0.05
        
        # ax[i].set_xlim(xlim)
        # ax[i].set_ylim(ylim)

        ax[i].set_xlim(-1,1)
        ax[i].set_ylim(-1,1)
        
        #y = x line 
        # line_x = np.linspace(xlim[0], xlim[1], 100)
        line_x = np.linspace(-1, 1, 100)
        line_y = line_x
        ax[i].plot(line_x, line_y, color='black')

        title = f"{params[i][1]:.2g} Fx + {params[i][2]:.2g} Fy + {params[i][0]:.2g}"
        ax[i].grid(True)
        ax[i].set_xlabel("Linear combination of forces")
        ax[i].set_ylabel(f"Message element {i+1}")
        ax[i].set_title(f"R2 Score {r2:.5g}\n{title}")
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/epoch_{epochs}.png', dpi = 300)
    return fig

def plot_linear_representation (model, input_data, sim='spring', model_type = 'L1', epochs = 30):

    #get message features
    print('Extracting message features.')
    df, msg_array = get_message_features(model, input_data)
    
    #fit the model using linear regression
    print('Fitting the forces to the two most important messages.')
    r2_scores, lin_combos, params1, params2 ,msgs_to_compare = fit_messages(df, msg_array, sim)
    
    #plot linear representation of messages
    save_path = f'linrepr_plots/{sim}/{model_type}'
    os.makedirs(save_path, exist_ok=True)
    fig = plot_force_components(r2_scores, lin_combos, msgs_to_compare, (params1, params2), save_path, epochs)
    
    print(f"Message 1 R2 Score: {r2_scores[0]:.5g}")
    print(f"Message 2 R2 Score: {r2_scores[1]:.5g}")
    
    return r2_scores, fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--num_epoch", type=str, required=True)
    parser.add_argument("--cutoff", type=int, default=0)

    args = parser.parse_args()

    model = load_model(dataset_name=args.dataset_name, model_type=args.model_type, num_epoch=args.num_epoch)
    _, _, test_data = load_data(args.dataset_name)
    X_test, y_test = test_data
    cutoff = args.cutoff #option to use a smaller subset of the test set 
    if cutoff != 0:
        X_test = X_test[:cutoff]
        y_test = y_test[:cutoff]
    plot_linear_representation(model, (X_test, y_test), sim=args.dataset_name, model_type=args.model_type, epochs = args.num_epoch)


if __name__ == "__main__":
    main()


