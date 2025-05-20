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
                messages = messages[:, 0::2] #for KL model, take means of the messages
            message_info = torch.cat((source_nodes, target_nodes, messages), dim=1)
            all_message_info.append(message_info) #collect message info for all edges in each graph
    
    #combine message info
    message_info = torch.cat(all_message_info, dim=0) #shape [num_edges, num_features*2 + msg_dim]
    message_info = message_info.numpy()
    
    #create a dataframe to store node info
    node_info = ['x', 'y', 'nu_x', 'nu_y', 'q', 'm'] #only in 2d for now
    source_cols = [f'{f}1' for f in node_info]
    target_cols = [f'{f}2' for f in node_info]

    msg_dim = messages.shape[-1] #this is 100 usually
    message_cols = [f'e{i}' for i in range(msg_dim)] 
    columns = source_cols + target_cols + message_cols
    
    #make a dataframe with all the relevent information 
    df = pd.DataFrame(message_info, columns=columns)
    
    #calculate distances
    df['dx'] = df.x1 - df.x2
    df['dy'] = df.y1 - df.y2
    df['r'] = np.sqrt(df.dx**2 + df.dy**2)
    df['bd'] = df.r + 1e-2 #TODO: Double check what exactly this is. Related to spring force
    
    #calculate relative velocities 
    df['dnu_x'] = df.nu_x1 - df.nu_x2
    df['dnu_y'] = df.nu_y1 - df.nu_y2
    df['dnu'] = np.sqrt(df.dnu_x**2 + df.dnu_y**2)
    
    #extract just the message features
    msg_array = df[message_cols].values #of shape [num_datapoints*(2*num_edges), msg_dim]
    
    return df, msg_array

def fit_messages(df, msg_array, sim='spring', dim=2):

    #find important features
    msg_importance = msg_array.std(axis=0) #computes stds for each msg_dim over all datapoints
    most_important = np.argsort(msg_importance)[-dim:] #msg of highest std
    msgs_to_compare = msg_array[:, most_important]
    
    #normalise the message elements
    msgs_to_compare = (msgs_to_compare - np.average(msgs_to_compare, axis=0)) / np.std(msgs_to_compare, axis=0)
    
    #calculate forces based on simulation
    if sim == 'spring':
        dir_cols = ['dx', 'dy']
        bd_array = np.array(df['bd'])
        dir_array = np.array(df[dir_cols])
        #true force for spring (returns force for each edge x and y direction)
        expected_forces = -(bd_array - 1)[:, np.newaxis] * dir_array / bd_array[:, np.newaxis]
    else:
        raise ValueError(f"Unknown simulation type: {sim}")

    #fit linear model with bias: msg1 = a0 + a1 * Fx + a2 * Fy
    reg = LinearRegression()
    reg.fit(expected_forces, msgs_to_compare)

    lin_combo = reg.predict(expected_forces) 
    lin_combo1 = lin_combo[:, 0]
    lin_combo2 = lin_combo[:, 1]

    #return back the parameters too
    params = reg.coef_
    biases = reg.intercept_  
    a0_1, a1_1, a2_1 = biases[0], params[0,0], params[0,1] #msg1 params
    a0_2, a1_2, a2_2 = biases[1], params[1,0], params[1,1] #msg2 params

    #get a score depending on similarity to the actual msg
    def percentile_mse(a, b):
        diffs = np.square(a - b)
        return np.mean([np.mean(diffs[:, 0]), np.mean(diffs[:, 1])]) #ave for both msg
    score = percentile_mse(msgs_to_compare, lin_combo)
    print(f'linear fit mse score: {score}')
    
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
        
        #set limits
        xlim = np.array([np.percentile(x, q) for q in [10, 90]])
        ylim = np.array([np.percentile(y, q) for q in [10, 90]])
        
        #add padding to the limits (same as in colab)
        xlim[0], xlim[1] = xlim[0] - (xlim[1] - xlim[0])*0.05, xlim[1] + (xlim[1] - xlim[0])*0.05
        ylim[0], ylim[1] = ylim[0] - (ylim[1] - ylim[0])*0.05, ylim[1] + (ylim[1] - ylim[0])*0.05
        
        ax[i].set_xlim(xlim)
        ax[i].set_ylim(ylim)
        
        #y = x line
        line_x = np.linspace(xlim[0], xlim[1], 100)
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


