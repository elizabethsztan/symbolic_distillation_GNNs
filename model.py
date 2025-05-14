import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
import wandb
import os
from datetime import datetime
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

def get_edge_index(num_nodes): #edge index for fully connected graph
    idx = torch.arange(num_nodes)
    edge_index = torch.cartesian_prod(idx, idx)
    edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]]
    return edge_index.t() #output dimension [2, E]

def load_data(path): #load dataset 
    data = torch.load(f"{path}.pt")
    return data['X'], data['y']

class NBodyDataset(Dataset):
    """
    Create pytorch dataset class for our simulation dataset.
    """
    def __init__(self, data, targets):
        """
        Args:
            data (torch.Tensor): shape is [no_datapoints, no_nodes, no_node_features]
            targets (torch.Tensor): shape is [no_datapoints, no_nodes, 2d_acceleration]
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data) #how many samples in dataset to use later for batching

    def __getitem__(self, idx):
        nodes = self.data[idx]  # shape: [no_nodes, no_node_features] 
        acc = self.targets[idx]
        return nodes, acc #inputs and target variables

class NBodyGNN(MessagePassing):
    def __init__(self, node_dim = 6, acc_dim = 2, hidden_dim = 300):
        """ 
        N-body graph NN class.

        Args:
            node_dim (int): dimensionality of the node (in 2d case it is 6)
            acc_dim (int): dimensionality of the output of the network, which are accelerations so in 2d = 2
            hidden_dim (int): hidden layer dimensions - same for both edge and node MLPs

        """
    
        super().__init__(aggr='add')
         
        #edge model MLP
        self.edge_model = nn.Sequential(
            nn.Linear(2*node_dim, hidden_dim), #inputs = node information for two nodes
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 100) #output message features of dimension 100 for standard and L1 model 
        )
        #node model MLP
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + 100, hidden_dim), #inputs = sum of outputs of edge model and node features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, acc_dim) #output = predicted acc
        )

        self.message_features = None
        self.combined_messages = None

        self.node_dim_ = node_dim
        self.acc_dim_ = acc_dim
        self.hidden_dim_ = hidden_dim

    def message(self, x_i, x_j):
        x = torch.cat((x_i, x_j), dim = -1) #concat along final dimension = features. shape is [batch_size, no_edges, no_features]
        message_features = self.edge_model(x) #put thru MLP. MLP only transforms the feature (last) dimension
        
        if self.message_features == None:
            self.message_features = message_features
        
        else:
            self.message_features = torch.cat([self.message_features, message_features], dim=0)
        
        return message_features

    def forward(self, x, edge_index): 
        """Forward pass of this network

        Args:
            x (torch.Tensor): shape is [batch_size, no_nodes, no_node_features]
            edge_index (torch.Tensor): shape is [2, no_edges]
        """
        self.message_features = None
        edge_message =  self.propagate(edge_index, x = (x,x)) #use same feature matrix for both source and target nodes (undirected network)
        #x is shape [batch_size, no_nodes, no_features]
        acc_pred = self.node_model(torch.cat([x, edge_message], dim = -1)) #predict accelerations

        return acc_pred
    
    def get_messages(self):
        return self.message_features

def train(model, train_data, val_data, num_epoch, dataset_name = 'r2', model_type = 'standard', save = True, wandb_log = True):
    """ Train the GNN

    Args:
        model (NBodyGNN): the model class
        train_data (tuple): contains (input_data, accelerations)
        val_data (tuple): contains (input_data, accelerations)
        num_epoch (int): number of epochs to train on

    Returns:
        model (NBodyGNN object): final trained model
    """
    #training data
    input_data, acc = train_data

    if model.node_dim_ != input_data.shape[-1]:
        assert 'Mismatch in model and data node/particle dimensions.'
        
    dataset = NBodyDataset(input_data, acc)   
    batch_size = int(64*(4/input_data.shape[1])**2) #batch depends on num_nodes
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #validation data
    input_val, acc_val = val_data
    val_dataset = NBodyDataset(input_val, acc_val)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # node_dim = input_data.shape[-1]
    num_nodes = input_data.shape[1]
    acc_dim = acc.shape[-1]

    if wandb_log:
        mode = 'online'
    else:
        mode = 'disabled'

    wandb.init(project = f'{dataset_name}',
            name = f'{model_type}',
            config={'num_epochs': num_epoch, 'num_nodes': num_nodes, 'dim': acc_dim}, 
            mode = mode
            )

    #optimiser
    init_lr = 1e-3
    optimiser = torch.optim.Adam(model.parameters(), init_lr, weight_decay=1e-8) #L2 regulariser on params

    #load onto GPU
    accelerator = Accelerator()
    model, dataloader, val_dataloader, optimiser = accelerator.prepare(model, dataloader, val_dataloader, optimiser)

    print(f'Running on {accelerator.device}.')

    #learning rate scheduler
    scheduler = OneCycleLR(optimiser, max_lr = init_lr, steps_per_epoch = len(dataloader), epochs = num_epoch, final_div_factor = 1e5)
    criterion = nn.L1Loss() #MAE loss

    #regularisation 
    if model_type == 'standard':
        def regularisation(model):
            return 0.0
        
    elif model_type == 'L1':
        def regularisation (model):
            message_features = model.get_messages()
            return torch.mean(torch.abs(message_features))


    edge_index = get_edge_index(input_data.shape[1]).to(accelerator.device) #this never changes so we only calc once

    losses = []
    val_losses = []

    for epoch in range (num_epoch):
        total_loss = 0 #loss tracking
        
        #set model in training mode
        model.train()

        pbar = tqdm(dataloader, desc=f"Epoch: {epoch+1}/{num_epoch}")
        for nodes, acc in pbar:

            #training
            optimiser.zero_grad()

            acc_pred = model(nodes, edge_index) #automatically calls model.forward()
            
            loss = criterion(acc_pred, acc) + regularisation(model)

            accelerator.backward(loss)
            optimiser.step()
            scheduler.step()

            total_loss += loss.item()
            #pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        #validation once per epoch
        model.eval()
        val_loss = 0
        with torch.no_grad(): #stop computing gradients
            for nodes, acc in val_dataloader:
                acc_pred = model(nodes, edge_index) #run forward pass thru model
                val_loss += criterion(acc_pred, acc).item()

        avg_loss = total_loss/len(dataloader)
        avg_val_loss = val_loss/len(val_dataloader)
        print(f'training loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}')

        #log to wandb
        wandb.log({
            'epoch': epoch, 
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        })

        #save losses
        losses.append(avg_loss)
        val_losses.append(avg_val_loss)

    wandb.finish()

    if save:
        save_path = f'{script_dir}/model_weights/{dataset_name}/{model_type}'
        os.makedirs(save_path, exist_ok=True)
        #save model weights
        checkpoint = {
            'model_state_dict': model.state_dict(), 
            'node_dim': model.node_dim_,
            'acc_dim': model.acc_dim_, 
            'hidden_dim': model.hidden_dim_
        }
        torch.save(checkpoint, f'{save_path}/epoch_{num_epoch}_model.pth')
        #log training run in json file
        metrics = {'datetime':datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   'dataset_name': dataset_name,
                   'model_type': model_type,
                   'epochs': num_epoch,
                   'num_nodes': num_nodes,
                   'dimensions': acc_dim, 
                   'train_loss': losses,
                   'val_loss': val_losses
                   }
        with open(f'{save_path}/epoch_{num_epoch}_metrics.json', 'w') as json_file:
            json.dump(metrics, json_file, indent = 4)

    return model

def load_model(dataset_name, model_type, num_epoch):
    checkpoint = torch.load(f'{script_dir}/model_weights/{dataset_name}/{model_type}/epoch_{num_epoch}_model.pth')
    #create a new model
    model = NBodyGNN(
        node_dim=checkpoint['node_dim'],
        acc_dim=checkpoint['acc_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )

    #load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Model loaded successfully.')
    
    return model


def test(model, test_data, model_type = 'standard'):

    input_data, acc = test_data
    dataset = NBodyDataset(input_data, acc)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    edge_index = get_edge_index(input_data.shape[1])

    if model_type == 'standard':
        def regularisation(model):
            return 0.0
        
    elif model_type == 'L1':
        def regularisation (model):
            message_features = model.get_messages()
            return torch.mean(torch.abs(message_features))

    criterion = nn.L1Loss()
    
    model.eval()
    loss = 0
    with torch.no_grad():
        for nodes, acc in dataloader:
            acc_pred = model(nodes, edge_index)
            loss += criterion(acc_pred, acc).item() + regularisation(model)
    
    avg_loss = loss/len(dataloader)
    print('Avg loss: ', avg_loss)

    return avg_loss