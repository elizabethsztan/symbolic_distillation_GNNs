import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import wandb
import os
from datetime import datetime 
import json
import numpy as np
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = os.getcwd()

def get_edge_index(num_nodes):
    """
    Generate edge indices for a fully connected graph.
    
    Creates a complete graph where every node is connected to every other node
    (excluding self-connections). All particles in our system interact.
    
    Args:
        num_nodes (int): Number of nodes/particles in the graph.
        
    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges] where num_edges = 
                     num_nodes * (num_nodes - 1). The first row contains source node 
                     indices and the second row contains target node indices.
                     
    Example:
        >>> edge_index = get_edge_index(4)
        >>> print(edge_index)
        tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
        # This represents edges: 0->1, 0->2, 0->3, 1->0, 1->2, 1->3, 2->0, 2->1, 2->3, 3->0, 3->1, 3->2
    """
    idx = torch.arange(num_nodes)
    edge_index = torch.cartesian_prod(idx, idx)
    edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]]
    return edge_index.t() #output dimension [2, num_edges]

class NBodyGNN(MessagePassing):
    def __init__(self, node_dim = 6, acc_dim = 2, hidden_dim = 300):
        """
            Graph Neural Network for N-body physics simulations using message passing.
            
            This class implements a standard message-passing neural network designed for 
            predicting accelerations in N-body systems (e.g., gravitational or electrostatic 
            interactions between particles). The network uses an edge model to compute 
            messages between particles and a node model to aggregate these messages and 
            predict accelerations.
            
            The architecture consists of:
            - Edge model: MLP that processes pairs of node features to generate messages
            - Node model: MLP that combines node features with aggregated messages to predict accelerations
            
            Args:
                node_dim (int, optional): Dimensionality of node features. For 2D systems, 
                    this is typically 6 (x, y, velocity_x, velocity_y, charge, mass). 
                    Defaults to 6.
                acc_dim (int, optional): Dimensionality of acceleration output. For 2D 
                    systems, this is 2 (acceleration_x, acceleration_y). Defaults to 2.
                hidden_dim (int, optional): Hidden layer dimensions for both edge and node 
                    MLPs. Defaults to 300.
            
            Attributes:
                model_type_ (str): Model type identifier, set to 'standard'.
                message_dim_ (int): Dimensionality of messages passed between nodes (100).
                edge_model (nn.Sequential): MLP for processing edge features.
                node_model (nn.Sequential): MLP for processing node features and messages.
                node_dim_ (int): Stored node dimension.
                acc_dim_ (int): Stored acceleration dimension.
                hidden_dim_ (int): Stored hidden dimension.
            
            Example:
                >>> model = NBodyGNN(node_dim=6, acc_dim=2, hidden_dim=256)
            
            Note:
                This class inherits from PyTorch Geometric's MessagePassing and uses 
                'add' aggregation to sum messages from neighboring nodes.
            """
    
        super().__init__(aggr='add')

        self.model_type_ = 'standard'
        self.message_dim_ = 100
        
        #edge model MLP
        self.edge_model = nn.Sequential(
            nn.Linear(2*node_dim, hidden_dim), #inputs = node information for two nodes
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, self.message_dim_) #output message features of dimension 100 for standard and L1 model 
        )

        #node model MLP
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + self.message_dim_, hidden_dim), #inputs = sum of outputs of edge model and node features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, acc_dim) #output = predicted acc
        )

        self.node_dim_ = node_dim
        self.acc_dim_ = acc_dim
        self.hidden_dim_ = hidden_dim

    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, node_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            
        Returns:
            torch.Tensor: Predicted accelerations of shape [num_nodes, acc_dim].
        """
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        """
        Compute messages between connected nodes.
        
        Args:
            x_i (torch.Tensor): Source node features of shape [num_edges, node_dim].
            x_j (torch.Tensor): Target node features of shape [num_edges, node_dim].
            
        Returns:
            torch.Tensor: Messages of shape [num_edges, message_dim_].
        """
        tmp = torch.cat([x_i, x_j], dim=1)
        messages = self.edge_model(tmp)
        return messages

    def update(self, aggr_out, x=None):
        """
        Update node representations using aggregated messages.
        
        Args:
            aggr_out (torch.Tensor): Aggregated messages of shape [num_nodes, message_dim_].
            x (torch.Tensor): Original node features of shape [num_nodes, node_dim].
            
        Returns:
            torch.Tensor: Updated node features (predicted accelerations) of shape [num_nodes, acc_dim].
        """
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_model(tmp)
    
    def get_predictions(self, data, augment, augmentation = 3):
        x, edge_index = data.x, data.edge_index
        if augment:
            augmentation = torch.randn(1, self.acc_dim_)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(self.acc_dim_).to(x.device), augmentation) #add noise to the position coordinates

        return self.propagate(edge_index,size=(x.size(0), x.size(0)), x=x)

    def loss(self, data, augment = True):
        acc_pred = self.get_predictions(data, augment)
        loss = torch.sum(torch.abs(data.y - acc_pred)) #MAE loss 

        return loss
    
class BottleneckGN(NBodyGNN):
    """
    Bottleneck Graph Neural Network with constrained message dimensionality.
    
    Inherits from NBodyGNN but constrains message dimensions to match the 
    acceleration dimension (typically 2D), creating an information bottleneck 
    that forces the model to learn more efficient representations.
    
    Args:
        node_dim (int): Dimensionality of node features (default: 6)
        acc_dim (int): Dimensionality of acceleration output (default: 2)
        hidden_dim (int): Hidden layer dimensions for MLPs (default: 300)
        
    Attributes:
        model_type_ (str): Set to 'bottleneck'
        message_dim_ (int): Set to acc_dim to create information bottleneck
    """
    def __init__(self, node_dim = 6, acc_dim = 2, hidden_dim = 300):
        super().__init__(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
        self.model_type_ = 'bottleneck'
        self.message_dim_ = acc_dim

        self.edge_model = nn.Sequential(
            nn.Linear(2*node_dim, hidden_dim), #inputs = node information for two nodes
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, self.message_dim_) #output message features of dimension 2 for bottleneck 2d
        )

        #node model MLP
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + self.message_dim_, hidden_dim), #inputs = sum of outputs of edge model and node features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, acc_dim) #output = predicted acc
        )

        self.node_dim_ = node_dim
        self.acc_dim_ = acc_dim
        self.hidden_dim_ = hidden_dim

class KLGN(NBodyGNN):
    """
    Kullback-Leibler regularized Graph Neural Network with variational messages.
    
    Implements variational message passing where messages are sampled from 
    learned distributions. Uses KL divergence regularisation to encourage 
    sparse and interpretable message representations.
    
    Args:
        node_dim (int): Dimensionality of node features (default: 6)
        acc_dim (int): Dimensionality of acceleration output (default: 2)
        hidden_dim (int): Hidden layer dimensions for MLPs (default: 300)
        
    Attributes:
        model_type_ (str): Set to 'KL'
        
    Notes:
        - Edge model outputs both mean and log-variance for each message dimension
        - Uses reparameterisation trick during training for backpropagation as we sample msgs
        - Returns mean values during evaluation (no sampling)
    """
    def __init__(self, node_dim = 6, acc_dim = 2, hidden_dim = 300):
        super().__init__(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)

        self.model_type_ = 'KL'
        self.edge_model = nn.Sequential(
            nn.Linear(2*node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, int(self.message_dim_ * 2)))
        
    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        messages = self.edge_model(tmp)
        #for KL model you need to sample messages
        mu = messages[:, 0::2] #all evens are the mus
        logvar = messages[:,1::2]#all odds are the logvars

        if self.training:  #only sample during training
            noise = torch.randn(mu.shape, device=mu.device, requires_grad=False)
            std = torch.exp(logvar/2)
            messages = mu + noise * std #reparameterisation trick
            return messages
        else:
            return mu
        
    def loss(self, data, augment = True):
        acc_pred = self.get_predictions(data, augment)
        loss = torch.sum(torch.abs(data.y - acc_pred)) #MAE loss 

        source_nodes = data.x[data.edge_index[0]]
        target_nodes = data.x[data.edge_index[1]]

        tmp = torch.cat([source_nodes, target_nodes], dim=1)
        messages = self.edge_model(tmp) 
        #get mus and logvars
        mu = messages[:, 0::2]
        logvar = messages[:,1::2]

        reg_str = 1 #regularisation strength
        unscaled_reg = reg_str * torch.sum(torch.exp(logvar) + mu**2 - logvar)/2 #reg term is kL div

        return loss, unscaled_reg
    
class L1GN(NBodyGNN):
    """
    L1-regularised Graph Neural Network for sparse message learning.
    
    Applies L1 regularisation to message activations to encourage sparsity
    and interpretability in the learned message representations.
    
    Args:
        node_dim (int): Dimensionality of node features (default: 6)
        acc_dim (int): Dimensionality of acceleration output (default: 2)
        hidden_dim (int): Hidden layer dimensions for MLPs (default: 300)
        
    Attributes:
        model_type_ (str): Set to 'L1'
        
    Notes:
        - Uses regularisation strength of 1e-2 for L1 penalty
        - Loss function returns both prediction loss and L1 regularisation term
    """
    def __init__(self, node_dim = 6, acc_dim = 2, hidden_dim = 300):
        super().__init__(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
        self.model_type_ = 'L1'

    def loss(self, data, augment = True):
        acc_pred = self.get_predictions(data, augment)
        loss = torch.sum(torch.abs(data.y - acc_pred)) #MAE loss 

        source_nodes = data.x[data.edge_index[0]]
        target_nodes = data.x[data.edge_index[1]]

        messages = self.message(source_nodes, target_nodes) #collect the messages between all nodes
        reg_str = 1e-2 #regularisation strength
        unscaled_reg = reg_str * torch.sum(torch.abs(messages))

        return loss, unscaled_reg #later scale it according to batch size and num_nodes
    
class PruningGN(NBodyGNN):
    """
    Pruning Graph Neural Network with dynamic message dimension reduction.
    
    Implements structured pruning of message dimensions during training based
    on message importance (measured by standard deviation). Gradually reduces
    message dimensionality according to a specified schedule.
    
    Args:
        node_dim (int): Dimensionality of node features (default: 6)
        acc_dim (int): Dimensionality of acceleration output (default: 2)
        hidden_dim (int): Hidden layer dimensions for MLPs (default: 300)
        
    Attributes:
        model_type_ (str): Set to 'pruning'
        initial_message_dim (int): Starting message dimension (100)
        target_message_dim (int): Final message dimension (acc_dim)
        current_message_dim (int): Current active message dimension
        pruning_schedule (dict): Epoch -> target dimension mapping
        pruning_mask (torch.Tensor): Boolean mask for active message dimensions
    """
    def __init__(self, node_dim = 6, acc_dim = 2, hidden_dim = 300):
        super().__init__(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
        self.model_type_ = 'pruning'

        self.initial_message_dim = self.message_dim_
        self.target_message_dim = acc_dim #towards the end of training, prune all except 2 messages
        self.current_message_dim = self.message_dim_
        self.pruning_schedule = None 
        self.pruning_mask = torch.ones(self.message_dim_, dtype=torch.bool) #this is set during training

    def set_pruning_schedule(self, total_epochs, schedule='exp', end_epoch_frac = 0.75):
        """
        Configure the pruning schedule for gradual dimension reduction.
        
        Args:
            total_epochs (int): Total number of training epochs
            schedule (str): Pruning schedule type - 'exp', 'linear', or 'cosine'
            end_epoch_frac (float): Fraction of training when pruning ends
            
        Schedule Types:
            - 'exp': Exponential decay with faster initial pruning
            - 'linear': Linear reduction over time
            - 'cosine': Cosine annealing schedule
            
        Notes:
            - Pruning stops at end_epoch_frac * total_epochs
            - Remaining epochs maintain target_message_dim dimensions
            - Creates self.pruning_schedule dict mapping epochs to target dimensions
        """
        prune_end_epoch = int(end_epoch_frac * total_epochs)
        # print(prune_end_epoch)
        prune_epochs = prune_end_epoch

        dims_to_prune = self.initial_message_dim - self.target_message_dim
        schedule_dict = {}

        #different pruning schedules
        #exponential decay
        if schedule == 'exp':
            decay_rate = 3.0
            max_decay = 1 - math.exp(-decay_rate)

            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                raw_decay = 1 - math.exp(-decay_rate * progress)
                decay_factor = raw_decay / max_decay

                dims_pruned = math.ceil(dims_to_prune * decay_factor)
                target_dims = max(self.initial_message_dim - dims_pruned, self.target_message_dim)
                schedule_dict[epoch] = target_dims

        #linear decay
        elif schedule == 'linear':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                dims_pruned = math.ceil(dims_to_prune * progress)
                target_dims = max(self.initial_message_dim - dims_pruned, self.target_message_dim)
                schedule_dict[epoch] = target_dims
        #cosine decay
        elif schedule == 'cosine':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                dims_pruned = math.ceil(dims_to_prune * (1 - cosine_decay))
                target_dims = max(self.initial_message_dim - dims_pruned, self.target_message_dim)
                schedule_dict[epoch] = target_dims

        #keep target_message_dim for the last part of training
        for epoch in range(prune_end_epoch, total_epochs):
            schedule_dict[epoch] = self.target_message_dim

        self.pruning_schedule = schedule_dict


    def update_pruning_mask(self, epoch, sample_data):
        """
        Update the pruning mask based on message importance at scheduled epochs.
        
        Args:
            epoch (int): Current training epoch
            sample_data (list): List of validation Data objects for importance calculation
            
        Process:
            1. Computes messages for all sample data points
            2. Calculates importance as standard deviation across messages
            3. Selects top-k most important message dimensions
            4. Updates pruning mask to keep only important dimensions
            
        Notes:
            - Only called at epochs specified in pruning_schedule
            - Uses validation data to avoid overfitting to training patterns
            - Updates both pruning_mask and current_message_dim
        """
        target_dims = self.pruning_schedule[epoch] #dims we need to reduce active units to

        with torch.no_grad():
            all_messages = []
            for datapoint in sample_data: #collect the messages from the validation set 
                x, edge_index = datapoint.x, datapoint.edge_index
                source_nodes = x[edge_index[0]]
                target_nodes = x[edge_index[1]]
                messages = self.message(source_nodes, target_nodes)
                all_messages.append(messages)

            msg_array = torch.cat(all_messages, dim=0)  #[num_messages, message_dim]
            msg_importance = msg_array.std(dim=0)  #computes stds for each msg_dim over all datapoints
            most_important = torch.argsort(msg_importance)[-target_dims:] #chooses the messages w highest std

            new_mask = torch.zeros_like(self.pruning_mask) 
            new_mask[most_important] = True #mask all unimportant messages
            self.pruning_mask = new_mask
            self.current_message_dim = target_dims

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        messages = self.edge_model(tmp)
        messages = messages * self.pruning_mask.to(messages.device).float() #apply pruning mask
        return messages
        


def train(model, train_data, val_data, num_epoch, dataset_name = 'r2', save = True, wandb_log = True):
    """
    Train a GNN model with comprehensive logging and validation.
    
    Args:
        model: GNN model instance to train (any of the model types)
        train_data: Tuple of (input_data, acceleration_data) for training
        val_data: Tuple of (input_data, acceleration_data) for validation
        num_epoch (int): Number of training epochs
        dataset_name (str): Name of dataset for logging and saving (default: 'r2')
        save (bool): Whether to save model weights and metrics (default: True)
        wandb_log (bool): Whether to log to Weights & Biases (default: True)
        
    Returns:
        Trained model instance
        
    Features:
        - Adaptive batch size: 64*(4/num_nodes)^2 This is always 64 though because we always have 4 particles
        - Dynamic learning rate scheduling (Cosine for >30 epochs, OneCycle otherwise)
        - Special handling for regularized models (KL, L1) and pruning models
        - Comprehensive logging of training/validation losses
        - Model checkpointing with full state preservation
        
    Notes:
        - Uses Accelerate for distributed training support
        - Applies data augmentation during training (position noise)
        - Validates without augmentation for clean performance metrics
    """

    model_type = model.model_type_

    #setup accelerator
    accelerator = Accelerator()
    
    #training data
    input_data, acc = train_data
    edge_index = get_edge_index(input_data.shape[1]) #this never changes so we only calc once

    # node_dim = input_data.shape[-1]
    num_nodes = input_data.shape[1]
    acc_dim = acc.shape[-1]

    assert model.node_dim_ == input_data.shape[-1], 'Mismatch in model and data node/particle dimensions.'

    if model_type == 'bottleneck':
        assert model.message_dim_ == acc_dim, 'Bottleneck model, but message dimensions do not match dimensionality of system.' 
    
    dataset = [Data(x=input_data[i], edge_index=edge_index, y=acc[i]) for i in range(len(input_data))]
    batch_size = int(64*(4/input_data.shape[1])**2) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #validation data
    input_val, acc_val = val_data
    val_dataset = [Data(x=input_val[i], edge_index=edge_index, y=acc_val[i]) for i in range(len(input_val))]
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

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
    model, dataloader, val_dataloader, optimiser = accelerator.prepare(model, dataloader, val_dataloader, optimiser)


    print(f'Running on {accelerator.device}.')

    #learning rate scheduler
    if num_epoch > 30:
        scheduler = CosineAnnealingLR(optimiser, T_max = len(dataloader)*num_epoch)
        scheduler_name = 'CosineAnnealingLR'
    else: #OneCycleLR for quick training
        scheduler = OneCycleLR(optimiser, max_lr = init_lr, steps_per_epoch = len(dataloader), epochs = num_epoch, final_div_factor = 1e5)
        scheduler_name = 'OneCycleLR'

    losses = []
    val_losses = []
    active_dims_history = [] #track active message dimensions for pruning

    batch_per_epoch = int(len(input_data)/batch_size)

    if model_type == 'pruning':
         #use the validation data to check important messages for pruning
        idx = np.random.choice(len(val_dataset), size=10_240, replace=False)
        sample_data = [
            Data(
                x=val_dataset[i].x.clone(),
                edge_index=val_dataset[i].edge_index.clone(), 
                y=val_dataset[i].y.clone()
            ).to(accelerator.device) 
            for i in idx
        ]

    for epoch in range(num_epoch):

        if model_type == 'pruning' and epoch in model.pruning_schedule:
           #update mask at specific epochs using val data
            model.update_pruning_mask(epoch, sample_data)
            
        if model_type == 'pruning':
            active_dims_history.append(model.current_message_dim)

        total_loss = 0 #loss tracking
        i = 0
        samples = 0
        model.train() #set model in training mode
        while i < batch_per_epoch:
            #training
            for datapoints in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epoch}"):
                if i >= batch_per_epoch:
                    break
                optimiser.zero_grad()
                cur_bs = int(datapoints.batch[-1] + 1) #current batch size 

                if model_type in {'standard', 'bottleneck', 'pruning'}:
                    base_loss = model.loss(datapoints)
                    loss = base_loss/cur_bs
                else:
                    base_loss, unscaled_reg = model.loss(datapoints)
                    loss = (base_loss + unscaled_reg/ num_nodes)/cur_bs
                
                accelerator.backward(loss)
                optimiser.step()
                scheduler.step()

                total_loss += base_loss.item()
                samples += cur_bs
                i+=1

        avg_loss = total_loss/samples

        #validation once per epoch
        model.eval()
        val_loss = 0
        val_samples = 0
        
        with torch.no_grad():
            for datapoints in val_dataloader:
                cur_bs  = int(datapoints.batch[-1] + 1)
                if model_type in {'standard', 'bottleneck', 'pruning'}:
                    loss = model.loss(datapoints, augment=False) #validation loss don't add reg or noise
                else: 
                    loss,_ = model.loss(datapoints, augment=False)
                val_loss += loss.item()
                val_samples += cur_bs 

        avg_val_loss = val_loss/val_samples

        #log to wandb
        log_dict = {
            'epoch': epoch, 
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }

        if model_type == 'pruning':
            print(f'training loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}, active msg dims: {model.current_message_dim}')
            log_dict['active_message_dims'] = model.current_message_dim

        else:
            print(f'training loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}')

        wandb.log(log_dict)


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
            'hidden_dim': model.hidden_dim_, 
            'model_type': model.model_type_,
            'message_dim': model.message_dim_
        }

        # Add pruning-specific info
        if model_type == 'pruning':
            checkpoint['pruning_mask'] = model.pruning_mask
            checkpoint['current_message_dim'] = model.current_message_dim
            checkpoint['initial_message_dim'] = model.initial_message_dim
            checkpoint['target_message_dim'] = model.target_message_dim

        torch.save(checkpoint, f'{save_path}/epoch_{num_epoch}_model.pth')
        #log training run in json file
        metrics = {'datetime':datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   'dataset_name': dataset_name,
                   'model_type': model_type,
                   'epochs': num_epoch,
                   'scheduler': scheduler_name,
                   'num_nodes': num_nodes,
                   'dimensions': acc_dim, 
                   'train_loss': losses,
                   'val_loss': val_losses
                   }
        if model_type == 'pruning':
            metrics['active_dims_history'] = active_dims_history
            metrics['final_message_dims'] = model.current_message_dim
        with open(f'{save_path}/epoch_{num_epoch}_metrics.json', 'w') as json_file:
            json.dump(metrics, json_file, indent = 4)

    return model

def create_model(model_type, node_dim=6, acc_dim=2, hidden_dim=300):
    """
    Factory function to create GNN models of different types.
    
    Args:
        model_type (str): Type of model to create
        node_dim (int): Dimensionality of node features (default: 6)
        acc_dim (int): Dimensionality of acceleration output (default: 2)
        hidden_dim (int): Hidden layer dimensions for MLPs (default: 300)
        
    Returns:
        GNN model instance of specified type
        
    Supported Model Types:
        - 'standard': Basic NBodyGNN with 100-dimensional messages
        - 'bottleneck': BottleneckGN with acc_dim-dimensional messages
        - 'KL': KLGN with variational message passing and KL regularization
        - 'L1': L1GN with L1 regularization on messages
        - 'pruning': PruningGN with dynamic message dimension reduction
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'standard':
        return NBodyGNN(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
    elif model_type == 'bottleneck':
        return BottleneckGN(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
    elif model_type == 'KL':
        return KLGN(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
    elif model_type == 'L1':
        return L1GN(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
    elif model_type == 'pruning':
        return PruningGN(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of: 'standard', 'bottleneck', 'KL', 'L1', 'pruning'")
    

def load_model(dataset_name, model_type, num_epoch):
    """
    Load a trained model from checkpoint with CPU mapping for compatibility.
    Basically if you train models on colab and move them here it works.
    
    Args:
        dataset_name (str): Name of dataset the model was trained on
        model_type (str): Type of model ('standard', 'bottleneck', 'KL', 'L1', 'pruning')
        num_epoch (int): Number of epochs the model was trained for
        
    Returns:
        Loaded model in evaluation mode with restored state
        
    Features:
        - Automatic GPU-to-CPU tensor mapping for cross-device compatibility
        - Restores model architecture from checkpoint metadata
        - Special handling for pruning models (restores pruning state)
        - Sets model to evaluation mode
        
    File Structure:
        Expects: model_weights/{dataset_name}/{model_type}/epoch_{num_epoch}_model.pth
    """
    checkpoint = torch.load(
        f'{script_dir}/model_weights/{dataset_name}/{model_type}/epoch_{num_epoch}_model.pth',
        map_location=torch.device('cpu')  #maps GPU tensors to CPU
    )
    
    #create a new model
    model = create_model(
        model_type=model_type,
        node_dim=checkpoint['node_dim'],
        acc_dim=checkpoint['acc_dim'],
        hidden_dim=checkpoint['hidden_dim'], 
    )

    #load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #load pruning-specific attributes for pruning model
    if model_type == 'pruning':
        model.pruning_mask = checkpoint['pruning_mask']
        model.current_message_dim = checkpoint['current_message_dim']
        model.initial_message_dim = checkpoint['initial_message_dim']
        model.target_message_dim = checkpoint['target_message_dim']

    print(f'Model loaded successfully.')
    
    return model

def test(model, test_data):
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: Trained GNN model instance
        test_data: Tuple of (input_data, acceleration_data) for testing
        
    Returns:
        float: Average test loss across all test samples
        
    Features:
        - Uses same loss computation as validation (no augmentation)
        - Handles different model types (standard vs regularised)
        - Batch processing for memory efficiency
        - Returns normalised loss per sample
        
    Notes:
        - Model is set to evaluation mode
        - Uses batch size of 1024 for efficient processing
        - Prints test loss for immediate feedback
    """
    input_data, acc = test_data #load data
    model_type = model.model_type_
    edge_index = get_edge_index(input_data.shape[1])
    #set up dataloader
    dataset = [Data(x=input_data[i], edge_index=edge_index, y=acc[i]) for i in range(len(input_data))]
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    model.eval()
    total_loss = 0
    samples = 0
    
    with torch.no_grad():
        for datapoints in dataloader:
            #get batch size
            cur_bs = int(datapoints.batch[-1] + 1)
            
            #handle different model types like in validation
            if model_type in {'standard', 'bottleneck', 'pruning'}:
                loss = model.loss(datapoints, augment=False)
            else: 
                loss, _ = model.loss(datapoints, augment=False)
            
            total_loss += loss.item()
            samples += cur_bs
    
    avg_loss = total_loss / samples
    print(f'test Loss: {avg_loss:.4f}')
    
    return avg_loss