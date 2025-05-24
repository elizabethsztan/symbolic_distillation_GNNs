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

# script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.getcwd()

def get_edge_index(num_nodes): #edge index for fully connected graph
    idx = torch.arange(num_nodes)
    edge_index = torch.cartesian_prod(idx, idx)
    edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]]
    return edge_index.t() #output dimension [2, num_edges]

def load_data(path): #load dataset 
    data = torch.load(f"{path}.pt")
    return data['X'], data['y']

class NBodyGNN(MessagePassing):
    def __init__(self, model_type = 'standard', node_dim = 6, acc_dim = 2, hidden_dim = 300):
        """ 
        N-body graph NN class.

        Args:
            node_dim (int): dimensionality of the node (in 2d case it is 6)
            acc_dim (int): dimensionality of the output of the network, which are accelerations so in 2d = 2
            hidden_dim (int): hidden layer dimensions - same for both edge and node MLPs

        """
    
        super().__init__(aggr='add')

        if model_type == 'bottleneck':
            message_dim = acc_dim #this is the dimensionality of the system
        
        elif model_type == 'pruning':
            message_dim = 100
            self.initial_message_dim = message_dim
            self.target_message_dim = acc_dim #towards the end of training, prune all except 2 messages
            self.current_message_dim = message_dim  

        else: 
            message_dim = 100
         
        #edge model MLP
        if model_type == 'KL':
            self.edge_model = nn.Sequential(
                nn.Linear(2*node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, int(message_dim * 2)) #double dimensionality because predicts mean and logvar of messages
            )

        else:
            self.edge_model = nn.Sequential(
                nn.Linear(2*node_dim, hidden_dim), #inputs = node information for two nodes
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, message_dim) #output message features of dimension 100 for standard and L1 model 
            )

        #node model MLP
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + message_dim, hidden_dim), #inputs = sum of outputs of edge model and node features
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
        self.model_type_ = model_type
        self.message_dim_ = message_dim

        if model_type == 'pruning':
            self.pruning_schedule = None 
            self.pruning_mask = torch.ones(message_dim, dtype=torch.bool) #this is set during training
    
    def set_pruning_schedule(self, total_epochs):
        prune_start_epoch = total_epochs // 4      #start pruning after 1/4 of training
        prune_end_epoch = 3 * total_epochs // 4    #stop pruning at 3/4 of training
        prune_epochs = prune_end_epoch - prune_start_epoch
        
        dims_to_prune = self.initial_message_dim - self.target_message_dim  #only leave 2 dims
        
        schedule = {}
        decay_rate = 3.0
        max_decay = 1 - math.exp(-decay_rate)
        
        for epoch in range(prune_start_epoch, prune_end_epoch):
            progress = (epoch - prune_start_epoch) / prune_epochs
            raw_decay = 1 - math.exp(-decay_rate * progress)
            decay_factor = raw_decay / max_decay
            
            dims_pruned = math.ceil(dims_to_prune * decay_factor)
            target_dims = self.initial_message_dim - dims_pruned
            target_dims = max(target_dims, self.target_message_dim)
            schedule[epoch] = target_dims
        
        #for the last quarter of training, keep at target_message_dim
        for epoch in range(prune_end_epoch, total_epochs):
            schedule[epoch] = self.target_message_dim
        
        self.pruning_schedule = schedule

    def update_pruning_mask(self, epoch, sample_data):
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

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        messages = self.edge_model(tmp)
        if self.model_type_ != 'KL':
            #apply pruning mask if this is a pruning model
            if self.model_type_ == 'pruning':
                messages = messages * self.pruning_mask.to(messages.device).float()
            return messages
        
        else: #for KL model you need to sample messages
            mu = messages[:, 0::2] #all evens are the mus
            logvar = messages[:,1::2]#all odds are the logvars
            if self.training:  #only sample during training
                noise = torch.randn(mu.shape, device=mu.device, requires_grad=False)
                std = torch.exp(logvar/2)
                messages = mu + noise * std #reparameterisation trick
                return messages
            else:
                return mu
    
    def update(self, aggr_out, x=None):

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

        source_nodes = data.x[data.edge_index[0]]
        target_nodes = data.x[data.edge_index[1]]

        if self.model_type_ == 'L1':

            messages = self.message(source_nodes, target_nodes) #collect the messages between all nodes
            reg_str = 1e-2 #regularisation strength
            unscaled_reg = reg_str * torch.sum(torch.abs(messages))
            return loss, unscaled_reg #later scale it according to batch size and num_nodes
        
        elif self.model_type_ == 'KL':
            tmp = torch.cat([source_nodes, target_nodes], dim=1)
            messages = self.edge_model(tmp) 
            mu = messages[:, 0::2]
            logvar = messages[:,1::2]
            reg_str = 1
            unscaled_reg = reg_str * torch.sum(torch.exp(logvar) + mu**2 - logvar)/2
            return loss, unscaled_reg
        else:
            return loss


def train(model, train_data, val_data, num_epoch, dataset_name = 'r2', save = True, wandb_log = True):

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

    if model_type == 'pruning':
        model.set_pruning_schedule(num_epoch)
    
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
        idx = np.random.choice(len(val_dataset), size=1024, replace=False)
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
                    loss = base_loss
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

def load_model(dataset_name, model_type, num_epoch):
    checkpoint = torch.load(f'{script_dir}/model_weights/{dataset_name}/{model_type}/epoch_{num_epoch}_model.pth')
    #create a new model
    model = NBodyGNN(
        node_dim=checkpoint['node_dim'],
        acc_dim=checkpoint['acc_dim'],
        hidden_dim=checkpoint['hidden_dim'], 
        # message_dim=checkpoint['message_dim'], 
        model_type=checkpoint['model_type']
    )

    #load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load pruning-specific attributes
    if model_type == 'pruning':
        model.pruning_mask = checkpoint['pruning_mask']
        model.current_message_dim = checkpoint['current_message_dim']
        model.initial_message_dim = checkpoint['initial_message_dim']
        model.target_message_dim = checkpoint['target_message_dim']

    print(f'Model loaded successfully.')
    
    return model

def test(model, test_data):

    input_data, acc = test_data
    edge_index = get_edge_index(input_data.shape[1])
    dataset = [Data(x=input_data[i], edge_index=edge_index, y=acc[i]) for i in range(len(input_data))]
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    model.eval()
    total_loss = 0
    samples = 0
    
    with torch.no_grad():
        for datapoints in dataloader:
            # Get batch size
            cur_bs = int(datapoints.batch[-1] + 1)
            loss = model.loss(datapoints, augment=False)
            
            total_loss += loss.item()
            samples += cur_bs
    
    avg_loss = total_loss / samples
    print(f'test Loss: {avg_loss:.4f}')
    
    return avg_loss