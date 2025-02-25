import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def get_edge_index(num_nodes): #edge index for fully connected graph
    idx = torch.arange(num_nodes)
    edge_index = torch.cartesian_prod(idx, idx)
    edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]]
    return edge_index.t() #output dimension [2, E]

def load_data(path): #load dataset 
    data = torch.load(f"{path}.pt")
    return data['X'], data['y']

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

        self.node_dim0 = node_dim
        self.acc_dim0 = acc_dim
        self.hidden_dim0 = hidden_dim

    def message(self, x_i, x_j):
        x = torch.cat((x_i, x_j), dim = -1) #concat along final dimension = features. shape is [batch_size, no_edges, no_features]
        x = self.edge_model(x) #put thru MLP. MLP only transforms the feature (last) dimension
        return x

    def forward(self, x, edge_index): 
        """Forward pass of this network

        Args:
            x (torch.Tensor): shape is [batch_size, no_nodes, no_node_features]
            edge_index (torch.Tensor): shape is [2, no_edges]
        """

        edge_message =  self.propagate(edge_index, x = (x,x)) #use same feature matrix for both source and target nodes (undirected network)
        #x is shape [batch_size, no_nodes, no_features]
        acc_pred = self.node_model(torch.cat([x, edge_message], dim = -1)) #predict accelerations

        return acc_pred
    
class NBodyGNN_L1(NBodyGNN):
    def __init__(self, node_dim=6, acc_dim=2, hidden_dim=300):
        super().__init__(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
        self.message_features = None
        self.combined_messages = None
    
    def message(self, x_i, x_j):
        message_features = super().message(x_i, x_j)

        if self.message_features == None:
            self.message_features = message_features

        else:
            self.message_features = torch.cat([self.message_features, message_features], dim=0)

        return message_features
    
    def forward(self, x, edge_index):
        self.message_features = None

        acc_pred = super().forward(x, edge_index)

        return acc_pred
    
    def get_messages(self):
        return self.message_features
    
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

def train (train_data, val_data, num_epoch, hidden_dim=300, patience = 10):
    """ Train the GNN

    Args:
        train_data (tuple): contains (input_data, accelerations)
        val_data (tuple): contains (input_data, accelerations)
        num_epoch (int): number of epochs to train on
        patience (int): number of epochs to wait before implementing early stopping

    Returns:
        model (NBodyGNN object): final trained model
    """

    #training data
    input_data, acc = train_data
    dataset = NBodyDataset(input_data, acc)   
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #validation data
    input_val, acc_val = val_data
    val_dataset = NBodyDataset(input_val, acc_val)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    node_dim = input_data.shape[-1]
    acc_dim = acc.shape[-1]

    model = NBodyGNN(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
    optimiser = torch.optim.Adam(model.parameters(), weight_decay=1e-8) #L2 regulariser on params
    criterion = nn.L1Loss() #MAE loss

    edge_index = get_edge_index(input_data.shape[1]) #this never changes so we only calc once

    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    for epoch in range (num_epoch):
        total_loss = 0 #loss tracking
        
        #set model in training mode
        model.train()

        pbar = tqdm(dataloader, desc=f"Epoch: {epoch+1}/{num_epoch}")
        for nodes, acc in pbar:

            #training
            optimiser.zero_grad()

            acc_pred = model(nodes, edge_index) #automatically calls model.forward()
            
            loss = criterion(acc_pred, acc)

            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            #pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        #validation 
        model.eval()
        val_loss = 0
        with torch.no_grad(): #stop computing gradients
            for nodes, acc in val_dataloader:
                acc_pred = model(nodes, edge_index) #run forward pass thru model
                val_loss += criterion(acc_pred, acc).item()

        avg_loss = total_loss/len(dataloader)
        avg_val_loss = val_loss/len(val_dataloader)
        print(f'training loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}')

        #check if this is the best loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            print(f"new best validation loss: {best_val_loss:.4f}")
        
        else: 
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")

        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")


    return model

def train_L1 (train_data, val_data, num_epoch, hidden_dim=300, patience = 10):
    """ Train the GNN with L1 regularisation on the messages

    Args:
        train_data (tuple): contains (input_data, accelerations)
        val_data (tuple): contains (input_data, accelerations)
        num_epoch (int): number of epochs to train on
        patience (int): number of epochs to wait before implementing early stopping

    Returns:
        model (NBodyGNN object): final trained model
    """

    #training data
    input_data, acc = train_data
    dataset = NBodyDataset(input_data, acc)   
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #validation data
    input_val, acc_val = val_data
    val_dataset = NBodyDataset(input_val, acc_val)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    node_dim = input_data.shape[-1]
    acc_dim = acc.shape[-1]

    model = NBodyGNN_L1(node_dim=node_dim, acc_dim=acc_dim, hidden_dim=hidden_dim)
    optimiser = torch.optim.Adam(model.parameters(), weight_decay=1e-8) #L2 regulariser on params
    criterion = nn.L1Loss() #MAE loss

    edge_index = get_edge_index(input_data.shape[1]) #this never changes so we only calc once
    #num_edges = int(len(edge_index[0])/2)

    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    for epoch in range (num_epoch):
        total_loss = 0 #loss tracking
        
        #set model in training mode
        model.train()

        pbar = tqdm(dataloader, desc=f"Epoch: {epoch+1}/{num_epoch}")
        for nodes, acc in pbar:

            #training
            optimiser.zero_grad()

            acc_pred = model(nodes, edge_index) #automatically calls model.forward()
            message_features = model.get_messages() #get message features. of shape [batch_size, num_nodes, 100]
            
            loss = criterion(acc_pred, acc) + 1e-2 * torch.mean(torch.abs(message_features)) #add L1 regulariser on messages

            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            #pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        #validation 
        model.eval()
        val_loss = 0
        with torch.no_grad(): #stop computing gradients
            for nodes, acc in val_dataloader:
                acc_pred = model(nodes, edge_index) #run forward pass thru model
                val_loss += criterion(acc_pred, acc).item()

        avg_loss = total_loss/len(dataloader)
        avg_val_loss = val_loss/len(val_dataloader)
        print(f'training loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}')
        #here it prints the loss INCLUDING the L1 loss term
        #you may want to have it just printing the accuracy? doesnt actually matter

        #check if this is the best loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            print(f"new best validation loss: {best_val_loss:.4f}")
        
        else: 
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")

        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")


    return model

def save_model(model, directory = "testing_models/model.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),     # Model weights
        'edge_model': model.edge_model.state_dict(),  # Edge MLP state
        'node_model': model.node_model.state_dict(),  # Node MLP state
        'node_dim': model.node_dim0,        # Your model's dimensions from __init__
        'acc_dim': model.acc_dim0,         # Acceleration dimensions (2 for 2D)
        'hidden_dim': model.hidden_dim0    # Hidden layer dimensions
        } 
    torch.save(checkpoint, f"{directory}")
    print('Model saved successfully')

def load_model(directory):
    checkpoint = torch.load(f"{directory}")
    model = NBodyGNN(
        node_dim=checkpoint['node_dim'],
        acc_dim=checkpoint['acc_dim'],
        hidden_dim=checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model loaded successfully')

    return model

def test(test_data, model):
    """
    Get loss for a test dataset
    """

    input_data, acc = test_data
    dataset = NBodyDataset(input_data, acc)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    edge_index = get_edge_index(input_data.shape[1])

    criterion = nn.L1Loss() #can add optionality to change this later on...
    
    model.eval()
    loss = 0
    with torch.no_grad():
        for nodes, acc in dataloader:
            acc_pred = model(nodes, edge_index)
            loss += criterion(acc_pred, acc).item()
    
    avg_loss = loss/len(dataloader)
    print('Avg loss: ', avg_loss)

    return avg_loss



