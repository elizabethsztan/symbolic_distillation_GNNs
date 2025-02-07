import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.utils.data import Dataset, DataLoader

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

def train (train_data, val_data, num_epoch, hidden_dim=300):
    """ Train the GNN

    Args:
        train_data (tuple): contains (input_data, accelerations)
        val_data (tuple): contains (input_data, accelerations)
        num_epoch (int): number of epochs to train on

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

    for epoch in range (num_epoch):
        total_loss = 0 #loss tracking
        
        #set model in training mode
        model.train()
        for nodes, acc in dataloader:

            #training
            optimiser.zero_grad()

            acc_pred = model(nodes, edge_index) #automatically calls model.forward()
            
            loss = criterion(acc_pred, acc)

            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        #validation 
        model.eval()
        val_loss = 0
        with torch.no_grad(): #stop computing gradients
            for nodes, acc in val_dataloader:
                acc_pred = model(nodes, edge_index) #run forward pass thru model
                val_loss += criterion(acc_pred, acc).item()

        
        avg_loss = total_loss/len(dataloader)
        avg_val_loss = val_loss/len(val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epoch}], training loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}')

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

def message_features(test_data, model):
    """
    Gets aggregated message features for each node in test data.
    Essentially aggregated outputs of the edge_model MLP for each node.
    """
    input_data, _ = test_data
    edge_index = get_edge_index(input_data.shape[1])
    edge_message =  model.propagate(edge_index, x = (input_data,input_data))
    #shape is [batch_size, no_nodes, 100]
    
    return edge_message

def kl_div (latents): #check if this works
    return (latents.softmax(-1) * (latents.log_softmax(-1) - (torch.ones_like(latents)/latents.size(-1)).log())).sum(-1)
