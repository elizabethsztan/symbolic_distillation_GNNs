import torch
# import torch.nn as nn
# from torch_geometric.nn import MessagePassing
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
import argparse
from utils import load_and_process

from model import get_edge_index, PruningGN
script_dir = os.path.dirname(os.path.abspath(__file__))
    
def train_pruning_models(model, train_data, val_data, num_epoch, dataset_name = 'charge', schedule = 'exp', end_epoch_frac = 0.75, save = True, wandb_log = True):

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

    #set up the pruning config
    model.set_pruning_schedule(num_epoch, schedule=schedule, end_epoch_frac=end_epoch_frac)
    
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

    wandb.init(project = f'pruning_experiments_{dataset_name}',
            name = f'{schedule}_end_epoch_frac{end_epoch_frac}',
            config={'num_epochs': num_epoch, 'num_nodes': num_nodes, 'dim': acc_dim, 'schedule': schedule, 'end_epoch_frac': end_epoch_frac}, 
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

        if epoch in model.pruning_schedule:
           #update mask at specific epochs using val data
            model.update_pruning_mask(epoch, sample_data)
            
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

                base_loss = model.loss(datapoints)
                loss = base_loss/cur_bs
                
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
                loss = model.loss(datapoints, augment=False) #validation loss don't add reg or noise
                val_loss += loss.item()
                val_samples += cur_bs 

        avg_val_loss = val_loss/val_samples

        #log to wandb
        log_dict = {
            'epoch': epoch, 
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'active_message_dims': model.current_message_dim
        }

        print(f'training loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}, active msg dims: {model.current_message_dim}')

        wandb.log(log_dict)


        #save losses
        losses.append(avg_loss)
        val_losses.append(avg_val_loss)

    wandb.finish()

    if save:
        save_path = f'{script_dir}/model_weights/pruning_experiments/{dataset_name}/{schedule}'
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

        checkpoint['pruning_mask'] = model.pruning_mask
        checkpoint['current_message_dim'] = model.current_message_dim
        checkpoint['initial_message_dim'] = model.initial_message_dim
        checkpoint['target_message_dim'] = model.target_message_dim

        torch.save(checkpoint, f'{save_path}/end_epoch_frac{end_epoch_frac}_epoch{num_epoch}_model.pth')
        #log training run in json file
        metrics = {'datetime':datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   'dataset_name': dataset_name,
                   'model_type': model_type,
                   'pruning_schedule': schedule,
                   'end_epoch_frac': end_epoch_frac,
                   'epochs': num_epoch,
                   'scheduler': scheduler_name,
                   'num_nodes': num_nodes,
                   'dimensions': acc_dim, 
                   'train_loss': losses,
                   'val_loss': val_losses
                   }
        metrics['active_dims_history'] = active_dims_history
        metrics['final_message_dims'] = model.current_message_dim
        with open(f'{save_path}/end_epoch_frac{end_epoch_frac}_epoch{num_epoch}_metrics.json', 'w') as json_file:
            json.dump(metrics, json_file, indent = 4)

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--epoch', type = int, required= True)
    
    args = parser.parse_args()
    
    seed = 290402 
    #run these experiments on the charge dataset
    data_path = 'datasets/charge_n=4_dim=2_nt=1000_dt=0.001.pt'

    train_data, val_data, _ = load_and_process(data_path, seed)

    schedules = ['exp', 'linear', 'cosine']
    # end_epoch_fracs = [0.65, 0.75, 0.85]
    end_epoch_fracs = [0.65, 0.75]

    for schedule in schedules:
        for end_epoch_frac in end_epoch_fracs:
            model = PruningGN()
            model = train_pruning_models(model, train_data=train_data, val_data=val_data, dataset_name = 'charge', schedule=schedule, 
                                        end_epoch_frac=end_epoch_frac, num_epoch=args.epoch, save = args.save, wandb_log = args.wandb_log)
    

if __name__ == "__main__":
    main()