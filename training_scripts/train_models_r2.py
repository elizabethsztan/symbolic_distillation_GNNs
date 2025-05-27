import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import *
from utils import *
import argparse

#split the data into train and test sets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--epoch', type = int, required= True)
    
    args = parser.parse_args()
    
    seed = 290402 
    data_path = 'datasets/r2_n=4_dim=2_nt=1000_dt=0.01.pt'

    train_data, val_data, _ = load_and_process(data_path, seed)

    # print("\n=== Testing standard Model ===")
    # model_type = 'standard'
    # model = create_model(model_type = model_type)
    # model = train(model, train_data=train_data, val_data=val_data, dataset_name = 'r2', num_epoch=args.epoch,
    #               save=args.save, wandb_log=args.wandb_log)

    print("\n=== Testing L1 Model ===")
    model_type = 'L1'
    model = create_model(model_type = model_type)
    model = train(model, train_data=train_data, val_data=val_data, dataset_name = 'r2', num_epoch=args.epoch,
                  save=args.save, wandb_log=args.wandb_log)

    print("\n=== Testing bottleneck Model ===")
    model_type = 'bottleneck'
    model = create_model(model_type = model_type)
    model = train(model, train_data=train_data, val_data=val_data, dataset_name = 'r2', num_epoch=args.epoch,
                  save=args.save, wandb_log=args.wandb_log)

    print("\n=== Testing KL Model ===")
    model_type = 'KL'
    model = create_model(model_type = model_type)
    model = train(model, train_data=train_data, val_data=val_data, dataset_name = 'r2', num_epoch=args.epoch,
                  save=args.save, wandb_log=args.wandb_log)

    print("\n=== Testing pruning Model ===")
    model_type = 'pruning'
    model = create_model(model_type = model_type)
    model = train(model, train_data=train_data, val_data=val_data, dataset_name = 'r2', num_epoch=args.epoch,
                  save=args.save, wandb_log=args.wandb_log)
    

if __name__ == "__main__":
    main()