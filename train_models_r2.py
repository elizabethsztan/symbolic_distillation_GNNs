from model import *
from utils import *
import argparse

#split the data into train and test sets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--wandb_log', action='store_true')
    
    args = parser.parse_args()
    
    seed = 290402 
    data_path = 'datasets/r2_n=4_dim=2_nt=1000_dt=0.001.pt'

    train_data, val_data, test_data = load_and_process(data_path, seed)

    # print("\n=== Testing 'standard' Model ===")
    # model = NBodyGNN()
    # model = train(model, train_data=train_data, val_data=val_data, num_epoch=3, 
    #              model_type='standard', save=args.save, wandb_log=args.wandb_log)

    # print("\n=== Testing L1 Model ===")
    # model = NBodyGNN()
    # model = train(model, train_data=train_data, val_data=val_data, num_epoch=3, 
    #              model_type='L1', save=args.save, wandb_log=args.wandb_log)

if __name__ == "__main__":
    main()