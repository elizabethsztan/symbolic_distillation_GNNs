from sklearn.model_selection import train_test_split
from model import *

#split the data into train and test sets

def main():
    seed = 290402 
    data = torch.load('datasets/r2_n=4_dim=2_nt=1000_dt=0.001.pt')
    X, y = data['X'], data['y']
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size = 0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X, y, shuffle=True, test_size = 0.5, random_state=seed)

    print("\n=== Testing 'standard' Model ===")
    #train 'standard' model
    model = NBodyGNN()
    model = train(model, train_data=(X_train, y_train), val_data=(X_val, y_val), num_epoch=100, model_type='standard', save = True, wandb_log = False)

    #train 'L1' model
    print("\n=== Testing L1 Model ===")
    model = NBodyGNN()
    model = train(model, train_data=(X_train, y_train), val_data=(X_val, y_val), num_epoch=100, model_type='L1',save = True, wandb_log = False)

if __name__ == "__main__":
    main()