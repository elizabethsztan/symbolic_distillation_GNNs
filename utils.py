import os
from sklearn.model_selection import train_test_split
import torch

def load_and_process(data_dir, seed):
    filename = os.path.basename(data_dir).split('_')[0]
    data_save_path = f'train_val_test_data/{filename}'
    
    if not os.path.exists(data_save_path): #checks if the data has already been split
        os.makedirs(data_save_path, exist_ok=True)
        
        data = torch.load(data_dir)
        X, y = data['X'], data['y']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, shuffle=True, test_size=0.5, random_state=seed)
        
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
        
        torch.save(train_data, os.path.join(data_save_path, 'train_data.pt'))
        torch.save(val_data, os.path.join(data_save_path, 'val_data.pt'))
        torch.save(test_data, os.path.join(data_save_path, 'test_data.pt'))

        return train_data, val_data, test_data
    else: #load the data if the data has already been split
        train_data = torch.load(os.path.join(data_save_path, 'train_data.pt'))
        val_data = torch.load(os.path.join(data_save_path, 'val_data.pt'))
        test_data = torch.load(os.path.join(data_save_path, 'test_data.pt'))
    
        return train_data, val_data, test_data

def load_data (dataset_name): #load the data
    path = f'train_val_test_data/{dataset_name}'

    train_data = torch.load(os.path.join(path, 'train_data.pt'))
    val_data = torch.load(os.path.join(path, 'val_data.pt'))
    test_data = torch.load(os.path.join(path, 'test_data.pt'))

    return train_data, val_data, test_data
