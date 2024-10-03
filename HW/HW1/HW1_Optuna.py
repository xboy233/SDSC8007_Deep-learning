# Auto tuning the hyper-parameters

import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os
import logging

log_path = './HW/HW1/optuna.log'
optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(log_path))
optuna.logging.set_verbosity(optuna.logging.INFO)
# Set random seed for reproducibility
myseed = 9797
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    print("Cuda Training")

# Get device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# Data processing class
class Dataprocess(Dataset):
    def __init__(self, path, mode='train', modify=True):
        self.mode = mode
        with open(path, 'r') as f:
            data = list(csv.reader(f))
            data = np.array(data[1:])[:, 1:].astype(float)
        if modify == False:
            feats = list(range(0, 93))
        else:
            feats_symp1 = list(range(40,44))    # day1 symptom like Covid-19
            feats_testp1 = [57]                 # day2 tested_positive
            feats_symp2 = list(range(58,62))    # day2 symptom like Covid-19
            feats_testp2 = [75]                 # day2 tested_positive 
            feats_symp3 = list(range(76,80))    # day3 symptom like Covid-19

            feats = feats_symp1 + feats_testp1 + feats_symp2 + feats_testp2 + feats_symp3
        if mode == 'test':
            feats = [f + 1 for f in feats]
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(dim=0, keepdim=True)
        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'.format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

# Data loader
def dataloader(path, mode, batch_size, n_jobs=0, modify=False):
    dataset = Dataprocess(path, mode=mode, modify=modify)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=n_jobs, pin_memory=True)
    return dataloader

# Neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons):
        super(NeuralNet, self).__init__()
        
        # Try to modify this DNN to achieve better performance
        # xby_revised
        # BN + Dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_neurons),
            nn.BatchNorm1d(n_neurons),
            nn.LeakyReLU(),
            nn.Dropout(p=0.35),
            nn.Linear(n_neurons, 1)
        )

        # Loss function MSE
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

# Training function
def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    # xby_revised
    # learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, verbose=False)
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            # print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1

        # xby revised
        # scheduler
        scheduler.step(dev_mse)

        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

# Validation function
def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss


# Main program
device = torch.device('cpu')
os.makedirs('models', exist_ok=True)
modify = True


# train_path = './HW/HW1/HW1.train.csv'  # path to training data
# test_path = './HW/HW1/HW1.test.csv'   # path to testing data
train_path = './HW/HW1/HW1.train.csv'  # path to training data
test_path = './HW/HW1/HW1.test.csv'   # path to testing data



def objective(trial):
    # Define hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    n_epochs = trial.suggest_int('n_epochs', 1000, 3000,log=True)
    # early_stop = trial.suggest_int('early_stop', 100, 500,log=True)
    # n_layers = trial.suggest_int('n_layers', 1, 1)

    n_neurons = trial.suggest_int('n_neurons', 64, 1024, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    # Define data loaders
    train_set = dataloader(train_path, 'train', batch_size, modify=modify)
    val_set = dataloader(train_path, 'dev', batch_size, modify=modify)
    
    # Define model
    model = NeuralNet(train_set.dataset.dim, 1, n_neurons).to(device)

    config = {
    'n_epochs': n_epochs,                # maximum number of epochs
    'batch_size': batch_size,               # mini-batch size for dataloader
    'optimizer': optimizer_name,              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': lr,                 # learning rate of SGD
    },
    'early_stop': 300,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': './HW/HW1/models/model_optuna.pth'           # your model will be saved here
}

    model_loss, model_loss_record = train(train_set, val_set, model, config, device)
 
    return model_loss

# Create Optuna study
study = optuna.create_study(direction='minimize')
# Run optimization
study.optimize(objective, n_trials=1000)

# Output best hyperparameters
print('Best hyperparameters: ', study.best_params)
