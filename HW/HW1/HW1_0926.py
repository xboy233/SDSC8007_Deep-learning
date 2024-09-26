#pytorch
# PyTorch
import HW1_Dataprocess
from HW1_Dataprocess import Dataprocess
from HW1_Dataprocess import dataloader
from HW1_Dataprocess import save_pred

# For data preprocess
import os

# DNN
import HW1_DNN
from HW1_DNN import torch
from HW1_DNN import NeuralNet
from HW1_DNN import get_device
from HW1_DNN import train
from HW1_DNN import test

# Plot
import HW1_Plotting
from HW1_Plotting import plot_learning_curve
from HW1_Plotting import plot_pred

save_path = 'HW/HW1/models/model.pth'
train_path = './HW/HW1/HW1.train.csv'  # path to training data
test_path = './HW/HW1/HW1.test.csv'   # path to testing data

# Tune these hyper-parameters to improve your model
# Hyper-Parameters for DNN
device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
modify = False                        # Need selection
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': save_path           # your model will be saved here
}

# Read the Dataset
train_set = dataloader(train_path, 'train', config['batch_size'], modify=modify)
validation_set = dataloader(train_path, 'dev', config['batch_size'], modify=modify)
test_set = dataloader(test_path, 'test', config['batch_size'], modify=modify)

# Train DNN
model = NeuralNet(train_set.dataset.dim).to(device)  # Construct model and move to device
model_loss, model_loss_record = train(train_set, validation_set, model, config, device)

# Plot
plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(train_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(validation_set, model, device)  # Show prediction on the validation set

# Testing
preds = test(test_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'prediction.csv')         # save prediction file to pred.csv