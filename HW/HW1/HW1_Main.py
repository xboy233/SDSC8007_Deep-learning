# Main File
# PyTorch
from HW1_Dataprocess import dataloader
from HW1_Dataprocess import save_pred

# For record
import os
from datetime import datetime
curDatetime = datetime.now()

# DNN
from HW1_DNN import torch
from HW1_DNN import NeuralNet
from HW1_DNN import get_device
from HW1_DNN import train
from HW1_DNN import test

# Plot
from HW1_Plotting import plot_learning_curve
from HW1_Plotting import plot_pred
from HW1_Plotting import plt

datestr = curDatetime.strftime("%m%d-%H%M")

model_name = 'ADAM-CosineAnnealingWarmRestarts_Pearson_' + datestr     # modelname:model_DNN_Features

save_path = './HW/HW1/models/'
model_path = save_path + model_name + '.pth' # path to new model
train_path = './HW/HW1/HW1.train.csv'  # path to training data
test_path = './HW/HW1/HW1.test.csv'   # path to testing data
record_path = './HW/HW1/result_trainingLoss.txt' # path to record of training loss
figure_path = './HW/HW1/figures/' # dir of figures
prediction_result_path = './HW/HW1/prediction_result/prediction_' + model_name + '.csv' # path to prediction_result

# Tune these hyper-parameters to improve your model
# Hyper-Parameters for DNN
device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs(save_path, exist_ok=True)  # The trained model will be saved to ./models/
os.makedirs(figure_path, exist_ok=True)
modify = True                        # Need selection
# tune by Optuna
config = {
    'n_epochs': 2000,                # maximum number of epochs
    'batch_size': 128,               # mini-batch size for dataloader
    'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 9e-4,   # learning rate of SGD
    },
    'early_stop': 300,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': model_path           # your model will be saved here
}


# Read the Dataset
train_set = dataloader(train_path, 'train', config['batch_size'], modify=modify)
validation_set = dataloader(train_path, 'dev', config['batch_size'], modify=modify)
test_set = dataloader(test_path, 'test', config['batch_size'], modify=modify)

# Train DNN
model = NeuralNet(train_set.dataset.dim).to(device)  # Construct model and move to device
model_loss, model_loss_record = train(train_set, validation_set, model, config, device)

with open(record_path,"a", encoding='utf-8') as f:
    f.write(model_name + '\n')
    f.write('\tepochs: ' + str(config['n_epochs']) + '\n\tbatch_size: ' + str(config['batch_size']) + '\n\toptimizer: ' + str(config['optimizer']) + '\n\tlearning_rates: ' + str(config['optim_hparas']) + '\n\tearly_stop: ' + str(config['early_stop']) + '\n')
    f.write('Training Loss: ' + str(model_loss) + '\n\n')

# Plot
plot_learning_curve(model_loss_record, title='deep model')
plt.savefig(figure_path + 'training_loss_' + model_name)
plt.show()

del model
model = NeuralNet(train_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(validation_set, model, device)  # Show prediction on the validation set
plt.savefig(figure_path + 'validation' + model_name)
plt.show() 

# Testing
preds = test(test_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, prediction_result_path)         # save prediction file to pred.csv