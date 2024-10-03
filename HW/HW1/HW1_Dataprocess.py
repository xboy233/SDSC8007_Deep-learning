# PyTorch
from HW1_DNN import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
# ## Data and Preprocess
# The function below used to:
# 1. Read the csv files into python
# 2. Choose features (you can choose yourself)
# 3. Split data into training and validation sets.
# 4. Normalization
myseed = 9797  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    print("Cuda Training")


class Dataprocess(Dataset):
    def __init__(self,
                 path,
                 mode='train',
                 modify=False):
        self.mode = mode

        # Read csv file
        with open(path, 'r') as f:
            data = list(csv.reader(f))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if modify == False:
            feats = list(range(0,93))
        else:
            # Hint:Feature Selection
            # feats_sta = list(range(40))         # 40 states
            # xby_revised
            # Selected by Pearson
            feats_symp1 = list(range(40,44))    # day1 symptom like Covid-19
            feats_testp1 = [57]                 # day2 tested_positive
            feats_symp2 = list(range(58,62))    # day2 symptom like Covid-19
            feats_testp2 = [75]                 # day2 tested_positive 
            feats_symp3 = list(range(76,80))    # day3 symptom like Covid-19
            
            feats = feats_symp1 + feats_testp1 + feats_symp2 + feats_testp2 + feats_symp3

            pass

        if mode == 'test':
            # Testing set
            feats = [f+1 for f in feats]
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training set
            target = data[:, -1]
            data = data[:, feats]

            # Splitting data into training and validation sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
    
# ## Dataloader
# Loads data into batches.
def dataloader(path, mode, batch_size, n_jobs=0, modify=False):
    dataset = Dataprocess(path, mode=mode, modify=modify)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader


# Save prediction results
def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])