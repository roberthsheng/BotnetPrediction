import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import pickle
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import pickle

from scipy.sparse import csr_matrix

from utils import *
from utilsModel import *

PARENT_PATH = 'data'
TRAIN_FILE_NAME = f'{PARENT_PATH}/final_train.pkl'
TEST_FILE_NAME = f'{PARENT_PATH}/final_test.pkl'

saved_dict = {}
PREPROCESSED_PATH = '../data'

FIGURE_PATH = f'images'



def loadDataAndPrepFiles():
    x_train, y_train, y_train_label = pickle.load(open(TRAIN_FILE_NAME, 'rb'))
    x_test, y_test, y_test_label = pickle.load(open(TEST_FILE_NAME, 'rb'))

    # Dictionaries
    saved_dict = pickle.load(open(f'{PARENT_PATH}/saved_dict.pkl', 'rb'))
    mode_dict = pickle.load(open(f'{PARENT_PATH}/mode_dict.pkl', 'rb'))

    # Standard scaler
    scaler = pickle.load(open(f'{PARENT_PATH}/scaler.pkl', 'rb'))

    # Onehot/Label encoders
    ohe_dtos = pickle.load(open(f'{PARENT_PATH}/ohe_dtos.pkl', 'rb'))
    ohe_stos = pickle.load(open(f'{PARENT_PATH}/ohe_stos.pkl', 'rb'))
    ohe_dir = pickle.load(open(f'{PARENT_PATH}/ohe_dir.pkl', 'rb'))
    ohe_proto = pickle.load(open(f'{PARENT_PATH}/ohe_proto.pkl', 'rb'))

    train_filt = x_train.isna().any(axis=1)
    if train_filt.sum() > 0:
        print(f"filtered out {train_filt.sum()} records containing NaNs in training set")
        x_train = x_train.loc[~train_filt, :]
        y_train = y_train.loc[~train_filt]
    # Making train data sparse matrix
    x_train_csr = csr_matrix(x_train.values)

    test_filt = x_test.isna().any(axis=1)
    if test_filt.sum() > 0:
        print(f"filtered out {train_filt.sum()} records containing NaNs in testing set")
        x_test = x_test.loc[~test_filt, :]
        y_test = y_train.loc[~test_filt]
    # Making test data sparse matrix
    x_test_csr = csr_matrix(x_test.values)

    return x_train_csr, y_train, x_test_csr, y_test


class MLP(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(MLP, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

    def forward(self, x):
        return self.layer(x).squeeze(1).float()


# Parameters
input_dim = 39  # input dimension
hidden_dim = 100  # hidden layer dimension
output_dim = 1  # output dimension
batch_size = 1024  # Adjust as per your computational resource
learning_rate = 0.01

# Preprocess data
print("Load and preprocess data")
x_train_csr, y_train, x_test_csr, y_test = loadDataAndPrepFiles()  # Assuming these functions are already defined as in your code

# Convert to PyTorch tensors
print("Convert to PyTorch tensors")
x_train = torch.tensor(x_train_csr.toarray()).float()
y_train = torch.tensor(y_train.values).float()
x_test = torch.tensor(x_test_csr.toarray()).float()
y_test = torch.tensor(y_test.values).float()

# Tensor datasets
print("Create datasets")
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Data loaders
print("Create dataloaders")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = MLP(input_dim, hidden_dim, output_dim, device).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()

    # Wrap train_loader with tqdm for a progress bar
    train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
    for i, (inputs, labels) in enumerate(train_loop):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Update the progress bar description
        train_loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        train_loop.set_postfix(loss=loss.item())


    # Evaluation
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, total=len(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)  # if using BCEWithLogitsLoss
            predictions = probabilities > 0.5  # or use a different threshold

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Convert lists to numpy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculating Metrics
    f1 = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    fpr = fp / (fp + tn)

    print(f'F1 Score: {f1}')
    print(f'ROC AUC Score: {roc_auc}')
    print(f'False Positive Rate: {fpr}')
