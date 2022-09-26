import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def single_predict(molecule, model):
    if molecule is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, 2048)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.from_numpy(arr).float().to(device)
    return model(a).cpu().detach().numpy()


def triple_predict(molecule, model1, model2, model3):
    pred1 = single_predict(molecule, model1)
    pred2 = single_predict(molecule, model2)
    pred3 = single_predict(molecule, model3)
    return pred1, pred2, pred3


def train(model, X_train, y_train, X_valtest, y_valtest):
    # Split train into train-val
    X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

    # Defining datasets
    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    '''Defining model and training parameters'''

    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    #input_length = 2048
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_stats = {
        'train': [],
        "val": []
    }

    ''' Training and Validation Phase'''

    print("Training model.")
    for e in range(1, EPOCHS + 1):

        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

                val_epoch_loss += val_loss.item()
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f}')

    '''Evaluating Model '''

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    mse = mean_squared_error(y_test, y_pred_list)
    r_square = r2_score(y_test, y_pred_list)
    print("Mean Squared Error :", mse)
    print("R^2 :", r_square)

    return model


def multiple_train(model, X_train, y_train, X_valtest, y_valtest):
    # Split train into train-val
    X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

    # Defining datasets
    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    '''Defining model and training parameters'''

    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    #input_length = 2048
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_stats = {
        'train': [],
        "val": []
    }

    ''' Training and Validation Phase'''

    print("Training model.")
    for e in range(1, EPOCHS + 1):

        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f}')

    '''Evaluating Model '''

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    mse = mean_squared_error(y_test, y_pred_list)
    r_square = r2_score(y_test, y_pred_list)
    print("Average mean Squared Error :", mse)
    print("Average R^2 :", r_square)

    for i in range(len(y_pred_list[0])):
        y_pred_list_i = []
        with torch.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                y_pred_list_i.append(y_test_pred.cpu().numpy()[0][i])
        y_pred_list_i = [a.squeeze().tolist() for a in y_pred_list_i]
        mse = mean_squared_error([item[i] for item in y_test], y_pred_list_i)
        print(f"Mean Squared Error of output {i}:", mse)
        print(f"R^2 of output {i}:", r_square)

    return model
