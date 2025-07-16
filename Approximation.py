import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ThermalCountoursActualData import data,XData,YData,TempXColumns,TempYColumns


def prepare_training_data():
    training_data = []
    maxTimestamp = np.max(data["Timestamp"])
    for i in range(1,maxTimestamp+1):
        idx = data.index[data['Timestamp']==i][0]
        for j in range(6):
            newData = []
            TempYData = data[TempYColumns[j]][idx]
            for k in range(11):
                TempXData = data[TempXColumns[k]][idx]
                avg = (TempXData + TempYData) / 2
                training_data.append([XData[k],YData[j],i, avg])
    
    return np.array(training_data)


def train_model(training_data : np.ndarray):
    X = training_data[:,0:3]
    Y = training_data[:,3]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    # define the model
    model = nn.Sequential(
        nn.Linear(3, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.ReLU()
    )
    # train the model
    loss_fn   = nn.MSELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 100
    batch_size = 10
    
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')
    
    
    y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")




