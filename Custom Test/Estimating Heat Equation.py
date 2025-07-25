import scipy
import scipy.io
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import autograd
import time
import pandas as pd
import os
import csv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


steps=100000

# === Constants ===
POWER_APPLIED = 0.15
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
Q = 2.192
THERMAL_CONDUCTIVITY = 10
INITIAL_TEMP = 21.23
NUM_DIV_X=10
NUM_DIV_Y=10
DURATION = 10

WIDTH=1
HEIGHT=1
TIMESTEP=0.1


# === Data Loading ===
def load_data(filepath="temperature_output_from_pinn.csv"):
    df = pd.read_csv(filepath)
            
    return df

# === Prepare Training Data ===

def prepare_training_data(df):
    training_data = []
    
    
    xValues = set()
    yValues = set()
    tValues = set()
    tempValues = set()
    all_columns = df.columns.tolist()

    all_coords = []
    all_temps = []
    for idx in range(len(df)):
        t_raw = df["Timestamp"].iloc[idx]
        for i in range(2, len(all_columns)):
            column = all_columns[i]
            x, y = map(float, column.strip("()").split(","))
            temp = df.iloc[idx, i]

            
            xValues.add(x)
            yValues.add(y)
            tValues.add(t_raw)
            tempValues.add(temp)
            all_coords.append([x, y, t_raw])
            all_temps.append([temp])
    return (
        np.array(training_data),
        sorted(xValues), sorted(yValues), sorted(tValues),all_coords,all_temps
    )



df = load_data()
data,xValues,yValues,tValues,coords_array,temps_array = prepare_training_data(df)
# Assuming xValues, yValues, tValues are sorted lists of unique coordinates
coords_array = np.array(coords_array)
temps_array = np.array(temps_array)

minX, maxX = xValues[0], xValues[-1]
minY, maxY = yValues[0], yValues[-1]
minT,maxT = tValues[0],tValues[-1]
xScale = maxX-minX
yScale = maxY-minY
tScale = maxT-minT

coords_array[:,0] = (coords_array[:,0]-minX)/xScale
coords_array[:,1] = (coords_array[:,1]-minY)/yScale
coords_array[:,2] = (coords_array[:,2]-minT)/tScale

 
X_train_Nu = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(temps_array).float().to(device)




bc_t_points = []
for x in xValues:
    for y in yValues:
        bc_t_points.append([(x - minX)/xScale, (y - minY)/yScale, 0])

X_bc_t = torch.from_numpy(np.array(bc_t_points)).float().to(device)
f_hat = torch.zeros(X_train_Nu.shape[0],1).to(device)
     


#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.ReLU()
    
        'Initialize neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
    
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,x):
              
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                             
        #convert to float
        a = x.float()
        
        for i in range(len(layers)-2):
            
            z = self.linears[i](a)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a
    

k=1.0

class FCN():
    def __init__(self,layers):

        self.iter=0
        'Call our DNN'
        self.dnn = DNN(layers).to(device)
        'Initialize our parameters'
        self.k = torch.tensor([k],requires_grad=True).float().to(device)

        self.k = nn.Parameter(self.k)

        'Call our DNN'
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('k',self.k)

        'Loss Function'

        self.loss_function = nn.MSELoss(reduction='mean')

    def loss_data(self,x,y):
                
        loss_u = self.loss_function(self.dnn(x), y)
      
        return loss_u
    # Inside FCN class
    def loss_PDE(self, X_train_Nu):
        k = self.k
        g = X_train_Nu.clone()
        g.requires_grad = True
        u = self.dnn(g)
        u_x_y_t = torch.autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        u_xx_yy_tt = torch.autograd.grad(u_x_y_t, g, torch.ones_like(u_x_y_t), create_graph=True)[0]
        u_xx_norm = u_xx_yy_tt[:, [0]]
        u_yy_norm = u_xx_yy_tt[:, [1]]
        u_t_norm = u_x_y_t[:, [2]]
        d2Tdx2 = (u_xx_norm) / (xScale**2)
        d2Tdy2 = (u_yy_norm) / (yScale**2)
        dTdt = ( u_t_norm) / tScale
        residual = d2Tdx2 + d2Tdy2 - ((DENSITY * SPECIFIC_HEAT_CAPACITY) / k) * dTdt + (Q / k)
        loss_f = self.loss_function(residual , f_hat)
        return loss_f
    
    # In FCN, create separate loss methods or handle within one
    def loss_BC(self, T_bc):

        g_t = T_bc.clone(); g_t.requires_grad = True
        u_t = self.dnn(g_t)
        loss_t = self.loss_function(u_t-INITIAL_TEMP, torch.zeros_like(u_t))
        return  loss_t
    # Then in your total loss function
    # In your FCN class
    def loss(self, x_data, y_data,X_bc_t):
        loss_u = self.loss_data(x_data, y_data)
        loss_f = self.loss_PDE(x_data)
        loss_bc = self.loss_BC(X_bc_t)
        # Adaptive weights
        w_data = 1.0
        w_physics = 1e-4
        w_bc = 1.0
        loss_val = w_data * loss_u + w_physics * loss_f + w_bc * loss_bc
        print(f"Iter {self.iter}: Data: {w_data * loss_u:.6f}, Physics: {w_physics * loss_f:.6f}, "
            f"BC: {w_bc * loss_bc:.6f}, k: {self.k.item():.6f}")
        self.iter += 1
        return loss_val
            
    
            
    def test(self,x_data):
        return self.dnn(x_data)


# Early stopping parameters
early_stopping_patience = 1000  # How many steps to wait for improvement
best_loss = float('inf')
patience_counter = 0
best_model_state = None

layers = np.array([3, 32, 32, 32, 32, 1])
PINN = FCN(layers)

# Training loop
optimizer = torch.optim.Adam(PINN.dnn.parameters(), lr=0.001)
k_history = []
csv_file = "k_history.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "k"])

while 1:
    optimizer.zero_grad()
    loss = PINN.loss(X_train_Nu, U_train_Nu, X_bc_t)
    loss.backward()
    optimizer.step()

    k_value = PINN.k.item()
    k_history.append(k_value)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([len(k_history) - 1, k_value])


    