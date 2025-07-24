import scipy
import scipy.io
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import autograd
import time
import pandas as pd



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


steps=20000

# === Constants ===
POWER_APPLIED = 0.15
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
TIME_STEP = 0.01
Q = 2.192


# === Data Loading ===
def load_data(filepath="temperature_output.csv"):
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
minTemp,maxTemp = np.min(temps_array),np.max(temps_array)
xScale = maxX-minX
yScale = maxY-minY
tScale = maxT-minT
tempScale = maxTemp-minTemp

coords_array[:,0] = (coords_array[:,0]-minX)/xScale
coords_array[:,1] = (coords_array[:,1]-minY)/yScale
coords_array[:,2] = (coords_array[:,2]-minT)/tScale


temps_array = (temps_array - minTemp) / tempScale

print(temps_array,coords_array)

X_train_Nu = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(temps_array).float().to(device)
l_b = X_train_Nu[0]
u_b = X_train_Nu[-1]




# Vertical boundaries (x=minX and x=maxX)
bc_v_points = [] 
for y in yValues:
    for t in tValues:
        bc_v_points.append([0, (y - minY)/yScale, (t-minT)/tScale])
        bc_v_points.append([1, (y - minY)/yScale, (t-minT)/tScale])

# Horizontal boundaries (y=minY and y=maxY)
bc_h_points = []
for x in xValues:
    for t in tValues:
        bc_h_points.append([(x - minX)/xScale, 0, (t-minT)/tScale])
        bc_h_points.append([(x - minX)/xScale, 1, (t-minT)/tScale])

X_bc_v = torch.from_numpy(np.array(bc_v_points)).float().to(device)
X_bc_h = torch.from_numpy(np.array(bc_h_points)).float().to(device)
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
    

k=2.0

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
    
    def loss_PDE(self, X_train_Nu):
            k = self.k
            g = X_train_Nu.clone()
            g.requires_grad = True
            
            u = self.dnn(g)
            
            # First derivatives
            u_x_y_t= torch.autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
            u_t_norm = u_x_y_t[:, [2]]

            # Second derivatives (Laplacian) - CORRECTED
            u_xx_yy_tt = torch.autograd.grad(u_x_y_t, g, torch.ones_like(u_x_y_t), create_graph=True)[0]
            # ... (rest of your chain rule code is fine) ...
            u_xx_norm = u_xx_yy_tt[:,[0]]
            u_yy_norm = u_xx_yy_tt[:,[1]]
            u_t_norm = u_x_y_t[:,[2]]
            d2Tdx2 =  (tempScale*u_xx_norm)/(xScale**2)
            d2Tdy2 =  (tempScale*u_yy_norm)/(yScale**2)
            dTdt =  (tempScale* u_t_norm) / tScale 

            residual = d2Tdx2 + d2Tdy2 - ((DENSITY * SPECIFIC_HEAT_CAPACITY) / k) * dTdt + (Q / k)
            
            loss_f = self.loss_function(residual, f_hat)
            return loss_f
    
    # In FCN, create separate loss methods or handle within one
    def loss_BC(self, X_bc_vertical, X_bc_horizontal):
        # --- Vertical Walls ---
        g_v = X_bc_vertical.clone(); g_v.requires_grad = True
        u_v = self.dnn(g_v)
        u_x_v = (tempScale* torch.autograd.grad(u_v, g_v, torch.ones_like(u_v), create_graph=True)[0][:, [0]]) / xScale
        loss_v = self.loss_function(u_x_v, torch.zeros_like(u_x_v))

        # --- Horizontal Walls ---
        g_h = X_bc_horizontal.clone(); g_h.requires_grad = True
        u_h = self.dnn(g_h)
        u_y_h = (tempScale * torch.autograd.grad(u_h, g_h, torch.ones_like(u_h), create_graph=True)[0][:, [1]]) / yScale
        loss_h = self.loss_function(u_y_h, torch.zeros_like(u_y_h))

        return loss_v + loss_h
    # Then in your total loss function
    # In your FCN class
    def loss(self, x_data, y_data, x_bc_v, x_bc_h): # Make sure to pass correct BC points
        loss_u = self.loss_data(x_data, y_data)
        loss_f = self.loss_PDE(x_data)
        loss_bc = self.loss_BC(x_bc_v, x_bc_h)

        # --- NEW WEIGHTS FOR LOSS BALANCING ---
        w_data = 1.0
        w_physics = 1.0  # Start with this
        w_bc = 1.0         # Start with this

        loss_val = (w_data * loss_u) + (w_physics * loss_f) + (w_bc * loss_bc)
        
        print(f"Iter {self.iter}: "
            f"Weighted Losses -> Data: {w_data * loss_u:.3f}, "
            f"Physics: {w_physics * loss_f:.3f}, "
            f"BC: {w_bc * loss_bc:.3f}, "
            f"K Value: {self.k}") 
        
        return loss_val
           
    
    'test neural network'
    def test(self):
                
        u_pred = self.dnn(X_train_Nu)
        
        error_vec = torch.linalg.norm((U_train_Nu-u_pred),2)/torch.linalg.norm(U_train_Nu,2)        # Relative L2 Norm of the error (Vector)
        
                
        return error_vec


layers = np.array([3,20,20,20,20,20,20,20,20,1])
PINN = FCN(layers)

params = list(PINN.dnn.parameters())
# Replace LBFGS with Adam
optimizer = torch.optim.Adam(params, lr=1e-3) # Use a smaller learning rate for Adam

start_time = time.time()
# Your training loop will need to change from a closure-based one
# to a standard loop:
for i in range(steps):
    optimizer.zero_grad()
    loss = PINN.loss(X_train_Nu, U_train_Nu, X_bc_v, X_bc_h) # Pass correct BC points
    loss.backward()
    optimizer.step()

    
    
elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))


''' Model Accuracy ''' 
error_vec, u_pred = PINN.test()

print('Test Error: %.5f'  % (error_vec))