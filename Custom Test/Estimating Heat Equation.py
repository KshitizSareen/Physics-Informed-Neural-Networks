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
lr=0.1

# === Constants ===
POWER_APPLIED = 0.15
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
TIME_STEP = 0.01
Q = 2.192


# === Data Loading ===
def load_data(filepath="temperature_output.csv"):
    df = pd.read_csv(filepath)
    all_columns = df.columns.tolist()
    minTemp = float("inf")
    maxTemp = float("-inf")
    for i in range(2,len(all_columns)):
        column_name = all_columns[i]
        temp_column = df[column_name]
        if temp_column.min()<minTemp:
            minTemp = temp_column.min()
        if temp_column.max()>maxTemp:
            maxTemp = temp_column.max()
            
    return df,minTemp,maxTemp

# === Prepare Training Data ===

def prepare_training_data(df, minTemp, maxTemp):
    training_data = []
    
    # Properly compute min/max without casting
    min_time = df["Timestamp"].min()
    max_time = df["Timestamp"].max()
    
    # Spatial bounds
    minX, maxX = 0.0, 1.0
    minY, maxY = 0.0, 1.0
    
    xValues = set()
    yValues = set()
    tValues = set()
    
    all_columns = df.columns.tolist()

    all_coords = []
    all_temps = []
    
    for idx in range(len(df)):
        t_raw = df["Timestamp"].iloc[idx]
        for i in range(2, len(all_columns)):
            column = all_columns[i]
            x, y = map(float, column.strip("()").split(","))
            temp = df.iloc[idx, i]
            if x <= maxX and y <= maxY and t_raw <= max_time:
                # Normalize each value
                norm_x = (x - minX) / (maxX - minX)
                norm_y = (y - minY) / (maxY - minY)
                norm_t = (t_raw - min_time) / (max_time - min_time)
                norm_temp = (temp -minTemp) / (maxTemp-minTemp)

                training_data.append([norm_x, norm_y, norm_t, norm_temp])
                
                xValues.add(norm_x)
                yValues.add(norm_y)
                tValues.add(norm_t)
    
    return (
        np.array(training_data),
        minX, maxX, minY, maxY,
        min_time, max_time,
        sorted(xValues), sorted(yValues), sorted(tValues)
    )



df,minTemp,maxTemp = load_data()
data,minX,maxX,minY,maxY,min_time,max_time,xValues,yValues,tValues = prepare_training_data(df,minTemp,maxTemp)
X,Y,T = np.meshgrid(xValues,yValues,tValues)

X_true = np.hstack((X.flatten()[:,None],Y.flatten()[:,None],T.flatten()[:,None]))


lb = X_true[0]
ub = X_true[-1]


total_points = len(xValues)*len(yValues)*len(tValues)

N_u = 2000000


idx = np.random.choice(total_points,N_u,replace=False)


U_true = data[:, 3].flatten('F')[:,None]


X_train_Nu = X_true[idx]
U_train_Nu = U_true[idx]

X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)

X_true = torch.from_numpy(X_true).float().to(device)
U_true = torch.from_numpy(U_true).float().to(device)

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
        
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        #preprocessing input 
        x = (x - l_b)/(u_b - l_b) #feature scaling              
        
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
        
        # Derivatives wrt NORMALIZED coordinates
        u_grads = torch.autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        u_t_norm = u_grads[:, [2]]
        # Second derivatives wrt NORMALIZED coordinates
        u_laplacian_grads = torch.autograd.grad(u_grads, g, torch.ones_like(u_grads), create_graph=True)[0]
        u_xx_norm = u_laplacian_grads[:, [0]]
        u_yy_norm = u_laplacian_grads[:, [1]]

        # --- Apply the Chain Rule ---
        # Get scaling factors from bounds
        x_scale = maxX - minX
        y_scale = maxY - minY
        t_scale = max_time - min_time
        temp_scale = maxTemp-minTemp

        # Scale derivatives back to physical space
        d2Tdx2 = (temp_scale* u_xx_norm) / (x_scale**2)
        d2Tdy2 =(temp_scale* u_yy_norm) / (y_scale**2)
        dTdt = (temp_scale* u_t_norm) / t_scale

        # The PDE is now dimensionally consistent
        residual = k*((d2Tdx2) + (d2Tdy2)) - (((DENSITY * SPECIFIC_HEAT_CAPACITY) ) * dTdt) + (Q)
        
        loss_f = self.loss_function(residual, f_hat)
        return loss_f

    def loss(self,x,y):

        loss_u = self.loss_data(x,y)
        loss_f = self.loss_PDE(x)
        
        loss_val = loss_u + loss_f
        
        return loss_val
     
    'callable for optimizer'                                       
    def closure(self):
        
        optimizer.zero_grad()
        
        loss = self.loss(X_train_Nu, U_train_Nu)
        
        loss.backward()
                
        self.iter += 1
        

        error_vec = PINN.test()
    
        print(
            'Relative Error(Test): %.5f , 𝜆_real = [0.015,], k_PINN = [%.5f]' %
            (
                error_vec.cpu().detach().numpy(),
                self.k.item(),
            )
        )
            

        return loss        
    
    'test neural network'
    def test(self):
                
        u_pred = self.dnn(X_true)
        
        error_vec = torch.linalg.norm((U_true-u_pred),2)/torch.linalg.norm(U_true,2)        # Relative L2 Norm of the error (Vector)
        
                
        return error_vec


layers = np.array([3,20,20,20,20,20,20,20,20,1])
PINN = FCN(layers)

params = list(PINN.dnn.parameters())
'L-BFGS Optimizer'
'L-BFGS Optimizer'
optimizer = torch.optim.LBFGS(params, lr, 
                              max_iter = steps, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')

start_time = time.time()

optimizer.step(PINN.closure)
    
    
elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))


''' Model Accuracy ''' 
error_vec, u_pred = PINN.test()

print('Test Error: %.5f'  % (error_vec))