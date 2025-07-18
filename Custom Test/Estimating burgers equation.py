import scipy
import scipy.io
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

data = scipy.io.loadmat('Burgers.mat')

x= data['x']
t=data['t']
usol = data['usol']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


steps=20000
lr=1e-1
layers = np.array([2,20,20,20,20,20,20,20,20,1]) #8 hidden layers
N_u = 100 #Total number of data points for 'u'
N_f = 10_000 #Total number of collocation points 
nu = 0.01/np.pi #diffusion coefficient

def plot3D(x,t,y):
  x_plot =x.squeeze(1) 
  t_plot =t.squeeze(1)
  X,T= np.meshgrid(x_plot,t_plot)
  F_xt = y
  fig,ax=plt.subplots(1,1)
  cp = ax.contourf(T.transpose(),X.transpose(), F_xt,20,cmap="rainbow")
  fig.colorbar(cp) # Add a colorbar to a plot
  ax.set_title('F(x,t)')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  plt.show()
  ax = plt.axes(projection='3d')
  ax.plot_surface(T.transpose(),X.transpose(), F_xt,cmap="rainbow")
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_zlabel('f(x,t)')
  plt.show()

  
def plot3D_Matrix(x,t,y):
  X,T= x,t
  F_xt = y
  fig,ax=plt.subplots(1,1)
  cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
  fig.colorbar(cp) # Add a colorbar to a plot
  ax.set_title('F(x,t)')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  plt.show()
  ax = plt.axes(projection='3d')
  ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="rainbow")
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_zlabel('f(x,t)')
  plt.show()

plot3D(x,t,usol)

print(x.shape,t.shape)

X,T = np.meshgrid(x,t)

X_true = np.hstack((X.flatten()[:,None],T.flatten()[:,None]))


lb = X_true[0]
ub = X_true[-1]

print(lb,ub)

total_points = len(x)*len(t)

N_u = 10000

idx = np.random.choice(total_points,N_u,replace=False)

U_true = usol.flatten('F')[:,None]

X_train_Nu = X_true[idx]
U_train_Nu = U_true[idx]

print(total_points,N_u)

X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)

X_true = torch.from_numpy(X_true).float().to(device)
U_true = torch.from_numpy(U_true).float().to(device)


#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.Tanh()
    
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
    

lambda1 = 2.0

lambda2 = 0.02

class FCN():
    def __init__(self,layers):
        'Call our DNN'
        self.dnn = DNN(layers).to(device)
        'Initialize our parameters'
        self.lambda1 = torch.tensor([lambda1],requires_grad=True).float().to(device)
        self.lambda2 = torch.tensor([lambda2],requires_grad=True).float().to(device)

        self.lambda1 = nn.Parameter(self.lambda1)
        self.lambda2 = nn.Parameter(self.lambda2)

        'Call our DNN'
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda1',self.lambda1)
        self.dnn.register_parameter('lambda2',self.lambda2)

        'Loss Function'

        self.loss_function = nn.MSELoss(reduction='mean')

    def loss_data(self,x,y):
                
        loss_u = self.loss_function(self.dnn(x), y)
      
        return loss_u
    
    def loss_PDE(self, X_train_Nu):
                        
        lambda1=self.lambda1

        lambda2=self.lambda2

        g = X_train_Nu.clone()
                        
        g.requires_grad = True
        
        u = self.dnn(g)
                
        u_x_t = autograd.grad(u,g,torch.ones([X_train_Nu.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
                                
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(X_train_Nu.shape).to(device), create_graph=True)[0]
                                                            
        u_x = u_x_t[:,[0]]
        
        u_t = u_x_t[:,[1]]
        
        u_xx = u_xx_tt[:,[0]]
                                        
        f = u_t + (lambda1)*(self.dnn(g))*(u_x) - (lambda2)*u_xx 
        
        loss_f = self.loss_function(f,f_hat)
                
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
        
        if self.iter % 100 == 0:

            error_vec, _ = PINN.test()
        
            print(
                'Relative Error(Test): %.5f , 𝜆_real = [1.0,  %.5f], 𝜆_PINN = [%.5f,  %.5f]' %
                (
                    error_vec.cpu().detach().numpy(),
                    nu,
                    self.lambda1.item(),
                    self.lambda2.item()
                )
            )
            

        return loss        
    
    'test neural network'
    def test(self):
                
        u_pred = self.dnn(X_true)
        
        error_vec = torch.linalg.norm((U_true-u_pred),2)/torch.linalg.norm(U_true,2)        # Relative L2 Norm of the error (Vector)
        
        u_pred = u_pred.cpu().detach().numpy()
        
        u_pred = np.reshape(u_pred,(x.shape[0],t.shape[0]),order='F')
                
        return error_vec, u_pred

