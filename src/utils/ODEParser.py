import torch
import torch.nn as nn
import numpy as np
import os

import torch.multiprocessing as mp
from functools import partial
from pathlib import Path
import torch.optim as optim

# Set environment variable to handle OpenMP runtime conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import faiss

def cond_alpha(t,dt): # in the training paper: it should be related to  b(\tau) in formula (3.1)
    return 1-t+dt

def cond_sigma2(t,dt):
    return t+dt

def f(t,dt):
    alpha_t = cond_alpha(t,dt)
    f_t = -1.0/(alpha_t)
    return f_t

def g2(t,dt):
    dsigma2_dt = 1.0
    f_t = f(t,dt)
    sigma2_t = cond_sigma2(t,dt)
    g2 = dsigma2_dt - 2*f_t*sigma2_t
    return g2
def g(t,dt):
    return (g2(t,dt))**0.5


def ODE_solver(zt,x_sample,z_sample,x0_test,
               ODESOLVER_TIME_STEPS:int=2000):
    t_vec = torch.linspace(1.0,0.0,ODESOLVER_TIME_STEPS+1)
    log_weight_likelihood = -1.0* torch.sum( (x0_test[:,None,:]-x_sample)**2/2 , axis = 2, keepdims= False)
    weight_likelihood =torch.exp(log_weight_likelihood)
    for j in range(ODESOLVER_TIME_STEPS): 
        if j% 100 == 0:
            print(f'this is {j} times / overall {ODESOLVER_TIME_STEPS} times')
        t = t_vec[j+1]
        dt = t_vec[j] - t_vec[j+1]
        #print()
        score_gauss = -1.0*(zt[:,None,:]-cond_alpha(t,dt)*z_sample)/cond_sigma2(t,dt)

        log_weight_gauss= -1.0* torch.sum( (zt[:,None,:]-cond_alpha(t,dt)*z_sample)**2/(2*cond_sigma2(t,dt)) , axis =2, keepdims= False)
        weight_temp = torch.exp( log_weight_gauss )
        weight_temp = weight_temp*weight_likelihood
        weight = weight_temp/ torch.sum(weight_temp,axis=1, keepdims=True)
        score = torch.sum(score_gauss*weight[:,:,None],axis=1, keepdims= False)  
        ## score is followed by the formula 3.11
        
        zt= zt - (f(t,dt)*zt-0.5*g2(t, dt)*score) *dt
    return zt

class FN_Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim, self.hid_size)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)  # Additional layer
        self.output = nn.Linear(self.hid_size, self.output_dim)
        
        # Initialize weights with better initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_fc2_weight = torch.clone(self.fc2.weight.data)
        self.best_fc2_bias = torch.clone(self.fc2.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Additional activation
        x = self.output(x)
        return x

    def update_best(self):
        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_fc2_weight = torch.clone(self.fc2.weight.data)
        self.best_fc2_bias = torch.clone(self.fc2.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):
        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.fc2.weight.data = self.best_fc2_weight
        self.fc2.bias.data = self.best_fc2_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias