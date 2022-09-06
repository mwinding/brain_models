# script from Michael Clayton
# %% 
import torch
import numpy as np
import matplotlib.pylab as plt

# Import processed connectivity data
import os
import sys
#mb_modelling_path = os.getenv("mb_modelling_path")
#sys.path.append(mb_modelling_path)
#import setup.emNetworkInfo as info

# Helper functions
from_np = lambda x : torch.from_numpy(x)
tens2np = lambda tens : [ten.detach().numpy() for ten in tens]
# fill_us_weights = lambda wdu, wou : from_np(np.vstack(tens2np([torch.zeros(size=(len(info.mnames),2)),wdu,wou])))
fill_us_weights = lambda wou : from_np(np.vstack(tens2np([torch.zeros(size=(len(info.mnames)+len(info.dnames),2)),wou])))
copyTensor = lambda tens : tens.clone()

# MB network
class Network:

    # -----------------------------------------
    # Initialisation
    # -----------------------------------------
    def __init__(self, inner_weights=[], biases=[], input_weights=[], dt=.5, timepoints=1, batches=30):
        '''Define core network constants and variables'''
        # Define network properties
        self.dt = dt
        self.n_neurons = len(inner_weights)
        self.batches = batches
        self.timepoints = timepoints
        # Define static network values
        self.inner_weights = inner_weights
        self.input_weights = input_weights if len(input_weights)!=0 else torch.zeros(size=(self.n_neurons,1))
        self.biases = biases
        # Initialise dynamic network values
        self.input_rates = torch.zeros(size=(self.n_neurons, self.batches))
        self.inner_rates = torch.zeros(size=(self.n_neurons, self.batches))
        # Define non-linear, positive rectification function
        self.nonlinear = lambda x : torch.relu(torch.tanh(x))
        # Initialise recordings
        self.recordings = {'rates': np.zeros((self.timepoints,self.n_neurons,self.batches))}

    # -----------------------------------------
    # Dynamics
    # -----------------------------------------

    # Equation 1
    def updateFiringRates(self):
        # Define variable mappings
        w, b, w_i = [copyTensor(tens) for tens in [self.inner_weights, self.biases, self.input_weights]] # copy variables that are due to be trained
        r, i = self.inner_rates, self.input_rates
        # Define inputs
        I_internal = torch.mm(w,r) + b #  (w.mm(r) + b).clone() # (weights * rates) + biases
        I_external = torch.einsum('ib,jb->ib',(w_i, i)) # KC inputs
        # Return new firing rate (i.e. Equation 1 in original paper)
        self.inner_rates = (1-self.dt)*r + self.dt*self.nonlinear(I_internal + I_external)

    # -----------------------------------------
    # Recordings
    # -----------------------------------------

    def record(self,t):
        self.recordings['rates'][t,:,:] = self.inner_rates

if __name__=="__main__":
    # Define network
    n_neurons = 4
    totalTime = 50
    inner_weights = torch.rand(size=(n_neurons, n_neurons))-.5
    biases = torch.rand(size=(n_neurons,1))
    input_weights = torch.rand(size=(n_neurons, 1))
    nn = Network(inner_weights, biases, input_weights=input_weights, timepoints=totalTime)

    for t in range(totalTime):
        nn.input_rates = torch.zeros(size=(nn.n_neurons,nn.batches))
        if 20 < t < 25:
            nn.input_rates = torch.rand(size=(nn.n_neurons,nn.batches))
        nn.updateFiringRates()
        nn.record(t)

    rates = nn.recordings["rates"]
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(rates[:,:,0].T, aspect="auto")
    ax[1].plot(rates[:,:,0])
    plt.show()
# %%
