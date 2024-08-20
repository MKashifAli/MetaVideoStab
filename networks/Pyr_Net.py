import torch
import torch.nn as nn
import torch.nn.functional as F

class PyrNet(nn.Module):

    def __init__(self, opts, nc_in, nc_out, n_pyr):
        super(PyrNet, self).__init__()
        #### Layers = [For n in n_pyr: BasicNet()]
    def forward(self, X, prev_state):
        #### op from Layers....
        pass

class BasicNet(nn.Module):

    def __init__(self, nc_in, nc_out):
        super(BasicNet, self).__init__()
        #### Basic layer of the pyramid with I(t)^(k), U(O(t-1)) and U(I(t)^(k-1))

    def forward(self, X):
        pass