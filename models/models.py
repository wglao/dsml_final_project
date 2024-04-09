import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.lin_in(x))
        for hidden in self.hidden_list:
            x = F.relu(hidden(x))
        x = self.lin_out(x)
        return x

class Siren(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers, freq_mod: float = 30.):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)

        hidden_freqs =  freq_mod*torch.ones((layers,1,1))
        self.hidden_freqs = nn.Parameter(hidden_freqs)
        
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)
    
    def forward(self, x):
        x = torch.sin(self.lin_in(x))
        for hidden, a in zip(self.hidden_list, self.hidden_freqs):
            x = torch.sin(hidden(a*x))
        x = self.lin_out(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.lin_in(x))
        for hidden in self.hidden_list:
            x = x + F.relu(hidden(x))
        x = self.lin_out(x)
        return x

class TrunkNet(nn.Module):
    def __init__(self, hidden_size, basis_dims, layers):
        super().__init__()
        lift = torch.empty((hidden_size,1))
        lift_bias = torch.empty((1,hidden_size))
        nn.init.kaiming_uniform_(lift)
        nn.init.kaiming_uniform_(lift_bias)
        self.lift = nn.Parameter(lift)
        self.lift_bias = nn.Parameter(lift_bias)
        self.body = MLP(hidden_size,hidden_size,basis_dims,layers-1)
    
    def forward(self, t):
        v = F.relu((t @ self.lift.T) + self.lift_bias)
        v = self.body(v)
        return v
    
class SirenTrunk(nn.Module):
    def __init__(self, basis_dims, layers, freq_mod: float=30.):
        super().__init__()
        lift = torch.empty((basis_dims,1))
        lift_bias = torch.empty((1,basis_dims))
        recomb_weights = torch.empty((layers+1,1,1))
        nn.init.kaiming_uniform_(lift)
        nn.init.kaiming_uniform_(lift_bias)
        nn.init.kaiming_uniform_(recomb_weights)
        self.lift = nn.Parameter(lift)
        self.lift_bias = nn.Parameter(lift_bias)
        self.recomb_weights = nn.Parameter(recomb_weights)
        hidden_freqs =  freq_mod*torch.ones((layers,1,1))
        self.hidden_freqs = nn.Parameter(hidden_freqs)

        hidden_list = [nn.Linear(basis_dims, basis_dims) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.bias = nn.Parameter(torch.zeros([1,1]))
    
    def forward(self,t):
        x = torch.sin((t @ self.lift.T) + self.lift_bias)
        v = self.recomb_weights[0]*x

        for hidden, a, w in zip(self.hidden_list, self.hidden_freqs, self.recomb_weights[1:]):
            x = torch.sin(hidden(a*x))
            v += w*x

        v += self.bias
        return v


class DeepONet(nn.Module):
    def __init__(self, branch_dict: dict, trunk_dict: dict):
        super().__init__()
        self.branch_net = branch_dict["Net"](**branch_dict["Args"])
        self.trunk_net = trunk_dict["Net"](**trunk_dict["Args"])
        
    def forward(self, x, t):
        c = self.branch_net(x)
        v = self.trunk_net(t)
        if len(c.shape) == 1:
            y = v @ c[:,None]
        elif c.shape[1] == v.shape[1]:
            y = v @ c.T
        else:
            y = v @ c
        return y