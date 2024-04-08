import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers) -> None:
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

class ResNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers) -> None:
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        # batch_norm_list = [nn.BatchNorm1d(hidden_size) for _ in range(layers)]
        # self.batch_norm_list = nn.ModuleList(batch_norm_list)
        self.lin_out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.lin_in(x))
        # for hidden, batch_norm in zip(self.hidden_list, self.batch_norm_list):
            # x = x + hidden(x)
            # x = F.relu(batch_norm(x))
        for hidden in self.hidden_list:
            x = F.relu(x + hidden(x))
        x = self.lin_out(x)
        return x

class TrunkNet(nn.Module):
    def __init__(self, in_size, hidden_size, basis_dims, layers) -> None:
        super().__init__()
        lift = torch.empty((hidden_size,))
        lift_bias = torch.empty((in_size,1))
        nn.init.kaiming_normal_(lift)
        nn.init.kaiming_uniform_(lift_bias)
        self.lift = nn.Parameter(lift)
        self.lift_bias = nn.Parameter(lift_bias)
        self.body = MLP(hidden_size,hidden_size,basis_dims,layers-1)
    
    def forward(self, t):
        v = F.relu(torch.outer(t,self.lift) + self.lift_bias)
        v = self.body(v)
        return v


class DeepONet(nn.Module):
    def __init__(self, branch_dict: dict, trunk_dict: dict) -> None:
        super().__init__()
        self.branch_net = branch_dict["Net"](**branch_dict["Args"])
        self.trunk_net = trunk_dict["Net"](**trunk_dict["Args"])
        
    def forward(self, x, t):
        c = self.branch_net(x)
        v = self.trunk_net(t)
        y = v @ c
        return y