import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers, act: callable=F.relu):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)
        self.act = act

    def forward(self, x, sigmoid: bool = False):
        x = self.act(self.lin_in(x))
        for hidden in self.hidden_list:
            x = self.act(hidden(x))
        x = self.lin_out(x)

        if sigmoid:
            return F.sigmoid(x)
        return x

class MLPFunction(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers, act: callable=F.relu):
        super().__init__()
        t_embed = torch.empty((1,hidden_size))
        embed_bias = torch.empty((1,hidden_size))
        nn.init.kaiming_normal_(t_embed)
        nn.init.kaiming_normal_(embed_bias)
        self.t_embed = nn.Parameter(t_embed)
        self.embed_bias = nn.Parameter(embed_bias)

        self.x_embed = nn.Linear(in_size, hidden_size)
        self.lin_in = nn.Linear(hidden_size * 2, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)
        self.act = act

    def forward(self, x, t, sigmoid: bool = False):
        d1 = x.shape[0]
        d2 = t.shape[0]
        t = self.act(t @ self.t_embed + self.embed_bias)
        x = self.act(self.x_embed(x))

        # run times in parallel
        x = torch.vstack([self.lin_in(torch.concat((torch.repeat_interleave(xi,d2,0),t),dim=1)) for xi in x])

        for hidden in self.hidden_list:
            x = self.act(hidden(x))
        x = self.lin_out(x)

        # reshape to recover timeseries
        x = torch.reshape(d1,d2)

        if sigmoid:
            return F.sigmoid(x)
        return x


class Siren(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers, freq_mod: float = 30.0):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)

        hidden_freqs = freq_mod * torch.ones((layers, 1, 1))
        self.hidden_freqs = nn.Parameter(hidden_freqs)

        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, sigmoid: bool = False):
        x = torch.sin(self.lin_in(x))
        for hidden, a in zip(self.hidden_list, self.hidden_freqs):
            x = torch.sin(hidden(a * x))
        x = self.lin_out(x)

        if sigmoid:
            return F.sigmoid(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, sigmoid: bool = False):
        x = F.relu(self.lin_in(x))
        for hidden in self.hidden_list:
            x = x + F.relu(hidden(x))
        x = self.lin_out(x)

        if sigmoid:
            return F.sigmoid(x)
        return x


class TrunkNet(nn.Module):
    def __init__(self, hidden_size, basis_dims, layers, act : callable=F.relu):
        super().__init__()
        lift = torch.empty((1, hidden_size))
        lift_bias = torch.empty((1, hidden_size))
        nn.init.kaiming_uniform_(lift)
        nn.init.kaiming_uniform_(lift_bias)
        self.lift = nn.Parameter(lift)
        self.lift_bias = nn.Parameter(lift_bias)
        self.body = MLP(hidden_size, hidden_size, basis_dims, layers - 1, act)

    def forward(self, t):
        v = F.silu((t @ self.lift) + self.lift_bias)
        v = self.body(v)
        return v


class SirenTrunk(nn.Module):
    def __init__(self, hidden_size, basis_dims, layers, freq_mod: float = 30.0):
        super().__init__()
        lift = torch.empty((1, hidden_size))
        lift_bias = torch.empty((1, hidden_size))
        nn.init.kaiming_uniform_(lift)
        nn.init.kaiming_uniform_(lift_bias)
        self.lift = nn.Parameter(lift)
        self.lift_bias = nn.Parameter(lift_bias)
        self.body = Siren(hidden_size, hidden_size, basis_dims, layers - 1, freq_mod)

    def forward(self, t):
        v = torch.sin((t @ self.lift) + self.lift_bias)
        v = self.body(v)
        # unit norm basis functions
        # v = v / torch.norm(v,2,0)
        return v


class DeepONet(nn.Module):
    def __init__(self, branch_dict: dict, trunk_dict: dict):
        super().__init__()
        self.branch_net = branch_dict["Net"](**branch_dict["Args"])
        self.trunk_net = trunk_dict["Net"](**trunk_dict["Args"])


    def forward(self, x, t, sigmoid: bool = False):
        c = self.branch_net(x, sigmoid=False)
        v = self.trunk_net(t)
        y = c @ v.T
        if sigmoid:
            return F.sigmoid(y)
        return y
