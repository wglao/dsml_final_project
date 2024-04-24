import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, PyTree
from typing_extensions import Sequence
import equinox as eqx


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers, act: callable=F.relu):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)
        self.act = act

    def forward(self, x, final_act: callable = None):
        x = self.act(self.lin_in(x))
        for hidden in self.hidden_list:
            x = self.act(hidden(x))
        x = self.lin_out(x)

        if final_act is not None:
            return final_act(x)
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

    def forward(self, x, t, final_act: callable = None):
        d1 = x.shape[0]
        d2 = t.shape[0]
        t = self.act(t @ self.t_embed + self.embed_bias)
        x = self.act(self.x_embed(x))

        # run times in parallel
        x = torch.vstack([self.lin_in(torch.concat((torch.repeat_interleave(xi[None,:],d2,0),t),dim=1)) for xi in x])

        for hidden in self.hidden_list:
            x = self.act(hidden(x))
        x = self.lin_out(x)

        # reshape to recover timeseries
        x = torch.reshape(x,(d1,d2))

        if final_act is not None:
            return final_act(x)
        return x

class eFCONet(eqx.Module):
    t_embed: Array
    embed_bias: Array
    x_embed: Array
    lin_in: eqx.Module
    hidden_list: Sequence[eqx.Module]
    lin_out: eqx.Module
    act: callable

    def __init__(self, in_size, hidden_size, out_size, layers, key: jrd.PRNGKey, act: callable=jax.nn.relu):
        super().__init__()
        keys = jrd.split(key,5+layers)
        init = jax.nn.initializers.kaiming_normal()
        self.t_embed = init(keys[0],(1,hidden_size))
        self.embed_bias = init(keys[1],(1,hidden_size))
        self.x_embed = eqx.nn.Linear(in_size,hidden_size,key=keys[2])
        self.lin_in = eqx.nn.Linear(2*hidden_size, hidden_size, key=keys[3])
        self.hidden_list = [eqx.nn.Linear(hidden_size, hidden_size, key=keys[i+4]) for i in range(layers)]
        self.lin_out = eqx.nn.Linear(hidden_size, out_size, key=keys[-1])
        self.act = act

    def __call__(self, x, t):
        t = self.act(t @ self.t_embed + self.embed_bias)
        x = self.act(self.x_embed(x))

        # run times in parallel
        feed_lin_in = lambda x, t: self.lin_in(jnp.concat(x,t))
        x = self.act(jax.vmap(feed_lin_in, in_axes=(None,0))(x, t))

        for hidden in self.hidden_list:
            x = self.act(hidden(x))
        x = self.lin_out(x)

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

    def forward(self, x, final_act: callable = None):
        x = torch.sin(self.lin_in(x))
        for hidden, a in zip(self.hidden_list, self.hidden_freqs):
            x = torch.sin(hidden(a * x))
        x = self.lin_out(x)

        if final_act is not None:
            return final_act(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers):
        super().__init__()
        self.lin_in = nn.Linear(in_size, hidden_size)
        hidden_list = [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        self.hidden_list = nn.ModuleList(hidden_list)
        self.lin_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, final_act: callable = None):
        x = F.relu(self.lin_in(x))
        for hidden in self.hidden_list:
            x = x + F.relu(hidden(x))
        x = self.lin_out(x)

        if final_act is not None:
            return final_act(x)
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
        self.act=act

    def forward(self, t):
        v = self.act((t @ self.lift) + self.lift_bias)
        v = self.body(v)
        return v
    
class eMLP(eqx.Module):
    lin_in: eqx.Module
    hidden_list: Sequence[eqx.Module]
    lin_out: eqx.Module
    act: callable

    def __init__(self, in_size, hidden_size, out_size, layers, key: jrd.PRNGKey, act: callable=F.relu):
        super().__init__()
        keys = jrd.split(key, 2+layers)
        self.lin_in = eqx.nn.Linear(in_size, hidden_size, key=keys[0])
        self.hidden_list = [nn.Linear(hidden_size, hidden_size, key=keys[i+1]) for i in range(layers)]
        self.lin_out = nn.Linear(hidden_size, out_size, key=keys[-1])
        self.act = act

    def __call__(self, x):
        x = self.act(self.lin_in(x))
        for hidden in self.hidden_list:
            x = self.act(hidden(x))
        x = self.lin_out(x)
        return x


class eTrunk(eqx.Module):
    body: eqx.Module
    lift: Array
    lift_bias: Array

    def __init__(self, hidden_size, basis_dims, layers, key: jrd.PRNGKey, act: callable=jax.nn.relu):
        super().__init__()
        keys = jrd.split(key, 3)
        init = jax.nn.initializers.kaiming_normal()
        self.lift = init(keys[0], (1, hidden_size))
        self.bias = init(keys[1], (1, hidden_size))
        self.body = eMLP(hidden_size, hidden_size, basis_dims, layers, key=keys[2], act=act)

    def __call__(self,t):
        v = self.act((t @ self.lift) + self.lift_bias)
        v = self.body(v)
        return v
    
class eONet(eqx.Module):
    branch_net: eqx.Module
    trunk_net: eqx.Module

    def __init__(self, branch_dict: dict, trunk_dict: dict, key: jrd.PRNGKey):
        super().__init__()
        keys = jrd.split(key,2)
        self.branch_net = branch_dict["Net"](**branch_dict["Args"], key=keys[0])
        self.trunk_net = trunk_dict["Net"](**trunk_dict["Args"], key=keys[1])


    def __call__(self, x, t):
        c = self.branch_net(x)
        v = self.trunk_net(t)
        y = c @ v.T
        return y



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


    def forward(self, x, t, final_act: callable = None):
        c = self.branch_net(x)
        v = self.trunk_net(t)
        y = c @ v.T
        if final_act is not None:
            return final_act(y)
        return y
