import torch
import torch.nn as nn
import torch.nn.functional as F

class NaiveMLP(nn.Module):
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
        x = self.out(x)
        return x