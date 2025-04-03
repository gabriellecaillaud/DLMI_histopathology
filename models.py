import torch.nn as nn
import torch


class HistoClassifierHead(nn.Module):

    def __init__(self, dim_input, hidden_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(dim_input, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.layernorm1 = nn.LayerNorm(dim_input)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.layernorm3 = nn.LayerNorm(hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x  =self.dropout(self.layernorm1(x))
        x = self.relu(self.layer1(x))
        x = self.layernorm2(self.dropout(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(self.layernorm3(x)))

        return x 

