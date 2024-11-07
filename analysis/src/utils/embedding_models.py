import torch
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

# Define the RGCN model
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return x
    

from torch_geometric.nn import GCNConv

# adapted from colab 2 of the homework 
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim if i == 0 else hidden_dim, output_dim if i == num_layers - 1 else hidden_dim) 
             for i in range(num_layers)]
        )
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        H = x
        for i in range(self.num_layers - 1):
            H = self.convs[i](H, adj_t)
            H = self.bns[i](H)
            H = torch.nn.functional.relu(H)
            H = torch.nn.functional.dropout(H, p=self.dropout, training=self.training)
        H = self.convs[self.num_layers - 1](H, adj_t)
        return H if self.return_embeds else self.softmax(H)
