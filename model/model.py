import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import Linear, HeteroConv, RGATConv

from .conv import HGTConv


class RGAT(torch.nn.Module):
    def __init__(self, author_features=768, paper_features=768, venue_features=768, out_channels=64):
        super().__init__()
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): RGATConv(in_channels=(author_features, paper_features),
                                                     out_channels=out_channels),
            ('paper', 'cites', 'paper'): RGATConv(in_channels=paper_features,
                                                   out_channels=out_channels),
            ('venue', 'publishes', 'paper'): RGATConv(in_channels=(venue_features, paper_features),
                                                       out_channels=out_channels),
            # ('paper', 'rev_writes', 'author'): RGATConv(in_channels=(paper_features, author_features),
            #                                          out_channels=out_channels),
            # ('paper', 'rev_publishes', 'venue'): RGATConv(in_channels=(paper_features, venue_features),
            #                                            out_channels=out_channels),                                   
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # x_dict is a dictionary of node type to node features
        # edge_index_dict is a dictionary of edge type to edge indices
        return self.conv1(x_dict, edge_index_dict)


class HGT(torch.nn.Module):
    # TODO: add RTE. 
    def __init__(self, hidden_channels, num_heads, num_layers, data=None, use_RTE=False):
        super().__init__()
        self.use_RTE = use_RTE
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            # TODO: check if -1 is the right dimension
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # TODO: check paras of HGTConv, data.metadata(), and group.
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum',use_RTE=use_RTE)
            self.convs.append(conv)
        
        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_time_dict) if self.use_RTE else conv(x_dict, edge_index_dict)
        # return self.lin(x_dict[self.task])
        return x_dict


class CitationPredictor(torch.nn.Module):
    """
    return regression and classification results.
    """
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.lin_reg = Linear(in_channels, 1)
        self.lin_cls = Linear(in_channels, num_classes)
        
    def forward(self, emb):
        return self.lin_reg(emb), self.lin_cls(emb)

class CooperationPredictor(torch.nn.Module):
    """
    return xxx.
    """
    # TODO
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.lin_reg = Linear(in_channels, 1)
        self.lin_cls = Linear(in_channels, num_classes)
        
    def forward(self, emb):
        return self.lin_reg(emb), self.lin_cls(emb)