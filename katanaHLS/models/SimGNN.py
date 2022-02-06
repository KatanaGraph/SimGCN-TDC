from dataclasses import dataclass

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCN, GAT

from .predictor import Predictor

__all__ =['SimGCN', 'SimGNNConfig']

@dataclass
class SimGNNConfig:
    in_channels:int
    gnn: str = 'GCN' 
    hidden_channels:int = 128
    num_layers: int = 2     
    batchnorm: bool = False
    dropout: float = 0.0
    cached: float = False
    predictor_hidden_feats:int = 128
    n_tasks:int = 1
    activation=torch.nn.ReLU(inplace=True)

class SimGCN(torch.nn.Module):
    def __init__(self, sim_GCN_config:SimGNNConfig):
        super(SimGCN, self).__init__()
        torch.manual_seed(12345)
        
        self.config = sim_GCN_config
        bn = None
        if sim_GCN_config.batchnorm:
            bn = torch.nn.BatchNorm1d(sim_GCN_config.hidden_channels)
        if sim_GCN_config.gnn == 'GCN':
            self.gnn = GCN(in_channels=sim_GCN_config.in_channels,
                        hidden_channels=sim_GCN_config.hidden_channels,
                        num_layers=sim_GCN_config.num_layers,
                        act=sim_GCN_config.activation,                       
                        norm=bn,
                        jk='cat',
                        dropout=sim_GCN_config.dropout
                        )
        elif sim_GCN_config.gnn == 'GAT':
            self.gnn = GAT(in_channels=sim_GCN_config.in_channels,
                        hidden_channels=sim_GCN_config.hidden_channels,
                        num_layers=sim_GCN_config.num_layers,
                        act=sim_GCN_config.activation,                       
                        norm=bn,
                        jk='cat',
                        dropout=sim_GCN_config.dropout
                        )
        else: 
            raise ValueError("Invalid similairty GNN string.")
        gnn_out_feats = self.gnn.out_channels
        self.predictor = Predictor(gnn_out_feats, 
                                   sim_GCN_config.predictor_hidden_feats, 
                                   sim_GCN_config.n_tasks)
        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gnn.reset_parameters()
        self.predictor.reset_parameters()
        
    def forward(self, x, edge_index):
        # 1. Obtain node embeddings 
        emb = self.gnn(x, edge_index)
        
        # 2. Run the classifier to make the final prediction
        pred = self.predictor(emb)
        
        if self.training:
            return pred
        else:
            return pred, emb

