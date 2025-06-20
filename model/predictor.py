import torch
import torch.nn as nn
from .gnn import RoadGNN
from .transformer import get_transformer_encoder

class TrafficFlowPredictor(nn.Module):
    def __init__(self, gnn_in_dim, gnn_hidden_dim, event_dim, transformer_dim, nhead, num_layers, seq_len, pred_len):
        super().__init__()
        self.gnn = RoadGNN(gnn_in_dim, gnn_hidden_dim)
        self.event_fc = nn.Linear(event_dim, transformer_dim)
        self.input_embedding = nn.Linear(gnn_hidden_dim + transformer_dim, transformer_dim)
        self.transformer_encoder = get_transformer_encoder(transformer_dim, nhead, num_layers)
        self.predictor = nn.Linear(transformer_dim, gnn_in_dim)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, traffic_x, edge_index, event_emb):
        batch_size, seq_len, num_nodes, feat_dim = traffic_x.shape
        traffic_x = traffic_x.view(-1, num_nodes, feat_dim).permute(1,0,2)
        gnn_out = self.gnn(traffic_x, edge_index)
        gnn_out = gnn_out.permute(1,0,2).view(batch_size, seq_len, num_nodes, -1)
        gnn_out = gnn_out.view(batch_size, seq_len, -1)
        event_emb = self.event_fc(event_emb).unsqueeze(1).repeat(1, seq_len, 1)
        x = torch.cat([gnn_out, event_emb], dim=-1)
        x = self.input_embedding(x)
        x = x.permute(1,0,2)
        enc_out = self.transformer_encoder(x)
        last_hidden = enc_out[-1]
        out = self.predictor(last_hidden)
        out = out.view(batch_size, num_nodes, feat_dim)
        return out
