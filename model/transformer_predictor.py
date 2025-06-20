import torch.nn as nn

class TrafficEventTransformer(nn.Module):
    def __init__(self, gnn_feat, event_dim, tr_dim, nhead, layers, seq_len, pred_len):
        super().__init__()
        self.event_fc = nn.Linear(event_dim, tr_dim)
        self.input_embed = nn.Linear(gnn_feat + tr_dim, tr_dim)
        encoder = nn.TransformerEncoderLayer(d_model=tr_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=layers)
        self.predictor = nn.Linear(tr_dim, gnn_feat)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, traffic_seq, event_emb):
        e = self.event_fc(event_emb).unsqueeze(1).repeat(1, self.seq_len, 1)
        x = torch.cat([traffic_seq, e], dim=-1)
        x = self.input_embed(x).permute(1,0,2)
        out = self.encoder(x)[-1]
        return self.predictor(out)
