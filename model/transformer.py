import torch.nn as nn

def get_transformer_encoder(transformer_dim, nhead, num_layers):
    encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    return transformer_encoder
