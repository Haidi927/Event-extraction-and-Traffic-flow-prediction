import torch.nn as nn

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

    def forward(self, h, r, t):
        return (self.entity_emb(h) + self.relation_emb(r) - self.entity_emb(t)).norm(p=1, dim=1)
