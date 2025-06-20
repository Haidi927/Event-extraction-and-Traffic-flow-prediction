import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.kg_embedding import TransE
from models.gnn import RoadGCN
from models.transformer_predictor import TrafficEventTransformer
from utils.data_loader import MetrLaDataset
from utils.metrics import mae, rmse
import json
import yaml

# 1. 加载配置
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载实体、关系映射
with open('data/entity2id.json') as f:
    entity2id = json.load(f)
with open('data/relation2id.json') as f:
    relation2id = json.load(f)

# 3. 加载数据
train_set = MetrLaDataset(
    cfg['data']['h5_file'],
    cfg['data']['edge_index_file'],
    cfg['data']['event_triplet_file'],
    entity2id, relation2id,
    seq_len=cfg['model']['seq_len'],
    pred_len=cfg['model']['pred_len'],
    split='train'
)
train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True)

# 4. 初始化模型
kg_model = TransE(len(entity2id), len(relation2id), cfg['model']['event_emb_dim']).to(device)
gnn_model = RoadGCN(cfg['model']['gnn_input_dim'], cfg['model']['gnn_hidden_dim']).to(device)
predictor = TrafficEventTransformer(
    gnn_feat=cfg['model']['gnn_hidden_dim'] * cfg['data']['num_nodes'],
    event_dim=cfg['model']['event_emb_dim'],
    tr_dim=cfg['model']['transformer_dim'],
    nhead=cfg['model']['nhead'],
    layers=cfg['model']['layers'],
    seq_len=cfg['model']['seq_len'],
    pred_len=cfg['model']['pred_len']
).to(device)

optimizer = optim.Adam(list(kg_model.parameters()) + list(gnn_model.parameters()) + list(predictor.parameters()),
                       lr=cfg['train']['lr'])
mse_loss = nn.MSELoss()

# 5. 训练循环
for epoch in range(cfg['train']['epochs']):
    predictor.train()
    kg_model.train()
    total_loss = 0
    for traffic_x, traffic_y, triplets, edge_index in train_loader:
        traffic_x = traffic_x.to(device)      # (B, T, N)
        traffic_y = traffic_y.to(device)      # (B, pred_len, N)
        edge_index = edge_index.to(device)
        triplets = triplets.to(device)

        # KG嵌入提取
        h, r, t = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        kg_loss = kg_model(h, r, t).mean()
        event_emb = kg_model.entity_emb(h) + kg_model.relation_emb(r) + kg_model.entity_emb(t)
        event_emb = event_emb.view(traffic_x.size(0), -1)  # (B, event_emb_dim)

        # GNN提取路网特征（只用最后一帧）
        gnn_inputs = traffic_x[:, -1, :]  # (B, N)
        gnn_outputs = gnn_model(gnn_inputs.transpose(0, 1), edge_index)  # (N, hidden)
        gnn_outputs = gnn_outputs.transpose(0, 1).unsqueeze(1).repeat(1, cfg['model']['seq_len'], 1)  # (B, T, N*H)

        # 预测未来一帧
        y_pred = predictor(gnn_outputs, event_emb)  # (B, N)

        # ground truth
        y_true = traffic_y[:, -1, :]  # (B, N)
        loss = mse_loss(y_pred, y_true) + 0.1 * kg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")
