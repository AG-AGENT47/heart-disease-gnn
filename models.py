# models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class InductiveGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(InductiveGCN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        # FIX: Handle both model(data) and model(x, edge_index)
        if edge_index is None:
            # If only one arg is passed, it's the 'data' object
            data = x
            x = data.x
            edge_index = data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classifier Head
        x = self.fc(x)
        return F.log_softmax(x, dim=1)