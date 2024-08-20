import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.data import Data, Batch

class GraphWaveletConv(nn.Module):
    def __init__(self, in_channels, out_channels, scales):
        super(GraphWaveletConv, self).__init__()
        self.scales = scales
        self.conv = nn.ModuleList([GCNConv(in_channels, out_channels) for _ in range(scales)])

    def forward(self, x, edge_index, edge_weight):
        # Normalize edge weights (example: min-max scaling)
        min_weight = edge_weight.min()
        max_weight = edge_weight.max()
        normalized_edge_weight = (edge_weight - min_weight) / (max_weight - min_weight)

        wavelets = []
        for i, conv in enumerate(self.conv):
            wavelet = conv(x, edge_index, normalized_edge_weight)  # Use normalized edge weights
            wavelets.append(wavelet)
        return torch.cat(wavelets, dim=1)

class AdaptiveGraphPooling(nn.Module):
    def __init__(self, in_channels, ratio):
        super(AdaptiveGraphPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.att = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, 1)
        )

        # Initialize weights using Kaiming initialization
        for layer in self.att:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x, edge_index, edge_weight, batch):
        scores = self.att(x)
        scores = scores.squeeze()
        scores = F.softmax(scores, dim=0)
        num_nodes = x.size(0)
        num_keep = max(1, min(int(num_nodes * self.ratio), num_nodes))
        
        values, indices = torch.topk(scores, num_keep)
        x = x[indices]

        # Create a mapping from old indices to new indices
        new_indices = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
        new_indices[indices] = torch.arange(num_keep, device=x.device)

        # Update the edge indices based on the new node indices
        edge_index = new_indices[edge_index]

        # Remove edges that point to nodes that have been removed
        mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

        # Update the batch vector based on the new node indices
        if batch is not None:
            batch = batch[indices]

        return x, edge_index, edge_weight, batch, indices

    def filter_adj(self, edge_index, edge_weight, indices):
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        for idx in indices:
            mask |= (edge_index[0] == idx) | (edge_index[1] == idx)
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
        return edge_weight

class SpatioTemporalFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatioTemporalFusion, self).__init__()
        self.spatial_att = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, 1)
        )
        self.temporal_att = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, 1)
        )

        # Initialize weights using Kaiming initialization
        for att in [self.spatial_att, self.temporal_att]:
            for layer in att:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        self.conv = GATConv(in_channels * 2, out_channels)

    def forward(self, x_spatial, x_temporal, edge_index, edge_weight):
        spatial_scores = self.spatial_att(x_spatial)
        temporal_scores = self.temporal_att(x_temporal)
        scores = torch.softmax(torch.cat([spatial_scores, temporal_scores], dim=1), dim=1)
        x_fused = torch.cat([x_spatial * scores[:, 0:1], x_temporal * scores[:, 1:2]], dim=1)
        x_fused = self.conv(x_fused, edge_index)
        return x_fused

class AMST_GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, scales, pooling_ratios):
        super(AMST_GNN, self).__init__()
        self.scales = scales
        self.pooling_ratios = pooling_ratios
        self.conv1 = GraphWaveletConv(in_channels, hidden_channels, scales)
        self.pool1 = AdaptiveGraphPooling(hidden_channels * scales, pooling_ratios[0])
        self.conv2 = GraphWaveletConv(hidden_channels * scales, hidden_channels, scales)
        self.pool2 = AdaptiveGraphPooling(hidden_channels * scales, pooling_ratios[1])
        self.fusion = SpatioTemporalFusion(hidden_channels * scales, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, out_channels)

    def forward(self, x_spatial, x_temporal, edge_index_spatial, edge_weight_spatial, edge_index_temporal, edge_weight_temporal, batch):
        x_spatial = self.conv1(x_spatial, edge_index_spatial, edge_weight_spatial)
        x_temporal = self.conv1(x_temporal, edge_index_temporal, edge_weight_temporal)
        x_spatial, edge_index_spatial, edge_weight_spatial, batch_spatial, _ = self.pool1(x_spatial, edge_index_spatial, edge_weight_spatial, batch)
        x_temporal, edge_index_temporal, edge_weight_temporal, batch_temporal, _ = self.pool1(x_temporal, edge_index_temporal, edge_weight_temporal, batch)
        x_spatial = self.conv2(x_spatial, edge_index_spatial, edge_weight_spatial)
        x_temporal = self.conv2(x_temporal, edge_index_temporal, edge_weight_temporal)
        x_spatial, edge_index_spatial, edge_weight_spatial, batch_spatial, _ = self.pool2(x_spatial, edge_index_spatial, edge_weight_spatial, batch_spatial)
        x_temporal, edge_index_temporal, edge_weight_temporal, batch_temporal, _ = self.pool2(x_temporal, edge_index_temporal, edge_weight_temporal, batch_temporal)
        x_fused = self.fusion(x_spatial, x_temporal, edge_index_spatial, edge_weight_spatial)
        x_fused = self.conv3(x_fused, edge_index_spatial)
        x_fused = F.relu(x_fused)
        x_fused = torch.mean(x_fused, dim=0)
        return x_fused

# Test AMST_GNN with mock data
if __name__ == "__main__":
    # Mock data setup
    num_nodes = 100
    num_features = 16
    num_classes = 3
    scales = 3
    pooling_ratios = [0.8, 0.6]

    # Create mock spatial and temporal data
    x_spatial = torch.randn((num_nodes, num_features))
    x_temporal = torch.randn((num_nodes, num_features))

    # Create mock edge indices and weights
    edge_index_spatial = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_weight_spatial = torch.randn((num_nodes * 2,))
    edge_index_temporal = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_weight_temporal = torch.randn((num_nodes * 2,))

    # Mock batch vector for handling graph batching
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Instantiate AMST_GNN model
    model = AMST_GNN(in_channels=num_features,
                     hidden_channels=32,
                     out_channels=num_classes,
                     scales=scales,
                     pooling_ratios=pooling_ratios)

    # Forward pass with mock data
    out = model(x_spatial, x_temporal, edge_index_spatial, edge_weight_spatial,
                edge_index_temporal, edge_weight_temporal, batch)

    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")
