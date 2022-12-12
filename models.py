import torch.nn as nn
import torch_geometric.nn as pyg_nn

class iVGAE_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv0 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv1 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.lin_mean = nn.Linear(hidden_channels, out_channels)
        self.lin_logstd = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv0(x, edge_index)
        h = nn.ReLU()(h)
        h = self.conv1(h, edge_index)
        h = nn.ReLU()(h)
        mean = self.lin_mean(h)
        logstd = self.lin_logstd(h)
        return mean, logstd

class iVGAE_Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv0 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv1 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, z, edge_index, sigmoid=True):
        h = self.conv0(z, edge_index)
        h = nn.ReLU()(h)
        h = self.conv1(h, edge_index)
        h = nn.ReLU()(h)
        out = self.linear(h)
        if sigmoid:
            out = nn.Sigmoid()(out)
        return out

class iVGAE(pyg_nn.VGAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def decode(self, z, pos_edge_index):
        x_gen = self.decoder(z, pos_edge_index)
        return x_gen

    def forward(self, x, pos_edge_index):
        z = self.encode(x, pos_edge_index)
        x_gen = self.decode(z, pos_edge_index)
        return x_gen, z
