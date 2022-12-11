import torch.nn as nn
import torch.optim as optim

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges

from models import iVGAE_Encoder, iVGAE_Decoder, iVGAE

if __name__ == "__main__":
    # Hyperparameters
    hidden_channels = 500
    out_channels = 50
    epochs = 300
    lr = 0.01

    # Create Dataset
    dataset = Planetoid("data/citeseer", "CiteSeer", transform=T.NormalizeFeatures())
    data = dataset[0]
    print(data.x[0][-1])
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)


    # Build iVGAE Model
    encoder = iVGAE_Encoder(data.x.shape[1], hidden_channels, out_channels)
    decoder = iVGAE_Decoder(out_channels, hidden_channels, data.x.shape[1])
    model = iVGAE(encoder, decoder)

    # Setup Training
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # Train
    for epoch in range(epochs):
        model.train()
        x_gen, z = model(data.x, data.train_pos_edge_index)
        loss = loss_fn(data.x, x_gen)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1} - Loss: {loss.item()}")
