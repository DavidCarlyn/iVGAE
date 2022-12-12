import torch
import torch.nn as nn
import torch.optim as optim

import torch_geometric.transforms as T

import matplotlib.pyplot as plt

from datasets import ASIA
from utils import set_seed
from models import iVGAE_Encoder, iVGAE_Decoder, iVGAE

if __name__ == "__main__":
    # Hyperparameters
    hidden_channels = 10
    out_channels = 10
    epochs = 6000
    lr = 0.001

    # Create Dataset
    set_seed(2022)
    train_dset = ASIA(num_samples=8000, transform=T.NormalizeFeatures())
    test_dset = ASIA(num_samples=1000, transform=T.NormalizeFeatures())
    in_channels = train_dset.data_list[0].x.shape[1]

    # Build iVGAE Model
    encoder = iVGAE_Encoder(in_channels, hidden_channels, out_channels)
    decoder = iVGAE_Decoder(out_channels, hidden_channels, in_channels)
    model = iVGAE(encoder, decoder).cuda()

    # Setup Training
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # Train
    all_losses = []
    all_train_acc = []
    all_test_acc = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        for data in train_dset.data_list:
            x = data.x.cuda()
            x_gen, z = model(x, data.edge_index.cuda())
            loss = loss_fn(x_gen, x)
            kl_loss = (1 / data.num_nodes) * model.kl_loss()

            loss = loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.zeros(x.shape[0]).unsqueeze(1)
            preds[x_gen >= 0.5] = 1

            acc = (x == preds.cuda()).sum() / x.shape[0]
            total_acc += acc.item()

        total_loss /= len(train_dset.data_list)
        total_acc /= len(train_dset.data_list)
        print(f"Epoch: {epoch + 1} - Loss: {total_loss}, Train Acc: {total_acc}")
        all_losses.append(total_loss)
        all_train_acc.append(total_acc)

        model.eval()
        total_acc = 0
        with torch.no_grad():
            for data in test_dset.data_list:
                x = data.x.cuda()
                x_gen, z = model(x, data.edge_index.cuda())
                preds = torch.zeros(x.shape[0]).unsqueeze(1)
                preds[x_gen >= 0.5] = 1

                acc = (x == preds.cuda()).sum() / x.shape[0]
                total_acc += acc.item()
        total_acc /= len(test_dset.data_list)
        print(f"Epoch: {epoch + 1} - Test Acc: {total_acc}")
        all_test_acc.append(total_acc)

        if (epoch+1) % 1 == 0:
            plt.figure(figsize=(16, 9))
            plt.plot(all_losses)
            plt.savefig("loss.png")
            plt.close()
            
            plt.figure(figsize=(16, 9))
            plt.plot(all_train_acc, label="Train Accuracy")
            plt.plot(all_test_acc, label="Test Accuracy")
            plt.legend()
            plt.savefig("accuracy.png")
            plt.close()


    # TODO: Perform on true testing dataset
    # TODO: Create

