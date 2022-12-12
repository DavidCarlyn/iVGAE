import torch
import torch.nn as nn
import torch.optim as optim

import torch_geometric.transforms as T

import matplotlib.pyplot as plt

from datasets import ASIA, Earthquake
from utils import set_seed, get_args
from models import iVGAE_Encoder, iVGAE_Decoder, iVGAE

def run(args, interventions={}):
    exp_name = f"exp1_{args.dset}_method_1"
    if len(interventions.keys()) > 0:
        exp_name += "_interv_" + "_".join(interventions.keys())

    # Create Dataset
    set_seed(args.seed)
    if args.dset == "asia":
        train_dset = ASIA(num_samples=args.train_samples, interventions=interventions, transform=T.NormalizeFeatures())
        val_dset = ASIA(num_samples=args.val_samples, interventions=interventions, transform=T.NormalizeFeatures())
    elif args.dset == "earthquake":
        train_dset = Earthquake(num_samples=args.train_samples, interventions=interventions, transform=T.NormalizeFeatures())
        val_dset = Earthquake(num_samples=args.val_samples, interventions=interventions, transform=T.NormalizeFeatures())

    in_channels = train_dset.data_list[0].x.shape[1]

    # Build iVGAE Model
    encoder = iVGAE_Encoder(in_channels, args.hidden_channels, args.out_channels)
    decoder = iVGAE_Decoder(args.out_channels, args.hidden_channels, in_channels)
    model = iVGAE(encoder, decoder).cuda()

    # Setup Training
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    # Train
    all_losses = []
    all_train_acc = []
    all_val_acc = []
    best_val_acc = 0
    for epoch in range(args.epochs):
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
            for data in val_dset.data_list:
                x = data.x.cuda()
                x_gen, z = model(x, data.edge_index.cuda())
                preds = torch.zeros(x.shape[0]).unsqueeze(1)
                preds[x_gen >= 0.5] = 1

                acc = (x == preds.cuda()).sum() / x.shape[0]
                total_acc += acc.item()
        total_acc /= len(val_dset.data_list)
        print(f"Epoch: {epoch + 1} - Val Acc: {total_acc}")
        all_val_acc.append(total_acc)
        if total_acc >= best_val_acc:
            torch.save(model.state_dict(), f"{exp_name}_model.pt")

        if (epoch+1) % 1 == 0:
            plt.figure(figsize=(16, 9))
            plt.plot(all_losses)
            plt.savefig(f"{exp_name}_loss.png")
            plt.close()
            
            plt.figure(figsize=(16, 9))
            plt.plot(all_train_acc, label="Train Accuracy")
            plt.plot(all_val_acc, label="Validation Accuracy")
            plt.legend()
            plt.savefig(f"{exp_name}_accuracy.png")
            plt.close()

if __name__ == "__main__":
    args = get_args()

    interventions = {}
    for var, prob in zip(args.i_vars, args.i_probs):
        interventions[var] = prob
    
    run(args, interventions)

