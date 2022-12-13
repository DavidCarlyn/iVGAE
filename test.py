from tqdm import tqdm

import torch
import numpy as np

import torch_geometric.transforms as T

import matplotlib.pyplot as plt

from datasets import ASIA, Earthquake
from utils import set_seed, get_test_args
from models import iVGAE_Encoder, iVGAE_Decoder, iVGAE

if __name__ == "__main__":
    args = get_test_args()

    interventions = {}
    for var, prob in zip(args.i_vars, args.i_probs):
        interventions[var] = prob
    
    # Create Dataset
    set_seed(args.seed)
    all_accuracy_runs = []
    for run in tqdm(range(args.num_of_runs), desc=f"Running {args.num_of_runs} times"):
        if args.dset == "asia":
            dset = ASIA(num_samples=args.test_samples, interventions=interventions, transform=T.NormalizeFeatures())
        elif args.dset == "earthquake":
            dset = Earthquake(num_samples=args.test_samples, interventions=interventions, transform=T.NormalizeFeatures())

        in_channels = dset.data_list[0].x.shape[1]

        # Load iVGAE Model
        encoder = iVGAE_Encoder(in_channels, args.hidden_channels, args.out_channels)
        decoder = iVGAE_Decoder(args.out_channels, args.hidden_channels, in_channels)
        model = iVGAE(encoder, decoder)
        model.load_state_dict(torch.load(args.model))
        model.cuda()
        model.eval()

        # Run inference
        total_acc = 0
        true_densities = None
        gen_densities = None
        with torch.no_grad():
            for data in dset.data_list:
                x = data.x.cuda()
                x_gen, z = model(x, data.edge_index.cuda())

                # Accuracy
                preds = torch.zeros(x.shape[0]).unsqueeze(1)
                preds[x_gen >= 0.5] = 1
                acc = (x == preds.cuda()).sum() / x.shape[0]
                total_acc += acc.item()

                # Densities
                if true_densities is None:
                    true_densities = np.zeros_like(x.cpu().detach().numpy()[:, 0])
                    gen_densities = np.zeros_like(preds.cpu().detach().numpy()[:, 0])

                true_densities += x.cpu().detach().numpy()[:, 0]
                gen_densities += preds.cpu().detach().numpy()[:, 0]

        total_acc /= len(dset.data_list)
        all_accuracy_runs.append(total_acc)
    
    #! Plot/print results
    all_accuracy_runs = np.array(all_accuracy_runs)
    print(f"Test Accuracy: {all_accuracy_runs.mean()} +- {all_accuracy_runs.std()}")

    # Can't think of a way to aggregate the densities without lossing information/interpreting properly
    # Will just use the densities from the last run to plot.
    
    # Normalize
    true_densities /= len(dset.data_list)
    gen_densities /= len(dset.data_list)
    node_names = dset.nodes

    if args.dset == "asia":
        rows = 2
        cols = 4
    elif args.dset == "earthquake":
        rows = 1
        cols = 5

    gt_color = "lightblue"
    colors = ["pink", "red", "aqua", "yellow", "blue", "green", "purple", "orange"]
    
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(12, 6))

    for i, (node, true_den, gen_den) in enumerate(zip(node_names, true_densities, gen_densities)):
        row = i // cols
        col = i % cols
        axes[row, col].set_title(node, fontsize = 20)
        axes[row, col].set_ylim(0.0, 1.0)
        axes[row, col].bar([0.05, 0.85], [1 - true_den, true_den], label="GT", width=0.1, color=gt_color)
        axes[row, col].bar([0.15, 0.95], [1 - gen_den, gen_den], label="Prediction", width=0.1, color=colors[i])
        axes[row, col].legend()

    inters = "None"
    save_txt = ""
    if len(interventions.keys()) > 0:
        save_txt = "_interv_" + "_".join(interventions.keys())
        inters = ""
        for k, v in interventions.items():
            inters += f"{k} ({v}) "
    fig.suptitle(f"Dataset: {args.dset} | Intervention on: {inters}", fontsize=32)
    fig.tight_layout()
    plt.savefig(f"inference_{args.dset}{save_txt}.png")

