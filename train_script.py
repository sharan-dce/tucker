import argparse
import data_loader
import numpy as np
from training_routine import train, measure_performance
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tucker", nargs="?",
                        help="Which model to use: tucker, distmult or rescal.")
    parser.add_argument("--dataset", type=str, default="FB15k", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=20, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=64, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")

    args = parser.parse_args()
    dl = data_loader.DataLoader(args.dataset)

    if args.model == 'tucker':
        from models import tucker
        model = tucker.TuckER(
            len(dl.entities),
            len(dl.relations),
            np.random.normal(size=[args.edim, args.rdim, args.edim])
        ).to(device)
    elif args.model == 'rescal':
        from models import rescal
        model = rescal.RESCAL(
            len(dl.entities),
            len(dl.relations),
            args.edim
        ).to(device)
    elif args.model == 'distmult':
        from models import distmult
        model = distmult.DistMult(
            len(dl.entities),
            len(dl.relations),
            args.edim
        ).to(device)
    else:
        raise Exception("Model not defined!")

    train(model, data_loader=dl, epochs=args.num_iterations, lr=args.lr, lr_decay=0.99, batch_size=args.batch_size, label_smoothing_rate=args.label_smoothing)
