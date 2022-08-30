import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import torchio as tio
import argparse
from denoising_diffusion_pytorch import Trainer, Unet, GaussianDiffusion


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Single cell generational models")

    parser.add_argument(
        "--dataset_path",
        default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/mvries/Kaggle/hpa-single-cell-image-classification",
        type=str,
        help="Please provide the path to the dataset of 3D tif images",
    )
    parser.add_argument(
        "--dataframe",
        default="all_data_removedwrong_ori_removedTwo.csv",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mvries/Documents/DiffusionHPA/",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--learning_rate",
        default=8e-5,
        type=float,
        help="Please provide the learning rate " "for the autoencoder training.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Please provide the batch size.",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI"
        "/DYNCESYS/mvries/Projects/TearingNetNew/Reconstruct_dgcnn_cls_k20_plane/models/shapenetcorev2_250.pkl",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )
    parser.add_argument(
        "--num_steps",
        default=700000,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--image_size",
        default=244,
        type=int,
        help="Input image size.",
    )
    args = parser.parse_args()

    model = Unet(dim=64, channels=3).cuda()
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=1000,  # number of steps
        loss_type="l1",  # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        folder=args.dataset_path,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision,
        results_folder=args.output_dir + "/results/"
    )

    trainer.train()
