import argparse

import torch 

import pytorch_lightning as pl 
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger
from transformers import data
from multiconer.config import (
    WNUT_BIO_ID, 
    BIO_TAGS
)

from multiconer.dataset import CoNLLDataModule
from multiconer.model import MultiCoNER

if __name__=="__main__":

    # Default seed
    seed_everything(42)

    # Parse Args
    parser = argparse.ArgumentParser(description="Process some parameters and Hyperparameters")

    # Hyperparams
    parser.add_argument('--lr', type=float, default=3e-5, help="learning_rate")
    parser.add_argument('--epochs', type=int, default=5, help="epochs")
    parser.add_argument('--base_model', type=str, default="xlm-roberta-base", help="Base model")
    parser.add_argument('--seq_length', type=int, default=32, help="max_seq_length")
    parser.add_argument('--padding', type=str, default="max_length", help="paddind type")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--tagging_scheme', type=str, default='wnut', help="Tagging scheme")

    # Hardware args
    parser.add_argument('--gpu', type=int, default=1, help="GPUS")
    parser.add_argument('--workers', type=int, default=4, help="CPU threads")
    parser.add_argument('--nodes', type=int, default=1, help="num nodes")
    parser.add_argument('--strategy', type=str, default="ddp", help="cluster training strategy")


    # Paths
    parser.add_argument('--dataset', type=str, default=None, required=True, help="dataset path")
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--name', type=str, default="trails", help="run names")
    parser.add_argument('--checkpoint', type=bool, default=False, help="Checkpointing")


    args = parser.parse_args()

    tag_scheme = WNUT_BIO_ID
    if args.tagging_scheme == 'bio':
        tag_scheme = BIO_TAGS    
    else:
        tag_scheme = WNUT_BIO_ID

    dataset_name = None 
    if args.dataset_name == 'twitter':
        dataset_name = 'twitter'
    else:
        dataset_name = None
    
    # INIT DATA
    dm = CoNLLDataModule(
        model_name_or_path=args.base_model, 
        path_dataset=args.dataset, 
        dataset_name=dataset_name,
        tag_to_id=tag_scheme,
        num_workers=args.workers, 
        max_seq_length=args.seq_length, 
        padding=args.padding,
        batch_size=args.batch_size
    )

    dm.setup("fit")
    # print(next(iter(dm.train_dataloader())))


    #INIT MODEL
    model = MultiCoNER(
        model_name_or_path=args.base_model, 
        tag_to_id=tag_scheme,
        learning_rate=args.lr, 
        max_seq_len=args.seq_length, 
        padding=args.padding
    )

    # USE TRAINER
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=args.epochs, 
        gpus=args.gpu, 
        logger=WandbLogger(project="Code-Mix NER", name=args.name), 
        num_nodes=args.nodes, 
        strategy=args.strategy, 
        checkpoint_callback=args.checkpoint
    )

    trainer.fit(model, dm)
