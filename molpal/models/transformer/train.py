from molpal.models.transformer.model import LitTransformer, collate_smiles, PropertyPredictionDataModule
from molpal.models.transformer.tokenizer.tokenizer import MolTranBertTokenizer, get_vocab_path
from pytorch_lightning.callbacks import ModelCheckpoint
from molpal.models.transformer.utils import StandardScaler, split_data
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from timeit import default_timer as time
import argparse, os


# KWARGS = {
#     'config': 'config_trans.yaml', 'clear_tensorboard_dir': False, 'seed': None, 
#     'verbose': 0, 'ncpu': 24, 'write_final': True, 'scores_csvs': None, 'fingerprint': 'pair', 
#     'radius': 2, 'length': 2048, 'pool': 'lazy', 'libraries': ['libraries/Enamine50k.csv'], 
#     'delimiter': ',', 'smiles_col': 0, 'cxsmiles': False, 'fps': None, 'cluster': False, 
#     'cache': False, 'invalid_idxs': None, 'metric': 'greedy', 'init_size': 0.01, 
#     'batch_sizes': [0.01], 'epsilon': 0.0, 'objective': 'lookup', 'minimize': True, 
#     'objective_config': 'examples/objective/Enamine50k_lookup.ini', 'model': 'transformer', 
#     'epochs': 5, 'test_batch_size': 8000, 'model_seed': None, 'init_lr': 0.0001, 'max_lr': 0.001, 
#     'final_lr': 0.0001, 'ddp': True, 'precision': 32, 'conf_method': 'none', 'title_line': True, 
#     'path': 'run', 'log_dir': 'run/log'
# }

train_config = {
    "uncertainty": 'none',
    "dataset_type": 'regression',
    "batch_size": 600,
    "max_epochs": 50,
    "init_lr": 1.6e-4,
    "metric": "rmse",
    "n_head": 12,
    "n_layer": 12,
    "n_embd": 768,
    "d_dropout": 0.1,
    "dropout": 0.1,
    "lr_start": 1.6e-4,
    "lr_multiplier": 1,
    "num_feats": 32,
    "dims": [768,768,768,1],
    "mode": 'avg',
    "aug": True,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', type=int)
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-u', '--uncertainty', type=str, default='none')
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    args = parser.parse_args()
    return args

def get_n_node(ngpu):
    assert ngpu in {1, 2, 3, 4, 6}, "Currrently, DDP only support 1, 2, 3, 4, or 6 GPUs"
    if ngpu <= 4:
        return 1, ngpu
    return 2, ngpu // 2


def main(args):
    ngpu = args.ngpu
    train_config['lr_start'] = train_config['lr_start'] * ngpu
    # ngpu = 1
    ncpu = ngpu * 8
    n_nodes, tasks_per_node = get_n_node(ngpu)
    if args.uncertainty == 'none' or args.uncertainty == 'mve':
        train_config['uncertainty'] = args.uncertainty
    else:
        print('Uncertainty type not supported! Use default "none" instead.')

    train_df = pd.read_csv(os.path.join(args.dir, 'run/train_files/train_data_iter%d.csv.gz'%args.iter), compression='gzip', header=0)
    val_df = pd.read_csv(os.path.join(args.dir, 'run/train_files/val_data_iter%d.csv.gz'%args.iter), compression='gzip', header=0)
    train_x, train_y = train_df['smiles'].to_numpy(), train_df['score'].to_numpy()
    val_x, val_y = val_df['smiles'].to_numpy(), val_df['score'].to_numpy()
    train_y, val_y = train_y.astype(float), val_y.astype(float)

    model_ckpt_path = Path(os.path.join(args.dir, 'run/chkpts/iter_%d/model'%args.iter))
    model_ckpt_path.mkdir(parents=True, exist_ok=True)


    datamodule = PropertyPredictionDataModule(train_config, workdir=args.dir)

    datamodule.prepare_data(train_x, train_y, val_x, val_y)
    train_dataloader = datamodule.train_dataloader(train_config['batch_size'], 8, shuffle=True)
    val_dataloader = datamodule.val_dataloader(train_config['batch_size'], 8, shuffle=False)

    tokenizer = MolTranBertTokenizer(get_vocab_path(args.dir))
    model = LitTransformer(
        train_config, 
        tokenizer = MolTranBertTokenizer(get_vocab_path(args.dir))
    ).load_from_checkpoint(
        os.path.join(args.dir, 'molpal/models/transformer/pretrained_ckpt/molformer_weights.ckpt'), strict=False, config=train_config, 
        tokenizer=tokenizer, vocab=len(tokenizer.vocab)
    )

    callbacks = [
        EarlyStopping("val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            monitor='val_loss', dirpath=os.path.join(args.dir, 'run/chkpts/iter_%d/model'%args.iter), 
            filename='molformer_iter%d'%args.iter, save_weights_only=True
        )
    ]

    # log_dir = Path('log')
    log_dir = os.path.join(args.dir, 'run/log/iter_%d'%args.iter)
    logger = TensorBoardLogger(save_dir=log_dir, name='train_ddp')
    trainer = PlTrainer(
        max_epochs=train_config['max_epochs'],
        accelerator='gpu',
        logger=logger,
        # devices=ngpu,
        devices=tasks_per_node,
        num_nodes=n_nodes,
        precision=32,
        strategy='ddp',
        callbacks=callbacks,
        enable_progress_bar = False,
        enable_model_summary= False,
        enable_checkpointing= True, 
    )
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
   args = parse_args()
   main(args)