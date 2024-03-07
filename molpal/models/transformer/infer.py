from molpal.models.transformer.model import LitTransformer, InferenceDataset, collate_smiles
from molpal.models.transformer.tokenizer.tokenizer import MolTranBertTokenizer, get_vocab_path
from molpal.models.transformer.utils import StandardScaler
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple, Union
from pytorch_lightning.callbacks import BasePredictionWriter
import numpy as np
import pandas as pd
import tempfile, os, shutil, argparse
from pathlib import Path
from functools import partial
from timeit import default_timer as time

KWARGS = {
    'config': 'config_trans.yaml', 'clear_tensorboard_dir': False, 'seed': None, 
    'verbose': 0, 'ncpu': 24, 'write_final': True, 'scores_csvs': None, 'fingerprint': 'pair', 
    'radius': 2, 'length': 2048, 'pool': 'lazy', 'libraries': ['libraries/Enamine50k.csv'], 
    'delimiter': ',', 'smiles_col': 0, 'cxsmiles': False, 'fps': None, 'cluster': False, 
    'cache': False, 'invalid_idxs': None, 'metric': 'greedy', 'init_size': 0.01, 
    'batch_sizes': [0.01], 'epsilon': 0.0, 'objective': 'lookup', 'minimize': True, 
    'objective_config': 'examples/objective/Enamine50k_lookup.ini', 'model': 'transformer', 
    'epochs': 5, 'test_batch_size': 8000, 'model_seed': None, 'init_lr': 0.0001, 'max_lr': 0.001, 
    'final_lr': 0.0001, 'ddp': True, 'precision': 32, 'conf_method': 'none', 'title_line': True, 
    'path': 'run', 'log_dir': 'run/log'
}

infer_config = {
    # "model": self.model,
    "uncertainty": 'none',
    "dataset_type": 'regression',
    "batch_size": 4096,
    "max_epochs": 5,
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

class CustomWriter(BasePredictionWriter):
    """Pytorch Lightning Callback that saves predictions and the corresponding batch
    indices in a temporary folder when using multigpu inference.

    Args:
        write_interval (str): When to perform write operations. Defaults to 'epoch'
    """
    def __init__(self, infer_dir, n_iter, write_interval="epoch") -> None:
        super().__init__(write_interval)
        self.infer_dir = infer_dir
        self.temp_dir = os.path.join(infer_dir, 'tmp_folder')
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        self.n_iter = n_iter

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Saves predictions after running inference on all samples."""

        # We need to save predictions in the most secure manner possible to avoid
        # multiple users and processes writing to the same folder.
        # For that we will create a tmp folder that will be shared only across
        # the DDP processes that were created
        if trainer.is_global_zero:
            temp_fname= tempfile.mkdtemp(dir=self.temp_dir)
            output_dir = [
                temp_fname,
            ]
        else:
            output_dir = [
                None,
            ]

        torch.distributed.broadcast_object_list(output_dir)

        # Make sure every process received the output_dir from RANK=0
        torch.distributed.barrier()  
        # Now that we have a single output_dir shared across processes we can save
        # prediction along with their indices.
        self.output_dir = output_dir[0]
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank        
        print('Global rank:', trainer.global_rank)
        torch.save(
            predictions, os.path.join(self.output_dir, f"pred_{trainer.global_rank}.pt")
        )
        # Make sure every process has saved the prediction .pt file
        torch.distributed.barrier()  
        if trainer.is_global_zero:
            print('My temp files is in:', self.output_dir)
            self.gather_all_predictions(trainer)
            self.cleanup(trainer)
        return
    
    def gather_all_predictions(self, trainer):
        if trainer.is_global_zero:
            idx = []
            preds = []
            for rank in range(trainer.world_size):
                pred_fname = os.path.join(self.output_dir, f"pred_{rank}.pt")
                print('Load prediciton from:', pred_fname)
                results = torch.load(pred_fname)[0]
                idx.extend([i[0] for i in results])
                preds.extend([i[1] for i in results])
            idx = np.concatenate(idx)
            print(idx.shape, trainer.world_size)
            predictions = torch.cat(preds)
            argsort_idx = np.argsort(idx)
            predictions = predictions[argsort_idx].cpu().numpy()
            # predictions = predictions[predictions[:, 0].sort()[1]].cpu().numpy()
            np.savez(os.path.join(self.infer_dir, 'Iter%d_preds.npz'%self.n_iter), idx=idx[argsort_idx], pred=predictions)
        return 
    
    def cleanup(self, trainer):
        """Cleans temporary files."""
        if trainer.is_global_zero:
            shutil.rmtree(self.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', type=int)
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-u', '--uncertainty', type=str, default='none')
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    args = parser.parse_args()
    return args

def get_n_node(ngpu):
    assert ngpu in {1, 2, 3, 4, 6, 8, 9, 12}, "Currrently, DDP only support 1, 2, 3, 4, 6, 8, 9, or 12 GPUs"
    n_node = (ngpu-1)//4 + 1
    if ngpu <= 4:
        return n_node, ngpu
    return n_node, ngpu // n_node


def main(args):
    ngpu = args.ngpu
    ncpu = ngpu * 8
    n_nodes, tasks_per_node = get_n_node(ngpu)
    if args.uncertainty == 'none' or args.uncertainty == 'mve':
        infer_config['uncertainty'] = args.uncertainty
    else:
        print('Uncertainty type not supported! Use default "none" instead.')
    
    # df = pd.read_csv(os.path.join(args.dir, 'run/infer_files/infer_data_iter%d.csv.gz'%args.iter), compression='gzip', header=0)
    df = np.load(os.path.join(args.dir, 'run/infer_files/infer_data_iter%d.npz'%args.iter), allow_pickle=True)

    tokenizer = MolTranBertTokenizer(get_vocab_path(args.dir))
    model = LitTransformer(
        infer_config, 
        tokenizer = MolTranBertTokenizer(get_vocab_path(args.dir))
    ).load_from_checkpoint(
        os.path.join(args.dir, 'run/chkpts/iter_%d/model/molformer_iter%d.ckpt'%(args.iter, args.iter)), strict=False, config=infer_config, 
        tokenizer=tokenizer, vocab=len(tokenizer.vocab)
    )

    callbacks = [
        CustomWriter(infer_dir=os.path.join(args.dir, 'run/infer_files'), n_iter=args.iter),
    ]

    log_dir = os.path.join(args.dir, 'run/log/iter_%d'%args.iter)
    logger = TensorBoardLogger(save_dir=log_dir, name='infer_ddp')
    trainer = PlTrainer(
        accelerator='gpu',
        logger=logger,
        # devices=ngpu,
        devices=tasks_per_node,
        num_nodes=n_nodes,
        precision=32,
        strategy='ddp',
        callbacks=callbacks,
        enable_progress_bar = True,
        enable_model_summary= False,
        enable_checkpointing= False, 
    )

    inferdataset = InferenceDataset(df['smiles'].astype(str), range(len(df['smiles'])), ncpu=8, aug=infer_config['aug'])
    pred_dataloader = DataLoader(
        dataset=inferdataset,
        batch_size=infer_config['batch_size'],
        num_workers=8,
        shuffle=False,
        collate_fn=partial(collate_smiles, workdir=args.dir),
    )
    model.eval()
    # TO-DO: Implement checkpoint load and predict using the ModelCheckpointAtEpochEnd
    trainer.predict(model=model, dataloaders=pred_dataloader, return_predictions=False)
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
