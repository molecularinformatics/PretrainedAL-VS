import time
from typing import Any, Optional
import torch
from torch import nn
import args
import torch.nn.functional as F
import os
import numpy as np
from p_tqdm import p_imap
import random
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from .tokenizer.tokenizer import MolTranBertTokenizer
from fast_transformers.masking import LengthMask as LM
from .rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
# from apex import optimizers
from molpal.models.transformer.tokenizer.tokenizer import get_vocab_path
from molpal.models.transformer.utils import Lamb, get_loss_func, normalize_smiles
import subprocess
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
# from molpal.models import transformer


class LitTransformer(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super(LitTransformer, self).__init__()
        #####################################
        config = config or {}

        self.uncertainty = config.get("uncertainty", "none")
        self.dataset_type = config.get("dataset_type", "regression")

        # self.warmup_epochs = config.get("warmup_epochs", 2.0)
        self.max_epochs = config.get("max_epochs", 50)
        # self.num_lrs = 1
        # self.init_lr = config.get("init_lr", 1e-4)
        # self.max_lr = config.get("max_lr", 1e-3)
        # self.final_lr = config.get("final_lr", 1e-4)

        self.criterion = get_loss_func(self.dataset_type, self.uncertainty)
        self.metric = {
            "mse": lambda X, Y: F.mse_loss(X, Y, reduction="none"),
            "rmse": lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction="none")),
        }.get(config.get("metric", "rmse"), "rmse")
        self.n_valid_steps = 0 
        #####################################
        self.config = config

        # self.hparams = config
        self.hparams.update(config)
        # self.hparams.update(vars(config))
        self.mode = config['mode']
        self.save_hyperparameters(config)
        self.tokenizer=tokenizer
        self.min_loss = {
            'Predicted docking score' + "min_valid_loss": torch.finfo(torch.float32).max,
            'Predicted docking score' + "min_epoch": 0,
        }

        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config['n_embd']
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config['n_layer'],
            n_heads=config['n_head'],
            query_dimensions=config['n_embd']//config['n_head'],
            value_dimensions=config['n_embd']//config['n_head'],
            feed_forward_dimensions=config['n_embd'],
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config['num_feats']),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config['n_embd'])
        self.drop = nn.Dropout(config['d_dropout'])
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config['n_embd'], n_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []  
        # self.loss = torch.nn.L1Loss()
        self.net = self.Net(
            config['n_embd'], dims=config['dims'], 
            dropout=config['dropout'], uncertainty=self.uncertainty
        )

    class Net(nn.Module):
        dims = [150, 50, 50, 2]
        def __init__(self, smiles_embed_dim, dims=dims, dropout=0.2, uncertainty = 'none'):
            super().__init__()
            self.desc_skip_connection = True 
            self.fcs = []  # nn.ModuleList()
            print('dropout is {}'.format(dropout))
            out_dim = 1
            self.uncertainty = uncertainty
            if self.uncertainty == 'mve':
                out_dim = 2
            

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, out_dim)

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)

            # If using MVE, cap the variance using softplus
            if self.uncertainty == 'mve':
                pred_vars = z[:, 1::2]
                capped_vars = nn.functional.softplus(pred_vars)
                z = torch.clone(z)
                z[:, 1::2] = capped_vars
            return z

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor
    
    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config['lr_start'] * self.train_config['lr_multiplier']
        # optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        optimizer = Lamb(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        idx, mask, Y = batch

        invalid = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y, nan=0.0).float()
        class_weights = torch.ones_like(Y)

        b, t = idx.size()
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        # self.blocks is the pretrained transformer model
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        Y_pred = self.net.forward(loss_input)
        if self.uncertainty == "mve":
            Y_pred_mean = Y_pred[:, 0::2]
            Y_pred_var = Y_pred[:, 1::2]
            L = self.criterion(Y_pred_mean, Y_pred_var, Y) * class_weights * invalid
        else:
            L = self.criterion(Y_pred, Y) * class_weights * invalid
        output = L.sum() / invalid.sum()
        # self.logger.log_metrics({"batch_train_loss": output}, step=self.global_step)
        return output

    def training_epoch_end(self, outputs):
        losses = [d["loss"] for d in outputs]
        # train_loss = losses
        train_loss = torch.stack(losses, dim=0).mean()
        # self.log("train_loss", train_loss, on_epoch=True)
        # self.logger.log_metrics({"epoch_train_loss": train_loss}, step=self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        idx, mask, Y = val_batch

        b, t = idx.size()
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        Y_pred = self.net.forward(loss_input)

        if self.uncertainty == "mve":
            Y_pred = Y_pred[:, 0::2]
        output = self.metric(Y_pred, Y)
        self.logger.log_metrics({"batch_val_loss": output.mean()}, step=self.n_valid_steps)
        self.n_valid_steps += 1
        return output
        
    def validation_epoch_end(self, outputs):
        # val_loss = torch.cat(self.all_gather(outputs)).mean()
        outputs = self.all_gather(outputs)
        if len(outputs[0].shape) < 3:
            val_loss = torch.cat(outputs, axis=0).mean()
        else:
            val_loss = torch.cat(outputs, axis=1).mean()
        """
        There is a confirmed bug here, where the current_epoch value stays at 0 for first two epochs
        Check the ref link:
        https://github.com/Lightning-AI/lightning/issues/3974
        """
        # self.logger.log_metrics({"epoch_val_loss": val_loss}, step=self.current_epoch)
        self.log("val_loss", val_loss, logger=None, sync_dist=True)
        if self.trainer.is_global_zero:
            print(f"Epoch {self.current_epoch} Validation loss: {val_loss}")
    
    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        idx, mask, smis, indices = batch

        b, t = idx.size()
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        Y_pred = self.net.forward(loss_input)
        return indices, Y_pred
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())

        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        output = (batches // effective_accum) * self.trainer.max_epochs
        return output
    

def get_dataset(xs, ys, aug=True):
    dataset = PropertyPredictionDataset(xs, ys, aug)
    return dataset

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys, aug=True):
        # df = df.dropna()  # TODO - Check why some rows are na
        # self.df = df
        # all_smiles = df["smiles"].tolist()
        all_smiles = xs
        self.aug=aug
        self.original_smiles = []
        if self.aug:
            self.original_canonical_map = {
                smi: normalize_smiles(smi, canonical=True, isomeric=False) for smi in all_smiles
            }

        all_measures = ys
        self.measure_map = {all_smiles[i]: all_measures[i] for i in range(len(all_smiles))}
        # Get the canonical smiles
        # Convert the keys to canonical smiles if not already

        for i in range(len(all_smiles)):
            smi = all_smiles[i]
            if smi in self.original_canonical_map.keys():
                self.original_smiles.append(smi)

        print(f"Embeddings not found for {len(all_smiles) - len(self.original_smiles)} molecules")


    def __getitem__(self, index):
        original_smiles = self.original_smiles[index]
        if self.aug:
            canonical_smiles = self.original_canonical_map[original_smiles]
        else:
            canonical_smiles = original_smiles
        return canonical_smiles, self.measure_map[original_smiles]

    def __len__(self):
        return len(self.original_smiles)

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, xs, indices, ncpu=1, aug=True):
        all_smiles = xs
        self.indices = indices
        self.aug = aug
        # processed = p_imap(
        #     partial(normalize_smiles, canonical=True, isomeric=False), 
        #     all_smiles, num_cpus=ncpu, mininterval=30
        # )
        self.original_smiles = []
        # self.original_canonical_map = {
        #     all_smiles[i]:p for i, p in enumerate(processed)
        # }
        if self.aug:
            self.original_canonical_map = {
                smi: normalize_smiles(smi, canonical=True, isomeric=False) for smi in all_smiles
            }

        for i in range(len(all_smiles)):
            smi = all_smiles[i]
            if smi in self.original_canonical_map.keys():
                self.original_smiles.append(smi)

        # print(f"Embeddings not found for {len(all_smiles) - len(self.original_smiles)} molecules")

    def __getitem__(self, index):
        original_smiles = self.original_smiles[index]
        if self.aug:
            canonical_smiles = self.original_canonical_map[original_smiles]
        else:
            canonical_smiles = original_smiles
        return canonical_smiles, self.indices[index]

    def __len__(self):
        return len(self.original_smiles)

def collate_smiles(batch, workdir=None):
    batch, indices = map(list, zip(*batch))
    tokenizer = MolTranBertTokenizer(get_vocab_path(workdir))
    # tokens = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
    tokens = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
    return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), batch, indices)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams, workdir=None):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        #self.smiles_emb_size = hparams.n_embd
        self.workdir = workdir
        self.tokenizer = MolTranBertTokenizer(get_vocab_path(workdir))
        # self.dataset_name = hparams.dataset_name

    def prepare_data(self, train_x, train_y, val_x, val_y):
        print("Inside prepare_dataset")

        train_ds = get_dataset(
            xs = train_x, ys = train_y,
            aug=self.hparams.aug,
        )

        val_ds = get_dataset(
            xs = val_x, ys = val_y,
            aug=self.hparams.aug,
        )

        self.train_ds = train_ds
        self.val_ds = val_ds


    def collate(self, batch):
        # tokens = self.tokenizer.batch_encode_plus([smile[0] for smile in batch], padding=True, add_special_tokens=True)
        tokens = self.tokenizer.batch_encode_plus([smile[0] for smile in batch], padding=True, add_special_tokens=True)
        return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor([smile[1] for smile in batch]).view(-1, 1))

    def val_dataloader(self, batch_size, ncpus, shuffle=False):
        return DataLoader(
            self.val_ds,
            batch_size=batch_size,
            num_workers=ncpus,
            shuffle=shuffle,
            collate_fn=self.collate,
        )

    def train_dataloader(self, batch_size, ncpus, shuffle=True):
        return DataLoader(
            self.train_ds,
            batch_size=batch_size,
            num_workers=ncpus,
            shuffle=shuffle,
            collate_fn=self.collate,
        )
    
class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
                #filename = f"{self.prefix}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")