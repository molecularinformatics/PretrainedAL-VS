import functools, random, os
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset, DataLoader
import pytorch_lightning as pl
from molpal.models.molclr.utils import get_loss_func, normalize_smiles, split_data

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from argparse import ArgumentParser, Namespace

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + \
            self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, 
        uncertainty='none', num_layer=5, emb_dim=300, feat_dim=512, 
        drop_ratio=0, pool='mean', pred_n_layer=2, pred_act='softplus'
    ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.uncertainty = uncertainty

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        print(self.uncertainty)
        if self.uncertainty == 'none':
            out_dim = 1
        elif self.uncertainty == 'mve':
            out_dim = 2
        
        self.pred_n_layer = max(1, pred_n_layer)

        if pred_act == 'relu':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                    nn.ReLU(inplace=True),
                ])
            pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        elif pred_act == 'softplus':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.Softplus()
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                    nn.Softplus()
                ])
        else:
            raise ValueError('Undefined activation function')
        
        pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        self.pred_head = nn.Sequential(*pred_head)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        z = self.pred_head(h)
        # If using MVE, cap the variance using softplus
        if self.uncertainty == 'mve':
            pred_vars = z[:, 1::2]
            capped_vars = nn.functional.softplus(pred_vars)
            z = torch.clone(z)
            z[:, 1::2] = capped_vars
        return z

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class LitMolCLR(pl.LightningModule):

    def __init__(self, gin, config):
        super(LitMolCLR, self).__init__()
        #####################################
        self.gin = gin
        config = config or {}

        self.uncertainty = config.get("uncertainty", "none")
        self.dataset_type = config.get("dataset_type", "regression")

        # self.warmup_epochs = config.get("warmup_epochs", 2.0)
        self.max_epochs = config.get("max_epochs", 50)

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
        self.pool = config['pool']
        self.save_hyperparameters(config)
        self.min_loss = {
            'Predicted docking score' + "min_valid_loss": torch.finfo(torch.float32).max,
            'Predicted docking score' + "min_epoch": 0,
        }
    
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

    def configure_optimizers(self):
        layer_list = []
        for name, param in self.gin.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, self.gin.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, self.gin.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=self.config['weight_decay']
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        Xs = batch
        Y = batch.y
    
        mask = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y, nan=0.0)
        class_weights = torch.ones_like(Y)
        Y_pred = self.gin(Xs)
        
        if self.uncertainty == "mve":
            Y_pred_mean = Y_pred[:, 0::2]
            Y_pred_var = Y_pred[:, 1::2]
            L = self.criterion(Y_pred_mean, Y_pred_var, Y)
        else:
            L = self.criterion(Y_pred, Y) * class_weights * mask
        output = L.sum() / mask.sum()
        self.logger.log_metrics({"batch_train_loss": output}, step=self.global_step)
        # print(output)
        return output

    def training_epoch_end(self, outputs):
        losses = [d["loss"] for d in outputs]
        # train_loss = losses
        train_loss = torch.stack(losses, dim=0).mean()
        # self.log("train_loss", train_loss, on_epoch=True)
        self.logger.log_metrics({"epoch_train_loss": train_loss}, step=self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        Xs = val_batch
        Y = val_batch.y
        Y_pred = self.gin(Xs)
        
        if self.uncertainty == "mve":
            Y_pred = Y_pred[:, 0::2]
        output = self.metric(Y_pred, Y)
        self.logger.log_metrics({"batch_val_loss": output.mean()}, step=self.n_valid_steps)
        self.n_valid_steps += 1
        return output
        
    def validation_epoch_end(self, outputs):
        val_loss = torch.cat(outputs).mean()
        """
        There is a confirmed bug here, where the current_epoch value stays at 0 for first two epochs
        Check the ref link:
        https://github.com/Lightning-AI/lightning/issues/3974
        """
        self.logger.log_metrics({"epoch_val_loss": val_loss}, step=self.current_epoch)
        self.log("val_loss", val_loss, logger=None)
        print(f"Epoch {self.current_epoch} Validation loss: {val_loss}")
    
    def predict_step(self, batch, batch_idx):
        Xs = batch
        Y_pred = self.gin(Xs)
        return Y_pred
    
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


class MolTestDataset(Dataset):
    def __init__(self, xs, ys, task='regression'):
        super().__init__()
        self.smiles_data, self.labels = xs, ys
        self.task = task

        self.conversion = 1

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)
    
    def len(self):
        return self.__len__()
    
    @functools.lru_cache(maxsize=None)
    def get(self, idx):
        return self.__getitem__(idx)
    

class MolInferDataset(Dataset):
    def __init__(self, xs, task='regression'):
        super().__init__()
        self.smiles_data = xs
        self.task = task

        self.conversion = 1

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)
    
    def len(self):
        return self.__len__()
    
    @functools.lru_cache(maxsize=None)
    def get(self, idx):
        return self.__getitem__(idx)


class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        # self.hparams = hparams
        self.hparams.update(vars(hparams))

    def prepare_data(self, train_x, train_y, val_x, val_y):
        print("Inside prepare_dataset")
        train_ds = MolTestDataset(
            xs = train_x, ys = train_y,
        )

        val_ds = MolTestDataset(
            xs = val_x, ys = val_y,
        )

        self.train_ds = train_ds
        self.val_ds = val_ds

    def val_dataloader(self, batch_size, ncpus, shuffle=False):
        return DataLoader(
            self.val_ds,
            batch_size=batch_size,
            num_workers=ncpus,
            shuffle=shuffle,
        )

    def train_dataloader(self, batch_size, ncpus, shuffle=True):
        return DataLoader(
            self.train_ds,
            batch_size=batch_size,
            num_workers=ncpus,
            shuffle=shuffle,
        )