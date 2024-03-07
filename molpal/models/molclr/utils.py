import debugpy
import socket
import glob
import pandas as pd
import numpy as np
from typing import List, Optional, Any
from rdkit import Chem

import collections
import math
from random import Random

import torch
# from tensorboardX import SummaryWriter
from torch.optim import Optimizer
from torch import clamp, nn


def getipaddress():
    return socket.gethostbyname(socket.getfqdn())


def debug():
    print("Waiting for debugger to connect")
    if (
        socket.getfqdn().startswith("dcc")
        or socket.getfqdn().startswith("mol")
        or socket.getfqdn().startswith("ccc")
    ):
        debugpy.listen(address=(getipaddress(), 3000))
        debugpy.wait_for_client()
    debugpy.breakpoint()


class ListDataset:
    def __init__(self, seqs):
        self.seqs = seqs

    def __getitem__(self, index):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)


def transform_single_embedding_to_multiple(smiles_z_map):
    """Transforms an embedding map of the format smi->embedding to
    smi-> {"canonical_embeddings":embedding}. This function exists
    as a compatibility layer

    Args:
        smiles_z_map ([type]): [description]
    """
    retval = dict()
    for key in smiles_z_map:
        retval[key] = {"canonical_embeddings": smiles_z_map[key]}
    return retval


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized


def get_all_proteins(affinity_dir: str):
    files = glob.glob(affinity_dir + "/*.csv")
    all_proteins = []
    print(files)
    for file in files:
        df = pd.read_csv(file)
        all_proteins.extend(df["protein"].tolist())
    return set(all_proteins)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")


def write_to_file(filename, line):
    with open(filename, "w") as f:
        f.write(line + "\n")


def get_loss_func(dataset_type: str, uncertainty_method: Optional[str] = None) -> nn.Module:
    """Get the loss function corresponding to a given dataset type

    Parameters
    ----------
    dataset_type : str
        the type of dataset
    uncertainty_method : Optional[str]
        the uncertainty method being used

    Returns
    -------
    loss_function : nn.Module
        a PyTorch loss function

    Raises
    ------
    ValueError
        if is dataset_type is neither "classification" nor "regression"
    """
    if dataset_type == "classification":
        return nn.BCEWithLogitsLoss(reduction="none")

    elif dataset_type == "regression":
        if uncertainty_method == "mve":
            return negative_log_likelihood

        return nn.MSELoss(reduction="none")

    raise ValueError(f'Unsupported dataset type: "{dataset_type}."')


def negative_log_likelihood(means, variances, targets):
    """The NLL loss function as defined in:
    Nix, D.; Weigend, A. ICNN’94. 1994; pp 55–60 vol.1"""
    variances = clamp(variances, min=1e-5)
    return (variances.log() + (means - targets) ** 2 / variances) / 2

def split_data(xs, ys, val_ratio, seed=None):
    random = Random(seed)

    indices = list(range(len(xs)))
    random.shuffle(indices)

    train_size = int(1-val_ratio * len(xs))
    train_x = [xs[i] for i in indices[:train_size]]
    train_y = [ys[i] for i in indices[:train_size]]
    val_x = [xs[i] for i in indices[train_size:]]
    val_y = [ys[i] for i in indices[train_size:]]

    return train_x, train_y, val_x, val_y

    
class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(
        self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None
    ):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> "StandardScaler":
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )

        return transformed_with_none
    