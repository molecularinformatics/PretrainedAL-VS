"""This module contains Model implementations that utilize the MPNN model as their underlying
model"""
from __future__ import annotations

from functools import partial
import json, os, random
import logging
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import ray
import torch
from tqdm import tqdm

from molpal.utils import batches
from molpal.models import molclr
from molpal.models.base import Model
from molpal.models.molclr.model import PropertyPredictionDataModule
from molpal.models.molclr.utils import StandardScaler
from molpal.models.molclr.model import GINet
# from molpal.models.transformer.predict import predict
# from molpal.models.chemprop.data.utils import split_data
from molpal.models.molclr.utils import split_data

# logging.getLogger("lightning").setLevel(logging.FATAL)
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
warnings.filterwarnings('ignore', category=UserWarning, message="Your `val_dataloader` has `shuffle=True`.*")
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

class MolCLR:
    """A message-passing neural network wrapper class that uses Chemprop as the underlying model

    Attributes
    ----------
    ncpu : int
        the number of cores over which to parallelize input batch preparation
    ddp : bool
        whether to train the model over a distributed setup. Only works with CUDA >= 11.0
    precision : int
        the bit precision with which to train the model
    model : MoleculeModel
        the underlying chemprop model on which to train and make predictions
    uncertainty : Optional[str], default=None
        the uncertainty quantification method the model uses. None if it does not use any
        uncertainty quantification
    loss_func : Callable
        the loss function used in model training
    batch_size : int
        the size of each minibatch during training
    epochs : int
        the number of epochs over which to train
    dataset_type : str
        the type of dataset. Choices: ('regression')
        TODO: add support for classification
    use_gpu : bool
        whether the GPU will be used.
        NOTE: If a GPU is detected, it will be used. If this is undesired, set the
        CUDA_VISIBLE_DEVICES environment variable to be empty
    num_workers : int
        the number of workers to distribute model training over. Equal to the number of GPUs
        detected, or if none are available, the ratio of total CPUs on detected on the ray cluster
        over the number of CPUs to dedicate to each dataloader
    train_config : Dict
        a dictionary containing the configuration of training variables: learning rates, maximum
        epochs, validation metric, etc.
    scaler : StandardScaler
        a scaler to normalize target data before training and validation and to reverse transform
        prediction outputs
    """

    def __init__(
        self,
        batch_size: int = 32,
        test_batch_size: Optional[int] = None,
        uncertainty: Optional[str] = None,
        dataset_type: str = "regression",
        num_layer: int = 5,
        emb_dim: int = 300,
        feat_dim: int = 512,
        dropout: float = 0.0,
        lr_multiplier: int = 1,
        pool: str = 'mean',
        metric: str = "rmse",
        epochs: int = 50,
        init_lr: float = 0.0005,
        init_base_lr: float = 0.0002,
        weight_decay: float = 1e-6,
        ncpu: int = 1,
        ddp: bool = False,
        precision: int = 32,
        show_progress_bar: bool = False,
        model_seed: Optional[int] = None,
        log_dir: Optional[Union[str, Path]] = None,
        seed_path: str = '/home/zcao/act_learn/molecular-active-learning/molpal/models/molclr/pretrained_gin/checkpoints/model.pth',
    ):
        self.seed_path = seed_path
        self.ncpu = ncpu
        self.ddp = ddp
        if precision not in (16, 32):
            raise ValueError(f'arg: "precision" can only be (16, 32)! got: {precision}')
        self.precision = precision
        self.log_dir = log_dir


        self.uncertainty = uncertainty if uncertainty is not None else "none"
        self.dataset_type = dataset_type

        self.epochs = epochs
        self.batch_size = batch_size
        """
        MVE loss will be extremely large for large batch sizes
        The temporary solution is to hard code the batch size to be 50 when using MVE
        Also set the learning rate to be lower (5e-5)
        """
        if test_batch_size is not None:
            self.test_batch_size = test_batch_size
        else:
            self.test_batch_size = self.batch_size

        self.scaler = None

        ngpu = int(ray.cluster_resources().get("GPU", 0))
        if ngpu > 0:
            self.use_gpu = True
        #     self._predict = mpnn.predict_.options(num_cpus=ncpu, num_gpus=1)
        #     self.num_workers = ngpu
        else:
            self.use_gpu = False
        #     self._predict = mpnn.predict_.options(num_cpus=ncpu)
        #     self.num_workers = int(ray.cluster_resources()["CPU"] // self.ncpu)
        self._predict_noray = molclr.predict

        self.show_progress_bar = show_progress_bar
        self.seed = model_seed
        if model_seed is not None:
            torch.manual_seed(model_seed)
        print('Learning rate = ', init_lr)
        self.train_config = {
            # "model": self.model,
            "uncertainty": self.uncertainty,
            "dataset_type": dataset_type,
            "batch_size": self.batch_size,
            "max_epochs": self.epochs,
            "init_lr": init_lr,
            "init_base_lr": init_base_lr,
            "metric": metric,
            "num_layer": num_layer,
            "emb_dim": emb_dim,
            "drop_ratio": dropout,
            "lr_multiplier": lr_multiplier,
            "feat_dim": feat_dim,
            "pool": pool,
            "weight_decay": weight_decay,
        }

        self.model = GINet(
            uncertainty=self.train_config['uncertainty'], num_layer=self.train_config['num_layer'], 
            emb_dim=self.train_config['emb_dim'], feat_dim=self.train_config['feat_dim'], 
            drop_ratio=self.train_config['drop_ratio'], pool=self.train_config['pool'],
        )
        try:
            state_dict = torch.load(seed_path, map_location='cpu')
            # model.load_state_dict(state_dict)
            self.model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")


    def train(self, smis: Iterable[str], targets: np.ndarray, n_iter: int) -> bool:
        """Train the model on the inputs SMILES with the given targets"""
        # train_data, val_data = self.make_datasets(smis, targets)
        train_x, train_y, val_x, val_y = self.make_datasets(smis, targets)

        if self.ddp:
            raise NotImplementedError("DDP training with Transformer has not been implemented")

        datamodule = PropertyPredictionDataModule(self.train_config)
        datamodule.prepare_data(train_x, train_y, val_x, val_y)
        train_dataloader = datamodule.train_dataloader(self.batch_size, self.ncpu, shuffle=True)
        val_dataloader = datamodule.val_dataloader(self.batch_size, self.ncpu, shuffle=False)

        self.model = GINet(
            uncertainty=self.train_config['uncertainty'], num_layer=self.train_config['num_layer'], 
            emb_dim=self.train_config['emb_dim'], feat_dim=self.train_config['feat_dim'], 
            drop_ratio=self.train_config['drop_ratio'], pool=self.train_config['pool'],
        )
        try:
            state_dict = torch.load(self.seed_path, map_location='cpu')
            # model.load_state_dict(state_dict)
            self.model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        lit_model = molclr.LitMolCLR(self.model, self.train_config)

        callbacks = [
            EarlyStopping("val_loss", patience=10, mode="min"),
        ]
        logger = False
        if self.log_dir:
            logger = TensorBoardLogger(save_dir=self.log_dir, name='Iter_%d'%n_iter)

        self.trainer = PlTrainer(
            logger=logger,
            max_epochs=self.epochs,
            callbacks=callbacks,
            gpus=1 if self.use_gpu else 0,
            precision=self.precision,
            log_every_n_steps=10,
            enable_progress_bar = self.show_progress_bar,
            enable_model_summary=False,
            enable_checkpointing=False,  # TODO: reimplement trainer checkpointing later
        )
        self.trainer.fit(lit_model, train_dataloader, val_dataloader)

        return True

    def make_datasets(
        self, xs: Iterable[str], ys: np.ndarray
    ):
        """Split xs and ys into train and validation datasets"""
        # data = zip(xs, ys.reshape(-1, 1))
        # data = MoleculeDataset([MoleculeDatapoint([x], y) for x, y in zip(xs, ys.reshape(-1, 1))])
        # train_data, val_data, _ = split_data(data, "random", (0.8, 0.2, 0.0), seed=self.seed)
        train_x, train_y, val_x, val_y = split_data(xs, ys, val_ratio=0.2, seed=self.seed)
        self.scaler = StandardScaler().fit(train_y)
        train_y = self.scaler.transform(train_y)
        val_y = self.scaler.transform(val_y)
        # self.scaler = train_data.normalize_targets()
        # val_data.scale_targets(self.scaler)
        return train_x, train_y, val_x, val_y
    
    ############################## The prediction without Ray ######################################
    def predict(self, smis: Iterable[str]) -> np.ndarray:
        """Generate predictions for the inputs xs

        Parameters
        ----------
        smis : Iterable[str]
            the SMILES strings for which to generate predictions

        Returns
        -------
        np.ndarray
            an array of shape `n x m`, where `n` is the number of SMILES strings and `m` is the
            number of tasks
        """
        scaler = self.scaler
        Y_pred_batches = []

        Y_pred = self._predict_noray(
                self.trainer,
                smis,
                self.test_batch_size,
                self.ncpu,
                self.uncertainty,
                scaler,
                self.use_gpu,
                True,
        )
        Y_pred_batches.append(Y_pred)

        # for i, smiss in enumerate(batches(smis, self.test_batch_size)):
        #     Y_pred = self._predict_noray(
        #         model,
        #         smiss,
        #         self.test_batch_size,
        #         self.ncpu,
        #         self.uncertainty,
        #         scaler,
        #         self.use_gpu,
        #         True,
        #     )
        #     Y_pred_batches.append(Y_pred)
        #     if (i+1) % 10 == 0:
        #         print(f"Predicted {i*self.test_batch_size} molecules")
        Y_pred = np.concatenate(Y_pred_batches)

        if self.scaler is not None:
            if self.uncertainty == "mve":
                Y_pred[:, 0::2] = Y_pred[:, 0::2] * self.scaler.stds + self.scaler.means
                Y_pred[:, 1::2] *= self.scaler.stds**2
            else:
                Y_pred = Y_pred * self.scaler.stds + self.scaler.means
        return Y_pred

    ##################################################################################################
    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = f"{path}/model.pt"
        torch.save(self.model.state_dict(), model_path)

        state_path = f"{path}/state.json"
        try:
            state = {
                "model_path": model_path,
                "means": self.scaler.means.tolist(),
                "stds": self.scaler.stds.tolist(),
            }
        except AttributeError:
            state = {"model_path": model_path}

        json.dump(state, open(state_path, "w"), indent=4)

        return state_path

    def load(self, path):
        state = json.load(open(path, "r"))

        self.model.load_state_dict(torch.load(state["model_path"]))
        try:
            self.scaler = StandardScaler(state["means"], state["stds"])
        except KeyError:
            pass


class MolCLRModel(Model):
    """Message-passing model that learns feature representations of inputs and
    passes these inputs to a feed-forward neural network to predict means"""

    def __init__(
        self,
        test_batch_size: Optional[int] = 1000000,
        ncpu: int = 1,
        ddp: bool = False,
        precision: int = 32,
        epochs: int = 50,
        model_seed: Optional[int] = None,
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(
            MolCLR, test_batch_size=test_batch_size,
            ncpu=ncpu, ddp=ddp, 
            precision=precision, epochs=epochs, 
            model_seed=model_seed,
            log_dir=log_dir
        )
        self.model = self.build_model()

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {"means"}

    @property
    def type_(self):
        return "molclr"

    def train(self, xs: Iterable[str], ys: np.ndarray, *, n_iter: int = 0, retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, n_iter)

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        preds = self.model.predict(xs)
        return preds[:, 0]  # assume single-task

    def get_means_and_vars(self, xs: List) -> NoReturn:
        raise TypeError("MolCLRModel cannot predict variance!")

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)

class MolCLRTwoOutputModel(Model):
    """Message-passing network model that predicts means and variances through mean-variance
    estimation"""

    def __init__(
        self,
        test_batch_size: Optional[int] = 1000000,
        ncpu: int = 1,
        ddp: bool = False,
        precision: int = 32,
        epochs: int = 50,
        model_seed: Optional[int] = None,
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 1000000
        self.build_model = partial(
            MolCLR, test_batch_size=test_batch_size,
            uncertainty="mve", ncpu=ncpu, ddp=ddp, 
            precision=precision, epochs=epochs,
            model_seed=model_seed,
            log_dir=log_dir
        )
        self.model = self.build_model()

        super().__init__(test_batch_size, **kwargs)

    @property
    def type_(self):
        return "molclr"

    @property
    def provides(self):
        return {"means", "vars"}

    def train(self, xs: Iterable[str], ys: np.ndarray, *, n_iter: int = 0, retrain: bool = False, **kwargs) -> bool:
        # When retrain is True, model is completely reinitialized.
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, n_iter)

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        means, _ = self._get_predictions(xs)
        return means

    def get_means_and_vars(self, xs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        means, variances = self._get_predictions(xs)
        return means, variances

    def _get_predictions(self, xs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Get both the means and the variances for the xs"""
        preds = self.model.predict(xs)
        # assume single task prediction now
        means, variances = preds[:, 0::2], preds[:, 1::2]
        # means, variances = preds[:, 0], preds[:, 1]  #
        return means, variances

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)

