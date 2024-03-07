"""This module contains Model implementations that utilize the MPNN model as their underlying
model"""
from __future__ import annotations

from functools import partial
import json, os
import logging
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple, Union
import warnings
import getpass
from pyslurmutils.client import SlurmScriptRestClient

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import requests, time
from tqdm import tqdm
from timeit import default_timer as timer

from molpal.utils import batches
from molpal.models import transformer
from molpal.models.base import Model
from molpal.models.transformer.model import PropertyPredictionDataModule
from molpal.models.transformer.utils import StandardScaler
from molpal.models.transformer.tokenizer.tokenizer import MolTranBertTokenizer, get_vocab_path
from molpal.models.transformer.utils import split_data


# logging.getLogger("lightning").setLevel(logging.FATAL)
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
warnings.filterwarnings('ignore', category=UserWarning, message="Your `val_dataloader` has `shuffle=True`.*")
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


class Transformer:
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
    num_tasks : int
        the number of training tasks
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
        n_iter: int = 0,
        uncertainty: Optional[str] = 'none',
        ngpu: int = 1,
        ddp: bool = False,
        slurm_token: Optional[str] = None,
        log_dir: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        weight_path: str = 'molpal/models/transformer/pretrained_ckpt/pretrained_weights.ckpt',
        seed: Optional[int] = None 
    ):  
        self.n_iter = n_iter
        self.CPU_PER_GPU = 8
        self.seed_path = os.path.join(work_dir, weight_path)
        self.ngpu = ngpu
        self.ncpu = ngpu*self.CPU_PER_GPU
        self.n_node, self.tasks_per_node = self.get_n_node(ngpu)
        self.ddp = ddp
        self.log_dir = log_dir
        self.seed = seed
        self.work_dir = work_dir
        self.slurm_url = 'your Slurm mgt node'
        self.user_name = os.environ.get("SLURM_USER", getpass.getuser())
        self.version = 'v0.0.38'
        self.errorTolerance = 20
        
        if slurm_token == None:
            self.slurm_token = self.get_slurm_token(days=1)

        self.uncertainty = uncertainty
        self.scaler = StandardScaler()



    def train(self, smis: Iterable[str], targets: np.ndarray, n_iter: int) -> bool:
        """Train the model on the inputs SMILES with the given targets"""
        # train_data, val_data = self.make_datasets(smis, targets)
        start = timer()
        self.n_iter = n_iter
        train_x, train_y, val_x, val_y = self.make_datasets(smis, targets)
        train_df = pd.DataFrame({'smiles': train_x, 'score': train_y})
        val_df = pd.DataFrame({'smiles': val_x, 'score': val_y})

        train_files_directory = os.path.join(self.work_dir, 'run/train_files')
        if not os.path.exists(train_files_directory):
            os.mkdir(train_files_directory)
        train_path = os.path.join(self.work_dir, 'run/train_files/train_data_iter%d.csv.gz'%n_iter)
        val_path = os.path.join(self.work_dir, 'run/train_files/val_data_iter%d.csv.gz'%n_iter)
        train_df.to_csv(train_path, compression='gzip', index=False)
        val_df.to_csv(val_path, compression='gzip', index=False)

        np.savez(os.path.join(self.work_dir, 'run/chkpts/iter_%d/model/scaler_params.npz'%n_iter), mean=self.scaler.means, std=self.scaler.stds)

        log_directory = os.path.join(self.work_dir, 'run/slurm_log/train_infer')
        if not os.path.exists(log_directory):
            os.mkdir(log_directory)

        SCRIPT = """#!/bin/bash -l
                module load Anaconda3
                module load NCCL
                conda activate yourENV
                srun --ntasks-per-node=%d python %s/molpal/models/transformer/train.py -i %d -d %s -n %d -u %s
                """%(self.tasks_per_node, self.work_dir, n_iter, self.work_dir, self.ngpu, self.uncertainty)

        parameter = {
            'name': 'AL_T%d'%self.n_iter,
            'partition':'gpu',
            'nodes': self.n_node,
            'gpus_per_node': self.tasks_per_node,
            'cpus_per_gpu': self.CPU_PER_GPU,
            'tasks_per_node': self.tasks_per_node,
            'memory_per_cpu': '16GB',
            'time_limit': '14-00:00:00',
            # 'qos': 'compchem-gpu-qos',
            'environment': {'_DUMMY_VAR': 'dummy_value'},
            'standard_input': '/dev/null',
            'standard_output': log_directory + '/train_it%d.out'%self.n_iter,
            'standard_error': log_directory + '/train_it%d.err'%self.n_iter,
        }

        job_id = self.submit_job(SCRIPT, parameter)

        status = self.wait_done(job_id)

        # Remove unnecessary files
        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(val_path):
            os.remove(val_path)
        
        if status != "COMPLETED":
            print('Training iter %d failed due to: %s'%(self.n_iter, status))
            return False
        stop = timer()
        (m, s, h, d) = self.calcTime(start, stop)
        print(f"Total time for AL_T{self.n_iter}: {d}d {h}h {m}m {s:0.2f}s", flush=True)
        return True

    def make_datasets(
        self, xs: Iterable[str], ys: np.ndarray
    ):
        """Split xs and ys into train and validation datasets"""
        train_x, train_y, val_x, val_y = split_data(xs, ys, val_ratio=0.2, seed=self.seed)
        self.scaler.fit(train_y)
        train_y = self.scaler.transform(train_y)
        val_y = self.scaler.transform(val_y)
        return train_x, train_y, val_x, val_y
    
    def get_n_node(self, ngpu):
        assert ngpu in {1, 2, 3, 4, 6}, "Currrently, DDP only support 1, 2, 3, 4, or 6 GPUs"
        if ngpu <= 4:
            return 1, ngpu
        return 2, ngpu // 2

    def calcTime(self, start, stop):
        m, s = divmod(stop - start, 60)
        h, m = divmod(int(m), 60)
        d, h = divmod(h, 24)
        return (m, s, h, d)

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
        start = timer()
        smiles = [s for s in smis]
        infer_directory = os.path.join(self.work_dir, 'run/infer_files')
        if not os.path.exists(infer_directory):
            os.mkdir(infer_directory)
            os.mkdir(os.path.join(infer_directory, 'tmp_folder'))
            
        test_path = os.path.join(self.work_dir, infer_directory, 'infer_data_iter%d.npz'%self.n_iter)
        np.savez_compressed(test_path, smiles=np.array(smiles, dtype=object))
        # pd.DataFrame({'smiles': smiles}).to_csv(test_path, compression='gzip', index=False)

        log_directory = os.path.join(self.work_dir, 'run/slurm_log/train_infer')
        if not os.path.exists(log_directory):
            os.mkdir(log_directory)

        SCRIPT = """#!/bin/bash -l
                module load Anaconda3
                module load NCCL
                conda activate yourENV
                srun --ntasks-per-node=%d python %s/molpal/models/transformer/infer.py -i %d -d %s -n %d -u %s
                """%(self.tasks_per_node, self.work_dir, self.n_iter, self.work_dir, self.ngpu, self.uncertainty)

        parameter = {
            'name': 'AL_I%d'%self.n_iter,
            'partition':'gpu',
            'nodes': self.n_node,
            'gpus_per_node': self.tasks_per_node,
            'cpus_per_gpu': self.CPU_PER_GPU,
            'tasks_per_node': self.tasks_per_node,
            'memory_per_cpu': '16GB',
            'time_limit': '14-00:00:00',
            # 'qos': 'compchem-gpu-qos',
            'environment': {'_DUMMY_VAR': 'dummy_value'},
            'standard_input': '/dev/null',
            'standard_output': log_directory + '/infer_it%d.out'%self.n_iter,
            'standard_error': log_directory + '/infer_it%d.err'%self.n_iter,
        }

        job_id = self.submit_job(SCRIPT, parameter)

        status = self.wait_done(job_id)

        if status != "COMPLETED":
            print('Inference iter %d failed due to: %s'%(self.n_iter, status))
            return None

        try:
            Y_pred = np.load(os.path.join(self.work_dir, 'run/infer_files/Iter%d_preds.npz'%self.n_iter), allow_pickle=True)['pred']
        except:
            raise Exception("Inference failed, no prediction file found!")
        
        assert len(Y_pred) == len(smiles), "Length of predicted value does not match length of input SMILES, something is off."
        if self.scaler is not None:
            if self.uncertainty == "mve":
                Y_pred[:, 0::2] = Y_pred[:, 0::2] * self.scaler.stds + self.scaler.means
                Y_pred[:, 1::2] *= self.scaler.stds**2
            else:
                Y_pred = Y_pred * self.scaler.stds + self.scaler.means
        os.remove(test_path)
        stop = timer()
        (m, s, h, d) = self.calcTime(start, stop)
        print(f"Total time for AL_I{self.n_iter}: {d}d {h}h {m}m {s:0.2f}s", flush=True)
        return Y_pred
        

    ##################################################################################################
    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = f"{path}/molformer_iter%d.ckpt"%self.n_iter
        # torch.save(self.model.state_dict(), model_path)

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

        # self.model.load_state_dict(torch.load(state["model_path"]))
        try:
            self.scaler = StandardScaler(state["means"], state["stds"])
        except KeyError:
            pass
    
    ######################## SLURM REST API ###################################
    def get_slurm_token(self, days=1):
        command = 'scontrol token lifespan=$((3600*24*%d))'%(days)
        token = os.popen(command).read().strip('\n').strip('SLURM_JWT=')
        return token
    
    def get_job_status(self, job_id):
        errorCount = 0
        while True:
            response = requests.get(
            f'{self.slurm_url}/slurm/{self.version}/job/{job_id}',
            headers={
                'X-SLURM-USER-NAME': f'{self.user_name}',
                'X-SLURM-USER-TOKEN': f'{self.slurm_token}'
            })

            if response.status_code == 200:
                break
            elif response.status_code == 500:
                errorCount += 1
                print("HTTP 500 Internal Server Error received during get_job_status. Retrying in 2 seconds...", flush=True)
                time.sleep(2)
                if errorCount > self.errorTolerance:
                    print('Error count > %d. Break the loop.'%self.errorTolerance, flush=True)
                    break
                # If a 500 error is caught, renew token before next getting next response.
                self.slurm_token = self.get_slurm_token(days=1)
            else:
                print(f"Received HTTP status code {response.status_code}. Raising an exception.")
                raise Exception("Unexpected HTTP status code in get job status")


        response.raise_for_status()
        job = response.json()
        job_status = job["jobs"][0]['job_state']

        return job_status

    def submit_job(self, scripts, parameters):
        self.slurm_token = self.get_slurm_token(days=1)
        response = requests.post(
            f'{self.slurm_url}/slurm/{self.version}/job/submit',
            headers={
                'X-SLURM-USER-NAME': f'{self.user_name}',
                'X-SLURM-USER-TOKEN': f'{self.slurm_token}'
            },
            json={
                'script': scripts,
                'job': parameters})
        
        errorCount = 0
        while True:
            if response.status_code == 200:
                break
            elif response.status_code == 500:
                print("HTTP 500 Internal Server Error received during submitting job. Retrying in 5 seconds...", flush=True)
                time.sleep(5)
                errorCount += 1
                if errorCount > (self.errorTolerance//2):
                    print('Error count > 5. Job submission failed.', flush=True)
                    return
                print('Resubmitting job for the %d time'%errorCount, flush=True)
                self.slurm_token = self.get_slurm_token(days=1)
                response = requests.post(
                    f'{self.slurm_url}/slurm/{self.version}/job/submit',
                    headers={
                        'X-SLURM-USER-NAME': f'{self.user_name}',
                        'X-SLURM-USER-TOKEN': f'{self.slurm_token}'
                    },
                    json={
                        'script': scripts,
                        'job': parameters}
                )
            else:
                print(f"Received HTTP status code {response.status_code}. Raising an exception.")
                raise Exception("Unexpected HTTP status code in job submission")

        job_id = response.json()["job_id"]
        print("{} submitted, Job ID: {}".format(parameters['name'], job_id))
        return job_id

    def wait_done(self, job_id):
        while True:
            status = self.get_job_status(job_id)
            if status in ["FAILED", "COMPLETED", "CANCELLED","TIMEOUT"]:
                return status
            else:
                time.sleep(1)


class TransformerModel(Model):
    """Message-passing model that learns feature representations of inputs and
    passes these inputs to a feed-forward neural network to predict means"""

    def __init__(
        self,
        ngpu: int = 1,
        ddp: bool = False,
        log_dir: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):

        self.build_model = partial(
            Transformer, 
            ngpu = ngpu,
            ddp=ddp, 
            log_dir=log_dir,
            work_dir=work_dir
        )
        self.model = self.build_model()

        super().__init__(**kwargs)

    @property
    def provides(self):
        return {"means"}

    @property
    def type_(self):
        return "transformer"

    def train(self, xs: Iterable[str], ys: np.ndarray, *, n_iter: int = 0, retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, n_iter)

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        preds = self.model.predict(xs)
        return preds[:, 0]  # assume single-task

    def get_means_and_vars(self, xs: List) -> NoReturn:
        raise TypeError("TransformerModel cannot predict variance!")

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)

class TransformerTwoOutputModel(Model):
    """Message-passing network model that predicts means and variances through mean-variance
    estimation"""

    def __init__(
        self,
        ngpu: int = 1,
        ddp: bool = False,
        log_dir: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.build_model = partial(
            Transformer,
            uncertainty="mve", ngpu=ngpu, ddp=ddp, 
            log_dir=log_dir, work_dir=work_dir
        )
        self.model = self.build_model()

        super().__init__(**kwargs)

    @property
    def type_(self):
        return "transformer"

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

