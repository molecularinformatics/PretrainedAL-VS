"""A featurizer transforms input representations into uncompressed feature representations for use
with clustering and model training/prediction."""
from dataclasses import dataclass
from itertools import chain
import math
from typing import List, Optional

import numpy as np
import ray
from p_tqdm import p_map
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

try:
    from map4 import map4
except ImportError:
    pass

from molpal.utils import batches


@dataclass
class Featurizer:
    fingerprint: str = "pair"
    radius: int = 2
    length: int = 2048

    def __post_init__(self):
        if self.fingerprint == "maccs":
            self.radius = 0
            self.length = 167

    def __len__(self):
        return self.length

    def __call__(self, smi: str) -> Optional[np.ndarray]:
        return featurize(smi, self.fingerprint, self.radius, self.length)


def featurize(smi, fingerprint, radius, length) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fingerprint == "morgan":
        fp = rdmd.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, useChirality=True)
    elif fingerprint == "pair":
        fp = rdmd.GetHashedAtomPairFingerprintAsBitVect(
            mol, minLength=1, maxLength=1 + radius, nBits=length
        )
    elif fingerprint == "rdkit":
        fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=1 + radius, fpSize=length)
    elif fingerprint == "maccs":
        fp = rdmd.GetMACCSKeysFingerprint(mol)
    elif fingerprint == "map4":
        fp = map4.MAP4Calculator(dimensions=length, radius=radius, is_folded=True).calculate(mol)
    else:
        raise NotImplementedError(f'Unrecognized fingerprint: "{fingerprint}"')

    X = np.empty(len(fp))
    ConvertToNumpyArray(fp, X)
    return X


def featurize_batch(smis, fingerprint, radius, length) -> List[np.ndarray]:
    return p_map(featurize, smis, fingerprint, radius, length)


# @ray.remote
# def featurize_batch(smis, fingerprint, radius, length) -> List[np.ndarray]:
#     return [featurize(smi, fingerprint, radius, length) for smi in smis]


def feature_matrix(smis, featurizer, disable: bool = False) -> List[np.ndarray]:
    fingerprint = featurizer.fingerprint
    radius = featurizer.radius
    length = len(featurizer)
    feats = featurize_batch(smis, fingerprint, radius, length)
    return list(feats)
    # chunksize = int(math.sqrt(ray.cluster_resources()["CPU"]) * 1024)
    # refs = [
    #     featurize_batch.remote(smis, fingerprint, radius, length)
    #     for smis in batches(smis, chunksize)
    # ]
    # fps_chunks = [
    #     ray.get(r) for r in tqdm(refs, "Featurizing", leave=False, disable=disable, unit="smi")
    # ]

    # return list(chain(*fps_chunks))
