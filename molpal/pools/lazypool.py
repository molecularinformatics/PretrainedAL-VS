from typing import Iterator, Sequence
import warnings

import numpy as np
import ray

from molpal.utils import batches
from molpal.featurizer import feature_matrix
from molpal.pools.base import MoleculePool


class LazyMoleculePool(MoleculePool):
    """A LazyMoleculePool does not precompute fingerprints for the pool

    Attributes (only differences with EagerMoleculePool are shown)
    ----------
    featurizer : Featurizer
        an Featurizer to generate uncompressed representations on the fly
    fps : None
        no fingerprint file is stored for a LazyMoleculePool
    chunk_size : int
        the buffer size of calculated fingerprints during pool iteration
    cluster_ids : None
        no clustering can be performed for a LazyMoleculePool
    cluster_sizes : None
        no clustering can be performed for a LazyMoleculePool
    """

    def get_fp(self, idx: int) -> np.ndarray:
        smi = self.get_smi(idx)
        return self.featurizer(smi)

    def get_fps(self, idxs: Sequence[int]) -> np.ndarray:
        smis = self.get_smis(idxs)
        return np.array([self.featurizer(smi) for smi in smis])

    def fps(self) -> Iterator[np.ndarray]:
        for fps_batch in self.fps_batches():
            for fp in fps_batch:
                yield fp

    def fps_batches(self) -> Iterator[np.ndarray]:
        for smis in batches(self.smis(), self.chunk_size):
            yield np.array(feature_matrix(smis, self.featurizer, True))

    def prune(
        self, threshold: float, Y_mean: np.ndarray, Y_var: np.ndarray, min_hit_prob: float = 0.025
    ) -> np.ndarray:
        idxs = self.prune_prob(threshold, Y_mean, Y_var, min_hit_prob)

        self.smis_ = self.get_smis(idxs)
        self.size = len(self.smis_)

        return idxs

    def _encode_mols(self, *args, **kwargs):
        """Do not precompute any feature representations"""
        try:
            self.chunk_size = int(ray.cluster_resources()["CPU"] * 4096)
        except:
            self.chunk_size = int(12 * 4096)
        self.fps_ = None

    def _cluster_mols(self, *args, **kwargs) -> None:
        """A LazyMoleculePool can't cluster molecules

        Doing so would require precalculating all uncompressed representations, which is what a
        LazyMoleculePool is designed to avoid. If clustering is desired, use the base
        (Eager)MoleculePool
        """
        warnings.warn(
            "Clustering is not possible for a LazyMoleculePool. No clustering will be performed!"
        )
