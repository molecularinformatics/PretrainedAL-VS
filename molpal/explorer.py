"""This module contains the Explorer class, which is an abstraction for batch Bayesian
optimization."""
from collections.abc import Iterable
import csv
import heapq
import json
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from molpal import acquirer, featurizer, models, objectives, pools
from molpal.exceptions import IncompatibilityError, InvalidExplorationError

T = TypeVar("T")


class Explorer:
    """An Explorer explores a pool of inputs using Bayesian optimization

    Attributes
    ----------
    pool : MoleculePool
        the pool of inputs to explore
    featurizer : Featurizer
        the featurizer this explorer will use convert molecules from SMILES strings into feature
        representations
    acquirer : Acquirer
        an acquirer which selects molecules to explore next using a prior distribution over the
        inputs
    objective : Objective
        an objective calculates the objective function of a set of inputs
    model : Model
        a model that generates a posterior distribution over the inputs using observed data
    retrain_from_scratch : bool
        whether the model will be retrained from scratch at each iteration. If False, train the
        model online. NOTE: The definition of 'online' is model-specific.
    iter : int
        the current iteration of exploration. I.e., the loop iteration the explorer has yet to
        start. This means that the current predictions will be the ones used in the previous
        iteration (because they have yet to be updated for the current iteration)
    scores : Dict[T, float]
        a dictionary mapping an input's identifier to its corresponding objective function value
    failed : Dict[T, None]
        a dictionary containing the inputs for which the objective function failed to evaluate
    adjustment : int
        the number of results that have been read from a file as opposed to being actually
        calculated
    new_scores : Dict[T, float]
        a dictionary mapping an input's identifier to its corresponding objective function value
        for the most recent batch of labeled inputs
    updated_model : bool
        whether the predictions are currently out-of-date with the model
    top_k_avg : float
        the average of the top-k explored inputs
    Y_pred : np.ndarray
        a list parallel to the pool containing the mean predicted score for an input
    Y_var : np.ndarray
        a list parallel to the pool containing the variance in the predicted score for an input.
        Will be empty if model does not provide variance
    recent_avgs : List[float]
        a list containing the recent top-k averages
    delta : float
        the minimum acceptable fractional difference between the current average and the moving
        average in order to continue exploration
    max_iters : int
        the maximum number of batches to explore
    window_size : int
        the number of recent averages from which to calculate a moving average
    write_final : bool
        whether the list of explored inputs and their scores should be written to a file at the end
        of exploration
    write_intermediate : bool
        whether the list of explored inputs and their scores should be written to a file after each
        round of exploration
    save_preds : bool
        whether the predictions should be written after each exploration batch
    verbose : int
        the level of output the Explorer prints
    config : str
        the filepath of a configuration file containing the options necessary to recreate this
        Explorer. NOTE: this does not necessarily ensure reproducibility if a random seed was not
        set

    Parameters
    ----------
    name : str
    k : Union[int, float], default=0.01
    window_size : int, default=3
        the number of top-k averages from which to calculate a moving average
    delta : float, default=0.01
    max_iters : int, default=10
    budget : Union[int, float], default=1.
    root : str, default='.'
    write_final : bool, default=True
    write_intermediate : bool, default=False
    save_preds : bool, default=False
    retrain_from_scratch : bool, default=False
    previous_scores : Optional[str], default=None
        the filepath of a CSV file containing previous scoring data which will be treated as the
        initialization batch (instead of randomly selecting from the bool.)
    verbose : int, default=0
    **kwargs
        keyword arguments to initialize an Encoder, MoleculePool, Acquirer, Model, and Objective
        classes

    Raises
    ------
    ValueError
        if k is less than 0
        if budget is less than 0
    """

    def __init__(
        self,
        path: Union[str, Path] = "molpal",
        work_dir: Union[str, Path] = None,
        k: Union[int, float] = 0.01,
        window_size: int = 3,
        delta: float = 0.01,
        max_iters: int = 10,
        budget: Union[int, float] = 1.0,
        prune: bool = False,
        prune_min_hit_prob: float = 0.025,
        use_observed_threshold: bool = False,
        write_intermediate: bool = False,
        chkpt_freq: int = 0,
        checkpoint_file: Optional[str] = None,
        retrain_from_scratch: bool = False,
        previous_scores: Optional[str] = None,
        **kwargs,
    ):
        args = locals()

        self.path = path
        kwargs["path"] = self.path
        self.log_dir = Path(self.path) / 'log'
        kwargs['log_dir'] = self.log_dir
        print(work_dir)
        kwargs['work_dir'] = work_dir
        self.verbose = kwargs.get("verbose", 0)

        self.featurizer = featurizer.Featurizer(
            kwargs["fingerprint"], kwargs["radius"], kwargs["length"]
        )
        self.pool = pools.pool(featurizer=self.featurizer, **kwargs)
        self.acquirer = acquirer.Acquirer(size=len(self.pool), **kwargs)

        if self.acquirer.metric == "thompson":
            kwargs["dropout_size"] = 1
        self.model = models.model(input_size=len(self.featurizer), **kwargs)
        self.acquirer.stochastic_preds = "stochastic" in self.model.provides
        self.retrain_from_scratch = retrain_from_scratch

        self.objective = objectives.objective(**kwargs)

        # stopping attributes
        self.k = k
        self.window_size = window_size
        self.delta = delta
        self.max_iters = max_iters
        self.budget = budget
        self.exhausted_pool = False

        # get top k prediction rate when retro study
        self.retro = False
        print(kwargs['objective'])
        if kwargs['objective'] == 'lookup':
            obj_path = self.objective.path
            self.retro = True
            self.real_topK = self.get_real_topK(self.k, obj_path)

        # pruning attributes
        self.prune = prune
        self.prune_min_hit_prob = prune_min_hit_prob
        self.use_observed_threshold = use_observed_threshold
        self.full_pool_size = len(self.pool)

        # logging attributes
        self.write_intermediate = write_intermediate
        self.chkpt_freq = chkpt_freq if chkpt_freq >= 0 else float("inf")
        self.previous_chkpt_iter = -float("inf")

        # stateful attributes (not including model)
        self.iter = 0
        self.scores = {}
        self.new_scores = {}
        self.adjustment = 0
        self.updated_model = None
        self.recent_avgs = []
        self.Y_mean = np.array([])
        self.Y_var = np.array([])

        kwargs['n_iter'] = self.iter
        self._validate_model()

        if previous_scores:
            self.load_scores(previous_scores)

        args.pop("self")
        args.update(**args.pop("kwargs"))
        args["fps"] = self.pool.fps_
        args["invalid_idxs"] = list(self.pool.invalid_idxs)
        args["k"] = self.k
        args["budget"] = self.budget
        args.pop("config", None)
        self.write_config(args)

        if checkpoint_file:
            self.load(checkpoint_file)

    def __len__(self) -> int:
        """The total number of objective evaluations"""
        return len(self.scores) - self.adjustment

    @property
    def path(self) -> Path:
        """The directory containing all automated output of this Explorer"""
        return self.__path

    @path.setter
    def path(self, path: Union[str, Path]):
        self.__path = Path(path)
        self.__path.mkdir(parents=True, exist_ok=True)

    @property
    def k(self) -> int:
        """the number of top-scoring inputs from which to calculate an average"""
        return self.__k

    @k.setter
    def k(self, k: Union[int, float]):
        """Set k either as an integer or as a fraction of the pool.

        NOTE: Specifying either a fraction greater than 1 or or a number larger than the pool size
        will default to using the full pool.
        """
        if isinstance(k, float):
            k = int(k * len(self.pool))
        if k <= 0:
            raise ValueError(f"k(={k}) must be greater than 0!")

        self.__k = min(k, len(self.pool))

    @property
    def budget(self) -> int:
        """the maximum budget expressed in terms of the number of allowed objective function
        evaluations"""
        return self.__budget

    @budget.setter
    def budget(self, budget: Union[int, float]):
        """Set budget either as an integer or as a fraction of the pool.

        NOTE: Specifying either a fraction greater than 1 or or a number larger than the pool size
        will default to using the full pool.
        """
        if isinstance(budget, float):
            budget = int(budget * len(self.pool))
        if budget <= 0.0:
            raise ValueError(f"budget(={budget}) must be greater than 0!")

        self.__budget = budget

    @property
    def top_k_avg(self) -> Optional[float]:
        """The most recent top-k average of the explored inputs. None if k inputs have not yet been
        explored"""
        try:
            return self.recent_avgs[-1]
        except IndexError:
            return None

    @property
    def status(self) -> str:
        """The current status of the exploration in string format"""
        if self.top_k_avg:
            ave = f"{self.top_k_avg:0.3f}"
        else:
            if len(self.scores) > 0:
                ave = f"{self.avg():0.3f} (only {len(self.scores)} scores)"
            else:
                ave = "N/A (no scores)"

        return (
            f"ITER: {self.iter}/{self.max_iters} | "
            f"TOP-{self.k} AVE: {ave} | "
            f"BUDGET: {len(self)}/{self.budget}"
        )

    @property
    def completed(self) -> bool:
        """whether the explorer fulfilled one of its stopping conditions

        Stopping Conditions
        -------------------
        a. explored the entire pool
            (not implemented right now due to complications with warm starting)
        b. explored for at least <max_iters> iters
        c. exceeded the maximum budget
        d. the current top-k average is within a fraction <delta> of the moving top-k average. This
            requires two sub-conditions to be met:
            1. the explorer has successfully explored at least k inputs
            2. the explorer has completed at least <window_size> iters after
                sub-condition (1) has been met

        Returns
        -------
        bool
            whether a stopping condition has been met
        """
        if self.iter > self.max_iters or len(self) >= self.budget or self.exhausted_pool:
            return True
        if len(self.recent_avgs) < self.window_size:
            return False

        sma = sum(self.recent_avgs[-self.window_size :]) / self.window_size
        return (self.top_k_avg - sma) / sma <= self.delta

    @property
    def should_chkpt(self) -> bool:
        """whether it is time to checkpoint"""
        return (self.iter - self.previous_chkpt_iter) > self.chkpt_freq

    def explore(self):
        self.run()

    def run(self):
        """Explore the MoleculePool until the stopping condition is met"""
        if self.iter == 0:
            print("Starting exploration...")
            print(f"{self.status}.", flush=True)
            self.explore_initial()
        else:
            print("Resuming exploration...")
            print(f"{self.status}.", flush=True)
            self.explore_batch()

        while not self.completed:
            if self.retro:
                hitrate = self.topK_rate(self.k, self.top_explored(self.k))
                print('Top-%d molecules retrieval rate: %.3f%%'%(self.k, hitrate*100), flush=True)
                
            print(f"{self.status}. Continuing...", flush=True)
            print('--------------------------------------------------------', flush=True)
            self.explore_batch()

        print("Finished exploring!")
        print(f"Final status: {self.status}.", flush=True)
        print("Final averages")
        print("--------------")
        for k in [0.0001, 0.0005, 0.001, 0.005]:
            print(f"TOP-{k:0.2%}: {self.avg(k):0.3f}")

        if self.retro:
            hitrate = self.topK_rate(self.k, self.top_explored(self.k))
            print('Top-%d molecules retrieval rate: %.3f%%'%(self.k, hitrate*100), flush=True)

        self.write_scores(final=True)

    def explore_initial(self) -> float:
        """Perform an initial round of exploration

        Must be called before explore_batch()

        Returns
        -------
        Optional[float]
            the average score of the batch. None if no objective values were calculated, either due
            to each input failing or no inputs being acquired
        """
        inputs = self.acquirer.acquire_initial(
            self.pool.smis(), self.pool.cluster_ids(), self.pool.cluster_sizes
        )
        self.exhausted_pool = len(inputs) == 0

        self.new_scores = self.objective(inputs)
        self.scores.update(self.new_scores)

        if len(self.scores) >= self.k:
            self.recent_avgs.append(self.avg())

        if self.write_intermediate:
            self.write_scores()

        self.iter += 1

        if self.should_chkpt:
            self.checkpoint()
            self.previous_chkpt_iter = self.iter

        valid_scores = [y for y in self.new_scores.values() if y is not None]

        try:
            return sum(valid_scores) / len(valid_scores)
        except ZeroDivisionError:
            return None

    def explore_batch(self) -> Optional[float]:
        """Perform a round of exploration

        Returns
        -------
        Optional[float]
            the average score of the batch. None if no objective values were calculated, either due
            to each input failing or no inputs being acquired

        Raises
        ------
        InvalidExplorationError
            if called before explore_initial or load_scores
        """
        if self.iter == 0:
            raise InvalidExplorationError("Cannot explore a batch before initialization!")

        if self.exhausted_pool:
            print("MoleculePool has been exhausted! No additional exploration will be performed.")
            return None

        self.fit_model()
        self.update_predictions()

        if self.prune:
            if self.use_observed_threshold:
                threshold = self.top_explored(self.k)[-1][1]
            else:
                threshold = np.partition(self.Y_mean, -self.k)[-self.k]

            idxs = self.pool.prune(threshold, self.Y_mean, self.Y_var, self.prune_min_hit_prob)
            expected_tp = pools.MoleculePool.expected_positives_pruned(
                threshold, self.Y_mean, self.Y_var, idxs
            )
            chkpt_dir = Path(self.path / "chkpts" / f"iter_{self.iter}")
            np.save(chkpt_dir / "retained_idxs.npy", idxs)

            if self.verbose >= 1:
                print(f"Pruned pool to {len(self.pool)} molecules!")
                print(f"Expected number of true positives pruned: {expected_tp:0.2f}")

            self.Y_mean = self.Y_mean[idxs]
            self.Y_var = self.Y_var[idxs]

        inputs = self.acquirer.acquire_batch(
            self.pool.smis(),
            self.Y_mean,
            self.Y_var,
            self.scores,
            self.k,
            self.pool.cluster_ids(),
            self.pool.cluster_sizes,
            self.iter - 1,
        )
        self.exhausted_pool = len(inputs) < self.acquirer.batch_size(self.iter - 1)

        self.new_scores = self.objective(inputs)
        self.scores.update(self.new_scores)

        if len(self.scores) >= self.k:
            self.recent_avgs.append(self.avg())

        if self.write_intermediate:
            self.write_scores()

        self.iter += 1

        if self.should_chkpt:
            self.checkpoint()
            self.previous_chkpt_iter = self.iter

        valid_scores = [y for y in self.new_scores.values() if y is not None]

        try:
            return sum(valid_scores) / len(valid_scores)
        except ZeroDivisionError:
            return None

    def avg(self, k: Union[int, float, None] = None) -> float:
        """Calculate the average of the top k molecules

        Parameter
        ---------
        k : Union[int, float, None], default=None
            the number of molecules to consider when calculating the average, expressed either as an
            integer or as a fraction of the pool. If the value specified is greater than the
            number of successfully evaluated inputs, return the average of all succesfully
            evaluated inputs. If None, use self.k

        Returns
        -------
        float
            the top-k average. NOTE: if there are not at least `k` valid scores, `None` scores will
            be included in the top-k as a score of 0 and deflate the top-`k` average.
        """
        k = k or self.k
        if isinstance(k, float):
            k = int(k * self.full_pool_size)
        k = min(k, len(self.scores))

        if k == len(self.scores):
            return sum(score or 0 for _, score in self.scores.items()) / k

        return sum(score or 0 for _, score in self.top_explored(k)) / k

    def top_explored(self, n: Union[int, float, None] = None) -> List[Tuple]:
        """Get the top-n explored molecules

        Parameter
        ---------
        n : Union[int, float, None], default=None
            the number of top-scoring molecules to get, expressed either as a
            specific number or as a fraction of the pool. If the value
            specified is greater than the number of successfully evaluated
            inputs, return all explored inputs. If None, use self.k

        Returns
        -------
        top_explored : List[Tuple[T, float]]
            a list of tuples containing the identifier and score of the
            top-k inputs, sorted by their score
        """
        n = n or self.k
        if isinstance(n, float):
            n = int(n * self.full_pool_size)
        n = min(n, len(self.scores))

        if n / len(self.scores) < 0.3:
            top_explored = heapq.nlargest(
                n, self.scores.items(), key=lambda xy: xy[1] if xy[1] is not None else -float("inf")
            )
        else:
            top_explored = sorted(
                self.scores.items(),
                key=lambda xy: xy[1] if xy[1] is not None else -float("inf"),
                reverse=True,
            )[:n]

        return top_explored

    def top_preds(self, n: Union[int, float, None] = None) -> List[Tuple]:
        """Get the top-n predicted molecules ranked by their predicted means

        Parameter
        ---------
        n : Union[int, float, None], default=None
            the number of molecules to consider when calculating the average, expressed either as an
            integer or as a fraction of the pool. If the value specified is greater than the
            number of successfully evaluated inputs, return the average of all succesfully
            evaluated inputs. If None, use self.k

        Returns
        -------
        List[Tuple[T, float]]
            a list of tuples containing the identifier and predicted score of
            the top-k predicted inputs, sorted by their predicted score
        """
        n = n or self.k
        if isinstance(n, float):
            n = int(n * len(self.pool))
        n = min(n, len(self.scores))

        idxs = np.argpartition(self.Y_mean, -n)[-n:]
        smis = self.pool.get_smis(idxs)
        y_means = self.Y_mean[idxs]
        selected = zip(smis, y_means)

        return sorted(selected, key=lambda xy: xy[1], reverse=True)

    def write_scores(self, final: bool = False):
        """Write all scores to a CSV file

        Writes a CSV file of the explored inputs with the input ID and the respective objective
        function value in the order in which they were acquired. If final is true, the CSV will be
        sorted order, with the best objective function values at the top

        Parameters
        ----------
        final : bool, default=False
            Whether the explorer has finished. If true, write all explored
            inputs (both successful and failed) and name the output CSV file
            "all_explored_final.csv"
        """
        n = len(self)

        p_data = self.path / "data"
        p_data.mkdir(parents=True, exist_ok=True)

        if final:
            p_scores = self.path / "all_explored_final.csv"
            points = self.top_explored(n)
        else:
            p_scores = p_data / f"top_{n}_explored_iter_{self.iter}.csv"
            points = self.scores.items()

        with open(p_scores, "w") as fid:
            writer = csv.writer(fid)
            writer.writerow(["smiles", "score"])
            writer.writerows(points)

        if self.verbose > 0:
            print(f'Results were written to "{p_scores}"')

    def load_scores(self, previous_scores: str) -> None:
        """Load the scores CSV located at saved_scores.

        If this is being called during initialization, treat the data as the
        initialization batch.

        Parameter
        ---------
        previous_scores : str
            the filepath of a CSV file containing previous scoring information. The 0th column of
            this CSV must contain the input identifier and the 1st column must contain a float
            corresponding to its score. A failure to parse the 1st column as a float will treat
            that input as a failure.
        """
        if self.verbose > 0:
            print(f'Loading scores from "{previous_scores}" ... ', end="")

        scores = self._read_scores(previous_scores)
        self.adjustment += len(scores)

        self.scores.update(scores)

        if self.iter == 0:
            self.iter = 1

        if self.verbose > 0:
            print("Done!")

    def checkpoint(self, path: Optional[str] = None) -> str:
        """write a checkpoint file for the explorer's current state and return the corresponding
        filepath

        Parameters
        ----------
        path : Optional[str], default=None
            the directory to under which all checkpoint files should be written

        Returns
        -------
        str
            the path of the JSON file containing all state information
        """
        path = path or self.path / "chkpts" / f"iter_{self.iter}"
        chkpt_dir = Path(path)
        chkpt_dir.mkdir(parents=True, exist_ok=True)

        scores_pkl = chkpt_dir / "scores.pkl"
        scores_pkl.write_bytes(pickle.dumps(self.scores))

        new_scores_pkl = chkpt_dir / "new_scores.pkl"
        new_scores_pkl.write_bytes(pickle.dumps(self.new_scores))

        preds_npz = chkpt_dir / "preds.npz"
        np.savez(preds_npz, Y_pred=self.Y_mean, Y_var=self.Y_var)

        state = {
            "iter": self.iter,
            "scores": str(scores_pkl.absolute()),
            "new_scores": str(new_scores_pkl.absolute()),
            "adjustment": self.adjustment,
            "updated_model": self.updated_model,
            "recent_avgs": self.recent_avgs,
            "preds": str(preds_npz.absolute()),
            "model": self.model.save(chkpt_dir / "model"),
        }

        p_chkpt = chkpt_dir / "state.json"
        json.dump(state, open(p_chkpt, "w"), indent=4)

        if self.verbose > 1:
            print(f'Checkpoint file saved to "{p_chkpt}".')

        return str(p_chkpt)

    def load(self, chkpt_file: str):
        """Load in the state of a previous Explorer's checkpoint"""

        if self.verbose > 0:
            print("Loading in previous state ... ", end="")

        state = json.load(open(chkpt_file))

        self.iter = state["iter"]

        self.scores = pickle.load(open(state["scores"], "rb"))
        self.new_scores = pickle.load(open(state["new_scores"], "rb"))
        self.adjustment = state["adjustment"]

        self.updated_model = state["updated_model"]
        self.recent_avgs.extend(state["recent_avgs"])

        preds_npz = np.load(state["preds"])
        self.Y_mean = preds_npz["Y_pred"]
        self.Y_var = preds_npz["Y_var"]

        self.model.load(state["model"])

        if self.verbose > 0:
            print("Done!")

    def write_config(self, args) -> str:
        args["top-k"] = args.pop("k")
        args["no_title_line"] = not args.pop("title_line")

        for k, v in list(args.items()):
            if v is None:
                args.pop(k)

        config_file = self.path / "config.ini"
        with open(config_file, "w") as fid:
            for k, v in args.items():
                if v is None or v is False:
                    continue
                if isinstance(v, Iterable) and not isinstance(v, str):
                    v = map(str, v)
                    v = "[" + ", ".join(v) + "]"
                fid.write(f'{k.replace("_", "-")} = {v}\n')

        return str(config_file)

    def fit_model(self):
        """fit the surrogate model on the observed data and empty the `new_scores` dictionary"""
        if len(self.new_scores) == 0:
            self.updated_model = False
            return

        if self.retrain_from_scratch:
            xs_ys = self.scores.items()
        else:
            xs_ys = self.new_scores.items()
        xs, ys = zip(*[(x, y) for x, y in xs_ys if y is not None])

        train_success = self.model.train(
            xs, np.array(ys), 
            featurizer=self.featurizer, n_iter=self.iter,
            retrain=self.retrain_from_scratch,
        )
        if not train_success:
            raise Exception("Training failed at iteration %d"%self.iter)

        self.new_scores = {}
        self.updated_model = True

    def update_predictions(self):
        """Update the predictions over the pool with the new model"""
        if not self.updated_model and self.Y_mean.size > 0:
            if self.verbose > 1:
                print("Model hasn't been updated since previous prediction update! Skipping...")
            return

        self.Y_mean, self.Y_var = self.model.apply(
            self.pool.smis(),
            self.pool.fps(),
            None,
            len(self.pool),
            mean_only="vars" not in self.acquirer.needs,
        )

        self.updated_model = False

    def _validate_model(self):
        """Ensure that the model provides necessary values for the Acquirer and during pruning"""
        if self.acquirer.needs > self.model.provides:
            raise IncompatibilityError(
                f"{self.acquirer.metric} metric needs: {self.acquirer.needs} "
                + f"but {self.model.type_} only provides: {self.model.provides}"
            )
        if self.prune and ("vars" not in self.model.provides):
            pass

    def _read_scores(self, scores_csv: str) -> Dict:
        """read the scores contained in the file located at scores_csv"""
        scores = {}
        with open(scores_csv) as fid:
            reader = csv.reader(fid)
            next(reader)
            for row in reader:
                k = row[0]
                try:
                    v = float(row[1])
                except ValueError:
                    v = None
                scores[k] = v

        return scores

    def get_real_topK(self, k, filename):
        # labeled_fname = filename.split('.')[0].replace('libraries', 'data') + '_scores.csv.gz'
        labeled_fname = filename
        df = pd.read_csv(labeled_fname, compression='gzip', header=0)
        if 'AmpC' in labeled_fname or 'D4' in labeled_fname:
            # Clean out the non-numeric dockscore in the AmpC and D4 datasets
            df = df[pd.to_numeric(df['dockscore'], errors='coerce').notnull()]
            df['dockscore'] = df['dockscore'].astype(float)
            real_topK = set(df.nsmallest(k, 'dockscore')['smiles'])
        else:
            real_topK = set(df.nsmallest(k, 'score')['smiles'])
        return real_topK
    
    def topK_rate(self, k, top_explored):
        top_explored = [x[0] for x in top_explored]
        if len(top_explored) < k:
            print('Top molecules explored is less than specified (%d), possibly due to limited iterations'%(k))
            print('Hit rate based on top-%d instead of top-%d'%(len(top_explored), k))
            k = len(top_explored)
        hit = 0
        for i in top_explored:
            if i in self.real_topK:
                hit +=1
        return hit/k