import atexit
import dataclasses
import csv
from typing import Dict, Iterable, Optional

import numpy as np
import pyscreener as ps

from molpal.objectives.base import Objective


class DockingObjective(Objective):
    """A DockingObjective calculates the objective function by calculating the
    docking score of a molecule

    Attributes
    ----------
    c : int
        the min/maximization constant, depending on the objective
    virtual_screen : pyscreener.docking.DockingVirtualScreen
        the VirtualScreen object that calculated docking scores of molecules against a given
        receptor with specfied docking parameters

    Parameters
    ----------
    objective_config : str
        the path to a pyscreener config file containing the options for docking calculations
    path : str, default="."
        the path under which docking inputs/outputs should be collected
    verbose : int, default=0
        the verbosity of pyscreener
    minimize : bool, default=True
        whether this objective should be minimized
    **kwargs
        additional and unused keyword arguments
    """

    def __init__(
        self,
        objective_config: str,
        path: str = ".",
        verbose: int = 0,
        minimize: bool = True,
        **kwargs,
    ):
        args = ps.args.gen_args(f"--config {objective_config}")

        metadata_template = ps.build_metadata(args.screen_type, args.metadata_template)
        self.virtual_screen = ps.virtual_screen(
            args.screen_type,
            args.receptors,
            args.center,
            args.size,
            metadata_template,
            args.pdbids,
            args.docked_ligand_file,
            args.buffer,
            args.ncpu,
            args.base_name,
            path,
            args.reduction,
            args.receptor_reduction,
            args.k,
        )

        atexit.register(self.cleanup)

        super().__init__(minimize=minimize)

    def forward(self, smis: Iterable[str], **kwargs) -> Dict[str, Optional[float]]:
        """Calculate the docking scores for a list of SMILES strings

        Parameters
        ----------
        smis : List[str]
            the SMILES strings of the molecules to dock
        **kwargs
            additional and unused positional and keyword arguments

        Returns
        -------
        scores : Dict[str, Optional[float]]
            a map from SMILES string to docking score. Ligands that failed
            to dock will be scored as None
        """
        Y = self.c * self.virtual_screen(smis)
        Y = np.where(np.isnan(Y), None, Y)

        return dict(zip(smis, Y))

    def cleanup(self):
        results = self.virtual_screen.results()
        self.virtual_screen.collect_files()

        with open(self.virtual_screen.path / "extended.csv", "w") as fid:
            writer = csv.writer(fid)
            writer.writerow(field.name for field in dataclasses.fields(results[0]))
            writer.writerows(dataclasses.astuple(r) for r in results)
