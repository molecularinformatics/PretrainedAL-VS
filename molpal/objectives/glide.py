import atexit
import dataclasses
import csv, os, sys
from typing import Dict, Iterable, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem
from molpal.objectives.base import Objective
from configargparse import ArgumentParser
import requests, time
import getpass

class GlideObjective(Objective):
    """A GlideObjective calculates the objective function by calculating the
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
        minimize: bool = True,
        **kwargs,
    ): 
        self.slurm_url = 'Your slurm mgt node'
        self.user_name = os.environ.get("SLURM_USER", getpass.getuser())
        self.slurm_token = self.get_slurm_token(days=1)
        self.version = 'v0.0.38'
        ncpu, utils_dir, targetfname = parse_config(objective_config)
        self.utils_dir = utils_dir
        # self.ncpu = 20
        self.ncpu = ncpu
        self.targetfname = targetfname
        self.errorTolerance = 20
        super().__init__(minimize=minimize)
        self.word_pair = {
            'UTILSPATH': self.utils_dir,
            'TARGETFNAME': self.targetfname
        }

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
        smis = np.array(smis)
        df = pd.DataFrame(smis, columns=['SMILES'])
        df['ID'] = ['d%d'%i for i in range(len(smis))]
        to_sdf_fname = os.path.join(self.utils_dir, 'to_sdf.py')
        input_fname = os.path.join(self.utils_dir, 'temp', 'tobedocked.csv')
        titrated_fname = os.path.join(self.utils_dir, 'temp', 'titrated.smi')
        ligprep_fname = os.path.join(self.utils_dir, 'ligprep.inp')
        glide_fname = os.path.join(self.utils_dir, 'glide-dock_SP_B2.in')
        df.to_csv(input_fname, header=['SMILES', 'ID'], index=False)
        oldfnames = self.get_existing_fnames(directory=self.utils_dir)
        self.modify_config_files(fname=os.path.join(self.utils_dir, 'glide-dock_SP_B2_template.in'), word_pairs=self.word_pair)
        self.modify_config_files(fname=os.path.join(self.utils_dir, 'ligprep_template.inp'), word_pairs=self.word_pair)


        SCRIPT = [
            '#!/bin/bash',
            '#SBATCH --job-name=dock',
            '#SBATCH --partition=cpu',
            '#SBATCH --ntasks=1',
            '#SBATCH --cpus-per-task=%d'%self.ncpu,
            '#SBATCH --mem=192GB',
            '#SBATCH --time=14-00:00:00',
            '#SBATCH -e %s/temp/dock.err'%self.utils_dir,
            '#SBATCH -o %s/temp/dock.out'%self.utils_dir,
            '#SBATCH -D %s'%self.utils_dir,
            'module load Schrodinger',
            'module load chemaxon/20.7',
            "molconvert sdf %s | cxcalc -S -t MAJORMS_7.4 -g majorms -H 7.4 -M true | molconvert smiles:-TId:MAJORMS_7.4 |awk '{print $3,$2}' > %s"%(input_fname, titrated_fname),
            'python %s %s %s'%(to_sdf_fname, titrated_fname, self.utils_dir),
            '$SCHRODINGER/ligprep -inp %s -HOST "localhost:20" -NJOBS %d  -WAIT -LOCAL'%(ligprep_fname, self.ncpu),
            '$SCHRODINGER/glide %s -OVERWRITE  -HOST "localhost:20" -NJOBS %d -WAIT'%(glide_fname, self.ncpu),
        ]
        
        bash_fname = os.path.join(self.utils_dir, 'dock.sh')
        SCRIPT = [i+'\n' for i in SCRIPT]
        with open(bash_fname, "w") as sh_file:
            sh_file.writelines(SCRIPT)

        

        job_id = int(os.popen('sbatch %s'%bash_fname).read().strip('\n').split(' ')[-1])
        print('Docking job %d submitted!'%job_id, flush=True)
        status = self.wait_done(job_id)
        if status != "COMPLETED":
            print('Docking Failed!!')
            return None

        scores = self.read_results(dataframe=df)

        newfnames = self.get_existing_fnames(directory=self.utils_dir)
        for f in newfnames:
            if f not in oldfnames:
                os.remove(f)
        return scores

    def to_sdf(self, smis):
        mergedSDF_OUT = Chem.SDWriter('%sinput2D.sdf'%self.utils_dir)
        for s in smis:
            try:
                mol = Chem.MolFromSmiles(s)
                mol.SetProp("_Name", s)
                mergedSDF_OUT.write(mol)
            except:
                print('Not able to read compound or get SMILES or ID')
        mergedSDF_OUT.close()
        return

    def read_results(self, dataframe):
        res = pd.read_csv(os.path.join(self.utils_dir, 'glide-dock_SP_B2.csv'))
        res = res.rename(columns={'title':'ID'})
        res = res.drop(columns=['SMILES'])
        res['ID'] = res['ID'].str.replace("\"","")
        res = res.drop_duplicates(subset=['ID'])

        df = pd.merge(dataframe, res, on=['ID'], how='left')
        df['r_i_docking_score'] = df['r_i_docking_score'].fillna(0)
        scores = {}
        for index, row in df.iterrows():
            row['r_i_docking_score'] = min(0, row['r_i_docking_score'])
            scores[row['SMILES']] = self.c*row['r_i_docking_score']
        return scores
    
    def modify_config_files(self, fname: Union[str, Path], word_pairs:dict):
        with open(fname, 'r') as file: 
            text = file.read() 
            for k, v in word_pairs.items():
                text = text.replace(k, v) 
        with open(fname.replace('_template', ''), 'w') as file: 
            file.write(text)
        return

    def get_existing_fnames(self, directory=None):
        if not directory:
            directory = os.getcwd()
        files = [f for f in os.listdir(directory) if os.path.isfile(f)]
        return set(files)
    
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

def parse_config(config: str):
    """parse a LookupObjective configuration file

    Parameters
    ----------
    config : str
        the config file to parse

    Returns
    -------
    path : str
        the filepath of the lookup CSV file
    sep : str
        the CSV separator
    title_line : bool
        is there a title in in the lookup file?
    smiles_col : int
        the column containing the SMILES string in the lookup file
    data_col : int
        the column containing the desired data in the lookup file
    """
    parser = ArgumentParser()
    parser.add_argument("config", is_config_file=True)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--utils-dir", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)

    args = parser.parse_args(config)
    return (args.ncpu, args.utils_dir, args.target)
