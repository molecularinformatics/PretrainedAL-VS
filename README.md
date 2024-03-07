

This repository provides pytorch source code, and data associated with our Journal of Chemical Information and Modeling publication, "Large-Scale Pretraining Improves Sample Efficiency of Active Learning-Based Virtual Screening".

Paper: [JCIM Link](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01938) / [Arxiv Link](https://arxiv.org/abs/2309.11687)

## PretrainedAL-VS

**PretrainedAL-VS** is an Active learning framework with the pretained large Language model added as surrogate model.

<center>

<div align=center><img width="800" height="500" src="https://github.com/molecularinformatics/PretrainedAL-VS/blob/90ed9f9b2981385d3794684d579f29848e916240/assets/fig.png"/></div>
</center>  


## Installation 

Our code is based on MolPal and MolFormer, please check the installation of MolPal and MolFormer.

* MolPal: https://github.com/coleygroup/molpal
* MolFormer:  https://github.com/IBM/molformer


## Quick Start

The model training, inference, and molecular docking parts are customized to run on HPC via SLURM RESTAPI. Please check the slide for the detail [link](https://docs.google.com/presentation/d/1Hg2poYb1eFRWk5aNHCY-awUyPc3_GFr2/edit?usp=sharing&ouid=113885279755421323328&rtpof=true&sd=true)

### Download MolFormer 
Download MolFormer from [link](https://ibm.ent.box.com/v/MoLFormer-data), and put it in `molpal/models/transformer/pretrained_ckpt/`
### Config slurm API server 

Config slurm API server for model training and inference in `molpal/models/transformermodels.py`
```python 
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
        self.slurm_url = 'Your Slurm Management Node ULR and port'
        self.user_name = os.environ.get("SLURM_USER", getpass.getuser())
        self.version = 'v0.0.38'
        self.errorTolerance = 20
        
        if slurm_token == None:
            self.slurm_token = self.get_slurm_token(days=1)

        self.uncertainty = uncertainty

```
Config slurm API server for SCHRODINGER/glide docking in `molpal/objectives/glide.py` if you want to use SCHRODINGER glide
```python
def __init__(
        self,
        objective_config: str,
        minimize: bool = True,
        **kwargs,
    ): 
        self.slurm_url = 'http://edge-hpc-mgt-101.hpc.biogen.com:6820'
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
```

``` sbatch run_molpal.sh```
## Contact
Ye Wang (ye.wang@biogen.com)


