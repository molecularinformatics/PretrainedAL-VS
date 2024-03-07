#!/bin/bash
#SBATCH --job-name=dock
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=192GB
#SBATCH --time=14-00:00:00
#SBATCH -e pathto/PretrainedAL-VS/molpal/objectives/glide_utils/temp/dock.err
#SBATCH -o pathto/PretrainedAL-VS/molpal/objectives/glide_utils/temp/dock.out
#SBATCH -D pathto/PretrainedAL-VS/molpal/molpal/objectives/glide_utils
module load Schrodinger
module load chemaxon/20.7
molconvert sdf pathto/PretrainedAL-VS/molpal/objectives/glide_utils/temp/tobedocked.csv/tobedocked.csv | cxcalc -S -t MAJORMS_7.4 -g majorms -H 7.4 -M true | molconvert smiles:-TId:MAJORMS_7.4 |awk '{print $3,$2}' > pathto/PretrainedAL-VS/molpal/objectives/glide_utils/temp/titrated.smi
python pathto/PretrainedAL-VS/molpal/objectives/glide_utils/to_sdf.py pathto/PretrainedAL-VS/molpal/objectives/glide_utils/temp/titrated.smi pathto/PretrainedAL-VS/molpal/objectives/glide_utils
$SCHRODINGER/ligprep -inp pathto/PretrainedAL-VS/molpal/objectives/glide_utils/ligprep.inp -HOST "localhost:20" -NJOBS 40  -WAIT -LOCAL
$SCHRODINGER/glide pathto/PretrainedAL-VS/molpal/objectives/glide_utils/your_glide.in -OVERWRITE  -HOST "localhost:20" -NJOBS 40 -WAIT
