#!/bin/bash -l                                                                                                       

#SBATCH --job-name="job_name"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G                                                                                   

#SBATCH --account=Education-TPM-MSc-EPA                                                                          

module load 2023rc1
module load 2023rc1-gcc11
module load python
module load py-pandas
module load py-geopandas
module load --user-networkx
module load py-scipy
module load py-numpy
module load py-shapely
module load --user-multiprocess
module load --user-igraph



srun python FlowAssign_GapBenefit.py