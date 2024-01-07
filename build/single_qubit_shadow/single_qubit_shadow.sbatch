#!/bin/bash
# Job name:
#SBATCH --job-name=single_qubit_shadow
#
# Partition:
#SBATCH --partition=normal
#
# Processors:
#SBATCH --ntasks=32
#SBATCH --nodes=2
#
# Wall clock limit:
#SBATCH --time=4-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o single_qubit_shadow.o%j
#------------------------------------------------------

echo "Starting job single_qubit_shadow"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=single_qubit_shadow.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job single_qubit_shadow complete"
