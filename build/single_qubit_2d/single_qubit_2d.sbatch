#!/bin/bash
# Job name:
#SBATCH --job-name=single_qubit_2d
#
# Partition:
#SBATCH --partition=cmt
#
# Processors:
#SBATCH --ntasks=16
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=7-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o single_qubit_2d.o%j
#------------------------------------------------------

echo "Starting job single_qubit_2d"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=single_qubit_2d.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job single_qubit_2d complete"
