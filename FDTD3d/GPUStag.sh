#!/bin/bash
#SBATCH --account=def-spichard
#SBATCH --nodes=1                # NUMBER OF NODES
#SBATCH --gres=gpu:4              # request GPU "generic resource", 4 on Cedar, 2 on Graham
#SBATCH --mem=64GB               # memory per node
#SBATCH --time=0-00:10            # time (DD-HH:MM)
#SBATCH --output=%N-%j.out       # %N for node name., %j for jobID
./FDTD3d
