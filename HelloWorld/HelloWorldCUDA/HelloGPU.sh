#!/bin/bash
#SBATCH --account=def-spichard
#SBATCH --gres=gpu:1              # request GPU "generic resource", 4 on Cedar, 2 on Graham
#SBATCH --mem=256MB               # memory per node
#SBATCH --time=0-00:2            # time (DD-HH:MM) hahahaha
./hello
./deviceQuery
