#!/bin/bash
#SBATCH --account=def-spichard
#SBATCH --gres=gpu:1              # request GPU "generic resource", 4 on Cedar, 2 on Graham
#SBATCH --mem=16GB               # memory per node
#SBATCH --time=0-00:1            # time (DD-HH:MM) haha
./hello
-I/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/cuda/9.0.176/extras/demo_suite ./deviceQuery
