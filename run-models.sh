ml load py-numpy/1.19.2_py36
ml load py-numba/0.53.1_py36
ml load cuda/11.2.0
ml load py-scipy/1.4.1_py36
srun python3 cs1500_singleGPU.py
srun python3 cs1500_manyGPU.py
srun python3 cs3000_singleGPU.py
srun python3 cs3000_manyGPU.py
srun python3 cs15000_singleGPU.py
srun python3 cs15000_manyGPU.py
srun python3 cs50000_singleGPU.py
srun python3 cs50000_manyGPU.py
srun python3 cs100000_singleGPU.py
srun python3 cs100000_manyGPU.py